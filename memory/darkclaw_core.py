"""
╔═══════════════════════════════════════════════════════════╗
║          DARKCLAW v1.0 — Context Graph Memory Engine      ║
║          Built for Darkclaw Neural Stack                  ║
║          Bradley Allen Foster / Darkclaw Project          ║
╚═══════════════════════════════════════════════════════════╝

Architecture:
  - Dual-layer memory: NetworkX graph (relational) + TF-IDF (semantic)
  - Stale-fact protection: supersedes (subject, predicate, object) triples
    BUT allows multiple objects per predicate when semantically distinct
    (e.g. ROUTES_TO → [CT210, Anthropic] are both valid simultaneously)
  - Agent-scoped: each agent gets its own context namespace
  - Harness-aware: maps to Claude Code s05 Knowledge-on-Demand pattern
  - Persistent: SQLite backing for graph + audit log
  - Tool-result format: returns JSON payloads ready for LiteLLM injection

Three memory tiers (mirrors Darkclaw Green/Yellow/Red classifier):
  GREEN  — direct fact lookup, high confidence
  YELLOW — distant fact or moderate confidence
  RED    — join query or low confidence / fallback
"""

import networkx as nx
import os
import sqlite3
import json
import threading
import time
import uuid
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Fact:
    subject: str
    predicate: str
    object: str
    agent_id: str
    turn_id: int
    fact_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    superseded_by: Optional[str] = None
    multi_valued: bool = False  # True = predicate allows multiple objects

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Turn:
    turn_id: int
    agent_id: str
    speaker: str
    text: str
    turn_type: str = "DISTRACTOR"      # FACT | DISTRACTOR | QUERY
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    multi_valued: bool = False          # allow multiple objects for this predicate
    query_type: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class QueryResult:
    answer: str
    confidence: float
    method: str                  # graph_direct | graph_join | vector | fallback
    tokens_used: int
    facts_used: List[str]
    hop_count: int = 0
    latency_ms: float = 0.0

    def to_tool_result(self) -> dict:
        """Format as LiteLLM tool_result payload — s05 Knowledge-on-Demand."""
        return {
            "type": "tool_result",
            "memory_tier": self._classify_tier(),
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "tokens": self.tokens_used,
            "hops": self.hop_count,
            "latency_ms": round(self.latency_ms, 2),
            "source_facts": self.facts_used,
        }

    def _classify_tier(self) -> str:
        if self.confidence >= 0.85:
            return "GREEN"
        elif self.confidence >= 0.45:
            return "YELLOW"
        else:
            return "RED"


# ─────────────────────────────────────────────
#  MULTI-VALUED PREDICATE REGISTRY
# ─────────────────────────────────────────────

# Predicates that naturally have multiple valid targets simultaneously.
# These should NOT trigger stale-fact removal when a new target is added.
MULTI_VALUED_PREDICATES = {
    "ROUTES_TO", "DEPENDS_ON", "CONTAINS", "INCLUDES",
    "CONNECTS_TO", "SUPPORTS", "PROVIDES", "MANAGES",
    "LINKED_TO", "COMPATIBLE_WITH",
}

# Predicates that are single-valued — new value supersedes old.
SINGLE_VALUED_PREDICATES = {
    "HAS_STATUS", "HAS_PRIORITY", "HAS_VERSION", "ASSIGNED_TO",
    "RUNS_ON", "HAS_IP", "HAS_PORT", "USES", "BUILT_WITH",
    "RUNS_IN", "LLC_NUMBER", "STORES", "REPORTS_TO",
}


# ─────────────────────────────────────────────
#  ALIAS REGISTRY
# ─────────────────────────────────────────────

class AliasRegistry:
    """
    Resolves vocabulary mismatches at write time.
    In production: replace _resolve() with LLM entity-linking.
    For Darkclaw: domain rules for Book Burrow + homelab vocabulary.
    """

    def __init__(self):
        self._aliases: Dict[str, str] = {}
        self._load_defaults()

    def _load_defaults(self):
        defaults = {
            # Book Burrow
            "the bookstore": "Book_Burrow",
            "book burrow": "Book_Burrow",
            "our store": "Book_Burrow",
            "316 s buell": "Book_Burrow",
            "ingram": "Ingram_Supplier",
            "the supplier": "Ingram_Supplier",
            "pos": "POS_System",
            "the pos": "POS_System",
            "shop-bot": "SHOP_BOT",
            "shopbot": "SHOP_BOT",
            "phi3": "Phi3_Mini",
            "phi-3": "Phi3_Mini",
            "phi 3": "Phi3_Mini",

            # Darkclaw homelab
            "the homelab": "Darkclaw_Stack",
            "darkclaw": "Darkclaw_Stack",
            "litellm": "LiteLLM_Gateway",
            "litellm gateway": "LiteLLM_Gateway",
            "the gateway": "LiteLLM_Gateway",
            "ct212": "CT212_LiteLLM",
            "ct210": "CT210_Ollama",
            "ct213": "CT213_Website",
            "t5810": "T5810_Server",
            "the server": "T5810_Server",
            "precision": "T5810_Server",
            "daddypi": "RaspberryPi5",
            "the pi": "RaspberryPi5",
            "chromadb": "ChromaDB",
            "networkx": "NetworkX",
            "darkclaw": "Darkclaw",

            # Prosthetic
            "the arm": "Prosthetic_Arm",
            "prosthetic": "Prosthetic_Arm",
            "the elbow": "Titanium_Elbow",
            "myoware": "MyoWare_Sensor",
            "emg": "EMG_System",

            # Agents
            "main agent": "Darkclaw_Main",
            "guardian": "Guardian_Agent",
            "monitoring": "Guardian_Agent",
        }
        for alias, canonical in defaults.items():
            self._aliases[alias.lower()] = canonical

    def register(self, alias: str, canonical: str):
        self._aliases[alias.lower()] = canonical

    def resolve(self, text: str) -> str:
        lower = text.lower().strip()
        if lower in self._aliases:
            return self._aliases[lower]
        best, best_len = text, 0
        for alias, canonical in self._aliases.items():
            if alias in lower and len(alias) > best_len:
                best, best_len = canonical, len(alias)
        return best

    def extract_entities(self, text: str) -> List[str]:
        found = []
        lower = text.lower()
        for alias, canonical in self._aliases.items():
            if alias in lower:
                found.append(canonical)
        return list(set(found))


# ─────────────────────────────────────────────
#  CONTEXT GRAPH
# ─────────────────────────────────────────────

class ContextGraph:
    """
    NetworkX directed multigraph — facts stored as (subject, predicate, object) triples.

    Stale-fact rule (refined from benchmark bugs):
      - SINGLE_VALUED predicates: new value drops old edge (e.g. HAS_STATUS)
      - MULTI_VALUED predicates: new value ADDS alongside old (e.g. ROUTES_TO)
      - Unknown predicates: default to single-valued (safer)
    """

    def __init__(self, agent_id: str, alias_registry: AliasRegistry):
        self.agent_id = agent_id
        self.graph = nx.MultiDiGraph()
        self.aliases = alias_registry
        self._facts: Dict[str, Fact] = {}

    def ingest(self, turn: Turn) -> Optional[Fact]:
        if turn.turn_type != "FACT" or turn.subject is None:
            return None

        subject = self.aliases.resolve(turn.subject)
        obj     = self.aliases.resolve(str(turn.object))
        pred    = turn.predicate
        multi   = turn.multi_valued or pred in MULTI_VALUED_PREDICATES

        # ── Stale-fact protection (single-valued predicates only) ──────
        if not multi:
            stale = [
                (u, v, k, d)
                for u, v, k, d in self.graph.edges(keys=True, data=True)
                if u == subject and d.get("predicate") == pred
            ]
            for u, v, k, d in stale:
                old_fid = d.get("fact_id")
                if old_fid in self._facts:
                    self._facts[old_fid].superseded_by = "pending"
                self.graph.remove_edge(u, v, key=k)
        # ──────────────────────────────────────────────────────────────

        self.graph.add_node(subject)
        self.graph.add_node(obj)

        fact = Fact(
            subject=subject, predicate=pred, object=obj,
            agent_id=turn.agent_id, turn_id=turn.turn_id,
            timestamp=turn.timestamp, multi_valued=multi,
        )
        self.graph.add_edge(
            subject, obj,
            predicate=pred, fact_id=fact.fact_id,
            timestamp=fact.timestamp, turn_id=turn.turn_id,
        )
        self._facts[fact.fact_id] = fact

        # Fix pending superseded_by references
        for u, v, k, d in (stale if not multi else []):
            old_fid = d.get("fact_id")
            if old_fid and old_fid in self._facts:
                self._facts[old_fid].superseded_by = fact.fact_id

        return fact

    def query_direct(self, subject: str, predicate: str) -> Optional[QueryResult]:
        """Single-hop lookup — returns all matching values."""
        t0 = time.perf_counter()
        subject = self.aliases.resolve(subject)
        matches = []
        for u, v, data in self.graph.edges(data=True):
            if u == subject and data.get("predicate") == predicate:
                matches.append((v, data.get("fact_id", "")))

        if not matches:
            return None

        values = [m[0] for m in matches]
        fact_ids = [m[1] for m in matches]
        answer = values[0] if len(values) == 1 else ", ".join(values)
        text = f"{subject} {predicate} {answer}"
        return QueryResult(
            answer=answer,
            confidence=1.0,
            method="graph_direct",
            tokens_used=max(1, len(text) // 4),
            facts_used=fact_ids,
            hop_count=1,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def query_join(self, query_text: str) -> Optional[QueryResult]:
        """
        Two-hop traversal for join queries.
        Also handles inverse lookups: "what X uses Y?" → find subject with Y as object.
        """
        t0 = time.perf_counter()

        # Collect candidate entities from query
        entities = self.aliases.extract_entities(query_text)
        for node in self.graph.nodes():
            if node.lower().replace("_", " ") in query_text.lower():
                entities.append(node)
        entities = list(set(entities))

        # ── Inverse single-hop: entity appears as OBJECT, find its SUBJECT ──
        # e.g. "what uses Phi3_Mini?" → walk in-edges to Phi3_Mini
        for entity in entities:
            if entity not in self.graph:
                continue
            in_edges = list(self.graph.in_edges(entity, data=True, keys=True))
            for u, v, _, data in in_edges:
                pred = data.get("predicate", "")
                score = self._score_relevance(query_text, pred, "", u)
                path_text = f"{u} {pred} {entity}"
                result = QueryResult(
                    answer=u,
                    confidence=min(0.92, score + 0.3),  # boost: direct inverse
                    method="graph_join",
                    tokens_used=max(1, len(path_text) // 4),
                    facts_used=[data.get("fact_id", "")],
                    hop_count=1,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                # Only return if query contains a predicate hint matching this edge
                # (extract_predicate_hints is defined at module scope below)
                hints = extract_predicate_hints(query_text)
                if pred in hints or not hints:
                    return result

        best_result = None
        best_score  = 0.0

        for entity in entities:
            if entity not in self.graph:
                continue

            out_edges = list(self.graph.out_edges(entity, data=True, keys=True))
            in_edges  = list(self.graph.in_edges(entity, data=True, keys=True))

            intermediates = (
                [(v, d) for _, v, _, d in out_edges] +
                [(u, d) for u, _, _, d in in_edges]
            )

            for intermediate, hop1_data in intermediates:
                if intermediate == entity:
                    continue
                hop2_edges = (
                    list(self.graph.out_edges(intermediate, data=True, keys=True)) +
                    list(self.graph.in_edges(intermediate, data=True, keys=True))
                )
                for edge in hop2_edges:
                    if len(edge) != 4:
                        continue
                    src, tgt, _, data = edge
                    target = tgt if src == intermediate else src
                    if target == entity:
                        continue
                    score = self._score_relevance(
                        query_text,
                        hop1_data.get("predicate", ""),
                        data.get("predicate", ""),
                        target,
                    )
                    if score > best_score:
                        best_score = score
                        path_text = f"{entity} → {intermediate} → {target}"
                        best_result = QueryResult(
                            answer=target,
                            confidence=score,
                            method="graph_join",
                            tokens_used=max(1, len(path_text) // 4),
                            facts_used=[
                                hop1_data.get("fact_id", ""),
                                data.get("fact_id", ""),
                            ],
                            hop_count=2,
                            latency_ms=(time.perf_counter() - t0) * 1000,
                        )
        return best_result

    def get_entity_context(self, entity: str) -> dict:
        entity = self.aliases.resolve(entity)
        if entity not in self.graph:
            return {"entity": entity, "facts": [], "found": False}
        facts = []
        for u, v, data in self.graph.edges(data=True):
            if u == entity:
                facts.append({
                    "predicate": data.get("predicate"),
                    "value": v,
                    "fact_id": data.get("fact_id"),
                })
        return {"entity": entity, "facts": facts, "found": True}

    def _score_relevance(self, query: str, pred1: str, pred2: str, target: str) -> float:
        q_words = set(query.lower().split())
        p_words = set((pred1 + " " + pred2).lower().split())
        t_words = set(target.lower().replace("_", " ").split())
        overlap = len(q_words & (p_words | t_words))
        return min(0.95, 0.3 + overlap * 0.2)

    def stats(self) -> dict:
        active = sum(1 for f in self._facts.values() if f.superseded_by is None)
        return {
            "agent_id": self.agent_id,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "total_facts": len(self._facts),
            "active_facts": active,
            "stale_facts": len(self._facts) - active,
        }


# ─────────────────────────────────────────────
#  VECTOR MEMORY
# ─────────────────────────────────────────────

_EMBED_URL   = os.environ.get("DARKCLAW_MEMORY_NODE_URL",
                              "http://192.168.1.130:11434").rstrip("/")
_EMBED_MODEL = os.environ.get("DARKCLAW_EMBED_MODEL", "nomic-embed-text:latest")


def _embed_batch(texts: List[str], prefix: str) -> Optional[List[List[float]]]:
    """Batch-embed via the memory node. Returns None on any failure."""
    import urllib.request
    try:
        out: List[List[float]] = []
        for i in range(0, len(texts), 64):
            body = json.dumps({
                "model": _EMBED_MODEL,
                # nomic-embed is asymmetric: prefixes matter for quality
                "input": [f"{prefix}: {t[:2000]}" for t in texts[i:i + 64]],
            }).encode()
            req = urllib.request.Request(
                f"{_EMBED_URL}/api/embed", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                out.extend(json.loads(r.read().decode())["embeddings"])
        return out if len(out) == len(texts) else None
    except Exception:
        return None


class VectorMemory:
    """
    Semantic retrieval over ingested turns.

    Primary path: real embeddings (nomic-embed-text on the memory node),
    computed lazily in batch on first retrieve and cached — new chunks
    embed incrementally. Fallback path: the original TF-IDF refit, used
    whenever the memory node is unreachable so retrieval never breaks.
    """

    def __init__(self):
        self._chunks: List[str] = []
        self._metadata: List[dict] = []
        self._emb: List[List[float]] = []   # aligned with _chunks[:len(_emb)]
        self._node_ok = True
        # Retrieves now run on threads (asyncio.to_thread), so the cache
        # build races itself: two threads both see it empty, both embed the
        # whole store, both extend → _emb longer than _chunks → IndexError
        # on argsort indexes. One lock: single builder, consistent snapshots.
        self._lock = threading.Lock()

    def ingest(self, text: str, metadata: dict = None):
        with self._lock:
            self._chunks.append(text)
            self._metadata.append(metadata or {})

    def _ensure_embedded_locked(self) -> bool:
        """Embed chunks past the cached prefix. Caller must hold _lock."""
        if not self._node_ok:
            return False
        pending = self._chunks[len(self._emb):]
        if pending:
            got = _embed_batch(pending, "search_document")
            if got is None:
                self._node_ok = False   # degrade to TF-IDF this process
                return False
            self._emb.extend(got)
        return bool(self._emb)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float, dict]]:
        with self._lock:
            if not self._chunks:
                return []
            usable = self._ensure_embedded_locked()
            if usable:
                # snapshot with lengths pinned equal — new ingests after
                # this point wait for the next retrieve
                n = len(self._emb)
                chunks, metas = self._chunks[:n], self._metadata[:n]
                emb = list(self._emb)
        if usable:
            # query embed is a network call — deliberately outside the lock
            got = _embed_batch([query], "search_query")
            if got:
                import numpy as np
                q = np.asarray(got[0])
                m = np.asarray(emb)
                sims = (m @ q) / (
                    (np.linalg.norm(m, axis=1) * np.linalg.norm(q)) + 1e-9)
                idx = sims.argsort()[::-1][:top_k]
                return [(chunks[i], float(sims[i]), metas[i])
                        for i in idx if sims[i] > 0]
            self._node_ok = False
        return self._retrieve_tfidf(query, top_k)

    def _retrieve_tfidf(self, query: str, top_k: int) -> List[Tuple[str, float, dict]]:
        try:
            with self._lock:
                chunks = list(self._chunks)
                metas = list(self._metadata)
            corpus = chunks + [query]
            vec = TfidfVectorizer(stop_words="english")
            mat = vec.fit_transform(corpus)
            sims = cosine_similarity(mat[-1], mat[:-1]).flatten()
            idx = sims.argsort()[::-1][:top_k]
            return [(chunks[i], float(sims[i]), metas[i])
                    for i in idx if sims[i] > 0]
        except Exception:
            return []


# ─────────────────────────────────────────────
#  PERSISTENCE LAYER
# ─────────────────────────────────────────────

class DarkclawPersistence:
    """
    SQLite persistence layer.  Uses a single persistent connection with WAL mode
    so we never accumulate open file descriptors — each `sqlite3.connect()` call
    normally opens the DB file plus a WAL journal file.  Under load (hundreds of
    fact-ingests per minute) that was the primary FD leak source.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        # Single connection — asyncio is single-threaded so no locking needed.
        # check_same_thread=False is a safety valve for any off-loop calls.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id      TEXT PRIMARY KEY,
                agent_id     TEXT NOT NULL,
                subject      TEXT NOT NULL,
                predicate    TEXT NOT NULL,
                object       TEXT NOT NULL,
                turn_id      INTEGER,
                timestamp    REAL,
                multi_valued INTEGER DEFAULT 0,
                superseded_by TEXT,
                raw_json     TEXT
            );
            CREATE TABLE IF NOT EXISTS turns (
                turn_id   INTEGER,
                agent_id  TEXT NOT NULL,
                speaker   TEXT,
                text      TEXT,
                turn_type TEXT,
                timestamp REAL,
                PRIMARY KEY (turn_id, agent_id)
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                agent_id   TEXT,
                fact_id    TEXT,
                detail     TEXT,
                timestamp  REAL DEFAULT (unixepoch('now','subsec'))
            );
            CREATE INDEX IF NOT EXISTS idx_facts_subpred
                ON facts(agent_id, subject, predicate);
        """)
        self._conn.commit()

    def save_fact(self, fact: Fact):
        self._conn.execute("""
            INSERT OR REPLACE INTO facts
            (fact_id,agent_id,subject,predicate,object,turn_id,
             timestamp,multi_valued,superseded_by,raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (fact.fact_id, fact.agent_id, fact.subject, fact.predicate,
              fact.object, fact.turn_id, fact.timestamp,
              int(fact.multi_valued), fact.superseded_by,
              json.dumps(fact.to_dict())))
        self._conn.commit()

    def save_turn(self, turn: Turn):
        self._conn.execute("""
            INSERT OR REPLACE INTO turns
            (turn_id,agent_id,speaker,text,turn_type,timestamp)
            VALUES (?,?,?,?,?,?)
        """, (turn.turn_id, turn.agent_id, turn.speaker,
              turn.text, turn.turn_type, turn.timestamp))
        self._conn.commit()

    def mark_superseded(self, old_fid: str, new_fid: str):
        self._conn.execute("UPDATE facts SET superseded_by=? WHERE fact_id=?",
                           (new_fid, old_fid))
        self._conn.execute(
            "INSERT INTO audit_log(event_type,fact_id,detail) VALUES(?,?,?)",
            ("SUPERSEDE", old_fid, f"replaced_by:{new_fid}"))
        self._conn.commit()

    def load_active_facts(self, agent_id: str) -> List[dict]:
        rows = self._conn.execute(
            "SELECT raw_json FROM facts WHERE agent_id=? AND superseded_by IS NULL",
            (agent_id,)).fetchall()
        return [json.loads(r[0]) for r in rows]


# ─────────────────────────────────────────────
#  PREDICATE HINT EXTRACTION
# ─────────────────────────────────────────────

PREDICATE_PATTERNS = [
    (r"\broute[sd]?\b|\bforward[sd]?\b|\bsend[s]?\b\s+to\b", "ROUTES_TO"),
    (r"\bdepend[s]?\b|\brequire[s]?\b|\bneed[s]?\b",          "DEPENDS_ON"),
    (r"\brun[s]?\b|\bhost[s]?\b|\bexecut[es]?\b",             "RUNS_ON"),
    (r"\brun[s]?\s+in\b|\binside\b|\bcontainer\b",            "RUNS_IN"),
    (r"\buse[sd]?\b|\butilize[sd]?\b|\bleverag[es]?\b",       "USES"),
    (r"\bstatus\b|\bstate\b",                                  "HAS_STATUS"),
    (r"\bpriority\b|\burgency\b",                              "HAS_PRIORITY"),
    (r"\bversion\b|\bv\d\b",                                   "HAS_VERSION"),
    (r"\bassign[ed]?\b|\bowner\b|\bresponsib\b",               "ASSIGNED_TO"),
    (r"\bcontain[s]?\b|\binclude[s]?\b",                       "CONTAINS"),
    (r"\bconnect[s]?\b|\blink[s]?\b",                          "CONNECTS_TO"),
    (r"\bstore[sd]?\b|\bsave[sd]?\b\|\bpersist",              "STORES"),
    (r"\bbuilt.with\b|\btechnolog\b|\bstack\b|\blanguage\b",   "BUILT_WITH"),
    (r"\bllc\b|\bregistration\b|\bnumber\b",                   "LLC_NUMBER"),
]

def extract_predicate_hints(query: str) -> List[str]:
    hints = []
    for pattern, pred in PREDICATE_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            hints.append(pred)
    return hints if hints else list({p for _, p in PREDICATE_PATTERNS})


# ─────────────────────────────────────────────
#  DARKCLAW ENGINE  (unified interface)
# ─────────────────────────────────────────────

class DarkclawEngine:
    """
    Darkclaw v1.0 — Unified dual-layer memory for Darkclaw multi-agent stack.

    Maps to Claude Code harness:
      s05 Knowledge-on-Demand  → to_tool_result() injection format
      s07 Persistent Tasks     → SQLite backing
      Three-tier classifier    → GREEN / YELLOW / RED per query confidence
    """

    def __init__(self, db_path: str = "/home/claude/darkclaw/darkclaw.db"):
        import os; os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.aliases     = AliasRegistry()
        self.persistence = DarkclawPersistence(db_path)
        self._graphs:  Dict[str, ContextGraph] = {}
        self._vectors: Dict[str, VectorMemory] = {}
        self._turn_ctr: Dict[str, int] = {}
        self._reload_from_db()

    def _reload_from_db(self):
        """Warm up the in-memory graph and vector stores from persisted facts."""
        try:
            conn = self.persistence._conn
            # Reload all active facts into graphs
            rows = conn.execute(
                "SELECT agent_id, subject, predicate, object, "
                "turn_id, timestamp, multi_valued, fact_id, raw_json "
                "FROM facts WHERE superseded_by IS NULL"
            ).fetchall()
            for row in rows:
                aid, subj, pred, obj, tid, ts, mv, fid, raw = row
                self._turn_ctr[aid] = max(self._turn_ctr.get(aid, 0), tid or 0)
                graph = self._graph(aid)
                try:
                    fact = Fact(
                        fact_id=fid, agent_id=aid, subject=subj,
                        predicate=pred, object=obj, turn_id=tid or 0,
                        timestamp=ts or 0.0, multi_valued=bool(mv),
                    )
                    graph.graph.add_node(subj)
                    graph.graph.add_node(obj)
                    graph.graph.add_edge(subj, obj, predicate=pred, fact=fact)
                except Exception:
                    pass

            # Reload turns into vector stores for semantic search
            rows = conn.execute(
                "SELECT agent_id, text, turn_type, turn_id "
                "FROM turns WHERE text IS NOT NULL AND text != ''"
            ).fetchall()
            for aid, text, ttype, tid in rows:
                vec = self._vector(aid)
                vec.ingest(text, {"turn_id": tid, "type": ttype or "TURN"})
        except Exception:
            pass   # fresh DB — nothing to reload

    # ── internal helpers ───────────────────────────────────────────────

    def _graph(self, agent_id: str) -> ContextGraph:
        if agent_id not in self._graphs:
            self._graphs[agent_id] = ContextGraph(agent_id, self.aliases)
        return self._graphs[agent_id]

    def _vector(self, agent_id: str) -> VectorMemory:
        if agent_id not in self._vectors:
            self._vectors[agent_id] = VectorMemory()
        return self._vectors[agent_id]

    def _next_turn(self, agent_id: str) -> int:
        self._turn_ctr[agent_id] = self._turn_ctr.get(agent_id, 0) + 1
        return self._turn_ctr[agent_id]

    # ── public API ─────────────────────────────────────────────────────

    def ingest_turn(self, turn: Turn) -> Optional[Fact]:
        self.persistence.save_turn(turn)
        vec = self._vector(turn.agent_id)
        if turn.turn_type == "FACT":
            fact = self._graph(turn.agent_id).ingest(turn)
            if fact:
                self.persistence.save_fact(fact)
            vec.ingest(turn.text, {"turn_id": turn.turn_id, "type": "FACT"})
            return fact
        elif turn.turn_type == "DISTRACTOR":
            vec.ingest(turn.text, {"turn_id": turn.turn_id, "type": "DISTRACTOR"})
        return None

    def ingest_fact(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        object_val: str,
        speaker: str = "system",
        text: str = None,
        multi_valued: bool = None,
    ) -> Optional[Fact]:
        """Convenience method for direct structured fact injection."""
        mv = multi_valued if multi_valued is not None else predicate in MULTI_VALUED_PREDICATES
        turn = Turn(
            turn_id=self._next_turn(agent_id),
            agent_id=agent_id,
            speaker=speaker,
            text=text or f"{subject} {predicate} {object_val}",
            turn_type="FACT",
            subject=subject,
            predicate=predicate,
            object=object_val,
            multi_valued=mv,
        )
        return self.ingest_turn(turn)

    def query(self, query_text: str, agent_id: str = "main") -> QueryResult:
        """
        Unified query: graph direct → graph join → vector fallback.
        Returns QueryResult with .to_tool_result() for LiteLLM injection.
        """
        t0    = time.perf_counter()
        graph = self._graph(agent_id)
        vec   = self._vector(agent_id)

        # Collect candidate entities
        entities = self.aliases.extract_entities(query_text)
        for node in graph.graph.nodes():
            if node.lower().replace("_", " ") in query_text.lower():
                entities.append(node)
        entities = list(set(entities))

        hints = extract_predicate_hints(query_text)

        # ── Direct graph lookup ────────────────────────────────────────
        for entity in entities:
            for pred in hints:
                result = graph.query_direct(entity, pred)
                if result:
                    result.latency_ms = (time.perf_counter() - t0) * 1000
                    return result

        # ── Join (two-hop) ─────────────────────────────────────────────
        join = graph.query_join(query_text)
        if join and join.confidence > 0.4:
            join.latency_ms = (time.perf_counter() - t0) * 1000
            return join

        # ── Vector fallback ────────────────────────────────────────────
        chunks = vec.retrieve(query_text, top_k=3)
        if chunks:
            top_text, top_score, _ = chunks[0]
            return QueryResult(
                answer=top_text,
                confidence=top_score * 0.7,
                method="vector",
                tokens_used=max(1, sum(len(c) for c, _, _ in chunks) // 4),
                facts_used=[],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        return QueryResult(
            answer="No memory found.",
            confidence=0.0,
            method="fallback",
            tokens_used=4,
            facts_used=[],
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def get_entity_context(self, agent_id: str, entity: str) -> dict:
        """All facts about an entity — inject as tool_result for s05."""
        return self._graph(agent_id).get_entity_context(entity)

    def register_alias(self, alias: str, canonical: str):
        self.aliases.register(alias, canonical)

    def stats(self, agent_id: str = None) -> dict:
        if agent_id:
            return self._graph(agent_id).stats()
        return {aid: self._graph(aid).stats() for aid in self._graphs}


# ─────────────────────────────────────────────
#  SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("/home/claude/darkclaw", exist_ok=True)
    # Fresh DB for test
    db = "/home/claude/darkclaw/darkclaw_test.db"
    if os.path.exists(db):
        os.remove(db)

    print("╔══════════════════════════════════════╗")
    print("║      DARKCLAW v1.0 — Self-Test       ║")
    print("╚══════════════════════════════════════╝\n")

    engine = DarkclawEngine(db_path=db)

    # Seed Darkclaw-domain facts
    seed_facts = [
        # Single-valued (RUNS_ON, USES, RUNS_IN)
        ("Darkclaw_Stack",   "RUNS_ON",    "T5810_Server",     False),
        ("CT210_Ollama",     "RUNS_ON",    "T5810_Server",     False),
        ("CT212_LiteLLM",    "RUNS_ON",    "T5810_Server",     False),
        ("CT213_Website",    "RUNS_ON",    "T5810_Server",     False),
        ("SHOP_BOT",         "USES",       "Phi3_Mini",        False),
        ("SHOP_BOT",         "RUNS_IN",    "CT131",            False),
        ("Book_Burrow",      "LLC_NUMBER", "1580647",          False),
        ("Prosthetic_Arm",   "HAS_STATUS", "v1.2_in_progress", False),
        # Multi-valued (ROUTES_TO, DEPENDS_ON)
        ("LiteLLM_Gateway",  "ROUTES_TO",  "CT210_Ollama",     True),
        ("LiteLLM_Gateway",  "ROUTES_TO",  "Anthropic_API",    True),
        ("Darkclaw",         "DEPENDS_ON", "NetworkX",         True),
        ("Darkclaw",         "DEPENDS_ON", "ChromaDB",         True),
        ("Book_Burrow",      "USES",       "Ingram_Supplier",  False),
        # Stale-fact test: priority changes
        ("Book_Burrow",      "HAS_PRIORITY", "high",           False),
    ]
    for subj, pred, obj, mv in seed_facts:
        engine.ingest_fact("main", subj, pred, obj, multi_valued=mv)

    # Now supersede the priority
    engine.ingest_fact("main", "Book_Burrow", "HAS_PRIORITY", "critical", multi_valued=False)

    # Distractors
    for i, txt in enumerate([
        "Sounds good, will look into it.",
        "No blockers on my end.",
        "Let's revisit after the weekend.",
        "Schedule is confirmed.",
    ]):
        engine.ingest_turn(Turn(i+100, "main", "agent_b", txt, "DISTRACTOR"))

    # Queries
    tests = [
        # (query, expected_substring_or_list, description)
        ("What does Darkclaw depend on?",                    ["NetworkX", "ChromaDB"],  "multi-valued DEPENDS_ON"),
        ("What does the LiteLLM Gateway route to?",          ["CT210_Ollama", "Anthropic_API"], "multi-valued ROUTES_TO"),
        ("What is Book Burrow's current priority?",          ["critical"],              "stale-fact: priority"),
        ("Which system does CT210 run on?",                  ["T5810_Server"],          "direct RUNS_ON"),
        ("What agent uses Phi3 Mini?",                       ["SHOP_BOT"],              "join via USES"),
        ("What is the LLC number for Book Burrow?",          ["1580647"],               "direct LLC_NUMBER"),
        ("What is the status of the prosthetic arm?",        ["v1.2"],                  "direct HAS_STATUS"),
    ]

    print(f"{'Test':<45} {'Method':<16} {'Tier':<8} {'Conf':<6} {'ms':<7} {'Pass'}")
    print("─" * 95)

    passed = 0
    for query_text, expected, desc in tests:
        result = engine.query(query_text, agent_id="main")
        tier = result._classify_tier()
        ok = any(e.lower() in result.answer.lower() for e in expected)
        mark = "✓" if ok else "✗"
        if ok:
            passed += 1
        print(f"{mark} {desc:<43} {result.method:<16} {tier:<8} {result.confidence:<6.2f} {result.latency_ms:<7.2f}")
        if not ok:
            print(f"  Expected one of {expected} | Got: '{result.answer}'")

    # Stale-fact explicit check
    pr = engine.query("What is the current priority of Book Burrow", "main")
    stale_ok = "critical" in pr.answer.lower() and "high" not in pr.answer.lower()

    print()
    print("── Stale-fact protection ─────────────────────────────────────────")
    print(f"{'✓' if stale_ok else '✗'} HAS_PRIORITY: 'high' → superseded by 'critical' → got: '{pr.answer}'")
    if stale_ok:
        passed += 1

    # Multi-valued check
    lm = engine.query("What does the LiteLLM Gateway route to?", "main")
    multi_ok = "CT210_Ollama" in lm.answer and "Anthropic_API" in lm.answer
    print(f"{'✓' if multi_ok else '✗'} ROUTES_TO both values preserved: '{lm.answer}'")

    print()
    print("── Graph stats ───────────────────────────────────────────────────")
    s = engine.stats("main")
    print(f"  Nodes: {s['nodes']}  |  Active: {s['active_facts']}  |  Stale: {s['stale_facts']}")

    print()
    total = len(tests) + 1
    print(f"Result: {passed}/{total} passed {'✓ All clear' if passed == total else '— some need attention'}")
