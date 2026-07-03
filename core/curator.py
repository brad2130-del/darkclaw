"""
darkclaw/core/curator.py

Memory Curator — the housekeeping SLM loop.

The RAG extractor builds memory up organically but noisily: crude
regex triples, MENTIONED_IN breadcrumbs, and RAG_RESPONSE blobs pile
up forever with nothing deduping, resolving, or expiring them. The
curator is the counterpart process that keeps the store ordered:

  pass 1  expire   — RAG breadcrumbs past their TTL are retrieval
                     debris, not knowledge; supersede them
  pass 2  dedupe   — identical (subject, predicate, object) triples
                     collapse to the newest copy
  pass 3  resolve  — single-valued predicates keep only the newest
                     object; older values are superseded
  pass 4  refine   — an SLM on the memory node rewrites clusters of
                     crude extracted triples into canonical ones
                     (bounded per cycle; skipped if the node is down)

Design rules:
  * soft-delete only — everything is superseded_by="curator:<why>",
    never DELETEd, so any pass is auditable and reversible in SQL
  * deterministic passes never depend on the SLM; a dead memory node
    degrades the curator to passes 1-3, it never breaks it
  * every cycle is idempotent — running twice changes nothing new
  * bounded work per cycle so the loop never monopolizes the node
"""
import asyncio
import json
import os
import re
import time
import urllib.request

from core.event_bus import emit, EventType
from memory.darkclaw_core import SINGLE_VALUED_PREDICATES

MEMORY_NODE_URL = os.environ.get("DARKCLAW_MEMORY_NODE_URL",
                                 "http://192.168.1.130:11434").rstrip("/")
CURATOR_MODEL   = os.environ.get("DARKCLAW_CURATOR_MODEL", "llama3.2:latest")
RAG_TTL_DAYS    = float(os.environ.get("DARKCLAW_RAG_TTL_DAYS", "7"))
INTERVAL_SEC    = int(os.environ.get("DARKCLAW_CURATOR_INTERVAL", "900"))

# Facts whose predicate is retrieval debris rather than knowledge.
_EPHEMERAL_PREDICATES = ("RAG_RESPONSE", "MENTIONED_IN")

# Crude predicates the rag_extractor emits that are worth refining.
_CRUDE_PREDICATES = ("IS",)

# Canonical predicates the SLM is allowed to emit.
_ALLOWED_PREDICATES = sorted(SINGLE_VALUED_PREDICATES | {
    "ROUTES_TO", "DEPENDS_ON", "CONTAINS", "INCLUDES", "CONNECTS_TO",
    "SUPPORTS", "PROVIDES", "MANAGES", "LINKED_TO", "COMPATIBLE_WITH",
    "IS_A", "LOCATED_AT", "PART_OF",
})

_MAX_REFINE_SUBJECTS = 6    # SLM batches per cycle
_MAX_FACTS_PER_SUBJ  = 10


def _norm(s: str) -> str:
    return re.sub(r"[\s_]+", " ", (s or "").strip().lower())


def _default_llm(prompt: str) -> str:
    """Blocking call to the memory-node SLM. Raises on any failure."""
    body = json.dumps({
        "model": CURATOR_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_ctx": 2048},
    }).encode()
    req = urllib.request.Request(
        f"{MEMORY_NODE_URL}/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["message"]["content"]


class Curator:
    """One curator per DarkclawEngine. llm_fn is injectable for tests."""

    def __init__(self, engine, llm_fn=None):
        self.engine = engine
        self.llm_fn = llm_fn or _default_llm

    # ── deterministic passes ────────────────────────────────────────────

    def _supersede(self, cur, fact_ids, reason: str) -> int:
        n = 0
        for fid in fact_ids:
            cur.execute(
                "UPDATE facts SET superseded_by=? "
                "WHERE fact_id=? AND superseded_by IS NULL",
                (f"curator:{reason}", fid))
            n += cur.rowcount
            cur.execute(
                "INSERT INTO audit_log(event_type,fact_id,detail) VALUES(?,?,?)",
                ("CURATE", fid, reason))
        return n

    def _pass_expire(self, cur) -> int:
        cutoff = time.time() - RAG_TTL_DAYS * 86400
        rows = cur.execute(
            "SELECT fact_id FROM facts WHERE superseded_by IS NULL "
            f"AND predicate IN ({','.join('?' * len(_EPHEMERAL_PREDICATES))}) "
            "AND timestamp < ?",
            (*_EPHEMERAL_PREDICATES, cutoff)).fetchall()
        return self._supersede(cur, [r[0] for r in rows], "expired")

    def _pass_dedupe(self, cur) -> int:
        rows = cur.execute(
            "SELECT fact_id, agent_id, subject, predicate, object, timestamp "
            "FROM facts WHERE superseded_by IS NULL "
            "ORDER BY timestamp DESC, fact_id").fetchall()
        seen, losers = set(), []
        for fid, aid, s, p, o, _ts in rows:
            key = (aid, _norm(s), p, _norm(o))
            if key in seen:
                losers.append(fid)
            else:
                seen.add(key)
        return self._supersede(cur, losers, "duplicate")

    def _pass_resolve(self, cur) -> int:
        """Single-valued predicate: newest object wins."""
        preds = tuple(SINGLE_VALUED_PREDICATES)
        rows = cur.execute(
            "SELECT fact_id, agent_id, subject, predicate, timestamp "
            "FROM facts WHERE superseded_by IS NULL "
            f"AND predicate IN ({','.join('?' * len(preds))}) "
            "ORDER BY timestamp DESC, fact_id", preds).fetchall()
        seen, losers = set(), []
        for fid, aid, s, p, _ts in rows:
            key = (aid, _norm(s), p)
            if key in seen:
                losers.append(fid)
            else:
                seen.add(key)
        return self._supersede(cur, losers, "conflict")

    # ── SLM refinement pass ─────────────────────────────────────────────

    def _refine_prompt(self, subject: str, facts: list) -> str:
        lines = "\n".join(f"- {f['object']}" for f in facts)
        preds = ", ".join(_ALLOWED_PREDICATES)
        return (
            "You clean a knowledge graph. Below are raw statements observed "
            f"about the entity \"{subject}\". Rewrite them as clean triples.\n"
            f"Allowed predicates: {preds}\n"
            "Rules: merge redundant statements; drop vague ones; objects are "
            "short noun phrases (max 8 words); at most 4 triples.\n"
            "Answer ONLY with JSON: "
            '{"triples": [{"predicate": "...", "object": "..."}]}\n\n'
            f"Raw statements about {subject}:\n{lines}"
        )

    def _pass_refine(self, cur) -> tuple[int, int]:
        """Returns (facts_superseded, facts_created)."""
        rows = cur.execute(
            "SELECT fact_id, agent_id, subject, object FROM facts "
            "WHERE superseded_by IS NULL "
            f"AND predicate IN ({','.join('?' * len(_CRUDE_PREDICATES))}) "
            "ORDER BY subject", _CRUDE_PREDICATES).fetchall()

        by_subj: dict[tuple, list] = {}
        for fid, aid, s, o in rows:
            by_subj.setdefault((aid, s), []).append(
                {"fact_id": fid, "object": o})
        # Only clusters worth an SLM call
        clusters = [(k, v) for k, v in by_subj.items() if len(v) >= 3]
        clusters = clusters[:_MAX_REFINE_SUBJECTS]

        superseded = created = 0
        for (aid, subj), facts in clusters:
            facts = facts[:_MAX_FACTS_PER_SUBJ]
            try:
                raw = self.llm_fn(self._refine_prompt(subj, facts))
                triples = json.loads(raw).get("triples", [])
            except Exception:
                continue   # node down / bad JSON → leave cluster untouched

            valid = []
            for t in triples[:4]:
                pred = str(t.get("predicate", "")).strip().upper().replace(" ", "_")
                obj  = str(t.get("object", "")).strip()[:80]
                if pred in _ALLOWED_PREDICATES and 2 < len(obj):
                    valid.append((pred, obj))
            if not valid:
                continue

            for pred, obj in valid:
                self.engine.ingest_fact(
                    agent_id=aid, subject=subj, predicate=pred,
                    object_val=obj, speaker="curator",
                    text=f"curated: {subj} {pred} {obj}")
                created += 1
            superseded += self._supersede(
                cur, [f["fact_id"] for f in facts], "refined")
            emit(EventType.TEACH_INGEST, "curator",
                 subject=subj, triples=len(valid), agent=aid)
        return superseded, created

    # ── cycle ───────────────────────────────────────────────────────────

    def run_cycle(self, with_slm: bool = True) -> dict:
        """One full curation cycle. Blocking — call via asyncio.to_thread."""
        t0 = time.perf_counter()
        conn = self.engine.persistence._conn
        cur = conn.cursor()

        report = {
            "expired":  self._pass_expire(cur),
            "deduped":  self._pass_dedupe(cur),
            "resolved": self._pass_resolve(cur),
            "refined":  0,
            "created":  0,
        }
        if with_slm:
            report["refined"], report["created"] = self._pass_refine(cur)
        conn.commit()

        changed = sum(v for k, v in report.items() if k != "created")
        if changed:
            # Superseded facts still live in the in-memory graphs; rebuild
            # from the now-clean DB so queries stop seeing them immediately.
            self.engine._graphs.clear()
            self.engine._vectors.clear()
            self.engine._reload_from_db()
            emit(EventType.MEMORY_SUPERSEDE, "curator", **report)

        report["active_facts"] = cur.execute(
            "SELECT COUNT(*) FROM facts WHERE superseded_by IS NULL"
        ).fetchone()[0]
        report["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return report


async def curator_loop(engine, interval: int = None):
    """Background loop for the server process. Never raises."""
    curator = Curator(engine)
    interval = interval or INTERVAL_SEC
    await asyncio.sleep(60)   # let the server settle before the first pass
    while True:
        try:
            report = await asyncio.to_thread(curator.run_cycle)
            emit(EventType.SYSTEM_START, "curator",
                 msg="curation cycle", **report)
        except Exception:
            pass   # housekeeping must never take the server down
        await asyncio.sleep(interval)
