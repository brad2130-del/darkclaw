"""
DARKCLAW v1.0 — LiteLLM Integration Shim
CT212 middleware: intercepts messages, injects memory as tool_results.

Drop-in for Darkclaw Neural Stack:
  - Wraps any LiteLLM completion call
  - Extracts facts from assistant responses (rule-based; LLM extraction optional)
  - Injects relevant entity context as tool_results before sending to model
  - Returns enriched response + memory diff
"""

import sys
import os
import re
import json
import time

# Import works both as a package (memory.darkclaw_litellm) and standalone CLI
# (python memory/darkclaw_litellm.py). Prefer the package path so we don't load
# a second, separate copy of the module when imported by the orchestrator.
try:
    from memory.darkclaw_core import (
        DarkclawEngine, Turn, MULTI_VALUED_PREDICATES, DEFAULT_DB)
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from darkclaw_core import (
        DarkclawEngine, Turn, MULTI_VALUED_PREDICATES, DEFAULT_DB)


# ─────────────────────────────────────────────
#  FACT EXTRACTOR  (rule-based, no LLM needed)
# ─────────────────────────────────────────────

# Patterns: (regex, subject_group, predicate, object_group)
EXTRACTION_PATTERNS = [
    # "X uses Y" / "X is using Y"
    (r"(\w[\w_]+)\s+(?:uses?|is using|utilizes?)\s+([\w_][\w\s_]+?)(?:\s+for|\.|,|$)",
     1, "USES", 2),

    # "X depends on Y" / "X requires Y"
    (r"(\w[\w_]+)\s+(?:depends? on|requires?)\s+([\w_][\w\s_]+?)(?:\s+for|\.|,|$)",
     1, "DEPENDS_ON", 2),

    # "X runs on Y" / "X is running on Y"
    (r"(\w[\w_]+)\s+(?:runs? on|is running on|hosted on)\s+([\w_][\w\s_]+?)(?:\.|,|$)",
     1, "RUNS_ON", 2),

    # "X status is Y" / "X is now Y"
    (r"(\w[\w_]+)\s+(?:status is|state is|is now|changed to)\s+([\w_][\w\s_]+?)(?:\.|,|$)",
     1, "HAS_STATUS", 2),

    # "X priority is Y"
    (r"(\w[\w_]+)\s+priority\s+(?:is|set to|changed to)\s+([\w_]+)",
     1, "HAS_PRIORITY", 2),

    # "X routes to Y"
    (r"(\w[\w_]+)\s+routes? to\s+([\w_][\w\s_]+?)(?:\.|,|$)",
     1, "ROUTES_TO", 2),

    # "X is assigned to Y"
    (r"(\w[\w_]+)\s+(?:is assigned to|assigned to|owned by)\s+([\w_][\w\s_]+?)(?:\.|,|$)",
     1, "ASSIGNED_TO", 2),

    # "assign X to Y" (imperative)
    (r"assign\s+([\w_][\w\s_]+?)\s+to\s+([\w_][\w\s_]+?)(?:\.|,|$)",
     1, "ASSIGNED_TO", 2),
]


def extract_facts_from_text(text: str, agent_id: str, turn_id: int) -> list[Turn]:
    """
    Rule-based fact extraction from free text.
    Returns list of FACT turns ready for DarkclawEngine.ingest_turn().
    
    In production: replace with LLM extraction call via LiteLLM.
    """
    turns = []
    for pattern, subj_grp, predicate, obj_grp in EXTRACTION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            subject = match.group(subj_grp).strip().replace(" ", "_")
            obj     = match.group(obj_grp).strip().replace(" ", "_")
            if len(subject) < 2 or len(obj) < 2:
                continue
            turns.append(Turn(
                turn_id=turn_id,
                agent_id=agent_id,
                speaker="auto_extract",
                text=text[:200],
                turn_type="FACT",
                subject=subject,
                predicate=predicate,
                object=obj,
                multi_valued=predicate in MULTI_VALUED_PREDICATES,
            ))
    return turns


# ─────────────────────────────────────────────
#  DARKCLAW MIDDLEWARE
# ─────────────────────────────────────────────

class DarkclawMiddleware:
    """
    Wraps LiteLLM completion calls with Darkclaw memory injection.

    Usage in CT212 (Darkclaw LiteLLM container):

        from darkclaw_litellm import DarkclawMiddleware
        import litellm

        dc = DarkclawMiddleware()

        # Instead of litellm.completion(...)
        response, memory_diff = dc.completion(
            model="ollama/devstral",
            messages=messages,
            agent_id="main",
        )

    What it does per call:
      1. Scans last user message for entity mentions
      2. Injects relevant graph context as a system memory block (tool_result style)
      3. Calls litellm.completion()
      4. Extracts new facts from assistant response
      5. Ingests new facts into Darkclaw
      6. Returns (response, memory_diff)
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        inject_context: bool = True,
        auto_extract: bool = True,
        verbose: bool = False,
    ):
        self.engine = DarkclawEngine(db_path=db_path)
        self.inject_context = inject_context
        self.auto_extract   = auto_extract
        self.verbose        = verbose
        self._call_counter  = 0

    def completion(
        self,
        model: str,
        messages: list[dict],
        agent_id: str = "main",
        **litellm_kwargs,
    ) -> tuple[any, dict]:
        """
        Drop-in wrapper around litellm.completion().
        Returns (litellm_response, memory_diff).
        """
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm not installed. Run: pip install litellm --break-system-packages"
            )

        self._call_counter += 1
        turn_id_base = self._call_counter * 1000
        memory_diff = {"injected": [], "extracted": [], "tokens_saved": 0}

        # ── 1. Inject memory context ──────────────────────────────────
        if self.inject_context and messages:
            enriched_messages, injected = self._inject_memory(
                messages, agent_id, turn_id_base
            )
            memory_diff["injected"] = injected
        else:
            enriched_messages = messages

        # ── 2. LiteLLM call — via the resilient fleet layer so every
        #    middleware consumer gets node/model failover for free ──────
        from core.llm_call import resilient_completion
        response, serve_info = resilient_completion(
            model=model,
            messages=enriched_messages,
            agent_id=agent_id,
            api_base=litellm_kwargs.pop("api_base", None),
            options=litellm_kwargs.pop("options", None),
            **litellm_kwargs,
        )
        memory_diff["served_by"] = serve_info

        # ── 3. Extract new facts from response ────────────────────────
        if self.auto_extract:
            assistant_text = ""
            try:
                assistant_text = response.choices[0].message.content or ""
            except Exception:
                pass
            if assistant_text:
                new_turns = extract_facts_from_text(
                    assistant_text, agent_id, turn_id_base + 500
                )
                for t in new_turns:
                    fact = self.engine.ingest_turn(t)
                    if fact:
                        memory_diff["extracted"].append({
                            "fact_id": fact.fact_id,
                            "triple": f"{fact.subject} {fact.predicate} {fact.object}",
                        })

        if self.verbose and memory_diff["extracted"]:
            print(f"[Darkclaw] Extracted {len(memory_diff['extracted'])} facts")

        return response, memory_diff

    def query(self, query_text: str, agent_id: str = "main") -> dict:
        """Direct memory query — returns tool_result payload."""
        result = self.engine.query(query_text, agent_id)
        return result.to_tool_result()

    def ingest_fact(self, agent_id: str, subject: str, predicate: str, obj: str, **kw):
        """Direct fact injection."""
        return self.engine.ingest_fact(agent_id, subject, predicate, obj, **kw)

    def stats(self, agent_id: str = None) -> dict:
        return self.engine.stats(agent_id)

    # ── Private ───────────────────────────────────────────────────────

    def _inject_memory(
        self, messages: list[dict], agent_id: str, turn_id_base: int
    ) -> tuple[list[dict], list[dict]]:
        """
        Scan last user message for entity mentions.
        Fetch relevant graph context.
        Prepend as memory block in system message.
        """
        injected = []
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            ""
        )
        if not last_user:
            return messages, injected

        # Find entities mentioned
        entities = self.engine.aliases.extract_entities(last_user)
        graph = self.engine._graph(agent_id)
        for node in graph.graph.nodes():
            if node.lower().replace("_", " ") in last_user.lower():
                entities.append(node)
        entities = list(set(entities))[:5]  # cap at 5 entities per call

        if not entities:
            return messages, injected

        # Build memory block
        memory_lines = ["[DARKCLAW MEMORY — injected context]"]
        for entity in entities:
            ctx = self.engine.get_entity_context(agent_id, entity)
            if ctx["found"] and ctx["facts"]:
                for f in ctx["facts"]:
                    line = f"  {entity} {f['predicate']} {f['value']}"
                    memory_lines.append(line)
                    injected.append({"entity": entity, "fact": line.strip()})

        if len(memory_lines) <= 1:
            return messages, injected

        memory_block = "\n".join(memory_lines)

        # Inject into system message (or prepend one)
        enriched = list(messages)
        if enriched and enriched[0].get("role") == "system":
            enriched[0] = {
                **enriched[0],
                "content": enriched[0]["content"] + "\n\n" + memory_block,
            }
        else:
            enriched.insert(0, {"role": "system", "content": memory_block})

        return enriched, injected


# ─────────────────────────────────────────────
#  STANDALONE QUERY CLI  (useful for testing)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Darkclaw Memory Query CLI")
    parser.add_argument("query", nargs="?", help="Query text")
    parser.add_argument("--agent", default="main", help="Agent ID")
    parser.add_argument("--add-fact", nargs=3, metavar=("SUBJECT","PREDICATE","OBJECT"),
                        help="Ingest a fact: subject predicate object")
    parser.add_argument("--stats", action="store_true", help="Show stats")
    parser.add_argument("--extract", metavar="TEXT", help="Extract facts from text")
    args = parser.parse_args()

    dc = DarkclawMiddleware(verbose=True)

    if args.add_fact:
        subj, pred, obj = args.add_fact
        fact = dc.ingest_fact(args.agent, subj, pred, obj)
        print(f"✓ Ingested: {fact.fact_id} | {subj} {pred} {obj}")

    elif args.extract:
        turns = extract_facts_from_text(args.extract, args.agent, 1)
        print(f"Extracted {len(turns)} facts:")
        for t in turns:
            print(f"  {t.subject} {t.predicate} {t.object}")
            dc.engine.ingest_turn(t)

    elif args.query:
        result = dc.engine.query(args.query, args.agent)
        payload = result.to_tool_result()
        print(json.dumps(payload, indent=2))

    if args.stats:
        print(json.dumps(dc.stats(args.agent), indent=2))
