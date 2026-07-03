"""
darkclaw/core/context_budget.py

Budget-enforced system-prompt builder, shared by the worker path and the
UI /stream path (previously two near-identical copies that could drift).

Why: agents run with num_ctx as low as 2048, but memory + doc context +
telemetry are injected unbounded. When the prompt overflows num_ctx,
Ollama silently truncates the HEAD of the prompt — the agent quietly
loses its instructions and no error fires anywhere. This module estimates
tokens and trims the lowest-value context first (Claude Code
micro-compaction pattern: clear old bulk content before touching
anything load-bearing), emitting a HEALTH_WARN so trims are visible.

Priority (lower number = kept longest):
  0  correction prompt   — the heal engine put it there for a reason
  1  live telemetry      — measured numbers; losing it causes hallucination
  2  Darkclaw memory     — useful, re-queryable
  3  document context    — bulkiest, lowest marginal value per token
"""
from core.event_bus import EventType, emit

# chars-per-token heuristic; ~4 for English, biased low (3.5) so we
# overestimate tokens and never trust exactly-at-budget prompts
_CHARS_PER_TOKEN = 3.5

# tokens reserved out of num_ctx for the model's own output + chat framing
OUTPUT_RESERVE = 512

# module counters for /api/hardening
stats = {"prompts_built": 0, "trims": 0, "parts_dropped": 0}


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def fit_parts(parts: list[tuple[int, str, str]], budget: int) -> tuple[list, list]:
    """
    parts: [(priority, label, text)]. Drops highest-priority-number parts
    until the rest fit; if a single remaining part still overflows, its
    tail is truncated. Returns (kept_parts, dropped_labels).
    """
    kept = sorted(parts, key=lambda p: p[0])
    dropped = []
    while kept and sum(estimate_tokens(t) for _, _, t in kept) > budget:
        if len(kept) == 1:
            prio, label, text = kept[0]
            keep_chars = int(budget * _CHARS_PER_TOKEN)
            kept[0] = (prio, label, text[:keep_chars] + "\n[...trimmed to fit context]")
            break
        prio, label, text = kept.pop()   # highest priority number = least valuable
        dropped.append(label)
    return kept, dropped


def build_system_prompt(context: dict, task: str, model: str,
                        agent_id: str = "main") -> str | None:
    """
    Assemble the system prompt from every context source, enforcing the
    model's real context window. Returns None when there is nothing to say.
    """
    parts: list[tuple[int, str, str]] = []

    correction = context.get("correction_prompt")
    if correction:
        parts.append((0, "correction", correction))

    tel = context.get("live_telemetry")
    if tel:
        parts.append((1, "telemetry",
                      "[Live System Telemetry — measured seconds ago; report ONLY "
                      f"these numbers, do not invent metrics]\n{tel}"))

    mem = context.get("injected_memory", {})
    if isinstance(mem, dict):
        ans = mem.get("answer", "")
        if ans and ans != "No memory found.":
            parts.append((2, "memory", f"[Darkclaw Memory]\n{ans}"))

    doc = context.get("doc_memory", {})
    if isinstance(doc, dict):
        doc_ans = doc.get("answer", "")
        if doc_ans and doc_ans != "No memory found.":
            parts.append((3, "documents", f"[Uploaded Document Context]\n{doc_ans}"))

    stats["prompts_built"] += 1
    if not parts:
        return None

    from core.model_router import router
    num_ctx = router.ollama_options(model).get("num_ctx", 4096)
    budget = num_ctx - OUTPUT_RESERVE - estimate_tokens(task)
    if budget < 128:          # pathological task length; keep a floor
        budget = 128

    kept, dropped = fit_parts(parts, budget)
    if dropped:
        stats["trims"] += 1
        stats["parts_dropped"] += len(dropped)
        emit(EventType.HEALTH_WARN, agent_id,
             issue="context over budget — trimmed",
             model=model, num_ctx=num_ctx, dropped=dropped)

    return "\n\n".join(text for _, _, text in kept)
