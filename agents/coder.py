"""
darkclaw/agents/coder.py

CoderAgent — two-model teach loop.

Primary:  deepseek-coder (local, fast, free)
Teacher:  claude-sonnet-4-6 (cloud, authoritative)

Workflow per task:
  1. Call primary model locally
  2. Score confidence of the response
  3. If confidence < threshold OR local memory is thin on this pattern:
       → also call teacher model
       → store teacher answer as ground truth in Darkclaw memory
       → emit TEACH_INGEST so Maple Creek / dashboard show learning
       → return the teacher's (better) answer
  4. If confidence is good AND memory has a strong hit:
       → return the primary answer solo
       → increment solo_streak
  5. Over time solo_streak grows, Claude calls decrease automatically.

The TeachEngine's run_eval() can be pointed at coding ground-truth
pairs to track when the local model "graduates" off needing the teacher.
"""
import asyncio
import hashlib
import json
import os
import time

from agents.base_agent import BaseAgent, AgentConfig, TaskResult
from core.event_bus import emit, EventType


# ── Confidence heuristics ──────────────────────────────────────────────

def _score_confidence(text: str) -> float:
    """Rough quality signal for a coding response. Returns 0.0–1.0."""
    if not text or len(text) < 40:
        return 0.0
    score = 0.4
    if "```" in text:                                   score += 0.25
    if len(text) > 200:                                 score += 0.10
    if any(k in text for k in ("def ", "function ", "class ", "return ", "import ")): score += 0.10
    low = text.lower()
    if any(k in low for k in ("i cannot", "i can't", "i don't know", "i'm sorry", "as an ai")): score -= 0.40
    if any(k in low for k in ("error", "exception", "traceback")):                              score -= 0.10
    return max(0.0, min(1.0, score))


def _task_key(task: str) -> str:
    return "coding_" + hashlib.md5(task[:80].encode()).hexdigest()[:10]


# ── CoderAgent ─────────────────────────────────────────────────────────

class CoderAgent(BaseAgent):
    """
    Local-first coding agent that leans on Claude until it has learned enough.

    AgentConfig fields used:
        model          → primary (local) model   e.g. ollama/deepseek-coder-v2:...
        teacher_model  → fallback/teacher model  e.g. claude-sonnet-4-6
        confidence_threshold  → float, default 0.65
        solo_graduation       → int, default 10  (solo successes before reducing teacher calls)
    """

    CONFIDENCE_THRESHOLD = 0.65
    SOLO_GRADUATION      = 10     # streak needed before skipping teacher on "known" tasks
    MAX_TOOL_ROUNDS      = 6      # tool-call rounds before forcing a text answer

    def __init__(self, config: AgentConfig, memory=None, tool_registry=None):
        super().__init__(config)
        self.memory         = memory
        self.tools          = tool_registry
        self.teacher_model  = getattr(config, "teacher_model", "claude-sonnet-4-6")
        self._middleware    = None   # lazy DarkclawMiddleware
        self._solo_streak   = 0
        self._teach_count   = 0

    async def run(self, task: str, context: dict = None) -> TaskResult:
        context = dict(context or {})
        t0 = time.perf_counter()
        self.status = "running"
        emit(EventType.AGENT_TASK_START, self.agent_id,
             task=task[:80], role=self.config.role, model=self.config.model)

        try:
            output, used_teacher = await self._generate(task, context)
            self._task_count += 1
            self.status = "idle"
            emit(EventType.AGENT_TASK_DONE, self.agent_id,
                 task=task[:40], used_teacher=used_teacher,
                 solo_streak=self._solo_streak, teach_count=self._teach_count)
            return TaskResult(
                success=True, output=output, agent_id=self.agent_id,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            self._error_count += 1
            self.status = "error"
            self._error_ts = time.time()   # lets _route retry after cooldown
            emit(EventType.SYSTEM_ERROR, self.agent_id, msg=str(e)[:200])
            return TaskResult(
                success=False, output=None, agent_id=self.agent_id,
                duration_ms=(time.perf_counter() - t0) * 1000, error=str(e),
            )

    # ── Generation logic ───────────────────────────────────────────────

    async def _generate(self, task: str, context: dict) -> tuple[str, bool]:
        """Returns (output, used_teacher)."""
        mw = self._get_middleware()

        # ── 1. Check memory for a known pattern ────────────────────────
        memory_hit = None
        mem_confidence = 0.0
        if self.memory:
            qr = self.memory.query(task, self.agent_id)
            mem_confidence = {"GREEN": 0.9, "YELLOW": 0.5, "RED": 0.1}.get(
                qr._classify_tier(), 0.0)
            if mem_confidence > 0.7:
                memory_hit = qr.to_tool_result().get("answer", "")

        # ── 2. Call local model (with tools when a registry is wired) ──
        local_output = await self._call_tools(mw, self.config.model, task, context)
        local_conf   = _score_confidence(local_output)

        # ── 3. Decide whether teacher is needed ────────────────────────
        strong_solo = (
            local_conf >= self.CONFIDENCE_THRESHOLD
            and (mem_confidence >= 0.7 or self._solo_streak >= self.SOLO_GRADUATION)
        )

        if strong_solo:
            self._solo_streak += 1
            return local_output, False

        # ── 4. Call teacher ────────────────────────────────────────────
        if not self.teacher_model or not os.environ.get("ANTHROPIC_API_KEY"):
            # No teacher available — return best we have
            return local_output, False

        teacher_output = await self._call_tools(mw, self.teacher_model, task, context)

        # ── 5. Store teaching signal in Darkclaw memory ────────────────
        self._ingest_teach_signal(task, local_output, teacher_output)
        self._solo_streak = 0

        return teacher_output, True

    def _call(self, mw, model: str, task: str, context: dict) -> str:
        messages = []
        if context.get("correction_prompt"):
            messages.append({"role": "system", "content": context["correction_prompt"]})
        messages.append({"role": "user", "content": task})

        extra = {}
        if model.startswith("ollama/"):
            url = os.environ.get("OLLAMA_API_BASE") or os.environ.get("OLLAMA_BASE_URL", "")
            if url:
                extra["api_base"] = url.rstrip("/")

        try:
            resp, _ = mw.completion(model=model, messages=messages,
                                    agent_id=self.agent_id, **extra)
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"[{model} error: {e}]"

    # ── tool-use loop ──────────────────────────────────────────────────

    @staticmethod
    def _extra_kwargs(model: str) -> dict:
        extra = {}
        if model.startswith("ollama"):
            url = os.environ.get("OLLAMA_API_BASE") or os.environ.get("OLLAMA_BASE_URL", "")
            if url:
                extra["api_base"] = url.rstrip("/")
        return extra

    async def _call_tools(self, mw, model: str, task: str, context: dict) -> str:
        """
        Agentic loop: the model may call registered tools (search_files,
        grep_search, run_command, memory_*) any number of rounds before
        answering. Falls back to the plain single-shot _call when no
        registry is wired, no tools are registered, or the provider
        rejects the tools= parameter — the coder must never get worse
        because tools exist.
        """
        schemas = self.tools.get_schemas() if self.tools else []
        if not schemas:
            return await asyncio.to_thread(self._call, mw, model, task, context)

        messages = []
        if context.get("correction_prompt"):
            messages.append({"role": "system", "content": context["correction_prompt"]})
        messages.append({"role": "user", "content": task})
        extra = self._extra_kwargs(model)

        for _round in range(self.MAX_TOOL_ROUNDS):
            try:
                resp, _ = await asyncio.to_thread(
                    mw.completion, model=model, messages=messages,
                    agent_id=self.agent_id, tools=schemas, **extra)
            except Exception:
                # Provider/model without tool support — degrade to plain call
                return await asyncio.to_thread(self._call, mw, model, task, context)

            msg = resp.choices[0].message
            calls = getattr(msg, "tool_calls", None) or []
            if not calls:
                return msg.content or ""

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    c.model_dump() if hasattr(c, "model_dump") else dict(c)
                    for c in calls
                ],
            })
            for c in calls:
                fn = getattr(c, "function", None) or (c.get("function", {}) if isinstance(c, dict) else {})
                name = getattr(fn, "name", None) or (fn.get("name", "") if isinstance(fn, dict) else "")
                raw = getattr(fn, "arguments", None) or (fn.get("arguments", "{}") if isinstance(fn, dict) else "{}")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result = await self.tools.dispatch(self.agent_id, name, args)
                call_id = getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None) or name
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": str(result)[:8000],
                })

        # Rounds exhausted — force a final text answer, tools withheld.
        messages.append({"role": "user",
                         "content": "Answer now using the tool results above."})
        try:
            resp, _ = await asyncio.to_thread(
                mw.completion, model=model, messages=messages,
                agent_id=self.agent_id, **extra)
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"[{model} error: {e}]"

    def _ingest_teach_signal(self, task: str, local: str, teacher: str):
        """Store the teacher's answer as ground truth for this task pattern."""
        if not self.memory:
            return
        key = _task_key(task)
        self.memory.ingest_fact(
            agent_id=self.agent_id,
            subject=key,
            predicate="CLAUDE_SOLUTION",
            object_val=teacher[:300],
            speaker="teach_engine",
            text=task[:200],
        )
        self.memory.ingest_fact(
            agent_id=self.agent_id,
            subject=key,
            predicate="LOCAL_ATTEMPT",
            object_val=local[:200],
            speaker=self.agent_id,
            text=task[:200],
        )
        self._teach_count += 1
        emit(EventType.TEACH_INGEST, self.agent_id,
             subject=key, predicate="CLAUDE_SOLUTION",
             object=teacher[:80], teach_count=self._teach_count)

    def _get_middleware(self):
        if self._middleware is None:
            from memory.darkclaw_litellm import DarkclawMiddleware
            db = os.environ.get("DARKCLAW_DB",
                                os.path.expanduser("~/.config/darkclaw/darkclaw.db"))
            self._middleware = DarkclawMiddleware(db_path=db)
        return self._middleware

    def health(self) -> dict:
        h = super().health()
        h.update({
            "teacher_model":  self.teacher_model,
            "solo_streak":    self._solo_streak,
            "teach_count":    self._teach_count,
            "graduation_at":  self.SOLO_GRADUATION,
        })
        return h
