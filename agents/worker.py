"""
darkclaw/agents/worker.py
Generic worker agent — does tasks, backed by Darkclaw memory.
Rosie and Kit in Maple Creek.

A worker:
  - receives a task string + context dict
  - uses the Darkclaw memory context injected by the orchestrator
    (or queries memory itself if running standalone)
  - calls an LLM through the darkclaw_litellm middleware *if* litellm is
    installed and DARKCLAW_USE_LLM=1; otherwise it answers from memory
    directly (echo mode) so the whole system runs with zero API keys
  - emits AGENT_TASK_START / MEMORY_QUERY / AGENT_TASK_DONE events
"""
import os
import time

from agents.base_agent import BaseAgent, AgentConfig, TaskResult
from core.event_bus import emit, EventType


def _llm_enabled() -> bool:
    """LLM mode only if explicitly turned on and litellm is importable."""
    if os.environ.get("DARKCLAW_USE_LLM", "0") not in ("1", "true", "True"):
        return False
    try:
        import litellm  # noqa: F401
        return True
    except ImportError:
        return False


class WorkerAgent(BaseAgent):
    """A general task worker. Subclass or instantiate directly."""

    def __init__(self, config: AgentConfig, memory=None, middleware=None):
        super().__init__(config)
        self.memory = memory
        self._middleware = middleware  # optional DarkclawMiddleware (lazy-built)

    async def run(self, task: str, context: dict = None) -> TaskResult:
        context = dict(context or {})
        t0 = time.perf_counter()
        self.status = "running"
        emit(EventType.AGENT_TASK_START, self.agent_id,
             task=task[:80], role=self.config.role)
        try:
            injected = context.get("injected_memory")
            if injected is None and self.memory:
                qr = self.memory.query(task, self.agent_id)
                injected = qr.to_tool_result()
                emit(EventType.MEMORY_QUERY, self.agent_id,
                     query=task[:80], tier=injected["memory_tier"], method=injected["method"])

            output = self._respond(task, context, injected)

            self._task_count += 1
            self.status = "idle"
            emit(EventType.AGENT_TASK_DONE, self.agent_id,
                 task=task[:40], tier=(injected or {}).get("memory_tier"))
            return TaskResult(
                success=True, output=output, agent_id=self.agent_id,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            self._error_count += 1
            self.status = "error"
            self._error_ts = time.time()   # lets _route retry after cooldown
            emit(EventType.SYSTEM_ERROR, self.agent_id, msg=str(e)[:160])
            return TaskResult(
                success=False, output=None, agent_id=self.agent_id,
                duration_ms=(time.perf_counter() - t0) * 1000, error=str(e),
            )

    # ── response generation ─────────────────────────────────────────────

    def _respond(self, task: str, context: dict, injected: dict) -> str:
        model = context.get("model_override", self.config.model)
        correction = context.get("correction_prompt")

        if _llm_enabled():
            return self._respond_llm(task, context, model, correction)

        # ── echo / degraded mode (no API keys needed) ───────────────────
        tier = (injected or {}).get("memory_tier", "RED")
        answer = (injected or {}).get("answer", "")
        if answer and answer != "No memory found.":
            human = answer.replace("_", " ")
            return f"[{self.agent_id}] {human}  (memory: {tier})"
        return (f"[{self.agent_id}] I don't have that in memory yet — "
                f"let's figure it out together. (memory: {tier})")

    def _respond_llm(self, task, context, model, correction) -> str:
        """Call the LLM through the Darkclaw memory middleware."""
        if self._middleware is None:
            from memory.darkclaw_litellm import DarkclawMiddleware
            db = os.environ.get("DARKCLAW_DB",
                                os.path.expanduser("~/darkclaw/data/darkclaw.db"))
            self._middleware = DarkclawMiddleware(db_path=db)

        # Build system prompt from all context sources — via the shared
        # budget-enforced builder so the prompt can never silently overflow
        # num_ctx (Ollama truncates the head, losing instructions first).
        from core.context_budget import build_system_prompt
        if correction:
            context = {**context, "correction_prompt": correction}
        sys_prompt = build_system_prompt(context, task, model,
                                         agent_id=self.agent_id)

        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": task})

        # Build litellm kwargs — Claude models need no Ollama options
        extra = {}
        if model.startswith("ollama/"):
            model_opts = dict(context.get("model_options", {}))
            # _api_base overrides the default Ollama URL (e.g. Pi5 endpoint)
            pi5_url = model_opts.pop("_api_base", None)
            url = pi5_url or os.environ.get("OLLAMA_API_BASE") or os.environ.get("OLLAMA_BASE_URL", "")
            if url:
                extra["api_base"] = url.rstrip("/")
            if model_opts:
                extra["options"] = model_opts

        response, _diff = self._middleware.completion(
            model=model, messages=messages, agent_id=self.agent_id, **extra,
        )
        try:
            return response.choices[0].message.content or ""
        except Exception:
            return str(response)
