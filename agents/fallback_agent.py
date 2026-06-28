"""
darkclaw/agents/fallback_agent.py

Emergency fallback — always uses Claude Haiku 4.5.
Registered as agent_id="fallback" in the orchestrator.
Guardian auto-promotes it when ≥50% of primary agents are in error state.

Never quarantined, never downgraded. Costs API tokens but never goes dark.
"""
import os
import time

from agents.base_agent import BaseAgent, AgentConfig, TaskResult
from core.event_bus import emit, EventType


class FallbackAgent(BaseAgent):
    """Stateless Claude Haiku wrapper. No Ollama dependency."""

    MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, memory=None):
        super().__init__(AgentConfig(
            agent_id="fallback",
            role="fallback",
            model=self.MODEL,
        ))
        self.memory = memory

    async def run(self, task: str, context: dict = None) -> TaskResult:
        context = context or {}
        t0 = time.perf_counter()
        self.status = "running"
        emit(EventType.AGENT_TASK_START, self.agent_id,
             task=task[:80], role="fallback", model=self.MODEL)
        try:
            import litellm

            sys_parts = []

            # Inject memory context
            for key in ("injected_memory", "doc_memory"):
                mem = context.get(key, {})
                if isinstance(mem, dict):
                    ans = mem.get("answer", "")
                    if ans and ans != "No memory found.":
                        sys_parts.append(ans)

            if self.memory:
                qr = self.memory.query(task, self.agent_id)
                ans = qr.to_tool_result().get("answer", "")
                if ans and ans != "No memory found.":
                    sys_parts.insert(0, f"[Darkclaw Memory]\n{ans}")

            messages = []
            if sys_parts:
                messages.append({"role": "system", "content": "\n\n".join(sys_parts)})
            messages.append({"role": "user", "content": task})

            response = litellm.completion(
                model=self.MODEL,
                messages=messages,
                timeout=20,
            )
            output = response.choices[0].message.content or ""

            self._task_count += 1
            self.status = "idle"
            emit(EventType.AGENT_TASK_DONE, self.agent_id, task=task[:40], model=self.MODEL)
            return TaskResult(True, output, self.agent_id,
                              (time.perf_counter() - t0) * 1000)

        except Exception as e:
            self._error_count += 1
            self.status = "error"
            emit(EventType.SYSTEM_ERROR, self.agent_id, msg=str(e)[:160])
            return TaskResult(False, None, self.agent_id,
                              (time.perf_counter() - t0) * 1000, error=str(e))
