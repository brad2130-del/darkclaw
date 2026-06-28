"""
darkclaw/agents/base_agent.py
Base agent interface — implement this to add new agents.
See docs/SCOPE.md for the full interface spec.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import time, uuid

@dataclass
class AgentConfig:
    agent_id: str
    model: str = "claude-haiku-4-5"
    role: str = "worker"
    fallback_model: str = "ollama/phi3-mini"
    teacher_model: str = ""          # CoderAgent: Claude assist until local model graduates

@dataclass
class TaskResult:
    success: bool
    output: Any
    agent_id: str
    duration_ms: float
    tokens_used: int = 0
    error: Optional[str] = None

class BaseAgent(ABC):
    """
    Implement this to add a new agent to Darkclaw.
    
    Minimum implementation:
        class MyAgent(BaseAgent):
            async def run(self, task: str, context: dict) -> TaskResult:
                # your logic here
                return TaskResult(success=True, output="done", ...)
    """
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self.status = "idle"
        self._task_count = 0
        self._error_count = 0
        self._start_time = time.time()

    @abstractmethod
    async def run(self, task: str, context: dict) -> TaskResult:
        """Execute a task. Override this."""
        pass

    def health(self) -> dict:
        uptime = time.time() - self._start_time
        return {
            "agent_id":    self.agent_id,
            "status":      self.status,
            "role":        self.config.role,
            "model":       self.config.model,
            "tasks_run":   self._task_count,
            "errors":      self._error_count,
            "error_rate":  round(self._error_count / max(1, self._task_count), 3),
            "uptime_s":    round(uptime, 1),
        }
