"""
tests/test_orchestrator.py
Sprint 1 integration tests — the orchestrator, worker, guardian, and the
self-healing + self-teaching loops working together.

Run: python -m pytest tests/test_orchestrator.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asyncio
import pytest

from core.orchestrator import Orchestrator
from core.event_bus import bus, EventType
from agents.base_agent import BaseAgent, AgentConfig, TaskResult


@pytest.fixture
def orc(tmp_path):
    return Orchestrator.bootstrap(db_path=str(tmp_path / "dc.db"))


def _collect(types=None):
    """Return event types currently in bus history (optionally filtered)."""
    return [e.type for e in bus.recent(500, types)]


@pytest.mark.asyncio
async def test_bootstrap_wires_everything(orc):
    assert set(orc.agents) == {"rosie", "kit", "sage", "bea", "coder", "fallback"}
    assert orc.guardian is not None
    assert set(orc.guardian.watched) == set(orc.agents)
    assert "memory_query" in orc.tools._tools
    assert "memory_teach" in orc.tools._tools
    assert "search_files" in orc.tools._tools
    assert "grep_search" in orc.tools._tools
    # coder gets the registry so its tool loop can dispatch
    assert orc.agents["coder"].tools is orc.tools


@pytest.mark.asyncio
async def test_submit_answers_from_memory(orc):
    orc.memory.ingest_fact("rosie", "Book_Burrow", "LLC_NUMBER", "1580647")
    result = await orc.submit("What is the LLC number for Book Burrow?", agent_id="rosie")
    assert result.success
    assert "1580647" in result.output


@pytest.mark.asyncio
async def test_self_heal_recovers_flaky_task(orc):
    """A worker that fails once then succeeds should be healed, not surfaced as a failure."""

    class FlakyAgent(BaseAgent):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.calls = 0

        async def run(self, task, context=None):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("downstream timed out")
            return TaskResult(success=True, output="recovered", agent_id=self.agent_id,
                              duration_ms=1.0)

    orc.register_agent(FlakyAgent(AgentConfig(agent_id="flaky")))
    result = await orc.submit("do the thing", agent_id="flaky")
    assert result.success
    assert result.output == "recovered"
    await asyncio.sleep(0.05)  # let the fire-and-forget event bus flush
    assert EventType.HEAL_SUCCESS in _collect()


@pytest.mark.asyncio
async def test_guardian_flags_errored_agent(orc):
    class BrokenAgent(BaseAgent):
        async def run(self, task, context=None):
            return TaskResult(False, None, self.agent_id, 1.0, error="boom")

    broken = BrokenAgent(AgentConfig(agent_id="broken"))
    broken.status = "error"
    orc.register_agent(broken)
    await orc.guardian.check_once()
    await asyncio.sleep(0.05)  # let the fire-and-forget event bus flush
    assert EventType.HEALTH_FAIL in _collect()


@pytest.mark.asyncio
async def test_demo_runs_end_to_end(orc):
    await orc.run_demo()
    types = set(_collect())
    # The demo must exercise memory, healing, and teaching for real.
    assert EventType.MEMORY_INGEST in types
    assert EventType.HEAL_SUCCESS in types
    assert EventType.TEACH_WIN in types
