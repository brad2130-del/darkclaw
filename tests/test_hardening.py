"""
Tests for the hardening stack: circuit breaker, dangerous-command gate,
context budget, query guard — and their integration into the heal engine
and the resilient LLM layer.
"""
import pytest

from core.guardrails import CircuitBreaker, is_dangerous_command, guard_tool_input
from core.context_budget import estimate_tokens, fit_parts, build_system_prompt
from core.query_guard import QueryGuard, GuardRegistry


# ── CircuitBreaker ─────────────────────────────────────────────────────

def test_breaker_opens_after_consecutive_failures():
    b = CircuitBreaker(max_consecutive=3, cooldown_s=9999)
    assert b.allows("x")
    for _ in range(3):
        b.record_failure("x")
    assert not b.allows("x")
    assert b.snapshot()["x"]["open"] is True


def test_breaker_success_resets_consecutive():
    b = CircuitBreaker(max_consecutive=3, cooldown_s=9999)
    b.record_failure("x"); b.record_failure("x")
    b.record_success("x")
    b.record_failure("x"); b.record_failure("x")
    assert b.allows("x")


def test_breaker_half_opens_after_cooldown():
    b = CircuitBreaker(max_consecutive=1, cooldown_s=0.0)
    b.record_failure("x")
    # cooldown elapsed immediately → probe allowed again
    assert b.allows("x")


def test_breaker_total_limit():
    b = CircuitBreaker(max_consecutive=100, max_total=5, cooldown_s=9999)
    for _ in range(5):
        b.record_failure("x")
        b.record_success("x")   # consecutive keeps resetting
    assert not b.allows("x")    # ...but total in window trips it


# ── Dangerous-command gate ─────────────────────────────────────────────

def test_dangerous_command_detection():
    assert is_dangerous_command("python3 -c 'import os'")
    assert is_dangerous_command("  sudo rm -rf /")
    assert is_dangerous_command("bash")
    assert is_dangerous_command("npm run build")
    assert not is_dangerous_command("ls -la")
    assert not is_dangerous_command("git status")
    # prefix must be a whole word: "pythonscript" is not "python"
    assert not is_dangerous_command("pythonscript.bin --help")


def test_guard_tool_input_blocks_and_passes():
    assert guard_tool_input("run_shell", {"command": "python3 evil.py"}) is not None
    assert guard_tool_input("run_shell", {"command": "df -h"}) is None
    assert guard_tool_input("memory_query", {"query": "python books"}) is None


# ── Context budget ─────────────────────────────────────────────────────

def test_fit_parts_drops_lowest_value_first():
    parts = [
        (0, "correction", "fix it"),
        (2, "memory", "m" * 4000),
        (3, "documents", "d" * 4000),
    ]
    kept, dropped = fit_parts(parts, budget=1500)   # ~5250 chars max
    assert dropped == ["documents"]
    assert [label for _, label, _ in kept] == ["correction", "memory"]


def test_fit_parts_truncates_last_survivor():
    kept, dropped = fit_parts([(1, "telemetry", "t" * 10000)], budget=100)
    assert dropped == []
    assert "[...trimmed to fit context]" in kept[0][2]
    assert estimate_tokens(kept[0][2]) <= 120   # small slack for the marker


def test_build_system_prompt_trims_docs_on_small_ctx():
    # llama3.2 profile: num_ctx 2048 → docs (huge) must be dropped,
    # memory + telemetry survive
    context = {
        "injected_memory": {"answer": "Book Burrow LLC is 1580647"},
        "doc_memory": {"answer": "x" * 40000},
        "live_telemetry": "GPU 34C, RAM 8/32GB",
    }
    out = build_system_prompt(context, "what is our LLC number?",
                              "ollama/llama3.2:latest")
    assert "1580647" in out
    assert "GPU 34C" in out
    assert "x" * 100 not in out


def test_build_system_prompt_returns_none_when_empty():
    assert build_system_prompt({}, "hi", "ollama/llama3.2:latest") is None


# ── QueryGuard ─────────────────────────────────────────────────────────

def test_guard_blocks_reentry_and_releases():
    g = QueryGuard()
    gen = g.try_start()
    assert gen is not None
    assert g.try_start() is None          # busy
    assert g.end(gen) is True
    assert g.try_start() is not None      # free again


def test_stale_generation_cannot_end_newer_query():
    g = QueryGuard()
    gen1 = g.try_start()
    g.force_end()                         # cancel invalidates gen1
    gen2 = g.try_start()
    assert g.end(gen1) is False           # stale finalizer: no-op
    assert g.is_active                    # gen2 still running
    assert g.end(gen2) is True


def test_registry_one_guard_per_agent():
    reg = GuardRegistry()
    assert reg.get("rosie") is reg.get("rosie")
    assert reg.get("rosie") is not reg.get("kit")
    reg.get("rosie").try_start()
    assert reg.snapshot() == {"rosie": True, "kit": False}


# ── Integration: breaker inside resilient_completion ──────────────────

def test_llm_call_skips_ladder_when_breaker_open():
    from core.llm_call import resilient_completion
    from tests.test_fleet import make_registry, make_completion

    breaker = CircuitBreaker(max_consecutive=1, cooldown_s=9999)
    breaker.record_failure("llm:ollama/llama3.2:1b")

    reg = make_registry()
    fake = make_completion({})
    resp, info = resilient_completion(
        "ollama/llama3.2:1b", [], api_base="http://pi5:11434",
        completion_fn=fake, registry=reg, breaker=breaker)
    # ladder skipped: pi5 never called, went straight to the fallback chain
    assert all(url != "http://pi5:11434" for _, url in fake.calls)
    assert info["healed"] is True
    assert info["model"] == "ollama/llama3.2:latest"


def test_llm_call_success_feeds_breaker_recovery():
    from core.llm_call import resilient_completion
    from tests.test_fleet import make_registry, make_completion

    breaker = CircuitBreaker(max_consecutive=3, cooldown_s=9999)
    breaker.record_failure("llm:ollama/llama3.2:latest")
    breaker.record_failure("llm:ollama/llama3.2:latest")

    reg = make_registry()
    fake = make_completion({})
    resilient_completion("ollama/llama3.2:latest", [],
                         api_base="http://mem:11434",
                         completion_fn=fake, registry=reg, breaker=breaker)
    # success reset the consecutive count
    assert breaker.allows("llm:ollama/llama3.2:latest")
    breaker.record_failure("llm:ollama/llama3.2:latest")
    assert breaker.allows("llm:ollama/llama3.2:latest")


# ── Integration: breaker + diminishing returns in HealEngine ──────────

@pytest.mark.asyncio
async def test_heal_escalates_immediately_when_breaker_open():
    from core.heal_engine import HealEngine, RepairStrategy

    breaker = CircuitBreaker(max_consecutive=1, cooldown_s=9999)
    breaker.record_failure("heal:rosie:TOOL_TIMEOUT")
    engine = HealEngine(breaker=breaker)

    result = await engine.heal("rosie", TimeoutError("timed out"), {},
                               retry_fn=lambda: "should never run")
    assert result.success is False
    assert result.strategy == RepairStrategy.ESCALATE
    assert result.attempts == 0
    assert result.teach_signal["breaker_open"] is True
    assert len(engine.escalation_queue()) == 1


@pytest.mark.asyncio
async def test_heal_stops_on_identical_repeated_errors():
    from core.heal_engine import HealEngine

    engine = HealEngine(breaker=CircuitBreaker(cooldown_s=9999))
    calls = {"n": 0}

    def retry_fn():
        calls["n"] += 1
        raise RuntimeError("connection refused to 10.0.0.9")

    result = await engine.heal("rosie", TimeoutError("timed out"), {},
                               retry_fn=retry_fn)
    assert result.success is False
    # TOOL_TIMEOUT has 3 strategies, but identical errors on attempts 1+2
    # stop the loop before the 3rd
    assert calls["n"] == 2
    # the exhausted heal fed the breaker
    assert engine.breaker.snapshot()["heal:rosie:TOOL_TIMEOUT"]["consecutive"] == 1
