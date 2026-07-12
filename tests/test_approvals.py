"""
Tests for the human-in-the-loop approval gate.

The property under test is 'fails closed': every ambiguous path must end in
a refusal, never in an execution. A permission system that defaults to yes
when confused is not a permission system.
"""
import asyncio

import pytest

from core.approvals import ApprovalQueue, Decision, classify
from core.system_tools import _run_command, register_system_tools
from core.tool_registry import ToolRegistry


# ── classification ─────────────────────────────────────────────────────

@pytest.mark.parametrize("cmd", [
    "ls -la /home/brad",
    "cat /etc/os-release",
    "nvidia-smi",
    "git status",
    "git log --oneline -5",
    "journalctl --user-unit darkclaw -n 20",
    "df -h",
])
def test_readonly_commands_auto_allow(cmd):
    decision, _ = classify(cmd)
    assert decision == Decision.AUTO


@pytest.mark.parametrize("cmd", [
    "git push origin main",          # allowlisted binary, mutating subcommand
    "git commit -m wip",
    "pip install requests",
    "systemctl restart darkclaw",
    "curl https://example.com",      # not on the allowlist at all
    "ls -la; rm -rf ~/notes",        # metacharacter smuggling
    "cat /etc/passwd > /tmp/leak",   # redirection
    "echo hi && git push",
    "ls `whoami`",                   # command substitution
])
def test_state_changing_commands_ask(cmd):
    decision, _ = classify(cmd)
    assert decision == Decision.ASK, f"{cmd!r} must not auto-run"


def test_unknown_binary_asks_rather_than_allows():
    """Fail closed: a command we've never heard of is not automatically safe."""
    decision, _ = classify("some-tool-we-have-never-seen --do-a-thing")
    assert decision == Decision.ASK


@pytest.mark.parametrize("cmd", [
    "find /home/brad -name '*.tmp' -delete",   # destroys without a metacharacter
    "find . -exec rm {} +",
    "find /var -okdir rm {} ;",
])
def test_find_that_mutates_is_not_auto(cmd):
    """`find` lists files; `find -delete` empties them, and needs no shell syntax."""
    assert classify(cmd)[0] == Decision.ASK


@pytest.mark.parametrize("cmd", ["pytest", "pytest tests/ -q", "python3 evil.py"])
def test_code_execution_never_auto_allowed(cmd):
    """pytest runs whatever is in tests/. That is code execution, not inspection."""
    assert classify(cmd)[0] != Decision.AUTO


def test_empty_command_denied():
    assert classify("")[0] == Decision.DENY
    assert classify("   ")[0] == Decision.DENY


# ── queue mechanics ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_request_blocks_until_approved():
    q = ApprovalQueue(timeout_s=5)
    task = asyncio.create_task(q.request("coder", "git push", "mutating"))
    await asyncio.sleep(0.05)

    pending = q.pending()
    assert len(pending) == 1
    assert pending[0]["command"] == "git push"
    assert not task.done()                    # the agent is genuinely waiting

    assert q.resolve(pending[0]["request_id"], True)
    granted, _ = await task
    assert granted
    assert q.pending() == []                  # and it's cleaned up


@pytest.mark.asyncio
async def test_denial_returns_refusal():
    q = ApprovalQueue(timeout_s=5)
    task = asyncio.create_task(q.request("coder", "rm -rf ~/build", "mutating"))
    await asyncio.sleep(0.05)
    q.resolve(q.pending()[0]["request_id"], False)

    granted, note = await task
    assert not granted
    assert "Denied" in note


@pytest.mark.asyncio
async def test_timeout_fails_closed():
    """An unattended request must expire denied, never approved."""
    q = ApprovalQueue(timeout_s=0.1)
    granted, note = await q.request("coder", "git push", "mutating")
    assert not granted
    assert "time out closed" in note or "Denied" in note
    assert q.pending() == []


def test_resolve_unknown_request_is_false():
    q = ApprovalQueue()
    assert q.resolve("nope", True) is False


# ── end-to-end through the tool ────────────────────────────────────────

@pytest.mark.asyncio
async def test_readonly_command_runs_without_approval(monkeypatch, tmp_path):
    monkeypatch.setenv("DARKCLAW_TERMINAL_POLICY", "ask")
    out = await _run_command({"command": "echo approved-not-needed",
                              "cwd": str(tmp_path)})
    assert "[exit 0]" in out and "approved-not-needed" in out


@pytest.mark.asyncio
async def test_mutating_command_waits_then_runs_when_approved(monkeypatch, tmp_path):
    monkeypatch.setenv("DARKCLAW_TERMINAL_POLICY", "ask")
    import core.approvals as ap
    q = ApprovalQueue(timeout_s=5)
    monkeypatch.setattr(ap, "queue", q)

    marker = tmp_path / "written.txt"
    task = asyncio.create_task(_run_command(
        {"command": f"touch {marker}", "cwd": str(tmp_path)}, agent_id="coder"))
    await asyncio.sleep(0.05)

    assert not marker.exists()               # nothing ran before the human said yes
    assert len(q.pending()) == 1
    q.resolve(q.pending()[0]["request_id"], True)

    out = await task
    assert "[exit 0]" in out
    assert marker.exists()                   # and it ran only after


@pytest.mark.asyncio
async def test_mutating_command_never_runs_when_denied(monkeypatch, tmp_path):
    monkeypatch.setenv("DARKCLAW_TERMINAL_POLICY", "ask")
    import core.approvals as ap
    q = ApprovalQueue(timeout_s=5)
    monkeypatch.setattr(ap, "queue", q)

    marker = tmp_path / "never.txt"
    task = asyncio.create_task(_run_command(
        {"command": f"touch {marker}", "cwd": str(tmp_path)}, agent_id="coder"))
    await asyncio.sleep(0.05)
    q.resolve(q.pending()[0]["request_id"], False)

    out = await task
    assert "Denied" in out
    assert not marker.exists()               # the whole point


@pytest.mark.asyncio
async def test_policy_auto_bypasses_gate(monkeypatch, tmp_path):
    """The escape hatch works — and is opt-in, not the default."""
    monkeypatch.setenv("DARKCLAW_TERMINAL_POLICY", "auto")
    marker = tmp_path / "auto.txt"
    out = await _run_command({"command": f"touch {marker}", "cwd": str(tmp_path)})
    assert "[exit 0]" in out
    assert marker.exists()


@pytest.mark.asyncio
async def test_dispatch_passes_agent_id_to_gated_handler(monkeypatch, tmp_path):
    """The approval must be attributed to the real caller, not a default."""
    monkeypatch.setenv("DARKCLAW_TERMINAL", "1")
    monkeypatch.setenv("DARKCLAW_TERMINAL_POLICY", "ask")
    import core.approvals as ap
    q = ApprovalQueue(timeout_s=5)
    monkeypatch.setattr(ap, "queue", q)

    reg = ToolRegistry()
    register_system_tools(reg)
    task = asyncio.create_task(
        reg.dispatch("sage", "run_command", {"command": "git push", "cwd": str(tmp_path)}))
    await asyncio.sleep(0.05)

    assert q.pending()[0]["agent_id"] == "sage"
    q.resolve(q.pending()[0]["request_id"], False)
    await task
