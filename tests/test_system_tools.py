"""Tests for core/system_tools.py — search, grep, and the gated terminal."""
import os

import pytest

from core.system_tools import (
    _grep_search,
    _is_destructive,
    _run_command,
    _search_files,
    register_system_tools,
)
from core.tool_registry import ToolRegistry


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    return 'darkclaw'\n")
    (tmp_path / "notes.md").write_text("remember the maple creek theme\n")
    return tmp_path


@pytest.mark.asyncio
async def test_search_files_finds_by_glob(tree):
    out = await _search_files({"pattern": "*.py", "root": str(tree)})
    assert str(tree / "src" / "app.py") in out


@pytest.mark.asyncio
async def test_search_files_requires_pattern(tree):
    out = await _search_files({"root": str(tree)})
    assert out.startswith("Error")


@pytest.mark.asyncio
async def test_grep_search_returns_path_line_text(tree):
    out = await _grep_search({"pattern": "maple creek", "root": str(tree)})
    assert "notes.md:1:" in out


@pytest.mark.asyncio
async def test_grep_search_glob_filter(tree):
    out = await _grep_search({"pattern": "darkclaw", "root": str(tree), "glob": "*.md"})
    assert "No matches" in out


@pytest.mark.asyncio
async def test_run_command_executes_and_reports_exit(tree):
    out = await _run_command({"command": "echo hello", "cwd": str(tree)})
    assert "[exit 0]" in out and "hello" in out


@pytest.mark.asyncio
async def test_run_command_blocks_destructive():
    out = await _run_command({"command": "dd if=/dev/zero of=/dev/sda"})
    assert out.startswith("Blocked")


def test_destructive_patterns():
    assert _is_destructive("rm -rf /")
    assert _is_destructive("mkfs.ext4 /dev/sdb1")
    assert _is_destructive("shutdown -h now")
    assert not _is_destructive("rm -rf ./build")
    assert not _is_destructive("ls -la /")


def test_terminal_registration_gated_by_env(monkeypatch):
    monkeypatch.delenv("DARKCLAW_TERMINAL", raising=False)
    reg = ToolRegistry()
    register_system_tools(reg)
    assert "search_files" in reg._tools
    assert "grep_search" in reg._tools
    assert "run_command" not in reg._tools

    monkeypatch.setenv("DARKCLAW_TERMINAL", "1")
    reg2 = ToolRegistry()
    register_system_tools(reg2)
    assert "run_command" in reg2._tools


@pytest.mark.asyncio
async def test_dispatch_gate_blocks_shell_prefix(monkeypatch):
    """The registry-level guardrail must veto interpreter/shell prefixes."""
    monkeypatch.setenv("DARKCLAW_TERMINAL", "1")
    reg = ToolRegistry()
    register_system_tools(reg)
    out = await reg.dispatch("coder", "run_command", {"command": "bash -c 'id'"})
    assert "Blocked" in out
