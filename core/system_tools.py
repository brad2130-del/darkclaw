"""
darkclaw/core/system_tools.py

System-access tools — file search, content grep, and terminal execution.

Gives tool-capable agents (the coder loop first) real eyes and hands on
the host: find files, grep contents, run commands as the darkclaw user.

Safety model, layered:
  1. ToolRegistry.dispatch already gates every input through
     guardrails.guard_tool_input — interpreter/shell/sudo prefixes are
     blocked there before any handler runs.
  2. run_command adds a destructive-pattern refusal (mkfs, dd to a
     device, rm -rf on root paths, power control) for commands that are
     not code-execution escapes but still unrecoverable.
  3. run_command only registers when DARKCLAW_TERMINAL=1. The read-only
     tools (search_files, grep_search) always register. A shop-node
     deploy that never sets the env var never grows a terminal.

Registration: register_system_tools(registry) from Orchestrator.bootstrap.
"""
import asyncio
import fnmatch
import os
import re
import subprocess
from pathlib import Path

HOME = os.path.expanduser("~")

# Directories that make walks slow and results useless.
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".cache",
    ".cargo", ".rustup", ".npm", ".local/share/Trash", "site-packages",
    ".snapshots", "build",
}

MAX_RESULTS = 50          # search_files cap
MAX_MATCHES = 40          # grep_search cap
MAX_WALK = 200_000        # entries visited before a walk gives up
MAX_FILE_BYTES = 2_000_000  # grep skips files larger than this
MAX_OUTPUT = 8_000        # run_command combined stdout+stderr cap
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 120

# Not shell escapes (guardrails handles those) — these are commands that
# destroy data or take the host down and can never be un-run.
DESTRUCTIVE_PATTERNS = (
    re.compile(r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(/|/boot|/etc|/home|/usr|/var|~)\s*(\s|$)"),
    re.compile(r"\bmkfs(\.\w+)?\b"),
    re.compile(r"\bdd\b.*\bof=/dev/"),
    re.compile(r"\b(shutdown|reboot|poweroff|halt)\b"),
    re.compile(r"\bsystemctl\s+(stop|disable|mask)\s"),
    re.compile(r":\s*\(\s*\)\s*{"),          # fork bomb
    re.compile(r"\bchmod\s+(-[a-zA-Z]+\s+)*777\s+/"),
    re.compile(r">\s*/dev/sd[a-z]"),
)


def _is_destructive(cmd: str) -> str | None:
    for pat in DESTRUCTIVE_PATTERNS:
        if pat.search(cmd):
            return (f"Blocked: matches destructive pattern "
                    f"'{pat.pattern}' — needs human approval.")
    return None


def _walk(root: str):
    """os.walk with skip-dirs pruned and a visit budget."""
    visited = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith(".git")]
        for name in filenames:
            visited += 1
            if visited > MAX_WALK:
                return
            yield dirpath, name


# ── handlers ───────────────────────────────────────────────────────────

async def _search_files(inp: dict) -> str:
    pattern = inp.get("pattern", "")
    root = os.path.expanduser(inp.get("root") or HOME)
    if not pattern:
        return "Error: 'pattern' is required (glob, e.g. '*.py' or 'router*')."
    if not os.path.isdir(root):
        return f"Error: root '{root}' is not a directory."

    def run():
        hits = []
        for dirpath, name in _walk(root):
            if fnmatch.fnmatch(name, pattern):
                hits.append(os.path.join(dirpath, name))
                if len(hits) >= MAX_RESULTS:
                    hits.append(f"... capped at {MAX_RESULTS} results")
                    break
        return "\n".join(hits) if hits else f"No files matching '{pattern}' under {root}"
    return await asyncio.to_thread(run)


async def _grep_search(inp: dict) -> str:
    pattern = inp.get("pattern", "")
    root = os.path.expanduser(inp.get("root") or HOME)
    name_glob = inp.get("glob") or "*"
    if not pattern:
        return "Error: 'pattern' is required (regex)."
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"Error: bad regex: {e}"
    if not os.path.isdir(root):
        return f"Error: root '{root}' is not a directory."

    def run():
        hits = []
        for dirpath, name in _walk(root):
            if not fnmatch.fnmatch(name, name_glob):
                continue
            path = os.path.join(dirpath, name)
            try:
                if os.path.getsize(path) > MAX_FILE_BYTES:
                    continue
                with open(path, "r", errors="ignore") as f:
                    for lineno, line in enumerate(f, 1):
                        if rx.search(line):
                            hits.append(f"{path}:{lineno}: {line.strip()[:200]}")
                            if len(hits) >= MAX_MATCHES:
                                hits.append(f"... capped at {MAX_MATCHES} matches")
                                return "\n".join(hits)
            except OSError:
                continue
        return "\n".join(hits) if hits else f"No matches for /{pattern}/ under {root}"
    return await asyncio.to_thread(run)


async def _run_command(inp: dict, agent_id: str = "coder") -> str:
    cmd = inp.get("command", "")
    if not cmd or not cmd.strip():
        return "Error: 'command' is required."
    refusal = _is_destructive(cmd)
    if refusal:
        return refusal
    timeout = min(int(inp.get("timeout") or DEFAULT_TIMEOUT), MAX_TIMEOUT)
    cwd = os.path.expanduser(inp.get("cwd") or HOME)
    if not os.path.isdir(cwd):
        return f"Error: cwd '{cwd}' is not a directory."

    # ── permission gate ────────────────────────────────────────────────
    # Read-only commands run straight through; everything else parks for a
    # human. DARKCLAW_TERMINAL_POLICY=auto skips the gate entirely — for a
    # trusted single-user box only, never a shop node.
    if os.environ.get("DARKCLAW_TERMINAL_POLICY", "ask").lower() != "auto":
        from core.approvals import classify, queue, Decision
        decision, reason = classify(cmd)
        if decision == Decision.DENY:
            return f"Blocked: {reason}."
        if decision == Decision.ASK:
            granted, note = await queue.request(agent_id, cmd, reason, cwd)
            if not granted:
                return note

    def run():
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=cwd, timeout=timeout,
                capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s."
        out = proc.stdout or ""
        err = proc.stderr or ""
        body = out + (("\n[stderr]\n" + err) if err.strip() else "")
        if len(body) > MAX_OUTPUT:
            body = body[:MAX_OUTPUT] + f"\n... truncated at {MAX_OUTPUT} chars"
        return f"[exit {proc.returncode}]\n{body}".strip()
    return await asyncio.to_thread(run)


# ── registration ───────────────────────────────────────────────────────

def register_system_tools(registry):
    """Register file-search + grep always; terminal only if DARKCLAW_TERMINAL=1."""
    registry.register({
        "name": "search_files",
        "description": ("Find files by name glob under a directory. "
                        "Example: pattern='*.service', root='~/.config'."),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Filename glob, e.g. '*.py'"},
                "root": {"type": "string", "description": f"Directory to search (default {HOME})"},
            },
            "required": ["pattern"],
        },
        "handler": _search_files,
    })
    registry.register({
        "name": "grep_search",
        "description": ("Search file contents by regex under a directory. "
                        "Returns path:line: text matches."),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex to search for"},
                "root": {"type": "string", "description": f"Directory to search (default {HOME})"},
                "glob": {"type": "string", "description": "Only files matching this name glob, e.g. '*.py'"},
            },
            "required": ["pattern"],
        },
        "handler": _grep_search,
    })
    if os.environ.get("DARKCLAW_TERMINAL", "0") in ("1", "true", "True"):
        registry.register({
            "name": "run_command",
            "description": ("Run a shell command on the host and return its output. "
                            "Read-only commands (ls, cat, grep, git status, "
                            "nvidia-smi, journalctl...) run immediately. Anything "
                            "that could change state pauses for human approval — "
                            "so prefer a single clear command over a chain, and "
                            "expect a wait. Interpreter/shell/sudo prefixes and "
                            "destructive commands are always refused."),
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run"},
                    "cwd": {"type": "string", "description": f"Working directory (default {HOME})"},
                    "timeout": {"type": "integer", "description": f"Seconds, max {MAX_TIMEOUT} (default {DEFAULT_TIMEOUT})"},
                },
                "required": ["command"],
            },
            "handler": _run_command,
        })
