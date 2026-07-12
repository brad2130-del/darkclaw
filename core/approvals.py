"""
darkclaw/core/approvals.py

Human-in-the-loop approval gate for privileged tool calls.

The denylist model (block known-bad, run everything else) fails on the
command nobody thought of. This is the rook_sync.sh pattern instead —
show the proposed command, wait for a human 'y' — with a third tier so
the common read-only cases don't nag:

    AUTO   — allowlisted read-only commands, run immediately
    ASK    — anything else: parked here until a human decides
    DENY   — guardrails / destructive patterns, never offered at all

Fails closed on every edge: unknown command → ASK, timeout → denied,
server restart with a pending request → denied (the future never resolves
because the process is gone, and the agent gets a refusal string).

The queue is process-local by design. An approval is a live decision about
a command that is about to run *now*; persisting them across restarts would
mean resurrecting a 'yes' whose context is gone.
"""
import asyncio
import re
import shlex
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.event_bus import emit, EventType

# Read-only commands that answer questions without changing anything.
# Matched on the *resolved binary name only* — arguments are what make a
# command dangerous, so anything with a shell metacharacter is kicked to
# ASK regardless of how safe the binary looks.
AUTO_ALLOW = {
    # inspection
    "ls", "pwd", "cat", "head", "tail", "wc", "file", "stat", "find", "tree",
    "grep", "rg", "which", "whereis", "readlink", "realpath", "basename", "dirname",
    # system state
    "uname", "uptime", "date", "whoami", "id", "hostname", "df", "du", "free",
    "ps", "top", "lscpu", "lsblk", "lsusb", "lspci", "lsmod", "dmesg",
    "nvidia-smi", "sensors", "journalctl", "systemctl",
    # dev (read-only usage only — see MUTATING_SUBCOMMANDS)
    "git", "pip", "npm", "echo", "printf", "true",
}
# Deliberately NOT auto-allowed, though they look harmless:
#   pytest / python / node / bash  — these execute arbitrary code from disk.
#   A test file is code. guardrails.DANGEROUS_PREFIXES already refuses the
#   interpreters outright; pytest would have slipped past it wearing a
#   read-only costume.

# Subcommands and flags that make an otherwise-safe binary a mutation.
# `git status` is read-only; `git push` is not. `find` lists files; `find
# -delete` does not — and it needs no shell metacharacter to do it, so the
# metacharacter check alone would have let it through.
MUTATING_SUBCOMMANDS = {
    "git": {"push", "commit", "reset", "rebase", "merge", "checkout", "clean",
            "rm", "mv", "revert", "cherry-pick", "stash", "tag", "remote",
            "config", "apply", "am", "filter-branch", "gc", "prune"},
    "find": {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint",
             "-fprintf", "-fls"},
    "pip": {"install", "uninstall", "download"},
    "npm": {"install", "uninstall", "publish", "run", "exec"},
    "systemctl": {"start", "stop", "restart", "enable", "disable", "mask",
                  "unmask", "edit", "set-property", "kill"},
    "journalctl": {"--vacuum-size", "--vacuum-time", "--rotate", "--flush"},
}

# Shell metacharacters. Their presence means we cannot reason about what
# actually runs (`ls; rm -rf ~`), so the whole line goes to a human.
SHELL_META = re.compile(r"[;&|><`$(){}\[\]\n\\]|\|\||&&")


class Decision:
    AUTO = "auto"
    ASK = "ask"
    DENY = "deny"


def classify(command: str) -> tuple[str, str]:
    """
    Return (decision, reason) for a command string.

    Note this runs *after* guardrails.guard_tool_input has already refused
    interpreter/shell/sudo prefixes and system_tools has refused destructive
    patterns — so DENY here is only for what those miss.
    """
    cmd = (command or "").strip()
    if not cmd:
        return Decision.DENY, "empty command"

    if SHELL_META.search(cmd):
        return Decision.ASK, "contains shell metacharacters — cannot be statically read"

    try:
        parts = shlex.split(cmd)
    except ValueError:
        return Decision.ASK, "unparseable quoting"
    if not parts:
        return Decision.DENY, "empty command"

    binary = parts[0].rsplit("/", 1)[-1]

    if binary not in AUTO_ALLOW:
        return Decision.ASK, f"'{binary}' is not on the read-only allowlist"

    muts = MUTATING_SUBCOMMANDS.get(binary)
    if muts:
        for arg in parts[1:]:
            if arg in muts:
                return Decision.ASK, f"'{binary} {arg}' modifies state"

    # Redirection can't reach here (SHELL_META), so an allowlisted binary
    # with plain args really is read-only.
    return Decision.AUTO, f"'{binary}' is read-only"


@dataclass
class ApprovalRequest:
    request_id: str
    agent_id: str
    command: str
    reason: str
    cwd: str = ""
    created: float = field(default_factory=time.time)
    future: Optional[asyncio.Future] = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "command": self.command,
            "reason": self.reason,
            "cwd": self.cwd,
            "created": self.created,
            "age_s": round(time.time() - self.created, 1),
        }


class ApprovalQueue:
    """Pending human decisions. One per process; the UI drives it."""

    def __init__(self, timeout_s: float = 300.0):
        self.timeout_s = timeout_s
        self._pending: Dict[str, ApprovalRequest] = {}

    def pending(self) -> List[dict]:
        return [r.to_dict() for r in self._pending.values()]

    async def request(self, agent_id: str, command: str, reason: str,
                      cwd: str = "") -> tuple[bool, str]:
        """
        Park a command until a human decides. Returns (granted, note).

        Blocks the calling agent — which is the point. An agent that wants
        to run a command it cannot justify should wait.
        """
        req = ApprovalRequest(
            request_id=uuid.uuid4().hex[:8],
            agent_id=agent_id, command=command, reason=reason, cwd=cwd,
            future=asyncio.get_running_loop().create_future(),
        )
        self._pending[req.request_id] = req
        emit(EventType.APPROVAL_REQUEST, agent_id,
             request_id=req.request_id, command=command[:200], reason=reason, cwd=cwd)

        try:
            granted = await asyncio.wait_for(req.future, timeout=self.timeout_s)
        except asyncio.TimeoutError:
            emit(EventType.APPROVAL_EXPIRED, agent_id,
                 request_id=req.request_id, command=command[:200])
            return False, (f"Denied: no human approved this within "
                           f"{int(self.timeout_s)}s. Commands time out closed.")
        except asyncio.CancelledError:
            return False, "Denied: request cancelled."
        finally:
            self._pending.pop(req.request_id, None)

        if granted:
            return True, "approved"
        return False, "Denied by human operator."

    def resolve(self, request_id: str, granted: bool) -> bool:
        """Approve or deny a pending request. Returns False if unknown/stale."""
        req = self._pending.get(request_id)
        if not req or not req.future or req.future.done():
            return False
        req.future.get_loop().call_soon_threadsafe(req.future.set_result, granted)
        emit(EventType.APPROVAL_GRANTED if granted else EventType.APPROVAL_DENIED,
             req.agent_id, request_id=request_id, command=req.command[:200])
        return True


# Process-wide singleton — system_tools requests, ui/server resolves.
queue = ApprovalQueue()
