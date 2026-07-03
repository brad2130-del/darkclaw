"""
darkclaw/core/guardrails.py

Two guardrails ported (as patterns, reimplemented) from Claude Code's
harness analysis:

CircuitBreaker — consecutive/total failure limits per signature, with a
  cooldown half-open. Stops the system re-healing the same failure forever
  (the June-2026 guardian auto-revert loop was exactly this). Shared by:
    - HealEngine:  signature "heal:<agent>:<failure_type>"
    - llm_call:    signature "llm:<model>"  (skip a known-dead ladder)

Dangerous-command gate — interpreter/shell prefixes that turn a "narrow"
  tool permission into arbitrary code execution. Consulted by
  ToolRegistry.dispatch before any handler that executes commands runs.
  Load-bearing the day an agent can touch a shop owner's machine.
"""
import threading
import time


class CircuitBreaker:
    """
    Per-signature failure limits (Claude Code denial-tracking pattern):
      - max_consecutive failures  → open
      - max_total failures inside window_s → open
    Open circuits half-open again cooldown_s after the last failure, so
    recovery is automatic once the underlying cause is fixed.
    """

    def __init__(self, max_consecutive: int = 3, max_total: int = 20,
                 window_s: float = 3600.0, cooldown_s: float = 600.0):
        self.max_consecutive = max_consecutive
        self.max_total = max_total
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self._lock = threading.Lock()
        # sig → {"consecutive": int, "failures": [ts...], "last": ts}
        self._state: dict[str, dict] = {}

    def record_failure(self, sig: str):
        now = time.time()
        with self._lock:
            s = self._state.setdefault(sig, {"consecutive": 0, "failures": [], "last": 0.0})
            s["consecutive"] += 1
            s["failures"] = [t for t in s["failures"] if now - t < self.window_s] + [now]
            s["last"] = now

    def record_success(self, sig: str):
        with self._lock:
            s = self._state.get(sig)
            if s:
                s["consecutive"] = 0

    def allows(self, sig: str) -> bool:
        """False while the circuit is open (limits hit, cooldown not elapsed)."""
        now = time.time()
        with self._lock:
            s = self._state.get(sig)
            if not s:
                return True
            if now - s["last"] > self.cooldown_s:
                # half-open: allow one probe through; failure re-opens instantly
                s["consecutive"] = 0
                return True
            recent = [t for t in s["failures"] if now - t < self.window_s]
            return (s["consecutive"] < self.max_consecutive
                    and len(recent) < self.max_total)

    def snapshot(self) -> dict:
        """Live state for the /api/hardening endpoint."""
        now = time.time()
        out = {}
        with self._lock:
            for sig, s in self._state.items():
                recent = [t for t in s["failures"] if now - t < self.window_s]
                if not recent and s["consecutive"] == 0:
                    continue
                open_ = not (s["consecutive"] < self.max_consecutive
                             and len(recent) < self.max_total) \
                    and now - s["last"] <= self.cooldown_s
                out[sig] = {
                    "consecutive": s["consecutive"],
                    "recent_failures": len(recent),
                    "open": open_,
                    "reopens_in_s": max(0, round(self.cooldown_s - (now - s["last"])))
                    if open_ else 0,
                }
        return out


# ── Dangerous-command gate ─────────────────────────────────────────────

# Command prefixes that grant arbitrary code execution no matter how
# narrow the tool permission looks. An allowlist entry for any of these
# is an allowlist entry for everything.
DANGEROUS_PREFIXES: tuple = (
    # interpreters
    "python", "python3", "python2", "node", "deno", "ruby", "perl",
    "php", "lua",
    # package runners
    "npx", "bunx", "npm run", "yarn run", "pnpm run", "bun run",
    # shells + escape hatches
    "bash", "sh", "zsh", "fish", "eval", "exec", "env", "xargs",
    "sudo", "ssh",
)


def is_dangerous_command(cmd: str) -> bool:
    """True if cmd starts with an arbitrary-code-execution prefix."""
    first = cmd.strip().lower()
    for p in DANGEROUS_PREFIXES:
        if first == p or first.startswith(p + " ") or first.startswith(p + "\t"):
            return True
    return False


def guard_tool_input(tool_name: str, tool_input: dict) -> str | None:
    """
    Returns a refusal string if the tool input must not run, else None.
    Checked in ToolRegistry.dispatch for every tool, so a future shell
    tool can't be added without inheriting the gate.
    """
    for key in ("command", "cmd", "shell", "script"):
        val = tool_input.get(key)
        if isinstance(val, str) and is_dangerous_command(val):
            return (f"Blocked: '{val.split()[0]}' grants arbitrary code "
                    f"execution and requires human approval (tool={tool_name}).")
    return None


# Singletons — one shared failure ledger for the whole process
breaker = CircuitBreaker()
