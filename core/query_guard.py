"""
darkclaw/core/query_guard.py

Per-agent query lifecycle guard (Claude Code QueryGuard pattern).

Prevents two failure modes in /stream that nothing guarded before:
  - re-entry: double-clicking Send runs two generations concurrently
    against the same agent, racing on status and RAG attribution
  - stale cleanup: a cancelled/failed query's finally block clobbering
    the state of the newer query that replaced it

The generation counter is the whole trick: end(gen) is a no-op unless
gen is still current, so stale finalizers can't touch fresh state.
"""
import threading


class QueryGuard:
    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._generation = 0

    def try_start(self) -> int | None:
        """Claim the guard. Returns a generation token, or None if busy."""
        with self._lock:
            if self._running:
                return None
            self._running = True
            self._generation += 1
            return self._generation

    def end(self, generation: int) -> bool:
        """Release iff `generation` is still current (stale ends are no-ops)."""
        with self._lock:
            if not self._running or generation != self._generation:
                return False
            self._running = False
            return True

    def force_end(self):
        """Cancel path: release unconditionally and invalidate stale finalizers."""
        with self._lock:
            self._running = False
            self._generation += 1

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._running


class GuardRegistry:
    """One guard per agent, created on first use."""

    def __init__(self):
        self._lock = threading.Lock()
        self._guards: dict[str, QueryGuard] = {}

    def get(self, agent_id: str) -> QueryGuard:
        with self._lock:
            if agent_id not in self._guards:
                self._guards[agent_id] = QueryGuard()
            return self._guards[agent_id]

    def snapshot(self) -> dict:
        with self._lock:
            return {aid: g.is_active for aid, g in self._guards.items()}


guards = GuardRegistry()
