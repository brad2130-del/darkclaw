"""
darkclaw/agents/guardian.py
Guardian agent — health monitor and watchdog.
Pip in Maple Creek. The hedgehog who never sleeps.

The Guardian polls every registered agent on an interval and:
  - emits HEALTH_OK when an agent recovers or first reports healthy
  - emits HEALTH_WARN when error_rate crosses a threshold
  - emits HEALTH_FAIL when an agent is in an error state or unreachable,
    and asks the HealEngine to step in
  - HEALTH_WARN / HEALTH_FAIL surface in Maple Creek as Pip's "HEY LISTEN"

It only emits HEALTH_OK on a *state change* so the event stream stays quiet
when everything is fine.
"""
import asyncio
import time

from core.event_bus import emit, EventType


class GuardianAgent:
    WARN_ERROR_RATE = 0.3
    # Minimum gap between guardian-triggered heal attempts for the same agent.
    # Without this, a persistently-broken agent generates a heal storm (one
    # heal cycle per watch_loop tick = every 12s = hundreds of retries/hour).
    HEAL_COOLDOWN_S = 120

    def __init__(self, orchestrator=None, heal_engine=None):
        self.orc = orchestrator
        self.healer = heal_engine
        self.watched = {}
        self._last_status = {}  # agent_id -> last reported level string
        self._heal_last:  dict = {}  # agent_id -> monotonic time of last heal trigger
        self._running = False

    def register(self, agent):
        self.watched[agent.agent_id] = agent

    # ── single pass (also handy for tests) ──────────────────────────────

    async def check_once(self):
        for agent_id, agent in list(self.watched.items()):
            try:
                h = agent.health()
            except Exception as e:
                await self._report(agent_id, "fail", reason=f"unreachable: {e}")
                continue

            data = {k: v for k, v in h.items() if k != "agent_id"}
            if h["status"] == "error":
                await self._report(agent_id, "fail", reason="agent in error state", **data)
            elif h["error_rate"] > self.WARN_ERROR_RATE:
                await self._report(agent_id, "warn", **data)
            else:
                await self._report(agent_id, "ok", **data)

    async def _report(self, agent_id, level, **data):
        changed = self._last_status.get(agent_id) != level
        self._last_status[agent_id] = level

        if level == "ok":
            if changed:   # only announce recovery / first-seen-healthy
                emit(EventType.HEALTH_OK, agent_id, **data)
            return

        if level == "warn":
            emit(EventType.HEALTH_WARN, agent_id, **data)
            return

        # level == "fail"
        emit(EventType.HEALTH_FAIL, agent_id, **data)
        if self.healer:
            now = time.monotonic()
            if now - self._heal_last.get(agent_id, 0) < self.HEAL_COOLDOWN_S:
                return  # still in cooldown — don't pile on

            self._heal_last[agent_id] = now

            # Nothing safe to retry on a dead agent — let the healer classify
            # and escalate to the human review queue.
            async def _noop():
                raise RuntimeError(data.get("reason", "agent unreachable"))
            try:
                repair = await self.healer.heal(
                    agent_id, RuntimeError(data.get("reason", "agent failed")),
                    {"task_text": "guardian health check"}, _noop)
                if self.orc and getattr(self.orc, "teacher", None):
                    self.orc.teacher.ingest_heal_signal(agent_id, repair)
            except Exception as e:
                emit(EventType.SYSTEM_ERROR, agent_id, msg=f"guardian heal failed: {e}")

    # ── fallback promotion ──────────────────────────────────────────────

    def _maybe_promote_fallback(self):
        """
        If ≥50% of primary agents are in error state, log and let the
        orchestrator's _route() naturally fall through to 'fallback'.
        The fallback agent is always registered — this just announces it.
        """
        if not self.orc:
            return
        primary = {aid: a for aid, a in self.orc.agents.items()
                   if aid not in ("fallback", "pip")}
        if not primary:
            return
        errored = sum(1 for a in primary.values() if a.status == "error")
        if errored / len(primary) >= 0.5:
            emit(EventType.SYSTEM_START, "pip",
                 msg="fallback-promoted",
                 errored=errored, total=len(primary))

    # ── continuous watch loop ───────────────────────────────────────────

    async def watch_loop(self, interval_s: float = 12.0):
        self._running = True
        emit(EventType.AGENT_STARTED, "pip", role="guardian", model="watchdog")
        while self._running:
            await asyncio.sleep(interval_s)
            await self.check_once()
            self._maybe_promote_fallback()

    def stop(self):
        self._running = False
