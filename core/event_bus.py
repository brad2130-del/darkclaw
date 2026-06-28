"""
darkclaw/core/event_bus.py

Internal pub/sub event bus.
Every agent action, memory write, heal attempt, and teach cycle
publishes here. The UI WebSocket server subscribes and streams
events to the browser in real time.

Design: simple asyncio-based, no external broker needed.
For distributed deployments: swap _subscribers with Redis pub/sub.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional
from enum import Enum


class EventType(str, Enum):
    # Agent lifecycle
    AGENT_STARTED     = "agent.started"
    AGENT_STOPPED     = "agent.stopped"
    AGENT_TASK_START  = "agent.task.start"
    AGENT_TASK_DONE   = "agent.task.done"
    AGENT_TOOL_CALL   = "agent.tool.call"
    AGENT_TOOL_RESULT = "agent.tool.result"

    # Health / Guardian
    HEALTH_OK         = "health.ok"
    HEALTH_WARN       = "health.warn"
    HEALTH_FAIL       = "health.fail"

    # Healing
    HEAL_TRIGGERED    = "heal.triggered"
    HEAL_STRATEGY     = "heal.strategy"
    HEAL_ATTEMPT      = "heal.attempt"
    HEAL_SUCCESS      = "heal.success"
    HEAL_FAILED       = "heal.failed"

    # Memory (Darkclaw)
    MEMORY_INGEST     = "memory.ingest"
    MEMORY_QUERY      = "memory.query"
    MEMORY_HIT        = "memory.hit"
    MEMORY_MISS       = "memory.miss"
    MEMORY_SUPERSEDE  = "memory.supersede"

    # Teaching
    TEACH_EXTRACT     = "teach.extract"
    TEACH_INGEST      = "teach.ingest"
    TEACH_EVAL        = "teach.eval"
    TEACH_WIN         = "teach.win"
    TEACH_QUARANTINE  = "teach.quarantine"

    # System
    SYSTEM_START      = "system.start"
    SYSTEM_STOP       = "system.stop"
    SYSTEM_ERROR      = "system.error"


@dataclass
class Event:
    type: EventType
    agent_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({
            "event_id":  self.event_id,
            "type":      self.type,
            "agent_id":  self.agent_id,
            "data":      self.data,
            "timestamp": self.timestamp,
            "ts_human":  time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
        })

    @property
    def severity(self) -> str:
        if self.type in (EventType.HEALTH_FAIL, EventType.HEAL_FAILED, EventType.SYSTEM_ERROR):
            return "error"
        if self.type in (EventType.HEALTH_WARN, EventType.HEAL_TRIGGERED, EventType.TEACH_QUARANTINE):
            return "warn"
        if self.type in (EventType.HEAL_SUCCESS, EventType.TEACH_WIN):
            return "success"
        return "info"


class EventBus:
    """
    Async pub/sub event bus.

    Publish from synchronous code:
        bus.publish_sync(Event(EventType.AGENT_STARTED, "main", {"model": "devstral"}))

    Subscribe from async code (WebSocket handler):
        async for event in bus.subscribe():
            await ws.send_text(event.to_json())

    Subscribe to specific event types:
        async for event in bus.subscribe(types=[EventType.HEAL_TRIGGERED]):
            ...
    """

    def __init__(self, max_history: int = 500):
        self._queues: List[asyncio.Queue] = []
        self._history: List[Event] = []
        self._max_history = max_history
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stats: Dict[str, int] = {}

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    async def publish(self, event: Event):
        """Publish an event (async)."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        self._stats[event.type] = self._stats.get(event.type, 0) + 1
        for q in self._queues:
            await q.put(event)

    def publish_sync(self, event: Event):
        """Publish from synchronous context."""
        loop = self._get_loop()
        if loop.is_running():
            asyncio.ensure_future(self.publish(event), loop=loop)
        else:
            loop.run_until_complete(self.publish(event))

    async def subscribe(
        self,
        types: Optional[List[EventType]] = None,
        since: Optional[float] = None,
    ):
        """
        Async generator — yields events as they arrive.
        Replays history since `since` timestamp before live events.
        """
        q: asyncio.Queue = asyncio.Queue()
        self._queues.append(q)

        # Replay history for late subscribers (e.g. browser reconnect)
        if since is not None:
            for event in self._history:
                if event.timestamp >= since:
                    if types is None or event.type in types:
                        yield event

        try:
            while True:
                event = await q.get()
                if types is None or event.type in types:
                    yield event
        finally:
            self._queues.remove(q)

    def recent(self, n: int = 50, types: Optional[List[EventType]] = None) -> List[Event]:
        """Return recent events from history (for UI initial load)."""
        events = self._history if types is None else [
            e for e in self._history if e.type in types
        ]
        return events[-n:]

    def stats(self) -> dict:
        return {
            "total_events": sum(self._stats.values()),
            "by_type": dict(self._stats),
            "history_size": len(self._history),
            "subscribers": len(self._queues),
        }


# Singleton — import this everywhere
bus = EventBus()


# ── Convenience helpers ────────────────────────────────────────────────

def emit(event_type: EventType, agent_id: str, **data):
    """One-liner emit for use throughout the codebase."""
    bus.publish_sync(Event(event_type, agent_id, data))


def emit_heal(agent_id: str, failure_type: str, strategy: str, **data):
    emit(EventType.HEAL_TRIGGERED, agent_id,
         failure_type=failure_type, strategy=strategy, **data)


def emit_memory(agent_id: str, fact_type: str, subject: str, predicate: str, obj: str):
    emit(EventType.MEMORY_INGEST, agent_id,
         fact_type=fact_type, subject=subject, predicate=predicate, object=obj)
