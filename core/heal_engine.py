"""
darkclaw/core/heal_engine.py

Self-healing engine. Classifies agent failures and applies repair strategies.

Every failure type has:
  - A detector (how do we know this happened?)
  - A repair strategy (what do we do about it?)
  - A max_attempts limit (when do we give up and escalate?)
  - A teach signal (what does the teach engine learn from this?)

Failure taxonomy (extend freely):
  STALE_CONTEXT    — agent answered from outdated memory
  TOOL_TIMEOUT     — tool call exceeded time limit
  BAD_OUTPUT       — output failed validation (schema, length, content)
  ROUTING_MISS     — wrong model/agent selected for task
  CONTEXT_OVERFLOW — token budget exceeded
  MEMORY_MISS      — query returned nothing, should have hit
  UNKNOWN          — unclassified, goes to human review queue
"""

import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from core.event_bus import bus, Event, EventType, emit, emit_heal


class FailureType(str, Enum):
    STALE_CONTEXT    = "STALE_CONTEXT"
    TOOL_TIMEOUT     = "TOOL_TIMEOUT"
    BAD_OUTPUT       = "BAD_OUTPUT"
    ROUTING_MISS     = "ROUTING_MISS"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    MEMORY_MISS      = "MEMORY_MISS"
    RESOURCE_ERROR   = "RESOURCE_ERROR"   # OS/socket errors: ENOFILE, ECONNREFUSED, etc.
    UNKNOWN          = "UNKNOWN"


class RepairStrategy(str, Enum):
    REFRESH_MEMORY       = "REFRESH_MEMORY"       # re-query Darkclaw, inject fresh context
    RETRY_WITH_BACKOFF   = "RETRY_WITH_BACKOFF"   # exponential backoff retry
    INJECT_CORRECTION    = "INJECT_CORRECTION"    # add corrective prompt, re-run
    REROUTE              = "REROUTE"              # send to different model/agent
    COMPACT_CONTEXT      = "COMPACT_CONTEXT"      # summarize old turns, retry
    FORCE_MEMORY_QUERY   = "FORCE_MEMORY_QUERY"   # bypass cache, hit graph directly
    ESCALATE             = "ESCALATE"             # add to human review queue
    QUARANTINE           = "QUARANTINE"           # isolate agent, spawn replacement


@dataclass
class Failure:
    agent_id: str
    failure_type: FailureType
    error_msg: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    attempt: int = 1
    resolved: bool = False
    resolution: Optional[str] = None


@dataclass
class RepairResult:
    success: bool
    strategy: RepairStrategy
    attempts: int
    output: Any = None
    teach_signal: Optional[dict] = None   # fed to TeachEngine
    notes: str = ""


# ── Failure classifiers ────────────────────────────────────────────────

def classify_failure(error: Exception, context: dict) -> FailureType:
    """
    Map an exception + context to a FailureType.
    Contributors: add new patterns here.
    """
    msg = str(error).lower()

    if "timeout" in msg or "timed out" in msg:
        return FailureType.TOOL_TIMEOUT

    if "token" in msg and ("limit" in msg or "exceed" in msg or "overflow" in msg):
        return FailureType.CONTEXT_OVERFLOW

    if "stale" in msg or context.get("memory_age_seconds", 0) > 3600:
        return FailureType.STALE_CONTEXT

    if context.get("memory_tier") == "RED" and context.get("query_confidence", 1.0) < 0.3:
        return FailureType.MEMORY_MISS

    if context.get("output_validation_failed"):
        return FailureType.BAD_OUTPUT

    if context.get("wrong_model_detected"):
        return FailureType.ROUTING_MISS

    # OS-level and network resource errors — reroute to fallback model rather than escalate
    if isinstance(error, OSError) or \
       "errno" in msg or "too many open files" in msg or \
       "connection refused" in msg or "connection reset" in msg or \
       "broken pipe" in msg or "network" in msg and "error" in msg:
        return FailureType.RESOURCE_ERROR

    # Schema / null violations from data pipeline errors (KDNuggets Workflow 5 patterns)
    if isinstance(error, (KeyError, ValueError)) or \
       "missing column" in msg or "schema" in msg or "keyerror" in msg:
        return FailureType.BAD_OUTPUT
    if "integrityerror" in msg or ("null" in msg and "violation" in msg):
        return FailureType.BAD_OUTPUT
    if isinstance(error, (ImportError, ModuleNotFoundError)) or \
       "no module named" in msg:
        return FailureType.RESOURCE_ERROR

    return FailureType.UNKNOWN


# ── Strategy selector ──────────────────────────────────────────────────

STRATEGY_MAP: Dict[FailureType, List[RepairStrategy]] = {
    FailureType.STALE_CONTEXT:    [RepairStrategy.REFRESH_MEMORY,     RepairStrategy.FORCE_MEMORY_QUERY, RepairStrategy.ESCALATE],
    FailureType.TOOL_TIMEOUT:     [RepairStrategy.RETRY_WITH_BACKOFF, RepairStrategy.REROUTE,            RepairStrategy.ESCALATE],
    FailureType.BAD_OUTPUT:       [RepairStrategy.INJECT_CORRECTION,  RepairStrategy.RETRY_WITH_BACKOFF, RepairStrategy.ESCALATE],
    FailureType.ROUTING_MISS:     [RepairStrategy.REROUTE,            RepairStrategy.INJECT_CORRECTION,  RepairStrategy.ESCALATE],
    FailureType.CONTEXT_OVERFLOW: [RepairStrategy.COMPACT_CONTEXT,    RepairStrategy.REFRESH_MEMORY,     RepairStrategy.ESCALATE],
    FailureType.MEMORY_MISS:      [RepairStrategy.FORCE_MEMORY_QUERY, RepairStrategy.REFRESH_MEMORY,     RepairStrategy.ESCALATE],
    FailureType.RESOURCE_ERROR:   [RepairStrategy.REROUTE,            RepairStrategy.RETRY_WITH_BACKOFF, RepairStrategy.ESCALATE],
    FailureType.UNKNOWN:          [RepairStrategy.RETRY_WITH_BACKOFF, RepairStrategy.ESCALATE,           RepairStrategy.QUARANTINE],
}

MAX_ATTEMPTS = 3

# Failure types that have a known repair path and are worth retrying.
# UNKNOWN is excluded — burning 3 attempts on an unclassified error just
# delays the human review that's actually needed.  Pattern borrowed from
# KDNuggets "parse_pipeline_error → auto_fixable" (Workflow 5).
AUTO_FIXABLE: set = {
    FailureType.STALE_CONTEXT,
    FailureType.TOOL_TIMEOUT,
    FailureType.BAD_OUTPUT,
    FailureType.ROUTING_MISS,
    FailureType.CONTEXT_OVERFLOW,
    FailureType.MEMORY_MISS,
    FailureType.RESOURCE_ERROR,
}


# ── Repair executor ────────────────────────────────────────────────────

class HealEngine:
    """
    Classifies failures and applies repair strategies.

    Usage:
        engine = HealEngine(memory_engine=darkclaw, orchestrator=orc)

        try:
            result = agent.run(task)
        except Exception as e:
            repair = await engine.heal(agent_id, e, context, retry_fn=lambda: agent.run(task))
            if repair.success:
                result = repair.output
    """

    def __init__(self, memory_engine=None, orchestrator=None, breaker=None):
        self.memory   = memory_engine
        self.orc      = orchestrator
        self._history: List[Failure] = []
        self._escalation_queue: List[Failure] = []
        if breaker is None:
            from core.guardrails import breaker as _default_breaker
            breaker = _default_breaker
        self.breaker  = breaker

    async def heal(
        self,
        agent_id: str,
        error: Exception,
        context: dict,
        retry_fn: Callable,
    ) -> RepairResult:
        """
        Main entry point. Classify → select strategy → attempt repair → log.
        """
        failure_type = classify_failure(error, context)
        strategies   = STRATEGY_MAP.get(failure_type, [RepairStrategy.ESCALATE])

        failure = Failure(
            agent_id=agent_id,
            failure_type=failure_type,
            error_msg=str(error),
            context=context,
        )
        self._history.append(failure)

        # Circuit breaker: if this exact failure signature keeps recurring,
        # stop auto-healing and hand it to a human — repeatedly "fixing" the
        # same thing unattended is how the June guardian loop happened.
        breaker_sig = f"heal:{agent_id}:{failure_type.value}"
        if not self.breaker.allows(breaker_sig):
            self._escalation_queue.append(failure)
            emit(EventType.HEAL_FAILED, agent_id,
                 failure_type=failure_type, breaker_open=True,
                 queued_for_review=True)
            return RepairResult(
                success=False,
                strategy=RepairStrategy.ESCALATE,
                attempts=0,
                teach_signal={
                    "failure_type": failure_type,
                    "strategy": None,
                    "attempts": 0,
                    "success": False,
                    "breaker_open": True,
                },
            )

        # Non-auto-fixable failures skip retries and go straight to the human
        # queue — no point burning attempts on errors we can't repair in code.
        if failure_type not in AUTO_FIXABLE:
            self._escalation_queue.append(failure)
            emit(EventType.HEAL_FAILED, agent_id,
                 failure_type=failure_type, auto_fixable=False,
                 queued_for_review=True)
            return RepairResult(
                success=False,
                strategy=RepairStrategy.ESCALATE,
                attempts=0,
                teach_signal={
                    "failure_type": failure_type,
                    "strategy": None,
                    "attempts": 0,
                    "success": False,
                    "auto_fixable": False,
                },
            )

        emit(EventType.HEAL_TRIGGERED, agent_id,
             failure_type=failure_type,
             error=str(error)[:200],
             strategies=[s.value for s in strategies])

        prev_err_sig = None
        for attempt, strategy in enumerate(strategies[:MAX_ATTEMPTS], 1):
            failure.attempt = attempt
            emit(EventType.HEAL_ATTEMPT, agent_id,
                 strategy=strategy, attempt=attempt)

            try:
                result = await self._apply_strategy(
                    strategy, agent_id, failure, context, retry_fn
                )
                if result.success:
                    failure.resolved  = True
                    failure.resolution = strategy.value
                    self.breaker.record_success(breaker_sig)
                    emit(EventType.HEAL_SUCCESS, agent_id,
                         strategy=strategy, attempts=attempt,
                         output_preview=str(result.output)[:100])
                    result.teach_signal = {
                        "failure_type": failure_type,
                        "strategy":     strategy,
                        "attempts":     attempt,
                        "success":      True,
                    }
                    return result

            except Exception as repair_err:
                emit(EventType.SYSTEM_ERROR, agent_id,
                     msg=f"Repair attempt {attempt} failed: {repair_err}")
                # Diminishing returns: two identical errors in a row means
                # further strategies are burning time, not making progress.
                err_sig = str(repair_err)[:120]
                if err_sig == prev_err_sig:
                    emit(EventType.HEAL_FAILED, agent_id,
                         failure_type=failure_type,
                         diminishing_returns=True, attempts=attempt)
                    break
                prev_err_sig = err_sig
                continue

        # All strategies exhausted
        self.breaker.record_failure(breaker_sig)
        self._escalation_queue.append(failure)
        emit(EventType.HEAL_FAILED, agent_id,
             failure_type=failure_type,
             queued_for_review=True)

        return RepairResult(
            success=False,
            strategy=RepairStrategy.ESCALATE,
            attempts=len(strategies),
            teach_signal={
                "failure_type": failure_type,
                "strategy": None,
                "attempts": len(strategies),
                "success": False,
            },
        )

    async def _apply_strategy(
        self,
        strategy: RepairStrategy,
        agent_id: str,
        failure: Failure,
        context: dict,
        retry_fn: Callable,
    ) -> RepairResult:

        if strategy == RepairStrategy.RETRY_WITH_BACKOFF:
            delay = 2 ** (failure.attempt - 1)   # 1s, 2s, 4s
            import asyncio; await asyncio.sleep(delay)
            output = await self._run(retry_fn)
            return RepairResult(True, strategy, failure.attempt, output)

        elif strategy == RepairStrategy.REFRESH_MEMORY:
            if self.memory:
                # Force re-query — drop any cached context
                task_text = context.get("task_text", "")
                fresh = self.memory.query(task_text, agent_id)
                emit(EventType.MEMORY_QUERY, agent_id,
                     query=task_text[:80], tier=fresh._classify_tier())
                context["injected_memory"] = fresh.to_tool_result()
            output = await self._run(retry_fn)
            return RepairResult(True, strategy, failure.attempt, output)

        elif strategy == RepairStrategy.FORCE_MEMORY_QUERY:
            if self.memory:
                task_text = context.get("task_text", "")
                fresh = self.memory.query(task_text, agent_id)
                context["injected_memory"] = fresh.to_tool_result()
                context["bypass_cache"] = True
            output = await self._run(retry_fn)
            return RepairResult(True, strategy, failure.attempt, output)

        elif strategy == RepairStrategy.INJECT_CORRECTION:
            correction = self._build_correction_prompt(failure)
            context["correction_prompt"] = correction
            output = await self._run(retry_fn)
            return RepairResult(True, strategy, failure.attempt, output)

        elif strategy == RepairStrategy.COMPACT_CONTEXT:
            context["force_compact"] = True
            output = await self._run(retry_fn)
            return RepairResult(True, strategy, failure.attempt, output)

        elif strategy == RepairStrategy.REROUTE:
            # Swap to fallback model
            fallback = context.get("fallback_model", "claude-haiku-4-5")
            context["model_override"] = fallback
            output = await self._run(retry_fn)
            return RepairResult(True, strategy, failure.attempt, output)

        elif strategy == RepairStrategy.ESCALATE:
            return RepairResult(False, strategy, failure.attempt,
                                notes="Added to human review queue.")

        elif strategy == RepairStrategy.QUARANTINE:
            # Guardian will handle agent replacement
            emit(EventType.HEALTH_FAIL, agent_id,
                 reason="quarantined after repeated heal failures")
            return RepairResult(False, strategy, failure.attempt,
                                notes="Agent quarantined.")

        raise ValueError(f"Unknown strategy: {strategy}")

    def _build_correction_prompt(self, failure: Failure) -> str:
        prompts = {
            FailureType.BAD_OUTPUT: (
                "Your previous output did not meet the required format. "
                "Please respond with valid JSON matching the expected schema. "
                "Do not include markdown code fences."
            ),
            FailureType.STALE_CONTEXT: (
                "The information you used may be outdated. "
                "Fresh context has been injected above. "
                "Please re-evaluate using only the updated information."
            ),
            FailureType.ROUTING_MISS: (
                "This task was routed incorrectly. "
                "Focus only on: " + failure.context.get("task_text", "the assigned task.")
            ),
        }
        return prompts.get(failure.failure_type, "Please retry the previous task carefully.")

    @staticmethod
    async def _run(fn: Callable) -> Any:
        import inspect
        if inspect.iscoroutinefunction(fn):
            return await fn()
        result = fn()
        # retry_fn is often a lambda returning a coroutine (e.g. lambda: agent.run(...))
        if inspect.isawaitable(result):
            return await result
        return result

    def escalate_external(self, source: str, error_msg: str,
                          context: dict = None) -> Failure:
        """
        File an escalation reported from outside the agent loop (the
        sentinel daemon, future watchers). Lands in the same human review
        queue and Escalate tab as agent-loop failures.
        """
        failure = Failure(
            agent_id=source,
            failure_type=FailureType.UNKNOWN,
            error_msg=error_msg,
            context=context or {},
        )
        self._history.append(failure)
        self._escalation_queue.append(failure)
        emit(EventType.HEAL_FAILED, source,
             failure_type=FailureType.UNKNOWN, external=True,
             queued_for_review=True, error=error_msg[:200])
        return failure

    def escalation_queue(self) -> List[Failure]:
        return list(self._escalation_queue)

    def clear_escalation(self, failure_id: int):
        if 0 <= failure_id < len(self._escalation_queue):
            self._escalation_queue.pop(failure_id)

    def stats(self) -> dict:
        resolved   = sum(1 for f in self._history if f.resolved)
        by_type    = {}
        for f in self._history:
            by_type[f.failure_type] = by_type.get(f.failure_type, 0) + 1
        return {
            "total_failures": len(self._history),
            "resolved":       resolved,
            "unresolved":     len(self._history) - resolved,
            "escalated":      len(self._escalation_queue),
            "by_type":        by_type,
            "heal_rate":      round(resolved / max(1, len(self._history)), 3),
        }
