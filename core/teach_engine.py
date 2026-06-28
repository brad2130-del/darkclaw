"""
darkclaw/core/teach_engine.py

Self-teaching engine. Extracts facts from interactions,
updates Darkclaw memory, and runs periodic accuracy evals.

The teach loop:
  1. Every agent response → extract_facts()
  2. Ingest into Darkclaw
  3. Periodically → run_eval() against known ground truth
  4. Accuracy improved → promote facts, emit TEACH_WIN
  5. Accuracy dropped  → quarantine recent facts, emit TEACH_QUARANTINE
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.event_bus import emit, EventType


# ── Extraction patterns (mirrors darkclaw_litellm.py) ─────────────────

EXTRACTION_PATTERNS = [
    (r"(\w[\w_]+)\s+(?:uses?|is using|utilizes?)\s+([\w_][\w\s_]+?)(?:\s+for|\.|,|$)",
     1, "USES", 2),
    (r"(\w[\w_]+)\s+(?:depends? on|requires?)\s+([\w_][\w\s_]+?)(?:\s+for|\.|,|$)",
     1, "DEPENDS_ON", 2),
    (r"(\w[\w_]+)\s+(?:runs? on|is running on)\s+([\w_][\w\s_]+?)(?:\.|,|$)",
     1, "RUNS_ON", 2),
    (r"(\w[\w_]+)\s+(?:status is|state is|is now|changed to)\s+([\w_][\w\s_]+?)(?:\.|,|$)",
     1, "HAS_STATUS", 2),
    (r"(\w[\w_]+)\s+priority\s+(?:is|set to|changed to)\s+([\w_]+)",
     1, "HAS_PRIORITY", 2),
    (r"(\w[\w_]+)\s+routes? to\s+([\w_][\w\s_]+?)(?:\.|,|$)",
     1, "ROUTES_TO", 2),
    (r"(\w[\w_]+)\s+(?:is assigned to|owned by)\s+([\w_][\w\s_]+?)(?:\.|,|$)",
     1, "ASSIGNED_TO", 2),
    (r"assign\s+([\w_][\w\s_]+?)\s+to\s+([\w_][\w\s_]+?)(?:\.|,|$)",
     1, "ASSIGNED_TO", 2),
    (r"(\w[\w_]+)\s+version\s+(?:is|=|:)\s*([\w\.\-]+)",
     1, "HAS_VERSION", 2),
]


@dataclass
class ExtractedFact:
    subject: str
    predicate: str
    object: str
    source_text: str
    confidence: float = 0.8
    agent_id: str = "teach_engine"
    timestamp: float = field(default_factory=time.time)


@dataclass
class EvalResult:
    accuracy: float
    total_queries: int
    passed: int
    failed: int
    delta: float                      # vs previous eval
    quarantined_facts: List[str]      # fact_ids quarantined
    timestamp: float = field(default_factory=time.time)


class TeachEngine:
    """
    Self-teaching loop.

    Usage:
        teach = TeachEngine(memory_engine=darkclaw)
        teach.ingest_from_text(agent_id, assistant_response)
        eval_result = teach.run_eval(agent_id, ground_truth_pairs)
    """

    def __init__(self, memory_engine=None):
        self.memory = memory_engine
        self._eval_history: List[EvalResult] = []
        self._ingested_count = 0
        self._quarantined: List[str] = []

    def extract_facts(self, text: str, agent_id: str) -> List[ExtractedFact]:
        """Rule-based fact extraction from free text."""
        facts = []
        for pattern, subj_grp, predicate, obj_grp in EXTRACTION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subj = match.group(subj_grp).strip().replace(" ", "_")
                obj  = match.group(obj_grp).strip().replace(" ", "_")
                if len(subj) < 2 or len(obj) < 2:
                    continue
                facts.append(ExtractedFact(
                    subject=subj,
                    predicate=predicate,
                    object=obj,
                    source_text=text[:200],
                    agent_id=agent_id,
                ))
        return facts

    def ingest_from_text(self, agent_id: str, text: str) -> List[ExtractedFact]:
        """Extract facts from text and ingest into Darkclaw."""
        if not self.memory:
            return []

        facts = self.extract_facts(text, agent_id)
        ingested = []
        for fact in facts:
            try:
                self.memory.ingest_fact(
                    agent_id=agent_id,
                    subject=fact.subject,
                    predicate=fact.predicate,
                    object_val=fact.object,
                    speaker="teach_engine",
                    text=fact.source_text,
                )
                ingested.append(fact)
                self._ingested_count += 1
                emit(EventType.TEACH_INGEST, agent_id,
                     subject=fact.subject,
                     predicate=fact.predicate,
                     object=fact.object)
            except Exception as e:
                emit(EventType.SYSTEM_ERROR, agent_id,
                     msg=f"TeachEngine ingest failed: {e}")

        if ingested:
            emit(EventType.TEACH_EXTRACT, agent_id,
                 count=len(ingested),
                 facts=[f"{f.subject} {f.predicate} {f.object}" for f in ingested])

        return ingested

    def ingest_heal_signal(self, agent_id: str, repair_result) -> None:
        """
        Learn from a healing outcome.
        Called by HealEngine after every repair attempt.
        """
        if not repair_result or not repair_result.teach_signal:
            return

        sig = repair_result.teach_signal
        if sig.get("success") and self.memory:
            # Record: this failure_type was solved by this strategy
            self.memory.ingest_fact(
                agent_id="__system__",
                subject=f"FailureType_{sig['failure_type']}",
                predicate="HEALED_BY",
                object_val=str(sig.get("strategy", "unknown")),
                speaker="teach_engine",
            )

        emit(
            EventType.TEACH_INGEST if sig.get("success") else EventType.TEACH_QUARANTINE,
            agent_id,
            **sig,
        )

    def run_eval(
        self,
        agent_id: str,
        ground_truth: List[Tuple[str, str]],   # [(query, expected_substring), ...]
    ) -> EvalResult:
        """
        Run accuracy evaluation against known ground truth.
        Quarantines recent facts if accuracy drops.
        """
        if not self.memory:
            return EvalResult(0.0, 0, 0, 0, 0.0, [])

        passed = 0
        failed_queries = []

        for query, expected in ground_truth:
            result = self.memory.query(query, agent_id)
            if expected.lower() in result.answer.lower():
                passed += 1
            else:
                failed_queries.append((query, expected, result.answer))

        total    = len(ground_truth)
        accuracy = passed / max(1, total)
        prev_acc = self._eval_history[-1].accuracy if self._eval_history else 0.0
        delta    = accuracy - prev_acc

        quarantined = []
        if delta < -0.1 and self._eval_history:
            # Accuracy dropped more than 10% — quarantine recent facts
            quarantined = self._quarantine_recent_facts(agent_id)
            emit(EventType.TEACH_QUARANTINE, agent_id,
                 accuracy=accuracy,
                 delta=delta,
                 quarantined_count=len(quarantined),
                 failed_queries=[q for q, _, _ in failed_queries[:3]])
        elif delta > 0:
            emit(EventType.TEACH_WIN, agent_id,
                 accuracy=accuracy,
                 delta=delta,
                 passed=passed,
                 total=total)

        eval_result = EvalResult(
            accuracy=accuracy,
            total_queries=total,
            passed=passed,
            failed=total - passed,
            delta=delta,
            quarantined_facts=quarantined,
        )
        self._eval_history.append(eval_result)

        emit(EventType.TEACH_EVAL, agent_id,
             accuracy=round(accuracy, 3),
             passed=passed,
             total=total,
             delta=round(delta, 3))

        return eval_result

    def _quarantine_recent_facts(self, agent_id: str, n: int = 5) -> List[str]:
        """
        Mark the n most recent facts as superseded (effectively quarantined).
        This is a conservative rollback — future evals will re-promote if accuracy recovers.
        """
        if not self.memory:
            return []

        graph = self.memory._graph(agent_id)
        recent = sorted(
            graph._facts.values(),
            key=lambda f: f.timestamp,
            reverse=True,
        )[:n]

        quarantined_ids = []
        for fact in recent:
            if fact.superseded_by is None:
                fact.superseded_by = "__quarantined__"
                quarantined_ids.append(fact.fact_id)
                self._quarantined.append(fact.fact_id)

        return quarantined_ids

    def stats(self) -> dict:
        last_eval = self._eval_history[-1] if self._eval_history else None
        return {
            "facts_ingested":    self._ingested_count,
            "evals_run":         len(self._eval_history),
            "quarantined_total": len(self._quarantined),
            "last_accuracy":     last_eval.accuracy if last_eval else None,
            "last_delta":        last_eval.delta    if last_eval else None,
        }
