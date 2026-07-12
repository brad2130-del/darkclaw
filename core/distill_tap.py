"""
darkclaw/core/distill_tap.py

Distillation tap — captures real Darkclaw traffic as canonical corpus
records (see ~/distillation/data/SCHEMA.md) for the sequence-level
distillation pipeline. This is priority zero for run 2: until this fires,
the student trains on synthetic seeds only.

Fire-and-forget, mirrors core/rag_extractor.schedule_rag: never block or
break the orchestrator loop over a logging write. Appends directly into
the current cycle's consolidated corpus file so judge_filter.py's output
and real traffic share one append-only stream per 60-day beat.
"""
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

CORPUS_DIR = Path(os.environ.get(
    "DISTILL_CORPUS_DIR",
    os.path.expanduser("~/distillation/data/corpus"),
))


def _cycle() -> str:
    return time.strftime("%Y%m%d")


def _record_id(task: str, ts: float) -> str:
    return hashlib.sha256(f"{task}|{ts}".encode()).hexdigest()[:16]


async def _write(record: dict):
    def _append():
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        path = CORPUS_DIR / f"corpus_{_cycle()}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
    await asyncio.to_thread(_append)


def schedule_tap(task: str, answer: str, agent_id: str, *,
                  memory_tier: str = "na",
                  retrieved_context: str | None = None,
                  teacher_model: str | None = None):
    """Fire-and-forget: queue a real-traffic record for the corpus.

    Only call this on a genuinely successful, non-empty result — the tap
    exists to capture what Darkclaw actually got right, not its failures
    (those belong in teacher_edu's correction queue, not here).
    """
    ts = time.time()
    tier_norm = (memory_tier or "").lower()
    record = {
        "id": _record_id(task, ts),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
        "source": "tap",
        "prompt": task,
        "retrieved_context": retrieved_context,
        "answer": answer,
        "reasoning": None,
        "memory_tier": tier_norm if tier_norm in ("green", "yellow", "red") else "na",
        "teacher_model": teacher_model or f"darkclaw/{agent_id}",
        "judge_verdict": None,
        "corrected_by": None,
        "embed_status": "pending",
    }
    try:
        asyncio.create_task(_write(record))
    except RuntimeError:
        # no running loop (e.g. called from sync test code) — write inline
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        path = CORPUS_DIR / f"corpus_{_cycle()}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
