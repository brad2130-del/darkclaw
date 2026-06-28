# Self-Healing in Darkclaw

## Failure Taxonomy

| Type | Detector | Repair | Max Attempts |
|---|---|---|---|
| STALE_CONTEXT | memory_age > 3600s OR explicit stale flag | REFRESH_MEMORY → FORCE_MEMORY_QUERY | 3 |
| TOOL_TIMEOUT | exception contains "timeout" | RETRY_WITH_BACKOFF (1s/2s/4s) → REROUTE | 3 |
| BAD_OUTPUT | output_validation_failed in context | INJECT_CORRECTION → RETRY | 3 |
| ROUTING_MISS | wrong_model_detected in context | REROUTE → INJECT_CORRECTION | 3 |
| CONTEXT_OVERFLOW | token limit exception | COMPACT_CONTEXT → REFRESH_MEMORY | 3 |
| MEMORY_MISS | RED tier + conf < 0.3 | FORCE_MEMORY_QUERY → REFRESH_MEMORY | 3 |
| UNKNOWN | catch-all | RETRY_WITH_BACKOFF → ESCALATE → QUARANTINE | 3 |

## Teach Signal

Every repair outcome feeds back:
```python
teach_signal = {
    "failure_type": FailureType.STALE_CONTEXT,
    "strategy":     RepairStrategy.REFRESH_MEMORY,
    "attempts":     1,
    "success":      True,
}
```

TeachEngine ingests this as a fact: `FailureType_STALE_CONTEXT HEALED_BY REFRESH_MEMORY`
Over time, the system learns which strategies work for which failure types in which contexts.
