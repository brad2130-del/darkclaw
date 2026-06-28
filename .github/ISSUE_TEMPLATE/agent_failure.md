---
name: Agent Failure Report
about: An agent failed and couldn't heal itself
labels: agent-failure, needs-triage
---

**Which agent failed?**

**Failure type (if known):**
- [ ] STALE_CONTEXT
- [ ] TOOL_TIMEOUT
- [ ] BAD_OUTPUT
- [ ] ROUTING_MISS
- [ ] CONTEXT_OVERFLOW
- [ ] MEMORY_MISS
- [ ] UNKNOWN

**What was the agent trying to do?**

**Heal attempts (from event stream):**
```
paste heal.triggered / heal.attempt events here
```

**Escalation queue entry:**
```
paste here
```

**Suggested fix / new healing strategy:**
