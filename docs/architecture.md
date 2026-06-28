# Darkclaw Architecture

## The 12-Layer Harness

Based on analysis of Claude Code v2.1.88 (sanbuphy/learn-coding-agent):

| Layer | Component | Status |
|---|---|---|
| s01 | Agent loop (while-true + tool dispatch) | 🔲 orchestrator.py |
| s02 | Tool registry (buildTool factory) | 🔲 tool_registry.py |
| s03 | Planning (TodoWrite + plan mode) | 🔲 Sprint 2 |
| s04 | Sub-agents (AgentTool + forkSubagent) | 🔲 Sprint 2 |
| s05 | Knowledge on-demand (lazy CLAUDE.md) | ✅ darkclaw_litellm.py |
| s06 | Context compression (autoCompact) | 🔲 Sprint 2 |
| s07 | Persistent tasks (file-based task graph) | ✅ SQLite in darkclaw |
| s08 | Background tasks (DreamTask) | 🔲 Sprint 2 |
| s09 | Event bus | ✅ event_bus.py |
| s10 | Health monitoring | 🔲 guardian.py |
| s11 | Autonomous agents (coordinator mode) | 🔲 Sprint 3 |
| s12 | Workspace isolation | 🔲 Sprint 3 |

## Memory Architecture

See "Vector RAG Isn't Enough" (Towards Data Science, 2026) for the benchmark
that validated this design. Graph outperforms vector-only on join queries 80% vs 20-40%.

```
Query → AliasRegistry.resolve()
     → ContextGraph.query_direct()   [single-hop, conf 1.0]
     → ContextGraph.query_join()     [two-hop, conf 0.4-0.95]
     → VectorMemory.retrieve()       [TF-IDF fallback, conf ×0.7]
     → fallback                      [conf 0.0]
```

Key invariants:
- SINGLE_VALUED predicates: new fact supersedes old (stale-fact protection)
- MULTI_VALUED predicates: new fact adds alongside old (ROUTES_TO, DEPENDS_ON)
- Entity vocabulary resolved at WRITE time (not query time) via AliasRegistry
