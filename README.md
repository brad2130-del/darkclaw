# Darkclaw

**A self-healing, self-teaching multi-agent harness with a live UI.**

Darkclaw is a production-grade agent orchestration framework built around two ideas:

1. **Self-healing** — agents monitor each other, detect failures, and attempt repair without human intervention. Every failure is logged, diagnosed, and fed back as a learning signal.
2. **Self-teaching** — a dual-layer memory (context graph + vector store) learns from every interaction. The system gets measurably better at routing, retrieval, and task completion over time.

It ships with a zero-friction UI so you can see what's happening, teach the system new facts, and intervene — without touching config files.

---

## Why this exists

Most agent frameworks give you the loop but not the harness. You get tool dispatch and a memory store, but nothing that handles:

- An agent that silently returns wrong answers because its context is stale
- A routing decision made at 2am that nobody noticed until morning
- A new contributor who wants to understand what the system is doing without reading 10,000 lines of TypeScript

Darkclaw is the harness. It's the 12 layers around the loop.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Darkclaw UI                          │
│   Live topology · Event stream · Teach interface        │
└────────────────────┬────────────────────────────────────┘
                     │ WebSocket
┌────────────────────▼────────────────────────────────────┐
│                 Orchestrator                            │
│   Task graph · Agent registry · Health monitor         │
└──────┬────────────┬───────────────┬─────────────────────┘
       │            │               │
┌──────▼──┐  ┌──────▼──┐  ┌────────▼────────┐
│  Agent  │  │  Agent  │  │  Guardian Agent  │
│  (main) │  │ (worker)│  │  (watchdog)      │
└──────┬──┘  └──────┬──┘  └────────┬─────────┘
       │            │               │
┌──────▼────────────▼───────────────▼─────────┐
│              Darkclaw Memory                 │
│   Context graph (NetworkX) + Vector (TF-IDF) │
│   SQLite persistence · Audit log             │
└──────────────────────────────────────────────┘
```

### The 5 subsystems

| Subsystem | What it does | Key file |
|---|---|---|
| **Orchestrator** | Runs the agent loop, dispatches tools, manages task graph | `core/orchestrator.py` |
| **Guardian** | Monitors agent health, triggers healing routines | `agents/guardian.py` |
| **Darkclaw** | Dual-layer memory: graph for relations, vector for semantics | `memory/darkclaw_core.py` |
| **Heal Engine** | Classifies failures, selects repair strategy, retries | `core/heal_engine.py` |
| **Teach Engine** | Extracts facts from interactions, updates memory, runs evals | `core/teach_engine.py` |

### Self-healing loop

```
Agent fails
    │
    ▼
Guardian detects (health check / error event)
    │
    ▼
HealEngine classifies failure
    ├── STALE_CONTEXT   → force memory refresh, re-query Darkclaw
    ├── TOOL_TIMEOUT    → retry with backoff, escalate if 3x
    ├── BAD_OUTPUT      → inject correction prompt, re-run
    ├── ROUTING_MISS    → update LiteLLM route weights
    └── UNKNOWN         → log + alert, human review queue
    │
    ▼
Retry with repair applied
    │
    ▼
Outcome logged → TeachEngine learns from it
```

### Self-teaching loop

```
Every interaction
    │
    ▼
TeachEngine extracts facts (rule-based or LLM)
    │
    ▼
Darkclaw ingests (graph + vector)
    │
    ▼
Periodic eval: query accuracy benchmark
    │
    ├── Accuracy improved → promote new facts, log win
    └── Accuracy dropped  → quarantine recent facts, alert Guardian
```

---

## Quickstart

```bash
git clone https://github.com/brad2130-del/darkclaw
cd darkclaw

# Isolated environment (reuses system networkx/scikit-learn if present)
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

# Start the server — echo mode, no API key needed
.venv/bin/uvicorn ui.server:app --host 0.0.0.0 --port 7430
```

Then open:

- **http://localhost:7430** → operator dashboard
- **http://localhost:7430/daydream** → Maple Creek

Click **Demo** (or `curl -X POST http://localhost:7430/demo`) and watch the
real system respond in both views at once — memory ingests, a self-heal, and a
teach-win all stream live over the WebSocket.

For full mode with real LLM routing:

```bash
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY and/or OLLAMA_BASE_URL
DARKCLAW_USE_LLM=1 .venv/bin/uvicorn ui.server:app --host 0.0.0.0 --port 7430
```

Run the tests:

```bash
.venv/bin/python -m pytest -q     # 14 passed
```

---

## Project structure

```
darkclaw/
├── core/
│   ├── orchestrator.py      # Main agent loop + task graph
│   ├── heal_engine.py       # Failure classification + repair
│   ├── teach_engine.py      # Fact extraction + memory update
│   ├── tool_registry.py     # Tool definitions + dispatch
│   └── event_bus.py         # Internal pub/sub for UI streaming
│
├── agents/
│   ├── base_agent.py        # Agent interface + lifecycle
│   ├── guardian.py          # Health monitor + watchdog
│   └── worker.py            # Generic task worker agent
│
├── memory/
│   ├── darkclaw_core.py     # Context graph + vector memory engine
│   ├── darkclaw_litellm.py  # LiteLLM middleware shim
│   └── alias_registry.py    # Entity vocabulary resolver
│
├── ui/
│   ├── server.py            # FastAPI + WebSocket server
│   ├── static/
│   │   └── index.html       # Single-file UI (no build step)
│   └── ws_handler.py        # Real-time event streaming
│
├── docs/
│   ├── architecture.md      # Deep dive: how the harness works
│   ├── contributing.md      # How to add agents, tools, memory backends
│   ├── self-healing.md      # Failure taxonomy + repair strategies
│   └── self-teaching.md     # Memory pipeline + eval loop
│
├── tests/
│   ├── test_darkclaw.py     # Memory accuracy benchmark (deterministic)
│   ├── test_heal_engine.py  # Failure injection + repair verification
│   └── test_orchestrator.py # Agent loop integration tests
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── agent_failure.md  # Special template for reporting agent failures
│   └── workflows/
│       └── ci.yml            # Test + lint on every PR
│
├── .env.example
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

---

## Contributing

Darkclaw is designed to be extended. The three most useful things to contribute:

**New healing strategies** — add a class to `core/heal_engine.py` that handles a failure type. See `docs/self-healing.md`.

**New memory backends** — implement the `MemoryBackend` interface in `memory/`. ChromaDB, Neo4j, and Redis backends are all welcome.

**UI panels** — the frontend is a single HTML file with no build step. Add a panel, open a PR.

See `CONTRIBUTING.md` for the full guide.

---

## Inspired by

- [sanbuphy/learn-coding-agent](https://github.com/sanbuphy/learn-coding-agent) — reverse-engineered analysis of Claude Code's 12-layer harness
- *Vector RAG Isn't Enough* (Towards Data Science, 2026) — the benchmark that proved context graphs beat flat vector stores on join queries
- The Darkclaw Neural Stack — Bradley Foster's homelab that runs this in production

---

## License

MIT. Build on it, fork it, ship it.
