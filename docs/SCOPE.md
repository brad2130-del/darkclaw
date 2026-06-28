# Darkclaw — Project Scope & Cowork Handoff

**Version:** 0.1 (Foundation Sprint)
**Author:** Bradley Allen Foster
**Date:** June 2026
**Status:** Active development — foundation built; **Sprint 1 (Make It Run) complete and verified** (14/14 tests passing, live WebSocket demo working)

---

## What Darkclaw Is

Darkclaw is a **self-healing, self-teaching multi-agent AI harness** with two audiences and one shared codebase.

### Audience 1 — Developers / Homelab builders
A production-grade agent orchestration framework that wraps any LLM (local Ollama, Anthropic API, or both via LiteLLM) with:
- A 12-layer agent harness (inspired by Claude Code's architecture)
- Dual-layer memory: **Darkclaw** context graph (NetworkX) + vector (TF-IDF)
- Self-healing loop: Guardian agent detects failures, HealEngine repairs them, TeachEngine learns from every outcome
- Live UI dashboard with event stream, memory query, teach interface
- WebSocket-connected so the UI reflects real running system state

### Audience 2 — Small business owners / everyday people
A friendly companion that sits beside someone opening a business — someone who has no lawyer, no accountant, no technical co-founder — and helps them:
- File correctly (LLC, EIN, licenses, trademarks)
- Track their inventory, suppliers, and contacts
- Understand what they're signing
- Get answers without needing to know how to ask
- Feel supported, not overwhelmed

The **Daydream Mode** (Maple Creek) makes this real for non-technical users: a cozy Zelda-style village screensaver where the AI agents are visible characters going about their work. When the system learns a new fact, Sage the owl opens a treasure chest. When something needs fixing, Pip the hedgehog rings the alarm. The Inn lets anyone leave feedback directly to the creator. No UI jargon. No dashboards. Just a warm little world that shows "the computer is working for you."

---

## The Vision (Why This Matters)

We want to see more people open businesses. Mom-and-pop shops. Local bookstores. Bakeries. Repair shops. The kind of businesses that build neighborhoods.

The people most likely to open those businesses — women, first-generation entrepreneurs, people in rural communities, people without networks — are also the people least likely to have access to lawyers, accountants, and tech people. They have ideas and drive but no support system.

Darkclaw is that support system. It doesn't replace professionals, but it holds your hand until you can afford them, makes sure you don't miss the important steps, and remembers everything so you don't have to.

**Book Burrow** (Bradley and Taylor Foster's bookstore, LLC #1580647, Perryville KY) is the first real-world test case. Everything built here has been validated against a real business being opened in real time.

---

## What's Already Built (Foundation Sprint)

### Memory Layer — Darkclaw v1.0 ✅
`memory/darkclaw_core.py` — **8/8 tests passing**

- NetworkX context graph for relational queries ("what does X depend on?")
- TF-IDF vector layer for semantic/fuzzy queries
- SQLite persistence with full audit log
- Stale-fact protection: `HAS_PRIORITY: high → critical` supersedes correctly
- Multi-valued predicates: `ROUTES_TO` keeps both values, `HAS_STATUS` only keeps the latest
- AliasRegistry resolves vocabulary mismatches at write time ("the bookstore" → Book_Burrow)
- `to_tool_result()` outputs LiteLLM-injectable tool_result payloads

`memory/darkclaw_litellm.py` — LiteLLM middleware shim
- Drop-in wrapper for `litellm.completion()`
- Injects memory context into system prompt before each call
- Auto-extracts facts from assistant responses
- Returns `(response, memory_diff)` with what was injected and what was learned

### Agent Harness — Core Systems ✅
`core/event_bus.py`
- Async pub/sub, typed EventType enum
- History replay for late subscribers (browser reconnect)
- `emit()` helper for one-liner publishing anywhere

`core/heal_engine.py`
- 7 failure types: STALE_CONTEXT, TOOL_TIMEOUT, BAD_OUTPUT, ROUTING_MISS, CONTEXT_OVERFLOW, MEMORY_MISS, UNKNOWN
- Priority-ordered repair strategies per failure type
- Async repair executor with backoff, memory refresh, correction injection, rerouting
- Human escalation queue for unresolvable failures
- Every outcome feeds back to TeachEngine

`core/teach_engine.py`
- Fact extraction from free text (rule-based, LLM-optional)
- Periodic accuracy eval against ground truth
- Automatic quarantine of recent facts when accuracy drops
- Re-promotion on recovery

### UI Layer ✅
`ui/static/index.html` — Live operator dashboard
- Three-column layout: agent health tree, live topology canvas + event stream, memory query + teach interface
- WebSocket connected (port 7430)
- Filters: All / Heal / Memory / Teach / Error
- Memory query panel with graph_direct / graph_join / vector / fallback display
- Teach panel: structured fact input + free-text extraction
- Escalation panel for human review queue
- GREEN / YELLOW / RED tier classification visible on every query result

`daydream/daydream.html` — Maple Creek screensaver
- GBC-era 16×16 pixel sprites (Zelda village palette)
- 6 characters: Rosie (main helper), Pip (hedgehog guardian), Sage (owl/knowledge), Kit (cat shopkeeper), Bea (bee worker), Hero (orchestrator)
- Village map: cobblestone paths, warm stone buildings, terracotta roofs, flower patches, stone well, tree canopy, water
- System events → world behaviors: memory ingest = Sage opens chest, heal triggered = Pip alerts + screen flash, heal success = green particle burst
- HEY LISTEN popup (Pip) for escalations
- Typewriter dialog box (Zelda style) with character names
- Inn feedback form → mailto with pre-filled subject/body
- F key fullscreen screensaver, any key to wake
- WebSocket auto-connect to live system (ws://localhost:7430/ws)
- `window.DarkclawDaydream.handleEvent()` public API

### Sprites (CC0 Original Art) ✅
`daydream/sprites/` — all 8 files
- 6 character sheets: 48×64px each, 3-frame walk × 4 directions
- 1 tileset: 256×64px, 16 tiles (grass×4, stone path, dirt, water×3, tree, wall×2, floor, roof, fence, well)
- 1 item sheet: 128×16px, 8 items (chest×2, heart, rupee×2, star, sparkle, flower)
- All original art, warm Zelda village palette, legally clean

---

## Critical Path — ✅ DONE (Sprint 1)

These are the files that make Darkclaw actually run. All built, wired, and
verified end-to-end (`POST /demo` streams real memory/heal/teach events to both
the dashboard and Maple Creek over the WebSocket).

| Component | File | What It Does | Status |
|---|---|---|---|
| Orchestrator | `core/orchestrator.py` | Outer loop: memory injection → run → self-heal → self-teach; `bootstrap()` wires the whole system; `run_demo()` scenario | ✅ |
| Base Agent | `agents/base_agent.py` | Agent interface, lifecycle, health reporting | ✅ |
| Guardian Agent | `agents/guardian.py` | Health watchdog loop, emits OK/WARN/FAIL, triggers HealEngine | ✅ |
| Worker Agent | `agents/worker.py` | Memory-backed task worker; LLM via middleware or echo mode (no key needed) | ✅ |
| UI Server | `ui/server.py` | FastAPI: serves both UIs, `/ws` event stream, `/teach` `/query` `/submit` `/demo` `/health` | ✅ |
| Tool Registry | `core/tool_registry.py` | Tool dispatch (s02); `memory_query` + `memory_teach` registered by default | ✅ |

**Foundation bugs fixed during Sprint 1 integration:**
- `darkclaw_core.py` inverse-join did a bare `from darkclaw_core import …` that only resolved with `memory/` as cwd — broke when imported as a package. (8th memory test now passes under pytest, not just the self-test harness.)
- `HealEngine._run` never awaited `retry_fn` lambdas that return coroutines — every healed result was a coroutine object instead of the real value. Self-healing now actually returns recovered output.
- `darkclaw_litellm.py` import hardened to avoid loading a second copy of the memory module.

## What's NOT Built Yet (Next Sprint)

### Small Business Features — The Mission-Critical Stuff

| Feature | What It Does | Notes |
|---|---|---|
| Business Setup Wizard | Step-by-step: LLC → EIN → Resale cert → License → Trademark | The thing that actually helps Taylor and people like her |
| Document Checker | Upload a contract/lease/invoice, get plain-English summary + red flags | Huge for people signing things they don't understand |
| Compliance Calendar | Track filing deadlines, renewal dates, quarterly taxes | Prevents the $500 "I forgot to renew" moments |
| Supplier Memory | Track vendors, terms, contacts, reorder points | Ingram integration for Book Burrow |
| SHOP-BOT | Inventory queries, book recommendations, POS integration | Already running on Phi-3 Mini in CT131, needs Darkclaw wired in |

### Daydream v2 — Maple Creek Expansion

| Feature | What It Does |
|---|---|
| Zelda village tileset integration | Use the new zelda/tileset sprites in the map renderer |
| Map scrolling | Camera follows characters, world larger than viewport |
| Day/night cycle | Amber windows glow at night, fireflies, softer palette |
| Seasonal flowers | Spring/summer/fall/winter tile variants |
| Notice board | In-world feedback UI (walk up to the board, press A) |
| NPC dialogue trees | Kit gives business tips, Sage quotes wisdom, Pip gives system status |
| Grant's cameo | Tiny sprite of a kid watching from the fence |

### Infrastructure

| Component | Notes |
|---|---|
| `requirements.txt` | networkx, fastapi, uvicorn, litellm, scikit-learn, sqlite3 |
| `.env.example` | ANTHROPIC_API_KEY, OLLAMA_BASE_URL, LITELLM_PORT, DB_PATH |
| `docker-compose.yml` | Container setup for CT212-style deployment |
| GitHub Actions CI | Test + lint on PR |
| `tests/test_darkclaw.py` | Already designed (8 ground truth pairs), needs pytest wrapper |
| `tests/test_heal_engine.py` | Failure injection + repair verification |

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    DARKCLAW UI (port 7430)                   │
│         Live dashboard  ←→  Maple Creek Daydream            │
└─────────────────────┬────────────────────────────────────────┘
                      │ WebSocket
┌─────────────────────▼────────────────────────────────────────┐
│                   ORCHESTRATOR                               │
│         Task graph · Agent registry · Tool dispatch          │
└──────┬──────────────┬─────────────────┬──────────────────────┘
       │              │                 │
┌──────▼───┐   ┌──────▼───┐   ┌────────▼──────────┐
│  Worker  │   │  Worker  │   │  Guardian Agent   │
│  Agent   │   │  Agent   │   │  (Pip / watchdog) │
│ (Rosie)  │   │ (SHOP-BOT│   └────────┬──────────┘
└──────┬───┘   └──────┬───┘            │
       │              │          ┌──────▼──────┐
       └──────┬────────┘         │ HealEngine  │
              │                 └──────┬──────┘
┌─────────────▼──────────────────────┐│
│          DARKCLAW MEMORY           ││
│  Context Graph (NetworkX)          ││◄── TeachEngine
│  Vector Layer (TF-IDF)             │
│  SQLite persistence                │
│  LiteLLM middleware shim           │
└────────────────────────────────────┘
              │
    ┌─────────┴─────────┐
    │   LiteLLM Gateway  │
    ├───────────┬────────┤
    │  Ollama   │Anthropic│
    │(CT210/    │  API    │
    │ local)    │(cloud)  │
    └───────────┴────────┘
```

---

## The Two Modes in One System

```
DEVELOPER MODE                    EVERYDAY MODE
─────────────────                 ──────────────────────
Live topology canvas              Maple Creek village
Event stream (typed)              Characters doing things
Memory query panel                Sage opens treasure chests
Teach interface                   Pip rings the alarm
Escalation queue                  Inn feedback form
GREEN/YELLOW/RED tiers            Warm dialog boxes
WebSocket JSON events             "Rosie: All fixed! ♥"
```

Same system. Same events. Same WebSocket. Different skin.

---

## Immediate Next Steps for Cowork

### Sprint 1 — Make It Run (3-5 days)
1. `core/orchestrator.py` — minimal working loop: task in → agent runs → result out → log event
2. `agents/base_agent.py` + `agents/guardian.py` — health check loop, emit events
3. `ui/server.py` — FastAPI with `/ws` WebSocket endpoint, serve static files
4. Wire `darkclaw_litellm.py` into orchestrator so every agent call goes through memory
5. `requirements.txt` + `.env.example`
6. Test: open browser, load Maple Creek, press Demo, watch events appear in both UI and daydream simultaneously

### Sprint 2 — Business Features (1-2 weeks)
1. Business Setup Wizard (guided flow, checkboxes, reminders)
2. Document plain-English explainer (upload PDF → Darkclaw extracts key facts → Rosie explains)
3. Compliance deadline tracker (SQLite table, Guardian monitors, Pip alerts)
4. SHOP-BOT Darkclaw integration (wire existing CT131 instance)

### Sprint 3 — Polish + Community (ongoing)
1. Maple Creek v2 (scrolling map, day/night, notice board)
2. GitHub repo public launch
3. Contributor guide + first Issues posted
4. Demo video: Taylor using it for Book Burrow

---

## The Ask for Contributors

**Three easy entry points:**

**1. New healing strategies** — `core/heal_engine.py` already has the interface. Add a new `FailureType` and a repair strategy. Good first PR.

**2. New memory backends** — implement the `MemoryBackend` interface. ChromaDB drop-in wanted. Neo4j for production scale. Redis for distributed.

**3. Daydream sprites** — draw a new character or tile in the Zelda village palette (16×16, 4 colors + transparent). Any art tool works. Opens the door for non-developers to contribute.

---

## File Manifest

```
darkclaw/
├── README.md                    ← Project intro (contributor-facing)
├── docs/
│   ├── SCOPE.md                 ← This document
│   ├── architecture.md          ← Deep dive: how the harness works
│   ├── self-healing.md          ← Failure taxonomy + repair strategies
│   └── self-teaching.md         ← Memory pipeline + eval loop
├── core/
│   ├── event_bus.py             ✅ Built
│   ├── heal_engine.py           ✅ Built
│   ├── teach_engine.py          ✅ Built
│   ├── orchestrator.py          ✅ Built (Sprint 1)
│   └── tool_registry.py         ✅ Built (Sprint 1)
├── agents/
│   ├── base_agent.py            ✅ Built (Sprint 1)
│   ├── guardian.py              ✅ Built (Sprint 1)
│   └── worker.py                ✅ Built (Sprint 1)
├── memory/
│   ├── darkclaw_core.py         ✅ Built (8/8 tests passing)
│   └── darkclaw_litellm.py      ✅ Built
├── ui/
│   ├── server.py                ✅ Built (Sprint 1)
│   └── static/
│       └── index.html           ✅ Built (live dashboard)
├── daydream/
│   ├── daydream.html            ✅ Built (Maple Creek)
│   ├── sprites.json             ✅ Built (base64 embedded)
│   └── sprites/                 ✅ Built (8 PNG files, CC0)
│       ├── rosie.png
│       ├── hero.png
│       ├── pip.png
│       ├── sage.png
│       ├── kit.png
│       ├── bea.png
│       ├── tileset.png
│       └── items.png
└── tests/
    ├── test_darkclaw.py         ✅ Built (8/8 passing)
    ├── test_heal_engine.py      ✅ Built
    └── test_orchestrator.py     ✅ Built (Sprint 1 integration)
```

---

## Personal Note

This project started as a homelab experiment on a Dell Precision T5810 in Stanford, Kentucky. It's being built alongside a real bookstore opening (Book Burrow, Perryville KY) by a family that believes in doing things right and helping others do the same.

The goal isn't to build another AI tool. It's to build the system that stands beside people who are building something — and makes sure they don't fall through the cracks.

If you're reading this and you want to help: welcome. The Issues tab is open. The code is clean. Jump in wherever makes sense.

— Bradley
