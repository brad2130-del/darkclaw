"""
darkclaw/main.py
Boot the full Darkclaw system.

Starts:
  - Darkclaw memory layer (context graph + vector)
  - HealEngine + TeachEngine
  - Orchestrator with built-in tools
  - Rosie (worker agent) — handles user tasks
  - Guardian (Pip) — health monitor, runs watch loop
  - FastAPI + WebSocket server on port 7430

Usage:
  python main.py

Then open:
  http://localhost:7430/          ← operator dashboard
  http://localhost:7430/daydream  ← Maple Creek screensaver

Environment variables (.env or export):
  ANTHROPIC_API_KEY    — for claude-* models
  OLLAMA_BASE_URL      — Ollama server (default: http://localhost:11434)
  DEFAULT_MODEL        — model for Rosie (default: claude-haiku-4-5)
  FALLBACK_MODEL       — fallback on failure (default: ollama/phi3)
  DB_PATH              — SQLite path (default: darkclaw.db)
  PORT                 — server port (default: 7430)
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

DB_PATH       = os.getenv("DB_PATH",        "darkclaw.db")
PORT          = int(os.getenv("PORT",        7430))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL",  "claude-haiku-4-5")
FALLBACK_MODEL= os.getenv("FALLBACK_MODEL", "ollama/phi3")
OLLAMA_URL    = os.getenv("OLLAMA_BASE_URL","http://localhost:11434")


async def build_system():
    """Construct and wire the full Darkclaw agent harness."""

    # ── 1. Memory layer ───────────────────────────────────────────────────
    from memory.darkclaw_litellm import DarkclawMiddleware
    memory = DarkclawMiddleware(db_path=DB_PATH, verbose=False)

    # ── 2. Core systems ───────────────────────────────────────────────────
    from core.heal_engine import HealEngine
    from core.teach_engine import TeachEngine
    from core.orchestrator import Orchestrator

    healer  = HealEngine(memory_engine=memory.engine)
    teacher = TeachEngine(memory_engine=memory.engine)
    orc     = Orchestrator(
        memory=memory,
        heal_engine=healer,
        teach_engine=teacher,
    )

    # ── 3. Built-in tools ─────────────────────────────────────────────────
    _register_tools(orc, memory)

    # ── 4. Worker agent (Rosie) ───────────────────────────────────────────
    from agents.base_agent import AgentConfig
    from agents.worker import WorkerAgent

    rosie = WorkerAgent(
        config=AgentConfig(
            agent_id="Rosie",
            model=DEFAULT_MODEL,
            role="worker",
            fallback_model=FALLBACK_MODEL,
        ),
        memory=memory,
        tool_registry=orc.tools,
    )
    orc.register_agent(rosie)

    # ── 5. Guardian (Pip) ─────────────────────────────────────────────────
    from agents.guardian import GuardianAgent

    guardian = GuardianAgent(
        config=AgentConfig(
            agent_id="Guardian",
            role="guardian",
            model="devstral-small",
            fallback_model=FALLBACK_MODEL,
        ),
        orchestrator=orc,
        heal_engine=healer,
    )
    healer.orc = orc
    guardian.register(rosie)
    orc.register_agent(guardian)

    return orc, guardian, memory


def _register_tools(orc, memory):
    """Register the built-in tool set available to all worker agents."""

    async def memory_query(inp: dict) -> str:
        result = memory.query(inp.get("query", ""), "tool")
        # result is a tool_result dict — pull the answer text
        try:
            return result["content"][0]["text"]
        except (KeyError, IndexError):
            return str(result)

    async def memory_teach(inp: dict) -> str:
        subj = inp.get("subject", "")
        pred = inp.get("predicate", "")
        obj  = inp.get("object", "")
        memory.ingest_fact("tool", subj, pred, obj)
        return f"Learned: {subj} {pred} {obj}"

    async def read_file(inp: dict) -> str:
        path = inp.get("path", "")
        try:
            with open(path) as f:
                return f.read()[:4000]
        except Exception as e:
            return f"Error reading {path}: {e}"

    async def check_compliance(inp: dict) -> str:
        topic = inp.get("topic", "")
        state = inp.get("state", "Kentucky")
        result = memory.query(f"{topic} compliance {state}", "tool")
        try:
            return result["content"][0]["text"]
        except (KeyError, IndexError):
            return f"No compliance data found for: {topic} in {state}"

    orc.register_tool({
        "name": "memory_query",
        "description": "Search Darkclaw memory for facts about a person, entity, or topic",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look up"}},
            "required": ["query"],
        },
        "handler": memory_query,
    })
    orc.register_tool({
        "name": "memory_teach",
        "description": "Add a fact to Darkclaw memory as a subject-predicate-object triple",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject":   {"type": "string"},
                "predicate": {"type": "string"},
                "object":    {"type": "string"},
            },
            "required": ["subject", "predicate", "object"],
        },
        "handler": memory_teach,
    })
    orc.register_tool({
        "name": "read_file",
        "description": "Read a local file and return its contents",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute or relative file path"}},
            "required": ["path"],
        },
        "handler": read_file,
    })
    orc.register_tool({
        "name": "check_compliance",
        "description": "Look up business compliance requirements for a topic and US state",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "e.g. LLC formation, resale certificate, food license"},
                "state": {"type": "string", "description": "US state name, e.g. Kentucky"},
            },
            "required": ["topic"],
        },
        "handler": check_compliance,
    })


async def main():
    print("╔══════════════════════════════════╗")
    print("║   Darkclaw v0.1  — booting…      ║")
    print("╚══════════════════════════════════╝")

    orc, guardian, memory = await build_system()

    import ui.server as srv
    srv.wire(orc, memory)

    print(f"  Dashboard : http://localhost:{PORT}/")
    print(f"  Daydream  : http://localhost:{PORT}/daydream")
    print(f"  Health    : http://localhost:{PORT}/health")
    print(f"  Model     : {DEFAULT_MODEL}")
    print(f"  DB        : {DB_PATH}")
    print()

    import uvicorn
    config = uvicorn.Config(
        "ui.server:app",
        host="0.0.0.0",
        port=PORT,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    await asyncio.gather(
        server.serve(),
        orc.run_forever(),
        guardian.watch_loop(interval_s=15.0),
    )


if __name__ == "__main__":
    asyncio.run(main())
