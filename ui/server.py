"""
darkclaw/ui/server.py
FastAPI server — serves the UIs and streams the live event bus.

Endpoints:
  GET  /            → operator dashboard (ui/static/index.html)
  GET  /daydream    → Maple Creek screensaver (daydream/daydream.html)
  WS   /ws          → live event stream (event_bus → browser)
  POST /teach       → add a fact to Darkclaw memory
  POST /query       → query Darkclaw memory
  POST /submit      → submit a task to an agent
  POST /demo        → run the scripted demo scenario through the real system
  GET  /health      → system health summary

Run:
  cd ~/darkclaw
  uvicorn ui.server:app --host 0.0.0.0 --port 7430 --reload
"""
import asyncio
import os

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from core.event_bus import bus, emit, EventType
from core.orchestrator import Orchestrator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_HTML = os.path.join(ROOT, "ui", "static", "index.html")
DAYDREAM_HTML = os.path.join(ROOT, "daydream", "daydream.html")

app = FastAPI(title="Darkclaw", version="0.1")

# Built once at startup, shared by all requests.
orc: Orchestrator = None


# ── request models ──────────────────────────────────────────────────────

class TeachIn(BaseModel):
    subject: str
    predicate: str
    object: str
    agent_id: str = "main"


class QueryIn(BaseModel):
    query: str
    agent_id: str = "main"


class SubmitIn(BaseModel):
    task: str
    agent_id: str = None


# ── lifecycle ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    global orc
    orc = Orchestrator.bootstrap()
    # Start the Guardian watchdog in the background.
    if orc.guardian:
        asyncio.create_task(orc.guardian.watch_loop())
    emit(EventType.SYSTEM_START, "darkclaw", msg="Darkclaw online", port=7430)


# ── pages ─────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)


@app.get("/daydream")
async def daydream():
    return FileResponse(DAYDREAM_HTML)


# ── live event stream ──────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    # Replay recent history so a freshly-opened browser isn't blank.
    for event in bus.recent(40):
        await websocket.send_text(event.to_json())
    try:
        async for event in bus.subscribe():
            await websocket.send_text(event.to_json())
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ── memory + task API ──────────────────────────────────────────────────────

@app.post("/teach")
async def teach(body: TeachIn):
    fact = orc.memory.ingest_fact(body.agent_id, body.subject, body.predicate, body.object)
    emit(EventType.MEMORY_INGEST, body.agent_id,
         subject=body.subject, predicate=body.predicate, object=body.object)
    triple = f"{body.subject} {body.predicate} {body.object}"
    return {"ok": bool(fact), "fact": triple}


@app.post("/query")
async def query(body: QueryIn):
    result = orc.memory.query(body.query, body.agent_id)
    emit(EventType.MEMORY_QUERY, body.agent_id,
         query=body.query[:80], tier=result._classify_tier(), method=result.method)
    return result.to_tool_result()


@app.post("/submit")
async def submit(body: SubmitIn):
    result = await orc.submit(body.task, agent_id=body.agent_id)
    return {
        "success": result.success,
        "output": result.output,
        "agent_id": result.agent_id,
        "duration_ms": round(result.duration_ms, 1),
        "error": result.error,
    }


@app.post("/demo")
async def demo():
    # Run in the background so the HTTP call returns immediately and events
    # stream over the WebSocket as the scenario plays out.
    asyncio.create_task(orc.run_demo())
    return JSONResponse({"ok": True, "msg": "demo scenario started"})


@app.get("/health")
async def health():
    return orc.health()
