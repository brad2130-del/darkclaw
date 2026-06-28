"""
darkclaw/ui/server.py
FastAPI server — serves the UIs and streams the live event bus.

Endpoints:
  GET  /                    → operator dashboard (redirects to /setup if unconfigured)
  GET  /setup               → first-run setup wizard
  GET  /daydream            → Maple Creek screensaver
  WS   /ws                  → live event stream (event_bus → browser)
  POST /teach               → add a fact to Darkclaw memory
  POST /query               → query Darkclaw memory
  POST /submit              → submit a task to an agent
  POST /demo                → run the scripted demo scenario
  GET  /health              → system health summary
  GET  /api/models          → list available models from all backends
  POST /api/test-connection → probe a model backend before saving config
  POST /api/setup           → save configuration to user config dir

Run:
  cd ~/darkclaw
  uvicorn ui.server:app --host 0.0.0.0 --port 7430 --reload
"""
import asyncio
import os

# ── Config loading: prefer XDG config dir over project .env ─────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(
    os.environ.get("DARKCLAW_CONFIG", ""),
) or os.path.join(
    os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
    "darkclaw", "darkclaw.env"
)
ENV_PATH = os.path.join(ROOT, ".env")

from dotenv import load_dotenv
if os.path.exists(CONFIG_PATH):
    load_dotenv(CONFIG_PATH)
else:
    load_dotenv(ENV_PATH)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from core.event_bus import bus, emit, EventType
from core.orchestrator import Orchestrator

INDEX_HTML   = os.path.join(ROOT, "ui", "static", "index.html")
DAYDREAM_HTML = os.path.join(ROOT, "daydream", "daydream.html")
SETUP_HTML   = os.path.join(ROOT, "ui", "static", "setup.html")


def is_configured() -> bool:
    """True if at least one model backend is configured."""
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OLLAMA_BASE_URL"))

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


class SetupIn(BaseModel):
    backend: str                                      # anthropic | ollama | both
    anthropic_key: str = ""
    ollama_url: str = "http://localhost:11434"
    default_model: str = ""


class TestConnIn(BaseModel):
    type: str                                         # anthropic | ollama
    key: str = ""
    url: str = ""


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
    if not is_configured():
        return RedirectResponse("/setup")
    return FileResponse(INDEX_HTML)


@app.get("/setup")
async def setup_page():
    return FileResponse(SETUP_HTML)


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


# ── Setup / config API ─────────────────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    """List available models from all configured backends."""
    import httpx
    result: dict = {"anthropic": [], "ollama": [], "configured": is_configured()}

    if os.getenv("ANTHROPIC_API_KEY"):
        result["anthropic"] = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"]

    ollama_url = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
    if ollama_url:
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                r = await client.get(f"{ollama_url}/api/tags")
                result["ollama"] = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass

    return result


@app.post("/api/test-connection")
async def test_connection(body: TestConnIn):
    """Probe a model backend — called by the setup wizard before saving."""
    import httpx

    if body.type == "ollama":
        url = body.url.rstrip("/") or "http://localhost:11434"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{url}/api/tags")
                models = [m["name"] for m in r.json().get("models", [])]
                return {"ok": True, "models": models}
        except Exception as e:
            return {"ok": False, "error": str(e), "models": []}

    if body.type == "anthropic":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=body.key)
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            return {"ok": True, "models": ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"]}
        except Exception as e:
            return {"ok": False, "error": str(e), "models": []}

    return {"ok": False, "error": "unknown backend type", "models": []}


@app.post("/api/setup")
async def do_setup(body: SetupIn):
    """Save config to user config dir and reload env vars."""
    cfg_dir = os.path.dirname(CONFIG_PATH)
    os.makedirs(cfg_dir, exist_ok=True)

    lines = ["# Darkclaw configuration — written by setup wizard\n", f"PORT=7430\n"]

    if body.backend in ("anthropic", "both") and body.anthropic_key:
        lines.append(f"ANTHROPIC_API_KEY={body.anthropic_key}\n")
        lines.append("DARKCLAW_USE_LLM=1\n")

    if body.backend in ("ollama", "both") and body.ollama_url:
        lines.append(f"OLLAMA_BASE_URL={body.ollama_url}\n")

    if body.default_model:
        lines.append(f"DEFAULT_MODEL={body.default_model}\n")

    with open(CONFIG_PATH, "w") as f:
        f.writelines(lines)

    # Reload into the running process
    load_dotenv(CONFIG_PATH, override=True)

    return {"ok": True, "config_path": CONFIG_PATH}
