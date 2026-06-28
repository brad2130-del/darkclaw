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

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from core.event_bus import bus, emit, EventType
from core.orchestrator import Orchestrator
from core.doc_store import ingest_document, list_documents, delete_document

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
    providers: list[str]                              # ["anthropic","openai","xai","gemini","mistral","ollama"]
    anthropic_key: str = ""
    openai_key:    str = ""
    xai_key:       str = ""
    gemini_key:    str = ""
    mistral_key:   str = ""
    ollama_url:    str = "http://localhost:11434"
    default_model: str = ""


class TestConnIn(BaseModel):
    type: str                                         # anthropic | openai | xai | gemini | mistral | ollama
    key: str = ""
    url: str = ""


# ── lifecycle ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    global orc
    orc = Orchestrator.bootstrap()
    if orc.guardian:
        asyncio.create_task(orc.guardian.watch_loop())
    emit(EventType.SYSTEM_START, "darkclaw", msg="Darkclaw online", port=7430)
    # Re-announce all agents so any browser that connects after startup
    # gets AGENT_STARTED events in the history replay window.
    for aid, agent in orc.agents.items():
        emit(EventType.AGENT_STARTED, aid,
             role=agent.config.role, model=agent.config.model)
    asyncio.create_task(_heartbeat_loop())


async def _heartbeat_loop():
    """Emit a snapshot every 30s so the topology stays live."""
    from core.model_router import router
    while True:
        await asyncio.sleep(30)
        vram = router.vram_state()
        agent_snap = {
            aid: {"role": a.config.role, "model": a.config.model,
                  "status": a.status, "tasks": a._task_count}
            for aid, a in orc.agents.items()
        }
        emit(EventType.HEALTH_OK, "darkclaw",
             msg="heartbeat", agents=agent_snap, vram=vram)


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
    # Send current agent snapshot immediately so topology renders on connect.
    import json as _json, time as _time
    if orc:
        for aid, agent in orc.agents.items():
            snap = _json.dumps({
                "event_id": f"snap-{aid}",
                "type": "agent.started",
                "agent_id": aid,
                "data": {"role": agent.config.role, "model": agent.config.model,
                         "status": agent.status},
                "timestamp": _time.time(),
                "ts_human": _time.strftime("%H:%M:%S"),
            })
            await websocket.send_text(snap)
    # Replay recent event history.
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


@app.post("/stream")
async def stream_submit(body: SubmitIn):
    """
    Streaming version of /submit — returns SSE token stream.
    Format: data: {"type": "meta"|"token"|"done"|"error", ...}
    """
    import json as _json

    async def generate():
        try:
            import litellm

            # Route + pick model
            target_id = body.agent_id or orc._route(body.task)
            agent     = orc.agents.get(target_id) or next(iter(orc.agents.values()))

            from core.model_router import router, score_complexity
            complexity    = score_complexity(body.task)
            model, opts   = router.pick(agent.config.role, body.task, complexity)

            # Build context with memory + docs
            context = {}
            if orc.memory:
                qr = orc.memory.query(body.task, target_id)
                context["injected_memory"] = qr.to_tool_result()
                doc_qr = orc.memory.query(body.task, "docs")
                if doc_qr.confidence > 0.1 and doc_qr.answer != "No memory found.":
                    context["doc_memory"] = doc_qr.to_tool_result()

            # Build messages
            sys_parts = []
            mem_ans = (context.get("injected_memory") or {}).get("answer", "")
            if mem_ans and mem_ans != "No memory found.":
                sys_parts.append(f"[Darkclaw Memory]\n{mem_ans}")
            doc_ans = (context.get("doc_memory") or {}).get("answer", "")
            if doc_ans and doc_ans != "No memory found.":
                sys_parts.append(f"[Document Context]\n{doc_ans}")

            messages = []
            if sys_parts:
                messages.append({"role": "system", "content": "\n\n".join(sys_parts)})
            messages.append({"role": "user", "content": body.task})

            # Ollama routing — include placement options (num_gpu, num_ctx)
            extra = {}
            if model.startswith("ollama/"):
                url = os.environ.get("OLLAMA_API_BASE") or os.environ.get("OLLAMA_BASE_URL", "")
                if url:
                    extra["api_base"] = url.rstrip("/")
                if opts:
                    extra["options"] = opts

            # Announce agent + model
            yield f"data: {_json.dumps({'type':'meta','agent_id':target_id,'model':model})}\n\n"

            # Stream tokens
            full_text = ""
            response = litellm.completion(
                model=model, messages=messages, stream=True, **extra
            )
            for chunk in response:
                delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                if delta:
                    full_text += delta
                    yield f"data: {_json.dumps({'type':'token','text':delta})}\n\n"

            yield f"data: {_json.dumps({'type':'done','agent_id':target_id})}\n\n"

            # Fire RAG + update router heat
            if full_text and orc.memory:
                router.record_use(model)
                from core.rag_extractor import schedule_rag
                schedule_rag(body.task, full_text, target_id, orc.memory)
                emit(EventType.AGENT_TASK_DONE, target_id, task=body.task[:40])

        except Exception as e:
            yield f"data: {_json.dumps({'type':'error','error':str(e)[:200]})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


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


# ── Document store API ────────────────────────────────────────────────────

@app.post("/api/docs/upload")
async def upload_doc(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        return JSONResponse({"ok": False, "error": "File too large (20 MB max)"}, status_code=413)
    record = ingest_document(file.filename, raw, orc.memory)
    emit(EventType.MEMORY_INGEST, "docs",
         subject=f"doc:{record['doc_id']}",
         predicate="IS_DOCUMENT",
         object=file.filename,
         chunks=record["chunks"],
         chunk_types=record.get("chunk_types", {}))
    return {"ok": True, **record}


@app.get("/api/docs")
async def get_docs():
    return {"docs": list_documents()}


@app.delete("/api/docs/{doc_id}")
async def delete_doc(doc_id: str):
    ok = delete_document(doc_id, orc.memory)
    return {"ok": ok}


@app.post("/api/docs/{doc_id}/analyze")
async def analyze_doc(doc_id: str):
    """
    Self-improvement: compare an uploaded doc against Darkclaw's current
    config/codebase and propose concrete improvements.
    Uses the coder agent (deepseek + Claude teacher).
    """
    docs = {d["doc_id"]: d for d in list_documents()}
    if doc_id not in docs:
        return JSONResponse({"ok": False, "error": "doc not found"}, status_code=404)

    record = docs[doc_id]
    filename = record["filename"]

    # Retrieve the doc's chunks from memory
    doc_qr = orc.memory.query(filename, "docs")
    doc_text = doc_qr.answer if doc_qr.answer != "No memory found." else ""

    # Read Darkclaw's own relevant source files for comparison context
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    own_files = []
    for rel in ["core/orchestrator.py", "agents/base_agent.py",
                "core/doc_store.py", ".env"]:
        fpath = os.path.join(root, rel)
        if os.path.exists(fpath):
            try:
                content = open(fpath).read()[:3000]   # cap per file
                own_files.append(f"=== {rel} ===\n{content}")
            except Exception:
                pass

    own_context = "\n\n".join(own_files)

    task = (
        f"You are reviewing an uploaded document against the Darkclaw AI harness "
        f"codebase to suggest concrete self-improvements.\n\n"
        f"UPLOADED FILE: {filename}\n"
        f"CONTENT:\n{doc_text[:2000]}\n\n"
        f"CURRENT DARKCLAW CONFIG:\n{own_context[:3000]}\n\n"
        f"Task: Compare the uploaded document against the current Darkclaw config. "
        f"Identify gaps, conflicts, or improvements. Output a concise numbered list "
        f"of specific, actionable changes Darkclaw could make to better align with "
        f"or incorporate this document. Be specific about which file/setting to change."
    )

    result = await orc.submit(task, agent_id="coder")
    return {
        "ok": result.success,
        "filename": filename,
        "analysis": result.output,
        "agent_id": result.agent_id,
        "duration_ms": round(result.duration_ms, 1),
    }


# ── Setup / config API ─────────────────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    """List available models from all configured backends."""
    result: dict = {"configured": is_configured(), "providers": {}}

    if os.getenv("ANTHROPIC_API_KEY"):
        result["providers"]["anthropic"] = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"]
    if os.getenv("OPENAI_API_KEY"):
        result["providers"]["openai"] = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    if os.getenv("XAI_API_KEY"):
        result["providers"]["xai"] = ["xai/grok-3-mini", "xai/grok-3"]
    if os.getenv("GEMINI_API_KEY"):
        result["providers"]["gemini"] = ["gemini/gemini-2.0-flash-lite", "gemini/gemini-2.0-flash", "gemini/gemini-1.5-pro"]
    if os.getenv("MISTRAL_API_KEY"):
        result["providers"]["mistral"] = ["mistral/mistral-small-latest", "mistral/mistral-large-latest"]

    ollama_url = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
    if ollama_url:
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                r = await client.get(f"{ollama_url}/api/tags")
                result["providers"]["ollama"] = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            result["providers"]["ollama"] = []

    return result


# Known model lists for cloud providers (returned without making a billable call)
_CLOUD_MODELS = {
    "anthropic": ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"],
    "openai":    ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
    "xai":       ["xai/grok-3-mini", "xai/grok-3"],
    "gemini":    ["gemini/gemini-2.0-flash-lite", "gemini/gemini-2.0-flash", "gemini/gemini-1.5-pro"],
    "mistral":   ["mistral/mistral-small-latest", "mistral/mistral-large-latest"],
}

# Endpoints to validate each cloud API key with a lightweight call
async def _probe_cloud(client: "httpx.AsyncClient", prov: str, key: str) -> dict:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        if prov == "anthropic":
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
                timeout=8.0,
            )
            ok = r.status_code in (200, 400)   # 400 = bad request but key valid
        elif prov == "openai":
            r = await client.get("https://api.openai.com/v1/models", headers=headers, timeout=8.0)
            ok = r.status_code == 200
        elif prov == "xai":
            r = await client.get("https://api.x.ai/v1/models", headers=headers, timeout=8.0)
            ok = r.status_code == 200
        elif prov == "gemini":
            r = await client.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={key}", timeout=8.0
            )
            ok = r.status_code == 200
        elif prov == "mistral":
            r = await client.get("https://api.mistral.ai/v1/models", headers=headers, timeout=8.0)
            ok = r.status_code == 200
        else:
            return {"ok": False, "error": "unknown provider"}
        if ok:
            return {"ok": True, "models": _CLOUD_MODELS.get(prov, [])}
        return {"ok": False, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/test-connection")
async def test_connection(body: TestConnIn):
    """Probe a model backend — called by the setup wizard before saving."""
    if body.type == "ollama":
        url = (body.url or "http://localhost:11434").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{url}/api/tags")
                models = [m["name"] for m in r.json().get("models", [])]
                return {"ok": True, "models": models}
        except Exception as e:
            return {"ok": False, "error": str(e), "models": []}

    if body.type in _CLOUD_MODELS:
        async with httpx.AsyncClient() as client:
            return await _probe_cloud(client, body.type, body.key)

    return {"ok": False, "error": "unknown provider type", "models": []}


@app.post("/api/setup")
async def do_setup(body: SetupIn):
    """Save config to user config dir and reload env vars."""
    cfg_dir = os.path.dirname(CONFIG_PATH)
    os.makedirs(cfg_dir, exist_ok=True)

    lines = ["# Darkclaw configuration — written by setup wizard\n", "PORT=7430\n"]

    key_map = {
        "anthropic": ("ANTHROPIC_API_KEY", body.anthropic_key),
        "openai":    ("OPENAI_API_KEY",    body.openai_key),
        "xai":       ("XAI_API_KEY",       body.xai_key),
        "gemini":    ("GEMINI_API_KEY",    body.gemini_key),
        "mistral":   ("MISTRAL_API_KEY",   body.mistral_key),
    }
    has_cloud = False
    for prov in body.providers:
        if prov in key_map:
            env_var, val = key_map[prov]
            if val:
                lines.append(f"{env_var}={val}\n")
                has_cloud = True
        elif prov == "ollama" and body.ollama_url:
            lines.append(f"OLLAMA_BASE_URL={body.ollama_url}\n")

    if has_cloud:
        lines.append("DARKCLAW_USE_LLM=1\n")
    if body.default_model:
        lines.append(f"DEFAULT_MODEL={body.default_model}\n")

    with open(CONFIG_PATH, "w") as f:
        f.writelines(lines)

    load_dotenv(CONFIG_PATH, override=True)
    return {"ok": True, "config_path": CONFIG_PATH}
