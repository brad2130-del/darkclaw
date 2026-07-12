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
import subprocess
import time

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
    asyncio.create_task(_fleet_preflight_loop())
    if orc.memory:
        from core.curator import curator_loop
        asyncio.create_task(curator_loop(orc.memory))
        asyncio.create_task(_embed_warmup())


async def _embed_warmup():
    """
    Pre-build the vector-memory embedding cache in the background.
    Cold cache = the first query after every restart re-embeds the whole
    store through the memory node (minutes) — that wait belongs to boot,
    not to the first person who says hello.
    """
    try:
        t0 = time.perf_counter()
        await asyncio.to_thread(orc.memory.query, "warmup: prebuild embedding cache", "system")
        emit(EventType.HEALTH_OK, "darkclaw", msg="embed cache warm",
             warmup_s=round(time.perf_counter() - t0, 1))
    except Exception as e:
        emit(EventType.HEALTH_WARN, "darkclaw", msg=f"embed warmup failed: {e}")


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


async def _fleet_preflight_loop():
    """
    Config-drift watchdog: verify every configured model actually exists on
    its assigned node (startup + every DARKCLAW_PREFLIGHT_INTERVAL seconds).
    A finding here is a user-visible failure waiting to happen — surface it
    as HEALTH_WARN before a request trips over it.
    """
    from core.fleet import fleet
    from core.model_router import expected_placement
    interval = float(os.environ.get("DARKCLAW_PREFLIGHT_INTERVAL", "600"))
    while True:
        try:
            findings = await asyncio.to_thread(fleet.preflight, expected_placement())
            for f in findings:
                emit(EventType.HEALTH_WARN, "fleet", **f)
            if not findings:
                emit(EventType.HEALTH_OK, "fleet",
                     msg="fleet preflight clean", nodes=list(fleet.nodes))
        except Exception as e:
            emit(EventType.SYSTEM_ERROR, "fleet", msg=f"preflight error: {e}")
        await asyncio.sleep(interval)


def _read_cyd_gpu() -> dict:
    """Compact GPU snapshot for the CYD dashboard — same nvidia-smi query as
    BashVault/health_check.py, reshaped to numeric fields for on-device gauges."""
    try:
        raw = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            encoding="utf-8", timeout=3,
        ).strip().split(",")
        
        def safe_float(v):
            cleaned = v.strip()
            return 0.0 if cleaned == "[N/A]" or not cleaned else float(cleaned)

        return {
            "online": True,
            "temp_c": safe_float(raw[0]),
            "util_pct": safe_float(raw[1]),
            "vram_used_mb": safe_float(raw[2]),
            "vram_total_mb": safe_float(raw[3]),
            "power_w": safe_float(raw[4]),
            "power_limit_w": safe_float(raw[5]),
        }
    except Exception:
        return {"online": False}


def _read_cyd_cpu_ram() -> dict:
    with open("/proc/loadavg") as f:
        load1, load5, load15 = f.read().split()[:3]
    freq_mhz = None
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            freq_mhz = round(int(f.read().strip()) / 1000, 0)
    except Exception:
        pass
    mem_total = mem_avail = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_avail = int(line.split()[1])
    used_pct = round(100 * (1 - mem_avail / mem_total), 1) if mem_total and mem_avail else None
    return {
        "load1": float(load1), "load5": float(load5), "load15": float(load15),
        "freq_mhz": freq_mhz, "ram_used_pct": used_pct,
    }


@app.get("/api/cyd")
async def api_cyd():
    """Single compact JSON payload for the ESP32 CYD dashboard's Acer page —
    keeps the on-device HTTP client to one request per poll cycle."""
    gpu = await asyncio.to_thread(_read_cyd_gpu)
    cpu_ram = await asyncio.to_thread(_read_cyd_cpu_ram)
    return {"gpu": gpu, **cpu_ram}


@app.get("/api/fleet")
async def api_fleet():
    """Live fleet map + config drift — the 'which node serves what' trace."""
    from core.fleet import fleet
    from core.model_router import expected_placement
    snap  = await asyncio.to_thread(fleet.snapshot)
    drift = await asyncio.to_thread(fleet.preflight, expected_placement())
    return {"nodes": snap, "drift": drift}


class SentinelIn(BaseModel):
    source: str
    signature: str
    count: int
    window_s: int = 900
    sample: str = ""


@app.post("/api/sentinel/report")
async def sentinel_report(body: SentinelIn):
    """
    Receiving end of the sentinel daemon: a recurring error signature
    crossed its threshold somewhere in the fleet. Escalate it for human
    review and (optionally) fire an async LLM analysis on the fleet —
    the sentinel itself never spends model time.
    """
    msg = (f"[{body.source}] '{body.signature}' seen {body.count}x "
           f"in {body.window_s // 60} min")
    emit(EventType.HEALTH_WARN, "sentinel",
         issue=msg, sample=body.sample[:200])
    if orc and orc.healer:
        orc.healer.escalate_external(
            "sentinel", msg,
            context={"signature": body.signature, "sample": body.sample},
        )
    analyzed = False
    if orc and os.environ.get("DARKCLAW_SENTINEL_ANALYZE", "1") == "1":
        task = (
            f"The fleet sentinel flagged a recurring issue on {body.source}: "
            f"the error signature below occurred {body.count} times in "
            f"{body.window_s // 60} minutes.\n\nSignature: {body.signature}\n"
            f"Sample line: {body.sample[:300]}\n\n"
            "Diagnose the likely root cause and recommend one concrete "
            "repair step. Be brief and specific."
        )
        asyncio.create_task(orc.submit(task, agent_id="sage"))
        analyzed = True
    return {"ok": True, "escalated": True, "analysis_started": analyzed}


@app.get("/api/hardening")
async def api_hardening():
    """
    One surface for the whole hardening stack: circuit breakers, context
    budget trims, per-agent query guards, and heal-engine stats — so 'is
    the self-healing actually earning its keep' is one curl away.
    """
    from core.guardrails import breaker
    from core.context_budget import stats as budget_stats
    from core.query_guard import guards
    return {
        "circuit_breakers": breaker.snapshot(),
        "context_budget":   dict(budget_stats),
        "query_guards":     guards.snapshot(),
        "heal_engine":      orc.healer.stats() if orc and orc.healer else {},
    }


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


# ── Slash-command handler ─────────────────────────────────────────────────

async def _run_command(cmd: str) -> str:
    """
    Process /slash commands typed in the Rosie chat box.
    Returns a plain text response string (streamed token-by-token by /stream).

    Commands:
      /fast    — switch all agents to Claude exclusively (haiku/sonnet by complexity)
      /return  — return to local Ollama routing
      /plan    — ask Claude Sonnet to analyse current system health and produce a repair plan
    """
    import json as _json
    from core.model_router import (enable_fast_mode, disable_fast_mode, is_fast_mode,
                                   CLAUDE_FAST, CLAUDE_SMART, OPENCLAW_PI5_URL, OPENCLAW_PI5_MODEL)

    name = cmd.strip().lstrip("/").split()[0].lower()

    if name == "fast":
        enable_fast_mode()
        emit(EventType.SYSTEM_START, "darkclaw", msg="fast-mode-on")
        return (
            "⚡ FAST MODE ON\n\n"
            f"All agents are now routed exclusively to Claude.\n"
            f"  Simple queries  → {CLAUDE_FAST}\n"
            f"  Complex tasks   → {CLAUDE_SMART}\n\n"
            "Claude has full visibility: memory context, uploaded docs, heal history,\n"
            "and agent state are all injected before every call.\n\n"
            "Type /return to restore local Ollama routing.\n"
            "Type /plan  to get a Claude-generated repair plan for the current system."
        )

    if name == "return":
        disable_fast_mode()
        emit(EventType.SYSTEM_START, "darkclaw", msg="fast-mode-off")
        pi5_note = f"\n  Pi5 offload → {OPENCLAW_PI5_MODEL} at {OPENCLAW_PI5_URL}" if OPENCLAW_PI5_URL else ""
        return (
            "🌿 LOCAL MODE RESTORED\n\n"
            "Routing returned to Ollama / P100:\n"
            "  Rosie, Kit, Bea  → llama3.2 (CPU)\n"
            "  Sage             → openclaw-brain-v3 (GPU)\n"
            "  Coder            → deepseek-coder 16B (GPU)" + pi5_note + "\n\n"
            "Claude remains available as Coder's teacher and via /fast."
        )

    if name == "plan":
        return await _generate_plan()

    return (
        f"Unknown command: /{name}\n\n"
        "Available commands:\n"
        "  /fast    — switch to Claude-exclusive mode\n"
        "  /return  — restore local Ollama routing\n"
        "  /plan    — Claude analyses system health and proposes repairs"
    )


async def _generate_plan() -> str:
    """
    Gather system health snapshot and ask Claude Sonnet for a repair plan.
    This is what /plan runs — Claude sees everything: agents, heals, teach stats,
    escalation queue, memory size, and recent failure types.
    """
    import json as _json, litellm

    health = orc.health()

    # Flatten agent states
    agents_snap = {
        aid: {
            "status":     a.get("status"),
            "error_rate": a.get("error_rate"),
            "tasks":      a.get("tasks_run", a.get("tasks")),
            "model":      a.get("model"),
        }
        for aid, a in health.get("agents", {}).items()
    }

    # Pull recent failures from heal engine history
    healer = getattr(orc, "healer", None)
    recent_failures, escalated = [], []
    if healer:
        for f in getattr(healer, "_history", [])[-12:]:
            recent_failures.append({
                "agent":      f.agent_id,
                "type":       str(f.failure_type),
                "error":      f.error_msg[:120],
                "resolved":   f.resolved,
                "resolution": f.resolution,
            })
        for f in getattr(healer, "_escalation_queue", [])[-5:]:
            escalated.append({"agent": f.agent_id, "error": f.error_msg[:80]})

    report = {
        "agents":           agents_snap,
        "heal":             health.get("heal", {}),
        "teach":            health.get("teach", {}),
        "memory_facts":     health.get("memory", {}).get("facts", "?"),
        "recent_failures":  recent_failures,
        "escalated":        escalated,
    }

    system_prompt = (
        "You are Darkclaw's autonomous repair planner. You have full visibility into "
        "the system and its agents. When given a health report, produce a numbered "
        "action plan — each step must say WHAT to fix, WHY it's broken, and HOW to fix it "
        "(name the file, method, or config key). Be concise and direct. No preamble.\n\n"
        "Agent roster: Rosie (helper/phi3.5), Kit (shopkeeper/llama3.2), "
        "Sage (memory/openclaw-brain-v3 GPU), Bea (worker/llama3.1), "
        "Coder (deepseek-coder 16B GPU + Claude teacher), Pip (guardian/watchdog), "
        "Fallback (claude-haiku-4-5, always-on cloud safety net).\n"
        "Stack: FastAPI + asyncio, Ollama P100 16GB, SQLite memory, litellm routing."
    )

    try:
        response = litellm.completion(
            model="claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":
                 f"Health report:\n```json\n{_json.dumps(report, indent=2)}\n```\n\n"
                 "Analyse this and give me a numbered repair plan. "
                 "If everything looks healthy, say so clearly."},
            ],
            timeout=45,
        )
        plan = response.choices[0].message.content or "No plan returned."
        return f"📋 DARKCLAW REPAIR PLAN\n\n{plan}"
    except Exception as e:
        return (
            f"⚠️  Plan generation failed ({e})\n\n"
            f"Raw health snapshot:\n```json\n{_json.dumps(report, indent=2)}\n```"
        )


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

            # ── Slash-command interception ───────────────────────────
            if body.task.strip().startswith("/"):
                yield f"data: {_json.dumps({'type':'meta','agent_id':'darkclaw','model':'system'})}\n\n"
                resp = await _run_command(body.task.strip())
                for word in resp.split(" "):
                    yield f"data: {_json.dumps({'type':'token','text':word+' '})}\n\n"
                    await asyncio.sleep(0.008)
                yield f"data: {_json.dumps({'type':'done','agent_id':'darkclaw'})}\n\n"
                emit(EventType.AGENT_TASK_DONE, "darkclaw", task=body.task[:40])
                return

            # Route + pick model
            target_id = body.agent_id or orc._route(body.task)
            agent     = orc.agents.get(target_id) or next(iter(orc.agents.values()))

            # Re-entrancy guard: one in-flight query per agent. A second
            # submit while the agent is busy gets a polite error instead of
            # racing the first on status and RAG attribution.
            from core.query_guard import guards
            guard = guards.get(target_id)
            gen_token = guard.try_start()
            if gen_token is None:
                yield f"data: {_json.dumps({'type':'error','error':f'{target_id} is still working on the previous message — give it a moment'})}\n\n"
                return
            try:
                from core.model_router import router, score_complexity
                complexity    = score_complexity(body.task)
                model, opts   = router.pick(agent.config.role, body.task, complexity)

                # Build context with memory + docs — in a thread: these are
                # blocking calls, and blocking the event loop here freezes
                # every other request (including the guard's busy replies)
                context = {}
                if orc.memory:
                    qr = await asyncio.to_thread(orc.memory.query, body.task, target_id)
                    context["injected_memory"] = qr.to_tool_result()
                    doc_qr = await asyncio.to_thread(orc.memory.query, body.task, "docs")
                    if doc_qr.confidence > 0.1 and doc_qr.answer != "No memory found.":
                        context["doc_memory"] = doc_qr.to_tool_result()

                # System prompt via the shared budget-enforced builder —
                # same code path as the worker, can't overflow num_ctx.
                from core.context_budget import build_system_prompt
                sys_prompt = build_system_prompt(context, body.task, model,
                                                 agent_id=target_id)
                messages = []
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": body.task})

                # Ollama routing — router opts carry _api_base (memory node / Pi5)
                opts = dict(opts)
                api_base = opts.pop("_api_base", None)

                # Announce agent + model
                yield f"data: {_json.dumps({'type':'meta','agent_id':target_id,'model':model})}\n\n"

                # Stream tokens through the resilient fleet layer (node/model
                # failover + HEAL events). Always close the response to release
                # the Ollama socket, even if the SSE client disconnects or an
                # exception is thrown mid-stream.
                full_text = ""
                from core.llm_call import resilient_completion
                response, serve_info = await asyncio.to_thread(
                    lambda: resilient_completion(
                        model=model, messages=messages, agent_id=target_id,
                        api_base=api_base, options=opts or None, stream=True,
                    ))
                try:
                    # litellm's stream iterator blocks per chunk — pull each
                    # chunk in a thread so the loop stays responsive
                    _END = object()
                    _it = iter(response)
                    while True:
                        chunk = await asyncio.to_thread(next, _it, _END)
                        if chunk is _END:
                            break
                        delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                        if delta:
                            full_text += delta
                            yield f"data: {_json.dumps({'type':'token','text':delta})}\n\n"
                finally:
                    try:
                        response.close()
                    except Exception:
                        pass

                # done carries the serve trace: which model/node really answered
                # (may differ from meta if the fleet layer healed a reroute)
                yield f"data: {_json.dumps({'type':'done','agent_id':target_id, **serve_info})}\n\n"

                # Fire RAG + update router heat
                if full_text and orc.memory:
                    router.record_use(serve_info["model"])
                    from core.rag_extractor import schedule_rag
                    schedule_rag(body.task, full_text, target_id, orc.memory)
                    emit(EventType.AGENT_TASK_DONE, target_id,
                         task=body.task[:40], **serve_info)
            finally:
                guard.end(gen_token)

        except Exception as e:
            # full traceback to the journal — the SSE error line alone has
            # cost us real debugging time ("list index out of range", July 3)
            import traceback
            traceback.print_exc()
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


@app.post("/curate")
async def curate():
    """Run one curation cycle on demand (the loop also runs it periodically)."""
    from core.curator import Curator
    if not orc or not orc.memory:
        return {"error": "memory engine not ready"}
    report = await asyncio.to_thread(Curator(orc.memory).run_cycle)
    return {"success": True, **report}


@app.get("/health")
async def health():
    return orc.health()


# ── Document store API ────────────────────────────────────────────────────

@app.post("/api/docs/upload")
async def upload_doc(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        return JSONResponse({"ok": False, "error": "File too large (20 MB max)"}, status_code=413)
    # Off the event loop: image files block on a CPU moondream call that
    # can take minutes; inline it would freeze every websocket and query.
    record = await asyncio.to_thread(ingest_document, file.filename, raw, orc.memory)
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
