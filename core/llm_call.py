"""
darkclaw/core/llm_call.py

resilient_completion() — the single litellm call site for all agent traffic.

Healing ladder (each rung emits HEAL_* events so the UI Healing panel shows
reroutes as they happen):

  1. preferred node   — the router's _api_base, or the env default
  2. REROUTE_NODE     — any other node that actually serves the model
                        (fresh fleet probe; catches config drift)
  3. REROUTE_MODEL    — fallback chain: llama3.2 → llama3.1 → Claude Haiku
  4. raise the original error (caller's HealEngine / UI handles it)

Every call site that builds its own litellm.completion() is a future
routing bug (that's how the /stream 'model not found' happened) — route
new code through here instead.
"""
import os

from core.event_bus import EventType, emit
from core.fleet import fleet as _global_fleet


def _is_conn_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in (
        "connection refused", "connection reset", "connect error",
        "timed out", "timeout", "unreachable", "no route to host",
    ))


def _fallback_chain(model: str) -> list[str]:
    """Cheapest-first substitutes; Claude only if a key is configured."""
    from core.model_router import FAST_MODEL, SYSTEM_FALLBACK, CLAUDE_FAST
    chain = [m for m in (FAST_MODEL, SYSTEM_FALLBACK) if m != model]
    if os.environ.get("ANTHROPIC_API_KEY"):
        chain.append(CLAUDE_FAST)
    return chain


def _router_opts(model: str) -> dict:
    from core.model_router import router
    return router.ollama_options(model)


def resilient_completion(
    model: str,
    messages: list,
    *,
    agent_id: str = "main",
    api_base: str = None,
    options: dict = None,
    stream: bool = False,
    completion_fn=None,      # injectable for tests
    registry=None,           # injectable for tests
    **kw,
):
    """
    Drop-in for litellm.completion() with node/model failover.
    Returns (response, serve_info) where serve_info records which model and
    node actually answered — the trace that makes routing bugs self-evident.
    """
    # Import litellm only when we'll really call it — importing it runs
    # load_dotenv(), which would leak .env (DARKCLAW_USE_LLM=1) into
    # hermetic test runs that inject completion_fn.
    if completion_fn is None:
        import litellm
        call = litellm.completion
    else:
        call = completion_fn
    reg = registry or _global_fleet

    def _attempt(m: str, url: str | None):
        extra = dict(kw)
        if m.startswith("ollama/"):
            if url:
                extra["api_base"] = url
            opts = options if m == model else _router_opts(m)
            if opts:
                extra["options"] = {k: v for k, v in opts.items()
                                    if not k.startswith("_")}
        return call(model=m, messages=messages, stream=stream, **extra)

    def _info(m, url, healed):
        return {"model": m,
                "node": reg.name_of(url) if url else "cloud",
                "healed": healed}

    # Cloud models have no node to fail over between — call straight through.
    if not model.startswith("ollama/"):
        return _attempt(model, None), _info(model, None, False)

    primary = (api_base or os.environ.get("OLLAMA_API_BASE")
               or os.environ.get("OLLAMA_BASE_URL", "")).rstrip("/") or None

    tried = []

    def _note(m, url, e):
        tried.append({"model": m,
                      "node": reg.name_of(url) if url else "cloud",
                      "error": str(e)[:120]})
        if url and _is_conn_error(e):
            reg.mark_bad(url)

    # ── rung 1: primary node ────────────────────────────────────────────
    try:
        return _attempt(model, primary), _info(model, primary, False)
    except Exception as e:
        first_err = e
        _note(model, primary, e)

    emit(EventType.HEAL_TRIGGERED, agent_id,
         failure_type="LLM_CALL", model=model,
         node=reg.name_of(primary) if primary else "default",
         error=str(first_err)[:200])

    # ── rung 2: same model, different node (fresh probe → sees drift) ───
    for url in reg.locate(model, fresh=True):
        if url == primary:
            continue
        emit(EventType.HEAL_ATTEMPT, agent_id,
             strategy="REROUTE_NODE", model=model, node=reg.name_of(url))
        try:
            resp = _attempt(model, url)
            emit(EventType.HEAL_SUCCESS, agent_id,
                 strategy="REROUTE_NODE", model=model, node=reg.name_of(url))
            return resp, _info(model, url, True)
        except Exception as e:
            _note(model, url, e)

    # ── rung 3: fallback models ─────────────────────────────────────────
    for fb in _fallback_chain(model):
        if fb.startswith("ollama/"):
            for url in reg.locate(fb):
                emit(EventType.HEAL_ATTEMPT, agent_id,
                     strategy="REROUTE_MODEL", model=fb, node=reg.name_of(url))
                try:
                    resp = _attempt(fb, url)
                    emit(EventType.HEAL_SUCCESS, agent_id,
                         strategy="REROUTE_MODEL", model=fb,
                         node=reg.name_of(url))
                    return resp, _info(fb, url, True)
                except Exception as e:
                    _note(fb, url, e)
        else:
            emit(EventType.HEAL_ATTEMPT, agent_id,
                 strategy="REROUTE_MODEL", model=fb, node="cloud")
            try:
                resp = _attempt(fb, None)
                emit(EventType.HEAL_SUCCESS, agent_id,
                     strategy="REROUTE_MODEL", model=fb, node="cloud")
                return resp, _info(fb, None, True)
            except Exception as e:
                _note(fb, None, e)

    emit(EventType.HEAL_FAILED, agent_id,
         failure_type="LLM_CALL", model=model, tried=tried)
    raise first_err
