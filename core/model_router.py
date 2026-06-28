"""
darkclaw/core/model_router.py

P100-aware dynamic model router with CPU/GPU placement, fast-mode override,
OpenClaw Pi5 offload, and Claude cloud models.

Routing tiers:
  /fast mode  → Claude Haiku (simple) or Sonnet (complex) — session override
  Pi5 offload → daddypi OpenClaw (100.77.235.30) for lightweight queries
  Local GPU   → deepseek-coder (9.5GB) + openclaw-brain-v2 (5GB) on P100
  Local CPU   → llama3.2, phi3.5, llama3.1 — free VRAM for GPU models
"""
import os
import time
import threading

# ── Fast-mode session flag ────────────────────────────────────────────
# Set via /fast command in the UI. Resets to False on service restart.
# When True, every model pick returns a Claude model regardless of role.

_fast_mode: bool = False
_fast_lock = threading.Lock()

def enable_fast_mode() -> None:
    global _fast_mode
    with _fast_lock:
        _fast_mode = True

def disable_fast_mode() -> None:
    global _fast_mode
    with _fast_lock:
        _fast_mode = False

def is_fast_mode() -> bool:
    with _fast_lock:
        return _fast_mode


# ── Hardware / endpoint constants ─────────────────────────────────────
P100_VRAM_GB   = 16.0
KEEP_ALIVE_SEC = 300

# OpenClaw Pi5 — daddypi (Tailscale 100.77.235.30, RPi5 BCM2712)
# Always-hot small model; used for lightweight general queries
OPENCLAW_PI5_URL   = os.environ.get("OPENCLAW_PI5_URL",   "http://100.77.235.30:11434")
OPENCLAW_PI5_MODEL = os.environ.get("OPENCLAW_PI5_MODEL", "openclaw-brain")

# Named model constants
FAST_MODEL      = "ollama/llama3.2:latest"
SYSTEM_MODEL    = "ollama/openclaw-brain-v2:latest"
SYSTEM_FALLBACK = "ollama/llama3.1:latest"
CODER_MODEL     = "ollama/deepseek-coder-v2:16b-lite-instruct-q4_K_M"
PI5_MODEL       = f"ollama/{OPENCLAW_PI5_MODEL}"
CLAUDE_FAST     = "claude-haiku-4-5-20251001"    # fastest Claude, ~50ms
CLAUDE_SMART    = "claude-sonnet-4-6"             # complex reasoning + repair plans


# ── Model placement profiles ──────────────────────────────────────────
# num_gpu=0  → CPU + RAM only  (fast enough for small models, frees P100)
# num_gpu=99 → all layers on GPU (needed for 7B+)
# num_ctx    → context window cap (smaller = faster prefill)
# Claude models have no Ollama opts — routed via litellm's Anthropic backend

MODEL_PROFILES: dict[str, dict] = {
    "ollama/phi3.5:latest": {
        "vram_gb": 0, "num_gpu": 0, "num_ctx": 2048,
    },
    "ollama/llama3.2:latest": {
        "vram_gb": 0, "num_gpu": 0, "num_ctx": 2048,
    },
    "ollama/llama3.1:latest": {
        "vram_gb": 0, "num_gpu": 0, "num_ctx": 2048,
    },
    "ollama/openclaw-brain-v2:latest": {
        "vram_gb": 5.0, "num_gpu": 99, "num_ctx": 4096,
    },
    "ollama/deepseek-coder-v2:16b-lite-instruct-q4_K_M": {
        "vram_gb": 9.5, "num_gpu": 99, "num_ctx": 4096,
    },
    # Pi5 — routed to daddypi Ollama; no P100 VRAM cost
    PI5_MODEL: {
        "vram_gb": 0, "num_gpu": 0, "num_ctx": 4096, "_pi5": True,
    },
    # Cloud models — no Ollama options; litellm uses ANTHROPIC_API_KEY
    CLAUDE_FAST:  {"vram_gb": 0, "num_gpu": 0, "num_ctx": 8192},
    CLAUDE_SMART: {"vram_gb": 0, "num_gpu": 0, "num_ctx": 16384},
}


class ModelRouter:
    """
    Routes tasks to models and returns an options dict the caller passes
    to litellm as kwargs.  Pi5 and Claude models carry special keys
    (_api_base) that callers must extract before passing as Ollama options.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._gpu_hot: dict[str, float] = {
            CODER_MODEL: time.time(),
        }

    # ── public API ─────────────────────────────────────────────────────

    def pick(self, role: str, task: str, complexity: float = 0.5) -> tuple[str, dict]:
        """
        Returns (model_string, options_dict).

        options_dict may contain:
          num_gpu, num_ctx  → Ollama placement (pass as extra["options"])
          _api_base         → override api_base (Pi5); callers must pop this
        """
        with self._lock:
            self._expire_stale()

            # /fast mode: all traffic → Claude
            if is_fast_mode():
                m = CLAUDE_FAST if complexity < 0.5 else CLAUDE_SMART
                return m, {}

            # Pi5 offload: very simple helper / shopkeeper queries → daddypi
            if role in ("helper", "shopkeeper") and complexity < 0.25 and OPENCLAW_PI5_URL:
                return self._use_pi5()

            if role in ("helper", "shopkeeper", "worker"):
                return self._use(FAST_MODEL)

            if role == "memory":
                if self._gpu_fits(SYSTEM_MODEL):
                    return self._use(SYSTEM_MODEL)
                return self._use(SYSTEM_FALLBACK)

            if role == "coder":
                if complexity < 0.3:
                    return self._use(FAST_MODEL)
                return self._use(CODER_MODEL)

            # Fallback agent always uses Claude Haiku
            if role == "fallback":
                return CLAUDE_FAST, {}

            return self._use(FAST_MODEL)

    def record_use(self, model: str):
        prof = MODEL_PROFILES.get(model, {})
        if prof.get("vram_gb", 0) > 0:
            with self._lock:
                self._gpu_hot[model] = time.time()

    def ollama_options(self, model: str) -> dict:
        prof = MODEL_PROFILES.get(model, {})
        opts = {}
        if "num_gpu" in prof:
            opts["num_gpu"] = prof["num_gpu"]
        if "num_ctx" in prof:
            opts["num_ctx"] = prof["num_ctx"]
        return opts

    def vram_state(self) -> dict:
        with self._lock:
            self._expire_stale()
            gpu_models = list(self._gpu_hot.keys())
            used = sum(MODEL_PROFILES.get(m, {}).get("vram_gb", 0) for m in gpu_models)
            return {
                "fast_mode":     is_fast_mode(),
                "gpu_models":    gpu_models,
                "cpu_models":    [k for k, v in MODEL_PROFILES.items()
                                  if v.get("num_gpu", 0) == 0 and not v.get("_pi5")
                                  and not k.startswith("claude-")],
                "pi5_model":     PI5_MODEL if OPENCLAW_PI5_URL else None,
                "pi5_url":       OPENCLAW_PI5_URL or None,
                "cloud_models":  [CLAUDE_FAST, CLAUDE_SMART],
                "vram_used_gb":  round(used, 1),
                "vram_total_gb": P100_VRAM_GB,
                "vram_free_gb":  round(P100_VRAM_GB - used, 1),
            }

    # ── internals ──────────────────────────────────────────────────────

    def _expire_stale(self):
        now = time.time()
        self._gpu_hot = {m: t for m, t in self._gpu_hot.items()
                         if now - t < KEEP_ALIVE_SEC}

    def _gpu_fits(self, model: str) -> bool:
        used = sum(MODEL_PROFILES.get(m, {}).get("vram_gb", 0) for m in self._gpu_hot)
        need = MODEL_PROFILES.get(model, {}).get("vram_gb", 0)
        return used + need <= P100_VRAM_GB

    def _use(self, model: str) -> tuple[str, dict]:
        prof = MODEL_PROFILES.get(model, {})
        if prof.get("vram_gb", 0) > 0:
            self._gpu_hot[model] = time.time()
        if model.startswith("claude-"):
            return model, {}
        return model, self.ollama_options(model)

    def _use_pi5(self) -> tuple[str, dict]:
        """Route to daddypi's OpenClaw — inject _api_base so caller overrides endpoint."""
        opts = self.ollama_options(PI5_MODEL)
        opts["_api_base"] = OPENCLAW_PI5_URL
        return PI5_MODEL, opts


# ── Complexity scoring ────────────────────────────────────────────────

def score_complexity(task: str) -> float:
    t = task.lower()
    score = 0.3
    if len(task) > 200:                                                   score += 0.2
    if any(k in t for k in ("debug","refactor","architect","design")):    score += 0.3
    if any(k in t for k in ("explain","what is","how does")):             score -= 0.1
    if t.count("```") > 0 or "def " in t:                                score += 0.2
    if any(k in t for k in ("fix","bug","error","traceback")):            score += 0.1
    return max(0.0, min(1.0, score))


# Singleton
router = ModelRouter()
