"""
darkclaw/core/model_router.py

P100-aware dynamic model router with CPU/GPU placement control.

Strategy (revised after benchmarking 2026-06-28):
  Small models (llama3.2, phi3.5) → CPU inference (num_gpu=0)
    - User's server has enough RAM; 3B models are fast on CPU (~1-2s)
    - Frees the entire P100 for heavy models that truly benefit from GPU
  Heavy models (deepseek-coder 16b) → full P100 GPU (num_gpu=99)
    - 16B needs GPU; CPU would be 60s+ per response
  System model (openclaw-brain-v2) → full P100 GPU, load on demand
    - Can now load without OOM since small models vacated VRAM

Benchmark results (P100, 2026-06-28):
  phi3.5:latest                    cold=10.7s  eval=195ms   ~2.2GB
  llama3.2:latest                  hot=242ms   eval=281ms   ~2.0GB
  llama3.1:latest                  cold=20.1s  eval=634ms   ~4.7GB
  openclaw-brain-v2:latest         OOM w/ deepseek → needs dedicated VRAM
  deepseek-coder-v2:16b-lite-q4   hot=138ms   eval=225ms   ~9.5GB
"""
import time
import threading

# ── Model placement profiles ──────────────────────────────────────────
# num_gpu=0  → CPU + RAM only  (fast enough for small models, frees P100)
# num_gpu=99 → all layers on GPU (needed for 7B+)
# num_ctx    → context window cap (smaller = faster prefill)

MODEL_PROFILES = {
    "ollama/phi3.5:latest": {
        "vram_gb": 0,           # CPU — no VRAM cost
        "num_gpu": 0,
        "num_ctx": 2048,
    },
    "ollama/llama3.2:latest": {
        "vram_gb": 0,           # CPU — no VRAM cost
        "num_gpu": 0,
        "num_ctx": 2048,
    },
    "ollama/llama3.1:latest": {
        "vram_gb": 0,           # CPU fallback for system queries
        "num_gpu": 0,
        "num_ctx": 2048,
    },
    "ollama/openclaw-brain-v2:latest": {
        "vram_gb": 5.0,         # GPU — has trained weights, needs fast inference
        "num_gpu": 99,
        "num_ctx": 4096,
    },
    "ollama/deepseek-coder-v2:16b-lite-instruct-q4_K_M": {
        "vram_gb": 9.5,         # GPU — 16B needs P100
        "num_gpu": 99,
        "num_ctx": 4096,
    },
}

P100_VRAM_GB   = 16.0
KEEP_ALIVE_SEC = 300

FAST_MODEL      = "ollama/llama3.2:latest"
SYSTEM_MODEL    = "ollama/openclaw-brain-v2:latest"
SYSTEM_FALLBACK = "ollama/llama3.1:latest"
CODER_MODEL     = "ollama/deepseek-coder-v2:16b-lite-instruct-q4_K_M"


class ModelRouter:
    """
    Routes tasks to models and returns Ollama options dict
    (num_gpu, num_ctx) so each call lands on the right hardware.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # Only GPU-resident models tracked here (CPU models don't count)
        self._gpu_hot: dict[str, float] = {
            CODER_MODEL: time.time(),   # was hot at benchmark
        }

    # ── public API ─────────────────────────────────────────────────────

    def pick(self, role: str, task: str, complexity: float = 0.5) -> tuple[str, dict]:
        """
        Returns (model_string, ollama_options_dict).
        Caller should pass options to litellm as extra kwargs.
        """
        with self._lock:
            self._expire_stale()

            if role in ("helper", "shopkeeper", "worker"):
                return self._use(FAST_MODEL)

            if role == "memory":
                # openclaw-brain-v2 has trained weights — use it on GPU
                # It can now load since small models vacated VRAM
                if self._gpu_fits(SYSTEM_MODEL):
                    return self._use(SYSTEM_MODEL)
                # If something heavy is blocking, fall back to CPU llama3.1
                return self._use(SYSTEM_FALLBACK)

            if role == "coder":
                if complexity < 0.3:
                    return self._use(FAST_MODEL)
                return self._use(CODER_MODEL)

            return self._use(FAST_MODEL)

    def record_use(self, model: str):
        prof = MODEL_PROFILES.get(model, {})
        if prof.get("vram_gb", 0) > 0:
            with self._lock:
                self._gpu_hot[model] = time.time()

    def ollama_options(self, model: str) -> dict:
        """Return the options dict to pass with every Ollama call."""
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
            used = sum(MODEL_PROFILES.get(m, {}).get("vram_gb", 0)
                       for m in gpu_models)
            return {
                "gpu_models":   gpu_models,
                "cpu_models":   [k for k, v in MODEL_PROFILES.items()
                                 if v.get("num_gpu", 0) == 0],
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
        used = sum(MODEL_PROFILES.get(m, {}).get("vram_gb", 0)
                   for m in self._gpu_hot)
        need = MODEL_PROFILES.get(model, {}).get("vram_gb", 0)
        return used + need <= P100_VRAM_GB

    def _use(self, model: str) -> tuple[str, dict]:
        prof = MODEL_PROFILES.get(model, {})
        if prof.get("vram_gb", 0) > 0:
            self._gpu_hot[model] = time.time()
        return model, self.ollama_options(model)


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
