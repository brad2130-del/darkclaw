"""
darkclaw/core/model_router.py

P100-aware dynamic model router.

The Tesla P100 (16GB HBM2) on the PVE homelab cannot hold all models
simultaneously. This router tracks estimated VRAM occupancy and makes
routing decisions that minimize cold-load latency.

Benchmark results (2026-06-28):
  phi3.5:latest                    cold=10.7s  eval=195ms   size≈2.2GB
  llama3.2:latest                  cold=242ms  eval=281ms   size≈2.0GB  ← always hot
  llama3.1:latest                  cold=20.1s  eval=634ms   size≈4.7GB
  openclaw-brain-v2:latest         OOM when deepseek resident          size≈?GB
  deepseek-coder-v2:16b-lite-q4   cold=138ms  eval=225ms   size≈9.5GB  ← hot for coding

Strategy:
  FAST tier  (≤2GB, coexist with anything):  llama3.2
  MEDIUM tier (4-5GB):                       llama3.1, openclaw-brain-v2
  HEAVY tier  (≥9GB):                        deepseek-coder-v2

Rules:
  • Rosie, Kit, Bea → always llama3.2 (instant, never evicts anything)
  • Sage (system queries) → openclaw-brain-v2 when deepseek is idle,
    else llama3.1 as fallback (avoids OOM)
  • Coder → deepseek-coder when complexity warrants it; llama3.2 for
    trivial snippets to avoid unnecessary load
"""
import time
import threading
from typing import Optional

# ── VRAM profile for each model (GB, estimated) ──────────────────────
MODEL_VRAM = {
    "ollama/phi3.5:latest":                                    2.2,
    "ollama/llama3.2:latest":                                  2.0,
    "ollama/llama3.1:latest":                                  4.7,
    "ollama/openclaw-brain-v2:latest":                         5.0,   # estimate
    "ollama/deepseek-coder-v2:16b-lite-instruct-q4_K_M":       9.5,
}

P100_VRAM_GB   = 16.0
KEEP_ALIVE_SEC = 300   # Ollama default keep_alive

# ── Fixed fast model (always hot, used as universal fallback) ─────────
FAST_MODEL   = "ollama/llama3.2:latest"
SYSTEM_MODEL = "ollama/openclaw-brain-v2:latest"
SYSTEM_FALLBACK = "ollama/llama3.1:latest"
CODER_MODEL  = "ollama/deepseek-coder-v2:16b-lite-instruct-q4_K_M"


class ModelRouter:
    """
    Tracks estimated VRAM state and routes agent tasks to the best
    available model without causing OOM evictions that spike latency.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # model → last_used timestamp; we evict oldest when VRAM full
        self._hot: dict[str, float] = {
            FAST_MODEL:  time.time(),    # always assume llama3.2 is hot
            CODER_MODEL: time.time(),    # deepseek was hot at benchmark
        }

    # ── public API ─────────────────────────────────────────────────────

    def pick(self, role: str, task: str, complexity: float = 0.5) -> str:
        """
        Return the best model string for a given agent role + task.

        complexity: 0.0 = trivial, 1.0 = deep reasoning needed
        """
        with self._lock:
            self._expire_stale()

            if role in ("helper", "shopkeeper", "worker"):
                # Always fast — llama3.2 stays hot
                return self._use(FAST_MODEL)

            if role == "memory":
                # Sage needs openclaw-brain-v2 (trained weights)
                # Only use it if deepseek is idle (won't cause OOM)
                if self._fits(SYSTEM_MODEL):
                    return self._use(SYSTEM_MODEL)
                # Deepseek is resident and VRAM is tight — use fallback
                return self._use(SYSTEM_FALLBACK)

            if role == "coder":
                if complexity < 0.3:
                    # Trivial snippet — don't burn deepseek load time
                    return self._use(FAST_MODEL)
                return self._use(CODER_MODEL)

            return self._use(FAST_MODEL)

    def record_use(self, model: str):
        """Call after a successful model response to refresh hot timestamp."""
        with self._lock:
            self._hot[model] = time.time()

    def vram_state(self) -> dict:
        """Returns current estimated VRAM occupancy for health endpoint."""
        with self._lock:
            self._expire_stale()
            hot = list(self._hot.keys())
            used = sum(MODEL_VRAM.get(m, 2.0) for m in hot)
            return {
                "hot_models":  hot,
                "vram_used_gb": round(used, 1),
                "vram_total_gb": P100_VRAM_GB,
                "vram_free_gb":  round(P100_VRAM_GB - used, 1),
            }

    # ── internals ──────────────────────────────────────────────────────

    def _expire_stale(self):
        now = time.time()
        self._hot = {m: t for m, t in self._hot.items()
                     if now - t < KEEP_ALIVE_SEC}

    def _fits(self, model: str) -> bool:
        used = sum(MODEL_VRAM.get(m, 2.0) for m in self._hot)
        return used + MODEL_VRAM.get(model, 2.0) <= P100_VRAM_GB

    def _use(self, model: str) -> str:
        self._hot[model] = time.time()
        return model


# ── Complexity scoring ─────────────────────────────────────────────────

def score_complexity(task: str) -> float:
    """
    Heuristic 0.0–1.0 complexity score from task text.
    Used to decide whether coder needs the heavy model or can use fast.
    """
    t = task.lower()
    score = 0.3   # baseline
    if len(task) > 200:                                      score += 0.2
    if any(k in t for k in ("debug", "refactor", "architect", "design")): score += 0.3
    if any(k in t for k in ("explain", "what is", "how does")):           score -= 0.1
    if t.count("```") > 0 or t.count("def ") > 0:           score += 0.2
    if any(k in t for k in ("fix", "bug", "error", "traceback")):         score += 0.1
    return max(0.0, min(1.0, score))


# Singleton
router = ModelRouter()
