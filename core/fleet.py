"""
darkclaw/core/fleet.py

Live fleet registry — which Ollama node actually serves which model, right now.

Why this exists: the July-2026 outages all traced to static config drifting
from reality (llama3.2 requests sent to a node that never had it, Pi5 bound
to loopback, openclaw-brain configured but installed nowhere). The registry
probes each node's /api/tags, caches briefly, and lets callers ask "who
really serves X?" instead of trusting MODEL_PROFILES.

Node failures get a cooldown (same pattern as the agent sticky-failover fix)
so one hiccup doesn't blacklist a node forever, but we also don't hammer a
dead node on every request.
"""
import os
import threading
import time

import httpx

PROBE_TIMEOUT = 3.0


def _norm(url: str) -> str:
    return (url or "").rstrip("/")


def _nodes_from_env() -> dict[str, str]:
    """
    name → url for every configured Ollama node; empty env vars drop out.
    Defaults must match core/model_router.py exactly — CT 130's fleet map
    once omitted pi5 while the drift check included it, which reads as a
    contradiction on the dashboard.
    """
    nodes = {}
    p100 = _norm(os.environ.get("OLLAMA_API_BASE") or os.environ.get("OLLAMA_BASE_URL", ""))
    mem  = _norm(os.environ.get("DARKCLAW_MEMORY_NODE_URL", "http://192.168.1.130:11434"))
    pi5  = _norm(os.environ.get("OPENCLAW_PI5_URL", "http://100.77.235.30:11434"))
    if p100:
        nodes["p100"] = p100
    if mem:
        nodes["memory-node"] = mem
    if pi5:
        nodes["pi5"] = pi5
    return nodes


class FleetRegistry:
    """
    Probes and caches the model inventory of every Ollama node.

    probe_fn is injectable for tests: (url) -> set[str] of model tags,
    raising on an unreachable node.
    """

    def __init__(self, nodes: dict[str, str] = None, probe_fn=None,
                 cache_ttl: float = None, node_cooldown: float = None):
        self.nodes = nodes if nodes is not None else _nodes_from_env()
        self._probe_fn = probe_fn or self._probe_http
        self.cache_ttl = cache_ttl if cache_ttl is not None else float(
            os.environ.get("DARKCLAW_FLEET_TTL", "60"))
        self.node_cooldown = node_cooldown if node_cooldown is not None else float(
            os.environ.get("DARKCLAW_NODE_COOLDOWN", "120"))
        self._lock = threading.Lock()
        # url → (probe_ts, models | None)   None = node was unreachable
        self._cache: dict[str, tuple[float, set | None]] = {}
        # url → ts of last failure (probe or completion)
        self._bad: dict[str, float] = {}

    # ── probing ─────────────────────────────────────────────────────────

    @staticmethod
    def _probe_http(url: str) -> set:
        r = httpx.get(f"{url}/api/tags", timeout=PROBE_TIMEOUT)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}

    def models_on(self, url: str, fresh: bool = False) -> set | None:
        """Model tags served by `url`, or None if unreachable. Cached."""
        url = _norm(url)
        now = time.time()
        with self._lock:
            hit = self._cache.get(url)
            if hit and not fresh and now - hit[0] < self.cache_ttl:
                return hit[1]
        try:
            models = self._probe_fn(url)
        except Exception:
            models = None
        with self._lock:
            self._cache[url] = (now, models)
            if models is None:
                self._bad[url] = now
            else:
                self._bad.pop(url, None)
        return models

    # ── health bookkeeping ──────────────────────────────────────────────

    def mark_bad(self, url: str):
        """Record a completion failure against a node and drop its cache."""
        url = _norm(url)
        with self._lock:
            self._bad[url] = time.time()
            self._cache.pop(url, None)

    def is_healthy(self, url: str) -> bool:
        with self._lock:
            t = self._bad.get(_norm(url))
        return t is None or time.time() - t > self.node_cooldown

    def name_of(self, url: str) -> str:
        url = _norm(url)
        for name, u in self.nodes.items():
            if u == url:
                return name
        return url or "default"

    # ── queries ─────────────────────────────────────────────────────────

    def locate(self, model: str, preferred: str = None, fresh: bool = False) -> list[str]:
        """
        Node urls that actually serve `model` (litellm "ollama/tag" or bare
        tag), preferred url first. Nodes in cooldown are skipped unless
        fresh=True re-probes them (a fresh probe that succeeds clears the
        cooldown, so recovery is automatic).
        """
        tag = model.split("/", 1)[-1]
        urls = list(self.nodes.values())
        if preferred:
            p = _norm(preferred)
            urls = [p] + [u for u in urls if u != p]
        found = []
        for u in urls:
            if not fresh and not self.is_healthy(u):
                continue
            models = self.models_on(u, fresh=fresh)
            if models and tag in models:
                found.append(u)
        return found

    def preflight(self, expected: dict[str, str]) -> list[dict]:
        """
        Config-drift check. expected: {model: assigned_url}. Returns one
        finding per model whose assigned node is down or missing it —
        exactly the two failure modes of 2026-07-02/03.
        """
        findings = []
        for model, url in expected.items():
            tag = model.split("/", 1)[-1]
            models = self.models_on(url, fresh=True)
            if models is None:
                findings.append({
                    "model": tag, "node": self.name_of(url),
                    "issue": "node unreachable",
                })
            elif tag not in models:
                findings.append({
                    "model": tag, "node": self.name_of(url),
                    "issue": "model missing on assigned node",
                    "available_on": [self.name_of(u) for u in self.locate(model)],
                })
        return findings

    def snapshot(self) -> dict:
        """Live fleet map for the /api/fleet endpoint and telemetry."""
        out = {}
        for name, url in self.nodes.items():
            models = self.models_on(url)
            out[name] = {
                "url": url,
                "healthy": models is not None and self.is_healthy(url),
                "models": sorted(models) if models else [],
            }
        return out


# Singleton — built from env at import, like model_router.router
fleet = FleetRegistry()
