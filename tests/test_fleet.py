"""
Tests for core/fleet.py + core/llm_call.py — the fleet-aware healing layer.

Fakes stand in for both the node probes and litellm, so these run with no
network and no Ollama.
"""
import pytest

from core.fleet import FleetRegistry
from core.llm_call import resilient_completion

NODES = {
    "p100":        "http://p100:11434",
    "memory-node": "http://mem:11434",
    "pi5":         "http://pi5:11434",
}

INVENTORY = {
    "http://p100:11434": {"openclaw-brain-v3:latest", "qwen2.5-coder:7b", "llama3.1:latest"},
    "http://mem:11434":  {"llama3.2:latest", "phi3.5:latest"},
    "http://pi5:11434":  {"llama3.2:1b"},
}


def make_registry(down=(), inventory=None):
    inv = inventory or INVENTORY
    def probe(url):
        if url in down:
            raise ConnectionError("connection refused")
        return set(inv[url])
    return FleetRegistry(nodes=dict(NODES), probe_fn=probe,
                         cache_ttl=0.0, node_cooldown=120.0)


class FakeResponse:
    def __init__(self, text="ok"):
        self.text = text


def make_completion(behavior):
    """
    behavior: {(model, api_base_or_None): exception-or-response}.
    Unlisted combos succeed. Records every attempt.
    """
    calls = []
    def completion(model, messages, stream=False, **kw):
        key = (model, kw.get("api_base"))
        calls.append(key)
        out = behavior.get(key, FakeResponse())
        if isinstance(out, Exception):
            raise out
        return out
    completion.calls = calls
    return completion


# ── FleetRegistry ──────────────────────────────────────────────────────

def test_locate_finds_only_nodes_serving_the_model():
    reg = make_registry()
    assert reg.locate("ollama/llama3.2:latest") == ["http://mem:11434"]
    assert reg.locate("ollama/nonexistent:latest") == []


def test_locate_puts_preferred_node_first():
    inv = dict(INVENTORY)
    inv["http://p100:11434"] = INVENTORY["http://p100:11434"] | {"llama3.2:latest"}
    reg = make_registry(inventory=inv)
    urls = reg.locate("ollama/llama3.2:latest", preferred="http://mem:11434")
    assert urls[0] == "http://mem:11434"
    assert "http://p100:11434" in urls


def test_down_node_enters_cooldown_and_is_skipped():
    reg = make_registry(down=("http://mem:11434",))
    assert reg.models_on("http://mem:11434") is None
    assert not reg.is_healthy("http://mem:11434")
    # cooldown skips it without fresh=True
    assert reg.locate("ollama/llama3.2:latest") == []


def test_preflight_flags_missing_model_and_down_node():
    reg = make_registry(down=("http://pi5:11434",))
    findings = reg.preflight({
        "ollama/llama3.2:latest": "http://p100:11434",   # drift: lives on mem
        "ollama/llama3.2:1b":     "http://pi5:11434",    # node down
        "ollama/qwen2.5-coder:7b": "http://p100:11434",  # fine
    })
    issues = {f["model"]: f["issue"] for f in findings}
    assert issues["llama3.2:latest"] == "model missing on assigned node"
    assert issues["llama3.2:1b"] == "node unreachable"
    assert "qwen2.5-coder:7b" not in issues
    drift = next(f for f in findings if f["model"] == "llama3.2:latest")
    assert drift["available_on"] == ["memory-node"]


# ── resilient_completion ───────────────────────────────────────────────

def test_success_on_primary_is_not_healed():
    reg = make_registry()
    fake = make_completion({})
    resp, info = resilient_completion(
        "ollama/llama3.2:latest", [], api_base="http://mem:11434",
        completion_fn=fake, registry=reg)
    assert info == {"model": "ollama/llama3.2:latest",
                    "node": "memory-node", "healed": False}
    assert fake.calls == [("ollama/llama3.2:latest", "http://mem:11434")]


def test_model_not_found_reroutes_to_node_that_has_it():
    """The exact 2026-07-02 /stream failure: llama3.2 sent to the P100."""
    reg = make_registry()
    fake = make_completion({
        ("ollama/llama3.2:latest", "http://p100:11434"):
            RuntimeError('OllamaException - {"error":"model \'llama3.2:latest\' not found"}'),
    })
    resp, info = resilient_completion(
        "ollama/llama3.2:latest", [], api_base="http://p100:11434",
        completion_fn=fake, registry=reg)
    assert info["healed"] is True
    assert info["node"] == "memory-node"


def test_node_down_falls_back_to_substitute_model(monkeypatch):
    """Pi5-style outage: the only node with the model refuses connections."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reg = make_registry(down=("http://pi5:11434",))
    fake = make_completion({
        ("ollama/llama3.2:1b", "http://pi5:11434"):
            ConnectionError("connection refused"),
    })
    resp, info = resilient_completion(
        "ollama/llama3.2:1b", [], api_base="http://pi5:11434",
        completion_fn=fake, registry=reg)
    assert info["healed"] is True
    # FAST_MODEL (llama3.2:latest) lives on the memory node
    assert info == {"model": "ollama/llama3.2:latest",
                    "node": "memory-node", "healed": True}
    assert not reg.is_healthy("http://pi5:11434")


def test_claude_fallback_when_no_ollama_node_works(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    reg = make_registry(down=tuple(NODES.values()))
    err = ConnectionError("connection refused")
    fake = make_completion({
        ("ollama/llama3.2:latest", "http://mem:11434"): err,
        ("ollama/llama3.2:latest", "http://p100:11434"): err,
        ("ollama/llama3.1:latest", "http://p100:11434"): err,
    })
    resp, info = resilient_completion(
        "ollama/llama3.2:latest", [], api_base="http://mem:11434",
        completion_fn=fake, registry=reg)
    assert info["model"] == "claude-haiku-4-5-20251001"
    assert info["node"] == "cloud"
    assert info["healed"] is True


def test_all_rungs_exhausted_raises_original_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reg = make_registry(down=tuple(NODES.values()))
    first = ConnectionError("connection refused (primary)")
    def always_fail(model, messages, stream=False, **kw):
        raise first if kw.get("api_base") == "http://mem:11434" else ConnectionError("down")
    resp = pytest.raises(ConnectionError, resilient_completion,
                         "ollama/llama3.2:latest", [],
                         api_base="http://mem:11434",
                         completion_fn=always_fail, registry=reg)
    assert "primary" in str(resp.value)


def test_cloud_model_calls_straight_through():
    reg = make_registry()
    fake = make_completion({})
    resp, info = resilient_completion(
        "claude-haiku-4-5-20251001", [], completion_fn=fake, registry=reg)
    assert info == {"model": "claude-haiku-4-5-20251001",
                    "node": "cloud", "healed": False}
    assert fake.calls == [("claude-haiku-4-5-20251001", None)]
