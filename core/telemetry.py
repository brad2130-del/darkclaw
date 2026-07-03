"""
darkclaw/core/telemetry.py

Live system telemetry for health questions.

Small local models will confidently invent GPU temperatures and RAM
numbers if asked about system health with nothing to ground them.
Instead of trusting the model, the orchestrator calls gather_health()
when a task looks like a health question and injects the result into
the system prompt as measured fact.

Every section fails soft: an unreachable node reports as unreachable
rather than raising, so a dead box never breaks the answer about the
live ones.
"""
import os
import shutil
import subprocess
import urllib.request
import json


_TIMEOUT = 3  # seconds per probe — health answers should stay snappy

OLLAMA_URL      = os.environ.get("OLLAMA_BASE_URL", "http://192.168.1.210:11434").rstrip("/")
MEMORY_NODE_URL = os.environ.get("DARKCLAW_MEMORY_NODE_URL", "http://192.168.1.130:11434").rstrip("/")


def _http_json(url: str):
    with urllib.request.urlopen(url, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode())


def _ollama_section(name: str, base: str) -> str:
    try:
        ps = _http_json(f"{base}/api/ps")
        models = ps.get("models", [])
        if not models:
            return f"{name}: online, no models loaded"
        lines = []
        for m in models:
            vram = m.get("size_vram", 0)
            size = m.get("size", 0)
            place = "GPU" if vram >= size * 0.9 else ("CPU" if vram == 0 else f"split {vram/1e9:.1f}GB GPU")
            lines.append(f"  {m['name']}: {size/1e9:.1f}GB loaded on {place}")
        return f"{name}: online\n" + "\n".join(lines)
    except Exception as e:
        return f"{name}: UNREACHABLE ({e.__class__.__name__})"


def _local_section() -> str:
    parts = []
    try:
        with open("/proc/loadavg") as f:
            parts.append(f"load {f.read().split()[0]}")
    except Exception:
        pass
    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                mem[k] = int(v.strip().split()[0])
        used = (mem["MemTotal"] - mem["MemAvailable"]) / 1024 / 1024
        parts.append(f"RAM {used:.1f}/{mem['MemTotal']/1024/1024:.1f}GB")
    except Exception:
        pass
    try:
        du = shutil.disk_usage("/")
        parts.append(f"disk {du.used/1e9:.0f}/{du.total/1e9:.0f}GB")
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=_TIMEOUT)
            if out.returncode == 0 and out.stdout.strip():
                parts.append("GPU " + out.stdout.strip().splitlines()[0])
        except Exception:
            pass
    return "this node: " + ", ".join(parts) if parts else "this node: no data"


def gather_health() -> str:
    """Return a compact plain-text snapshot of the whole inference fleet."""
    return "\n".join([
        _local_section(),
        _ollama_section("T5810/P100 Ollama (" + OLLAMA_URL + ")", OLLAMA_URL),
        _ollama_section("Lenovo memory node (" + MEMORY_NODE_URL + ")", MEMORY_NODE_URL),
    ])
