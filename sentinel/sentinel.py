"""
darkclaw/sentinel/sentinel.py

Fleet sentinel — the 24/7 secretary. Watches the systemd journal and
probes the inference fleet, counting normalized error signatures in a
sliding window. When a signature crosses the threshold ("I've seen X
error N times") it files ONE report with DarkClaw, which escalates it
for LLM analysis. The sentinel itself never calls a model: detection is
deterministic and nearly free; intelligence is spent only on analysis,
and only once per signature per cooldown.

Runs as a root systemd service on the always-on node (Lenovo CT 130,
next to the DarkClaw production instance it reports to).

Env knobs (all optional):
  SENTINEL_DARKCLAW_URL   report target      (default http://127.0.0.1:7430)
  SENTINEL_WINDOW_S       sliding window     (default 900 = 15 min)
  SENTINEL_THRESHOLD      hits to report     (default 3)
  SENTINEL_COOLDOWN_S     re-report cooldown (default 3600)
  SENTINEL_PROBE_S        fleet probe period (default 60)
"""
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections import deque

import httpx

DARKCLAW_URL = os.environ.get("SENTINEL_DARKCLAW_URL", "http://127.0.0.1:7430").rstrip("/")
WINDOW_S     = float(os.environ.get("SENTINEL_WINDOW_S", "900"))
THRESHOLD    = int(os.environ.get("SENTINEL_THRESHOLD", "3"))
COOLDOWN_S   = float(os.environ.get("SENTINEL_COOLDOWN_S", "3600"))
PROBE_S      = float(os.environ.get("SENTINEL_PROBE_S", "60"))

HOSTNAME = socket.gethostname()

# journal lines worth counting even at warning priority
_INTERESTING = re.compile(
    r"error|fail|traceback|refused|timeout|timed out|denied|critical"
    r"|panic|oom|segfault|corrupt|unreachable", re.I)

# noise that recurs forever and means nothing actionable
_IGNORE = re.compile(r"sentinel|audit\[|pam_unix|session (opened|closed)", re.I)


def normalize(line: str) -> str:
    """
    Collapse a log line to a stable signature: hex, uuids, numbers and
    whitespace runs become placeholders so 'refused to 10.0.0.7:11434'
    and 'refused to 10.0.0.9:11434' count as the same failure.
    """
    s = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
               "<uuid>", line, flags=re.I)
    s = re.sub(r"0x[0-9a-f]+", "<hex>", s, flags=re.I)
    s = re.sub(r"\d+", "#", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:160]


class SigCounter:
    """Sliding-window signature counter with per-signature report cooldown."""

    def __init__(self, window_s: float = WINDOW_S, threshold: int = THRESHOLD,
                 cooldown_s: float = COOLDOWN_S):
        self.window_s = window_s
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._lock = threading.Lock()
        self._hits: dict[str, deque] = {}
        self._reported: dict[str, float] = {}

    def hit(self, sig: str, now: float = None) -> int:
        """
        Record one occurrence. Returns the in-window count when this hit
        crosses the threshold and the signature isn't in report cooldown —
        i.e. a nonzero return means "report now". Returns 0 otherwise.
        """
        now = now if now is not None else time.time()
        with self._lock:
            q = self._hits.setdefault(sig, deque())
            q.append(now)
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) < self.threshold:
                return 0
            last = self._reported.get(sig)
            if last is not None and now - last < self.cooldown_s:
                return 0
            self._reported[sig] = now
            return len(q)


def report(source: str, signature: str, count: int, sample: str = ""):
    """File one report with DarkClaw. Failure to deliver is non-fatal —
    the cooldown was already claimed, so we log and move on."""
    try:
        httpx.post(f"{DARKCLAW_URL}/api/sentinel/report", json={
            "source": source,
            "signature": signature,
            "count": count,
            "window_s": int(WINDOW_S),
            "sample": sample[:400],
        }, timeout=10)
        print(f"[sentinel] reported: {signature!r} x{count}", flush=True)
    except Exception as e:
        print(f"[sentinel] report failed ({e}): {signature!r}", flush=True)


# ── journal watcher ────────────────────────────────────────────────────

def journal_loop(counter: SigCounter):
    """Follow the journal; count err-priority lines and interesting warnings."""
    proc = subprocess.Popen(
        ["journalctl", "-f", "-p", "warning", "-o", "json", "--no-pager", "-n", "0"],
        stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        msg = entry.get("MESSAGE") or ""
        if not isinstance(msg, str) or not msg or _IGNORE.search(msg):
            continue
        try:
            pri = int(entry.get("PRIORITY", 6))
        except (TypeError, ValueError):
            pri = 6
        if pri > 3 and not _INTERESTING.search(msg):
            continue
        unit = entry.get("_SYSTEMD_UNIT") or entry.get("SYSLOG_IDENTIFIER") or "?"
        sig = f"{unit}: {normalize(msg)}"
        n = counter.hit(sig)
        if n:
            report(f"journal:{HOSTNAME}", sig, n, sample=msg)


# ── fleet prober ───────────────────────────────────────────────────────

def probe_loop(counter: SigCounter):
    """Reachability + config-drift probes across the inference fleet."""
    from core.fleet import FleetRegistry
    from core.model_router import expected_placement
    reg = FleetRegistry()
    while True:
        for name, url in reg.nodes.items():
            if reg.models_on(url, fresh=True) is None:
                sig = f"probe: node {name} unreachable"
                n = counter.hit(sig)
                if n:
                    report("sentinel-probe", sig, n, sample=url)
        try:
            for f in reg.preflight(expected_placement()):
                sig = f"probe: {f['model']} {f['issue']} ({f['node']})"
                n = counter.hit(sig)
                if n:
                    report("sentinel-probe", sig, n, sample=str(f))
        except Exception as e:
            print(f"[sentinel] preflight error: {e}", flush=True)
        try:
            httpx.get(f"{DARKCLAW_URL}/health", timeout=5)
        except Exception:
            # can't report darkclaw-down TO darkclaw; log for the journal
            # (another node's sentinel probing this host would catch it)
            print("[sentinel] darkclaw unreachable", flush=True)
        time.sleep(PROBE_S)


def main():
    print(f"[sentinel] up on {HOSTNAME} → {DARKCLAW_URL} "
          f"(window {WINDOW_S:.0f}s, threshold {THRESHOLD}, "
          f"cooldown {COOLDOWN_S:.0f}s)", flush=True)
    counter = SigCounter()
    threads = [
        threading.Thread(target=journal_loop, args=(counter,), daemon=True),
        threading.Thread(target=probe_loop, args=(counter,), daemon=True),
    ]
    for t in threads:
        t.start()
    while all(t.is_alive() for t in threads):
        time.sleep(5)
    print("[sentinel] a watcher thread died — exiting for systemd restart",
          flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
