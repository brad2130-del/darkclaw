#!/usr/bin/env python3
"""wiki_watchdog — keep the openclaw-wiki pipeline from stalling silently.

The pipeline is a chain, and on 2026-08-04 three of its links broke at once
while nothing complained:

    Pi5 (daddypi)  ->  BlackArmor NAS  ->  proxmox-ha re-export
                                              |
                                              v
                                    Acer /mnt/blackarmor-wiki
                                              |
                                              v
                              CT 135 second-brain (wiki-ingest.timer)

The outage was found by hand, two days late, by noticing the newest wiki file
was stale. That staleness check is the one signal that catches a break at ANY
layer, so it is the primary probe here; the per-layer probes exist to say
*which* link went down once staleness has told us that something did.

WHAT IT HEALS AND WHAT IT WILL NOT.

Two failures are safe to repair automatically because they are idempotent and
touch only this machine's view of the world:

  - a stale/absent NFS mount  -> re-trigger the automount by listing the dir
  - a wedged automount unit   -> systemctl restart the .automount unit

Everything else is REPORTED, NOT TOUCHED. Re-exporting on the NAS, restarting
CT 135, or power-cycling the Pi are cross-machine actions on shared infra; a
watchdog that retries those on a timer can turn one dead link into a flapping
cluster. Those become DarkClaw reports for a human to act on.

Reports go to DarkClaw's /api/sentinel/report, the same endpoint sentinel.py
uses, so wiki alerts land in the existing escalate + LLM-analysis path rather
than a second notification channel. Each distinct signature reports at most
once per cooldown (default 6h) — a pipeline that is down stays down for hours,
and re-alerting every 15 minutes trains you to ignore it.

Env knobs (all optional):
  WIKI_WATCHDOG_DARKCLAW_URL  report target      (default http://127.0.0.1:7430)
  WIKI_WATCHDOG_PATH          wiki mount         (default /mnt/blackarmor-wiki)
  WIKI_WATCHDOG_MAX_AGE_H     staleness ceiling  (default 36)
  WIKI_WATCHDOG_COOLDOWN_S    re-report cooldown (default 21600 = 6h)
  WIKI_WATCHDOG_STATE         state file         (default ~/.local/state/wiki-watchdog.json)
  WIKI_WATCHDOG_NO_HEAL       set to 1 to probe/report only, never remount
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

DARKCLAW_URL = os.environ.get("WIKI_WATCHDOG_DARKCLAW_URL",
                              "http://127.0.0.1:7430").rstrip("/")
WIKI_PATH    = Path(os.environ.get("WIKI_WATCHDOG_PATH", "/mnt/blackarmor-wiki"))
MAX_AGE_H    = float(os.environ.get("WIKI_WATCHDOG_MAX_AGE_H", "36"))
COOLDOWN_S   = float(os.environ.get("WIKI_WATCHDOG_COOLDOWN_S", "21600"))
STATE_PATH   = Path(os.environ.get(
    "WIKI_WATCHDOG_STATE",
    Path.home() / ".local/state/wiki-watchdog.json"))
NO_HEAL      = os.environ.get("WIKI_WATCHDOG_NO_HEAL") == "1"

# The systemd automount backing WIKI_PATH. Derived rather than hardcoded so a
# different WIKI_WATCHDOG_PATH still restarts the right unit.
AUTOMOUNT_UNIT = str(WIKI_PATH).strip("/").replace("-", r"\x2d").replace("/", "-") \
                 + ".automount"

# Second Brain's health endpoint — the consumer end of the pipeline.
BRAIN_URL = os.environ.get("WIKI_WATCHDOG_BRAIN_URL",
                           "http://192.168.1.132:8000").rstrip("/")

# Subdir whose files are written on a daily cadence; the most sensitive
# staleness signal we have, since a gap here means the writer stopped.
DAILY_SUBDIR = "Daily-Reports"


def now() -> float:
    return time.time()


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── state (report cooldowns) ───────────────────────────────────────────

def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict):
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, separators=(",", ":")))
        tmp.replace(STATE_PATH)
    except OSError as e:
        print(f"[wiki-watchdog] could not persist state: {e}", flush=True)


def should_report(state: dict, signature: str) -> bool:
    """True when this signature is not inside its re-report cooldown."""
    last = state.get("reported", {}).get(signature)
    return last is None or (now() - last) >= COOLDOWN_S


def claim_report(state: dict, signature: str):
    state.setdefault("reported", {})[signature] = now()


def clear_report(state: dict, signature: str):
    """Drop a signature's cooldown so a recurrence alerts immediately."""
    state.get("reported", {}).pop(signature, None)


# ── reporting ──────────────────────────────────────────────────────────

def report(signature: str, detail: str, healed: bool = False):
    """File one report with DarkClaw. Delivery failure is non-fatal — the
    cooldown is already claimed, so we log and move on rather than spin."""
    try:
        httpx.post(f"{DARKCLAW_URL}/api/sentinel/report", json={
            "source": "wiki-watchdog",
            "signature": signature,
            "count": 1,
            "window_s": int(COOLDOWN_S),
            "sample": detail[:400],
        }, timeout=10)
        print(f"[wiki-watchdog] reported: {signature} — {detail}", flush=True)
    except Exception as e:
        print(f"[wiki-watchdog] report failed ({e}): {signature} — {detail}",
              flush=True)


# ── probes ─────────────────────────────────────────────────────────────

def is_mounted() -> bool:
    """True when WIKI_PATH is a real mountpoint (not just an empty dir)."""
    try:
        return subprocess.run(["mountpoint", "-q", str(WIKI_PATH)],
                              timeout=15).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        # A timeout here IS the symptom: a hung NFS mount blocks stat().
        return False


def readable() -> tuple[bool, str]:
    """Can we actually list the mount? Catches stale handles, which present
    as EACCES/ESTALE even to root while the mount still 'exists'."""
    try:
        entries = subprocess.run(["ls", "-1", str(WIKI_PATH)],
                                 capture_output=True, text=True, timeout=20)
        if entries.returncode != 0:
            return False, (entries.stderr or "ls failed").strip()
        return True, f"{len(entries.stdout.split())} entries"
    except subprocess.TimeoutExpired:
        return False, "timed out listing (mount is hung)"
    except OSError as e:
        return False, str(e)


def newest_file(root: Path) -> tuple[Path | None, float]:
    """Newest regular file under root, by mtime. Walks rather than globs so a
    writer that only touches a nested subdir still counts as activity."""
    newest, newest_ts = None, 0.0
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                p = Path(dirpath) / fn
                try:
                    ts = p.stat().st_mtime
                except OSError:
                    continue
                if ts > newest_ts:
                    newest, newest_ts = p, ts
    except OSError:
        pass
    return newest, newest_ts


def brain_healthy() -> tuple[bool, str]:
    """Second Brain up and ready — the ingest half of the pipeline."""
    try:
        r = httpx.get(f"{BRAIN_URL}/health", timeout=8)
        body = r.json()
        if r.status_code == 200 and body.get("ready"):
            return True, f"{body.get('memories_loaded', '?')} memories"
        return False, f"HTTP {r.status_code} {str(body)[:120]}"
    except Exception as e:
        return False, f"{e.__class__.__name__}: {e}"


# ── heal ───────────────────────────────────────────────────────────────

def heal_mount() -> tuple[bool, str]:
    """Re-establish the wiki mount. Both steps are idempotent and local to
    this machine — see the module docstring for why nothing here reaches
    across to the NAS, CT 135, or the Pi."""
    if NO_HEAL:
        return False, "healing disabled (WIKI_WATCHDOG_NO_HEAL=1)"

    # 1. Simply touching the path is enough when the automount is merely idle;
    #    systemd mounts on access. Cheapest possible repair, so try it first.
    try:
        subprocess.run(["ls", str(WIKI_PATH)], capture_output=True, timeout=25)
    except (subprocess.TimeoutExpired, OSError):
        pass
    if is_mounted():
        ok, detail = readable()
        if ok:
            return True, f"automount re-triggered on access ({detail})"

    # 2. Otherwise the unit itself is wedged — restart it. Needs root; if we
    #    are unprivileged this fails cleanly and becomes a report instead.
    if shutil.which("systemctl"):
        try:
            r = subprocess.run(
                ["systemctl", "restart", AUTOMOUNT_UNIT],
                capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                return False, f"restart {AUTOMOUNT_UNIT} failed: " \
                              f"{(r.stderr or '').strip()[:160]}"
        except (subprocess.TimeoutExpired, OSError) as e:
            return False, f"restart {AUTOMOUNT_UNIT} errored: {e}"
        try:
            subprocess.run(["ls", str(WIKI_PATH)], capture_output=True, timeout=25)
        except (subprocess.TimeoutExpired, OSError):
            pass
        if is_mounted():
            ok, detail = readable()
            if ok:
                return True, f"restarted {AUTOMOUNT_UNIT} ({detail})"

    return False, "mount still unavailable after remount attempts"


# ── main check ─────────────────────────────────────────────────────────

def check(verbose: bool = False) -> int:
    """Run every probe. Returns 0 when the pipeline is healthy, 1 otherwise.

    Ordered outermost-inward: a dead mount makes staleness unknowable, so we
    never report 'wiki is stale' when the real answer is 'we cannot see it'.
    """
    state = load_state()
    problems = []

    def note(signature: str, detail: str, healed: bool = False):
        problems.append(signature)
        if healed:
            # A self-heal is worth recording but must not consume the cooldown:
            # if it breaks again tomorrow we want to hear about it again.
            print(f"[wiki-watchdog] HEALED {signature} — {detail}", flush=True)
            clear_report(state, signature)
            return
        if should_report(state, signature):
            claim_report(state, signature)
            report(signature, detail)
        elif verbose:
            print(f"[wiki-watchdog] {signature} — {detail} (in cooldown)",
                  flush=True)

    # --- layer 1: is the mount there and readable? ---
    mount_ok = is_mounted()
    read_ok, read_detail = (readable() if mount_ok else (False, "not mounted"))

    if not (mount_ok and read_ok):
        healed, heal_detail = heal_mount()
        if healed:
            note("wiki mount was down, self-healed", heal_detail, healed=True)
            mount_ok, (read_ok, read_detail) = True, readable()
        else:
            note("wiki mount unavailable",
                 f"{WIKI_PATH}: {read_detail}; heal: {heal_detail}")
            # Staleness is unknowable without the mount — stop here rather
            # than emit a second, misleading alert about file age.
            save_state(state)
            return 1
    elif verbose:
        print(f"[wiki-watchdog] mount OK ({read_detail})", flush=True)

    # --- layer 2: is anything actually being written? ---
    # This is the signal that would have caught the Aug-4 outage on day one.
    daily = WIKI_PATH / DAILY_SUBDIR
    probe_root = daily if daily.is_dir() else WIKI_PATH
    newest, newest_ts = newest_file(probe_root)
    if newest is None:
        note("wiki has no files", f"{probe_root} is empty or unreadable")
    else:
        age_h = (now() - newest_ts) / 3600
        if age_h > MAX_AGE_H:
            note("wiki content stale",
                 f"newest file {newest.name} is {age_h:.1f}h old "
                 f"(ceiling {MAX_AGE_H:.0f}h, mtime {iso(newest_ts)}) — "
                 f"the writer (Pi5 daddypi) or an upstream link has stopped")
        elif verbose:
            print(f"[wiki-watchdog] freshest: {newest.name} "
                  f"({age_h:.1f}h old)", flush=True)

    # --- layer 3: is the consumer alive to ingest it? ---
    brain_ok, brain_detail = brain_healthy()
    if not brain_ok:
        note("second-brain unreachable",
             f"{BRAIN_URL}/health: {brain_detail} — CT 135 may be stopped; "
             f"wiki pages will not be ingested until it returns")
    elif verbose:
        print(f"[wiki-watchdog] second-brain OK ({brain_detail})", flush=True)

    state["last_check"] = now()
    state["last_result"] = "ok" if not problems else "; ".join(problems)
    save_state(state)

    if not problems:
        if verbose:
            print("[wiki-watchdog] pipeline healthy", flush=True)
        return 0
    return 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every probe result, not just failures")
    ap.add_argument("--status", action="store_true",
                    help="show last recorded result and active cooldowns")
    args = ap.parse_args()

    if args.status:
        state = load_state()
        last = state.get("last_check")
        print(f"last check : {iso(last) if last else 'never'}")
        print(f"last result: {state.get('last_result', 'unknown')}")
        cooling = state.get("reported", {})
        if cooling:
            print("cooldowns:")
            for sig, ts in sorted(cooling.items()):
                left = max(0, COOLDOWN_S - (now() - ts)) / 3600
                print(f"  {sig:<40} {left:.1f}h left")
        else:
            print("cooldowns : none")
        return 0

    return check(verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
