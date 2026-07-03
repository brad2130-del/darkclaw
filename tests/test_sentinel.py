"""
Tests for sentinel/sentinel.py — signature normalization and the
sliding-window counter. No network, no journal.
"""
from sentinel.sentinel import SigCounter, normalize


def test_normalize_collapses_variable_parts():
    a = normalize("connect to 10.0.0.7:11434 failed after 3 retries")
    b = normalize("connect to 10.0.0.9:11434 failed after 17 retries")
    assert a == b
    assert "#" in a and "10.0.0.7" not in a


def test_normalize_hex_uuid_whitespace():
    s = normalize("worker  a1b2c3d4-e5f6-7890-abcd-ef0123456789 died at 0xDEADBEEF")
    assert "<uuid>" in s
    assert "<hex>" in s
    assert "  " not in s


def test_counter_reports_at_threshold_once():
    c = SigCounter(window_s=60, threshold=3, cooldown_s=3600)
    assert c.hit("x", now=0) == 0
    assert c.hit("x", now=1) == 0
    assert c.hit("x", now=2) == 3          # threshold crossed → report
    assert c.hit("x", now=3) == 0          # cooldown: no re-report spam


def test_counter_window_expires_old_hits():
    c = SigCounter(window_s=10, threshold=3, cooldown_s=3600)
    c.hit("x", now=0)
    c.hit("x", now=1)
    # first two hits fell out of the window — this is hit #1 again
    assert c.hit("x", now=100) == 0


def test_counter_rereports_after_cooldown():
    c = SigCounter(window_s=1000, threshold=2, cooldown_s=50)
    c.hit("x", now=0)
    assert c.hit("x", now=1) == 2
    c.hit("x", now=2)
    assert c.hit("x", now=3) == 0          # in cooldown
    assert c.hit("x", now=60) == 5         # cooldown over, still recurring


def test_counter_signatures_are_independent():
    c = SigCounter(window_s=60, threshold=2, cooldown_s=3600)
    c.hit("a", now=0)
    assert c.hit("b", now=1) == 0
    assert c.hit("a", now=2) == 2
