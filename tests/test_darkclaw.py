"""
tests/test_darkclaw.py
Darkclaw memory accuracy benchmark.
All 8 tests designed and verified — wrap in pytest.

Run: python -m pytest tests/test_darkclaw.py -v
Or:  python memory/darkclaw_core.py  (built-in self-test)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from memory.darkclaw_core import DarkclawEngine

@pytest.fixture
def engine(tmp_path):
    return DarkclawEngine(db_path=str(tmp_path/"test.db"))

@pytest.fixture
def seeded_engine(engine):
    facts = [
        ("Darkclaw_Stack","RUNS_ON","T5810_Server",False),
        ("CT210_Ollama","RUNS_ON","T5810_Server",False),
        ("LiteLLM_Gateway","ROUTES_TO","CT210_Ollama",True),
        ("LiteLLM_Gateway","ROUTES_TO","Anthropic_API",True),
        ("SHOP_BOT","USES","Phi3_Mini",False),
        ("SHOP_BOT","RUNS_IN","CT131",False),
        ("Darkclaw","DEPENDS_ON","NetworkX",True),
        ("Darkclaw","DEPENDS_ON","ChromaDB",True),
        ("Book_Burrow","LLC_NUMBER","1580647",False),
        ("Book_Burrow","HAS_PRIORITY","high",False),
        ("Prosthetic_Arm","HAS_STATUS","v1.2_in_progress",False),
    ]
    for s,p,o,mv in facts:
        engine.ingest_fact("main",s,p,o,multi_valued=mv)
    engine.ingest_fact("main","Book_Burrow","HAS_PRIORITY","critical",multi_valued=False)
    return engine

def test_direct_lookup(seeded_engine):
    r = seeded_engine.query("What does Darkclaw depend on?","main")
    assert "NetworkX" in r.answer or "ChromaDB" in r.answer

def test_multi_valued_preserved(seeded_engine):
    r = seeded_engine.query("What does LiteLLM Gateway route to?","main")
    assert "CT210_Ollama" in r.answer and "Anthropic_API" in r.answer

def test_stale_fact_superseded(seeded_engine):
    r = seeded_engine.query("What is Book Burrow's current priority?","main")
    assert "critical" in r.answer.lower()
    assert "high" not in r.answer.lower()

def test_inverse_join(seeded_engine):
    r = seeded_engine.query("What agent uses Phi3 Mini?","main")
    assert "SHOP_BOT" in r.answer

def test_runs_on_direct(seeded_engine):
    r = seeded_engine.query("Which system does CT210 run on?","main")
    assert "T5810_Server" in r.answer

def test_llc_number(seeded_engine):
    r = seeded_engine.query("What is the LLC number for Book Burrow?","main")
    assert "1580647" in r.answer

def test_status_query(seeded_engine):
    r = seeded_engine.query("What is the status of the prosthetic arm?","main")
    assert "v1.2" in r.answer

def test_empty_query(engine):
    r = engine.query("What is the capital of France?","main")
    assert r.confidence == 0.0 or r.method == "fallback"
