"""Curator tests — deterministic passes, SLM refinement, idempotency."""
import json
import time

import pytest

from core.curator import Curator
from memory.darkclaw_core import DarkclawEngine


@pytest.fixture
def engine(tmp_path):
    return DarkclawEngine(db_path=str(tmp_path / "test.db"))


def _active(engine, agent="a1"):
    return engine.persistence.load_active_facts(agent)


def test_expire_old_rag_breadcrumbs(engine):
    engine.ingest_fact("a1", "rag:abc", "RAG_RESPONSE", "some old blob")
    engine.ingest_fact("a1", "Proxmox", "MENTIONED_IN", "rag:abc")
    # Age them past the TTL
    engine.persistence._conn.execute(
        "UPDATE facts SET timestamp = ?", (time.time() - 30 * 86400,))
    engine.persistence._conn.commit()

    report = Curator(engine).run_cycle(with_slm=False)
    assert report["expired"] == 2
    assert _active(engine) == []


def test_fresh_rag_breadcrumbs_survive(engine):
    engine.ingest_fact("a1", "rag:xyz", "RAG_RESPONSE", "fresh blob")
    report = Curator(engine).run_cycle(with_slm=False)
    assert report["expired"] == 0
    assert len(_active(engine)) == 1


def test_dedupe_keeps_newest(engine):
    engine.ingest_fact("a1", "Ollama", "RUNS_ON", "T5810")
    time.sleep(0.01)
    engine.ingest_fact("a1", "ollama", "RUNS_ON", "t5810")   # same, normalized
    report = Curator(engine).run_cycle(with_slm=False)
    # one survives (dedupe or single-valued resolve may both claim it)
    assert report["deduped"] + report["resolved"] == 1
    assert len(_active(engine)) == 1


def test_single_valued_conflict_newest_wins(engine):
    engine.ingest_fact("a1", "Business_License", "HAS_STATUS", "Pending")
    time.sleep(0.01)
    engine.ingest_fact("a1", "Business_License", "HAS_STATUS", "Filed")
    Curator(engine).run_cycle(with_slm=False)
    facts = _active(engine)
    status = [f for f in facts if f["predicate"] == "HAS_STATUS"]
    assert len(status) == 1
    assert status[0]["object"] == "Filed"


def test_multi_valued_predicates_untouched(engine):
    engine.ingest_fact("a1", "Router", "ROUTES_TO", "sage")
    engine.ingest_fact("a1", "Router", "ROUTES_TO", "kit")
    report = Curator(engine).run_cycle(with_slm=False)
    assert report["resolved"] == 0
    assert len(_active(engine)) == 2


def test_refine_consolidates_crude_facts(engine):
    for obj in ("a hypervisor on the Lenovo", "running Proxmox VE",
                "the box hosting CT 130"):
        engine.ingest_fact("a1", "Lenovo_M910", "IS", obj)

    def fake_llm(prompt):
        assert "Lenovo_M910" in prompt
        return json.dumps({"triples": [
            {"predicate": "IS_A", "object": "Proxmox hypervisor"},
            {"predicate": "RUNS_ON", "object": "Lenovo M910"},
            {"predicate": "BAD_PREDICATE", "object": "should be dropped"},
        ]})

    report = Curator(engine, llm_fn=fake_llm).run_cycle()
    assert report["refined"] == 3          # crude facts superseded
    assert report["created"] == 2          # only whitelisted predicates land
    preds = {f["predicate"] for f in _active(engine)}
    assert preds == {"IS_A", "RUNS_ON"}


def test_refine_survives_dead_slm(engine):
    for obj in ("x1", "x2", "x3"):
        engine.ingest_fact("a1", "Thing", "IS", obj + " value here")

    def dead_llm(prompt):
        raise ConnectionError("memory node down")

    report = Curator(engine, llm_fn=dead_llm).run_cycle()
    assert report["refined"] == 0 and report["created"] == 0
    assert len(_active(engine)) == 3       # untouched


def test_cycle_is_idempotent(engine):
    engine.ingest_fact("a1", "Ollama", "RUNS_ON", "T5810")
    engine.ingest_fact("a1", "Ollama", "RUNS_ON", "T5810")
    engine.ingest_fact("a1", "Job", "HAS_STATUS", "old")
    engine.ingest_fact("a1", "Job", "HAS_STATUS", "new")

    c = Curator(engine)
    first = c.run_cycle(with_slm=False)
    assert first["deduped"] + first["resolved"] > 0
    second = c.run_cycle(with_slm=False)
    assert second["expired"] == second["deduped"] == second["resolved"] == 0


def test_queries_stop_seeing_superseded_facts(engine):
    engine.ingest_fact("a1", "Server", "HAS_STATUS", "offline")
    time.sleep(0.01)
    engine.ingest_fact("a1", "Server", "HAS_STATUS", "online")
    Curator(engine).run_cycle(with_slm=False)
    res = engine.query("what is the status of the Server?", "a1")
    assert "online" in res.answer


def test_errored_agent_rejoins_routing_after_cooldown():
    """One backend hiccup must not fail traffic over to the API forever."""
    import time as _t
    from core.orchestrator import Orchestrator

    orc = Orchestrator.bootstrap(with_default_agents=True)
    sage = orc.agents["sage"]
    sage.status = "error"

    sage._error_ts = _t.time()               # just failed
    assert orc._route("how is the gpu health?") != "sage"

    sage._error_ts = _t.time() - 300         # cooldown elapsed
    assert orc._route("how is the gpu health?") == "sage"
