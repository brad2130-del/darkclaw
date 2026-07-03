"""Vision ingestion tests — images flow through the doc pipeline."""
import pytest

from core import vision
from core.doc_store import extract_text
from memory.darkclaw_core import DarkclawEngine


def test_image_routes_through_moondream(monkeypatch):
    monkeypatch.setattr(vision, "describe_image",
                        lambda raw, prompt=None: "a red bookshop storefront "
                                                 "with sign text Book Burrow")
    out = extract_text("shopfront.jpg", b"\xff\xd8fakejpegbytes")
    assert "Book Burrow" in out
    assert "moondream" in out


def test_dead_memory_node_degrades_softly(monkeypatch):
    def boom(raw, prompt=None):
        raise ConnectionError("node down")
    monkeypatch.setattr(vision, "describe_image", boom)
    out = extract_text("photo.png", b"\x89PNGfake")
    assert "failed" in out.lower()
    assert "ConnectionError" in out


def test_non_image_files_unaffected():
    out = extract_text("notes.txt", b"plain text still works")
    assert out == "plain text still works"


def test_image_description_lands_in_memory(monkeypatch, tmp_path):
    from core import doc_store
    monkeypatch.setattr(doc_store, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(doc_store, "META_FILE", tmp_path / "docs" / "meta.json")
    monkeypatch.setattr(vision, "describe_image",
                        lambda raw, prompt=None: "an invoice from Ingram for "
                                                 "42 paperback books, total $312.50")
    engine = DarkclawEngine(db_path=str(tmp_path / "t.db"))
    record = doc_store.ingest_document("invoice.png", b"\x89PNGfake", engine)
    assert record["chunks"] >= 1

    # The description text must be persisted in the docs namespace so the
    # vector tier can retrieve it (graph tier may rank summary facts first).
    rows = engine.persistence._conn.execute(
        "SELECT text FROM turns WHERE agent_id='docs'").fetchall()
    assert any("312.50" in (r[0] or "") for r in rows)
