"""
Tests for the persistent embedding cache.

The bug this guards: embeddings were held only in RAM, so every process
start re-embedded every turn ever recorded (4k+ on the live box) over the
network on the first query. Embeddings are deterministic — that work was
pure waste, and it grew with the corpus.
"""
import memory.darkclaw_core as dc
from memory.darkclaw_core import DarkclawPersistence, VectorMemory


class FakeEmbedder:
    """Deterministic stand-in for the memory node; counts texts embedded."""

    def __init__(self):
        self.calls = 0
        self.texts_embedded = 0

    def __call__(self, texts, prefix):
        self.calls += 1
        self.texts_embedded += len(texts)
        # Cheap deterministic vector, distinct per text.
        return [[float(len(t)), float(sum(map(ord, t[:8]))), 1.0] for t in texts]


def _store(tmp_path):
    return DarkclawPersistence(str(tmp_path / "dc.db"))


def test_roundtrip_preserves_vectors(tmp_path):
    store = _store(tmp_path)
    store.save_embeddings([("k1", [0.5, -1.25, 3.0])], "test-model")
    got = store.get_embeddings(["k1", "missing"])
    assert "missing" not in got
    assert got["k1"] == [0.5, -1.25, 3.0]


def test_cache_key_changes_with_model(monkeypatch):
    k1 = dc._cache_key("hello", "search_document")
    monkeypatch.setattr(dc, "_EMBED_MODEL", "some-other-model")
    k2 = dc._cache_key("hello", "search_document")
    # A model swap must never serve a stale vector from the old model.
    assert k1 != k2


def test_cache_key_changes_with_prefix():
    # nomic-embed is asymmetric — a doc vector must not answer as a query vector.
    assert (dc._cache_key("hello", "search_document")
            != dc._cache_key("hello", "search_query"))


def test_second_process_embeds_nothing(tmp_path, monkeypatch):
    """The load-bearing test: a restart must not re-embed a warm corpus."""
    fake = FakeEmbedder()
    monkeypatch.setattr(dc, "_embed_batch", fake)
    store = _store(tmp_path)

    vm = VectorMemory(store=store)
    for text in ("the shop opens at nine", "kit runs the register", "sage knows the homelab"):
        vm.ingest(text)
    with vm._lock:
        assert vm._ensure_embedded_locked()
    assert fake.texts_embedded == 3          # cold: all three embedded

    # Simulate a restart: brand-new VectorMemory, same on-disk store.
    vm2 = VectorMemory(store=store)
    for text in ("the shop opens at nine", "kit runs the register", "sage knows the homelab"):
        vm2.ingest(text)
    with vm2._lock:
        assert vm2._ensure_embedded_locked()

    assert fake.texts_embedded == 3          # warm: nothing re-embedded
    assert vm2._emb == vm._emb               # and the vectors are identical


def test_only_new_chunks_embed(tmp_path, monkeypatch):
    fake = FakeEmbedder()
    monkeypatch.setattr(dc, "_embed_batch", fake)
    store = _store(tmp_path)

    vm = VectorMemory(store=store)
    vm.ingest("first")
    with vm._lock:
        vm._ensure_embedded_locked()
    assert fake.texts_embedded == 1

    vm.ingest("second")
    with vm._lock:
        vm._ensure_embedded_locked()
    assert fake.texts_embedded == 2          # only the new one
    assert len(vm._emb) == 2                 # still index-aligned with _chunks


def test_duplicate_text_embeds_once(tmp_path, monkeypatch):
    """Identical text shares a content key — it must not double-insert."""
    fake = FakeEmbedder()
    monkeypatch.setattr(dc, "_embed_batch", fake)
    store = _store(tmp_path)

    vm = VectorMemory(store=store)
    vm.ingest("same text")
    vm.ingest("same text")
    with vm._lock:
        assert vm._ensure_embedded_locked()

    assert len(vm._emb) == 2                 # both chunks get a vector
    assert vm._emb[0] == vm._emb[1]
    assert len(store.get_embeddings([dc._cache_key("same text", "search_document")])) == 1


def test_node_failure_still_degrades_to_tfidf(tmp_path, monkeypatch):
    """A dead memory node must not break retrieval — the fallback still holds."""
    monkeypatch.setattr(dc, "_embed_batch", lambda texts, prefix: None)
    store = _store(tmp_path)

    vm = VectorMemory(store=store)
    vm.ingest("book burrow sells used books")
    vm.ingest("the p100 runs inference")
    # TF-IDF is lexical, not semantic — the query must share literal terms.
    hits = vm.retrieve("used books", top_k=1)
    assert hits and "book burrow" in hits[0][0]
    assert vm._node_ok is False   # and it noticed the node was dead
