"""Tests for the ChromaDB-backed VectorStore in rag/embed_store.py (M5/M6)."""

import pytest

from rag.embed_store import VectorStore
from rag.ingest import Chunk

CHUNKS = [
    Chunk(chunk_id="a::0", doc_title="Diabetes", text="Diabetes is a chronic disease affecting blood sugar levels.", source_org="WHO"),
    Chunk(chunk_id="b::0", doc_title="Influenza", text="Influenza is a viral infection causing fever, cough, and fatigue.", source_org="CDC"),
    Chunk(chunk_id="c::0", doc_title="Hypertension", text="Hypertension, or high blood pressure, increases risk of heart disease.", source_org="NIH"),
]


@pytest.fixture
def store(tmp_path) -> VectorStore:
    """A VectorStore backed by a throwaway persist directory, so tests never
    touch the real chroma_db/ index built by scripts/build_index.py."""
    s = VectorStore(persist_dir=str(tmp_path / "chroma"))
    s.build(CHUNKS)
    return s


def test_build_persists_all_chunks(store: VectorStore) -> None:
    assert store.count() == len(CHUNKS)


def test_query_before_build_raises(tmp_path) -> None:
    empty_store = VectorStore(persist_dir=str(tmp_path / "empty_chroma"))
    with pytest.raises(RuntimeError):
        empty_store.query("diabetes symptoms")


def test_query_returns_top_k_ranked_by_similarity(store: VectorStore) -> None:
    results = store.query("What causes high blood pressure?", top_k=2)
    assert len(results) == 2
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    top_chunk, top_score = results[0]
    assert top_chunk.doc_title == "Hypertension"
    assert -1.0 <= top_score <= 1.0


def test_query_preserves_chunk_metadata(store: VectorStore) -> None:
    results = store.query("blood sugar disease", top_k=1)
    chunk, _ = results[0]
    assert chunk.doc_title == "Diabetes"
    assert chunk.source_org == "WHO"
    assert chunk.source_url is None


def test_semantic_match_beats_keyword_mismatch(store: VectorStore) -> None:
    """A paraphrased query with no shared keywords should still retrieve the
    right chunk first — the reason we moved off TF-IDF."""
    results = store.query("flu-like symptoms and body temperature", top_k=1)
    assert results[0][0].doc_title == "Influenza"


def test_rebuild_replaces_prior_index(tmp_path) -> None:
    """build() should fully replace the index, not append to it."""
    s = VectorStore(persist_dir=str(tmp_path / "chroma"))
    s.build(CHUNKS)
    s.build(CHUNKS[:1])
    assert s.count() == 1
