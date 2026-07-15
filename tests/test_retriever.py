"""Tests for the similarity-threshold retrieval guard in rag/retriever.py (M7)."""

import pytest

from rag.embed_store import VectorStore
from rag.ingest import Chunk
from rag.retriever import retrieve

CHUNKS = [
    Chunk(chunk_id="a::0", doc_title="Diabetes", text="Diabetes is a chronic disease affecting blood sugar levels.", source_org="WHO"),
    Chunk(chunk_id="b::0", doc_title="Influenza", text="Influenza is a viral infection causing fever, cough, and fatigue.", source_org="CDC"),
]


@pytest.fixture
def store(tmp_path) -> VectorStore:
    s = VectorStore(persist_dir=str(tmp_path / "chroma"))
    s.build(CHUNKS)
    return s


def test_retrieve_passes_relevant_chunks_with_lenient_threshold(store: VectorStore) -> None:
    results = retrieve(store, "What causes diabetes?", top_k=2, similarity_threshold=0.3)
    assert len(results) >= 1
    assert all(score >= 0.3 for _, score in results)


def test_retrieve_returns_empty_when_threshold_is_unreachable(store: VectorStore) -> None:
    results = retrieve(store, "What causes diabetes?", top_k=2, similarity_threshold=0.999)
    assert results == []


def test_retrieve_respects_top_k_before_filtering(store: VectorStore) -> None:
    results = retrieve(store, "medical condition", top_k=1, similarity_threshold=0.0)
    assert len(results) == 1


def test_retrieve_uses_config_defaults_when_not_overridden(store: VectorStore) -> None:
    # Should not raise, and should behave like an explicit call with config values.
    results = retrieve(store, "diabetes symptoms")
    assert isinstance(results, list)
