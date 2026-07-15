"""Tests for the embedding-based VectorStore in rag/embed_store.py (M5)."""

import numpy as np
import pytest

from rag.embed_store import VectorStore
from rag.ingest import Chunk

CHUNKS = [
    Chunk(chunk_id="a::0", doc_title="Diabetes", text="Diabetes is a chronic disease affecting blood sugar levels.", source_org="WHO"),
    Chunk(chunk_id="b::0", doc_title="Influenza", text="Influenza is a viral infection causing fever, cough, and fatigue.", source_org="CDC"),
    Chunk(chunk_id="c::0", doc_title="Hypertension", text="Hypertension, or high blood pressure, increases risk of heart disease.", source_org="NIH"),
]


@pytest.fixture(scope="module")
def store() -> VectorStore:
    s = VectorStore()
    s.build(CHUNKS)
    return s


def test_build_creates_normalized_embeddings(store: VectorStore) -> None:
    assert store.embeddings is not None
    assert store.embeddings.shape[0] == len(CHUNKS)
    norms = np.linalg.norm(store.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_query_before_build_raises() -> None:
    empty_store = VectorStore()
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


def test_semantic_match_beats_keyword_mismatch(store: VectorStore) -> None:
    """A paraphrased query with no shared keywords should still retrieve the
    right chunk first — the reason we moved off TF-IDF."""
    results = store.query("flu-like symptoms and body temperature", top_k=1)
    assert results[0][0].doc_title == "Influenza"
