"""Tests for the refusal guard and extractive mode in rag/generate.py (M7)."""

from rag.generate import NO_EVIDENCE_MESSAGE, extractive_answer, generate_answer
from rag.ingest import Chunk

CHUNK = Chunk(chunk_id="x::0", doc_title="Test Doc", text="Some retrieved text.", source_org="WHO")


def test_generate_answer_refuses_when_nothing_retrieved() -> None:
    assert generate_answer("anything", []) == NO_EVIDENCE_MESSAGE


def test_extractive_answer_refuses_when_nothing_retrieved() -> None:
    assert extractive_answer("anything", []) == NO_EVIDENCE_MESSAGE


def test_generate_answer_extractive_mode_uses_retrieved_chunks() -> None:
    answer = generate_answer("q", [(CHUNK, 0.9)], mode="extractive")
    assert "Test Doc" in answer
    assert "Some retrieved text." in answer


def test_generate_answer_does_not_refuse_when_evidence_exists() -> None:
    answer = generate_answer("q", [(CHUNK, 0.9)], mode="extractive")
    assert answer != NO_EVIDENCE_MESSAGE
