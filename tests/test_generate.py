"""Tests for the refusal guard, extractive mode, and prompt wiring in
rag/generate.py (M7/M8)."""

from rag.generate import NO_EVIDENCE_MESSAGE, extractive_answer, generate_answer, llm_answer
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


def test_llm_answer_falls_back_to_extractive_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    answer = llm_answer("q", [(CHUNK, 0.9)])
    assert "not configured" in answer
    assert "Some retrieved text." in answer  # extractive fallback content


def test_llm_answer_builds_prompt_when_api_key_is_set(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    answer = llm_answer("What is in the test doc?", [(CHUNK, 0.9)])
    assert "What is in the test doc?" in answer
    assert "Source: Test Doc (WHO)" in answer
    assert "Never invent" in answer  # confirms the real system prompt was used
