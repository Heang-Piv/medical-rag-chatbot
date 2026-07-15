"""Tests for the refusal guard, extractive mode, provider branching, and
confidence scoring in rag/generate.py (M7/M8/M9)."""

from types import SimpleNamespace

import anthropic
import pytest

from rag.generate import (
    NO_EVIDENCE_MESSAGE,
    confidence_level,
    extractive_answer,
    generate_answer,
    llm_answer,
)
from rag.ingest import Chunk

CHUNK = Chunk(chunk_id="x::0", doc_title="Test Doc", text="Some retrieved text.", source_org="WHO")
CHUNK_B = Chunk(chunk_id="y::0", doc_title="Other Doc", text="More retrieved text.", source_org="CDC")


def _fake_config(**overrides) -> SimpleNamespace:
    base = dict(
        llm_provider="anthropic",
        anthropic_api_key="",
        openai_api_key="",
        llm_model="claude-sonnet-4-6",
        llm_temperature=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, text: str, stop_reason: str = "end_turn"):
        self.stop_reason = stop_reason
        self.content = [_FakeTextBlock(text)] if text else []


class _FakeAnthropicMessages:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.last_call_kwargs = None

    def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        if self._error:
            raise self._error
        return self._response


class _FakeAnthropicClient:
    def __init__(self, messages: _FakeAnthropicMessages):
        self.messages = messages


# --- Guard 1: refusal when nothing retrieved ---

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


# --- llm_answer: missing API key falls back gracefully ---

def test_llm_answer_falls_back_to_extractive_without_anthropic_key(monkeypatch) -> None:
    monkeypatch.setattr("rag.generate.config", _fake_config(anthropic_api_key=""))
    answer = llm_answer("q", [(CHUNK, 0.9)])
    assert "not configured" in answer
    assert "Some retrieved text." in answer  # extractive fallback content


def test_llm_answer_falls_back_to_extractive_without_openai_key(monkeypatch) -> None:
    monkeypatch.setattr("rag.generate.config", _fake_config(llm_provider="openai", openai_api_key=""))
    answer = llm_answer("q", [(CHUNK, 0.9)])
    assert "not configured" in answer
    assert "Some retrieved text." in answer


def test_llm_answer_reports_unknown_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag.generate.config",
        _fake_config(llm_provider="ollama", anthropic_api_key="", openai_api_key=""),
    )
    answer = llm_answer("q", [(CHUNK, 0.9)])
    assert "Configuration error" in answer
    assert "ollama" in answer


# --- llm_answer: real call construction and response parsing (mocked SDK) ---

def test_anthropic_answer_sends_built_prompt_and_parses_response(monkeypatch) -> None:
    fake_messages = _FakeAnthropicMessages(
        response=_FakeAnthropicResponse("Grounded answer citing (Source: Test Doc).")
    )
    monkeypatch.setattr("rag.generate.config", _fake_config(anthropic_api_key="test-key"))
    monkeypatch.setattr(
        "rag.generate.anthropic.Anthropic",
        lambda api_key: _FakeAnthropicClient(fake_messages),
    )

    answer = llm_answer("What is in the test doc?", [(CHUNK, 0.9)])

    assert answer == "Grounded answer citing (Source: Test Doc)."
    sent = fake_messages.last_call_kwargs
    assert sent["model"] == "claude-sonnet-4-6"
    assert "What is in the test doc?" in sent["messages"][0]["content"]
    assert "Never invent" in sent["system"]  # confirms the real system prompt was used


def test_anthropic_answer_handles_refusal_stop_reason(monkeypatch) -> None:
    fake_messages = _FakeAnthropicMessages(
        response=_FakeAnthropicResponse("", stop_reason="refusal")
    )
    monkeypatch.setattr("rag.generate.config", _fake_config(anthropic_api_key="test-key"))
    monkeypatch.setattr(
        "rag.generate.anthropic.Anthropic",
        lambda api_key: _FakeAnthropicClient(fake_messages),
    )

    answer = llm_answer("q", [(CHUNK, 0.9)])
    assert "declined" in answer.lower()


def test_anthropic_answer_handles_connection_error_gracefully(monkeypatch) -> None:
    fake_messages = _FakeAnthropicMessages(error=anthropic.APIConnectionError(request=None))
    monkeypatch.setattr("rag.generate.config", _fake_config(anthropic_api_key="test-key"))
    monkeypatch.setattr(
        "rag.generate.anthropic.Anthropic",
        lambda api_key: _FakeAnthropicClient(fake_messages),
    )

    answer = llm_answer("q", [(CHUNK, 0.9)])
    assert "temporarily unavailable" in answer
    assert "Traceback" not in answer  # never expose a stack trace


# --- confidence_level: Guard 4, deterministic ---

def test_confidence_level_is_none_when_nothing_retrieved() -> None:
    assert confidence_level([]) is None


def test_confidence_level_high_for_multiple_consistent_sources() -> None:
    assert confidence_level([(CHUNK, 0.85), (CHUNK_B, 0.80)]) == "High"


def test_confidence_level_moderate_for_single_source_or_middling_score() -> None:
    assert confidence_level([(CHUNK, 0.65)]) == "Moderate"


def test_confidence_level_low_for_weak_evidence() -> None:
    assert confidence_level([(CHUNK, 0.52), (CHUNK_B, 0.50)]) == "Low"
