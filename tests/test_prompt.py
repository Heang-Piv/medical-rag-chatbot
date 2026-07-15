"""Tests for prompt construction in rag/prompt.py (M8)."""

from rag.ingest import Chunk
from rag.prompt import SYSTEM_PROMPT, build_context, build_prompt, build_user_message

CHUNKS = [
    (Chunk(chunk_id="a::0", doc_title="Diabetes", text="Diabetes affects blood sugar.", source_org="WHO"), 0.9),
    (Chunk(chunk_id="b::0", doc_title="Flu", text="Flu causes fever and cough.", source_org="CDC"), 0.8),
]


def test_system_prompt_covers_required_behaviors() -> None:
    required_phrases = [
        "ONLY",
        "I could not find sufficient information in the provided document collection.",
        "Never invent",
        "Never cite",
        "not a substitute for professional medical advice",
    ]
    for phrase in required_phrases:
        assert phrase in SYSTEM_PROMPT


def test_build_context_numbers_and_labels_each_chunk() -> None:
    context = build_context(CHUNKS)
    assert "[1] Source: Diabetes (WHO)" in context
    assert "[2] Source: Flu (CDC)" in context
    assert "Diabetes affects blood sugar." in context
    assert "Flu causes fever and cough." in context


def test_build_context_handles_missing_source_org() -> None:
    chunk = Chunk(chunk_id="c::0", doc_title="Mystery", text="Some text.", source_org=None)
    context = build_context([(chunk, 0.5)])
    assert "Unknown source" in context


def test_build_context_empty_list_returns_empty_string() -> None:
    assert build_context([]) == ""


def test_build_user_message_includes_question_and_context() -> None:
    message = build_user_message("What causes diabetes?", CHUNKS)
    assert "What causes diabetes?" in message
    assert "[1] Source: Diabetes (WHO)" in message


def test_build_prompt_returns_system_and_user_parts() -> None:
    system, user = build_prompt("What causes diabetes?", CHUNKS)
    assert system == SYSTEM_PROMPT
    assert "What causes diabetes?" in user
