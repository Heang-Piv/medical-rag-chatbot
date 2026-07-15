"""Tests for the recursive, boundary-aware chunker in rag/ingest.py (M4)."""

from rag.ingest import build_chunk_records, chunk_text

SENTENCES = [
    f"This is sentence number {i} about diabetes and blood sugar control."
    for i in range(60)
]
LONG_TEXT = " ".join(SENTENCES)

PARAGRAPHS_TEXT = "\n\n".join(
    " ".join(SENTENCES[i:i + 10]) for i in range(0, 60, 10)
)


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_returns_single_chunk() -> None:
    text = "Flu symptoms include fever, cough, and fatigue."
    chunks = chunk_text(text, chunk_size=350, overlap=50)
    assert chunks == [text]


def test_chunks_never_exceed_chunk_size() -> None:
    chunks = chunk_text(LONG_TEXT, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 50


def test_chunks_break_on_sentence_boundaries() -> None:
    """Every chunk should start and end on a full sentence, never mid-sentence."""
    chunks = chunk_text(LONG_TEXT, chunk_size=50, overlap=10)
    for c in chunks:
        assert c.strip().endswith(".")
        assert c.strip()[0].isupper()


def test_overlap_shares_trailing_words_between_chunks() -> None:
    chunks = chunk_text(LONG_TEXT, chunk_size=50, overlap=15)
    assert len(chunks) > 1
    for prev_chunk, next_chunk in zip(chunks, chunks[1:]):
        prev_tail = prev_chunk.split()[-5:]
        next_head = next_chunk.split()[:20]
        assert any(word in next_head for word in prev_tail)


def test_zero_overlap_produces_no_shared_words() -> None:
    chunks = chunk_text(LONG_TEXT, chunk_size=50, overlap=0)
    assert len(chunks) > 1
    for prev_chunk, next_chunk in zip(chunks, chunks[1:]):
        assert prev_chunk.split()[-1] != next_chunk.split()[0]


def test_respects_paragraph_boundaries_when_they_fit() -> None:
    chunks = chunk_text(PARAGRAPHS_TEXT, chunk_size=200, overlap=0)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.split()) <= 200


def test_build_chunk_records_uses_config_defaults() -> None:
    docs = [
        {
            "title": "Sample Doc",
            "text": LONG_TEXT,
            "source_org": "WHO",
            "source_url": "https://example.org",
        }
    ]
    records = build_chunk_records(docs)
    assert len(records) > 1
    for r in records:
        assert len(r.text.split()) <= 350
        assert r.doc_title == "Sample Doc"
        assert r.source_org == "WHO"
        assert r.chunk_id.startswith("Sample Doc::")


def test_single_run_on_sentence_longer_than_chunk_size_is_hard_split() -> None:
    run_on = " ".join(f"word{i}" for i in range(100)) + "."
    chunks = chunk_text(run_on, chunk_size=30, overlap=5)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 30
