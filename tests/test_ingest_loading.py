"""Tests for load_documents()'s graceful error handling in rag/ingest.py (M10)."""

import logging

from rag.ingest import load_documents


def test_load_documents_skips_corrupted_pdf_instead_of_crashing(tmp_path, caplog) -> None:
    (tmp_path / "good.txt").write_text("This is a valid, readable document.")
    (tmp_path / "broken.pdf").write_bytes(b"not a real pdf file")

    with caplog.at_level(logging.WARNING):
        docs = load_documents(str(tmp_path))

    titles = [d["title"] for d in docs]
    assert "Good" in titles
    assert len(docs) == 1  # the corrupted PDF was skipped, not raised
    assert any("broken.pdf" in record.message for record in caplog.records)


def test_load_documents_returns_empty_list_for_empty_folder(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        docs = load_documents(str(tmp_path))
    assert docs == []
    assert any("No documents found" in record.message for record in caplog.records)
