"""Tests for load_documents()'s graceful error handling and file-upload
support in rag/ingest.py (M10 / upload feature)."""

import logging
import os

import pytest

from rag.ingest import load_documents, save_uploaded_file


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


def test_save_uploaded_file_writes_content_to_disk(tmp_path) -> None:
    folder = str(tmp_path / "uploaded")
    save_uploaded_file(folder, "note.txt", b"hello world")
    with open(os.path.join(folder, "note.txt"), "rb") as f:
        assert f.read() == b"hello world"


def test_save_uploaded_file_creates_folder_if_missing(tmp_path) -> None:
    folder = str(tmp_path / "does" / "not" / "exist" / "yet")
    save_uploaded_file(folder, "note.txt", b"content")
    assert os.path.exists(os.path.join(folder, "note.txt"))


def test_save_uploaded_file_rejects_oversized_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("rag.ingest.MAX_UPLOAD_BYTES", 10)
    with pytest.raises(ValueError, match="MB limit"):
        save_uploaded_file(str(tmp_path / "uploaded"), "big.txt", b"x" * 11)
    assert not os.path.exists(tmp_path / "uploaded" / "big.txt")


def test_save_uploaded_file_saved_document_is_then_loadable(tmp_path) -> None:
    save_uploaded_file(str(tmp_path), "new_doc.txt", b"Content about a new medical topic.")
    docs = load_documents(str(tmp_path))
    assert len(docs) == 1
    assert docs[0]["title"] == "New Doc"
    assert docs[0]["source_org"] is None  # not a WHO/CDC/NIH subfolder
