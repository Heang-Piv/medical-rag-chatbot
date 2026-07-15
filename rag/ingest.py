"""
Ingestion: load raw documents from disk and split them into overlapping chunks.

Supports .txt, .md, and .pdf, and walks subfolders recursively so a corpus
organized as data/medical/{who,cdc,nih}/ is picked up in one pass. If a
manifest.json is present in the root data folder (see data/medical/manifest.json),
it supplies title, source_org, and source_url for each file by relative path;
files not listed in the manifest fall back to a title derived from the
filename and a source_org guessed from the immediate parent folder name.

Upgrade path (still open for later milestones):
- Swap the naive word-count chunker below for a sentence- or paragraph-aware
  recursive splitter (M4) — the chunk_text(text) -> List[str] interface stays
  the same so nothing downstream has to change again.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from pypdf import PdfReader

_HEADER_PREFIXES = ("Title:", "Source:", "URL:", "Fetched:", "Published:")
_KNOWN_SOURCE_ORGS = ("who", "cdc", "nih")


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    text: str
    source_org: Optional[str] = None
    source_url: Optional[str] = None


def _load_manifest(folder: str) -> dict:
    """Load <folder>/manifest.json if present, keyed by relative filename."""
    manifest_path = os.path.join(folder, "manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["filename"]: entry for entry in entries}


def _strip_header(text: str) -> str:
    """Drop the Title/Source/URL/Fetched header block we write into fetched
    documents, so it isn't embedded as if it were part of the document body.
    Files with no such header (e.g. the original sample_docs) pass through
    unchanged.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (lines[i].strip() == "" or lines[i].startswith(_HEADER_PREFIXES)):
        i += 1
    return "\n".join(lines[i:]).strip()


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def load_documents(folder: str) -> List[dict]:
    """Recursively load every .txt/.md/.pdf file under `folder`.

    Returns a list of {"title", "text", "source_org", "source_url", "path"} dicts.
    """
    manifest = _load_manifest(folder)
    docs = []
    for root, _dirs, files in os.walk(folder):
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in (".txt", ".md", ".pdf"):
                continue
            path = os.path.join(root, filename)
            rel_path = os.path.relpath(path, folder).replace(os.sep, "/")

            if ext == ".pdf":
                text = _read_pdf(path)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    text = _strip_header(f.read())
            if not text:
                continue

            entry = manifest.get(rel_path)
            if entry:
                title = entry["title"]
                source_org = entry["source_org"]
                source_url = entry["url"]
            else:
                title = os.path.splitext(filename)[0].replace("_", " ").title()
                parent = os.path.basename(root).lower()
                source_org = parent.upper() if parent in _KNOWN_SOURCE_ORGS else None
                source_url = None

            docs.append({
                "title": title,
                "text": text,
                "source_org": source_org,
                "source_url": source_url,
                "path": rel_path,
            })
    return docs


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> List[str]:
    """Split text into overlapping word-count chunks (simple, dependency-free).

    Placeholder splitter through M3 — doesn't respect sentence or paragraph
    boundaries, so a chunk can end mid-sentence. M4 replaces this with a
    recursive, boundary-aware splitter driven by config.chunk_size_tokens /
    config.chunk_overlap_tokens.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_chunk_records(docs: List[dict], chunk_size: int = 80, overlap: int = 20) -> List[Chunk]:
    """Turn loaded documents into a flat list of Chunk records ready for embedding."""
    records = []
    for doc in docs:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            records.append(Chunk(
                chunk_id=f"{doc['title']}::{i}",
                doc_title=doc["title"],
                text=piece,
                source_org=doc.get("source_org"),
                source_url=doc.get("source_url"),
            ))
    return records
