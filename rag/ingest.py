"""
Ingestion: load raw documents from disk and split them into overlapping chunks.

Supports .txt, .md, and .pdf, and walks subfolders recursively so a corpus
organized as data/medical/{who,cdc,nih}/ is picked up in one pass. If a
manifest.json is present in the root data folder (see data/medical/manifest.json),
it supplies title, source_org, and source_url for each file by relative path;
files not listed in the manifest fall back to a title derived from the
filename and a source_org guessed from the immediate parent folder name.

Chunking splits recursively on paragraph, then sentence, then word boundaries,
so chunks only break mid-sentence when a single sentence exceeds chunk_size on
its own.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from pypdf import PdfReader

from config import config

_HEADER_PREFIXES = ("Title:", "Source:", "URL:", "Fetched:", "Published:")
_KNOWN_SOURCE_ORGS = ("who", "cdc", "nih")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


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


def _word_count(text: str) -> int:
    """Approximate token count as whitespace-separated word count."""
    return len(text.split())


def _split_into_units(text: str, max_size: int) -> List[str]:
    """Recursively break text into pieces of at most max_size words each.

    Tries paragraph boundaries first, then sentence boundaries, and only
    falls back to a hard word-count split for the rare single sentence that
    exceeds max_size on its own.
    """
    units: List[str] = []
    for paragraph in (p.strip() for p in _PARAGRAPH_RE.split(text)):
        if not paragraph:
            continue
        if _word_count(paragraph) <= max_size:
            units.append(paragraph)
            continue
        for sentence in (s.strip() for s in _SENTENCE_RE.split(paragraph)):
            if not sentence:
                continue
            if _word_count(sentence) <= max_size:
                units.append(sentence)
                continue
            words = sentence.split()
            for start in range(0, len(words), max_size):
                units.append(" ".join(words[start:start + max_size]))
    return units


def _tail_units(units: List[str], overlap: int) -> List[str]:
    """Return the trailing units of `units` whose combined size is <= overlap."""
    if overlap <= 0:
        return []
    tail: List[str] = []
    size = 0
    for unit in reversed(units):
        unit_size = _word_count(unit)
        if tail and size + unit_size > overlap:
            break
        tail.insert(0, unit)
        size += unit_size
    return tail


def chunk_text(
    text: str,
    chunk_size: int = config.chunk_size_tokens,
    overlap: int = config.chunk_overlap_tokens,
) -> List[str]:
    """Split text into overlapping chunks along paragraph/sentence boundaries.

    Recursively decomposes the text into paragraph- or sentence-sized units
    (see _split_into_units), then greedily packs those units into chunks of
    at most chunk_size words, carrying the trailing ~overlap words of each
    chunk into the next one so retrieval doesn't lose context at chunk edges.
    """
    units = _split_into_units(text, max_size=chunk_size)
    if not units:
        return []

    chunks: List[str] = []
    current_units: List[str] = []
    current_size = 0

    for unit in units:
        unit_size = _word_count(unit)
        while current_units and current_size + unit_size > chunk_size:
            chunks.append(" ".join(current_units))
            tail = _tail_units(current_units, overlap)
            tail_size = sum(_word_count(u) for u in tail)
            if tail_size >= current_size:
                # Carried-over overlap alone already fills a chunk (e.g. a
                # hard-split fragment) — drop it rather than loop forever.
                tail, tail_size = [], 0
            current_units, current_size = tail, tail_size
        current_units.append(unit)
        current_size += unit_size

    if current_units:
        chunks.append(" ".join(current_units))
    return chunks


def build_chunk_records(
    docs: List[dict],
    chunk_size: int = config.chunk_size_tokens,
    overlap: int = config.chunk_overlap_tokens,
) -> List[Chunk]:
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
