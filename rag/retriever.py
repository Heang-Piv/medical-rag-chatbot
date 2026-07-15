"""
Retrieval: run a vector search and apply a similarity threshold so weak
matches never reach generation.

This is Guard 1 of hallucination prevention: if nothing retrieved clears
config.similarity_threshold, there isn't enough evidence to answer, and the
caller should refuse rather than ask the LLM (or the extractive summarizer)
to make something out of loosely related chunks.
"""

import re
from typing import List, Tuple

from config import config

from .embed_store import VectorStore
from .ingest import Chunk
from .utils import get_logger

_logger = get_logger(__name__)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "what", "how", "why", "does", "do", "did", "can", "could",
    "should", "would", "this", "that", "it", "its", "with", "by", "from", "as",
    "be", "been", "which", "who", "whom", "at", "i", "you", "your", "my", "me",
    "we", "us", "our", "they", "their", "about",
}


def retrieve(
    store: VectorStore,
    query: str,
    top_k: int = config.top_k_default,
    similarity_threshold: float = config.similarity_threshold,
) -> List[Tuple[Chunk, float]]:
    """Return the retrieved (chunk, score) pairs that clear similarity_threshold.

    Requests up to top_k chunks from the store, then drops any chunk whose
    score falls below similarity_threshold. The result can therefore be
    shorter than top_k, including empty — an empty result means the caller
    should refuse to answer rather than guess.
    """
    results = store.query(query, top_k=top_k)
    passed = [(chunk, score) for chunk, score in results if score >= similarity_threshold]
    _logger.info(
        "Retrieval executed: %d/%d chunks cleared threshold %.2f for query: %r",
        len(passed), len(results), similarity_threshold, query[:60],
    )
    return passed


def _significant_words(text: str) -> set:
    """Lowercased, stopword-free words of length > 2, for term-overlap checks."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def explain_chunk(query: str, chunk: Chunk) -> str:
    """A lightweight, deterministic explanation of why a chunk was retrieved.

    Highlights literal shared terms between the query and the chunk text when
    present. Semantic (embedding-based) retrieval often matches with zero
    literal word overlap by design — that's the whole point of moving off
    keyword search (M5) — so when no terms are shared, this says so honestly
    rather than pretending to explain the embedding model's internal
    reasoning. No LLM call, per the "no complex AI explanation systems"
    requirement.
    """
    shared = sorted(_significant_words(query) & _significant_words(chunk.text))
    if shared:
        terms = ", ".join(shared[:5])
        return f'This passage from "{chunk.doc_title}" shares key terms with your question: {terms}.'
    return (
        f'This passage from "{chunk.doc_title}" was the closest semantic match to your '
        f"question among the indexed documents, even though it doesn't share exact wording."
    )
