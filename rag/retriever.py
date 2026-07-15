"""
Retrieval: run a vector search and apply a similarity threshold so weak
matches never reach generation.

This is Guard 1 of hallucination prevention: if nothing retrieved clears
config.similarity_threshold, there isn't enough evidence to answer, and the
caller should refuse rather than ask the LLM (or the extractive summarizer)
to make something out of loosely related chunks.
"""

from typing import List, Tuple

from config import config

from .embed_store import VectorStore
from .ingest import Chunk


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
    return [(chunk, score) for chunk, score in results if score >= similarity_threshold]
