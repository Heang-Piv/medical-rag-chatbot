"""
Vector store: turn chunks into vectors and support similarity search over them.

Backed by a persistent ChromaDB collection (config.chroma_persist_dir), so the
corpus is embedded once by scripts/build_index.py and every later app startup
just reopens the existing index instead of re-embedding the whole collection.
Similarity uses a local sentence-transformers model (config.embedding_model) so
matches reflect meaning rather than exact word overlap — needed because medical
queries are often phrased differently than the source text (e.g. "high blood
pressure" vs. "hypertension").
"""

from functools import lru_cache
from typing import List, Optional, Tuple

import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

from config import config

from .ingest import Chunk
from .utils import get_logger

_logger = get_logger(__name__)

_COLLECTION_NAME = "medical_chunks"
_COLLECTION_METADATA = {"hnsw:space": "cosine"}

# BAAI/bge-* models are trained to expect this instruction prefix on queries
# (not on passages) for query-to-passage retrieval; the model card calls it
# out as necessary for good retrieval quality, and it costs nothing to add.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Fixed reference set for the upload domain gate (see domain_relevance_score
# below) — short descriptions of what medical/health content looks like,
# spanning disease, treatment, public health, anatomy, and mental health so a
# reasonably wide range of real medical documents lands near at least one of
# them.
_DOMAIN_REFERENCE_SENTENCES = [
    "This document describes a disease, illness, or medical condition, including its causes, symptoms, or how it is diagnosed.",
    "This document explains a medical treatment, therapy, medication, dosage, or clinical procedure.",
    "This document provides public health guidance, disease prevention advice, or vaccination information.",
    "This document discusses human anatomy, physiology, or how a part of the body functions.",
    "This document covers a health condition's risk factors, complications, or long-term outcomes.",
    "This document is about a virus, bacteria, infection, or how a disease spreads between people.",
    "This document discusses mental health, psychological conditions, or emotional well-being.",
    "This document provides nutrition or lifestyle guidance related to preventing disease.",
    "This document describes a medical test, diagnostic procedure, or screening for a health condition.",
    "This document is published or reviewed by a health organization such as WHO, CDC, or NIH.",
]


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load and cache the embedding model so repeated VectorStore instances
    (e.g. across Streamlit reruns or tests) don't reload it from disk."""
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def _domain_reference_vectors(model_name: str):
    """Embed _DOMAIN_REFERENCE_SENTENCES once per model, cached like the
    model itself so re-checking uploads across Streamlit reruns is cheap."""
    return _load_model(model_name).encode(_DOMAIN_REFERENCE_SENTENCES, normalize_embeddings=True)


def domain_relevance_score(text: str) -> float:
    """Cosine similarity (0-1) between `text` and the closest of a fixed set
    of "this is medical/health content" reference sentences.

    This is the upload-time counterpart to Guard 1's retrieval threshold:
    same local embedding model, same cosine-similarity technique, just
    compared against a domain reference instead of a live query. No LLM
    call, no external API, no extra cost — reuses the model already loaded
    for indexing/retrieval.

    Calibrated against this project's real corpus: full WHO/CDC/NIH
    documents scored 0.60-0.76, while unrelated reference text (course
    material) scored 0.48-0.58 — see config.upload_relevance_threshold.
    Like Guard 1, this is a threshold-based heuristic, not a classifier: very
    short snippets embed noisily and can land near the boundary either way.
    """
    model = _load_model(config.embedding_model)
    ref_vecs = _domain_reference_vectors(config.embedding_model)
    vec = model.encode(text[:2000], normalize_embeddings=True)
    return float((ref_vecs @ vec).max())


def _chunk_metadata(chunk: Chunk) -> dict:
    """Build a Chroma-safe metadata dict (Chroma rejects None values)."""
    return {
        "doc_title": chunk.doc_title,
        "source_org": chunk.source_org or "",
        "source_url": chunk.source_url or "",
    }


class VectorStore:
    def __init__(self, persist_dir: Optional[str] = None):
        """persist_dir overrides config.chroma_persist_dir — used by tests to
        avoid touching the real, shared index on disk."""
        self.model = _load_model(config.embedding_model)
        self.client = chromadb.PersistentClient(path=persist_dir or config.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            _COLLECTION_NAME, metadata=_COLLECTION_METADATA
        )
        _logger.info(
            "Database loaded: collection '%s' has %d vectors (%s)",
            _COLLECTION_NAME, self.collection.count(), persist_dir or config.chroma_persist_dir,
        )

    def count(self) -> int:
        """Number of chunks currently persisted in the index."""
        return self.collection.count()

    def build(self, chunks: List[Chunk]) -> None:
        """Embed all chunks and persist them to ChromaDB, replacing any
        existing index. This is the expensive step — call it from
        scripts/build_index.py, not on every app startup.
        """
        try:
            self.client.delete_collection(_COLLECTION_NAME)
        except NotFoundError:
            pass  # no prior index to delete
        self.collection = self.client.create_collection(
            _COLLECTION_NAME, metadata=_COLLECTION_METADATA
        )
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[_chunk_metadata(c) for c in chunks],
        )
        _logger.info("Embedding creation: embedded and persisted %d chunks", len(chunks))

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, similarity_score) pairs for a query string.

        Similarity is cosine similarity: the collection is configured for
        cosine distance, and Chroma's cosine distance is 1 - cosine_similarity.
        """
        if self.collection.count() == 0:
            raise RuntimeError(
                "VectorStore index is empty — run scripts/build_index.py first."
            )
        query_vec = self.model.encode(
            _QUERY_INSTRUCTION + query_text, normalize_embeddings=True
        )
        result = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks_and_scores = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_title=meta["doc_title"],
                text=text,
                source_org=meta["source_org"] or None,
                source_url=meta["source_url"] or None,
            )
            chunks_and_scores.append((chunk, 1.0 - distance))
        return chunks_and_scores
