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


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load and cache the embedding model so repeated VectorStore instances
    (e.g. across Streamlit reruns or tests) don't reload it from disk."""
    return SentenceTransformer(model_name)


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
