"""
Vector store: turn chunks into vectors and support similarity search over them.

Uses a local sentence-transformers embedding model (config.embedding_model) so
similarity reflects semantic meaning rather than exact word overlap — needed
because medical queries are often phrased differently than the source text
(e.g. "high blood pressure" vs. "hypertension").

Upgrade path (still open for later milestones):
- Swap the in-memory dot-product search below for ChromaDB once the pipeline
  works end-to-end (M6). Keep the VectorStore interface (`build`, `query`) the
  same so app.py doesn't change.
"""

from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from config import config

from .ingest import Chunk

# BAAI/bge-* models are trained to expect this instruction prefix on queries
# (not on passages) for query-to-passage retrieval; the model card calls it
# out as necessary for good retrieval quality, and it costs nothing to add.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load and cache the embedding model so repeated VectorStore instances
    (e.g. across Streamlit reruns or tests) don't reload it from disk."""
    return SentenceTransformer(model_name)


class VectorStore:
    def __init__(self):
        self.model = _load_model(config.embedding_model)
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        """Embed all chunk text and store the resulting normalized matrix."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, similarity_score) pairs for a query string.

        Similarity is cosine similarity, computed as a dot product since both
        chunk and query embeddings are L2-normalized.
        """
        if self.embeddings is None:
            raise RuntimeError("VectorStore.build() must be called before query().")
        query_vec = self.model.encode(
            _QUERY_INSTRUCTION + query_text, normalize_embeddings=True
        )
        scores = self.embeddings @ query_vec
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx]
