"""
Vector store: turn chunks into vectors and support similarity search over them.

This starter ships with a TF-IDF backend (same technique from the Week 14 lab) so
the whole project runs immediately with zero API keys and no model downloads.

Upgrade path (for your final project — do this once the pipeline works end-to-end):
- Swap TfidfVectorizer for real embeddings, e.g.:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(texts)
- Swap the in-memory cosine_similarity search below for FAISS or Chroma once your
  chunk count grows past a few thousand.
- Keep the VectorStore interface (`build`, `query`) the same so app.py doesn't change.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingest import Chunk


class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        """Fit the vectorizer on all chunk text and store the resulting matrix."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, similarity_score) pairs for a query string."""
        if self.matrix is None:
            raise RuntimeError("VectorStore.build() must be called before query().")
        query_vec = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx]
