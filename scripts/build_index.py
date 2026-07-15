"""
Standalone script that (re)builds the persistent ChromaDB index.

Run this once whenever the document collection, chunking, or embedding
config changes:

    python scripts/build_index.py

app.py only ever *reads* the index this script produces — it never
re-embeds the corpus on its own, so Streamlit restarts stay fast. Re-run
this script any time data/medical/, chunk_size_tokens, chunk_overlap_tokens,
or embedding_model changes.
"""

import os
import sys

# Allow running as `python scripts/build_index.py` from the project root
# without needing `python -m` or an installed package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config  # noqa: E402
from rag.embed_store import VectorStore  # noqa: E402
from rag.ingest import build_chunk_records, load_documents  # noqa: E402


def main() -> None:
    docs = load_documents(config.data_folder)
    chunks = build_chunk_records(docs)
    store = VectorStore()
    store.build(chunks)
    print(
        f"Indexed {len(docs)} documents -> {len(chunks)} chunks "
        f"into '{config.chroma_persist_dir}' ({store.count()} vectors stored)."
    )


if __name__ == "__main__":
    main()
