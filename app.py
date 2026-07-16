"""
Medical RAG Search System — Streamlit interface.

Run with:
    streamlit run app.py

Wires together ingestion (rag/ingest.py), embeddings + persistent vector
store (rag/embed_store.py), similarity-threshold retrieval (rag/retriever.py),
and grounded generation (rag/generate.py) behind a query box, an answer
panel, and a sources panel.
"""

import os

import streamlit as st

# On Streamlit Community Cloud, API keys are set via the dashboard's Secrets
# manager (st.secrets), not a committed .env file. Bridge them into os.environ
# so config.py's plain os.environ.get() calls work unchanged in both places.
# Wrapped in try/except: st.secrets raises if no secrets.toml/Cloud secrets
# exist at all, which is the normal case for local development with .env.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

from config import config
from rag.ingest import load_documents, build_chunk_records
from rag.embed_store import VectorStore
from rag.retriever import explain_chunk, retrieve
from rag.generate import capability_answer, confidence_level, detect_intent, generate_answer, GREETING_RESPONSE
from rag.utils import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="RAG Search", page_icon="🔎", layout="wide")

_CONFIDENCE_DISPLAY = {
    "High": st.success,
    "Moderate": st.info,
    "Low": st.warning,
}


@st.cache_resource(show_spinner="Loading index...")
def load_store():
    docs = load_documents(config.data_folder)
    chunks = build_chunk_records(docs)
    store = VectorStore()
    if store.count() == 0 and chunks:
        # First run on this environment (e.g. a fresh Streamlit Cloud container
        # with no persisted chroma_db/) — build once. Local dev normally skips
        # this by running scripts/build_index.py ahead of time, which is still
        # the right way to deliberately rebuild after a config/data change.
        logger.info("No persisted index found — building now (%d chunks)", len(chunks))
        store.build(chunks)
    return store, docs, chunks


try:
    store, docs, chunks = load_store()
except Exception:
    logger.exception("Failed to load the document index")
    st.error(
        "Something went wrong while loading the search index. This usually means "
        "the embedding model couldn't be downloaded or the index files are "
        "unreadable. Check the terminal logs for details, then try again."
    )
    st.stop()

if store.count() == 0:
    st.error(
        f"No documents were found under '{config.data_folder}', so there's nothing to "
        "index. Check that the data folder exists and contains .txt/.md/.pdf files."
    )
    st.stop()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=config.top_k_default)
    mode = st.radio("Answer mode", ["extractive", "llm"], index=0,
                     help=f"Extractive works with no setup. LLM mode calls {config.llm_provider} "
                          f"and needs its API key set (.env locally, or Secrets on Streamlit Cloud).")
    st.divider()
    st.caption(f"Indexed **{len(docs)}** documents → **{len(chunks)}** chunks")
    with st.expander("Documents in this index"):
        for d in docs:
            st.write(f"- {d['title']}")

st.title("🔎 RAG-Based AI Search System")
st.caption("Ask a question about the indexed medical documents below (WHO / CDC / NIH sources only).")
st.caption(
    "⚠️ **Medical disclaimer:** answers are summarized from the retrieved documents only "
    "and are not a substitute for professional medical advice, diagnosis, or treatment."
)

query = st.text_input("Your question", placeholder="e.g. How does content-based filtering rank items?")
search_clicked = st.button("Search", type="primary")

if search_clicked and query.strip():
    intent = detect_intent(query)

    if intent == "greeting":
        st.subheader("Answer")
        st.write(GREETING_RESPONSE)
    elif intent == "capability":
        logger.info("Handled as capability query, skipped retrieval")
        st.subheader("Answer")
        st.write(capability_answer(docs))
    else:
        try:
            retrieved = retrieve(store, query, top_k=top_k)
            answer = generate_answer(query, retrieved, mode=mode)
        except Exception:
            logger.exception("Search failed for query: %r", query)
            st.error("Something went wrong while processing your question. Please try again.")
        else:
            st.subheader("Answer")
            st.write(answer)

            confidence = confidence_level(retrieved)
            if confidence:
                _CONFIDENCE_DISPLAY[confidence](f"Confidence: {confidence}")

            st.subheader("Sources")
            if retrieved:
                for chunk, score in retrieved:
                    with st.expander(f"{chunk.doc_title}  ·  similarity {score:.2f}"):
                        st.write(chunk.text)
                        st.caption(explain_chunk(query, chunk))
            else:
                st.caption("No sources cleared the similarity threshold for this query.")
elif search_clicked:
    st.warning("Type a question first.")
