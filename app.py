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
from rag.ingest import load_documents, build_chunk_records, save_uploaded_file, MAX_UPLOAD_BYTES
from rag.embed_store import VectorStore
from rag.retriever import explain_chunk, retrieve
from rag.generate import capability_answer, confidence_level, detect_intent, generate_answer, GREETING_RESPONSE
from rag.utils import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Medical RAG Search", page_icon=":material/health_and_safety:", layout="wide")

_CONFIDENCE_DISPLAY = {
    "High": st.success,
    "Moderate": st.info,
    "Low": st.warning,
}

UPLOAD_SUBFOLDER = "uploaded"

# The same 10 questions documented in docs/evaluation.md — kept identical so
# the in-app Evaluation tab and the written evaluation report describe the
# same test set, not two different ones that could quietly drift apart.
EVALUATION_QUESTIONS = [
    ("Easy", "What are common symptoms of the flu?"),
    ("Easy", "How is malaria transmitted?"),
    ("Easy", "What is asthma?"),
    ("Easy", "What causes tuberculosis?"),
    ("Medium", "What are the risk factors for type 2 diabetes?"),
    ("Medium", "How are high blood pressure and heart disease related?"),
    ("Medium", "What lifestyle changes can help prevent cardiovascular disease?"),
    ("Hard", "Can a specific experimental gene therapy cure diabetes?"),
    ("Hard", "What is the recommended dosage of ibuprofen for a child's fever?"),
    ("Hard", "Does eating chocolate cause acne?"),
]


@st.cache_resource(show_spinner="Loading index...")
def load_store():
    docs = load_documents(config.data_folder)
    store = VectorStore()
    if store.count() == 0:
        # First run on this environment (e.g. a fresh Streamlit Cloud container
        # with no persisted chroma_db/) — build once. Local dev normally skips
        # this by running scripts/build_index.py ahead of time, which is still
        # the right way to deliberately rebuild after a config/data change.
        chunks = build_chunk_records(docs)
        if chunks:
            logger.info("No persisted index found — building now (%d chunks)", len(chunks))
            store.build(chunks)
    return store, docs


try:
    store, docs = load_store()
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
    st.caption(f"Indexed **{len(docs)}** documents on disk → **{store.count()}** chunks in the index")
    with st.expander("Documents in this index"):
        for d in docs:
            st.write(f"- {d['title']}")

    st.divider()
    st.subheader("Chunking & index")
    chunk_size = st.slider(
        "Chunk size (tokens)", min_value=50, max_value=800, value=config.chunk_size_tokens, step=10,
        help="Words per chunk. Changing this has no effect until you click Rebuild index below.",
    )
    chunk_overlap = st.slider(
        "Chunk overlap (tokens)", min_value=0, max_value=300, value=config.chunk_overlap_tokens, step=10,
        help="Words repeated between consecutive chunks.",
    )

    st.subheader("Add documents")
    st.caption(
        "Uploaded files are added to the same index but are not vetted WHO/CDC/NIH "
        "sources — treat answers grounded in them accordingly."
    )
    uploaded_files = st.file_uploader(
        "Upload .txt / .md / .pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        help=f"Max {MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MB per file.",
    )
    if "saved_uploads" not in st.session_state:
        st.session_state.saved_uploads = set()
    for uploaded in uploaded_files or []:
        if uploaded.name in st.session_state.saved_uploads:
            continue
        try:
            save_uploaded_file(
                os.path.join(config.data_folder, UPLOAD_SUBFOLDER), uploaded.name, uploaded.getvalue()
            )
            st.session_state.saved_uploads.add(uploaded.name)
            st.success(f"Saved '{uploaded.name}'. Click Rebuild index to include it.")
        except ValueError as e:
            st.error(str(e))

    if st.button("Rebuild index", icon=":material/refresh:",
                 help="Re-chunks and re-embeds every document using the settings above."):
        with st.spinner("Rebuilding index..."):
            try:
                rebuild_docs = load_documents(config.data_folder)
                rebuild_chunks = build_chunk_records(rebuild_docs, chunk_size=chunk_size, overlap=chunk_overlap)
                VectorStore().build(rebuild_chunks)
                logger.info(
                    "Manual rebuild: %d documents -> %d chunks (chunk_size=%d, overlap=%d)",
                    len(rebuild_docs), len(rebuild_chunks), chunk_size, chunk_overlap,
                )
            except Exception:
                logger.exception("Manual index rebuild failed")
                st.error("Rebuild failed. Check the logs for details.")
            else:
                load_store.clear()
                st.rerun()

search_tab, eval_tab = st.tabs(["Search", "Evaluation"])

with search_tab:
    st.title(":material/health_and_safety: Medical RAG Search System")
    st.caption("Ask a question about the indexed medical documents below (WHO / CDC / NIH sources only).")
    st.caption(
        ":material/warning: **Medical disclaimer:** answers are summarized from the retrieved documents only "
        "and are not a substitute for professional medical advice, diagnosis, or treatment."
    )

    query = st.chat_input("Ask a medical question, e.g. What are common flu symptoms?")

    if query and query.strip():
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

with eval_tab:
    st.subheader("Retrieval & Generation Evaluation")
    st.caption(
        "Runs the same 10 test questions documented in docs/evaluation.md through the "
        "live pipeline, using the current sidebar settings (top-k, answer mode). "
        "Results are shown here for this session only, not saved to disk."
    )

    if st.button("Run evaluation", icon=":material/play_arrow:"):
        progress = st.progress(0.0, text="Running evaluation questions...")
        results = []
        for i, (category, eval_query) in enumerate(EVALUATION_QUESTIONS):
            eval_retrieved = retrieve(store, eval_query, top_k=top_k)
            eval_answer = generate_answer(eval_query, eval_retrieved, mode=mode)
            eval_confidence = confidence_level(eval_retrieved)
            results.append((category, eval_query, eval_retrieved, eval_answer, eval_confidence))
            progress.progress((i + 1) / len(EVALUATION_QUESTIONS), text=f"Ran {i + 1}/{len(EVALUATION_QUESTIONS)}")
        progress.empty()
        st.session_state.eval_results = results
        logger.info("Ran in-app evaluation: %d questions, mode=%s", len(EVALUATION_QUESTIONS), mode)

    eval_results = st.session_state.get("eval_results")
    if eval_results:
        for category, eval_query, eval_retrieved, eval_answer, eval_confidence in eval_results:
            with st.expander(f"[{category}] {eval_query}"):
                st.write("**Answer**")
                st.write(eval_answer)
                if eval_confidence:
                    _CONFIDENCE_DISPLAY[eval_confidence](f"Confidence: {eval_confidence}")
                st.write("**Retrieved sources**")
                if eval_retrieved:
                    for chunk, score in eval_retrieved:
                        st.write(f"- {chunk.doc_title} (similarity {score:.2f})")
                else:
                    st.write("- No sources cleared the similarity threshold.")
