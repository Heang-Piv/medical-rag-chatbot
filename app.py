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
from rag.ingest import (
    load_documents,
    build_chunk_records,
    save_uploaded_file,
    extract_text_from_bytes,
    MAX_UPLOAD_BYTES,
)
from rag.embed_store import VectorStore, domain_relevance_score
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
        "Uploads are automatically checked for medical/health relevance (semantic "
        "similarity to the existing corpus) before being added — off-topic files are "
        "rejected. Accepted uploads still aren't vetted WHO/CDC/NIH sources; treat "
        "answers grounded in them accordingly."
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
        # Marked as seen up front (success or not) so a rejected/oversized
        # file isn't re-embedded and re-checked on every later rerun (e.g.
        # moving a slider) for as long as it sits in the uploader widget.
        st.session_state.saved_uploads.add(uploaded.name)
        content = uploaded.getvalue()
        if len(content) > MAX_UPLOAD_BYTES:
            st.error(f"'{uploaded.name}' is over the {MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MB limit — not added.")
            continue
        text = extract_text_from_bytes(uploaded.name, content)
        if not text.strip():
            st.error(f"'{uploaded.name}' couldn't be read (empty or unreadable file) — not added.")
            continue
        score = domain_relevance_score(text)
        if score < config.upload_relevance_threshold:
            st.error(
                f"'{uploaded.name}' doesn't look medical/health-related "
                f"(relevance {score:.2f}, needs ≥{config.upload_relevance_threshold:.2f}) — not added."
            )
            continue
        save_uploaded_file(os.path.join(config.data_folder, UPLOAD_SUBFOLDER), uploaded.name, content)
        st.success(f"Saved '{uploaded.name}' (relevance {score:.2f}). Click Rebuild index to include it.")

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
            st.session_state.last_search = {"query": query, "intent": intent}
        elif intent == "capability":
            logger.info("Handled as capability query, skipped retrieval")
            st.session_state.last_search = {"query": query, "intent": intent}
        else:
            try:
                retrieved = retrieve(store, query, top_k=top_k)
                answer = generate_answer(query, retrieved, mode=mode)
            except Exception:
                logger.exception("Search failed for query: %r", query)
                st.error("Something went wrong while processing your question. Please try again.")
            else:
                # Recorded here (not in rag/generate.py) since this is purely a UI
                # display concern, not RAG pipeline logic.
                st.session_state.last_search = {
                    "query": query,
                    "intent": intent,
                    "mode": mode,
                    "top_k": top_k,
                    "answer": answer,
                    "confidence": confidence_level(retrieved),
                    "retrieved": retrieved,
                }

    # Rendered from session_state, not the `query` variable above: st.chat_input
    # always clears itself right after submission, and any later widget
    # interaction anywhere on the page (e.g. a sidebar slider) reruns the whole
    # script with query back to None. Reading from session_state is what keeps
    # the last question and answer visible instead of disappearing on the next
    # rerun.
    last_search = st.session_state.get("last_search")
    if not last_search:
        st.caption("Ask a question below to see the answer here.")
    else:
        st.subheader("Question")
        st.write(last_search["query"])
        st.subheader("Answer")

        if last_search["intent"] == "greeting":
            st.write(GREETING_RESPONSE)
        elif last_search["intent"] == "capability":
            st.write(capability_answer(docs))
        else:
            st.write(last_search["answer"])

            confidence = last_search["confidence"]
            if confidence:
                _CONFIDENCE_DISPLAY[confidence](f"Confidence: {confidence}")

            st.subheader("Sources")
            retrieved = last_search["retrieved"]
            if retrieved:
                for chunk, score in retrieved:
                    with st.expander(f"{chunk.doc_title}  ·  similarity {score:.2f}"):
                        st.write(chunk.text)
                        st.caption(explain_chunk(last_search["query"], chunk))
            else:
                st.caption("No sources cleared the similarity threshold for this query.")

with eval_tab:
    st.subheader("Evaluation of your last question")
    last_search = st.session_state.get("last_search")

    if not last_search:
        st.info("Ask a question in the Search tab first — a detailed evaluation "
                 "breakdown of that query will appear here.")
    elif last_search["intent"] != "question":
        st.write(f"**Question:** {last_search['query']}")
        st.write(f"**Handled as:** {last_search['intent']} — greetings and capability "
                 "questions bypass retrieval entirely, so there's nothing to evaluate.")
    else:
        retrieved = last_search["retrieved"]
        confidence = last_search["confidence"]
        st.write(f"**Question:** {last_search['query']}")
        st.write(f"**Settings used:** top_k={last_search['top_k']}, mode={last_search['mode']}")
        st.write(f"**Refused (no evidence found)?** {'Yes' if not retrieved else 'No'}")

        if retrieved:
            distinct_docs = {chunk.doc_title for chunk, _ in retrieved}
            avg_score = sum(score for _, score in retrieved) / len(retrieved)
            st.write(f"**Chunks retrieved above threshold:** {len(retrieved)}")
            st.write(f"**Distinct source documents:** {len(distinct_docs)}")
            st.write(f"**Average similarity:** {avg_score:.3f}")
            if confidence:
                _CONFIDENCE_DISPLAY[confidence](
                    f"Confidence: {confidence} — {len(distinct_docs)} distinct source(s), "
                    f"average similarity {avg_score:.2f}"
                )

            st.write("**Retrieved chunks (full detail):**")
            for chunk, score in retrieved:
                with st.expander(f"{chunk.doc_title}  ·  similarity {score:.2f}"):
                    st.write(chunk.text)
                    st.caption(explain_chunk(last_search["query"], chunk))
                    if chunk.source_url:
                        st.caption(f"Source URL: {chunk.source_url}")
        else:
            st.write("No chunks cleared the similarity threshold for this query — the "
                     "system correctly refused rather than guessing from weak evidence.")

    st.divider()
    st.subheader("Fixed 10-question evaluation set")
    st.caption(
        "Optional: runs the same 10 test questions documented in docs/evaluation.md "
        "through the live pipeline, using the current sidebar settings (top-k, answer "
        "mode). Results are shown here for this session only, not saved to disk."
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
