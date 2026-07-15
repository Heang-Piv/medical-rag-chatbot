"""
Prompt construction for grounded, medical-safe LLM generation.

Strategy: "Structured Context" — the model is instructed to judge whether
each retrieved excerpt actually supports the question (not just shares its
topic) before using it, then summarize and cite only what qualifies. This
was chosen over a simpler "cite whatever was retrieved" prompt because
similarity-threshold retrieval (rag/retriever.py) can pass chunks that are
topically related but don't actually answer the question — e.g. a query
about an unproven treatment can retrieve a same-disease chunk that never
mentions that treatment. A relevance-judgment step catches that case; a
hidden chain-of-thought approach was rejected as needing response parsing
to keep the reasoning out of the visible answer, for a benefit that's hard
to measure on a corpus this size.

Kept separate from rag/generate.py so the prompt can be edited without
touching call logic, per the "modular, easy-to-edit prompt" requirement.
"""

from typing import List, Tuple

from .ingest import Chunk

SYSTEM_PROMPT = """You are a medical information assistant. Answer questions using ONLY the numbered excerpts provided with each question — never rely on prior knowledge or training data about medicine.

For every question:
1. Read each excerpt and judge whether it actually addresses the question, not merely shares its topic.
2. Base your answer only on excerpts that directly support it. Disregard excerpts that are topically related but don't answer the question.
3. If no excerpt directly supports an answer, say exactly: "I could not find sufficient information in the provided document collection." Do not guess or fill gaps with outside knowledge.
4. Summarize supporting excerpts in your own words — do not copy long passages verbatim.
5. Cite every claim with the source document title it came from, e.g. (Source: <title>).
6. Never invent a diagnosis, treatment, medication, or recommendation that isn't in the excerpts.
7. Never cite a source that wasn't provided.
8. Write in concise, professional language, and end with a brief reminder that this is not a substitute for professional medical advice."""


def build_context(retrieved: List[Tuple[Chunk, float]]) -> str:
    """Format retrieved chunks into numbered, source-labeled excerpts."""
    lines = []
    for i, (chunk, _score) in enumerate(retrieved, start=1):
        source = chunk.source_org or "Unknown source"
        lines.append(f"[{i}] Source: {chunk.doc_title} ({source})\n{chunk.text}")
    return "\n\n".join(lines)


def build_user_message(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """Build the user-turn content: the question plus its retrieved excerpts."""
    return f"Question: {query}\n\nRetrieved excerpts:\n{build_context(retrieved)}"


def build_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> Tuple[str, str]:
    """Return (system_prompt, user_message), ready to send to any LLM provider."""
    return SYSTEM_PROMPT, build_user_message(query, retrieved)
