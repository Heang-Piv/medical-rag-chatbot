"""
Generation: turn retrieved chunks + a query into a final answer.

Two modes are provided:
- "extractive" (default): no API key needed, works immediately. Just stitches
  together the retrieved chunks so you can verify retrieval quality before wiring
  up an LLM.
- "llm": calls an LLM to write a grounded answer from the retrieved context.
  TODO: fill in your provider of choice (Anthropic, OpenAI, a local model via
  Ollama, etc). A minimal Anthropic example is sketched below — install the
  `anthropic` package and set the ANTHROPIC_API_KEY environment variable to use it.
"""

import os
from typing import List, Tuple

from .ingest import Chunk

# Guard 1 (hallucination prevention): shown whenever retrieval finds nothing
# that clears config.similarity_threshold — see rag/retriever.py.
NO_EVIDENCE_MESSAGE = "I could not find sufficient information in the provided document collection."


def extractive_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    if not retrieved:
        return NO_EVIDENCE_MESSAGE
    lines = [f"Top passages related to: “{query}”\n"]
    for chunk, score in retrieved:
        lines.append(f"[{chunk.doc_title}, score={score:.2f}] {chunk.text}\n")
    return "\n".join(lines)


def llm_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """TODO: replace this with a real LLM call once retrieval is working well."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            "[LLM mode not configured] Set ANTHROPIC_API_KEY (or wire up your own "
            "provider in rag/generate.py) to enable grounded LLM answers. "
            "Falling back to extractive mode:\n\n" + extractive_answer(query, retrieved)
        )

    context = "\n\n".join(f"Source: {c.doc_title}\n{c.text}" for c, _ in retrieved)
    prompt = (
        "Answer the question using ONLY the sources below. Cite the source title(s) "
        f"you used.\n\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    # TODO: uncomment once the `anthropic` package is installed
    # import anthropic
    # client = anthropic.Anthropic(api_key=api_key)
    # response = client.messages.create(
    #     model="claude-sonnet-4-6",
    #     max_tokens=500,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # return response.content[0].text

    return "[TODO] Wire up your LLM call in rag/generate.py using the prompt below:\n\n" + prompt


def generate_answer(query: str, retrieved: List[Tuple[Chunk, float]], mode: str = "extractive") -> str:
    if not retrieved:
        return NO_EVIDENCE_MESSAGE
    if mode == "llm":
        return llm_answer(query, retrieved)
    return extractive_answer(query, retrieved)
