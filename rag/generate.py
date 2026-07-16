"""
Generation: turn retrieved chunks + a query into a final answer.

Two modes are provided:
- "extractive" (default): no API key needed, works immediately. Just stitches
  together the retrieved chunks so you can verify retrieval quality before wiring
  up an LLM.
- "llm": passes the grounded prompt (rag/prompt.py) to the configured provider
  (config.llm_provider: anthropic | openai) and returns its answer. The
  "openai" branch also works against any OpenAI-compatible endpoint via
  config.openai_base_url (e.g. NVIDIA's free-tier catalog).

Guard 1 (retrieval requirement) is enforced in generate_answer(): if nothing
was retrieved, neither mode is even attempted.
"""

import re
from typing import List, Optional, Tuple

import anthropic
import openai

from config import config

from .ingest import Chunk
from .prompt import build_prompt
from .utils import get_logger

_logger = get_logger(__name__)

# Guard 1 (hallucination prevention): shown whenever retrieval finds nothing
# that clears config.similarity_threshold — see rag/retriever.py.
NO_EVIDENCE_MESSAGE = "I could not find sufficient information in the provided document collection."

GREETING_RESPONSE = (
    "Hi! Ask me a medical question and I'll answer using only the indexed "
    "WHO/CDC/NIH documents — I won't rely on outside knowledge."
)

# Anchored to the whole query (not a substring search) so a real medical
# question that happens to contain these words — e.g. "what do you know
# about diabetes?" — falls through to "question" instead of being
# short-circuited before retrieval ever runs.
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|greetings|good (morning|afternoon|evening))[\s!.,]*$",
    re.IGNORECASE,
)
_CAPABILITY_RE = re.compile(
    r"^\s*(what (do you|can you|does (this|the system)) "
    r"(know|do|cover|answer|help with)|"
    r"what (topics|documents|sources|subjects) "
    r"(do you have|are (indexed|covered|available)))\s*\??\s*$",
    re.IGNORECASE,
)


def detect_intent(query: str) -> str:
    """Classify a query as "greeting", "capability", or "question" via plain
    pattern matching — no LLM call, so this can never itself hallucinate or
    misroute a real medical question on a whim. Only matches when the query
    IS the whole greeting/meta-question; a topic tacked on falls through to
    "question" and goes through normal retrieval + generation.
    """
    stripped = query.strip()
    if _GREETING_RE.match(stripped):
        return "greeting"
    if _CAPABILITY_RE.match(stripped):
        return "capability"
    return "question"


def capability_answer(docs: List[dict]) -> str:
    """Deterministic (no LLM call) description of what's actually indexed.

    Even a meta-question about the system's own knowledge is answered from
    the real corpus, not a guess — grounding applies here too.
    """
    if not docs:
        return "No documents are currently indexed."
    orgs = sorted({d["source_org"] for d in docs if d.get("source_org")})
    titles = sorted({d["title"] for d in docs})
    preview = ", ".join(titles[:8])
    if len(titles) > 8:
        preview += f", and {len(titles) - 8} more"
    org_text = f" from {', '.join(orgs)}" if orgs else ""
    return (
        f"I have {len(docs)} documents indexed{org_text}, covering topics "
        f"including: {preview}. Ask me a specific question and I'll answer "
        f"using only this material."
    )

_LLM_ERROR_MESSAGE = (
    "The AI service is temporarily unavailable. Please try again in a moment, "
    "or switch to extractive mode."
)
_LLM_MAX_TOKENS = 1024


def extractive_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    if not retrieved:
        return NO_EVIDENCE_MESSAGE
    lines = [f"Top passages related to: “{query}”\n"]
    for chunk, score in retrieved:
        lines.append(f"[{chunk.doc_title}, score={score:.2f}] {chunk.text}\n")
    return "\n".join(lines)


def _anthropic_answer(system_prompt: str, user_message: str) -> str:
    """Call Anthropic's Messages API and return the grounded answer text."""
    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    try:
        response = client.messages.create(
            model=config.llm_model,
            max_tokens=_LLM_MAX_TOKENS,
            temperature=config.llm_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        _logger.warning("Anthropic call failed: %s", e)
        return _LLM_ERROR_MESSAGE

    if response.stop_reason == "refusal":
        return "The AI declined to answer this question."
    return next((b.text for b in response.content if b.type == "text"), _LLM_ERROR_MESSAGE)


def _openai_answer(system_prompt: str, user_message: str) -> str:
    """Call an OpenAI-compatible Chat Completions API and return the answer text.

    config.openai_base_url lets this target any OpenAI-compatible endpoint
    (e.g. NVIDIA's free-tier catalog) instead of OpenAI's own API — same SDK
    and call shape, just a different host, model name, and key.
    """
    client = openai.OpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url or None)
    try:
        response = client.chat.completions.create(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=_LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
    except (openai.APIStatusError, openai.APIConnectionError) as e:
        _logger.warning("OpenAI call failed: %s", e)
        return _LLM_ERROR_MESSAGE

    return response.choices[0].message.content or _LLM_ERROR_MESSAGE


def llm_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """Generate a grounded answer via the configured LLM provider.

    Falls back to a clear, user-facing message (never a stack trace or raw
    exception) if the provider's API key is missing or the call itself fails.
    """
    if config.llm_provider == "anthropic" and not config.anthropic_api_key:
        return (
            "[LLM mode not configured] Set ANTHROPIC_API_KEY to enable grounded "
            "LLM answers. Falling back to extractive mode:\n\n"
            + extractive_answer(query, retrieved)
        )
    if config.llm_provider == "openai" and not config.openai_api_key:
        return (
            "[LLM mode not configured] Set OPENAI_API_KEY to enable grounded "
            "LLM answers. Falling back to extractive mode:\n\n"
            + extractive_answer(query, retrieved)
        )

    system_prompt, user_message = build_prompt(query, retrieved)

    if config.llm_provider == "anthropic":
        return _anthropic_answer(system_prompt, user_message)
    if config.llm_provider == "openai":
        return _openai_answer(system_prompt, user_message)
    return f"[Configuration error] Unknown LLM_PROVIDER '{config.llm_provider}'."


def generate_answer(query: str, retrieved: List[Tuple[Chunk, float]], mode: str = "extractive") -> str:
    if not retrieved:
        _logger.info("Refused: no evidence retrieved for query: %r", query[:60])
        return NO_EVIDENCE_MESSAGE
    answer = llm_answer(query, retrieved) if mode == "llm" else extractive_answer(query, retrieved)
    _logger.info("Generation completed: mode=%s provider=%s", mode, config.llm_provider if mode == "llm" else "n/a")
    return answer


def confidence_level(retrieved: List[Tuple[Chunk, float]]) -> Optional[str]:
    """Guard 4 (hallucination prevention): a simple, deterministic confidence
    label computed from retrieval evidence alone — never asked of the LLM.

    Returns None when nothing was retrieved (the caller should show the
    refusal message instead of a confidence label). Otherwise:
    - "High": chunks from 2+ distinct source documents, averaging well above
      the similarity threshold — multiple consistent sources agree.
    - "Moderate": supporting evidence exists, but from a single source or
      with middling similarity.
    - "Low": evidence barely cleared the similarity threshold.
    """
    if not retrieved:
        return None
    scores = [score for _, score in retrieved]
    avg_score = sum(scores) / len(scores)
    distinct_docs = {chunk.doc_title for chunk, _ in retrieved}

    if len(distinct_docs) >= 2 and avg_score >= 0.75:
        return "High"
    if avg_score >= 0.6:
        return "Moderate"
    return "Low"
