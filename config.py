"""
Central configuration for the medical RAG system.

Every tunable value lives here, sourced from environment variables (via .env)
with a sensible default. Nothing in rag/ or app.py should hardcode a model
name, path, or threshold directly — import it from here instead, so the whole
system can be reconfigured without touching code.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _get_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Config:
    # --- Data & storage ---
    data_folder: str = os.environ.get("DATA_FOLDER", "data/medical")
    chroma_persist_dir: str = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db")

    # --- Chunking ---
    chunk_size_tokens: int = _get_int("CHUNK_SIZE_TOKENS", 350)
    chunk_overlap_tokens: int = _get_int("CHUNK_OVERLAP_TOKENS", 50)

    # --- Embeddings ---
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # --- Retrieval ---
    top_k_default: int = _get_int("TOP_K_DEFAULT", 4)
    similarity_threshold: float = _get_float("SIMILARITY_THRESHOLD", 0.5)

    # --- Upload domain gate ---
    # Minimum rag.embed_store.domain_relevance_score() an uploaded document
    # must reach to be indexed, so the corpus stays medical/health content
    # rather than whatever a user happens to upload. Calibrated against this
    # project's real corpus (see domain_relevance_score's docstring).
    upload_relevance_threshold: float = _get_float("UPLOAD_RELEVANCE_THRESHOLD", 0.58)

    # --- LLM generation ---
    llm_provider: str = os.environ.get("LLM_PROVIDER", "anthropic")  # anthropic | openai
    llm_model: str = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
    llm_temperature: float = _get_float("LLM_TEMPERATURE", 0.0)
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    # Empty = OpenAI's own API. Override to point the same OpenAI-SDK call at
    # any OpenAI-compatible endpoint instead — e.g. NVIDIA's free-tier model
    # catalog at https://integrate.api.nvidia.com/v1 (set LLM_MODEL to a
    # model it hosts, e.g. "openai/gpt-oss-120b", and OPENAI_API_KEY to your
    # NVIDIA key).
    openai_base_url: str = os.environ.get("OPENAI_BASE_URL", "")

    # --- Logging ---
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")


config = Config()
