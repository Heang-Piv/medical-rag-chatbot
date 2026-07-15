# Project status — Medical RAG chatbot (CS382 final project)

Read this first in a new session. It's the handoff between planning (done in
claude.ai) and implementation (Claude Code, from here on).

## Assignment

- Official brief: CS382 Final Project — RAG-based AI search system. Presentation
  day 29 July 2026. Full text: `docs/official_brief.md`.
- Course meta-prompt (collaboration style, milestone roadmap, medical-domain
  requirements, quality checklists): `docs/course_meta_prompt.md`.
- Domain chosen: medical information retrieval (WHO / CDC / NIH sources only).
  Chatbot must refuse to answer when evidence isn't in the retrieved documents.
- Full requirement analysis, architecture proposal, and milestone plan were
  worked out in a prior claude.ai conversation — this file carries forward only
  the decisions and current state. Read the two docs above for full detail;
  read this file for what's actually been decided and built so far.

## Decisions locked in (don't re-litigate without a reason)

| Area | Decision | Why (short) |
|---|---|---|
| Vector store | ChromaDB, persistent | Avoids re-embedding the corpus on every app restart; native metadata support |
| Embedding model | `BAAI/bge-small-en-v1.5` | Same size/speed as MiniLM, trained for retrieval specifically; bge-base gains aren't worth 2-3x the size for this corpus |
| Chunking | Recursive, paragraph/sentence-aware, ~350 tokens / ~50 token overlap | Current word-count chunker (M2-era) doesn't respect boundaries — this is M4's job |
| LLM providers | Anthropic (primary) + OpenAI (secondary), config-driven `if/elif`, no plugin framework | Two real providers proves the abstraction without overbuilding |
| UI | Streamlit (unchanged from starter) | Already satisfies interface requirements; no reason to switch |
| Config | `config.py` reads from `.env`, no hardcoded values anywhere else | Required by both the brief and course meta-prompt |

## Milestones

| # | Milestone | Status |
|---|---|---|
| M0 | Codebase analysis | Done |
| M1 | Environment setup (`config.py`, `.env.example`, `.gitignore`) | Done |
| M2 | Medical dataset — 24 docs, WHO/CDC/NIH, `data/medical/{who,cdc,nih}/` + `manifest.json` | Done |
| M3 | Ingestion upgrade — recursive loader, `.txt`/`.md`/`.pdf`, metadata (source_org, source_url) on every `Chunk` | Done |
| M4 | Chunking — replace word-count `chunk_text()` with recursive/boundary-aware splitter | **Next** |
| M5 | Embeddings — swap TF-IDF in `rag/embed_store.py` for `bge-small-en-v1.5` | Not started |
| M6 | ChromaDB — persistent store, separate build step from app startup | Not started |
| M7 | Retrieval — configurable top-k (already have the UI slider), similarity-threshold refusal | Not started |
| M8 | Prompt engineering — compare 3 strategies, pick one, `rag/prompt.py` | Not started |
| M9 | Generation — real LLM call (2 providers), citations, confidence statement (High/Moderate/Low, deterministic) | Not started |
| M10 | UI polish — explainability text per chunk, confidence badge, medical disclaimer, error handling, logging | Not started |
| — | Evaluation — 10 test questions (easy/medium/hard), results table | Not started |
| — | Documentation — README, architecture diagram, design-decisions doc, presentation outline, demo script | Not started |

## Known gaps / things to watch

- Two of the 24 sourced documents (WHO diabetes page, and 6 CDC/NIH pages) were
  built from strong prior knowledge rather than a freshly verified fetch —
  worth a quick manual click-check on their URLs (see `data/medical/manifest.json`)
  before relying on them being byte-for-byte current.
- `rag/embed_store.py` still runs TF-IDF (unchanged since the starter). M5 is
  the milestone that replaces it — don't be surprised retrieval quality is
  still mediocre until then.
- `chunk_text()` in `rag/ingest.py` is still the naive word-count splitter.
  Its interface (`chunk_text(text) -> List[str]`) is meant to stay stable
  through the M4 rewrite so nothing downstream has to change.

## Working style (carried over from the course meta-prompt)

- One milestone at a time: explain the objective, implement, test, self-review,
  summarize, then stop and wait for a go-ahead before the next one.
- Keep it simple — no microservices, no unnecessary abstractions, no provider
  frameworks for LLMs beyond a plain `if/elif` on `config.llm_provider`.
- Every new function needs type hints, a docstring, and a real reason to exist.
- Git commits should be one-per-milestone (see the meta-prompt's suggested
  commit sequence), not one giant "final version" commit.
