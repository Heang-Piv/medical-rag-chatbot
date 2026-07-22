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
| M4 | Chunking — replace word-count `chunk_text()` with recursive/boundary-aware splitter | Done |
| M5 | Embeddings — swap TF-IDF in `rag/embed_store.py` for `bge-small-en-v1.5` | Done |
| M6 | ChromaDB — persistent store, separate build step from app startup | Done |
| M7 | Retrieval — configurable top-k (already have the UI slider), similarity-threshold refusal | Done |
| M8 | Prompt engineering — compare 3 strategies, pick one, `rag/prompt.py` | Done |
| M9 | Generation — real LLM call (2 providers), citations, confidence statement (High/Moderate/Low, deterministic) | Done |
| M10 | UI polish — explainability text per chunk, confidence badge, medical disclaimer, error handling, logging | Done |
| — | Evaluation — 10 test questions (easy/medium/hard), results table | Done — `docs/evaluation.md` |
| — | Documentation — README, architecture diagram, design decisions | Done — `README.md` |
| — | Presentation outline, demo script, Q&A prep | Done — `docs/presentation.md` |

Deployed live on Streamlit Community Cloud: https://medical-rag-chatbot-peow.streamlit.app/

## Known gaps / things to watch

- **Re-run the 3 "hard" evaluation questions in `mode="llm"` with a real API key
  before presenting.** `docs/evaluation.md` was written using extractive mode
  only (no API key was available in that environment), so Guard 2/3's
  relevance-judgment refusal behavior is unit-tested
  (`tests/test_generate.py`, `tests/test_prompt.py`) but not yet confirmed
  against a live model. This is the single most important verification step
  left — see the README's Evaluation section and `docs/evaluation.md`'s
  Limitations section for detail.
- Two of the 24 sourced documents (WHO diabetes page, and several CDC/NIH
  pages) were built from strong prior knowledge rather than a freshly
  verified fetch — worth a quick manual click-check on their URLs (see
  `data/medical/manifest.json`) before relying on them being byte-for-byte
  current.

## What's left before presentation day (29 July 2026)

1. Re-run hard-tier questions in LLM mode with a real API key (see above) and
   update `docs/evaluation.md` with the result — right now it's the one claim
   in the whole project that's argued from design/unit-tests rather than
   demonstrated live.
2. Turn `docs/presentation.md`'s outline into actual slides.
3. Rehearse the demo script in `docs/presentation.md` against the live
   Streamlit deployment, not just localhost.

## Working style (carried over from the course meta-prompt)

- One milestone at a time: explain the objective, implement, test, self-review,
  summarize, then stop and wait for a go-ahead before the next one.
- Keep it simple — no microservices, no unnecessary abstractions, no provider
  frameworks for LLMs beyond a plain `if/elif` on `config.llm_provider`.
- Every new function needs type hints, a docstring, and a real reason to exist.
- Git commits should be one-per-milestone (see the meta-prompt's suggested
  commit sequence), not one giant "final version" commit.
