# Presentation materials

Slide outline, demo script, and Q&A prep for the CS382 final presentation
(**29 July 2026**). This is content to build slides from, not a slide deck
itself — turn the outline below into actual slides in whatever tool you like
(Google Slides, PowerPoint, Canva).

**Before you build slides:** re-run the 3 hard-tier evaluation questions in
`mode="llm"` with a real API key (see `docs/evaluation.md`'s Limitations
section and `PROJECT_STATUS.md`). Slide 8 and the demo script below assume
you have a real result to show, not a hypothetical.

---

## Slide outline (12 slides)

### 1. Title

- **Medical RAG Search System** — a Retrieval-Augmented Generation chatbot
  grounded in WHO / CDC / NIH documents
- Your name, CS382 — Search Engines & Information Retrieval, 29 July 2026
- Live demo link + GitHub repo link

**Presenter notes:** Keep this on screen while people settle in. State the
one-sentence pitch out loud: "A chatbot that only answers from real medical
documents, and says so when it can't."

### 2. Problem statement

- General-purpose chatbots answer medical questions from opaque training
  data — no way to verify the source, no way to know if it's current
- Medical misinformation from an LLM is a real, specific-shaped harm, not a
  generic "AI can be wrong" caveat
- The ask: build a search system where every answer is traceable to a real,
  named document

**Presenter notes:** This is the "why does this project exist" slide — spend
30 seconds max, don't over-explain something the audience already knows.

### 3. Objectives

- Answer medical questions using **only** retrieved WHO/CDC/NIH content
- Cite the source document for every claim
- Refuse to answer when there isn't enough evidence, instead of guessing
- Make the retrieval pipeline itself visible and explainable, not hidden
  behind a chat bubble — this is a **search** project first

**Presenter notes:** The last bullet is what separates this from "I called
an LLM API." Emphasize it briefly.

### 4. System architecture

- Four separable layers: **ingestion → vector store/retrieval → generation →
  interface**
- Show the module responsibility table from the README (one row per file)
- Each layer swappable independently (e.g. embedding model, LLM provider)
  without touching the others

**Presenter notes:** Point at the actual files in the repo if presenting
live, or the architecture diagram from the README.

### 5. RAG pipeline (the two Mermaid diagrams from the README)

- **Offline:** documents → chunk → embed → persist to ChromaDB
- **Online:** query → embed → vector search → similarity guard → context
  build → LLM → cited answer
- Call out where each hallucination guard sits in the flow

**Presenter notes:** This is the technical core of the talk — slow down
here. Trace a single example query through the diagram out loud.

### 6. Technologies used

- Embeddings: `BAAI/bge-small-en-v1.5` (sentence-transformers, local)
- Vector store: ChromaDB (persistent)
- LLM: config-driven Anthropic/OpenAI abstraction — live demo runs NVIDIA
  NIM's free-tier `openai/gpt-oss-120b` through the OpenAI code path
- Interface: Streamlit, deployed on Streamlit Community Cloud
- One line each on *why*, not just *what* — see README's Design Decisions
  table

**Presenter notes:** Don't read the table verbatim. Pick the 2 decisions
you find most interesting to justify out loud (embedding model comparison
and prompt strategy comparison are the strongest "I made a real engineering
choice" stories).

### 7. Hallucination prevention (the 4 guards)

- Guard 1: similarity-threshold refusal
- Guard 2: LLM judges relevance, not just topic match, before using a chunk
- Guard 3: mandatory citation, never cite an unprovided source
- Guard 4: deterministic confidence label (High/Moderate/Low)

**Presenter notes:** This is likely to be the most-scrutinized part of the
whole project, given the medical domain. Be ready to explain *why*
similarity threshold alone isn't enough (a query can be "about" a covered
topic without the docs answering the specific question) — that's exactly
what Guard 2 exists to catch.

### 8. Live demo

- (See full demo script below — this slide is just a placeholder/segue)

**Presenter notes:** Actually switch to the live app here. Don't demo from
a recording if you can avoid it — a live refusal on a hard question is the
single most convincing moment in the whole presentation.

### 9. Evaluation results

- 10 test questions, 3 tiers (easy/medium/hard), against the real 24-doc
  corpus
- Easy + medium: all 7 retrieved correct source documents, several with
  genuine cross-document synthesis (e.g. diabetes across WHO/CDC/NIH)
- Hard tier: **state your actual LLM-mode result here** — did it refuse
  correctly on all 3? This is the number that matters most
- Link to `docs/evaluation.md` for full detail

**Presenter notes:** Be honest if the LLM-mode re-verification surfaced a
gap — a professor grading LLO2 ("justify design decisions and *performance
outcomes*") will trust an honestly-reported limitation more than a claim
that everything worked perfectly.

### 10. Limitations

- Small corpus (24 docs / ~44 chunks) — enough to demonstrate the pipeline,
  not a comprehensive medical reference
- Similarity-threshold refusal alone isn't sufficient — needs the LLM
  relevance-judgment layer on top (Guard 2/3)
- English-only, text-only, no conversation memory
- Not a substitute for professional medical advice

**Presenter notes:** Full list is in the README — pick 3-4, don't read all
of them.

### 11. Future improvements

- Hybrid retrieval (BM25 + embeddings) for exact-term matches (drug names,
  dosages)
- Cross-encoder re-ranking for sharper top-of-list precision
- Larger corpus, targeted at the gaps evaluation surfaced
- Conversation memory for multi-turn follow-ups

**Presenter notes:** Frame these as "here's what I'd do with more time,"
not "here's what's broken."

### 12. Lessons learned / conclusion

- What surprised you during development (pick 1-2 genuine ones — e.g.
  similarity-threshold alone not catching the "topically adjacent but
  doesn't answer the question" failure mode)
- One sentence on why retrieval-first design mattered more than prompt
  cleverness for a trustworthy system
- Thank you / Q&A

**Presenter notes:** This slide is genuinely yours to fill in — the most
convincing "lessons learned" answers are specific to something that
actually surprised you while building it, not generic RAG talking points.

---

## Demo script (5-7 minutes)

Rehearse this against the **live Streamlit deployment**
(https://medical-rag-chatbot-peow.streamlit.app/), not localhost — confirm
it still loads and responds before presentation day.

1. **Introduction (30s).** "This is a medical RAG chatbot that only answers
   from WHO, CDC, and NIH documents — 24 of them, indexed as ChromaDB
   vectors. It never uses the LLM's own medical knowledge."

2. **Show the architecture (30s).** Point at the sidebar: top-k slider,
   answer mode toggle, indexed document count. "Every one of these is a
   real, adjustable setting, not a hardcoded value."

3. **Easy question, extractive mode (1 min).** Ask *"What are common
   symptoms of the flu?"* Expand a source chunk. Point out: document title,
   similarity score, the "why this chunk was retrieved" explanation.

4. **Same question, LLM mode (1 min).** Switch Answer mode to `llm`, re-ask.
   Show the generated answer with inline citations and the confidence badge.
   "Same retrieved evidence, now summarized and cited by the model instead
   of shown verbatim."

5. **Cross-document synthesis (1 min).** Ask *"What are the risk factors for
   type 2 diabetes?"* Show that sources are pulled from multiple
   organizations (WHO + CDC + NIH), not just one document.

6. **Unsupported question — the refusal (1-1.5 min, the payoff moment).**
   Ask one of the hard-tier questions, e.g. *"Does eating chocolate cause
   acne?"* or *"Can a specific experimental gene therapy cure diabetes?"* in
   `llm` mode. Narrate what should happen: the system should either refuse
   outright or clearly flag Low confidence with unsupported evidence. **This
   is the moment your hallucination-prevention design gets tested live —
   know in advance which of the 3 hard questions gives the cleanest
   refusal, and lead with that one.**

7. **Explainability + settings (30s).** Briefly show the chunk-size/overlap
   sliders and mention the document upload domain gate (upload something
   obviously off-topic if time allows, to show the rejection).

8. **Conclusion (30s).** "Every answer traces back to a named source
   document with a similarity score, and the system says so honestly when
   it can't find supporting evidence." Hand back to slides for evaluation
   results / Q&A.

---

## Q&A prep

Concise, interview-style answers to likely questions.

**Why did you choose ChromaDB?**
It's a persistent, embedded vector database with native metadata support —
no separate server process to run or deploy, which matters for a project
graded on a live local/hosted demo. For a corpus of this size (tens of
thousands of vectors, not millions), it performs comparably to heavier
options while being far simpler to operate.

**Why not PostgreSQL with pgvector?**
pgvector is the right call at production scale with existing relational
data to join against. Here there's no relational data model to speak of —
just chunks and their embeddings — so running a full Postgres instance
would add operational overhead (connection management, schema migrations)
with no corresponding benefit for a ~44-chunk corpus.

**Why use embeddings instead of keyword search?**
Medical queries are frequently phrased differently than source text (e.g.
"high blood pressure" vs. "hypertension"). Keyword/TF-IDF search misses
that connection entirely; embedding-based semantic search matches on
meaning. The project actually started on TF-IDF (the instructor's starter
baseline) specifically so this upgrade's impact was directly comparable.

**What is Retrieval-Augmented Generation (RAG)?**
Instead of asking an LLM to answer from its training data alone, you first
retrieve relevant passages from a trusted document collection, then give
the LLM those passages as context and ask it to answer *only* from them.
It grounds generation in verifiable, current, domain-specific evidence
instead of the model's frozen, unverifiable training knowledge.

**Why does the chatbot refuse some questions?**
Because a medical chatbot that confidently fabricates an answer is more
dangerous than one that says "I don't know." Refusal is enforced at two
layers: a similarity-threshold floor before generation is even attempted,
and a prompt-level instruction for the LLM to judge whether retrieved
evidence actually answers the question (not just shares its topic) before
using it.

**Why use Top-K retrieval?**
It bounds how much context is sent to the LLM (cost and latency) while
still surfacing multiple corroborating sources when they exist — which
directly feeds the confidence heuristic (2+ distinct sources agreeing
raises confidence from Moderate to High). Top-k is user-adjustable in the
sidebar rather than hardcoded.

**Why did you choose your embedding model?**
Compared BGE-small, BGE-base, and MiniLM on size/speed/retrieval quality.
BGE-small is trained specifically for asymmetric retrieval (short query →
long passage matching), which is exactly this use case, at roughly
MiniLM's size and speed. BGE-base's quality gain didn't justify 2-3x the
model size for a corpus this small.

**Why did you choose your LLM provider?**
The system supports Anthropic Claude and OpenAI natively, plus any
OpenAI-compatible endpoint via `OPENAI_BASE_URL`. For the live demo I'm
running NVIDIA NIM's free-tier `gpt-oss-120b` through that same OpenAI code
path — same abstraction, zero-cost inference, which matters for a student
project that needs to stay demoable without a paid API key.

**How would you improve this project in the future?**
Hybrid retrieval (BM25 + embeddings) to catch exact-term matches like drug
names that pure semantic search can miss, and cross-encoder re-ranking to
sharpen precision at the top of the retrieved list — both are standard
next steps in production RAG systems that were out of scope for a
3-4-week project.

**What challenges did you encounter?**
Similarity-threshold refusal alone isn't sufficient — retrieval can return
chunks that are topically related to a question without actually answering
it (e.g. retrieving general diabetes-overview text for a question about an
unproven gene therapy). That required a second guard at the prompt level
that explicitly judges relevance, not just topic match, before the model is
allowed to use an excerpt.

**What trade-offs did you make?**
Kept the LLM provider integration to a plain `if/elif` on a config value
instead of a plugin/adapter framework — two real providers prove the
abstraction is sound without overbuilding for a project this size. Also
kept the confidence score fully deterministic (computed from retrieval
statistics) rather than LLM-generated, since an LLM-reported confidence
score is itself an unverified claim, not real evidence.
