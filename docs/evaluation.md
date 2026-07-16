# Retrieval & Generation Evaluation

10 test questions across three difficulty tiers, run against the real,
built index (24 documents → 44 chunks from `data/medical/{who,cdc,nih}/`).
Retrieval used the actual pipeline (`rag.retriever.retrieve`, top_k=4,
similarity_threshold=0.5 — both from `config`); generation used **extractive
mode** (`rag.generate.extractive_answer`), which stitches retrieved passages
together verbatim with no LLM call. Every question below is real output from
the running system, not a hypothetical.

## Why extractive mode, not LLM mode

This evaluation environment has no configured `ANTHROPIC_API_KEY` /
`OPENAI_API_KEY`, so `mode="llm"` (M9) could not be exercised live end to
end — only unit-tested against a mocked client (`tests/test_generate.py`).
That matters for the "hard" tier below: extractive mode has no way to judge
whether a retrieved passage actually answers the question (that judgment is
the system prompt's job — Guard 2/3 in `rag/prompt.py`, M8). **Before your
presentation, re-run the three hard questions with `mode="llm"` and a real
API key** to confirm the LLM actually refuses where extractive mode
couldn't. That's the real test of this project's hallucination-prevention
design, not something extractive mode alone can demonstrate.

## Results

| # | Tier | Question | Expected source(s) | Retrieved correctly? | Confidence | Correct/refused as expected? | Notes |
|---|------|----------|---------------------|----------------------|------------|-------------------------------|-------|
| 1 | Easy | What are common symptoms of the flu? | CDC *Flu*, WHO *Influenza (seasonal)* | Yes — both docs, top 2 hits (0.84, 0.78) | High | Yes | Clean, on-topic retrieval from both flu documents. |
| 2 | Easy | How is malaria transmitted? | WHO *Malaria* | Partial — Malaria doc is top hit (0.80), but 2 of 4 slots went to HIV docs (0.66–0.67) | Moderate | Yes (answer itself is correct) | top_k=4 padded with lower-relevance HIV chunks once malaria-specific chunks ran out; correctly reflected as Moderate, not High, confidence. |
| 3 | Easy | What is asthma? | CDC *Asthma* | Partial — Asthma doc is a clear top hit (0.85), remaining 3 slots are filler (flu, malaria, depression, 0.60–0.64) | Moderate | Yes (top passage is correct) | Corpus has limited asthma-specific content; only one genuinely relevant chunk exists. |
| 4 | Easy | What causes tuberculosis? | WHO *Tuberculosis* | Yes — both TB chunks in top 2 (0.81, 0.71) | Moderate | Yes | Solid, on-topic. |
| 5 | Medium | What are the risk factors for type 2 diabetes? | WHO/CDC/NIH diabetes docs | Yes — all 3 orgs' diabetes docs retrieved | High | Yes | Good cross-document synthesis; exactly the "combine multiple chunks" case this tier is meant to test. |
| 6 | Medium | How are high blood pressure and heart disease related? | CDC/NIH high blood pressure + CDC heart disease | Yes — all 4 slots are on-topic across 3 docs | High | Yes | Correctly pulled both conditions' documents together. |
| 7 | Medium | What lifestyle changes can help prevent cardiovascular disease? | WHO *Cardiovascular diseases*, NIH *Heart-Healthy Living* | Yes — both expected docs present, plus a plausible bonus (NIH *Healthy Aging*) | High | Yes | Retrieval reasonably extended to a related-but-unrequested doc; still on-topic. |
| 8 | Hard | Can a specific experimental gene therapy cure diabetes? | *(none — not discussed in corpus)* | N/A | Moderate (0.64 avg) | **No — should have refused, did not** | Retrieved general diabetes-overview text that never mentions gene therapy. Not a hallucination (real text, verbatim), but not a correct or useful answer either — a naive extractive-only system would present this as if relevant. |
| 9 | Hard | What is the recommended dosage of ibuprofen for a child's fever? | *(none)* | N/A | Low (0.56 avg) | **No — should have refused, did not** | Retrieved malaria-prevention text with zero relevance to the question. Lowest-confidence tier of all 10 questions, which is at least directionally honest. |
| 10 | Hard | Does eating chocolate cause acne? | *(none)* | N/A | Low (0.54 avg) | **No — should have refused, did not** | Retrieved obesity statistics; no connection to the actual question. Weakest similarity scores of the set (0.53–0.55), just barely above the 0.5 threshold. |

## Strengths

- **Easy and medium tiers performed well.** All 7 non-hard questions retrieved genuinely relevant documents, and the medium tier's cross-document synthesis (diabetes across all 3 source orgs, blood-pressure↔heart-disease linkage) is exactly the behavior that tier is meant to test.
- **The confidence heuristic (Guard 4) tracks retrieval quality sensibly**, not just superficially. It's lowest exactly on the 3 hard, unsupported questions (Moderate/Low) and on the 2 easy questions where the corpus only has one truly relevant chunk (asthma, malaria) — even though those two still produced a correct top answer. Confidence is measuring evidence *strength*, not answer correctness, and that distinction held up under real testing.
- **No hallucination occurred at the retrieval/extractive layer**, because extractive mode is architecturally incapable of it — it only ever echoes real retrieved text. This is a hard guarantee, not a probabilistic one.

## Limitations — read before the presentation

- **Similarity-threshold refusal (Guard 1) alone did not refuse any of the 3 hard questions.** All three retrieved topically-adjacent chunks (diabetes, malaria, obesity) that cleared 0.5 similarity purely on topical relevance, without ever addressing the actual claim in the question. This is the known, previously-flagged gap (see M7's self-review): a query can be "about" a covered topic without the documents answering the *specific* question asked.
- **This is precisely what M8's system prompt (Guard 2/3) exists to catch** — it explicitly instructs the model to judge whether each excerpt actually supports the question before using it, and to refuse with the exact required wording otherwise. That logic is unit-tested (`tests/test_prompt.py`, `tests/test_generate.py`) but **not exercised against a live model in this evaluation**, since no API key was available. Re-running questions 8–10 with `mode="llm"` and a real key is the single most important verification step remaining before the demo — if the LLM doesn't refuse these three, the hallucination-prevention story has a real hole, not just a theoretical one.
- **top_k=4 sometimes pads results with weaker filler** once a narrow topic runs out of genuinely relevant chunks (asthma, malaria) — harmless here since the top hit was still correct, but worth knowing if top_k is raised for the demo.
- **10 questions against 44 chunks is a small sample.** These results indicate the pipeline behaves sensibly, not that it's exhaustively validated — don't present this as more rigorous than it is.
