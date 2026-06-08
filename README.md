# NBA CBA RAG System

A retrieval-augmented generation system for the 2023 NBA Collective Bargaining Agreement. Given a natural language question, it retrieves the relevant CBA sections and generates a grounded, citation-level answer using Claude.

---

## Architecture

```
PDF → [Chunking] → [BM25 fit] → [Dense embed] → Pinecone
                                              ↓
Question → [Jargon expansion] → [Hybrid query] → [Dedup] → Chunks
                                                              ↓
                                                         [Generate]
                                                              ↓
                                                     Answer + citations
```

The system has three layers: ingestion, retrieval, and generation. Each is independently runnable.

---

## Design Decisions

### 1. Chunking: structure-aware, not fixed-size

Most RAG tutorials use fixed-size character or token windows. That works for prose. Legal text is different — the CBA is organized into Articles and Sections, and every subsection (a)(b)(i)(ii)...) belongs to a specific provision. Splitting at arbitrary boundaries severs the relationship between a sub-rule and its parent clause, making it impossible to answer "what are the exceptions to X?" without also retrieving the rule X itself.

**The chunking pipeline has three passes:**

**Pass 1 — Structural split (`chunk_by_section`):** Splits only at `ARTICLE` and `Section N.` boundaries using regex. Subsections accumulate in the parent section's body rather than becoming independent chunks. This keeps legal provisions whole.

**Pass 2 — Size normalization (`post_process_chunks`):** Sections exceeding 1,500 characters are recursively split on `\n\n`, `\n`, then `. ` with 200-character overlap between pieces. Sub-minimum stubs (<150 chars) are absorbed into the previous same-section chunk rather than dropped, preserving text that would otherwise disappear.

**Pass 3 — Section intro injection (`inject_section_intros`):** Large sections (e.g. Article VII Section 6, which covers every salary cap exception) produce multiple chunks. A sub-rule chunk about the "Early Bird Exception" doesn't carry the words "salary cap exceptions" — only the first chunk of that section does. This causes it to score poorly for queries about the section's broader topic.

The fix: bake the first 300 characters of each section's opening chunk into the `embed_text` of every subsequent chunk in that section. This means every Article VII Section 6 sub-chunk carries the section purpose in its embedding. The `chunk_store.json` display text also prepends a truncated section header so Claude sees which section a sub-rule belongs to.

**Result:** 787 chunks across 42 articles.

---

### 2. Two-store design

Pinecone enforces a 40KB metadata limit per vector. Long legal clauses hit this limit easily, and storing full text in metadata also bloats index size and query payloads unnecessarily.

**Solution:** Pinecone stores vectors + lightweight metadata (article, section, label, chunk_id). Full chunk text lives in a local `chunk_store.json`, keyed by `chunk_id`. At query time, Pinecone returns IDs; the retriever hydrates text from the local store. The two stores must stay in sync — `--reset` clears Pinecone before reingest.

---

### 3. Hybrid retrieval with tuned alpha

Pure dense retrieval struggles with precise legal terminology — the embedding model doesn't know that "Bird Rights" and "qualifying veteran free agent exception" are the same concept. Pure BM25 struggles with paraphrasing and synonym queries.

**Hybrid search** (Pinecone's built-in dense + BM25 sparse) combines both signals. The balance is controlled by `alpha` (1.0 = pure dense, 0.0 = pure BM25). Alpha was tuned via grid search over the eval set, selecting `alpha = 0.68` based on Hit@3 — the metric that matters most since the top-3/5 chunks feed the LLM.

**Query expansion** handles the terminology gap directly: a 30-term jargon dictionary maps colloquial NBA terms to their CBA equivalents before embedding. Normalization handles punctuation variants; a fuzzy sliding-window match (difflib, threshold 0.85) catches multi-word key misspellings. Single-word keys use exact match only to avoid false positives.

**Over-fetching:** The retriever queries Pinecone for `top_k × 8` candidates. This is necessary because large sections generate many chunks, and the most relevant chunk for a specific sub-rule query might rank 20th overall but first within its section after deduplication.

**Section-level deduplication:** Candidates are grouped by `(article, section)`. Within each group, chunks are ranked by keyword overlap with the expanded query, and the top 2 are kept (`MAX_PER_SECTION = 2`). This prevents a high-scoring but off-topic sibling chunk from displacing the most relevant one, while still allowing multi-chunk sections to contribute more than one result when the query spans sub-rules.

**Retrieval metrics (historical baseline, 37 annotated queries, alpha=0.68):**

| Hit@1 | Hit@3 | Hit@5 | MRR   | Recall@5 |
|-------|-------|-------|-------|----------|
| 0.730 | 0.946 | 1.000 | 0.838 | 0.804    |

---

### 4. Agentic generation with parallel tool calls

The CBA is a heavily cross-referenced document. A question about sign-and-trades may require Article VII Section 8 (the rule), Article VII Section 6 (cap exceptions), and Article XI (free agency eligibility). A single retrieve pass doesn't reliably surface all of them.

The agentic pipeline (`rag_agentic`) addresses this:

1. **Seed context:** Run one retrieve pass and give Claude the initial chunks.
2. **Tool-use loop:** Offer Claude a `retrieve` tool for up to `MAX_TOOL_ROUNDS = 5` additional rounds. The system prompt instructs it to issue all needed lookups in the same message ("they run in parallel").
3. **Parallel execution:** When Claude issues multiple tool calls in one turn, they're dispatched concurrently via `ThreadPoolExecutor`, each hitting Pinecone independently. New chunks are deduplicated against `seen_ids` across rounds.
4. **Forced answer:** On the final round, the tool is withheld and a summary instruction is injected into the last user turn, ensuring Claude produces a text answer rather than continuing to search.

The non-agentic `rag()` pipeline (single retrieve → single generate) is kept for eval scripts where determinism and cost matter more than completeness.

A cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is wired up but disabled by default. It was trained on MS MARCO web search and hurts ranking on legal text. A domain-matched reranker (e.g. Cohere Rerank) would be needed for this to help.

---

### 5. Evaluation

**Retrieval:** `eval_retrieval.py` runs 37 hand-crafted queries covering salary cap rules, trades, extensions, free agency, drug policy, and CBA governance. Metrics: Hit@K, MRR, Recall@5.

**Generation:** `opik_eval.py` runs LLM-as-a-judge scoring via Opik against a dataset of 8 seed queries. Three metrics:
- **AnswerRelevance** — is the answer on-topic?
- **Hallucination** — does the answer stay grounded in the retrieved excerpts?
- **KeyInfoPresent** (custom) — does the answer contain a specified required fact? Claude Haiku scores each output with a YES/NO + one-sentence explanation. Queries without a `must_include` fact skip this check (score 1.0).

Every `retrieve()`, `generate()`, and `rag*()` call is decorated with `@opik.track()`, so all pipeline runs — including ad-hoc queries — are logged to Opik automatically.

**Caveat — training-data leakage:** The 2023 NBA CBA is a public document that predates Claude's training cutoff, so it is almost certainly present in the model's training data. This means Claude can answer many CBA questions from parametric memory alone, without relying on the retrieved excerpts. Two consequences for evaluation:

- **Generation metrics are optimistic.** A correct, relevant answer doesn't prove the retrieval pipeline surfaced the right context — the model may have known the answer already. AnswerRelevance and KeyInfoPresent scores partly measure Claude's prior knowledge, not just this system's retrieval quality.
- **Hallucination/grounding is harder to assess.** An answer can be factually correct yet not actually grounded in the provided chunks, which a grounding judge may still pass.

The retrieval metrics (Hit@K, MRR, Recall@5) are unaffected — they measure the retriever directly against annotated ground truth and never touch the LLM. For a leakage-free read on the generation layer, the system should be evaluated against a corpus the model has *not* seen (e.g. a private or post-cutoff agreement), or with retrieval deliberately ablated to isolate its contribution.

---

## Next Steps

The next logical step is to layer a graph database over the corpus, modeling the CBA's key concepts (defined terms, articles, sections) as nodes and its cross-references as edges. Because the CBA is a densely cross-referenced document, traversing these relationships at query time would pull in the supporting rules and definitions an answer depends on — even when they share little vocabulary with the question — improving retrieval quality on multi-hop questions.

---

## Stack

| Layer | Technology |
|---|---|
| PDF extraction | PyMuPDF |
| Chunking | Custom regex parser (3-pass) |
| Embedding | `multi-qa-mpnet-base-dot-v1` (768d, dot product, Q&A-optimized) |
| Sparse vectors | Pinecone BM25Encoder (fit on corpus) |
| Vector store | Pinecone serverless (AWS us-east-1) |
| Hybrid alpha tuning | Jupyter notebook, grid search |
| Generation | Anthropic Claude (Haiku default, Sonnet supported) |
| Observability | Opik (traces every pipeline call) |
| LLM eval | Opik evaluate + custom `KeyInfoPresent` metric |

---

## Project Structure

```
src/
  ingestion/
    ingest.py          — PDF → 3-pass chunking → BM25 fit → Pinecone upsert
  retrieval/
    retrieve.py        — jargon expansion → hybrid query → dedup → RankedChunk[]
  generation/
    generate.py        — rag() single-shot + rag_agentic() tool-use loop
  eval/
    eval_retrieval.py  — 37-query retrieval sweep (Hit@K, MRR, Recall@5)
    opik_eval.py       — LLM-as-a-judge: AnswerRelevance, Hallucination, KeyInfoPresent

data/
  raw/                 — nba_cba_2023.pdf (gitignored)
  processed/           — chunk_store.json, bm25_params.json (gitignored)

notebooks/
  tune_alpha.ipynb     — alpha grid search over eval set

app.py                 — minimal Streamlit web UI over the RAG pipeline
```

## Quickstart

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Add API keys to .env.local
# PINECONE_API_KEY=...
# ANTHROPIC_API_KEY=...
# OPIK_API_KEY=...

# Ingest
cd src/ingestion
python ingest.py --pdf ../../data/raw/nba_cba_2023.pdf

# Ask a question (agentic, default)
cd ../..
.venv/bin/python src/generation/generate.py "How do Bird Rights work?"

# Or launch the web UI
.venv/bin/streamlit run app.py

# Single-shot mode
.venv/bin/python src/generation/generate.py "What is the luxury tax threshold?" --no-agent

# Retrieval only
.venv/bin/python src/retrieval/retrieve.py "What are the rules for two-way contracts?" --top-k 5

# Run LLM eval
.venv/bin/python src/eval/opik_eval.py --experiment "baseline"
```
