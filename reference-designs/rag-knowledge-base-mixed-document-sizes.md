# RAG Knowledge Base for Large Document Collections of Mixed Sizes

| Field | Value |
|-------|-------|
| Created | 2026-05-04 |
| Last Updated | 2026-05-30 |
| Version | 1.2 |

---

- [1. Scope and Design Goals](#1-scope-and-design-goals)
- [2. Reference Architecture](#2-reference-architecture)
- [3. Ingestion Pipeline](#3-ingestion-pipeline)
- [4. Storage Layer](#4-storage-layer)
- [5. Retrieval Pipeline](#5-retrieval-pipeline)
- [6. Answer Generation](#6-answer-generation)
- [7. Reference Parameters](#7-reference-parameters)
- [8. Anti-Patterns](#8-anti-patterns)
- [9. Open-Source Reference Stack](#9-open-source-reference-stack)
- [10. Hyperscaler Managed-Service Mappings](#10-hyperscaler-managed-service-mappings)
- [11. Evaluation Methodology](#11-evaluation-methodology)
- [12. When to Deviate from This Design](#12-when-to-deviate-from-this-design)
- [13. Engineering Readiness Pack](#13-engineering-readiness-pack)
- [References](#references)

This is the canonical reference design for a retrieval-augmented generation (RAG) knowledge base that must serve a large number of documents of widely varying sizes (single-page memos through 600+ page technical manuals) without per-document tuning. It consolidates the production design validated through 14 evaluation iterations on the `doc-agent` branch of the ba-ai-discovery codebase, with the broader open-source landscape as fallback options. The design is platform-neutral; hyperscaler equivalents are listed in §10.

For background and surveys of the field, see [`rag-and-context-engineering.md`](../rag/rag-and-context-engineering.md) and [`large-document-llm-methods.md`](../rag/large-document-llm-methods.md).

---

## 1. Scope and Design Goals

### Use cases this design targets

- Knowledge bases containing **dozens to thousands of documents**.
- Document sizes spanning **at least two orders of magnitude** (e.g. 1 page to 600 pages).
- Mixed content: structured reports, technical manuals, contracts, narrative documents, OCR'd scans.
- A single retrieval pipeline that must work across all of the above without per-document configuration.

### Goals

| Goal | Why |
|---|---|
| **Scale-invariant retrieval quality** | A 35-page memo and a 585-page manual must both produce useful answers from the same pipeline. |
| **Predictable latency** | Sub-second retrieval, sub-3s end-to-end is the production target. |
| **Citation-grounded answers** | Every answer cites the source chunk, document, and section. |
| **One database, one query path** | Avoid multi-system distributed-search architectures unless cardinality demands it. |
| **Pluggable models** | Embedding model, reranker, and answer LLM are independently swappable. |
| **Backwards-compatible rollout** | Every component change is feature-flagged and A/B tested. |

### Non-goals

- Real-time index updates measured in milliseconds (indexing is a batch operation).
- Cross-language retrieval at frontier quality (multilingual is supported but English is the design target).
- Replacing structured database queries — RAG is for unstructured content.

---

## 2. Reference Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION (offline)                          │
│                                                                      │
│  Source Document → Parse → Distil Structure → Section-Aware Chunk    │
│                                              ↓                       │
│                                    Contextual Embedding Prefix       │
│                                              ↓                       │
│                                          Embed                       │
│                                              ↓                       │
│                                   Store: vectors + tsvector          │
│                                          (one row per chunk)         │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        STORAGE (PostgreSQL)                          │
│                                                                      │
│   chunks(id, doc_id, section_id, text, embedding vector(d),          │
│          tsvector, chunk_index, metadata jsonb)                      │
│   sections(id, doc_id, title, summary, level, parent_id)             │
│   documents(id, title, type, summary, page_count, ...)               │
└─────────────────────────────────────────────────────────────────────┘
                                  ↑                ↑
                       ┌──────────┘                └──────────┐
                       │                                      │
┌─────────────────────────────────────────────────────────────────────┐
│                       RETRIEVAL (per query)                          │
│                                                                      │
│   Query → [Decompose?] → ┬─ Dense Search (cosine, top-40) ─┐         │
│                          └─ BM25 Search (ts_rank_cd, top-40)┤        │
│                                                              ↓       │
│                                            RRF Merge (k=60, 0.7/0.3) │
│                                                              ↓       │
│                                         Cross-Encoder Rerank top-20  │
│                                                              ↓       │
│                                          Select top-5 + ±1 neighbours│
│                                                              ↓       │
│                                  Enrich with section/doc summaries   │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       ANSWER GENERATION                              │
│                                                                      │
│   System prompt + enriched context + question → Answer LLM           │
│                       → Cited answer (chunk IDs + relevance scores)  │
└─────────────────────────────────────────────────────────────────────┘
```

The pipeline is deliberately small. Every box is independently testable. There is no orchestration framework, no agentic loop, no graph database — those belong in extensions, not the reference core.

---

## 3. Ingestion Pipeline

### 3.1 Document parsing

Route documents by type at upload:

| Input | Parser |
|---|---|
| Clean digital PDF (extractable text) | `pymupdf` / `pypdf` |
| Scanned PDF, complex layout, tables | VLM-based: **Docling** (Apache 2.0) or **MinerU** (Apache 2.0) |
| Office (DOCX, PPTX) | `python-docx`, `python-pptx` |
| Markdown, plain text | Pass through |
| HTML | `trafilatura` or `readability-lxml` for main-content extraction |

Output of every parser is **markdown** — this gives a uniform downstream pipeline and makes section detection trivial via ATX headings (`#`, `##`, ...).

For complex/scanned documents, invest in VLM parsing. The accuracy gap between Tesseract-class OCR (65–78%) and modern VLM parsers (85–95%) propagates downstream as retrieval failures that no clever ranking can recover. The ICCV 2025 "OCR Hinders RAG" paper documents this.

### 3.2 Knowledge distillation

Before chunking, send the first ~12,000 characters and detected headings to an approved fast LLM from the deployment's verified model shortlist. It returns:

- Document **title**, **type**, and a **3–5 sentence summary**
- **Section hierarchy** with titles, levels, and per-section summaries

Heading depth scales with document length to prevent section explosion on large documents:

| Document size | Heading depth retained |
|---|---|
| < 50 pages | Levels 1–3 |
| 50–200 pages | Levels 1–2 |
| 200+ pages | Level 1 only |

This metadata is **reused at three downstream stages**: as embedding prefix material (§3.4), as context enrichment at answer time (§6), and as a soft section-relevance signal if you later add re-ranking by section.

### 3.3 Section-aware chunking

1. Split first at detected section boundaries.
2. Within each section, apply fixed-size chunking (1000 chars, 200 overlap) using `RecursiveCharacterTextSplitter` or equivalent.

This sequencing prevents a single chunk from straddling two unrelated topics — the most common failure mode of pure fixed-size chunking on structured documents. Chunk size of 1000 characters is a reasonable default; section-aware splitting matters more than the exact size.

For documents without detectable structure (raw OCR, pure narrative), fall through to recursive character splitting at paragraph and sentence boundaries.

### 3.4 Contextual embedding prefix

Before embedding, prepend each chunk with:

```
Document: <document_type> — <document_title>
Section: <section_title>
Section summary: <section_summary>

<chunk_text>
```

The prefix is used **only for embedding**. The stored chunk text is the clean original — the prefix is reconstructed on the fly at index time from the metadata tables.

This is a lighter version of Anthropic's contextual retrieval (which prepends an LLM-generated chunk-specific situating sentence). The full Anthropic version reduces retrieval failures by up to 67% combined with hybrid search and reranking, but costs one LLM call per chunk at indexing time. The lightweight prefix here gets most of the benefit at zero per-chunk cost.

**Upgrade path:** If retrieval quality remains insufficient on hard documents, add the full Anthropic contextual-retrieval LLM call per chunk with prompt caching to amortise the cost (~10x reduction with caching).

### 3.5 Embedding

**Default:** Gemini Embedding 001 (3072-dim) via LiteLLM proxy (or your hyperscaler embedding gateway).

**Open-source default:** **BGE-M3** (BAAI, MIT licence, 1024-dim) — the strongest general-purpose open-source embedding model that supports dense, sparse, and ColBERT-style multi-vector outputs from a single forward pass.

**Other validated options:** `nomic-embed-text-v1.5` (Apache 2.0, 768-dim, fully open weights/data/training), `jina-embeddings-v3` (CC BY-NC 4.0 — commercial licence required for production use; supports late chunking and task-specific LoRA adapters).

**Critical pgvector detail.** pgvector 0.8.x caps HNSW and IVFFlat indexes at **2000 dimensions**. For 3072-dim Gemini embeddings, either:

- Sequential scan only (acceptable up to ~10k chunks).
- Use Matryoshka truncation to 1536 dimensions and index normally.
- Use a vector store that supports higher dimensions natively (Qdrant, Milvus, Weaviate).

Above ~100k chunks, sequential scan is no longer viable — index or shard.

### 3.6 Storage write

Write atomically per document: insert document row, section rows, then chunk rows. Failed ingests must be fully reversible (idempotent re-ingest by document ID).

---

## 4. Storage Layer

A single PostgreSQL database with the `pgvector` extension is the reference storage layer. It holds vectors and full-text indexes in the same table, which is the single biggest simplification this design makes.

### Schema sketch

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id           UUID PRIMARY KEY,
    title        TEXT NOT NULL,
    doc_type     TEXT,
    summary      TEXT,
    page_count   INT,
    metadata     JSONB,
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE sections (
    id           UUID PRIMARY KEY,
    doc_id       UUID REFERENCES documents(id) ON DELETE CASCADE,
    title        TEXT,
    summary      TEXT,
    level        INT,
    parent_id    UUID REFERENCES sections(id),
    ordinal      INT
);

CREATE TABLE chunks (
    id           UUID PRIMARY KEY,
    doc_id       UUID REFERENCES documents(id) ON DELETE CASCADE,
    section_id   UUID REFERENCES sections(id),
    chunk_index  INT NOT NULL,                   -- ordering within doc
    text         TEXT NOT NULL,
    embedding    vector(1536),                   -- dims per chosen model
    tsv          tsvector
                 GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    metadata     JSONB
);

CREATE INDEX chunks_embed_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_gin
    ON chunks USING gin (tsv);
CREATE INDEX chunks_doc_idx
    ON chunks (doc_id, chunk_index);
```

### Why pgvector (not a dedicated vector DB)

| Reason | Explanation |
|---|---|
| **One transaction boundary** | Inserts are atomic across vector + tsvector + metadata. |
| **One query** | Hybrid search merges results from the same table — no cross-system join. |
| **Operational simplicity** | One database to back up, monitor, secure, and audit. |
| **RBAC alignment** | Row-level security applies uniformly to vector and full-text results. |
| **Sufficient at ≤10M chunks** | Most knowledge bases never exceed this. |

**Switch to a dedicated vector DB (Qdrant, Milvus, Weaviate) when:**
- Chunk count exceeds ~10M and HNSW build/recall on pgvector becomes a bottleneck.
- You need >2000-dim vectors with indexing and don't want to truncate.
- You need filtering features (e.g. complex metadata pre-filtering at scale) that PostgreSQL doesn't deliver efficiently.

### Sharding strategy at scale

If a single tenant exceeds ~10M chunks, shard by tenant or document type before sharding by chunk hash. Tenant-aligned shards preserve the locality of typical queries and simplify per-tenant deletion (GDPR, data-retention).

---

## 5. Retrieval Pipeline

### 5.1 Optional: query decomposition

A small LLM decides whether to decompose the query into 2–4 focused sub-queries. Simple factual queries pass through unchanged; multi-part or broad-scope queries get split, each retrieves independently, and candidate sets are merged (deduplicated by chunk ID) before reranking.

Most valuable on **large documents** where relevant information is scattered across sections. Adds one LLM call before retrieval. Skip this for first-pass implementations and add later if retrieval misses on multi-part queries.

### 5.2 Two-signal retrieval

Run **in parallel**:

**Dense vector search.** Cosine similarity against the query embedding. Captures semantic matches — "income inequality" matches "wealth concentration", "shutdown procedure" matches "decommissioning steps". Pool 40 candidates.

**BM25 full-text search.** PostgreSQL `tsvector` with `ts_rank_cd` (cover density ranking, not the simpler `ts_rank`). Captures exact matches: acronyms, proper nouns, error codes, version numbers. Use AND-first matching with OR fallback for recall on rare-term queries. Pool 40 candidates.

```sql
-- Dense (embedding from query encoder)
SELECT id, embedding <=> $1 AS distance
FROM chunks
ORDER BY distance ASC
LIMIT 40;

-- BM25
SELECT id, ts_rank_cd(tsv, query) AS score
FROM chunks, plainto_tsquery('english', $1) query
WHERE tsv @@ query
ORDER BY score DESC
LIMIT 40;
```

Pool size matters. Pool of 10 caused score collapse in evaluation; 40 gives enough room for the correct chunk to survive into reranking even when ranked low by both signals individually.

### 5.3 Reciprocal Rank Fusion (RRF)

Merge the two result sets by **rank position**, not score:

```
RRF(d) = w_dense / (k + rank_dense(d)) + w_bm25 / (k + rank_bm25(d))
```

- `k = 60` (Cormack et al. 2009 standard)
- Weights: `0.7` dense, `0.3` BM25

```python
def rrf_merge(dense_hits, bm25_hits, w_dense=0.7, w_bm25=0.3, k=60):
    scores = {}
    for rank, hit in enumerate(dense_hits):
        scores[hit.id] = scores.get(hit.id, 0) + w_dense / (k + rank + 1)
    for rank, hit in enumerate(bm25_hits):
        scores[hit.id] = scores.get(hit.id, 0) + w_bm25 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
```

**This is the single most important design decision.** Cosine similarity returns 0.3–0.9; `ts_rank_cd` returns 0.0–0.1. A naive weighted score combination (`0.7 * cosine + 0.3 * ts_rank`) makes the BM25 signal effectively invisible. RRF uses only ranks, so score-scale differences don't matter — and that's what makes the pipeline scale-invariant across document sizes (a small document and a large document produce different absolute score distributions but compatible rank distributions).

### 5.4 Cross-encoder reranking

Re-score the **top 20 RRF candidates** with a cross-encoder. Cross-encoders jointly attend to query and chunk and produce far more accurate relevance scores than bi-encoders, but they cannot scale to every chunk.

| Reranker | Notes |
|---|---|
| **BGE-reranker-v2-m3** (BAAI, Apache 2.0) | **Default.** ~568MB. Multilingual. Spreads scores across 0.13–0.63 — good discrimination. |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 80MB, CPU-friendly, English-only. |
| FlashRank (small) | 110MB, ONNX-runtime, fastest CPU option. |
| ColBERT v2 / RAGatouille | Late-interaction; faster than cross-encoder, less accurate. |

**Avoid:** FlashRank (nano, ms-marco-MultiBERT). In evaluation it scored all candidates ≈0.999 with no discrimination between relevant and irrelevant chunks. Always smoke-test a reranker by checking score spread on a held-out set before adopting it.

Select **top 5** for answer generation. Top-10 added only +0.1 average judge score at 2× the token cost.

### 5.5 Context window expansion

For each of the 5 selected chunks, fetch the **±1 adjacent chunks** (by `chunk_index` within the same document) and concatenate. This restores continuity that fixed-size splitting destroys without ballooning context size.

The "small-to-large" or parent-document-retrieval pattern is a more elaborate variant: embed small chunks (~200 tokens), return larger parents (~1000 tokens or full section). It improves accuracy on queries needing broad context by 15–30% in some reports. Adopt it as an upgrade if ±1 adjacency is insufficient on your eval set.

### 5.6 Context enrichment

For each retrieved chunk, assemble the prompt context as:

```
[Source: <doc_title>, Section: <section_title>, Chunk: <chunk_index>, Score: <rerank_score>]
Document summary: <document_summary>
Section summary: <section_summary>

<chunk_text_with_neighbours>
```

The summaries are read from the `documents` and `sections` tables (the distillation output from §3.2). They are cheap — already computed at ingest time — and they give the answer LLM the necessary frame to interpret a chunk that might otherwise look ambiguous.

---

## 6. Answer Generation

Send the assembled context plus the user's question to the answer LLM with a system prompt that:

1. Instructs the model to answer **only from the provided context**.
2. Requires inline citations with chunk IDs (e.g. `[chunk 42]`).
3. Tells the model to say "I don't know" when the context is insufficient — and to be specific about *what* is missing.

### Model selection

| Tier | Recommendation |
|---|---|
| Frontier (default) | Use the best latency/cost/quality model from the organisation's verified model shortlist. Internal evals showed answer-model quality mattered more than any single retrieval change. |
| Open-source | Llama 3.3 70B Instruct or Qwen 2.5 72B Instruct — competitive with frontier mid-tier models, deployable on a single H100 or via vLLM. |
| Cost-sensitive | Use a cheaper approved model only after measuring the quality drop on the target eval set. |

**Note on chain-of-thought prompting:** Helpful for some answer models in internal evaluation and neutral for others. Test on your own eval before adding it as a standard prompt feature; do not expose hidden reasoning or require chain-of-thought in user-facing outputs.

---

## 7. Reference Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Chunk size | 1000 chars, 200 overlap | Standard. Section-aware splitting matters more than exact size. |
| Heading depth (distillation) | L1–3 (<50pp), L1–2 (50–200pp), L1 (200+pp) | Prevents section explosion on large documents. |
| Embedding model | Gemini Embedding 001 (3072-d) or BGE-M3 (1024-d) | Validated head-to-head against nomic-embed-text. |
| Pool size per signal | 40 | Pool of 10 caused score collapse; 40 is the smallest pool that consistently survived to rerank with the right chunk present. |
| RRF `k` | 60 | Industry standard (Cormack et al. 2009). |
| RRF weights | 0.7 dense / 0.3 BM25 | Semantic-first, keywords as safety net. |
| Reranker | BGE-reranker-v2-m3 | Best score spread among open-source rerankers tested. |
| Rerank candidates | 20 | Top-20 from RRF → reranker → top-5. |
| Final top-k | 5 | Top-10 added only +0.1 avg judge score at 2× token cost. |
| Context window | ±1 chunks | Provides continuity without bloat. |
| Answer model | Verified approved model for the deployment | Single biggest single-component impact on answer quality in internal evals. |

These defaults are validated across documents from 10 to 600 pages without per-document tuning. If your eval shows regressions on a specific document size, change one parameter at a time and re-evaluate.

---

## 8. Anti-Patterns

These were tested and rejected during evaluation. Do not adopt without strong evidence on your own data.

| Anti-pattern | Why it fails |
|---|---|
| **Weighted score combination** (e.g. `0.7 * cosine + 0.3 * ts_rank`) | Score scales are incompatible. BM25 signal becomes invisible. **Use RRF.** |
| **Hard section pre-filtering** | Filtering chunks to "relevant" sections before retrieval drops the right chunk when the section classifier is wrong. Caused a 14.0 → 10.2 regression. Use section information as a soft signal at most. |
| **Pure cosine similarity, no BM25** | Fails on acronyms, proper nouns, error codes, version numbers, and domain-specific vocabulary. Hybrid is mandatory. |
| **Reranker without score-spread check** | FlashRank (nano) scored everything ≈0.999. Always smoke-test that the reranker actually discriminates. |
| **Top-k = 10** | Doubles context size for marginal quality gain. Top-5 is the sweet spot. |
| **CoT answer prompt without measurement** | Helps weaker answer models, no help for capable reasoners. Measure before adding. |
| **Chunk-only context (no section/doc summaries)** | Chunks read out of context are often ambiguous. The cheap pre-computed summaries from distillation are a free quality lift. |
| **Adding SPLADE / sparse-learned vectors as a third RRF signal** | +0.2 average judge score for substantial pipeline complexity. Within evaluation noise. Not worth it at this scale. |
| **GraphRAG before measuring base RAG** | High build cost, slow ingest, complex infra. Only justified after a base RAG pipeline measurably fails on relational queries. |
| **Skipping evaluation infrastructure** | Every parameter above was set by measurement. Without eval, you cannot tell which changes help. |

---

## 9. Open-Source Reference Stack

| Layer | Choice | Licence |
|---|---|---|
| Document parsing (text PDFs) | `pymupdf`, `pypdf` | AGPL / BSD |
| Document parsing (scanned/complex) | **Docling** (docling-project, originally IBM) or **MinerU** | Apache 2.0 |
| Markdown / heading parsing | `mistune`, `markdown-it-py` | BSD / MIT |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | MIT |
| Embeddings | **BGE-M3** (BAAI), nomic-embed-text-v1.5 | MIT / Apache 2.0 |
| Vector + full-text store | **PostgreSQL + pgvector** | PostgreSQL Licence |
| Reranker | **BGE-reranker-v2-m3** (BAAI) | Apache 2.0 |
| LLM proxy | **LiteLLM** | MIT |
| LLM (answer) | Llama 3.3 70B Instruct, Qwen 2.5 72B Instruct, Mistral Large | Llama Licence / Apache 2.0 / MRL |
| LLM (distillation, fast) | Llama 3.1 8B Instruct, Qwen 2.5 7B Instruct | Llama Licence / Apache 2.0 |
| Evaluation framework | **RAGAS**, DeepEval, custom LLM-as-judge | Apache 2.0 |
| Orchestration (optional) | Direct Python — no framework needed for the reference design |

The reference design intentionally avoids LangChain/LlamaIndex/Haystack as load-bearing dependencies. They are useful for prototyping but lock you into their abstractions for retrieval, reranking, and prompt assembly — every one of which you will eventually want to control directly. Use them for components (text splitters, document loaders) but not for the pipeline.

---

## 10. Hyperscaler Managed-Service Mappings

The reference design maps cleanly onto each major cloud's managed services. The trade-off is operational simplicity versus per-component control and per-page cost.

| Component | AWS | Azure | GCP | IBM | Oracle |
|---|---|---|---|---|---|
| **Document parsing** | Textract | AI Document Intelligence | Document AI | Datacap, watsonx Document Understanding | OCI Document Understanding |
| **Embeddings** | Bedrock (Titan Embeddings V2) | Azure OpenAI (text-embedding-3-large) | Vertex AI (gemini-embedding-001) | watsonx.ai (Granite Embedding) | OCI Generative AI Embeddings |
| **Vector + hybrid search** | OpenSearch Serverless; Aurora PostgreSQL with pgvector | AI Search (vector + hybrid + semantic ranker); Azure Database for PostgreSQL with pgvector | Vertex AI Vector Search; AlloyDB AI; Cloud SQL PostgreSQL with pgvector | watsonx.data (Milvus / DataStax) | Oracle Database 23ai (native AI Vector Search) |
| **Reranking** | Bedrock Rerank API (Cohere Rerank 3) | AI Search semantic ranker | Vertex AI ranking API | Self-host BGE-reranker | OCI Generative AI rerank endpoint |
| **Answer LLM** | Bedrock (Claude, Llama, Mistral) | Azure OpenAI; Azure AI Foundry | Vertex AI (Gemini, Claude) | watsonx.ai (Granite, Llama) | OCI Generative AI |
| **Turnkey RAG platform** | Bedrock Knowledge Bases | Azure AI Search + AI Foundry | Vertex AI RAG Engine | watsonx.ai | OCI Generative AI Agents |

### When to use a turnkey RAG platform vs. assembling components

**Use the turnkey platform when** you need RAG as a feature in a broader product, accept the platform's chunking and retrieval defaults, and value managed scaling over algorithmic control. Bedrock Knowledge Bases, Azure AI Search with semantic ranker, and Vertex AI RAG Engine all provide RRF-style hybrid retrieval and reranking out of the box.

**Assemble components when** you need to control chunking strategy, embedding/reranker choice, or context assembly — i.e. when retrieval quality is a product differentiator. Every default the reference design specifies (RRF weights, pool size, rerank candidate count, ±1 neighbours) is something a turnkey platform will hide.

A reasonable hybrid: use the platform's vector store and embedding model, but apply your own retrieval logic (RRF merge, custom reranker, context assembly) on top via the platform's lower-level APIs.

---

## 11. Evaluation Methodology

This design was set by measurement, and any deviation should be set by measurement too.

### LLM-as-judge framework

For each query, the answer model's response is scored against a reference answer on:
- **Correctness** (1–5)
- **Specificity** (1–5)
- **Relevance** (1–5)

Maximum 15 per query. Same model answers and judges to reduce variance.

### Eval set

Build a starting eval set of **15–50 queries spanning the document size range**. The reference pipeline was tuned on 15 queries across 5 documents from 35 to 585 pages. Add queries as production users surface failure cases.

Cover at minimum:
- **Direct factual lookups** ("What is the GDP target for 2026?")
- **Multi-part queries** ("Compare X to Y and explain the reasoning behind Z")
- **Acronym / proper-noun queries** (BM25 stress tests)
- **Synonym / paraphrase queries** (semantic-search stress tests)
- **Queries where the answer is in the middle of a long document** (lost-in-the-middle stress tests)
- **Queries with no answer in the corpus** (the model should refuse, not hallucinate)

### Iteration discipline

- Change **one variable at a time**.
- Judge variance is ~2 points between runs on the same retrieval. **Only differences > 2 points are meaningful.**
- Run a final hold-out evaluation on unseen queries before promoting changes.
- Feature-flag every change so you can A/B test in production.

### Production telemetry

In addition to offline evaluation, log:
- Retrieval recall@k (was the cited chunk in the candidate pool?).
- Reranker score distribution (early-warning signal for reranker degradation).
- User feedback (thumbs up/down) joined to chunk IDs and scores.
- Per-document and per-query-type quality metrics (so a regression on one document type is visible).

---

## 12. When to Deviate from This Design

This reference design is the **default starting point**. Deviate when measurement justifies it.

| Symptom | Consider |
|---|---|
| Recall on multi-part queries is poor | Add query decomposition (§5.1). |
| Scattered information across long documents | Add RAPTOR-style hierarchical summarisation indexing on top of chunk indexing. |
| Heavy entity-relationship queries ("who reports to whom across these org charts?") | Add a knowledge graph layer (GraphRAG, Neo4j, Microsoft GraphRAG OSS). |
| Visual content (charts, diagrams, layout-critical pages) is being missed | Add a ColPali / ColQwen vision retrieval signal alongside text retrieval. |
| Chunks are too narrow to answer broad questions | Move from ±1 adjacency to parent-document retrieval (small-to-large). |
| Index size exceeds ~10M chunks and pgvector latency is climbing | Move vectors to Qdrant / Milvus / Weaviate; keep BM25 in PostgreSQL. |
| Same query asked thousands of times against a small static corpus | Consider Cache-Augmented Generation (CAG): preload corpus into model context. |
| Reasoning over the entire document is required, not retrieval | Use long-context models (Gemini 2.5 Pro 1M, Qwen2.5-1M, Llama 4 Scout 10M) — RAG is the wrong tool. |

For deeper coverage of these alternatives and when each is appropriate, see [`large-document-llm-methods.md`](../rag/large-document-llm-methods.md) and [`rag-and-context-engineering.md`](../rag/rag-and-context-engineering.md).

---

## 13. Engineering Readiness Pack

This design is closest to implementation-ready, but production use still requires source traceability, concrete contracts, and reproducible evaluation artefacts.

### Evidence and claim ledger

Maintain a `claim-ledger.md` with `claim`, `section`, `source`, `source type`, `last verified`, `confidence`, `owner`, and `recheck trigger`.

| Claim class | Current status | Required handling |
|---|---|---|
| Internal 14-iteration evaluation results | Internal benchmark | Store raw eval set, run logs, prompts, model versions, and scorer rubric with the implementation. |
| Model-specific recommendations | High churn | Revalidate against current model cards/pricing before deployment. |
| pgvector dimensional/index limits | Source-linked technical claim | Verify against the exact pgvector and PostgreSQL versions used. |
| RAG method papers and vendor claims | External research | Keep as supporting evidence, not proof of target-corpus performance. |
| Cost/latency targets | Design targets | Replace with measured p50/p95 latency and cost on target infrastructure. |

### Implementation artefacts

Required before handoff:

- C4 context/container/deployment diagrams showing ingestion workers, parser service, metadata distiller, embedding service, PostgreSQL/pgvector, retrieval API, reranker, answer service, eval runner, and observability.
- DDL/migration files for documents, sections, chunks, eval cases, retrieval traces, feedback, and model/version metadata.
- API contracts for ingest, re-ingest, delete, query, feedback, eval run, and citation lookup.
- Parser contract specifying supported MIME types, failure modes, metadata extraction, OCR confidence, and layout/table handling.
- ADRs for pgvector vs vector DB, embedding model, reranker, chunk size, RRF settings, and answer model.
- Data-retention and tenant-isolation model, including row-level security if multiple tenants or permission domains share one database.

### Anti-hallucination controls

The answer model must only answer from retrieved, authorised, current source chunks.

Controls:

- Server-side context builder with immutable source IDs; the model cannot invent citation IDs or source URLs.
- Citation validator that maps every citation to a chunk and checks answer sentences for source support.
- No-answer/refusal path when retrieval confidence is low or the corpus does not contain the answer.
- Source freshness and access-control filters applied before retrieval and again before context assembly.
- Retrieval trace stored with dense hits, BM25 hits, RRF ranking, reranker scores, selected chunks, neighbour expansion, and final citations.
- Prompt-injection tests in source documents; retrieved text is treated as untrusted data.

### Threat model

| Threat | Control |
|---|---|
| Prompt injection in indexed documents | Content sanitisation, instruction isolation, and adversarial eval corpus. |
| Citation fabrication | Server-side citation mapping and validation. |
| Stale or unauthorised chunk retrieval | Metadata filters, RLS, source freshness, and audit logs. |
| Low OCR quality causes false answer | OCR confidence propagation and refusal/escalation below threshold. |
| Reranker degradation | Score-spread monitoring and held-out eval smoke tests before model changes. |
| Silent retrieval regression | CI evals on recall@k, citation accuracy, hallucination, and no-answer behaviour. |

### Evaluation and acceptance gates

Minimum production gate:

- 50+ curated eval questions before pilot; 150+ before broad production if the corpus is high-risk.
- Test buckets for direct lookup, acronyms/proper nouns, multi-hop synthesis, long-document middle answers, visual/OCR content, permission filtering, and no-answer refusal.
- Metrics: recall@40 before rerank, recall@5 after rerank, citation precision, answer faithfulness, refusal correctness, latency, and cost.
- Hold-out set for model/retriever changes; no promotion if the change improves aggregate score but regresses a critical query class.
- Human review of every hallucination or false refusal in the eval report before release.

### Operational runbook

Runbooks must cover parser failure, partial ingest rollback, embedding provider outage, index rebuild, vector dimension migration, source deletion/right-to-erasure, reranker rollback, tenant permission incident, and eval-gate failure.

---

## References

- Anthropic (2024). *Introducing Contextual Retrieval.* https://www.anthropic.com/news/contextual-retrieval
- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.* SIGIR. https://dl.acm.org/doi/10.1145/1571941.1572114
- Liu, N. F. et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* TACL. https://arxiv.org/abs/2307.03172
- Microsoft Research (2024). *GraphRAG.* https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-datasets/
- Sarthi, P. et al. (2024). *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.* https://github.com/parthsarthi03/raptor
- Chen, T. et al. (2024). *Dense X Retrieval: What Retrieval Granularity Should We Use?* EMNLP. https://aclanthology.org/2024.emnlp-main.845/
- BAAI. *BGE-M3 and BGE-Reranker-v2-m3.* https://huggingface.co/BAAI/bge-m3 · https://huggingface.co/BAAI/bge-reranker-v2-m3
- Docling Project (originally IBM). *Docling.* https://github.com/docling-project/docling
- OpenDataLab. *MinerU.* https://github.com/opendatalab/MinerU
- Jina AI (2024). *Late Chunking in Long-Context Embedding Models.* https://jina.ai/news/late-chunking-in-long-context-embedding-models/
- Faysse, M. et al. (2024). *ColPali: Efficient Document Retrieval with Vision Language Models.* https://arxiv.org/abs/2407.01449
- pgvector. https://github.com/pgvector/pgvector
- LiteLLM. https://github.com/BerriAI/litellm
- AWS. *Amazon Bedrock Knowledge Bases.* https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html
- AWS. *Amazon Bedrock Reranking.* https://docs.aws.amazon.com/bedrock/latest/userguide/rerank.html
- Microsoft. *Azure AI Search — Hybrid Search and Semantic Ranker.* https://learn.microsoft.com/en-us/azure/search/semantic-search-overview
- Google Cloud. *Vertex AI RAG Engine.* https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/retrieval-and-ranking
- IBM. *watsonx.ai Granite Embedding Models.* https://www.ibm.com/granite/docs/models/embedding
- Oracle. *AI Vector Search in Oracle Database 23ai.* https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/retrieval-augmented-generation1.html
- Zhang, J. et al. (2025). *OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation.* ICCV 2025. https://arxiv.org/abs/2412.02592
- Internal: ba-ai-discovery `doc-agent` branch. RAG Pipeline Engineering Summary and 14-iteration evaluation history (rag-pipeline-engineering-summary.md, doc-agent-summary.md, .scripts/research-rag-retrieval-methods-2026-03-19.md).
- NIST AI Risk Management Framework and Generative AI Profile. https://www.nist.gov/itl/ai-risk-management-framework
- OWASP Top 10 for LLM Applications 2025. https://owasp.org/www-project-top-10-for-large-language-model-applications/
- OpenAI Evals and graders documentation. https://platform.openai.com/docs/guides/evals
- Azure Well-Architected Framework for AI workloads. https://learn.microsoft.com/en-us/azure/well-architected/ai/
