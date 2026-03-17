# RAG & Context Engineering — State of the Art Reference

| Field | Value |
|-------|-------|
| Created | 2026-03-17 |
| Last Updated | 2026-03-17 |
| Version | 1.0 |

---

A concise reference for engineers building document-grounded AI systems. Covers the production-ready techniques that have displaced first-generation RAG, and the emerging paradigms that complement or replace it.

## 1. Document Parsing

### First-generation (avoid for complex docs)
- pypdf, python-docx, pdfplumber — text extraction only, loses tables/layout/images.
- Tesseract OCR — 65–78% accuracy on handwriting, poor on mixed layouts.

### Current best practice
- **Vision-Language Models** process pages as images, preserving layout, tables, and handwriting at 85–95% accuracy.
- **Docling** (IBM, open-source) and **MinerU 2.5** are the production-ready options. Both output structured markdown with table preservation (~95% fidelity).
- **Hybrid approach:** Use fast text extraction for clean digital documents; route scanned/complex documents through a VLM parser. Classify at upload time based on whether the PDF contains extractable text or is image-based.

### Key decision
If your documents are predominantly clean digital PDFs/DOCX, traditional extraction is fine. If you handle scanned documents, forms, tables, or handwritten notes, a VLM parser is no longer optional — it's the difference between 65% and 95% accuracy.

---

## 2. Chunking

### First-generation (the baseline)
- `RecursiveCharacterTextSplitter` with fixed size (e.g. 1000 chars, 200 overlap).
- Simple, fast, but fragments meaning at arbitrary boundaries. Chunks lack surrounding context.

### Current best practice

**Semantic chunking** — split at natural topic boundaries rather than character counts:
- Use sentence embeddings to detect topic shifts (cosine similarity between adjacent sentences drops below a threshold).
- LangChain's `SemanticChunker` and LlamaIndex's `SemanticSplitterNodeParser` implement this.
- Produces variable-length chunks that respect document structure.

**Contextual retrieval** (Anthropic, 2024) — prepend a short LLM-generated summary to each chunk:
- Before embedding, pass each chunk + surrounding document context to an LLM with a prompt like: *"Given the full document, write a concise context sentence for this chunk."*
- The context sentence is prepended to the chunk text before embedding.
- Reduces retrieval failures by ~67% in Anthropic's benchmarks.
- Trade-off: requires one LLM call per chunk at index time (use a fast/cheap model).

**Late chunking** (Jina, 2024) — embed the full document first, then split the embedding sequence:
- Preserves cross-chunk token attention that per-chunk embedding loses.
- Requires a long-context embedding model (e.g. jina-embeddings-v3).

### Recommended stack
Semantic chunking + contextual retrieval. The combination addresses both boundary quality (semantic) and context loss (contextual). Fixed-size chunking should only be used as a fallback for very large documents where semantic chunking is too slow.

---

## 3. Retrieval

### First-generation
- Pure vector similarity search (cosine distance against embeddings).
- Fails on keyword-specific queries, exact matches, and queries that don't textually resemble the answer.

### Current best practice — Hybrid search + reranking

**Stage 1: Dual retrieval**
- **Vector search** (semantic similarity) — catches paraphrases and conceptual matches.
- **BM25 / keyword search** — catches exact terms, acronyms, IDs, proper nouns.
- Typical weighting: 0.7 vector + 0.3 keyword (tune per domain).
- pgvector supports both via `<=>` (cosine) for vectors and `ts_rank` / `tsvector` for full-text search in the same database.

**Stage 2: Reranking**
- Retrieve 20–50 candidates from Stage 1, then rerank with a cross-encoder.
- Cross-encoders (e.g. Cohere Rerank, `cross-encoder/ms-marco-MiniLM-L-6-v2`, or an LLM-as-reranker) score query-document pairs jointly — much more accurate than bi-encoder similarity.
- Return top 5–10 after reranking.
- This two-stage approach (cheap retrieval then expensive reranking) is the standard production pattern.

**Stage 3 (optional): Contextual compression**
- After reranking, use an LLM to extract only the relevant sentences from each chunk.
- Reduces token usage and noise in the final context window.

### Key metrics to track
- **Recall@K** — are the right chunks in the candidate set?
- **MRR (Mean Reciprocal Rank)** — is the best chunk ranked first?
- **Context relevance** — does the retrieved context actually answer the query? (Use LLM-as-judge.)

---

## 4. Beyond RAG — The Context Engineering Landscape

RAG is not being replaced by a single successor. It is being decomposed into a component within a broader discipline called **context engineering** — managing what information reaches the model, how, and in what order.

> Over 70% of LLM errors come from incomplete, irrelevant, or poorly structured context — not from model limitations.

### Alternative paradigms

| Paradigm | Core idea | Best for | Trade-offs |
|---|---|---|---|
| **Cache-Augmented Generation (CAG)** | Preload entire corpus into context, skip retrieval | Small static corpora (<100 pages), high query volume | Context window limits; cost scales with corpus size |
| **Recursive Language Models** | Model navigates documents programmatically via self-calls | Large documents, complex multi-hop queries | Higher latency; requires careful recursion depth control |
| **Agentic Document Workflows** | Stateful multi-step processing with tool use and actions | Cross-document workflows, compliance, approvals | Complexity; harder to debug and test |
| **Native Multimodal (Docopilot-style)** | Process documents as images directly, no OCR/chunking pipeline | Visually complex documents, charts, tables | Token-expensive; limited by vision model quality |
| **Knowledge Graphs (KAG/GraphRAG)** | Graph traversal replaces vector similarity | Relational reasoning, regulated domains, entity-centric queries | Expensive to build and maintain; ingestion is slow |
| **Tool-Augmented Generation (TAG)** | Function/tool calling replaces retrieval for structured data | Databases, APIs, real-time data | Requires well-defined tool interfaces |
| **Memory-Augmented (MemGPT/Letta)** | Multi-tier self-editing agent memory | Long-running sessions, personalisation | Complex memory management; potential for drift |
| **Knowledge Distillation Pyramids** | LLM-powered ingestion into hierarchical knowledge summaries | Large enterprise corpora, table-heavy documents | High ingestion cost; summaries may lose detail |
| **Compound AI Systems (DSPy)** | Modular multi-component orchestration with auto-optimisation | Complex pipelines needing systematic tuning | Learning curve; framework lock-in |

### The emerging enterprise stack (2026)

```
Query Router
  |
  +-- 80% simple queries --> RAG (hybrid search + rerank)
  |
  +-- 15% relational queries --> GraphRAG / Knowledge Graph
  |
  +-- 5% complex workflows --> Agentic AI (multi-step, tool-using)
```

The three tiers map to: **RAG** (knowledge retrieval) + **MCP/tools** (structured interaction) + **Agentic AI** (orchestration). Most production systems should start with excellent RAG and only add the other tiers when retrieval alone demonstrably fails for specific query types.

---

## 5. Implementation Priorities

For a team upgrading a first-generation RAG system, this is the recommended order based on effort-to-impact ratio:

| Priority | Change | Effort | Impact |
|---|---|---|---|
| 1 | **Hybrid search** — add BM25/keyword alongside vector, weighted combination | Medium | High |
| 2 | **Reranking** — cross-encoder or LLM reranker on top-K candidates | Low–Medium | High |
| 3 | **Semantic chunking** — replace fixed-size with topic-boundary splitting | Low | High |
| 4 | **Contextual retrieval** — LLM-generated context prepended to chunks at index time | Medium | High |
| 5 | **VLM parsing** — Docling/MinerU for complex documents | Medium | Medium (depends on doc types) |
| 6 | **GraphRAG** — knowledge graph for entity-centric and relational queries | High | Medium |
| 7 | **Query routing** — classify queries and route to appropriate retrieval strategy | Medium | Medium |
| 8 | **Agentic workflows** — multi-step reasoning with tool use | High | High (for complex cases) |

---

## 6. Anti-Patterns

- **Config without implementation** — defining hybrid search weights and reranking settings in config but only implementing pure vector search. The system reports capabilities it doesn't have.
- **Bypassing existing capabilities** — having semantic chunking code in the document processor but wiring the indexing pipeline to a separate fixed-size splitter.
- **Over-engineering before measuring** — adding GraphRAG or agentic workflows before establishing that basic retrieval is failing. Measure recall and relevance first.
- **Embedding model mismatch** — using a general-purpose embedding model (e.g. text-embedding-ada-002) for domain-specific content without evaluating domain-tuned alternatives.
- **No evaluation framework** — upgrading retrieval components without a way to measure whether accuracy actually improved. Build a small eval set (50–100 query/answer pairs) before changing anything.

---

## 7. Key Tools & Libraries (March 2026)

| Category | Options |
|---|---|
| **Parsing** | Docling, MinerU 2.5, Unstructured.io, marker-pdf |
| **Chunking** | LangChain SemanticChunker, LlamaIndex SemanticSplitter, custom sentence-embedding approach |
| **Embeddings** | text-embedding-3-large (OpenAI), jina-embeddings-v3, Cohere embed-v4, BGE-M3 (open-source) |
| **Vector DB** | pgvector (PostgreSQL), Qdrant, Weaviate, Pinecone, Milvus |
| **Keyword search** | PostgreSQL tsvector/ts_rank, Elasticsearch/OpenSearch, BM25 via rank_bm25 |
| **Reranking** | Cohere Rerank, cross-encoder/ms-marco models, LLM-as-reranker, Jina Reranker |
| **Graph** | Neo4j + LangChain GraphRAG, Microsoft GraphRAG, Graphiti |
| **Orchestration** | LangGraph, DSPy, CrewAI, Autogen |
| **Evaluation** | RAGAS, DeepEval, LangSmith, custom LLM-as-judge |

---

## References

- Anthropic (2024). *Introducing Contextual Retrieval.* https://www.anthropic.com/news/contextual-retrieval
- Jina AI (2024). *Late Chunking.* https://jina.ai/news/late-chunking-in-long-context-embedding-models/
- Microsoft Research (2024). *GraphRAG: Unlocking LLM discovery on narrative private datasets.* https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-datasets/
- LangChain (2025). *Not all RAG is the same.* https://blog.langchain.dev/not-all-rag-is-the-same/
- Anthropic (2025). *Building effective agents.* https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts
- IBM Docling. https://github.com/DS4SD/docling
- MinerU. https://github.com/opendatalab/MinerU
