# RAG & Context Engineering — State of the Art Reference

| Field | Value |
|-------|-------|
| Created | 2026-03-17 |
| Last Updated | 2026-03-17 |
| Version | 2.0 |

---

- [1. Document Parsing](#1-document-parsing)
- [2. Chunking](#2-chunking)
- [3. Retrieval](#3-retrieval)
- [4. Beyond RAG — The Context Engineering Landscape](#4-beyond-rag--the-context-engineering-landscape)
- [5. Implementation Priorities](#5-implementation-priorities)
- [6. Anti-Patterns](#6-anti-patterns)
- [7. Key Tools & Libraries](#7-key-tools--libraries)
- [References](#references)

A concise reference for engineers building document-grounded AI systems. Covers the production-ready techniques that have displaced first-generation RAG, and the emerging paradigms that complement or replace it.

---

## 1. Document Parsing

### First-generation (avoid for complex docs)
- pypdf, python-docx, pdfplumber — text extraction only, loses tables/layout/images.
- Tesseract OCR (open-source) — 65–78% accuracy on handwriting, poor on mixed layouts.

### Current best practice
- **Vision-Language Models** process pages as images, preserving layout, tables, and handwriting at 85–95% accuracy. Frontier VLMs (e.g. GPT-4o, Claude, Gemini) are best-in-class for complex layouts, but open-source VLMs are narrowing the gap.
- **Docling** (IBM, open-source, Apache 2.0) and **MinerU 2.5** (open-source) are the production-ready open-source options. Both output structured markdown with table preservation (~95% fidelity).
- **PaddleOCR 3.0** (open-source) is a strong alternative for multilingual OCR and layout analysis.
- **Hybrid approach:** Use fast text extraction for clean digital documents; route scanned/complex documents through a VLM parser. Classify at upload time based on whether the PDF contains extractable text or is image-based.

### Managed services (hyperscaler)

| Provider | Service |
|---|---|
| AWS | Amazon Textract |
| Azure | Azure AI Document Intelligence |
| GCP | Google Cloud Document AI |
| IBM | IBM Datacap, watsonx Document Understanding |
| Oracle | OCI Document Understanding |

These managed services handle OCR, table extraction, and form parsing without infrastructure management. For teams already on a hyperscaler, they offer faster time-to-production than self-hosted open-source alternatives, though at higher per-page cost.

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
- LangChain's `SemanticChunker` and LlamaIndex's `SemanticSplitterNodeParser` implement this (both open-source).
- Produces variable-length chunks that respect document structure.

**Contextual retrieval** (Anthropic, 2024) — prepend a short LLM-generated summary to each chunk:
- Before embedding, pass each chunk + surrounding document context to an LLM with a prompt like: *"Given the full document, write a concise context sentence for this chunk."*
- The context sentence is prepended to the chunk text before embedding.
- Reduces retrieval failures by ~67% in Anthropic's benchmarks.
- Trade-off: requires one LLM call per chunk at index time (use a fast/cheap model).

**Late chunking** (Jina, 2024) — embed the full document first, then split the embedding sequence:
- Preserves cross-chunk token attention that per-chunk embedding loses.
- Requires a long-context embedding model (e.g. jina-embeddings-v3, open-source).

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
- pgvector (open-source) supports both via `<=>` (cosine) for vectors and `ts_rank` / `tsvector` for full-text search in the same database.

**Embedding models:**

| Type | Options |
|---|---|
| **Open-source** | BGE-M3 (BAAI, multi-functional: dense + sparse + multi-vector), Granite Embedding (IBM, Apache 2.0, top-10 MTEB), GTE (Alibaba), Nomic Embed, jina-embeddings-v3 |
| **Managed (hyperscaler)** | AWS Bedrock (Titan Embeddings V2), Azure OpenAI Service (text-embedding-3-large), GCP Vertex AI (gemini-embedding-001), IBM watsonx.ai (Granite Embedding), Oracle OCI Generative AI |

Frontier proprietary models (e.g. text-embedding-3-large via Azure OpenAI) currently lead on general-purpose benchmarks, but open-source models like BGE-M3 and Granite Embedding are competitive and preferred when data sovereignty or cost is a concern.

**Stage 2: Reranking**
- Retrieve 20–50 candidates from Stage 1, then rerank with a cross-encoder.
- Cross-encoders score query-document pairs jointly — much more accurate than bi-encoder similarity.
- Return top 5–10 after reranking.
- This two-stage approach (cheap retrieval then expensive reranking) is the standard production pattern.

| Type | Options |
|---|---|
| **Open-source** | BGE-reranker-v2-m3 (BAAI, multilingual), cross-encoder/ms-marco-MiniLM-L-6-v2, FlashRank (lightweight), ColBERT v2 (late interaction) |
| **Managed (hyperscaler)** | AWS Bedrock Rerank API, Azure AI Search semantic ranker, GCP Vertex AI ranking API, Oracle OCI Generative AI (rerank endpoint) |

Open-source rerankers like BGE-reranker-v2-m3 perform comparably to proprietary alternatives for most use cases and can be self-hosted for full control.

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
- **Embedding model mismatch** — using a general-purpose embedding model for domain-specific content without evaluating domain-tuned alternatives. Consider fine-tuning open-source models like BGE-M3 or Granite Embedding on your domain data.
- **No evaluation framework** — upgrading retrieval components without a way to measure whether accuracy actually improved. Build a small eval set (50–100 query/answer pairs) before changing anything.

---

## 7. Key Tools & Libraries

### Open-source

| Category | Options |
|---|---|
| **Parsing** | Docling (IBM, Apache 2.0), MinerU 2.5, PaddleOCR 3.0, marker-pdf, Unstructured.io |
| **Chunking** | LangChain SemanticChunker, LlamaIndex SemanticSplitter, custom sentence-embedding approach |
| **Embeddings** | BGE-M3 (BAAI), Granite Embedding (IBM, Apache 2.0), GTE (Alibaba), Nomic Embed, jina-embeddings-v3 |
| **Vector DB** | pgvector (PostgreSQL), Qdrant, Weaviate, Milvus, Chroma |
| **Keyword search** | PostgreSQL tsvector/ts_rank, OpenSearch, BM25 via rank_bm25 |
| **Reranking** | BGE-reranker-v2-m3 (BAAI), cross-encoder/ms-marco models, FlashRank, ColBERT v2 |
| **Graph** | Neo4j, Microsoft GraphRAG (open-source), Graphiti, Apache TinkerPop |
| **Orchestration** | LangGraph, DSPy, CrewAI, Autogen |
| **Evaluation** | RAGAS, DeepEval, Giskard, custom LLM-as-judge |

### Managed services (hyperscaler)

| Category | AWS | Azure | GCP | IBM | Oracle |
|---|---|---|---|---|---|
| **Parsing/IDP** | Textract | AI Document Intelligence | Document AI | Datacap, watsonx Doc Understanding | OCI Document Understanding |
| **Embeddings** | Bedrock (Titan Embeddings V2) | Azure OpenAI (text-embedding-3-large) | Vertex AI (gemini-embedding-001) | watsonx.ai (Granite Embedding) | OCI Generative AI |
| **Vector search** | OpenSearch Serverless, Aurora pgvector | AI Search (vector + hybrid) | Vertex AI Vector Search, AlloyDB | watsonx.data (DataStax vectors) | Database 23ai (native vectors) |
| **Reranking** | Bedrock Rerank API | AI Search semantic ranker | Vertex AI ranking API | — (self-host open-source) | OCI Generative AI (rerank) |
| **Graph DB** | Neptune | Cosmos DB (Gremlin API) | Spanner Graph | Db2 Graph | Oracle Graph |
| **RAG platform** | Bedrock Knowledge Bases | Azure AI Search + AI Foundry | Vertex AI RAG Engine | watsonx.ai | OCI Generative AI Agents |

---

## References

- Anthropic (2024). *Introducing Contextual Retrieval.* https://www.anthropic.com/news/contextual-retrieval
- Jina AI (2024). *Late Chunking.* https://jina.ai/news/late-chunking-in-long-context-embedding-models/
- Microsoft Research (2024). *GraphRAG: Unlocking LLM discovery on narrative private datasets.* https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-datasets/
- LangChain (2025). *Not all RAG is the same.* https://blog.langchain.dev/not-all-rag-is-the-same/
- Anthropic (2025). *Building effective agents.* https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts
- IBM Docling. https://github.com/DS4SD/docling
- MinerU. https://github.com/opendatalab/MinerU
- IBM (2025). *Granite Embedding Models.* https://www.ibm.com/granite/docs/models/embedding
- BAAI. *BGE-M3 and BGE-Reranker.* https://huggingface.co/BAAI/bge-m3
- AWS. *Amazon Bedrock Reranking.* https://docs.aws.amazon.com/bedrock/latest/userguide/rerank.html
- AWS. *Amazon Titan Text Embeddings.* https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html
- Microsoft. *Azure AI Search — Semantic Ranking.* https://learn.microsoft.com/en-us/azure/search/semantic-search-overview
- Microsoft. *Azure Cosmos DB — AI Knowledge Graphs.* https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/cosmos-ai-graph
- Google Cloud. *Vertex AI RAG Engine — Retrieval and Ranking.* https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/retrieval-and-ranking
- Google Cloud. *Spanner Graph.* https://cloud.google.com/products/spanner/graph
- Oracle. *OCI Document Understanding.* https://docs.oracle.com/en-us/iaas/Content/document-understanding/using/home.htm
- Oracle. *OCI Generative AI — Embedding Models.* https://docs.oracle.com/en-us/iaas/Content/generative-ai/embed-models.htm
- Oracle. *Oracle AI Vector Search.* https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/retrieval-augmented-generation1.html
- Agentset AI. *Awesome Rerankers.* https://github.com/agentset-ai/awesome-rerankers
