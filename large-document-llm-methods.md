# Large Document Methods for LLM-Based Chatbots

| Field | Value |
|-------|-------|
| Created | 2026-03-20 |
| Last Updated | 2026-03-20 |
| Version | 1.1 |

---

- [1. The Large Document Problem](#1-the-large-document-problem)
- [2. Strategy 1 — Retrieval-Augmented Generation (RAG)](#2-strategy-1--retrieval-augmented-generation-rag)
- [3. Strategy 2 — Long-Context Models](#3-strategy-2--long-context-models)
- [4. Strategy 3 — Hierarchical Summarisation and Map-Reduce](#4-strategy-3--hierarchical-summarisation-and-map-reduce)
- [5. Strategy 4 — Context Compression](#5-strategy-4--context-compression)
- [6. Strategy 5 — Graph-Based RAG](#6-strategy-5--graph-based-rag)
- [7. Strategy 6 — Agentic Document Processing](#7-strategy-6--agentic-document-processing)
- [8. Strategy 7 — Prompt and Context Caching](#8-strategy-7--prompt-and-context-caching)
- [9. Hybrid Approaches — The Emerging Best Practice](#9-hybrid-approaches--the-emerging-best-practice)
- [10. RAG vs Long Context — Head-to-Head](#10-rag-vs-long-context--head-to-head)
- [11. Multimodal Document Processing](#11-multimodal-document-processing)
- [12. Open-Source Chatbot Platforms](#12-open-source-chatbot-platforms)
- [13. Security Considerations](#13-security-considerations)
- [14. Decision Framework](#14-decision-framework)
- [15. Open Problems](#15-open-problems)
- [16. Key Tools and Libraries](#16-key-tools-and-libraries)
- [References](#references)

A practical reference for engineers building chatbots that must reason over very large documents — contracts, books, regulatory filings, codebases, and multi-hundred-page PDFs. Covers every major approach from chunked retrieval to million-token context windows, with open-source alternatives for each.

---

## 1. The Large Document Problem

LLM-based chatbots face a fundamental tension: users want to ask questions about documents that exceed the model's context window, or that — even when they technically fit — degrade the model's performance through information overload.

**Scale of the problem.** Over 2.5 trillion PDF documents exist on the public web as of 2025 (MMORE, 2025). Enterprise use cases routinely involve 100–1000+ page documents: SEC 10-K filings, legal contracts, technical manuals, medical records, and regulatory submissions.

**Why simply increasing the context window is not enough:**

1. **Lost-in-the-middle effect.** LLMs exhibit a U-shaped performance curve — they attend best to information at the beginning and end of the context, with 10–20%+ accuracy drops for information in the middle (Liu et al., 2024). This is a well-replicated finding across model families.

2. **Effective context length < advertised maximum.** The RULER benchmark and Databricks' 2000+ experiment study found that most models degrade well before their stated limit. For example, Llama-3.1-405b performance starts to decrease after 32k tokens despite a 128k limit (Databricks, 2024). These thresholds are model-specific and improve with newer releases, but the gap between advertised and effective context remains a persistent pattern.

3. **Smaller needles are harder to find.** Recent research (Bianchi et al., 2025) shows that LLM performance drops sharply when the relevant text segment (the "needle") is shorter — a critical factor for large documents where answers may be a single sentence buried in hundreds of pages. Shorter gold contexts also amplify the positional sensitivity of the lost-in-the-middle effect.

4. **Cost and latency.** Processing 100k tokens on every query is expensive and slow. Standard transformer attention has an O(n²) component in sequence length, and practitioners report latencies of 30–60 seconds at high token counts versus ~1 second for well-optimised RAG pipelines.

Seven distinct strategies have emerged to address this problem. Most production systems combine several of them.

---

## 2. Strategy 1 — Retrieval-Augmented Generation (RAG)

The most widely deployed approach. The document is split into chunks, embedded into a vector store, and relevant chunks are retrieved per query.

### Chunking strategies

The choice of chunking strategy is critical. Nine main approaches exist, from simple to sophisticated:

| Strategy | How it works | Best for |
|----------|-------------|----------|
| **Fixed-size** | Split at N tokens with overlap | Baseline; fast and predictable |
| **Recursive character** | Split at natural boundaries (paragraphs, sentences) then recurse | General purpose; LangChain default |
| **Semantic** | Use embedding similarity between adjacent sentences to detect topic shifts | Variable-length, meaning-preserving chunks |
| **Contextual retrieval** | Prepend LLM-generated context summary to each chunk before embedding | Reduces retrieval failures ~67% (Anthropic, 2024) |
| **Late chunking** | Embed full document through long-context embedding model, then split the embedding sequence | Preserves cross-chunk attention; best retrieval quality |
| **Document-structure** | Split along headings, sections, tables | Structured documents (legal, technical) |
| **Hierarchical** | Maintain parent-child chunk relationships; retrieve child, return parent for context | Complex documents with nested structure |
| **Agentic** | LLM agent analyses document and selects chunking strategy per section | Mixed-format documents |
| **LLM-based** | Use an LLM to identify chunk boundaries based on semantic meaning | Highest quality; highest cost |

**Recommended defaults.** Chunk size of 256–512 tokens with 10–20% overlap. Semantic chunking + contextual retrieval is the current best-practice combination for most use cases. Fixed-size chunking remains a reasonable fallback for very large corpora where semantic chunking is too slow.

### Retrieval methods

**Dense retrieval** (embedding similarity) is standard but should be combined with **sparse retrieval** (BM25 or SPLADE) in a hybrid approach. Hybrid lexical + dense retrieval with lightweight reranking yields the largest gains under fixed token budgets. The LiveRAG 2025 benchmark confirmed that combining BM25 with dense retrieval (E5) outperforms either method alone.

**Fusion strategy.** Use Reciprocal Rank Fusion (RRF) with k=60 as a zero-configuration default. With 50+ labelled query pairs, switch to convex combination with a tuned alpha parameter. Add a cross-encoder reranker after fusion for best results.

**SPLADE** (Sparse Lexical and Expansion) is an advanced sparse retrieval method that learns term expansion, bridging the gap between BM25 and dense retrieval. It can serve as a drop-in BM25 replacement in hybrid pipelines with improved recall.

**ColBERT** (late-interaction retrieval) encodes documents at the token level rather than as single vectors, scoring individual token-level similarities via a MaxSim operator. This is more expressive than bi-encoders while remaining far more efficient than cross-encoders — searching large collections in tens of milliseconds. ColBERTv2 reduces index storage 6–10x versus the original. ColPali and ColQwen extend late interaction to multimodal (vision-language) document retrieval. Open-source implementations exist via RAGatouille and the ColBERT project (Stanford).

### Open-source embedding models

| Model | Context | Licence | Notes |
|-------|---------|---------|-------|
| **nomic-embed-text-v1.5** | 8192 | Apache 2.0 | Fully open weights, data, and training code |
| **jina-embeddings-v3** | 8192 | Apache 2.0 | Supports late chunking natively |
| **BGE-M3** | 8192 | MIT | Multilingual, supports dense + sparse + ColBERT |
| **GTE-Qwen2** | 8192 | Apache 2.0 | Strong on MTEB benchmarks |
| **mxbai-embed-large** | 512 | Apache 2.0 | Lightweight; good for resource-constrained environments |

### Open-source vector stores

Chroma, Milvus, Qdrant, Weaviate, pgvector (PostgreSQL extension), and FAISS are all production-ready. Choice depends on scale, deployment model, and existing infrastructure.

---

## 3. Strategy 2 — Long-Context Models

Feed the entire document (or as much as possible) directly into the model's context window.

### Open-source long-context models (as of early 2026)

| Model | Context window | Parameters | Notes |
|-------|---------------|------------|-------|
| **Qwen2.5-1M** | 1,000,000 | 7B, 14B | First open-source 1M-context model |
| **Llama 3.1** | 128,000 | 8B, 70B, 405B | Meta's flagship; widely supported |
| **Llama 4 Scout** | 10,000,000 | 17B active (109B total) | MoE architecture; very long context |
| **Mistral Large** | 128,000 | Various | Strong multilingual support |
| **DeepSeek-V3** | 128,000 | 671B MoE | Competitive with frontier models |
| **Command R+** | 128,000 | 104B | Designed for RAG and tool use |

### When long context works well

- **Single-document analysis** where the entire document fits in context (contracts, manuscripts, meeting transcripts).
- **Comparing 2–5 documents** where everything fits and holistic reasoning is needed.
- **One-off research tasks** where building retrieval infrastructure is not justified.

### When long context breaks down

- Documents exceed even the largest context window.
- Repeated queries against the same document (cost scales linearly with queries).
- Information density is low — most of the document is irrelevant to the query.
- Accuracy requirements are high for specific facts located anywhere in the document.

### Key limitation: effective vs advertised context

The RULER benchmark found that effective context length — the length at which performance starts to degrade — is often much shorter than the maximum. Newer models are improving, but always benchmark against your specific task and document type rather than trusting the headline number.

---

## 4. Strategy 3 — Hierarchical Summarisation and Map-Reduce

For documents that exceed any context window, or when you need a global understanding of the entire document.

### Map-Reduce

1. **Map phase:** Split the document into chunks. Summarise each chunk independently with an LLM.
2. **Reduce phase:** Combine the summaries into a single summary. If the combined summaries still exceed the context window, recursively summarise.

LangChain provides `chain_type="map_reduce"` out of the box. Benefits: predictable cost, bounded latency, full document coverage, parallelisable.

**LLM×MapReduce** (ACL 2025) extends this pattern with a Structured Information Protocol for inter-chunk communication and In-Context Confidence Calibration that lets chunks signal when they lack sufficient information. This training-free framework handles cross-chunk dependencies that naive map-reduce misses.

### Refine

Process chunks sequentially. The LLM summarises the first chunk, then receives the running summary plus the next chunk, and refines. Produces higher-quality summaries than map-reduce but is not parallelisable and takes longer.

LangChain provides `chain_type="refine"` for this pattern.

### Hierarchical indexing

Build multi-level indices: document-level summaries → section summaries → paragraph chunks. At query time, search the top level first to identify relevant sections, then drill down. LlamaIndex's `DocumentSummaryIndex` and structured hierarchical retrieval implement this pattern.

This is particularly effective for very large document collections where the chatbot must first identify *which* document is relevant before finding the specific passage.

---

## 5. Strategy 4 — Context Compression

Reduce the token count of retrieved context before sending it to the LLM, preserving the essential information.

### Approaches

| Method | How it works | Open-source tool |
|--------|-------------|-----------------|
| **Token pruning** | Remove low-information tokens based on perplexity scoring | LLMLingua, LongLLMLingua |
| **Selective Context** | Score and filter sentences by information density | Selective Context |
| **Extractive compression** | Select the most relevant sentences from retrieved chunks | RECOMP |
| **LLM-based compression** | Use a smaller LLM to compress context for a larger one | Various |

**LLMLingua** (Microsoft, open-source) is the most mature tool. It uses a small language model to score token importance and can achieve 2–10x compression ratios while preserving most downstream performance. **LongLLMLingua** extends this to long-context scenarios with question-aware compression.

### Caveats

Context compression works well for factual QA and summarisation but can degrade performance on reasoning-heavy tasks where the pruned tokens carried implicit logical structure. Performance varies significantly by document type and downstream task. Treat the commonly cited "60–80% reduction" figures as achievable upper bounds, not guarantees.

---

## 6. Strategy 5 — Graph-Based RAG

Standard RAG retrieves isolated chunks. Graph-based approaches preserve relationships between entities and concepts across the entire document.

### GraphRAG (Microsoft, MIT licence)

1. **Extract:** LLM identifies entities and relationships from the document, building a knowledge graph.
2. **Community detection:** Graph is partitioned into hierarchical communities of related entities.
3. **Summarise:** LLM generates summaries for each community at multiple levels.
4. **Query:** For local queries, retrieve relevant entities and their neighbourhoods. For global queries, use community summaries.

**Strengths:** Excels at global/thematic queries ("What are the main themes in this document?"), cross-reference questions, and sensemaking over entire corpora. Outperforms standard RAG for these query types.

**Weaknesses:** High indexing cost (many LLM calls per document), slower index build time, more complex infrastructure. Standard RAG may still be better for direct factual lookups.

### Open-source alternatives

- **LangChain GraphRAG** — integration with Neo4j and other graph databases
- **LlamaIndex Knowledge Graph Index** — property graph-based retrieval
- **nano-graphrag** — lightweight Python implementation

---

## 7. Strategy 6 — Agentic Document Processing

Embed an autonomous agent into the retrieval pipeline that can plan, iterate, and self-correct.

### How it differs from traditional RAG

Traditional RAG follows a fixed linear pipeline: query → retrieve top-k → concatenate into prompt → generate. This assumes a single retrieval suffices, which fails for ambiguous queries, evolving goals, and multi-step reasoning. Agentic RAG replaces this with a looped, agent-driven control flow:

1. **Plan:** Decompose complex queries into sub-questions.
2. **Retrieve:** Select appropriate retrieval strategies per sub-question.
3. **Evaluate:** Assess whether retrieved information is sufficient.
4. **Iterate:** Refine queries, try alternative search strategies, or fetch additional context.
5. **Use tools:** Invoke external tools (web search, code execution, SQL, vector stores) as first-class operations.
6. **Synthesise:** Combine answers with citations.

The agent maintains bifurcated memory: short-term memory tracks dialogue and task state, while long-term memory (vector stores, KV databases) supports cross-session continuity. Selective memory loading ranks stored chunks to stay within context limits.

### When to use agentic RAG

- Multi-hop questions requiring information from different parts of a large document.
- Complex analytical queries where the first retrieval may not surface the right chunks.
- Multi-document reasoning across a corpus.
- Tasks requiring validation and fact-checking of retrieved information.

### Open-source frameworks

| Framework | Agentic support | Notes |
|-----------|----------------|-------|
| **LangGraph** | Full agent orchestration | Graph-based agent workflows |
| **LlamaIndex** | Agent + tool abstractions | QueryPipeline, SubQuestionQueryEngine |
| **Haystack** | Pipeline-based agents | Modular, production-oriented |
| **CrewAI** | Multi-agent collaboration | Role-based agent teams |
| **RAGFlow** | Built-in agent templates | Deep document understanding |

### Limitations

Agentic approaches are slower (multiple LLM calls per query), more expensive, harder to debug, and can enter infinite loops without proper guardrails. Use for complex queries; fall back to standard RAG for simple lookups.

---

## 8. Strategy 7 — Prompt and Context Caching

When the same large document is queried repeatedly, caching avoids reprocessing the same tokens.

### How it works

The LLM provider stores the computed key-value (KV) cache for a prompt prefix. Subsequent requests with the same prefix reuse the cached computation, paying only for the new tokens.

### Impact

- **Cost reduction:** 50–90% on cached tokens (e.g. cached reads at $0.30/M tokens vs $3.00/M fresh on some providers).
- **Latency reduction:** 80–85% for the cached prefix, since the model skips prefill computation.
- **Best case:** A chatbot where the system prompt + document is fixed across many user queries. The entire document can be cached after the first request.

### Open-source considerations

For self-hosted deployments, **vLLM** supports automatic KV cache management with prefix caching. **SGLang** provides RadixAttention for efficient prefix sharing across requests. Both are open-source inference servers that make prompt caching practical without relying on managed API providers.

### Design implications

Structure your prompts with the large document as a prefix and the user query as a suffix. This maximises cache hit rates. Avoid reordering or modifying the document portion between requests.

---

## 9. Hybrid Approaches — The Emerging Best Practice

The strongest pattern emerging from research and practice is to combine strategies rather than choosing one.

### The "retrieval for finding, long context for reasoning" pattern

1. **RAG narrows the search space.** Retrieve the most relevant chunks from the document.
2. **Long context enables reasoning.** Feed the retrieved chunks (potentially many of them) into a long-context model for synthesis.
3. **Prompt caching amortises cost.** Cache the document-level context across queries.

This hybrid approach avoids RAG's precision problem (missing relevant chunks) and long context's cost problem (paying for irrelevant tokens on every query).

### Self-Route (Li et al., EMNLP 2024)

A specific hybrid method: the model self-reflects on whether a query can be answered from RAG-retrieved chunks. If yes, use RAG. If not, fall back to long context. Achieves LC-comparable performance at significantly lower cost.

### Practical hybrid stack

```
User query
    ↓
[Query router] — decides approach based on query type
    ↓                          ↓
[RAG pipeline]           [Long-context pipeline]
    ↓                          ↓
[Reranker] — filters and orders results
    ↓
[LLM generation with cached document prefix]
    ↓
[Citation extraction and response]
```

---

## 10. RAG vs Long Context — Head-to-Head

Based on multiple independent studies (Li et al. 2024a, Li et al. 2024b, Databricks 2024, LaRA benchmark):

| Dimension | RAG | Long Context |
|-----------|-----|-------------|
| **QA accuracy** | Good; varies with chunking quality | Better when resourced sufficiently |
| **Dialogue/conversational** | **Better** — more natural for multi-turn | Adequate but expensive per turn |
| **Global/thematic queries** | Weak (chunks lose global context) | **Better** for single-document |
| **Latency** | ~1s typical (retrieval + generation) | 30–60s at high token counts |
| **Cost per query** | Low (only retrieved tokens processed) | High (all tokens processed each time) |
| **Cost at scale** | Flat after indexing | Scales linearly with queries × doc size |
| **Multi-document** | Natural fit with corpus indexing | Limited by context window |
| **Accuracy on specific facts** | Depends on retrieval quality | Good if within effective context |
| **Infrastructure** | Vector store + embedding model + LLM | LLM only |

**Key insight for chatbot builders:** Since chatbots are inherently dialogue-based and serve repeated queries, RAG's advantages in dialogue handling and cost efficiency make it the stronger default. Use long context as a complement for holistic reasoning, not as a wholesale replacement for retrieval.

---

## 11. Multimodal Document Processing

Real-world large documents are rarely plain text. PDFs contain tables, images, diagrams, headers, footers, and complex layouts.

### Document parsing

| Tool | Type | Capabilities |
|------|------|-------------|
| **Docling** (IBM) | Open-source, Apache 2.0 | PDF/DOCX to structured markdown, table preservation ~95% |
| **MinerU 2.5** | Open-source | Layout analysis, table extraction, formula recognition |
| **PaddleOCR 3.0** | Open-source | Multilingual OCR, layout analysis |
| **Marker** | Open-source | PDF to markdown with high fidelity |
| **Unstructured** | Open-source (community) | Multi-format parsing, partitioning |

### Multimodal RAG

For documents with embedded images, charts, and diagrams:

1. **Extract and describe:** Use a vision-language model (VLM) to generate text descriptions of images and charts, then embed those descriptions alongside text chunks.
2. **Direct multimodal embedding:** Use multimodal embedding models (e.g. nomic-embed-vision-v1) that embed both text and images into the same vector space.
3. **Page-as-image RAG:** Treat each page as an image, use ColPali (a visual document retriever) to find relevant pages, then pass them to a VLM for question answering.

The MMORE pipeline (2025) provides an end-to-end open-source solution for multimodal document RAG.

---

## 12. Open-Source Chatbot Platforms

Several open-source platforms provide built-in large-document handling:

| Platform | Document handling | RAG approach | Notes |
|----------|------------------|-------------|-------|
| **Open WebUI** | File upload → RAG; large text auto-converted to file | Built-in chunking + embedding | Most popular self-hosted UI |
| **LibreChat** | Upload as text (direct injection) or RAG via PGVector | LangChain + PGVector indexing | Multi-provider support |
| **RAGFlow** | Deep document understanding with visual/layout analysis | Converged context engine | Enterprise-focused |
| **AnythingLLM** | Drag-and-drop document upload | Built-in vector store, multiple embedding options | Easy setup |
| **Danswer/Onyx** | Workspace-based document management | Hybrid retrieval (dense + sparse) | Team collaboration features |

All support self-hosted deployment with open-source LLMs via Ollama or vLLM.

---

## 13. Security Considerations

Document-processing chatbots introduce specific security risks that must be addressed in production.

### Indirect prompt injection via documents

User-uploaded documents may contain hidden instructions that manipulate the LLM's behaviour. This is a well-documented vulnerability class:

- **Attack:** Malicious text embedded in a PDF (possibly in white-on-white text, metadata, or image alt-text) instructs the LLM to ignore previous instructions, exfiltrate data, or produce harmful outputs.
- **RAG-specific risk:** Poisoned chunks in the vector store can be surfaced for any query, affecting all users.
- **Impact:** Data exfiltration, behaviour manipulation, privilege escalation.

### Mitigations (OWASP guidance)

1. **Input sanitisation.** Strip or flag suspicious patterns in uploaded documents before ingestion.
2. **Privilege separation.** Run document-derived content at a lower trust level than system instructions.
3. **Output filtering.** Validate LLM outputs against expected formats and content policies.
4. **Access control.** Ensure users can only query their own documents in multi-tenant systems.
5. **Monitoring.** Log and audit unusual patterns in retrieval and generation.

### Data leakage

In multi-tenant deployments, ensure vector store isolation between users. Shared embedding spaces can leak information across tenants through similarity search.

---

## 14. Decision Framework

Choose your approach based on document characteristics and use-case requirements:

```
Is the document < model's effective context length?
  YES → Long context (with prompt caching for repeated queries)
  NO  ↓

Is the document < model's maximum context window?
  YES → Long context may work, but benchmark first
        Consider RAG for cost/latency-sensitive applications
  NO  ↓

Single document or corpus?
  SINGLE → RAG with hierarchical chunking
            + map-reduce for summarisation tasks
  CORPUS → RAG with document-level routing
            + GraphRAG for thematic queries

Query complexity?
  SIMPLE LOOKUP → Standard RAG
  MULTI-HOP     → Agentic RAG or hybrid RAG + LC
  GLOBAL THEME  → GraphRAG or hierarchical summarisation

Repeated queries on same document?
  YES → Add prompt caching or pre-computed summaries
  NO  → Standard approach per above

Multimodal content (tables, images)?
  YES → Use VLM-based document parsing (Docling, MinerU)
        Consider ColPali for visual retrieval
```

---

## 15. Open Problems

Several challenges remain genuinely unsolved:

1. **Multi-document synthesis.** Both RAG and long-context approaches struggle when answers require integrating information across many documents simultaneously (Loong benchmark, EMNLP 2024). This is a hard problem without clean solutions.

2. **Evaluation.** There is no standard benchmark for large-document chatbot quality. Existing benchmarks (Natural Questions, HotPotQA, LongBench) test specific aspects but miss the full picture of conversational document QA.

3. **Rapid obsolescence of benchmarks.** Performance thresholds and model comparisons become outdated within months. Any specific numbers cited in this article should be validated against current models.

4. **Fine-tuning for document domains.** For repeatedly queried documents (company policies, product manuals), fine-tuning on the document content is a viable but under-explored alternative to both RAG and long context.

5. **Robust multimodal understanding.** Most RAG systems still treat documents as flat text, losing structural and visual information. Multimodal pipelines are maturing but not yet production-standard for all document types.

---

## 16. Key Tools and Libraries

### Frameworks

| Tool | Purpose | Licence |
|------|---------|---------|
| **LangChain** | RAG pipelines, agents, summarisation chains | MIT |
| **LlamaIndex** | Document indexing, retrieval, agent workflows | MIT |
| **Haystack** (deepset) | Production RAG pipelines | Apache 2.0 |
| **RAGFlow** (InfiniFlow) | Enterprise RAG with deep document understanding | Apache 2.0 |

### Inference servers

| Tool | Purpose | Licence |
|------|---------|---------|
| **vLLM** | High-throughput LLM serving with prefix caching | Apache 2.0 |
| **SGLang** | Efficient serving with RadixAttention | Apache 2.0 |
| **Ollama** | Easy local model deployment | MIT |
| **llama.cpp** | CPU/GPU inference for GGUF models | MIT |

### Compression and retrieval

| Tool | Purpose | Licence |
|------|---------|---------|
| **LLMLingua / LongLLMLingua** | Prompt compression | MIT |
| **ColBERT / RAGatouille** | Late-interaction retrieval | MIT/Apache 2.0 |
| **GraphRAG** (Microsoft) | Knowledge graph RAG | MIT |
| **Docling** (IBM) | Document parsing and conversion | Apache 2.0 |

---

## References

1. Liu, N. F. et al. (2024). "Lost in the Middle: How Language Models Use Long Contexts." *TACL*. DOI: 10.1162/tacl_a_00638. https://arxiv.org/abs/2307.03172

2. Li, X. et al. (2024). "Long Context vs. RAG for LLMs: An Evaluation and Revisits." *arXiv:2501.01880*. https://arxiv.org/abs/2501.01880

3. Li, Z. et al. (2024). "Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach." *EMNLP 2024 Industry Track*. DOI: 10.18653/v1/2024.emnlp-industry.66. https://aclanthology.org/2024.emnlp-industry.66/

4. Edge, D. et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *arXiv:2404.16130*. https://microsoft.github.io/graphrag/

5. Günther, M. et al. (2024). "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models." *arXiv:2409.04701*. https://arxiv.org/abs/2409.04701

6. Nussbaum, Z. et al. (2024). "Nomic Embed: Training a Reproducible Long Context Text Embedder." *arXiv:2402.01613*. https://arxiv.org/abs/2402.01613

7. Databricks (2024). "Long Context RAG Performance of LLMs." https://www.databricks.com/blog/long-context-rag-performance-llms

8. Jiang, H. et al. (2024). "Prompt Compression for Large Language Models: A Survey." *NAACL 2025*. https://arxiv.org/abs/2410.12388

9. Singh, A. et al. (2025). "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG." *arXiv:2501.09136*. https://arxiv.org/abs/2501.09136

10. Wang, M. et al. (2024). "Loong: Leave No Document Behind — Benchmarking Long-Context LLMs with Extended Multi-Doc QA." *EMNLP 2024 (Oral)*. https://github.com/MozerWang/Loong

11. OWASP (2025). "LLM Prompt Injection Prevention Cheat Sheet." https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html

12. Zhao, Q. et al. (2024). "LongRAG: A Dual-Perspective Retrieval-Augmented Generation Paradigm for Long-Context Question Answering." *EMNLP 2024*. DOI: 10.18653/v1/2024.emnlp-main.1259

13. Qwen Team (2025). "Qwen2.5-1M: Deploy Your Own Qwen with Context Length up to 1M Tokens." https://qwenlm.github.io/blog/qwen2.5-1m/

14. Chen, Z. et al. (2025). "Securing AI Agents Against Prompt Injection Attacks in RAG Systems." *arXiv:2511.15759*. https://arxiv.org/html/2511.15759v1

15. MMORE Authors (2025). "MMORE: Massive Multimodal Open RAG & Extraction." *arXiv:2509.11937*. https://arxiv.org/html/2509.11937

16. LangCopilot (2025). "Document Chunking for RAG: 9 Strategies Tested." https://langcopilot.com/posts/2025-10-11-document-chunking-for-rag-practical-guide

17. Weaviate (2024). "Late Chunking: Balancing Precision and Cost in Long Context Retrieval." https://weaviate.io/blog/late-chunking

18. LibreChat Documentation. "Upload Files as Text." https://www.librechat.ai/docs/features/upload_as_text

19. Open WebUI Documentation. "Features." https://docs.openwebui.com/features/

20. RAGFlow — Open-source RAG Engine. https://github.com/infiniflow/ragflow

21. Bianchi, F. et al. (2025). "Lost in the Haystack: Smaller Needles are More Difficult for LLMs to Find." *arXiv:2505.18148*. https://arxiv.org/abs/2505.18148

22. Zhou, Y. et al. (2024). "LLM×MapReduce: Simplified Long-Sequence Processing using Large Language Models." *ACL 2025*. https://arxiv.org/abs/2410.09342

23. Khattab, O. & Zaharia, M. (2020). "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." *SIGIR 2020*. https://arxiv.org/abs/2004.12832

24. IEEE Computer Society (2025). "Agentic RAG: Embedding Autonomous Agents into Retrieval-Augmented Generation." https://www.computer.org/publications/tech-news/trends/agentic-rag

25. LiveRAG Challenge (2025). "Evaluating Hybrid Retrieval Augmented Generation using Dynamic Test Sets." https://arxiv.org/abs/2506.22644

26. CoTHSSum Authors (2025). "Structured long-document summarization via chain-of-thought reasoning and hierarchical segmentation." *Journal of King Saud University*. https://link.springer.com/article/10.1007/s44443-025-00041-2
