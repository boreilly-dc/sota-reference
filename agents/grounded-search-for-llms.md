# Grounded Search for LLMs — State of the Art Reference

| Field | Value |
|-------|-------|
| Created | 2026-06-24 |
| Last Updated | 2026-06-24 |
| Version | 1.1 |

---

- [1. What "Grounded Search" Means](#1-what-grounded-search-means)
- [2. Reference Architecture](#2-reference-architecture)
- [3. The Retrieval & Search-Backend Layer](#3-the-retrieval--search-backend-layer)
- [4. Citation & Attribution Generation](#4-citation--attribution-generation)
- [5. Grounding Verification & Faithfulness](#5-grounding-verification--faithfulness)
- [6. Agentic & RL-Trained Search](#6-agentic--rl-trained-search)
- [7. Open-Source Frameworks & Answer Engines](#7-open-source-frameworks--answer-engines)
- [8. The Generation Core: Self-Hostable LLMs](#8-the-generation-core-self-hostable-llms)
- [9. Hyperscaler Managed Grounding](#9-hyperscaler-managed-grounding)
- [10. Proprietary Systems (Examples)](#10-proprietary-systems-examples)
- [11. Evaluation](#11-evaluation)
- [12. Production Considerations](#12-production-considerations)
- [13. Legal, Compliance & Adversarial Robustness](#13-legal-compliance--adversarial-robustness)
- [14. Build Recommendations](#14-build-recommendations)
- [15. Areas of Uncertainty & Caveats](#15-areas-of-uncertainty--caveats)
- [References](#references)

---

**Scope.** This article is a build-it-yourself reference for *grounded search*: systems that answer a user's question by retrieving live evidence (usually from the web), then generating an answer that is **attributable** to that evidence through inline citations. Proprietary systems (Perplexity, Gemini, ChatGPT search, Claude) are covered as examples in [Section 10](#10-proprietary-systems-examples), but the focus is the research, reference architectures, and open-source frameworks you need to build your own. Per repository conventions, every component category presents an open-source option first; managed equivalents are limited to the major hyperscalers (AWS, Azure, GCP, IBM, Oracle).

---

## 1. What "Grounded Search" Means

**Grounding** is the broader category of techniques that connect an LLM's output to verifiable external sources. Plain retrieval-augmented generation (RAG) is *one* technique inside it: RAG retrieves documents to inform generation, but does not inherently verify or attribute the output. Grounded *search* adds two things on top of retrieval:

1. **Live retrieval** — evidence is fetched at query time from the open web (or a fresh index), not just from a static pre-ingested corpus.
2. **Attribution** — each claim in the answer is traceable to a specific source span, surfaced as an inline citation, and ideally verified against that source.

The conceptual foundation is **AIS (Attributable to Identified Sources)** (Rashkin et al., *Computational Linguistics* 2023): every statement an NLG system makes about the external world should be verifiable against an identified source. Modern grounded-search systems operationalise AIS through the pipeline in [Section 2](#2-reference-architecture).

Why it matters: search grounding measurably reduces hallucination (vendors report large reductions, though independent studies still find non-trivial error rates — e.g. RAG-powered legal tools hallucinating in 17–33% of queries, and a Columbia Journalism Review audit putting Perplexity's error rate near 37%). **Grounding reduces but does not eliminate hallucination**, which is exactly why the verification stage ([Section 5](#5-grounding-verification--faithfulness)) is a first-class component, not an afterthought.

---

## 2. Reference Architecture

### 2.1 The canonical pipeline

Every grounded-search system — proprietary or open — follows the same component flow. Build around these stages:

```
            ┌──────────────┐
  query ──▶ │ 1. Query     │  rewrite / decompose / classify intent
            │   understanding│  (HyDE, multi-query, step-back, decomposition)
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ 2. Search    │  web search API or self-hosted meta-search
            │   (retrieval)│  → ranked list of (title, url, snippet)
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ 3. Read /    │  selective full-text extraction of top hits
            │   extract    │  (Trafilatura / Jina Reader / Firecrawl)
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ 4. Chunk &   │  split, embed, hybrid-retrieve, then
            │   rerank     │  cross-encoder rerank to top-k passages
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ 5. Grounded  │  LLM writes answer constrained to the
            │   generation │  passages, emitting inline citations
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ 6. Verify    │  NLI / attribution check each cited claim;
            │   citations  │  down-grade or drop unsupported statements
            └──────┬───────┘
                   ▼
              cited answer
```

Stages 1, 4, and 6 are what separate a *grounded* system from a naive "search → stuff results into prompt" hack. Skipping query understanding hurts recall; skipping reranking floods the generator with noise; skipping verification is how hallucinated citations ship.

### 2.1.1 The data contract (carry provenance end-to-end)

The single hardest engineering problem in grounded search is keeping a claim traceable to a source *span* through every stage. Define the data objects up front and thread their provenance fields through the whole pipeline — retrofitting offsets later is impractical:

| Object | Key fields |
|---|---|
| `SearchResult` | `url`, `canonical_url`, `title`, `snippet`, `rank`, `engine`, `retrieved_at` |
| `ExtractedDocument` | `canonical_url`, `content`, `content_hash`, `extractor`, `extractor_version`, `fetched_at`, `mime_type` |
| `Chunk` | `doc_id`, `text`, `char_start`, `char_end`, `byte_start`, `byte_end`, `embedding`, `source_trust` |
| `CitationSpan` | `chunk_id`, `quote`, `char_start`, `char_end`, `url` |
| `Claim` | `text`, `citation_span_ids[]`, `requires_aggregation` |
| `VerificationVerdict` | `claim_id`, `label` (entailed/contradicted/unverifiable), `score`, `verifier`, `threshold` |

Two rules fall out of this contract: store the **canonical URL** (resolve redirects/AMP/tracking params) so citations dedupe and resolve later; and record `extractor_version` and `content_hash` so a citation can be re-validated against the exact text it was drawn from when the live page changes.

### 2.2 Three reference profiles

The pipeline scales down, but be honest about what each tier actually delivers. The key distinction is **whether citations are merely *displayed* or actually *verified against fetched source text***. Snippets alone (a search API's 1–2 sentence preview) rarely contain enough context — or stable character offsets — to support claim-level entailment, so a snippet-only system produces *attributed-looking answers*, not the span-verified grounding defined in [Section 1](#1-what-grounded-search-means).

| | **A — Snippet-attributed** | **B — Standard grounded** | **C — Verified grounded** |
|---|---|---|---|
| Attribution strength | weak (link-level) | source-level | span-level, entailment-checked |
| Query understanding | none / single rewrite | classify intent → HyDE/decompose | + multi-query fusion |
| Search | SearXNG (snippets) | SearXNG + paid API fallback | same |
| Extraction | none (snippet only) | selective full-text + offsets | full-text + preserved offsets (required) |
| Retrieval | BM25 over snippets | hybrid BM25 + dense + rerank | same |
| Generation | 8B open model, prompt citations | larger model / reasoning core | citation-format-tuned model |
| Verification | none / heuristic | NLI on each claim | NLI + abstention + audit sampling |
| Hardware | single GPU (~12 GB VRAM) | multi-GPU / managed inference | + verifier serving |

Profile **A** is a legitimate *low-stakes* starting point (FAQ-style answers, internal tooling) and the dominant cost saving is **not fetching every result** — decoupling search from extraction and letting the model select which URLs deserve full-text reading cuts extraction cost an estimated 70–90% (codenote.net, 2026). But **A is not "grounded" in the strict sense**: do not present span-level "verified" citations from snippets alone. Any product that claims verified citations must be **profile B or C**, which require full-text extraction with **preserved character/byte offsets** ([Section 4.3](#43-citation-granularity)). Treat A→B→C as a maturity path, not interchangeable options.

---

## 3. The Retrieval & Search-Backend Layer

This is the layer most teams underestimate. **Retrieval quality is the primary bottleneck of a grounded-search system, not LLM capability** — a finding echoed across both the open literature and reverse-engineering of Perplexity.

### 3.1 Web search backends

> **Scope note on the hyperscaler-only rule.** This repository limits *managed-service* recommendations to AWS, Azure, GCP, IBM, and Oracle. The search APIs below are listed as **external data providers** (a raw input to your own pipeline), not as managed platforms that replace it — the same category as a public dataset or news feed. The open-source option (SearXNG) is presented first and is sufficient to build the whole system; the paid APIs are optional inputs. Managed *grounding platforms* are confined to the five hyperscalers in [Section 9](#9-hyperscaler-managed-grounding).

The web-search API market shifted materially in 2025–26, and several defaults that older tutorials assume are now gone:

- **Bing Web Search API — retired 11 August 2025** (official Microsoft lifecycle notice). The replacement, *Grounding with Bing Search* inside Azure AI Agents, costs roughly **$35 per 1,000 queries** — far more than the old API. Do not design new systems around the Bing API.
- **Google Custom Search JSON API — closing to new customers, full retirement 1 January 2027** (per Google's own deprecation guidance; existing customers retain access until that date). Google directs new builders to Vertex AI Search. Not a viable foundation for a new project — confirm current status on Google's docs before relying on it.

That leaves a field of LLM-oriented search APIs and one strong open-source option (prices are 2026 vendor-page snapshots — **re-check before committing**, they move with tier changes and acquisitions):

| Backend | Type | Free tier | Indicative paid price /1K | Notes |
|---|---|---|---|---|
| **SearXNG** (open source) | Self-hosted meta-search *interface* | n/a (self-host) | Free (VPS only) | Aggregates 200+ engines behind one JSON API; the open-source default. It is a **proxy, not an index** (see 3.5). |
| Brave Search API | Independent index | $5/mo free credits | ~$4–5 (+~$5/M tokens for the AI-grounding endpoint) | Own multi-billion-page index (not scraped from others); SOC 2 Type II; `/llm/context`-style endpoint returns LLM-shaped content. |
| Tavily | LLM-native search+extract | limited | ~$8 (basic) / $16 (adv) | De-facto LangChain default; returns clean content. Acquisition by Nebius announced Feb 2026 — avoid deep lock-in. |
| Exa | Neural/semantic | 20,000 req/mo | ~$7 search + $1/1K contents | Embedding-based search with a `/findSimilar` endpoint; weaker on high-freshness content (news, prices). |
| Serper.dev | Raw Google SERP | trial credits | ~$0.30–1 | Cheapest, but a *Google v. SerpApi* lawsuit (filed Dec 2025) is pending — possible legal exposure for SERP-scraping resellers (the degree varies by provider; not legal advice). |
| You.com | Search + Web-LLM | sales-led | ~$5 (sales-led) | Returns full content; also offers an end-to-end RAG endpoint. |

**Recommendation for builders:** default to **SearXNG** for development and low-volume production (open-source, no per-query cost, no vendor lock-in), with a paid **independent-index** provider (Brave) as the scale/quality fallback. Avoid building *solely* on Google-SERP-reselling providers given the pending litigation.

### 3.2 Content extraction (the "read" stage)

Once you have URLs, you need clean main-text. Open-source first:

- **Trafilatura** (open source, Python) — consistently the most accurate open-source main-text extractor in published benchmarks (beats readability-lxml, Goose3, Newspaper4k). Outputs MD/JSON/XML. Best default for static HTML.
- **Jina Reader** (Apache-2.0, self-hostable) — prefix `r.jina.ai/` to any URL to get clean Markdown; trivially simple, token-priced when hosted (~$0.10/1K).
- **Firecrawl** (open source core) — handles JavaScript-rendered pages and anti-bot circumvention, which Trafilatura cannot. Use it specifically for SPA/JS-heavy sites.

A robust extraction stage uses Trafilatura first and falls back to Firecrawl when the page is JS-rendered or extraction is empty.

### 3.3 Query transformation

Not every query needs transformation — short, precise keyword queries work fine raw. Route dynamically based on query characteristics:

- **HyDE** (hypothetical document embeddings): generate a hypothetical answer, embed *that* for retrieval. Best for short/vague queries; risk: if the model hallucinates, retrieval is pulled off-course.
- **Multi-query**: generate several reformulations, retrieve each, fuse with **Reciprocal Rank Fusion (RRF, k=60)**. Best for broad topics.
- **Step-back prompting**: abstract a specific question to a more general one before retrieving. Best for "why"/diagnostic questions.
- **Decomposition**: split a multi-part question into atomic sub-queries. Best for comparison and multi-hop questions — and the natural interface to agentic search ([Section 6](#6-agentic--rl-trained-search)).

### 3.4 Hybrid retrieval & reranking

**Hybrid BM25 + dense retrieval is the 2026 production standard.** Dense-only retrieval fails on roughly 40% of real-world queries — exact terms, code identifiers, product SKUs, rare entities — where lexical match is essential. Combine a sparse signal (BM25, or learned-sparse **SPLADE**) with dense embeddings, and fuse with RRF.

Then **rerank**. The standard two-stage pattern: retrieve top-50/100 candidates cheaply, then re-sort with a cross-encoder to the top-10 the generator actually sees. A cross-encoder scores the (query, passage) pair jointly, capturing token-level relevance a bi-encoder cannot; hybrid + reranking has been reported to lift recall@10 from ~78% to ~91%.

Open-source rerankers (all self-hostable):

| Reranker | Licence | Notes |
|---|---|---|
| **BGE-reranker-v2-m3** (BAAI) | Apache-2.0 | Multilingual, strong MIRACL results; safe default. |
| **Qwen3-Reranker** (0.6B/4B/8B) | Apache-2.0 | 100+ languages, 32K context, code retrieval; scales by budget. |
| **mxbai-rerank-large-v2** (mixedbread) | Apache-2.0 | Multilingual; hosted option available. |
| **ColBERTv2** (Stanford) | MIT | Late-interaction; can serve as single-stage retriever with reranker-like quality, at higher index/storage cost. |
| Jina Reranker v2/v3 | CC-BY-NC-4.0 | v3 (0.6B) is a strong listwise reranker, but **non-commercial weights** — commercial use needs the paid Jina API. |

Managed equivalent: **Cohere Rerank 4** (closed, 100+ languages) — available directly and as a first-party option inside Oracle OCI and others. Prefer an Apache-2.0/MIT open reranker unless you have a reason not to.

**Embedding models** (open-weight, for the dense signal): **BGE-M3** (multi-functional: dense + sparse + ColBERT-style in one model, a strong default), **E5 / multilingual-E5**, **GTE**, **Nomic Embed** (open, long-context), and **Qwen3-Embedding** to standardise on one family with the reranker. Validate on your own queries — MTEB rank does not predict web-grounding performance.

**Where to run retrieval.** Web grounding is unusual: results are *ephemeral per query*, so you build a small index on the fly rather than maintaining a giant persistent one. Two patterns:

- *Ephemeral in-memory index* — embed the handful of fetched passages per query, score, discard. Simplest; no infrastructure. Best for profiles A/B.
- *Persistent hybrid store* — when you cache/accumulate sources. Open-source options: **OpenSearch** or **Elasticsearch** (BM25 + dense in one engine), **Vespa** (best-in-class hybrid + ranking), **Qdrant / Milvus / Weaviate / LanceDB** (vector-first), **pgvector** (if already on Postgres), and **Lucene/Tantivy** for the lexical side. SPLADE for learned-sparse.

**Three things the happy path omits but a builder must handle:**

- **Deduplication** — the same story appears across syndicating sites; dedupe by canonical URL and near-duplicate content hashing (MinHash/SimHash) *before* reranking, or you waste context on copies.
- **Freshness ranking** — for time-sensitive queries, boost by source recency; do not let a high-similarity stale page outrank a fresh one. Carry `retrieved_at`/publish date into the ranker.
- **Source trust scoring** — weight by domain reputation (a credibility prior like this article's own rubric) so SEO spam and content farms are demoted before they reach the generator (see [Section 13](#13-legal-compliance--adversarial-robustness)).

### 3.5 Resolving the SearXNG scalability caveat

Be precise about what SearXNG *is*: a **self-hosted meta-search interface**, not a **self-owned index**. It owns no crawl or corpus — it forwards each query to upstream engines and merges their results. So its reliability, freshness, and legal posture are inherited from those upstreams: high-volume Google sub-queries trigger CAPTCHAs and 429s, and commercial use sits within the upstreams' terms of service. The only way to *own* your search (no upstream dependency, no rate-limit ceiling, no ToS grey area) is to run your own crawler + index (e.g. an OpenSearch/Vespa corpus, or a hosted independent index like Brave's) — a far larger undertaking that most teams should defer.

Practical mitigations for SearXNG in production: (a) prefer engines that tolerate automation (Brave, DuckDuckGo, Wikipedia) over Google; (b) run multiple instances behind rate-limiting; (c) keep a paid independent-index API (Brave) as a fallback when results are thin. Treat **pure-SearXNG-on-Google as a development-tier configuration**; for production at volume, budget for an independent-index API or your own index.

---

## 4. Citation & Attribution Generation

This is the heart of "grounded" — how the answer gets tied to sources. There are three architectural paradigms; they are not mutually exclusive.

### 4.1 Cite-as-you-write (inline, during generation)

The model emits citations as it generates, constrained to the retrieved passages. This is the dominant production approach and what the proprietary APIs do internally.

- **Self-RAG** (Asai et al., NeurIPS 2023) trains the model to emit *reflection tokens*: `Retrieve` (should I retrieve now?), `IsRel` (is this passage relevant?), `IsSup` (is my sentence supported by the passage? — an inline groundedness check), `IsUse` (overall utility). Retrieval becomes on-demand rather than mandatory, and the `IsSup` token performs support-checking *during* generation. Outperforms vanilla RAG and ChatGPT on factuality and citation accuracy.
- **AGREE** (Google, NAACL 2024) fine-tunes a base LLM to self-ground with citations using synthetic training data whose citations are assigned automatically by an NLI model, plus a test-time adaptation loop that iteratively retrieves to fill gaps. Reports >30% relative improvement in citation precision/recall over prompting and post-hoc baselines.
- **SelfCite** (MIT CSAIL + Meta FAIR, ICML 2025) trains sentence-level citation with **no human labels**, using a self-supervised *context-ablation* reward: a citation is good if removing the cited text changes the model's output (*necessity*, "Prob-Drop") and keeping only the cited text preserves it (*sufficiency*, "Prob-Hold"). The reward needs just two forward passes. An 8B model reaches near-parity with Anthropic's Citations API on LongBench-Cite — a strong result for self-hosters.

The cheapest entry point requires no training at all: **prompt-based inline citation** (the ALCE "ICLCite" baseline) — few-shot instruct a capable instruction-tuned model to cite passage IDs inline. Quality is lower than trained approaches but it is the fastest path to a working system, and it is what LlamaIndex's `CitationQueryEngine` ([Section 7](#7-open-source-frameworks--answer-engines)) does out of the box.

### 4.2 Retrieve-then-cite vs. post-hoc attribution

- **Post-hoc attribution** adds citations *after* a draft is written. The foundational system is **RARR** (Gao et al., ACL 2023, Google): for each claim it generates probing questions, searches the web, checks agreement with an NLI/entailment model, and **revises** unsupported content while preserving the original text. RARR is model-agnostic — it bolts onto any generator's output — which makes it attractive when you cannot fine-tune the generation model.
- **Corrective retrieval (CRAG)** (2024) adds a lightweight retrieval evaluator that grades retrieved docs as *Correct / Incorrect / Ambiguous*; *Incorrect* triggers a corrective web search, *Ambiguous* triggers knowledge-strip filtering. It is plug-and-play on an existing RAG pipeline.

**Architectural trade-off:** cite-as-you-write (Self-RAG) calibrates *internally* (lower latency, needs training or a capable model); post-hoc/corrective (RARR, CRAG) corrects *externally* (no generator training, but extra search+NLI passes add latency and cost). Many production systems combine them: inline citations during generation, then a post-hoc verification sweep ([Section 5](#5-grounding-verification--faithfulness)).

### 4.3 Citation granularity

- **Answer-level** — whole answer attributed to a source set (weakest, easiest).
- **Sentence-level** — each sentence carries citations. The dominant granularity in ALCE, AGREE, and SelfCite.
- **Claim/sub-sentence-level** — **LAQuer** (ACL 2025) lets users verify specific claims *within* a sentence; **FullCite** (2026 preprint) generates document-level *and* verbatim evidence-span citations simultaneously. More precise, more engineering.

A practical engineering point the literature underweights: **provenance must be preserved through the whole pipeline.** To cite a span, you have to carry byte/character offsets from raw HTML → extracted text → chunk → generated sentence. Design your chunker to retain source URL + offset metadata on every chunk, or sentence-level citation becomes impossible downstream.

---

## 5. Grounding Verification & Faithfulness

Grounding reduces hallucination but does not remove it, so verify before you ship the answer. The workhorse is **Natural Language Inference (NLI) entailment checking**: given (claim, cited passage), classify as *entailed / contradicted / neutral (unverifiable)*.

- **Google T5-XXL-TRUE** is a widely used NLI verifier and underpins ALCE's automatic scoring.
- A 2025 study (SDP workshop) found **DeBERTa-V3-large fine-tuned on NLI corpora outperforms LLM prompting** for reference-grounded hallucination detection on *that benchmark*, while being far cheaper and faster. Read this as scoped evidence, not a universal law — it makes DeBERTa-V3 a strong, self-hostable *default* verifier, not the right tool for every claim type (see "verification is more than NLI" below).
- **AttrScore** (OSU, EMNLP Findings 2023) classifies attribution errors into *attributable / extrapolatory (goes beyond source) / contradictory*; `AttrScore-Flan-T5` (3B) is a compact open evaluator. **AttributionBench** (ACL Findings 2024) confirms NLI-fine-tuned models lead this task.

**Verifier options, by cost/accuracy:**

| Approach | Latency/cost | Notes |
|---|---|---|
| DeBERTa-V3 NLI (open, ~400M) | Lowest | Best accuracy-per-dollar for entailment; recommended default. |
| AttrScore-Flan-T5 (open, 3B) | Low | Gives an error *type*, not just a binary. |
| SelfCite context-ablation | Medium (2 fwd passes/claim) | No extra model; reuses the generator. |
| LLM-as-judge | Highest | Flexible but slow, costly, and itself prone to sycophancy. Use sparingly. |

The verification stage should **down-grade or remove** any claim whose citation does not entail it, and surface abstention ("the sources do not confirm X") rather than paper over gaps. Calibrated abstention is a quality signal.

### 5.1 Verification is more than NLI

Sentence-pair NLI handles the common case but is brittle exactly where grounded answers fail. A production verifier is a small pipeline, not a single model:

- **Claim segmentation** — decompose the answer into atomic, individually-checkable claims first; a sentence often bundles several.
- **Multi-citation / aggregation** — a claim supported by *combining* two sources ("X is the largest of A, B, C") entails against no single passage. Verify against the *union* of cited spans, and flag `requires_aggregation` claims for stricter handling.
- **Numeric, temporal & tabular claims** — NLI is weak on "23% vs 0.23", date arithmetic, unit conversions, and table lookups. Add targeted checks (extract the number/date from the source span and compare) rather than trusting entailment.
- **Quotation integrity** — for verbatim quotes, do an exact-substring check against the source, not entailment.
- **Citation-span validation** — confirm the cited offsets actually exist in the (hashed) source and contain the quoted text — catches both model hallucination and silently-changed pages.
- **Threshold calibration & contradiction handling** — calibrate the entailment threshold on a labelled set per claim class; treat *contradicted* differently from *unverifiable* (contradiction is a hard removal).
- **Human audit sampling** — sample a small fraction of shipped answers for human review; this is your ground-truth signal for drift and the only check that catches systematic verifier blind spots.

---

## 6. Agentic & RL-Trained Search

So far the pipeline is mostly static. The frontier question is **when and what to search** — letting the model decide, iterate, and stop. Two families: prompt-orchestrated agentic patterns, and models trained (via RL) to search.

### 6.1 Prompt-orchestrated patterns (no training)

These are achievable today with any capable instruction model and an orchestration framework:

- **ReAct** (Yao et al., 2023) — the foundational *Thought → Act → Observation* loop. The model reasons, issues a search, observes results, reasons again. Grounded reasoning beats pure chain-of-thought on knowledge-intensive tasks.
- **Self-Ask** — decompose a multi-hop question into follow-up sub-questions, answer each (with search), compose. Closes the "compositional gap".
- **IRCoT** (ACL 2023) — interleave chain-of-thought with retrieval: each reasoning step generates the next retrieval query, and retrieved content informs the next step.
- **FLARE** (EMNLP 2023) — *active* retrieval: generate a tentative next sentence, and only retrieve if it contains low-confidence tokens, using the tentative sentence as a forward-looking query. Avoids unnecessary searches.

### 6.2 RL-trained search models

> **This is frontier research, not core build guidance.** Most teams building a grounded-search system will *not* train their own search model — the prompt-orchestrated patterns in §6.1 plus a good retrieval/verification stack get you most of the way. Treat this subsection as context on where the field is heading and what to adopt *if* you hit a ceiling and have the data + RL infrastructure. Reported gains come from fast-moving 2025–26 work on overlapping benchmarks with limited independent replication.

The 2025–26 research wave trains the model itself to search well, rather than scripting it.

- **WebGPT** (OpenAI, 2022) — the progenitor: fine-tuned GPT-3 to browse a text interface (search/click/quote/cite) via imitation learning then RLHF. Established that RL-from-human-feedback can optimise web-grounded QA with citations.
- **Search-R1** (UIUC, 2025) — DeepSeek-R1-style RL teaching an LLM to emit `<search>` queries inside its reasoning. Two key tricks: an **outcome-only reward** (just answer correctness — no process/format reward needed) and **retrieved-token masking** (mask retrieved passages during gradient computation so the model is rewarded for its own reasoning/queries, not for copying retrieved text). +41% over RAG baselines on Qwen2.5-7B; **GRPO outperformed PPO**.
- **R1-Searcher / R1-Searcher++** (RUC, 2025) — two-stage (SFT cold-start → RL) training for *dynamic knowledge acquisition*: the model learns to choose between its own parametric knowledge and external search, reducing both over- and under-retrieval.
- **DeepResearcher** (Shanghai AI Lab, EMNLP 2025) — the first end-to-end RL training **in the live web** (not a frozen RAG corpus), with a reasoning agent plus browsing agents. RL produced *emergent* behaviours: planning, cross-validation across sources, self-reflection, and honest abstention. +28.9 over prompt-engineering and +7.2 over RAG-corpus RL. Its thesis — that training in the real, noisy web is a requirement, not an implementation detail — is the most important recent finding for builders.
- **ZeroSearch** (Alibaba, 2025) — cuts RL training cost by replacing the live search engine *during training* with a 3B LLM that simulates retrieval, using a curriculum that progressively degrades document quality. Real search is used only at inference. Removes the API-cost and quality-variance problems of live-search RL.
- **WebThinker** (RUC, NeurIPS 2025) — equips large reasoning models (DeepSeek-R1, o1-class) to interleave search, navigation, and report drafting inside the thinking process; trained with iterative online DPO.
- **Search-o1** (RUC, EMNLP 2025) — names the *Reasoning-Retrieval Dilemma*: injecting raw retrieved documents disrupts a long reasoning chain. Its *Reason-in-Documents* module distils retrieved content into reasoning-compatible form before injection.

A practical note on reward design: these systems optimise mainly for **answer correctness** and report that useful search behaviour emerges without an explicit citation reward. But answer correctness does **not** guarantee citation *faithfulness* — a model can reach the right answer while citing the wrong span or no span at all. Keep citation training and citation evaluation ([Sections 4](#4-citation--attribution-generation)–[5](#5-grounding-verification--faithfulness)) as a separate concern; do not assume a correctness-optimised search model produces trustworthy citations. If you train your own, start with outcome reward + retrieved-token masking (Search-R1), then add explicit citation supervision/eval rather than expecting it for free.

### 6.3 Train or orchestrate?

| | Prompt-orchestrated (ReAct, etc.) | RL-trained (Search-R1, etc.) |
|---|---|---|
| Up-front cost | Low — works today | High — needs RL infra + data |
| Best for | Prototyping, simple/single-turn, no training budget | Multi-turn, noisy web, non-obvious strategies |
| Reliability | Decays multiplicatively with steps (~pⁿ) | Directly optimises end-to-end trajectory |
| Open recipe? | Yes (frameworks, §7) | Yes (Search-R1, DeepResearcher, OpenDeepSearch code released) |

The survey *RL-based Agentic Search* (2025) frames the central risk: **trajectory reliability decays multiplicatively** — a 10-step agent where each step is 95% reliable succeeds ~60% of the time. This is the strongest argument both for RL training (which optimises the whole trajectory) and for the deep-research design principle of *parallelising breadth over serialising depth*. If you orchestrate rather than train, keep chains short and fan out.

> **Caveat (from critique):** the RL-search results above come largely from one fast-moving research community evaluating on overlapping benchmarks, mostly in English, with limited independent replication. Treat the *magnitude* of reported gains as indicative, not settled, and benchmark on your own data.

---

## 7. Open-Source Frameworks & Answer Engines

Two layers of open source matter: **end-to-end answer engines** (fork-and-run "Perplexity clones") and **orchestration frameworks** (build-your-own).

### 7.1 End-to-end answer engines

| Project | Licence | Stack & grounding approach | Best for |
|---|---|---|---|
| **Perplexica** | MIT | SearXNG → embedding-similarity rerank → pluggable LLM (Ollama/OpenAI/Claude/Gemini), inline cited sources. Most mature OSS answer engine (~29K stars). | Fastest path to a self-hosted Perplexity-style UX; fully local via Ollama. |
| **GPT-Researcher** | Apache-2.0 | Plan-and-solve deep-research agent; cited multi-source reports from 20+ sources; provider-agnostic. Reports #1 on CMU DeepResearchGym. | Long-form, multi-source *research reports* rather than quick answers. |
| **OpenDeepSearch** | Apache-2.0 | Open Search Tool + Open Reasoning Agent on HuggingFace **SmolAgents**; runs on DeepSeek-R1/Qwen. Paper-backed (arXiv:2503.20201). | An open, paper-grounded agentic search stack you can extend. |
| **Khoj** | AGPL-3.0 | Personal-document semantic search + web search, agents, cited research. | A personal "second brain" combining private docs and web. |
| **Morphic** | Apache-2.0 | Next.js + Vercel AI SDK + **Tavily** (paid) search; generative cited UI. | A polished front-end; note Tavily dependency (not fully self-hostable as-is). |
| **Farfalle** | Apache-2.0 | Local LLMs via LiteLLM (llama/gemma/mistral/phi); lighter clone. | A minimal starter; less maintained. |
| LeptonAI `search_with_lepton` | Apache-2.0 | Reference *demo* (search API + streaming cited answers). | Reading the minimal pattern; not production. |

For most teams wanting a working system quickly, **Perplexica** (interactive answers) or **GPT-Researcher** (reports) are the two to evaluate first; both are permissively licensed and LLM-agnostic.

Star counts are a weak signal. When you actually evaluate one of these, score it on what matters for production: **maintenance activity** (recent commits/releases), **deployment model** (Docker/k8s, single-user vs multi-tenant), **citation granularity** (link vs span) and whether it has any **verification** at all, **search backend** (self-hostable vs paid-API-locked — Morphic, for instance, is Tavily-dependent), **local-only feasibility**, **licence risk** (note Khoj is AGPL-3.0), and **observability/security posture**. Most of these projects do *display* citations but do *not* verify them — you will likely add the §5 verification stage yourself.

### 7.2 Orchestration frameworks (build-your-own)

- **LangGraph / LangChain** (MIT) — agentic RAG with retriever-as-tool, query-rewriting loops, self-correction, and hallucination-detection patterns. No built-in citation *engine*, but the most flexible substrate for the agentic loops in [Section 6](#6-agentic--rl-trained-search). Ships a `SearxSearchWrapper` for SearXNG.
- **LlamaIndex** (MIT) — the only mainstream framework with a **batteries-included `CitationQueryEngine`** that auto-maps answer spans to source chunks. Best starting point if inline citations are your priority and you don't want to build attribution from scratch.
- **Haystack** (deepset, Apache-2.0) — explicit, modular, production-oriented pipelines (retrievers/readers/generators) over Elasticsearch/FAISS/OpenSearch. Best when you want declarative, inspectable pipeline graphs.
- **DSPy** (Stanford, MIT) — programmatic, *optimisable* pipelines: define `retrieve → rerank → answer` as modules with signatures, then let optimisers tune the prompts/weights. Best when you want to systematically optimise quality rather than hand-tune prompts.

**Picking one:** citations-first → LlamaIndex; agentic loops / custom control flow → LangGraph; inspectable production pipelines → Haystack; systematic optimisation → DSPy. They are composable (e.g. DSPy modules inside a LangGraph node).

---

## 8. The Generation Core: Self-Hostable LLMs

Grounded search is bottlenecked by retrieval, not generation, so you can use a **smaller open-weight model** than you might assume — especially when the answer is constrained to retrieved passages. As of mid-2026, the open-weight options (per the repo's own model surveys) include:

- **Qwen3** family — strong general/coding performance; the 0.6–8B reranker siblings make it convenient to standardise a stack on one model family.
- **DeepSeek-V3.2 / R1** — strong reasoning and tool orchestration; R1 is the reasoning core used by OpenDeepSearch and WebThinker-style stacks.
- **Llama 4 Scout** — long context (useful for many extracted pages at once) and multimodal.
- **Gemma 4** — efficient smaller models for the minimal-viable profile.

No single model dominates every axis. For grounded search specifically, weight your selection on criteria that generic benchmarks miss:

- **Citation-format adherence** — does it reliably emit the inline citation syntax you asked for, every time? (This, not raw reasoning, is often the deciding factor.)
- **Structured-output & tool-use reliability** — for agentic loops and JSON citation objects.
- **Context length under many sources** — and *quality* at long context, not just the advertised window, when you stuff 10–20 extracted pages.
- **Runtime/serving** — quantisation support, vLLM/TGI/llama.cpp compatibility, batching, latency, and cost at your volume.

For the minimal profile (extractive, few sources), an 8B-class instruction model with prompt-based inline citation plus a DeBERTa NLI verifier is a credible starting point — but qualify that by task complexity: multi-hop synthesis and high citation fidelity push you toward larger models. Validate on your own grounded-search eval ([Section 11](#11-evaluation)); see the repository's frontier- and open-model surveys for current rankings.

---

## 9. Hyperscaler Managed Grounding

If you would rather buy the grounding layer, the major clouds offer managed equivalents. Coverage is uneven — AWS, GCP, and Azure are mature; IBM and Oracle are thinner.

**Distinguish two different things these vendors sell**, because only one is *grounded web search*: (1) **live open-web grounding** — answers grounded in real-time public search results (GCP *Grounding with Google Search*, Azure *Grounding with Bing*); versus (2) **enterprise-corpus RAG** — grounding over *your configured* documents/connectors (AWS Bedrock Knowledge Bases, GCP *Vertex AI Search*, Azure Foundry IQ, IBM OpenRAG). Bedrock's web-crawler connector ingests pages into your corpus; it is not the same as querying the live open web. Pick the category that matches your problem before comparing vendors.

- **AWS — Bedrock Knowledge Bases.** Managed RAG with auto-ingestion/embedding/reranking, **citations in responses**, and *agentic retrieval* (multi-hop query decomposition). Connectors for S3, SharePoint, Confluence, web crawler. **Contextual Grounding guardrails** detect hallucination/ungrounded output as a first-class feature. Integrates with AgentCore Gateway (MCP-compatible). The most complete managed grounding stack for the build-vs-buy comparison.
- **GCP — Vertex AI grounding.** *Grounding with Google Search* grounds Gemini in real-time results with byte-indexed `groundingMetadata` citations and a dynamic-retrieval threshold; *Grounding with Vertex AI Search* grounds on your own data (combinable, up to 10 data sources); *Enterprise Web Search* offers web grounding **without query logging** for regulated industries. Now under the Gemini Enterprise Agent Platform.
- **Azure — Foundry Agent Service + Foundry IQ.** Note: the older *Azure OpenAI "On Your Data"* is **deprecated and retires October 2026**; new builds should target **Foundry Agent Service with Foundry IQ** for grounded answers (Azure AI Search remains the retrieval backend). *Grounding with Bing Search* is the web-grounding option (≈$35/1K — see §3.1).
- **IBM — watsonx.** *OpenRAG* on watsonx.data grounds agents in governed enterprise knowledge with document processing, hybrid search, and agentic retrieval; watsonx.ai provides a RAG toolkit. *(Evidence here is thinner than for AWS/GCP/Azure — validate current capabilities directly with IBM before committing.)*
- **Oracle — OCI Generative AI + AI Database 26ai.** OCI offers Cohere Rerank 4 for retrieval and *Select AI* as an orchestration layer over vector search in AI Database 26ai. *(Least mature of the five for grounded *web* search specifically; treat as directional.)*

> **Editorial note:** the open-source-first rule is fully satisfiable for every layer above (SearXNG, Trafilatura/Jina/Firecrawl, BGE/Qwen3/ColBERT rerankers, Perplexica/LlamaIndex/Haystack, open-weight LLMs, DeBERTa NLI). Managed services are a convenience/scale choice, not a necessity.

---

## 10. Proprietary Systems (Examples)

These are reference points, not blueprints. The useful observation is that **they all implement the same Section 2 pipeline**; the differences are in index quality, training, and how citations are surfaced via API.

**Developer-facing grounding APIs** (well-documented, and the closest thing to a buildable spec):

- **Google — Grounding with Google Search (Gemini API).** Add the `google_search` tool; the model decides whether/what to search, issues one or more queries, and returns the answer with inline `url_citation` annotations (character `startIndex`/`endIndex` → source URLs) plus a search-suggestion widget. Gemini 3 bills per search query executed.
- **OpenAI — Responses API `web_search`.** Three modes: non-reasoning lookup, agentic iterative search (reasoning models decide whether to keep searching), and deep research. Returns `web_search_call` items exposing the actions taken (`searching`/`open_page`/`find_in_page`) plus a message with an `annotations` array (URL, title, character offsets). `search_context_size` controls retrieval depth; domain allow/block lists supported.
- **Anthropic — Claude `web_search` tool.** A server-side tool Claude invokes autonomously (possibly several times per turn). Citations are **always on**; results carry `encrypted_content`/`encrypted_index` for multi-turn citation continuity. *Dynamic filtering* (recent Claude models) lets Claude **write and run code to filter search results before they enter the context window** — a notable efficiency pattern worth emulating. ~$10 per 1,000 searches.

The common, copyable lesson: **return citations as structured annotations with character/byte offsets**, so the application can render inline, clickable citations — not as free-text "[1]" markers the model might fabricate.

**System examples:**

- **Microsoft Copilot** uses a *Prometheus* orchestrator bridging the Bing index and GPT, with a 5-stage loop (input → safety filters → iterative context-retrieval/LLM/API loop → response → output filters) and short sanitised 2–3 keyword Bing queries for privacy.

> **Perplexity — speculated design (treat as directional, not authoritative).** Perplexity is fully closed; the following is **reverse-engineered** from third-party analysis (some with commercial interest) and Perplexity's own research blog, *not* official architecture documentation. Reportedly: a 6-stage RAG pipeline over its **own** search index (hundreds of billions of pages, moved off the Bing API), custom `pplx-embed` embedding models (0.6B/4B, Qwen3-based), hybrid BM25 + dense retrieval, a 3-tier ML reranker (XGBoost at L3, ~0.7 quality threshold with a fail-safe re-query), and citations **pre-embedded before generation** so the synthesis model is constrained to the assembled sources. Its *Sonar* model is said to be built on Llama 3.1 70B. The credible, transferable idea here is *pre-binding the answer to assembled, ranked sources* rather than citing post-hoc — but the specifics should not be relied on as an engineering reference.

---

## 11. Evaluation

You cannot improve grounding you do not measure. Use both offline attribution metrics and online production signals.

**Offline / attribution metrics:**

- **ALCE** (Princeton, EMNLP 2023) — the standard citation-generation benchmark. Reports **citation precision** (fraction of cited passages that actually support the claim) and **citation recall** (fraction of claims supported by ≥1 citation), scored automatically via NLI (TRUE), alongside fluency and correctness. This precision/recall pair is the core metric to track for any grounded-search system.
- **AIS** (CL 2023) — the human-annotation framework underneath ALCE; binary attributability per statement.
- **AttrScore / AttributionBench** — automatic attribution evaluation with an error taxonomy (attributable / extrapolatory / contradictory).
- **LongBench-Cite** — long-context citation F1 (used by SelfCite).
- **DeepResearchGym / Deep Research Bench** — for agentic, long-form research-report systems (GPT-Researcher tops DeepResearchGym).

**Online / production metrics** (the literature underweights these — instrument them yourself):

- Citation **verification rate** (% of shipped claims that pass the NLI check).
- Citation **click-through** and answer rejection/regeneration rate.
- **Freshness** (age of cited sources) and abstention rate.
- Drift monitoring (citations that stop resolving over time).

**Evaluation matrix** — the metrics worth tracking, by stage:

| Stage | Metric | What it catches |
|---|---|---|
| Retrieval | source recall@k | did the evidence even get retrieved? |
| Retrieval | freshness (median source age) | stale answers on time-sensitive queries |
| Citation | citation precision / recall (ALCE) | do citations support claims / are claims cited? |
| Citation | span IoU vs gold | are the cited *offsets* right, not just the doc? |
| Verification | unsupported-claim rate | hallucinated/ungrounded statements shipped |
| Verification | contradiction rate | claims that conflict with their own source |
| Verification | abstention calibration | does it abstain when it should (not too much/little)? |
| System | latency (P50/P95), cost/query | operational viability |

Break every metric down **by query class** (factual lookup vs comparison vs multi-hop vs time-sensitive) — aggregate numbers hide the failure modes that matter.

A pragmatic harness: run ALCE-style precision/recall on a gold set in CI, and track verification rate + abstention live. Beware "citation washing" — inline citations can create an *illusion* of reliability; the verification stage is what makes them real.

---

## 12. Production Considerations

The research literature is light on operations. The following are the engineering realities that decide whether a grounded-search system survives contact with real traffic.

- **Latency budgets.** Each web hop adds ~1–5s (search + fetch + extract). Agentic multi-turn compounds this. Target sub-500ms for the *retrieval+rerank* core (achievable locally), and parallelise fetches/extractions. Stream the answer so perceived latency tracks first-token, not full-trajectory, time.
- **Semantic caching.** In production, a large share of queries are near-duplicates. A semantic cache (e.g. GPTCache-style, or your framework's cache) over both search results and generated answers materially cuts cost and latency — design cache keys on the *normalised/rewritten* query, and set TTLs short enough to preserve freshness.
- **Selective extraction.** As in §2.2/§3.1: show the model titles+snippets first, extract full text only for the URLs it selects. This is the single biggest cost lever (≈70–90% extraction savings).
- **Failure modes & graceful degradation.** Search-API outages, extraction timeouts, reranker/NLI OOM. Degrade gracefully: snippet-only answer with a "could not fully verify" flag beats a hard failure. Keep a fallback search provider configured (§3.5).
- **Cost modelling.** At 100K–1M queries/day, per-query search ($0.30–35/1K) and extraction dominate over self-hosted inference. Model cost at realistic volume *before* choosing managed vs. self-hosted; the cheap-API economics that hold at 1K/day invert at scale.
- **Observability.** Log the full trajectory (queries issued, sources retrieved, rerank scores, verification verdicts) per answer. This is both your debugging surface and your audit trail (§13).

---

## 13. Legal, Compliance & Adversarial Robustness

Grounded search reaches out to the live web and republishes others' content with attribution — this carries legal and security exposure that the model-research literature largely ignores.

**Legal & compliance:**

- **SERP-scraping risk.** The *Google v. SerpAPI* lawsuit (filed Dec 2025) targets providers that scrape Google results; it creates downstream risk for anyone building on Serper-style backends. Prefer providers with **independent indexes** (Brave) or licensed APIs, and keep a swap-out path.
- **robots.txt and ToS.** Respect `robots.txt` when fetching/extracting pages, and review search-provider terms — self-hosted SearXNG scraping of Google sits in a grey area at commercial scale.
- **Data residency & retention.** Cached web content and query logs may contain personal data. For regulated workloads use grounding paths that **don't log queries** (GCP Enterprise Web Search) and define retention/right-to-erasure handling for your caches.
- **Transparency obligations.** Regimes such as the EU AI Act impose transparency duties on AI-generated content; reliable, resolvable citations and a stored trajectory (§12) help evidence compliance. (This is general guidance, not legal advice — confirm obligations for your jurisdiction and use case.)

**Adversarial robustness:**

- **Prompt injection via retrieved content.** Retrieved web text frequently contains instructions aimed at the model ("ignore previous instructions…"). **Treat all fetched content as data, never as instructions.** Keep retrieved text in clearly delimited, non-authoritative context; never let it alter system/developer instructions or tool permissions. This is the single most important security control in a grounded-search system.
- **SEO spam & source poisoning.** Adversaries optimise content to rank and to be cited. Mitigate with **source-reputation scoring** (credibility weighting, as in this article's own rubric), preferring primary/established domains, and cross-checking a claim against ≥2 independent sources before stating it with confidence.
- **Citation integrity.** Verify that cited URLs resolve and that the cited span actually entails the claim (§5) — both to catch model hallucination and to catch poisoned/altered sources.

**Fetcher hardening (you are running a server-side URL fetcher — treat it as attack surface):**

- **SSRF prevention.** A grounded-search fetcher will dereference URLs an attacker can influence (via search results or user input). Block requests to private/link-local ranges (`127.0.0.0/8`, `10/8`, `172.16/12`, `192.168/16`, `169.254.169.254` cloud metadata, IPv6 ULA), resolve-then-validate to defeat DNS rebinding, and disallow non-`http(s)` schemes (`file://`, `gopher://`).
- **Sandboxed fetch/render.** Run extraction (especially JS-rendering headless browsers) in an isolated, network-egress-restricted, ephemeral sandbox with no credentials and no access to internal services. Never reuse an authenticated browser profile for grounding fetches.
- **Content limits & sanitisation.** Enforce MIME-type allowlists, max response size, and timeouts; sandbox/scan PDFs and office files (malware, embedded scripts); strip `<script>`/HTML before the text reaches the model; reject binaries you do not intend to parse.
- **URL allow/deny lists** for known-malicious or out-of-policy domains, applied before fetch.

**Content reuse:**

- **Copyright, caching & opt-outs.** Quoting for citation is generally defensible, but storing large verbatim copies, ignoring publisher opt-out signals (e.g. AI-usage directives), or serving long passages can create exposure. Cache the minimum needed for verification, honour opt-outs, and bound quotation length. (General guidance, not legal advice.)

---

## 14. Build Recommendations

A concrete, open-source-first stack and the decision points that matter:

1. **Start with the canonical pipeline (§2), not an agent.** Get query→search→extract→rerank→cited-generation→verify working end-to-end before adding agentic loops. Most quality comes from retrieval + reranking + verification, not from clever agency.
2. **Default open-source stack:** SearXNG (search) → Trafilatura + Firecrawl fallback (extract) → hybrid BM25+dense → BGE/Qwen3 cross-encoder rerank → open-weight LLM (8B+ for minimal, larger for quality) with prompt-based or LlamaIndex `CitationQueryEngine` inline citations → DeBERTa-V3 NLI verification. Fork **Perplexica** (answers) or **GPT-Researcher** (reports) if you want a head start.
3. **Make verification non-optional.** A cheap NLI verifier that down-grades unsupported claims is the difference between "grounded" and "grounded-looking".
4. **Decouple search from extraction** and cache aggressively — the two biggest cost/latency levers.
5. **Add agency deliberately.** Use ReAct/decomposition (§6.1) via LangGraph for multi-hop; keep chains short (trajectory reliability ≈ pⁿ). Only invest in **RL training** (Search-R1 / DeepResearcher recipes) once a prompt-orchestrated system is hitting a clear ceiling and you have the data/infra — and follow DeepResearcher's lesson of training in the real web.
6. **Treat fetched content as untrusted data.** Bake in injection resistance and source-reputation scoring from day one.
7. **Buy the grounding layer (AWS Bedrock KB / GCP Vertex grounding / Azure Foundry IQ) when** speed-to-market and managed compliance outweigh control and per-query cost at your volume.

### 14.1 Build checklist

A launch-readiness sequence — each item maps to a section above:

- [ ] **Search backend** chosen, with a fallback provider (§3.1, §3.5)
- [ ] **Fetch policy**: snippet-only vs selective vs full extraction, matched to your profile (§2.2)
- [ ] **Extraction** with fallback (Trafilatura → Firecrawl) and **offsets preserved** (§3.2, §2.1.1)
- [ ] **Chunk schema** carries `url`/`canonical_url`/offsets/`content_hash`/trust (§2.1.1)
- [ ] **Retrieval + rerank** config: hybrid weights, RRF k, candidate/return counts, dedup + freshness + trust scoring (§3.4)
- [ ] **Citation format** the generator must emit, and granularity target (§4)
- [ ] **Verifier** + calibrated thresholds + abstention policy + claim-type routing (§5, §5.1)
- [ ] **Eval set** with gold answers; ALCE-style citation precision/recall in CI (§11)
- [ ] **Security**: SSRF blocklist, sandboxed fetch, content sanitisation, injection tests (§13)
- [ ] **Caching** with freshness-aware TTLs (§12)
- [ ] **Observability**: full per-answer trajectory logged (§12)
- [ ] **Launch gates**: min citation precision, max unsupported-claim rate, latency/cost budgets

---

## 15. Areas of Uncertainty & Caveats

- **Reverse-engineered internals.** Perplexity's architecture (§10) is reconstructed from third-party analysis with possible commercial bias — directional only.
- **Preprint-heavy frontier.** Much of §6 (Search-R1, DeepResearcher, ZeroSearch, R1-Searcher, WebThinker, Search-o1) is 2025–26 work, some not yet peer-reviewed and reproduced; reported gains are indicative.
- **Vendor pricing is volatile.** All per-query prices (§3, §9) are 2026 snapshots from vendor pages and shift with acquisitions (Nebius–Tavily), tier changes, and litigation. Re-check the vendor page before committing — several figures here were already stale within months of first writing.
- **English-centric evidence.** Retrieval quality, extraction accuracy, and NLI-verifier availability are materially worse in low-resource languages; this article's sources are overwhelmingly English.
- **Thin IBM/Oracle coverage.** Managed grounding evidence for IBM watsonx and Oracle OCI is limited; validate current capabilities directly.
- **Limited independent benchmarking.** Cross-system, apples-to-apples comparisons (RL vs. prompt-orchestrated; backend A vs. B on identical queries) are scarce — benchmark on your own workload.
- **Search-provider sustainability.** The economics and acceptable-use posture of LLM-driven query volume against commercial search engines remain in flux.

---

## References

Sources are graded by type: **[peer-reviewed]** academic venue, **[official]** vendor/primary documentation, **[vendor]** vendor marketing/pricing page, **[secondary]** third-party blog/analysis (indicative, cross-checked where possible), **[speculative]** reverse-engineering. Treat pricing/benchmark figures from **[vendor]**/**[secondary]** sources as indicative and re-verify against primary sources before relying on them.

**Grounding, attribution & verification — [peer-reviewed] except as noted**

1. Rashkin et al. — Measuring Attribution in NLG Models (AIS), *Computational Linguistics* 2023 — https://aclanthology.org/2023.cl-4.2/
2. Gao et al. — RARR: Researching and Revising What Language Models Say, ACL 2023 — https://arxiv.org/abs/2210.08726
3. Gao et al. — Enabling LLMs to Generate Text with Citations (ALCE), EMNLP 2023 — https://aclanthology.org/2023.emnlp-main.398/
4. Asai et al. — Self-RAG, NeurIPS 2023 — https://arxiv.org/abs/2310.11511
5. Yan et al. — Corrective Retrieval-Augmented Generation (CRAG), 2024 — https://arxiv.org/abs/2401.15884
6. Chuang et al. — SelfCite, ICML 2025 — https://selfcite.github.io/
7. AGREE: Adaptation for Grounding Enhancement, Google Research / NAACL 2024 — https://research.google/blog/effective-large-language-model-adaptation-for-improved-grounding/
8. Yue et al. — AttrScore, EMNLP Findings 2023 — https://arxiv.org/abs/2305.06311
9. AttributionBench, OSU NLP, ACL Findings 2024 — https://osu-nlp-group.github.io/AttributionBench/
10. LAQuer: Localized Attribution Queries, ACL 2025 — https://aclanthology.org/2025.acl-long.746/
11. From RAG to Reality: Hallucination Detection via NLI Fine-Tuning, SDP 2025 — https://aclanthology.org/2025.sdp-1.34/

**Agentic & RL-trained search (academic)**

12. Yao et al. — ReAct, ICLR 2023 — https://arxiv.org/abs/2210.03629
13. Press et al. — Self-Ask, 2022 — https://arxiv.org/abs/2210.03350
14. Trivedi et al. — IRCoT, ACL 2023 — https://arxiv.org/abs/2212.10509
15. Jiang et al. — FLARE, EMNLP 2023 — https://arxiv.org/abs/2305.06983
16. Nakano et al. — WebGPT, OpenAI 2022 — https://arxiv.org/abs/2112.09332
17. Jin et al. — Search-R1, 2025 — https://arxiv.org/abs/2503.09516
18. Song et al. — R1-Searcher, 2025 — https://arxiv.org/abs/2503.05592 ; R1-Searcher++ — https://arxiv.org/abs/2505.17005
19. Zheng et al. — DeepResearcher, EMNLP 2025 — https://arxiv.org/abs/2504.03160
20. ZeroSearch, Alibaba 2025 — https://arxiv.org/abs/2505.04588
21. Li et al. — WebThinker, NeurIPS 2025 — https://arxiv.org/abs/2504.21776
22. Li et al. — Search-o1, EMNLP 2025 — https://arxiv.org/abs/2501.05366
23. Open Deep Search (Sentient), 2025 — https://arxiv.org/abs/2503.20201 ; code: https://github.com/sentient-agi/OpenDeepSearch
24. A Comprehensive Survey on RL-based Agentic Search, 2025 — https://arxiv.org/abs/2510.16724

**Retrieval, search backends & reranking**

25. [official] Bing Search APIs Retiring Aug 2025, Microsoft Lifecycle — https://learn.microsoft.com/en-us/lifecycle/announcements/bing-search-api-retirement
26. [official] Custom Search JSON API — Google for Developers (deprecation/overview; confirm current status) — https://developers.google.com/custom-search/v1/overview
27. [vendor] Brave Search API — pricing/free tier — https://brave.com/search/api/
28. [vendor] Exa — pricing/free tier — https://exa.ai/pricing
29. [secondary] Search & Extraction APIs for AI Agents — Cost Comparison (2026) — https://codenote.net/en/posts/tavily-alternatives-cost-comparison-search-extract-api/
30. [secondary] Exa/Tavily/Serper/Brave comparison — https://rhumb.dev/blog/exa-vs-tavily-vs-serper-vs-brave-search
31. [official] SearXNG — https://github.com/searxng/searxng
32. [official] Trafilatura evaluation — https://trafilatura.readthedocs.io/en/latest/evaluation.html
33. [secondary] Query transformation (HyDE / Multi-Query / Step-Back / Decomposition) — https://neelmishra.github.io/blog/mlops/rag/query-transformation.html
34. [secondary] Best Rerankers for RAG in 2026 — https://futureagi.com/blog/best-rerankers-for-rag-2026/
35. [peer-reviewed] jina-reranker-v3 — https://arxiv.org/abs/2509.25085
36. [secondary] Hybrid Search & Re-ranking in Production RAG 2026 — https://appscale.blog/en/blog/hybrid-search-and-reranking-production-rag-bm25-dense-cross-encoder-2026

**Open-source frameworks & answer engines — [official] repos/docs**

37. Perplexica — https://github.com/ItzCrazyKns/Perplexica
38. GPT-Researcher — https://github.com/assafelovic/gpt-researcher
39. Morphic — https://github.com/miurla/morphic
40. Farfalle — https://github.com/rashadphz/farfalle
41. Khoj — https://github.com/khoj-ai/khoj
42. LeptonAI search_with_lepton — https://github.com/leptonai/search_with_lepton
43. LangGraph Agentic RAG — https://docs.langchain.com/oss/python/langgraph/agentic-rag
44. LlamaIndex CitationQueryEngine — https://developers.llamaindex.ai/python/examples/query_engine/citation_query_engine/
45. Haystack (deepset) — https://github.com/deepset-ai/haystack
46. DSPy RAG — https://dspy.ai/tutorials/rag/
47. SearXNG self-hosted grounding pipeline (reference impl) — https://github.com/TadMSTR/searxng-mcp

**Proprietary systems & managed grounding**

48. [official] Grounding with Google Search (Gemini API) — https://ai.google.dev/gemini-api/docs/google-search
49. [official] OpenAI Responses API — Web search — https://developers.openai.com/api/docs/guides/tools-web-search
50. [official] Anthropic Claude web search tool — https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
51. [speculative] How Perplexity AI Answers Work (third-party reverse-engineering, directional) — https://ziptie.dev/blog/how-perplexity-ai-answers-work/
52. [vendor] Perplexity Research — Architecting an AI-First Search API — https://research.perplexity.ai/articles/architecting-and-evaluating-an-ai-first-search-api
53. [secondary] How Microsoft Copilot Search Works (third-party) — https://rankly.substack.com/p/how-microsoft-copilot-search-works
54. [official] AWS Bedrock Knowledge Bases — https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html
55. [official] GCP Grounding with Google Search (Vertex) — https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search
56. [official] Azure OpenAI "On Your Data" deprecation — https://learn.microsoft.com/en-us/azure/foundry-classic/openai/concepts/use-your-data
57. [official] IBM watsonx.data (OpenRAG) — https://www.ibm.com/products/watsonx-data
58. [vendor] Oracle AI updates (June 2026) — https://blogs.oracle.com/ai-and-datascience/whats-new-in-ai-june-2026

**Open-weight models for the generation core — [secondary]**

59. DeepSeek V3 vs Llama 4 vs Qwen 3 (2026) — https://appscale.blog/en/blog/deepseek-v3-vs-llama-4-vs-qwen-3-open-weight-comparison-2026
60. Best Open-Source LLM 2026 — https://codersera.com/blog/best-open-source-llm-2026-llama-4-qwen-3-5-deepseek-v4-gemma-4-mistral/
