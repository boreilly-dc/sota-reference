# Grounded Search for LLMs — State of the Art Reference

| Field | Value |
|-------|-------|
| Created | 2026-06-24 |
| Last Updated | 2026-06-24 |
| Version | 1.0 |

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

### 2.2 Two reference profiles

The full pipeline scales down. Pick a profile by your compute budget and reliability needs.

| | **Minimal viable pipeline** | **Full canonical pipeline** |
|---|---|---|
| Query understanding | none / single rewrite | classify intent → HyDE or decomposition |
| Search | SearXNG (snippets only) | SearXNG + paid API fallback |
| Extraction | snippet-only (no fetch) | selective full-text (Trafilatura/Firecrawl) |
| Retrieval | BM25 over snippets | hybrid BM25 + dense + rerank |
| Generation | 8B open model, prompt-based citations | larger model / reasoning core |
| Verification | DeBERTa-V3 NLI on each claim | NLI + LLM-judge + abstention |
| Hardware | single GPU (~12 GB VRAM) | multi-GPU / managed inference |

The minimal profile is a legitimate production starting point for low query volumes — the dominant cost saving is **not fetching and extracting every result**. Decoupling search from extraction and letting the model select which URLs are worth full-text reading cuts extraction cost by an estimated 70–90% (codenote.net comparison, 2026). Start snippet-only; add full extraction only where snippets prove insufficient.

---

## 3. The Retrieval & Search-Backend Layer

This is the layer most teams underestimate. **Retrieval quality is the primary bottleneck of a grounded-search system, not LLM capability** — a finding echoed across both the open literature and reverse-engineering of Perplexity.

### 3.1 Web search backends

The web-search API market shifted materially in 2025–26, and several defaults that older tutorials assume are now gone:

- **Bing Web Search API — retired 11 August 2025** (official Microsoft lifecycle notice). The replacement, *Grounding with Bing Search* inside Azure AI Agents, costs roughly **$35 per 1,000 queries** — far more than the old API. Do not design new systems around the Bing API.
- **Google Custom Search JSON API — closed to new customers (2025), full retirement 1 January 2027.** Google directs new builders to Vertex AI Search. Not a viable foundation for a new project.

That leaves a field of LLM-oriented search APIs and one strong open-source option:

| Backend | Type | Indicative price /1K | Notes |
|---|---|---|---|
| **SearXNG** (open source) | Self-hosted meta-search | Free (VPS only) | Aggregates 200+ engines behind one JSON API; the open-source default. Google sub-queries hit CAPTCHAs at scale (see 3.5). |
| Brave Search API | Independent index | ~$5 | Own 30B-page index (not scraped from others); SOC 2 Type II; `/llm/context` endpoint returns LLM-shaped content. Retired its free tier Feb 2026. |
| Tavily | LLM-native search+extract | ~$8 (basic) / $16 (adv) | De-facto LangChain default; returns clean content. Being acquired by Nebius (Feb 2026) — avoid deep lock-in. |
| Exa | Neural/semantic | ~$7 search + $1 contents | Embedding-based search with a `/findSimilar` endpoint; weaker on high-freshness content (news, prices). |
| Serper.dev | Raw Google SERP | ~$0.30–1 | Cheapest, but exposed to the *Google v. SerpAPI* lawsuit (filed Dec 2025) — legal risk for all Google-scraping providers. |
| You.com | Search + Web-LLM | ~$5 (sales-led) | Returns full content; also offers an end-to-end RAG endpoint. |

**Recommendation for builders:** default to **SearXNG** for development and low-volume production (open-source, no per-query cost, no vendor lock-in), with a paid **independent-index** provider (Brave) as the scale/quality fallback. Avoid building solely on Google-SERP-scraping providers given the active litigation.

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

### 3.5 Resolving the SearXNG scalability caveat

SearXNG is the right open-source default, but it proxies public engines, so high-volume Google sub-queries trigger CAPTCHAs and 429s. In production: (a) configure SearXNG to prefer engines that tolerate automation (Brave, DuckDuckGo, Wikipedia) over Google; (b) run multiple instances behind rate-limiting; (c) keep a paid independent-index API (Brave) as a fallback when SearXNG returns thin results. Treat pure-SearXNG-on-Google as a **development-tier** configuration, not a scale tier.

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
- A 2025 study (SDP workshop) found **DeBERTa-V3-large fine-tuned on NLI corpora outperforms LLM prompting** for reference-grounded hallucination detection, while being far cheaper and faster — a strong, self-hostable default verifier.
- **AttrScore** (OSU, EMNLP Findings 2023) classifies attribution errors into *attributable / extrapolatory (goes beyond source) / contradictory*; `AttrScore-Flan-T5` (3B) is a compact open evaluator. **AttributionBench** (ACL Findings 2024) confirms NLI-fine-tuned models lead this task.

**Verifier options, by cost/accuracy:**

| Approach | Latency/cost | Notes |
|---|---|---|
| DeBERTa-V3 NLI (open, ~400M) | Lowest | Best accuracy-per-dollar for entailment; recommended default. |
| AttrScore-Flan-T5 (open, 3B) | Low | Gives an error *type*, not just a binary. |
| SelfCite context-ablation | Medium (2 fwd passes/claim) | No extra model; reuses the generator. |
| LLM-as-judge | Highest | Flexible but slow, costly, and itself prone to sycophancy. Use sparingly. |

The verification stage should **down-grade or remove** any claim whose citation does not entail it, and surface abstention ("the sources do not confirm X") rather than paper over gaps. Calibrated abstention is a quality signal.

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

The 2025–26 research wave trains the model itself to search well, rather than scripting it. This is the "build your own" frontier if you can fine-tune.

- **WebGPT** (OpenAI, 2022) — the progenitor: fine-tuned GPT-3 to browse a text interface (search/click/quote/cite) via imitation learning then RLHF. Established that RL-from-human-feedback can optimise web-grounded QA with citations.
- **Search-R1** (UIUC, 2025) — DeepSeek-R1-style RL teaching an LLM to emit `<search>` queries inside its reasoning. Two key tricks: an **outcome-only reward** (just answer correctness — no process/format reward needed) and **retrieved-token masking** (mask retrieved passages during gradient computation so the model is rewarded for its own reasoning/queries, not for copying retrieved text). +41% over RAG baselines on Qwen2.5-7B; **GRPO outperformed PPO**.
- **R1-Searcher / R1-Searcher++** (RUC, 2025) — two-stage (SFT cold-start → RL) training for *dynamic knowledge acquisition*: the model learns to choose between its own parametric knowledge and external search, reducing both over- and under-retrieval.
- **DeepResearcher** (Shanghai AI Lab, EMNLP 2025) — the first end-to-end RL training **in the live web** (not a frozen RAG corpus), with a reasoning agent plus browsing agents. RL produced *emergent* behaviours: planning, cross-validation across sources, self-reflection, and honest abstention. +28.9 over prompt-engineering and +7.2 over RAG-corpus RL. Its thesis — that training in the real, noisy web is a requirement, not an implementation detail — is the most important recent finding for builders.
- **ZeroSearch** (Alibaba, 2025) — cuts RL training cost by replacing the live search engine *during training* with a 3B LLM that simulates retrieval, using a curriculum that progressively degrades document quality. Real search is used only at inference. Removes the API-cost and quality-variance problems of live-search RL.
- **WebThinker** (RUC, NeurIPS 2025) — equips large reasoning models (DeepSeek-R1, o1-class) to interleave search, navigation, and report drafting inside the thinking process; trained with iterative online DPO.
- **Search-o1** (RUC, EMNLP 2025) — names the *Reasoning-Retrieval Dilemma*: injecting raw retrieved documents disrupts a long reasoning chain. Its *Reason-in-Documents* module distils retrieved content into reasoning-compatible form before injection.

A practical note on reward design: across these systems, **citation quality is an emergent property of optimising answer correctness** — none use an explicit citation reward. If you train your own, start with outcome reward + retrieved-token masking (Search-R1) before engineering anything fancier.

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

No single model dominates every axis; choose by your dominant workload (reasoning depth vs. throughput vs. context length) and validate on your own grounded-search eval. For the minimal profile, an 8B-class instruction model with prompt-based inline citation plus a DeBERTa NLI verifier is a credible production starting point. (See the repository's frontier- and open-model surveys for current rankings.)

---

## 9. Hyperscaler Managed Grounding

If you would rather buy the grounding layer, the major clouds offer managed equivalents. Coverage is uneven — AWS, GCP, and Azure are mature; IBM and Oracle are thinner.

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

---

## 15. Areas of Uncertainty & Caveats

- **Reverse-engineered internals.** Perplexity's architecture (§10) is reconstructed from third-party analysis with possible commercial bias — directional only.
- **Preprint-heavy frontier.** Much of §6 (Search-R1, DeepResearcher, ZeroSearch, R1-Searcher, WebThinker, Search-o1) is 2025–26 work, some not yet peer-reviewed and reproduced; reported gains are indicative.
- **Vendor pricing is volatile.** All per-query prices (§3, §9) are 2026 snapshots from vendor pages and shift with acquisitions (Nebius–Tavily), tier changes (Brave free-tier retirement), and litigation. Re-check before committing.
- **English-centric evidence.** Retrieval quality, extraction accuracy, and NLI-verifier availability are materially worse in low-resource languages; this article's sources are overwhelmingly English.
- **Thin IBM/Oracle coverage.** Managed grounding evidence for IBM watsonx and Oracle OCI is limited; validate current capabilities directly.
- **Limited independent benchmarking.** Cross-system, apples-to-apples comparisons (RL vs. prompt-orchestrated; backend A vs. B on identical queries) are scarce — benchmark on your own workload.
- **Search-provider sustainability.** The economics and acceptable-use posture of LLM-driven query volume against commercial search engines remain in flux.

---

## References

**Grounding, attribution & verification (academic)**

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

25. Bing Search APIs Retiring Aug 2025, Microsoft Lifecycle — https://learn.microsoft.com/en-us/lifecycle/announcements/bing-search-api-retirement
26. Google Custom Search JSON API pricing/retirement — https://blog.expertrec.com/google-custom-search-json-api-simplified/
27. Search & Extraction APIs for AI Agents — Cost Comparison (2026) — https://codenote.net/en/posts/tavily-alternatives-cost-comparison-search-extract-api/
28. Exa/Tavily/Serper/Brave comparison — https://rhumb.dev/blog/exa-vs-tavily-vs-serper-vs-brave-search
29. SearXNG — https://github.com/searxng/searxng
30. Trafilatura evaluation — https://trafilatura.readthedocs.io/en/latest/evaluation.html
31. Query transformation (HyDE / Multi-Query / Step-Back / Decomposition) — https://neelmishra.github.io/blog/mlops/rag/query-transformation.html
32. Best Rerankers for RAG in 2026 — https://futureagi.com/blog/best-rerankers-for-rag-2026/
33. jina-reranker-v3 — https://arxiv.org/html/2509.25085v3
34. Hybrid Search & Re-ranking in Production RAG 2026 — https://appscale.blog/en/blog/hybrid-search-and-reranking-production-rag-bm25-dense-cross-encoder-2026

**Open-source frameworks & answer engines**

35. Perplexica — https://github.com/ItzCrazyKns/Perplexica
36. GPT-Researcher — https://github.com/assafelovic/gpt-researcher
37. Morphic — https://github.com/miurla/morphic
38. Farfalle — https://github.com/rashadphz/farfalle
39. Khoj — https://github.com/khoj-ai/khoj
40. LeptonAI search_with_lepton — https://github.com/leptonai/search_with_lepton
41. LangGraph Agentic RAG — https://docs.langchain.com/oss/python/langgraph/agentic-rag
42. LlamaIndex CitationQueryEngine — https://developers.llamaindex.ai/python/examples/query_engine/citation_query_engine/
43. Haystack (deepset) — https://github.com/deepset-ai/haystack
44. DSPy RAG — https://dspy.ai/tutorials/rag/
45. SearXNG self-hosted grounding pipeline (reference impl) — https://github.com/TadMSTR/searxng-mcp

**Proprietary systems & managed grounding**

46. Grounding with Google Search (Gemini API) — https://ai.google.dev/gemini-api/docs/google-search
47. OpenAI Responses API — Web search — https://developers.openai.com/api/docs/guides/tools-web-search
48. Anthropic Claude web search tool — https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
49. How Perplexity AI Answers Work (third-party, directional) — https://ziptie.dev/blog/how-perplexity-ai-answers-work/
50. Perplexity Research — Architecting an AI-First Search API — https://research.perplexity.ai/articles/architecting-and-evaluating-an-ai-first-search-api
51. How Microsoft Copilot Search Works (third-party) — https://rankly.substack.com/p/how-microsoft-copilot-search-works
52. AWS Bedrock Knowledge Bases — https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html
53. GCP Grounding with Google Search (Vertex) — https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search
54. Azure OpenAI "On Your Data" deprecation — https://learn.microsoft.com/en-us/azure/foundry-classic/openai/concepts/use-your-data
55. IBM watsonx.data (OpenRAG) — https://www.ibm.com/products/watsonx-data
56. Oracle AI updates (June 2026) — https://blogs.oracle.com/ai-and-datascience/whats-new-in-ai-june-2026

**Open-weight models for the generation core**

57. DeepSeek V3 vs Llama 4 vs Qwen 3 (2026) — https://appscale.blog/en/blog/deepseek-v3-vs-llama-4-vs-qwen-3-open-weight-comparison-2026
58. Best Open-Source LLM 2026 — https://codersera.com/blog/best-open-source-llm-2026-llama-4-qwen-3-5-deepseek-v4-gemma-4-mistral/
