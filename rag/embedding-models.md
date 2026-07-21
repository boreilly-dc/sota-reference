# Embedding Models: Current Best Models by Domain and Size

| Field | Value |
|-------|-------|
| Created | 2026-03-19 |
| Last Updated | 2026-07-21 |
| Version | 4.1 |

---

- [Executive Summary](#executive-summary)
- [How to Read the Rankings](#how-to-read-the-rankings)
- [Size Classes](#size-classes)
- [Best-in-Class Matrix](#best-in-class-matrix)
- [Best Open-Weight Models](#best-open-weight-models)
- [Best Fully Open Models](#best-fully-open-models)
- [Domain Recommendations](#domain-recommendations)
- [Model Profiles by Size](#model-profiles-by-size)
- [Retrieval Architecture Matters](#retrieval-architecture-matters)
- [Multimodal Embeddings](#multimodal-embeddings)
- [Managed Embedding APIs](#managed-embedding-apis)
- [Deployment and Hardware](#deployment-and-hardware)
- [Evaluation and Selection Process](#evaluation-and-selection-process)
- [Areas of Uncertainty](#areas-of-uncertainty)
- [References](#references)

## Executive Summary

There is no single best embedding model for every workload. The best model changes with the domain, parameter budget, language, input length, modality, and retrieval architecture. The live MTEB leaderboard now contains separate suites for English, multilingual text, law, medicine, code, long-context retrieval, visual documents, and the beta Retrieval Embedding Benchmark (RTEB). This separation is useful because an overall MTEB average can hide a large domain-specific gap.

As of **21 July 2026**, the strongest shortlists are shown in three views: the absolute benchmark leader, the best open-weight submission, and the best fully-open submission. The open-weight view includes downloadable models with non-commercial or custom licences; the fully-open view requires permissive weights, training code, and training data.

The main findings are:

- **General English text:** `jina-embeddings-v5-text-nano` at 212M, `Jasper-Token-Compression-600M` at 607M, and the proprietary `ingot-8b-r3` at 7.6B lead their size classes on MTEB English v2. For a permissively licensed large model, use `Qwen3-Embedding-8B` as the safer production default.
- **Multilingual text:** Microsoft `harrier-oss-v1-270m`, `harrier-oss-v1-0.6b`, and `harrier-oss-v1-27b` lead the small, medium, and very-large classes on MTEB Multilingual v2. The 27B model is MIT licensed and has a 131K-token limit.
- **Code:** LightOn `LateOn-Code-edge` (17M) and `LateOn-Code` (149M) lead the two smallest classes. CodeFuse `F2LLM-v2-1.7B` and `F2LLM-v2-14B` lead the medium and very-large classes on both MTEB Code and CoIR. These systems do not all use the same index design: LateOn is late-interaction, while F2LLM is dense.
- **Legal:** `dinghy-law-0.6b-v1` is the strongest compact specialist on MTEB Law. `Euler-Legal-Embedding-V1` leads that suite overall. RTEB Legal gives different winners, so legal teams must test both benchmark styles on their own matter types.
- **Finance and healthcare retrieval:** NVIDIA's July 2026 `Nemotron-3-Embed-8B` leads RTEB Finance and is the strongest submitted dense model in its size class on RTEB Healthcare. RTEB is still beta, and the model's custom NVIDIA licence needs a legal review.
- **Long documents:** `GTE-ModernColBERT-v1` (149M) leads its class and ranks second overall on LongEmbed. Its late-interaction index uses many token vectors per document, so it is not storage-equivalent to a 149M dense encoder.
- **Multimodal:** `jina-embeddings-v5-omni-small` (1.63B) leads MIEB Multilingual. `Nemotron-3-Embed-8B` leads ViDoRe v3.1 for visually rich document retrieval. The suites measure different tasks and cannot be merged into one ranking.
- **Fully open:** CodeFuse F2LLM-v2 provides the broadest set of size-class leaders with permissive weights, training code, and training data. Jasper 600M leads the fully-open medium class for English v2. LateOn-Code leads the fully-open tiny and small code classes. GTE-ModernColBERT leads the fully-open small class on LongEmbed.

Use these results to form a shortlist. Do not select a production model from a leaderboard alone. Run a held-out evaluation on the target corpus and measure nDCG@10, Recall@k, MRR, latency, peak memory, vector-index size, and cost.

## How to Read the Rankings

### Evidence rules

This article applies the following rules:

1. A model is a size-class leader only when the live MTEB API reports it as the leader for that exact benchmark and parameter interval.
2. Scores from different benchmark suites or versions are not compared numerically.
3. A dense encoder, sparse encoder, late-interaction model, and reranker are identified separately. Their storage and latency costs differ.
4. Open weights do not mean open source. The licence, training code, and training data are separate properties.
5. In this article, **fully open** means downloadable weights, a permissive commercial licence, public training code, and public training data. A paper and model card support reproducibility but do not replace code or data.
6. A benchmark winner is not always the recommended production model. Licence, reproducibility, deployment support, and data leakage can change the recommendation.

### Confidence labels

| Label | Meaning |
|-------|---------|
| **High** | Mature benchmark, enough submitted models, primary metadata available, and the result is suitable for the stated task. |
| **Medium** | Useful primary result, but the suite is narrow, new, beta, or sensitive to architecture differences. |
| **Low** | Sparse comparable evidence or only vendor-reported evidence. Treat the model as a candidate, not a winner. |

### Important benchmark boundaries

- **MTEB English v2** covers 41 tasks across classification, clustering, retrieval, reranking, pair classification, and semantic similarity.
- **MTEB Multilingual v2** covers 131 tasks. Its aggregate is not comparable with English v2.
- **MTEB Law v1** has eight legal tasks. **MTEB Medical v1** has 12 medical tasks.
- **MTEB Code v1** has 12 tasks. **CoIR** has ten code retrieval datasets.
- **LongEmbed** has six long-context retrieval tasks.
- **MIEB** measures image and image-text embeddings. **ViDoRe** focuses on visually rich document retrieval.
- **RTEB** uses fresh retrieval data to reduce contamination, but it remains a beta benchmark as of July 2026.

## Size Classes

The classes reflect deployment hardware rather than the old three-tier convention.

| Class | Parameters | Typical deployment | Approximate FP16 weight memory |
|-------|------------|--------------------|--------------------------------|
| **Tiny** | Under 100M | Mobile, edge, CPU service, CI | Under 0.2GB |
| **Small** | 100M to 500M | Laptop CPU, edge GPU, small server | 0.2–1GB |
| **Medium** | Over 500M to 2B | Laptop with 16GB memory, consumer GPU | 1–4GB |
| **Large** | Over 2B to 10B | Workstation or datacentre GPU | 4–20GB |
| **Very large / API** | Over 10B, or managed model with undisclosed size | Multi-GPU, large unified memory, or API | Over 20GB or provider-managed |

The bounds use total parameters, not active mixture-of-experts parameters. Runtime memory also includes activations, framework overhead, and batching. Late-interaction systems can have a small model but a much larger index than dense systems.

## Best-in-Class Matrix

The table gives the leader returned by the live MTEB API on 21 July 2026 for each class. **—** means that the suite does not contain enough comparable evidence for a useful recommendation. Model names are shortened; links and licence details appear in later sections and the references. The snapshot can change when results are added or corrected.

| Domain and benchmark | Tiny (<100M) | Small (100–500M) | Medium (>500M–2B) | Large (>2B–10B) | Very large / API | Confidence |
|----------------------|--------------|-------------------|--------------------|-----------------|------------------|------------|
| **General English** — MTEB English v2 | GIST-small (33M) | Jina v5 text nano (212M) | Jasper TC 600M (607M) | ingot-8b-r3 (7.6B, proprietary) | F2LLM-v2-14B | High |
| **General retrieval** — BEIR | MongoDB leaf-ir (23M) | Stella 400M v5 (435M) | Stella 1.5B v5 | Qwen3-Embedding-8B | F2LLM-v2-14B | High |
| **Multilingual** — MTEB Multilingual v2 | F2LLM-v2-80M | Harrier 270M | Harrier 0.6B | Llama Embed Nemotron 8B | Harrier 27B | High |
| **Code** — MTEB Code / CoIR | LateOn-Code-edge (17M) | LateOn-Code (149M) | F2LLM-v2-1.7B | C2LLM-7B | F2LLM-v2-14B | High |
| **Legal** — MTEB Law v1 | Ivysaur (23M) | Arctic Embed M v2 (305M) | Dinghy Law 0.6B | Euler Legal 8B | F2LLM-v2-14B | Medium |
| **Finance retrieval** — RTEB Finance beta | MongoDB leaf-ir (23M) | Jina v5 text nano | Nemotron-3 Embed 1B | Nemotron-3 Embed 8B | F2LLM-v2-14B | Medium |
| **Medical, broad tasks** — MTEB Medical v1 | F2LLM-v2-80M | F2LLM-v2-330M | GTE-Qwen2-1.5B | GTE-Qwen2-7B | F2LLM-v2-14B | Medium |
| **Healthcare retrieval** — RTEB Healthcare beta | F2LLM-v2-80M | Jina v5 text nano | Nemotron-3 Embed 1B | Nemotron-3 Embed 8B | F2LLM-v2-14B | Medium |
| **Long-context retrieval** — LongEmbed | Granite 97M R2 | GTE-ModernColBERT (149M) | BidirLM 1B | F2LLM-v2-8B | F2LLM-v2-14B | Medium |
| **Image-text, multilingual** — MIEB | Nomic Embed Vision 1.5 (93M) | SigLIP Base 512 (204M) | Jina v5 Omni Small (1.63B) | E5-V (8.36B) | — | Medium |
| **Visual documents** — ViDoRe v3.1 | — | EmbeddingGemma 300M | Nemotron-3 Embed 1B | Nemotron-3 Embed 8B | — | Medium |
| **Hybrid dense+sparse** | — | — | BGE-M3 (568M) | — | Cohere Embed v4 API | Medium |
| **Classification / STS / topic routing** | GIST-small | Jina v5 text nano | Jasper TC 600M | Evaluate task adapter | Evaluate managed API | Medium |
| **Scientific/technical literature** | — | — | — | — | — | Low |

### What the matrix does not imply

- A higher parameter class does not always have a better score. The very-large entry is only the best model *inside that class*.
- The scientific row is intentionally unfilled. SciFact, NFCorpus, arXiv, bioRxiv, and related tasks exist, but the current public leaderboard does not provide a clean, independent scientific-domain aggregate. Use a corpus-specific evaluation.
- The legal, finance, and health rows show that benchmark definition matters. MTEB Law and RTEB Legal have different leaders. MTEB Medical and RTEB Healthcare also differ.
- The matrix includes non-commercial and proprietary models when they lead. The two matrices below make the open alternatives explicit.

## Best Open-Weight Models

This matrix filters each benchmark and size class to models with downloadable weights. **P** means a permissive commercial licence, **NC** means non-commercial, and **C** means a custom licence that needs review. Open weights do not imply public training code or data. Sparse baselines and rerankers are excluded so the entries remain embedding retrievers.

| Domain and benchmark | Tiny (<100M) | Small (100–500M) | Medium (>500M–2B) | Large (>2B–10B) | Very large (>10B) |
|----------------------|--------------|-------------------|--------------------|-----------------|-------------------|
| **General English** — MTEB English v2 | GIST-small, P | Jina v5 nano, NC | Jasper 600M, P | QZhou-Embedding, P | F2LLM-v2-14B, P |
| **General retrieval** — BEIR | MongoDB leaf-ir, P | Stella 400M v5, P | Stella 1.5B v5, P | Qwen3-Embedding-8B, P | F2LLM-v2-14B, P |
| **Multilingual** — MTEB Multilingual v2 | F2LLM-v2-80M, P | Harrier 270M, P | Harrier 0.6B, P | Llama Embed Nemotron 8B, C | Harrier 27B, P |
| **Code** — MTEB Code v1 | LateOn-Code-edge, P | LateOn-Code, P | F2LLM-v2-1.7B, P | C2LLM-7B, P | F2LLM-v2-14B, P |
| **Legal** — MTEB Law v1 | Ivysaur, P | Arctic Embed M v2, P | Dinghy Law 0.6B, P | Euler Legal 8B, P | F2LLM-v2-14B, P |
| **Finance retrieval** — RTEB beta | MongoDB leaf-ir, P | Jina v5 nano, NC | Nemotron-3 Embed 1B, C | Nemotron-3 Embed 8B, C | F2LLM-v2-14B, P |
| **Medical** — MTEB Medical v1 | F2LLM-v2-80M, P | F2LLM-v2-330M, P | GTE-Qwen2-1.5B, P | GTE-Qwen2-7B, P | F2LLM-v2-14B, P |
| **Healthcare retrieval** — RTEB beta | F2LLM-v2-80M, P | Jina v5 nano, NC | Nemotron-3 Embed 1B, C | Nemotron-3 Embed 8B, C | F2LLM-v2-14B, P |
| **Long-context retrieval** — LongEmbed | Granite 97M R2, P | GTE-ModernColBERT, P | BidirLM 1B, P | F2LLM-v2-8B, P | F2LLM-v2-14B, P |
| **Image-text, multilingual** — MIEB | Nomic Embed Vision 1.5, P | SigLIP Base 512, P | Jina v5 Omni Small, NC | E5-V, licence unclear | — |
| **Visual documents** — ViDoRe v3.1 | — | EmbeddingGemma 300M, P | Nemotron-3 Embed 1B, C | Nemotron-3 Embed 8B, C | — |

The open-weight matrix is often the practical shortlist. It still includes non-commercial and custom licences because the weights are downloadable. Use the fully-open matrix when auditability, modification, or reproducible training is a requirement.

## Best Fully Open Models

These entries pass the strict article test: downloadable weights, a permissive commercial licence, public training code, and public training data. The result can be lower than the open-weight leader because many strong models do not publish their complete training pipeline. **—** means no qualifying submitted model was found in that class. It does not prove that no fully-open model exists outside the submitted results. Missing MTEB artefact links are treated as missing evidence, not as proof that an artefact is unavailable.

| Domain and benchmark | Tiny (<100M) | Small (100–500M) | Medium (>500M–2B) | Large (>2B–10B) | Very large (>10B) |
|----------------------|--------------|-------------------|--------------------|-----------------|-------------------|
| **General English** — MTEB English v2 | F2LLM-v2-80M | F2LLM-v2-330M | Jasper 600M | F2LLM-v2-8B | F2LLM-v2-14B |
| **General retrieval** — BEIR | F2LLM-v2-80M | ColBERT-Zero | F2LLM-v2-1.7B | BGE-en-ICL | F2LLM-v2-14B |
| **Multilingual** — MTEB Multilingual v2 | F2LLM-v2-80M | F2LLM-v2-330M | F2LLM-v2-1.7B | F2LLM-v2-8B | F2LLM-v2-14B |
| **Code** — MTEB Code v1 | LateOn-Code-edge | LateOn-Code | F2LLM-v2-1.7B | F2LLM-v2-4B | F2LLM-v2-14B |
| **Legal** — MTEB Law v1 | F2LLM-v2-80M | F2LLM-v2-330M | F2LLM-v2-1.7B | F2LLM-v2-8B | F2LLM-v2-14B |
| **Finance retrieval** — RTEB beta | F2LLM-v2-80M | F2LLM-v2-330M | F2LLM-v2-1.7B | F2LLM-v2-8B | F2LLM-v2-14B |
| **Medical** — MTEB Medical v1 | F2LLM-v2-80M | F2LLM-v2-330M | Jasper Vision-Language v1 | F2LLM-v2-8B | F2LLM-v2-14B |
| **Healthcare retrieval** — RTEB beta | F2LLM-v2-80M | F2LLM-v2-330M | F2LLM-v2-1.7B | F2LLM-v2-8B | F2LLM-v2-14B |
| **Long-context retrieval** — LongEmbed | F2LLM-v2-80M | GTE-ModernColBERT | F2LLM-v2-1.7B | F2LLM-v2-8B | F2LLM-v2-14B |
| **Image-text, multilingual** — MIEB | — | LAION CLIP-L DataComp XL | LAION CLIP-H laion2B | LAION CLIP-bigG laion2B | — |
| **Visual documents** — ViDoRe v3.1 | — | ColModernVBERT | — | Nomic Embed Multimodal 7B | — |

F2LLM appears frequently because CodeFuse publishes the Apache-2.0 weights, training code, and training dataset for a broad family of sizes. This is a transparency advantage, not evidence that one family is optimal for every production corpus. Jasper 600M, LateOn-Code, GTE-ModernColBERT, LAION CLIP, and selected BGE/Nomic models provide independent fully-open alternatives.

## Domain Recommendations

### General RAG and enterprise search

**Recommended default:** `Qwen3-Embedding-8B` when a large GPU is available; `Jasper-Token-Compression-600M` for a medium local and fully-open model; `jina-embeddings-v5-text-nano` when its non-commercial licence is acceptable. Use F2LLM-v2 when a fully-open model is required at another size.

Qwen3-Embedding-8B leads the large class on BEIR and has an Apache 2.0 licence, a 32K-token limit, Matryoshka dimensions, broad language coverage, and mature Sentence Transformers support. Jasper leads the medium class on MTEB English v2 and is MIT licensed with published training code and data. Jina v5 nano leads the small class on English v2, but its CC-BY-NC-4.0 licence excludes many commercial deployments.

For commercial CPU deployment, shortlist `ibm-granite/granite-embedding-311m-multilingual-r2`, Snowflake Arctic Embed v2, and permissively licensed 300–500M models even when a non-commercial model has a higher leaderboard score.

### Multilingual and cross-lingual retrieval

**Recommended default:** Harrier OSS v1.

- `harrier-oss-v1-270m` leads the small class.
- `harrier-oss-v1-0.6b` leads the medium class.
- `harrier-oss-v1-27b` leads the benchmark overall with a mean task score of 0.7427.

The family uses an MIT licence. The 27B model supports a 131K-token limit and 100+ language or script variants. Its weights need about 51.5GB according to MTEB metadata, so the 270M and 0.6B versions are more practical.

Do not treat one multilingual average as proof for every language. Evaluate each production language, code-switching pattern, and script. Test cross-lingual query-to-document retrieval separately from same-language retrieval.

### Code search

**Recommended by scale:**

- **Tiny:** `LateOn-Code-edge` (17M).
- **Small:** `LateOn-Code` (149M, Apache 2.0).
- **Medium:** `F2LLM-v2-1.7B` (Apache 2.0).
- **Large:** `C2LLM-7B` for the live score; also test `F2LLM-v2-8B` for a fully documented family.
- **Very large:** `F2LLM-v2-14B`.

LateOn-Code is a late-interaction model. It keeps one 128-dimensional vector per token rather than one vector per passage. This design improves token-level code matching, but it increases index size and changes query cost. F2LLM is a dense family and supports major programming languages. The 14B model leads MTEB Code and CoIR, but the difference from C2LLM-7B is small enough that hardware cost can decide the choice.

Evaluate natural-language-to-code, code-to-code, repository-level retrieval, symbol search, and language mix separately. CodeSearchNet alone is not enough for modern agentic coding systems.

### Legal retrieval

**Recommended default:** `dinghy-law-0.6b-v1` for compact commercial use; `Euler-Legal-Embedding-V1` when maximum MTEB Law quality matters.

Dinghy Law is a 597M Apache 2.0 model derived from Qwen3-Embedding-0.6B. It publishes its training dataset and ranks eighth overall and first in its size class on MTEB Law. Euler Legal is an Apache 2.0 7.6B model and leads MTEB Law, but its model card does not expose training data or code.

RTEB Legal gives `Octen-Embedding-8B` the top large-class score, not Euler. This disagreement is a warning: legal retrieval varies by jurisdiction, document type, citation style, and time. Test statutes, cases, contracts, opinions, and internal advice as separate strata. Do not use embedding similarity as a legal correctness measure.

### Finance

**Recommended default:** `Nemotron-3-Embed-8B` for a high-quality shortlist and `Nemotron-3-Embed-1B` where memory is constrained. Review the NVIDIA licence before commercial deployment.

The models were released on 16 July 2026. The 8B model leads RTEB Finance beta at 0.8793. The 1B variant leads the medium class at 0.7736. Both have a 32K-token limit and include finance retrieval data in training metadata. This creates a possible overlap concern, so the fresh RTEB result is useful but still needs independent replication.

FinE5 remains an important finance specialist because it was developed with FinMTEB, a 64-dataset English and Chinese benchmark. However, a model developed against a benchmark is not automatically the best independent test result. Use financial filings, tables, footnotes, dates, units, and entity aliases from the target workload.

### Biomedical and healthcare

**Recommended default:** test both `gte-Qwen2-7B-instruct` and `Nemotron-3-Embed-8B`.

GTE-Qwen2-7B leads the broad MTEB Medical v1 suite. Nemotron-3-Embed-8B is the strongest submitted dense model in its size class on RTEB Healthcare beta. These suites test different distributions. Medical search also needs terminology, abbreviation, evidence-date, and patient-safety controls that an embedding score does not provide.

For local deployment, `F2LLM-v2-330M` leads the small class on MTEB Medical, while Jina v5 nano leads the small class on RTEB Healthcare. Jina is non-commercial. Consider permissive alternatives and fine-tuning on in-domain pairs.

### Scientific and technical literature

There is no defensible universal winner from one current aggregate. Shortlist models that are strong on BEIR, SciFact, NFCorpus, arXiv, bioRxiv, and visual-document tasks. Then evaluate on the target field.

For text-only papers, start with Qwen3-Embedding-8B, GTE-Qwen2-7B, and a smaller model such as Jasper 600M. For papers with equations, charts, and tables, add Jina v5 Omni Small and Nemotron-3 Embed. Use metadata and citation links as structured retrieval signals rather than forcing all information into one vector.

### Long documents

**Recommended default:** `GTE-ModernColBERT-v1` when index growth is acceptable; BidirLM 1B or F2LLM-v2 when a single dense vector is required.

GTE-ModernColBERT leads the small class and ranks second overall on LongEmbed with 0.9058. It is Apache 2.0 and fully open, but it is a late-interaction model. Its 8K model limit is not the same as embedding an unlimited document. Long documents still need section-aware chunking, overlap control, metadata, or hierarchical retrieval.

A declared context window does not guarantee that the model preserves information at the end of that window. Test length buckets and position sensitivity on the target corpus.

### Classification, STS, and semantic routing

Overall MTEB English v2 is a useful first filter because it includes classification and semantic-similarity tasks. It is not a routing benchmark. For topic routing or guardrails, compare a small encoder with a trained linear classifier against direct cosine similarity.

Use `GIST-small` for a tiny baseline, Jina v5 nano for a high-quality non-commercial small model, and Jasper 600M for a permissive medium model. Calibrate thresholds per model and per intent. Never copy a cosine threshold from another model.

### Hybrid and late-interaction retrieval

`BGE-M3` remains the most practical open model when one encoder must produce dense, sparse, and ColBERT-style outputs. It is MIT licensed, multilingual, and 568M parameters. It is not the top dense-only model in its class, but the single-model hybrid design can improve total system quality.

Use reciprocal rank fusion to combine independent dense and lexical results. Add a cross-encoder reranker when latency permits. Compare the complete pipeline, not only the encoder score.

## Model Profiles by Size

### Tiny: under 100M

| Model | Params | Type | Context | Licence | Openness | Best fit |
|-------|--------|------|---------|---------|----------|----------|
| `LateOn-Code-edge` | 17M | Late interaction | Check model card | Apache 2.0 | Fully open | Code retrieval |
| `MongoDB/mdbr-leaf-ir` | 23M | Dense | Check model card | Check model card | Open weights | General and finance retrieval baseline |
| `GIST-small-Embedding-v0` | 33M | Dense | 512 | MIT | Open weights | General English and routing baseline |
| `F2LLM-v2-80M` | 80M | Dense | 40K | Apache 2.0 | Fully open | Multilingual, medical, code-capable edge use |
| `granite-embedding-97m-multilingual-r2` | 97M | Dense | 8K | Apache 2.0 | Open weights | Commercial multilingual CPU and long-context baseline |

Tiny models are useful for high-volume routing and first-stage retrieval. They give up recall on difficult semantic and cross-lingual queries. Quantise only after measuring the change in neighbour ordering.

### Small: 100M to 500M

| Model | Params | Type | Context | Licence | Openness | Best fit |
|-------|--------|------|---------|---------|----------|----------|
| `LateOn-Code` | 149M | Late interaction | 8K | Apache 2.0 | Fully open | Code retrieval |
| `GTE-ModernColBERT-v1` | 149M | Late interaction | 8K | Apache 2.0 | Fully open | Long-document retrieval |
| `jina-embeddings-v5-text-nano` | 212M | Dense + task adapters | 8K | CC-BY-NC-4.0 | Open weights, NC | English and multilingual quality where non-commercial terms fit |
| `harrier-oss-v1-270m` | 268M | Dense | 32K | MIT | Open weights | Multilingual commercial use |
| `snowflake-arctic-embed-m-v2.0` | 305M | Dense | 8K | Apache 2.0 | Open weights | Legal and general retrieval |
| `F2LLM-v2-330M` | 334M | Dense | 40K | Apache 2.0 | Fully open | Medical and multilingual retrieval |
| `stella_en_400M_v5` | 435M | Dense | 8K | MIT | Open weights | English BEIR retrieval |

Jina v5 nano has a strong score but a non-commercial licence. Harrier, Granite, Arctic, F2LLM, and Stella are safer starting points for commercial self-hosting.

### Medium: over 500M to 2B

| Model | Params | Type | Context | Licence | Openness | Best fit |
|-------|--------|------|---------|---------|----------|----------|
| `BGE-M3` | 568M | Dense + sparse + late interaction | 8K | MIT | Open weights | Hybrid multilingual retrieval |
| `dinghy-law-0.6b-v1` | 597M | Dense | 32K | Apache 2.0 | Open weights | Legal retrieval |
| `harrier-oss-v1-0.6b` | 596M | Dense | 32K | MIT | Open weights | Multilingual retrieval |
| `Jasper-Token-Compression-600M` | 607M | Dense | 32K | MIT | Fully open | General English |
| `BidirLM-1B-Embedding` | 1.0B | Dense | 32K | Check model card | Open weights | Long-context dense retrieval |
| `Nemotron-3-Embed-1B` | 1.14B | Dense | 32K | NVIDIA licence | Open weights, custom | Finance, health, visual documents |
| `gte-Qwen2-1.5B-instruct` | 1.54B | Dense | 32K | Apache 2.0 | Open weights | Medical and general retrieval |
| `stella_en_1.5B_v5` | 1.54B | Dense | 8K | MIT | Open weights | English BEIR retrieval |
| `jina-embeddings-v5-omni-small` | 1.63B | Dense multimodal | 32K | Check model card | Open weights, NC | Multilingual image-text retrieval |
| `F2LLM-v2-1.7B` | 1.72B | Dense | 40K | Apache 2.0 | Fully open | Code and multilingual text |

This class is the best local quality-to-cost range. Most models fit in 2–4GB of FP16 weights and can run on a consumer GPU or Apple Silicon with sufficient unified memory.

### Large: over 2B to 10B

| Model | Params | Type | Context | Licence | Openness | Best fit |
|-------|--------|------|---------|---------|----------|----------|
| `Qwen3-Embedding-8B` | 7.57B | Dense | 32K | Apache 2.0 | Open weights | General retrieval and multilingual baseline |
| `Euler-Legal-Embedding-V1` | 7.57B | Dense | 1.5K | Apache 2.0 | Open weights | MTEB Law |
| `C2LLM-7B` | 7.67B | Dense | Check model card | Check model card | Open weights | Code retrieval |
| `Nemotron-3-Embed-8B` | 7.95B | Dense | 32K | NVIDIA licence | Open weights, custom | Finance, health, visual documents |
| `F2LLM-v2-8B` | 7.57B | Dense | 40K | Apache 2.0 | Fully open | Long-context and multilingual retrieval |
| `E5-V` | 8.36B | Dense multimodal | Check model card | Check model card | Open weights | MIEB image-text tasks |

Use Qwen3-Embedding-8B as the permissive general-purpose control. Specialist winners should beat this control on a held-out domain test before adoption.

### Very large and managed

| Model | Size | Type | Context | Licence/access | Openness | Best fit |
|-------|------|------|---------|----------------|----------|----------|
| `F2LLM-v2-14B` | 14B | Dense | 40K | Apache 2.0 | Fully open | Code and broad multilingual tasks |
| `harrier-oss-v1-27b` | 27B | Dense | 131K | MIT | Open weights | Best MTEB Multilingual v2 aggregate |
| Gemini Embedding 2 | Undisclosed | Multimodal dense API | Provider-defined by modality | GCP API | Closed API | Text, image, audio, video, and PDF in one space |
| Cohere Embed v4 | Undisclosed | Text/image API | 128K | AWS, Azure, Oracle | Closed API | Enterprise multimodal and compressed outputs |
| OpenAI text-embedding-3 | Undisclosed | Dense API | 8K | Azure API | Closed API | Mature managed text embeddings |

A larger model can lose to a smaller specialist. Select this class only when measured quality offsets serving, latency, and re-indexing costs.

## Retrieval Architecture Matters

### Dense bi-encoders

A dense encoder produces one vector for each query and passage. It has simple indexes, fast approximate nearest-neighbour search, and predictable storage. Most Qwen, Harrier, Jasper, GTE, Jina text, Granite, and Nemotron text results use this design.

### Sparse encoders

A sparse encoder produces weighted vocabulary features. It preserves exact terms and can work well for identifiers, product codes, citations, and rare terminology. SPLADE and BM25 are important controls. Sparse scores should usually be fused with dense scores rather than treated as a replacement.

### Late-interaction models

ColBERT-style models keep one vector per token and compute a MaxSim-style score. LateOn-Code and GTE-ModernColBERT lead important size classes, but their indexes are larger than a single-vector dense index. Compare bytes per document and query latency, not parameter count alone.

### Rerankers

A cross-encoder scores a query and candidate together. It is more expensive and usually more accurate than first-stage embedding similarity. Use it after dense, sparse, or hybrid retrieval. Do not place reranker scores in an embedding-model table as if they were directly interchangeable.

## Multimodal Embeddings

Multimodal ranking depends on the input and retrieval direction.

| Need | Current shortlist | Evidence |
|------|-------------------|----------|
| Multilingual image-text retrieval | Jina v5 Omni Small | First on MIEB Multilingual; 1.63B |
| Visually rich PDF/page retrieval | Nemotron-3 Embed 8B and 1B | First and second on ViDoRe v3.1 |
| Tiny image embeddings | Nomic Embed Vision v1.5 | Tiny-class leader on MIEB Multilingual |
| Small image-text embeddings | SigLIP Base Patch16 512 | Small-class leader on MIEB Multilingual |
| Five modalities through an API | Gemini Embedding 2 | Text, image, video, audio, and PDF |
| Open text/image/document/video pipeline | Qwen3-VL-Embedding 2B or 8B | Apache 2.0 model family and technical report |

MIEB and ViDoRe measure different things. MIEB has broad image and image-text tasks. ViDoRe tests retrieval from visually rich documents. A document system must also test OCR errors, table retrieval, chart labels, page order, and text-to-page versus page-to-page retrieval.

## Managed Embedding APIs

Prices and regional availability change frequently. Check the provider's current page before procurement.

| Hyperscaler | Managed options | Use when |
|-------------|-----------------|----------|
| **AWS** | Amazon Titan Text Embeddings, Cohere Embed through Bedrock; open models through SageMaker | Data and operations are already on AWS |
| **Azure** | Azure OpenAI text-embedding models; Cohere and selected models through Azure AI model catalogue | Azure governance and private networking are required |
| **GCP** | Gemini Embedding and text embedding models through Vertex AI | Native multimodal embedding or GCP integration is required |
| **IBM** | Granite embedding models through watsonx.ai; open Granite weights for self-hosting | IBM governance or Granite deployment is preferred |
| **Oracle** | Cohere Embed through OCI Generative AI; open models on OCI compute | OCI data locality is required |

Managed APIs remove model-serving work, but they add network latency, data-governance review, provider limits, and re-embedding risk. Store the model name, version, dimensions, task prefix, normalisation setting, and source text with every index build.

## Deployment and Hardware

### Weight-memory guide

| Model size | FP16 weights | INT8 weights | 4-bit weights | Practical host |
|------------|--------------|--------------|---------------|----------------|
| 100M | ~0.2GB | ~0.1GB | ~0.05GB | Any modern CPU or mobile-class GPU |
| 500M | ~1GB | ~0.5GB | ~0.25GB | Laptop CPU, Apple Silicon, entry GPU |
| 1B | ~2GB | ~1GB | ~0.5GB | 16GB laptop or consumer GPU |
| 8B | ~16GB | ~8GB | ~4GB | 16–32GB unified memory or 12GB+ GPU |
| 14B | ~28GB | ~14GB | ~7GB | 24GB GPU or 32GB+ unified memory |
| 27B | ~54GB | ~27GB | ~14GB | 64GB unified memory or datacentre GPU |

These figures cover weights only. Add runtime overhead and batch activations. Embedding encoders do not have the same autoregressive KV-cache growth as generative LLMs, but long sequences and large batches still increase activation memory.

### Serving frameworks

| Framework | Best fit |
|-----------|----------|
| Sentence Transformers | Development, evaluation, fine-tuning, broad model compatibility |
| Hugging Face Text Embeddings Inference | Production dense embedding and reranking services |
| vLLM pooling models | High-throughput decoder-derived encoders on NVIDIA GPUs |
| FastEmbed / ONNX Runtime | CPU-first serving and quantised small models |
| PyLate | Late-interaction training, indexing, and serving |
| FlagEmbedding | BGE dense, sparse, multi-vector, and reranker pipelines |
| Ollama / llama.cpp | Simple local dense embeddings when the model has a supported conversion |
| MLX | Apple Silicon deployments with a verified model implementation |

Do not assume that a model works correctly in every framework. Check pooling, task instruction, padding side, maximum length, normalisation, and Matryoshka truncation against the official model card.

## Evaluation and Selection Process

### Minimum evaluation set

Build a held-out set from production-like data. Include:

- at least 200 queries for a first comparison and more for stable subgroup results;
- judged relevant and hard-negative documents;
- short, medium, and long queries;
- each production language and domain;
- identifiers, abbreviations, misspellings, tables, and date-sensitive questions;
- adversarial and out-of-domain queries where routing or guardrails use the embeddings.

Keep this set separate from fine-tuning data. Check model training metadata for known overlap.

### Metrics

| Measure | Purpose |
|---------|---------|
| nDCG@10 | Rewards correct ordering with graded relevance |
| Recall@k | Measures whether enough relevant evidence enters the reranker or LLM context |
| MRR | Measures the rank of the first relevant result |
| Precision@k | Useful where only a small result set is shown |
| p50/p95 latency | Captures normal and tail query performance |
| Documents or tokens per second | Measures indexing throughput |
| Peak RAM/VRAM | Determines deployment fit |
| Bytes per document | Exposes dense, sparse, and late-interaction index cost |
| Cost per million documents and queries | Supports total-cost comparison |

Measure retrieval both before and after reranking. A weaker first stage can appear acceptable if the evaluation ignores relevant documents that never reach the reranker.

### Selection gates

1. **Quality gate:** The candidate must improve the primary retrieval metric or meet a fixed quality target.
2. **Safety gate:** The candidate must not regress protected-language, jurisdiction, medical-safety, or policy subsets.
3. **Licence gate:** The model and its dependencies must permit the intended commercial and redistribution use.
4. **Operations gate:** The model must meet p95 latency, throughput, memory, and index-size limits.
5. **Migration gate:** The re-embedding and rollback plan must be tested before replacement.

### Recommended trial set

For a new text RAG system, start with three structurally different controls:

1. a permissive dense model that fits the target hardware, such as Granite 97M/311M, Jasper 600M, or Qwen3-Embedding-8B;
2. a lexical baseline such as BM25;
3. a hybrid or late-interaction option such as BGE-M3 or GTE-ModernColBERT.

Add one domain specialist. Use reciprocal rank fusion and a reranker as separate experiment factors. This design shows whether gains come from the encoder, retrieval architecture, or reranker.

## Areas of Uncertainty

- **RTEB is beta.** Its finance, healthcare, legal, code, and multilingual boards are useful because they use newer retrieval data, but they are not yet as mature as long-standing MTEB suites.
- **Leaderboard contamination remains possible.** MTEB metadata lists training datasets, but not every model publishes complete data. A high score can reflect overlap or close synthetic derivatives.
- **Some live leaders are proprietary or non-commercial.** `ingot-8b-r3`, Jina v5 text models, zembed-1, and some API models need access or licence checks.
- **Model cards vary in quality.** New July 2026 models have limited independent production evidence.
- **Parameter count is incomplete cost information.** Late-interaction and sparse indexes can dominate storage. Long-context models can dominate activation memory.
- **Domain labels are broad.** “Medical”, “legal”, and “financial” contain many distinct tasks and jurisdictions.
- **Scientific retrieval lacks one clean aggregate.** Do not infer a universal scientific winner from a general benchmark average.
- **Multimodal suites differ.** MIEB, ViDoRe, MMEB, and vendor tests use different modalities, directions, and distractors.
- **A declared context limit is not effective context.** Position sensitivity and information dilution need separate tests.
- **Scores change.** The MTEB service is live. Record the date, benchmark name, model revision, and evaluation command with each decision.

## References

1. [MTEB Leaderboard](https://leaderboard.mteb.org/) — live benchmark explorer used for the 21 July 2026 snapshot.
2. [MTEB Models Catalogue](https://leaderboard.mteb.org/models) — parameter, architecture, context, language, and availability metadata.
3. [MTEB Benchmarks Catalogue](https://leaderboard.mteb.org/benchmarks) — benchmark task, modality, language, and domain definitions.
4. [MTEB source repository](https://github.com/embeddings-benchmark/mteb) — evaluation framework and task implementations.
5. [MTEB v2 introduction](https://huggingface.co/blog/isaacchung/mteb-v2) — updated input schemas and evaluation design.
6. [RTEB introduction](https://huggingface.co/blog/rteb) — fresh retrieval benchmark design and contamination motivation.
7. [Harrier OSS v1 27B model card](https://huggingface.co/microsoft/harrier-oss-v1-27b) — MIT licence, 27B metadata, languages, and context limit.
8. [Microsoft Harrier announcement](https://blogs.bing.com/search/April-2026/Microsoft-Open-Sources-Industry-Leading-Embedding-Model) — official family announcement.
9. [Jina Embeddings v5 text paper](https://arxiv.org/abs/2602.15547) — task-targeted distillation and adapters.
10. [Jina v5 text nano model card](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) — 212M metadata and CC-BY-NC-4.0 licence.
11. [Jasper Token Compression 600M model card](https://huggingface.co/infgrad/Jasper-Token-Compression-600M) — MIT licence, training code, data, and 32K context.
12. [Qwen3 Embedding repository](https://github.com/QwenLM/Qwen3-Embedding) — Apache 2.0 text embedding and reranking family.
13. [Dinghy Law 0.6B model card](https://huggingface.co/Hanno-Labs/dinghy-law-0.6b-v1) — legal specialist metadata and Apache 2.0 licence.
14. [Dinghy Law training report](https://huggingface.co/blog/Hanno-Labs/fine-tuning-a-legal-embedding-model) — training method and MTEB Law position.
15. [Euler Legal Embedding model card](https://huggingface.co/Mira190/Euler-Legal-Embedding-V1) — legal 8B model metadata.
16. [FinMTEB paper](https://aclanthology.org/2025.emnlp-main.179/) — 64-dataset finance benchmark and FinE5.
17. [FinMTEB repository](https://github.com/yixuantt/FinMTEB) — benchmark datasets and evaluation code.
18. [Nemotron-3 Embed 8B model card](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16) — July 2026 model metadata, languages, and training datasets.
19. [GTE-Qwen2-7B model card](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct) — medical-suite leader and long-context dense encoder.
20. [Towards Domain Specification of Embedding Models in Medicine](https://arxiv.org/abs/2507.19407) — limits of broad medical embedding evaluation.
21. [LateOn-Code model card](https://huggingface.co/lightonai/LateOn-Code) — Apache 2.0 late-interaction code model with open code and data.
22. [F2LLM-v2 paper](https://arxiv.org/abs/2603.19223) — multilingual and code-capable embedding family.
23. [F2LLM-v2 repository](https://github.com/codefuse-ai/CodeFuse-Embeddings/tree/main/F2LLM) — training and inference code.
24. [GTE-ModernColBERT model card](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) — Apache 2.0 long-document late-interaction model.
25. [LongEmbed paper](https://arxiv.org/abs/2404.12096) — long-context retrieval evaluation.
26. [BGE-M3 paper](https://arxiv.org/abs/2402.03216) — unified dense, sparse, and multi-vector retrieval.
27. [FlagEmbedding repository](https://github.com/FlagOpen/FlagEmbedding) — BGE model and reranker implementations.
28. [Jina Embeddings v5 Omni Small model card](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small) — multimodal model metadata.
29. [Qwen3-VL-Embedding paper](https://arxiv.org/abs/2601.04720) — open multimodal embedding and reranking family.
30. [Gemini Embedding 2 announcement](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/) — five-modality managed embedding model.
31. [Cohere Embed v4 documentation](https://docs.cohere.com/docs/cohere-embed) — multimodal input, dimensions, and output types.
32. [Sentence Transformers documentation](https://www.sbert.net/) — embedding and reranker evaluation and deployment.
33. [Hugging Face Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) — production embedding server.
34. [PyLate documentation](https://lightonai.github.io/pylate/) — late-interaction training and retrieval.
35. [AWS Bedrock model documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) — managed embedding availability.
36. [Azure AI model catalogue](https://ai.azure.com/explore/models) — managed and deployable model catalogue.
37. [Vertex AI embeddings documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings) — Google managed embedding service.
38. [IBM Granite Embedding Multilingual R2](https://arxiv.org/abs/2605.13521) — IBM's 2026 multilingual embedding family.
39. [OCI Generative AI documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm) — Oracle managed generative AI and embedding services.
40. [F2LLM-v2 training dataset](https://huggingface.co/datasets/codefuse-ai/F2LLM-v2) — public training data for the fully-open CodeFuse embedding family.
41. [Jasper training repository](https://github.com/DunZhang/Jasper-Token-Compression-Training) — public training code for Jasper Token Compression 600M.
42. [Jasper distillation dataset](https://huggingface.co/datasets/infgrad/jasper_text_distill_dataset) — public training data for Jasper 600M.
43. [LateOn-Code training dataset](https://huggingface.co/datasets/lightonai/nv-embed-supervised-distill-dedup-code) — public code-retrieval training data.
44. [OpenCLIP repository](https://github.com/mlfoundations/open_clip) — training code and model definitions for LAION CLIP models.
45. [LAION-2B dataset](https://laion.ai/blog/laion-5b/) — public dataset family used by the fully-open LAION CLIP entries.
46. [Nomic Embed Multimodal](https://huggingface.co/nomic-ai/nomic-embed-multimodal-7b) — open visual-document embedding model with published artefact links.
47. [ColModernVBERT](https://huggingface.co/ModernVBERT/colmodernvbert) — compact fully-open visual-document retriever.
