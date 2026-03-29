# Embedding-Based Pre-Screening for Chatbot Topic Relevance

| Field | Value |
|-------|-------|
| Created | 2026-03-26 |
| Last Updated | 2026-03-26 |
| Version | 1.0 |

---

- [Executive Summary](#executive-summary)
- [Why Pre-Screen with Embeddings?](#why-pre-screen-with-embeddings)
- [Core Techniques](#core-techniques)
  - [Zero-Shot Classification via Embeddings](#zero-shot-classification-via-embeddings)
  - [Two-Stage Architecture: Embedding + Classifier](#two-stage-architecture-embedding--classifier)
  - [Canonical Form Matching (NeMo Guardrails)](#canonical-form-matching-nemo-guardrails)
  - [Hybrid Zero-Shot Topic Assignment (BERTopic)](#hybrid-zero-shot-topic-assignment-bertopic)
- [Embedding Model Selection](#embedding-model-selection)
  - [Open-Source Models](#open-source-models)
  - [Commercial API Embeddings](#commercial-api-embeddings)
  - [Multilingual Models](#multilingual-models)
  - [Model Selection Decision Framework](#model-selection-decision-framework)
- [Similarity Threshold Calibration](#similarity-threshold-calibration)
  - [Why Fixed Thresholds Fail](#why-fixed-thresholds-fail)
  - [Cosine Adapter: Query-Dependent Mapping](#cosine-adapter-query-dependent-mapping)
  - [Calibration Best Practices](#calibration-best-practices)
- [Production Architectures](#production-architectures)
  - [Three-Tier Guardrail Architecture](#three-tier-guardrail-architecture)
  - [Vector Database Integration](#vector-database-integration)
  - [Latency and Resource Considerations](#latency-and-resource-considerations)
  - [False Positive Compounding](#false-positive-compounding)
  - [Managed Service Options](#managed-service-options)
- [Adversarial Robustness](#adversarial-robustness)
- [Comparison with Alternative Approaches](#comparison-with-alternative-approaches)
- [Implementation Guidance](#implementation-guidance)
- [Areas of Uncertainty](#areas-of-uncertainty)
- [References](#references)

## Executive Summary

Embedding-based pre-screening uses lightweight vector similarity to determine whether a user's chatbot query falls within the chatbot's intended topic scope *before* the query reaches the primary LLM. This approach sits in the 20-100ms latency tier — orders of magnitude faster than LLM-based guardrails (500ms-8s) — making it practical as a first-pass filter on every request.

The core idea is straightforward: embed the user's query, compare it against reference embeddings that represent the chatbot's allowed topics, and reject or escalate queries that fall below a similarity threshold. Variations of this pattern are used in production by NVIDIA NeMo Guardrails, BERTopic's zero-shot topic modelling, and custom two-stage architectures combining embeddings with lightweight classifiers (SVM, Random Forest, XGBoost).

Key findings from this research:

- **Two-stage architectures** (embedding model + lightweight classifier) achieve 85-89% accuracy on safety/topic classification at ~50ms latency, compared to 500ms-8s for LLM judges — verified across 4+ independent sources.
- **Zero-shot classification** requires no training data: embed topic label descriptions, compute cosine similarity with the input, and assign the highest-similarity label. Effective for rapid prototyping and low-resource scenarios.
- **Threshold calibration is critical**: cosine similarity thresholds are model-dependent and not transferable between embedding models. Raw scores from contrastive-trained models are not directly interpretable — query-dependent mapping functions significantly improve precision.
- **Open-weight models now match commercial APIs** on MTEB benchmarks (March 2026), with NV-Embed-v2 and Qwen3-Embedding-8B leading. For latency-critical filtering, MiniLM-L6-v2 (22M params, ~15ms/1K tokens) remains the pragmatic choice.
- **Domain-specific fine-tuning** almost always outperforms general-purpose models for specialised topic classification, though zero-shot approaches provide a strong baseline.

## Why Pre-Screen with Embeddings?

LLM-based chatbots are expensive to run and vulnerable to topic hijacking — users asking about subjects outside the chatbot's intended scope, whether through curiosity, social engineering, or adversarial attack. Every off-topic query that reaches the primary LLM wastes compute, risks generating inappropriate responses, and may expose the model to prompt injection vectors.

Embedding-based pre-screening addresses this by providing a fast, cheap first gate:

| Approach | Latency | GPU Required | Accuracy Range |
|----------|---------|-------------|----------------|
| Rule-based (regex/keywords) | <10ms | No | High precision, low recall |
| Embedding similarity | 20-100ms | Optional (CPU viable) | 77-89% |
| LLM judge (Llama Guard) | 500ms-8s | A100/H100 | 85-95% |
| Full LLM classification | 1-10s | Yes | 90-98% |

The embedding approach occupies a sweet spot: fast enough to run on every request, accurate enough to filter the majority of off-topic queries, and cheap enough to deploy without dedicated GPU infrastructure for smaller models.

## Core Techniques

### Zero-Shot Classification via Embeddings

The simplest embedding-based pre-screening technique requires no training data at all. The approach, documented in the OpenAI Cookbook and implemented in BERTopic, works as follows:

1. **Define topic labels** as natural language descriptions (e.g., "customer support for billing enquiries", "technical troubleshooting for network issues")
2. **Embed each label** using the same embedding model used for user queries
3. **For each incoming query**, embed it and compute cosine similarity against all label embeddings
4. **Assign the highest-similarity label** as the predicted topic; if the highest score falls below a threshold, flag as off-topic

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Define allowed topics
topic_labels = [
    "billing and payment enquiries",
    "technical support for software issues",
    "account management and settings",
]
off_topic_label = "unrelated or off-topic question"

# Pre-compute label embeddings (do once at startup)
all_labels = topic_labels + [off_topic_label]
label_embeddings = model.encode(all_labels)

def classify_query(query: str, threshold: float = 0.35) -> tuple[str, float]:
    query_embedding = model.encode(query)
    similarities = np.dot(label_embeddings, query_embedding) / (
        np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score < threshold:
        return "off-topic", best_score
    return all_labels[best_idx], best_score
```

**Strengths:** No training data needed; easy to add/remove topics by editing label descriptions; fast iteration during development.

**Limitations:** Lower accuracy than trained classifiers; sensitive to how labels are worded; threshold selection is non-trivial (see [Similarity Threshold Calibration](#similarity-threshold-calibration)).

### Two-Stage Architecture: Embedding + Classifier

For higher accuracy, a two-stage architecture trains a lightweight classifier on top of embedding vectors. This approach was validated in a peer-reviewed study at COLING 2025 (Zheng et al.), which fine-tuned Sentence-BERT (distilbert-base-uncased, 67M parameters) and trained an SVM/neural network classifier on the resulting 768-dimensional embeddings.

**Architecture:**

```
User Query → Embedding Model → 768-d vector → Classifier (SVM/RF/XGBoost) → On-topic / Off-topic
```

**Results from Zheng et al. (COLING 2025):**

| Approach | Accuracy | F1 | AUPRC | Latency |
|----------|----------|-----|-------|---------|
| McEMcC (multi-class embedding + multi-class classifier) | 88.83% | 0.89 | 0.946 | ~50ms |
| LlamaGuard (7B params) | ~87% | — | — | 500ms-8s |

The best configuration used:
- **Triplet-soft loss** for embedding fine-tuning (better than standard triplet or contrastive loss)
- **Multi-class framing** for both embedding training and classification (outperforms binary safe/unsafe)
- **Mean pooling** over token embeddings to produce the 768-d vector
- **SVM or shallow neural network** as the downstream classifier

The same pattern — embeddings fed to Random Forest or XGBoost — has also been shown effective for prompt injection detection (Ayub & Majumdar, 2024), where embedding-based classifiers outperformed encoder-only neural networks.

### Canonical Form Matching (NeMo Guardrails)

NVIDIA's NeMo Guardrails uses a more structured embedding-based approach. Rather than comparing against label descriptions, it indexes *canonical user message forms* — representative example utterances for each allowed intent — in a vector database (Annoy or FAISS).

**How it works:**

1. Developers define canonical forms in Colang DSL:
   ```colang
   define user ask about billing
     "What's my current balance?"
     "How do I pay my invoice?"
     "When is my next payment due?"
   ```

2. At startup, all canonical form examples are embedded (default model: all-MiniLM-L6-v2) and indexed in a vector store.

3. When a user message arrives:
   - The message is embedded
   - KNN search retrieves the **top-5 most similar** canonical forms
   - These 5 examples are passed to the LLM as few-shot context for intent classification
   - The LLM classifies the user intent based on the retrieved examples

**Performance:** 77-83% accuracy on the Banking NLU dataset (77 intents). Requires at least **k=3 example utterances per canonical form** for reliable performance.

**Key insight:** NeMo uses embeddings for *retrieval*, not direct classification. The embedding search narrows the candidate space; the LLM makes the final classification decision. This hybrid approach is more robust than pure embedding similarity but adds one LLM call of latency.

### Hybrid Zero-Shot Topic Assignment (BERTopic)

BERTopic offers a hybrid approach that combines zero-shot embedding classification with unsupervised clustering:

1. **Zero-shot phase:** Embed predefined topic labels and compute cosine similarity with each document embedding. If similarity exceeds a user-defined threshold, assign the zero-shot topic.
2. **Clustering fallback:** Documents that don't match any predefined topic above the threshold enter the standard BERTopic clustering pipeline (UMAP + HDBSCAN).

This is particularly useful for chatbot pre-screening because it handles both **known topics** (your chatbot's domain) and **unknown topics** (novel off-topic queries that don't match any predefined pattern). The clustering step can surface emerging categories of off-topic queries for future refinement.

## Embedding Model Selection

### Open-Source Models

The following table summarises key open-source embedding models suitable for topic pre-screening, ordered by the speed/accuracy trade-off:

| Model | Params | Embedding Dim | Latency (1K tokens) | BEIR Accuracy | GPU Memory | Licence |
|-------|--------|---------------|---------------------|---------------|------------|---------|
| all-MiniLM-L6-v2 | 22M | 384 | ~15ms | 78.1% | ~1.2 GB | Apache 2.0 |
| E5-Base-v2 | 110M | 768 | ~20ms | 83.5% | ~2.0 GB | MIT |
| BGE-Base-v1.5 | 110M | 768 | ~23ms | 84.7% | ~2.1 GB | MIT |
| Nomic Embed v1 | 137M | 768 | ~42ms | 86.2% | ~4.8 GB | Apache 2.0 |
| stella_en_1.5B_v5 | 1.5B | 1024 | — | — | — | MIT |
| NV-Embed-v2 | 7B | 4096 | — | MTEB 72.31 | — | CC-BY-NC-4.0 |
| Qwen3-Embedding-8B | 8B | — | — | MTEB 70.58 | — | Apache 2.0 |

**Recommendations:**

- **Speed-critical / high-volume:** all-MiniLM-L6-v2. This is also NeMo Guardrails' default. At 22M parameters it runs efficiently on CPU. The 5-8% accuracy gap vs larger models is acceptable when used as a first-pass filter with LLM fallback.
- **Balanced:** BGE-Base-v1.5 or E5-Base-v2. These 110M-parameter models fit comfortably on a T4 GPU and provide meaningfully better classification margins.
- **Maximum accuracy:** Nomic Embed v1 or stella_en_1.5B_v5. Use when topic boundaries are ambiguous and classification margins matter.

### Commercial API Embeddings

| Provider | Model | MTEB Avg | Price (per M tokens) |
|----------|-------|----------|---------------------|
| Google | Gemini Embedding 001 | 68.32 | — |
| OpenAI | text-embedding-3-large | 64.60 | $0.13 |
| OpenAI | text-embedding-3-small | — | $0.02 |

**Managed service equivalents:**
- **AWS:** Amazon Bedrock Embeddings (Titan Embeddings, Cohere Embed)
- **Azure:** Azure OpenAI Embeddings (text-embedding-3-large/small)
- **GCP:** Vertex AI Text Embeddings (Gemini Embedding 001, textembedding-gecko)

Commercial APIs add network latency (typically 50-200ms round-trip) which may negate the speed advantage of embedding-based pre-screening. For latency-sensitive filtering, self-hosted models are strongly preferred.

### Multilingual Models

For chatbots serving non-English users, multilingual embedding models are available:

- **Multilingual E5** (Microsoft Research): Available in small/base/large sizes. Follows the English E5 training recipe adapted for multilingual corpus. Open-source, widely adopted.
- **BGE-M3** (BAAI): Supports 100+ languages with multi-granularity embeddings (dense, sparse, multi-vector).
- **Qwen3-Embedding** (Alibaba): Available in 0.6B, 4B, and 8B sizes with multilingual support.

**Caveat:** Most benchmarking evidence for embedding pre-screening is English-only. Threshold calibration and accuracy claims may not transfer directly to other languages — independent evaluation on target-language data is essential.

### Model Selection Decision Framework

```
Start
  │
  ├── Need <20ms latency? → MiniLM-L6-v2 (CPU-viable)
  │
  ├── Need multilingual? → Multilingual E5 or BGE-M3
  │
  ├── Have GPU budget?
  │     ├── T4 / consumer GPU → BGE-Base-v1.5 or E5-Base-v2
  │     └── A10+ / production GPU → Nomic Embed v1 or stella_en_1.5B_v5
  │
  └── Prefer managed service? → Cloud provider embedding API
        (but factor in network latency)
```

## Similarity Threshold Calibration

### Why Fixed Thresholds Fail

A common mistake is choosing a single cosine similarity threshold (e.g., 0.7) and applying it universally. This fails for two reasons:

1. **Thresholds are model-dependent.** A threshold of 0.79 that works well for OpenAI's ada-002 model produces completely different behaviour with text-embedding-3-large. Each model has its own score distribution characteristics determined by training data, loss function, and dimensionality.

2. **Raw cosine scores are not directly interpretable.** Models trained with contrastive or ranking losses optimise for *relative ordering* of scores, not absolute magnitudes. A cosine similarity of 0.82 does not inherently mean "82% relevant" — it is only meaningful relative to other scores from the same model on similar queries.

### Cosine Adapter: Query-Dependent Mapping

The Cosine Adapter (Rossi et al., CIKM 2024), developed and production-validated at Walmart, addresses this by learning a **query-dependent mapping function** that transforms raw cosine scores into interpretable relevance scores:

```
raw_cosine_score → Cosine Adapter → interpretable_score → global_threshold → accept/reject
```

Key findings from the paper:
- A global threshold on *adapted* scores significantly increases precision with only a small recall trade-off
- The mapping function accounts for the fact that different queries have inherently different score distributions
- Validated via A/B testing on production search traffic at Walmart

### Calibration Best Practices

For production chatbot topic filtering, recommended calibration steps:

1. **Collect a labelled evaluation set** — at minimum 200-500 queries with on-topic/off-topic labels, including edge cases and adversarial examples.

2. **Compute similarity scores** for all evaluation queries against your topic reference embeddings.

3. **Plot ROC and precision-recall curves** to visualise the trade-off:
   - **High-precision regime** (threshold ≈ 0.8+): Few false positives but may reject legitimate on-topic queries
   - **High-recall regime** (threshold ≈ 0.5-0.6): Catches most on-topic queries but lets some off-topic through
   - For pre-screening, **favour recall** (don't reject legitimate users) and rely on downstream LLM guardrails for precision

4. **Use the threshold that maximises F1** on your evaluation set as a starting point, then adjust based on business requirements.

5. **Monitor and recalibrate** in production. Topic distributions drift, new topics emerge, and adversarial patterns evolve. Set up logging to track similarity score distributions and periodically re-evaluate the threshold.

6. **Consider per-topic thresholds** if your chatbot covers topics with very different semantic densities. Technical topics may cluster tightly (high similarity scores between related queries) while conversational topics may be more diffuse.

## Production Architectures

### Three-Tier Guardrail Architecture

Production chatbot systems typically implement guardrails in a layered architecture with early exit at each tier:

```
┌──────────────────────────────────────────────────┐
│                  User Request                     │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│         Tier 1: Rule-Based (<10ms)               │
│  • Regex patterns (API keys, credit cards)       │
│  • Keyword blocklists                            │
│  • Format validation                             │
│  • Known injection patterns                      │
└────────────────────────┬─────────────────────────┘
                         ▼ (if passed)
┌──────────────────────────────────────────────────┐
│      Tier 2: ML Classifiers (20-100ms)           │
│  • Embedding similarity for topic relevance  ◄── │
│  • Toxicity detection (fine-tuned BERT)           │
│  • PII detection (NER models)                     │
│  • Prompt injection classifier                    │
└────────────────────────┬─────────────────────────┘
                         ▼ (if uncertain or flagged)
┌──────────────────────────────────────────────────┐
│       Tier 3: LLM Judge (500ms-8s)               │
│  • Llama Guard content classification            │
│  • LLM-as-judge for nuanced evaluation           │
│  • Context-dependent policy decisions            │
└──────────────────────────────────────────────────┘
```

**Embedding-based topic relevance filtering sits in Tier 2.** It runs after cheap rule-based checks but before expensive LLM evaluation. The early-exit design means that clearly on-topic queries pass through in <100ms, while only ambiguous or flagged queries incur the cost of LLM-based judgement.

### Vector Database Integration

For canonical form matching (NeMo-style), a vector database stores and indexes reference embeddings:

| Vector Store | Type | Latency (KNN) | Notes |
|-------------|------|---------------|-------|
| Annoy | Library (in-process) | <1ms | NeMo default; read-only after build; memory-mapped |
| FAISS | Library (in-process) | <1ms | NeMo supported; GPU-accelerated option |
| Qdrant | Standalone server | 1-5ms | Filtering, payload storage, real-time updates |
| Weaviate | Standalone server | 1-5ms | Hybrid search, multi-tenancy |
| pgvector | PostgreSQL extension | 5-20ms | If already using PostgreSQL; simpler ops |

For pre-screening with a small number of topic reference embeddings (tens to hundreds), in-process libraries like Annoy or FAISS are sufficient and avoid network overhead. Standalone vector databases become valuable when managing thousands of canonical forms across multiple chatbot instances.

**Managed service equivalents:**
- **AWS:** Amazon OpenSearch Service (with k-NN plugin), Amazon MemoryDB
- **Azure:** Azure AI Search (vector search), Azure Cosmos DB (vector indexing)
- **GCP:** Vertex AI Vector Search, AlloyDB (with pgvector)

### Latency and Resource Considerations

| Component | CPU Latency | GPU Latency | Notes |
|-----------|-------------|-------------|-------|
| MiniLM-L6-v2 encoding | 15-30ms | 5-10ms | Viable on CPU for moderate traffic |
| BGE-Base-v1.5 encoding | 50-100ms | 15-25ms | GPU recommended for >100 QPS |
| Annoy/FAISS KNN lookup | <1ms | <1ms | Negligible vs encoding |
| SVM/RF classification | <1ms | — | CPU only; negligible |
| **Total Tier 2 pipeline** | **20-100ms** | **10-30ms** | **Including all components** |

NeMo Guardrails' embedding-based routing runs on T4 GPUs (entry-level), while LLM guardrails like Llama Guard require A100 or H100 GPUs — a significant cost differential for self-hosted deployments.

### False Positive Compounding

When stacking multiple guardrails (topic filter + toxicity check + PII detection + injection detection), false positive rates compound:

| Number of Guards | Per-Guard Accuracy | System False Positive Rate |
|-----------------|-------------------|---------------------------|
| 1 | 90% | 10% |
| 3 | 90% | 27% |
| 5 | 90% | 41% |
| 5 | 95% | 23% |
| 5 | 99% | 5% |

This means each individual guard needs **95%+ accuracy** to keep the overall system false positive rate manageable with 5 guards. For topic pre-screening specifically, this favours a high-recall, lower-precision configuration — it's better to let borderline queries through to the LLM than to frustrate users with false rejections.

### Managed Service Options

For teams that prefer not to self-host embedding models and vector stores:

- **AWS Amazon Bedrock Guardrails:** Built-in topic filtering via denied topics configuration. Uses embedding similarity internally. No model management required.
- **Azure AI Content Safety:** Custom categories with few-shot examples. Integrates with Azure OpenAI Service.
- **GCP Vertex AI:** Custom content classifiers via AutoML or embedding similarity on Vertex AI Vector Search.

These managed options trade customisation and latency control for operational simplicity.

## Adversarial Robustness

Embedding-based classifiers are not immune to adversarial attacks. Key concerns:

**Paraphrasing attacks:** Adversaries can rephrase off-topic queries to increase semantic similarity with on-topic reference embeddings. For example, framing a financial advice question as a "customer support enquiry about account features" may bypass a banking chatbot's topic filter.

**Encoding tricks:** Character substitutions, Unicode homoglyphs, and token-level perturbations can shift embedding vectors in ways that evade similarity thresholds while preserving human-readable meaning.

**Prompt injection via embedding space:** Research by Ayub & Majumdar (2024) found that embedding-based classifiers (Random Forest and XGBoost on embeddings) can detect prompt injection attacks, outperforming encoder-only neural networks. This suggests the embedding+classifier approach has some inherent robustness to injection patterns, though it should not be the sole defence.

**Mitigation strategies:**

1. **Layer with rule-based checks** (Tier 1) that catch common injection patterns before the embedding stage
2. **Use a trained classifier** rather than raw cosine threshold — classifiers learn decision boundaries that are harder to evade than a single threshold
3. **Include adversarial examples** in your training/evaluation set to harden the classifier
4. **Add an LLM fallback** (Tier 3) for queries that pass Tier 2 with marginal confidence scores
5. **Monitor embedding score distributions** for anomalous patterns that suggest evasion attempts

## Comparison with Alternative Approaches

| Approach | Accuracy | Latency | Training Data | Maintenance |
|----------|----------|---------|---------------|-------------|
| **Embedding cosine similarity (zero-shot)** | 75-83% | 20-50ms | None | Low — edit labels |
| **Embedding + classifier (trained)** | 85-89% | 20-100ms | 500-5000 labelled examples | Medium — retrain on drift |
| **TF-IDF + SVM** | 50-72% | <10ms | 500+ labelled examples | Medium |
| **Fine-tuned BERT classifier** | 85-92% | 50-200ms | 1000+ labelled examples | High — retrain whole model |
| **LLM-as-judge** | 85-95% | 500ms-8s | None (prompt-based) | Low — edit prompt |
| **Keyword/regex rules** | High precision, low recall | <1ms | None | High — manual rule creation |

Key comparisons:

- **TF-IDF vs embeddings:** GloVe/transformer embeddings consistently outperform TF-IDF for text classification. In one study, SVM with GloVe embeddings achieved 72.1% accuracy vs 50.1% with TF-IDF on the same task (Petridis, 2024). Embeddings capture semantic meaning that bag-of-words approaches miss.
- **Embeddings vs fine-tuned BERT:** Fine-tuned BERT achieves the highest accuracy (85-92%) but requires more training data and is harder to update. The embedding+classifier approach is nearly as accurate while being more modular — swap the embedding model without retraining the classifier.
- **Embeddings vs LLM judge:** LLM judges are more accurate for nuanced cases but 10-100x slower and more expensive. The recommended approach is to use embeddings as a fast filter and reserve LLM judgement for uncertain cases.

## Implementation Guidance

**Starting point for a new chatbot:**

1. **Begin with zero-shot classification** using all-MiniLM-L6-v2. Define your topic labels as descriptive sentences. This gives you a working filter in hours, not weeks.

2. **Log all queries with their similarity scores** in production. This builds your training set for the next stage.

3. **After collecting 500+ labelled examples**, train a two-stage classifier (embedding + SVM/Random Forest). This typically improves accuracy by 5-10% over zero-shot.

4. **Calibrate thresholds** using ROC/precision-recall curves on a held-out evaluation set. Set the threshold to favour recall (don't reject legitimate users).

5. **Deploy in the three-tier architecture** with rule-based checks before and LLM fallback after the embedding filter.

6. **Monitor and iterate.** Track false positive/negative rates, recalibrate thresholds quarterly, and fine-tune embeddings if you have domain-specific data.

**What not to do:**

- Don't copy cosine similarity thresholds from blog posts or other projects — they are model-dependent and dataset-dependent.
- Don't rely on embedding similarity as your sole guardrail — always layer with other defences.
- Don't use a large embedding model (7B+) for pre-screening — it defeats the latency purpose. Save large models for the primary LLM task.
- Don't skip adversarial testing — include paraphrased and injection-style queries in your evaluation set.

## Areas of Uncertainty

- **Chatbot-specific benchmarks are sparse.** Most evidence comes from adjacent domains (safety classification, content moderation, search relevance). There is no standardised benchmark for chatbot topic pre-screening specifically.
- **Adversarial robustness is understudied** in the context of topic filtering. Most adversarial research focuses on safety/toxicity classifiers, not topic relevance.
- **Multilingual performance** of embedding-based pre-screening has limited benchmarking evidence. Threshold calibration and accuracy claims may not transfer across languages.
- **Long-term drift** in user query distributions is not well-studied. How often thresholds need recalibration in practice is an open question.
- **The 3000x latency advantage** claimed by Zheng et al. (COLING 2025) for BERT-based guardrails vs LlamaGuard is likely overstated. Independent benchmarks place Llama Guard latency at 500ms-8s (not 163s), suggesting the real advantage is 10-100x — still significant, but not three orders of magnitude.

## References

1. Zheng, S. et al. (2025). [Lightweight Safety Guardrails Using Fine-tuned BERT Embeddings](https://arxiv.org/abs/2411.14398). COLING 2025 Industry Track.
2. Rebedea, T. et al. (2023). [NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails](https://arxiv.org/abs/2310.10501). EMNLP 2023 Demo Track.
3. Rossi, A. et al. (2024). [Relevance Filtering for Embedding-based Retrieval (Cosine Adapter)](https://arxiv.org/abs/2408.04887). CIKM 2024.
4. Dong, Y. et al. (2024). [Building Guardrails for Large Language Models](https://arxiv.org/abs/2402.01822). arXiv preprint.
5. Ayub, M. & Majumdar, S. (2024). [Embedding-based classifiers can detect prompt injection attacks](https://arxiv.org/abs/2410.22284). arXiv preprint.
6. Petridis, C. (2024). [Text Classification: Neural Networks VS Machine Learning Models VS Pre-trained Models](https://arxiv.org/abs/2412.21022). arXiv preprint.
7. Wang, L. et al. (2024). [Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/abs/2402.05672). arXiv preprint.
8. NVIDIA. [Colang Architecture Guide — NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/latest/reference/colang-architecture-guide.html). Official documentation.
9. OpenAI. [Zero-shot classification with embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Zero-shot_classification_with_embeddings.ipynb). OpenAI Cookbook.
10. Grootendorst, M. [Zero-shot Topic Modeling — BERTopic](https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html). BERTopic documentation.
11. Jalan, A. (2026). [Production LLM Guardrails: NeMo, Guardrails AI, Llama Guard Compared](https://blog.premai.io/production-llm-guardrails-nemo-guardrails-ai-llama-guard-compared/). PremAI Blog.
12. [Embedding Model Leaderboard: MTEB Rankings March 2026](https://awesomeagents.ai/leaderboards/embedding-model-leaderboard-mteb-march-2026/). AwesomeAgents.
13. [Top Embedding Models on the MTEB Leaderboard](https://modal.com/blog/mteb-leaderboard-article). Modal Blog, October 2025.
14. [Best Open-Source Embedding Models Benchmarked and Ranked](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/). Supermemory Blog.
15. [Rule of Thumb Cosine Similarity Thresholds?](https://community.openai.com/t/rule-of-thumb-cosine-similarity-thresholds/693670). OpenAI Developer Community.
