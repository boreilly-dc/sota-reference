# Benchmarks for Testing RAG-Powered Chatbots

| Field | Value |
|-------|-------|
| Created | 2026-03-20 |
| Last Updated | 2026-03-20 |
| Version | 1.0 |

---

- [1. Why RAG Evaluation is Hard](#1-why-rag-evaluation-is-hard)
- [2. Evaluation Taxonomy](#2-evaluation-taxonomy)
- [3. Core Metrics](#3-core-metrics)
- [4. Purpose-Built RAG Benchmarks](#4-purpose-built-rag-benchmarks)
- [5. Retrieval-Stage Benchmarks](#5-retrieval-stage-benchmarks)
- [6. QA Datasets Commonly Used for RAG](#6-qa-datasets-commonly-used-for-rag)
- [7. Multi-Turn and Conversational Benchmarks](#7-multi-turn-and-conversational-benchmarks)
- [8. Domain-Specific Benchmarks](#8-domain-specific-benchmarks)
- [9. Open-Source Evaluation Frameworks](#9-open-source-evaluation-frameworks)
- [10. Managed RAG Evaluation Services](#10-managed-rag-evaluation-services)
- [11. LLM-as-a-Judge: Caveats](#11-llm-as-a-judge-caveats)
- [12. Choosing a Benchmark Strategy](#12-choosing-a-benchmark-strategy)
- [13. Limitations of Current Benchmarks](#13-limitations-of-current-benchmarks)
- [References](#references)

A practical reference for engineers selecting benchmarks, metrics, and tooling to evaluate retrieval-augmented generation (RAG) chatbots. Covers academic benchmarks, open-source frameworks, and managed cloud services.

---

## 1. Why RAG Evaluation is Hard

RAG systems combine two fundamentally different components — a **retriever** and a **generator** — each with its own failure modes. Evaluating them requires assessing:

- **Retrieval quality**: Did the retriever find relevant documents?
- **Generation quality**: Did the generator use the retrieved context faithfully?
- **End-to-end correctness**: Is the final answer accurate and well-grounded?

These components interact: a perfect generator cannot compensate for poor retrieval, and a perfect retriever is wasted if the generator hallucinates. Most production RAG failures originate in the retrieval stage, making independent retrieval evaluation essential.

---

## 2. Evaluation Taxonomy

RAG evaluation approaches split into two broad categories, each with distinct trade-offs:

### Reference-required evaluation

Benchmarks that compare system output against ground-truth answers. Examples: RGB, CRUD-RAG, MultiHop-RAG.

- **Strengths**: Objective, reproducible, language-agnostic metrics (accuracy, F1, ROUGE, BLEU).
- **Weaknesses**: Expensive to create, static (answers may become outdated), limited to the tasks in the dataset.

### Reference-free evaluation

Frameworks that use an LLM judge to assess output quality without ground-truth answers. Examples: RAGAS, ARES, TruLens RAG Triad.

- **Strengths**: No labelled data needed, can evaluate any query, scalable.
- **Weaknesses**: LLM judges exhibit known biases (see [Section 11](#11-llm-as-a-judge-caveats)), may be unreliable when retrieved context is low quality.

A robust evaluation strategy uses **both** approaches: reference-required benchmarks for baseline measurement, reference-free metrics for continuous production monitoring.

---

## 3. Core Metrics

The following metrics are consistently cited across academic literature, open-source frameworks (RAGAS, DeepEval), and managed services (AWS Bedrock, GCP Vertex AI):

### Retrieval metrics

| Metric | What it measures |
|--------|-----------------|
| **Context Precision** | Proportion of retrieved documents that are relevant to the query |
| **Context Recall** | Proportion of relevant documents that were successfully retrieved |
| **nDCG@K** | Quality of document ranking (standard IR metric) |
| **MRR** (Mean Reciprocal Rank) | How high the first relevant document appears |
| **MAP** (Mean Average Precision) | Average precision across all relevant documents |

### Generation metrics

| Metric | What it measures |
|--------|-----------------|
| **Faithfulness** | Whether the response is grounded in retrieved documents (not hallucinated) |
| **Answer Relevancy** | Whether the response addresses the user's query |
| **Answer Correctness** | Factual accuracy compared to ground truth |
| **Citation Precision** | Whether cited sources actually support the claims made |
| **Citation Coverage** | Whether all claims in the response are backed by citations |

### Additional metrics

| Metric | What it measures |
|--------|-----------------|
| **Noise Robustness** | Performance when retrieved documents contain irrelevant content |
| **Negative Rejection** | Ability to refuse answering when no relevant context exists |
| **Latency** | End-to-end response time |
| **Cost** | Token/API cost per query |

---

## 4. Purpose-Built RAG Benchmarks

### RGB (Retrieval-Augmented Generation Benchmark)

- **Paper**: Chen et al., AAAI 2024
- **What it tests**: Four fundamental RAG abilities — noise robustness, negative rejection, information integration, counterfactual robustness.
- **Size**: 600 base questions + 200 additional per ability, in English and Chinese.
- **Key finding**: LLMs show decent noise robustness but struggle significantly with negative rejection (failing to say "I don't know") and tend to trust retrieved information over their own knowledge even when warned about factual errors.
- **Code**: [github.com/chen700564/RGB](https://github.com/chen700564/RGB)

### CRUD-RAG

- **Paper**: Lyu et al., ACM J. ACM 2024
- **What it tests**: Four categories of RAG interaction — Create (text continuation), Read (QA), Update (error correction), Delete (summarisation).
- **Why it matters**: Most RAG benchmarks focus exclusively on QA. CRUD-RAG extends evaluation to content generation, error correction, and summarisation tasks.
- **Metrics**: ROUGE, BLEU, RAGQuestEval.
- **Also tests**: Effects of chunk size, embedding model, retrieval algorithm, and LLM choice on overall performance.
- **Code**: [github.com/IAAR-Shanghai/CRUD_RAG](https://github.com/IAAR-Shanghai/CRUD_RAG)

### MultiHop-RAG

- **Paper**: Tang & Yang, 2024
- **What it tests**: Multi-hop reasoning requiring retrieval from multiple documents to answer complex questions.
- **Metrics**: MAP, MRR, Hit@K for retrieval; LLM-as-judge for generation.

### RECALL

- **Paper**: Liu et al., 2023
- **What it tests**: Response quality and robustness using EventKG and UJ datasets.
- **Metrics**: BLEU, ROUGE-L.

### FeB4RAG

- **Paper**: Wang et al., 2024
- **What it tests**: Four dimensions — consistency, correctness, clarity, coverage.
- **Evaluation**: Human evaluation alongside automated metrics.

### DomainRAG

- **Paper**: Gu et al., 2024
- **What it tests**: Six abilities — conversational, structural information, faithfulness, denoising, time-sensitive problem solving, multi-document understanding.

### ReEval

- **Paper**: Shen et al., 2024
- **What it tests**: Hallucination detection using prompt chaining to create dynamic test cases. Uses RealTimeQA and Natural Questions.

---

## 5. Retrieval-Stage Benchmarks

Evaluating retrieval quality independently is critical. Most production RAG failures stem from the retriever, not the generator.

### BEIR (Benchmarking Information Retrieval)

The standard benchmark for evaluating retrieval models in zero-shot and domain-agnostic settings.

- **Datasets**: 18 diverse datasets including MS MARCO, Natural Questions, TREC-COVID, SciFact, and domain-specific corpora (medical, legal, financial).
- **Primary metric**: nDCG@10.
- **Why it matters**: Tests whether retrieval models generalise across domains — essential for RAG chatbots that handle diverse queries.
- **Code**: [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir)

### MTEB (Massive Text Embedding Benchmark)

Broader than BEIR, MTEB evaluates embedding models across retrieval, classification, clustering, and semantic similarity tasks. The retrieval subset is directly relevant to RAG.

- **Leaderboard**: [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

### IRSC

- **Paper**: 2025
- **Why it matters**: Addresses the gap between general embedding benchmarks (MTEB) and the specialised requirements of embeddings in RAG pipelines.

---

## 6. QA Datasets Commonly Used for RAG

These datasets were not designed specifically for RAG but are widely used to test RAG systems:

| Dataset | Type | Size | Notes |
|---------|------|------|-------|
| **Natural Questions** (Google) | Open-domain QA | 300K+ | Real Google search queries with Wikipedia answers |
| **HotpotQA** | Multi-hop QA | 113K | Requires reasoning across multiple documents |
| **TriviaQA** | Open-domain QA | 95K | Trivia questions with evidence documents |
| **MS MARCO** | Passage ranking + QA | 1M+ | Real Bing queries, large-scale retrieval benchmark |
| **SQuAD 2.0** | Reading comprehension | 150K | Includes unanswerable questions |
| **FEVER** | Fact verification | 185K | Classifies claims as supported/refuted/not enough info |

---

## 7. Multi-Turn and Conversational Benchmarks

Single-turn QA benchmarks miss the challenges of real chatbot conversations: coreference resolution, topic drift, context accumulation, and follow-up questions.

### MTRAG

- **Paper**: Katsis et al., IBM Research, TACL 2025
- **What it tests**: End-to-end multi-turn RAG pipeline evaluation with human-generated conversations.
- **Size**: 110 conversations averaging 7.7 turns each (842 tasks total) across 4 domains.
- **Key findings**: Even state-of-the-art systems struggle with later turns, unanswerable questions, non-standalone questions, and cross-domain conversations.
- **Code**: [github.com/ibm/mt-rag-benchmark](https://github.com/ibm/mt-rag-benchmark)

### ChatRAG Bench

- **Source**: NVIDIA, 2024
- **What it tests**: Conversational RAG with 10 datasets covering various domains.

### Conversational QA datasets (repurposed for RAG)

| Dataset | Focus |
|---------|-------|
| **QuAC** | Multi-turn conversational QA with information-seeking dialogues |
| **CoQA** | Conversational QA requiring understanding of conversation history |
| **OR-QuAC** | Open-retrieval conversational QA combining retrieval with QuAC |

---

## 8. Domain-Specific Benchmarks

Generic RAG benchmarks are insufficient for specialised domains. Optimal RAG configurations vary significantly by domain.

### Medical

- **MedRAG** (Xiong et al., ACL Findings 2024): Systematic toolkit covering 5 medical corpora, 4 retrievers, 6 LLMs. Demonstrates that RAG improves medical QA but optimal retriever-LLM pairings differ by task.
- **MIRAGE**: Medical information retrieval benchmark used by MedRAG.
- **Code**: [github.com/Teddy-XiongGZ/MedRAG](https://github.com/Teddy-XiongGZ/MedRAG)

### Multi-domain

- **RAGEval**: Framework for generating domain-specific evaluation datasets for finance, healthcare, and legal sectors.
- **CDQA**: Cross-domain QA benchmark testing RAG across heterogeneous knowledge sources.

### Temporal knowledge

- **HoH Benchmark** (ACL 2025): Measures how outdated information affects RAG reliability — a critical concern for chatbots in fast-moving domains.

---

## 9. Open-Source Evaluation Frameworks

### RAGAS

The most widely adopted open-source RAG evaluation framework.

- **Metrics**: 35+ metrics covering context precision, context recall, faithfulness, answer relevancy, noise sensitivity, summarisation score, and more.
- **Approach**: Reference-free (LLM-as-judge) for most metrics; some metrics support ground-truth references.
- **Integrations**: LangChain, LlamaIndex, Haystack, Langfuse, Arize Phoenix.
- **Code**: [github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas)

### TruLens (Snowflake)

Pioneered the **RAG Triad** — three metrics that together assess RAG quality:

1. **Context relevance**: Are retrieved documents relevant to the query?
2. **Groundedness**: Is the response grounded in retrieved documents?
3. **Answer relevance**: Does the response address the query?

- **Approach**: LLM-as-judge with OpenTelemetry-based tracing.
- **Code**: [github.com/truera/trulens](https://github.com/truera/trulens)

### DeepEval (Confident AI)

- **Focus**: CI/CD integration for automated RAG testing in development pipelines.
- **Metrics**: Contextual precision, contextual recall, faithfulness, answer relevancy, hallucination detection.
- **Strengths**: GitHub Actions workflow support, regression testing, hyperparameter tracking (chunk size, top-K, embedding model).
- **Code**: [github.com/confident-ai/deepeval](https://github.com/confident-ai/deepeval)

### ARES

- **Paper**: Saad-Falcon et al., 2023
- **Approach**: Uses LLM judges with human preference data for context relevance, answer faithfulness, and answer relevance. Trains lightweight classifiers on LLM judgements for scalable evaluation.

### Arize Phoenix

- **Focus**: Open-source AI observability with RAG-specific tracing and retrieval analysis.
- **Features**: OpenTelemetry-based instrumentation, retrieval evals, embedding space analysis.
- **Code**: [github.com/Arize-ai/phoenix](https://github.com/Arize-ai/phoenix)

### Langfuse

- **Focus**: Open-source LLM observability with RAG-specific tracing.
- **Features**: Session-level analysis for multi-turn interactions, human annotation workflows, native LangChain/LlamaIndex support.
- **Code**: [github.com/langfuse/langfuse](https://github.com/langfuse/langfuse)

### Comparison

| Framework | Type | Core Strength | RAG Metrics | Production Monitoring |
|-----------|------|---------------|-------------|----------------------|
| **RAGAS** | Eval library | Broadest metric set | 35+ | No (pair with observability tool) |
| **TruLens** | Eval + tracing | RAG Triad, OTel | ~10 | Limited |
| **DeepEval** | Eval + CI/CD | Automated testing | ~10 | Limited |
| **ARES** | Eval framework | LLM+classifier | 3 | No |
| **Phoenix** | Observability | Retrieval debugging | ~5 | Yes |
| **Langfuse** | Observability | Tracing + experiments | Via integrations | Yes |

---

## 10. Managed RAG Evaluation Services

All three major hyperscalers now offer RAG evaluation capabilities:

| Provider | Service | RAG Metrics | Key Feature |
|----------|---------|-------------|-------------|
| **AWS** | Amazon Bedrock Evaluations | Context relevance, answer faithfulness, answer relevance, citation precision, citation coverage | Bring-your-own-inference (BYOI) — evaluate RAG systems from any provider |
| **Azure** | Azure AI Foundry | RAG Triad metrics, evaluation dashboards | Agentic Retrieval API with up to 40% better relevance for complex queries |
| **GCP** | Vertex AI Gen AI Evaluation Service | Custom evaluation criteria, built-in metrics | Managed pipeline for RAG evaluation with SQuAD 2.0 integration |

IBM and Oracle do not yet offer dedicated RAG evaluation services, though IBM's watsonx.ai platform supports custom evaluation workflows, and IBM Research produced the MTRAG benchmark.

---

## 11. LLM-as-a-Judge: Caveats

Reference-free evaluation frameworks (RAGAS, ARES, TruLens) rely on LLMs to judge output quality. This approach has known limitations:

**Known biases:**
- **Positional bias**: Preference for certain answer positions in comparisons.
- **Verbosity bias**: Longer responses rated higher regardless of quality.
- **Self-enhancement bias**: Models rate their own outputs more favourably.
- **Limited reasoning**: LLM judges may not detect subtle factual errors.

**Mitigation strategies:**
- Use structured rubrics with clear criteria.
- Employ multi-judge panels (multiple LLMs or human+LLM).
- Calibrate scores against human judgements on a held-out set.
- Combine LLM-as-judge with reference-required metrics where ground truth exists.

**Bottom line**: LLM-as-judge is useful for rapid iteration and production monitoring but should not be the sole evaluation method. Validate against human judgement periodically.

---

## 12. Choosing a Benchmark Strategy

### For a quick start

Use **RAGAS** with a sample of real user queries. Measure faithfulness, answer relevancy, context precision, and context recall. This gives a baseline without labelled data.

### For rigorous evaluation

1. **Retrieval**: Benchmark your retriever against BEIR or the MTEB retrieval subset using nDCG@10.
2. **Generation**: Run RGB or CRUD-RAG to test specific RAG abilities (noise robustness, negative rejection, etc.).
3. **End-to-end**: Use RAGAS or TruLens RAG Triad for holistic assessment.

### For conversational chatbots

Add **MTRAG** to test multi-turn handling, coreference resolution, and unanswerable question rejection.

### For domain-specific applications

Use domain benchmarks (MedRAG for medical, RAGEval for finance/legal) alongside generic benchmarks.

### For production monitoring

Pair a reference-free framework (RAGAS or DeepEval) with an observability tool (Langfuse or Phoenix) for continuous evaluation. Use a managed service (Bedrock, Vertex AI, or AI Foundry) if already on that cloud platform.

---

## 13. Limitations of Current Benchmarks

Current RAG benchmarks have several well-documented limitations:

1. **QA-centric**: Most benchmarks test only question answering, ignoring other RAG applications (content generation, summarisation, error correction). CRUD-RAG partially addresses this.

2. **Static knowledge**: Benchmarks use fixed document collections that don't reflect the dynamic nature of real knowledge bases. The HoH Benchmark (ACL 2025) is beginning to address temporal knowledge.

3. **Missing retrieval isolation**: Many benchmarks evaluate retrieval and generation together, making it difficult to diagnose which component failed. Evaluate retrieval independently with BEIR/MTEB.

4. **LLM-as-judge reliability**: Reference-free metrics depend on LLM judges with known biases. Correlation with human judgement varies by task and domain.

5. **No agentic RAG coverage**: Modern RAG systems increasingly involve tool use, multi-step retrieval, and agentic loops. No current benchmark evaluates this paradigm.

6. **Cost and latency gaps**: Academic benchmarks focus on quality metrics, ignoring operational concerns (latency, token cost, retrieval overhead) that are often more decision-relevant in production.

7. **Chunking/indexing strategy evaluation**: How documents are chunked, embedded, and indexed massively affects RAG quality, but only CRUD-RAG evaluates this dimension.

---

## References

1. Chen, J., Lin, H., Han, X. & Sun, L. (2024). "Benchmarking Large Language Models in Retrieval-Augmented Generation." *AAAI 2024*. [https://doi.org/10.1609/aaai.v38i16.29728](https://doi.org/10.1609/aaai.v38i16.29728)

2. Yu, H., Gan, A., Zhang, K. et al. (2024). "Evaluation of Retrieval-Augmented Generation: A Survey." *arXiv:2405.07437*. [https://arxiv.org/abs/2405.07437](https://arxiv.org/abs/2405.07437)

3. Lyu, Y., Li, Z., Niu, S. et al. (2024). "CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models." *J. ACM*. [https://doi.org/10.1145/3701228](https://doi.org/10.1145/3701228)

4. Katsis, Y., Rosenthal, S., Fadnis, K. et al. (2025). "MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems." *TACL*. [https://arxiv.org/abs/2501.03468](https://arxiv.org/abs/2501.03468)

5. Xiong, G., Jin, Q. & Lu, Z. (2024). "Benchmarking Retrieval-Augmented Generation for Medicine." *ACL Findings 2024*. [https://arxiv.org/abs/2402.13178](https://arxiv.org/abs/2402.13178)

6. Thakur, N., Reimers, N., Rücklé, A. et al. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." *NeurIPS 2021 Datasets & Benchmarks*. [https://arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663)

7. Es, S., James, J., Espinosa-Anke, L. & Schockaert, S. (2023). "RAGAS: Automated Evaluation of Retrieval Augmented Generation." *arXiv:2309.15217*. [https://docs.ragas.io](https://docs.ragas.io)

8. Saad-Falcon, J., Khattab, O., Potts, C. & Zaharia, M. (2023). "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems." *arXiv:2311.09476*.

9. Gu, J. et al. (2024). "A Survey on LLM-as-a-Judge." *arXiv:2411.15594*. [https://arxiv.org/abs/2411.15594](https://arxiv.org/abs/2411.15594)

10. AWS. (2025). "Evaluate models or RAG systems using Amazon Bedrock Evaluations — Now Generally Available." [https://aws.amazon.com/blogs/machine-learning/evaluate-models-or-rag-systems-using-amazon-bedrock-evaluations-now-generally-available/](https://aws.amazon.com/blogs/machine-learning/evaluate-models-or-rag-systems-using-amazon-bedrock-evaluations-now-generally-available/)

11. Microsoft. (2025). "Evaluating and Optimizing RAG Agents with Azure AI Foundry." [https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/the-future-of-ai-evaluating-and-optimizing-custom-rag-agents-using-azure-ai-foun/4455215](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/the-future-of-ai-evaluating-and-optimizing-custom-rag-agents-using-azure-ai-foun/4455215)

12. Google. (2025). "Evaluate RAG Systems with Vertex AI." [https://codelabs.developers.google.com/codelabs/production-ready-ai-with-gc/6-ai-evaluation/evaluate-rag-systems-with-vertex-ai](https://codelabs.developers.google.com/codelabs/production-ready-ai-with-gc/6-ai-evaluation/evaluate-rag-systems-with-vertex-ai)

13. Evidently AI. (2025). "7 RAG Benchmarks." [https://www.evidentlyai.com/blog/rag-benchmarks](https://www.evidentlyai.com/blog/rag-benchmarks)

14. Gao, Y., Xiong, Y., Gao, X. et al. (2023). "Retrieval-Augmented Generation for Large Language Models: A Survey." *arXiv:2312.10997*. [https://arxiv.org/abs/2312.10997](https://arxiv.org/abs/2312.10997)
