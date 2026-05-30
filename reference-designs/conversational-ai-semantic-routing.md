# Conversational AI with Tiered Semantic Routing

| Field | Value |
|-------|-------|
| Created | 2026-05-27 |
| Last Updated | 2026-05-30 |
| Version | 1.3 |

---

- [Problem Statement](#problem-statement)
- [Architecture Overview](#architecture-overview)
- [Tier 0: Deterministic Fast Paths](#tier-0-deterministic-fast-paths)
- [Tier 1: Embedding Semantic Router](#tier-1-embedding-semantic-router)
- [Tier 2: Slot Guards](#tier-2-slot-guards)
- [Tier 3: Lightweight Router LLM](#tier-3-lightweight-router-llm)
- [Tier 4: Full Agentic Planner Fallback](#tier-4-full-agentic-planner-fallback)
- [Multi-Intent Handling](#multi-intent-handling)
- [Route Handler Protocol](#route-handler-protocol)
- [Evaluation Methodology](#evaluation-methodology)
- [Phased Rollout Strategy](#phased-rollout-strategy)
- [Technology Options](#technology-options)
- [Scale Guidance](#scale-guidance)
- [When NOT to Use This Pattern](#when-not-to-use-this-pattern)
- [Engineering Readiness Pack](#engineering-readiness-pack)
- [References](#references)

## Problem Statement

Most conversational AI systems route every inbound request through a full agentic planner loop: session lookup, intent classification via LLM, tool selection, tool execution, final LLM response generation, and scoring. This architecture is correct for complex, multi-step requests — but production traffic analysis consistently reveals that the majority of requests invoke exactly one tool (or zero tools). The full planner adds unnecessary latency and cost for these straightforward cases: a planner LLM call, tool execution, a final LLM call, session hydration, and quality scoring — all for a request whose destination was obvious from the utterance alone.

A tiered semantic routing system addresses this by classifying requests early and routing high-confidence, single-intent messages directly to focused handlers, reserving the full planner for genuinely complex or ambiguous cases. Empirical research supports this approach: Wang et al. (2025) demonstrated that semantic routing in LLM serving systems reduces response latency by 47% and token consumption by 49% while improving accuracy, and Arora et al. (2024) showed that hybrid encoder-plus-LLM routing achieves within 2% of native LLM accuracy at 50% less latency.

## Architecture Overview

The system is structured as a cascading series of tiers, evaluated top-to-bottom. Each tier either resolves the request (routes it to a handler) or passes it down to the next tier. Early tiers are fast and cheap; later tiers are more capable but more expensive.

```
User message
    │
    ▼
┌─────────────────────────────────────┐
│  Tier 0: Deterministic fast paths   │ ─── match ──→ Handler (immediate)
└─────────────────────────────────────┘
    │ no match
    ▼
┌─────────────────────────────────────┐
│  Tier 1: Embedding semantic router  │ ─── high confidence ──→ Tier 2
└─────────────────────────────────────┘
    │ uncertain
    ▼
┌─────────────────────────────────────┐
│  Tier 3: Lightweight router LLM     │ ─── classified ──→ Tier 2
└─────────────────────────────────────┘
    │ ambiguous / multi-intent
    ▼
┌─────────────────────────────────────┐
│  Tier 4: Full agentic planner       │ ─── resolve ──→ Tool orchestration
└─────────────────────────────────────┘
```

Tier 2 (slot guards) sits between classification and handler execution — it validates that required parameters are present before dispatching.

## Tier 0: Deterministic Fast Paths

Pattern-matched responses for utterances that never require tool invocation:

- **Greetings and politeness**: "hello", "thanks", "goodbye" — respond with templated replies.
- **Capability questions**: "what can you do?", "help" — return a static or semi-static capability summary.
- **Out-of-scope declarations**: utterances that match known out-of-scope patterns — return a polite decline.

Implementation is typically regex or keyword matching against a curated list. This tier handles 5-15% of production traffic at near-zero latency and zero LLM cost.

## Tier 1: Embedding Semantic Router

The core of the routing system. An embedding-based classifier that maps user utterances to domain routes using vector similarity. This approach has been validated in multiple domains: Manias et al. (2024) demonstrated that semantic routing with embedding encoders improves both accuracy and efficiency compared to standalone LLM prompting architectures in intent-based network management.

### Label Set Design

Start with 8-12 route labels maximum. Each label corresponds to a domain handler (e.g., `account_lookup`, `document_search`, `recommendation`, `status_check`). A two-level classification can separate conversation-level intent (question, command, clarification) from domain route.

### Example Library

For each label, curate 20-100 representative utterances. These are embedded offline and stored as reference vectors. Quality of examples matters more than quantity — cover paraphrases, colloquialisms, and edge phrasings.

### Runtime Pipeline

1. **Normalise**: lowercase, strip excess whitespace, expand common abbreviations.
2. **Embed**: encode the user utterance using the same model used for the example library.
3. **Compare**: compute cosine similarity against all reference embeddings.
4. **Aggregate**: average (or max) similarity scores per label.
5. **Threshold check**: apply dual-threshold logic.

### Dual Threshold Logic

Two conditions must both be satisfied for a confident route:

- **Absolute threshold**: the top label's score must exceed a minimum (e.g., 0.78).
- **Margin threshold**: the gap between the top and second-best label must exceed a minimum (e.g., 0.10).

Example scenarios:
- Top = 0.84, second = 0.68, margin = 0.16 → confident, route directly.
- Top = 0.74, second = 0.72, margin = 0.02 → uncertain, escalate to Tier 3.
- Top = 0.65, second = 0.40, margin = 0.25 → below absolute threshold, escalate.

Thresholds should be calibrated conservatively at launch and relaxed as evaluation data accumulates.

## Tier 2: Slot Guards

Once a route is selected (by Tier 1 or Tier 3), slot guards validate that the required parameters for that route's handler are present in the utterance or session context.

Each route declares its required slots. Examples:

| Route | Required Slots |
|-------|---------------|
| `recommendation` | occupation, state/region |
| `document_compare` | 2+ document references |
| `account_lookup` | account identifier or authenticated session |
| `schedule_meeting` | date, time, participant |

If slots are missing, the system generates a clarifying question targeting the missing information rather than routing to the handler or falling back to the planner. Slot extraction uses lightweight methods: regex patterns, spaCy NER, or session context lookup.

## Tier 3: Lightweight Router LLM

A small, fast language model invoked only when the embedding router is uncertain. This model's sole job is classification — it never generates user-facing responses. This two-stage pattern (fast encoder first, LLM fallback for uncertain cases) is validated by Arora et al. (2024), who showed that uncertainty-based routing from sentence transformers to LLMs combines the latency advantage of encoders with the accuracy of larger models.

### Output Schema

The router LLM returns structured metadata only:

```json
{
  "intent": "document_search",
  "confidence": 0.87,
  "slots": {"query": "annual report 2025", "doc_type": "pdf"},
  "is_multi_intent": false,
  "reason": "Single clear request for document retrieval"
}
```

### Model Selection

Use the smallest model that achieves acceptable classification accuracy. Quantised models at 3-8B parameters typically suffice for routing tasks (e.g., Phi-4 Mini at 3.8B, Mistral 7B, or Llama 4 Scout at 17B active parameters for more complex taxonomies). The model should be fine-tuned or few-shot prompted specifically for the routing schema — generic instruction-following models are unnecessarily large for this role.

## Tier 4: Full Agentic Planner Fallback

The existing full-capability agent loop, reserved for:

- Requests the router LLM flags as multi-intent.
- Requests where no tier achieves sufficient confidence.
- Domains not yet covered by route handlers.
- Complex orchestration requiring multiple tools in sequence.

This tier remains the safety net. The goal is not to eliminate it but to reduce its invocation rate from ~100% to 20-40% of traffic.

## Multi-Intent Handling

Multi-intent markers include conjunctions linking distinct requests ("check my balance and also find documents about..."), topic shifts mid-utterance, or multiple question marks addressing different domains.

When detected:
1. **Preferred**: escalate to the full planner (Tier 4), which can orchestrate multiple tools.
2. **Alternative**: ask the user which request to handle first, then process sequentially.

Never force a multi-intent utterance through a single-route handler — this produces incomplete or incorrect responses. For systems with complex intent disambiguation requirements, ontology-guided routing frameworks such as iCARE (Wiratunga et al., 2024) demonstrate how Case-Based Reasoning can complement embedding-based approaches for ambiguous multi-domain queries.

## Route Handler Protocol

Each handler implements a common interface:

```
Handler(utterance, slots, session_context) → Response
```

Handlers are focused functions that bypass the full agent loop. Common handler types:

- **Low-intent responder**: returns templated or lightly-generated responses for simple acknowledgements.
- **RAG lookup**: embeds the query, retrieves from a knowledge base, generates a grounded answer.
- **Single-tool executor**: calls exactly one tool with extracted slots, formats the result.
- **Clarification generator**: produces a targeted follow-up question for missing slots.

Handlers should be stateless where possible, receiving all necessary context as input. Session state (conversation history, user profile) is passed in rather than fetched internally.

## Evaluation Methodology

### Shadow Mode Evaluation

Before enabling direct routing in production:

1. Pull historical messages paired with their actual tool outcomes.
2. Run the router against this traffic offline.
3. Compare the router's predicted route against what the planner actually did.

### Key Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Direct-route rate | % of requests handled without planner | 40-70% |
| Fallback rate | % escalated to full planner | 30-60% |
| False direct-route rate | % incorrectly routed (would have needed planner) | < 2% |
| Latency saved | p50/p95 latency reduction for direct-routed requests | Measured |

### Core Principle

False direct-routing must be kept very low. Falling back to the planner is acceptable and expected — it adds cost but not incorrectness. Misrouting a request to the wrong handler produces a wrong answer, which is far worse than a slower correct answer. Calibrate thresholds to minimise misroutes, not to maximise direct-route rate.

## Phased Rollout Strategy

**Phase 1** — Deploy deterministic fast paths (Tier 0) only. Immediate latency and cost wins for trivial traffic with zero risk of misrouting.

**Phase 2** — Deploy the embedding router in shadow mode. Log predicted routes alongside actual planner outcomes. Measure accuracy without affecting production responses.

**Phase 3** — Enable high-confidence direct routes behind an allowlist of the safest, highest-volume routes. Monitor false direct-route rate closely.

**Phase 4** — Add the lightweight router LLM for cases where the embedding router is uncertain. Expand the allowlist as confidence grows.

**Phase 5** — Gradually reduce fallback dependence by adding new route handlers, improving example libraries, and relaxing thresholds based on accumulated evaluation data.

## Technology Options

### Open-Source Tools

| Component | Options |
|-----------|---------|
| Embedding libraries | sentence-transformers (Python framework for embedding models) |
| Embedding models | all-MiniLM-L6-v2, BGE (BAAI), E5 (intfloat/Microsoft) |
| Semantic router libraries | semantic-router (Aurelio Labs, Python) |
| Vector similarity | FAISS, Annoy, hnswlib |
| Slot extraction | spaCy NER, regex, lightweight fine-tuned models |
| Router LLM | Llama 4 Scout (17B active, 16 experts, MoE), Mistral 7B, Phi-4 Mini (3.8B, quantised for speed) |
| LLM output validation | Guardrails AI (output validation and guardrails for LLM responses) |
| Evaluation | DeepEval, RAGAS (Vibrant Labs AI, for RAG handlers) |

### Managed Service Options

| Provider | Embeddings | Vector Store | Router LLM |
|----------|-----------|--------------|------------|
| **Azure** | Azure OpenAI Embeddings | Azure AI Search | Azure OpenAI (GPT-4o-mini) |
| **AWS** | Amazon Bedrock (Titan Text Embeddings V2, Cohere Embed) | Amazon OpenSearch Serverless (vector engine) | Amazon Bedrock (Llama/Mistral) |
| **GCP** | Vertex AI Text Embeddings (gemini-embedding-001) | Gemini Agent Platform Vector Search (formerly Vertex AI Vector Search) | Vertex AI (Gemma, Llama) |

## Scale Guidance

### Small Chatbots (<1,000 daily messages, <5 routes)

- Skip the embedding router entirely. Use Tier 0 (deterministic fast paths) and go directly to Tier 4 (full planner) for everything else.
- The overhead of maintaining an example library and calibrating thresholds is not justified until you have enough traffic to measure false-route rates.
- Start by instrumenting your existing planner to log which tool was actually called per request. Only invest in routing when logs prove >60% of requests call a single predictable tool.

### Medium Chatbots (1,000-50,000 daily messages, 5-15 routes)

- Implement the full tiered architecture as described.
- Shadow mode evaluation is critical — run for 2-4 weeks before enabling direct routing.
- Threshold calibration requires at least 500 messages per route for statistical confidence.
- One engineer can maintain the routing system alongside other responsibilities.

### Large Chatbots (50,000+ daily messages, 15+ routes)

- Invest in automated threshold tuning (Bayesian optimisation over historical traffic).
- Per-route A/B testing infrastructure to measure quality impact of direct routing vs planner.
- Dedicated embedding model fine-tuned on your domain's utterances (general sentence-transformers plateau at high route counts).
- Consider per-route handler versioning and canary deployments.
- Real-time monitoring dashboards for false-route rates, latency percentiles, and cost per route.

## When NOT to Use This Pattern

This architecture adds complexity. It is not appropriate when:

- **Every request genuinely requires multi-tool orchestration** — if traffic analysis shows most requests use 3+ tools, the routing overhead provides no benefit.
- **Very small scale** — if latency and cost are not meaningful concerns, the simpler monolithic planner is easier to maintain.
- **Rapidly changing intent taxonomy** — if new intents are added weekly and example libraries cannot keep pace, the embedding router will produce excessive fallbacks, negating its value.
- **Safety-critical domains with low error tolerance** — if even a 1-2% false direct-route rate is unacceptable, the full planner's additional validation may be required for every request.

## Engineering Readiness Pack

This design becomes engineering-ready when the router is specified, tested, and observable as a production control plane component rather than a prompt-only optimisation.

### Evidence and claim ledger

Maintain a `claim-ledger.md` for all performance claims, model recommendations, and threshold defaults.

| Claim class | Current status | Required handling |
|---|---|---|
| Latency and token savings from semantic routing | Literature-supported but workload-dependent | Re-measure in shadow mode on the target traffic mix. |
| Default thresholds such as 0.78 absolute / 0.10 margin | Starting assumptions | Calibrate per embedding model, route taxonomy, and false-route tolerance. |
| Direct-route rate targets | Directional | Set from production traffic analysis, not from generic benchmarks. |
| Model/library choices | Time-sensitive | Verify licences, maintenance status, and inference latency before adoption. |

### Route contract

Each route must be registered in a machine-readable catalog:

```json
{
  "route_id": "document_search",
  "owner": "knowledge-platform",
  "description": "Search approved document corpus",
  "required_slots": ["query"],
  "optional_slots": ["document_type", "date_range"],
  "allowed_tools": ["search_documents"],
  "max_side_effect": "read_only",
  "handler_version": "v1",
  "fallback_route": "planner",
  "eval_suite": "router_document_search_v1"
}
```

Handlers with write actions, financial consequences, safety impact, or irreversible side effects must not be direct-routed until their own approval and rollback controls are proven.

### Anti-hallucination and routing controls

Routing mistakes create downstream hallucination risk because the wrong handler may receive the wrong context. Implement:

- Server-side structured outputs for router LLM results with schema validation and rejected extra fields.
- Direct-route allowlist; new routes start in shadow mode and cannot self-enable.
- Multi-intent detector that escalates rather than compressing distinct requests into one route.
- Handler-level "unsupported request" response when required slots are missing or route assumptions do not hold.
- Prompt-injection tests where user text attempts to override routing labels, tool policies, or fallback behaviour.
- Per-route confidence calibration, not a single global threshold for all routes.

### Threat model

| Threat | Control |
|---|---|
| Route gaming by users or embedded instructions | Treat user text as data; never allow it to set route IDs, tool names, or confidence values. |
| False direct-routing to a capable but wrong tool | Conservative thresholds, allowlist rollout, and false-route alerts. |
| Missing slot causes fabricated defaults | Slot guards must ask clarification questions; handlers cannot invent required slots. |
| Route taxonomy drift | Version route catalogs, example libraries, and eval sets together. |
| Sensitive tool exposure | Route catalog must include `max_side_effect`; policy engine blocks direct routes to disallowed tools. |

### Evaluation and acceptance gates

Before production direct-routing:

- Use at least 500 historical messages per high-volume route or mark confidence as low.
- Measure top-1 accuracy, false direct-route rate, false fallback rate, multi-intent miss rate, p50/p95 latency, and cost per routed request.
- Run adversarial tests for prompt injection, ambiguous utterances, missing slots, out-of-scope requests, and near-neighbour intents.
- Promotion gate: false direct-route rate below the route-specific threshold, no severity-1 side-effect errors, and rollback tested by disabling the route catalog entry.

### Observability and runbook

Log message ID, selected tier, candidate labels and scores, route ID, margin, slots, handler version, fallback reason, outcome, latency, and user correction/override. Dashboards should expose direct-route rate, false-route investigations, fallback causes, top ambiguous label pairs, and per-route drift.

Runbooks must cover threshold rollback, example-library corruption, embedding model change, handler outage, route disabled by policy, and emergency full-planner fallback.

## References

- Aurelio AI. *semantic-router: Superfast AI decision making and intelligent processing of multi-modal data*. GitHub. https://github.com/aurelio-labs/semantic-router
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019. https://arxiv.org/abs/1908.10084
- Johnson, J., Douze, M., & Jegou, H. (2017). *Billion-scale similarity search with GPUs (FAISS)*. GitHub. https://github.com/facebookresearch/faiss
- Spotify. *Annoy: Approximate Nearest Neighbors Oh Yeah*. GitHub. https://github.com/spotify/annoy
- Hnswlib. *Fast approximate nearest neighbor search*. GitHub. https://github.com/nmslib/hnswlib
- Microsoft. *Azure AI Search vector search documentation*. https://learn.microsoft.com/en-us/azure/search/vector-search-overview
- AWS. *Amazon Titan Text Embeddings models*. https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html
- AWS. *Vector search collections — Amazon OpenSearch Serverless*. https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-vector-search.html
- Google Cloud. *Gemini Agent Platform Vector Search documentation (formerly Vertex AI Vector Search)*. https://cloud.google.com/vertex-ai/docs/vector-search/overview
- Meta. *The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation*. https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- Microsoft. *Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models*. https://arxiv.org/abs/2503.01743
- Confident AI. *DeepEval: The open-source LLM evaluation framework*. GitHub. https://github.com/confident-ai/deepeval
- Vibrant Labs AI. *RAGAS: Supercharge Your LLM Application Evaluations*. GitHub. https://github.com/vibrantlabsai/ragas
- Guardrails AI. *Adding guardrails to large language models*. GitHub. https://github.com/guardrails-ai/guardrails
- Manias, D. M., Chouman, A., & Shami, A. (2024). *Semantic Routing for Enhanced Performance of LLM-Assisted Intent-Based 5G Core Network Management and Orchestration*. IEEE GlobeCom 2024. https://arxiv.org/abs/2404.15869
- Wang, C., Liu, X., Liu, Y., Zhu, Y., Mo, X., Jiang, J., & Chen, H. (2025). *When to Reason: Semantic Router for vLLM*. Workshop on ML for Systems, NeurIPS 2025. https://arxiv.org/abs/2510.08731
- Arora, G., Jain, S., & Merugu, S. (2024). *Intent Detection in the Age of LLMs*. Proceedings of EMNLP 2024 Industry Track, pp. 1559–1570. https://aclanthology.org/2024.emnlp-industry.114/
- Wiratunga, N. et al. (2024). *iCARE: Ontology-Guided Intent Routing for Multi-Agent LLM-Based Dialogue Systems*. CEUR Workshop Proceedings, Vol. 4178. https://ceur-ws.org/Vol-4178/paper11.pdf
- NIST AI Risk Management Framework and Generative AI Profile. https://www.nist.gov/itl/ai-risk-management-framework
- OWASP Top 10 for LLM Applications 2025. https://owasp.org/www-project-top-10-for-large-language-model-applications/
- OpenAI Evals and graders documentation. https://platform.openai.com/docs/guides/evals
