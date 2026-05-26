# Best Practices for AI Development on Azure in 2026

| Field | Value |
|-------|-------|
| Created | 2026-05-26 |
| Last Updated | 2026-05-26 |
| Version | 1.0 |

---

- [Executive Summary](#executive-summary)
- [Platform Architecture: Microsoft Foundry](#platform-architecture-microsoft-foundry)
- [Model Selection Strategy](#model-selection-strategy)
- [RAG and Knowledge Retrieval](#rag-and-knowledge-retrieval)
- [Document Processing Pipelines](#document-processing-pipelines)
- [Agent Development](#agent-development)
- [AI Gateway with Azure API Management](#ai-gateway-with-azure-api-management)
- [Hosting and Compute Patterns](#hosting-and-compute-patterns)
- [Security and Networking](#security-and-networking)
- [Responsible AI and Content Safety](#responsible-ai-and-content-safety)
- [Cost Management](#cost-management)
- [Observability and Monitoring](#observability-and-monitoring)
- [Evaluation and Testing](#evaluation-and-testing)
- [Infrastructure as Code](#infrastructure-as-code)
- [CI/CD for AI Applications](#cicd-for-ai-applications)
- [Copilot Studio and Low-Code Scenarios](#copilot-studio-and-low-code-scenarios)
- [Architecture Patterns by Business Size](#architecture-patterns-by-business-size)
- [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
- [References](#references)

## Executive Summary

Azure's AI platform underwent a major consolidation in 2025–2026. Azure AI Studio became **Microsoft Foundry** (also called Azure AI Foundry) — a unified platform combining the model catalogue, prompt flow orchestration, evaluation tools, agent hosting, and responsible AI guardrails. Separately, Microsoft merged **Semantic Kernel** and **AutoGen** into the **Microsoft Agent Framework (MAF)**, which reached GA in April 2026 as the single SDK for building AI agents on Azure in .NET and Python.

For contractors building cloud AI systems, the 2026 Azure landscape has three tiers of engagement:

1. **Pro-code (Foundry + MAF)** — full control over architecture, model selection, retrieval, and orchestration. Suitable for enterprise RAG systems, multi-agent platforms, and bespoke AI applications.
2. **Low-code (Copilot Studio)** — rapid delivery of conversational agents with SharePoint/Dataverse knowledge grounding, deployed into Teams or web channels. Suitable for internal productivity tools and simple Q&A bots.
3. **Hybrid** — Copilot Studio as the user-facing surface, with Azure AI Foundry Agent Service providing the reasoning backend and MCP-based tool integrations.

This guide covers production best practices across all three, with emphasis on the pro-code path where professional services teams spend most of their time.

---

## Platform Architecture: Microsoft Foundry

Microsoft Foundry (formerly Azure AI Studio) is the control plane for AI development on Azure. It provides:

| Component | Purpose |
|-----------|---------|
| **Foundry Hub** | Shared infrastructure: networking, identity, key vault, storage. One per environment (dev/staging/prod). |
| **Foundry Project** | Workload isolation within a hub. Maps 1:1 to an application or engagement. |
| **Model Catalogue** | 1,800+ models — Azure OpenAI (GPT-5.x, GPT-4o, o3), Meta Llama 4, Mistral, Cohere, Phi-4, DeepSeek, and more. |
| **Foundry Agent Service** | Managed runtime for hosting agents with built-in tool calling, knowledge grounding, and thread management. |
| **Foundry IQ** | Knowledge base abstraction over Azure AI Search — agentic retrieval with query planning, sub-query decomposition, and re-ranking. |
| **Evaluation** | Built-in evaluators for groundedness, relevance, coherence, fluency, and safety — plus custom evaluator support. |

### Hub/Project topology

```
Foundry Hub (shared infra)
├── Project: app-dev        (development)
├── Project: app-staging    (pre-production)
└── Project: app-prod       (production)
```

**Best practices:**

- One hub per environment boundary (dev/prod) — hubs share networking config and managed VNet.
- One project per application or customer engagement — projects isolate connections, deployments, and evaluation data.
- Use managed identity (system-assigned on the hub) for all service-to-service authentication.
- Enable managed VNet on the hub with private endpoints for dependent services (AI Search, Storage, Cosmos DB).

---

## Model Selection Strategy

The Azure model catalogue now spans frontier proprietary models and open-source alternatives. Selection should be driven by task requirements, not brand loyalty.

### Decision matrix

| Use Case | Recommended Model | Alternative (Open-Source) | Deployment Type |
|----------|-------------------|---------------------------|-----------------|
| Complex reasoning / long-context | GPT-5 / GPT-5.2 | Llama 4 Maverick (128K) | Standard or PTU |
| High-volume chat / simple Q&A | GPT-4o-mini | Phi-4 | Standard (pay-per-token) |
| Multimodal (vision + text) | GPT-4o / GPT-5 | Llama 4 Scout | Standard |
| Code generation | GPT-5 / Claude Sonnet 4 | DeepSeek-V3 | Standard |
| Embedding | text-embedding-3-large | Cohere Embed v4 | Standard |
| Edge / on-device | Phi-4-mini | Llama 4 Nano | Serverless API or self-hosted |
| Structured extraction | GPT-4o (JSON mode) | Mistral Large | Standard |

### Model selection principles

1. **Start with GPT-4o-mini** for cost-efficient prototyping — upgrade to larger models only when evaluation metrics demand it.
2. **Use open-source models** (deployed as Serverless API or on Managed Compute) when data sovereignty, cost, or fine-tuning requirements preclude proprietary options.
3. **Pin model versions** in production (`gpt-4o-2024-11-20`, not `gpt-4o`) — auto-upgrades break prompts.
4. **Benchmark on your data** — public leaderboard rankings do not predict task-specific performance. Use Foundry Evaluation or DeepEval to compare models on your actual workload before committing.
5. **Plan for model retirement** — Azure retires model versions with 90 days notice. Maintain a fallback model in your deployment config.

---

## RAG and Knowledge Retrieval

### Azure AI Search — the default retrieval layer

Azure AI Search is the managed retrieval engine for RAG on Azure. In 2026, it supports three retrieval paradigms:

| Paradigm | When to Use |
|----------|-------------|
| **Classic hybrid search** (BM25 + vector + semantic ranker) | Standard RAG with pre-indexed content. Predictable, well-understood. |
| **Agentic retrieval** (Foundry IQ knowledge bases) | Agent-driven scenarios requiring query planning, sub-query decomposition, and multi-index reasoning. |
| **Direct vector search** | Embedding-only workloads where keyword matching adds no value (e.g., image similarity). |

### Hybrid search configuration

For most enterprise RAG deployments, hybrid search with semantic ranker is the correct starting point:

```
Query → BM25 (keyword) + Vector (embedding) → RRF fusion → Semantic Ranker → Top-K chunks → LLM
```

**Best practices:**

- **Always enable semantic ranker** — it re-ranks the fused results using a cross-encoder model for significantly better relevance. The cost is marginal ($1/1000 queries at Standard tier).
- **Use `text-embedding-3-large` at 1536 dimensions** as the default embedding model — best cost/quality trade-off on Azure.
- **Enable integrated vectorisation** — let AI Search call the embedding model during both indexing and query time, eliminating client-side embedding logic.
- **Set `k=50` for vector search and `top=50` for BM25** before fusion, then return `top=5–10` final results to the LLM.
- **Use metadata filters** to scope retrieval (by document type, customer, date range) — this prevents irrelevant results and reduces token spend.

### Agentic retrieval (Foundry IQ)

Agentic retrieval is the new pattern for agent-to-search interaction, GA in the `2026-04-01` API version:

- A **knowledge base** object wraps one or more AI Search indexes with query configuration.
- The **knowledge agent** decomposes complex user queries into sub-queries, executes them against the knowledge base, and synthesises results.
- Supports **MCP protocol** — the knowledge base is exposed as an MCP tool that any agent can invoke.

**When to use agentic retrieval over classic RAG:**

- Queries span multiple document types or indexes.
- The user's question requires multi-hop reasoning (e.g., "compare policy A with policy B").
- You're building a multi-agent system where retrieval is delegated to a specialist agent.

**When to stick with classic RAG:**

- Simple Q&A over a single document corpus.
- Latency-sensitive applications (agentic retrieval adds query planning overhead).
- Cost-constrained deployments (additional LLM calls for query decomposition).

### Chunking strategy

| Document Type | Chunking Approach | Chunk Size |
|---------------|-------------------|------------|
| Structured reports (PDF, DOCX) | Document Intelligence layout-aware chunking | 512–1024 tokens |
| Legal / compliance documents | Semantic chunking (paragraph boundaries from Document Intelligence markdown output) | 256–512 tokens |
| Code documentation | Fixed-size with overlap | 512 tokens, 128 overlap |
| Tabular data (Excel, CSV) | Row-per-chunk or table-per-chunk | Varies |

---

## Document Processing Pipelines

### Azure Document Intelligence (formerly Form Recognizer)

Document Intelligence is the entry point for all document ingestion pipelines on Azure. The **Layout model** (prebuilt) handles most enterprise document types.

**Best practices:**

1. **Use the Layout model with markdown output** — it preserves document structure (headings, tables, lists) that downstream chunking can exploit.
2. **Enable figure extraction** for documents with diagrams or charts — extracted figures can be processed by a vision model for captioning.
3. **Use the `2024-11-30` GA API** (or later) — it includes improved table extraction and semantic chunking support.
4. **Process asynchronously** — use Azure Functions with a Storage Queue trigger for batch ingestion. Document Intelligence has rate limits (15 concurrent requests on S0).
5. **Store raw extracted content in Blob Storage** alongside the original document — enables re-chunking without re-extraction if your strategy changes.

### Production pipeline architecture

```
Source (Blob Storage)
  → Event Grid trigger
    → Azure Function: submit to Document Intelligence
      → Document Intelligence: extract layout + markdown
        → Azure Function: chunk + embed
          → Azure AI Search: index chunks
            → Cosmos DB: store metadata + processing status
```

- Use **Durable Functions** for orchestrating multi-step pipelines with retry logic.
- Tag each chunk with source document ID, page number, and section heading for citation generation.
- For large corpora (10,000+ documents), use Document Intelligence **batch API** with a dedicated S0 instance.

---

## Agent Development

### Microsoft Agent Framework (MAF) 1.0

MAF is the unified SDK for building AI agents on Azure, combining the best of Semantic Kernel (production reliability, enterprise plugins) and AutoGen (multi-agent orchestration, group chat patterns).

**Key concepts:**

| Concept | Description |
|---------|-------------|
| **Agent** | A unit of work with a system prompt, tools, and an LLM backend. |
| **Kernel** | The runtime that manages plugins, memory, and LLM connections. |
| **Plugin** | A collection of related tools (functions) exposed to the agent. |
| **Workflow** | Orchestration patterns: sequential, concurrent, handoff, group chat. |
| **MCP Server** | Standard protocol for exposing tools — Azure Functions now supports MCP natively. |

### Agent orchestration patterns

From Microsoft's AI Architecture Centre, the five canonical patterns:

1. **Sequential** — agents execute in a fixed pipeline (e.g., extract → validate → enrich).
2. **Concurrent** — multiple agents work in parallel on different sub-tasks, results are merged.
3. **Handoff** — a router agent delegates to specialist agents based on intent classification.
4. **Group chat** — agents collaborate iteratively on a shared problem (useful for code review, planning).
5. **Magentic (supervisor)** — a supervisor agent plans, delegates, and evaluates sub-agent outputs.

### Best practices for production agents

1. **Keep agents single-purpose** — one agent, one job. Compose complex behaviours through orchestration, not monolithic prompts.
2. **Use MCP for tool interfaces** — Azure Functions with MCP triggers provide a standard, testable, deployable surface for tools. This decouples tool logic from agent logic.
3. **Implement OBO (On-Behalf-Of) token flow** for tools that access user-scoped data (CRM, SharePoint, databases). The agent acts with the user's identity, not a service principal with broad access.
4. **Set token budgets and timeout limits** per agent — prevent runaway reasoning loops from consuming PTU capacity or blocking other requests.
5. **Log every tool invocation and LLM call** with OpenTelemetry spans — you cannot debug agents without observability.
6. **Use Foundry Agent Service for managed hosting** when you don't need fine-grained control over the runtime. It handles thread management, tool dispatch, and conversation state.

### When to use Foundry Agent Service vs. self-hosted MAF

| Criterion | Foundry Agent Service | Self-hosted (Container Apps / AKS) |
|-----------|----------------------|-------------------------------------|
| Time to production | Days | Weeks |
| Custom orchestration logic | Limited (sequential, tool-calling) | Full control |
| Multi-agent workflows | Basic (single agent with tools) | Complex (group chat, handoff, supervisor) |
| Networking | Managed VNet only | Any topology |
| Cost at scale | Higher per-request | Lower (own compute) |
| State management | Built-in threads | BYO (Cosmos DB, Redis) |

---

## AI Gateway with Azure API Management

Azure API Management (APIM) functions as a **GenAI Gateway** — a centralised control plane for all LLM traffic.

### Why every production deployment needs an AI Gateway

- **Load balancing** across multiple Azure OpenAI deployments (PTU primary + consumption fallback).
- **Token rate limiting** per consumer/team to prevent budget overruns.
- **Semantic caching** to reduce redundant LLM calls for repeated queries.
- **Content safety enforcement** at the gateway level (pre-request filtering).
- **Observability** — centralised token usage metrics, latency tracking, and cost attribution.
- **Circuit breaking** — graceful failover when a backend deployment returns 429s.

### Configuration pattern

```
Client → APIM (AI Gateway policies)
           ├── Backend Pool: PTU deployment (priority 1)
           ├── Backend Pool: Standard deployment (priority 2, spillover)
           └── Backend Pool: Secondary region (priority 3, disaster recovery)
```

**Best practices:**

1. **Use backend pools with priority-based routing** — PTU first, standard consumption as spillover. APIM handles 429 retry automatically.
2. **Emit token metrics to Application Insights** — use the `emit-token-metric` policy for per-consumer cost attribution.
3. **Apply `azure-openai-token-limit`** policy per subscription key to enforce per-team budgets.
4. **Enable semantic caching** for deterministic queries (FAQ-style, document summarisation) — saves 30–60% on repeated traffic patterns.
5. **Deploy APIM in the same VNet as your Foundry hub** — use internal mode with private endpoints for zero-egress traffic.

---

## Hosting and Compute Patterns

### Decision matrix

| Workload Characteristic | Recommended Service | Rationale |
|------------------------|---------------------|-----------|
| Event-driven, short-lived (<5 min) | Azure Functions (Flex Consumption) | Scale-to-zero, pay-per-execution, MCP server support |
| Long-running agent sessions (5–30 min) | Azure Container Apps | GPU support, scale-to-zero, Dapr integration |
| High-throughput API with steady traffic | Azure App Service (P1v3+) | Predictable cost, simple deployment |
| Complex multi-container, custom networking | AKS | Full control, KEDA autoscaling |
| Frontend (React/Next.js) | Azure Static Web Apps | Global CDN, integrated auth, API backend |

### Azure Functions for AI workloads (2026 patterns)

- **Flex Consumption plan** is the default for AI — supports VNet integration, larger instance sizes (up to 4 vCPU / 16 GB), and longer timeouts (up to 30 min for Durable Functions).
- **MCP server trigger** (GA January 2026) — host MCP-compliant tools as Azure Functions with built-in OBO authentication and streamable HTTP transport.
- **Durable Functions** for orchestrating multi-step AI pipelines (document ingestion, batch evaluation, multi-agent workflows).

### Azure Container Apps for AI workloads

- **GPU workloads** — Container Apps supports GPU-enabled containers for running local models (Phi-4, Whisper, embedding models).
- **Scale-to-zero** with custom KEDA scalers — scale on queue depth, HTTP concurrency, or custom metrics.
- **Dapr integration** for service-to-service communication in multi-agent architectures.
- **Jobs** (scheduled or event-triggered) for batch processing, evaluation runs, and data pipelines.

---

## Security and Networking

### Network architecture

All production AI deployments on Azure should use a **hub-spoke VNet topology** with private endpoints:

```
Hub VNet (shared services)
├── Azure Firewall / NVA
├── Bastion
└── APIM (internal mode)

Spoke VNet (AI workload)
├── Foundry Hub (managed VNet with private endpoints)
│   ├── PE: Azure OpenAI
│   ├── PE: Azure AI Search
│   ├── PE: Cosmos DB
│   ├── PE: Storage Account
│   └── PE: Key Vault
├── Container Apps Environment (internal)
└── Private DNS Zones
```

### Identity and access

| Principle | Implementation |
|-----------|---------------|
| No API keys in application code | Managed Identity (system-assigned) for all Azure service access |
| User-scoped data access | OBO token exchange through Entra ID |
| Least privilege | Custom RBAC roles: `Cognitive Services OpenAI User` for inference, not `Contributor` |
| Key rotation (where keys are unavoidable) | Key Vault with automatic rotation policies |
| Cross-tenant access | Entra External ID or B2B guest access — never shared service principals |

### Security checklist for AI deployments

- [ ] Public network access **disabled** on Azure OpenAI, AI Search, Storage, Cosmos DB.
- [ ] Managed Identity used for all service-to-service auth.
- [ ] APIM acts as the single ingress point for LLM traffic — no direct client-to-OpenAI calls.
- [ ] Content Safety filters enabled at both the APIM gateway and the Azure OpenAI deployment level.
- [ ] Data encryption at rest with CMK (customer-managed keys) for regulated workloads.
- [ ] Diagnostic logs shipped to Log Analytics for audit trail.
- [ ] No PII in prompt text — use redaction before sending to the model, or Azure AI Content Safety PII detection.

---

## Responsible AI and Content Safety

### Azure AI Content Safety

Azure AI Content Safety provides guardrails at multiple levels:

| Layer | Capability | Configuration |
|-------|-----------|---------------|
| **Input filtering** | Hate, violence, sexual, self-harm detection | Severity thresholds (0–6) per category |
| **Prompt Shields** | Jailbreak detection, indirect injection detection | Enable on all user-facing deployments |
| **Groundedness detection** | Detect hallucinated claims not supported by retrieved context | Use in evaluation pipelines and optionally at runtime |
| **Protected material detection** | Detect verbatim reproduction of copyrighted text | Enable for public-facing applications |
| **Custom blocklists** | Domain-specific blocked terms/phrases | Per-deployment configuration |

### Implementation best practices

1. **Layer your defences** — Content Safety at the APIM gateway (pre-request), built-in filters on the Azure OpenAI deployment, and application-level validation of outputs.
2. **Set severity thresholds conservatively** for customer-facing applications — start at threshold 2 (block medium and above) and relax only if false positive rate is unacceptable.
3. **Always enable Prompt Shields** — jailbreak attempts are common in any public-facing LLM application.
4. **Use groundedness detection in evaluation** — run it against your test dataset to measure hallucination rate before go-live.
5. **Log all blocked requests** — content safety blocks should trigger alerts for security review, not just silent drops.
6. **Document your responsible AI posture** — Azure requires a use case application for GPT-4/GPT-5 access. Maintain a Responsible AI Impact Assessment per engagement.

---

## Cost Management

### Pricing tiers

| Deployment Type | Best For | Pricing Model |
|-----------------|----------|---------------|
| **Standard (pay-per-token)** | Variable/unpredictable workloads, development, PoC | Per 1M tokens (input/output priced separately) |
| **Provisioned Throughput (PTU)** | Steady-state production with predictable volume | Hourly per PTU (1 PTU ≈ 6 RPM for GPT-4o) |
| **Global deployment** | Maximum throughput, Microsoft-routed | Same token pricing, higher rate limits |
| **Data Zone deployment** | Data residency (US/EU) with higher limits | Same token pricing, geographic guarantee |

### Cost optimisation strategies

1. **Right-size your model** — GPT-4o-mini is 30x cheaper than GPT-4o for input tokens. Use it for classification, routing, extraction, and simple Q&A. Reserve GPT-4o/GPT-5 for complex reasoning.
2. **Prompt caching** — Azure OpenAI automatically caches identical prompt prefixes. Structure your system prompts as a stable prefix (cached) + dynamic suffix (user context). Cached tokens are 50% cheaper.
3. **PTU with consumption spillover** — provision PTUs for your baseline load (~P50), use standard consumption for burst traffic via APIM backend pool priority routing.
4. **Semantic caching at the gateway** — APIM can cache semantically similar queries. Effective for FAQ-style applications.
5. **Batch API for non-latency-sensitive work** — evaluation runs, bulk summarisation, and data enrichment jobs should use the Batch API at 50% discount.
6. **Monitor token waste** — track input/output token ratios. If output tokens consistently exceed input, your prompts may be too open-ended. If input tokens are very high relative to output, you may be over-stuffing context.

### Cost estimation rules of thumb (May 2026)

| Workload | Monthly Cost Estimate |
|----------|----------------------|
| Low-volume RAG chatbot (1K queries/day, GPT-4o-mini) | $50–150 NZD |
| Medium-volume RAG chatbot (10K queries/day, GPT-4o) | $800–2,000 NZD |
| Enterprise multi-agent platform (50K interactions/day) | $5,000–15,000 NZD (PTU recommended) |
| Document processing pipeline (10K docs/month) | $300–800 NZD (Document Intelligence + embedding) |
| Azure AI Search (Standard S1, 1M documents) | $400–600 NZD/month |

---

## Observability and Monitoring

### OpenTelemetry-native stack

Azure's AI observability is built on OpenTelemetry semantic conventions for GenAI (`gen_ai.*` attributes):

```
Application (MAF / custom code)
  → OpenTelemetry SDK (traces + metrics)
    → Azure Monitor Exporter
      → Application Insights
        → Log Analytics Workspace
```

### What to instrument

| Signal | What to Capture | Tool |
|--------|-----------------|------|
| **Traces** | Full request lifecycle: user query → retrieval → LLM call → tool invocations → response | OpenTelemetry distributed traces |
| **Metrics** | Token usage (input/output/cached), latency (P50/P95/P99), error rate, cache hit ratio | APIM `emit-token-metric` + custom metrics |
| **Logs** | Content safety blocks, evaluation failures, tool errors, model version changes | Structured logging to Log Analytics |
| **Evaluations** | Groundedness, relevance, citation accuracy — sampled from production traffic | Azure AI Evaluation SDK or DeepEval |

### Application Insights Agents view

The new **Agents view** (GA April 2026) in Application Insights provides a purpose-built dashboard for AI agent workloads:

- Agent execution timeline with tool call breakdown.
- Token consumption per agent/tool/conversation.
- Error clustering by agent type and failure mode.
- End-to-end latency decomposition (LLM time vs. tool time vs. retrieval time).

### Alerting thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| P95 latency (end-to-end) | >5s | >15s |
| Groundedness score (sampled) | <0.7 | <0.5 |
| Content safety blocks / hour | >10 | >50 |
| Token budget utilisation | >70% | >90% |
| 429 rate (throttling) | >5% of requests | >15% |

---

## Evaluation and Testing

### Two-track evaluation strategy

1. **Offline evaluation** (CI/CD) — run against a curated test dataset before deployment.
2. **Online evaluation** (production sampling) — continuously sample production traffic and evaluate quality.

### Tooling options

| Tool | Strengths | When to Use |
|------|-----------|-------------|
| **Azure AI Evaluation SDK** (`azure-ai-evaluation`) | Native Foundry integration, built-in evaluators, no extra infra | Default choice for Azure-native teams |
| **DeepEval** (open-source) | Richer metric library, CI/CD-native (pytest-style), provider-agnostic | When you need custom metrics, run evaluations locally, or support multi-cloud |
| **PromptFoo** (open-source) | Prompt comparison, A/B testing, model comparison | Prompt engineering phase, model selection |

### Evaluation metrics by use case

| Use Case | Key Metrics |
|----------|-------------|
| RAG Q&A | Groundedness, Answer Relevancy, Faithfulness, Citation Accuracy |
| Document summarisation | Coherence, Fluency, Completeness |
| Classification / routing | Accuracy, Precision, Recall, F1 |
| Agent task completion | Task Success Rate, Tool Call Accuracy, Steps to Completion |
| Content safety | False Positive Rate, False Negative Rate (adversarial testing) |

### Best practices

1. **Build your evaluation dataset from day one** — collect 30–50 representative Q&A pairs during requirements gathering. Expand to 200+ for production baselines.
2. **Automate evaluation in CI/CD** — every prompt change, model upgrade, or retrieval config change triggers an evaluation run. Gate deployments on threshold pass.
3. **Use LLM-as-judge for subjective metrics** — Azure AI Evaluation SDK uses GPT-4o as a judge for groundedness and relevance. Pin the judge model version.
4. **Red-team before launch** — test adversarial inputs (prompt injection, jailbreaks, off-topic queries) as a dedicated evaluation pass.
5. **Track metrics over time** — quality regressions are gradual. Dashboard your evaluation scores and alert on downward trends.

---

## Infrastructure as Code

### Bicep as the default

Bicep is the first-class IaC language for Azure. It has immediate support for new Azure resource types (including AI services) without waiting for Terraform provider updates.

### Module structure for AI deployments

```
infra/
├── main.bicep              (orchestrator)
├── parameters/
│   ├── dev.bicepparam
│   ├── staging.bicepparam
│   └── prod.bicepparam
├── modules/
│   ├── foundry-hub.bicep       (AI Foundry hub + managed VNet)
│   ├── foundry-project.bicep   (per-app project)
│   ├── openai.bicep            (Azure OpenAI + deployments)
│   ├── ai-search.bicep         (search service + indexes)
│   ├── cosmos-db.bicep         (database)
│   ├── apim.bicep              (API Management + AI policies)
│   ├── container-apps.bicep    (compute environment)
│   ├── functions.bicep         (serverless compute)
│   ├── networking.bicep        (VNet, private endpoints, DNS)
│   ├── monitoring.bicep        (App Insights, Log Analytics)
│   └── identity.bicep          (managed identities, RBAC)
└── scripts/
    └── deploy.sh
```

### Best practices

1. **Use Bicep modules from the Azure Verified Modules (AVM) registry** where available — they encode Well-Architected Framework recommendations.
2. **Parameterise environment differences** (SKU, capacity, network mode) — same templates across dev/staging/prod with different `.bicepparam` files.
3. **Deploy Azure OpenAI model deployments as Bicep resources** — pin model versions and capacity in code, not through the portal.
4. **Use deployment stacks** for lifecycle management — they prevent resource drift and enable clean teardown of entire environments.
5. **Store Bicep modules in a private registry** (Azure Container Registry) for reuse across engagements.

### Bicep MCP Server

Microsoft provides a **Bicep MCP server** that integrates with AI coding assistants (Claude Code, GitHub Copilot, VS Code). It provides:

- Resource type documentation lookup.
- Syntax validation and auto-completion context.
- Best practice recommendations for Bicep authoring.

Use it during development to generate correct Bicep on the first pass.

---

## CI/CD for AI Applications

### Pipeline architecture

```
Code Push → Build & Lint → Unit Tests → Deploy to Dev
  → Integration Tests → Evaluation Run (DeepEval/Azure AI Eval)
    → Gate: metrics pass thresholds?
      → Yes → Deploy to Staging → Smoke Tests → Manual Approval → Deploy to Prod
      → No → Fail pipeline, report metrics
```

### AI-specific CI/CD considerations

1. **Prompt changes are code changes** — store prompts in version control alongside application code. Treat prompt modifications as PRs requiring evaluation.
2. **Evaluation as a gate** — add an evaluation step after integration tests. The pipeline should fail if groundedness drops below threshold or hallucination rate exceeds tolerance.
3. **Model deployment is infrastructure** — model version upgrades should go through the same Bicep deployment pipeline as other infra changes.
4. **Separate data pipelines from application pipelines** — document ingestion (re-indexing AI Search) should be independently deployable.
5. **Blue/green for AI backends** — use APIM traffic splitting to gradually shift traffic to a new model version (10% → 50% → 100%) while monitoring quality metrics.

### GitHub Actions example (evaluation gate)

```yaml
- name: Run AI Evaluation
  run: |
    python -m deepeval test run tests/eval/ \
      --model azure/gpt-4o \
      --threshold groundedness=0.8 \
      --threshold relevancy=0.75
  env:
    AZURE_OPENAI_ENDPOINT: ${{ secrets.AZURE_OPENAI_ENDPOINT }}
```

---

## Copilot Studio and Low-Code Scenarios

### When to use Copilot Studio

| Scenario | Copilot Studio | Pro-Code (Foundry) |
|----------|---------------|---------------------|
| Internal Q&A bot over SharePoint/Dataverse | Yes | Overkill |
| Customer-facing compliance assistant | No | Yes |
| Teams-integrated workflow automation | Yes | Overhead |
| Complex multi-agent reasoning | No | Yes |
| Rapid PoC for stakeholder demo | Yes | Slower |
| Regulated industry with audit requirements | Maybe (with governance) | Yes |

### Copilot Studio best practices

1. **Use Topics for deterministic flows** — don't rely on generative AI for business-critical routing. Define explicit topic triggers for high-stakes intents.
2. **Connect knowledge sources carefully** — SharePoint sites, Dataverse tables, and uploaded files. Test retrieval quality before exposing to users.
3. **Implement DLP policies** — use Power Platform DLP to restrict which connectors the agent can access.
4. **Deploy through managed environments** — use solution-aware agents for ALM (dev → test → prod promotion).
5. **Set up analytics** — Copilot Studio provides conversation analytics. Export to Power BI for deeper analysis.

### Hybrid pattern: Copilot Studio + Foundry Agent Service

For scenarios requiring both low-code convenience and pro-code reasoning:

1. Copilot Studio handles the user interface, conversation management, and Teams deployment.
2. A custom **Power Automate connector** or **Azure Function** bridges to Foundry Agent Service.
3. The Foundry agent performs complex reasoning, tool calling, and multi-source retrieval.
4. Results flow back through Copilot Studio to the user with adaptive cards.

---

## Architecture Patterns by Business Size

### Small business (< 50 users, < 5K queries/day)

**Pattern: Serverless RAG**

```
Static Web Apps (React frontend)
  → Azure Functions (Flex Consumption)
    → Azure OpenAI (GPT-4o-mini, standard deployment)
    → Azure AI Search (Basic tier)
    → Blob Storage (documents)
    → Document Intelligence (S0, pay-per-page)
```

- **Monthly cost:** $150–400 NZD
- **Networking:** Public endpoints with API key auth (acceptable for internal tools) or Entra ID auth for customer-facing.
- **IaC:** Single Bicep template, deployed from GitHub Actions.
- **Evaluation:** Manual + PromptFoo for prompt iteration.

### Medium business (50–500 users, 5K–50K queries/day)

**Pattern: Container Apps RAG with AI Gateway**

```
Static Web Apps (React frontend)
  → APIM (AI Gateway, Standard v2)
    → Container Apps (API layer, Dapr)
      → Azure OpenAI (GPT-4o, PTU + standard spillover)
      → Azure AI Search (Standard S1)
      → Cosmos DB (conversation history, metadata)
      → Document Intelligence (S0)
      → Azure Functions (document ingestion pipeline)
```

- **Monthly cost:** $2,000–5,000 NZD
- **Networking:** VNet integration, private endpoints for data services, APIM in internal mode.
- **IaC:** Modular Bicep with per-environment parameters.
- **Evaluation:** DeepEval in CI/CD + Azure AI Evaluation for production sampling.

### Enterprise (500+ users, 50K+ queries/day, multi-agent)

**Pattern: Enterprise AI Platform**

```
Foundry Hub (managed VNet)
  ├── Foundry Project: knowledge-agents
  │   ├── Foundry IQ (agentic retrieval)
  │   ├── Azure AI Search (Standard S2, multiple indexes)
  │   └── Knowledge agents (specialist per domain)
  ├── Foundry Project: orchestration
  │   ├── MAF agents (Container Apps, multi-replica)
  │   ├── Supervisor agent (planning + delegation)
  │   └── MCP servers (Azure Functions)
  └── Foundry Project: shared-services
      ├── APIM (Premium, multi-region)
      ├── Azure OpenAI (PTU, multi-region)
      ├── Cosmos DB (multi-region, strong consistency)
      └── Application Insights (Agents view)
```

- **Monthly cost:** $10,000–30,000 NZD
- **Networking:** Hub-spoke VNet, Azure Firewall, private DNS zones, ExpressRoute for on-prem connectivity.
- **IaC:** Bicep modules in private registry, deployment stacks, multi-subscription landing zone.
- **Evaluation:** Continuous production evaluation, red-teaming sprints, A/B model testing via APIM traffic splitting.

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails | Better Approach |
|-------------|-------------|-----------------|
| Calling Azure OpenAI directly from the client | No rate limiting, no observability, key exposure | Route through APIM or a backend API |
| Using a single Azure OpenAI deployment for all workloads | One consumer's burst starves others | Separate deployments per workload class + APIM routing |
| Embedding API keys in application config | Key rotation breaks deployments, security risk | Managed Identity everywhere |
| Skipping evaluation ("the demo works") | Quality degrades silently in production | Automated evaluation in CI/CD + production sampling |
| Over-indexing AI Search (dump everything in one index) | Retrieval quality degrades, irrelevant results | Separate indexes by document type/domain, use metadata filters |
| Building custom orchestration when Foundry Agent Service suffices | Maintenance burden, reinventing thread management | Start with managed service, graduate to self-hosted only when you hit limits |
| Using GPT-4o/GPT-5 for every call | 10–30x cost increase for tasks that don't need it | Model routing: classify intent → select appropriate model tier |
| Ignoring prompt caching | Paying full price for repeated system prompts | Structure prompts with stable prefix for automatic caching |
| Deploying without content safety filters | Regulatory risk, reputational damage | Always-on Content Safety + Prompt Shields |
| Manual infrastructure provisioning through the portal | Configuration drift, unreproducible environments | Bicep from day one, even for PoCs |

---

## References

- Microsoft Learn — AI Architecture Guidance for Azure PaaS: https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/ai/platform/architectures
- Microsoft Learn — AI Agent Orchestration Design Patterns: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns
- Microsoft Learn — Azure AI Security Best Practices: https://learn.microsoft.com/en-us/azure/security/fundamentals/ai-security-best-practices
- Microsoft Learn — AI Gateway Capabilities in APIM: https://learn.microsoft.com/en-us/azure/api-management/genai-gateway-capabilities
- Microsoft Learn — Agentic Retrieval Overview (Azure AI Search): https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview
- Microsoft Learn — RAG and Generative AI with Azure AI Search: https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview
- Microsoft Learn — RAG Chunking Phase (Architecture Centre): https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase
- Microsoft Learn — Observability in Microsoft Foundry: https://learn.microsoft.com/en-us/azure/foundry/concepts/observability
- Microsoft Learn — Responsible AI for Microsoft Foundry: https://learn.microsoft.com/en-us/azure/foundry/responsible-use-of-ai-overview
- Microsoft Learn — Foundry Guardrails Overview: https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/foundry/guardrails/guardrails-overview.md
- Microsoft Learn — Provisioned Throughput (PTU) Onboarding: https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/provisioned-throughput-onboarding
- Microsoft Learn — Run Evaluations in Foundry Portal: https://learn.microsoft.com/en-us/azure/foundry/how-to/evaluate-generative-ai-app
- Microsoft Learn — Bicep MCP Server: https://learn.microsoft.com/en-us/azure/azure-resource-manager/bicep/bicep-mcp-server
- Microsoft Learn — Foundry Models Sold by Azure: https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure
- Microsoft Agent Framework 1.0 GA announcement: https://blog.imseankim.com/microsoft-agent-framework-1-0-semantic-kernel-autogen-unified-multi-agent-sdk/
- Azure-Samples/AI-Gateway (reference implementation): https://deepwiki.com/Azure-Samples/AI-Gateway
- Azure AI Bicep Modules (enterprise reference): https://github.com/kbabbington-ms/azure-ai-bicep-modules-kmb
- Enterprise AI on Azure in 2026 — What Actually Changed: https://genioct.be/en/blog/azure-ai-enterprise-architecture-2026/
- The True Cost of AI on Azure — FinOps Deep Dive: https://itnext.io/the-true-cost-of-ai-on-azure-a-finops-deep-dive-into-tokens-ptus-and-the-gen-ai-gateway-505d90148768
- Building Production AI Agents with Microsoft Foundry: https://medium.com/codex/building-production-ai-agents-with-microsoft-foundry-architecture-tools-governance-and-f836560abffd
- Azure Functions MCP Support (InfoQ): https://www.infoq.com/news/2026/01/azure-functions-mcp-support/
- DeepEval — LLM Evaluation Framework: https://deepeval.com/
- OpenTelemetry GenAI Semantic Conventions: https://openobserve.ai/blog/opentelemetry-for-llms/
- LLM Observability with OpenTelemetry + Azure Monitor (GitHub sample): https://github.com/robcamer/llm-observability-otel
- Azure AI Foundry Responsible AI Guardrails Implementation: https://thecloudarchitect.io/en/articles/azure-ai-foundry-responsible-ai-guardrails-a-complete-implementation-guide/
- MCP Best Practices — 12 Rules for Production Deployment: https://apigene.ai/blog/mcp-best-practices
- Azure AI Deployment Patterns (7 patterns with Bicep): https://github.com/KrishnaDistributedcomputing/AzureAIDeployments
