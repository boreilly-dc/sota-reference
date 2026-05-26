# Best Practices for AI Development on Azure in 2026

| Field | Value |
|-------|-------|
| Created | 2026-05-26 |
| Last Updated | 2026-05-27 |
| Version | 1.1 |

---

- [Executive Summary](#executive-summary)
- [Evidence Basis and Status Labels](#evidence-basis-and-status-labels)
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
- [Production Readiness Gaps to Close](#production-readiness-gaps-to-close)
- [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
- [References](#references)

## Executive Summary

Azure's AI platform underwent a major consolidation in 2025-2026. Microsoft now positions **Microsoft Foundry** as the primary platform for building, evaluating, deploying, and monitoring AI apps and agents on Azure. The current Foundry architecture centres on **Foundry resources** and **projects**; older hub-based projects still exist for compatibility, but should not be the default starting point for new pro-code work unless a required feature or migration path depends on them. [1]

Separately, **Microsoft Agent Framework (MAF)** is Microsoft's pro-code SDK for building agents and multi-agent workflows. Use MAF when you need control over workflow, tools, state, and hosting; use Azure AI Foundry Agent Service when the managed service fits the requirements and its current availability/status constraints. [2][3]

For contractors building cloud AI systems, the 2026 Azure landscape has three tiers of engagement:

1. **Pro-code (Foundry + MAF)** — full control over architecture, model selection, retrieval, and orchestration. Suitable for enterprise RAG systems, multi-agent platforms, and bespoke AI applications.
2. **Low-code (Copilot Studio)** — rapid delivery of conversational agents with SharePoint/Dataverse knowledge grounding, deployed into Teams or web channels. Suitable for internal productivity tools and simple Q&A bots.
3. **Hybrid** — Copilot Studio as the user-facing surface, with Azure AI Foundry Agent Service providing the reasoning backend and MCP-based tool integrations.

This guide covers production best practices across all three, with emphasis on the pro-code path where professional services teams spend most of their time.

---

## Evidence Basis and Status Labels

This playbook treats current Microsoft Learn, Azure product documentation, and Microsoft developer documentation as normative. Non-Microsoft blog posts, community examples, and classic/legacy documentation are not used as evidence for the recommendations in this version.

Feature status matters on Azure because Foundry, agents, and GenAI gateway capabilities are evolving quickly. Use these labels when applying the guidance:

| Label | Meaning | Delivery rule |
|-------|---------|---------------|
| **GA** | Generally available in current Microsoft documentation | Suitable for production, subject to standard regional availability and quota checks |
| **Preview** | Public preview or preview-labelled documentation/API | Use behind an explicit risk decision; avoid as a hard dependency for regulated production unless accepted by the client |
| **Classic** | Older hub-based or compatibility path | Use only for migration, compatibility, or a documented feature requirement |
| **Region-dependent** | Availability varies by geography, model, SKU, or resource type | Confirm in the Azure portal, model catalogue, and quota docs before committing to design |

Current status checkpoints to validate before delivery:

| Area | Current playbook stance |
|------|-------------------------|
| Foundry project architecture | Prefer Foundry resource + project for new work; use hub-based projects only when required. [1] |
| Foundry Agent Service | Use when the documented service capabilities and region/model support match the workload; verify project type and model limitations. [3] |
| Azure AI Search agentic retrieval | GA for the stable programmatic API; portal/Foundry experiences and richer synthesis behaviours may differ by API version. [4] |
| APIM GenAI Gateway | Use current `llm-*` policies for token limits, metrics, and semantic cache. [5] |
| Azure Monitor Agents view and Foundry agent monitoring | Treat preview-labelled monitoring surfaces as useful but not contractual for production runbooks. [6] |
| Azure Functions MCP extension | Verify current extension status and hosting constraints before using it as a production MCP surface. [7] |
| GPU on Azure Container Apps | Supported through documented GPU workload profiles; availability and quota are region/SKU dependent. [8] |

---

## Platform Architecture: Microsoft Foundry

Microsoft Foundry is the control plane for AI application and agent development on Azure. In the current architecture, the primary organisational units are:

| Component | Purpose |
|-----------|---------|
| **Foundry resource** | Top-level Azure resource for building AI apps and agents; owns shared configuration for projects. |
| **Foundry project** | Workload boundary for agents, model deployments, evaluation assets, connections, and observability. Use one project per application, customer engagement, or bounded workload. |
| **Model catalogue / Foundry Models** | Region-dependent catalogue for Azure OpenAI and other model providers. Do not hardcode catalogue counts or model availability in architecture decisions. |
| **Azure AI Foundry Agent Service** | Managed agent service with documented support for agents, tools, threads, and model integrations. Validate project type, model, and region support before committing. |
| **Azure AI Search knowledge agents** | Agentic retrieval layer over Azure AI Search for query planning and retrieval against search indexes. |
| **Evaluation and tracing** | Foundry evaluation, tracing, and monitoring surfaces. Treat preview-labelled agent monitoring surfaces as optional accelerators, not the only source of operational truth. |

### Recommended project topology

```
Azure subscription / landing zone
├── Resource group: app-dev
│   └── Foundry resource
│       └── Project: app-dev
├── Resource group: app-staging
│   └── Foundry resource
│       └── Project: app-staging
└── Resource group: app-prod
    └── Foundry resource
        └── Project: app-prod
```

**Best practices:**

- Use a separate Foundry resource/project per environment where isolation, RBAC, networking, quota, or change control needs to differ.
- Use one project per application or customer engagement. Do not overload a single project with unrelated tenants or workloads.
- Use managed identities and Microsoft Entra ID for service-to-service authentication wherever the target service supports it.
- Use managed virtual networks/private endpoints where documented and required for the data classification. Pair private endpoints with private DNS and explicit diagnostic settings.
- Keep hub-based projects out of new designs unless current Microsoft documentation identifies a specific compatibility reason.

---

## Model Selection Strategy

The Azure model catalogue spans Microsoft, Azure OpenAI, and partner/open-source model families, but availability changes by region, deployment type, quota, and commercial status. Selection should be driven by task requirements and measured evaluation results, not brand loyalty or static leaderboard rankings. [9][10]

### Decision matrix pattern

| Use Case | Selection Criteria | Deployment Notes |
|----------|--------------------|------------------|
| Complex reasoning / long-context | Reasoning quality, context length, latency, tool-use support, cost per successful task | Compare current frontier models available in the target region; consider provisioned throughput only after load modelling. |
| High-volume chat / simple Q&A | Cost, latency, answer quality on representative data | Start with the smallest model that passes evaluation thresholds; route only complex requests to larger models. |
| Multimodal (vision + text) | Image/PDF capability, structured extraction accuracy, latency | Validate input size, image limits, and safety filters for the chosen model. |
| Code generation | Task success rate on your repository/tests, tool calling, latency | Use an eval harness tied to unit/integration tests rather than public coding benchmarks alone. |
| Embedding | Retrieval quality, dimensions, index size, multilingual/domain performance | Confirm model availability through Azure OpenAI or Foundry Models and benchmark against your corpus. |
| Edge / on-device | Data residency, offline use, hardware fit, acceptable quality loss | Use Foundry Models/managed compute only when the hosting pattern and licensing fit. |
| Structured extraction | Schema adherence, validation pass rate, retry cost | Prefer structured outputs/tool schemas and downstream validation over prompt-only parsing. |

### Model selection principles

1. **Start with the smallest available model that can meet the eval threshold** — upgrade only when quality, safety, or task success metrics justify the cost.
2. **Use open-source models** (deployed as Serverless API or on Managed Compute) when data sovereignty, cost, or fine-tuning requirements preclude proprietary options.
3. **Pin model versions where the deployment type supports it** — auto-upgrades and retirements can change behaviour, prompts, and evaluation baselines.
4. **Benchmark on your data** — public leaderboard rankings do not predict task-specific performance. Use Foundry Evaluation or an approved evaluation harness to compare models on your actual workload before committing.
5. **Plan for model retirement and regional unavailability** — track the official model retirement/deprecation documentation, maintain a fallback deployment, and rehearse rollback before production changes. [10]

---

## RAG and Knowledge Retrieval

### Azure AI Search — the default retrieval layer

Azure AI Search is the default managed retrieval engine for RAG on Azure when the workload needs keyword search, vector search, semantic ranking, integrated vectorisation, and index-level governance. [11][12]

| Paradigm | When to Use |
|----------|-------------|
| **Hybrid search** (BM25 + vector + optional semantic ranker) | Standard RAG with pre-indexed content. Predictable, well-understood. |
| **Agentic retrieval** (Azure AI Search knowledge agents) | Agent-driven scenarios requiring query planning and retrieval over one or more search indexes. |
| **Direct vector search** | Embedding-only workloads where keyword matching adds no value (e.g., image similarity). |

### Hybrid search configuration

For most enterprise RAG deployments, hybrid search with semantic ranker is the correct starting point:

```
Query → BM25 (keyword) + Vector (embedding) → RRF fusion → Semantic Ranker → Top-K chunks → LLM
```

**Best practices:**

- **Enable semantic ranker when quality matters and the service tier supports it** — it reranks eligible text results after retrieval and is especially useful for answer generation and captions. [13]
- **Choose embeddings through evaluation** — Azure OpenAI embedding models are a strong default on Azure, but dimensions, language support, latency, and index size should be measured against your corpus. [9][12]
- **Use integrated vectorisation when it simplifies operations** — AI Search can vectorise content during indexing and vectorise queries at query time through configured vectorizers. [12]
- **Tune `k`, `top`, filters, and semantic ranker inputs** — `k=50` and final `top=5-10` are starting points, not universal defaults.
- **Use metadata filters** to scope retrieval (by document type, customer, date range) — this prevents irrelevant results and reduces token spend.

### Agentic retrieval (Azure AI Search knowledge agents)

Agentic retrieval is the Azure AI Search pattern for agent-to-search interaction. The stable `2026-04-01` API provides the GA programmatic contract for knowledge-agent retrieval; preview API versions and portal experiences may expose additional behaviours and should be labelled accordingly in delivery designs. [4]

- A **knowledge agent** defines retrieval instructions and targets Azure AI Search indexes.
- The retrieval process can plan and execute subqueries for complex information needs.
- Treat generated answers/synthesis as the responsibility of the calling agent or application unless the selected API/service contract explicitly provides synthesis.

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

Document Intelligence is the default Azure service for extracting structure from documents before RAG indexing. The prebuilt **Layout** model supports text, tables, selection marks, document structure, and Markdown output in the current v4.0 documentation. [14]

**Best practices:**

1. **Use the Layout model with markdown output** — it preserves document structure (headings, tables, lists) that downstream chunking can exploit.
2. **Enable figure extraction** for documents with diagrams or charts — extracted figures can be processed by a vision model for captioning.
3. **Use the current GA API for new builds** — v4.0 (`2024-11-30`) is the current GA line in the official docs at the time of this update. [14]
4. **Process asynchronously** — use Azure Functions with a Storage Queue trigger for batch ingestion, and size concurrency from the current Document Intelligence quota/limit documentation rather than hardcoded concurrency assumptions. [15]
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
- For large corpora, partition work through queues and orchestration, respect documented rate limits, and test throughput in the target region/SKU before committing to ingestion SLAs.

---

## Agent Development

### Microsoft Agent Framework (MAF) 1.0

MAF is Microsoft's pro-code SDK direction for building AI agents and multi-agent workflows in .NET and Python. Use it when you need explicit control of orchestration, state, tools, testing, and hosting. [2]

**Key concepts:**

| Concept | Description |
|---------|-------------|
| **Agent** | A unit of work with a system prompt, tools, and an LLM backend. |
| **Thread / conversation state** | State boundary for a multi-turn agent interaction. |
| **Tool** | Function, API, MCP server, or connector exposed to the agent under explicit policy. |
| **Workflow** | Orchestration pattern for sequential, concurrent, handoff, or supervisor-style work. |
| **MCP Server** | Standard protocol surface for tools. Azure Functions has an MCP extension; verify its current status and hosting support before using it as a production dependency. [7] |

### Agent orchestration patterns

Microsoft Agent Framework documents these built-in orchestration patterns: [28]

1. **Sequential** — agents execute in a fixed pipeline (e.g., extract → validate → enrich).
2. **Concurrent** — multiple agents work in parallel on different sub-tasks, results are merged.
3. **Handoff** — a router agent delegates to specialist agents based on intent classification.
4. **Group chat** — agents collaborate iteratively on a shared problem (useful for code review, planning).
5. **Magentic (supervisor)** — a supervisor agent plans, delegates, and evaluates sub-agent outputs.

### Best practices for production agents

1. **Keep agents single-purpose** — one agent, one job. Compose complex behaviours through orchestration, not monolithic prompts.
2. **Use MCP for tool interfaces where it fits** — MCP provides a standard, testable tool surface. Azure Functions can host MCP-style tools through its documented extension, but preview/hosting constraints must be checked in the current docs. [7]
3. **Implement OBO (On-Behalf-Of) token flow** for tools that access user-scoped data (CRM, SharePoint, databases). The agent acts with the user's identity, not a service principal with broad access. [29]
4. **Set token budgets and timeout limits** per agent — prevent runaway reasoning loops from consuming PTU capacity or blocking other requests.
5. **Log every tool invocation and LLM call** with OpenTelemetry spans — you cannot debug agents without observability.
6. **Use Foundry Agent Service for managed hosting** when you do not need fine-grained control over runtime internals and the documented project/model/tool support fits the workload. [3]

### When to use Foundry Agent Service vs. self-hosted MAF

| Criterion | Foundry Agent Service | Self-hosted (Container Apps / AKS) |
|-----------|----------------------|-------------------------------------|
| Time to production | Lower operational setup | Higher operational setup |
| Custom orchestration logic | Limited (sequential, tool-calling) | Full control |
| Multi-agent workflows | Basic (single agent with tools) | Complex (group chat, handoff, supervisor) |
| Networking | Depends on Foundry project/resource capabilities | Any supported app networking topology |
| Cost at scale | Managed service cost profile | Own compute and operations cost profile |
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

1. **Use backend pools with priority-based routing** — send baseline traffic to provisioned or primary deployments, then fall back to standard or secondary deployments through APIM policy and backend configuration. [5]
2. **Emit token metrics to Application Insights** — use the current `llm-emit-token-metric` policy for per-consumer usage and cost attribution. [5]
3. **Apply `llm-token-limit`** per product, subscription, consumer, or other APIM expression to enforce budget boundaries. [5]
4. **Enable semantic caching selectively** with `llm-semantic-cache-lookup` and `llm-semantic-cache-store` for workloads where approximate cache hits are acceptable. Validate correctness before using it for regulated or high-stakes answers. [5]
5. **Deploy APIM according to the network boundary** — internal mode, private endpoints, and VNet integration are appropriate for private workloads, but the exact topology should follow the landing-zone/networking design rather than the old hub-only pattern.

---

## Hosting and Compute Patterns

### Decision matrix

| Workload Characteristic | Recommended Service | Rationale |
|------------------------|---------------------|-----------|
| Event-driven, short-lived work | Azure Functions (Flex Consumption) | Scale-to-zero, pay-per-execution, VNet integration, higher resource options than classic Consumption |
| Long-running agent sessions / containerised services | Azure Container Apps | Scale-to-zero, Dapr integration, jobs, and documented GPU workload profiles where available |
| High-throughput API with steady traffic | Azure App Service (P1v3+) | Predictable cost, simple deployment |
| Complex multi-container, custom networking | AKS | Full control, KEDA autoscaling |
| Frontend (React/Next.js) | Azure Static Web Apps | Global CDN, integrated auth, API backend |

### Azure Functions for AI workloads (2026 patterns)

- **Flex Consumption plan** is the default starting point for serverless AI workloads that need VNet integration, larger instance sizes, and scale-to-zero. Confirm timeout, memory, regional, and language-stack limits in the current Functions docs. [16]
- **MCP extension** — Azure Functions can expose MCP tools through the documented extension. Confirm current extension status, authentication model, and hosting limitations before relying on it for production tool surfaces. [7]
- **Durable Functions** for orchestrating multi-step AI pipelines (document ingestion, batch evaluation, multi-agent workflows).

### Azure Container Apps for AI workloads

- **GPU workloads** — Container Apps supports GPU-enabled workload profiles in documented regions/SKUs for model inference and AI workloads. [8]
- **Scale-to-zero** with custom KEDA scalers — scale on queue depth, HTTP concurrency, or custom metrics.
- **Dapr integration** for service-to-service communication in multi-agent architectures.
- **Jobs** (scheduled or event-triggered) for batch processing, evaluation runs, and data pipelines.

---

## Security and Networking

### Network architecture

Production Azure AI deployments should use landing-zone-aligned networking, commonly a hub-spoke VNet topology with private endpoints for regulated or private workloads. [25][26]

```
Hub VNet (shared services)
├── Azure Firewall / NVA
├── Bastion
└── APIM (internal mode)

Spoke VNet (AI workload)
├── Foundry resource / workload services
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

### Governance controls

Production Azure AI landing zones should include:

- Azure Policy assignments for required tags, allowed regions, diagnostic settings, public network access restrictions, private endpoint usage, and approved SKUs. [25][26]
- Management group and subscription boundaries that separate dev/test, production, regulated data, and shared platform services.
- Budgets and cost alerts at subscription/resource group level, plus token-level usage metrics from APIM or application telemetry.
- Defender for Cloud, Service Health alerts, and Log Analytics retention policies aligned to the client's compliance requirements.

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

Azure AI Content Safety and Azure OpenAI safety systems provide guardrails at multiple levels. Apply them as defence-in-depth, not as a replacement for application policy, retrieval scoping, and adversarial testing. [17][18]

| Layer | Capability | Configuration |
|-------|-----------|---------------|
| **Input filtering** | Hate, violence, sexual, self-harm detection | Severity thresholds (0–6) per category |
| **Prompt Shields** | Jailbreak detection, indirect injection detection | Enable on all user-facing deployments |
| **Groundedness detection** | Detect claims not supported by provided grounding context where the feature is available | Use in evaluation pipelines and optionally at runtime after latency/cost testing |
| **Protected material detection** | Reduce risk of unwanted protected material output where supported by the chosen model/service | Enable for public-facing applications when available |
| **Custom categories / blocklists** | Domain-specific blocked terms, classes, or phrases | Per-application configuration |

### Implementation best practices

1. **Layer your defences** — Content Safety at the APIM gateway (pre-request), built-in filters on the Azure OpenAI deployment, and application-level validation of outputs.
2. **Set severity thresholds conservatively** for customer-facing applications — start at threshold 2 (block medium and above) and relax only if false positive rate is unacceptable.
3. **Always enable Prompt Shields** — jailbreak attempts are common in any public-facing LLM application.
4. **Use groundedness detection in evaluation** — run it against your test dataset to measure hallucination rate before go-live.
5. **Log all blocked requests** — content safety blocks should trigger alerts for security review, not just silent drops.
6. **Document your responsible AI posture** — maintain a Responsible AI Impact Assessment, safety evaluation results, known limitations, human escalation paths, and monitoring plan per engagement.

---

## Cost Management

### Pricing tiers

| Deployment Type | Best For | Pricing Model |
|-----------------|----------|---------------|
| **Standard (pay-per-token)** | Variable/unpredictable workloads, development, PoC | Per 1M tokens (input/output priced separately) |
| **Provisioned Throughput (PTU)** | Steady-state production with predictable volume | Hourly capacity reservation; effective throughput varies by model, latency target, prompt shape, and output length |
| **Global deployment** | Higher availability/throughput where global routing is acceptable | Check current model/deployment documentation and data handling constraints. [30] |
| **Data Zone deployment** | Data residency requirements for supported zones | Check current model/deployment documentation and regional availability. [30] |

### Cost optimisation strategies

1. **Right-size your model** — use smaller/lower-cost models for classification, routing, extraction, and simple Q&A when evaluation confirms they meet the quality threshold. Reserve larger reasoning models for tasks that need them.
2. **Prompt caching** — Azure OpenAI supports prompt caching for eligible requests/models. Structure long stable prefixes so they can benefit from caching, but validate cache eligibility and pricing in current docs. [19]
3. **PTU with consumption spillover** — provision PTUs for your baseline load (~P50), use standard consumption for burst traffic via APIM backend pool priority routing.
4. **Semantic caching at the gateway** — APIM can cache semantically similar queries. Effective for FAQ-style applications.
5. **Batch API for non-latency-sensitive work** — evaluation runs, bulk summarisation, and data enrichment jobs should use batch processing where the selected model/API supports it and the pricing model is favourable.
6. **Monitor token waste** — track input/output token ratios. If output tokens consistently exceed input, your prompts may be too open-ended. If input tokens are very high relative to output, you may be over-stuffing context.

### Cost estimation workflow

1. Estimate requests/day, input tokens, output tokens, cacheable prefix size, retrieval calls, document pages, and search index/storage size.
2. Price the workload with the current Azure Pricing Calculator and target-region pricing pages.
3. Run a load test with representative prompts and outputs; PTU sizing must be benchmarked because throughput is model- and workload-dependent. [20]
4. Add APIM, Azure AI Search, Document Intelligence, storage, Cosmos DB, Container Apps/Functions, monitoring, private networking, and Log Analytics ingestion/retention.
5. Set budgets and alerts before production launch, then reconcile token metrics against Azure Cost Management weekly during the first month.

---

## Observability and Monitoring

### OpenTelemetry-native stack

Azure AI observability should be based on OpenTelemetry traces/metrics, Azure Monitor, Application Insights, and Log Analytics. Use Foundry tracing and agent monitoring where supported, but keep application-owned telemetry so production runbooks are not dependent on preview-only portal views. [6][21]

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
| **Metrics** | Token usage (input/output/cached), latency (P50/P95/P99), error rate, cache hit ratio | APIM `llm-emit-token-metric` + custom metrics |
| **Logs** | Content safety blocks, evaluation failures, tool errors, model version changes | Structured logging to Log Analytics |
| **Evaluations** | Groundedness, relevance, citation accuracy — sampled from production traffic | Azure AI Evaluation SDK or approved evaluation harness |

### Application Insights Agents view

The **Agents view** in Application Insights provides a purpose-built dashboard for AI agent workloads, but current Microsoft documentation labels this surface as preview. Use it for diagnosis and visibility, while keeping durable traces, metrics, logs, and alerts in Azure Monitor/Log Analytics. [6]

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
| **Azure AI Evaluation SDK** (`azure-ai-evaluation`) | Native Foundry integration and documented built-in evaluators | Default choice for Azure-native teams |
| **Provider-agnostic test harness** | Custom metrics, local/CI execution, multi-cloud portability | When Microsoft-native evaluation does not cover the required metric or workflow |
| **Prompt/model comparison harness** | A/B prompt testing and model comparison | Prompt iteration and migration testing |

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
3. **Use LLM-as-judge for subjective metrics carefully** — choose a documented evaluator/model, pin the judge model version, and record evaluator configuration with every run. [22]
4. **Red-team before launch** — test adversarial inputs (prompt injection, jailbreaks, off-topic queries) as a dedicated evaluation pass.
5. **Track metrics over time** — quality regressions are gradual. Dashboard your evaluation scores and alert on downward trends.

---

## Infrastructure as Code

### Bicep as the default

Bicep is the first-class Azure-native IaC language. Use it when you want direct alignment with Azure Resource Manager, Azure Verified Modules, deployment stacks, and current Azure resource schemas. Validate newly released AI resource types against current Bicep/ARM documentation before assuming full schema coverage. [23][24]

### Module structure for AI deployments

```
infra/
├── main.bicep              (orchestrator)
├── parameters/
│   ├── dev.bicepparam
│   ├── staging.bicepparam
│   └── prod.bicepparam
├── modules/
│   ├── foundry-resource.bicep  (Microsoft Foundry resource)
│   ├── foundry-project.bicep   (per-app/project resource)
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

1. **Use Bicep modules from the Azure Verified Modules (AVM) registry** where available — they provide Microsoft-supported reusable modules aligned to Azure best practices. [23]
2. **Parameterise environment differences** (SKU, capacity, network mode) — same templates across dev/staging/prod with different `.bicepparam` files.
3. **Deploy Azure OpenAI model deployments as Bicep resources** — pin model versions and capacity in code, not through the portal.
4. **Use deployment stacks** for lifecycle management — they provide managed resource lifecycle operations for resources deployed by the stack. [24]
5. **Store Bicep modules in a private registry** (Azure Container Registry) for reuse across engagements.

### Bicep MCP Server

Microsoft provides a **Bicep MCP server** that integrates with AI coding assistants (Claude Code, GitHub Copilot, VS Code). It provides: [27]

- Resource type documentation lookup.
- Syntax validation and auto-completion context.
- Best practice recommendations for Bicep authoring.

Use it during development to generate correct Bicep on the first pass.

---

## CI/CD for AI Applications

### Pipeline architecture

```
Code Push → Build & Lint → Unit Tests → Deploy to Dev
  → Integration Tests → Evaluation Run (Azure AI Evaluation or approved harness)
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
    python scripts/run_ai_evaluation.py \
      --dataset tests/eval/golden.jsonl \
      --threshold groundedness=0.8 \
      --threshold relevance=0.75
  env:
    AZURE_AI_PROJECT_ENDPOINT: ${{ secrets.AZURE_AI_PROJECT_ENDPOINT }}
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
    → Document Intelligence (pay-per-page tier)
```

- **Cost posture:** Usually consumption-first; validate with Pricing Calculator and a representative load test.
- **Networking:** Public endpoints may be acceptable for low-risk internal PoCs; production customer-facing systems should prefer Entra ID, private networking, and explicit data-classification review.
- **IaC:** Single Bicep template, deployed from GitHub Actions.
- **Evaluation:** Manual review plus the approved evaluation harness for prompt iteration.

### Medium business (50–500 users, 5K–50K queries/day)

**Pattern: Container Apps RAG with AI Gateway**

```
Static Web Apps (React frontend)
  → APIM (AI Gateway, Standard v2)
    → Container Apps (API layer, Dapr)
      → Azure OpenAI (GPT-4o, PTU + standard spillover)
      → Azure AI Search (Standard S1)
      → Cosmos DB (conversation history, metadata)
      → Document Intelligence
      → Azure Functions (document ingestion pipeline)
```

- **Cost posture:** Usually APIM + standard/provisioned model mix; validate PTU only after baseline load is measurable.
- **Networking:** VNet integration, private endpoints for data services, APIM in internal mode.
- **IaC:** Modular Bicep with per-environment parameters.
- **Evaluation:** CI/CD evaluation gates plus Azure AI Evaluation for production sampling where supported.

### Enterprise (500+ users, 50K+ queries/day, multi-agent)

**Pattern: Enterprise AI Platform**

```
Foundry resource / project estate
  ├── Foundry Project: knowledge-agents
  │   ├── Azure AI Search knowledge agents
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
      └── Application Insights / Log Analytics
```

- **Cost posture:** Usually provisioned/model-capacity planning plus explicit quota, budget, and chargeback governance.
- **Networking:** Hub-spoke VNet, Azure Firewall, private DNS zones, ExpressRoute for on-prem connectivity.
- **IaC:** Bicep modules in private registry, deployment stacks, multi-subscription landing zone.
- **Evaluation:** Continuous production evaluation, red-teaming sprints, A/B model testing via APIM traffic splitting.

---

## Production Readiness Gaps to Close

Use this section as the delivery checklist for turning the reference architecture into an implementation plan.

### Reliability and disaster recovery

- Define RTO/RPO for the application, search indexes, conversation state, document source data, and telemetry.
- Confirm regional availability for every model, SKU, APIM tier, Azure AI Search tier, Document Intelligence region, Container Apps workload profile, and private networking feature.
- Implement retry/backoff for model throttling and transient failures; fail over through APIM backend pools only after validating semantic compatibility of the fallback model.
- Rehearse model retirement, model rollback, APIM backend failure, AI Search outage, quota exhaustion, and document-ingestion backlog scenarios. [5][10][20]

### Data residency, privacy, and retention

- Choose global, data-zone, or regional model deployments based on client data residency requirements and official deployment-type documentation. [30]
- Define what prompt, response, retrieval context, file content, trace, and evaluation data is logged; redact or suppress sensitive content where logs are not approved for that data class.
- Configure Log Analytics retention, storage lifecycle policies, CMK requirements, and purge workflows before production launch.
- Treat retrieved context and semantic-cache entries as sensitive tenant data; include `tenant_id`, data classification, and retention policy in cache/index design.

### Governance and operations

- Add Azure Policy, Defender for Cloud, budgets, diagnostic settings, Service Health alerts, and tagging conventions to the landing-zone design.
- Maintain an operational calendar for model retirement, quota renewals, certificate/key rotation, evaluation dataset refresh, and dependency updates.
- Create runbooks for content-safety escalations, evaluation regressions, cost anomalies, 429 spikes, private endpoint/DNS failures, and failed document-ingestion batches.
- Store architecture decision records for preview feature adoption, model selection, safety thresholds, retention settings, and fallback behaviour. [18][21][25]

### Implementation artifacts still needed

- Bicep modules for Foundry resources/projects, Azure OpenAI deployments, AI Search, APIM GenAI policies, networking/private DNS, monitoring, identities/RBAC, Functions, and Container Apps.
- APIM policy snippets for `llm-token-limit`, `llm-emit-token-metric`, semantic cache, backend failover, and content-safety routing.
- CI/CD templates for infrastructure deployment, app deployment, prompt/model evaluation gates, and staged APIM traffic shifts.
- Evaluation seed datasets, adversarial test cases, model comparison reports, and production-sampling dashboards.

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
| Using the largest model for every call | Cost and capacity are consumed by tasks that do not need frontier reasoning | Model routing: classify intent → select appropriate model tier |
| Ignoring prompt caching | Paying full price for repeated system prompts | Structure prompts with stable prefix for automatic caching |
| Deploying without content safety filters | Regulatory risk, reputational damage | Always-on Content Safety + Prompt Shields |
| Manual infrastructure provisioning through the portal | Configuration drift, unreproducible environments | Bicep from day one, even for PoCs |

---

## References

All normative references in this version are official Microsoft/Azure documentation. Classic/legacy docs and non-Microsoft posts are intentionally excluded.

1. Microsoft Learn — Microsoft Foundry architecture: https://learn.microsoft.com/en-us/azure/foundry/concepts/architecture
2. Microsoft Learn — Microsoft Agent Framework: https://learn.microsoft.com/en-us/agent-framework/overview/
3. Microsoft Learn — Microsoft Foundry Agent Service: https://learn.microsoft.com/en-us/azure/foundry/agents/overview
4. Microsoft Learn — Agentic retrieval in Azure AI Search: https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview
5. Microsoft Learn — AI Gateway capabilities in Azure API Management: https://learn.microsoft.com/en-us/azure/api-management/genai-gateway-capabilities
6. Microsoft Learn — Application Insights Agents view: https://learn.microsoft.com/en-us/azure/azure-monitor/app/agents-view
7. Microsoft Learn — Azure Functions MCP extension: https://learn.microsoft.com/en-us/azure/azure-functions/functions-bindings-mcp
8. Microsoft Learn — Azure Container Apps workload profiles: https://learn.microsoft.com/en-us/azure/container-apps/workload-profiles-overview
9. Microsoft Learn — Foundry Models sold by Azure: https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure
10. Microsoft Learn — Foundry Models lifecycle and support policy: https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-retirements
11. Microsoft Learn — Hybrid search in Azure AI Search: https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview
12. Microsoft Learn — Integrated vectorization in Azure AI Search: https://learn.microsoft.com/en-us/azure/search/vector-search-integrated-vectorization
13. Microsoft Learn — Semantic ranking in Azure AI Search: https://learn.microsoft.com/en-us/azure/search/semantic-search-overview
14. Microsoft Learn — Document Intelligence Layout model: https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/layout
15. Microsoft Learn — Azure AI services quotas and limits: https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/service-limits
16. Microsoft Learn — Azure Functions Flex Consumption plan: https://learn.microsoft.com/en-us/azure/azure-functions/flex-consumption-plan
17. Microsoft Learn — Azure AI Content Safety Prompt Shields: https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/prompt-shields
18. Microsoft Learn — Responsible AI for Microsoft Foundry: https://learn.microsoft.com/en-us/azure/foundry/responsible-use-of-ai-overview
19. Microsoft Learn — Azure OpenAI prompt caching: https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/prompt-caching
20. Microsoft Learn — Provisioned throughput onboarding: https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/provisioned-throughput-onboarding
21. Microsoft Learn — Observability in Microsoft Foundry: https://learn.microsoft.com/en-us/azure/foundry/concepts/observability
22. Microsoft Learn — Azure AI evaluation evaluators: https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/general-purpose-evaluators
23. Microsoft Learn — Azure Verified Modules: https://learn.microsoft.com/en-us/community/content/azure-verified-modules
24. Microsoft Learn — Azure deployment stacks: https://learn.microsoft.com/en-us/azure/azure-resource-manager/bicep/deployment-stacks
25. Microsoft Learn — Azure AI security best practices: https://learn.microsoft.com/en-us/azure/security/fundamentals/ai-security-best-practices
26. Microsoft Learn — AI platform architecture in Cloud Adoption Framework: https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/ai/platform/architectures
27. Microsoft Learn — Bicep MCP server: https://learn.microsoft.com/en-us/azure/azure-resource-manager/bicep/bicep-mcp-server
28. Microsoft Learn — Agent Framework orchestration patterns: https://learn.microsoft.com/en-us/agent-framework/workflows/orchestrations/
29. Microsoft Learn — Microsoft identity platform OAuth 2.0 on-behalf-of flow: https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-on-behalf-of-flow
30. Microsoft Learn — Azure OpenAI deployment types: https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/deployment-types
