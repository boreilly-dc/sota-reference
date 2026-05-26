# Best Practices for AI Development on Google Cloud in 2026

| Field | Value |
|-------|-------|
| Created | 2026-05-26 |
| Last Updated | 2026-05-27 |
| Official Docs Verified | 2026-05-26 |
| Version | 1.1 |

---

- [Executive Summary](#executive-summary)
- [Evidence Basis and Status Labels](#evidence-basis-and-status-labels)
- [Platform Architecture: Gemini Enterprise Agent Platform](#platform-architecture-gemini-enterprise-agent-platform)
- [Model Selection Strategy](#model-selection-strategy)
- [RAG and Knowledge Retrieval](#rag-and-knowledge-retrieval)
- [Document Processing Pipelines](#document-processing-pipelines)
- [Agent Development](#agent-development)
- [AI Gateway and Traffic Management](#ai-gateway-and-traffic-management)
- [Hosting and Compute Patterns](#hosting-and-compute-patterns)
- [Security and Networking](#security-and-networking)
- [Responsible AI and Content Safety](#responsible-ai-and-content-safety)
- [Cost Management](#cost-management)
- [Observability and Monitoring](#observability-and-monitoring)
- [Evaluation and Testing](#evaluation-and-testing)
- [Infrastructure as Code](#infrastructure-as-code)
- [CI/CD for AI Applications](#cicd-for-ai-applications)
- [Agent Studio and Low-Code Scenarios](#agent-studio-and-low-code-scenarios)
- [Architecture Patterns by Business Size](#architecture-patterns-by-business-size)
- [Implementation Gaps to Fill](#implementation-gaps-to-fill)
- [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
- [References](#references)

## Executive Summary

Google Cloud has consolidated agent development under the **Gemini Enterprise Agent Platform** while **Vertex AI remains an active Google Cloud platform and API surface** for machine learning and generative AI. Treat Agent Platform as the current agent-building control plane, and treat Vertex AI resource names and APIs as still relevant where the official docs and Terraform provider use them. [1][2]

The most significant current platform developments:

1. **Gemini Enterprise Agent Platform** — current Google Cloud surface for agent development, deployment, evaluation, and governance. [1]
2. **Agent Development Kit (ADK)** — open-source, code-first framework for agents. Current ADK documentation lists Python, TypeScript, Go, Java, and Kotlin support; confirm maturity per language SDK before production use. [3]
3. **Agent Runtime** — managed runtime for deploying and scaling agents. Some adjacent services and APIs are preview-labelled, so production designs need a feature-by-feature launch-stage check. [4][5]
4. **RAG Engine** — managed retrieval infrastructure with mode-specific trade-offs: Serverless mode is public preview and Spanner mode is the more mature stateful option for production RAG that needs tighter control. [6][7]
5. **Model Armor** — prompt/response screening, prompt injection and jailbreak detection, sensitive data protection, and floor settings for central policy enforcement. [8]
6. **Remote MCP servers and MCP Toolbox** — official Google Cloud paths for connecting agents to tools and databases, including BigQuery and Cloud SQL patterns. [9][10]
7. **Model Garden and partner models** — Gemini, Gemma, and selected partner models remain accessible through current Google Cloud model surfaces, with availability and launch stage varying by model and region. [11][12]

For contractors building cloud AI systems, the GCP landscape has three tiers of engagement:

1. **Pro-code (Agent Platform + ADK)** — full control over architecture, model selection, retrieval, orchestration, and deployment. Suitable for enterprise RAG systems, multi-agent platforms, and bespoke AI applications.
2. **Low-code (Agent Studio)** — visual builder for conversational agents with enterprise data grounding, deployed to web, mobile, or internal channels. Suitable for internal productivity tools and Q&A applications.
3. **Hybrid** — Agent Studio for user-facing surfaces with ADK-built agents providing reasoning backends, connected via A2A protocol and MCP tool integrations.

This guide covers production best practices across all three, with emphasis on the pro-code path where professional services teams spend most of their time.

---

## Evidence Basis and Status Labels

This playbook uses current official documentation from Google Cloud, Google AI, Terraform Registry, and Anthropic's Claude-on-Vertex documentation as normative sources. Third-party blogs and community guides are intentionally excluded from the reference list.

Feature status matters because the Agent Platform is evolving rapidly from the Vertex AI migration. Use these labels:

| Label | Meaning | Delivery rule |
|-------|---------|---------------|
| **GA** | Generally available in current documentation | Suitable for production |
| **Preview** | Public preview or preview-labelled API | Use behind an explicit risk decision; avoid as a hard dependency for regulated production |
| **Legacy** | Explicitly legacy-labelled documentation or product surface | Use only for migration; plan to move off |
| **Region-dependent** | Availability varies by geography, model, or TPU type | Confirm in the console before committing to design |
| **Pre-GA** | Marked by Google as Preview, Experimental, or otherwise subject to Pre-GA terms | Avoid as a hard dependency for regulated production unless the risk is accepted |

Current status checkpoints:

| Area | Current stance |
|------|---------------|
| Gemini Enterprise Agent Platform | Current official agent-building platform surface. [1] |
| Vertex AI | Still active; do not describe it as retired. Many Agent Platform resources still use Vertex AI API or Terraform names. [2][20] |
| Agent Runtime | Current managed runtime name in official docs; check adjacent capabilities individually because some are preview-labelled. [4][5] |
| ADK | Official ADK docs list Python, TypeScript, Go, Java, Kotlin; confirm maturity per language SDK because product pages can lag SDK announcements. [3] |
| Model Garden | Current model catalogue; model availability and launch stage are region- and model-dependent. [11] |
| Gemini 3.1 Pro | Pre-GA in official docs and currently documented with a 1M token context window. [12] |
| RAG Engine Serverless mode | Public preview; important limitations include regional and security-feature constraints. [6] |
| Model Armor | Current official security service for prompt and response screening. [8] |
| Remote MCP servers / MCP Toolbox | Official integration pattern for agents and database tools; exact service coverage should be confirmed per server. [9][10] |

---

## Platform Architecture: Gemini Enterprise Agent Platform

The Gemini Enterprise Agent Platform is the current Google Cloud control plane for agent development. It sits alongside active Vertex AI APIs and resources rather than replacing every Vertex AI surface. Use the current Agent Platform docs for agent lifecycle decisions, and use current Vertex AI docs where the service, SDK, or Terraform resource remains Vertex AI-named. [1][2]

| Component | Purpose |
|-----------|---------|
| **Model Garden** | 200+ foundation models: Gemini family, Gemma (open-source), Anthropic Claude, Meta Llama, Mistral, and others |
| **Agent Development Kit (ADK)** | Open-source code-first SDK for building agents. Official language support: Python, TypeScript, Go, Java, Kotlin |
| **Agent Studio** | Visual low-code agent builder for rapid prototyping and simple use cases |
| **Agent Runtime** | Managed runtime for deploying and scaling agents; some adjacent runtime capabilities are preview-labelled |
| **Memory Bank** | Managed long-term memory service for context-aware agent interactions across sessions |
| **Sessions** | Multi-day persistent agent sessions with state management |
| **RAG Engine / Search** | Managed retrieval, indexing, and grounding surfaces; mode and feature support varies |
| **Evaluation Suite** | Model and agent evaluation: automated benchmarks, trajectory evaluation, and human-in-the-loop scoring |
| **Model Armor** | Content safety, prompt injection detection, and responsible AI guardrails |
| **Agent Garden** | Prebuilt agents and templates; check launch stage before production dependency |

### Recommended project topology

```
Google Cloud Organisation
├── Folder: ai-workloads
│   ├── Project: ai-dev
│   │   ├── Agent Platform (development agents + Model Garden access)
│   │   ├── Cloud Storage (document corpus, training data)
│   │   ├── Cloud Run (development compute)
│   │   └── AlloyDB / Cloud SQL (operational data + vector embeddings)
│   ├── Project: ai-staging
│   │   └── Mirror of production with lower capacity
│   └── Project: ai-prod
│       ├── Agent Platform (production agents)
│       ├── Agent Runtime (managed agent runtime)
│       ├── RAG Engine / Search (enterprise retrieval)
│       ├── Cloud Run (application layer)
│       ├── AlloyDB (production vector store + operational data)
│       ├── VPC with Private Service Connect
│       └── Cloud Armor (WAF + DDoS protection)
├── Folder: shared-services
│   ├── Project: networking-hub
│   │   └── Shared VPC host project
│   ├── Project: security
│   │   └── Security Command Center, VPC-SC perimeters, KMS
│   └── Project: observability
│       └── Cloud Monitoring, Cloud Trace, Cloud Logging (centralised)
```

**Best practices:**

- Use **separate GCP projects** per environment (dev/staging/prod) within an organisational folder — this provides hard isolation for IAM, networking, and billing attribution.
- Deploy AI resources in the **same region as your compute** to minimise latency. Use Gemini's global endpoint for automatic routing when latency tolerance allows.
- Enable **model invocation logging** at the project level from day one — logs to Cloud Logging for audit, debugging, and evaluation.
- Use **service accounts** (not user credentials) for all service-to-service authentication. Follow the principle of least privilege with fine-grained IAM roles.
- Configure **organisation policies** to restrict which models and APIs can be used in production projects.
- Use **Shared VPC** for network centralisation across environments — AI projects use the shared network, networking team manages firewall rules and Private Service Connect.

---

## Model Selection Strategy

The Model Garden hosts 200+ models spanning Google's own Gemini and Gemma families, third-party frontier models (Anthropic Claude, Meta Llama, Mistral), and task-specific models for embeddings, vision, and speech.

### Decision matrix

| Use Case | Recommended Model | Alternative (Open-Source) | Notes |
|----------|-------------------|---------------------------|-------|
| Complex reasoning / long-context | Gemini 3.1 Pro where Pre-GA is acceptable; otherwise use the latest GA Gemini Pro-class model available in the target region | Claude on Vertex where available | Gemini 3.1 Pro is Pre-GA and documented with a 1M token context window; do not treat it as a default regulated-production dependency. [12] |
| High-volume chat / simple Q&A | Current Gemini Flash-class GA model available in region | Gemma on self-managed GKE / Vertex AI where supported | Use current pricing and model cards at design time; do not hard-code a stale Flash generation. [13] |
| Multimodal (vision + text) | Current Gemini multimodal model available in region | Partner/open models where supported | Confirm input modalities, launch stage, and regional availability per model. [11][12] |
| Code generation | Gemini Pro-class model after evaluation | Claude on Vertex where available | Choose by measured quality on the client codebase, not provider defaults. [12][14] |
| Embedding | `gemini-embedding-001` for text; `gemini-embedding-2` where multimodal embeddings are required and available | Existing `text-embedding-005` indexes during migration | Google documents `gemini-embedding-001` as the unified successor to earlier specialised embedding models such as `text-embedding-005`; Google AI documents Gemini Embedding 2 for multimodal embedding use cases. [15][16] |
| Edge / lightweight | Gemma / Gemini Nano where officially supported for the target runtime | Small open models on GKE | Confirm model distribution, hardware, and licence at design time. |
| Structured extraction | Gemini model with function calling / structured output support | Partner model with JSON/schema support | Validate schema adherence in CI. |
| Speech-to-text | Chirp 2 | Whisper (GKE) | Supports 100+ languages |
| Text-to-speech | Cloud Text-to-Speech / current Gemini audio-capable model where supported | — | Confirm supported modalities and regions in current model docs |
| Image generation | Imagen / current Gemini image-capable model where supported | Stable Diffusion (GKE) | Confirm model availability, safety controls, and launch stage in current model docs |

### Model selection principles

1. **Start with the current Gemini Flash-class model** for cost-efficient prototyping — upgrade to Pro-class or partner models only when evaluation metrics demand it. Confirm the exact model generation, launch stage, region, and price from current official model and pricing docs before committing. [11][13]
2. **Use Gemini's native capabilities first** — native multimodal, function calling, grounding, and code execution are deeply integrated with the Agent Platform and require less glue code.
3. **Pin model versions** in production — use explicit version strings where the serving API supports them rather than aliases that auto-update.
4. **Benchmark on your data** — use the Evaluation Suite or open-source tools (DeepEval, RAGAS) to compare models on your actual workload before committing.
5. **Leverage prompt caching** for repeated context — Gemini supports context caching for prompts with reusable prefixes, reducing cost by up to 75% on cached tokens.
6. **Consider Claude on Vertex for specific strengths** — Anthropic documents Claude access through Vertex AI. Use Claude only when evaluation shows it outperforms Gemini for the specific task and the required model is available in the target region. [14]
7. **Use Gemma for self-hosted or open-weight scenarios** — when you need model weights on your own infrastructure, confirm the exact Gemma release, licence, and supported serving stack.

---

## RAG and Knowledge Retrieval

### RAG Engine and Search — managed retrieval layers

Google Cloud now has multiple official retrieval paths. **RAG Engine** is the Agent Platform retrieval infrastructure for grounding generative AI apps, with Serverless and Spanner modes. **Vertex AI Search / Discovery Engine-style search resources** still appear in official APIs and Terraform provider resources for search apps and data stores. Pick the retrieval surface by launch stage, data-residency needs, security controls, and operational control rather than by product name alone. [6][7][26][27]

| Vector Store / Retrieval Backend | When to Use |
|----------------------------------|-------------|
| **RAG Engine, Spanner mode** | Default managed RAG path when you want Google-managed retrieval with stronger production-control expectations than Serverless mode. |
| **RAG Engine, Serverless mode** | Fastest setup for managed RAG experiments and lower-control workloads. Public preview; validate limitations before production. |
| **AlloyDB AI** | Teams already on PostgreSQL wanting relational + vector in one database. SQL-based embedding generation via Vertex AI integration. |
| **Cloud SQL for PostgreSQL (pgvector)** | Smaller-scale vector search alongside existing relational data. Lower cost than AlloyDB. |
| **Vertex AI Vector Search** | High-scale, low-latency approximate nearest neighbour (ANN) search. Best for large embedding corpora (100M+ vectors). |
| **BigQuery vector search** | Analytics-first teams wanting vector search alongside their data warehouse. Good for batch retrieval and analytics. |
| **Spanner** | Global, strongly consistent vector search for applications requiring multi-region availability. |
| **Memorystore for Redis** | Real-time, low-latency vector search for caching and session-based retrieval. |

### Retrieval architecture

For most managed RAG deployments, start by evaluating RAG Engine mode constraints:

```
Query → RAG Engine
         ├── Semantic search (embeddings)
         ├── Ranking / retrieval configuration
         └── Grounded context → LLM
```

For teams wanting full control over the retrieval pipeline with SQL-based access:

```
Query → Embedding (gemini-embedding-001 / gemini-embedding-2)
         → AlloyDB AI (pgvector ANN search + metadata filtering)
           → Re-rank (optional: cross-encoder or Gemini)
             → Top-K chunks → LLM
```

For massive-scale vector search (100M+ embeddings):

```
Query → Embedding → Vertex AI Vector Search (ScaNN index)
         → Top-K candidates → Re-rank → LLM
```

**Best practices:**

- **Use RAG Engine first for managed retrieval** when its launch stage, region, security, and feature constraints match the workload.
- **Do not treat RAG Engine Serverless mode as a default regulated-production choice** — it is public preview. If the workload needs compliance features such as CMEK or dedicated isolated database instances, evaluate Spanner mode and confirm metadata-search constraints in the current docs. [6][7][17]
- **Use AlloyDB AI when your data is already in PostgreSQL** — it can generate embeddings via SQL functions that call Vertex AI, eliminating the need for external embedding pipelines.
- **Use `gemini-embedding-001` for new text embedding pipelines** unless a stronger workload-specific reason exists. Use `gemini-embedding-2` for multimodal embedding requirements where available. Existing `text-embedding-005` indexes can remain during migration, but should not be the default for new designs. [15][16]
- **Enable grounding** in Gemini API calls — Gemini supports built-in grounding with Google Search or your own data stores, reducing hallucination without a full RAG pipeline.
- **Use metadata filters** to scope retrieval by document type, customer, date range, or access level.
- **Set retrieval to return 5–10 chunks** — more context doesn't always improve quality and increases token cost.
- **Implement ranking and retrieval evaluation** for quality-sensitive applications — measure retrieval precision/recall separately from final-answer quality.

### Multimodal RAG

For multimodal RAG, prefer official multimodal embedding and Gemini grounding capabilities where available, but validate whether the selected RAG Engine mode supports the required file types, metadata filters, and security controls.

- Images and diagrams are processed alongside text using Gemini's native vision capabilities.
- Tables are extracted and preserved as structured content.
- PDF layout is preserved with reading order detection.
- Charts and figures can be captioned by Gemini during indexing for semantic search.

### Chunking strategy

| Document Type | Chunking Approach | Configuration |
|---------------|-------------------|---------------|
| Structured reports (PDF, DOCX) | Layout-aware extraction before indexing | Preserve headings, sections, paragraphs |
| Legal / compliance documents | Hierarchical chunking | Parent: section, child: paragraph |
| Code documentation | Fixed-size with overlap | 512 tokens, 128 overlap |
| Tabular data | Row-per-chunk or table-per-chunk | Preserve column headers |
| Mixed multimedia | Multimodal embedding / captioning where supported | Validate support in selected retrieval mode |
| Web content | Document-aware | Respect HTML structure, strip boilerplate |

---

## Document Processing Pipelines

### Document AI

Document AI provides ML-powered document extraction across OCR, form parsing, and specialised processors. It's the primary entry point for document ingestion pipelines on GCP.

**Processor types:**

| Processor | Use Case |
|-----------|----------|
| **OCR (Enterprise)** | High-quality text extraction from scanned documents, handwriting |
| **Form Parser** | Key-value pair extraction from structured forms |
| **Layout Parser** | Document structure: headings, paragraphs, tables, lists, reading order |
| **Invoice Parser** | Invoice and receipt field extraction |
| **Identity Document Parser** | Passports, driving licences, national IDs |
| **Contract Parser** | Legal document clause extraction |
| **Lending Parser** | Loan document package processing |
| **Custom Document Extractor** | Train custom extraction models on your own document types |
| **Document Summarizer** | AI-powered document summarisation |

### Cloud Natural Language API

Provides NLP capabilities for post-extraction processing:

- Entity recognition (people, places, organisations, dates).
- Sentiment analysis and content classification.
- Syntax analysis (part-of-speech, dependency trees).
- Custom classification models via AutoML Natural Language.

### Healthcare Natural Language API

For healthcare and life sciences workloads:

- Medical entity extraction (conditions, medications, procedures).
- Relationship detection between medical concepts.
- FHIR-compatible structured output.
- PHI detection for de-identification workflows.

### Production pipeline architecture

```
Source (Cloud Storage Bucket)
  → Eventarc trigger (new object notification)
    → Cloud Workflows orchestration:
      1. Document AI: extract layout + tables + forms
      2. Cloud DLP: classify and detect PII/PHI
      3. Cloud Run job: chunk + embed (gemini-embedding-001 / gemini-embedding-2)
      4. RAG Engine / AlloyDB / Vector Search: index chunks
      5. Firestore / BigQuery: store processing metadata + status
```

**Best practices:**

1. **Use Cloud Workflows** (not Cloud Functions chaining) for orchestrating multi-step pipelines — provides retry logic, error handling, parallel processing, and a visual execution graph.
2. **Use Document AI's Layout Parser** for documents with complex formatting — it preserves reading order, headings, and structural hierarchy far better than OCR alone.
3. **Process asynchronously** — Document AI has per-project quotas. Use batch processing for large corpora and Pub/Sub notifications for completion rather than polling.
4. **Store raw extracted content in Cloud Storage** alongside the original document — enables re-chunking without re-extraction if your strategy changes.
5. **Tag each chunk with source metadata** — document ID, page number, section heading, extraction confidence — for citation generation downstream.
6. **Use Cloud DLP** before indexing for regulated workloads — redact or flag PII/PHI before content enters the vector store.
7. **For large corpora (10,000+ documents)**, use Document AI batch processing and Cloud Workflows parallel iteration for throughput.

---

## Agent Development

### Agent Development Kit (ADK)

ADK is Google's open-source, code-first framework for building agents. It is the primary pro-code path on Google Cloud and is designed to work with Agent Runtime for managed deployment. [3][4]

**Key characteristics:**

| Feature | Description |
|---------|-------------|
| **Multi-language** | Python, TypeScript, Go, Java, Kotlin |
| **Model-agnostic** | Works with Gemini, Claude, Llama, or any model backend |
| **Deployment-agnostic** | Deploy to Agent Runtime, Cloud Run, GKE, or other supported container platforms |
| **Built-in orchestration** | Sequential, parallel, loop, and conditional agent compositions |
| **MCP support** | Native MCP client for connecting to tool servers |
| **A2A support** | Native Agent-to-Agent protocol for inter-agent communication |
| **OpenTelemetry** | Built-in instrumentation for observability |
| **Evaluation** | Integrated evaluation framework for testing agent behaviour |

### Agent Runtime — the managed runtime

Agent Runtime is the managed runtime for deploying agents at scale. The underlying API and Terraform resources may still use Vertex AI or Reasoning Engine names, so do not infer product retirement from resource names. [4][20]

- **Autoscaling** — scales agents based on request volume.
- **Session isolation** — each user session runs in an isolated context.
- **IAM integration** — per-agent IAM bindings for access control.
- **Regional pinning** — deploy agents to specific regions for data residency.
- **Persistent sessions** — multi-day sessions with state management.
- **Memory Bank integration** — long-term memory across sessions.

### Memory Bank

Memory Bank is a managed service for agent memory:

| Memory Type | Purpose |
|-------------|---------|
| **Short-term** | Within-session conversation history and working state |
| **Long-term** | Cross-session facts, preferences, and learned context |
| **Semantic** | Embedding-based retrieval of relevant past interactions |
| **Episodic** | Time-ordered events and experiences |

### MCP on Google Cloud

MCP (Model Context Protocol) is a first-class citizen on the Agent Platform:

- **Remote MCP servers / MCP Toolbox** for Google Cloud services — BigQuery and Cloud SQL are explicitly documented in official Google Cloud MCP material. Confirm the exact server's supported services before designing around it. [9][10]
- **Cloud Run as MCP host** — deploy custom MCP servers as Cloud Run services with automatic scaling, IAM authentication, and private networking.
- **ADK MCP client** — native MCP client in ADK for connecting agents to any MCP-compliant tool server.
- **Tool discovery and governance** — maintain an internal registry of approved MCP servers and tool scopes, even where the platform provides discovery features.

### A2A Protocol (Agent-to-Agent)

A2A is the open protocol for agent interoperability, donated by Google to the Linux Foundation:

- **Agent discovery** — agents publish capability cards describing their skills.
- **Task delegation** — one agent sends a task to another and receives results.
- **Streaming** — real-time progress updates during long-running tasks.
- **Push notifications** — async completion callbacks.
- **Multi-framework** — works with ADK, LangGraph, CrewAI, AutoGen, and others.

MCP and A2A are complementary: **MCP connects agents to tools; A2A connects agents to each other.**

### Agent orchestration patterns

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| **Supervisor** | Orchestrator agent routes to specialist sub-agents | Complex workflows with distinct skill domains |
| **Sequential pipeline** | Agents execute in a fixed order | Extract → validate → enrich → respond |
| **Parallel fan-out** | Multiple agents work simultaneously | Independent sub-tasks that can be merged |
| **Peer-to-peer (A2A)** | Agents communicate directly without a central orchestrator | Cross-team or cross-organisation collaboration |
| **Hierarchical** | Nested supervisors managing sub-teams | Large-scale systems with multiple domains |
| **Loop** | Agent iterates until a condition is met | Refinement, self-correction, iterative research |

### Best practices for production agents

1. **Keep agents single-purpose** — one agent, one job. Compose complex behaviours through orchestration, not monolithic prompts.
2. **Use Agent Runtime for managed deployment** — it handles the managed-agent runtime path. Graduate to Cloud Run or GKE when you need custom networking, GPUs, unsupported frameworks, or capabilities not yet GA.
3. **Use official MCP servers / MCP Toolbox where available** — don't write custom integrations for BigQuery or Cloud SQL when an official MCP path satisfies the requirement.
4. **Scope IAM per agent** — each agent's service account should have only the permissions it needs. Use Workload Identity Federation for external service access.
5. **Set token budgets and timeout limits** — prevent runaway reasoning loops. Configure max output tokens and session timeouts per agent.
6. **Enable Memory Bank selectively** — short-term memory for all multi-turn agents; long-term memory only for agents that genuinely benefit from cross-session recall (personalisation, learning assistants).
7. **Instrument with OpenTelemetry** — ADK has built-in OTel support. Export traces to Cloud Trace and metrics to Cloud Monitoring from day one.
8. **Use A2A for cross-boundary communication** — when agents are built by different teams or need to communicate across trust boundaries, A2A provides the standardised protocol. Within a single team's codebase, direct function calls or ADK orchestration are simpler.

### When to use Agent Runtime vs. Cloud Run

| Criterion | Agent Runtime (managed) | Cloud Run (self-managed) |
|-----------|------------------------|--------------------------|
| Time to production | Days | Weeks |
| Custom networking (VPC, PSC) | Check current runtime feature support and launch stage | Full control |
| GPU workloads | Not supported | Supported (GPU node pools) |
| Framework support | ADK, LangGraph, CrewAI | Any framework |
| Autoscaling | Automatic | Configure min/max instances |
| Session management | Built-in (Sessions + Memory Bank) | BYO (Redis, Firestore) |
| Cost at low volume | Depends on runtime pricing and session use | Scale-to-zero |
| Cost at high volume | Depends on invocation profile and managed features used | Potentially lower with own compute |
| Observability | Built-in tracing + Cloud Trace export | Configure OTel collector |

---

## AI Gateway and Traffic Management

Google Cloud's approach to AI gateway patterns differs from AWS and Azure. Rather than a single "AI Gateway" product, the pattern is assembled from existing infrastructure services.

### Why every production deployment needs an AI Gateway

- **Authentication and authorisation** — validate identity before model access.
- **Rate limiting and quotas** — per-tenant throttling to prevent budget overruns.
- **Model routing** — select the optimal model/tier per request based on complexity.
- **Caching** — cache repeated prompts to reduce latency and cost.
- **Observability** — centralised request logging and token metering.
- **Cost attribution** — tag requests with customer/project identifiers for chargeback.
- **Failover** — route to alternative models or regions on failure.

### Architecture patterns

**Pattern 1: Cloud Run + Apigee (enterprise)**

```
Client → Apigee (API management, auth, quotas, analytics)
           → Cloud Run (routing logic, prompt transformation)
             ├── Gemini Pro-class model (complex queries)
             ├── Gemini Flash-class model (simple queries)
             └── Claude on Vertex (specific evaluated tasks)
```

Apigee provides enterprise API management: developer portals, API keys, OAuth, usage plans, and analytics. Use this for multi-tenant SaaS or when you need formal API products.

**Pattern 2: Cloud Load Balancer + Cloud Run (standard)**

```
Client → External Application Load Balancer (Cloud Armor WAF, SSL)
           → Cloud Run (routing, transformation, retry)
             ├── Gemini API (primary)
             ├── Claude on Vertex (fallback)
             └── Gemma on GKE (cost tier)
```

Simpler than Apigee — suitable for single-tenant deployments or internal applications where you don't need a full API management layer.

**Pattern 3: Service Mesh with GKE (microservices)**

```
Client → Ingress Gateway
           → Envoy sidecar (rate limiting, circuit breaking)
             → AI Router Service
               ├── Gemini (latency-sensitive)
               ├── Self-hosted Gemma (cost-sensitive)
               └── Specialist models (domain-specific)
```

For teams already on GKE with service mesh infrastructure.

**Best practices:**

1. **Use Apigee for multi-tenant SaaS** where you need per-customer API keys, usage plans, and developer portals. For single-tenant or internal apps, Cloud Load Balancer + Cloud Run is sufficient.
2. **Implement model routing in Cloud Run** — classify intent or complexity with a lightweight model (Gemini Nano or heuristics), then route to the appropriate tier.
3. **Use Gemini's context caching** for prompts with reusable system instructions or few-shot examples — up to 75% cost reduction on cached tokens.
4. **Apply Cloud Armor WAF rules** at the load balancer — protects against prompt injection at volume, DDoS, and API abuse.
5. **Enable request/response logging** to Cloud Logging — capture token usage, latency, and model selection per request for cost tracking and debugging.
6. **Use streaming (SSE)** for chat interfaces — Cloud Run supports server-sent events natively, providing better UX than polling.
7. **Implement circuit breakers** — if a model endpoint is degraded, fail fast and route to a fallback rather than queuing requests.

---

## Hosting and Compute Patterns

### Decision matrix

| Workload Characteristic | Recommended Service | Rationale |
|------------------------|---------------------|-----------|
| Event-driven, short-lived (<60 min) | Cloud Run | Scale-to-zero, pay-per-request, native Gemini SDK, up to 60-min timeout |
| Agent sessions (managed) | Agent Runtime | No infrastructure, built-in sessions + memory, ADK-native |
| Long-running or GPU workloads | GKE Autopilot | GPU node pools, persistent connections, full Kubernetes control |
| Custom model hosting (open-source) | GKE + vLLM / TGI | Full control over serving infrastructure, TPU/GPU support |
| Batch processing / evaluation | Cloud Run Jobs | Parallel task execution, no infrastructure, cost-efficient |
| Frontend (React/Next.js) | Cloud Run / Firebase Hosting | CDN, CI/CD integration, serverless |
| Data pipelines | Dataflow / Cloud Workflows | Managed pipeline orchestration, auto-scaling |
| Scheduled tasks | Cloud Scheduler + Cloud Run | Cron-like scheduling with serverless execution |

### Cloud Run for AI workloads

Cloud Run is the default compute for Agent Platform-backed applications:

- **60-minute timeout** — sufficient for complex multi-step agent interactions and orchestration loops.
- **32 GiB memory / 8 vCPU** — adequate for request handling (inference runs on Gemini/Agent Platform, not in your container).
- **GPU support** — L4 GPUs available for local model inference (Whisper, embeddings, small LLMs).
- **Streaming responses** — native SSE support for real-time chat interfaces.
- **Scale-to-zero** — no cost when idle. Minimum instances for latency-sensitive paths.
- **Startup CPU boost** — faster cold starts for JIT-compiled languages.
- **Direct VPC egress** — access private resources (AlloyDB, Memorystore) without a connector.

These Cloud Run limits and features should still be checked against current service documentation for the selected region and trigger type. [19][21][22][23]

**When Cloud Run isn't enough:**

- Workloads requiring persistent WebSocket connections (use GKE).
- Custom model serving requiring multiple GPUs or TPUs (use GKE with node pools).
- Workloads needing fine-grained Kubernetes networking (NetworkPolicy, service mesh).

### GKE Autopilot for AI workloads

- **GPU node pools** — A100, H100, L4 for model inference. TPU node pools for Gemma/custom models.
- **No timeout limits** — suitable for long-running processes and persistent connections.
- **Autoscaling** — KEDA for custom metrics, HPA for standard scaling.
- **Service mesh** — Istio/Envoy for traffic management between microservices.
- **Workload Identity** — map Kubernetes service accounts to GCP IAM service accounts.
- **Spot VMs** — 60–91% cost savings for fault-tolerant workloads (batch inference, evaluation).

### Google Cloud custom silicon

| Chip | Purpose | Key Benefit |
|------|---------|-------------|
| **Ironwood (TPU v7)** | Inference-first | Google-published TPU generation for AI inference workloads; confirm availability and quotas by region. [32] |
| **Trillium (TPU v6e)** | Training + inference | Production-ready, cost-effective for medium-scale |
| **TPU v5p** | Large-scale training | UltraClusters for frontier model training |
| **Axion (Arm)** | General compute | Best price/performance for non-GPU workloads |
| **NVIDIA L4** | Cost-efficient inference | Available on Cloud Run and GKE |
| **NVIDIA H100/A100** | High-performance inference | Available on GKE |

### Self-hosting open-source models

For scenarios requiring model weights on your own infrastructure:

| Framework | When to Use |
|-----------|-------------|
| **vLLM on GKE** | High-throughput serving with PagedAttention. Best for Gemma, Llama, Mistral. Supports TPU and GPU. |
| **Text Generation Inference (TGI)** | HuggingFace models, good ONNX/Safetensors support |
| **Triton Inference Server** | Multi-model serving, batching, ensemble pipelines |
| **Ollama** | Development and prototyping. Not for production at scale. |

**Best practices:**

1. **Default to Cloud Run** for all API-serving AI workloads — it scales to zero, handles bursts, and the 60-minute timeout covers most agent sessions.
2. **Use Agent Runtime** when you want managed agent deployment. Move to Cloud Run or GKE when you need custom networking, GPUs, unsupported frameworks, or features that are not GA in the managed runtime.
3. **Use GKE Autopilot with GPU pools** for self-hosted models — Autopilot handles node provisioning, you define pod resource requests.
4. **Use TPUs for Gemma** — Gemma models are optimised for TPU inference via JAX/FLAX. Cost-effective at scale compared to NVIDIA GPUs.
5. **Use Spot VMs for batch inference** — evaluation jobs, document processing, and non-latency-sensitive workloads save 60–91%.
6. **Use Cloud Run Jobs for batch processing** — parallel task execution (up to 10,000 tasks) with automatic retries, ideal for document processing and evaluation suites.

---

## Security and Networking

### Defence in depth for AI workloads

Security for AI on GCP follows the same zero-trust principles as traditional workloads, with additional considerations for model access, data exfiltration, and prompt injection.

```
┌─────────────────────────────────────────────────────┐
│ Organisation Policies (restrict APIs, regions, models)│
├─────────────────────────────────────────────────────┤
│ VPC Service Controls (perimeter around AI services)  │
├─────────────────────────────────────────────────────┤
│ IAM (least-privilege, service accounts, WIF)         │
├─────────────────────────────────────────────────────┤
│ Private Service Connect (private model endpoints)    │
├─────────────────────────────────────────────────────┤
│ Cloud Armor (WAF, DDoS, bot protection)              │
├─────────────────────────────────────────────────────┤
│ Model Armor (content safety, injection detection)    │
├─────────────────────────────────────────────────────┤
│ Cloud DLP (PII detection, de-identification)         │
├─────────────────────────────────────────────────────┤
│ CMEK (customer-managed encryption keys)              │
└─────────────────────────────────────────────────────┘
```

### VPC Service Controls for AI

VPC-SC creates security perimeters around supported Google Cloud services, helping reduce data exfiltration risk even if another control is misconfigured. Confirm that every target service and API is listed as supported before making VPC-SC a design dependency. [28]

**Configure VPC-SC perimeters around:**

- Vertex AI / Agent Platform services that are listed as VPC-SC supported products for the target feature
- Cloud Storage (document corpus, model artefacts)
- BigQuery (structured data, analytics)
- AlloyDB / Cloud SQL (operational data, vector stores)
- RAG/search indexes where the underlying service is VPC-SC-supported
- Cloud Logging (audit logs containing model interactions)

**Access levels** define who can cross the perimeter boundary:

- Corporate network CIDR ranges.
- Specific service accounts.
- Device trust (BeyondCorp Enterprise).
- Region restrictions.

### Private Service Connect (PSC)

PSC provides private connectivity patterns for Google APIs and producer services. Confirm that the selected API path and deployment mode support PSC before making it a production control. [29]

- **PSC endpoints** for Google APIs where Private Service Connect supports the target API path.
- **PSC/private networking for Agent Runtime** where the current official runtime feature set supports the deployment mode.
- **PSC for AlloyDB/Cloud SQL** — private vector store access from Cloud Run.

### IAM for AI workloads

| Role | Purpose | Assign to |
|------|---------|-----------|
| `roles/aiplatform.user` | Invoke models, deploy agents | Application service accounts |
| `roles/aiplatform.admin` | Full Agent Platform administration | Platform team (break-glass) |
| `roles/discoveryengine.editor` | Manage search apps and data stores | RAG pipeline service accounts |
| `roles/discoveryengine.viewer` | Query search apps | Application service accounts |
| `roles/documentai.apiUser` | Call Document AI processors | Pipeline service accounts |
| `roles/alloydb.client` | Connect to AlloyDB instances | Application service accounts |
| Custom role | Minimal permissions for specific agent | Per-agent service accounts |

**Best practices:**

1. **Deploy VPC-SC perimeters from day one** for regulated workloads — retrofitting is painful and error-prone.
2. **Use Private Service Connect where supported and required** for production model/API access — validate the exact API path before treating PSC as mandatory.
3. **One service account per agent** — scope permissions to exactly what each agent needs. Never share service accounts across agents with different trust levels.
4. **Use Workload Identity Federation** for external access — avoid service account keys. Map GitHub Actions, Azure AD, or AWS IAM to GCP service accounts.
5. **Enable Cloud Audit Logs** for supported AI services — Data Access logs are essential for model and data access investigations, but exact payload and retention behaviour must be validated per service. [30]
6. **Use CMEK where supported** for sensitive workloads — confirm service-level CMEK support for model artefacts, vector stores, logs, and session/memory data before committing to a compliance design.
7. **Implement egress controls** — use VPC firewall rules and organisation policies to prevent agents from exfiltrating data to unauthorised endpoints.
8. **Use Cloud Armor** with pre-configured WAF rules — the OWASP top 10 rule set plus custom rules for LLM-specific attacks (prompt injection patterns, jailbreak attempts).

---

## Responsible AI and Content Safety

### Model Armor

Model Armor is Google Cloud's prompt and response screening service for generative AI applications. It supports security and safety checks such as prompt injection and jailbreak detection, malicious URI detection, sensitive data protection, content safety filters, template-based configuration, and organisation-level floor settings. [8][18]

| Capability | Description |
|------------|-------------|
| **Prompt injection detection** | Identifies and blocks prompt injection attempts before they reach the model |
| **Content filtering** | Configurable thresholds for harmful content (hate speech, violence, sexual content, dangerous activities) |
| **PII detection** | Identifies personal information in prompts and responses |
| **Malicious URI detection** | Detects malicious URIs in prompts and responses |
| **Templates and floor settings** | Define reusable templates and centrally enforced minimum policy settings |
| **Logging integration** | Requires explicit operational design for Cloud Logging visibility, retention, and access controls |

### Implementing responsible AI

1. **Apply Model Armor to production user-facing agents** — configure appropriate thresholds per use case. A customer-facing chatbot needs stricter filtering than an internal code assistant.
2. **Use the safety settings API** — Gemini models support configurable safety thresholds per category (harassment, hate speech, sexually explicit, dangerous content).
3. **Implement human-in-the-loop for high-stakes decisions** — agents handling financial, legal, or medical advice should escalate to humans rather than acting autonomously.
4. **Log all model interactions** — maintain audit trails for compliance. Use Cloud Logging with appropriate retention policies.
5. **Test adversarial inputs** — include prompt injection, jailbreak attempts, and boundary-testing queries in your evaluation suite.
6. **Use Cloud DLP for output sanitisation** — scan model responses for accidental PII leakage before returning to users.
7. **Implement rate limiting per user** — prevents abuse of AI capabilities and limits blast radius of compromised accounts.
8. **Define clear escalation paths** — when Model Armor blocks a request, provide helpful guidance rather than opaque errors.
9. **Validate regional and integration constraints** — align Model Armor location, logging, and template/floor-settings design with the rest of the workload before production rollout. [18]

---

## Cost Management

### Pricing model overview

GCP AI pricing has three primary dimensions:

| Dimension | How it's charged | Key lever |
|-----------|-----------------|-----------|
| **Model invocation** | Per 1M input/output tokens (varies by model) | Model selection, prompt caching, prompt length |
| **Agent Platform services** | Per-request or feature-specific pricing for Agent Runtime, Sessions, Memory Bank, and related services | Session duration, memory retention policy |
| **Compute** | Per vCPU-second / GPU-hour / TPU chip-hour | Instance sizing, scale-to-zero, spot pricing |
| **Storage + retrieval** | Per GB stored + per query for RAG Engine, search, vector stores, and databases | Data lifecycle policies, query volume |

### Pricing source of truth

Do not hard-code token prices in architecture guidance. Gemini, partner model, embedding, batch, cache, and provisioned-throughput prices change by model generation, region, input/output tier, modality, and launch stage. Use Google AI / Google Cloud pricing pages and Anthropic's Claude-on-Vertex documentation as the source of truth at proposal time. [13][14]

### Cost optimisation strategies

1. **Use the current Gemini Flash-class model as the default low-cost tier** — only upgrade to Pro-class or partner models when evaluation shows the lower-cost model cannot meet quality requirements.
2. **Enable context caching** — for prompts with reusable system instructions, few-shot examples, or document context, caching reduces input token cost by up to 75%. Cached content must be >32K tokens and persists for a configurable TTL.
3. **Implement model routing** — use a lightweight classifier or heuristic to route simple queries to Flash and complex queries to Pro. This alone can reduce costs 40–60%.
4. **Use batch API for non-real-time workloads** — evaluation, document processing, and offline analysis can use batch pricing (typically 50% discount).
5. **Scale to zero** — Cloud Run scales to zero by default. Set minimum instances only for latency-critical paths.
6. **Use Spot VMs for batch workloads** — 60–91% savings on GKE nodes used for evaluation, batch inference, and data processing.
7. **Right-size Cloud Run instances** — most AI gateway workloads need minimal CPU/memory since inference runs remotely. 1 vCPU / 512 MiB is often sufficient.
8. **Set per-request token budgets** — configure `max_output_tokens` to prevent runaway generation. A 4,000-token cap on a summarisation task prevents accidentally generating 100K tokens.
9. **Monitor token usage in real-time** — export metrics to Cloud Monitoring and set alerts for unexpected spikes.
10. **Negotiate committed-use or private pricing only after usage modelling** — discounts and commitments are commercial terms, not architecture assumptions.

### Billing alerts and governance

```
Cloud Billing → Budget alerts (per-project, per-service)
  → Pub/Sub → Cloud Function (auto-disable non-essential services)
  → Email/Slack notifications at 50%, 80%, 100% thresholds
```

- Set **per-project budgets** for each environment.
- Use **labels** on all resources for cost attribution by customer, team, or project.
- Export billing data to **BigQuery** for detailed analysis and chargeback reporting.
- Use **quotas** to hard-cap token consumption per project or per API key.

---

## Observability and Monitoring

### The observability stack for AI on GCP

| Layer | Service | Purpose |
|-------|---------|---------|
| **Tracing** | Cloud Trace (OTLP) | Distributed traces across agent orchestrations, tool calls, and model invocations |
| **Logging** | Cloud Logging | Structured logs: prompts, responses, errors, model selection decisions |
| **Metrics** | Cloud Monitoring | Token usage, latency percentiles, error rates, cost per request |
| **Profiling** | Cloud Profiler | CPU/memory profiling for Cloud Run and GKE workloads |
| **Dashboards** | Cloud Monitoring Dashboards | Operational visibility for SRE and developers |
| **Alerting** | Cloud Monitoring Alerting | PagerDuty/Slack integration for SLO breaches |

### OpenTelemetry integration

ADK has built-in OpenTelemetry instrumentation. Configure the OTel SDK to export to Google Cloud:

```
ADK Agent
  → OpenTelemetry SDK (auto-instrumentation)
    → OTLP Exporter
      ├── Cloud Trace (traces + spans)
      ├── Cloud Monitoring (metrics)
      └── Cloud Logging (structured logs)
```

**What to trace:**

- Each user request end-to-end (from API gateway to response).
- Each model invocation (input tokens, output tokens, latency, model version).
- Each tool call (MCP server, duration, success/failure).
- Each agent step in multi-step orchestrations.
- Each retrieval operation (query, results count, relevance scores).

### Key metrics to monitor

| Metric | Alert threshold | Rationale |
|--------|----------------|-----------|
| Request latency (p95) | >10s for chat, >60s for complex tasks | User experience degradation |
| Error rate | >1% of requests | Model failures, quota exhaustion |
| Token usage per request | >2x baseline | Prompt injection or runaway generation |
| Daily token spend | >120% of budget | Cost overrun detection |
| Agent loop iterations | >20 per request | Stuck agent detection |
| Retrieval relevance score | <0.5 mean | RAG quality degradation |
| Memory Bank size per user | >10MB | Memory pollution |

### Best practices

1. **Export OTel from day one** — retrofitting observability is far harder than starting with it.
2. **Use structured logging** — log JSON with consistent fields (request_id, user_id, model, tokens_in, tokens_out, latency_ms). Use Cloud Logging's structured log format for native querying.
3. **Create SLOs** — define service level objectives for latency and availability. Cloud Monitoring SLOs integrate with error budgets and alerting.
4. **Trace agent reasoning** — each step in multi-agent orchestration should be a child span under the parent request. This is critical for debugging why an agent made a particular decision.
5. **Log model inputs/outputs selectively** — for debugging, log the first 500 characters of prompts and responses. For compliance, log everything to a separate, access-controlled log bucket.
6. **Set up cost dashboards** — combine billing export (BigQuery) with operational metrics to correlate cost with usage patterns.
7. **Use Cloud Trace for latency analysis** — identify whether latency is in your code, the model, or retrieval. Trace spans make this trivial.

---

## Evaluation and Testing

### The evaluation stack

| Level | What's Tested | Tool |
|-------|---------------|------|
| **Unit** | Individual functions, prompt templates, tool schemas | pytest / Jest + mocks |
| **Component** | Single agent: does it use the right tools and produce correct output? | ADK evaluation framework |
| **Trajectory** | Multi-step agent: does it follow the expected sequence of actions? | Agent Platform Evaluation Suite |
| **End-to-end** | Full pipeline: user input → final response quality | DeepEval / RAGAS + custom metrics |
| **Regression** | Did a model/prompt change break existing behaviour? | Golden dataset comparison |
| **Safety** | Does the agent resist adversarial inputs? | Model Armor + custom red-team datasets |

### Agent Platform Evaluation Suite

The built-in evaluation service supports:

- **Pointwise evaluation** — score a single response against criteria (coherence, groundedness, safety).
- **Pairwise evaluation** — compare two model/prompt configurations on the same input.
- **Trajectory evaluation** — verify that an agent followed the expected sequence of tool calls and reasoning steps.
- **Custom metrics** — define your own evaluation criteria with Gemini-as-judge.

### Evaluation best practices

1. **Build golden datasets from day one** — collect real user queries and expert-annotated ideal responses. Start with 50–100 examples per task type.
2. **Evaluate on every model/prompt change** — run your evaluation suite in CI before deploying changes. A 2% regression in accuracy should block deployment.
3. **Use trajectory evaluation for agents** — it's not enough to check the final answer. Verify the agent took the right actions in the right order, especially for tool-calling agents.
4. **Combine automated and human evaluation** — automated metrics catch regressions; periodic human review catches subtle quality issues that metrics miss.
5. **Test adversarial robustness** — include prompt injection attempts, out-of-scope queries, and boundary-testing inputs in your evaluation suite.
6. **Evaluate retrieval separately** — measure precision@k and recall@k for your RAG pipeline independently of the generation step. Poor retrieval produces poor answers regardless of model quality.
7. **Use DeepEval or RAGAS for RAG evaluation** — open-source frameworks that measure faithfulness, answer relevance, and context precision.
8. **Run evaluation at scale with Cloud Run Jobs** — parallelise evaluation across thousands of test cases using Cloud Run Jobs with up to 10,000 parallel tasks.

---

## Infrastructure as Code

### Terraform for GCP AI

Terraform is the primary IaC tool for Google Cloud infrastructure. Current Agent Platform resources may still appear under Vertex AI / Reasoning Engine names in the official Terraform provider, so align module names with product language while using provider resource names exactly as documented. [20]

**Key resources:**

| Terraform Resource | Purpose |
|-------------------|---------|
| `google_vertex_ai_endpoint` | Model serving endpoints |
| `google_discovery_engine_search_engine` | Search apps where Discovery Engine resources are the current documented path |
| `google_discovery_engine_data_store` | Search data stores |
| `google_vertex_ai_reasoning_engine` | Agent Runtime / reasoning-engine deployments |
| `google_alloydb_cluster` + `google_alloydb_instance` | AlloyDB vector store |
| `google_cloud_run_v2_service` | Application compute |
| `google_cloud_run_v2_job` | Batch processing |
| `google_document_ai_processor` | Document AI processors |
| `google_compute_network` + subnets | VPC networking |
| `google_access_context_manager_service_perimeter` | VPC-SC perimeters |
| `google_kms_crypto_key` | CMEK encryption keys |
| `google_service_account` | Per-agent service accounts |

### Module structure

```
terraform/
├── modules/
│   ├── agent-platform/      # Agent Runtime, RAG/search, Model Garden config
│   ├── networking/           # VPC, PSC, firewall rules, Cloud Armor
│   ├── data/                 # AlloyDB, BigQuery, Cloud Storage
│   ├── compute/              # Cloud Run services, GKE clusters
│   ├── security/             # IAM, KMS, VPC-SC, service accounts
│   └── observability/        # Monitoring, alerting, dashboards
├── environments/
│   ├── dev/
│   ├── staging/
│   └── prod/
├── backend.tf               # GCS remote state
└── variables.tf
```

### Agent Runtime deployment with Terraform

Agent Runtime deployments can be represented in Terraform through current Vertex AI / Reasoning Engine resources, but application packaging, agent release promotion, and runtime evaluation still need a CI/CD workflow. The recommended pattern:

1. **Terraform** provisions infrastructure: networking, databases, service accounts, IAM, monitoring.
2. **ADK / `gcloud` / Terraform-managed runtime resources** deploy agent code or runtime references, depending on the selected deployment path and current provider support.
3. **CI/CD pipeline** orchestrates both: Terraform applies first, then agent deployment.

### Best practices

1. **Use remote state in GCS** with state locking via Cloud Storage's generation-based preconditions.
2. **Separate state files per environment** — dev/staging/prod should never share Terraform state.
3. **Use `google-beta` only when the official provider docs require it** — otherwise prefer the stable `google` provider for resources available there.
4. **Pin provider versions** — GCP Terraform provider releases are frequent. Pin to a specific minor version and upgrade deliberately.
5. **Use Workload Identity Federation** for Terraform CI/CD — no service account keys. Map your CI runner's identity to a GCP service account.
6. **Store secrets in Secret Manager** — reference them in Terraform but never store values in state. Use `data "google_secret_manager_secret_version"` for lookups.

---

## CI/CD for AI Applications

### Pipeline architecture

AI applications require a dual-track CI/CD pipeline: one for infrastructure and one for agent/model code.

```
Source (GitHub / Cloud Source Repositories)
  → Trigger (push to branch / PR)
    → Cloud Build:
      ├── Track 1: Infrastructure (Terraform plan/apply)
      │   └── Networking, databases, IAM, monitoring
      └── Track 2: Application (agent code)
          ├── Lint + type check
          ├── Unit tests (prompt templates, tool schemas)
          ├── Build container image → Artifact Registry
          ├── Deploy to staging (Agent Runtime or Cloud Run)
          ├── Evaluation suite (golden datasets, trajectory tests)
          ├── Canary deployment to production (10% traffic)
          ├── Monitor error rate + latency
          └── Full rollout or rollback
```

### Cloud Build vs. GitHub Actions

| Aspect | Cloud Build | GitHub Actions |
|--------|-------------|----------------|
| GCP integration | Native (Workload Identity, VPC, Artifact Registry) | Via Workload Identity Federation |
| Pricing | $0.003/build-minute (free tier: 120 min/day) | Free for public repos; paid for private |
| Secrets | Secret Manager integration | GitHub Secrets + WIF for GCP access |
| Caching | Kaniko layer caching | Standard actions/cache |
| When to use | GCP-native teams, private networking requirements | GitHub-centric teams, multi-cloud |

### Evaluation in CI

The critical addition to AI CI/CD (vs. traditional software) is **automated evaluation gates**:

1. **Pre-merge**: run the evaluation suite on PR branches. Block merge if quality metrics regress beyond threshold.
2. **Post-deploy staging**: run end-to-end evaluation against the staging agent deployment. Block production promotion on failure.
3. **Canary monitoring**: after canary deployment, monitor real traffic metrics for 30–60 minutes. Auto-rollback on SLO breach.

### Prompt and model version management

- **Version prompts in Git** — system prompts, few-shot examples, and tool descriptions are code. Track them in the repository.
- **Pin model versions** — use explicit version strings in configuration, not aliases.
- **Use feature flags for model experiments** — route a percentage of traffic to a new model version and compare metrics before full rollout.
- **Maintain a prompt changelog** — document why each prompt change was made, what evaluation results showed.

### Best practices

1. **Use Cloud Build for GCP-native pipelines** — native VPC access, Secret Manager integration, and Artifact Registry push without additional auth configuration.
2. **Run evaluation in CI** — every PR that changes prompts, tools, or model configuration should trigger the evaluation suite. No exceptions.
3. **Use canary deployments** — Cloud Run supports traffic splitting natively. Route 10% to the new revision, monitor, then promote.
4. **Store container images in Artifact Registry** — region-local, IAM-integrated, vulnerability scanning included.
5. **Use Workload Identity Federation** — eliminate service account keys in CI/CD. Cloud Build has native WIF; GitHub Actions uses the `google-github-actions/auth` action.
6. **Implement rollback automation** — if canary metrics breach SLO, automatically roll back to the previous revision. Cloud Run revision-based routing makes this a single API call.

---

## Agent Studio and Low-Code Scenarios

### When to use Agent Studio

Agent Studio is the visual, low-code surface for building agents on the Gemini Enterprise Agent Platform. It's suitable for:

- Internal Q&A bots grounded on company documents.
- Simple task-completion agents with a fixed set of tools.
- Rapid prototyping before committing to pro-code development.
- Citizen developer use cases where non-engineers need to build and maintain agents.

### Agent Studio capabilities

| Feature | Description |
|---------|-------------|
| **Visual agent builder** | Drag-and-drop flow design for agent behaviour |
| **Data store grounding** | Connect to Cloud Storage, BigQuery, websites, or structured data for RAG |
| **Connectors and tools** | Google Workspace, approved MCP servers, and third-party tools where officially supported |
| **Multi-channel deployment** | Web widget, API, Dialogflow CX integration |
| **Conversation design** | Built-in conversation testing and simulation |
| **Version management** | Draft/publish workflow with rollback |
| **Analytics** | Conversation analytics, user satisfaction tracking |

### Hybrid: Agent Studio + ADK

For complex deployments, Agent Studio can serve as the user-facing surface while ADK-built agents handle backend reasoning:

```
User → Agent Studio (conversational UI, basic routing)
         → A2A call to ADK agent (complex reasoning, multi-tool orchestration)
           → MCP tools (BigQuery, Cloud SQL, external APIs)
             → Response returned through Agent Studio
```

This pattern lets non-technical teams manage the conversational experience while engineering teams control the reasoning backend.

### Dialogflow CX integration

Dialogflow CX remains available for teams with existing investments. It integrates with Agent Platform for:

- Telephony agents (IVR, contact centre).
- Strict flow-based conversation design (compliance-heavy).
- Multi-language support with explicit translation control.

For new projects, prefer Agent Studio unless you specifically need Dialogflow CX's telephony features or strict flow control.

---

## Architecture Patterns by Business Size

### Small business (1–50 employees, <$5K/month AI spend)

**Pattern: Gemini Flash-class model + Cloud Run + managed RAG**

```
Users → Cloud Run (Next.js / FastAPI frontend)
          → Gemini Flash-class API (reasoning)
          → RAG Engine / search data store (document retrieval)
          → Cloud Storage (document upload)
```

**Key decisions:**

- Use a current Gemini Flash-class model exclusively until evaluation proves a need for a larger or partner model.
- Managed RAG where the launch stage and security constraints fit — no vector database to manage.
- Cloud Run for everything — single service, scale-to-zero, minimal ops.
- Firebase Auth for user management.
- Minimal infrastructure: one GCP project, no VPC-SC, no separate environments.

**Typical use cases:** Document Q&A, customer support chatbot, internal knowledge base, email drafting assistant.

### Medium business (50–500 employees, $5K–$50K/month AI spend)

**Pattern: Multi-model + Cloud Run + AlloyDB + Agent Runtime**

```
Users → Cloud Run (application layer, model routing)
          ├── Gemini Flash-class model (simple queries, classification)
          ├── Gemini Pro-class model (complex reasoning, analysis)
          └── Claude on Vertex (specific tasks where it excels)
        → Agent Runtime (multi-step agents)
        → AlloyDB AI (vector store + operational data)
        → Document AI (document processing pipeline)
        → Cloud Workflows (orchestration)
```

**Key decisions:**

- Model routing to optimise cost/quality trade-off.
- Agent Runtime for managed agent deployment where required features are GA or accepted by risk decision.
- AlloyDB AI for combined relational + vector storage — single database for application state and embeddings.
- Separate dev/prod projects with VPC-SC in production.
- Cloud Build CI/CD with evaluation gates.
- Basic observability: Cloud Trace + Cloud Monitoring dashboards.

**Typical use cases:** Enterprise RAG, document processing automation, multi-agent workflows, customer service with human escalation, compliance document analysis.

### Enterprise (500+ employees, $50K+/month AI spend)

**Pattern: Full Agent Platform + GKE + multi-region + governance**

```
Users → Global Load Balancer (Cloud Armor WAF)
          → Apigee (API management, per-tenant quotas)
            → Cloud Run / GKE (application layer)
              ├── Agent Runtime (managed agents)
              ├── ADK agents on Cloud Run (custom agents)
              └── Self-hosted models on GKE (Gemma, domain-specific)
            → AlloyDB (multi-region, vector + relational)
            → RAG Engine / search / Vector Search (enterprise retrieval)
            → Spanner (global consistency for critical state)
            → BigQuery (analytics, evaluation datasets, billing)
```

**Key decisions:**

- Apigee for multi-tenant API management with per-customer quotas and analytics.
- GKE Autopilot for self-hosted models and custom workloads requiring GPUs/TPUs.
- Multi-region deployment for availability and data residency.
- VPC-SC perimeters, Private Service Connect, and CMEK where supported by the selected services and regions.
- Comprehensive evaluation pipeline with trajectory testing and human review.
- Dedicated SRE with SLOs, error budgets, and incident management.
- A2A protocol for cross-team agent collaboration.
- Committed-use or private-pricing review after measured usage modelling.

**Typical use cases:** Organisation-wide AI platform, multi-agent systems spanning business units, customer-facing AI products, regulated industry applications (financial services, healthcare, government).

---

## Implementation Gaps to Fill

This playbook is a design guide, not yet an executable delivery pack. Before using it as a client-facing implementation standard, add:

1. **Launch-stage matrix** — service, API/resource name, GA/Preview/Pre-GA status, supported regions, Terraform provider support, and replacement/migration path. Prioritise Agent Runtime, RAG Engine modes, Gemini model versions, Model Armor, and MCP servers. [1][4][6][8][12]
2. **Security baseline artifacts** — organisation policies, VPC-SC supported-product checks, Private Service Connect design, service-account/IAM role mappings, audit-log retention, and Model Armor logging/templates. [18][28][29][30][31]
3. **Operational runbooks** — quota exhaustion, model version rollback, prompt regression, RAG index rebuild, Model Armor false positives, retrieval-quality degradation, and runaway agent loops. Tie each runbook to Cloud Logging, Cloud Monitoring, Cloud Trace, and ADK OpenTelemetry signals. [25][30]
4. **CI/CD examples** — Terraform plan/apply, agent deployment, model/prompt version pinning, evaluation gates, canary rollout, and rollback automation. [20][26][27]
5. **Cost model template** — current official model pricing lookup, context caching assumptions, batch API eligibility, storage/retrieval costs, Cloud Run/GKE sizing, and billing export labels. [13]

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It's Bad | What to Do Instead |
|--------------|--------------|-------------------|
| **Calling all Vertex AI surfaces legacy** | Vertex AI remains active and many current resources still use Vertex AI names | Use Agent Platform docs for agent lifecycle decisions and current Vertex AI docs/provider resources where they remain official |
| **Defaulting to Gemini Pro for all requests** | More expensive models are often unnecessary for simple tasks | Implement model routing; start with a current Flash-class model, upgrade based on evaluation |
| **Skipping evaluation in CI** | Prompt/model changes break things silently; quality regressions reach production undetected | Run automated evaluation on every change; block deployment on regression |
| **Monolithic agents** | Hard to debug, test, and maintain; single failure point; token-hungry system prompts | Decompose into single-purpose agents orchestrated via ADK or A2A |
| **Storing vectors in Cloud SQL without pgvector tuning** | Default PostgreSQL config has terrible ANN performance at scale | Use AlloyDB AI (optimised for vector), tune `ivfflat` lists, or use Vertex AI Vector Search for large scale |
| **No token budgets on agents** | Runaway reasoning loops can consume entire daily budget in minutes | Set `max_output_tokens`, iteration limits, and timeout per agent |
| **Service account key files** | Security risk; hard to rotate; leak in git history | Use Workload Identity (GKE), Workload Identity Federation (external CI), or attached service accounts (Cloud Run) |
| **Single-project deployment** | No environment isolation; production changes tested on production | Separate GCP projects per environment with promotion pipeline |
| **No VPC-SC in regulated environments** | Data exfiltration risk even with correct IAM; compliance audit failure | Deploy VPC-SC perimeters from day one for any workload handling sensitive data |
| **Building custom integrations for standard GCP services** | Reinventing official MCP/database-tooling paths; maintenance burden | Use official MCP servers / MCP Toolbox where they cover the target service and security model |
| **Ignoring context caching** | Paying full input token price on every request for repeated system prompts | Enable context caching for prompts with >32K tokens of reusable context |
| **Using Dialogflow CX for new text-only agents** | More complex than needed; Agent Studio + ADK is the modern path | Use Agent Studio for low-code or ADK for pro-code; reserve Dialogflow CX for telephony |
| **No observability for agent reasoning** | Can't debug why an agent made a specific decision; incidents are undiagnosable | Instrument with OpenTelemetry from day one; trace every tool call and model invocation |

---

## References

1. Google Cloud documentation — Gemini Enterprise Agent Platform. https://docs.cloud.google.com/gemini-enterprise-agent-platform
2. Google Cloud documentation — Vertex AI. https://docs.cloud.google.com/vertex-ai/docs
3. Agent Development Kit documentation. https://adk.dev/
4. Google Cloud documentation — Agent Runtime. https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/runtime
5. Google Cloud documentation — Agent Gateway overview. https://docs.cloud.google.com/gemini-enterprise-agent-platform/govern/gateways/agent-gateway-overview
6. Google Cloud documentation — RAG Engine Serverless mode. https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/rag-engine/serverless-mode
7. Google Cloud documentation — RAG Engine deployment modes. https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/rag-engine/deployment-modes
8. Google Cloud documentation — Model Armor overview. https://docs.cloud.google.com/model-armor/overview
9. Google Cloud documentation — Use the BigQuery MCP server. https://docs.cloud.google.com/bigquery/docs/use-bigquery-mcp
10. Google Cloud documentation — Connect LLMs to BigQuery with MCP Toolbox. https://docs.cloud.google.com/bigquery/docs/pre-built-tools-with-mcp-toolbox
11. Google Cloud documentation — Model Garden. https://cloud.google.com/model-garden
12. Google Cloud documentation — Gemini 3.1 Pro. https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-pro
13. Google AI documentation — Gemini API pricing. https://ai.google.dev/gemini-api/docs/pricing
14. Anthropic documentation — Claude on Vertex AI. https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai
15. Google Cloud documentation — Get text embeddings. https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/get-text-embeddings
16. Google AI documentation — Embeddings. https://ai.google.dev/gemini-api/docs/embeddings
17. Google Cloud documentation — RAG Engine metadata search. https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/rag-engine/use-metadata-search
18. Google Cloud documentation — Model Armor templates. https://docs.cloud.google.com/model-armor/manage-templates
19. Google Cloud documentation — Cloud Run request timeout. https://docs.cloud.google.com/run/docs/configuring/request-timeout
20. Terraform Registry — `google_vertex_ai_reasoning_engine`. https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/vertex_ai_reasoning_engine
21. Google Cloud documentation — Cloud Run memory limits. https://docs.cloud.google.com/run/docs/configuring/services/memory-limits
22. Google Cloud documentation — Cloud Run GPU support. https://docs.cloud.google.com/run/docs/configuring/services/gpu
23. Google Cloud documentation — Cloud Run Direct VPC egress. https://docs.cloud.google.com/run/docs/configuring/vpc-direct-vpc
24. Google Cloud documentation — Document AI overview. https://docs.cloud.google.com/document-ai/docs/overview
25. Google Cloud documentation — Instrument ADK applications with OpenTelemetry. https://docs.cloud.google.com/stackdriver/docs/instrumentation/ai-agent-adk
26. Terraform Registry — `google_discovery_engine_search_engine`. https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/discovery_engine_search_engine
27. Terraform Registry — `google_discovery_engine_data_store`. https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/discovery_engine_data_store
28. Google Cloud documentation — VPC Service Controls. https://docs.cloud.google.com/vpc-service-controls/docs
29. Google Cloud documentation — Private Service Connect. https://docs.cloud.google.com/vpc/docs/private-service-connect
30. Google Cloud documentation — Cloud Audit Logs. https://docs.cloud.google.com/logging/docs/audit
31. Google Cloud documentation — Service accounts. https://docs.cloud.google.com/iam/docs/service-accounts
32. Google Cloud blog — Ironwood: The first Google TPU for the age of inference. https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/
