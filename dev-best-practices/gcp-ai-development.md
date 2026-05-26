# Best Practices for AI Development on Google Cloud in 2026

| Field | Value |
|-------|-------|
| Created | 2026-05-26 |
| Last Updated | 2026-05-26 |
| Version | 1.0 |

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
- [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
- [References](#references)

## Executive Summary

Google Cloud's AI platform underwent a major consolidation at Cloud Next 2026 (April 2026). **Vertex AI has been retired as a standalone product** and replaced by the **Gemini Enterprise Agent Platform** — a unified environment for building, scaling, governing, and optimising autonomous AI agents. The platform brings together model access (Model Garden), agent development (ADK and Agent Studio), managed runtime (Agent Engine), persistent context (Memory Bank and Sessions), and evaluation tooling under a single control plane. [1][2]

The most significant 2026 developments:

1. **Gemini Enterprise Agent Platform** (April 2026) — replaces Vertex AI as the central AI development surface. Unifies model building, agent orchestration, deployment, and governance.
2. **Agent Development Kit (ADK) 2.0** — open-source, code-first framework for building production agents. Available in Python, TypeScript, Go, Java, and Kotlin. Model-agnostic and deployment-agnostic.
3. **Ironwood TPU GA** (April 2026) — seventh-generation TPU, first designed primarily for inference. 10x peak performance over TPU v5p and 4x per-chip improvement over TPU v6e (Trillium).
4. **A2A Protocol v1.0** — open agent-to-agent communication protocol (donated to Linux Foundation), in production at 150+ organisations. Complements MCP for inter-agent collaboration.
5. **Managed MCP servers** — first-class MCP support integrated with Agent Engine, enabling standardised tool connectivity for agents across BigQuery, Cloud SQL, and other services.
6. **Model Garden expansion** — 200+ foundation models including Gemini 3.1 Pro, Gemini 3.5 Flash, Gemma 4, and third-party models (Anthropic Claude Opus/Sonnet/Haiku, Llama, Mistral).
7. **TPU 8 preview** — eighth-generation architecture splits into training-optimised (Sunfish/Broadcom) and inference-optimised (Zebrafish/MediaTek) chips, targeting late 2027.

For contractors building cloud AI systems, the GCP landscape has three tiers of engagement:

1. **Pro-code (Agent Platform + ADK)** — full control over architecture, model selection, retrieval, orchestration, and deployment. Suitable for enterprise RAG systems, multi-agent platforms, and bespoke AI applications.
2. **Low-code (Agent Studio)** — visual builder for conversational agents with enterprise data grounding, deployed to web, mobile, or internal channels. Suitable for internal productivity tools and Q&A applications.
3. **Hybrid** — Agent Studio for user-facing surfaces with ADK-built agents providing reasoning backends, connected via A2A protocol and MCP tool integrations.

This guide covers production best practices across all three, with emphasis on the pro-code path where professional services teams spend most of their time.

---

## Evidence Basis and Status Labels

This playbook uses current Google Cloud documentation, the Agent Platform product pages, and official Google developer blog posts as normative sources. Community guides and third-party blog posts are not used as primary evidence.

Feature status matters because the Agent Platform is evolving rapidly from the Vertex AI migration. Use these labels:

| Label | Meaning | Delivery rule |
|-------|---------|---------------|
| **GA** | Generally available in current documentation | Suitable for production |
| **Preview** | Public preview or preview-labelled API | Use behind an explicit risk decision; avoid as a hard dependency for regulated production |
| **Legacy** | Former Vertex AI surface still accessible but deprecated | Use only for migration; plan to move off |
| **Region-dependent** | Availability varies by geography, model, or TPU type | Confirm in the console before committing to design |

Current status checkpoints:

| Area | Current stance |
|------|---------------|
| Gemini Enterprise Agent Platform | GA — the primary development surface for all new AI work on GCP [1] |
| Agent Engine (formerly Deployments) | GA — managed runtime for deploying agents built with ADK, LangGraph, or CrewAI [3] |
| Memory Bank | GA — managed long-term memory for context-aware agents across sessions [3] |
| Sessions (multi-day) | GA — persistent agent sessions with state management [3] |
| ADK 2.0 | GA — open-source, code-first agent framework [4] |
| Agent Studio | GA — visual low-code agent builder [5] |
| Model Garden | GA — 200+ models, region-dependent availability [6] |
| Ironwood TPU | GA (April 2026) — seventh-gen inference-optimised TPU [7] |
| A2A Protocol v1.0 | GA — open standard for agent interoperability [8] |
| Managed MCP servers | GA for select Google Cloud services (BigQuery, Cloud SQL); expanding [9] |
| Vertex AI Search (now Agent Platform Search) | GA — enterprise search and retrieval [10] |

---

## Platform Architecture: Gemini Enterprise Agent Platform

The Gemini Enterprise Agent Platform is the unified control plane for AI development on Google Cloud. It replaces the former Vertex AI service and consolidates model access, agent building, deployment, governance, and evaluation.

| Component | Purpose |
|-----------|---------|
| **Model Garden** | 200+ foundation models: Gemini family, Gemma (open-source), Anthropic Claude, Meta Llama, Mistral, and others |
| **Agent Development Kit (ADK)** | Open-source code-first SDK for building agents. Python, TypeScript, Go, Java, Kotlin |
| **Agent Studio** | Visual low-code agent builder for rapid prototyping and simple use cases |
| **Agent Engine** | Managed runtime for deploying and scaling agents (autoscaling, IAM, regional pinning, persistent sessions) |
| **Memory Bank** | Managed long-term memory service for context-aware agent interactions across sessions |
| **Sessions** | Multi-day persistent agent sessions with state management |
| **Agent Platform Search** | Enterprise search and retrieval (formerly Vertex AI Search) — combines structured and unstructured data |
| **Evaluation Suite** | Model and agent evaluation: automated benchmarks, trajectory evaluation, and human-in-the-loop scoring |
| **Model Armor** | Content safety, prompt injection detection, and responsible AI guardrails |
| **Agent Garden** | Library of prebuilt agents and templates for common use cases |

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
│       ├── Agent Engine (managed agent runtime)
│       ├── Agent Platform Search (enterprise retrieval)
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
| Complex reasoning / long-context | Gemini 3.1 Pro / Claude Opus 4.6 | Llama 4 Maverick | Gemini 3.1 Pro: 2M token context window |
| High-volume chat / simple Q&A | Gemini 3.5 Flash / Claude Haiku 4.5 | Gemma 4 2B | Flash optimised for cost/speed |
| Multimodal (vision + text) | Gemini 3.1 Pro / Claude Sonnet 4.6 | Llama 4 Scout | Gemini native multimodal |
| Code generation | Gemini 3.1 Pro / Claude Opus 4.6 | DeepSeek-V3 / Gemma 4 Code | — |
| Embedding | text-embedding-005 | Gemma Embedding | 768 dimensions, native integration |
| Edge / lightweight | Gemma 4 2B / Gemini Nano | Phi-4-mini (GKE) | On-device or low-latency |
| Structured extraction | Gemini 3.1 Pro (function calling) | Mistral Large | JSON mode + structured output |
| Speech-to-text | Chirp 2 | Whisper (GKE) | Supports 100+ languages |
| Text-to-speech | Cloud TTS / Gemini 3.1 Pro | — | Neural voices or conversational |
| Image generation | Imagen 4 / Gemini 3.1 Flash Image | Stable Diffusion (GKE) | Enterprise safety filters built-in |

### Model selection principles

1. **Start with Gemini 3.5 Flash** for cost-efficient prototyping — upgrade to Pro or Claude only when evaluation metrics demand it. Flash is 10–20x cheaper than Pro for many tasks.
2. **Use Gemini's native capabilities first** — native multimodal, function calling, grounding, and code execution are deeply integrated with the Agent Platform and require less glue code.
3. **Pin model versions** in production — use explicit version strings (e.g., `gemini-3.1-pro-002`) rather than aliases that auto-update.
4. **Benchmark on your data** — use the Evaluation Suite or open-source tools (DeepEval, RAGAS) to compare models on your actual workload before committing.
5. **Leverage prompt caching** for repeated context — Gemini supports context caching for prompts with reusable prefixes, reducing cost by up to 75% on cached tokens.
6. **Consider Claude on Vertex for specific strengths** — Claude models are available natively in Model Garden. Use them when evaluation shows they outperform Gemini for your specific task (e.g., nuanced instruction following, long-form writing).
7. **Use Gemma for self-hosted scenarios** — when you need model weights on your own infrastructure (compliance, air-gapped, edge), Gemma 4 is the best-supported open model on GCP.

---

## RAG and Knowledge Retrieval

### Agent Platform Search — the managed retrieval layer

Agent Platform Search (formerly Vertex AI Search) provides end-to-end managed RAG: point at a data source, and the service handles chunking, embedding, indexing, and retrieval. It supports structured data (BigQuery, Cloud SQL), unstructured documents (Cloud Storage, websites), and hybrid configurations.

| Vector Store / Retrieval Backend | When to Use |
|----------------------------------|-------------|
| **Agent Platform Search** | Default for enterprise RAG. Managed chunking, embedding, hybrid search (semantic + keyword), and ranking. No infrastructure to manage. |
| **AlloyDB AI** | Teams already on PostgreSQL wanting relational + vector in one database. SQL-based embedding generation via Vertex AI integration. |
| **Cloud SQL for PostgreSQL (pgvector)** | Smaller-scale vector search alongside existing relational data. Lower cost than AlloyDB. |
| **Vertex AI Vector Search** | High-scale, low-latency approximate nearest neighbour (ANN) search. Best for large embedding corpora (100M+ vectors). |
| **BigQuery vector search** | Analytics-first teams wanting vector search alongside their data warehouse. Good for batch retrieval and analytics. |
| **Spanner** | Global, strongly consistent vector search for applications requiring multi-region availability. |
| **Memorystore for Redis** | Real-time, low-latency vector search for caching and session-based retrieval. |

### Retrieval architecture

For most enterprise RAG deployments, Agent Platform Search is the default starting point:

```
Query → Agent Platform Search
         ├── Semantic search (embeddings)
         ├── Keyword search (BM25)
         └── Fusion + Re-ranking → Top-K chunks → LLM
```

For teams wanting full control over the retrieval pipeline with SQL-based access:

```
Query → Embedding (text-embedding-005)
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

- **Use Agent Platform Search as the default** unless you have specific requirements for SQL-based retrieval or need to co-locate vectors with relational data. It handles chunking, embedding, indexing, and re-ranking with zero infrastructure.
- **Use AlloyDB AI when your data is already in PostgreSQL** — it can generate embeddings via SQL functions that call Vertex AI, eliminating the need for external embedding pipelines.
- **Use text-embedding-005** as the default embedding model — native GCP integration, 768 dimensions, strong multilingual performance.
- **Enable grounding** in Gemini API calls — Gemini supports built-in grounding with Google Search or your own data stores, reducing hallucination without a full RAG pipeline.
- **Use metadata filters** to scope retrieval by document type, customer, date range, or access level.
- **Set retrieval to return 5–10 chunks** — more context doesn't always improve quality and increases token cost.
- **Implement re-ranking** for quality-sensitive applications — use a cross-encoder model or the built-in Agent Platform Search ranker to improve precision.

### Multimodal RAG

Agent Platform Search supports multimodal ingestion:

- Images and diagrams are processed alongside text using Gemini's native vision capabilities.
- Tables are extracted and preserved as structured content.
- PDF layout is preserved with reading order detection.
- Charts and figures can be captioned by Gemini during indexing for semantic search.

### Chunking strategy

| Document Type | Chunking Approach | Configuration |
|---------------|-------------------|---------------|
| Structured reports (PDF, DOCX) | Layout-aware chunking (Agent Platform Search built-in) | Respects headings, sections, paragraphs |
| Legal / compliance documents | Hierarchical chunking | Parent: section, child: paragraph |
| Code documentation | Fixed-size with overlap | 512 tokens, 128 overlap |
| Tabular data | Row-per-chunk or table-per-chunk | Preserve column headers |
| Mixed multimedia | Multimodal chunking | Enable figure extraction + captioning |
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
      3. Cloud Run job: chunk + embed (text-embedding-005)
      4. Agent Platform Search / AlloyDB: index chunks
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

### Agent Development Kit (ADK) 2.0

ADK is Google's open-source, code-first framework for building production agents. It's the primary pro-code path on GCP and is designed to work with Agent Engine for managed deployment.

**Key characteristics:**

| Feature | Description |
|---------|-------------|
| **Multi-language** | Python, TypeScript, Go, Java, Kotlin |
| **Model-agnostic** | Works with Gemini, Claude, Llama, or any model backend |
| **Deployment-agnostic** | Deploy to Agent Engine, Cloud Run, GKE, or any container platform |
| **Built-in orchestration** | Sequential, parallel, loop, and conditional agent compositions |
| **MCP support** | Native MCP client for connecting to tool servers |
| **A2A support** | Native Agent-to-Agent protocol for inter-agent communication |
| **OpenTelemetry** | Built-in instrumentation for observability |
| **Evaluation** | Integrated evaluation framework for testing agent behaviour |

### Agent Engine — the managed runtime

Agent Engine (formerly Vertex AI Agent Builder Deployments) is the managed runtime for deploying agents at scale. It handles:

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

### MCP on Google Cloud (2026)

MCP (Model Context Protocol) is a first-class citizen on the Agent Platform:

- **Managed MCP servers** for Google Cloud services — BigQuery, Cloud SQL, Cloud Storage, Firestore, and others. Agents can query databases, read files, and interact with cloud services via standardised MCP tools.
- **Cloud Run as MCP host** — deploy custom MCP servers as Cloud Run services with automatic scaling, IAM authentication, and private networking.
- **ADK MCP client** — native MCP client in ADK for connecting agents to any MCP-compliant tool server.
- **MCP Tool Registry** — discover and manage MCP tools available to your agents.

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
2. **Use Agent Engine for managed deployment** — it handles autoscaling, session isolation, and IAM. Graduate to Cloud Run only when you need custom networking, GPUs, or unsupported frameworks.
3. **Use managed MCP servers for Google Cloud services** — don't write custom integrations for BigQuery, Cloud SQL, or Storage when an official MCP server exists.
4. **Scope IAM per agent** — each agent's service account should have only the permissions it needs. Use Workload Identity Federation for external service access.
5. **Set token budgets and timeout limits** — prevent runaway reasoning loops. Configure max output tokens and session timeouts per agent.
6. **Enable Memory Bank selectively** — short-term memory for all multi-turn agents; long-term memory only for agents that genuinely benefit from cross-session recall (personalisation, learning assistants).
7. **Instrument with OpenTelemetry** — ADK has built-in OTel support. Export traces to Cloud Trace and metrics to Cloud Monitoring from day one.
8. **Use A2A for cross-boundary communication** — when agents are built by different teams or need to communicate across trust boundaries, A2A provides the standardised protocol. Within a single team's codebase, direct function calls or ADK orchestration are simpler.

### When to use Agent Engine vs. Cloud Run

| Criterion | Agent Engine (managed) | Cloud Run (self-managed) |
|-----------|------------------------|--------------------------|
| Time to production | Days | Weeks |
| Custom networking (VPC, PSC) | Limited (PSC-I deployments available) | Full control |
| GPU workloads | Not supported | Supported (GPU node pools) |
| Framework support | ADK, LangGraph, CrewAI | Any framework |
| Autoscaling | Automatic | Configure min/max instances |
| Session management | Built-in (Sessions + Memory Bank) | BYO (Redis, Firestore) |
| Cost at low volume | Higher per-invocation | Scale-to-zero |
| Cost at high volume | Competitive | Lower (own compute) |
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
             ├── Gemini 3.1 Pro (complex queries)
             ├── Gemini 3.5 Flash (simple queries)
             └── Claude Sonnet 4.6 (specific tasks)
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
| Agent sessions (managed) | Agent Engine | No infrastructure, built-in sessions + memory, ADK-native |
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
| **Ironwood (TPU v7)** | Inference-first | 10x over v5p, 4x over v6e. 4.6 PFLOPS per chip. GA April 2026 |
| **Trillium (TPU v6e)** | Training + inference | Production-ready, cost-effective for medium-scale |
| **TPU v5p** | Large-scale training | UltraClusters for frontier model training |
| **TPU 8t "Sunfish" (preview)** | Training (Broadcom, 2nm) | Next-gen training, late 2027 |
| **TPU 8i "Zebrafish" (preview)** | Inference (MediaTek, 2nm) | Next-gen inference, late 2027 |
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
2. **Use Agent Engine** when you want zero infrastructure — it handles scaling, sessions, and memory automatically. Only move to Cloud Run when you need custom networking or GPUs.
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

VPC-SC creates security perimeters around Google Cloud services, preventing data exfiltration even if IAM is misconfigured.

**Configure VPC-SC perimeters around:**

- Agent Platform (model invocations, agent sessions)
- Cloud Storage (document corpus, model artefacts)
- BigQuery (structured data, analytics)
- AlloyDB / Cloud SQL (operational data, vector stores)
- Agent Platform Search (enterprise search indexes)
- Cloud Logging (audit logs containing model interactions)

**Access levels** define who can cross the perimeter boundary:

- Corporate network CIDR ranges.
- Specific service accounts.
- Device trust (BeyondCorp Enterprise).
- Region restrictions.

### Private Service Connect (PSC)

PSC provides private connectivity to Google APIs without traversing the public internet:

- **PSC endpoints** for Gemini/Agent Platform API calls — traffic stays on Google's backbone.
- **PSC for Agent Engine** — deploy agents with private networking (PSC-I deployment mode).
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
2. **Use Private Service Connect** for all model invocations in production — eliminates public internet traversal for API calls.
3. **One service account per agent** — scope permissions to exactly what each agent needs. Never share service accounts across agents with different trust levels.
4. **Use Workload Identity Federation** for external access — avoid service account keys. Map GitHub Actions, Azure AD, or AWS IAM to GCP service accounts.
5. **Enable Cloud Audit Logs** for all AI services — Data Access logs capture who invoked which model with what parameters.
6. **Use CMEK** (Customer-Managed Encryption Keys) for sensitive workloads — encrypt model artefacts, vector stores, and agent sessions with keys you control.
7. **Implement egress controls** — use VPC firewall rules and organisation policies to prevent agents from exfiltrating data to unauthorised endpoints.
8. **Use Cloud Armor** with pre-configured WAF rules — the OWASP top 10 rule set plus custom rules for LLM-specific attacks (prompt injection patterns, jailbreak attempts).

---

## Responsible AI and Content Safety

### Model Armor

Model Armor is the Agent Platform's built-in content safety layer, providing input and output filtering for responsible AI:

| Capability | Description |
|------------|-------------|
| **Prompt injection detection** | Identifies and blocks prompt injection attempts before they reach the model |
| **Content filtering** | Configurable thresholds for harmful content (hate speech, violence, sexual content, dangerous activities) |
| **PII detection** | Identifies personal information in prompts and responses |
| **Grounding checks** | Verifies model responses are grounded in provided context (reduces hallucination) |
| **Custom policies** | Define organisation-specific content rules |
| **Audit logging** | All filtering decisions logged for compliance review |

### Implementing responsible AI

1. **Apply Model Armor to all production agents** — configure appropriate thresholds per use case. A customer-facing chatbot needs stricter filtering than an internal code assistant.
2. **Use the safety settings API** — Gemini models support configurable safety thresholds per category (harassment, hate speech, sexually explicit, dangerous content).
3. **Implement human-in-the-loop for high-stakes decisions** — agents handling financial, legal, or medical advice should escalate to humans rather than acting autonomously.
4. **Log all model interactions** — maintain audit trails for compliance. Use Cloud Logging with appropriate retention policies.
5. **Test adversarial inputs** — include prompt injection, jailbreak attempts, and boundary-testing queries in your evaluation suite.
6. **Use Cloud DLP for output sanitisation** — scan model responses for accidental PII leakage before returning to users.
7. **Implement rate limiting per user** — prevents abuse of AI capabilities and limits blast radius of compromised accounts.
8. **Define clear escalation paths** — when Model Armor blocks a request, provide helpful guidance rather than opaque errors.

---

## Cost Management

### Pricing model overview

GCP AI pricing has three primary dimensions:

| Dimension | How it's charged | Key lever |
|-----------|-----------------|-----------|
| **Model invocation** | Per 1M input/output tokens (varies by model) | Model selection, prompt caching, prompt length |
| **Agent Platform services** | Per-request for Agent Engine, Sessions, Memory Bank | Session duration, memory retention policy |
| **Compute** | Per vCPU-second / GPU-hour / TPU chip-hour | Instance sizing, scale-to-zero, spot pricing |
| **Storage + retrieval** | Per GB stored + per query (Agent Platform Search, vector stores) | Data lifecycle policies, query volume |

### Token pricing tiers (indicative, May 2026)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Context caching |
|-------|----------------------|------------------------|-----------------|
| Gemini 3.5 Flash | ~$0.10 | ~$0.40 | 75% reduction on cached |
| Gemini 3.1 Pro | ~$1.25 | ~$5.00 | 75% reduction on cached |
| Claude Sonnet 4.6 (on Vertex) | ~$3.00 | ~$15.00 | Prompt caching available |
| Claude Haiku 4.5 (on Vertex) | ~$0.80 | ~$4.00 | Prompt caching available |
| text-embedding-005 | ~$0.00005 | — | — |

*Prices are indicative and region-dependent. Check the pricing page for current rates.*

### Cost optimisation strategies

1. **Use Gemini 3.5 Flash as the default** — it's 10–25x cheaper than Pro for many tasks. Only upgrade when evaluation shows Flash can't meet quality requirements.
2. **Enable context caching** — for prompts with reusable system instructions, few-shot examples, or document context, caching reduces input token cost by up to 75%. Cached content must be >32K tokens and persists for a configurable TTL.
3. **Implement model routing** — use a lightweight classifier or heuristic to route simple queries to Flash and complex queries to Pro. This alone can reduce costs 40–60%.
4. **Use batch API for non-real-time workloads** — evaluation, document processing, and offline analysis can use batch pricing (typically 50% discount).
5. **Scale to zero** — Cloud Run scales to zero by default. Set minimum instances only for latency-critical paths.
6. **Use Spot VMs for batch workloads** — 60–91% savings on GKE nodes used for evaluation, batch inference, and data processing.
7. **Right-size Cloud Run instances** — most AI gateway workloads need minimal CPU/memory since inference runs remotely. 1 vCPU / 512 MiB is often sufficient.
8. **Set per-request token budgets** — configure `max_output_tokens` to prevent runaway generation. A 4,000-token cap on a summarisation task prevents accidentally generating 100K tokens.
9. **Monitor token usage in real-time** — export metrics to Cloud Monitoring and set alerts for unexpected spikes.
10. **Negotiate committed use discounts** — at $1M+ annual AI spend, Google Cloud offers 25–50% discounts on Vertex AI / Agent Platform consumption as part of broader cloud commitments.

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

Terraform is the primary IaC tool for GCP AI infrastructure. The `google` and `google-beta` providers cover Agent Platform resources.

**Key resources:**

| Terraform Resource | Purpose |
|-------------------|---------|
| `google_vertex_ai_endpoint` | Model serving endpoints |
| `google_discovery_engine_search_engine` | Agent Platform Search apps |
| `google_discovery_engine_data_store` | Search data stores |
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
│   ├── agent-platform/      # Agent Engine, Search, Model Garden config
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

### Agent Engine deployment with Terraform

Agent Engine deployments are managed through the ADK CLI or `gcloud` rather than pure Terraform, since agents contain application code. The recommended pattern:

1. **Terraform** provisions infrastructure: networking, databases, service accounts, IAM, monitoring.
2. **ADK CLI / `agents deploy`** deploys agent code to Agent Engine, referencing Terraform-managed infrastructure.
3. **CI/CD pipeline** orchestrates both: Terraform applies first, then agent deployment.

### Best practices

1. **Use remote state in GCS** with state locking via Cloud Storage's generation-based preconditions.
2. **Separate state files per environment** — dev/staging/prod should never share Terraform state.
3. **Use the `google-beta` provider** for Agent Platform resources that are newly GA — the `google` provider lags behind on new resources.
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
          ├── Deploy to staging (Agent Engine or Cloud Run)
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
| **Prebuilt connectors** | Integration with Google Workspace, third-party SaaS via managed MCP servers |
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

**Pattern: Gemini Flash + Cloud Run + Agent Platform Search**

```
Users → Cloud Run (Next.js / FastAPI frontend)
          → Gemini 3.5 Flash API (reasoning)
          → Agent Platform Search (document retrieval)
          → Cloud Storage (document upload)
```

**Key decisions:**

- Use Gemini 3.5 Flash exclusively — cost-efficient and capable enough for most business tasks.
- Agent Platform Search for RAG — no vector database to manage.
- Cloud Run for everything — single service, scale-to-zero, minimal ops.
- Firebase Auth for user management.
- Minimal infrastructure: one GCP project, no VPC-SC, no separate environments.

**Typical use cases:** Document Q&A, customer support chatbot, internal knowledge base, email drafting assistant.

### Medium business (50–500 employees, $5K–$50K/month AI spend)

**Pattern: Multi-model + Cloud Run + AlloyDB + Agent Engine**

```
Users → Cloud Run (application layer, model routing)
          ├── Gemini 3.5 Flash (simple queries, classification)
          ├── Gemini 3.1 Pro (complex reasoning, analysis)
          └── Claude Sonnet 4.6 (specific tasks where it excels)
        → Agent Engine (multi-step agents)
        → AlloyDB AI (vector store + operational data)
        → Document AI (document processing pipeline)
        → Cloud Workflows (orchestration)
```

**Key decisions:**

- Model routing to optimise cost/quality trade-off.
- Agent Engine for managed agent deployment — avoid infrastructure overhead.
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
              ├── Agent Engine (managed agents)
              ├── ADK agents on Cloud Run (custom agents)
              └── Self-hosted models on GKE (Gemma, domain-specific)
            → AlloyDB (multi-region, vector + relational)
            → Agent Platform Search (enterprise search)
            → Spanner (global consistency for critical state)
            → BigQuery (analytics, evaluation datasets, billing)
```

**Key decisions:**

- Apigee for multi-tenant API management with per-customer quotas and analytics.
- GKE Autopilot for self-hosted models and custom workloads requiring GPUs/TPUs.
- Multi-region deployment for availability and data residency.
- Full VPC-SC perimeters, Private Service Connect, CMEK.
- Comprehensive evaluation pipeline with trajectory testing and human review.
- Dedicated SRE with SLOs, error budgets, and incident management.
- A2A protocol for cross-team agent collaboration.
- Committed use discounts negotiated with Google Cloud.

**Typical use cases:** Organisation-wide AI platform, multi-agent systems spanning business units, customer-facing AI products, regulated industry applications (financial services, healthcare, government).

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It's Bad | What to Do Instead |
|--------------|--------------|-------------------|
| **Using Vertex AI legacy APIs for new projects** | Deprecated; will not receive new features; migration debt accumulates | Use Gemini Enterprise Agent Platform APIs and SDKs |
| **Defaulting to Gemini Pro for all requests** | 10–25x more expensive than Flash; most queries don't need Pro-level reasoning | Implement model routing; start with Flash, upgrade based on evaluation |
| **Skipping evaluation in CI** | Prompt/model changes break things silently; quality regressions reach production undetected | Run automated evaluation on every change; block deployment on regression |
| **Monolithic agents** | Hard to debug, test, and maintain; single failure point; token-hungry system prompts | Decompose into single-purpose agents orchestrated via ADK or A2A |
| **Storing vectors in Cloud SQL without pgvector tuning** | Default PostgreSQL config has terrible ANN performance at scale | Use AlloyDB AI (optimised for vector), tune `ivfflat` lists, or use Vertex AI Vector Search for large scale |
| **No token budgets on agents** | Runaway reasoning loops can consume entire daily budget in minutes | Set `max_output_tokens`, iteration limits, and timeout per agent |
| **Service account key files** | Security risk; hard to rotate; leak in git history | Use Workload Identity (GKE), Workload Identity Federation (external CI), or attached service accounts (Cloud Run) |
| **Single-project deployment** | No environment isolation; production changes tested on production | Separate GCP projects per environment with promotion pipeline |
| **No VPC-SC in regulated environments** | Data exfiltration risk even with correct IAM; compliance audit failure | Deploy VPC-SC perimeters from day one for any workload handling sensitive data |
| **Building custom integrations for standard GCP services** | Reinventing what managed MCP servers already provide; maintenance burden | Use official managed MCP servers for BigQuery, Cloud SQL, Storage |
| **Ignoring context caching** | Paying full input token price on every request for repeated system prompts | Enable context caching for prompts with >32K tokens of reusable context |
| **Using Dialogflow CX for new text-only agents** | More complex than needed; Agent Studio + ADK is the modern path | Use Agent Studio for low-code or ADK for pro-code; reserve Dialogflow CX for telephony |
| **No observability for agent reasoning** | Can't debug why an agent made a specific decision; incidents are undiagnosable | Instrument with OpenTelemetry from day one; trace every tool call and model invocation |

---

## References

1. Google Cloud Blog — "Gemini Enterprise Agent Platform optimizes your agents" (April 2026). https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/gemini-enterprise-agent-platform/
2. Google Cloud Next 2026 announcements hub. https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/next-2026/
3. GCP Study Hub — "Vertex AI Is Now Gemini Enterprise Agent Platform: What Changed in 2026". https://gcpstudyhub.com/blog/vertex-ai-replaced-by-gemini-enterprise-agent-platform
4. Agent Development Kit (ADK) documentation. https://adk.dev/
5. Google Cloud documentation — Gemini Enterprise Agent Platform. https://docs.cloud.google.com/gemini-enterprise-agent-platform
6. Google Cloud — Model Garden. https://cloud.google.com/model-garden
7. Google Blog — "Ironwood: The first Google TPU for the age of inference". https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/
8. A2A Protocol specification. https://a2a-protocol.org/latest/
9. Google Cloud Blog — "Building Connected Agents with MCP and A2A". https://cloud.google.com/blog/topics/developers-practitioners/building-connected-agents-with-mcp-and-a2a
10. Google Cloud documentation — Agent Platform Search. https://docs.cloud.google.com/gemini-enterprise-agent-platform
11. Google Cloud documentation — VPC Service Controls. https://docs.cloud.google.com/vpc-service-controls/docs
12. Google Cloud documentation — Host AI agents on Cloud Run. https://docs.cloud.google.com/run/docs/ai-agents
13. Google Cloud documentation — Document AI overview. https://docs.cloud.google.com/document-ai/docs/overview
14. Google Cloud documentation — Instrument ADK applications with OpenTelemetry. https://docs.cloud.google.com/stackdriver/docs/instrumentation/ai-agent-adk
15. Terraform Registry — GoogleCloudPlatform/vertex-ai/google. https://registry.terraform.io/modules/GoogleCloudPlatform/vertex-ai/google/latest
16. Google Developers Blog — "Agents CLI in Agent Platform: create to production in one CLI". https://developers.googleblog.com/agents-cli-in-agent-platform-create-to-production-in-one-cli/
17. Google Cloud documentation — RAG infrastructure using Vertex AI and AlloyDB. https://docs.cloud.google.com/architecture/rag-capable-gen-ai-app-using-vertex-ai
18. InfoQ — "Google Cloud Brings Full OpenTelemetry Support to Cloud Monitoring" (March 2026). https://www.infoq.com/news/2026/03/google-cloud-opentelemetry/
19. Claude on Vertex AI documentation. https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai
20. Doolpa — "Google Ironwood TPU GA, TPU 8 Splits Training & Inference" (April 2026). https://doolpa.com/news/google-ironwood-tpu-general-availability-tpu-8-split-cloud-next-april-2026
