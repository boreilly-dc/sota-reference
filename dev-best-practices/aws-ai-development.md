# Best Practices for AI Development on AWS in 2026

| Field | Value |
|-------|-------|
| Created | 2026-05-26 |
| Last Updated | 2026-05-26 |
| Version | 1.0 |

---

- [Executive Summary](#executive-summary)
- [Platform Architecture: Amazon Bedrock](#platform-architecture-amazon-bedrock)
- [Model Selection Strategy](#model-selection-strategy)
- [RAG and Knowledge Retrieval](#rag-and-knowledge-retrieval)
- [Document Processing Pipelines](#document-processing-pipelines)
- [Agent Development](#agent-development)
- [AI Gateway with API Gateway](#ai-gateway-with-api-gateway)
- [Hosting and Compute Patterns](#hosting-and-compute-patterns)
- [Security and Networking](#security-and-networking)
- [Guardrails and Content Safety](#guardrails-and-content-safety)
- [Cost Management](#cost-management)
- [Observability and Monitoring](#observability-and-monitoring)
- [Evaluation and Testing](#evaluation-and-testing)
- [Infrastructure as Code](#infrastructure-as-code)
- [CI/CD for AI Applications](#cicd-for-ai-applications)
- [Architecture Patterns by Business Size](#architecture-patterns-by-business-size)
- [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
- [References](#references)

## Executive Summary

AWS's AI platform consolidated significantly through 2025–2026. **Amazon Bedrock** is now the unified foundation for building generative AI applications — providing a model catalogue of 103+ models from 18+ providers, managed RAG (Knowledge Bases), agent orchestration, guardrails, evaluation, and prompt management. At re:Invent 2025, AWS introduced **Amazon Bedrock AgentCore** — a full platform for deploying production agents with managed runtime, identity, gateway, and memory services — alongside the **Strands Agents SDK** (open-source Python SDK for agent development).

The most significant 2026 developments:

1. **OpenAI models on Bedrock** (April 2026) — GPT-5.4, GPT-5.5, and Codex are now available, ending Microsoft's Azure exclusivity. Usage counts toward AWS cloud commitments.
2. **Amazon S3 Vectors** (GA January 2026) — native vector storage in S3 at up to 90% lower cost than conventional vector databases. Changes the economics of RAG at scale.
3. **Stateful MCP support** (March 2026) — AgentCore Runtime supports the full MCP protocol including elicitation, sampling, and progress notifications.
4. **Amazon Nova 2 family** — Nova 2 Lite, Pro, Omni (multimodal in/out), and Sonic (speech-to-speech) with extended thinking capabilities.
5. **Kiro** — AWS's new spec-driven agentic IDE replacing Amazon Q Developer (end-of-support April 2027).

For contractors building cloud AI systems, the AWS landscape has three tiers:

1. **Pro-code (Bedrock + AgentCore + Strands SDK)** — full control over models, retrieval, orchestration, and deployment. Suitable for enterprise RAG, multi-agent platforms, and custom AI applications.
2. **Managed (Amazon Q Business)** — rapid deployment of knowledge assistants with 40+ data source connectors and permission-aware responses. Suitable for enterprise search and internal Q&A.
3. **Hybrid** — Q Business for user-facing search with Bedrock Agents providing reasoning backends and custom tool integrations.

This guide covers production best practices across all three, with emphasis on the pro-code path.

---

## Platform Architecture: Amazon Bedrock

Amazon Bedrock is the serverless foundation model platform on AWS. Unlike Azure's hub/project model, Bedrock is a regional service accessed directly via IAM-authenticated API calls.

| Component | Purpose |
|-----------|---------|
| **Model Catalogue** | 103+ models from Anthropic, OpenAI, Meta, Mistral, Cohere, Amazon, and others |
| **Knowledge Bases** | Managed RAG: ingestion, chunking, embedding, indexing, and retrieval |
| **Agents** | Orchestration runtime with planning, tool calling, memory, and knowledge base integration |
| **AgentCore** | Production platform: Runtime, Identity, Gateway, Memory services |
| **Guardrails** | Content filtering, PII detection, grounding checks, automated reasoning |
| **Flows** | Visual workflow builder for multi-step AI pipelines |
| **Prompt Management** | Version, evaluate, and deploy prompts with optimisation tooling |
| **Model Evaluation** | Built-in automated and human evaluation jobs |

### Account topology

```
AWS Organization
├── Shared Services Account
│   ├── API Gateway (AI Gateway)
│   ├── CloudWatch (centralised observability)
│   └── IAM Identity Center
├── AI Development Account
│   ├── Bedrock (dev models + Knowledge Bases)
│   ├── S3 (document storage)
│   └── Lambda / Container Apps (dev compute)
├── AI Staging Account
│   └── Mirror of production with lower capacity
└── AI Production Account
    ├── Bedrock (production model deployments)
    ├── AgentCore Runtime
    ├── OpenSearch Serverless / S3 Vectors
    ├── Lambda / ECS (application layer)
    └── VPC with private endpoints
```

**Best practices:**

- Use **separate AWS accounts** per environment (dev/staging/prod) within an AWS Organisation — this provides hard isolation for IAM, networking, and cost attribution.
- Deploy Bedrock resources in the **same region as your compute** to minimise latency. Use cross-region inference profiles for capacity failover.
- Enable **model invocation logging** at the account level from day one — logs to CloudWatch Logs and S3 for audit, debugging, and evaluation.
- Use **IAM roles** (not access keys) for all service-to-service authentication. Bedrock supports identity-based policies with per-model granularity.
- Configure **SCPs (Service Control Policies)** at the organisation level to restrict which models can be invoked in production accounts.

---

## Model Selection Strategy

The Bedrock catalogue spans frontier proprietary models and open-source alternatives across 30+ regions. Model selection should be driven by task requirements and cost constraints.

### Decision matrix

| Use Case | Recommended Model | Alternative (Open-Source) | Pricing Tier |
|----------|-------------------|---------------------------|--------------|
| Complex reasoning / long-context | Claude Opus 4.6 / GPT-5.4 | Llama 4 Maverick | Standard or Priority |
| High-volume chat / simple Q&A | Claude Haiku 4.5 / Nova 2 Lite | Mistral Small | Standard or Flex |
| Multimodal (vision + text) | Claude Sonnet 4.6 / Nova 2 Omni | Llama 4 Scout | Standard |
| Code generation | Claude Opus 4.6 / GPT-5.5 Codex | DeepSeek-V3 | Standard |
| Embedding | Titan Embeddings V2 | Cohere Embed v4 | Standard |
| Edge / lightweight | Nova Micro / Nova 2 Lite | Phi-4-mini (SageMaker) | Standard |
| Structured extraction | Claude Sonnet 4.6 (tool use) | Mistral Large | Standard |
| Speech-to-speech | Nova 2 Sonic | Whisper (SageMaker) | Standard |

### Model selection principles

1. **Start with Claude Haiku or Nova 2 Lite** for cost-efficient prototyping — upgrade to larger models only when evaluation metrics demand it.
2. **Use Intelligent Prompt Routing** for workloads with mixed complexity — Bedrock dynamically routes between models within a family based on predicted quality (up to 30% cost reduction).
3. **Pin model versions** in production (use full model IDs like `anthropic.claude-sonnet-4-6-20250514-v1:0`) — auto-upgrades break prompts.
4. **Benchmark on your data** — use Bedrock Model Evaluation or DeepEval to compare models on your actual workload before committing.
5. **Use cross-region inference profiles** for high-availability — Bedrock routes requests to the optimal region automatically.
6. **Leverage prompt caching** for repeated context — available on Claude and Nova models, up to 90% cost reduction on cached tokens.
7. **Consider service tiers** — Priority for latency-critical paths, Standard for everyday use, Flex (50% discount) for background processing.

---

## RAG and Knowledge Retrieval

### Bedrock Knowledge Bases — the managed retrieval layer

Bedrock Knowledge Bases provide end-to-end managed RAG: point at an S3 bucket, and the service handles chunking, embedding, indexing, and retrieval. In 2026, it supports multiple vector store backends and multimodal content.

| Vector Store | When to Use |
|--------------|-------------|
| **Amazon S3 Vectors** (GA Jan 2026) | Massive scale, up to 90% cost reduction, serverless, no provisioning. Default for new projects. |
| **OpenSearch Serverless** | Real-time hybrid search (keyword + vector), full-text capabilities, complex filtering |
| **Aurora PostgreSQL pgvector** | Teams already on Aurora wanting relational + vector in one database |
| **Amazon Neptune Analytics** | Graph + vector hybrid queries (knowledge graphs) |
| **Pinecone / Redis / MongoDB Atlas** | Third-party integrations via Knowledge Bases connectors |

### Retrieval architecture

For most enterprise RAG deployments, hybrid search with OpenSearch Serverless is the highest-quality starting point:

```
Query → OpenSearch Serverless
         ├── BM25 (keyword)
         ├── k-NN (vector)
         └── Fusion (RRF) → Re-rank → Top-K chunks → LLM
```

For cost-sensitive or high-scale deployments, S3 Vectors provides a simpler path:

```
Query → Embedding → S3 Vectors (ANN search) → Top-K chunks → LLM
```

**Best practices:**

- **Use S3 Vectors as the default for new projects** unless you need hybrid search (keyword + semantic). It's 90% cheaper than OpenSearch Serverless and requires zero provisioning.
- **Use OpenSearch Serverless when you need hybrid search** — combining BM25 keyword matching with vector similarity produces better results than either alone for enterprise document corpora.
- **Use Titan Embeddings V2 (1024 dimensions)** as the default embedding model on AWS — native integration, good quality/cost trade-off.
- **Enable integrated vectorisation** in Knowledge Bases — let the service handle embedding at both index and query time.
- **Use metadata filters** to scope retrieval by document type, customer, date range, or access level.
- **Set retrieval to return 5–10 chunks** — more context doesn't always improve quality and increases token cost.

### Multimodal RAG (re:Invent 2025)

Knowledge Bases now support multimodal ingestion:

- Images and diagrams are processed alongside text.
- Tables are extracted and preserved as structured content.
- Charts and figures can be captioned by a vision model during indexing.

Use this for document corpora containing technical drawings, architectural diagrams, or data visualisations.

### Chunking strategy

| Document Type | Chunking Approach | Configuration |
|---------------|-------------------|---------------|
| Structured reports (PDF, DOCX) | Semantic chunking (Knowledge Bases built-in) | Max 512 tokens, respect paragraph boundaries |
| Legal / compliance documents | Hierarchical chunking | Parent: section, child: paragraph |
| Code documentation | Fixed-size with overlap | 512 tokens, 128 overlap |
| Tabular data | Row-per-chunk or table-per-chunk | Preserve column headers in each chunk |
| Mixed multimedia | Multimodal chunking | Enable figure extraction |

---

## Document Processing Pipelines

### Amazon Textract

Textract provides ML-powered document extraction beyond OCR. It's the entry point for document ingestion pipelines on AWS.

**APIs:**

| API | Use Case |
|-----|----------|
| **DetectDocumentText** | Basic text extraction from scanned documents |
| **AnalyzeDocument** | Tables, forms, key-value pairs, layout detection |
| **AnalyzeExpense** | Invoice and receipt processing |
| **AnalyzeID** | Identity document extraction (passports, licences) |
| **Analyze Lending** | Loan document package processing |
| **Custom Queries** | Natural language questions about document content |

### Amazon Comprehend

Comprehend provides NLP capabilities for post-extraction processing:

- Entity recognition (people, places, organisations, dates).
- PII detection and classification (for redaction workflows).
- Sentiment analysis and key phrase extraction.
- Custom classification and entity models.
- **Comprehend Medical** — healthcare-specific NLP for PHI and medical entities.

### Production pipeline architecture

```
Source (S3 Bucket)
  → EventBridge rule (new object notification)
    → Step Functions workflow:
      1. Textract: extract layout + tables + forms
      2. Comprehend: classify document type + detect PII
      3. Lambda: chunk + embed (Titan Embeddings V2)
      4. Knowledge Base / OpenSearch: index chunks
      5. DynamoDB: store processing status + metadata
```

**Best practices:**

1. **Use Step Functions** (not Lambda chaining) for orchestrating multi-step pipelines — provides retry logic, error handling, parallel processing, and visual debugging.
2. **Use Textract's Layout API** for documents with complex formatting — it preserves reading order, headings, and structural hierarchy.
3. **Process asynchronously** — Textract has rate limits. Use SNS notifications for completion rather than polling.
4. **Store raw extracted content in S3** alongside the original document — enables re-chunking without re-extraction if your strategy changes.
5. **Tag each chunk with source metadata** — document ID, page number, section heading, extraction confidence — for citation generation downstream.
6. **Use Comprehend PII detection** before indexing for regulated workloads — redact or flag sensitive content before it enters the vector store.
7. **For large corpora (10,000+ documents)**, use Textract batch processing with dedicated throughput and Step Functions distributed map for parallel processing.

---

## Agent Development

### Amazon Bedrock Agents

Bedrock Agents provide managed orchestration: a model paired with a planner, action groups (tools), optional memory, and knowledge base access — all running serverlessly.

**Key concepts:**

| Concept | Description |
|---------|-------------|
| **Agent** | An LLM with a system prompt, action groups, and optional knowledge base access |
| **Action Group** | A set of tools defined as OpenAPI schemas or Lambda functions |
| **Knowledge Base** | Attached RAG retrieval — the agent decides when to search |
| **Memory** | Cross-session context retention (short-term and long-term) |
| **Guardrails** | Content safety and grounding checks applied per-invocation |
| **Session** | Conversation state with automatic context management |

### Amazon Bedrock AgentCore (2026)

AgentCore is the production platform for deploying and scaling AI agents. It provides:

| Service | Purpose |
|---------|---------|
| **AgentCore Runtime** | Serverless, secure hosting with session isolation for thousands of concurrent users |
| **AgentCore Identity** | Multi-IDP authentication via Cognito integration |
| **AgentCore Gateway** | Centralised MCP server connections, auth, and policy enforcement |
| **AgentCore Memory** | Short-term, long-term, semantic, episodic, and procedural memory strategies |

### Strands Agents SDK (open-source)

Strands is AWS's open-source Python SDK for building agents. It integrates with AgentCore Runtime for deployment but is not locked to it.

- Supports any model backend (Bedrock, OpenAI, Vertex AI, local models).
- Supports any protocol (MCP, A2A agent-to-agent).
- Built-in OpenTelemetry instrumentation.
- Memory management via AgentCore Memory.

### MCP support on AWS (2026)

MCP is a first-class citizen on AWS as of 2026:

- **AgentCore Gateway** centralises MCP server connections with OAuth authentication and policy enforcement.
- **Stateful MCP** (March 2026) — full protocol support including elicitation, sampling, and progress notifications.
- **Stateful MCP Client** (April 2026) — interactive multi-turn agent workflows.
- **A2A protocol** support — agents communicate with each other via standard protocol.
- **AG-UI protocol** (March 2026) — responsive, real-time user-facing agent applications.

### Agent orchestration patterns

From AWS guidance, the canonical multi-agent patterns:

1. **Supervisor** — an orchestrator agent routes queries to specialist agents based on intent, collects and synthesises responses.
2. **Sequential pipeline** — agents execute in a fixed order (extract → validate → enrich → respond).
3. **Parallel fan-out** — multiple agents work simultaneously on different sub-tasks, results are merged.
4. **Peer-to-peer (A2A)** — agents communicate directly via the A2A protocol without a central orchestrator.
5. **Hierarchical** — nested supervisor agents managing sub-teams of specialists.

### Best practices for production agents

1. **Keep agents single-purpose** — one agent, one job. Compose complex behaviours through orchestration, not monolithic prompts.
2. **Use AgentCore Gateway for MCP** — centralises tool connections, handles auth, and enforces policy across the organisation. Don't wire MCP servers directly into each agent.
3. **Scope IAM per action group** — each Lambda function behind an action group should have its own minimal-privilege IAM role. The agent's execution role controls which action groups it can invoke.
4. **Set token budgets and timeout limits** — prevent runaway reasoning loops from consuming capacity. Configure `maxTokens` and `idleSessionTTL` per agent.
5. **Enable memory selectively** — short-term memory for multi-turn conversations, long-term memory only for agents that genuinely benefit from cross-session recall (e.g., personalisation agents).
6. **Log every tool invocation** with OpenTelemetry spans via Strands SDK instrumentation — you cannot debug agents without observability.
7. **Use Bedrock Agents for simple tool-calling scenarios** — graduate to AgentCore + Strands only when you need custom orchestration, multi-agent patterns, or self-hosted compute.

### When to use Bedrock Agents vs. AgentCore + Strands

| Criterion | Bedrock Agents (managed) | AgentCore + Strands (self-managed) |
|-----------|--------------------------|-------------------------------------|
| Time to production | Days | Weeks |
| Custom orchestration | Limited (single agent, tool-calling) | Full control (multi-agent, custom routing) |
| Multi-agent workflows | Basic (supervisor with sub-agents) | Complex (A2A, hierarchical, custom patterns) |
| Model flexibility | Bedrock catalogue only | Any model (Bedrock, OpenAI, Vertex, local) |
| Cost at scale | Higher per-invocation | Lower (own compute via AgentCore Runtime) |
| State management | Built-in sessions + memory | BYO + AgentCore Memory |
| MCP support | Via AgentCore Gateway | Direct or via Gateway |

---

## AI Gateway with API Gateway

Amazon API Gateway serves as the **AI Gateway** — a centralised control plane for all LLM traffic — sitting in front of Bedrock.

### Why every production deployment needs an AI Gateway

- **Authorisation** — API keys, IAM, Cognito JWT validation per consumer/tenant.
- **Usage quotas and throttling** — per-tenant rate limiting to prevent budget overruns.
- **Real-time streaming** — API Gateway supports response streaming for chat interfaces.
- **WAF integration** — AWS WAF rules for input validation, IP restriction, and bot protection.
- **Centralised observability** — all requests logged and metered in one place.
- **Multi-model routing** — Lambda behind API Gateway selects the optimal model per request.
- **Cost attribution** — tag requests with customer/project identifiers for chargeback.

### Architecture patterns

**Pattern 1: API Gateway + Lambda (standard)**

```
Client → API Gateway (REST/WebSocket)
           → Lambda (routing, transformation, retry logic)
             ├── Bedrock: Claude Sonnet (primary)
             ├── Bedrock: Claude Haiku (fallback / simple queries)
             └── Bedrock: Nova 2 Lite (cost-optimised tier)
```

**Pattern 2: Cross-Region with Inference Profiles**

```
Client → API Gateway
           → Lambda
             → Bedrock Cross-Region Inference Profile
               ├── us-east-1 (primary)
               ├── us-west-2 (failover)
               └── eu-west-1 (GDPR workloads)
```

**Pattern 3: CloudFront + API Gateway (edge caching)**

```
Client → CloudFront (edge caching for repeated prompts)
           → API Gateway
             → Lambda
               → Bedrock
```

**Best practices:**

1. **Use REST API (not HTTP API)** for production AI gateways — REST API supports usage plans, API keys, request/response transformation, and WAF integration.
2. **Implement model routing in Lambda** — classify intent or complexity, then route to the appropriate model tier. Simple queries → Haiku/Nova; complex reasoning → Opus/GPT-5.
3. **Use cross-region inference profiles** for automatic failover — Bedrock handles region routing without custom logic. This is simpler than building your own failover.
4. **Apply usage plans per API key** — enforce per-tenant rate limits and quotas. Map API keys to customer identifiers for cost attribution.
5. **Enable request/response logging** to CloudWatch — capture token usage per request for cost tracking.
6. **Use WebSocket API** for streaming responses — provides a better UX for chat interfaces than polling.
7. **Enable WAF** with rate-based rules — protects against prompt injection at volume and API abuse.

---

## Hosting and Compute Patterns

### Decision matrix

| Workload Characteristic | Recommended Service | Rationale |
|------------------------|---------------------|-----------|
| Event-driven, short-lived (<15 min) | Lambda | Scale-to-zero, pay-per-invocation, native Bedrock SDK |
| Long-running agent sessions (15+ min) | ECS Fargate | No timeout limit, GPU support, per-task IAM roles |
| High-throughput API with steady traffic | ECS Fargate (always-on) | Predictable latency, cost-efficient at scale |
| Custom model hosting (open-source) | SageMaker Endpoints | Managed ML infrastructure, Inferentia/Trainium support |
| Complex multi-container orchestration | EKS | Full Kubernetes control, KEDA autoscaling, GPU node pools |
| Frontend (React/Next.js) | Amplify Hosting / CloudFront + S3 | Global CDN, CI/CD integration |
| Batch processing / evaluation | Step Functions + Lambda | Distributed map for parallelism, no infrastructure |

### Lambda for AI workloads

Lambda is the default compute for Bedrock-backed applications:

- **15-minute timeout** — sufficient for most single-turn LLM interactions and tool-calling loops.
- **10 GB memory / 6 vCPU** — adequate for request handling (inference runs on Bedrock, not in Lambda).
- **Streaming responses** — Lambda supports response streaming via function URLs or API Gateway WebSocket.
- **Provisioned concurrency** — eliminates cold starts for latency-sensitive paths.
- **SnapStart** (Java/Python) — reduces cold start to <200ms.

**When Lambda isn't enough:**

- Agent sessions that run >15 minutes (complex multi-step reasoning).
- Workloads requiring persistent WebSocket connections.
- Custom model inference requiring GPU/Inferentia.

### ECS Fargate for AI workloads

- **No timeout limits** — suitable for long-running agent sessions and complex orchestrations.
- **Per-task IAM roles** — fine-grained access control per container.
- **GPU support** — run local models (Phi-4, Whisper, embedding models) on GPU instances.
- **Service Connect** — simplified service-to-service communication for multi-agent architectures.
- **Auto-scaling** — scale on CPU/memory, custom CloudWatch metrics, or SQS queue depth.

### SageMaker for custom models

Use SageMaker when you need to host open-source models outside the Bedrock catalogue:

- **SageMaker JumpStart** — one-click deployment of popular open-source models (Llama, Mistral, Falcon).
- **Inferentia2 instances** — 40–60% cost savings vs GPU for inference workloads. Neuron SDK for model compilation.
- **Trainium instances** — for fine-tuning and training custom models.
- **Real-time endpoints** — managed auto-scaling with traffic-based policies.
- **Serverless inference** — scale-to-zero endpoints for intermittent workloads.

### AWS custom silicon

| Chip | Purpose | Instance Type | Key Benefit |
|------|---------|---------------|-------------|
| **Trainium3** (2026) | Training + inference | Trn3 | First 3nm AI chip, UltraServer packs 144 chips |
| **Trainium2** | Training + inference | Trn2 | Production-ready, UltraClusters for large models |
| **Inferentia2** | Cost-optimised inference | Inf2 | 40–60% savings vs comparable GPUs |
| **Graviton4** | General compute | C7g/M7g/R7g | Best price/performance for non-GPU workloads |

---

## Security and Networking

### Network architecture

All production AI deployments on AWS should use **VPC endpoints (PrivateLink)** to ensure traffic never traverses the public internet:

```
VPC (AI Workload)
├── Private Subnets
│   ├── Lambda (VPC-attached)
│   ├── ECS Tasks
│   └── SageMaker Endpoints
├── VPC Endpoints (PrivateLink)
│   ├── com.amazonaws.region.bedrock-runtime
│   ├── com.amazonaws.region.bedrock-agent-runtime
│   ├── com.amazonaws.region.s3 (gateway)
│   ├── com.amazonaws.region.secretsmanager
│   ├── com.amazonaws.region.logs
│   └── com.amazonaws.region.monitoring
├── NAT Gateway (for external API calls only)
└── Security Groups
    ├── sg-lambda: egress to VPC endpoints only
    ├── sg-ecs: egress to VPC endpoints + NAT
    └── sg-vpc-endpoints: ingress from compute SGs
```

### Identity and access

| Principle | Implementation |
|-----------|---------------|
| No API keys in application code | IAM roles for all Bedrock access — Lambda execution role, ECS task role |
| Per-model access control | IAM policy with `bedrock:InvokeModel` scoped to specific model ARNs |
| User-scoped data access | Cognito user pools + IAM role mapping for per-user retrieval scoping |
| Least privilege | Separate IAM roles per agent action group — each tool gets minimal permissions |
| Organisation-level governance | SCPs restricting which models/regions are available in production accounts |
| Cross-account access | IAM roles with trust policies — never shared credentials |

### Security checklist for AI deployments

- [ ] VPC endpoints configured for Bedrock, S3, and all dependent services — no public internet traffic.
- [ ] IAM roles (not access keys) for all service-to-service authentication.
- [ ] Per-model IAM policies — production accounts only allow approved models.
- [ ] API Gateway with WAF as the single ingress point for client traffic.
- [ ] Guardrails enabled on all user-facing model invocations.
- [ ] Model invocation logging enabled — audit trail for all LLM calls.
- [ ] KMS customer-managed keys for Knowledge Base data and S3 vectors at rest.
- [ ] CloudTrail enabled for Bedrock API calls.
- [ ] No PII in prompt text — use Comprehend PII detection or Guardrails PII masking before model invocation.
- [ ] VPC flow logs enabled for network audit trail.

---

## Guardrails and Content Safety

### Amazon Bedrock Guardrails

Guardrails provide multi-layered content safety applied uniformly across ALL models in the Bedrock catalogue — including third-party models.

| Layer | Capability | Configuration |
|-------|-----------|---------------|
| **Content filters** | Hate, insults, sexual, violence, misconduct detection | Severity thresholds (NONE/LOW/MEDIUM/HIGH) per category |
| **Denied topics** | Custom topic definitions that are blocked | Natural language topic descriptions |
| **PII detection** | Detect and mask/redact personally identifiable information | Per-entity-type configuration (name, SSN, email, etc.) |
| **Contextual grounding** | Detect hallucinated claims not supported by retrieved context | Grounding score threshold |
| **Automated Reasoning** (GA 2026) | Formal mathematical verification of compliance claims | Policy rules defined in natural language, verified with 99% accuracy |
| **Prompt attack detection** | Jailbreak and injection defence heuristics | Enable/disable per guardrail |
| **Word/phrase filters** | Exact match blocklists | Custom word lists |

### Automated Reasoning checks (new in 2026)

This is unique to AWS — formal mathematical verification of model outputs against compliance rules:

- Define policies in natural language (e.g., "employees with less than 1 year of service are entitled to 10 days annual leave").
- Guardrails mathematically verifies whether the model's output is consistent with the policy.
- Claims accuracy of 99% — significantly higher than LLM-based grounding checks.
- Ideal for compliance-critical applications: HR policy, insurance claims, regulatory Q&A.

### Implementation best practices

1. **Layer your defences** — Guardrails on Bedrock invocations + WAF on API Gateway + application-level output validation.
2. **Start with MEDIUM thresholds** for content filters — relax only if false positive rate is unacceptable with measured data.
3. **Always enable prompt attack detection** — jailbreak attempts are common in any user-facing LLM application.
4. **Use contextual grounding checks** for RAG applications — catches hallucinated claims that aren't supported by the retrieved context.
5. **Use Automated Reasoning** for compliance-critical domains — it's more reliable than LLM-based verification for policy-adherent responses.
6. **Apply Guardrails to both input and output** — filter harmful inputs before they reach the model, and validate outputs before they reach the user.
7. **Log all blocked requests** — guardrail violations should trigger alerts, not just silent drops. Use CloudWatch Alarms on block metrics.
8. **Version your guardrail configurations** — manage them as IaC alongside your application code.

---

## Cost Management

### Pricing tiers (2026 — four tiers)

| Tier | Description | Relative Cost | Best For |
|------|-------------|---------------|----------|
| **Priority** | Fastest response times, guaranteed performance | ~75% premium over Standard | Mission-critical, latency-sensitive paths |
| **Standard** | Consistent performance for everyday tasks | Baseline (published per-token rates) | General production workloads |
| **Flex** | Best-effort delivery, higher latency acceptable | ~50% discount vs Standard | Background processing, batch jobs |
| **Reserved** | Committed capacity with predictable pricing | Volume-based discount | Sustained high-volume workloads |

### Additional pricing models

| Model | Discount | Use Case |
|-------|----------|----------|
| **Batch Inference** | 50% vs on-demand | Bulk summarisation, evaluation runs, data enrichment |
| **Provisioned Throughput** | 15–40% at sustained volume | Dedicated capacity with guaranteed performance |
| **Prompt Caching** | Up to 90% on cached tokens | Repeated system prompts and context prefixes |
| **Cross-Region Inference** | No premium (capacity routing) | Automatic load distribution for availability |
| **Intelligent Prompt Routing** | Up to 30% savings | Mixed-complexity workloads routed dynamically |

### Cost optimisation strategies (ordered by implementation priority)

1. **Right-size your model** — Nova 2 Lite / Claude Haiku for classification, routing, extraction. Reserve Opus / GPT-5 for complex reasoning. This is the single biggest lever.
2. **Intelligent Prompt Routing** — let Bedrock route between model sizes within a family based on query complexity. Up to 30% savings with minimal quality impact.
3. **Prompt caching** — structure system prompts as a stable prefix (cached) + dynamic suffix (user context). Cached tokens cost up to 90% less.
4. **Batch Inference for non-real-time work** — evaluation runs, bulk document processing, and data enrichment at 50% discount.
5. **Service tier selection** — use Flex tier for background processing where latency doesn't matter.
6. **Cross-region inference** — ensures you're never throttled (which wastes developer time and retry cost).
7. **Provisioned Throughput** — lock in capacity for sustained high-volume workloads with predictable pricing.
8. **Token budget monitoring** — track input/output ratios. High input relative to output suggests over-stuffed context. High output suggests overly open-ended prompts.

### Cost estimation rules of thumb (May 2026, NZD)

| Workload | Monthly Cost Estimate |
|----------|----------------------|
| Low-volume RAG chatbot (1K queries/day, Claude Haiku) | $80–200 |
| Medium-volume RAG chatbot (10K queries/day, Claude Sonnet) | $1,000–2,500 |
| Enterprise multi-agent platform (50K interactions/day) | $5,000–15,000 (Provisioned recommended) |
| Document processing pipeline (10K docs/month) | $200–600 (Textract + embedding) |
| OpenSearch Serverless (vector + keyword, 1M documents) | $500–800/month |
| S3 Vectors (1M vectors) | $50–100/month |
| Knowledge Base managed RAG (including embedding) | $300–700/month at 10K queries/day |

---

## Observability and Monitoring

### Model invocation logging

Bedrock's native logging captures all model invocations:

```
Bedrock API Call
  → Model Invocation Log (CloudWatch Logs + S3)
    ├── Request: model ID, input tokens, prompt content (optional)
    ├── Response: output tokens, response content (optional), latency
    └── Metadata: region, request ID, guardrail actions
```

**Configuration:** Enable at the account level. Choose whether to log prompt/response content (disable for privacy-sensitive workloads, enable for debugging and evaluation).

### OpenTelemetry-native observability

```
Application (Strands SDK / custom code)
  → OpenTelemetry SDK (traces + metrics)
    → AWS Distro for OpenTelemetry (ADOT) Collector
      → CloudWatch (metrics + logs)
      → X-Ray (distributed traces)
```

### What to instrument

| Signal | What to Capture | Tool |
|--------|-----------------|------|
| **Traces** | Full request lifecycle: user query → retrieval → LLM call → tool invocations → response | X-Ray + ADOT |
| **Metrics** | Token usage (input/output/cached), latency (P50/P95/P99), error rate, throttle rate | CloudWatch Metrics |
| **Logs** | Guardrail blocks, tool errors, model version changes, cost per request | CloudWatch Logs |
| **Evaluations** | Groundedness, relevance, citation accuracy — sampled from production | Bedrock Model Evaluation or DeepEval |

### CloudWatch dashboards for AI workloads

Build a purpose-built dashboard with:

- **Token consumption** per model, per customer, per endpoint.
- **Latency decomposition** — LLM time vs. retrieval time vs. tool execution time.
- **Throttle rate** (429s) — indicates capacity issues requiring cross-region inference or provisioned throughput.
- **Guardrail block rate** — security signal; spikes indicate potential attack or misconfigured filters.
- **Cost burn rate** — projected monthly spend based on trailing 24h usage.

### Alerting thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| P95 latency (end-to-end) | >5s | >15s |
| Throttle rate (429s) | >5% of requests | >15% |
| Guardrail blocks / hour | >10 | >50 |
| Token budget utilisation | >70% of daily budget | >90% |
| Error rate (5xx from Bedrock) | >1% | >5% |
| Knowledge Base retrieval failures | >2% | >10% |

---

## Evaluation and Testing

### Two-track evaluation strategy

1. **Offline evaluation** (CI/CD) — run against a curated test dataset before deployment.
2. **Online evaluation** (production sampling) — continuously sample production traffic and evaluate quality.

### Tooling options

| Tool | Strengths | When to Use |
|------|-----------|-------------|
| **Bedrock Model Evaluation** | Native integration, built-in metrics, no extra infra | Default for Bedrock-native teams, model comparison |
| **DeepEval** (open-source) | Richer metric library, CI/CD-native (pytest-style), provider-agnostic | Custom metrics, multi-cloud support, automated CI gates |
| **PromptFoo** (open-source) | Prompt comparison, A/B testing, model comparison | Prompt engineering phase, model selection |
| **Ragas** (open-source) | RAG-specific evaluation (context relevance, faithfulness) | Detailed RAG pipeline analysis |

### Evaluation metrics by use case

| Use Case | Key Metrics |
|----------|-------------|
| RAG Q&A | Groundedness, Answer Relevancy, Faithfulness, Citation Accuracy |
| Document summarisation | Coherence, Fluency, Completeness |
| Classification / routing | Accuracy, Precision, Recall, F1 |
| Agent task completion | Task Success Rate, Tool Call Accuracy, Steps to Completion |
| Content safety | False Positive Rate, False Negative Rate (adversarial testing) |
| Compliance (with Automated Reasoning) | Policy Adherence Rate, Contradiction Rate |

### Best practices

1. **Build your evaluation dataset from day one** — collect 30–50 representative Q&A pairs during requirements gathering. Expand to 200+ for production baselines.
2. **Automate evaluation in CI/CD** — every prompt change, model upgrade, or retrieval config change triggers an evaluation run. Gate deployments on threshold pass.
3. **Use LLM-as-judge for subjective metrics** — Claude Sonnet or GPT-4o as a judge for groundedness and relevance. Pin the judge model version.
4. **Red-team before launch** — test adversarial inputs (prompt injection, jailbreaks, off-topic queries) as a dedicated evaluation pass.
5. **Track metrics over time** — quality regressions are gradual. Dashboard your evaluation scores and alert on downward trends.
6. **Separate retrieval evaluation from generation evaluation** — measure recall@k and precision@k for your retrieval layer independently of the LLM's generation quality.
7. **Use Bedrock Model Evaluation for model comparison** — when selecting between models for a new workload, run the same test set through multiple candidates.

---

## Infrastructure as Code

### CDK as the default

AWS CDK (Cloud Development Kit) is the recommended IaC tool for AWS-only AI workloads in 2026. It has first-class L2 constructs for Bedrock, AgentCore, and supporting services.

| Tool | Bedrock Support | Best For |
|------|----------------|----------|
| **AWS CDK** (TypeScript/Python) | Excellent — L2 constructs for Bedrock, AgentCore, Step Functions | AWS-only shops, developers who want programmatic IaC |
| **Terraform** (HCL) | Good — aws provider covers Bedrock resources | Multi-cloud organisations, existing Terraform teams |
| **CloudFormation** (YAML/JSON) | Full support including AgentCore | Compliance-heavy organisations requiring AWS-direct support |

**Note:** CDKTF (CDK for Terraform) is being deprecated — teams using it should migrate to native CDK or native Terraform.

### CDK project structure for AI deployments

```
infra/
├── bin/
│   └── app.ts                    (entry point)
├── lib/
│   ├── ai-platform-stack.ts      (orchestrator stack)
│   ├── constructs/
│   │   ├── bedrock-models.ts     (model access + guardrails)
│   │   ├── knowledge-base.ts     (RAG pipeline)
│   │   ├── agent.ts              (Bedrock Agent + action groups)
│   │   ├── api-gateway.ts        (AI gateway + WAF)
│   │   ├── compute.ts            (Lambda / ECS)
│   │   ├── networking.ts         (VPC, endpoints, security groups)
│   │   ├── observability.ts      (CloudWatch, X-Ray, alarms)
│   │   └── storage.ts            (S3, DynamoDB, OpenSearch)
│   └── config/
│       ├── dev.ts
│       ├── staging.ts
│       └── prod.ts
├── test/
│   └── ai-platform.test.ts
├── cdk.json
└── package.json
```

### Best practices

1. **Use CDK L2 constructs** where available — they encode AWS best practices (encryption, logging, least-privilege IAM) by default.
2. **Parameterise environment differences** (model IDs, capacity, network mode) through config files — same constructs across dev/staging/prod.
3. **Define Bedrock model access as IaC** — Guardrails, model permissions, and Knowledge Base configurations should be in code, not configured through the console.
4. **Use CDK Aspects** for security compliance — automatically enforce encryption, VPC attachment, and tagging policies across all resources.
5. **Deploy with CDK Pipelines** — self-mutating CI/CD pipeline that deploys infrastructure changes through environments with approval gates.
6. **Store reusable constructs in a private package** — publish to CodeArtifact for reuse across engagements.
7. **Use `cdk diff`** before every deployment — review infrastructure changes like code changes.

### Terraform alternative

For multi-cloud organisations or teams with existing Terraform expertise:

```hcl
resource "aws_bedrock_model_invocation_logging_configuration" "main" {
  logging_config {
    embedding_data_delivery_enabled = true
    text_data_delivery_enabled      = true
    cloudwatch_config {
      log_group_name = aws_cloudwatch_log_group.bedrock.name
    }
    s3_config {
      bucket_name = aws_s3_bucket.bedrock_logs.id
      key_prefix  = "invocation-logs/"
    }
  }
}
```

---

## CI/CD for AI Applications

### Pipeline architecture

```
Code Push → Lint & Type Check → Unit Tests → CDK Synth → Deploy to Dev
  → Integration Tests → Evaluation Run (DeepEval)
    → Gate: metrics pass thresholds?
      → Yes → Deploy to Staging → Smoke Tests → Manual Approval → Deploy to Prod
      → No → Fail pipeline, report metrics
```

### AI-specific CI/CD considerations

1. **Prompt changes are code changes** — store prompts in version control. Treat prompt modifications as PRs requiring evaluation runs.
2. **Evaluation as a deployment gate** — add a DeepEval or PromptFoo step after integration tests. Fail the pipeline if groundedness drops below threshold.
3. **Model version upgrades are infrastructure changes** — model ID changes go through the CDK pipeline with evaluation validation.
4. **Separate data pipelines from application pipelines** — Knowledge Base re-indexing (document ingestion) should be independently deployable and testable.
5. **Canary deployments for model changes** — use API Gateway canary releases or weighted routing to gradually shift traffic to a new model version (10% → 50% → 100%) while monitoring quality.
6. **Guardrail changes need testing** — tightening content filters can increase false positives. Test against your evaluation dataset before deploying guardrail updates.

### CodePipeline / GitHub Actions example (evaluation gate)

```yaml
- name: Run AI Evaluation
  run: |
    python3 -m deepeval test run tests/eval/ \
      --model bedrock/anthropic.claude-sonnet-4-6 \
      --threshold groundedness=0.8 \
      --threshold relevancy=0.75
  env:
    AWS_REGION: ap-southeast-2
    AWS_ROLE_ARN: ${{ secrets.EVAL_ROLE_ARN }}
```

---

## Architecture Patterns by Business Size

### Small business (< 50 users, < 5K queries/day)

**Pattern: Serverless RAG**

```
Amplify Hosting (React frontend)
  → API Gateway (REST)
    → Lambda (Bedrock SDK)
      → Bedrock: Claude Haiku (inference)
      → Knowledge Base + S3 Vectors (retrieval)
      → S3 (document storage)
      → Textract (document processing, on-demand)
```

- **Monthly cost:** $100–300 NZD
- **Networking:** Public endpoints with Cognito authentication.
- **IaC:** Single CDK stack, deployed from GitHub Actions.
- **Evaluation:** Manual + PromptFoo for prompt iteration.
- **Guardrails:** Basic content filter + denied topics.

### Medium business (50–500 users, 5K–50K queries/day)

**Pattern: API Gateway + Lambda with model routing**

```
Amplify Hosting (React frontend)
  → API Gateway (REST + WebSocket for streaming)
    → Lambda (model router + orchestration)
      → Bedrock: Claude Sonnet (complex queries)
      → Bedrock: Claude Haiku (simple queries, via Intelligent Prompt Routing)
      → Knowledge Base + OpenSearch Serverless (hybrid retrieval)
      → DynamoDB (conversation history, metadata)
      → Step Functions (document ingestion pipeline)
        → Textract → Comprehend → Knowledge Base
```

- **Monthly cost:** $1,500–4,000 NZD
- **Networking:** VPC with private endpoints for Bedrock and OpenSearch. API Gateway public with WAF.
- **IaC:** Multi-stack CDK with per-environment config.
- **Evaluation:** DeepEval in CI/CD + production sampling via model invocation logging.
- **Guardrails:** Full suite — content filters, PII detection, grounding checks, prompt attack detection.

### Enterprise (500+ users, 50K+ queries/day, multi-agent)

**Pattern: AgentCore Platform**

```
CloudFront (CDN)
  → API Gateway (REST + WebSocket)
    → Lambda (routing + auth)
      → AgentCore Runtime
        ├── Supervisor Agent (planning + delegation)
        ├── Knowledge Agent (specialist retrieval)
        │   └── Knowledge Base + OpenSearch Serverless (multiple indexes)
        ├── Action Agent (tool execution)
        │   └── AgentCore Gateway → MCP Servers (Lambda)
        └── Compliance Agent (policy verification)
            └── Guardrails (Automated Reasoning)
      → Bedrock (multi-model: Opus, Sonnet, Haiku via routing)
      → DynamoDB (session state, metadata)
      → ElastiCache Redis (semantic cache)
      → AgentCore Memory (cross-session recall)
```

- **Monthly cost:** $8,000–25,000 NZD
- **Networking:** Multi-AZ VPC, private subnets, VPC endpoints for all services, Transit Gateway for multi-account connectivity.
- **IaC:** CDK constructs in CodeArtifact private package, multi-account deployment via CDK Pipelines.
- **Evaluation:** Continuous production evaluation, red-teaming, A/B model testing via API Gateway canary.
- **Guardrails:** Full suite + Automated Reasoning for compliance domains + custom guardrail versions per agent.

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails | Better Approach |
|-------------|-------------|-----------------|
| Calling Bedrock directly from the client | No rate limiting, no observability, credential exposure | Route through API Gateway + Lambda |
| Using one model for all queries | 10–30x cost increase for tasks that don't need frontier reasoning | Intelligent Prompt Routing or explicit model tiering in Lambda |
| Hardcoding model IDs without version | Model behaviour changes break prompts silently | Pin full model version ARNs in config |
| Skipping evaluation ("the demo works") | Quality degrades silently in production | Automated DeepEval in CI/CD + production sampling |
| Single Knowledge Base for all content | Retrieval quality degrades with mixed-domain content | Separate Knowledge Bases per domain, use metadata filters |
| Building custom orchestration when Bedrock Agents suffice | Maintenance burden, reinventing session management | Start managed, graduate to AgentCore only when limits are hit |
| Ignoring prompt caching | Paying full price for repeated system prompts | Structure prompts with stable prefix for automatic caching |
| VPC endpoints skipped for "dev simplicity" | Data traverses public internet, compliance violation | VPC endpoints from day one — CDK makes this trivial |
| Deploying without Guardrails | Regulatory risk, reputational damage, prompt injection vulnerability | Always-on Guardrails with content filters + prompt attack detection |
| Manual infrastructure through the console | Configuration drift, unreproducible environments, no audit trail | CDK from day one, even for PoCs |
| Using ANTHROPIC_API_KEY directly with Bedrock | Unnecessary dependency, breaks IAM model | Use IAM roles — Bedrock authenticates via SigV4 natively |
| Over-provisioning OpenSearch Serverless | Costs spiral for small-to-medium workloads | Use S3 Vectors for simple vector search; OpenSearch only when you need hybrid |

---

## References

- AWS Architecture Blog — Building an AI Gateway to Amazon Bedrock with Amazon API Gateway: https://aws.amazon.com/blogs/architecture/building-an-ai-gateway-to-amazon-bedrock-with-amazon-api-gateway/
- AWS — Amazon Bedrock AgentCore Overview: https://cloudvisor.co/amazon-bedrock-agentcore/
- AWS — Amazon Bedrock AgentCore Stateful MCP Support: https://aws.amazon.com/about-aws/whats-new/2026/03/amazon-bedrock-agentcore-runtime-stateful-mcp/
- AWS — Amazon Bedrock Cross-Region Inference: https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
- AWS — Amazon Bedrock Guardrails: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html
- AWS — Amazon Bedrock Pricing and Service Tiers: https://aws.amazon.com/bedrock/pricing/
- AWS — Amazon Bedrock Service Tiers: https://aws.amazon.com/bedrock/service-tiers/
- AWS — Amazon S3 Vectors (GA January 2026): https://aws.amazon.com/blogs/aws/introducing-amazon-s3-vectors-first-cloud-storage-with-native-vector-support-at-scale/
- AWS — Build AI Agents with AgentCore using CloudFormation: https://aws.amazon.com/blogs/machine-learning/build-ai-agents-with-amazon-bedrock-agentcore-using-aws-cloudformation/
- AWS — Model Invocation Logging: https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html
- AWS — OpenAI Models on Bedrock (April 2026): https://aws.amazon.com/about-aws/whats-new/2026/04/bedrock-openai-models-codex-managed-agents/
- AWS — re:Invent 2025 AI Announcements: https://www.aboutamazon.com/news/aws/aws-re-invent-2025-ai-news-updates
- AWS — Trainium3 UltraServer: https://www.aboutamazon.com/news/aws/trainium-3-ultraserver-faster-ai-training-lower-cost
- AWS Samples — AI Gateway for Amazon Bedrock: https://github.com/aws-samples/sample-ai-gateway-for-amazon-bedrock
- AWS Samples — CloudWatch Generative AI Observability: https://github.com/aws-samples/sample-amazon-cloudwatch-generative-ai-observability
- AWS Solutions Library — Multi-Agent Orchestration using Bedrock AgentCore: https://github.com/aws-solutions-library-samples/guidance-for-multi-agent-orchestration-using-bedrock-agentcore-on-aws
- CDK vs Terraform Practical Comparison (2026): https://andrewodendaal.com/aws-cdk-vs-terraform-practical-comparison-2026/
- Cevo — AWS Vector Store for RAG Beyond OpenSearch: https://cevo.com.au/post/aws-vector-store-for-rag-beyond-opensearch/
- DeepEval — LLM Evaluation Framework: https://deepeval.com/
- FutureAGI — AWS Bedrock: The Future of AI Development on AWS: https://futureagi.com/blog/aws-bedrock-the-future-of-ai-development-on-aws/
- InfoQ — Amazon S3 Vectors GA: https://www.infoq.com/news/2026/01/aws-s3-vectors-ga/
- InterWorks — Securing Amazon Bedrock (2026): https://interworks.com/blog/2026/03/06/securing-amazon-bedrock-what-enterprises-need-to-get-right/
- K21 Academy — AWS Generative AI Cost Optimization: https://k21academy.com/aws-aiml/aws-generative-ai-cost-optimization/
- KMS ITC — AI Gateway as Enterprise Control Plane (2026): https://www.kmsitc.net/insights/ai-gateway-bedrock-enterprise-control-plane-2026
- OpenAI on AWS announcement: https://openai.com/index/openai-on-aws/
- OpenObserve — Monitoring AWS Bedrock: https://openobserve.ai/blog/monitoring-aws-bedrock/
- Strands Agents SDK — Deploy to Bedrock AgentCore: https://strandsagents.com/docs/user-guide/deploy/deploy_to_bedrock_agentcore/
- Nerova — Stateful MCP on Amazon Bedrock AgentCore: https://nerova.ai/guides/what-is-stateful-mcp-amazon-bedrock-agentcore-2026
