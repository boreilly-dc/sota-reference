# Modern AI Observability

| Field | Value |
|-------|-------|
| Created | 2026-05-26 |
| Last Updated | 2026-05-26 |
| Version | 1.0 |

---

- [Why AI Observability Is Different](#why-ai-observability-is-different)
- [Core Concepts](#core-concepts)
- [The OpenTelemetry GenAI Standard](#the-opentelemetry-genai-standard)
- [Observability for Agentic Systems](#observability-for-agentic-systems)
- [Multi-Agent and Swarm Observability](#multi-agent-and-swarm-observability)
- [Open-Source vs Proprietary Model Observability](#open-source-vs-proprietary-model-observability)
- [Evaluation in Production](#evaluation-in-production)
- [Privacy and Content Capture](#privacy-and-content-capture)
- [Implementation: Open-Source Tools](#implementation-open-source-tools)
- [Implementation: Hyperscaler Services](#implementation-hyperscaler-services)
- [Architecture Patterns](#architecture-patterns)
- [References](#references)

## Why AI Observability Is Different

Traditional Application Performance Monitoring (APM) tracks four signals: latency, traffic, errors, and saturation. These remain necessary for AI systems but are insufficient. AI observability extends traditional APM in three fundamental ways:

1. **Output quality is non-binary.** A REST API either returns correct data or throws an error. An LLM returns text that appears valid even when factually wrong — the system is "up" by every infrastructure metric while silently delivering poor outputs. An estimated 67% of LLM failures are silent: the model returns confident, fluent, completely wrong answers while APM shows green.

2. **Behaviour degrades continuously, not discretely.** Traditional software failures are step functions. LLM quality degradation is a slope: hallucination rates creep upward as prompt distributions drift, model checkpoints change upstream, or retrieval quality decays. Without baseline comparison, degradation is invisible until significant.

3. **Cost is a first-class operational metric.** Running LLMs at scale means paying per token, per model, per provider. A feature costing $0.002 per request at 1,000 daily users becomes a $72K/month line item at 100,000 users. Token cost monitoring belongs alongside latency and error rates.

| APM Category | Traditional Signal | LLM Equivalent |
|---|---|---|
| Latency | HTTP response time | TTFT + TPOT + end-to-end |
| Traffic | Requests per second | RPS + Goodput (RPS meeting SLOs) |
| Errors | 4xx/5xx rate | Refusal rate + timeout rate + provider errors |
| Saturation | CPU/memory utilisation | Queue depth + context window utilisation |
| Quality | *(not applicable)* | Hallucination rate + faithfulness + relevance |
| Cost | Infrastructure spend | Token cost per trace, per feature, per user |
| Drift | *(not applicable)* | Input drift + output drift + embedding centroid drift |

## Core Concepts

AI observability rests on three pillars inherited from traditional observability but with LLM-specific semantics:

### Metrics

Aggregated numerical measurements collected over time. LLM-specific metrics extend beyond latency and error rates:

- **Time to First Token (TTFT)** — interval between request and first streamed token. Targets: chat ≤300ms (p95), voice ≤150ms (p95), batch ≤2,000ms.
- **Time Per Output Token (TPOT)** — interval between tokens in streaming. Above 100ms produces a "stuttering" experience. Voice targets ≤50ms.
- **Goodput** — requests per second meeting *all* SLOs simultaneously (TTFT, TPOT, end-to-end). A system at 500 RPS with 30% exceeding TTFT SLO has goodput of only 350 RPS.
- **Token consumption** — input/output tokens per request, per feature, per user. Enables cost attribution.
- **Quality scores** — faithfulness, relevance, groundedness attached to production traces via evaluation pipelines.

### Traces

The execution path of a single request through a distributed system. In LLM applications, a trace spans the full pipeline:

```
TRACE: user_query_id=abc123
├── SPAN: retrieve_context (45ms)
│   ├── embedding_model: "text-embedding-3-small"
│   └── chunks_retrieved: 5
├── SPAN: construct_prompt (3ms)
│   └── template_version: "v2.4"
├── SPAN: llm_inference (387ms)
│   ├── gen_ai.provider.name: "openai"
│   ├── gen_ai.request.model: "gpt-4o"
│   ├── prompt_tokens: 1,243
│   ├── completion_tokens: 218
│   └── estimated_cost_usd: 0.00412
├── SPAN: guardrail_check (12ms)
│   └── toxicity_score: 0.02
└── SPAN: eval_async (background)
    ├── faithfulness_score: 0.91
    └── hallucination_risk: "low"
```

Every span carries enough context to reconstruct exactly what happened for any given request — critical for compliance audits and debugging quality regressions.

### Logs

Raw structured records of prompts, responses, and tool calls. In LLM systems, logs serve as ground truth for incident investigation and provide the data substrate for offline evaluation. The challenge is volume: a single agent interaction may produce dozens of LLM calls, each with kilobytes of prompt/completion text.

### Beyond the Three Pillars

The 2026 LLM observability stack adds four primitives:

1. **Span-attached evaluation scores** — quality judgements linked directly to trace spans
2. **Prompt and model version tracking** — enabling before/after comparison on version changes
3. **Token and cost telemetry** — per-span cost attribution at the trace level
4. **Retrieval and tool-call structure** — semantic classification of RAG retrieval and tool invocations

## The OpenTelemetry GenAI Standard

OpenTelemetry (OTel) is converging as the vendor-neutral standard for AI observability. The **GenAI Special Interest Group (GenAI SIG)**, formed in April 2024 under the OTel Semantic Conventions SIG, has expanded from basic LLM client call tracing to cover agent orchestration, MCP tool calling, content capture, and quality evaluation.

As of May 2026, the GenAI semantic conventions are at **v1.41** and remain in **Development status** with no public timeline for stabilisation. Despite this, the core concepts have settled and building on the spec is a reasonable bet.

### Six Layers of the Spec

| Layer | Purpose | Key Span/Attribute Types |
|---|---|---|
| 1. Client Spans | Standardising model calls | `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens` |
| 2. Agent & Workflow Spans | Beyond microservices | `invoke_agent`, `invoke_workflow`, `execute_tool`, `create_agent` |
| 3. MCP Conventions | Fixing broken traces | `mcp.method.name`, `mcp.session.id`, W3C Trace Context propagation |
| 4. Events & Content Capture | Privacy-aware logging | `gen_ai.client.inference.operation.details`, three recording modes |
| 5. Metrics | Operational histograms | `gen_ai.client.operation.duration`, `gen_ai.client.token.usage` |
| 6. Provider-Specific | Vendor extensions | OpenAI cache/reasoning tokens, Anthropic billing, AWS Bedrock |

### Spec Evolution

| Version | Key GenAI Changes |
|---|---|
| v1.37 | Chat history revamp; `gen_ai.system` replaced by `gen_ai.provider.name` |
| v1.38 | Evaluation event; tool definitions and call details; embeddings dimension |
| v1.39 | MCP semantic conventions |
| v1.40 | Retrieval span; cache token attributes; Anthropic input token calculation |
| v1.41 | `execute_tool` span naming; reasoning tokens; `invoke_workflow`; streaming metrics |

### Instrumentation in Practice

With the OpenAI Python SDK, instrumentation takes one line:

```python
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
OpenAIInstrumentor().instrument()
```

The **opentelemetry-instrumentation-openai-v2** package is the most mature GenAI instrumentation. Community libraries like **OpenLLMetry** cover Anthropic, Cohere, AWS Bedrock, and 28+ other providers with 31 Python packages and ~3.85M downloads/month.

## Observability for Agentic Systems

Agents introduce observability challenges that single LLM calls do not:

### The Agent Black Box Problem

Traditional traces show infrastructure-level operations. An agent's *reasoning process* — why it chose to call a tool, how it interpreted results, when it decided to retry — is invisible unless explicitly instrumented. The OTel GenAI spec addresses this with dedicated span types:

```
invoke_agent research-assistant (INTERNAL)
├── chat gpt-4o (CLIENT)          ← Model decides to search
├── execute_tool web_search (INTERNAL)  ← Tool executed
├── chat gpt-4o (CLIENT)          ← Reasoning continues
├── execute_tool summarize (INTERNAL)   ← Summarisation
└── chat gpt-4o (CLIENT)          ← Final answer
```

### Event Amplification

In multi-step agent workflows, a single user request triggers multiple agent actions. When an observability tool records detailed data at every step, total telemetry events grow rapidly with workflow depth. Benchmarking shows:

- **Langfuse** incurs ~15% overhead from detailed prompt/output/token tracing
- **LangSmith** has lighter trace artefacts and remains close to baseline
- **AgentOps** generates higher overhead in complex pipelines due to comprehensive event capture

Tighter integration between the observability tool and the agent framework reduces overhead by eliminating translation steps.

### Tool Call Auditing

For production agents that execute tools (database queries, API calls, code execution), observability must capture:

- What tool was called and why (the preceding reasoning)
- Tool input arguments and output results
- Latency and error status of tool execution
- Whether the agent correctly interpreted tool results

The OTel v1.41 spec requires tool names in span names (`execute_tool {gen_ai.tool.name}`) and records `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` when privacy policies permit.

## Multi-Agent and Swarm Observability

Multi-agent systems where multiple agents collaborate introduce the hardest observability challenges: tracing causality across agent boundaries.

### The Broken Trace Problem

In single-agent systems, one trace ID covers the entire interaction. In multi-agent systems:

- Agent A decides to delegate to Agent B
- Agent B calls an MCP server
- The MCP server invokes a tool that triggers Agent C

Without explicit context propagation, these produce disconnected traces — Trace A, Trace B, Trace C — with no causal link between them.

### Solutions Emerging in 2026

**W3C Trace Context propagation** — OTel's MCP semantic conventions (v1.39+) propagate trace context across the JSON-RPC boundary, linking agent-side and server-side spans into a unified trace.

**OTel Baggage** — Cisco's AGNTCY project uses OTel baggage to carry request-specific metadata (user IDs, session IDs) across agent boundaries in A2A (Agent-to-Agent) communications, maintaining session affinity across distributed invocations.

**Causal Trace IDs** — Microsoft's Agent Governance Toolkit implements hierarchical trace IDs encoding the full spawn tree (parent, child, sibling relationships), enabling root-cause analysis of multi-agent failures by traversing the tree backwards.

**AG2 (formerly AutoGen)** provides four instrumentation functions for complete multi-agent OTel tracing:

- `instrument_agent` — individual agent turns
- `instrument_llm_wrapper` — LLM calls
- `instrument_pattern` — group chat orchestration
- `instrument_a2a_server` — A2A protocol communication

Span types classify work semantically: `conversation`, `agent`, `llm`, `tool`, `code_execution`, `human_input`, `speaker_selection`.

### A Complete Multi-Agent Trace

```
invoke_agent support-router (INTERNAL, trace=t1)
│
├── chat gpt-4o (CLIENT)
│   gen_ai.usage.input_tokens = 1523
│   gen_ai.response.finish_reasons = ["tool_calls"]
│
├── tools/call query-orders (CLIENT)   ← MCP client
│   mcp.session.id = sess-abc
│   gen_ai.tool.name = query-orders
│   │
│   └── tools/call query-orders (SERVER)  ← MCP server
│
└── chat gpt-4o (CLIENT)
    gen_ai.usage.input_tokens = 2841
    gen_ai.usage.cache_read.input_tokens = 1523
    gen_ai.response.finish_reasons = ["stop"]
```

A single `trace_id` links the entire chain from agent decision through MCP execution to final response.

## Open-Source vs Proprietary Model Observability

The observability data available differs significantly between self-hosted open-source models and proprietary API services.

### Proprietary API Models (OpenAI, Anthropic, Google)

Proprietary providers return only response-level metadata:

| Signal | Available | Source |
|---|---|---|
| Token counts (input/output) | Yes | Response body |
| Model name (requested + actual) | Yes | Response body |
| Finish reason | Yes | Response body |
| Latency (TTFT, E2E) | Client-measured only | Instrumentation |
| Cost | Calculated from token × price | Published pricing |
| Cache tokens (OpenAI) | Yes (since 2025) | Response body |
| Reasoning tokens (o-series) | Yes | Response body |
| GPU utilisation | No | Not exposed |
| Batch scheduling / queue depth | No | Not exposed |
| KV cache state | No | Not exposed |
| Model version changes | No advance notice | Silent updates |

**Key limitation**: Provider-side model version drift is invisible. Providers silently update model weights, safety filters, or decoding parameters, causing behaviour shifts even when application code is unchanged. The only defence is continuous quality evaluation.

### Self-Hosted Open-Source Models (vLLM, TGI)

Self-hosted inference gives full access to infrastructure-level metrics that proprietary APIs cannot expose:

**vLLM** exposes a comprehensive Prometheus-compatible `/metrics` endpoint:

| Metric Category | Examples |
|---|---|
| Request lifecycle | `e2e_request_latency_seconds`, `time_to_first_token_seconds`, `request_queue_time_seconds` |
| Token generation | `inter_token_latency_seconds`, `request_generation_tokens`, `request_prompt_tokens` |
| KV Cache | `kv_block_idle_before_evict_seconds`, `kv_block_lifetime_seconds`, `kv_block_reuse_gap_seconds` |
| Inference phases | `request_prefill_time_seconds`, `request_decode_time_seconds` |
| Queue state | `num_requests_running`, `num_requests_waiting`, `num_requests_waiting_by_reason` |
| Speculative decoding | `spec_decode_num_accepted_tokens`, `spec_decode_num_draft_tokens` |
| GPU performance | `estimated_flops_per_gpu_total` (via `--enable-mfu-metrics`) |

**Key advantage**: Token-level visibility into the inference pipeline, flat-cost observability regardless of volume, data stays in-environment, and full control over retention and access policies.

### Practical Differences

| Dimension | Proprietary API | Self-Hosted OSS |
|---|---|---|
| Latency measurement | Client-side only | Server-side (TTFT, TPOT, prefill, decode) |
| Cost tracking | Token count × published price | Infrastructure cost (GPU-hours) |
| Cost model | Per-token, scales linearly | Fixed infrastructure, flat per-query |
| Drift detection | Must instrument outputs | Can monitor input distributions + model weights |
| Observability cost | Typically per-span/per-token fees | Flat (OTel + Prometheus is free) |
| Privacy | Data leaves your network | Data stays in-environment |
| Depth | Application-level only | Full stack (GPU → kernel → application) |

### The Hybrid Reality

Most production deployments use both proprietary and self-hosted models. The common pattern is a **hybrid observability stack**: commercial platforms (Langfuse Cloud, Helicone) handle proprietary API traces via proxy, while an OSS stack (OTel + Prometheus + Grafana) handles self-hosted inference metrics. The crossover point for going fully OSS is approximately $5K/month in LLM API spend.

## Evaluation in Production

Infrastructure metrics cannot detect hallucinations. Quality evaluation is a distinct observability concern requiring automated assessment of production outputs.

### The 3-Layer Eval Harness

A production eval strategy operates at three levels:

| Layer | What | Cadence | Cost | Coverage |
|---|---|---|---|---|
| 1. Prompt unit tests | Deterministic assertions on frozen fixtures | CI gate | $1–5/mo | Catches 80% of bugs |
| 2. Property tests | Schema/format/safety checks on sampled traffic | Hourly | $5–20/mo | Catches schema drift |
| 3. Drift detection | LLM-as-judge on nightly production traces | Nightly | $10–40/mo | Catches quality decay |

**Grader preference order**: deterministic assertions first, LLM-as-judge second, human-judge last. LLM-as-judge should only be used where deterministic checks cannot work.

### LLM-as-Judge Patterns

Over 50% of production agent teams now use judge LLMs at runtime. Six patterns exist:

1. **Offline eval harnesses** — batch evaluation of collected traces
2. **Online runtime verifiers** — synchronous blocking before delivery (used by Amazon Prime Video, Microsoft Bing)
3. **Self-consistency loops** — multiple generations compared for agreement
4. **Reflexion/reflection** — agent critiques own output with external grounding
5. **Constitutional AI / RLAIF** — values-based filtering
6. **Inference-time reward models** — scoring during generation

**Cost-efficient judges**: Small distilled models (Galileo Luna-2 3B-8B, Prometheus 2 7B, Patronus Lynx 8B) achieve 97% cost reduction at 0.88–0.95 accuracy versus GPT-4-based evaluation, enabling inline verification at scale.

**Judge placement**: Production teams should instrument judge checks at three boundaries: before user-facing output, before irreversible tool execution, and on writes to persistent memory.

### Drift Taxonomy

LLM drift manifests as subtle performance erosion rather than catastrophic failure:

- **Data drift** — shift in input distributions (users asking different questions)
- **Concept drift** — changed relationship between inputs and desired outputs
- **Provider/model version drift** — silent upstream model updates
- **Retrieval drift** — document corpus or embedding quality degradation
- **Prompt drift** — template edits inadvertently altering behaviour
- **Guardrail drift** — safety filter changes affecting response rates

## Privacy and Content Capture

Prompt and completion content is simultaneously the most valuable debugging data and the most sensitive. The OTel spec defines three content recording modes:

1. **Not recorded** (default) — content capture is off entirely. Gated behind `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`.

2. **On span attributes** — `gen_ai.input.messages` and `gen_ai.output.messages` stored as span attributes. Convenient for debugging but size-limited and visible to anyone with trace access.

3. **External storage with span reference** — full content stored in external storage (S3, dedicated database) with independent IAM and retention policies. The span holds only a reference URL. **Recommended for production** with significant telemetry volume or sensitive data.

### Compliance Considerations

- **GDPR/Privacy**: Prompt content may contain PII. PII detection and redaction should run as a pre-processing step before content enters the observability pipeline.
- **Retention**: Different jurisdictions require different retention periods. Decoupling content storage from trace storage enables independent lifecycle management.
- **Access control**: Not everyone who can view traces should see full prompts. External storage allows separate IAM policies.

## Implementation: Open-Source Tools

The open-source LLM observability ecosystem has matured rapidly. The market is consolidating — three major acquisitions occurred within 14 months (2025–2026).

### Platform Comparison

| Tool | Licence | Focus | Self-Hosted | OTel-Native | Differentiator |
|---|---|---|---|---|---|
| **Langfuse** | MIT | LLM tracing + eval | Yes (PG+CH+Redis) | Partial | Market leader, 26.6K stars, prompt management |
| **Arize Phoenix** | ELv2 | LLM + ML observability | Yes (single container) | Yes (OpenInference) | 30+ auto-instrumentations, drift detection |
| **OpenLLMetry** | Apache 2.0 | Instrumentation library | N/A (library) | Yes | 31 packages, routes to any backend |
| **OpenLIT** | Apache 2.0 | LLM + GPU monitoring | Yes (ClickHouse) | Yes | GPU metrics (NVIDIA+AMD), eBPF k8s controller |
| **AgentOps** | Proprietary | Agent debugging | No (SaaS) | Yes (foundation) | Time-travel debugging, session replay |
| **Lunary** | Open-source | Chatbot observability | Yes | Partial | Self-hostable, prompt + feedback management |
| **LangSmith** | Proprietary | LangChain ecosystem | Enterprise only | No | Zero-friction LangChain integration |

### Integration Approaches

Three architectural patterns exist for connecting applications to observability backends:

1. **Proxy-based** — route LLM API calls through a proxy gateway that captures telemetry transparently (Helicone pattern). One-line integration but adds a network hop.

2. **SDK-based** — instrument code with an SDK that wraps LLM client calls (Langfuse, LangSmith, AgentOps). Tighter control but requires code changes.

3. **OTel-native** — use OpenTelemetry auto-instrumentation that emits standard spans to any compatible backend (Phoenix, OpenLIT, OpenLLMetry). Most portable but requires OTel infrastructure.

### The Full OSS Stack

For teams running self-hosted models who want zero vendor dependency:

```
OpenTelemetry Collector
    ├── Metrics → Prometheus → Grafana dashboards
    ├── Traces → Grafana Tempo (or Jaeger)
    └── Logs → Grafana Loki

vLLM /metrics → Prometheus scrape
Application → OTel SDK → Collector
```

The **openinference-instrumentation** library (from Arize) provides automatic OTel span wrapping for OpenAI, Anthropic, and vLLM, capturing model identification, token counts, latency breakdown, and cost signals.

## Implementation: Hyperscaler Services

### AWS

**Amazon Bedrock** provides observability through:

- **Model invocation logging** — captures full request/response data, token counts, and metadata to CloudWatch Logs and/or S3. Disabled by default. Bodies up to 100KB inline; larger payloads stored separately in S3.
- **CloudWatch metrics** — `Invocations`, `InvocationLatency`, `InputTokenCount`, `OutputTokenCount`, `TimeToFirstToken` (added March 2026), `EstimatedTPMQuotaUsage` (added March 2026), `CacheReadInputTokens`, `CacheWriteInputTokens`, plus error/throttle counters.
- **Separate metrics for Agents, Guardrails, and Knowledge Bases** — each has dedicated CloudWatch dimensions.
- **EventBridge** — monitors job state changes.
- **CloudTrail** — audits all API calls.

AWS does not natively support OTel GenAI semantic conventions within Bedrock. OTel is supported broadly for application tracing (X-Ray/ADOT), but Bedrock-specific instrumentation uses proprietary CloudWatch conventions.

### Azure

**Microsoft Foundry** (formerly Azure AI Foundry) is the most advanced hyperscaler in natively supporting OpenTelemetry GenAI semantic conventions:

- **OTel-native tracing** — uses OpenTelemetry semantic conventions as a first-class tracing mechanism. GA for prompt agents (March 2026+), preview for workflow/hosted/custom agents.
- **Server-side auto-tracing** — no code changes required for prompt agents.
- **Client-side SDK** — `azure-ai-projects` + `opentelemetry-sdk` + `azure-core-tracing-opentelemetry`.
- **Application Insights** — traces stored with 90-day retention; conversation-level views showing dialogue history, token usage, run steps, and tool calls.
- **Local development** — Foundry Toolkit in VS Code with OTLP-compatible collector.
- **Multi-agent conventions** — Microsoft (with Cisco) has proposed new OTel spans specifically for multi-agent observability, integrated into Semantic Kernel and LangChain/LangGraph packages.

Azure supports OpenAI, Anthropic, and LangChain frameworks through OpenTelemetry instrumentation.

### GCP

**Gemini Enterprise Agent Platform** (successor to Vertex AI):

- **Observability tab** in Agent Registry with dashboards for operational health, performance, and infrastructure utilisation.
- **Metrics, traces, and logs** — standard three-pillar observability.
- **Vertex AI Model Monitoring** — centralised management for continuous production monitoring, including data drift detection for deployed endpoints.
- **Prebuilt model observability dashboard** — covers Gemini models and partner models with managed endpoints.

GCP's AI observability is integrated into its existing Cloud Operations suite (Cloud Monitoring, Cloud Trace, Cloud Logging) but details on native OTel GenAI semantic convention support are less documented than Azure's.

### Hyperscaler Comparison

| Capability | AWS | Azure | GCP |
|---|---|---|---|
| Native OTel GenAI conventions | No | Yes (GA for prompt agents) | Unclear |
| Model invocation logging | CloudWatch/S3 | Application Insights | Cloud Logging |
| TTFT metric | Yes (March 2026) | Yes (via OTel spans) | Yes (dashboard) |
| Token cost tracking | Via token count metrics | Via OTel span attributes | Via dashboard |
| Agent-specific metrics | Yes (Bedrock Agents) | Yes (multi-agent spans) | Yes (Agent Registry) |
| Auto-instrumentation | No | Yes (server-side) | Partial |
| Multi-agent tracing | Basic | Advanced (OTel + Semantic Kernel) | Basic |

## Architecture Patterns

### Pattern 1: Proprietary API with Commercial Observability

Best for: teams using OpenAI/Anthropic APIs who want minimal setup.

```
Application → Langfuse SDK → Langfuse Cloud
                                  ├── Traces
                                  ├── Evaluations
                                  └── Cost dashboards
```

### Pattern 2: Self-Hosted Inference with OSS Stack

Best for: teams running vLLM/TGI who need full control and flat costs.

```
vLLM /metrics ──────────────────→ Prometheus → Grafana
Application → OTel SDK → Collector → Tempo (traces)
                                   → Loki (logs)
```

### Pattern 3: Hybrid (Most Common in Production)

Best for: teams using both proprietary APIs and self-hosted models.

```
Proprietary API calls → Langfuse/Phoenix (eval + traces)
Self-hosted inference → Prometheus/Grafana (infra metrics)
Both → OTel Collector → unified trace backend
```

### Pattern 4: Multi-Agent with OTel

Best for: distributed agent systems with inter-agent communication.

```
Agent A → OTel SDK ─┐
Agent B → OTel SDK ─┼→ OTel Collector → Tempo/Jaeger
MCP Server → OTel SDK ─┘       (W3C Trace Context links all)
```

## References

1. [OpenTelemetry Semantic Conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Official OTel spec (Development status, v1.41)
2. [OpenTelemetry GenAI Semantic Conventions Repository](https://github.com/open-telemetry/semantic-conventions-genai) — GitHub repository for GenAI-specific conventions
3. [How OpenTelemetry Traces LLM Calls, Agent Reasoning, and MCP Tools](https://greptime.com/blogs/2026-05-09-opentelemetry-genai-semantic-conventions) — Greptime, May 2026
4. [vLLM Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/) — Official vLLM documentation
5. [Building Production-Ready Observability for vLLM](https://medium.com/ibm-data-ai/building-production-ready-observability-for-vllm-a2f4924d3949) — IBM Data Science in Practice, June 2025
6. [Amazon Bedrock Model Invocation Logging](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html) — AWS official documentation
7. [Amazon Bedrock Monitoring](https://docs.aws.amazon.com/bedrock/latest/userguide/monitoring.html) — AWS CloudWatch metrics documentation
8. [Set Up Tracing for AI Agents in Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry/observability/how-to/trace-agent-setup) — Microsoft Learn, May 2026
9. [Azure AI Foundry: Advancing OpenTelemetry for Multi-Agent Observability](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/azure-ai-foundry-advancing-opentelemetry-and-delivering-unified-multi-agent-obse/4456039) — Microsoft Tech Community
10. [Gemini Enterprise Agent Platform Observability](https://docs.cloud.google.com/gemini-enterprise-agent-platform/optimize/observability/overview) — Google Cloud documentation
11. [AI Observability in Multi-Agent Systems using OpenTelemetry](https://outshift.cisco.com/blog/ai-ml/ai-observability-multi-agent-systems-opentelemetry) — Cisco Outshift (AGNTCY)
12. [AG2 OpenTelemetry Tracing for Multi-Agent Systems](https://docs.ag2.ai/latest/docs/blog/2026/02/08/AG2-OpenTelemetry-Tracing/) — AG2 documentation
13. [Microsoft Agent Governance Toolkit: Observability & Tracing](https://microsoft.github.io/agent-governance-toolkit/tutorials/13-observability-and-tracing/) — Microsoft open-source toolkit
14. [AI Monitoring in Production 2026](https://valuestreamai.com/blog/ai-monitoring-in-production-guide-2026) — ValueStreamAI
15. [LLM Observability: The ML Engineer's Practical Guide](https://mlopslab.org/llm-observability/) — MLOpsLab, April 2026
16. [Best LLM Observability Platforms in 2026](https://chatforest.com/guides/best-llm-observability-platforms-2026/) — ChatForest comparison
17. [9 AI Observability Platforms Compared](https://softcery.com/lab/top-8-observability-platforms-for-ai-agents-in-2025) — Softcery, April 2026
18. [AgentOps](https://github.com/AgentOps-AI/agentops) — GitHub repository
19. [LLM-as-Judge in Production: Agent Reasoning Verification](https://zylos.ai/en/research/2026-04-10-llm-as-judge-production-agent-verification-2026) — Zylos AI, April 2026
20. [LLM Evaluation in Production: The 3-Layer Eval Harness](https://autoolize.com/blog/eval-suites-catch-drift/) — Autoolize
21. [Open Source LLM Monitoring Stack in 2026](https://stackpulsar.com/blog/open-source-llm-monitoring-stack/) — StackPulsar
