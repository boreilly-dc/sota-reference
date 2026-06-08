# Agentic Harnesses — Frameworks for Building Agentic Systems

| Field | Value |
|-------|-------|
| Created | 2026-06-08 |
| Last Updated | 2026-06-08 |
| Version | 1.0 |

---

- [1. What This Article Covers](#1-what-this-article-covers)
- [2. The 2026 Landscape at a Glance](#2-the-2026-landscape-at-a-glance)
- [3. Language Coverage Matrix](#3-language-coverage-matrix)
- [4. Gold-Standard Picks (TL;DR)](#4-gold-standard-picks-tldr)
- [5. Python Frameworks](#5-python-frameworks)
- [6. JavaScript / TypeScript Frameworks](#6-javascript--typescript-frameworks)
- [7. Go (Golang) Frameworks](#7-go-golang-frameworks)
- [8. C# / .NET Frameworks](#8-c--net-frameworks)
- [9. Vendor Agent SDKs](#9-vendor-agent-sdks)
- [10. Coding-Agent Harnesses](#10-coding-agent-harnesses)
- [11. Durable Execution & Workflow Engines](#11-durable-execution--workflow-engines)
- [12. Hyperscaler Managed Agent Services](#12-hyperscaler-managed-agent-services)
- [13. Interoperability Protocols (MCP, A2A, AGNTCY)](#13-interoperability-protocols-mcp-a2a-agntcy)
- [14. Beyond the Four Languages: the JVM](#14-beyond-the-four-languages-the-jvm)
- [15. Choosing a Framework — Decision Guide](#15-choosing-a-framework--decision-guide)
- [16. Anti-Patterns and When NOT to Use a Framework](#16-anti-patterns-and-when-not-to-use-a-framework)
- [17. Confidence, Caveats and Limitations](#17-confidence-caveats-and-limitations)
- [References](#references)

An **agentic harness** is the software layer that drives a large language model around the agent loop — calling tools, managing context and memory, coordinating sub-agents, handling errors, persisting state, and exposing the whole thing as something you can deploy and observe. The model supplies reasoning; the harness supplies *control*. This article surveys the frameworks available for building such harnesses as of June 2026, covering both proprietary vendor SDKs and open-source libraries, with explicit attention to four language ecosystems — **Python, JavaScript/TypeScript, Go, and C#/.NET** — plus the durable-execution engines, hyperscaler managed services, and interoperability protocols that production agents increasingly depend on.

The guiding principle throughout (per the SOTA Reference conventions) is **open-source first**: every category presents an open-source option, and managed-service mentions are limited to AWS, Azure, GCP, IBM, and Oracle.

---

## 1. What This Article Covers

The agentic-framework space has stratified into roughly six layers, and "which framework should I use" almost always means "which layers do I need":

1. **Agent application frameworks** — the agent loop, tools, handoffs, memory, multi-agent orchestration (LangGraph, CrewAI, Mastra, Microsoft Agent Framework, ADK …).
2. **Vendor agent SDKs** — opinionated, provider-aligned loops (OpenAI Agents SDK, Anthropic Claude Agent SDK, Google ADK, Mistral Agents API).
3. **Coding-agent harnesses** — specialised for editing repositories and running code (Claude Code, Codex CLI, OpenHands, Aider, SWE-agent).
4. **Durable-execution engines** — make long-running agents crash-proof and resumable (Temporal, Restate, Inngest, DBOS, Dapr).
5. **Hyperscaler managed runtimes** — host and operate agents (AWS Bedrock AgentCore, Microsoft Foundry Agent Service, Vertex AI Agent Engine, IBM watsonx Orchestrate, OCI Generative AI Agents).
6. **Interoperability protocols** — how tools and agents talk (MCP, A2A, AGNTCY).

Sections 5–8 are organised by language because that is the hard constraint most teams start from. Sections 9–13 are cross-cutting.

---

## 2. The 2026 Landscape at a Glance

Several structural shifts define the 2026 state of the art:

- **Frameworks have hit 1.0.** The churn-prone experiments of 2024–2025 have largely stabilised. LangGraph, CrewAI, Pydantic AI, Mastra, AWS Strands, Google ADK, and Microsoft Agent Framework all shipped stable major releases by mid-2026.
- **MCP is table stakes.** Native [Model Context Protocol](#13-interoperability-protocols-mcp-a2a-agntcy) support is now a baseline expectation for any serious framework — a tool that does not speak MCP is a tool that cannot reach the 10,000+ public MCP servers.
- **Durable execution moved from "nice to have" to "core reliability layer."** Agents that run for minutes-to-hours, pause for human approval, and call side-effecting tools cannot live in a single in-memory loop. Temporal, Restate, Inngest, DBOS and the framework-native checkpointers are now part of the standard production stack.
- **The big labs converged on the same shape.** OpenAI, Anthropic, Google and Microsoft all ship an "agent SDK" with the same primitives — agent loop, tools, handoffs/sub-agents, sessions, tracing, MCP — differing mainly in language coverage and which model they optimise for.
- **Go and .NET became first-class.** What was a Python-and-TypeScript story in 2024 is now genuinely polyglot: Google ADK Go 1.0, CloudWeGo Eino, and Microsoft Agent Framework (.NET) give the two "enterprise backend" languages credible, vendor-backed options.
- **Multi-agent scepticism matured.** A body of practitioner and research evidence (Anthropic's "Building Effective Agents," the MAST failure-mode taxonomy, production surveys) now actively warns against reaching for multi-agent architectures by default.

---

## 3. Language Coverage Matrix

Every framework below is positioned against the four target languages. "Native" means a first-party SDK in that language; "via SDK" means you build the loop yourself on the official model SDK; "—" means no meaningful support.

| Framework / layer | Python | JS/TS | Go | C#/.NET | Licence | Type |
|---|---|---|---|---|---|---|
| **LangGraph** | Native | Native (lags) | — | — | MIT | OSS orchestration |
| **OpenAI Agents SDK** | Native | Native | — | — | MIT | Vendor SDK |
| **Anthropic Claude Agent SDK** | Native | Native | via Go SDK | via C# SDK | SDK open; API proprietary | Vendor SDK |
| **Google ADK** | Native | Native | **Native (1.0)** | Native (Java/Kotlin) | Apache-2.0 | Vendor-OSS |
| **CrewAI** | Native | — | — | — | MIT | OSS orchestration |
| **Pydantic AI** | Native | — | — | — | MIT | OSS, type-safe |
| **LlamaIndex** | Native | Native (.TS) | — | — | MIT | OSS (RAG-first) |
| **smolagents** | Native | — | — | — | Apache-2.0 | OSS (code-as-action) |
| **Agno** | Native | — | — | — | MPL-2.0 | OSS full-stack |
| **AWS Strands** | Native (1.0) | Native (1.0) | — | — | Apache-2.0 | Vendor-OSS |
| **Mastra** | — | **Native (1.0)** | — | — | Apache-2.0 | OSS orchestration |
| **Vercel AI SDK** | — | Native (v6) | — | — | Apache-2.0 | OSS toolkit |
| **VoltAgent / AgentKit / Cloudflare Agents** | — | Native | — | — | OSS | OSS (TS) |
| **CloudWeGo Eino** | — | — | **Native** | — | Apache-2.0 | OSS orchestration |
| **Genkit** | Native | Native | **Native** | — | Apache-2.0 | Vendor-OSS |
| **LangChainGo** | — | — | **Native** | — | MIT | OSS (community) |
| **Microsoft Agent Framework** | Native | — | — | **Native (1.0)** | MIT | Vendor-OSS |
| **LlmTornado** | — | — | — | **Native** | MIT | OSS (community) |
| **Dapr Agents** | Native (1.0) | via Dapr | early SDK | via Dapr | Apache-2.0 (CNCF) | OSS durable |
| **Temporal** (durability) | Native | Native | **Native** | **Native** | MIT/Apache-2.0 | OSS durable |

The headline: **all four languages are well served**. Python and TypeScript have the deepest choice; Go's story crystallised in 2026 around ADK Go, Eino, Genkit and LangChainGo; and .NET is anchored by the GA Microsoft Agent Framework with LlmTornado as an independent alternative.

---

## 4. Gold-Standard Picks (TL;DR)

If you want a defensible default per language and per job, start here. Rationale is in the per-language sections.

| If you are building in… | Gold-standard default | Strong alternative | Lightweight option |
|---|---|---|---|
| **Python** | LangGraph (complex/stateful) | Pydantic AI (type-safe), Google ADK | OpenAI Agents SDK, smolagents |
| **JavaScript/TypeScript** | Mastra (full product) | Vercel AI SDK (UI/streaming) | OpenAI Agents SDK (TS) |
| **Go** | Google ADK Go 1.0 | CloudWeGo Eino (Go-idiomatic) | LangChainGo, Genkit Go |
| **C#/.NET** | Microsoft Agent Framework | LlmTornado (provider-agnostic) | Anthropic/OpenAI C# SDK + own loop |
| **Any (vendor-aligned)** | The model vendor's own SDK | — | — |
| **Coding agent** | Claude Code / Claude Agent SDK | OpenAI Codex CLI | OpenHands, Aider (OSS) |
| **Durability layer** | Temporal | Restate (lighter), Dapr (k8s) | DBOS (just Postgres) |
| **Managed runtime** | Match your cloud | AWS AgentCore / Azure Foundry / Vertex Agent Engine | — |

Two cross-cutting rules of thumb that hold across every language:

1. **Pick the orchestration layer for the *shape* of your control flow, not for the vendor.** Vendor SDKs give you the fastest path to a working tool-using agent; graph frameworks (LangGraph, Eino, Agent Framework Workflows) earn their weight only when you have branching, loops, checkpointing, and human-in-the-loop.
2. **Decide your durability story separately and early.** Whether durability comes from the framework (LangGraph checkpointers, Dapr) or an external engine (Temporal, Restate) is one of the highest-leverage architectural choices — and the hardest to retrofit.

---

## 5. Python Frameworks

Python remains the centre of gravity: the largest selection, the most mature tooling, and the reference implementation of nearly every protocol. The trade-off is fragmentation — at least a dozen credible frameworks, several overlapping.

### LangGraph — the production default for complex agents

- **Licence**: MIT · **Maintainer**: LangChain Inc · **Languages**: Python, TypeScript
- **Architecture**: directed state-machine graphs (inspired by Google Pregel / Apache Beam) — nodes are steps, edges are control flow, with typed shared state, conditional edges, and built-in checkpointing.

LangGraph is the most widely deployed Python agent framework in production, reported at ~34.5M monthly PyPI downloads and ~31K GitHub stars, and used in production by companies such as Klarna. Its differentiator is **explicit, inspectable control flow plus durable execution as a first-class concern** — checkpointers persist graph state at every superstep, enabling pause/resume, time-travel debugging, fault recovery, and native human-in-the-loop interrupts. The v1.x line added deep-agent templates and distributed-runtime support. It is the right tool when your agent has real branching, loops, and long-running state; it is over-engineered for a simple tool-using loop. The managed LangGraph Platform adds deployment, a task queue, and cron, but does not target serverless environments.

### Pydantic AI — type-safe, production-grade

- **Licence**: MIT · **Maintainer**: Pydantic Inc (Samuel Colvin) · **Languages**: Python
- **Architecture**: FastAPI-shaped, type-safe agents with validated dependencies, typed tool calls, and Pydantic-validated outputs.

Pydantic AI reached **V1 (stable API)** after ~15M downloads, with a V2 in beta. Its V1 headline features are **human-in-the-loop tool approval** and **durable execution via a Temporal integration** — durability designed in rather than bolted on. With 25+ model providers plus MCP and A2A support, it is the strongest pick for teams in regulated or correctness-sensitive domains who want compile-time-ish guarantees around tool I/O.

### Google ADK (Agent Development Kit) — the polyglot newcomer

- **Licence**: Apache-2.0 · **Maintainer**: Google · **Languages**: Python, TypeScript, Go, Java, Kotlin
- **Architecture**: code-first, with explicit workflow agents (Sequential, Parallel, Loop) plus LLM-driven dynamic routing; first-class evaluation framework; A2A protocol support; deploys to the managed Vertex AI Agent Engine.

ADK is Google's serious bet on the agent-framework category and is moving fast — Python hit **1.0 at I/O 2025**, and **2.0 is already on PyPI with breaking changes**. It is model-agnostic (Gemini-optimised but supports OpenAI/Anthropic) and uniquely **spans all four target languages plus the JVM**, making it the natural choice for multi-language organisations and GCP-aligned teams. Its built-in eval framework is a genuine differentiator. The flip side of its velocity is API instability between majors.

### CrewAI — role-based multi-agent, fastest to prototype

- **Licence**: MIT · **Maintainer**: CrewAI Inc · **Languages**: Python
- **Architecture**: a "crew" of role-playing agents (Role / Goal / Backstory / Tools) executing Tasks within a Process.

The most-starred Python framework (~50K stars). CrewAI shipped 1.x and, by v1.12, added agent "skills" and native OpenAI-compatible providers (OpenRouter, DeepSeek, Ollama, vLLM, Cerebras). It deploys straightforward role-decomposition use cases noticeably faster than LangGraph, at the cost of less control over execution. Best for rapid prototyping and clearly decomposable workflows.

### AWS Strands Agents — model-driven, AWS-native (but open)

- **Licence**: Apache-2.0 · **Maintainer**: AWS · **Languages**: Python, TypeScript
- **Architecture**: model-driven — the model decides tool use within an agent loop, rather than the developer hand-coding a graph.

Open-sourced by AWS in 2025, Strands reached **Python 1.0 (May 2026)** and **TypeScript 1.0 (April 2026)** with ~14M PyPI downloads in its first year. It integrates cleanly with Bedrock AgentCore, Bedrock Guardrails, and OpenTelemetry, positioning itself as "AWS's answer to LangGraph" while remaining genuinely open and cloud-portable. The 1.0 release added multi-agent (A2A, sessions) support. Best for AWS-native teams who still want an open SDK.

### The lighter / specialist Python options

- **OpenAI Agents SDK** (MIT, ~26K stars, ~25.7M monthly downloads): minimal primitives — Agents, Tools, Handoffs, Guardrails, Sessions, Tracing. The fastest path to a provider-native agent; the successor to the experimental Swarm. See [§9](#9-vendor-agent-sdks).
- **smolagents** (Hugging Face, Apache-2.0, ~27K stars): the **code-as-action** approach — the agent writes executable Python instead of emitting JSON tool calls. ~1,000 lines of core code; topped the open-source division of the GAIA benchmark. Lightweight and readable, but running model-generated code demands sandboxing.
- **Agno** (formerly Phidata, MPL-2.0, ~40K stars): an opinionated full-stack runtime treating Agents, Teams, Workflows, Storage, Memory and Knowledge as first-class objects; bidirectional MCP. One of the fastest-growing frameworks. Good when you want a single batteries-included Python stack.
- **LlamaIndex Workflows / AgentWorkflow** (MIT, ~49K stars): event-driven async graphs, but fundamentally a **data/RAG framework** (78 vector-store integrations, 100+ LLM providers) with agents layered on. Reach for it when retrieval is the core of the problem.
- **DSPy** (MIT, Stanford → MIT, ~34K stars): **not an orchestration framework** but a *prompt-optimisation/compilation* system — Signatures, Modules, and Optimizers that tune prompts and few-shot demos against metrics. Use it *alongside* LangGraph or Agno. In production at Databricks, Dropbox, Replit.
- **AG2** (formerly AutoGen, Apache-2.0): the community fork after Microsoft folded AutoGen into the Microsoft Agent Framework. Conversational multi-agent with many orchestration patterns, but a ground-up "AG2 Beta" rewrite is underway and it lacks first-party observability/enterprise features — treat as experimental in 2026.

**Python verdict**: LangGraph for complex stateful agents; Pydantic AI when type-safety and durability matter; Google ADK for multi-language/GCP shops; OpenAI Agents SDK or smolagents when you want minimalism.

---

## 6. JavaScript / TypeScript Frameworks

TypeScript is now a first-class agent ecosystem, not a port target — driven by the reality that most production AI *products* (chat UIs, copilots, SaaS features) are built in TS/Node. The 2026 story is a clear migration toward TypeScript-native frameworks over the lagging LangGraph.js port.

### Mastra — the TypeScript-first gold standard

- **Licence**: Apache-2.0 · **Maintainer**: Mastra (ex-Gatsby founders) · **Languages**: TypeScript
- **Architecture**: an integrated "Rails for agents" — agents, workflows (graph-based), memory, evals, RAG, and observability in one toolkit.

Mastra reached **1.0 GA in January 2026**, with ~22K GitHub stars and 300K+ weekly npm downloads. It is the most complete TypeScript-native option: batteries-included, with durable workflows and observability built in rather than bolted on. Practitioner reports describe TypeScript teams migrating off LangGraph.js to Mastra for the developer experience. Best for teams building production agent *products* entirely in TypeScript who want a single coherent framework.

### Vercel AI SDK — the streaming/UI default

- **Licence**: Apache-2.0 · **Maintainer**: Vercel · **Languages**: TypeScript
- **Architecture**: streaming-first primitives (`generateText` / `streamText`), Zod-typed tool schemas, provider abstraction (swap models via one string), and — as of **v6** — agent-loop primitives and tool approval via the v3 Language Model Specification.

The most-depended-on TypeScript AI package (npm `ai` at v6.0.x, ~4,800 dependent projects). Its sweet spot is React/Next.js chat and copilot UIs with model portability. Crucially, it provides **no native durable execution** — pair it with Inngest, Vercel Workflow, or Temporal for crash recovery. It is a toolkit more than an orchestration framework; for complex multi-step control flow, reach for Mastra or LangGraph.js.

### LangGraph.js — the same graph model, in TypeScript

- **Licence**: MIT · **Maintainer**: LangChain Inc.
- The TypeScript port of LangGraph, with the same state-machine + checkpointing model. The consistent caveat across sources is that **it lags the Python version** on features and documentation. Choose it when you specifically want LangGraph's graph/durability model in a TS stack and can live with the lag — otherwise Mastra is the smoother TS-native path.

### The rest of the TypeScript field

- **OpenAI Agents SDK (TS)** (MIT, `@openai/agents`): the official TypeScript port of OpenAI's SDK — provider-agnostic core, handoffs for multi-agent, lightweight. Still pre-1.0 (v0.5.x), so expect some churn. Best when committed to OpenAI platform tools (Realtime, file search, computer use).
- **Cloudflare Agents SDK** (open-source): stateful agents on Cloudflare Workers via **Durable Objects** — each agent gets persistent SQLite (10 GB as of Agents Week 2026), WebSockets, scheduling, MCP, and "Code Mode," hibernating when idle. The gold standard for *edge-deployed, stateful* agents — but platform-locked to Cloudflare.
- **Inngest AgentKit** (pre-1.0): a TS agent library backed by Inngest's durable-execution platform, so crash recovery, retries, and step memoisation come for free. Best for teams already on Inngest.
- **VoltAgent** (open-source, ~5K stars): observability-first, with Memory/RAG/Guardrails/Tools/MCP/Voice/Workflow primitives and a companion VoltOps console. Smaller community — promising but less battle-tested.
- **LlamaIndex.TS** (open-source): the TypeScript context-engineering/RAG framework; supports multi-agent workflows but, like its Python sibling, is data-first rather than orchestration-first.

**TypeScript verdict**: Mastra for full production products; Vercel AI SDK for streaming UIs and model portability; LangGraph.js if you specifically need the graph/durability model; Cloudflare Agents SDK for edge-native stateful agents.

---

## 7. Go (Golang) Frameworks

For two years Go agents meant "call the model SDK and write your own loop." That changed decisively in 2026: Go is now a **first-class** agent language with vendor-backed, 1.0-grade frameworks. The motivation is production-shaped — Go's concurrency model, small static binaries, and absence of a GIL suit high-throughput agent backends better than Python.

### Google ADK Go — the gold standard

- **Licence**: Apache-2.0 · **Maintainer**: Google · **Repo**: `google/adk-go`
- ADK Go reached **1.0 on 31 March 2026** — the first enterprise-grade Go agent framework at stable release. It ships native OpenTelemetry tracing, a plugin system with self-healing retry-and-reflect logic, human-in-the-loop confirmations, YAML-based agent configs, Sequential/Parallel/Loop agents, and the A2A protocol for cross-language (Go/Java/Python) multi-agent communication. For Go teams wanting an officially supported, production-stable framework — especially on GCP — this is the default.

### CloudWeGo Eino — the most Go-idiomatic

- **Licence**: Apache-2.0 · **Maintainer**: CloudWeGo (ByteDance) · **Repo**: `cloudwego/eino`
- Eino is purpose-built for Go conventions from the ground up. It provides typed **Components** (ChatModel, Tool, Retriever, ChatTemplate), graph/workflow **Composition**, and an Agent Development Kit with multi-agent coordination and **interrupt/resume** for human-in-the-loop. It draws lessons from LangChain and Google ADK but feels native to Go. Backed by ByteDance's production use, it is the strongest choice when you want idiomatic Go and high-scale performance; its community is smaller than LangChainGo's.

### LangChainGo — the community workhorse

- **Licence**: MIT · **Maintainer**: Travis Cline (`tmc/langchaingo`)
- The most widely adopted Go LLM framework by community metrics, with 20+ provider integrations and chain/agent abstractions. The recurring critique is that it can feel like a **Python port rather than Go-idiomatic**. Best for maximum provider flexibility and the largest body of Go examples/docs.

### Genkit Go and the SDK route

- **Genkit Go** (Apache-2.0, Google/Firebase, `genkit-ai/genkit-go`): a production AI SDK with a unified API for generation, structured output, streaming, tool calling, and agentic workflows; strong for rapid prototyping with a plugin architecture and first-class Go support.
- **Official model + protocol SDKs**: Anthropic (`anthropics/anthropic-sdk-go`) and OpenAI ship official Go SDKs with tool use, and the **official MCP Go SDK** (`modelcontextprotocol/go-sdk`, v1.4.0+) gives full client/server MCP. For many Go services the pragmatic answer is still "official SDK + a thin loop," now with first-class MCP.
- **SwarmGo** (community): a lightweight Go take on OpenAI's Swarm (Agents + handoffs) — testable and minimal, but individually maintained, not corporate-backed.

For **durable** Go agents, **Temporal** has a first-class Go SDK, and **Dapr Workflow** offers a Go SDK for durable orchestration (the higher-level Dapr Agents framework is Python-first at GA, with a community Go SDK emerging) — see [§11](#11-durable-execution--workflow-engines).

**Go verdict**: ADK Go 1.0 for enterprise/GCP and stability; Eino for idiomatic Go at scale; LangChainGo for provider breadth and community; official SDK + MCP Go SDK when you want minimal dependencies. (One honest caveat: Go's ecosystem, while now credible, is still far smaller than Python's.)

---

## 8. C# / .NET Frameworks

The .NET story in 2026 is dominated by a single decisive event: the **convergence of Semantic Kernel and AutoGen into the Microsoft Agent Framework**.

### Microsoft Agent Framework — the .NET gold standard

- **Licence**: MIT · **Maintainer**: Microsoft · **Languages**: .NET (C#) and Python, with a consistent API across both
- **Packages**: `dotnet add package Microsoft.Agents.AI` (.NET) · `pip install agent-framework` (Python)

Microsoft announced in **October 2025** that Semantic Kernel and AutoGen would both enter maintenance mode (security and bug fixes only) and merge into one framework. The new framework reached **Release Candidate on 20 February 2026** and **GA on 3 April 2026**. It takes AutoGen's multi-agent abstractions, layers them on Semantic Kernel's enterprise foundation (middleware, filters, telemetry), and adds a new graph-based **Workflow** engine with conditional routing, parallelism, checkpointing, streaming, and human-in-the-loop.

Core building blocks are **Agents** (with sessions, context providers, middleware, MCP clients) and **Workflows**; orchestration patterns include sequential, concurrent, handoff, group-chat, and the Magentic-One hierarchical pattern. It supports **MCP natively** (thousands of existing MCP servers work as tools), plus **A2A** and **AG-UI** interoperability, and agents can be declared in YAML and version-controlled. By Build 2026 it shipped connectors for Azure OpenAI, OpenAI, Anthropic Claude, Amazon Bedrock, Google Gemini, and Ollama (the provider set expanded between RC and GA). Microsoft published dedicated migration guides from both Semantic Kernel and AutoGen.

This is the unambiguous default for .NET and Azure-native teams. The trade-off vs LangGraph: practitioner comparisons still rate LangGraph as more battle-tested at scale, while Agent Framework wins decisively for .NET shops and Azure integration.

> **Migration note**: existing Semantic Kernel and AutoGen code keeps working (maintenance mode) but receives no new features. New .NET agent work should target Microsoft Agent Framework; the migration guides cover both source frameworks.

### LlmTornado — the independent OSS .NET alternative

- **Licence**: MIT · **Maintainer**: community (`lofcz/LlmTornado`) · **NuGet**: v3.8.x
- A provider-agnostic .NET SDK with 30+ built-in connectors (Anthropic, Azure, OpenAI, Cohere, DeepSeek, Google, Groq, Mistral, xAI, and local backends via vLLM/Ollama/LocalAI) and **no dependency on first-party provider SDKs**. It offers agent orchestration (Orchestrator/Runner/Advancer), MCP (`LlmTornado.Mcp`) and A2A (`LlmTornado.A2A`) packages, and full multimodality. It has been featured in the Microsoft .NET AI Community Standup. Best when you want a single vendor-neutral .NET SDK without committing to Microsoft's framework or any one model provider.

### Other .NET routes

- **Dapr Workflow** (CNCF, GA) has a .NET SDK for durable, workflow-backed orchestration on Kubernetes; the higher-level Dapr Agents framework is Python-first at GA — see [§11](#11-durable-execution--workflow-engines).
- **Official model SDKs**: Anthropic, OpenAI, and Azure OpenAI all ship C# SDKs; for a simple tool-using loop, "C# SDK + your own loop" remains perfectly viable.
- There is **no official LangChain/LangGraph .NET port** — for graph orchestration in .NET, Microsoft Agent Framework's Workflows is the equivalent.

**.NET verdict**: Microsoft Agent Framework is the gold standard, now GA and MIT-licensed; LlmTornado is the strong provider-agnostic OSS alternative; Dapr Agents adds durability for Kubernetes deployments.

---

## 9. Vendor Agent SDKs

The four major model labs all ship an agent SDK. They converge on the same primitives and differ mainly on language coverage and which model they optimise for. A key point: the **SDK** is typically open-source (MIT/Apache); the **model API** behind it is proprietary.

| Vendor SDK | Languages | Licence (SDK) | Distinctive features |
|---|---|---|---|
| **OpenAI Agents SDK** | Python, TS/JS | MIT | Agents, Tools, Handoffs, Guardrails, Sessions, Tracing; hosted tools (web/file search, code interpreter); sandbox/harness added Apr 2026 (Python-first) |
| **Anthropic Claude Agent SDK** | Python, TS + headless CLI | SDK open; API proprietary | Same agent loop/tools/context management as Claude Code; native MCP with **in-process server model** + lifecycle hooks |
| **Google ADK** | Python, TS, Go, Java, Kotlin | Apache-2.0 | Most language coverage; workflow agents; first-class eval framework; A2A; deploys to Vertex AI Agent Engine |
| **Mistral Agents API** | Python, TS | API-hosted | Built-in connectors (code execution, image gen, document/RAG library, web search); MCP; stateful conversations; handoffs |

**How vendor SDKs differ from orchestration frameworks like LangGraph.** This is the single most common point of confusion. A vendor SDK is an *opinionated agent application layer* — the fastest path to a working tool-using agent with handoffs, guardrails, tracing, and MCP. LangGraph (and Eino, and Agent Framework Workflows) are *workflow infrastructure* — durable execution, checkpointing, resumability, explicit state machines, and human-in-the-loop governance for long-running, failure-prone business processes. They solve **different problems and compose**: you can run OpenAI/Anthropic models inside a LangGraph-driven workflow. The common mistakes are choosing LangGraph for a simple tool loop, or choosing a bare vendor SDK for a complex long-running process that needs durability.

The "provider-agnostic vs locked" nuance: OpenAI's and Anthropic's SDKs technically support other providers, but their *value* concentrates around their own platform features. Google ADK and Pydantic AI are the most genuinely model-agnostic of the SDK-shaped options.

---

## 10. Coding-Agent Harnesses

Coding agents are a distinct category of harness. Beyond a generic agent loop they add: **diff-based file editing**, **sandboxed command execution**, **repository context management**, and **test-loop iteration**. Empirically the harness matters as much as the model — practitioner testing reports a "harness effect" of roughly **+16 points on SWE-bench** for the same model run inside a proper coding harness versus a raw API call (a single-source figure, directionally consistent with the category's premise).

### Proprietary / product harnesses

- **Claude Code** (Anthropic) and the **Claude Agent SDK**: Claude Code is the interactive product; the Claude Agent SDK (Python, TS, headless CLI) exposes the *same* agent loop, tools, and context management programmatically. The gold-standard coding harness in 2026, and notable as a general agent SDK because of its MCP-native, in-process-server design.
- **OpenAI Codex CLI**: **open-source (Apache-2.0)**, built in Rust, runs locally in the terminal with sandboxed execution, `AGENTS.md` project config, and MCP support; integrates GPT-5.x models and IDE plugins. (Codex is unusual: an OSS harness in front of a proprietary model.)
- **Cursor** and **Google Jules**: proprietary — Cursor is the IDE-integrated agent; Jules is a cloud/VM-sandboxed autonomous coding agent.

### Open-source coding harnesses

- **OpenHands** (formerly OpenDevin): a full platform with Docker sandboxing, LLM-agnostic, the most complete OSS option.
- **Aider**: lightweight, git-native pair-programming in the terminal; works with local LLMs.
- **SWE-agent**: research/benchmarking-oriented, the reference agent-computer interface for SWE-bench.
- **Plandex**, **OpenCode**, **Pi**: additional OSS harnesses spanning planning-heavy and build-your-own-agent philosophies.

Open-source coding agents reportedly reached ~79% on SWE-bench Verified in 2026, materially closing the gap with the proprietary leaders. For most teams the choice is: **Claude Code / Claude Agent SDK** (best results, proprietary model) vs **OpenHands or Aider** (fully open, model-agnostic, self-hostable). For a deeper treatment see the companion articles *Agentic Coding: Claude Code vs OpenAI Codex* and *Open Models for Coding Agents* in this repository.

---

## 11. Durable Execution & Workflow Engines

This is the layer that turns a fragile in-memory agent loop into a production system. It matters for agents *specifically* because: (1) LLM calls are expensive and non-deterministic, so replaying from scratch wastes money and changes results; (2) tool calls have side effects (emails, payments, PRs) that must not be duplicated; (3) human-approval gates can take hours or days, which no in-memory loop survives; and (4) crash recovery must resume from the exact step without re-running completed work. Across every engine, three invariants hold: **LLM outputs are recorded once and replayed (never re-called) on recovery; workflow code must be deterministic; and every side-effecting tool call must be idempotent.**

| Engine | Model | Languages | Licence | Notes |
|---|---|---|---|---|
| **Temporal** | Journal / deterministic replay | Go, Java, TS, Python, .NET, PHP, Ruby | MIT (OSS) + Temporal Cloud | The enterprise reference; 99.99% SLA, multi-region |
| **Restate** | Journal / deterministic replay | TS, Java/Kotlin, Go, Python, Rust | Apache-2.0 | Single Rust binary, low ops surface, exactly-once invocation |
| **Inngest** | Step memoisation / checkpointing | TS, Python, Go, Java | Apache-2.0 (self-host) | Serverless-first, event-driven, TS-native |
| **DBOS** | DB checkpointing (Postgres) | TS, Python, Java, Go | OSS library | A *library*, not a server — only needs Postgres |
| **Dapr Agents** | Durable workflow per invocation | Python (GA); Dapr Workflow: Go/.NET/Java/JS | Apache-2.0 (CNCF) | GA Mar 2026; Kubernetes-native |
| **LangGraph checkpointers** | Superstep state snapshots | Python, TS | MIT | Framework-native durability + time travel |

### The 2026 picture

- **Temporal** is the gold-standard enterprise engine and the broadest by language (notably the strongest **Go** *and* **.NET** durable option). At Replay 2026 it shipped agent-specific features: **Workflow Streams** (durable real-time LLM token streaming), **External Payload Storage** (S3 for large AI payloads), **Serverless Workers** on Lambda, and **GA integrations with the OpenAI Agents SDK and Google ADK** — wrapping LLM and tool calls as durable Activities automatically. OpenAI is reported to run Codex's agent infrastructure on Temporal. The main gotcha is event-history size limits, which large LLM payloads can saturate (hence External Payload Storage).
- **Restate** offers a similar journal-replay model with far less operational surface (one Rust binary) and exactly-once invocation guarantees that reduce idempotency plumbing. Best for lighter-weight deployments.
- **Inngest** is the developer-experience pick for TypeScript/serverless teams; step-based pricing can balloon under multi-model retries.
- **DBOS** is the minimalist's choice — a library that adds crash-proof workflows using only Postgres, no separate orchestrator.
- **Dapr Agents** (CNCF, GA March 2026) is the Kubernetes-native option, running *every* agent invocation as a durable workflow. The agent framework itself is **Python** at v1.0 GA, but its durability rests on **Dapr Workflow**, whose SDKs are genuinely polyglot (Go, .NET, Java, JS) — and a community `dapr-agents-go` SDK (v0.2.0) is emerging. So for non-Python languages today, the durable-workflow layer is first-class while the agent-framework layer is still maturing.
- **Framework-native**: LangGraph's checkpointers, Pydantic AI's Temporal integration, and the OpenAI Agents SDK's session/snapshot system mean you can sometimes get "good enough" durability without a separate engine. The caveat from the research: **checkpoints alone do not make an agent durable** — you still need idempotent tool boundaries and durable, audited human-approval records (who/when/what/hash), not just a chat message.

**Durability verdict**: Temporal for serious production (and the only engine spanning all four languages well); Restate for a lighter journal-replay engine; Dapr for Kubernetes/polyglot; DBOS when you just want Postgres; framework-native checkpointing for simpler needs.

---

## 12. Hyperscaler Managed Agent Services

Per the SOTA Reference conventions, only AWS, Azure, GCP, IBM, and Oracle are covered. The common pattern in 2026 is **framework-agnostic managed runtimes**: rather than forcing their own framework, the leading clouds host whatever you bring (LangGraph, CrewAI, Strands, ADK …) and add memory, identity, gateways, observability, and scaling around it.

### AWS — Amazon Bedrock AgentCore

GA since **13 October 2025**. A framework- and model-agnostic platform of six composable services: **Runtime** (sessions up to 8 hours, per-second billing, no charge during I/O wait), **Memory**, **Gateway** (turns APIs and Lambda functions into MCP-compatible tools), **Identity** (verifiable per-agent identities), **Built-in Tools** (managed Browser and Code Interpreter), and **Observability** (CloudWatch + OpenTelemetry). It explicitly hosts agents from CrewAI, LangGraph, LlamaIndex, Strands, LangChain, the OpenAI Agents SDK, and the Claude Agent SDK. Pairs naturally with the open-source **AWS Strands Agents SDK** ([§5](#5-python-frameworks)). re:Invent 2025 previews added Cedar-based policy controls, evaluations, episodic memory, and bidirectional streaming.

### Azure — Microsoft Foundry Agent Service

GA (reached GA at Build 2025 as Azure AI Foundry Agent Service; rebranded **Microsoft Foundry Agent Service** in 2026). A managed runtime with multi-agent orchestration — **Connected Agents** (point-to-point delegation) and **Multi-Agent Workflows** (stateful orchestration) — built on the converged Semantic Kernel/AutoGen runtime that underpins the **Microsoft Agent Framework** ([§8](#8-c--net-frameworks)). It supports **MCP and A2A**, exposes Python/.NET SDKs plus OpenAI-SDK compatibility, hosts CrewAI/LangGraph/LlamaIndex agents, and offers 1,400+ Logic Apps and first-party tools (SharePoint, Fabric, Bing) with built-in evaluation and AgentOps tracing. The natural choice for Azure-native and .NET shops.

### GCP — Vertex AI Agent Engine + ADK

The managed runtime, **Vertex AI Agent Engine**, reached GA in early 2026 and hosts agents built on **ADK, LangGraph, or CrewAI** with autoscaling, request-scoped tracing, IAM binding, regional pinning, and persistent sessions. It is paired with the open-source **Google ADK** (the only first-party SDK spanning Python/TS/Go/Java/Kotlin) and the **A2A** protocol that Google created and donated to the Linux Foundation. At Cloud Next 2026 the broader offering was rebranded under the **Gemini Enterprise Agent Platform**, bundling code-first ADK, a low-code Agent Studio, 200+ models, and Agent Engine. The strongest pick for GCP/Gemini teams and for genuinely multi-language agent estates.

### IBM — watsonx Orchestrate

watsonx Orchestrate spans **no-code** (visual drag-and-drop) to **pro-code** (an **ADK** delivered as a Python library + CLI + APIs), lets you prototype in Langflow, connects tools via OpenAPI and **MCP**, and provides multi-agent orchestration with intelligent routing and shared context, an **Agent Connect** governed catalogue, and AgentOps observability/governance. It can host agents built on other frameworks and connect to models from multiple providers (including Amazon Bedrock). The enterprise-integration and governance angle is its differentiator.

### Oracle — OCI Generative AI Agents

A fully managed service centred on **RAG**, with tools for RAG, SQL, Agent-as-a-Tool (sub-agents), custom function calling, and custom API endpoints, plus guardrails, human-in-the-loop, and multi-turn context. Access is via the OCI Python SDK, available in seven regions. As of mid-2026 its public docs show **no MCP/A2A support and no open-source-framework hosting** — it is narrower and more RAG-focused than the AWS/Azure/GCP offerings, best suited to Oracle-native teams with retrieval-centric use cases.

**Hyperscaler verdict**: AWS AgentCore, Azure Foundry Agent Service, and Vertex AI Agent Engine are all mature, framework-agnostic, MCP-aware runtimes — **choose by your existing cloud**. IBM watsonx Orchestrate leads on governance and no-code/pro-code breadth; Oracle's service is the least open and most RAG-specific.

---

## 13. Interoperability Protocols (MCP, A2A, AGNTCY)

Two complementary open standards now anchor agent interoperability, with a third providing infrastructure beneath them.

- **MCP (Model Context Protocol)** — solves **tool and resource access**: a standard way for an agent to connect to external data, tools, and workflows ("USB-C for AI"). Governed under the Linux Foundation (LF Projects, LLC) with a BDFL model, Apache-2.0, and a Specification Enhancement Proposal process; latest spec **2025-11-25**. Adopted across Claude, ChatGPT, VS Code, Cursor, and every major framework in this article, with 10,000+ public servers. (Reporting that MCP was placed under an "Agentic AI Foundation" with Block and OpenAI in December 2025 comes from secondary sources and should be treated as such.)
- **A2A (Agent2Agent)** — solves **agent-to-agent communication**: lets autonomous agents collaborate as peers (rather than one wrapping another as a "tool"), using HTTP, JSON-RPC and SSE, with **Agent Cards** for discovery and support for long-running, multi-turn, opaque interactions. Created by Google, donated to the Linux Foundation; partners include Microsoft, SAP, Zoom, Box, Auth0.
- **MCP and A2A are complementary, not competing.** The canonical pattern: an agentic application uses **A2A *between* agents** and **MCP *within* each agent** for tool access.
- **AGNTCY** ("Internet of Agents") — not a protocol but an **infrastructure stack** for agent collaboration: discovery (OASF), identity, messaging (SLIM), and observability. Led by Cisco/Outshift and backed by LangChain and LlamaIndex among others; it is protocol-agnostic and works *with* both MCP and A2A.

The practical upshot: **MCP support is now mandatory** when evaluating a framework, and **A2A support is the emerging differentiator** for multi-agent and cross-vendor scenarios. For a deeper treatment of tool-calling strategies and MCP alternatives, see *MCP Alternatives and Tool-Calling Strategies* in this repository.

---

## 14. Beyond the Four Languages: the JVM

Although out of the core scope, the JVM deserves a brief note for completeness, since "what about Java?" is the inevitable next question. Java/Kotlin is a viable, if secondary, agent ecosystem in 2026:

- **Spring AI** reached **1.0 GA (May 2025)**, bringing tool calling, RAG, and agent primitives to Spring Boot with idiomatic dependency injection.
- **LangChain4j 1.0** provides lower-level, Java-native chains, agents, memory, and tool use.
- **Google ADK** ships a first-party **Java/Kotlin** SDK, and **A2A** has an official Java implementation.
- Durable execution is well covered (Temporal, Restate, and Dapr all have JVM SDKs).

So the language coverage story extends cleanly to five ecosystems; Java teams are not stranded.

---

## 15. Choosing a Framework — Decision Guide

A short decision procedure that subsumes the per-language verdicts:

1. **Start from your language and platform constraint.** This eliminates most of the field immediately (see [§3](#3-language-coverage-matrix)). A .NET shop on Azure is choosing between Microsoft Agent Framework and LlmTornado, not pondering LangGraph vs CrewAI.

2. **Classify the control flow.**
   - *Single tool-using loop* → a **vendor SDK** (OpenAI/Anthropic/ADK) or a thin loop on the model SDK. Don't reach for a graph framework.
   - *Branching, loops, multiple coordinated agents, human-in-the-loop* → a **graph/orchestration framework** (LangGraph, Eino, Agent Framework Workflows, Mastra).
   - *Role-decomposable task* → CrewAI (Python) for speed of prototyping.

3. **Decide the durability requirement explicitly.** Does the agent run for minutes-plus, pause for human approval, or call irreversible tools? If yes, choose a durability layer now (Temporal/Restate/Dapr or framework-native checkpointing) — see [§11](#11-durable-execution--workflow-engines). Retrofitting durability is painful.

4. **Decide build vs managed runtime.** If you are already committed to a cloud and want operations handled, a managed runtime (AgentCore / Foundry Agent Service / Vertex Agent Engine) hosts your chosen open-source framework — you are not choosing *between* the framework and the runtime, you are stacking them.

5. **Insist on MCP; weigh A2A.** Treat MCP support as mandatory. Weigh A2A support if you anticipate multi-agent or cross-vendor interoperability.

6. **Default to the gold standard, deviate with a reason.** The picks in [§4](#4-gold-standard-picks-tldr) are defensible defaults; choosing something else should be a deliberate trade (lighter weight, specific feature, existing investment).

---

## 16. Anti-Patterns and When NOT to Use a Framework

The strongest 2026 guidance is, paradoxically, often *don't reach for a heavy framework — or for multi-agent — by default.*

- **Framework-for-its-own-sake.** Anthropic's widely cited *Building Effective Agents* argues that most successful implementations use **simple, composable patterns, not complex frameworks**; frameworks add abstraction layers that obscure the actual prompts and responses and make debugging harder. Start with direct LLM API calls and add framework machinery only when it demonstrably improves outcomes.

- **Reflexive multi-agent ("the multi-agent trap").** A single well-prompted agent often suffices. Multi-agent systems incur a **coordination tax** (reported as the largest share of multi-agent failures in the MAST taxonomy), **token-cost multiplication** (retry loops can burn real money for no output), and **compound reliability decay** (trajectory reliability falls off multiplicatively with step count). Research cited in practitioner write-ups suggests centralised supervisor control suppresses the error amplification that unstructured "bag of agents" networks produce. Practical mitigations: keep chains short, insert verification steps, set hard per-agent token budgets with circuit breakers, and treat inter-agent messages with the same suspicion as external user input. *(These specific quantitative figures come from secondary syntheses of the MAST study and should be read as directional, not precise.)*

- **Ignoring framework churn.** A 2026 production survey found that **~70% of regulated enterprises rebuild their agent stack roughly every three months**, and fewer than one in three teams are satisfied with their observability/guardrails — with only ~5% of surveyed organisations running agents in production at all. The lesson is to **design for modularity, not lock-in**: keep prompts, tools, and business logic separable from the framework so that swapping frameworks is a contained change. (Sample sizes are small; treat as directional.)

- **Treating checkpoints as durability.** As noted in [§11](#11-durable-execution--workflow-engines), saving graph state is necessary but not sufficient — without idempotent tool boundaries and durable, audited approvals, a "resumable" agent can still double-charge a customer.

- **Skipping observability.** Tracing and evaluation are not optional at production scale. This article does not cover the observability layer in depth; see the companion *Modern AI Observability* article (OpenTelemetry GenAI conventions, plus open-source tools like Langfuse/OpenLLMetry and the hyperscaler equivalents).

---

## 17. Confidence, Caveats and Limitations

**Overall confidence: high (~0.88)** for the load-bearing claims — which frameworks exist, their languages, licences, and architecture — because these are anchored in primary sources (official vendor docs/blogs, official repositories, PyPI/npm, CNCF, and the official MCP/A2A specifications), and corroborated across independent sources.

Specific caveats:

- **Version and date specifics** (e.g. exact GA dates, version numbers like Mastra 1.0 in Jan 2026, Strands 1.0 in May 2026, Microsoft Agent Framework GA on 3 April 2026, ADK Go 1.0 on 31 March 2026) are well-corroborated but evolve quickly. Re-verify against the official release notes before relying on them.
- **Popularity metrics** (GitHub stars, download counts) are point-in-time figures from secondary aggregators, some of them AI-generated content sites. They indicate rough scale, not precise ranking, and download counts can be inflated by transitive dependencies.
- **Quantitative multi-agent failure statistics** (coordination-tax percentages, error-amplification multiples, the "+16-point harness effect") come from single secondary syntheses and are presented as directional, not authoritative.
- **The "Agentic AI Foundation" governance claim for MCP** (December 2025) rests on a single secondary source and is flagged as reported, not confirmed.
- **Rapid obsolescence**: this is among the fastest-moving areas in software. Google ADK reaching 2.0 with breaking changes within months of 1.0 is emblematic. A 2026 snapshot will date faster than most SOTA topics.
- **Scope boundaries**: observability tooling, LLM model selection, and RAG pipeline internals are deliberately out of scope and covered by sibling articles in this repository.

**Areas of genuine uncertainty**: the precise maturity ordering among the newest 1.0 releases (Mastra, Strands, ADK Go, Microsoft Agent Framework) under real production load; the long-term governance trajectory of MCP/A2A; and whether the current framework consolidation holds or fragments again.

---

## References

Primary sources (official docs, repositories, vendor blogs, standards bodies) are listed first, followed by secondary analyses used for corroboration and colour.

### Primary — vendor documentation, repositories, and standards

1. [Pydantic AI v1 release](https://pydantic.dev/articles/pydantic-ai-v1) — official announcement (V1 stability, Temporal durable execution, HITL tool approval).
2. [Strands Agents — AWS Prescriptive Guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-frameworks/strands-agents.html) — model-driven, open-source, Python/TS.
3. [Amazon Bedrock AgentCore — AWS Prescriptive Guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-frameworks/amazon-bedrock-agentcore.html) — six services, framework-agnostic, MCP gateway.
4. [Amazon Bedrock AgentCore — product page](https://aws.amazon.com/bedrock/agentcore/) — supported frameworks.
5. [google-adk — PyPI](https://pypi.org/project/google-adk/) — Apache-2.0, code-first, ADK 2.0, multi-language.
6. [Agent Development Kit (ADK) — official site](https://adk.dev/) — Python/TS/Go/Java/Kotlin, eval framework, A2A.
7. [ADK Go 1.0 Arrives! — Google Developers Blog](https://developers.googleblog.com/adk-go-10-arrives/) — Go 1.0, OTel, HITL, A2A.
8. [What's new with Agents: ADK, Agent Engine, A2A — Google Developers Blog](https://developers.googleblog.com/en/agents-adk-agent-engine-a2a-enhancements-google-io/) — Python ADK 1.0, Agent Engine, A2A v0.2.
9. [Migrate Semantic Kernel and AutoGen to Microsoft Agent Framework RC — Microsoft DevBlogs](https://devblogs.microsoft.com/agent-framework/migrate-your-semantic-kernel-and-autogen-projects-to-microsoft-agent-framework-release-candidate/) — RC, MCP/A2A/AG-UI, migration guides.
10. [What is Microsoft Foundry Agent Service? — Microsoft Learn](https://learn.microsoft.com/en-us/azure/foundry/agents/overview) — managed runtime, multi-agent, A2A/MCP.
11. [Claude Agent SDK overview — Anthropic docs](https://code.claude.com/docs/en/agent-sdk/overview) — Python/TS/CLI, MCP-native, in-process server.
12. [anthropics/claude-agent-sdk-typescript — GitHub](https://github.com/anthropics/claude-agent-sdk-typescript) — official TS SDK.
13. [Claude API Client SDKs — Anthropic](https://platform.claude.com/docs/en/api/client-sdks) — official SDKs incl. Go and C#.
14. [openai/codex — GitHub](https://github.com/openai/codex) — Apache-2.0 Rust coding-agent CLI, MCP, sandbox.
15. [Build AI agents with the Mistral Agents API — Mistral AI](https://mistral.ai/news/agents-api/) — connectors, MCP, stateful, handoffs.
16. [cloudwego/eino — GitHub](https://github.com/cloudwego/eino) — Go-idiomatic LLM/agent framework (ByteDance), Apache-2.0.
17. [genkit-ai/genkit-go — GitHub](https://github.com/genkit-ai/genkit-go) — Google/Firebase Go AI SDK.
18. [tmc/langchaingo — GitHub](https://github.com/tmc/langchaingo) — Go LangChain implementation, MIT.
19. [Official MCP Go SDK — GitHub](https://github.com/modelcontextprotocol/go-sdk) — v1.4.0+, client/server, spec 2025-11-25.
20. [lofcz/LlmTornado — GitHub](https://github.com/lofcz/LlmTornado) — provider-agnostic .NET SDK, MIT, MCP/A2A.
21. [Temporal Replay 2026 product announcements](https://temporal.io/blog/replay-2026-product-announcements) — Workflow Streams, OpenAI Agents SDK + ADK integrations.
22. [Durable Agent with OpenAI Agents SDK — Temporal docs](https://docs.temporal.io/ai-cookbook/openai-agents-sdk-python) — durable agent integration.
23. [DBOS Transact — official](https://www.dbos.dev/dbos-transact) — open-source durable execution library on Postgres.
24. [General Availability of Dapr Agents — CNCF](https://www.cncf.io/announcements/2026/03/23/general-availability-of-dapr-agents-delivers-production-reliability-for-enterprise-ai/) — v1.0 GA, durable workflows, polyglot.
25. [Dapr Workflow — official](https://dapr.io/workflow/) — durable workflows, Go/Python/.NET/Java/JS.
26. [watsonx Orchestrate AI Agent Builder — IBM](https://www.ibm.com/products/watsonx-orchestrate/ai-agent-builder) — no-code/pro-code ADK, MCP, Agent Connect.
27. [watsonx Orchestrate ADK — IBM developer docs](https://developer.watson-orchestrate.ibm.com/) — Python library + CLI.
28. [OCI Generative AI Agents — Oracle docs](https://docs.oracle.com/en-us/iaas/Content/generative-ai-agents/overview.htm) — managed RAG-focused agent service.
29. [MCP Governance and Stewardship — official](https://modelcontextprotocol.io/community/governance.md) — Linux Foundation, BDFL, SEP process.
30. [What is MCP? — official intro](https://modelcontextprotocol.io/docs/getting-started/intro.md) — tool/resource access standard.
31. [What is A2A? — A2A Protocol](https://a2a-protocol.org/latest/topics/what-is-a2a/) — inter-agent comms, Linux Foundation.
32. [A2A and MCP comparison — A2A project](https://raw.githubusercontent.com/a2aproject/A2A/main/docs/topics/a2a-and-mcp.md) — complementary roles.
33. [AGNTCY — Internet of Agents](https://agntcy.org) — discovery/identity/messaging/observability infrastructure.
34. [Building Effective Agents — Anthropic Engineering](https://www.anthropic.com/engineering/building-effective-agents) — simple patterns over frameworks.
35. [AI SDK 6 — Vercel](https://vercel.com/blog/ai-sdk-6) — agent loop + tool approval.
36. [ai — npm (Vercel AI SDK)](https://www.npmjs.com/package/ai) — v6.0.x, dependents.
37. [mastra-ai/mastra — GitHub](https://github.com/mastra-ai/mastra) — TypeScript agent framework, Apache-2.0.
38. [openai/openai-agents-js — GitHub](https://github.com/openai/openai-agents-js) — official TS Agents SDK.
39. [cloudflare/agents — GitHub](https://github.com/cloudflare/agents) — Durable Objects-backed agents.
40. [AgentKit by Inngest](https://agentkit.inngest.com/) — TS agent orchestration on durable execution.
41. [VoltAgent/voltagent — GitHub](https://github.com/VoltAgent/voltagent) — observability-first TS framework.
42. [LlamaIndex.TS documentation](https://developers.llamaindex.ai/typescript/framework/) — TS context-engineering/RAG.
43. [Spring AI 1.0 GA Released — spring.io](https://spring.io/blog/2025/05/20/spring-ai-1-0-GA-released/) — JVM agent primitives.

### Secondary — analyses and comparisons (corroboration / colour)

44. [Choosing an agent framework — Speakeasy](https://www.speakeasy.com/blog/ai-agent-framework-comparison) — licences, TS comparison, durability notes.
45. [Best AI Agent Frameworks in 2026 — ChatForest](https://chatforest.com/guides/best-ai-agent-frameworks-2026/) — 14-framework comparison (AI-written; stats cross-checked).
46. [Definitive Guide to Agentic Frameworks in 2026 — Softmax Data](https://softmaxdata.com/blog/definitive-guide-to-agentic-frameworks-in-2026-langgraph-crewai-ag2-openai-and-more/) — version milestones.
47. [Mastra complete guide 2026 — Generative.inc](https://www.generative.inc/mastra-ai-the-complete-guide-to-the-typescript-agent-framework-2026) — Mastra metrics/history.
48. [OpenAI Agents SDK vs LangGraph 2026 — Nerova](https://nerova.ai/comparisons/openai-agents-sdk-vs-langgraph-2026) — vendor SDK vs orchestration framework.
49. [Microsoft Agent Framework / Build 2026 — ByteIota](https://byteiota.com/windows-agent-framework-azure-agent-mesh-build-2026/) — GA date, MIT, primitives.
50. [AutoGen + Semantic Kernel unified — Paperclipped](https://www.paperclipped.de/en/blog/microsoft-agent-framework-autogen-semantic-kernel/) — merger analysis.
51. [Durable Execution Engines in 2026 — Youngju Kim](https://www.youngju.dev/blog/culture/2026-05-14-durable-execution-engines-2026-temporal-restate-inngest-trigger-dev-dbos-deep-dive-2026.en) — engine comparison.
52. [Durable Execution for AI Agent Runtimes — Zylos Research](https://zylos.ai/research/2026-04-24-durable-execution-agent-runtimes/) — why agents need durability.
53. [Durable Execution for Agents: Temporal vs Inngest vs Restate — Particula](https://particula.tech/blog/durable-execution-ai-agents-temporal-inngest-restate) — decision framework.
54. [The Multi-Agent Trap — Towards Data Science](https://towardsdatascience.com/the-multi-agent-trap/) — MAST failure modes (figures directional).
55. [AI Agents in Production 2025 — Cleanlab](https://cleanlab.ai/ai-agents-in-production-2025/) — framework-churn survey.
56. [The Go Revolution: Golang in AI Agent Development — Mule AI](https://muleai.io/blog/2026-02-28-golang-ai-agent-frameworks-2026/) — Go framework survey (Go-focused bias noted).
57. [Amazon Strands Agents 1.0 — AgentMarketCap](https://agentmarketcap.ai/blog/2026/04/07/amazon-strands-agent-framework-aws-open-source-bedrock-agentcore) — Strands metrics.
58. [Azure AI Foundry Agent Service GA — MobileMonitoringSolutions](https://mobilemonitoringsolutions.com/azure-ai-foundry-agent-service-ga-introduces-multi-agent-orchestration-and-open-interoperability/) — Azure GA details.
59. [Evaluating Vertex AI Agent Engine in 2026 — FutureAGI](https://futureagi.com/blog/evaluating-vertex-ai-agent-engine-2026/) — Agent Engine capabilities.
60. [AI coding harnesses 2026 — thoughts.jock.pl](https://thoughts.jock.pl/p/ai-coding-harness-agents-2026) — "harness effect" (single-source figure).

### Related SOTA Reference articles

- [Frameworks for Research Agents](frameworks-research-agents.md) — deeper dive on research-agent-specific patterns.
- [Agentic Coding: Claude Code vs OpenAI Codex](agentic-coding-claude-vs-openai.md)
- [Open Models for Coding Agents](open-models-coding-agents.md)
- [MCP Alternatives and Tool-Calling Strategies](mcp-alternatives-and-tool-calling-strategies.md)
- [Modern AI Observability](../evaluation/ai-observability.md)
