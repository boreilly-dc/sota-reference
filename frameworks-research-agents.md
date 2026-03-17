# Frameworks for Research Agents — State of the Art Reference

| Field | Value |
|-------|-------|
| Created | 2026-03-17 |
| Last Updated | 2026-03-17 |
| Version | 1.1 |

---

- [1. What Makes a Research Agent Different](#1-what-makes-a-research-agent-different)
- [2. Agent Orchestration Frameworks](#2-agent-orchestration-frameworks)
- [3. Vendor Agent SDKs](#3-vendor-agent-sdks)
- [4. Hyperscaler Managed Services](#4-hyperscaler-managed-services)
- [5. Interoperability Protocols](#5-interoperability-protocols)
- [6. Reasoning Patterns](#6-reasoning-patterns)
- [7. Research-Agent Reference Architectures](#7-research-agent-reference-architectures)
- [8. Evaluation Frameworks](#8-evaluation-frameworks)
- [9. State Management and Memory](#9-state-management-and-memory)
- [10. Choosing a Framework — Decision Guide](#10-choosing-a-framework--decision-guide)
- [11. Anti-Patterns](#11-anti-patterns)
- [12. Critical Perspective](#12-critical-perspective)
- [References](#references)

Research agents go beyond single-turn Q&A: they decompose questions, plan multi-step searches, critique their own findings, and synthesise verified answers with citations. This article surveys the frameworks, protocols, reasoning patterns, and evaluation tools that matter when building them — from open-source orchestration libraries through to hyperscaler managed services.

---

## 1. What Makes a Research Agent Different

A research agent is distinguished from a generic LLM agent by several requirements:

- **Multi-step planning**: decomposing a broad question into sub-questions, then executing searches in sequence or in parallel.
- **Source management**: tracking provenance, assessing credibility, and deduplicating across sources.
- **Self-critique and verification**: the agent must be able to challenge its own findings, identify gaps, and seek contrary evidence.
- **Long-running state**: research sessions can span minutes to hours, requiring robust checkpointing and context management.
- **Structured output**: producing cited, confidence-scored findings rather than free-form text.

These requirements constrain framework choice more than most agent use cases.

---

## 2. Agent Orchestration Frameworks

### LangGraph (LangChain)

**Type**: Graph-based orchestration
**Licence**: MIT
**Language**: Python, TypeScript/JavaScript
**Best for**: Complex, stateful workflows with conditional branching

LangGraph models agent workflows as directed graphs where nodes are processing steps and edges define control flow. Key features for research agents:

- **Conditional edges**: route execution based on intermediate results (e.g., "if confidence < 0.7, run additional search").
- **Checkpointing**: built-in thread-scoped persistence via `MemorySaver` or database-backed stores, enabling pause/resume of long research sessions.
- **Subgraph composition**: nest graphs within graphs for modular architectures (e.g., a search subgraph, a verification subgraph).
- **Human-in-the-loop**: native `interrupt()` function to pause execution and collect human feedback before continuing.
- **Streaming**: token-level and step-level streaming for real-time progress reporting.
- **Platform**: LangGraph Platform (formerly LangGraph Cloud) provides managed deployment with a task queue, cron scheduling, and double-texting support.

LangGraph is arguably the most mature open-source option for research agents due to its explicit state management and graph-based control flow, though it carries the complexity cost of the broader LangChain ecosystem.

```python
# Simplified LangGraph research agent structure
from langgraph.graph import StateGraph, START, END

graph = StateGraph(ResearchState)
graph.add_node("plan", plan_research)
graph.add_node("search", execute_searches)
graph.add_node("critique", self_critique)
graph.add_node("verify", verify_claims)
graph.add_node("synthesise", produce_report)

graph.add_edge(START, "plan")
graph.add_edge("plan", "search")
graph.add_conditional_edges("search", should_critique_or_verify)
graph.add_edge("critique", "search")  # Loop back
graph.add_edge("verify", "synthesise")
graph.add_edge("synthesise", END)
```

**Managed equivalent**: LangGraph Platform (self-hosted or LangChain-hosted). No direct hyperscaler equivalent, though LangGraph agents can be deployed on any cloud.

### CrewAI

**Type**: Role-based multi-agent
**Licence**: MIT
**Language**: Python
**Best for**: Multi-agent collaboration with defined roles

CrewAI organises agents into "crews" with explicit roles, goals, and backstories. Agents can delegate tasks to each other and share context. Research-relevant features:

- **Role specialisation**: define a "Researcher" agent, a "Fact-Checker" agent, and an "Editor" agent, each with distinct system prompts and tool access.
- **Process types**: sequential (agents run in order) or hierarchical (a manager agent delegates to specialists).
- **Task delegation**: agents can autonomously decide to hand off subtasks.
- **Memory**: short-term (conversation), long-term (cross-session via a vector store), and entity memory.
- **Structured output**: Pydantic model support for enforcing output schemas.

CrewAI trades some of LangGraph's fine-grained control flow for faster prototyping of multi-agent teams. Research agents benefit from the natural mapping of research roles (searcher, critic, synthesiser) to CrewAI's agent abstraction.

**Limitation**: Less control over execution flow than graph-based approaches; debugging multi-agent delegation chains can be opaque.

### Microsoft Agent Framework (AutoGen + Semantic Kernel)

**Type**: Event-driven multi-agent
**Licence**: MIT
**Language**: Python, .NET, Java
**Best for**: Enterprise environments, complex multi-agent conversations

In October 2025, Microsoft merged AutoGen and Semantic Kernel into a unified Agent Framework. Key features:

- **Event-driven architecture**: agents communicate via an event bus, enabling loose coupling and scalability.
- **Conversation patterns**: supports group chat, nested chat, sequential pipelines, and custom topologies.
- **Code execution**: built-in sandboxed code execution for data analysis steps within research workflows.
- **Plugin system**: Semantic Kernel's plugin architecture provides a rich ecosystem of pre-built connectors.
- **Enterprise integration**: Deep integration with Azure services, but the core framework is fully open-source and model-agnostic.

The framework is powerful but carries significant complexity. It is best suited for teams already invested in the Microsoft ecosystem or building enterprise-grade multi-agent systems.

### SmolAgents (Hugging Face)

**Type**: Code-first minimal agent
**Licence**: Apache 2.0
**Language**: Python
**Best for**: Lightweight agents, rapid prototyping, open-model deployments

SmolAgents is Hugging Face's deliberately minimal agent framework (~1,000 lines of core code). The key differentiator is **code-as-action**: instead of selecting from predefined tools via JSON, the agent writes and executes Python code directly.

- **Code agents**: the LLM generates executable Python rather than tool-call JSON, enabling arbitrary computation without pre-defining every possible action.
- **Model flexibility**: first-class support for Hugging Face models, enabling fully open-source research agent stacks.
- **MCP support**: native Model Context Protocol integration for tool discovery.
- **Multi-agent**: supports hierarchical multi-agent setups where a manager agent delegates to specialist code agents.

SmolAgents is ideal when you want a research agent that can perform ad-hoc data analysis, transformation, or computation as part of its workflow without the overhead of a full orchestration framework.

**Limitation**: No built-in checkpointing or state persistence; you must implement your own for long-running research sessions.

### AWS Agent Squad (formerly Multi-Agent Orchestrator)

**Type**: Router-based multi-agent
**Licence**: Apache 2.0
**Language**: Python, TypeScript
**Best for**: Intent-based routing to specialist agents

AWS Agent Squad provides a lightweight orchestration layer focused on routing incoming queries to the most appropriate specialist agent:

- **Classifier-based routing**: an LLM-based classifier determines which agent should handle each query.
- **Agent types**: supports Bedrock agents, Lex bots, Lambda functions, LangChain agents, and custom agents as targets.
- **Conversation memory**: built-in per-agent and cross-agent memory management.
- **Streaming**: native response streaming support.

For research agents, Agent Squad is useful as a top-level router (e.g., routing different types of research questions to specialised sub-agents) rather than as the full orchestration layer.

### A Note on LlamaIndex

LlamaIndex is not an agent orchestration framework — it is a **data framework** for connecting private and public data to LLMs. However, it is a critical building block for research agents because of its strengths in document ingestion, indexing (vector, tree, keyword, knowledge graph), and multi-document retrieval. LlamaIndex tools can be integrated into agent frameworks such as CrewAI and LangGraph, making it complementary rather than competing. If your research agent needs sophisticated RAG over diverse data sources (PDFs, SQL, APIs, Notion, Slack), LlamaIndex is likely part of the stack even if it is not the orchestration layer.

---

## 3. Vendor Agent SDKs

These are model-provider SDKs optimised for their respective LLMs. They are not orchestration frameworks per se, but provide the building blocks (tool use, streaming, guardrails) that research agents need.

### Claude Agent SDK (Anthropic)

**Released**: September 2025
**Language**: Python, TypeScript
**Licence**: MIT

The Claude Agent SDK is essentially **Claude Code exposed as a library** — it provides the same agent loop, tool execution engine, and context management that power the Claude Code CLI, but in a programmable form.

- **Production-grade tooling out of the box**: file I/O, shell execution, code editing, web search, and web fetch are all built in. Most competing SDKs require developers to implement tool execution themselves.
- **Native MCP support**: first-class Model Context Protocol integration for connecting to any MCP-compatible tool server alongside built-in tools.
- **Subagent patterns**: spawn subordinate agents for task delegation (e.g., one subagent researches while another writes).
- **Hooks**: lifecycle hooks inject custom logic at specific points in the agent loop (before/after tool calls), useful for logging, approval gates, and observability.
- **Sessions**: persistent conversational state across multiple calls, with filesystem-based configuration (CLAUDE.md, skills, slash commands) for version-controlled agent behaviour.
- **Multi-provider auth**: supports Anthropic API, Amazon Bedrock, Google Vertex AI, and Azure AI Foundry.
- **Permissions system**: built-in tool-level allow-listing for constraining agent access.

The Claude Agent SDK is the most natural choice when building research agents powered by Claude models. Its combination of built-in tools and MCP integration means a research agent can be functional with minimal tool implementation.

### OpenAI Agents SDK

**Language**: Python
**Licence**: MIT

The entire framework reduces to four concepts: Agents, Handoffs, Guardrails, and Tools. Despite being built by OpenAI, it is model-agnostic — any provider exposing an OpenAI-compatible Chat Completions endpoint can be used as the backend.

- **Code-first**: minimal abstractions over the Responses API.
- **Handoffs**: first-class agent-to-agent delegation primitive — elegant for triage/routing patterns, though not true parallel orchestration.
- **Guardrails**: input/output validation via `@input_guardrail`/`@output_guardrail` decorators.
- **Tracing**: built-in execution tracing for debugging.
- **MCP integration**: first-class MCP support via `MCPServerStdio`, including hosted tools (web search, code interpreter).

### Google Agent Development Kit (ADK)

**Language**: Python
**Licence**: Apache 2.0

Google ADK treats agents as modular, testable, composable software components — closer to a microservices philosophy than a prompt-chaining one.

- **Multi-model**: supports Gemini natively but can work with other models.
- **Built-in orchestration patterns**: `SequentialAgent` (pipeline), `ParallelAgent` (concurrent execution with result merging), and `LoopAgent` (iterative refinement) cover ~90% of multi-agent workflows out of the box.
- **Agent-as-tool**: one agent can be exposed as a tool for another agent via `.as_tool()`, enabling flexible composition.
- **Bidirectional streaming**: real-time voice and video agent interactions, with native multimodal support (text, images, video, audio).
- **Session management**: built-in session architecture separating short-term memory (session state) from long-term memory (pluggable services), with `InMemorySessionService` for development and `DatabaseSessionService` with PostgreSQL for production.
- **A2A protocol support**: native Agent-to-Agent protocol integration.
- **Managed deployment**: Vertex AI and Cloud Run integration with auto-scaling.

---

## 4. Hyperscaler Managed Services

For teams preferring managed infrastructure over self-hosted frameworks:

| Service | Provider | Key Features | Best For |
|---------|----------|-------------|----------|
| **Azure AI Foundry** | Azure | AutoGen-based, enterprise connectors, Azure AI Search integration | Enterprise research agents in Microsoft ecosystem |
| **Amazon Bedrock Agents** | AWS | Widest third-party model selection (Claude, Llama, Mistral, Cohere, Titan), knowledge bases via OpenSearch, action groups via Lambda, guardrails, reserved throughput for 30–50% cost savings | AWS-native deployments with managed RAG; model flexibility to avoid vendor lock-in |
| **Google Vertex AI Agent Builder** | GCP | Gemini-optimised (2M token context window), grounding with Google Search, Vertex AI Search integration, AutoML, Feature Store, drift/skew detection, Kubeflow Pipelines | ML-heavy teams building custom models alongside agents; cost-sensitive (Gemini is cheapest) |
| **IBM watsonx Orchestrate** | IBM | Skills-based agent composition, enterprise automation focus | Regulated industries, workflow automation |
| **Oracle AI Agent Studio** | Oracle | Database-integrated agents, Oracle Cloud Infrastructure integration | Data-intensive research with Oracle backends |

**Key consideration**: Managed services reduce operational burden but limit customisation. Research agents often require complex, custom control flow (critique loops, verification passes) that managed services may not fully support. A hybrid approach — using managed services for infrastructure (model hosting, vector stores) while implementing orchestration logic in code — is often pragmatic.

---

## 5. Interoperability Protocols

Four protocols are shaping how research agents connect to tools and to each other. They address different layers of the stack and are complementary, not competing.

### Model Context Protocol (MCP)

**Origin**: Anthropic, November 2024
**Scope**: Tool integration (agent-to-tool)
**Transport**: JSON-RPC over stdio or HTTP+SSE
**Status**: Widely adopted; supported by Claude, OpenAI, Google ADK, SmolAgents, and most major frameworks

MCP standardises how an LLM agent discovers and invokes external tools. It defines:

- **Resources**: data the agent can read (files, database rows, API responses).
- **Tools**: actions the agent can execute (search, compute, write).
- **Prompts**: reusable prompt templates exposed by the server.
- **Sampling**: allows the MCP server to request LLM completions from the client.

For research agents, MCP is foundational — it allows a single agent to seamlessly use search engines, databases, document parsers, and specialised APIs through a uniform interface. The growing ecosystem of MCP servers means new research tools can be integrated without modifying agent code.

**Limitation**: MCP is point-to-point (one client to one server). It does not address agent-to-agent communication or multi-party coordination.

### Agent Communication Protocol (ACP)

**Origin**: IBM, contributed to Linux Foundation, March 2025
**Scope**: Agent-to-agent communication
**Transport**: REST/HTTP with multimodal content support
**Status**: Early adoption, growing Linux Foundation backing

ACP enables agents to communicate with each other through a standardised REST API. Key features:

- **Multimodal messages**: supports text, images, audio, and binary content in agent exchanges.
- **Asynchronous execution**: agents can submit tasks and poll for results, suitable for long-running research sub-tasks.
- **Agent discovery**: agents can advertise their capabilities for dynamic team composition.
- **Framework-agnostic**: any agent that exposes the ACP API can participate, regardless of underlying framework.

ACP is most relevant for research systems where multiple specialised agents (e.g., a search agent, a fact-checking agent, a domain-expert agent) need to collaborate without tight coupling.

### Agent-to-Agent Protocol (A2A)

**Origin**: Google, April 2025
**Scope**: Cross-organisational agent coordination
**Transport**: HTTP, JSON-RPC
**Status**: Enterprise-focused, growing adoption

A2A addresses scenarios where agents from different organisations or platforms need to coordinate:

- **Agent Cards**: JSON metadata files describing an agent's capabilities, published at a well-known URL.
- **Task lifecycle**: defines a standard lifecycle for delegated tasks (submitted, working, completed, failed).
- **Push notifications**: webhook-based notifications for task state changes.
- **Enterprise features**: authentication, authorisation, and audit trails for cross-boundary agent interactions.

For research agents, A2A is relevant when building federated research systems — e.g., an internal research agent that can delegate domain-specific questions to external expert agents.

### Agent Network Protocol (ANP)

**Origin**: Community-driven, 2025
**Scope**: Decentralised peer-to-peer agent networks
**Transport**: DID-based, peer-to-peer
**Status**: Experimental

ANP enables fully decentralised agent-to-agent communication without central registries:

- **Decentralised identity**: agents use Decentralised Identifiers (DIDs) for authentication.
- **Peer-to-peer**: no central server required.
- **Open networks**: designed for internet-scale agent ecosystems.

ANP is the most forward-looking of the four protocols and is not yet mature enough for production research agents, but it points towards a future where agents from different providers can collaborate without pre-arranged integrations.

### Protocol Support by Framework

| Framework / SDK | MCP | ACP | A2A |
|----------------|-----|-----|-----|
| LangGraph | Via LangChain | No | Via integration |
| CrewAI | Partial | No | No |
| MS Agent Framework | Via plugins | No | No |
| SmolAgents | Native | No | No |
| Claude Agent SDK | Native | No | No |
| OpenAI Agents SDK | Native | No | No |
| Google ADK | Limited | No | Native |

MCP has the broadest adoption. A2A is currently native only in Google ADK, with LangGraph offering integration support. ACP adoption in frameworks is still nascent — it is primarily used via direct REST integration rather than framework-level abstractions.

### Protocol Adoption Roadmap

Based on maturity and practical value, a phased adoption path for research agents:

1. **Now**: Implement **MCP** for tool integration — immediate, high-value, broad ecosystem support.
2. **Near-term**: Add **ACP** for internal multi-agent communication — useful when your research system grows beyond a single agent.
3. **Medium-term**: Adopt **A2A** for cross-organisational agent delegation — relevant for federated research scenarios.
4. **Future**: Monitor **ANP** for decentralised agent networks — experimental, watch for maturity signals.

---

## 6. Reasoning Patterns

The reasoning pattern defines how an agent thinks and acts. Different patterns suit different research tasks.

### ReAct (Reason + Act)

The foundational agent pattern: the LLM alternates between reasoning ("I need to find...") and acting (tool calls). Simple, robust, and the default in most frameworks.

**Best for**: Straightforward research queries where a linear search-reason-search loop suffices.

### Plan-and-Execute

Separates planning from execution: a planner LLM decomposes the question into a task list, then an executor LLM carries out each task. Replanning occurs after each step based on results.

**Best for**: Complex research questions that benefit from upfront decomposition.

### Plan-Execute-Verify-Replan (VMAO)

Extends Plan-and-Execute with explicit verification loops. After execution, a verifier checks results for accuracy, completeness, and consistency. If gaps or errors are found, the planner generates corrective sub-tasks.

**Best for**: High-stakes research where accuracy is critical. The verification loop maps naturally to self-critique in research workflows.

### Reflexion

The agent maintains an explicit self-reflection step after each action, building a growing "reflection memory" that improves subsequent decisions. Unlike simple ReAct, the agent explicitly reasons about what went wrong and how to improve.

**Best for**: Iterative research where early searches inform the strategy for later ones.

### LATS (Language Agent Tree Search)

Combines LLM reasoning with Monte Carlo Tree Search. The agent explores multiple reasoning paths in parallel, evaluates them, and selects the most promising. Provides backtracking when a path proves unproductive.

**Best for**: Research questions with high uncertainty where exploring multiple hypotheses simultaneously is valuable.

### Tree of Thoughts

Extends chain-of-thought prompting by exploring multiple reasoning branches at each step, using the LLM itself to evaluate which branches are most promising.

**Best for**: Research synthesis where multiple interpretive framings of the evidence are possible.

### REWOO (Reasoning Without Observation)

Plans all tool calls upfront before executing any of them, minimising the number of LLM invocations. The planner generates a complete execution plan with tool calls, then a worker executes all tools, and finally a solver synthesises the results.

**Best for**: Cost-sensitive research tasks with predictable tool-call patterns. REWOO can be 3-5x cheaper than ReAct for straightforward queries because it avoids the reasoning overhead between each tool call.

**Limitation**: Inflexible — if an early tool call returns unexpected results, the pre-planned subsequent calls may be wasted. Best combined with a verification step.

### LLM Compiler

Decomposes a task into a DAG of parallel sub-tasks with dependency tracking. Independent sub-tasks execute concurrently, with results merged at synchronisation points.

**Best for**: Research that involves many independent searches (e.g., investigating multiple aspects of a topic simultaneously). The parallel execution dramatically reduces wall-clock time.

### Pattern Combinations

These patterns are not mutually exclusive. Effective research agents often combine them:

- **Plan-Execute + Reflexion**: plan the research, execute it, then reflect on gaps and replan. The core loop of most production research agents.
- **ReAct + Reflexion** ("ReAction"): action with self-correction — the agent acts, reflects on what went wrong, and adjusts.
- **STORM + Plan-Execute**: multi-perspective planning followed by structured execution.
- **LLM Compiler + Verification**: parallel execution with a post-hoc verification pass to catch errors.

The trend is toward **dynamic orchestration** — agents that switch between reasoning modes depending on context (reflective when uncertain, reactive when time-critical, parallel when sub-tasks are independent).

### Pattern Selection Guide

| Pattern | Complexity | Parallelism | Self-Correction | Cost | Best Research Use |
|---------|-----------|-------------|-----------------|------|-------------------|
| ReAct | Low | None | Implicit | $$ | Simple fact-finding |
| REWOO | Medium | None | None | $ | Cost-sensitive queries |
| Plan-and-Execute | Medium | Sequential | Replanning | $$ | Structured investigations |
| VMAO | High | DAG-based | Explicit verification | $$$ | High-accuracy research |
| Reflexion | Medium | None | Reflection memory | $$$ | Iterative exploration |
| Tree of Thoughts | High | Branching | Branch evaluation | $$$ | Multi-framing synthesis |
| LATS | High | Tree-based | Backtracking | $$$$ | Hypothesis exploration |
| STORM | High | Perspective-parallel | Multi-expert critique | $$$ | Comprehensive articles (see [Section 7](#7-research-agent-reference-architectures)) |
| LLM Compiler | Medium | Full DAG | None (add separately) | $$ | Parallel multi-source search |

---

## 7. Research-Agent Reference Architectures

### STORM (Stanford)

**Paper**: "Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models" (2024)
**Architecture**: Multi-perspective expert simulation

STORM generates comprehensive, Wikipedia-quality articles through a distinctive approach:

1. **Perspective discovery**: given a topic, STORM identifies multiple expert perspectives that should be consulted (e.g., for "climate change" — a climate scientist, an economist, a policy analyst).
2. **Simulated interviews**: for each perspective, the system simulates a conversation between a "writer" and the "expert", with the expert grounded in retrieved sources.
3. **Outline generation**: synthesises the multi-perspective interviews into a structured article outline.
4. **Article generation**: produces the final article with inline citations.

STORM's key insight is that **diverse expert perspectives produce more comprehensive coverage** than a single agent searching iteratively. This directly addresses the confirmation bias problem common in single-agent research systems.

**Co-STORM** extends STORM with human-in-the-loop: a human collaborator can inject questions, redirect the conversation, and approve or challenge the simulated experts' claims during the interview phase. This is valuable for domain-specific research where the human brings expertise the LLM lacks.

### VMAO (Verified Multi-Agent Orchestration)

**Paper**: Zhang, X. et al. "Verified Multi-Agent Orchestration: A Plan-Execute-Verify-Replan Framework for Complex Query Resolution" — arXiv:2603.11445, March 2026
**Architecture**: DAG-based task decomposition with verification loops

VMAO operates through five sequential phases that form an iterative loop:

1. **Plan**: a QueryPlanner decomposes the incoming query into sub-questions, each assigned to a specialised agent type, with explicit dependencies represented as a DAG.
2. **Execute**: a DAGExecutor runs sub-questions through domain-specific agents in parallel "waves", processing the top-k ready tasks (those whose dependencies are met) concurrently.
3. **Verify**: a ResultVerifier (using a separate, more capable LLM) evaluates whether results adequately address the original query, scoring completeness 0–1 and flagging contradictions.
4. **Replan**: if verification identifies gaps, an AdaptiveReplanner generates corrective actions — retries of incomplete sub-questions or new queries targeting missing aspects.
5. **Synthesise**: once stop conditions are met (>=80% completeness, diminishing returns <5% improvement, token budget, or max iterations), results are merged into a coherent answer with source attribution.

Agents are organised into three tiers: data gathering (RAG, web search, financial, competitor), analysis (reasoning, raw data), and output (document, visualisation) — spanning 8 MCP servers exposing 42 tools.

**Benchmarks** (25 expert-curated market research queries, 1–5 scale):

| Method | Completeness | Source Quality | Avg Tokens | Avg Time (s) |
|--------|-------------|---------------|-----------|-------------|
| Single-Agent | 3.1 | 2.6 | 100K | 165 |
| Static Pipeline | 3.5 | 3.2 | 350K | 420 |
| VMAO | **4.2** | **4.1** | 850K | 900 |

VMAO achieves +35% completeness and +58% source quality over single-agent, at ~8.5x the token cost. In experiments, 75% of queries terminated via resource-based stop conditions (diminishing returns, max iterations, or token budget) rather than completeness thresholds, suggesting that most research queries are hard to fully resolve.

**Implementation stack**: LangGraph (workflow orchestration), Strands Agent framework (agent execution), AWS Bedrock (model hosting), MCP (tool integration).

**Limitations**: LLM-based verification evaluates completeness, not factual accuracy. Poor initial decomposition can propagate errors. The 8.5x token cost may be prohibitive for latency-sensitive settings.

### Router + Specialist Pattern

A practical architecture used in production research systems:

1. **Router agent**: classifies incoming queries and routes them to appropriate specialist agents.
2. **Specialist agents**: domain-specific agents with tailored prompts, tools, and knowledge bases.
3. **Aggregator**: merges specialist outputs into a coherent response.

This pattern maps naturally to AWS Agent Squad (for routing) combined with LangGraph or CrewAI (for specialist agent logic). It scales well as new research domains are added — just add a new specialist agent without modifying the router.

---

## 8. Evaluation Frameworks

Evaluating research agents is harder than evaluating single-turn LLM outputs because you must assess both the process (did the agent search effectively?) and the product (is the final report accurate and complete?).

### DeepEval

**Licence**: Apache 2.0
**Language**: Python (pytest-based)

DeepEval provides a pytest-like interface for evaluating agent traces across three layers:

**Reasoning layer**:
- `PlanQualityMetric`: whether the agent's plan is logical, complete, and efficient for the task.
- `PlanAdherenceMetric`: whether the agent actually follows its own stated plan during execution.

**Action layer**:
- `ToolCorrectnessMetric`: whether the agent selects the right tools (`correctness = correctly_used / total_called`). Supports configurable strictness: name matching, parameter matching, output matching, or exact matching.
- `ArgumentCorrectnessMetric`: whether tool arguments are correct (LLM-judged, referenceless — no expected values needed).

**Execution layer**:
- `TaskCompletionMetric`: the ultimate measure — did the agent accomplish the intended task?
- `StepEfficiencyMetric`: whether the agent completed the task without redundant steps, penalising unnecessary tool calls and reasoning loops.

All metrics operate on **execution traces** captured via the `@observe` decorator, analysing the full record of reasoning and actions rather than just final output. This is critical for research agents, where the process (search strategy, self-correction, source selection) matters as much as the product.

Additional features include G-Eval (LLM-as-judge with chain-of-thought), custom metrics for domain-specific criteria, and standard benchmark suites (TruthfulQA, HumanEval). The pytest integration makes it natural to incorporate into CI/CD workflows.

DeepEval is currently the most comprehensive open-source option for evaluating research agent pipelines.

### RAGAS

**Licence**: Apache 2.0
**Language**: Python

Originally focused on RAG evaluation, RAGAS has expanded to support agent evaluation:

- **RAG metrics**: context precision, context recall, faithfulness, answer relevancy.
- **Agent metrics**: topic adherence, tool call accuracy, agent goal accuracy.
- **Automated test generation**: can generate evaluation datasets from your documents.
- **Framework integrations**: works with LangChain, LlamaIndex, and custom pipelines.

RAGAS is particularly strong for evaluating the retrieval and grounding aspects of research agents — did the agent find the right sources and use them faithfully?

### Inspect (UK AI Safety Institute)

**Licence**: MIT
**Language**: Python

Inspect is designed for rigorous AI evaluation, originally built for safety assessments but applicable to agent evaluation:

- **Pipeline architecture**: Dataset -> Task -> Solver -> Scorer, providing clean separation of concerns.
- **Agent-native**: built-in support for evaluating tool-using agents.
- **Solver composition**: chain together agent reasoning steps for evaluation.
- **Reproducibility**: deterministic evaluation runs with full logging.
- **Multi-model**: evaluate the same agent architecture across different underlying models.

Inspect is particularly valuable for **comparing framework choices** — define your research task as an Inspect evaluation, then swap out the underlying agent framework to measure which performs best for your specific use case.

### Evaluation Strategy for Research Agents

A practical evaluation approach combines these tools:

1. **Unit-level** (RAGAS): Evaluate individual retrieval and grounding quality.
2. **Trace-level** (DeepEval): Evaluate complete research workflows — did the agent follow appropriate search strategies, self-correct, and produce faithful output?
3. **Comparative** (Inspect): Benchmark different framework/model combinations against the same research tasks.
4. **Human evaluation**: No automated metric fully captures research quality. Include human review for a sample of outputs, assessing completeness, accuracy, and usefulness.

---

## 9. State Management and Memory

Research agents must manage state across long-running sessions and, sometimes, across multiple sessions on related topics.

### Within-Session State

- **LangGraph checkpointing**: thread-scoped persistence using `MemorySaver` (in-memory), SQLite, or PostgreSQL backends. Enables pause/resume and time-travel debugging. This is the most mature open-source solution for within-session state.
- **CrewAI memory**: three-tier memory (short-term, long-term, entity) with automatic management. Less granular than LangGraph but easier to set up.
- **Custom state**: for simpler agents (e.g., built on SmolAgents or vendor SDKs), a simple state dictionary persisted to disk or a database often suffices.

### Cross-Session Memory

- **Vector stores**: store embeddings of previous research findings for retrieval in future sessions. Useful for building cumulative knowledge.
- **Structured knowledge bases**: maintain a structured store of verified claims, sources, and their relationships. More complex to build but enables richer cross-session reasoning.
- **Context summarisation**: for long-running sessions that approach context limits, summarise earlier conversation turns and inject the summary as context. LangGraph supports this pattern natively.

### Harness Patterns for Long-Running Agents

Anthropic's engineering team has documented patterns for agents that span multiple context windows — directly relevant to research sessions that exceed a single context:

- **Shift handover metaphor**: treat each context window as a "shift" — the agent must leave a clear handover for the next. Use a persistent progress file and descriptive git commits as the primary state mechanism.
- **Two-prompt architecture**: use a different prompt for the first session (initialiser — sets up environment, creates state files) versus subsequent sessions (worker — reads progress, does incremental work, updates state).
- **Structured state files over prose**: use JSON for structured data (feature lists, claim tracking) rather than Markdown — models are less likely to corrupt or inappropriately modify JSON.
- **Smoke test at session start**: run a quick validation before starting new work to catch regressions from prior sessions.
- **Incremental, completable units**: break research into discrete units (one research question per session) rather than attempting everything at once. Agents that try to "one-shot" a complex research task run out of context mid-way and leave work half-finished.
- **Compaction alone is insufficient**: even with automatic context compaction (summarising prior context), agents need explicit external state artefacts (files, databases) to maintain coherence across windows.

### Context Window Management

Research agents face a fundamental tension: they need to accumulate lots of information (sources, claims, critiques) but LLM context windows are finite. Strategies:

- **Progressive summarisation**: summarise completed research threads, keeping only key findings and source references in the active context.
- **External state stores**: track claims, sources, and their relationships in an external store (database, MCP server), querying as needed rather than keeping everything in context.
- **Hierarchical agents**: use a manager agent with a high-level view that delegates to worker agents with focused contexts.

---

## 10. Choosing a Framework — Decision Guide

### Decision Matrix

| Requirement | LangGraph | CrewAI | MS Agent Framework | SmolAgents | Vendor SDK |
|------------|-----------|--------|-------------------|------------|------------|
| Fine-grained control flow | Excellent | Limited | Good | Minimal | Minimal |
| Multi-agent roles | Good | Excellent | Excellent | Basic | Basic |
| Checkpointing/persistence | Built-in | Built-in | Partial | Manual | Manual |
| Open-model support | Yes | Yes | Yes | Excellent | Model-specific |
| Learning curve | Steep | Moderate | Steep | Low | Low |
| Production readiness | High | Moderate | High | Low-Moderate | High |
| MCP integration | Via LangChain | Partial | Via plugins | Native | SDK-specific |

### Recommendations by Use Case

**Simple research assistant** (single agent, search + synthesise):
- Start with a **vendor Agent SDK** (Claude Agent SDK if using Claude, OpenAI Agents SDK if using OpenAI) + MCP for tools.
- Graduate to **SmolAgents** if you need open models or code-as-action.

**Multi-perspective research system** (STORM-style):
- **LangGraph** for the orchestration graph + MCP for tool integration.
- Consider **CrewAI** if the role-based abstraction maps naturally to your research team design.

**Enterprise research platform**:
- **Microsoft Agent Framework** if in the Azure ecosystem.
- **LangGraph** + hyperscaler managed services (Bedrock Agents, Vertex AI) for cloud-native deployment.
- Invest in **evaluation** (DeepEval + Inspect) from day one.

**Experimental / prototyping**:
- **SmolAgents** for minimum overhead.
- **CrewAI** for quickly testing multi-agent designs.

### The Pragmatic View

Framework choice matters less than three other decisions:

1. **Prompt engineering**: How you instruct your agents to reason, critique, and synthesise determines research quality far more than which orchestration library wraps the LLM calls.
2. **Tool design**: The quality, reliability, and coverage of your research tools (search APIs, document parsers, databases) is the primary constraint on what your agent can discover.
3. **Evaluation strategy**: Without rigorous evaluation, you cannot know whether your research agent is actually producing accurate, comprehensive results — regardless of framework.

A well-designed research agent built on a simple ReAct loop with good tools and thorough evaluation will outperform a poorly-designed agent built on the most sophisticated orchestration framework.

---

## 11. Anti-Patterns

- **Framework tourism**: switching frameworks before understanding why the current one is insufficient. Each migration carries significant rewrite cost.
- **Over-agentification**: using multi-agent architectures when a single agent with good tools would suffice. Each agent boundary adds latency, cost, and debugging complexity.
- **Ignoring evaluation**: building increasingly complex agent pipelines without measuring whether they actually improve research quality.
- **Context stuffing**: dumping all retrieved content into the LLM context instead of summarising and selecting. This degrades reasoning quality and increases cost.
- **Blind tool trust**: assuming tool outputs (search results, API responses) are correct without verification. Research agents must treat all tool outputs as potentially incomplete or incorrect.
- **Protocol premature adoption**: implementing A2A or ANP before you have a working single-agent system with MCP. Get the basics right first.

---

## 12. Critical Perspective

The research agent framework space is immature and fast-moving. Several caveats are worth noting:

- **Most benchmarks are artificial**: framework comparison benchmarks typically use simple, well-defined tasks. Real research questions are messy, open-ended, and domain-specific. Benchmark performance may not predict real-world research quality. Marketing claims like "10,000x faster" are often based on agent instantiation time (microseconds), which is <0.01% of total execution time.
- **Framework overhead is real**: abstractions can add 20–50% to compute costs. One consulting engagement reported saving $3,000/month on AWS after migrating from LangChain to a leaner solution. Another reported $400+/day in debugging costs due to framework opacity. These are anecdotal but directionally consistent.
- **Custom solutions often win at scale**: organisations with high-volume research needs frequently find that custom-built orchestration — tailored to their specific data sources, quality requirements, and domain constraints — outperforms general-purpose frameworks. One case study reported a 10x throughput improvement (10K to 100K requests/day on the same infrastructure) after migrating from LangChain to LangGraph with custom optimisations. The frameworks are most valuable for getting started quickly and for medium-scale deployments.
- **The LLM is the bottleneck**: framework sophistication cannot compensate for limitations in the underlying LLM's reasoning ability. A research agent is only as good as the model's ability to formulate queries, assess source credibility, identify contradictions, and synthesise coherent narratives.
- **Cost scales non-linearly**: the VMAO benchmarks illustrate this concretely — 8.5x more tokens for a 35% completeness improvement. Multi-agent architectures, verification loops, and self-critique all multiply LLM calls. Design with cost awareness and consider which steps truly need LLM reasoning versus deterministic logic. Use traditional automation (Airflow, n8n) for deterministic sub-tasks rather than routing everything through an LLM.
- **Framework convergence is underway**: OpenAI is adding more structure to its SDK, Google ADK is becoming more flexible, and LangGraph is simplifying its API. The differences between frameworks are shrinking, which further supports the argument that prompt engineering, tool design, and evaluation matter more than framework choice.

---

## References

1. LangGraph Documentation — [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
2. CrewAI Documentation — [https://docs.crewai.com/](https://docs.crewai.com/)
3. Microsoft Agent Framework (AutoGen) — [https://github.com/microsoft/autogen](https://github.com/microsoft/autogen)
4. SmolAgents (Hugging Face) — [https://huggingface.co/docs/smolagents/](https://huggingface.co/docs/smolagents/)
5. AWS Agent Squad — [https://github.com/awslabs/agent-squad](https://github.com/awslabs/agent-squad)
6. Claude Agent SDK — [https://platform.claude.com/docs/en/agent-sdk/overview](https://platform.claude.com/docs/en/agent-sdk/overview)
7. OpenAI Agents SDK — [https://openai.github.io/openai-agents-python/](https://openai.github.io/openai-agents-python/)
8. Google Agent Development Kit — [https://google.github.io/adk-docs/](https://google.github.io/adk-docs/)
9. Model Context Protocol Specification — [https://spec.modelcontextprotocol.io/](https://spec.modelcontextprotocol.io/)
10. Agent Communication Protocol (ACP) — [https://agentcommunicationprotocol.dev/](https://agentcommunicationprotocol.dev/)
11. Agent-to-Agent Protocol (A2A) — [https://google.github.io/A2A/](https://google.github.io/A2A/)
12. Shtyrlin, K. et al. "A Survey of Agent Interoperability Protocols: Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent-to-Agent Protocol (A2A), and Agent Network Protocol (ANP)" — arXiv, 2025
13. STORM: Shao, Y. et al. "Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models" — Stanford, 2024. [https://storm.genie.stanford.edu/](https://storm.genie.stanford.edu/)
14. DeepEval Documentation — [https://docs.confident-ai.com/](https://docs.confident-ai.com/)
15. RAGAS Documentation — [https://docs.ragas.io/](https://docs.ragas.io/)
16. Inspect (UK AI Safety Institute) — [https://inspect.ai-safety-institute.org.uk/](https://inspect.ai-safety-institute.org.uk/)
17. Yao, S. et al. "ReAct: Synergizing Reasoning and Acting in Language Models" — ICLR 2023
18. Shinn, N. et al. "Reflexion: Language Agents with Verbal Reinforcement Learning" — NeurIPS 2023
19. Yao, S. et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" — NeurIPS 2023
20. Zhou, A. et al. "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models" — ICML 2024
21. Kim, S. et al. "An LLM Compiler for Parallel Function Calling" — 2024
22. Azure AI Foundry — [https://azure.microsoft.com/en-us/products/ai-foundry](https://azure.microsoft.com/en-us/products/ai-foundry)
23. Amazon Bedrock Agents — [https://aws.amazon.com/bedrock/agents/](https://aws.amazon.com/bedrock/agents/)
24. Google Vertex AI Agent Builder — [https://cloud.google.com/products/agent-builder](https://cloud.google.com/products/agent-builder)
25. Zhang, X. et al. "Verified Multi-Agent Orchestration: A Plan-Execute-Verify-Replan Framework for Complex Query Resolution" — arXiv:2603.11445, March 2026
26. Young, J. "Effective Harnesses for Long-Running Agents" — Anthropic Engineering Blog, November 2025. [https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
27. DeepEval Agent Evaluation Metrics — [https://deepeval.com/guides/guides-ai-agent-evaluation-metrics](https://deepeval.com/guides/guides-ai-agent-evaluation-metrics)
28. Agent Patterns Documentation — [https://agent-patterns.readthedocs.io/](https://agent-patterns.readthedocs.io/)
29. A2A and MCP Comparison — [https://agent2agent.info/docs/topics/a2a-and-mcp/](https://agent2agent.info/docs/topics/a2a-and-mcp/)
30. Roy, A. "AI Agent Frameworks: The Honest Comparison Nobody Talks About" — aankitroy.com, September 2025
31. Vicente, F. "Architecting AI Agent Systems: A Strategic Framework for Production Deployment" — dypsis.ai, November 2025
