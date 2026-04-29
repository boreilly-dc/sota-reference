# Alternatives to MCP: Tool Calling Strategies for LLM Integration (2025-2026)

| Field | Value |
|-------|-------|
| Created | 2026-03-30 |
| Last Updated | 2026-03-30 |
| Version | 1.1 |

---

- [Introduction](#introduction)
- [The Tool Integration Stack](#the-tool-integration-stack)
- [MCP: Current State and Adoption](#mcp-current-state-and-adoption)
- [Native Function Calling](#native-function-calling)
- [Competing Protocols](#competing-protocols)
- [Agent Frameworks with Built-in Tool Integration](#agent-frameworks-with-built-in-tool-integration)
- [Hyperscaler Managed Agent Services](#hyperscaler-managed-agent-services)
- [The "No Protocol" Baseline](#the-no-protocol-baseline)
- [MCP's Practical Problems](#mcps-practical-problems)
- [Security Considerations](#security-considerations)
- [Decision Framework](#decision-framework)
- [Areas of Uncertainty](#areas-of-uncertainty)
- [References](#references)

## Introduction

The Model Context Protocol (MCP), introduced by Anthropic in late 2024 and donated to the Linux Foundation's Agentic AI Foundation (AAIF) in December 2025, has become the dominant standard for connecting LLMs to external tools and data sources. By early 2026, MCP has reached 97 million monthly SDK downloads and 79,000 GitHub stars on its servers repository.

However, MCP is not the only option for LLM tool integration, and it is not always the right one. This article surveys the practical alternatives available in 2025-2026, spanning native function calling APIs, competing protocols (A2A, UTCP, ACP), agent frameworks, hyperscaler managed services, and the underrated strategy of simply not using a protocol at all.

The key insight from this research is that MCP, function calling, and REST APIs operate at **different layers of the stack** and are often complementary rather than competing. The real question is not "which protocol should I use?" but "how much protocol do I actually need?"

## The Tool Integration Stack

Tool integration for LLMs operates at three distinct layers:

| Layer | Technology | What It Does |
|-------|-----------|--------------|
| Transport | REST APIs, HTTP, gRPC | Raw data exchange between services |
| Model Capability | Function calling (OpenAI, Anthropic, Google) | LLM outputs structured JSON to request tool execution |
| Application Protocol | MCP, UTCP, A2A | Standardised tool discovery, auth, and execution across clients |

Understanding this layering is critical: choosing MCP does not mean abandoning function calling or REST APIs. MCP *uses* both under the hood. The question is whether the abstraction layer MCP adds is worth the complexity for your use case.

## MCP: Current State and Adoption

### Adoption Numbers (as of Q1 2026)

| Metric | Value | Source |
|--------|-------|--------|
| Monthly SDK downloads | 97 million | Multiple industry reports |
| GitHub stars (servers repo) | 79,000 | GitHub |
| Company-operated MCP servers | 1,412 | Bloomberry analysis of 2M companies |
| Growth rate (6 months) | 232% (425 to 1,412) | Bloomberry |
| Share of API subdomains | < 1% (vs 151,000 API subdomains) | Bloomberry |
| Avg tools per server | 1-4 | Bloomberry |
| Monthly new servers (Feb 2026) | 301 | Bloomberry |
| Total active servers (broader ecosystem) | 10,000+ | Ooty/PulseMCP/MCP.so |
| Remote server growth (May-Oct 2025) | ~4x | Ooty |

MCP adoption is real and accelerating, but remains early-stage. The 1,412 figure counts only company-operated servers scanned by Bloomberry; the broader ecosystem (community, open-source, and experimental servers) exceeds 10,000 — PulseMCP indexes over 8,600 and MCP.so lists over 17,000. Regardless, this represents less than 1% of the 151,000 companies with dedicated API subdomains. Adoption is led by smaller companies (81% have fewer than 200 employees) and B2B companies (70%).

All four major AI platforms adopted MCP within 14 months of its November 2024 launch: OpenAI (March 2025), Google DeepMind (April 2025), and Microsoft (May 2025), followed by the Linux Foundation donation in December 2025.

### Performance

MCP's latency overhead is negligible for most AI agent use cases:

| Setup | Added Latency | Source |
|-------|--------------|--------|
| Local MCP server (stdio) | < 1ms per tool call | Quickchat AI |
| Remote MCP server (same region) | 1-5ms per tool call | Quickchat AI |
| Remote MCP server (different region) | 10-50ms per tool call | Quickchat AI |
| Under load (JSON-RPC serialisation) | ~10-15% overhead | Microsoft Azure Architecture Blog |

Since LLM inference takes 500ms-5s, MCP overhead is typically dwarfed by the model and backend API latency. An independent benchmark found that MCP adds ~2x tool definition tokens due to richer schemas, but the impact on total cost and latency is modest. The recommendation: choose your integration approach based on architecture (same-process vs cross-boundary), not performance.

## Native Function Calling

Every major LLM provider offers native function/tool calling:

- **OpenAI**: Function calling in Chat Completions, built-in tools in Responses API (web search, file search, code interpreter, image generation), HostedMCPTool in Agents SDK
- **Anthropic**: Tool use API with JSON Schema definitions
- **Google**: Function calling in Gemini API, integrated with Vertex AI

### How It Works

The model receives tool definitions as JSON schemas, generates structured arguments when it wants to call a tool, and your code executes the call and returns the result. This is a **model capability**, not a protocol — the LLM proposes actions, your runtime executes them.

### Advantages

- **Simplicity**: No additional infrastructure (no servers, no protocol layer)
- **Direct control**: Full control over request construction, auth, error handling
- **Zero overhead**: No proxy, no serialisation layer, no additional network hops
- **Familiar**: Standard HTTP/REST patterns that any developer knows

### Limitations

- **Provider-locked**: Tool definitions must match each provider's format (though the formats are converging)
- **Static tool definitions**: All tools sent in every request, creating a "context tax"
- **No dynamic discovery**: Tools are hardcoded at development time
- **No credential isolation**: Your application code manages all auth

### OpenAI's Evolution

OpenAI's approach is noteworthy because it demonstrates convergence. Their Agents SDK now supports five categories of tools:

1. **Hosted OpenAI tools** — WebSearch, FileSearch, CodeInterpreter, ImageGeneration, **HostedMCPTool**
2. **Local/runtime tools** — ComputerTool, ShellTool, ApplyPatchTool
3. **Function calling** — wrap any Python function as a tool
4. **Agents as tools** — expose an agent as a callable tool
5. **Codex tool** — workspace-scoped code execution

The inclusion of `HostedMCPTool` shows that even OpenAI has embraced MCP rather than building a completely separate tool integration standard. Their `ToolSearchTool` addresses MCP's context overload problem by deferring tool loading until the model actually needs them.

## Competing Protocols

### Google Agent-to-Agent Protocol (A2A)

Launched April 2025 with 50+ technology partners (Atlassian, Salesforce, PayPal, SAP, ServiceNow). A2A is explicitly designed to **complement** MCP, not replace it:

- **MCP**: How an agent connects to tools and data sources (tool-to-agent)
- **A2A**: How agents communicate with each other (agent-to-agent)

A2A is built on existing standards (HTTP, SSE, JSON-RPC) and uses "Agent Cards" for capability discovery. It enables cross-vendor agent collaboration and focuses on task delegation, status tracking, and result aggregation between autonomous agents.

**When to use A2A**: Multi-agent systems where different agents (potentially from different vendors) need to collaborate on complex workflows.

**When to skip A2A**: Single-agent applications, simple tool calling, or environments where all agents are within the same framework.

### Universal Tool Calling Protocol (UTCP)

Emerged mid-2025 as a direct architectural alternative to MCP. The core difference: UTCP describes how to call existing tools at their **native endpoints** (HTTP, gRPC, WebSocket, CLI) without proxying through a dedicated MCP server.

- **Open source**: Implementations in Python, TypeScript, Go
- **Integrations**: LangChain/LangGraph, AWS Strands (strands-utcp, Jan 2026)
- **Design philosophy**: After discovery, the agent speaks directly to the tool — no wrapper server

**Adoption status**: Partially verified. UTCP exists with real implementations and PyPI packages, but adoption data is sparse compared to MCP. No GitHub star counts or production deployment reports were found during this research. It remains unclear whether UTCP has meaningful traction beyond its initial community.

**When to consider UTCP**: If you have existing HTTP/gRPC services that you want to expose to agents without building dedicated MCP servers, or if you want to avoid the proxy server overhead in latency-critical scenarios.

### IBM Agent Communication Protocol (ACP)

Published by the Linux Foundation with backing from IBM, Cisco, and Red Hat. ACP is positioned as an enterprise-focused protocol for multi-agent orchestration, specifically targeting commerce, compliance, and audit trails for regulated industries. It defines how agents negotiate service terms, process micropayments, and confirm transaction completion — capabilities absent from MCP.

**Adoption status**: The least mature of the three main protocols (MCP, A2A, ACP), with rare production deployments as of Q1 2026. DeepLearning.AI has created a dedicated course on ACP, suggesting growing educational interest.

### Other Protocols

The protocol landscape is fragmenting. Additional protocols announced in 2025-2026 include:

- **Google Agent Gateway Protocol (AGP)** — gateway-level agent routing
- **Cisco AGNTCY** — network-infrastructure-focused agent protocol
- **Zed Agent Client Protocol** — IDE-focused agent communication
- **ANP (Agent Network Protocol)** — decentralised agent networking
- **Agents.json** — stateless, OpenAPI-based spec where the agent manages all context (no MCP server)

Most of these have limited public adoption data. The historical pattern with competing standards (cf. XMPP vs proprietary messaging, RSS vs Atom) suggests that many will be abandoned or absorbed within 12-18 months. MCP's head start, Linux Foundation backing, and integration into OpenAI, Google, and Microsoft SDKs give it a significant moat.

## Agent Frameworks with Built-in Tool Integration

Agent frameworks provide their own tool integration patterns that can work **with or without** MCP. They operate at a higher level of abstraction — orchestrating agent behaviour, managing state, and coordinating multi-step workflows.

### LangGraph (LangChain)

LangGraph replaced LangChain's legacy agent abstraction as the recommended approach. It uses a state-machine model for agent orchestration with built-in persistence, checkpointing, and human-in-the-loop support. Tools are defined as Python functions decorated with `@tool`.

- **Best for**: Complex, stateful agent workflows with branching logic
- **Tool integration**: Native function tools + MCP via `langchain-mcp-adapters`
- **Open source**: Yes (MIT)

### CrewAI

Role-based multi-agent framework with the best developer experience for team-of-agents patterns. Agents have defined roles, goals, and tools, and are organised into Crews working on Tasks.

- **Best for**: Multi-agent collaboration with clear role separation
- **Tool integration**: Built-in tool decorator + MCP support
- **Open source**: Yes

### Microsoft AutoGen

Research-oriented framework from Microsoft Research, excelling at code generation and execution with Docker sandboxing. Agents can spin up containers, write code, run it, see errors, and iterate automatically.

- **Best for**: Code generation, data science workflows, research
- **Tool integration**: Function-based tools + Semantic Kernel integration
- **Open source**: Yes (MIT)

### Microsoft Semantic Kernel

Enterprise-focused SDK for integrating LLMs into .NET, Python, and Java applications. Strong Azure ecosystem integration, enterprise governance features, and plugin architecture.

- **Best for**: Enterprise .NET/Java environments with Azure
- **Tool integration**: Plugin system, OpenAPI spec ingestion, MCP support
- **Open source**: Yes (MIT)

### OpenAI Swarm

Intentionally minimal framework — agents are just instructions + functions, handoffs are function calls returning other agents. Beautifully simple but explicitly "educational" and not production-grade.

- **Best for**: Prototyping, learning agent concepts
- **Tool integration**: Python functions only, OpenAI-specific
- **Open source**: Yes

### Other Notable Frameworks

The framework landscape has consolidated from the 2023-2024 explosion, but several additional options are worth noting:

- **Google Agent Development Kit (ADK)** — Google-ecosystem-aligned, open-source, supports both A2A and MCP
- **Pydantic AI** — type-safe tool definitions using Pydantic models, popular with Python developers preferring minimal abstraction
- **AWS Strands Agents** — open-source SDK with UTCP integration, tightly coupled with Bedrock
- **Agno** and **Mastra** — newer entrants with growing communities

A Langfuse comparison identified 12+ major open-source agent frameworks as of early 2026, each with distinct tool registration patterns.

### Key Insight

These frameworks are **complementary to MCP**, not alternatives to it. MCP standardises how tools are discovered and invoked; frameworks orchestrate how agents use those tools. In practice, most frameworks now support MCP as one of their tool integration options.

## Hyperscaler Managed Agent Services

The major cloud providers offer fully managed agent services with built-in tool calling, providing an enterprise alternative to self-hosting MCP infrastructure.

### AWS

- **Bedrock Agents**: Managed agent orchestration with action groups (Lambda functions, API schemas)
- **AWS Strands**: Open-source agent SDK with UTCP integration
- **MCP support**: Bedrock supports MCP tool servers

### Azure

- **Azure AI Foundry**: Managed agent platform with tool integration
- **Semantic Kernel**: SDK with plugin architecture and Azure-native tools
- **MCP support**: Azure AI Agent Service supports MCP

### GCP

- **Vertex AI Agent Builder**: Managed agent platform
- **Agent Development Kit (ADK)**: Open-source SDK for building agents
- **A2A + MCP**: GCP supports both protocols, positioning A2A for inter-agent and MCP for tool integration

All three hyperscalers have converged on supporting MCP alongside their native tool integration patterns. The managed services are most valuable when you want minimal infrastructure management, built-in security/compliance, and tight integration with cloud-native services.

## The "No Protocol" Baseline

The most underrated alternative to MCP is **not using a protocol at all**. For many teams, straightforward function calling with custom wrappers is the right approach.

### When to Skip MCP

- **1-2 tool integrations**: Direct HTTP calls are simpler with fewer moving parts
- **Stable, predictable tool sets**: MCP's dynamic discovery adds no value if your tools don't change
- **Maximum control**: Direct API calls give full control over headers, auth, request construction
- **Latency-sensitive operations**: Every millisecond matters (e.g., real-time voice agents)
- **Non-AI HTTP calls**: Backend cron jobs, webhooks, system-to-system communication

As Qdrant noted: "Introducing MCP servers for context engineering makes sense only if we have enough knowledge that changes frequently, and when it is big enough that we prefer not to pass it on to each LLM call."

### The Bare SDK Approach

```python
# Direct function calling with Anthropic SDK — no MCP needed
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
]

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Auckland?"}]
)
```

This approach works, is debuggable, has zero infrastructure overhead, and is the dominant strategy in practice for simple agent applications.

## MCP's Practical Problems

### Context Overload

MCP's biggest practical problem in 2026 is **context overload**. Generic MCP servers are designed to cover as many use cases as possible, exposing all their tools to any connecting agent. A standard setup of Playwright, GitHub, and IDE MCP servers consumed over 20% of the context window before the agent even started working.

The rule of thumb is to stay below 40% total context usage. If tools alone consume half that budget, there is little room for files, instructions, and actual work. MCP adds ~2x tool definition tokens compared to minimal function calling schemas.

**Mitigations emerging in 2025-2026**:

- **Deferred tool loading**: Anthropic's tool discovery feature (late 2025) and OpenAI's `ToolSearchTool` load tools on demand
- **Sub-agents**: Isolate tool sets to specialised sub-agents that only carry the tools they need
- **Fine-grained tool selection**: Some clients (VS Code, Theia IDE) let users enable/disable individual functions from an MCP server
- **Tool namespaces**: OpenAI's namespace system groups related tools, loading entire namespaces on demand

### The Irony of MCP's Success

The two features that make MCP successful — broad, generic servers and easy integration — are also the features that contribute most to context overload. Server authors maximise coverage; users enable everything that looks useful; agents drown in irrelevant tool definitions.

## Security Considerations

MCP introduces distinct security attack vectors:

### Tool Poisoning

Malicious instructions embedded in tool metadata (descriptions, parameter schemas) that the AI model sees but users don't. A tool might present as "add_numbers" while secretly instructing the model to exfiltrate SSH keys. The MCPTox academic benchmark (arXiv, 2025) demonstrates this systematically.

### Prompt Injection via External Data

Indirect prompt injection becomes a distributed trust-boundary problem across tools, metadata, sessions, and external systems. The Supabase/Cursor breach (June 2025) combined privileged access, untrusted input, and an external communication channel.

### Vulnerability Taxonomy

Adversa AI has catalogued 25 distinct MCP vulnerability categories. The top 5 critical vulnerabilities:

| Vulnerability | Severity |
|--------------|----------|
| Prompt injection | Critical (10/10) |
| Command injection | Critical (10/10) |
| Remote code execution | Critical (10/10) |
| Tool poisoning (TPA) | Critical (9/10) |
| Unauthenticated access | Critical (9/10) |

Additional MCP-specific attack vectors include **rug pull attacks** (tool definitions change after initial user approval, so a tool approved as safe can later mutate its behaviour), **confused deputy attacks** via OAuth proxies, and **tool name spoofing** using homoglyph characters.

Equixly's independent security assessment of MCP implementations found command injection vulnerabilities in 43% of tested implementations, 30% vulnerable to SSRF, and 22% allowing arbitrary file access. Invariant Labs demonstrated a practical attack where a malicious MCP server exfiltrated WhatsApp message history through a legitimate server in the same agent environment.

### Real-World Security Posture

Bloomberry's analysis of 1,412 company-operated MCP servers found alarming security gaps:

| Issue | Prevalence |
|-------|-----------|
| No authentication | 38.7% |
| Wide-open CORS | 22.9% |
| Rate limiting in place | 2.4% |
| Leaked debug information | 3 servers |

Sensitive tools (financial transfers, KYC data, HR candidate lookups) were found on unauthenticated MCP servers. The MCP specification itself acknowledges the risk, recommending: "There SHOULD always be a human in the loop with the ability to deny tool invocations."

### Security Implications for Alternatives

- **Native function calling** reduces the attack surface (no tool metadata from untrusted servers) but doesn't eliminate prompt injection
- **UTCP's direct endpoint model** avoids the proxy server as an attack vector but shifts trust to the native endpoints
- **Managed hyperscaler services** generally have stronger auth, audit logging, and compliance controls than self-hosted MCP

## Decision Framework

Use this decision tree to choose your tool integration approach:

```
Is an AI agent calling the tool?
├── No → Use direct HTTP (standard backend code)
└── Yes
    ├── How many external tools?
    │   ├── 1-2 → Direct function calling (bare SDK)
    │   ├── 3-5 → Consider MCP if tools change frequently; otherwise function calling
    │   └── 5+ → MCP (reduces integration burden, enables portability)
    ├── Do agents need to talk to each other?
    │   ├── Yes → Add A2A for inter-agent communication
    │   └── No → MCP or function calling is sufficient
    ├── Enterprise/compliance requirements?
    │   ├── Yes → Hyperscaler managed services (Bedrock, Azure AI, Vertex AI)
    │   └── No → Self-hosted MCP or function calling
    ├── Need cross-platform portability?
    │   ├── Yes → MCP (works with Claude, ChatGPT, Cursor, etc.)
    │   └── No → Provider-specific function calling is fine
    └── Latency-critical (real-time voice, etc.)?
        ├── Yes → Direct HTTP + function calling
        └── No → MCP overhead is negligible
```

### Hybrid Approach (Most Common in Production)

Many production systems use both MCP and direct HTTP:

- **MCP** for tool integrations triggered by the AI agent during conversations (CRM lookups, ticket creation, knowledge base queries)
- **Direct HTTP** for internal backend operations, webhooks, and system-to-system communication

## Areas of Uncertainty

- **Protocol mortality**: It is unclear which of the 6+ announced protocols beyond MCP will survive 18 months. The historical pattern with competing standards suggests consolidation.
- **UTCP adoption trajectory**: UTCP has technical merit but adoption data is sparse. Whether it can challenge MCP's first-mover advantage is unknown.
- **Enterprise MCP security**: Whether the current security gaps (38.7% no-auth) represent growing pains or a fundamental protocol design issue is debated.
- **Context overload solutions**: Deferred tool loading is emerging but not yet mature. It remains to be seen whether it fully solves the problem or introduces new failure modes.
- **Non-Western ecosystem**: This analysis is anchored in the US/Anglophone tech ecosystem. Chinese AI frameworks (Dify, Coze, FastGPT) and their tool integration patterns were not covered and may present different alternatives.

## References

1. [6 Model Context Protocol alternatives to consider in 2026](https://www.merge.dev/blog/model-context-protocol-alternatives) - Merge.dev
2. [MCP vs Function Calling vs REST APIs: When to Use Each](https://mcpplaygroundonline.com/blog/mcp-vs-function-calling-vs-api-comparison) - MCP Playground
3. [Announcing the Agent2Agent Protocol (A2A)](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) - Google Developers Blog
4. [Function Calling vs. MCP vs. A2A: Developer's Guide](https://zilliz.com/blog/function-calling-vs-mcp-vs-a2a-developers-guide-to-ai-agent-protocols) - Zilliz
5. [MCP vs UTCP](https://nordicapis.com/model-context-protocol-mcp-vs-universal-tool-calling-protocol-utcp/) - Nordic APIs
6. [MCP security: Prompt injection and tool poisoning](https://securityboulevard.com/2026/01/mcp-security-how-to-prevent-prompt-injection-and-tool-poisoning-attacks/) - Security Boulevard
7. [Universal Tool Calling Protocol (UTCP)](https://github.com/universal-tool-calling-protocol) - GitHub
8. [OpenAI Agents SDK - Tools](https://openai.github.io/openai-agents-python/tools/) - OpenAI
9. [Best AI Agent Frameworks Compared (2026)](https://theoperatorcollective.org/blog/best-ai-agent-frameworks-compared) - The Operator Collective
10. [Agent-to-Agent Communication Protocol Standards](https://zylos.ai/research/2026-02-15-agent-to-agent-communication-protocols) - Zylos Research
11. [I analyzed 1400 MCP servers](https://bloomberry.com/blog/we-analyzed-1400-mcp-servers-heres-what-we-learned/) - Bloomberry
12. [MCP vs HTTP: When to Use Each](https://quickchat.ai/post/mcp-vs-http) - Quickchat AI
13. [Decision Matrix: API vs MCP Tools](https://techcommunity.microsoft.com/blog/azurearchitectureblog/decision-matrix-api-vs-mcp-tools-%E2%80%94-the-great-integration-showdown-%F0%9F%A5%8A/4499385) - Microsoft Tech Community
14. [MCP vs Direct Benchmark](https://github.com/odedha-dr/mcp-vs-direct-benchmark) - GitHub
15. [When Is MCP Actually Worth It?](https://thenewstack.io/when-is-mcp-actually-worth-it/) - The New Stack
16. [MCP and Context Overload](https://eclipsesource.com/blogs/2026/01/22/mcp-context-overload/) - EclipseSource
17. [MCP Security: TOP 25 MCP Vulnerabilities](https://adversa.ai/mcp-security-top-25-mcp-vulnerabilities/) - Adversa AI
18. [Securing AI's New Frontier: MCP Security](https://genai.owasp.org/2025/04/22/securing-ais-new-frontier-the-power-of-open-collaboration-on-mcp-security/) - OWASP GenAI Security Project
19. [MCP: Landscape, Security Threats, and Future Research Directions](https://arxiv.org/abs/2503.23278) - arXiv (academic paper)
20. [State of the MCP Ecosystem: 2026 Report](https://www.ooty.io/blog/state-of-mcp-ecosystem-2026) - Ooty
21. [AI Agent Protocols: MCP vs A2A vs ANP vs ACP](https://dev.to/dr_hernani_costa/ai-agent-protocols-mcp-vs-a2a-vs-anp-vs-acp-4k98) - Dr. Hernani Costa
22. [Comparing Open-Source AI Agent Frameworks](https://langfuse.com/blog/2025-03-19-ai-agent-comparison) - Langfuse
23. [Agentic AI Frameworks: Architectures, Protocols, and Design Challenges](https://arxiv.org/html/2508.10146v1) - arXiv (academic paper)
24. [MCP vs OpenAI Function Calling vs LangChain](https://docs.gostoa.dev/blog/mcp-vs-openai-function-calling-vs-langchain) - STOA Docs
25. [Plug, Play, and Prey: MCP Security Risks](https://techcommunity.microsoft.com/blog/microsoftdefendercloudblog/plug-play-and-prey-the-security-risks-of-the-model-context-protocol/4410829) - Microsoft Defender
