# Best Practices for Prompting Foundation Models in 2026

| Field | Value |
|-------|-------|
| Created | 2026-05-26 |
| Last Updated | 2026-05-26 |
| Version | 1.0 |

---

- [Executive Summary](#executive-summary)
- [Perspectives Analysed](#perspectives-analysed)
- [Key Findings](#key-findings)
- [Prompt Storage Patterns](#finding-5-three-patterns-for-prompt-storage-in-production)
- [Cloud-Specific Prompt Management](#finding-6-cloud-specific-prompt-management-2026-state)
- [Prompt Testing and CI/CD](#finding-7-prompt-testing-and-cicd)
- [Professional Services Patterns](#finding-8-professional-services-patterns)
- [Open-Source Tooling Landscape](#finding-9-open-source-tooling-landscape-2026)
- [Anti-Patterns to Avoid](#finding-10-anti-patterns-to-avoid-in-2026)
- [Areas of Uncertainty](#areas-of-uncertainty)
- [Caveats](#caveats)
- [References](#references)

## Executive Summary

The practice of prompting foundation models has undergone a paradigm shift in 2025–2026. The field has moved from "prompt engineering" — carefully crafting individual prompts — to **"context engineering"**: the systematic management of what information reaches the model, when, and in what structure. This shift is driven by three developments: (1) frontier models now perform chain-of-thought reasoning internally via API-controlled thinking parameters, making explicit "think step by step" instructions counterproductive; (2) structured output guarantees (JSON schemas, tool calling) have matured across all major providers; and (3) prompts have become production assets requiring the same version control, testing, and deployment rigour as application code.

For professional services companies operating across AWS, Azure, and GCP, the recommended architecture is a **two-tier prompt system**: a reusable core library of model-agnostic prompt templates versioned in Git, with cloud-specific adapter layers that handle provider-specific formatting, caching, and deployment. AWS Bedrock Prompt Management and GCP Vertex AI Prompt Registry are both GA and actively developed; Azure Prompt Flow is being retired (April 2027) and should be avoided for new work. Prompt testing should be automated via PromptFoo or DeepEval in CI/CD pipelines, with evaluation datasets maintained per client engagement.

The default recommendation for prompt storage is **file-based in Git alongside application code**, with deviation to database-backed registries only when non-engineering stakeholders need to modify prompts without code deployments, or when the number of prompts exceeds 10–20 per service.

---

## Perspectives Analysed

### 1. Framework Architect (Extensibility & Composability)
**Focus:** How to structure prompts as reusable, composable components across projects and clients.
**Key findings:** Template systems with variable injection (Jinja2, Mustache, or provider-native `{{variable}}` syntax) enable prompt reuse. The `.prompty` file format (Microsoft, open standard) provides a portable single-file prompt definition. Prompt registries become essential at 3–10+ distinct prompts per service.

### 2. Production Engineer (Reliability & Cost)
**Focus:** Caching, latency, deployment safety, rollback, and multi-cloud operational concerns.
**Key findings:** All three clouds support prompt caching with 50–75% cost savings. Feature flags and canary deployments prevent production incidents from prompt changes. Immutable prompt versions with git commit hash tracking enable precise reproducibility and rollback.

### 3. Academic/Researcher (Reasoning & Evaluation)
**Focus:** What the literature says about prompting effectiveness, evaluation methodology, and reasoning control.
**Key findings:** Extended thinking is now API-controlled, not prompt-controlled. LLM-as-judge is mainstream for evaluation. Multi-perspective prompting (STORM-style) outperforms single-viewpoint generation. Structured outputs are the single most impactful technique for production reliability.

### 4. Sceptic/Critic (Risks & Anti-patterns)
**Focus:** What to avoid, where the industry is wrong, and emerging failure modes.
**Key findings:** Over-engineering prompts is now counterproductive — frontier models perform worse with excessive instructions. Vendor lock-in through cloud-specific prompt management is a real risk. "Prompt engineering" as a distinct role is declining; the skill is being absorbed into general software engineering.

### 5. Professional Services Practitioner (Multi-tenant, Multi-cloud)
**Focus:** Patterns for consultancies delivering AI solutions to multiple clients across different cloud platforms.
**Key findings:** Two-tier IP architecture (core library + client configs) enables reuse without leaking client data. Tenant isolation must be enforced at the backend level, never relying on the LLM to segregate data. Prompt migration harnesses evaluate quality when moving between model versions or providers.

---

## Key Findings

### Finding 1: "Context Engineering" Has Replaced "Prompt Engineering"

**Confidence:** High
**Sources:** [1], [5], [8]
**Perspectives:** Framework Architect, Researcher, Sceptic

The 2026 paradigm shift recognises that the prompt is just one component of the context window. Context engineering encompasses: what documents to retrieve (RAG), what tools to expose, how to structure system instructions, what conversation history to retain, and how to manage the thinking budget. The most impactful optimisations are now about _what information reaches the model_, not _how you phrase the request_.

**Practical implication:** Invest in retrieval quality, tool selection logic, and context window management over prompt wording finesse.

### Finding 2: Extended Thinking Is API-Controlled, Not Prompt-Controlled

**Confidence:** High
**Sources:** [1], [2], [8]
**Perspectives:** Researcher, Production Engineer

All frontier models now have internal reasoning capabilities controlled via API parameters:

| Model | Parameter | Values |
|---|---|---|
| Claude 4.x | `thinking.budget_tokens` | low / medium / high / max |
| GPT-5.x | `reasoning_effort` | none / low / medium / high / xhigh |
| Gemini 2.5/3 | `thinking_config` | off / low / medium / high |

Writing "think step by step" or "reason carefully" in the prompt is **counterproductive** — it wastes output tokens on visible reasoning text instead of leveraging the model's private thinking tokens, which are cheaper and more effective. For open-weight models (Llama, Qwen) that lack API-level thinking control, explicit reasoning prompts remain useful.

### Finding 3: Model-Specific Formatting Preferences

**Confidence:** High
**Sources:** [1], [2], [5]
**Perspectives:** Framework Architect, Researcher

Each model family responds best to different structural patterns:

**Claude (Anthropic):**
- XML tags for content structure: `<context>`, `<instructions>`, `<examples>`
- System prompt for role and constraints
- Long documents placed before the query (up to 30% improvement)
- Tell it what to do, not what to avoid
- 3–5 diverse examples in `<example>` tags

**GPT-5.x (OpenAI):**
- Markdown formatting (headers, lists, code blocks)
- Outcome-first prompts: state the desired result before the process
- Output contracts: explicit JSON schema or format specification
- Retrieval budgets: tell it how many sources to consult
- Verification loops: ask it to check its own work

**Gemini 2.5/3 (Google):**
- Few-shot examples are especially important (more so than other models)
- Data in tables/structured format
- Questions placed at the end of context (after all reference material)
- System instructions for behaviour control
- Deep Think benefits from parallel hypothesis framing

**Open-weight (Llama 4, Qwen 3, Mistral):**
- Still benefit from explicit CoT prompting (no internal thinking API)
- Chat templates vary by model — use the model's native template
- Structured output via constrained decoding (vLLM, SGLang) rather than prompt alone
- Context windows smaller — be more aggressive with compression

### Finding 4: Structured Outputs Are the Most Impactful Production Technique

**Confidence:** High
**Sources:** [1], [2], [4], [5]
**Perspectives:** Production Engineer, Framework Architect

Forcing structured output (JSON schema enforcement, function/tool calling) is now the single highest-ROI prompting technique for production systems. It eliminates parsing failures, enables type-safe integration, and allows downstream validation. All major providers support it:

- **Claude:** Tool use with input schemas, or `response_format: { type: "json_schema", ... }`
- **GPT-5.x:** Structured Outputs (guaranteed schema conformance), function calling
- **Gemini:** `response_schema` parameter with JSON schema
- **Open-weight:** Constrained decoding via vLLM/SGLang outlines, or Instructor library

### Finding 5: Three Patterns for Prompt Storage in Production

**Confidence:** High
**Sources:** [3], [4], [5], [6]
**Perspectives:** Framework Architect, Production Engineer, Professional Services

| Pattern | When to Use | Pros | Cons |
|---|---|---|---|
| **File-based in Git** | Default; <10 prompts; engineering-only editors | Full audit trail, same PR process as code, no extra infra | Requires deploy for changes |
| **Database-backed registry** | Non-engineers edit prompts; A/B testing; >10 prompts | Hot-swap without deploy, RBAC, audit | Extra infra, migration complexity |
| **SaaS platform** (Braintrust, Langfuse) | Rapid iteration; need built-in eval; small teams | Fast setup, built-in analytics | Vendor lock-in, cost at scale |

**Recommended file structure for Git-based storage:**

```
prompts/
├── system/
│   ├── assistant-v2.yaml          # System prompt
│   └── assistant-v2.test.yaml    # Test cases
├── tasks/
│   ├── summarise.yaml
│   ├── extract-entities.yaml
│   └── classify-intent.yaml
├── templates/
│   └── _base.yaml                 # Shared template fragments
└── prompt.config.yaml             # Model mappings, defaults
```

Each prompt file should contain: the template text, variable definitions, model compatibility notes, and a version identifier. Use Jinja2 or Mustache for variable interpolation; `{{variable}}` is near-universal across cloud providers.

### Finding 6: Cloud-Specific Prompt Management (2026 State)

**Confidence:** High
**Sources:** [7], [8], [9], [10]
**Perspectives:** Production Engineer, Professional Services

#### AWS Bedrock Prompt Management (Recommended — Active Development)
- **Status:** GA, no extra charge (pay only for model tokens)
- **Features:** Visual Prompt Builder, versioning, prompt variants for A/B comparison, `{{variable}}` placeholders, integration with Bedrock Flows and Agents
- **Caching:** Explicit cache checkpoints (up to 4 per request), 5min or 1hr TTL, Amazon Nova has automatic implicit caching
- **Best for:** Teams already on AWS; production workloads needing managed versioning

#### GCP Vertex AI Prompt Registry (Recommended — Active Development)
- **Status:** GA (October 2025)
- **Features:** Prompts as first-class managed GCP resources, programmatic CRUD, automatic versioning on every update, seamless Vertex AI Studio ↔ SDK experience
- **Enterprise:** CMEK encryption, VPC Service Controls
- **Caching:** Both implicit (automatic) and explicit, 75% discount for Gemini 2.0+
- **Best for:** Enterprise/regulated environments; GCP-native teams

#### Azure Prompt Flow (AVOID — Being Retired)
- **Status:** Feature development ENDED April 20, 2026; full retirement April 20, 2027
- **What to use instead:** Azure AI Foundry (successor), or cloud-agnostic tools (Langfuse, MLflow)
- **Caching:** Azure OpenAI automatic prompt caching still operational (50% discount, min 1,024 tokens)
- **Implication for professional services:** Do not build new client solutions on Prompt Flow. Migrate existing clients to Azure AI Foundry or a cloud-agnostic alternative.

### Finding 7: Prompt Testing and CI/CD

**Confidence:** High
**Sources:** [4], [5], [6]
**Perspectives:** Production Engineer, Framework Architect

**Three-layer testing model:**

1. **Syntax/schema validation** — Does the prompt template render correctly with all variable combinations? Does the expected output schema validate?
2. **Regression testing** — Given known inputs, does the model produce outputs that pass quality assertions? Use a golden dataset of 20–50 input/expected-output pairs.
3. **A/B comparison** — When changing a prompt, compare new vs. old on the same inputs. Block merge if quality drops below threshold.

**Recommended tools:**

| Tool | License | Strengths | Notes |
|---|---|---|---|
| **PromptFoo** | MIT | CI/CD-native, YAML configs, GitHub Actions, 50+ assertion types | Acquired by OpenAI March 2026; still MIT |
| **DeepEval** | Apache 2.0 | 50+ metrics, pytest integration, LLM-as-judge | Good for teams already on pytest |
| **Ragas** | Apache 2.0 | RAG-specific (9 dimensions), context relevance, faithfulness | Use alongside general tools for RAG |
| **Langfuse** | MIT | Tracing, scoring, prompt management, 27K GitHub stars | Acquired by ClickHouse Jan 2026 |

**CI/CD integration pattern:**

```yaml
# .github/workflows/prompt-eval.yml
on:
  pull_request:
    paths: ['prompts/**']
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npx promptfoo eval --config prompts/eval.yaml
      - run: npx promptfoo assert --threshold 0.85
```

### Finding 8: Professional Services Patterns

**Confidence:** High
**Sources:** [3], [4], [5], [6]
**Perspectives:** Professional Services, Framework Architect, Sceptic

#### Two-Tier IP Architecture

```
┌─────────────────────────────────────┐
│         Core Prompt Library          │  ← Company IP (reusable)
│  • Model-agnostic templates         │
│  • Standard task patterns           │
│  • Evaluation harnesses             │
│  • Cloud adapter layers             │
└──────────────┬──────────────────────┘
               │ inherits / overrides
┌──────────────▼──────────────────────┐
│      Client-Specific Configs         │  ← Client IP (isolated)
│  • Domain terminology overrides      │
│  • Tone/brand guidelines            │
│  • Client-specific examples         │
│  • Regulatory constraints           │
└─────────────────────────────────────┘
```

- Core library owned by the professional services company, version-controlled internally
- Client configs stored in client-specific repos with appropriate access controls
- Never share client examples or data across engagements
- Evaluation datasets are per-client (they contain client domain data)

#### Multi-Tenant Isolation Rules

1. **`tenant_id` in all cache keys** — Prevents cross-tenant cache poisoning
2. **Namespace-per-tenant in vector stores** — Physical or logical isolation depending on regulatory requirements
3. **Backend resolves tenant context** — The LLM never decides which tenant's data to access
4. **Tiered isolation:** Physical separation for regulated industries (healthcare, finance); logical separation for standard commercial
5. **Audit trail:** Log prompt version + tenant_id + model + timestamp for every inference call

#### Prompt Migration Evaluation Harness

When moving a client from one model to another (e.g., GPT-4o → Claude 4.6, or upgrading within a family):

1. Run the full evaluation dataset against both models
2. Compare on: accuracy, format compliance, latency, cost
3. Flag regressions > 5% on any metric
4. Provide client with comparison report before switching
5. Maintain rollback capability for 30 days post-migration

#### Cross-Cloud Abstraction Layer

```python
# Simplified prompt resolution pattern
class PromptResolver:
    def resolve(self, prompt_id: str, cloud: str, model: str, variables: dict) -> str:
        template = self.registry.get(prompt_id, version="latest")
        adapted = self.cloud_adapter[cloud].format(template, model)
        return adapted.render(**variables)
```

- Adapter layer translates between cloud-specific placeholder syntaxes (`{{var}}` for AWS, `{var}` for GCP, Jinja2 for Azure)
- Model-specific formatting applied at resolution time (XML tags for Claude, Markdown for GPT, etc.)
- Single source of truth for prompt content; cloud differences handled in infrastructure

### Finding 9: Open-Source Tooling Landscape (2026)

**Confidence:** High
**Sources:** [3], [5], [6]
**Perspectives:** Framework Architect, Professional Services

| Tool | Category | Stars | License | Key Feature |
|---|---|---|---|---|
| **Langfuse** | Observability + Prompts | 27K | MIT | Full LLMOps platform; prompt management, tracing, scoring |
| **PromptFoo** | Testing/Eval | 13K+ | MIT | CI/CD-native eval; assertion framework; red-teaming |
| **MLflow Prompt Registry** | Enterprise Registry | — | Apache 2.0 | Databricks Unity Catalog integration; enterprise governance |
| **Agenta** | Prompt Management | ~3K | MIT | Git-like versioning; playground; deployment |
| **Pezzo** | Prompt Management | ~2K | Apache 2.0 | Version control; analytics; caching |
| **DeepEval** | Evaluation | ~4K | Apache 2.0 | 50+ metrics; pytest native; LLM-as-judge |
| **Ragas** | RAG Evaluation | ~8K | Apache 2.0 | 9-dimension RAG quality; faithfulness; relevance |

**Recommended stack for professional services:**
- **Observability:** Langfuse (self-hosted for client isolation)
- **Testing:** PromptFoo in CI/CD + DeepEval for pytest-native teams
- **Prompt management:** Git-based (default) or Langfuse (when non-engineers need access)
- **RAG evaluation:** Ragas
- **Enterprise governance:** MLflow (if already on Databricks)

### Finding 10: Anti-Patterns to Avoid in 2026

**Confidence:** High
**Sources:** [1], [2], [5], [8]
**Perspectives:** Sceptic, Researcher

1. **"Think step by step" on frontier models** — Wastes tokens; reasoning is internal. Only use on open-weight models without thinking APIs.
2. **Over-specified process instructions** — "First do X, then do Y, then do Z" constrains the model. State the desired outcome and quality criteria instead.
3. **Prompts in environment variables** — Unversioned, untestable, invisible to PR review. Use only for secrets and model API keys.
4. **Hardcoded prompts in application code** — Acceptable for simple cases, but creates deployment coupling. Extract to files once you have more than 2–3 prompts.
5. **Vendor-specific prompt management without abstraction** — Lock-in risk, especially with Azure Prompt Flow retirement. Always have a portable fallback.
6. **Single-model prompt libraries** — Prompts optimised for one model degrade on others. Maintain model-agnostic templates with model-specific adapters.
7. **Testing prompts only with unit tests** — Prompts are stochastic; you need statistical evaluation over multiple runs, not binary pass/fail assertions.
8. **Sharing evaluation datasets across clients** — Leaks domain knowledge; violates data isolation principles.

---

## Areas of Uncertainty

- **Azure AI Foundry maturity:** As the successor to Prompt Flow, Azure AI Foundry's prompt management capabilities are still evolving. Specific feature parity timelines are unclear.
- **PromptFoo post-acquisition direction:** Following OpenAI's acquisition (March 2026), it remains MIT-licensed, but long-term governance and neutrality are uncertain.
- **Optimal prompt caching strategies across providers:** Cache invalidation behaviour differs significantly between AWS (explicit TTL), Azure (automatic), and GCP (implicit + explicit). The optimal caching strategy for multi-model, multi-cloud deployments is still being established.
- **Open-weight model thinking APIs:** Whether Llama 5 and Qwen 4 will offer API-level thinking control (like proprietary models) is unknown as of May 2026.
- **Regulatory requirements for prompt auditability:** The EU AI Act's requirements around prompt documentation and auditability for high-risk systems are not yet fully clarified in implementation guidance.

---

## Caveats

- **Vendor documentation bias:** Official cloud provider docs (AWS, GCP, Azure) naturally present their own services favourably. Cross-referenced with independent sources where possible.
- **Recency of findings:** The prompt engineering landscape changes rapidly. Techniques described here reflect May 2026 state; validate against current model release notes before implementation.
- **Paywall limitations:** Some academic literature on prompt optimisation (particularly evaluation methodology papers) was behind paywalls and not accessed directly.
- **Geographic/language coverage:** Findings are primarily sourced from English-language documentation and North American/European practitioner communities.
- **Model-specific advice shelf life:** Specific model capabilities (e.g., Claude 4.6's adaptive thinking, GPT-5.5's reasoning_effort) are subject to change with model updates. The principles are durable; the parameter names may not be.

---

## References

1. [Anthropic — Be Clear and Direct (Prompt Engineering Guide)](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/be-clear-and-direct)
2. [Google Cloud — Introduction to Prompt Design (Vertex AI)](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/introduction-prompt-design)
3. [Braintrust — What is Prompt Management?](https://www.braintrust.dev/articles/what-is-prompt-management)
4. [Optivulnix — Prompt Engineering at Scale: Version Control, Testing, and Deployment](https://optivulnix.com/blog/prompt-engineering-version-control-production/)
5. [QubitTool — Enterprise LLMOps Architecture Guide 2026](https://qubittool.com/blog/enterprise-llmops-architecture-guide)
6. [Microsoft Learn — LLMOps: Operational Management of LLMs](https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/mlops-in-openai/)
7. [AWS — Prompt Management in Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-management.html)
8. [Azure — Prompt Flow in Microsoft Foundry Portal](https://learn.microsoft.com/en-us/azure/foundry-classic/concepts/prompt-flow)
9. [Google Cloud Blog — Manage your prompts using Vertex SDK](https://cloud.google.com/blog/products/ai-machine-learning/manage-your-prompts-using-vertex-sdk/)
10. [AWS — Prompt caching for faster model inference in Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
11. [Anthropic — Extended Thinking Documentation](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
12. [OpenAI — Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
13. [Langfuse Documentation](https://langfuse.com/docs)
14. [PromptFoo Documentation](https://promptfoo.dev/docs/intro)
15. [MLflow — Prompt Registry](https://mlflow.org/docs/latest/llms/prompt-registry/)
16. [Microsoft — .prompty File Format](https://prompty.ai)
17. [DeepEval Documentation](https://docs.confident-ai.com/)
18. [Ragas Documentation](https://docs.ragas.io/)
