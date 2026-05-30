# Automated Multi-Step AI Research Pipeline

| Field | Value |
|-------|-------|
| Created | 2026-05-27 |
| Last Updated | 2026-05-30 |
| Version | 1.3 |

---

- [Problem Statement](#problem-statement)
- [Reference Workflow](#reference-workflow)
- [Architecture Overview](#architecture-overview)
- [Workflow Execution Patterns](#workflow-execution-patterns)
- [Context Management Strategy](#context-management-strategy)
- [Data Sensitivity and Security](#data-sensitivity-and-security)
- [Technical Architecture (Azure-Native)](#technical-architecture-azure-native)
- [Open-Source Alternatives](#open-source-alternatives)
- [Cross-Platform Options](#cross-platform-options)
- [Scale Guidance](#scale-guidance)
- [Design Considerations](#design-considerations)
- [Engineering Readiness Pack](#engineering-readiness-pack)
- [Financial-Services Control Addendum](#financial-services-control-addendum)
- [Applicability Beyond Investment Research](#applicability-beyond-investment-research)
- [References](#references)

## Problem Statement

Analysts across investment research, compliance, and due diligence run structured multi-step processes that follow predictable sequences of reasoning. Today, they manually orchestrate between LLMs — each step's output is hand-fed as input to the next. The analyst becomes the glue layer, copying outputs between context windows, retrieving external data, verifying against structured models, and assembling final documents.

The pain points are concrete:

- **Manual context management**: Tracking which prior outputs feed which subsequent steps, managing context window limits, and deciding what to summarise versus include verbatim.
- **Copy-paste orchestration**: Manually transferring outputs between sessions, reformatting for the next prompt, and maintaining coherence across steps.
- **No resumability**: If a step fails or needs revision, there is no clean way to re-run from that point without reconstructing the entire chain.
- **Inconsistent quality**: Without standardised prompt templates and workflow definitions, output quality varies between runs and between analysts.

This reference architecture addresses these problems with an automated pipeline that chains LLM calls together, manages state between steps, integrates external data sources, and provides human-in-the-loop checkpoints where analyst judgement is required.

## Reference Workflow

The following seven-step investment research workflow illustrates the pattern. Each step's output feeds subsequent steps, with varying requirements for external data, structured analysis, and independent verification.

**Step 1 — Industry & Business Model Research**: Retrieve and synthesise information on the target company's segments, competitors, demand drivers, and market positioning. Requires web research and document ingestion (annual reports, industry publications).

**Step 2 — Financial Data Verification**: Audit a pre-populated Excel financial model for errors, inconsistencies, and missing data. Requires structured data parsing and numerical reasoning against the source filings.

**Step 3 — Historical Financial Commentary**: Generate analytical commentary on five-year financial performance (revenue growth, margins, capital allocation, returns). Consumes Step 2 outputs plus the verified model data.

**Step 4 — Competitive Advantages & Moat Assessment**: Qualitative analysis of durable competitive advantages supported by financial evidence (pricing power, switching costs, network effects, scale economics). Draws on Steps 1 and 3.

**Step 5 — Management Commentary**: Two sub-steps. (a) Generate a fresh-context prompt for management quality assessment, intentionally isolated from prior analysis to avoid confirmation bias. (b) Cross-reference the management assessment against independent evidence (insider transactions, capital allocation history, governance scores).

**Step 6 — Forward Outlook & Assumptions**: Two sub-steps. (a) Generate forecast assumptions and scenarios based on industry trends and company positioning. (b) Integrate and reconcile forecasts against the financial model, flagging inconsistencies.

**Step 7 — Investment Thesis Synthesis**: Produce a standalone one-page investment thesis in dense prose, drawing on all prior steps. This is the final deliverable — a concise, internally consistent argument for or against an investment position.

## Architecture Overview

The pipeline comprises seven components:

| Component | Responsibility |
|-----------|---------------|
| **Trigger Input** | Entity identifier + key URLs, data sources, and analyst parameters |
| **Workflow Engine** | Orchestrates step execution, manages state, handles human checkpoints and branching |
| **Step Executor** | Executes individual LLM calls with step-specific prompts and assembled context |
| **Context Manager** | Assembles relevant prior-step outputs + external data into each step's context window, applying summarisation where needed |
| **Data Connectors** | File upload (Excel, PDF), web research, API integrations (financial data providers) |
| **State Store** | Persists intermediate outputs, enables resume/retry from any step |
| **UI Layer** | Initiate workflows, review intermediate outputs at checkpoints, approve/modify/redirect, view final output |

Human checkpoints are configurable gates positioned after any step where the analyst can review the output, edit it directly, provide additional guidance, or reject and re-run with modified parameters.

## Workflow Execution Patterns

Four execution modes cover the range of analyst trust and workflow maturity:

**Fully automated**: All steps run sequentially without pause. Suitable for trusted, repeatable workflows where the analyst reviews only the final output. Typical for well-established templates after initial calibration.

**Step-by-step with checkpoints**: The pipeline pauses after each step (or after designated steps) for human review and approval. The analyst can edit outputs before they propagate downstream. This is the default for new workflows.

**Branching**: Some steps produce multiple variants (e.g., bull/bear/base case scenarios in Step 6). The analyst selects the preferred path, or the pipeline carries all branches forward for comparison in the synthesis step.

**Retry/edit**: The analyst edits a step's output and triggers re-execution of all subsequent steps from that point. The state store preserves the full history, enabling rollback and comparison between runs.

## Context Management Strategy

Context window management is critical — investment research steps can individually exceed 100k tokens when source documents are included.

- Each step has a **prompt template** specifying which prior outputs to include (verbatim or summarised) and which external data to retrieve.
- **Long outputs are summarised** before passing to subsequent steps. Summarisation uses a dedicated compression prompt that preserves key facts, figures, and conclusions.
- **External data** (Excel models, filings, reports) is parsed and structured before inclusion — raw file bytes are never passed directly to the LLM.
- **Intentional isolation**: Some steps (e.g., Step 5a) deliberately exclude prior analysis to enable independent verification. The context manager enforces this isolation per the workflow definition.
- **Token budgets**: Each step specifies a maximum input token allocation. If assembled context exceeds the budget, the context manager applies progressive summarisation of oldest/least-relevant prior outputs.

## Data Sensitivity and Security

Investment research data is commercially sensitive. Premature disclosure of research conclusions, position sizing, or target prices can constitute market abuse. Security requirements are non-negotiable:

- **Encryption**: All data encrypted in transit (TLS 1.3) and at rest (AES-256).
- **Tenant isolation**: No data shared across tenants. Dedicated compute and storage per organisation where required.
- **Data residency**: Configurable to meet jurisdictional requirements (e.g., data remains within AU/NZ region).
- **LLM data controls**: Use providers and deployment modes whose current contracts, data-processing terms, and service configuration meet the organisation's restrictions on training use, retention, and human review. Verify terms before production; do not rely on generic vendor summaries.
- **Audit logging**: Every LLM interaction logged with timestamp, user identity, input hash, and output hash. Immutable audit trail for regulatory compliance.
- **Access control**: Role-based access to workflows, step outputs, and final deliverables. Separation between research and trading functions where applicable.

## Technical Architecture (Azure-Native)

For organisations requiring enterprise audit compliance, an Azure-native deployment provides integrated identity, compliance, and data residency controls:

| Layer | Service |
|-------|---------|
| LLM Inference | Azure OpenAI Service (GPT-4o, o3 for reasoning-heavy steps) |
| Agent Orchestration | Microsoft Agent Framework / Microsoft Foundry Agent Service (agent orchestration, tool calling, hosting) |
| Workflow Engine | Azure Durable Functions or custom orchestrator on Container Apps |
| Application Hosting | Azure Container Apps (serverless scaling) |
| State Store | Azure Cosmos DB (workflow state, step outputs) or PostgreSQL Flexible Server |
| API Layer | Azure API Management (rate limiting, authentication, observability) |
| Secrets | Azure Key Vault |
| File Storage | Azure Blob Storage (uploaded documents, generated outputs) |
| Identity | Microsoft Entra ID (SSO, RBAC) |
| CI/CD | GitHub + GitHub Actions |
| Monitoring | Azure Monitor + Application Insights |

## Open-Source Alternatives

Every component has viable open-source alternatives for organisations that prefer self-hosted deployments or want to avoid vendor lock-in:

| Component | Open-Source Options |
|-----------|-------------------|
| Workflow Orchestration | LangGraph, CrewAI, Prefect, Temporal, Apache Airflow |
| LLM Inference | Llama 4 Scout/Maverick, Mistral Large, Qwen 3 (dense: 0.6B-32B; MoE: 30B-A3B, 235B-A22B) via Ollama or vLLM |
| Agent Framework | LangChain, Microsoft Agent Framework, or another maintained framework verified against the current official documentation |
| State Management | PostgreSQL, SQLite (simpler deployments), Redis (ephemeral state) |
| UI | Streamlit, Gradio, custom React/Next.js application |
| Document Parsing | Unstructured.io, marker-pdf, docling, pandas (Excel) |
| Web Research | Tavily (commercial API), SerpAPI (commercial API), or custom scraping with Playwright |
| Observability | OpenTelemetry, Langfuse, Phoenix (Arize) |

A minimal self-hosted deployment can run on a single server with PostgreSQL for state, vLLM for inference, LangGraph for orchestration, and Streamlit for the UI.

## Cross-Platform Options

**AWS**: Amazon Bedrock (LLM inference with Claude, Llama, Mistral), AWS Step Functions (workflow orchestration), ECS/Fargate (application hosting), DynamoDB or Aurora PostgreSQL (state store), S3 (document storage), Cognito (identity).

**GCP**: Vertex AI (Gemini, Claude, open models), Cloud Workflows or Cloud Composer (orchestration), Cloud Run (hosting), Firestore or Cloud SQL PostgreSQL (state store), Cloud Storage (documents), Identity Platform (authentication).

**IBM**: watsonx.ai (LLM inference), IBM Cloud Code Engine (hosting), IBM Cloud Databases for PostgreSQL (state store).

**Oracle**: OCI Generative AI Service (LLM inference), OCI Container Instances (hosting), Oracle Autonomous Database (state store).

## Scale Guidance

### Small Deployments (1-3 analysts, single workflow type)

- LangGraph or a simple Python script with sequential function calls is sufficient — skip Temporal/Airflow.
- Use Streamlit for the UI — fast to build, adequate for a small user base.
- SQLite or a single PostgreSQL database handles all state requirements.
- A single approved model for all steps is simpler to manage than per-step model tiering. Select from the organisation's verified model shortlist, not from unverified release claims.
- Skip APIM — not needed when the application is accessed by a handful of known users.
- Human checkpoints can be implemented as simple CLI prompts or Streamlit form inputs.
- Total cost: model inference fees only ($50-200/month for moderate usage).

### Large Deployments (10+ analysts, multiple workflow types, enterprise compliance)

- Full workflow engine (Temporal or Azure Durable Functions) with retry logic, timeout handling, and observability.
- Per-step model tiering to optimise cost (frontier for synthesis, cheap for extraction).
- Custom React/Next.js UI with role-based access, workflow templates, and audit dashboards.
- Centralised template library with version control and analyst-level customisation.
- APIM for rate limiting, authentication, and metering (cost-per-analyst reporting).
- Formal output review workflows with multi-level approval for high-stakes deliverables.
- Integration with enterprise data providers (Bloomberg, Refinitiv, FactSet) via dedicated connectors.
- Compliance audit trail with immutable logging and retention policies.
- Total cost: $2,000-10,000/month depending on analyst count, model usage, and data provider fees.

## Design Considerations

**Idempotency**: Steps must be re-runnable without side effects. Given the same inputs and parameters, a step should produce consistent (though not necessarily identical) outputs. External data retrieval should be cached per workflow run.

**Observability**: Log token usage, latency, cost, and quality metrics per step. Track completion rates, human override frequency, and time-to-completion across workflows. This data informs prompt optimisation and model selection.

**Cost management**: Assign token budgets per step. Use model tiering — higher-quality models for synthesis and judgement-heavy steps, cheaper models for extraction and summarisation. Monitor cost per workflow run and per analyst, and source all pricing from the provider's current pricing page for the target region.

**Output format**: Configurable per workflow and per step. Final outputs may be dense prose, structured tables, JSON for downstream systems, or formatted PDF via a rendering pipeline.

**Template library**: Maintain a library of reusable prompt templates and workflow definitions for different analysis types. Version-control templates alongside the application code. Enable analysts to fork and customise templates without engineering involvement.

**Model selection per step**: Not all steps require the same model. A well-designed pipeline uses cheaper, faster models for data extraction and summarisation, and reserves expensive reasoning models for synthesis and judgement-heavy steps.

## Engineering Readiness Pack

This design is not implementation-ready until the following artefacts exist in the repository next to the application code.

### Evidence and claim ledger

Maintain a `claim-ledger.md` with one row per factual claim that an engineer, compliance reviewer, or analyst might rely on. Required fields: `claim`, `document section`, `source`, `source type`, `last verified`, `confidence`, `owner`, and `expiry/recheck trigger`.

| Claim class | Current status | Required handling |
|---|---|---|
| Workflow pattern, resumability, and checkpointing | Design assumption grounded in workflow-engineering practice | Validate in pilot with at least three real research templates before production rollout. |
| Cloud and orchestration component choices | Source-linked but time-sensitive | Recheck official provider docs before procurement or build planning. |
| Open-weight model availability | High churn | Use only models verified from official model cards or vendor documentation; unsupported open-model release claims were removed on 2026-05-30. |
| Cost estimates | Directional | Replace with metered estimates from the target provider, region, and expected token volumes. |
| Financial-regulatory obligations | Jurisdiction-specific | Compliance owner must map to SEC, FINRA, FCA, ASIC, or local obligations before production use. |

### Architecture specification for engineering

Before build starts, create these implementation artefacts:

- C4 context and container diagrams showing analyst UI, workflow engine, model gateway, data connectors, state store, document store, eval service, and audit log.
- Step definition schema: `step_id`, `input_contract`, `prompt_template_version`, `allowed_tools`, `model_policy`, `token_budget`, `checkpoint_policy`, `output_schema`, `eval_suite`, and `retention_class`.
- Workflow state model covering queued, running, waiting_for_review, rejected, re-running_downstream, approved, failed, and archived states.
- Connector contracts for Excel models, filings, market-data APIs, web research, internal research libraries, and generated deliverables.
- ADRs for model-provider selection, data-residency posture, workflow engine choice, and whether analysts may edit intermediate outputs directly.

### Anti-hallucination controls

Every generated analytical claim must be traceable to one of four evidence types: source filing, structured financial model cell/range, licensed market-data API response, or analyst-approved prior output. The final investment thesis must reject unsupported assertions instead of smoothing over missing evidence.

Controls:

- Structured extraction from filings and spreadsheets before LLM synthesis; never ask the model to infer numbers from raw screenshots or unparsed files.
- Source-bound prompts requiring citation IDs for every material factual, financial, or valuation claim.
- Unsupported-claim detector that flags sentences without citations, stale data, or unresolved source conflicts.
- Independent verification step for financial figures, share counts, valuation multiples, management transactions, and market-size claims.
- "No recommendation" output path when evidence is insufficient or compliance review is not complete.
- Prompt-injection filtering for web and filing content before it enters the workflow context.

### Evaluation and acceptance gates

Minimum eval pack:

- 30-50 historical research tasks with source filings, model extracts, expected factual findings, and known traps.
- Numeric accuracy tests for extracted financial line items and model-derived ratios.
- Citation precision/recall tests: cited source must support the sentence, and all material claims must cite a source.
- Hallucination/refusal tests where the answer is absent, stale, conflicting, or outside permitted research coverage.
- Compliance tests for MNPI handling, disclosure language, research/trading separation, retention, and supervisory review.
- Regression suite triggered by prompt, model, connector, retrieval, or template changes.

Promotion gate: no severity-1 factual errors, no unsupported material claims, no citation fabricated by the model, and compliance sign-off for the target jurisdiction.

### Observability and runbook

Log workflow run ID, step ID, prompt/template version, model, tool calls, source IDs, token usage, latency, retry count, user edits, approval outcome, and final deliverable hash. Dashboards must show cost per workflow, failure rate per step, human override frequency, unsupported-claim rate, citation-failure rate, and stale-source rate.

Runbooks must cover failed data-provider calls, model outage, prompt-regression rollback, corrupted workflow state, source-document re-ingest, and compliance hold/removal of a generated report.

## Financial-Services Control Addendum

This architecture handles high-risk financial analysis and must be constrained as a decision-support system, not an autonomous recommendation engine.

Required controls before production:

- **Supervision and records**: retain prompts, source documents, intermediate outputs, approvals, final reports, and distribution records according to the applicable recordkeeping regime.
- **MNPI handling**: tag workflows and source documents by information barrier status; block cross-barrier retrieval and model context sharing.
- **AI-washing control**: marketing and user-facing materials must not imply predictive certainty, autonomous investment advice, or model capabilities that have not been validated.
- **Human approval**: final outputs require analyst and supervisory approval before distribution outside the research team.
- **Data licensing**: market-data and broker-research connectors must enforce licence terms on storage, redistribution, and derived outputs.
- **Conflict and disclosure checks**: final-report generation must include issuer coverage restrictions, holdings/conflicts, methodology, and required disclosures.

## Applicability Beyond Investment Research

This architecture applies to any structured analytical process with sequential reasoning steps:

- **Due diligence (M&A, vendor assessment)**: Multi-step investigation with document review, financial analysis, risk identification, and recommendation synthesis.
- **Competitive intelligence**: Market scanning, competitor profiling, trend analysis, and strategic implications.
- **Compliance audit reports**: Policy review, evidence gathering, gap analysis, finding classification, and report generation.
- **Insurance underwriting**: Risk assessment across multiple dimensions with data from diverse sources, culminating in a pricing recommendation.
- **Grant/proposal evaluation**: Multi-criteria assessment against rubrics, with evidence gathering and scoring justification.
- **Legal research**: Case law retrieval, argument construction, counter-argument identification, and brief synthesis.

The common pattern is: structured multi-step reasoning where each step builds on prior outputs, requires different types of input data, and benefits from human oversight at key decision points.

## References

- Microsoft Foundry documentation: https://learn.microsoft.com/en-us/azure/foundry/
- Microsoft Agent Framework: https://github.com/microsoft/agent-framework
- Microsoft Agent Framework blog: https://devblogs.microsoft.com/agent-framework/
- Azure Durable Functions: https://learn.microsoft.com/en-us/azure/azure-functions/durable-functions/durable-functions-overview
- LangGraph documentation: https://docs.langchain.com/oss/python/langgraph/overview
- LangGraph GitHub: https://github.com/langchain-ai/langgraph
- CrewAI framework: https://github.com/crewAIInc/crewAI
- Temporal workflow engine: https://temporal.io/
- Prefect orchestration: https://www.prefect.io/
- Amazon Bedrock: https://aws.amazon.com/bedrock/
- Google Vertex AI: https://cloud.google.com/vertex-ai
- OCI Generative AI Service: https://docs.oracle.com/en-us/iaas/Content/generative-ai/overview.htm
- Unstructured.io document parsing: https://github.com/Unstructured-IO/unstructured
- Docling (IBM) document parsing: https://github.com/docling-project/docling
- marker-pdf: https://pypi.org/project/marker-pdf/
- Langfuse observability: https://langfuse.com/
- Phoenix (Arize) observability: https://github.com/Arize-ai/phoenix
- vLLM inference engine: https://github.com/vllm-project/vllm
- Meta Llama 4 (Scout/Maverick): https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- Qwen 3 (Alibaba; released April 2025; dense 0.6B–32B, MoE 30B-A3B and 235B-A22B): https://github.com/QwenLM/Qwen3
- Qwen 3 technical report: https://arxiv.org/abs/2505.09388
- OpenAI o3 reasoning model: https://openai.com/index/introducing-o3-and-o4-mini/
- Azure OpenAI Service pricing: https://azure.microsoft.com/en-us/pricing/details/azure-openai/
- Microsoft Agent Framework documentation (MS Learn): https://learn.microsoft.com/en-us/agent-framework/
- Tavily web search API: https://www.tavily.com/
- Ollama (local LLM inference): https://ollama.com/
- Streamlit: https://streamlit.io/
- NIST AI Risk Management Framework and Generative AI Profile: https://www.nist.gov/itl/ai-risk-management-framework
- OWASP Top 10 for LLM Applications 2025: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- OpenAI Evals and graders documentation: https://platform.openai.com/docs/guides/evals
- SEC 2026 Examination Priorities: https://www.sec.gov/files/2026-exam-priorities.pdf
- SEC, "SEC Charges Two Investment Advisers with Making False and Misleading Statements About Their Use of Artificial Intelligence": https://www.sec.gov/newsroom/press-releases/2024-36
- FINRA Regulatory Notice 24-09, "Artificial Intelligence (AI) in the Securities Industry": https://www.finra.org/rules-guidance/notices/24-09
