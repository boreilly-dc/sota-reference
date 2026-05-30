# AI-Assisted Change Impact Assessment for Infrastructure Mega-Projects

| Field | Value |
|-------|-------|
| Created | 2026-05-27 |
| Last Updated | 2026-05-30 |
| Version | 1.3 |

---

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Industry Context and State of the Art](#industry-context-and-state-of-the-art)
- [Core Capabilities (MVP)](#core-capabilities-mvp)
- [Document Baseline Architecture](#document-baseline-architecture)
- [Change Request Workflow Integration](#change-request-workflow-integration)
- [Technical Architecture](#technical-architecture)
- [Design Principles for Low-Maturity Organisations](#design-principles-for-low-maturity-organisations)
- [Open-Source and Managed Service Options](#open-source-and-managed-service-options)
- [Scale Guidance](#scale-guidance)
- [Future State and Roadmap](#future-state-and-roadmap)
- [Risk Factors](#risk-factors)
- [Engineering Readiness Pack](#engineering-readiness-pack)
- [References](#references)

## Introduction

This reference design describes an agentic AI system for assessing change request impacts against regulatory and compliance document baselines on large infrastructure mega-projects ($1B+). **This is a pioneering architecture** — no verified end-to-end deployments exist in the literature for this specific pattern. The design is presented as a reusable blueprint with appropriate caveats regarding its novelty.

The value proposition is **risk reduction and accuracy**, not efficiency. On projects of this scale, an incorrectly assessed change request has consequences ranging from thousands to millions of dollars, or breach of externally-controlled regulatory approvals that may halt works entirely.

## Problem Statement

Infrastructure mega-projects operate within extraordinarily complex approval environments. A single project may have:

- Environmental approvals running to 1,000+ pages across multiple instruments
- Hundreds of secondary approvals (flora/fauna permits, heritage agreements, water licences)
- Native title and cultural heritage agreements with specific obligations
- Construction and environmental management plans (CEMPs, SEMPs, CTMPs) numbering in the dozens
- Interface agreements with adjacent projects, utilities, and transport authorities

When a change request is raised — whether triggered by design development, unforeseen site conditions, or stakeholder requirements — staff must assess its impact against this entire baseline. The approval environment is too complex for any individual to hold in their head. The result is systematic underestimation of impacts, missed approval triggers, and downstream non-compliance discovered only when it becomes expensive to rectify.

Australian transport infrastructure projects valued at $20M or more have historically exceeded promised costs by 21% on average, representing $34B in total overruns over the past two decades (Grattan Institute, 2020). While change management is not the sole cause, poorly assessed changes that cascade into compliance breaches and rework are a significant contributor.

## Industry Context and State of the Art

As of mid-2026, no verified end-to-end deployment of agentic AI for change request impact assessment at mega-project scale has been documented in peer-reviewed literature. While research into LLM-based compliance checking and construction document analysis is accelerating (particularly in BIM compliance and code checking), these remain isolated capabilities rather than integrated change impact systems.

The closest market analogues include:

- **Procore / Datagrid** (acquired January 2026): Datagrid is a vertical AI platform focused on data connectivity and autonomous workflow execution across construction systems, not full regulatory change impact assessment.
- **ALICE Technologies**: AI-driven construction scheduling and optimisation (generative scheduling, what-if scenario analysis), not document-level compliance.
- **nPlan**: AI-powered schedule risk analytics using deep learning and graph neural networks trained on 750,000+ historical project schedules.
- **Oracle Construction and Engineering Intelligence**: Combines managed ETL, pre-built data pipelines, generative AI visualisations, and predictive analytics for construction project risk mitigation.
- **CMiC**: Enterprise construction ERP built on a single-database architecture with integrated project, financial, and field management capabilities.

Emerging research shows promise in LLM-based approaches for BIM-based automated compliance checking (Automation in Construction, 2025) and knowledge graph construction techniques that could underpin structured regulatory knowledge bases (Choi & Jung, 2025), but these remain experimental and have not been validated at the scale of a full project approval environment.

This reference design therefore carries inherent uncertainty. Organisations implementing it should expect to iterate significantly and should plan for controlled piloting before relying on outputs for governance decisions.

## Core Capabilities (MVP)

### Content Library

A curated, restricted baseline document library containing all controlled documents against which change requests must be assessed. Documents are ingested from SharePoint, document management systems, or controlled document registers. Access controls mirror existing project permissions.

### Scope Quality Check

Before impact assessment begins, the system evaluates whether the change request contains sufficient information. It identifies gaps in scope description, affected areas, triggering events, and proposed solutions — then prompts the submitter for additional detail. This addresses the common problem of under-described changes entering the assessment pipeline.

### Impact Assessment

The core capability: compare the change description against the full baseline document library. The system identifies:

- Which regulatory instruments are potentially affected
- Which specific conditions, obligations, or clauses are engaged
- Which management plans require review or update
- Whether the change triggers notification or re-approval requirements

### Functional Area Reports

Impacts are grouped by functional lead (environment, planning, heritage, community, engineering, commercial) for their validation. Each functional lead receives only the impacts relevant to their domain, with source references into the baseline documents.

### Decision-Maker Summary

A consolidated summary with options, trade-offs, and a clear statement of which impacts are confirmed versus requiring further investigation. Designed for governance forums where time is constrained and decisions must be defensible.

### Confidence Scoring

A traffic-light system indicating:

- **Green**: High confidence — change is well-described, baseline documents are clear, impacts are straightforward
- **Amber**: Moderate confidence — some ambiguity in change scope or baseline interpretation
- **Red**: Low confidence — insufficient information, conflicting baseline documents, or novel regulatory territory requiring specialist review

### Interactive Querying

A chat interface for super users to deep-dive after the initial assessment: "What if the works extend 50m further south?", "Which conditions specifically require pre-clearance surveys?", "Has a similar change been assessed before?"

## Document Baseline Architecture

### Baseline Artefacts Register

A structured index of all controlled documents, capturing:

- Document ID (format: `BR.XX.NON.XXX-XXX-NNNNN` or project-specific convention)
- Document title, revision, and date
- Controlling authority (external regulator vs internal governance)
- Functional area owner
- Key obligations and conditions extracted as structured data

### Criticality Tiers

Documents are classified into two tiers with materially different consequences:

1. **Externally controlled** (regulatory): Breach has severe consequences — potential stop-work orders, prosecution, reputational damage, or loss of approval. Changes affecting these documents require external notification or re-approval.
2. **Internally controlled**: Updatable through the project's own governance processes. Consequences of non-compliance are internal (audit findings, process failures) but still require managed updates.

### Supported Document Formats

| Format | Use Case | Processing Approach |
|--------|----------|-------------------|
| PDF | Approvals, permits, reports | OCR + layout-aware extraction |
| Word/DOCX | Management plans, procedures | Direct text extraction with structure preservation |
| Excel/XLSX | Registers, compliance matrices, schedules | Tabular extraction with header inference |
| PowerPoint | Briefings, stakeholder presentations | Slide-level extraction |
| P6 (XER/XML) | Programme schedules | Structured parse of activities, logic, calendars |
| GIS (Shapefile/GeoJSON) | Spatial boundaries, exclusion zones | Geospatial indexing for location-based queries |

## Change Request Workflow Integration

The AI system integrates at specific points within the existing change management workflow:

1. **Change raised** in document management system (InEight, Aconex, Procore, or equivalent)
2. **Tiering assessment** against standard criteria: mandate alignment, interface complexity, approvals impact, delivery capability, community/reputation risk
3. **AI-assisted scoping**: System reviews the change request for completeness, flags gaps, suggests affected areas
4. **AI-assisted impact assessment**: Full baseline scan with confidence-scored outputs
5. **Functional area review**: Human validation of AI-identified impacts by subject matter experts
6. **Governance decision**: Human approval at all gates — the AI informs but does not decide

The system never bypasses human governance. It provides better information to decision-makers, not automated decisions.

## Technical Architecture

```
[Change Management System] ──API──> [Intake Agent]
                                         │
                                         v
                                  [Scope Validation Agent]
                                         │
                                         v
                              [Multi-Document Impact Scan]
                               (RAG over baseline library)
                                         │
                                         v
                              [Functional Area Router]
                                    /    |    \
                                   v     v     v
                          [Env]  [Plan] [Heritage] ... [Commercial]
                                    \    |    /
                                     v   v   v
                              [Confidence Scorer]
                                         │
                                         v
                              [Summary Generator]
                                         │
                                         v
                          [Decision-Maker Dashboard / Chat UI]
```

### Key Technical Components

- **RAG pipeline**: Hybrid search (BM25 keyword + dense vector retrieval) over the baseline document library, with re-ranking for precision
- **Chunking strategy**: Document-structure-aware chunking preserving clause/condition boundaries, not arbitrary token windows
- **Agentic workflow**: Multi-step orchestration where each agent has a defined role and structured outputs feed downstream agents
- **Structured output**: LLM responses constrained to impact matrices, obligation references, and confidence scores rather than free-form text
- **Audit trail**: Every assessment step logged with source citations for governance defensibility
- **Integration layer**: REST APIs to document management systems (InEight, Procore, Aconex) for change request ingestion and status updates

## Design Principles for Low-Maturity Organisations

Infrastructure mega-projects typically operate at low digital maturity relative to technology companies. The following principles ensure adoption:

1. **Meet the organisation where it is**: Target 10–20% stretch from current capability, not a transformation. If the organisation currently assesses changes in Word templates via email, the first deployment is a standalone chat interface — not an embedded automation.

2. **Flag, don't confront**: The system identifies gaps gently ("This change request may benefit from additional detail on the spatial extent of works") rather than accusatorily ("This change request is incomplete"). Language matters for adoption.

3. **Start basic, layer sophistication**: MVP delivers a simple impact summary. Subsequent iterations add confidence scoring, functional routing, and interactive querying.

4. **Narrow initial user base**: 3–4 super users who understand both the technology and the approval environment. They validate outputs and build trust before broader rollout.

5. **Standalone first, embedded later**: A chat interface accessible via browser is the first deployment. Embedded panels within document management systems come only after the core capability is proven and trusted.

6. **Human-in-the-loop always**: The system augments human judgement; it does not replace it. This is non-negotiable for regulatory compliance contexts.

## Open-Source and Managed Service Options

### Open-Source Stack

| Layer | Options |
|-------|---------|
| RAG framework | LangChain, LlamaIndex |
| Vector database | Qdrant, Weaviate, Milvus, pgvector |
| Embeddings | BGE-M3, GTE-Qwen2, nomic-embed-text |
| Document processing | Apache Tika, Unstructured.io, marker-pdf, Docling |
| LLM (on-prem/sovereign) | Llama 4 Scout/Maverick, Mistral Large, Qwen 2.5/3 family models verified for the deployment environment |
| Workflow orchestration | LangGraph, CrewAI, Prefect |
| Search (keyword) | OpenSearch, Apache Solr |
| Geospatial | PostGIS, GeoServer |

### Managed Services (Hyperscaler)

| Capability | Azure | AWS | GCP |
|-----------|-------|-----|-----|
| Vector search | AI Search | OpenSearch Serverless | Vertex AI Vector Search |
| Document processing | Document Intelligence | Textract | Document AI |
| LLM inference | Azure OpenAI Service | Bedrock | Vertex AI |
| Database | Cosmos DB (vCore) | Aurora PostgreSQL | AlloyDB |
| Object storage | Blob Storage | S3 | Cloud Storage |
| Orchestration | Azure AI Foundry Agent Service | Step Functions + Bedrock Agents | Vertex AI Agent Builder |

### Sovereign Deployment Considerations

For mega-projects subject to data sovereignty requirements (common in Australian government infrastructure), on-premises or sovereign cloud deployment may be mandatory for documents classified above OFFICIAL. The open-source stack supports air-gapped deployment; hyperscaler options require sovereign-capable regions or services:

- **Azure**: IRAP-assessed at PROTECTED level for Azure, Microsoft 365, and Dynamics 365 services in Australian regions (first hyperscaler to achieve PROTECTED certification, 2018).
- **AWS**: IRAP-assessed at PROTECTED level for services in the Asia Pacific (Sydney) region, with published IRAP PROTECTED reference architecture.
- **GCP**: IRAP-certified for a subset of services (assessed by the Australian Signals Directorate's ACSC), with Assured Workloads providing sovereignty controls for Australian government requirements.

## Scale Guidance

### Small Projects ($50M-$500M, <50 baseline documents)

- The full agentic multi-agent architecture is overkill. Use a simpler RAG chat interface over the baseline document library.
- Skip the tiering assessment automation — at this scale, manual tiering is fast enough.
- Deploy as a single Azure Function App or Container App with AI Search + Azure OpenAI.
- The content library fits comfortably in a single AI Search index without hierarchical retrieval.
- Confidence scoring can be simplified to a binary "well-supported / needs specialist review" output.
- 1-2 super users are sufficient; no functional area routing needed if one person covers all domains.
- Total infrastructure cost: $300-600/month (Azure AI Search Basic + Azure OpenAI consumption).

### Mega-Projects ($1B+, 100+ baseline documents, complex regulatory environment)

- The full architecture as described: multi-agent workflow with functional area routing and tiered confidence scoring.
- Hierarchical retrieval is essential: register → document → section → clause → obligation.
- Dedicated chunking strategy per document type (regulatory approvals chunked at condition boundaries, management plans at section headings, schedules at activity level).
- Consider knowledge graph augmentation alongside vector search for cross-document relationship traversal.
- Multiple document management system integrations (InEight, Procore, Aconex) with real-time change notification.
- GIS layer integration for spatial impact queries from MVP.
- Budget for 6-12 months of iterative refinement — this is pioneering work with no guaranteed first-time success.
- Total infrastructure cost: $2,000-5,000/month depending on document volume and query load.

## Future State and Roadmap

Beyond MVP, the following capabilities represent logical extensions:

- **Consistency checking**: Identify contradictions between baseline documents (e.g., an environmental approval condition that conflicts with a heritage management plan requirement)
- **Full document mark-up**: Generate proposed amendments to affected management plans, tracked as draft revisions
- **What-if exploratory mode**: "If we undertook works in this location, how many approvals would be triggered?" — enabling proactive planning
- **GIS layer integration**: Spatial queries overlaid on approval boundaries, exclusion zones, and sensitive receptor locations
- **P6 schedule integration**: Temporal impact assessment — does this change affect critical path activities?
- **3D model integration**: BIM/digital twin linkage for spatial change visualisation
- **Broader content library**: Legislation, codes of practice, Australian standards, design guidelines
- **Cross-project learning**: Patterns from assessed changes on one mega-project informing another (with appropriate data governance)

## Risk Factors

| Risk | Mitigation |
|------|-----------|
| **Novelty** — no established playbook for this use case | Controlled pilot with clear success criteria; accept iteration; maintain human override at all points |
| **Document quality** — baseline documents vary in structure, clarity, and internal consistency | Document-structure-aware processing; confidence scoring reflects source quality; human validation of edge cases |
| **Organisational change management** — staff may distrust or resist AI outputs | Super-user-led adoption; demonstrate value on real changes before broad rollout; "flag don't confront" principle |
| **Regulatory interpretation** — some obligations require legal/specialist judgement beyond AI capability | System identifies affected clauses but does not interpret legal meaning; always routes to human specialist |
| **Baseline conflicts** — controlled documents may contradict each other | Highlight conflicts rather than resolve them; this is itself valuable information for governance |
| **Hallucination risk** — LLM generates plausible but incorrect impact assessments | RAG with source citations; structured output constraints; confidence scoring; mandatory human review |
| **Scale** — document libraries of 10,000+ pages may challenge retrieval precision | Hierarchical retrieval (register → document → section → clause); re-ranking; iterative refinement |

## Engineering Readiness Pack

This design is explicitly novel. Engineering teams should treat it as a controlled pilot architecture until validated on real project baselines and historical change requests.

### Evidence and claim ledger

Maintain a `claim-ledger.md` with `claim`, `section`, `source`, `source type`, `last verified`, `confidence`, `owner`, and `recheck trigger`.

| Claim class | Current status | Required handling |
|---|---|---|
| No verified end-to-end deployments | Research assessment | Re-run market/literature scan before investment decisions. |
| Vendor analogues | Market context, not proof of this pattern | Label as adjacent capability only; do not claim validation of regulatory impact assessment. |
| Cost and schedule impact figures | External report context | Keep separate from claims that this AI system will reduce overruns. |
| Sovereign cloud/IRAP claims | Time-sensitive | Verify against official assurance portals and customer obligations before architecture sign-off. |
| Model suitability | Assumption | Validate against project-specific baseline documents and historical changes. |

### Implementation artefacts

Required before build:

- Obligation data model with `obligation_id`, source document, clause/condition, spatial extent, temporal trigger, responsible function, approval authority, criticality tier, and change-trigger rules.
- Change request schema covering scope, location/GIS geometry, schedule impact, design artefacts, affected assets, requested decision, and attachments.
- C4 context/container diagrams showing change-management system, baseline library, retrieval/indexing, obligation store, GIS layer, workflow engine, functional review queues, dashboard, and audit log.
- Integration contracts for Aconex/InEight/Procore or equivalent, GIS, P6 schedule data, SharePoint/EDRMS, identity provider, and reporting exports.
- ADRs for RAG vs knowledge graph, clause-boundary chunking, GIS query engine, workflow engine, model hosting, and audit-log immutability.

### Anti-hallucination and governance controls

The system must identify possible impacts, not decide compliance outcomes.

Controls:

- Structured output only: impact matrix rows must include source document ID, clause/condition, extracted text span, relevance rationale, confidence, affected function, and required human reviewer.
- No-citation/no-impact rule: a generated impact cannot be shown as actionable unless it is tied to a baseline source span.
- Conflict surfacing: where documents conflict, the system must present the conflict and route to humans, not reconcile it silently.
- Obligation extraction QA before impact assessment; poor OCR or low-confidence clause extraction lowers downstream confidence.
- Specialist review gates for externally controlled approvals, legal/regulatory interpretation, cultural heritage, environmental conditions, and safety-critical changes.
- Prompt-injection filtering for uploaded change descriptions and baseline documents.

### Threat model

| Threat | Control |
|---|---|
| Missed external approval trigger | Hierarchical retrieval, obligation register coverage tests, functional SME review, and red-confidence routing. |
| Fabricated impact or clause reference | Server-side source-span validation and structured output schemas. |
| Overconfident model on ambiguous scope | Scope quality check must produce missing-information questions and amber/red confidence. |
| Stale baseline documents | Controlled document register sync, revision metadata, and superseded-document blocking. |
| Unauthorised access to restricted project documents | Mirror project permissions in retrieval filters and audit every source access. |
| Spatial query error | GIS layer with deterministic geometry checks; LLM cannot infer boundaries from prose alone. |

### Evaluation and acceptance gates

Pilot eval pack:

- 30-50 historical change requests with known impact assessments, reviewed obligations, and final governance outcomes.
- Coverage tests for externally controlled approvals, internal management plans, GIS-triggered obligations, schedule-triggered obligations, and no-impact changes.
- Retrieval recall target set by SMEs for obligation clauses; any missed stop-work or re-approval trigger is severity 1.
- Hallucination tests with irrelevant clauses, superseded documents, conflicting requirements, vague scope, and malicious instructions embedded in attachments.
- Promotion gate: no fabricated obligations, no missed high-criticality historical triggers in the gold set, and SME acceptance of the false-positive burden.

### Operational runbook

Runbooks must cover baseline re-indexing, urgent document supersession, erroneous impact report withdrawal, SME review queue backlog, integration outage, GIS data mismatch, audit export for governance forum, and rollback to manual assessment.

## References

- Terrill, M., Emslie, O., & Moran, G. (2020). *The Rise of Megaprojects: Counting the Costs*. Grattan Institute Report No. 2020-15. https://grattan.edu.au/report/the-rise-of-megaprojects-counting-the-costs/
- Procore Technologies. (2026). *Procore Acquires Datagrid to Accelerate AI Strategy*. https://www.procore.com/press/procore-acquires-datagrid
- ALICE Technologies. (2026). *AI Construction Project Planning and Scheduling Software*. https://www.alicetechnologies.com/
- nPlan. (2026). *AI-Powered Schedule Risk Analysis for Construction*. https://www.nplan.io/
- Leveraging Large Language Models for BIM-based Automated Compliance Checking. (2025). *Automation in Construction*, 175. https://www.sciencedirect.com/science/article/pii/S0926580525007472
- Choi, S., & Jung, Y. (2025). Knowledge Graph Construction: Extraction, Learning, and Evaluation. *Applied Sciences*, 15(7), 3727. https://www.mdpi.com/2076-3417/15/7/3727
- Chishiki-AI. (2024). *AI-powered Civil Engineering Community*. University of Texas at Austin / Cornell University (NSF Award #2321040). https://www.chishiki-ai.org/
- Oracle. (2026). *Construction and Engineering Intelligence: Managed ETL, Pre-built Data Pipelines, and Predictive Analytics*. https://www.oracle.com/construction-engineering/intelligence/
- InEight. (2026). *Document Control for Capital Construction*. https://ineight.com/products/ineight-document/
- LangChain. (2026). *LangGraph: Agent Orchestration Framework*. https://github.com/langchain-ai/langgraph
- Unstructured.io. (2026). *Open-Source Document Processing*. https://github.com/Unstructured-IO/unstructured
- Microsoft. (2025). *Introducing Microsoft Agent Framework*. https://azure.microsoft.com/en-us/blog/introducing-microsoft-agent-framework/
- Google Cloud. (2026). *Vertex AI Agent Builder*. https://docs.cloud.google.com/agent-builder
- Microsoft. (2024). *Microsoft's Commitment to Trust in Australia: 2024 Azure, Dynamics 365, and Microsoft 365 IRAP Assessments*. https://servicetrust.microsoft.com/Viewpage/AustraliaIRAP
- AWS. (2025). *IRAP Compliance — Amazon Web Services*. https://aws.amazon.com/compliance/irap/
- Google Cloud. (2025). *IRAP Compliance — Google Cloud*. https://cloud.google.com/security/compliance/irap
- CMiC. (2026). *All-in-One Construction ERP | Project & Financial Management*. https://cmicglobal.com/products
- NIST AI Risk Management Framework and Generative AI Profile. https://www.nist.gov/itl/ai-risk-management-framework
- OWASP Top 10 for LLM Applications 2025. https://owasp.org/www-project-top-10-for-large-language-model-applications/
- OpenAI Evals and graders documentation. https://platform.openai.com/docs/guides/evals
- ISO/IEC/IEEE 42010, Architecture description. https://www.iso.org/standard/74393.html
