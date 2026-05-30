# AI-Assisted OIA/FOI Request Processing Pipeline

| Field | Value |
|-------|-------|
| Created | 2026-05-27 |
| Last Updated | 2026-05-30 |
| Version | 1.3 |

---

- [Problem Statement](#problem-statement)
- [Pipeline Stages](#pipeline-stages)
- [Architecture Overview](#architecture-overview)
- [Key Technology Components](#key-technology-components)
- [Existing Deployments and Evidence](#existing-deployments-and-evidence)
- [Human-in-the-Loop Requirements](#human-in-the-loop-requirements)
- [Privacy and Sovereignty](#privacy-and-sovereignty)
- [Open-Source Stack](#open-source-stack)
- [Managed Services](#managed-services)
- [Scale Guidance](#scale-guidance)
- [Existing Platforms in Market](#existing-platforms-in-market)
- [Regulatory Framework](#regulatory-framework)
- [Technology Readiness Assessment](#technology-readiness-assessment)
- [Engineering Readiness Pack](#engineering-readiness-pack)
- [References](#references)

## Problem Statement

Government agencies face rapidly growing volumes of Official Information Act (OIA) and Freedom of Information (FOI) requests. In New Zealand, request volumes grew from 26,505 (January–June 2023) to 39,809 (July–December 2024) per six-month reporting period — a 50% increase in 18 months (Public Service Commission OIA Statistics). Statutory deadlines — 20 working days in New Zealand, 20 working days in the UK, varying by US state — create significant compliance pressure.

Manual processing is labour-intensive across every stage: document discovery across fragmented repositories, relevance assessment of potentially thousands of documents, PII redaction, sensitivity classification against withholding grounds, legal reasoning about applicable exemptions, and response drafting. A single complex request may require 40+ hours of analyst time.

AI can assist at each stage of the pipeline while maintaining the mandatory human oversight that democratic accountability requires. This reference architecture describes a reusable blueprint for AI-assisted OIA/FOI processing, applicable to any jurisdiction with statutory information disclosure obligations.

## Pipeline Stages

Each stage incorporates AI assistance paired with a human checkpoint. No stage operates autonomously for final decisions.

### 1. Intake and Triage

AI classifies incoming requests by topic area, estimated complexity (simple/moderate/complex), and likely document volume. The system routes requests to the appropriate team based on subject matter, identifies similar prior requests for consistency, and flags potential consultation requirements (e.g., requests touching other agencies' interests).

**Human checkpoint**: Team lead confirms routing and complexity classification.

### 2. Document Discovery (RAG)

Semantic search operates over government document stores — email archives, SharePoint, records management systems, and shared drives. The system compares requests against prior OIA responses and the agency's public reading room to identify whether information has already been released. Hybrid search (vector + keyword) ensures both conceptual and exact-match retrieval.

**Human checkpoint**: Analyst reviews retrieved document set for completeness.

### 3. Relevance Assessment

AI classifies documents as relevant, not relevant, or partially relevant to the scope of the request. Confidence scores accompany each classification. Partially relevant documents are flagged for scope determination (i.e., which sections fall within scope).

**Human checkpoint**: Analyst confirms relevance decisions, particularly edge cases.

### 4. Sensitivity Classification

Documents are classified against statutory withholding grounds. In New Zealand, this means Sections 6 (conclusive reasons — national security, international relations, law enforcement), 7 (special reasons — defence, trade negotiations), and 9 (other reasons subject to public interest test — privacy, commercial sensitivity, free and frank advice) of the OIA 1982. The classifier flags which specific exemption subsections may be engaged.

**Human checkpoint**: Legal advisor confirms classification of withholding grounds.

### 5. PII Detection and Redaction

Automated detection identifies personal information: names, addresses, phone numbers, email addresses, IRD numbers, health information, and other identifiers. The system suggests redaction boundaries and applies exemption code watermarks on redaction boxes (e.g., "s9(2)(a)" for privacy). Redactions must be immutable — pixel-level removal of underlying content, not overlays.

**Human checkpoint**: Analyst reviews all suggested redactions before application.

### 6. Legal Reasoning Assistance

The LLM drafts initial assessments of applicable withholding grounds, referencing Ombudsman case note precedents. For Section 9 grounds, the system applies the public interest balancing test — weighing the reason for withholding against the public interest in release. The AI produces structured rationales citing relevant case law.

**Human checkpoint**: Legal advisor reviews and approves all withholding rationales.

### 7. Response Drafting

The system generates a draft response letter with appropriate language, including a schedule of documents released, partially released, and withheld with specific reasons for each. Templates adapt to the type of release (full, partial, refusal, transfer, extension).

**Human checkpoint**: Delegated authority approves and signs the response.

### 8. Quality Assurance

Automated checks verify consistency with prior decisions on similar topics, confirm all redactions are complete and immutable, and track compliance deadlines. The system alerts when statutory timeframes are at risk.

**Human checkpoint**: Senior reviewer signs off on complex or high-profile requests.

## Architecture Overview

The pipeline integrates five core subsystems:

1. **RAG pipeline** — Ingests and indexes agency document stores (email via PST/EWS, SharePoint via Graph API, records management via CMIS). Maintains vector and keyword indices for hybrid search.
2. **PII detection engine** — Combines named entity recognition (NER), regex patterns for structured identifiers, and ML classifiers for contextual PII.
3. **Sensitivity classification model** — Fine-tuned on the agency's historical OIA decisions to classify documents against statutory withholding grounds.
4. **Case management integration** — Connects to existing workflow systems for deadline tracking, approval routing, and correspondence management.
5. **Audit trail** — Immutable log of all AI-assisted decisions, human overrides, confidence scores, and approval chains. Essential for Ombudsman review and algorithmic transparency.

## Key Technology Components

**Document discovery**: RAG with hybrid search (dense vector embeddings + BM25 keyword matching). Chunking strategy must respect document boundaries and maintain metadata lineage. Re-ranking with cross-encoder models improves precision. Similar to the US State Department's approach operating across 4 billion+ artefacts.

**PII redaction**: Microsoft Presidio (open-source, MIT licence) provides the NER and pattern-matching backbone. For production deployments requiring immutable redaction, CaseGuard offers on-premises AI-powered redaction across documents, video, audio, and images supporting 750+ file types. Hyperscience provides AI-assisted redaction specifically targeting FOIA/PA workflows. Redaction must produce pixel-level content removal — PDF overlay redactions are insufficient and can be reversed.

**Sensitivity classification**: Fine-tuned classifier trained on the agency's historical OIA decisions and outcomes. Generative AI adds explainability by producing natural-language rationales for each classification. Training data: prior decisions with Ombudsman outcomes provide ground truth.

**Legal reasoning**: Large language model with OIA/FOI case law, Ombudsman guidance, and agency policy in retrieval context. Must operate human-in-the-loop — the AI assists reasoning but does not make withholding decisions.

**Document processing**: Layout-aware extraction handles PDFs (including scanned), PST email archives, Word documents, Excel spreadsheets, and presentations. OCR with layout preservation for scanned content.

## Existing Deployments and Evidence

- **Microsoft NZ**: RAG-based OIA solutions deployed at multiple New Zealand central and local government agencies. One agency reported a 71% reduction in hours spent reporting on ministerial products and a 70% reduction in allocation time (Microsoft NZ, "Optimising official information management with AI", May 2025).
- **US State Department**: The department's eRecords archive contains 4 billion+ artefacts (emails and cable traffic). An ML-driven declassification pilot (launched October 2022) demonstrated 97% parity with human reviewers in determining whether to declassify records. Subsequent pilots applied these capabilities to FOIA request processing (FedScoop, May 2024 — Microsoft-sponsored content; Partnership for Public Service case study).
- **US Federal FOIA/PA**: AI-assisted redaction (proof of concept with Hyperscience) achieved 188% productivity increase — 575 pages per hour compared to 200 pages manually — across 4,429 pages of service treatment records processed by three data keyers (Hyperscience blog, 2023).
- **Relativity aiR for Review**: Generally available since September 2024, this generative AI document review solution produces document citations with associated rationale and recommendations. Announced at Relativity Fest (October 2025) for inclusion in standard RelativityOne pricing from early 2026, demonstrating maturity of LLM-assisted review for eDiscovery and government use cases.

## Human-in-the-Loop Requirements

These requirements are non-negotiable for democratic accountability:

- All withholding decisions require human sign-off by a delegated authority
- PII redaction suggestions must be reviewed by an analyst before application
- Response letters must be approved by the statutory decision-maker
- AI provides recommendations with confidence scores; humans decide
- Override mechanisms must be available at every stage
- Audit trail must record where AI recommendations were accepted or rejected
- Ombudsman must be able to review the decision-making process including AI inputs

## Privacy and Sovereignty

**Data residency**: All data must remain within the relevant jurisdiction. For New Zealand, this means Azure New Zealand North (launched December 2024), AWS Asia Pacific New Zealand (ap-southeast-6, launched September 2025), or on-premises deployment. Cross-border data flows of request content are not acceptable.

**New Zealand Privacy Act 2020**: Information Privacy Principle 12 governs disclosure of personal information to foreign persons or entities. OIA processing systems must not transmit request content to overseas AI endpoints unless appropriate safeguards exist.

**Algorithm Charter for Aotearoa NZ**: Requires transparency about algorithmic decision-making, accountability for outcomes, consideration of Te Ao Maori perspectives, and public engagement on high-impact systems.

**Technical controls**: Confidential computing (e.g., Azure Confidential VMs, AMD SEV-SNP) for sensitive workloads. Bring-your-own-key (BYOK) encryption ensures government retains cryptographic control. Private endpoints eliminate public internet exposure of API traffic.

## Open-Source Stack

| Component | Open-Source Option | Licence |
|---|---|---|
| PII detection | Microsoft Presidio | MIT |
| Document processing | Unstructured.io, Apache Tika | Apache 2.0 |
| RAG framework | LangChain, LlamaIndex | MIT |
| Vector database | Qdrant, Milvus | Apache 2.0 |
| Vector database | Weaviate | BSD-3-Clause |
| LLM (on-premises) | Llama 4 Scout/Maverick, Mistral, Qwen | Various (Llama Community Licence, Apache 2.0) |
| Workflow orchestration | Prefect, Apache Airflow | Apache 2.0 |
| Workflow orchestration | Temporal | MIT |
| Search engine | OpenSearch | Apache 2.0 |
| OCR | Tesseract | Apache 2.0 |
| OCR | Surya | GPL-3.0 |

On-premises deployment using approved open-weight LLMs can satisfy data sovereignty requirements without cloud dependency. Treat model adequacy for classification and legal-reasoning assistance as an empirical question; validate on the agency's own historical decisions before production use.

## Managed Services

| Capability | Azure | AWS | GCP |
|---|---|---|---|
| Document extraction | Document Intelligence | Textract | Document AI |
| PII detection | Sensitive Data Protection | Comprehend, Macie | Sensitive Data Protection (DLP) |
| LLM inference | Azure OpenAI Service | Bedrock | Vertex AI |
| Vector search | AI Search | OpenSearch Serverless | AlloyDB, Vertex AI Search |
| Object storage | Blob Storage | S3 | Cloud Storage |
| Workflow | Logic Apps, Durable Functions | Step Functions | Cloud Workflows |

## Scale Guidance

### Small Agencies (<500 requests/year, <5 staff handling OIA)

- Deploy document discovery (RAG) and response drafting only — skip automated sensitivity classification and PII detection.
- Use the agency's existing case management system (SharePoint lists, Voco Digital, or even Excel) for workflow tracking.
- A single Azure Function App with AI Search + Azure OpenAI is sufficient.
- Manual redaction with PDF tools remains practical at this volume; AI-assisted redaction adds complexity without commensurate benefit.
- Focus investment on document discovery: even simple RAG over prior responses dramatically reduces duplicate work.
- Total cost: $300-500/month for AI services.

### Large Agencies (5,000+ requests/year, dedicated OIA teams)

- Full pipeline as described: all 8 stages with AI assistance and human checkpoints.
- Automated PII detection (Presidio or commercial) becomes essential at scale — manual scanning of thousands of pages is unsustainable.
- Sensitivity classification model trained on the agency's historical decisions — requires 200+ labelled prior decisions for acceptable accuracy.
- Integration with enterprise records management (EDRMS) and email archiving systems.
- Dedicated OIA processing environment with restricted access and audit logging.
- Consider on-premises LLM deployment for agencies handling Restricted-classified material.
- Real-time compliance dashboard tracking statutory deadlines across all active requests.
- Total cost: $3,000-8,000/month depending on volume, model usage, and infrastructure choices.

## Existing Platforms in Market

- **New Zealand**: Voco Digital (Power Platform/Dynamics 365 based), Microsoft NZ AI solutions for central and local government
- **United States**: NextRequest (CivicPlus), GovQA (Granicus), Tyler Technologies FOIA Management, MuckRock (requestor-side)
- **United Kingdom**: Civica Information Governance (powered by iCasework), eCase (Fivium)
- **Australia**: CGI Freedom of Information Solution, Objective FOI (purpose-built FOI management software), Resolve Software Group (complaints and case management)

## Regulatory Framework

This architecture is designed for New Zealand but adapts to any FOI jurisdiction by substituting the relevant legislation:

| Legislation | Jurisdiction | Key Provisions |
|---|---|---|
| Official Information Act 1982 | NZ | 20-day deadline, Sections 6/7/9 withholding grounds |
| Local Government Official Information and Meetings Act 1987 | NZ (local govt) | Mirrors OIA for councils |
| Privacy Act 2020 | NZ | IPP 12 cross-border, IPP 6 access rights |
| Algorithm Charter for Aotearoa NZ | NZ | Transparency, accountability, Te Ao Maori |
| Proactive release (Cabinet direction) | NZ | Ministers must proactively release Cabinet material |
| Freedom of Information Act 2000 | UK | 20-day deadline, absolute and qualified exemptions |
| Freedom of Information Act 1982 | Australia (Cth) | 30-day deadline, conditional exemptions |

The Chief Ombudsman provides compliance oversight in New Zealand, with powers to review decisions and recommend release. AI-assisted processes must maintain sufficient transparency for Ombudsman investigation.

## Technology Readiness Assessment

| Capability | Maturity | NZ Deployment Status |
|---|---|---|
| RAG / Semantic Search | Production-ready | Yes (Microsoft NZ, multiple agencies) |
| AI Redaction | Production-ready | Possible (on-premises options available) |
| Document Classification | Production-ready | Not yet OIA-specific in NZ |
| LLM Legal Reasoning | Emerging (human-in-loop required) | Early stage |
| Case Management Integration | Production-ready | Yes (Voco Digital, Microsoft) |
| End-to-end Pipeline | Emerging | Partial deployments only |

RAG-based document discovery and AI-assisted redaction are production-ready today. LLM-based legal reasoning remains in the human-assisted category — suitable for drafting rationales that humans review, not for autonomous decision-making. Full end-to-end pipelines integrating all stages are emerging but not yet deployed at scale in New Zealand.

## Engineering Readiness Pack

This pipeline is a high-accountability decision-support system. It is not engineering-ready until evidence, legal review, human approval, redaction QA, and auditability are specified as product requirements.

### Evidence and claim ledger

Maintain a `claim-ledger.md` with one row per operational, legal, vendor, or performance claim. Required fields: `claim`, `section`, `source`, `source type`, `last verified`, `confidence`, `owner`, and `recheck trigger`.

| Claim class | Current status | Required handling |
|---|---|---|
| NZ statutory deadlines and withholding grounds | Grounded in legislation | Recheck legislation and Ombudsman guidance before jurisdiction-specific deployment. |
| OIA volume statistics | Official statistics | Update when Public Service Commission publishes new periods. |
| Vendor productivity and deployment claims | Vendor or sponsored evidence | Label as vendor evidence; do not use as acceptance criteria without agency pilot data. |
| AI legal-reasoning capability | Emerging | Keep as drafting assistance only, with delegated human decision-maker approval. |
| On-prem/open-weight model suitability | Assumption | Validate on agency-specific historical cases and redaction/classification benchmarks. |

### Implementation artefacts

Required before build:

- Jurisdiction profile files for OIA, LGOIMA, Privacy Act, proactive release, and any agency-specific policy.
- Data model for request, scope, document, retrieved chunk, relevance decision, proposed withholding ground, redaction, approval, response letter, and audit event.
- API contracts for intake, document discovery, review queue, redaction review, legal rationale drafting, response drafting, and deadline alerts.
- Integration map for SharePoint, email/archive stores, EDRMS, case-management system, identity provider, and reading-room publication.
- C4 context/container diagrams plus deployment view showing sovereign/cloud/on-prem boundaries.
- ADRs for redaction engine, RAG store, case-management integration, model hosting, and audit-log immutability.

### Anti-hallucination and legal-decision controls

The system must never invent legislation, case notes, withholding grounds, document existence, or public-interest balancing factors.

Controls:

- Retrieval-only legal rationale: every suggested withholding ground must cite the request scope, document passage, statutory provision, and any Ombudsman/case-note source used.
- Server-side citation validator that rejects invented provision IDs, document IDs, pages, or redaction codes.
- "Insufficient basis" state when the relevant legislation, guidance, or document passage is absent.
- Separate model roles for document discovery, relevance classification, redaction detection, and rationale drafting; do not let one generation step make final legal decisions.
- Human approval workflow for every relevance exclusion, withholding ground, public-interest balance, redaction, extension, transfer, and final response.
- Redaction QA that verifies irreversible removal at file-content level, not just a visual overlay.

### Threat model

| Threat | Control |
|---|---|
| Prompt injection in emails or documents | Treat source text as untrusted evidence; strip executable instructions from context and test malicious documents. |
| Over-withholding due to model conservatism | Require public-interest balancing, reviewer sign-off, and audit sampling. |
| Under-redaction of personal information | Combine ML, regex, and human review; run post-redaction extraction tests. |
| Fabricated legal rationale | Citation validation against legislation/guidance corpus and delegated authority approval. |
| Cross-border data leakage | Jurisdiction-specific deployment boundary, private endpoints, BYOK where required, and approved model endpoints only. |
| Incomplete document discovery | Hybrid search, source inventory reconciliation, and analyst certification of search completeness. |

### Evaluation and acceptance gates

Minimum eval pack before pilot:

- 50-100 historical requests with known document sets, relevance decisions, redactions, final responses, and review outcomes.
- Separate tests for discovery recall, relevance precision, PII recall, redaction irreversibility, withholding-ground classification, refusal/insufficient-basis behaviour, and deadline calculations.
- Adversarial tests containing misleading documents, contradictory precedents, hidden prompt injection, scanned/OCR material, duplicate emails, and partially relevant documents.
- Promotion gate: no fabricated source/legal citation, no irreversible-redaction failure, no missed high-risk PII in the gold set, and delegated legal owner approval for the residual error profile.

### Operational runbook

Runbooks must cover urgent request escalation, model or search outage, accidental disclosure risk, redaction failure, source-index corruption, deadline miscalculation, Ombudsman review export, data-subject/privacy incident, and rollback to manual processing.

## References

### Legislation and Policy

- Official Information Act 1982 (NZ): https://www.legislation.govt.nz/act/public/1982/0156/latest/DLM64785.html
- Privacy Act 2020 (NZ): https://www.legislation.govt.nz/act/public/2020/0031/latest/LMS23223.html
- Algorithm Charter for Aotearoa NZ: https://data.govt.nz/toolkit/data-ethics/government-algorithm-transparency-and-accountability/algorithm-charter
- Office of the Ombudsman (NZ) — OIA guidance: https://www.ombudsman.parliament.nz/resources/official-information-act-guides-and-resources
- Public Service Commission OIA Statistics: https://www.publicservice.govt.nz/guidance/official-information/oia-statistics

### Evidence Base

- Microsoft NZ, "Optimising official information management with AI" (May 2025): https://news.microsoft.com/en-nz/2025/05/09/optimising-official-information-management-with-ai/
- FedScoop, "How the State Department used AI and machine learning to revolutionize records management" (May 2024): https://fedscoop.com/how-the-state-department-used-ai-and-machine-learning-to-revolutionize-records-management/
- Partnership for Public Service, "Leadership program inspires an AI revolution at the State Department": https://ourpublicservice.org/about/history-and-impact/leadership-program-inspires-an-ai-revolution-at-the-state-department
- Hyperscience, "How FOIA/PA Offices enhance efficiency with AI-Assisted Redaction": https://www.hyperscience.ai/blog/ai-assisted-redaction-how-foia-pa-offices-can-increase-throughput-reduce-over-under-redaction/
- Relativity aiR for Review GA announcement (September 2024): https://www.relativity.com/blog/a-year-of-air-reflecting-on-2024-and-what-lies-ahead/
- CaseGuard Studio: https://caseguard.com/how-it-works/

### Open-Source Tools

- Microsoft Presidio: https://github.com/microsoft/presidio
- Unstructured.io: https://github.com/Unstructured-IO/unstructured
- LangChain: https://github.com/langchain-ai/langchain
- LlamaIndex: https://github.com/run-llama/llama_index
- Qdrant: https://github.com/qdrant/qdrant
- Weaviate: https://github.com/weaviate/weaviate
- Apache Tika: https://tika.apache.org/
- Surya OCR: https://github.com/datalab-to/surya
- Prefect: https://github.com/PrefectHQ/prefect
- Temporal: https://github.com/temporalio/temporal

### Platforms

- Voco Digital (NZ): https://www.voco.digital/nz/
- CivicPlus NextRequest: https://www.civicplus.com/nextrequest-public-records-software/
- Granicus GovQA: https://granicus.com/product/records-request-management-govqa/
- Tyler Technologies FOIA: https://www.tylertech.com/solutions/courts-public-safety/courts-justice/investigations-audits/FOIA
- Civica Information Governance (iCasework): https://www.civica.com/en-gb/product-pages/case-management-software/information-governance-foi-request-handling-software/
- eCase (Fivium): https://www.ecase.co.uk/platform/
- Objective FOI (Australia): https://www.objective.com.au/solutions/freedom-of-information-software
- CGI Freedom of Information Solution (Australia): https://www.cgi.com/au/en-au/freedom-of-information-solution
- NIST AI Risk Management Framework and Generative AI Profile: https://www.nist.gov/itl/ai-risk-management-framework
- OWASP Top 10 for LLM Applications 2025: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- OpenAI Evals and graders documentation: https://platform.openai.com/docs/guides/evals
- New Zealand Privacy Commissioner, Information Privacy Principle 12: https://www.privacy.org.nz/privacy-act-2020/privacy-principles/12/
