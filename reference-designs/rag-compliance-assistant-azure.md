# RAG Compliance Assistant (Small-Scale Azure)

| Field | Value |
|-------|-------|
| Created | 2026-05-27 |
| Last Updated | 2026-05-30 |
| Version | 1.2 |

---

- [Scope and Positioning](#scope-and-positioning)
- [Architecture Overview](#architecture-overview)
- [RAG Pipeline](#rag-pipeline)
- [Citation Model](#citation-model)
- [Copy-to-Clipboard UX](#copy-to-clipboard-ux)
- [DeepEval Quality Assurance Pipeline](#deepeval-quality-assurance-pipeline)
- [Security Architecture](#security-architecture)
- [Cost Estimate](#cost-estimate)
- [Open-Source Alternatives](#open-source-alternatives)
- [Cross-Platform Managed Service Mappings](#cross-platform-managed-service-mappings)
- [When to Use This Design](#when-to-use-this-design)
- [Assumptions](#assumptions)
- [Engineering Readiness Pack](#engineering-readiness-pack)
- [References](#references)

## Scope and Positioning

This reference architecture targets compliance and advisory use cases where a small corpus of 100-500 regulatory or policy documents must be searchable via natural language. Advisors ask questions and receive cited, copy-pasteable answers suitable for inclusion in customer emails or reports.

This is not a general-purpose chatbot. It is a compliance-grade knowledge assistant with a constrained domain, verifiable citations, and quality gates that block deployment when answer fidelity degrades.

The design is serverless, low-ops, and deliverable by a solo developer or two-person team. It prioritises operational simplicity over horizontal scalability.

## Architecture Overview

The Azure-native stack consists of:

| Layer | Service | Role |
|-------|---------|------|
| Frontend | Azure Static Web Apps | React SPA with MSAL.js for Entra ID authentication |
| Backend | Azure Functions (Node.js/TypeScript) | Serverless API — chat, ingestion, admin endpoints |
| Generation | Azure OpenAI / Microsoft Foundry Models (GPT-4o) | Answer synthesis with inline citations |
| Embeddings | Azure OpenAI (text-embedding-3-small) | 1536-dimension vectors for semantic search |
| Search | Azure AI Search | Hybrid retrieval: BM25 keyword + vector, semantic ranker for re-ranking |
| Document Processing | Azure AI Document Intelligence | Layout-aware extraction preserving headings, tables, page numbers |
| Storage | Azure Blob Storage | Source documents (PDF, DOCX) |
| Database | Azure Cosmos DB (Serverless) | Conversation history, analytics, feedback |
| Identity | Microsoft Entra ID | MSAL.js with authorisation code + PKCE flow |

All services communicate over private endpoints or managed identity where supported. The architecture has no always-on compute — costs scale to zero during idle periods.

> **Note:** As of early 2026, Microsoft has consolidated Azure AI services under the "Microsoft Foundry" branding. Azure OpenAI Service is now accessed via the Foundry portal (ai.azure.com) and documentation has moved accordingly. The underlying APIs and service names remain functionally equivalent for deployment purposes.

## RAG Pipeline

### Query Path

1. User types a compliance question in the React SPA.
2. Frontend calls the Chat API (Azure Function) with the question and conversation context.
3. The function constructs a hybrid query against Azure AI Search — BM25 keyword matching combined with vector similarity on the embedded question.
4. AI Search returns the top 5-10 chunks, each carrying metadata (document name, page number, section heading, blob URL).
5. The function assembles a system prompt constraining the model to the compliance domain, appends retrieved chunks as numbered context blocks, and sends the request to Azure OpenAI GPT-4o.
6. GPT-4o returns an answer with inline citation markers referencing the numbered chunks.
7. The function maps citation markers to structured metadata and returns the response.
8. The frontend renders the answer with clickable citations and copy buttons.

### Document Ingestion Path

1. An authorised user uploads a document via the admin UI or drops it into the designated Blob container.
2. A Blob-triggered Azure Function sends the document to Document Intelligence (layout model) for structure-aware extraction.
3. The extracted text is chunked at 500-800 tokens with 100-token overlap, preserving section boundaries where possible.
4. Each chunk is embedded via text-embedding-3-small.
5. Chunks are upserted into Azure AI Search with metadata fields: `documentName`, `pageNumber`, `sectionHeading`, `blobUrl`, `chunkText`, `embedding`.

A full re-index of 500 documents completes in under 30 minutes and costs approximately $2-5 NZD in Document Intelligence and embedding tokens.

## Citation Model

Each citation returned by the API carries:

```json
{
  "documentName": "Privacy Act 2020 - Guidance.pdf",
  "pageNumber": 14,
  "sectionHeading": "Principle 6 - Access",
  "blobUrl": "https://storage.blob.core.windows.net/docs/privacy-act-guidance.pdf",
  "relevanceScore": 0.92,
  "snippetText": "An agency that holds personal information..."
}
```

The frontend renders citations as: *See [Privacy Act 2020 - Guidance], Section "Principle 6 - Access", Page 14* — with a clickable link to the source PDF (SAS-protected, short-lived, read-only).

## Copy-to-Clipboard UX

Two copy actions are presented alongside every answer:

- **Copy Answer** — plain text of the generated response, suitable for pasting into an email body.
- **Copy with Citations** — the answer text followed by a formatted references block listing each cited document, section, and page number.

This is critical for advisory workflows where compliance officers paste answers directly into customer correspondence. The copy function uses the Clipboard API with a fallback to `execCommand` for older browsers.

## DeepEval Quality Assurance Pipeline

A separate Python project sits alongside the TypeScript application, responsible for offline evaluation of answer quality.

### Test Dataset

30-50+ curated question-answer-source triplets derived from real compliance scenarios. Each triplet contains: the question, the expected answer (or acceptable answer range), and the source chunks that should be retrieved.

### Metrics and Thresholds

| Metric | Threshold | What It Measures |
|--------|-----------|------------------|
| Faithfulness | > 0.90 | Answer is grounded in retrieved context |
| Answer Relevancy | > 0.85 | Answer addresses the question asked |
| Contextual Precision | > 0.85 | Retrieved chunks are relevant to the question |
| Contextual Recall | > 0.80 | All necessary information is retrieved |
| Hallucination | < 0.10 | Answer does not fabricate claims |

### Execution Model

- Runs offline against the Chat API endpoint (staging environment), not in the production request path.
- Triggered in CI/CD on every pull request that modifies prompts, chunking logic, or search configuration.
- Deployment is blocked if any metric falls below its threshold.
- Cost per run: approximately $0.50-2.00 NZD in judge LLM tokens across 30-50 test cases.

### CI/CD Integration

```yaml
# GitHub Actions excerpt
- name: Run DeepEval suite
  run: |
    cd eval/
    pip install deepeval
    deepeval test run test_compliance_rag.py
```

Thresholds are defined within the test file itself on each metric object (e.g., `FaithfulnessMetric(threshold=0.9)`). A failing threshold causes the test to fail, which blocks the CI pipeline.

A failing evaluation produces a detailed report showing which test cases degraded and on which metrics, enabling targeted prompt or retrieval tuning.

## Security Architecture

- **Authentication**: Entra ID with MSAL.js using authorisation code flow with PKCE. No client secrets in the SPA.
- **Authorisation**: Azure Functions validate the JWT on every request. App roles defined in the Entra ID app registration control access tiers (e.g., `Compliance.Reader`, `Compliance.Admin`).
- **Blob access**: Azure Functions use managed identity to read documents. Frontend receives short-lived, read-only SAS tokens scoped to individual blobs for citation links.
- **Encryption**: All data encrypted at rest (Azure Storage Service Encryption, Cosmos DB encryption). TLS 1.2+ enforced for all transit.
- **Domain constraint**: The system prompt explicitly restricts responses to the compliance domain. This is a guardrail, not a guarantee; out-of-scope handling must also be tested with evals and monitored in production.
- **Audit**: All queries and responses logged to Cosmos DB with user identity for compliance audit trails.

## Cost Estimate

Monthly production costs in NZD at moderate usage (100-500 queries/day):

| Service | Tier | Monthly Cost (NZD) |
|---------|------|-------------------|
| Static Web Apps | Free | $0 |
| Azure Functions | Consumption | $15-30 |
| Azure OpenAI (GPT-4o + embeddings) | Pay-as-you-go | $80-200 |
| Azure AI Search | Basic (1 replica) | ~$120 |
| Document Intelligence | S0 | $2-5 |
| Blob Storage | LRS | ~$5 |
| Cosmos DB | Serverless | $10-25 |
| **Total** | | **$230-385** |

The dominant cost driver is Azure OpenAI token usage. Costs scale linearly with query volume. At very low usage (<50 queries/day), total costs drop to $150-200/month.

## Open-Source Alternatives

Every component in this architecture has an open-source equivalent:

| Layer | Open-Source Option |
|-------|-------------------|
| Frontend | Any React, Vue, or Svelte SPA |
| Backend | FastAPI (Python), Express (Node.js) |
| LLM | Llama 4 Scout, Llama 4 Maverick, Mistral Small/Large via Ollama or vLLM |
| Embeddings | sentence-transformers, nomic-embed-text |
| Vector search | Qdrant, Weaviate, Milvus, OpenSearch with k-NN |
| Document processing | Unstructured.io, marker-pdf, Apache Tika |
| Evaluation | DeepEval (MIT licence), Ragas |
| Auth | Keycloak, Authentik |

DeepEval itself is MIT-licensed and forms the quality backbone regardless of whether the rest of the stack is Azure-managed or self-hosted.

## Cross-Platform Managed Service Mappings

| Capability | Azure | AWS | GCP |
|-----------|-------|-----|-----|
| Static hosting | Static Web Apps | Amplify | Firebase Hosting |
| Serverless compute | Functions | Lambda | Cloud Functions |
| LLM / embeddings | Azure OpenAI (Foundry Models) | Bedrock | Vertex AI |
| Search | AI Search | OpenSearch Serverless / Kendra | Vertex AI Search / AlloyDB (pgvector) |
| Document extraction | Document Intelligence | Textract | Document AI |
| Identity | Entra ID | Cognito | Firebase Auth / IAP |
| Object storage | Blob Storage | S3 | Cloud Storage |
| NoSQL database | Cosmos DB | DynamoDB | Firestore |

## When to Use This Design

**Use this design when:**

- Document corpus is under 500 documents.
- Delivery team is 1-2 people.
- Use case is compliance, advisory, or policy Q&A.
- Copy-pasteable answers for customer correspondence are a core requirement.
- Serverless, low-ops architecture is preferred.
- Quality assurance with measurable thresholds is non-negotiable.

**Use the [large-scale mixed-document-sizes reference design](rag-knowledge-base-mixed-document-sizes.md) when:**

- Document corpus exceeds 1,000 documents.
- Complex chunking strategies are needed for varied document types and sizes.
- Multi-tenant isolation is required.
- Enterprise-grade retrieval with advanced re-ranking pipelines is needed.
- A dedicated platform team is available for ongoing operations.

## Assumptions

- A Microsoft Entra ID tenant exists and the team has permissions to register applications.
- Azure OpenAI quota has been approved for the target region.
- Source documents are in machine-readable format (PDF, DOCX) — scanned documents require OCR via Document Intelligence.
- Query volume is moderate: 100-500 queries per day.
- No on-premises deployment requirement — fully cloud-hosted.
- Users access the application via modern browsers (Chrome, Edge, Firefox, Safari — latest two versions).
- Compliance documents are in English (or a single language consistent with the embedding model).

## Engineering Readiness Pack

This design is suitable for build planning only after the following engineering and verification artefacts are produced.

### Evidence and claim ledger

Maintain a `claim-ledger.md` beside the implementation. Required fields: `claim`, `section`, `source`, `source type`, `last verified`, `confidence`, `owner`, and `recheck trigger`.

| Claim class | Current status | Required handling |
|---|---|---|
| Azure service capabilities | Source-linked but time-sensitive | Verify against Microsoft Learn and regional service availability before build. |
| Cost estimate | Directional | Replace with Azure Pricing Calculator output for the target region, model, AI Search tier, and query volume. |
| Eval thresholds | Design defaults | Calibrate against the organisation's own corpus and risk tolerance before making them deployment gates. |
| Citation rendering | Implementation requirement | Validate against real PDFs, section metadata, scanned documents, and access-controlled citation links. |

### Implementation artefacts

Required before handoff to engineering:

- C4 context/container diagram showing SPA, API, Azure AI Search, Azure OpenAI/Foundry Models, Document Intelligence, Blob Storage, Cosmos DB, Entra ID, and eval runner.
- API contract for `POST /chat`, `POST /admin/documents`, `GET /citations/{id}`, `POST /feedback`, and `POST /eval/run`.
- Search-index schema including text fields, vector fields, metadata fields, filters, scoring profiles, and semantic-ranker configuration.
- Prompt-template files under version control, not embedded directly in function code.
- IaC for all Azure resources, private endpoints, managed identities, app roles, diagnostic settings, and Key Vault references.
- ADRs for AI Search vs pgvector, model choice, chunking strategy, citation-link approach, and retention policy.

### Anti-hallucination controls

The assistant must be unable to present uncited compliance advice as authoritative. Implement:

- Context-only answer policy with explicit "insufficient source support" responses.
- Citation validator that checks every cited marker maps to a retrieved chunk and every material sentence has at least one supporting source.
- Source freshness metadata for regulatory documents, including effective date, superseded status, and jurisdiction.
- Refusal tests for out-of-domain, stale-law, missing-source, and adversarial prompt-injection cases.
- Retrieval recall tests that verify the expected source appears in the candidate set before answer generation.
- UI affordance that distinguishes generated wording from source excerpts and exposes source metadata on hover/click.

### Threat model

Cover at minimum:

| Threat | Control |
|---|---|
| Prompt injection inside uploaded documents | Strip/flag instruction-like text, isolate retrieved content as data, and test with malicious corpus entries. |
| Stale or superseded regulations | Maintain effective-date metadata and block answers from superseded sources unless explicitly requested. |
| Citation fabrication | Server-side citation mapping; the model may request citation IDs but cannot invent source URLs. |
| Over-permissive document access | Entra app roles, AI Search filters, and blob SAS generation must enforce the same authorisation model. |
| Sensitive query leakage | Private endpoints, managed identities, diagnostic redaction, and retention controls for query logs. |

### Evaluation and acceptance gates

Minimum release gate:

- At least 50 curated question-answer-source triplets before pilot, expanding to 150+ for production.
- Separate test buckets for direct lookup, multi-document synthesis, no-answer refusal, stale-source refusal, prompt injection, and citation accuracy.
- No fabricated citations, no unsupported material compliance claims, contextual recall above the calibrated threshold, and human reviewer sign-off on all high-risk failures.
- CI blocks changes to prompts, chunking, search config, model versions, or citation mapping when the eval suite regresses beyond the accepted tolerance.

### Production runbook

Include procedures for source re-indexing, emergency removal of a document, model rollback, failed eval gate, citation-link expiry, AI Search outage, Azure OpenAI quota exhaustion, and privacy incident response.

## References

- [Azure AI Search — Hybrid search](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)
- [Microsoft Foundry (Azure OpenAI) documentation](https://learn.microsoft.com/en-us/azure/foundry/)
- [Azure AI Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/)
- [MSAL.js — Authorization code flow with PKCE](https://learn.microsoft.com/en-us/entra/identity-platform/scenario-spa-app-configuration)
- [Azure Static Web Apps](https://learn.microsoft.com/en-us/azure/static-web-apps/)
- [Azure Functions — Node.js developer guide](https://learn.microsoft.com/en-us/azure/azure-functions/functions-reference-node)
- [Azure Cosmos DB Serverless](https://learn.microsoft.com/en-us/azure/cosmos-db/serverless)
- [DeepEval documentation](https://docs.confident-ai.com/)
- [Ragas — RAG evaluation framework](https://docs.ragas.io/)
- [Qdrant vector database](https://qdrant.tech/documentation/)
- [MSAL overview — Microsoft Authentication Library](https://learn.microsoft.com/en-us/entra/msal/overview)
- [Unstructured.io — Document processing](https://docs.unstructured.io/)
- [marker-pdf — PDF to markdown converter](https://github.com/datalab-to/marker)
- [nomic-embed-text — Open-source embedding model](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [NIST AI Risk Management Framework and Generative AI Profile](https://www.nist.gov/itl/ai-risk-management-framework)
- [OWASP Top 10 for LLM Applications 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [OpenAI Evals and graders documentation](https://platform.openai.com/docs/guides/evals)
- [Azure Well-Architected Framework for AI workloads](https://learn.microsoft.com/en-us/azure/well-architected/ai/)
