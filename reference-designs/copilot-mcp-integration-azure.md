# Microsoft 365 Copilot Enterprise Backend Integration via Microsoft Foundry and MCP

| Field | Value |
|-------|-------|
| Created | 2026-05-27 |
| Last Updated | 2026-05-30 |
| Version | 1.3 |

---

- [Component Maturity Status](#component-maturity-status)
- [Overview](#overview)
- [High-Level Architecture](#high-level-architecture)
- [Identity Architecture](#identity-architecture)
- [Integration Patterns](#integration-patterns)
- [Integration Pattern Decision Matrix](#integration-pattern-decision-matrix)
- [Cross-System Identity Mapping](#cross-system-identity-mapping)
- [MCP Server Design](#mcp-server-design)
- [Caching Strategy](#caching-strategy)
- [Security Model](#security-model)
- [Environment Isolation](#environment-isolation)
- [Scale Guidance](#scale-guidance)
- [Open-Source Alternatives](#open-source-alternatives)
- [Engineering Readiness Pack](#engineering-readiness-pack)
- [References](#references)

## Component Maturity Status

This architecture combines components at different stages of the Azure release lifecycle. The table below is a planning checklist as of 2026-05-30; production plans must verify each status against Microsoft Learn, Azure Updates, and the target tenant/region before procurement or implementation.

| Component | Status | Notes |
|-----------|--------|-------|
| Microsoft Foundry Agent Service | Verify official availability | Use only regions, SDK versions, and networking modes supported in Microsoft Learn for the target subscription. |
| Foundry REST API and SDKs | Verify official availability | Pin SDK versions in implementation ADRs and CI. |
| Foundry Agent Identity / Entra integration | Verify official availability | Confirm token audience, consent model, and RBAC flow before designing OBO. |
| Azure Functions MCP hosting | Verify official availability | Confirm supported triggers, transport, auth behaviour, cold-start profile, and regional availability. |
| Azure API Management — MCP server exposure | Preview/availability must be verified | Treat APIM MCP features as change-prone unless Microsoft publishes GA/SLA status for the target tier. |
| Azure API Center — MCP server registry | Verify official availability | Use as registry/discovery only after governance model is approved. |
| Foundry MCP Server (cloud-hosted) | Preview/availability must be verified | Appropriate for development only unless Microsoft publishes GA/SLA status and enterprise controls match requirements. |
| Microsoft 365 Copilot — MCP connector support | Tenant rollout must be verified | Check Microsoft 365 roadmap/admin centre for the exact tenant before committing to this channel. |
| Microsoft Entra Connect Sync (V2) | Established Microsoft identity component | Validate hybrid identity topology, sync scope, and deprecation notices. |
| Azure Container Apps | Established Azure runtime | Recommended default for production MCP servers when autoscaling, probes, and VNet controls are required. |
| Azure Cache for Redis / compatible cache | Established Azure cache component | Use only after cache-staleness risks are accepted for each permission type. |

**Practical implication:** Treat Container Apps or Functions-hosted MCP servers as the conservative production path when they satisfy the official service documentation and enterprise controls. Treat APIM MCP and cloud-hosted MCP features as gated dependencies until their support, SLA, and tenant availability are verified.

## Overview

This reference architecture describes a pattern for integrating Microsoft 365 Copilot with enterprise backend systems — CRM, legacy ledgers, data platforms, and third-party APIs — using Microsoft Foundry (formerly Azure AI Foundry) as the agent orchestration layer, MCP (Model Context Protocol) servers as the integration abstraction, and Azure API Management (APIM) as the security and governance gateway.

The architecture addresses a common enterprise reality: multiple backend systems with heterogeneous authentication models, inconsistent identity schemes, and varying levels of API maturity. Rather than forcing a single integration pattern, it provides three composable approaches selected per-backend based on their capabilities.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  User (Microsoft 365)                                               │
│  ┌──────────────────┐                                               │
│  │ Microsoft 365    │                                               │
│  │ Copilot          │                                               │
│  └────────┬─────────┘                                               │
│           │ (Entra ID SSO token)                                    │
├───────────┼─────────────────────────────────────────────────────────┤
│  Microsoft Foundry                                                  │
│  ┌────────▼─────────┐                                               │
│  │ Agent            │  Tool-calling / function invocation            │
│  │ (Orchestrator)   │                                               │
│  └────────┬─────────┘                                               │
│           │                                                         │
├───────────┼─────────────────────────────────────────────────────────┤
│  Azure API Management                                               │
│  ┌────────▼─────────┐                                               │
│  │ Gateway          │  JWT validation, rate limiting, routing        │
│  │                  │  MCP endpoint routing, tool discovery          │
│  └────────┬─────────┘                                               │
│           │                                                         │
├───────────┼─────────────────────────────────────────────────────────┤
│  MCP Server Layer (per-backend)                                     │
│  ┌────────▼──┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ MCP: CRM  │  │ MCP:     │  │ MCP:     │  │ MCP:     │          │
│  │ (OBO)     │  │ Legacy   │  │ Data Plat│  │ External │          │
│  │           │  │ (ACL)    │  │ (Gateway)│  │ (Gateway)│          │
│  └─────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│        │             │              │              │                 │
├────────┼─────────────┼──────────────┼──────────────┼────────────────┤
│  Backend Systems                                                    │
│  ┌─────▼────┐  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐          │
│  │ Dynamics │  │ Legacy   │  │ BI/Data  │  │ External │          │
│  │ CRM      │  │ Ledger   │  │ Platform │  │ APIs     │          │
│  │ (IaaS)   │  │ (SOAP)   │  │          │  │          │          │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

The key design principle: each MCP server encapsulates the integration complexity for its backend, exposing a uniform tool interface to the AI agent while handling auth, identity mapping, and response filtering internally.

## Identity Architecture

A common pattern in enterprise environments is "on-premises" infrastructure that has been lift-and-shifted to Azure IaaS. This simplifies networking (VNet peering rather than site-to-site VPN) while retaining the original application stack.

```
┌──────────────┐   Microsoft Entra Connect  ┌──────────────┐
│  On-Prem AD  │ ──────────────────────────▶│  Entra ID    │
│  (DC on IaaS)│     (sync + hybrid join)   │  (Cloud IdP) │
└──────────────┘                            └──────┬───────┘
                                                   │
                                    ┌──────────────┼──────────────┐
                                    │              │              │
                              ┌─────▼────┐  ┌─────▼────┐  ┌─────▼────┐
                              │ M365     │  │ Microsoft│  │ App Regs │
                              │ Copilot  │  │ Foundry  │  │ (MCP)    │
                              └──────────┘  └──────────┘  └──────────┘
```

**Key points:**

- The domain controller runs on Azure IaaS VMs, synchronised to Entra ID via Microsoft Entra Connect Sync (formerly Azure AD Connect).
- Microsoft 365 Copilot authenticates the user via Entra ID and provides a bearer token (JWT) to Microsoft Foundry.
- Microsoft Foundry agents receive an Entra ID Agent Identity (auto-provisioned, GA), enabling them to participate in OAuth flows and receive RBAC assignments.
- The user's JWT propagates through the call chain where OBO is supported.

## Integration Patterns

### Pattern 1: On-Behalf-Of (OBO) Token Exchange

**Use when:** The backend supports Entra ID authentication and has its own row-level security (RLS) or entity-level access control tied to user identity.

**Flow:**

1. Microsoft Foundry receives the user's access token (audience: Foundry).
2. The MCP server exchanges this token via the OBO flow for a new token scoped to the backend API (e.g., Dynamics CRM Web API).
3. The backend enforces its native security model — the MCP server does not need to filter results.

**Advantages:** Native RLS enforced at the backend; no permission replication; audit trail shows the actual user.

**Requirements:** Backend must accept Entra ID tokens; App Registration must have OBO consent for the downstream API.

### Pattern 2: Anti-Corruption Layer (ACL)

**Use when:** The backend is a legacy system (SOAP/WSDL, J2EE, custom protocols) that cannot participate in modern OAuth flows.

**Flow:**

1. The MCP server authenticates to the legacy backend using a service account (credentials stored in Azure Key Vault).
2. Before returning results, the MCP server queries the primary system of record for permissions (e.g., calls CRM to retrieve the user's access scope).
3. The MCP server filters the legacy system's response, returning only data the user is authorised to see.

**Advantages:** No modification to the legacy backend; access control enforced consistently with the primary system.

**Trade-offs:** Additional latency for permission lookups (mitigated by caching); the service account has broad access, requiring strict network isolation and audit logging.

### Pattern 3: Existing API Gateway Passthrough

**Use when:** The backend is already exposed via an API gateway (its own APIM instance, or a third-party gateway) with established auth mechanisms (API keys, mutual TLS, or OAuth client credentials).

**Flow:**

1. The MCP server authenticates to the existing gateway using its established mechanism.
2. The MCP server trusts that the downstream gateway enforces its own access control.
3. The MCP server may still apply response shaping or field filtering, but does not replicate permission logic.

**Advantages:** Minimal new infrastructure; leverages existing investment in API governance.

**Requirements:** The existing gateway's security model must be acceptable for AI-mediated access.

## Integration Pattern Decision Matrix

| Criterion | OBO | ACL | Gateway Passthrough |
|-----------|-----|-----|---------------------|
| Backend supports Entra ID auth | Yes | No | N/A |
| Backend has native RLS | Yes | No | Varies |
| Legacy protocol (SOAP, RPC) | No | Yes | Possible |
| Backend already behind API gateway | No | No | Yes |
| Need user-level audit at backend | Yes | No (service account) | Depends on gateway |
| Permission model matches primary system | Inherent | Must be enforced | Delegated |
| Implementation complexity | Low | High | Low |
| Latency overhead | Low (token exchange) | Medium (permission lookup) | Low |

**Decision heuristic:** Start with OBO where possible. Fall back to ACL for legacy systems. Use Gateway Passthrough for backends that already have mature API governance.

## Cross-System Identity Mapping

Enterprise backends rarely share a common entity identifier. A **Master Identity Service** resolves this by maintaining a mapping table across systems.

**Implementation:**

- **Storage:** Azure SQL Database (or PostgreSQL) with a mapping table: `(entity_type, system_code, system_id, canonical_id)`.
- **API:** A lightweight REST API (Azure Functions or Container Apps) exposed via APIM.
- **Usage:** MCP servers call the Identity Service to translate between system-specific IDs (e.g., CRM GUID → Legacy Ledger ID → External System Reference).

**Schema example:**

```
canonical_id | system     | system_entity_id
─────────────┼────────────┼──────────────────
uuid-001     | crm        | {CRM-GUID}
uuid-001     | ledger     | ACC-12345
uuid-001     | external   | EXT-REF-789
```

This service is called by MCP servers during request processing and is cached aggressively (entity mappings change infrequently).

## MCP Server Design

Each backend system gets a dedicated MCP server. The MCP server is responsible for:

1. **Auth pattern selection** — Implements OBO, ACL, or gateway passthrough as appropriate for its backend.
2. **Permission enforcement** — For ACL patterns, queries and caches the user's access scope.
3. **Identity mapping** — Translates entity IDs between the AI agent's context (typically CRM-centric) and the backend's native identifiers.
4. **Response filtering** — Removes fields or records the user is not authorised to see.
5. **Tool definition** — Exposes a well-defined set of MCP tools (functions) that the AI agent can invoke.

**Deployment:** MCP servers run as Azure Container Apps when production workloads require auto-scaling, health probes, VNet integration, and explicit runtime control. Azure Functions can be used where its MCP hosting, authentication, transport, and cold-start behaviour are officially supported for the workload. Both options can be registered behind APIM or the selected gateway when that gateway pattern is validated.

**APIM responsibilities (in front of MCP servers):**

> **Note:** Verify APIM MCP feature status in Microsoft Learn and Azure Updates before production use. If the feature is still preview for the target tenant/tier, plan for policy schema or routing changes and keep a direct-to-MCP fallback.

- JWT validation (ensures the caller is the authorised Microsoft Foundry agent).
- Rate limiting and throttling per-agent and per-user.
- Request/response logging for audit.
- MCP server endpoint routing (`/mcp` for Streamable HTTP transport).
- Tool discovery via Azure API Center (centralised MCP server registry — GA).
- Routing to the appropriate MCP server based on tool namespace.

## Caching Strategy

Redis (Azure Cache for Redis) provides low-latency caching for frequently-accessed, slowly-changing data:

| Data Type | Cache Duration | Rationale |
|-----------|---------------|-----------|
| User permission scopes | 30–60 minutes | Permissions change infrequently; acceptable staleness window for ACL enforcement |
| Entity ID mappings | 4–24 hours | Cross-system IDs are essentially static once created |
| Advisor/role codes | 30–60 minutes | Organisational assignments change daily at most |
| Portfolio/account mappings | 15–30 minutes | Near-real-time requirements for financial data |
| Real-time transactional data | No cache (or 1–5 min) | Must reflect current state |

**Cache invalidation:** Event-driven invalidation via Azure Service Bus where backends support change notifications. Otherwise, TTL-based expiry with the durations above.

**Cache key design:** `{user_oid}:{system}:{scope_type}` for permission data; `{canonical_id}:{system}` for identity mappings.

## Security Model

Enterprise environments commonly have multiple coexisting security models:

- **CRM security model** — Role-based with business unit hierarchy, team membership, and record-level sharing.
- **Data platform security model** — Attribute-based or group-based, often more permissive for analytics use cases.
- **Legacy system security model** — Application-level roles, often coarse-grained.

**Principle:** The AI agent should align with the security model most appropriate for the use case. For customer-facing advisory tasks, the CRM security model (most restrictive, most granular) is typically the correct choice.

**Implementation guidance:**

- Define explicitly which security model governs each MCP server's access decisions.
- Where models conflict, default to the most restrictive.
- Document security model alignment decisions in the MCP server's configuration, not in ad-hoc code.
- Use Conditional Access policies in Entra ID to restrict which users can access AI-mediated interfaces during rollout.

## Environment Isolation

| Environment | Entra ID Tenant | App Registrations | Conditional Access | Data |
|-------------|-----------------|-------------------|--------------------|------|
| Development | Shared tenant | Dev-specific App Regs | Dev user group only | Synthetic/anonymised |
| Test/UAT | Shared tenant | Test-specific App Regs | Test user group only | Subset of production (masked) |
| Production | Shared tenant | Prod-specific App Regs | Prod user group + MFA | Live data |

**Key controls:**

- Separate App Registrations per environment ensure tokens cannot cross environment boundaries.
- Conditional Access policies restrict which Entra ID groups can obtain tokens for each environment's App Registrations.
- Network isolation: each environment's MCP servers and backends reside in separate VNets (or subnets with NSGs).
- Secrets (service account credentials, API keys) are stored in per-environment Key Vaults with RBAC scoped to environment-specific managed identities.

## Scale Guidance

### Small Projects (1-2 backends, single team, POC)

- Deploy MCP servers as Azure Functions (consumption plan) rather than Container Apps.
- Skip APIM for the POC when the Foundry-to-MCP path is officially supported for the target tenant. Add APIM or another gateway when moving to production if governance, rate limiting, observability, and discovery requirements justify it.
- Use a single App Registration for dev/test; separate only for production.
- Skip the Master Identity Service if all backends share a common identifier (e.g., email address). Implement only when cross-system ID mapping is genuinely needed.
- Cache in-memory (application-level dictionary) rather than standing up Redis.
- A single MCP server can serve multiple backends if the integration patterns are identical.

### Large Projects (5+ backends, multiple teams, enterprise)

- Deploy MCP servers as Container Apps with auto-scaling and health probes per backend.
- APIM is mandatory — centralises JWT validation, rate limiting, observability, and tool discovery.
- Implement full environment isolation with per-environment Key Vaults, VNets, and Conditional Access.
- Stand up the Master Identity Service early — cross-system queries become the norm, not the exception.
- Consider MCP server versioning (v1/v2 endpoints) to enable backend upgrades without breaking the agent.
- Dedicated Redis cluster with geo-replication for multi-region deployments.
- Implement circuit breakers in MCP servers for graceful degradation when backends are unavailable.

## Open-Source Alternatives

| Component | Azure Managed | Open-Source Alternative |
|-----------|---------------|------------------------|
| Identity Provider | Entra ID | Keycloak, Authentik |
| API Gateway | Azure API Management | Kong Gateway, Tyk, Apache APISIX |
| Agent Orchestration | Microsoft Foundry | LangGraph, CrewAI, AutoGen (open-source) |
| MCP Server Runtime | Azure Container Apps | Any HTTP server implementing MCP spec (Node.js, Python, Go SDKs) |
| Cache | Azure Cache for Redis | Redis OSS, Valkey, Dragonfly |
| Identity Mapping DB | Azure SQL | PostgreSQL, YugabyteDB |
| Secret Management | Azure Key Vault | HashiCorp Vault |
| Message Bus | Azure Service Bus | RabbitMQ, Apache Kafka |
| Monitoring | Azure Monitor + App Insights | OpenTelemetry + Grafana + Prometheus |

**Cross-cloud equivalents:**

- **AWS:** Amazon Bedrock Agents + Amazon API Gateway + Amazon ElastiCache + Amazon Cognito.
- **GCP:** Vertex AI Agent Builder + Apigee + Memorystore + Cloud Identity.
- **IBM:** watsonx.ai + IBM API Connect + IBM Cloud Databases for Redis.

The MCP specification itself is open (created by Anthropic, released as an open standard). Server implementations exist in Python (`mcp` package on PyPI), TypeScript (`@modelcontextprotocol/sdk` on npm), C# (`ModelContextProtocol` on NuGet), Java, Go, and Rust. The protocol uses JSON-RPC 2.0 for messaging over Streamable HTTP (the current recommended transport) or the deprecated SSE transport. No vendor lock-in exists at the protocol level — any compliant MCP server can be exposed through this architecture regardless of the language it is implemented in.

## Engineering Readiness Pack

This design is engineering-ready only after identity, tool contracts, permission enforcement, and preview-service dependencies are specified and tested.

### Evidence and claim ledger

Maintain a `claim-ledger.md` with `claim`, `section`, `source`, `source type`, `last verified`, `confidence`, `owner`, and `recheck trigger`.

| Claim class | Current status | Required handling |
|---|---|---|
| Microsoft service maturity | Highly time-sensitive | Verify from Microsoft Learn/Azure Updates/admin centre before architecture sign-off. |
| Copilot tenant rollout | Tenant-specific | Confirm in the customer's Microsoft 365 admin centre and roadmap before commitment. |
| MCP protocol details | Open specification | Pin protocol version and transport in ADRs. |
| OBO and identity mapping behaviour | Implementation-specific | Validate with real app registrations, scopes, token audiences, and conditional access policies. |
| Cache TTLs | Design assumptions | Security owner must accept staleness windows for each permission class. |

### Implementation artefacts

Required before build:

- C4 context/container diagrams showing Copilot, Foundry/agent runtime, gateway, MCP servers, identity service, cache, Key Vault, and each backend.
- Tool catalog with tool name, schema, owning team, backend, auth pattern, read/write scope, rate limit, data classification, side-effect level, and rollback behaviour.
- Entra app-registration plan with scopes, audiences, consent model, managed identities, conditional-access policy, and environment separation.
- Permission-decision record format: user, route/tool, backend, requested entity, policy source, decision, cache hit/miss, and source timestamp.
- ADRs for OBO vs ACL per backend, gateway choice, cache staleness, MCP runtime, and preview-service fallback.
- Contract tests for every MCP tool using realistic tokens and least-privilege accounts.

### Anti-hallucination and tool-safety controls

The agent must not infer permissions, fabricate tool results, or silently degrade into broader access.

Controls:

- Tool schemas with strict input validation and server-side defaults; user/model text cannot select hidden fields or broaden filters.
- Server-side permission enforcement in MCP servers; never rely on the model to filter unauthorised records.
- Tool response provenance: every answer that uses backend data carries tool call ID, backend record IDs, and timestamp in audit metadata.
- "No result / not authorised / backend unavailable" states are distinct and must not be rewritten by the model as business facts.
- Prompt-injection tests through backend text fields and external API responses.
- Circuit breakers and policy-deny defaults for backend errors, cache misses, identity-service outage, and token-exchange failure.

### Threat model

| Threat | Control |
|---|---|
| Excessive agency via tool overreach | Per-tool side-effect classification, allowlists, approval gates for writes, and rate limits. |
| Permission bypass in ACL pattern | Deterministic server-side filtering, cache staleness limits, and audit sampling. |
| Token confusion/audience mismatch | Explicit token-audience validation and integration tests for every OBO flow. |
| Prompt injection from backend records | Treat backend content as untrusted data; never execute instructions returned by tools. |
| Preview gateway behaviour changes | Pin configuration, monitor Azure Updates, and maintain direct MCP fallback. |
| Cross-environment token/data leakage | Separate app registrations, Key Vaults, VNets/subnets, and diagnostics per environment. |

### Evaluation and acceptance gates

Minimum gate before production:

- Contract tests for all MCP tools, including invalid schema, missing scopes, wrong audience, expired token, and conditional-access denial.
- Permission tests for OBO, ACL, and gateway patterns using users from multiple roles/business units.
- Adversarial tests where backend data attempts to instruct the agent to reveal records or call different tools.
- Load tests for p95 latency, backend timeout handling, cache effectiveness, and rate limiting.
- Audit reconstruction test: given a final Copilot answer, engineers can reconstruct each tool call, permission decision, source record, and human/user action.

### Operational runbook

Runbooks must cover backend outage, token-exchange failure, identity mapping mismatch, permission-cache flush, MCP server rollback, gateway preview breaking change, tool schema versioning, compromised service account, and emergency tool disablement.

## References

- Microsoft Learn — "Overview of MCP servers in Azure API Management" (2025). https://learn.microsoft.com/en-us/azure/api-management/mcp-server-overview
- Microsoft Learn — "Connect to MCP Server Endpoints for agents" (2025). https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/tools/model-context-protocol
- Microsoft Learn — "Agent identity concepts in Microsoft Foundry" (2025). https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/agent-identity
- Microsoft Learn — "Microsoft identity platform and OAuth 2.0 On-Behalf-Of flow" (2025). https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-on-behalf-of-flow
- Microsoft Learn — "Introduction to Microsoft Entra Connect V2" (2025). https://learn.microsoft.com/en-us/entra/identity/hybrid/connect/whatis-azure-ad-connect-v2
- Microsoft Dev Blogs — "Announcing Foundry MCP Server (preview)" (2025). https://devblogs.microsoft.com/foundry/announcing-foundry-mcp-server-preview-speeding-up-ai-dev-with-microsoft-foundry/
- Microsoft Dev Blogs — "What's new in Microsoft Foundry | Dec 2025 & Jan 2026" (2026). https://devblogs.microsoft.com/foundry/whats-new-in-microsoft-foundry-dec-2025-jan-2026/
- Microsoft Dev Blogs — "What's new in Microsoft Foundry | March 2026" (2026). https://devblogs.microsoft.com/foundry/whats-new-in-microsoft-foundry-mar-2026/
- Model Context Protocol specification. https://modelcontextprotocol.io
- Azure MCP Server Registry (live example via Azure API Center). https://mcp.azure.com
- NIST AI Risk Management Framework and Generative AI Profile. https://www.nist.gov/itl/ai-risk-management-framework
- OWASP Top 10 for LLM Applications 2025. https://owasp.org/www-project-top-10-for-large-language-model-applications/
- OpenAI Evals and graders documentation. https://platform.openai.com/docs/guides/evals
