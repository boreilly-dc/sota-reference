# DAG Workflows: Orchestration Best Practices

| Field | Value |
|-------|-------|
| Created | 2026-05-27 |
| Last Updated | 2026-05-27 |
| Version | 1.1 |

---

- [Introduction](#introduction)
- [Fact-Check and Gap Assessment](#fact-check-and-gap-assessment)
- [How to Use This Guide](#how-to-use-this-guide)
- [Core Concepts](#core-concepts)
- [Orchestration Tools Landscape](#orchestration-tools-landscape)
- [Tool Comparison Matrix](#tool-comparison-matrix)
- [Use Cases](#use-cases)
- [Reference Architectures](#reference-architectures)
- [Design Patterns](#design-patterns)
- [Production Hardening Practices](#production-hardening-practices)
- [Tool-Specific Production Checklists](#tool-specific-production-checklists)
- [Anti-Patterns](#anti-patterns)
- [Testing Strategies](#testing-strategies)
- [Observability and Lineage](#observability-and-lineage)
- [Migration and Lifecycle Management](#migration-and-lifecycle-management)
- [Managed Services on Hyperscalers](#managed-services-on-hyperscalers)
- [Modern Trends (2025–2026)](#modern-trends-20252026)
- [Decision Framework](#decision-framework)
- [Claim-Source-Confidence Matrix](#claim-source-confidence-matrix)
- [Source Quality Notes](#source-quality-notes)
- [References](#references)

---

## Introduction

A **Directed Acyclic Graph (DAG)** is the foundational abstraction for workflow orchestration in data engineering, machine learning, and modern software delivery. DAGs model work as a set of discrete tasks connected by dependency edges — enabling deterministic execution order, safe parallelism, failure isolation, and reproducibility.

Evidence markers such as **{S1}** refer to the [Claim-Source-Confidence Matrix](#claim-source-confidence-matrix). They are used for claims that are volatile, tool-specific, or likely to be challenged during architecture review.

This guide covers:

1. The theory behind DAG-based orchestration and why it dominates pipeline design.
2. A current (2026) survey of open-source and managed orchestration tools.
3. Battle-tested design patterns and anti-patterns drawn from production systems.
4. Testing, observability, and lineage practices that keep DAGs reliable at scale.

The target audience is software engineers and data engineers building production pipelines — whether batch ETL, ML training workflows, event-driven processing, or AI agent orchestration.

---

## Fact-Check and Gap Assessment

**Verification date**: 2026-05-27. Version numbers and managed-service support change frequently; treat all tool versions in this document as point-in-time checks and re-verify before procurement, migration, or production upgrades.

### Summary Assessment

The original version was directionally sound on the core DAG model, the main orchestration categories, and the highest-value engineering practices: idempotency, retries, explicit dependencies, testing, and observability. The main gaps were not conceptual; they were production-readiness gaps and a few claims that needed tighter source grounding.

### Corrections Made in This Revision

| Area | Prior issue | Correction |
|------|-------------|------------|
| Airflow Task SDK | Described as enabling multi-language task authoring. | Corrected to "Python-native interfaces" and separated that from Airflow's task execution/API architecture. The stable Task SDK docs describe Python-native DAG/task interfaces, not a general multi-language DAG authoring API. |
| Airflow version | Said "3.2.x (May 2026)" without a precise source. | Updated to Apache Airflow 3.2.1, released 2026-04-22, with "verify latest before deployment." |
| Prefect version | Said 3.7.x without distinguishing stable and nightly releases. | Updated to stable 3.7.2 as of 2026-05-27, while noting 3.7.3 development builds exist. |
| Dagster version | Said 1.13.x without current patch. | Updated to 1.13.6 as of 2026-05-27. |
| Flyte 2.0 | Overstated Flyte 2 as generally production-ready OSS distributed orchestration. | Clarified that Flyte 2 is available locally/Devbox, production preview is hosted by Union.ai, and Flyte 1.x remains the conservative choice for OSS distributed execution at this verification date. |
| OpenLineage | Implied every listed system has the same first-party integration maturity. | Clarified OpenLineage as a standard with strong Airflow/Spark/dbt/Flink support; Dagster also has native asset lineage and may integrate through catalogue/OpenLineage paths depending on stack. |
| Cloud Composer | Claimed broad Airflow 3.0 feature support without support-stage nuance. | Clarified that Google Managed Service for Apache Airflow has Cloud Composer/Gen 3 and Airflow 3 support paths, but exact Airflow versions and support stage must be checked against the version list. |
| References | Included several weak secondary blogs as support for factual claims. | Replaced or demoted weak sources; references now prioritise official docs, release notes, PyPI/GitHub release pages, and vendor documentation. |

### Material Gaps Addressed

1. **Operational ownership**: The guide now covers ownership, runbooks, severity, service-level objectives, and escalation paths instead of treating alerting as just callbacks.
2. **Capacity and backpressure**: Added guidance for concurrency limits, pool/queue sizing, downstream rate limits, and fan-out controls. These are common sources of production DAG outages.
3. **Data contracts and quality gates**: Added schema contracts, row-count/freshness checks, quarantine paths, and explicit "silent success" controls.
4. **Security and secrets**: Added RBAC, secrets backend usage, least-privilege service accounts, artifact retention, PII handling, and supply-chain controls.
5. **Migration/lifecycle**: Added version pinning, dependency constraints, blue/green DAG deployment, rollback, replay/backfill compatibility, and deprecation policy.
6. **DAG vs workflow distinction**: Clarified that DAG schedulers, state machines, durable execution engines, asset orchestrators, and CI systems solve overlapping but different problems.
7. **AI-agent nuance**: Added caveats that open-ended agents often require loops and durable state; a pure acyclic DAG is a poor model for unbounded planning loops.

### Remaining Limits

- The document is a reference guide, not a benchmark. It does not measure throughput, scheduler latency, cost per 1,000 tasks, or failure recovery time across tools.
- Managed-service feature availability varies by region, account, tier, and date. Always check the provider's release notes and region matrix.
- "Best tool" recommendations assume typical team capabilities and workloads. Existing platform investment can outweigh theoretical tool fit.

---

## How to Use This Guide

Use this document as a decision and review aid, not as a substitute for a proof-of-concept. The fastest path depends on your role and workload.

| Reader | Start with | Then read |
|--------|------------|-----------|
| **Data engineer choosing Airflow/Dagster/Prefect** | [Tool Comparison Matrix](#tool-comparison-matrix) | [Data Engineering](#data-engineering--etlelt), [Production Hardening Practices](#production-hardening-practices), [Tool-Specific Production Checklists](#tool-specific-production-checklists) |
| **Analytics/data-platform lead** | [Asset-Based Scheduling](#1-asset-based-scheduling) | [Data Contracts and Quality Gates](#2-data-contracts-and-quality-gates), [Data Lineage with OpenLineage](#data-lineage-with-openlineage), [Migration and Lifecycle Management](#migration-and-lifecycle-management) |
| **Platform engineer** | [Managed Services on Hyperscalers](#managed-services-on-hyperscalers) | [Capacity, Backpressure, and Concurrency](#3-capacity-backpressure-and-concurrency), [Runtime Isolation](#5-runtime-isolation), [Choosing Managed vs Self-Hosted](#choosing-managed-vs-self-hosted) |
| **Application engineer orchestrating business processes** | [Long-Running Business Processes](#long-running-business-processes) | [Temporal](#temporal), [Human-in-the-Loop Gates](#7-human-in-the-loop-gates), [Anti-Patterns](#anti-patterns) |
| **ML/AI engineer** | [Machine Learning Pipelines](#machine-learning-pipelines) | [Flyte](#flyte), [Argo Workflows](#argo-workflows), [AI Agent Orchestration](#ai-agent-orchestration), [AI/ML-Specific Orchestration](#5-aiml-specific-orchestration) |
| **Architecture reviewer** | [Fact-Check and Gap Assessment](#fact-check-and-gap-assessment) | [Claim-Source-Confidence Matrix](#claim-source-confidence-matrix), [Reference Architectures](#reference-architectures), [Decision Framework](#decision-framework) |

### Evaluation Criteria

When comparing tools, score them against these concrete dimensions rather than generic popularity:

| Criterion | Question |
|-----------|----------|
| **Workload semantics** | Is this a scheduled batch DAG, an asset graph, a state machine, a durable saga, or an open-ended agent loop? |
| **Failure model** | What happens after worker death, scheduler restart, downstream throttling, partial side effects, or human non-response? |
| **Backfill/replay model** | Can historical work be re-run safely and cheaply with the correct logical date and code version? |
| **Observability** | Can operators trace input, output, owner, run status, logs, metrics, lineage, cost, and business impact? |
| **Security boundary** | Who can author, trigger, inspect, and mutate workflows? Which data can each worker identity access? |
| **Portability** | Can the workflow move across clouds or runtimes, and what semantics are lost during migration? |
| **Team fit** | Does the owning team prefer Python code, YAML, Kubernetes primitives, managed cloud services, or application code? |

---

## Core Concepts

### What is a DAG?

A DAG is a finite graph where:

- **Directed**: Every edge has a direction (A → B means "A must complete before B can start").
- **Acyclic**: No path through the graph leads back to its starting node — cycles are forbidden.

In orchestration, **nodes** represent tasks (units of work) and **edges** represent dependencies between them. The acyclic constraint ensures there is no dependency cycle that makes scheduling impossible; actual completion still depends on task correctness, infrastructure availability, and retry/error handling.

### Why DAGs?

| Property | Benefit |
|----------|---------|
| **Deterministic ordering** | Topological sort yields a valid execution sequence honouring all dependencies |
| **Safe parallelism** | Independent branches (no shared edges) execute concurrently without coordination |
| **Failure isolation** | A failed node blocks only its downstream dependants, not the entire pipeline |
| **Reproducibility** | Same DAG structure + same inputs = same execution path |
| **Lineage** | The graph structure inherently documents data flow and provenance |
| **Backfill** | Re-running for historical periods is structurally safe when tasks are idempotent |

### Terminology

| Term | Definition |
|------|-----------|
| **Node / Task / Step** | A discrete unit of work (extract data, train model, send notification) |
| **Edge / Dependency** | A directed connection A→B meaning "B requires A to complete first" |
| **Root node** | A node with no incoming edges — a starting point |
| **Leaf / Terminal node** | A node with no outgoing edges — an endpoint |
| **Topological sort** | An ordering of all nodes such that for every edge A→B, A appears before B |
| **Fan-out** | One node feeding multiple downstream nodes (enables parallelism) |
| **Fan-in** | Multiple upstream nodes converging to one downstream node (aggregation) |
| **Trigger rule** | Conditions for execution (all parents succeeded, one parent succeeded, all done regardless) |
| **Backfill** | Retroactively executing a DAG for historical time periods |
| **Materialisation** | Producing a concrete data artefact from an asset/task definition |
| **Idempotency** | Executing a task multiple times produces the same result |
| **Partition** | A logical slice of data a task operates on (e.g., one day, one region) |
| **Sensor / Trigger** | A node that waits for an external condition before proceeding |

### Execution Model

```text
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Extract │────▶│Transform│────▶│  Load   │
└─────────┘     └─────────┘     └─────────┘
      │                               ▲
      │         ┌─────────┐           │
      └────────▶│ Validate│───────────┘
                └─────────┘
```

A scheduler:
1. Parses the DAG definition and builds the dependency graph.
2. Topologically sorts the graph.
3. Schedules root nodes for execution.
4. As each node completes, evaluates trigger rules on its dependants.
5. Schedules dependants whose trigger rules are satisfied.
6. Repeats until all leaf nodes complete (success) or an unrecoverable failure occurs.

---

## Orchestration Tools Landscape

### Apache Airflow

**Version**: 3.0 GA (April 2025); 3.2.1 verified on PyPI as of 2026-05-27 (released 2026-04-22). {S1}{S2}
**Managed services**: AWS MWAA, GCP Cloud Composer 3, Astronomer
**Licence**: Apache 2.0

Airflow is a mature incumbent for scheduled batch orchestration and has a broad ecosystem of provider packages. It defines DAGs in Python and executes tasks via pluggable executors (Local, Celery, Kubernetes). {S1}

**Airflow 3.0 key changes:**

| Feature | Description |
|---------|-------------|
| **Task SDK** | Provides stable Python-native DAG/task authoring interfaces under `airflow.sdk`, reducing reliance on internal Airflow modules. {S1} |
| **DAG Versioning** | Track multiple versions of a DAG, view historical definitions, compare changes over time |
| **Remote/Hybrid Execution** | Task Execution Interface separates DAG parsing from task execution via a dedicated API Server — tasks run on Kubernetes, remote workers, or edge infrastructure |
| **Event-driven scheduling** | Native event-based triggers beyond cron (dataset events, external signals) |
| **Rebuilt UI** | Modern web interface with improved navigation, DAG dependency visualisation, and grid/graph views |
| **Backfill improvements** | More granular backfill control with partition-aware re-runs |

**When to use Airflow:**
- Scheduled batch ETL/ELT pipelines on a cadence.
- Complex multi-step data workflows requiring mature integrations (100+ provider packages).
- Organisations that need OpenLineage-compatible data lineage.
- Teams already invested in the Airflow ecosystem.

**Watch-outs:**
- Keep DAG files lightweight; scheduler parse latency is a real scaling limit.
- Avoid using sensors in `poke` mode for long waits; use deferrable operators or `reschedule` mode so workers are not occupied while waiting. {S3}
- Airflow is a scheduler/orchestrator, not a durable application runtime. Long-running human workflows and months-long state machines usually fit Temporal or a cloud state-machine service better.

**Example — basic DAG in Airflow 3.0:**

```python
from airflow.sdk import DAG, task
from datetime import datetime

with DAG(
    dag_id="etl_pipeline",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    @task
    def extract():
        return {"records": 1000}

    @task
    def transform(data: dict):
        return {**data, "cleaned": True}

    @task
    def load(data: dict):
        print(f"Loading {data['records']} records")

    raw = extract()
    cleaned = transform(raw)
    load(cleaned)
```

---

### Prefect

**Version**: 3.7.2 stable verified as of 2026-05-27. {S5}
**Managed service**: Prefect Cloud
**Licence**: Apache 2.0

Prefect follows a "negative engineering" philosophy — reduce boilerplate and let workflows emerge from function calls rather than explicit DAG definitions. Any Python function decorated with `@flow` or `@task` becomes an orchestrated unit.

**Key capabilities:**

- Pure Python decorator-based API (`@flow`, `@task`)
- Event-driven automations for real-time reactions. {S5}
- Transactional task execution with rollback support. {S5}
- Work pools for infrastructure abstraction. {S5}
- Native async support and concurrent task execution
- Automatic retries, caching, and result persistence
- Runs locally as a normal Python script — no infrastructure needed for development

**When to use Prefect:**
- New Python-focused projects wanting minimal boilerplate.
- Teams that value developer experience over ecosystem breadth.
- Rapid iteration and experimentation phases.
- Projects where flows are naturally expressed as function calls.

**Watch-outs:**
- Prefect's flexible Python control flow is productive, but production teams should still document expected dependency structure, retry policy, and data contracts explicitly.
- Its lineage story is lighter than Dagster's asset graph or Airflow + OpenLineage; pair it with catalogue/metadata tooling if lineage is a hard requirement.

**Example:**

```python
from prefect import flow, task

@task(retries=3, retry_delay_seconds=60)
def extract_data(source: str) -> dict:
    return {"source": source, "rows": 5000}

@task
def transform_data(raw: dict) -> dict:
    return {**raw, "transformed": True}

@task
def load_data(data: dict) -> None:
    print(f"Loaded {data['rows']} rows from {data['source']}")

@flow(name="etl-pipeline")
def etl_pipeline(source: str = "postgres"):
    raw = extract_data(source)
    transformed = transform_data(raw)
    load_data(transformed)

if __name__ == "__main__":
    etl_pipeline()
```

---

### Dagster

**Version**: 1.13.6 verified as of 2026-05-27 (released 2026-05-22). {S6}
**Managed service**: Dagster Cloud (Dagster+)
**Licence**: Apache 2.0

Dagster's core paradigm is **Software-Defined Assets (SDA)** — you define *what data should exist* rather than *what tasks should run*. The asset graph becomes your documentation, lineage, and orchestration layer simultaneously. {S6}

**Key capabilities:**

| Feature | Description |
|---------|-------------|
| **Software-Defined Assets** | Declarative data product definitions; the asset graph IS your documentation |
| **Declarative Automation** | Assets materialise based on freshness policies, not cron schedules |
| **Partitions** | First-class partition-aware processing with automatic backfill |
| **Type system** | Strong typing for data flowing between assets |
| **Integrated lineage** | Asset health, staleness detection, and dependency tracking built in |
| **dbt integration** | Native first-class support for dbt models as assets |
| **Virtual assets** (1.13) | Represent database views without materialisation. {S6} |
| **Testing** | Assets are plain Python functions — trivially unit-testable with `materialize_to_memory()` |

**When to use Dagster:**
- Data platform teams where lineage and observability are critical.
- Asset-centric architectures (medallion/lakehouse patterns, analytics engineering).
- dbt-heavy stacks needing orchestration beyond dbt Cloud.
- Teams that prioritise testability and type safety.

**Watch-outs:**
- Dagster is strongest when the organisation can name and own data assets. It can feel heavy for simple task chains with no durable data products.
- "Asset-first" modelling requires up-front naming, ownership, partitioning, and freshness decisions; weak asset modelling can produce a confusing graph.

**Example — asset-based pipeline:**

```python
from dagster import asset, Definitions, AssetExecutionContext

@asset
def raw_orders(context: AssetExecutionContext) -> list[dict]:
    context.log.info("Extracting orders")
    return [{"id": 1, "amount": 100}, {"id": 2, "amount": 200}]

@asset
def cleaned_orders(raw_orders: list[dict]) -> list[dict]:
    return [o for o in raw_orders if o["amount"] > 0]

@asset
def order_summary(cleaned_orders: list[dict]) -> dict:
    return {
        "count": len(cleaned_orders),
        "total": sum(o["amount"] for o in cleaned_orders),
    }

defs = Definitions(assets=[raw_orders, cleaned_orders, order_summary])
```

---

### Temporal

**Version**: Continuous server and SDK releases; verify server and language SDK compatibility before deployment
**Managed service**: Temporal Cloud
**Licence**: MIT

Temporal is a **durable execution** engine, not a DAG scheduler. Workflows are imperative code whose state is persisted to an event history that survives worker crashes, restarts, and deployments. Workflow code must be deterministic because Temporal replays event history to rebuild workflow state. {S7}

**Key capabilities:**

- Durable execution with automatic state persistence
- Multi-language SDKs: Go, Java, TypeScript, Python, .NET, PHP, Ruby
- Long-running workflows: native timers spanning days, weeks, or months
- Saga pattern with built-in compensating actions for distributed transactions
- Signals and queries for external interaction with running workflows. {S7}
- Worker Versioning: helps route workflow tasks to compatible worker code during deployments. {S7}
- Child workflows for hierarchical decomposition
- Strong fit for agent/tool loops where retry, timeout, and human-interaction state must survive process failure

**When to use Temporal:**
- Long-running stateful workflows (hours to months).
- Payment processing, order management, subscription billing.
- Saga/compensation patterns across distributed services.
- Human-in-the-loop approval chains.
- AI agent control loops requiring durable state.
- Microservice orchestration beyond batch scheduling.

**Watch-outs:**
- Workflow code has determinism constraints: do not call random, wall-clock time, network I/O, or non-deterministic libraries directly inside workflow logic; put side effects in activities.
- Temporal does not replace a data lineage system or batch scheduler. For daily data jobs, pair it with Airflow/Dagster only when durable state is actually required.

**Example — durable workflow:**

```python
from temporalio import workflow, activity
from datetime import timedelta

@activity.defn
async def process_payment(order_id: str) -> str:
    return f"payment_confirmed_{order_id}"

@activity.defn
async def ship_order(order_id: str, payment_ref: str) -> str:
    return f"shipped_{order_id}"

@workflow.defn
class OrderWorkflow:
    @workflow.run
    async def run(self, order_id: str) -> str:
        payment = await workflow.execute_activity(
            process_payment,
            order_id,
            start_to_close_timeout=timedelta(minutes=5),
        )
        shipment = await workflow.execute_activity(
            ship_order,
            args=[order_id, payment],
            start_to_close_timeout=timedelta(hours=24),
        )
        return shipment
```

---

### Flyte

**Version**: Flyte 2 available for local/Devbox use; Flyte 1.x remains the conservative OSS distributed runtime as of 2026-05-27. {S8}
**Managed service**: Union.ai (Union Cloud)
**Licence**: Apache 2.0 (LF AI & Data Foundation)

Flyte is purpose-built for ML/AI pipelines at scale. Tasks are defined with typed Python interfaces and can request resources such as CPU, memory, and GPU/accelerators. {S8}

**Flyte highlights:**

- Typed Python task/workflow definitions with containerised execution
- Resource-aware scheduling (GPU, memory, CPU constraints)
- Kubernetes-native with multi-tenant execution
- Built-in caching and versioned workflow runs
- Dynamic workflows and map tasks for ML-style fan-out
- Flyte 2 introduces a local-first execution model and a revised architecture, but check production-readiness by runtime and hosting option before adopting it for distributed workloads. {S8}

**When to use Flyte:**
- ML training pipelines at scale.
- Feature engineering with strong reproducibility requirements.
- GPU-intensive workloads requiring accelerator-aware scheduling.
- Kubernetes-first environments needing container isolation per task.
- Teams already committed to Kubernetes and typed Python ML workflows.

**Watch-outs:**
- Treat Flyte 2 as an adoption decision, not a drop-in upgrade. Validate scheduler/runtime maturity, migration tooling, and hosted-vs-OSS feature parity.
- Flyte is not the simplest choice for general business ETL where Airflow, Dagster, or Prefect already meet the requirement.

---

### Argo Workflows

**Version**: 4.0.5 verified as of 2026-05-27 (released 2026-04-23). {S9}
**Managed service**: Pipekit (commercial); self-hosted on any Kubernetes cluster
**Licence**: Apache 2.0 (CNCF)

Argo Workflows is a Kubernetes-native orchestrator implemented as a Custom Resource Definition (CRD). Workflows are defined in YAML, and each step runs as a container. {S9}

**Key capabilities:**

- Container-first: each step is an isolated container
- DAG and step-based workflow definitions in YAML. {S9}
- Artifact passing between steps via S3/GCS/MinIO. {S9}
- Retry strategies and conditional execution. {S9}
- Parallel execution with configurable concurrency limits
- Template reuse and composition
- Part of the broader Argo ecosystem (CD, Events, Rollouts)

**When to use Argo Workflows:**
- Kubernetes-native CI/CD pipelines.
- ML pipelines on Kubernetes (Kubeflow Pipelines uses Argo under the hood).
- Container-based batch processing.
- Teams preferring YAML/declarative definitions over Python.
- GitOps-driven workflow management.

**Watch-outs:**
- Argo inherits Kubernetes' operational surface: RBAC, service accounts, image security, namespace isolation, cluster autoscaling, and artifact storage are part of the production design.
- YAML reuse is powerful but can become difficult to reason about without templates, linting, and review conventions.

**Example — YAML DAG:**

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: etl-pipeline-
spec:
  entrypoint: etl
  templates:
    - name: etl
      dag:
        tasks:
          - name: extract
            template: run-step
            arguments:
              parameters: [{name: step, value: "extract"}]
          - name: transform
            template: run-step
            dependencies: [extract]
            arguments:
              parameters: [{name: step, value: "transform"}]
          - name: load
            template: run-step
            dependencies: [transform]
            arguments:
              parameters: [{name: step, value: "load"}]
    - name: run-step
      inputs:
        parameters: [{name: step}]
      container:
        image: etl-runner:2026-05-27
        command: [python, -m, pipeline]
        args: ["{{inputs.parameters.step}}"]
```

---

### Kestra

**Version**: Active development; open-source + Kestra Cloud
**Licence**: Apache 2.0

Kestra is a YAML-first, event-driven orchestrator that is language-agnostic — tasks can execute scripts in multiple runtimes rather than requiring Python-only task logic. {S10}

**Key capabilities:**

- Declarative YAML workflow definitions
- Event-driven architecture with trigger support for systems such as webhooks, Kafka, S3, and SQS. {S10}
- Visual UI for building and monitoring workflows. {S10}
- Language-agnostic task execution
- Infrastructure-as-Code approach
- Plugin ecosystem for integrations
- AI-assisted workflow generation where available in the deployed edition

**When to use Kestra:**
- Teams preferring YAML/declarative over Python.
- Event-driven and reactive workloads.
- Mixed-language environments (not Python-only).
- Platform/ops teams applying IaC principles to orchestration.

**Watch-outs:**
- Validate plugin maturity for your specific systems. Event-driven orchestration is only as reliable as the trigger, retry, and idempotency design around each source.
- Language-agnostic execution helps platform teams, but it can weaken shared testing conventions unless each task type has a standard harness.

---

### Lightweight and Complementary Tools

#### Apache Hamilton (Incubating)

**Version**: Active development; verify current release before adoption | **Licence**: BSD-3 (Apache incubating)

Hamilton is a *micro-orchestration* library — not a scheduler. Each Python function defines a node; function parameters define dependencies. It structures dataflow logic *within* your code and runs anywhere Python runs.

**When to use**: Structuring feature engineering logic, within-task dataflow (e.g., Hamilton inside an Airflow task), teams wanting testable/documented transformations without scheduler overhead.

```python
# features.py — each function is a node, parameters are edges
def age(birth_year: int, current_year: int = 2026) -> int:
    return current_year - birth_year

def is_adult(age: int) -> bool:
    return age >= 18

def risk_score(age: int, is_adult: bool) -> float:
    return 1.0 / age if is_adult else 0.0
```

#### Kedro

**Version**: Active development; verify current release before adoption | **Licence**: Apache 2.0 (LF AI & Data)

Kedro is a framework for structuring ML/data science code — not an orchestrator itself. It provides an opinionated project layout, a YAML-based data catalogue, modular pipelines, and deploys to any orchestrator (Airflow, Vertex AI, Argo).

**When to use**: Data science teams wanting production-ready code structure, when deploying the same pipeline to multiple orchestrators, when reproducibility and standardised layout matter.

#### Mage AI

**Version**: Active development (2026) | **Licence**: Apache 2.0

Mage combines a notebook-style interactive interface with workflow orchestration for data pipelines. Treat AI-assisted generation as a productivity feature to validate, not as a substitute for review, testing, and production hardening.

**When to use**: Small teams wanting rapid pipeline development, AI-assisted data engineering, simpler ETL/ELT workloads where interactive development is preferred.

#### Luigi (Legacy)

**Version**: Legacy project; verify current release and maintenance status before adoption | **Licence**: Apache 2.0

Originally from Spotify, Luigi is now primarily a legacy choice. For new projects, compare it carefully against Airflow, Dagster, and Prefect before adopting it, especially if you need active ecosystem growth, managed-service options, or modern lineage integrations.

---

## Tool Comparison Matrix

| Criterion | Airflow | Prefect | Dagster | Temporal | Flyte | Argo WF | Step Functions |
|-----------|---------|---------|---------|----------|-------|---------|----------------|
| **Primary paradigm** | Task-based DAG | Flow-based (Python) | Asset-based | Durable execution | Typed tasks (ML) | Container DAG | State machine |
| **Best for** | Batch ETL/ELT | Python pipelines | Data platforms | Stateful apps | ML/AI workloads | K8s-native CI/CD | Serverless AWS |
| **Scheduling** | Cron + dataset events | Flexible (cron, event, ad-hoc) | Freshness policies | Event/signal-driven | Cron + triggers | Cron + events | Event-driven |
| **Language** | Python DAG authoring | Python | Python | Go, Java, TS, Python, .NET, PHP, Ruby | Python | YAML | JSON/YAML (Amazon States Language) |
| **Learning curve** | Steep | Low–moderate | Moderate | Steep | Moderate | Moderate | Low–moderate |
| **Testing story** | Requires setup (`dag.test()`) | Built-in (functions are testable) | Best-in-class (`materialize_to_memory`) | Standard unit tests | Built-in | Standard | Limited |
| **Data lineage** | Via OpenLineage | Limited unless paired with catalogue tooling | Built-in asset graph | N/A | Built-in execution metadata | N/A unless integrated | CloudWatch/X-Ray/EventBridge metadata |
| **Long waits (hours+)** | Discouraged (ties up worker) | Possible | Not designed for | First-class (native timers) | Not designed for | Not designed for | Up to 1 year |
| **Dynamic DAGs** | `expand()` / dynamic task mapping | Natural (Python control flow) | `DynamicOut`, dynamic partitions | Fully dynamic imperative code | Dynamic workflows/map tasks | Conditional steps | Choice/Map/Distributed Map states |
| **Managed option** | MWAA, Composer 3, Astronomer | Prefect Cloud | Dagster Cloud | Temporal Cloud | Union.ai | Pipekit | AWS-native |
| **Maturity (since)** | 2014 | 2018 | 2018 | 2019 | 2017 (Lyft) | 2017 | 2016 |
| **Portability** | High across Airflow-compatible services | High for Python flows | High for Python assets; Dagster-specific model | Medium; Temporal runtime required | Medium; Kubernetes/Flyte runtime required | Medium; Kubernetes required | Low outside AWS |

---

## Use Cases

### Data Engineering — ETL/ELT

The canonical DAG use case. Extract data from sources, transform it, load it into a warehouse or lake.

| Stage | Pattern | Tools |
|-------|---------|-------|
| **Extract** | Parallel fan-out across sources (APIs, databases, files) | Airflow, Dagster, Prefect |
| **Transform** | Sequential or parallelised cleaning, enrichment, aggregation | dbt + orchestrator, Dagster assets |
| **Load** | Idempotent upsert into target (warehouse, lake, feature store) | All orchestrators |
| **Quality** | Post-load validation checks (row counts, schema, freshness) | Great Expectations, Dagster checks, Soda |

**Scheduling**: Typically cron-driven (`@daily`, `@hourly`) or event-driven (new file arrival in S3/GCS).

### Machine Learning Pipelines

ML pipelines extend ETL with training, evaluation, and deployment stages:

```text
Feature Engineering → Training → Evaluation → Model Registry → Deployment → Monitoring
```

| Requirement | Tool fit |
|-------------|----------|
| GPU-aware scheduling | Flyte, Argo Workflows |
| Experiment tracking | Flyte (built-in), Airflow + MLflow |
| Hyperparameter sweeps | Flyte (map tasks), Argo (parallel pods) |
| Model versioning | Flyte, Dagster (asset versioning) |
| A/B deployment | Argo Rollouts, Step Functions + SageMaker |

### CI/CD Pipelines

Build → test → deploy → validate chains are DAGs:

- **Argo Workflows**: Kubernetes-native, container-per-step, integrates with Argo CD for GitOps.
- **Step Functions**: Serverless, integrates with CodeBuild/CodeDeploy.
- **GitHub Actions / GitLab CI**: Built-in DAG support via `needs` / `dependencies`.

### Event-Driven Workflows

React to events rather than schedules:

- File arrives in storage → trigger processing pipeline
- API webhook → kick off validation and enrichment
- Message on Kafka topic → update downstream aggregates

**Best tools**: Kestra (event-first), Prefect (automations), Step Functions (EventBridge), Airflow 3.0 (dataset events).

### Long-Running Business Processes

Workflows spanning hours, days, or months with human interaction:

- Order fulfilment (payment → warehouse → shipping → delivery confirmation)
- Insurance claims (submission → assessment → approval → payment)
- Employee onboarding (multiple departments, multi-week)

**Best tool**: Temporal — purpose-built for durable, long-running, stateful execution.

### AI Agent Orchestration

LLM-powered agents with tool calls, reasoning loops, and human checkpoints:

| Pattern | Implementation |
|---------|---------------|
| Durable agent loop | Temporal workflow per agent session |
| Tool calls | Temporal activities (retries, timeouts) |
| Human-in-the-loop | Temporal signals for approval gates |
| Outer scheduling | Airflow/Dagster triggers agent runs on a cadence |

A common hybrid pattern: Airflow as the outer scheduler ("run this agent daily at 06:00") with Temporal as the inner runtime managing the agent's durable execution.

---

## Reference Architectures

These are starting-point architectures. They show where the workflow boundary should sit, what state belongs in the orchestrator, and which failure controls are mandatory.

### 1. Batch ETL / ELT Data Pipeline

```text
Source systems
  -> extract tasks
  -> raw landing zone
  -> validation gate
  -> transform/dbt or Python tasks
  -> warehouse/lakehouse table
  -> data quality checks
  -> lineage/catalogue event
  -> downstream notification
```

**Strong candidates**: Airflow, Dagster, Prefect. Use Dagster when data assets, partitions, freshness, and lineage are the organising model. Use Airflow when broad provider ecosystem and scheduled operational maturity matter most. Use Prefect when Python-first ergonomics and lightweight deployment matter most.

**Mandatory controls**:
- Partition by logical date or business key.
- Make extract and load idempotent.
- Validate source schema before transform.
- Enforce row-count, freshness, and uniqueness checks before publishing.
- Record code version, input snapshot, and output partition.
- Use pools/queues for warehouses, APIs, and shared source systems.

**Avoid**:
- Treating a successful task run as proof of valid data.
- Appending into canonical tables without idempotency keys.
- Calendar-only scheduling when upstream data arrival is variable.

### 2. Asset-Centric Data Platform

```text
Declared assets
  -> freshness policy
  -> partitioned materialisation
  -> asset checks
  -> asset metadata
  -> catalogue/lineage graph
  -> consumer-facing data product
```

**Strong candidates**: Dagster, Airflow with assets/datasets plus OpenLineage, dbt + orchestrator.

**Mandatory controls**:
- Define asset owner, freshness target, and downstream consumers.
- Use asset checks or equivalent data tests as publish gates.
- Make partitioning explicit and test backfill for representative historical partitions.
- Keep asset names stable; rename through a migration, not an incidental refactor.
- Record schema version and quality-check results with materialisation metadata.

**Avoid**:
- Modelling every internal helper as a business asset.
- Creating an asset graph that mirrors implementation details rather than business data products.

### 3. ML Training and Evaluation Pipeline

```text
Feature extraction
  -> training dataset snapshot
  -> train model
  -> evaluate model
  -> compare against baseline
  -> register model
  -> approval gate
  -> deploy/canary
  -> monitor drift
```

**Strong candidates**: Flyte, Vertex AI Pipelines, Argo Workflows, Dagster for asset-heavy ML platforms.

**Mandatory controls**:
- Version dataset, feature code, training code, model hyperparameters, and container image.
- Request GPU/accelerator resources explicitly.
- Cache deterministic feature steps, but invalidate cache on code/data/schema changes.
- Fail closed if evaluation does not beat baseline thresholds.
- Store model card, metrics, and lineage before deployment.

**Avoid**:
- Re-training from mutable "latest" datasets.
- Deploying from a workflow that does not preserve the evaluation artifact and model version.

### 4. Long-Running Human Approval Workflow

```text
Trigger
  -> validate request
  -> perform automated checks
  -> wait for human approval
  -> execute side effect
  -> compensation path if failed
  -> audit record
```

**Strong candidates**: Temporal, Azure Durable Functions, AWS Step Functions Standard.

**Mandatory controls**:
- Persist business state outside ephemeral worker memory.
- Record approver identity, timestamp, decision, and edited artifact hash.
- Define timeout behaviour for missing approval.
- Use idempotency keys for external side effects.
- Model compensation explicitly for failed distributed transactions.

**Avoid**:
- A scheduler task that waits for days or months.
- Manual UI actions as the only record of approval.

### 5. LLM Agent With Durable Inner Runtime

```text
Outer scheduler or event trigger
  -> create agent run
  -> durable runtime controls loop
      -> plan
      -> call tools
      -> checkpoint state
      -> ask human if needed
      -> enforce budget/stop condition
  -> persist report/result
  -> quality and audit checks
```

**Strong candidates**: Temporal for durable control loops, LangGraph for explicit agent control flow, Airflow/Dagster only as the outer scheduler when runs are periodic.

**Mandatory controls**:
- Set token, tool-call, wall-clock, and iteration budgets.
- Make tool calls idempotent or side-effect gated.
- Persist agent state and intermediate outputs.
- Record prompt, model, tool schema, and retrieval/source versions.
- Separate deterministic pre/post-processing from probabilistic agent reasoning.

**Avoid**:
- Hiding an unbounded agent loop inside one opaque DAG task.
- Retrying the whole task after partial tool side effects.

### 6. Cloud-Native Service Choreography

```text
EventBridge / Eventarc / Event Grid
  -> serverless workflow
  -> service call
  -> choice/branch
  -> callback or wait
  -> compensation/retry
  -> notification/audit
```

**Strong candidates**: AWS Step Functions, Google Cloud Workflows, Azure Logic Apps, Azure Durable Functions.

**Mandatory controls**:
- Bound retries and add backoff.
- Track state-machine cost, especially per-state-transition billing.
- Store payloads safely; avoid pushing large or sensitive documents through workflow state.
- Use cloud IAM/service identities with least privilege.
- Keep complex business logic in versioned application code, not only visual workflow steps.

**Avoid**:
- Using cloud-native state machines for portable data pipelines without accepting vendor lock-in.
- Treating visual workflow configuration as exempt from code review.

---

## Design Patterns

### 1. Fan-Out / Fan-In

One task produces output consumed by multiple parallel downstream tasks (fan-out), which converge at a single aggregation task (fan-in).

```text
                ┌──▶ Process Shard 1 ──┐
                │                       │
Extract ────────┼──▶ Process Shard 2 ──┼──▶ Aggregate
                │                       │
                └──▶ Process Shard 3 ──┘
```

**Use for**: Parallel processing of partitions, multi-source ingestion, distributed computation, map-reduce patterns.

**Implementation:**

```python
# Airflow — dynamic task mapping (fan-out/fan-in)
@task
def extract() -> list[str]:
    return ["shard_1", "shard_2", "shard_3"]

@task
def process(shard: str) -> dict:
    return {"shard": shard, "rows": 1000}

@task
def aggregate(results: list[dict]) -> int:
    return sum(r["rows"] for r in results)

shards = extract()
processed = process.expand(shard=shards)  # fan-out
total = aggregate(processed)               # fan-in
```

### 2. Dynamic DAGs

DAG structure determined at runtime based on data, configuration, or external state.

| Tool | Mechanism |
|------|-----------|
| Airflow 3.0 | `expand()` for dynamic task mapping; runtime-generated task lists |
| Dagster | Dynamic partitions, `DynamicOut` for runtime-determined asset counts |
| Temporal | Fully dynamic — imperative code, no predetermined structure |
| Prefect | Natural Python control flow (loops, conditionals generate tasks) |
| Flyte | Dynamic workflows and map tasks; validate Flyte 2 runtime maturity separately |

**Example — Prefect dynamic parallelism:**

```python
from prefect import flow, task

@task
def process_file(path: str) -> dict:
    return {"path": path, "status": "done"}

@flow
def process_all_files():
    # Discover files at runtime
    files = list_new_files()  # unknown count at definition time
    # Dynamic fan-out
    futures = process_file.map(files)
    return [f.result() for f in futures]
```

### 3. Conditional Branching

Execute different paths based on runtime conditions.

```text
                    ┌──▶ Path A (large file) ──┐
Check File Size ────┤                           ├──▶ Notify
                    └──▶ Path B (small file) ──┘
```

**Implementation across tools:**

```python
# Airflow — BranchPythonOperator / @task.branch
@task.branch
def choose_path(file_size: int):
    if file_size > 1_000_000:
        return "process_large"
    return "process_small"

# Dagster — conditional asset materialisation
@asset
def conditional_output(context, raw_data):
    if len(raw_data) > threshold:
        return heavy_transform(raw_data)
    return light_transform(raw_data)

# Step Functions — Choice state
# {
#   "Type": "Choice",
#   "Choices": [
#     {"Variable": "$.fileSize", "NumericGreaterThan": 1000000, "Next": "ProcessLarge"}
#   ],
#   "Default": "ProcessSmall"
# }
```

### 4. Retries with Exponential Backoff

Configure per-task retry policies with increasing delays to handle transient failures gracefully.

**Best practice configuration:**

```python
# Airflow
@task(
    retries=3,
    retry_delay=timedelta(minutes=2),
    retry_exponential_backoff=True,
    max_retry_delay=timedelta(minutes=30),
)
def call_external_api():
    ...

# Prefect
@task(
    retries=3,
    retry_delay_seconds=[60, 300, 900],  # explicit backoff schedule
    retry_jitter_factor=0.1,
)
def call_external_api():
    ...

# Temporal — per-activity retry policy
@workflow.defn
class MyWorkflow:
    @workflow.run
    async def run(self):
        await workflow.execute_activity(
            call_api,
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(minutes=5),
                maximum_attempts=5,
            ),
        )
```

**Guidelines:**
- Always configure retries for tasks that call external systems.
- Use exponential backoff to avoid overwhelming failing services.
- Add jitter to prevent thundering herd on recovery.
- Set a maximum retry delay and maximum attempts.
- Log each retry attempt for debugging.

### 5. Idempotency

Every task must produce the same result regardless of how many times it executes. This is the most critical property for reliable DAGs.

**Patterns for achieving idempotency:**

| Pattern | Description |
|---------|-------------|
| **Upsert, not insert** | Use `INSERT ... ON CONFLICT UPDATE` or merge operations |
| **Write-then-rename** | Write to a temporary path, then atomically rename to final location |
| **Deterministic output paths** | Include execution date/partition in path: `s3://bucket/orders/dt=2026-05-27/` |
| **Truncate-and-reload** | Clear the target partition before writing (simple, effective for small data) |
| **Deduplication keys** | Use natural keys or deterministic IDs to prevent duplicates |
| **Tombstone markers** | Track processed inputs; skip on re-execution |

**Anti-pattern** — non-idempotent task:

```python
# BAD: appends on every execution — duplicates on retry/backfill
def load_orders(orders):
    db.execute("INSERT INTO orders VALUES (%s)", orders)
```

**Idempotent equivalent:**

```python
# GOOD: upsert — safe to re-run
def load_orders(orders, execution_date):
    db.execute("""
        INSERT INTO orders (id, amount, loaded_at)
        VALUES (%s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET amount = EXCLUDED.amount
    """, [(o["id"], o["amount"], execution_date) for o in orders])
```

### 6. Data Lineage Integration

Track data provenance through the pipeline automatically.

**OpenLineage** is the open standard for runtime lineage events and has mature integrations across common data engines and orchestrators such as Airflow, Spark, dbt, and Flink. Dagster also provides native asset lineage through its asset graph; use OpenLineage or a catalogue integration when you need cross-platform lineage beyond Dagster itself. {S14}

```python
# Airflow — enable OpenLineage with a single config
# airflow.cfg
[openlineage]
transport = {"type": "http", "url": "http://marquez:5000/api/v1/lineage"}

# Dagster — asset lineage is automatic for Dagster assets
# Every asset's declared dependencies are visible in the asset graph
```

**Benefits of lineage:**
- Blast-radius analysis: "If this source fails, what dashboards break?"
- Incident triage: trace data quality issues to their origin.
- Compliance: demonstrate data provenance for audits.
- Impact assessment: understand downstream effects before changing a pipeline.

### 7. Parameterisation

Pass runtime parameters to DAGs for flexibility without code changes.

```python
# Airflow — runtime parameters via dag_run.conf
with DAG(
    dag_id="parameterised_etl",
    params={
        "source_table": Param("orders", type="string"),
        "date_override": Param(None, type=["null", "string"]),
    },
) as dag:
    @task
    def extract(params=None):
        table = params["source_table"]
        ...

# Dagster — RunConfig
@asset
def configurable_extract(context: AssetExecutionContext):
    source = context.op_config.get("source", "default_table")
    ...

# Prefect — just function parameters
@flow
def etl(source: str = "orders", batch_size: int = 1000):
    ...
```

### 8. Task Groups and Sub-DAGs

Organise related tasks into logical groups for readability and reuse.

```python
# Airflow — TaskGroup (preferred over deprecated SubDAGs)
from airflow.utils.task_group import TaskGroup

with DAG(...) as dag:
    with TaskGroup("extract") as extract_group:
        extract_orders = extract_from_db("orders")
        extract_products = extract_from_db("products")
        extract_customers = extract_from_db("customers")

    with TaskGroup("transform") as transform_group:
        clean = clean_data()
        enrich = enrich_data()
        clean >> enrich

    with TaskGroup("load") as load_group:
        load_warehouse = load_to_warehouse()

    extract_group >> transform_group >> load_group
```

```python
# Dagster — @graph for reusable sub-graphs
from dagster import graph, op

@op
def validate(data): ...

@op
def clean(data): ...

@graph
def cleaning_pipeline(raw_data):
    validated = validate(raw_data)
    return clean(validated)
```

### 9. Sensor / Trigger Pattern

Wait for an external condition before proceeding — avoids polling in the main DAG.

```python
# Airflow — sensors
from airflow.sensors.s3 import S3KeySensor

wait_for_file = S3KeySensor(
    task_id="wait_for_export",
    bucket_name="data-lake",
    bucket_key="exports/{{ ds }}/orders.parquet",
    poke_interval=300,  # check every 5 minutes
    timeout=3600,        # give up after 1 hour
    mode="reschedule",   # free up worker slot while waiting
)

# Dagster — asset sensors
from dagster import asset_sensor, RunRequest

@asset_sensor(asset_key=AssetKey("raw_orders"), job=transform_job)
def orders_sensor(context, asset_event):
    yield RunRequest(run_key=str(asset_event.timestamp))
```

### 10. Graceful Degradation

Handle partial failures without losing all progress.

**Patterns:**
- **Partial success trigger rules**: In Airflow, set `trigger_rule="none_failed_min_one_success"` to continue if at least one upstream succeeded.
- **Dead-letter queues**: Route failed records to a DLQ for later reprocessing rather than failing the entire task.
- **Checkpointing**: For long tasks, save progress periodically so retries resume from the last checkpoint rather than the beginning.
- **Circuit breakers**: If a downstream system is unhealthy, skip that branch and alert rather than blocking the entire DAG.

---

## Production Hardening Practices

Patterns describe how to build DAGs; production hardening describes how to keep them safe when data, infrastructure, and people fail.

### 1. Ownership, Runbooks, and SLOs

Every production workflow needs a named owner, severity classification, and operator procedure. Without this, retries and alerts only move confusion around.

| Practice | Minimum standard |
|----------|------------------|
| **Owner** | Team alias and escalation channel stored in DAG metadata/tags |
| **Purpose** | One-sentence business purpose and downstream consumers |
| **SLO** | Expected start time, expected completion time, and freshness target |
| **Runbook** | How to retry, backfill, disable, or quarantine the pipeline |
| **Failure mode** | What breaks downstream if this DAG misses its SLO |
| **Escalation** | Who gets paged vs. who gets a business-hours ticket |

For critical data products, define both **pipeline SLOs** (run completion) and **data SLOs** (freshness, completeness, schema validity). A DAG can complete successfully while still publishing bad or empty data.

### 2. Data Contracts and Quality Gates

Reliable orchestration depends on explicit contracts at task and dataset boundaries.

| Contract | Examples |
|----------|----------|
| **Schema** | Required columns, data types, nullability, semantic constraints |
| **Volume** | Expected row-count range, minimum records, max change percentage |
| **Freshness** | Input not older than N minutes/hours/days |
| **Uniqueness** | Natural key or compound key constraints |
| **Referential integrity** | Foreign keys, lookup tables, dimension presence |
| **PII classification** | Which columns contain personal, sensitive, or regulated data |

**Recommended gate placement:**
1. Validate inputs before expensive processing.
2. Validate transformations before loading to canonical storage.
3. Validate outputs before marking downstream assets fresh.
4. Quarantine invalid records separately from failed infrastructure runs.

Use Great Expectations, Soda, dbt tests, Dagster asset checks, or custom assertions depending on stack. The key requirement is not the tool; it is that data quality failures are first-class workflow outcomes, not buried in logs. {S6}{S16}

### 3. Capacity, Backpressure, and Concurrency

Most serious orchestration incidents are not graph-theory failures; they are capacity failures.

| Control | Purpose |
|---------|---------|
| **Global concurrency limit** | Prevent scheduler/executor overload |
| **Per-DAG concurrency** | Stop one busy workflow from starving the platform |
| **Per-task concurrency** | Prevent accidental fan-out explosions |
| **Pools/queues** | Protect scarce shared systems such as warehouses, APIs, GPUs, and mainframes |
| **Rate limits** | Respect third-party API quotas and downstream service capacity |
| **Batch sizing** | Keep task duration and memory bounded |
| **Dead-letter paths** | Avoid blocking the whole workflow on a small number of bad records |

Design fan-out from the downstream system backwards. If the target API permits 100 requests/minute, a 1,000-way mapped task is a denial-of-service bug even if the orchestrator can schedule it.

### 4. Security and Secrets

Workflow platforms often have broad data access. Treat them as privileged production systems.

| Area | Practice |
|------|----------|
| **Secrets** | Store credentials in secret backends or cloud secret managers, not DAG files or environment dumps |
| **RBAC** | Separate DAG authors, operators, viewers, and platform administrators |
| **Service accounts** | Use least-privilege identities per workflow or domain where feasible |
| **Network egress** | Restrict worker egress to required APIs/databases |
| **Artifact storage** | Encrypt logs, task outputs, and intermediate files; define retention |
| **PII** | Redact or hash sensitive values before logging, tracing, and lineage emission |
| **Images/dependencies** | Pin base images, scan containers, and lock dependency versions |
| **Audit trail** | Record who triggered, retried, modified, paused, or backfilled production workflows |

Do not assume the orchestrator's UI permission model protects the underlying data. A task running with a powerful service account can still exfiltrate data even if UI access looks restricted.

### 5. Runtime Isolation

Choose isolation based on blast radius:

| Isolation level | Fit |
|-----------------|-----|
| **Shared worker process** | Small trusted teams, lightweight Python tasks |
| **Dedicated worker queues** | Different workloads, priorities, or dependency sets |
| **Container per task** | Heterogeneous dependencies, security boundaries, reproducible ML jobs |
| **Namespace/project/account per domain** | Regulated data, multi-tenant platforms, strict chargeback |
| **Air-gapped/self-hosted runtime** | Sovereign, classified, or disconnected environments |

The higher the isolation, the higher the operational cost. Do not containerise every trivial task just to appear "cloud native"; do isolate workloads that have conflicting dependencies, sensitive data, or unpredictable resource use.

### 6. Cost Controls

Cost should be visible at the workflow and task level.

| Cost driver | Control |
|-------------|---------|
| Compute | Right-size workers/pods; set CPU/memory/GPU requests; autoscale from queue depth |
| Warehouse queries | Use partition pruning, incremental loads, and query budgets |
| LLM/API calls | Cache deterministic inputs, tier models per step, enforce token/request budgets |
| Storage | Expire intermediate artifacts and logs by retention policy |
| Retries | Cap retries and alert on retry storms |
| Backfills | Estimate cost before launching large historical replays |

Large backfills deserve change-review discipline. A 365-day backfill of a daily DAG can be equivalent to launching a year's worth of production load at once.

### 7. Human-in-the-Loop Gates

Approval gates must be explicit state, not an operator manually unpausing a downstream job.

| Requirement | Implementation pattern |
|-------------|------------------------|
| Approval identity | Persist approver user, timestamp, and decision |
| Editable output | Version the edited artifact and mark it as human-modified |
| Timeout | Define what happens if approval does not arrive |
| Rejection | Support re-run from the failed/edited step without losing history |
| Audit | Include inputs, model/tool version, output hash, and approval record |

Temporal signals, Durable Functions external events, Step Functions callbacks, and custom state stores are better fits for long human waits than keeping a scheduler task alive.

---

## Tool-Specific Production Checklists

Use these as pre-production review checklists. They are intentionally operational, not tutorial-oriented.

### Airflow

| Check | Why it matters |
|-------|----------------|
| DAG files import quickly with no top-level network/database calls | Scheduler parse performance and reliability. {S3} |
| Tasks use logical date / data interval, not wall-clock `now()` | Backfill and replay correctness |
| Long waits use deferrable operators or reschedule mode | Avoid worker-slot exhaustion. {S3} |
| Pools protect warehouses, APIs, and other scarce systems | Prevent downstream overload |
| Dynamic task mapping has max concurrency and input-size limits | Prevent fan-out explosions. {S4} |
| Connections/secrets use a secrets backend or controlled Airflow connections | Avoid credential leakage |
| `on_failure_callback`, SLAs/deadlines, and owner tags are set for production DAGs | Operational accountability |
| DAG import tests and `dag.test()`/integration tests run in CI | Catch graph and dependency breakage before deploy |
| Provider package versions are pinned and tested with the Airflow core version | Avoid runtime incompatibility |
| OpenLineage or equivalent metadata capture is configured if cross-system lineage is required | Traceability and audit. {S14} |

### Dagster

| Check | Why it matters |
|-------|----------------|
| Assets map to stable business/data products, not incidental helper steps | Keeps the asset graph understandable |
| Owners, partitions, freshness expectations, and asset checks are defined for critical assets | Data-product accountability. {S6} |
| `materialize_to_memory()` or equivalent unit tests cover core assets | Local testability. {S6} |
| Resources isolate production credentials from tests | Prevents accidental production access |
| Backfill policy is tested for representative partition ranges | Recovery and correction workflows |
| Asset metadata records row counts, schema, quality results, and source versions | Observability and audit |
| dbt integration boundaries are explicit if dbt is part of the stack | Avoid duplicate orchestration semantics |
| Automation rules are reviewed for unintended cascades | Prevents surprise materialisations |

### Prefect

| Check | Why it matters |
|-------|----------------|
| Flow parameters are typed and validated | Prevents invalid runtime configuration |
| Work pools and workers are separated by trust/resource class | Runtime isolation. {S5} |
| Task retries include backoff and stop conditions | Prevents retry storms |
| Transaction/rollback semantics are tested for side-effecting tasks | Recovery correctness. {S5} |
| Cache keys are deterministic and include relevant code/input version | Prevents stale cache reuse |
| Deployment schedules and event automations are documented | Operator clarity. {S5} |
| External observability/catalogue tooling is planned if lineage is required | Prefect lineage is not the primary strength |
| `.fn` task tests cover business logic separately from orchestration | Fast unit feedback |

### Temporal

| Check | Why it matters |
|-------|----------------|
| Workflow code is deterministic; side effects live in activities | Replay correctness. {S7} |
| Activity timeouts, retry policies, and cancellation behaviour are explicit | Failure containment. {S7} |
| Signals/updates are used for human or external interaction | Durable long waits. {S7} |
| Worker versioning/deployment compatibility is planned | Safe rolling deploys. {S7} |
| Long-running workflows have history-size controls via continue-as-new or child workflows where appropriate | Avoid unbounded event histories |
| Business state and audit records are persisted outside Temporal when required by domain policy | Separation of operational and business truth |
| Compensation paths are tested for saga-style workflows | Side-effect recovery |
| Namespace/task queue isolation matches data sensitivity and workload criticality | Security and blast-radius control |

### Argo Workflows and Flyte

| Check | Why it matters |
|-------|----------------|
| Container images use immutable tags or digests | Reproducibility |
| Kubernetes service accounts are least privilege | Cluster/data security |
| CPU, memory, GPU, and ephemeral storage requests/limits are set | Scheduling and cost control |
| Artifact repository retention and encryption are configured | Data protection. {S9} |
| Retry strategy distinguishes transient infrastructure failures from bad data/code | Avoid repeating deterministic failures |
| Workflow templates/tasks are linted and reviewed in Git | Maintainability |
| Cluster autoscaling and queue limits are load-tested | Prevent pod storms |
| Flyte 2 adoption is validated against required distributed/runtime features | Avoid premature migration. {S8} |

### AWS Step Functions

| Check | Why it matters |
|-------|----------------|
| Standard vs Express mode is chosen deliberately | Durability, cost, and observability tradeoffs. {S11} |
| Distributed Map concurrency is bounded to downstream capacity | Prevents high-scale overload. {S11} |
| Redrive behaviour is understood and tested for partial failures | Recovery without full restart. {S11} |
| HTTP Task usage includes authentication, timeout, and payload-size review | Safe API integration. {S11} |
| Large/sensitive payloads are stored externally and referenced by key | Avoid state bloat and data exposure |
| State transition cost is estimated for normal runs and backfills | Cost predictability |
| IAM roles are least privilege per state machine | Security |
| State machine definitions are version-controlled and reviewed | Change control |

### Azure Durable Functions and Google Cloud Workflows

| Check | Why it matters |
|-------|----------------|
| Durable orchestrator code avoids non-deterministic operations | Replay correctness. {S13} |
| Activities/functions own side effects and retries | Failure containment |
| External events/callbacks are used for human waits | Durable approval gates |
| Cloud Workflows YAML/JSON is version-controlled and reviewed | Governance. {S12} |
| Payload size and secret-handling limits are reviewed | Reliability and security |
| Region and service availability are checked before standardising | Managed-service variability |
| Monitoring and audit logs are connected to the same incident process as application services | Operational consistency |

---

## Anti-Patterns

### 1. Monolithic DAGs

**Symptom**: A single DAG with hundreds of tasks spanning multiple business domains.

**Consequences**: Slow parsing, poor failure isolation (one failing task blocks unrelated work), impossible to reason about, difficult to assign ownership.

**Fix**: Decompose into focused, domain-specific DAGs. Use dataset dependencies (Airflow) or asset dependencies (Dagster) to connect DAGs across domains without tight coupling.

### 2. Tight Coupling Between DAGs

**Symptom**: DAG A triggers DAG B via `TriggerDagRunOperator` with implicit assumptions about timing, data format, or shared state.

**Consequences**: Fragile chains that break when one DAG changes schedule or schema. Hidden dependencies not visible in any single DAG's graph.

**Fix**: Use explicit data contracts. Airflow datasets or Dagster asset dependencies make cross-DAG dependencies visible and enforceable. Define schemas at boundaries.

### 3. Over-Orchestration

**Symptom**: A three-task DAG that runs once, has no retries, no scheduling, and no monitoring — but still lives in the orchestrator.

**Consequences**: Operational overhead, scheduler load, alert fatigue from trivial DAGs.

**Fix**: Only orchestrate when you need scheduling, retries, dependency management, or observability. A simple script with a cron job is sometimes the right answer.

### 4. Calendar-Only Scheduling

**Symptom**: Every DAG runs on a fixed cron schedule regardless of whether upstream data has actually arrived.

**Consequences**: Pipelines process stale or missing data. Downstream consumers see empty tables. Wasteful runs produce nothing useful.

**Fix**: Combine time-based and event-based triggers. Use sensors (Airflow), freshness policies (Dagster), or event-driven automations (Prefect) to ensure data readiness before processing.

### 5. Ignoring Backfill

**Symptom**: Pipelines that only work for "today" — hardcoded `datetime.now()`, non-partitioned outputs, append-only loads.

**Consequences**: Impossible to reprocess historical data after a bug fix. Data corrections require manual intervention. Recovery from outages is painful.

**Fix**:
- Use the execution date / logical date as a parameter (never `datetime.now()`).
- Partition outputs by date/key.
- Ensure all tasks are idempotent.
- Test backfill explicitly during development.

### 6. Hardcoded Configuration

**Symptom**: Connection strings, file paths, API keys, and dates embedded directly in task code.

**Consequences**: Cannot deploy the same DAG to dev/staging/prod. Secrets leak into version control. Configuration changes require code changes.

**Fix**: Use the orchestrator's configuration mechanisms:
- Airflow: Variables, Connections, Params
- Dagster: Resources, RunConfig, environment variables
- Prefect: Blocks, Variables
- All: Environment-specific config files loaded at runtime

### 7. Non-Idempotent Tasks

**Symptom**: Tasks that append rows, send duplicate notifications, or leave partial state on failure.

**Consequences**: Retries create duplicates. Backfills corrupt data. Manual cleanup required after every failure.

**Fix**: See [Idempotency pattern](#5-idempotency) above. Every task must be safe to re-run.

### 8. Heavy Top-Level Code in DAG Files

**Symptom**: Import-time API calls, database queries, or expensive computation in the DAG definition file.

**Consequences**: Slow DAG parsing (the scheduler re-parses every heartbeat cycle). Failures at parse time are hard to debug. Unnecessary load on external systems.

**Fix**: Keep DAG files lightweight. Move all logic into task callables or modules imported only at task execution time. Never call APIs or query databases at the top level of a DAG file.

### 9. Missing Alerting and SLAs

**Symptom**: No `on_failure_callback`, no SLA configuration, no integration with alerting tools. Pipelines fail silently for hours or days.

**Consequences**: Downstream consumers operate on stale data without knowing. Business decisions made on incorrect information. Incidents discovered by users rather than operators.

**Fix**:
- Set `on_failure_callback` on every production DAG.
- Configure SLAs for critical pipelines (expected completion time).
- Integrate with Slack/PagerDuty/Opsgenie.
- Monitor for "silent success" — a DAG that succeeds but produces no output.

### 10. Mismatched Executor for Workload

**Symptom**: Using LocalExecutor for production (no parallelism beyond threading) or over-provisioning a Celery cluster for five DAGs.

**Consequences**: Performance bottlenecks, wasted infrastructure cost, unexpected resource contention.

**Fix**:
- **LocalExecutor**: Development and testing only.
- **CeleryExecutor**: Medium-scale teams with stable, predictable workloads.
- **KubernetesExecutor**: Dynamic scaling, heterogeneous resource needs, production workloads.
- **Managed services**: When operational overhead outweighs cost.

### 11. Unlimited Fan-Out

**Symptom**: Dynamic task mapping over every file, customer, API record, or partition with no maximum concurrency.

**Consequences**: Scheduler overload, warehouse/API throttling, GPU starvation, runaway cloud spend, and downstream incident cascades.

**Fix**: Put every high-cardinality fan-out behind an explicit concurrency limit, pool, queue, or batch-size control. Load-test the largest expected partition count before enabling automated backfills.

### 12. Hidden Side Effects

**Symptom**: Tasks send emails, mutate production records, trigger webhooks, or update case systems without idempotency keys or audit records.

**Consequences**: Retries duplicate external actions; backfills accidentally notify customers; rollback is impossible.

**Fix**: Treat side effects as durable commands with idempotency keys. Separate "prepare" from "commit" steps, and require explicit approval for irreversible external actions.

### 13. Treating Orchestrator State as Business State

**Symptom**: The only record of business progress is task status in Airflow/Prefect/Dagster UI.

**Consequences**: Replatforming, retention cleanup, or metadata-db recovery can destroy the business audit trail. Human approvals and external decisions become hard to reconstruct.

**Fix**: Store business state in an application database or event log. Let the orchestrator store operational state; let the domain system store domain truth.

### 14. Weak Dependency Pinning

**Symptom**: Workers install unpinned Python packages or mutable container tags such as `latest`.

**Consequences**: A dependency release changes task behaviour without a DAG code change. Historical backfills no longer reproduce prior results.

**Fix**: Pin dependencies with lockfiles or constraints, use immutable image digests for critical jobs, and record code/image/model versions in run metadata.

### 15. Mixing Batch DAGs with Unbounded Agent Loops

**Symptom**: A DAG node starts an LLM agent that can loop, call tools, and wait for humans with no durable state model or budget.

**Consequences**: The scheduler sees one opaque task while the real workflow is hidden inside it. Retries can replay expensive or irreversible tool calls.

**Fix**: Model bounded deterministic steps as DAG tasks. Model unbounded loops, human waits, and tool-call sagas in a durable runtime such as Temporal, Durable Functions, or Step Functions, then trigger that runtime from the outer DAG if scheduling is needed.

---

## Testing Strategies

### Unit Testing Tasks

Test each task function in isolation with known inputs and expected outputs.

```python
# Dagster — materialize_to_memory (strong local testability)
from dagster import materialize_to_memory

def test_cleaned_orders():
    result = materialize_to_memory(
        [raw_orders, cleaned_orders],
        resources={"db": mock_db_resource},
    )
    output = result.output_for_node("cleaned_orders")
    assert len(output) == 2
    assert all(o["amount"] > 0 for o in output)

# Prefect — tasks are just functions
def test_transform_data():
    raw = {"source": "test", "rows": 100}
    result = transform_data.fn(raw)  # .fn bypasses orchestration
    assert result["transformed"] is True

# Airflow — test task logic
def test_extract_logic():
    # Extract the business logic into a testable function
    result = extract_logic(source="test_table", date="2026-01-01")
    assert result["records"] > 0
```

### Integration Testing DAGs

Verify the complete DAG executes correctly in a controlled environment.

```python
# Airflow — DAG validation tests (run in CI)
import pytest
from airflow.models import DagBag

@pytest.fixture
def dag_bag():
    return DagBag(include_examples=False)

def test_no_import_errors(dag_bag):
    assert len(dag_bag.import_errors) == 0

def test_dag_loads(dag_bag):
    dag = dag_bag.get_dag("etl_pipeline")
    assert dag is not None
    assert len(dag.tasks) > 0

def test_no_cycles(dag_bag):
    for dag_id, dag in dag_bag.dags.items():
        # Topological sort will fail if there's a cycle
        assert dag.topological_sort()

# Airflow — full DAG test execution
def test_dag_execution():
    dag = dag_bag.get_dag("etl_pipeline")
    dag.test()  # Runs entire DAG synchronously
```

### Mocking External Systems

Swap real integrations for test doubles.

```python
# Dagster — Resources abstraction makes this trivial
from dagster import resource

@resource
def production_db():
    return PostgresClient(os.environ["DB_URL"])

@resource
def test_db():
    return InMemoryClient()

# Test with mock resource
result = materialize_to_memory(
    [my_asset],
    resources={"db": test_db},
)

# Prefect — standard pytest mocking
from unittest.mock import patch

def test_extract_with_mock():
    with patch("my_module.api_client.fetch") as mock_fetch:
        mock_fetch.return_value = [{"id": 1}]
        result = extract_data.fn(source="api")
        assert result["rows"] == 1
```

### CI Pipeline for DAGs

```yaml
# .github/workflows/dag-ci.yml
name: DAG CI
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - name: Validate DAG imports
        run: python -c "from airflow.models import DagBag; db = DagBag(); assert not db.import_errors, db.import_errors"
      - name: Run unit tests
        run: pytest tests/unit/ -v
      - name: Run integration tests
        run: pytest tests/integration/ -v --timeout=300
      - name: Static analysis
        run: ruff check dags/ --select=ALL
      - name: Type checking
        run: mypy dags/ --ignore-missing-imports
```

### Testing Checklist

| Check | Purpose |
|-------|---------|
| DAG imports without errors | Catch syntax/dependency issues |
| No cycles in graph | Validate DAG structure |
| All tasks have owners | Enforce accountability |
| SLAs configured for critical DAGs | Ensure monitoring coverage |
| Idempotency test (run twice, compare output) | Verify retry safety |
| Backfill test (run for historical date) | Verify parameterisation |
| Resource allocation within limits | Prevent OOM / cluster exhaustion |
| Secrets not in DAG files | Security hygiene |
| Dependency versions pinned | Reproducibility |
| Failure injection | Verify retries, compensation, and alerting |
| Data quality gate fails closed | Prevent bad data publication |
| Large fan-out test | Validate scheduler and downstream capacity |
| Human approval timeout | Verify stalled approvals do not hang forever |

### Failure-Injection Tests

Unit and integration tests prove the happy path. Production DAGs also need tests for the failure paths operators will actually see:

| Scenario | Expected behaviour |
|----------|--------------------|
| Source file missing | Sensor waits or run fails with actionable alert; downstream tasks do not publish empty outputs |
| Source schema changed | Quality gate fails before load; invalid data is quarantined |
| API returns 429/503 | Retry with backoff and jitter; no unbounded retry storm |
| Worker dies mid-task | Task retries from a safe checkpoint or idempotent restart point |
| Partial load succeeds then task fails | Re-run does not duplicate rows or double-send side effects |
| Human approval absent | Workflow times out to a defined state with notification |
| Downstream system unavailable | Circuit breaker or DLQ prevents unrelated branches from failing |

### Backfill and Replay Tests

Backfill is where many DAGs fail because developers accidentally encode "now" instead of the logical execution date.

Before production:
1. Run a historical partition with a fixed logical date.
2. Re-run the same partition and compare outputs.
3. Run adjacent partitions concurrently to expose shared-state bugs.
4. Re-run after a dependency/image/model version change and confirm expected differences.
5. Confirm external side effects are disabled, sandboxed, or idempotent during backfill.

---

## Observability and Lineage

### Logging

| Practice | Implementation |
|----------|---------------|
| **Structured logging** | JSON format with task_id, dag_id, execution_date, run_id for queryability |
| **Centralised aggregation** | Ship logs to ELK, Loki, or CloudWatch for cross-DAG search |
| **Remote log storage** | Airflow: configure remote logging to S3/GCS/Azure Blob (local logs lost with ephemeral workers) |
| **Log levels** | INFO for progress, WARNING for degraded state, ERROR for failures requiring action |
| **Correlation IDs** | Pass a unique ID through the DAG for end-to-end tracing |

### Alerting

**Tiered alerting strategy:**

| Tier | Condition | Action |
|------|-----------|--------|
| **P1 — Critical** | SLA breach on revenue-impacting pipeline | Page on-call (PagerDuty/Opsgenie) |
| **P2 — High** | Task failure with retries exhausted | Slack alert to team channel |
| **P3 — Medium** | Task failure with retries remaining | Log warning, auto-retry |
| **P4 — Low** | Non-critical DAG delay | Dashboard metric, no notification |

```python
# Airflow — on_failure_callback
def alert_on_failure(context):
    task_instance = context["task_instance"]
    slack_client.post(
        channel="#data-alerts",
        text=f"FAILED: {task_instance.dag_id}.{task_instance.task_id} "
             f"on {context['execution_date']}",
    )

with DAG(..., default_args={"on_failure_callback": alert_on_failure}):
    ...
```

### SLA Monitoring

```python
# Airflow — SLA configuration
@task(sla=timedelta(hours=2))
def critical_transform():
    ...

# Dagster — Freshness policies (declarative SLAs)
@asset(freshness_policy=FreshnessPolicy(maximum_lag_minutes=120))
def order_summary():
    ...
```

**Monitor for silent success**: A DAG that "succeeds" but produces empty or trivially small output is often worse than a visible failure. Add data quality checks as downstream tasks.

### Data Lineage with OpenLineage

**OpenLineage** is the open standard for runtime lineage collection. It captures: {S14}
- What datasets a task reads (inputs)
- What datasets a task writes (outputs)
- Task run metadata (duration, status, error)
- Schema information and data quality facets

**Architecture:**

```text
┌─────────┐     ┌─────────────────┐     ┌─────────────┐
│ Airflow │────▶│  OpenLineage    │────▶│   Marquez   │
│ Spark   │     │  Events (HTTP)  │     │  (Backend)  │
│ dbt     │     └─────────────────┘     └─────────────┘
│ Flink   │                                     │
└─────────┘                              ┌──────┴──────┐
                                         │ Lineage UI  │
                                         │ + Query API │
                                         └─────────────┘
```

**Key benefits:**
- Cross-orchestrator compatibility (one lineage graph spanning Airflow, Spark, dbt).
- Blast-radius analysis: "This source table changed — what's affected?"
- Incident triage: trace a data quality issue back to its root cause.
- Compliance: demonstrate provenance for regulatory audits.

**Tools:**
- **Marquez** (open-source): OpenLineage reference backend and UI.
- **Dagster** (built-in): Asset graph provides native lineage without requiring OpenLineage for Dagster-native assets.
- **DataHub** / **OpenMetadata**: Broader data catalogue tools that consume OpenLineage events.

**Integration maturity note**: OpenLineage support is strongest where there is a maintained integration for the engine or orchestrator in use. Airflow, Spark, dbt, and Flink are common first-class examples. For Dagster, native asset lineage may already solve the primary requirement; OpenLineage is usually relevant when you need a cross-platform catalogue that spans Dagster and non-Dagster systems.

### Metadata Management

Track operational metadata alongside lineage:

| Metadata | Purpose |
|----------|---------|
| Row counts per task | Detect data volume anomalies |
| Execution duration | Identify performance regression |
| Data freshness | Ensure downstream consumers see current data |
| Schema changes | Alert on unexpected column additions/removals |
| Cost per run | Track infrastructure spending per pipeline |

```python
# Dagster — first-class metadata on assets
from dagster import asset, MetadataValue

@asset
def order_summary(cleaned_orders: list[dict]) -> dict:
    summary = {"count": len(cleaned_orders), "total": sum(o["amount"] for o in cleaned_orders)}
    # Attach metadata for observability
    context.add_output_metadata({
        "row_count": len(cleaned_orders),
        "total_revenue": MetadataValue.float(summary["total"]),
        "schema": MetadataValue.json({"count": "int", "total": "float"}),
    })
    return summary
```

---

## Migration and Lifecycle Management

Workflow systems become hard to change because DAG definitions, task code, historical run metadata, data contracts, and external systems evolve together. Treat orchestration changes as production migrations.

### Versioning Strategy

| Item | What to version |
|------|-----------------|
| DAG/workflow definition | Git commit, DAG version, task graph hash |
| Task runtime | Container image digest or locked dependency set |
| Data contracts | Schema version and validation rules |
| Prompt/model calls | Prompt template version, model ID, tool schema version |
| Infrastructure | Executor/work pool/queue config, service account, resource limits |
| Outputs | Partition/date, source snapshot, transformation version |

Airflow 3 adds DAG version visibility, but versioning should not stop at the scheduler UI. Persist versions with outputs so a downstream consumer can answer: "Which code produced this table/report/model?"

### Deployment Patterns

| Pattern | Use when |
|---------|----------|
| **Parse-only CI** | Every DAG change; catches import errors and invalid graph definitions |
| **Canary DAG run** | New scheduler/executor/runtime versions |
| **Blue/green workers** | Dependency or base-image upgrades |
| **Paused rollout** | New DAGs that should be reviewed before first scheduled run |
| **Shadow run** | New logic should be compared with old logic before switching consumers |
| **Dual-write** | Output format/schema migration with downstream consumers on different schedules |

Never deploy a breaking DAG contract and a downstream consumer change as an uncoordinated pair. Prefer additive schema changes, compatibility windows, and explicit deprecation dates.

### Upgrade Checklist

1. Read orchestrator release notes and migration guide.
2. Pin the old runtime and export current metadata/backups.
3. Run parse/import tests against the new version.
4. Run representative historical backfills in a staging environment.
5. Validate custom operators, plugins, providers, secrets backends, and UI auth.
6. Confirm scheduler/executor capacity and database migrations.
7. Roll out to non-critical DAGs first.
8. Keep rollback instructions executable, not just documented.

### Deprecating DAGs

Deleting a DAG is also a migration.

| Step | Reason |
|------|--------|
| Mark deprecated in metadata/tags | Make ownership and timeline visible |
| Notify downstream consumers | Prevent silent data disappearance |
| Disable schedule before deletion | Prove no active dependency remains |
| Archive code and run history | Preserve auditability |
| Remove credentials and infrastructure | Reduce security and cost footprint |
| Remove catalogue/lineage entries or mark inactive | Avoid stale discovery results |

### When to Replatform

Replatform only when the current tool blocks a material requirement:

- Airflow to Dagster: asset lineage, partitioning, and data-quality ownership are more important than task scheduling familiarity.
- Airflow/Prefect to Temporal: long-running human workflows, durable application state, or sagas are becoming awkward in batch-task form.
- Python orchestrator to Argo/Flyte: container isolation, GPUs, and Kubernetes-native execution matter more than Python developer ergonomics.
- Self-hosted to managed: upgrades, availability, and platform operations are distracting the team from product/data work.
- Managed to self-hosted: air-gapped deployment, custom executors, or compliance controls are impossible in the managed service.

Migration cost is usually dominated by semantics, not syntax: retry behaviour, state persistence, backfill model, secrets, lineage, and alerting rarely map one-to-one.

---

## Managed Services on Hyperscalers

### AWS

| Service | Description | When to use |
|---------|-------------|-------------|
| **AWS MWAA** | Managed Apache Airflow | Teams with existing Airflow DAGs wanting reduced ops burden on AWS; verify supported Airflow versions before migration |
| **AWS Step Functions** | Serverless state machine orchestration (Standard + Express modes). {S11} | Serverless architectures, event-driven microservices, minimal ops overhead |
| **Step Functions Distributed Map** | High-scale parallel processing over S3 data sources. {S11} | High-volume parallel processing (log analysis, batch transforms) |
| **AWS Glue Workflows** | Managed ETL orchestration for Glue jobs | Simple ETL chains within the Glue ecosystem |

**Step Functions pricing model:**
- Standard: per state transition (long-running, complex)
- Express: per execution, duration, and memory (short, high-volume)

**Key Step Functions features (2025–2026):**
- Distributed Map: high-scale fan-out over S3 data sources with managed child executions. {S11}
- HTTP Task: call HTTPS APIs using EventBridge Connections without a Lambda wrapper for simple API calls. {S11}
- Redrive: retry failed Standard workflow executions from unsuccessful states without restarting all successful work. {S11}
- Intrinsic functions: data transformation without Lambda

**AWS watch-outs**:
- Step Functions Standard workflows are durable but priced per state transition; small state-machine changes can materially change cost.
- Express workflows fit high-volume short executions but have different durability, history, and observability tradeoffs.
- Distributed Map can exceed downstream capacity quickly; set concurrency limits and design idempotent item processors.

### Azure

| Service | Description | When to use |
|---------|-------------|-------------|
| **Azure Data Factory** | Managed ETL orchestration | Existing ADF users, simple copy/transform pipelines, hybrid integration runtime needs |
| **Microsoft Fabric Data Factory** | Fabric-native data integration and pipeline experience | New Fabric-centric analytics projects and teams standardising on OneLake/Fabric |
| **Azure Logic Apps** | Serverless workflow automation with a large connector ecosystem. {S13} | Integration workflows, SaaS-to-SaaS automation |
| **Azure Durable Functions** | Stateful serverless orchestration (C#, Python, JS). {S13} | Custom stateful workflows within Azure Functions |

**Fabric Data Factory**:
- AI-powered pipeline generation
- Integrated data governance via Microsoft Purview
- Unified with Power BI, Synapse, and OneLake
- Migration from ADF should be evaluated feature-by-feature; do not assume every ADF pattern has a one-click Fabric equivalent

**Azure watch-outs**:
- Durable Functions orchestrator code has determinism constraints similar in spirit to Temporal; put side effects in activities. {S13}
- Logic Apps is strong for integration workflows but can become hard to manage as application logic grows; keep business-critical logic versioned and tested.

### GCP

| Service | Description | When to use |
|---------|-------------|-------------|
| **Cloud Composer / Google Managed Service for Apache Airflow** | Managed Apache Airflow, including newer serverless/Gen 3 deployment options. {S12} | Complex data pipelines on GCP needing the full Airflow ecosystem; verify exact supported Airflow version and support stage |
| **Google Cloud Workflows** | Serverless workflow orchestration (YAML/JSON). {S12} | Lightweight GCP service orchestration, event-driven via Eventarc |
| **Vertex AI Pipelines** | Managed ML pipeline orchestration (KFP-based). {S12} | ML training and deployment pipelines on Vertex AI |
| **Cloud Dataflow** | Managed Apache Beam for streaming and batch | Real-time streaming ETL, unified batch/stream processing |

**Cloud Composer / Gen 3 direction**:
- Reduced cluster-management burden compared with earlier Composer generations
- Autoscaling workers based on queue depth
- Native BigQuery, Dataflow, and Vertex AI integrations
- Airflow 3 support depends on the managed service version list and may be in preview/GA depending on date and region. {S12}

**GCP watch-outs**:
- Cloud Workflows is excellent for service orchestration, but it is not a replacement for data-engine lineage, partition-aware backfills, or Python-native data transformation testing.
- Vertex AI Pipelines is a better fit than general workflow tools when the pipeline is primarily ML training/deployment inside Vertex AI.

### IBM

| Service | Description | When to use |
|---------|-------------|-------------|
| **IBM DataStage** | Enterprise ETL orchestration | Legacy IBM environments, mainframe integration |
| **IBM Cloud Pak for Data** | Unified data and AI platform with pipeline orchestration | IBM-centric enterprises with existing Cloud Pak investments |

### Oracle

| Service | Description | When to use |
|---------|-------------|-------------|
| **Oracle Data Integrator (ODI)** | Enterprise data integration and orchestration | Oracle database-centric environments |
| **OCI Data Integration** | Cloud-native ETL service with workflow support | Oracle Cloud workloads, Oracle DB sources/targets |

---

## Modern Trends (2025–2026)

### 1. Asset-Based Scheduling

One of the most significant paradigm shifts in data orchestration. Instead of defining only "what tasks should run and when", define "what data should exist and how fresh it should be."

**Key insight**: Business users care about data products (tables, reports, features), not the mechanics of the tasks that produce them. Asset-based orchestration aligns the system's model with business language.

| Aspect | Task-centric (traditional) | Asset-centric (modern) |
|--------|---------------------------|------------------------|
| **Core question** | "What should run at 06:00?" | "What data should exist, and is it fresh enough?" |
| **Scheduling trigger** | Cron expression | Freshness policy violation |
| **Documentation** | Separate from code | The asset graph IS the documentation |
| **Lineage** | Requires external tooling (OpenLineage) | Built into the system |
| **Testing** | Test task execution | Test data transformations directly |

**Adoption**: Dagster is the clearest asset-first orchestrator. Airflow has moved toward data-aware scheduling through datasets/assets, but its core model remains task/DAG oriented.

### 2. Event-Driven DAGs

Moving beyond pure cron: pipelines trigger on data arrival, API events, or system signals.

**Drivers:**
- Real-time analytics demands lower latency than hourly batch.
- Cloud-native architectures produce events naturally (S3 notifications, Kafka topics, webhooks).
- Cost efficiency: don't run pipelines when there's nothing to process.

**Implementation across tools:**
- Airflow 3.0: Dataset events/assets, external triggers via deferrable operators. {S1}{S4}
- Prefect: Event-driven automations with custom triggers. {S5}
- Kestra: Event-first architecture with webhook and messaging/storage triggers. {S10}
- Step Functions: EventBridge integration for reactive orchestration. {S11}

### 3. DAGs-as-Code and Declarative Pipelines

Infrastructure-as-Code principles applied to orchestration:

- Pipelines defined in version-controlled code, reviewed via pull request.
- Changes to pipeline structure require the same rigour as application code.
- Enables reproducibility, auditability, and collaboration.
- Declarative definitions (Kestra YAML, Argo YAML, Dagster assets) preferred over imperative scripts.

**Implication**: Teams adopt the same Git branching, PR review, and CI validation for DAGs as for application code. Pipeline changes go through code review.

### 4. Serverless Orchestration

Fully managed, pay-per-execution orchestration with zero infrastructure:

- AWS Step Functions (Standard and Express)
- Google Cloud Workflows
- Azure Logic Apps / Durable Functions
- Cloud Composer 3 (serverless Airflow)

**Tradeoff**: Reduced operational burden vs. limited customisation, potential vendor lock-in, and cost unpredictability at scale.

### 5. AI/ML-Specific Orchestration

Purpose-built tools for AI workloads are maturing:

- **Flyte / Union.ai**: Typed, resource-aware orchestration for ML pipelines; validate Flyte 2 maturity and hosted-vs-OSS feature parity before standardising. {S8}
- **Temporal + AI agents**: Durable execution for LLM agent loops, tool calls, and human checkpoints. {S7}
- **LangGraph**: DAG-based control flow for multi-step LLM reasoning.
- **GPU-aware scheduling**: First-class support for heterogeneous hardware (A100, H100, TPU).

**Emerging pattern**: The DAG model works well for bounded *batch ML* (training, feature engineering, evaluation). Open-ended agents often need loops, external events, and durable state; use DAGs for bounded outer scheduling and durable workflow runtimes for the inner agent loop.

### 6. OpenLineage as Universal Lineage Standard

OpenLineage adoption is accelerating as the cross-platform lineage protocol:

- Integrations with common engines and orchestrators such as Airflow, Spark, dbt, and Flink. {S14}
- Marquez as the open-source reference backend.
- Enables "one lineage graph" spanning multiple orchestrators and processing engines.
- Works best when paired with disciplined dataset naming and ownership; lineage events alone do not create usable governance.

### 7. Hybrid Orchestration Stacks

Production systems increasingly use multiple orchestrators together:

```text
┌────────────────────────────────────────────────────┐
│  Outer Scheduler (Airflow / Dagster)               │
│  - Cron-driven batch pipelines                     │
│  - Data quality checks                             │
│  - ML training runs                                │
├────────────────────────────────────────────────────┤
│  Inner Runtime (Temporal)                          │
│  - Long-running stateful workflows                 │
│  - AI agent execution                              │
│  - Human-in-the-loop approvals                     │
├────────────────────────────────────────────────────┤
│  Micro-Orchestration (Hamilton / Kedro)            │
│  - Within-task dataflow logic                      │
│  - Feature engineering DAGs                        │
│  - Testable transformation graphs                  │
└────────────────────────────────────────────────────┘
```

Each layer serves a different purpose; no single tool covers all requirements optimally.

---

## Decision Framework

Use this flowchart to select a starting point, then validate against existing platform investment, team skills, compliance requirements, and managed-service availability.

```text
START
  │
  ├─ Is the workflow long-running (hours/days/months)?
  │   └─ YES → Temporal
  │
  ├─ Is it AI agent orchestration with durable state?
  │   └─ YES → Temporal (+ outer scheduler for triggering)
  │
  ├─ Is it an ML training pipeline needing GPU scheduling?
  │   └─ YES → Flyte (or Argo Workflows if already on K8s)
  │
  ├─ Is it serverless on AWS with minimal ops?
  │   └─ YES → Step Functions
  │
  ├─ Is it Kubernetes-native CI/CD or container batch?
  │   └─ YES → Argo Workflows
  │
  ├─ Is it a data platform with lineage + quality focus?
  │   └─ YES → Dagster
  │
  ├─ Is it event-driven with mixed languages (not Python-only)?
  │   └─ YES → Kestra
  │
  ├─ Is it a new Python project wanting minimal boilerplate?
  │   └─ YES → Prefect
  │
  ├─ Is it scheduled batch ETL needing mature integrations?
  │   └─ YES → Airflow (managed: MWAA / Composer 3 / Astronomer)
  │
  └─ Do you just need to structure transform logic within a task?
      └─ YES → Hamilton or Kedro (complementary to any orchestrator)
```

### Workload-to-Tool Fit

| Workload shape | Strong candidates | Avoid by default |
|----------------|-------------------|------------------|
| Daily/hourly ETL with many integrations | Airflow, Dagster, Prefect | Temporal as the primary scheduler |
| Data products with freshness and lineage | Dagster, Airflow + OpenLineage | Untracked scripts |
| Python-first lightweight automation | Prefect | Heavy Airflow deployment for trivial flows |
| Months-long business process | Temporal, Durable Functions, Step Functions Standard | Airflow sensors waiting for months |
| AWS-native service choreography | Step Functions | Self-hosted orchestrator unless portability is required |
| Kubernetes-native container DAGs | Argo Workflows | Serverless state machine if task images and cluster locality matter |
| ML training with resource typing | Flyte, Vertex AI Pipelines, Argo | General ETL scheduler without resource isolation |
| LLM agent with loops/human gates | Temporal, LangGraph + durable state, Step Functions/Durable Functions | Pure acyclic DAG as the only runtime |

### Choosing Managed vs Self-Hosted

| Factor | Managed | Self-hosted |
|--------|---------|-------------|
| **Ops burden** | Provider handles upgrades, scaling, availability | Your team owns uptime, patches, capacity planning |
| **Cost** | Higher unit cost, predictable billing | Lower unit cost, unpredictable ops labour |
| **Customisation** | Limited to provider's configuration surface | Full control over plugins, executors, infrastructure |
| **Compliance** | Provider's certifications may cover you | You own the entire compliance surface |
| **Scale ceiling** | Provider limits (may be generous) | Limited by your infrastructure investment |
| **Vendor lock-in** | Medium (Airflow managed services are portable; Step Functions are not) | None |

**Recommendation**: Start with managed services unless you have specific requirements that demand self-hosting (custom executors, air-gapped environments, regulatory constraints requiring full infrastructure control).

---

## Claim-Source-Confidence Matrix

This matrix maps evidence markers used in the guide to the sources that support them. "Confidence" is about source support and stability as of 2026-05-27, not about future validity.

| ID | Claim area | Primary sources | Confidence | Notes |
|----|------------|-----------------|------------|-------|
| **S1** | Airflow 3.0 introduced major architecture changes including Task SDK, API server changes, DAG versioning, event-driven scheduling, UI updates, and backfill changes | Apache Airflow 3.0 announcement; Airflow Task SDK docs; Airflow assets docs | High | Airflow details are official but feature behaviour can vary by deployment and provider support |
| **S2** | Apache Airflow 3.2.1 was the verified current PyPI release when this guide was updated | PyPI `apache-airflow`; Apache Airflow announcements | High | Version claims are volatile; re-check before upgrade |
| **S3** | Airflow DAG files should avoid expensive top-level code; sensor modes/deferrable patterns affect worker-slot usage | Airflow best practices; Airflow sensors/deferrable operator docs | High | Operational impact depends on executor and deployment |
| **S4** | Airflow supports dynamic task mapping and data-aware scheduling through assets/datasets | Airflow dynamic task mapping docs; Airflow assets docs | High | Naming shifted from "datasets" toward "assets" in newer docs |
| **S5** | Prefect supports flows/tasks, work pools, automations, transactions, retries, caching, and current 3.x releases | Prefect v3 docs; Prefect GitHub releases; PyPI `prefect` | High | Prefect Cloud features may differ from open-source server/self-hosted deployments |
| **S6** | Dagster supports software-defined assets, partitions, automation, asset checks, local asset testing, virtual assets, and 1.13.x releases | Dagster docs; Dagster 1.13 release; Dagster GitHub releases; PyPI `dagster` | High | Dagster release cadence is active; verify patch version before production |
| **S7** | Temporal workflows use event history replay and require deterministic workflow code; activities, signals, retries, and worker versioning are core production concepts | Temporal workflow, determinism, activities, signals, retry policy, and worker-versioning docs | High | SDK-specific APIs differ by language |
| **S8** | Flyte is typed/resource-aware for ML workflows; Flyte 2 has a local-first/Devbox path and production-readiness should be verified separately | Flyte 2 OSS docs; Union.ai Devbox docs; Flyte task/workflow docs; Union.ai Flyte 2 announcement | Medium-High | Flyte 2 is the least stable/mature claim area in this guide; validate before adopting |
| **S9** | Argo Workflows is Kubernetes-native, YAML/CRD based, supports DAG templates, artifacts, retries, and 4.0.x releases | Argo Workflows docs and GitHub releases | High | Cluster operations/security are outside Argo docs but materially affect production readiness |
| **S10** | Kestra is YAML-first, event-driven, and supports triggers/plugins across multiple systems | Kestra docs; Kestra triggers docs; Kestra GitHub | Medium-High | Plugin maturity varies by integration |
| **S11** | AWS Step Functions supports Standard/Express modes, Distributed Map, HTTP Task, redrive, and state-transition pricing | AWS Step Functions docs and pricing | High | AWS service limits and regional availability can change |
| **S12** | Google Cloud has Cloud Composer / Managed Service for Apache Airflow, Cloud Workflows, and Vertex AI Pipelines; Airflow version support must be checked against provider version lists | Google Cloud Composer docs/version list; Cloud Workflows docs; Vertex AI Pipelines docs | High | Managed-service support stage can vary by region/date |
| **S13** | Azure Durable Functions has deterministic orchestrator constraints; Logic Apps and Fabric Data Factory cover integration/data-pipeline scenarios | Microsoft Durable Functions overview/code constraints; Logic Apps connector docs; Fabric Data Factory docs | High | Azure service naming and Fabric migration paths are evolving |
| **S14** | OpenLineage is a runtime lineage standard with integrations for common engines/orchestrators including Airflow, Spark, dbt, and Flink; Marquez is an OSS backend | OpenLineage docs/integrations; Marquez GitHub | High | Integration depth varies by engine and version |
| **S15** | GitHub Actions and GitLab CI support DAG-like job dependencies | GitHub Actions jobs docs; GitLab directed acyclic graph pipeline docs | High | CI DAG semantics are not equivalent to data-orchestration semantics |
| **S16** | Great Expectations, Soda, and Dagster asset checks are suitable data-quality gate mechanisms | Great Expectations docs; Soda docs; Dagster asset check docs | High | Tool fit depends on data platform and test ownership |

### Claims Deliberately Not Made

- No claim that one orchestrator is universally "best".
- No unsourced market-share claim such as "X organisations" or "Y monthly downloads".
- No throughput or cost benchmark across orchestrators.
- No claim that managed-service support for the newest open-source version is immediate.
- No claim that OpenLineage integration maturity is equal across all engines.

---

## Source Quality Notes

This revision prioritises primary sources:

1. Official project documentation and release notes.
2. Package indexes or GitHub releases for version checks.
3. Cloud-provider documentation for managed-service capabilities.
4. Secondary blogs only for interpretation, never as sole support for factual feature claims.

Claims that are deliberately phrased as judgement calls ("best fit", "watch-out", "conservative choice") are engineering synthesis based on the documented feature model, not vendor benchmark results. Where a claim depends on exact version or service tier, the document says to verify before deployment.

---

## References

### Apache Airflow

- Apache Airflow. "Airflow 3.0 Is Here." https://airflow.apache.org/blog/airflow-three-point-oh-is-here/
- Apache Airflow. "Announcements." https://airflow.apache.org/announcements/
- Apache Airflow. "Task SDK Documentation." https://airflow.apache.org/docs/task-sdk/stable/
- Apache Airflow. "Best Practices." https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html
- Apache Airflow. "Dynamic Task Mapping." https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/dynamic-task-mapping.html
- Apache Airflow. "Params." https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/params.html
- Apache Airflow. "Assets." https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/assets.html
- Apache Airflow. "Sensors." https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/sensors.html
- Apache Airflow. "Deferring execution." https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/deferring.html
- PyPI. "apache-airflow." https://pypi.org/project/apache-airflow/

### Prefect

- Prefect. "Flows." https://docs.prefect.io/v3/concepts/flows
- Prefect. "Tasks." https://docs.prefect.io/v3/concepts/tasks
- Prefect. "Work pools." https://docs.prefect.io/v3/concepts/work-pools
- Prefect. "Automations." https://docs.prefect.io/v3/automate/
- Prefect. "Transactions." https://docs.prefect.io/v3/advanced/transactions
- PrefectHQ. "Prefect releases." https://github.com/PrefectHQ/prefect/releases
- PyPI. "prefect." https://pypi.org/project/prefect/

### Dagster

- Dagster. "Assets." https://docs.dagster.io/guides/build/assets
- Dagster. "Partitions." https://docs.dagster.io/guides/build/partitions-and-backfills/partitioning-assets
- Dagster. "Automation." https://docs.dagster.io/guides/automate
- Dagster. "Asset checks." https://docs.dagster.io/guides/test/asset-checks
- Dagster. "Testing assets and ops." https://docs.dagster.io/guides/test/unit-testing-assets-and-ops
- Dagster. "Dagster 1.13 release." https://dagster.io/blog/dagster-1-13-octopuss-garden
- Dagster. "GitHub releases." https://github.com/dagster-io/dagster/releases
- PyPI. "dagster." https://pypi.org/project/dagster/

### Temporal

- Temporal. "Workflows." https://docs.temporal.io/workflows
- Temporal. "Workflow determinism." https://docs.temporal.io/workflow-definition
- Temporal. "Activities." https://docs.temporal.io/activities
- Temporal. "Signals." https://docs.temporal.io/develop/python/message-passing
- Temporal. "Retry policies." https://docs.temporal.io/encyclopedia/retry-policies
- Temporal. "Worker Versioning." https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning

### Flyte and Argo Workflows

- Flyte. "Flyte 2 OSS." https://flyte.org/platform/flyte-2
- Union.ai. "Run on the Devbox." https://www.union.ai/docs/v2/flyte/user-guide/run-modes/running-devbox/
- Flyte. "Tasks." https://docs.flyte.org/en/latest/user_guide/flyte_fundamentals/tasks.html
- Flyte. "Workflows." https://docs-legacy.flyte.org/en/latest/user_guide/concepts/main_concepts/workflows.html
- Union.ai. "Introducing Flyte 2.0." https://www.union.ai/blog-post/introducing-flyte-2-0-dynamic-crash-proof-resource-aware-ai-orchestration
- Argo Workflows. "DAG." https://argo-workflows.readthedocs.io/en/latest/walk-through/dag/
- Argo Workflows. "Artifacts." https://argo-workflows.readthedocs.io/en/latest/walk-through/artifacts/
- Argo Workflows. "Retrying failed or errored steps." https://argo-workflows.readthedocs.io/en/latest/retries/
- Argo Workflows. "Releases." https://github.com/argoproj/argo-workflows/releases

### Kestra and Complementary Tools

- Kestra. "Documentation." https://kestra.io/docs/
- Kestra. "Triggers." https://kestra.io/docs/workflow-components/triggers
- Kestra. "Namespace Files and IaC." https://kestra.io/docs/concepts/namespace-files
- Kestra. "GitHub repository." https://github.com/kestra-io/kestra
- Apache Hamilton. "Hamilton Documentation." https://hamilton.dagworks.io/
- Kedro. "Kedro Documentation." https://docs.kedro.org/
- Mage AI. "Mage GitHub." https://github.com/mage-ai/mage-ai
- Luigi. "Luigi GitHub." https://github.com/spotify/luigi

### Managed Services

- AWS. "What is AWS Step Functions?" https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html
- AWS. "Distributed Map state." https://docs.aws.amazon.com/step-functions/latest/dg/state-map-distributed.html
- AWS. "Redriving executions." https://docs.aws.amazon.com/step-functions/latest/dg/redrive-executions.html
- AWS. "Call HTTPS APIs." https://docs.aws.amazon.com/step-functions/latest/dg/call-https-apis.html
- AWS. "Step Functions pricing." https://aws.amazon.com/step-functions/pricing/
- Google Cloud. "Cloud Composer documentation." https://cloud.google.com/composer/docs
- Google Cloud. "Managed Service for Apache Airflow version list." https://docs.cloud.google.com/composer/docs/composer-versions
- Google Cloud. "Workflows overview." https://cloud.google.com/workflows/docs/overview
- Google Cloud. "Vertex AI Pipelines." https://cloud.google.com/vertex-ai/docs/pipelines/introduction
- Microsoft. "Azure Durable Functions overview." https://learn.microsoft.com/en-us/azure/azure-functions/durable/durable-functions-overview
- Microsoft. "Durable orchestrator code constraints." https://learn.microsoft.com/en-us/azure/durable-task/common/durable-task-code-constraints
- Microsoft. "Azure Logic Apps connectors." https://learn.microsoft.com/en-us/azure/connectors/apis-list
- Microsoft. "Data Factory in Microsoft Fabric." https://learn.microsoft.com/en-us/fabric/data-factory/data-factory-overview

### Lineage, Data Quality, and CI/CD

- OpenLineage. "OpenLineage Documentation." https://openlineage.io/docs/
- OpenLineage. "Airflow integration." https://openlineage.io/docs/integrations/airflow/
- OpenLineage. "Integrations." https://github.com/OpenLineage/OpenLineage/tree/main/integration
- Marquez. "Marquez GitHub." https://github.com/MarquezProject/marquez
- Great Expectations. "GX Core." https://docs.greatexpectations.io/docs/core/introduction/
- Soda. "Soda Core." https://docs.soda.io/soda-core/overview-main.html
- GitHub. "Using jobs in a workflow." https://docs.github.com/en/actions/using-jobs
- GitLab. "Make jobs start earlier with needs." https://docs.gitlab.com/ci/yaml/needs/
