# Edge-First AI Clinical Documentation

| Field | Value |
|-------|-------|
| Created | 2026-05-27 |
| Last Updated | 2026-05-30 |
| Version | 1.3 |

---

- [Overview](#overview)
- [Deployment Options — Decision Framework](#deployment-options--decision-framework)
- [Core AI Pipelines](#core-ai-pipelines)
- [Clinician-in-the-Loop Approval Workflow](#clinician-in-the-loop-approval-workflow)
- [Data Flow Architecture](#data-flow-architecture)
- [Edge Compute Requirements](#edge-compute-requirements)
- [Caching and Clinical Audit Trail](#caching-and-clinical-audit-trail)
- [Connectivity Patterns](#connectivity-patterns)
- [Medical Device Integration](#medical-device-integration)
- [Technology Stack](#technology-stack)
- [Safety and Regulatory Considerations](#safety-and-regulatory-considerations)
- [Scale Guidance](#scale-guidance)
- [Decision Matrix](#decision-matrix)
- [Engineering Readiness Pack](#engineering-readiness-pack)
- [References](#references)

## Overview

This reference architecture provides a reusable blueprint for deploying AI-assisted clinical documentation systems in environments with intermittent or unreliable connectivity. It addresses the core challenge of enabling real-time AI processing (speech-to-text, image OCR, clinical note structuring) while maintaining strict clinical governance requirements including clinician-in-the-loop approval and complete audit trails.

The architecture is derived from patterns proven in aeromedical retrieval, remote/rural health, disaster response, and military medical operations — any setting where clinicians must document care in connectivity-constrained environments and later synchronise with central Electronic Health Record (EHR) systems.

## Deployment Options — Decision Framework

Three deployment topologies are presented as a progressive spectrum. Organisations should select based on their connectivity profile, regulatory posture, and operational constraints.

### Option 1: Cloud-First

All AI inference runs in the cloud. The mobile application handles audio/image preprocessing, local queuing, and retry logic over satellite or intermittent cellular links.

**Architecture**: Mobile App → Connectivity Layer → API Gateway (STT + Image endpoints) → AI Services (transcription, OCR, LLM structuring) → Clinician Approval → EHR Updater → EHR System.

**Best for**: Environments with reliable connectivity (>90% uptime), urban clinics with cellular backup, telehealth workflows. Simplest to operate and update.

### Option 2: Cloud + Edge Hybrid

Introduces an edge compute layer between the mobile application and cloud services. The edge handles device connections, data packaging, local preprocessing, and hosts the Clinician Approval Manager. Heavy inference (large LLMs) remains in the cloud.

**Architecture**: Mobile App → Edge Compute (device management, preprocessing, approval workflow) → Connectivity Layer → Cloud (LLM inference, EHR integration).

**Best for**: Intermittent connectivity with predictable windows (e.g., scheduled satellite passes), environments needing local approval workflows during disconnection, settings where data must not leave a facility until approved.

### Option 3: Edge-First

The majority of AI processing executes on local edge hardware. The cloud is minimal — primarily identity/auth services and the EHR update endpoint. The full STT pipeline, image OCR, and record compilation run locally.

**Architecture**: Mobile App → Edge Compute (STT pipeline, OCR service, LLM structuring, Clinician Approval Manager, local cache) → Connectivity Layer (opportunistic) → Cloud (IAM/Auth, EHR Update Endpoint).

**Best for**: Unreliable or absent connectivity (satellite-only, airborne, maritime, disaster zones), high data-sensitivity environments, settings requiring sub-second response times regardless of connectivity state.

## Core AI Pipelines

### Speech-to-Text (STT) Pipeline

```
Voice Recording → Download Manager → Ingestion Service → Transcription Service
→ LLM Structuring → Output Dispatcher → Clinician Approval → EHR Updater
```

The ingestion service handles audio normalisation (sample rate, noise reduction, VAD segmentation). The transcription service produces raw text. The LLM structuring step converts free-text narration into structured clinical fields (presenting complaint, observations, interventions, medications administered).

### Image-to-Text (I2T) Pipeline

```
Photo Capture → Download Manager → Ingestion Service → OCR Service
→ Formatting Service → Output Dispatcher → Clinician Approval → EHR Updater
```

Handles photographs of handwritten notes, medication labels, monitor screens, paper forms, and patient identification. The formatting service maps extracted text into the target clinical document schema.

**Alternative approach**: A multimodal LLM (e.g., Llama 4 Scout with vision, GPT-4o) can replace the separate OCR + Formatting pipeline as a single model, reducing component count at the cost of significantly higher compute requirements on edge. Llama 4 Scout's MoE architecture (16 experts, 109B total parameters with 17B active per forward pass) requires all expert weights resident in memory, meaning ~50 GB VRAM even with int4 quantisation — making it impractical for typical edge hardware.

## Clinician-in-the-Loop Approval Workflow

All AI-generated outputs require explicit clinician confirmation before any EHR modification. This is non-negotiable in clinical settings — AI assists but never autonomously writes to the patient record.

**Confirmation API endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/documents/upload` | POST | Submit processed AI output for review |
| `/api/v1/documents/{id}/status` | GET | Check approval state |
| `/api/v1/documents/{id}` | GET | Retrieve full document for review |
| `/api/v1/documents/{id}/confirm` | POST | Clinician approves (with optional edits) |
| `/api/v1/documents/{id}/reject` | POST | Clinician rejects with reason |

The approval interface must function fully offline on the edge device. Approved documents are queued for EHR submission when connectivity is available.

## Data Flow Architecture

```
Clinician (Photos / Voice / Device Data)
         │
         ▼
    Mobile App (preprocessing, compression, queuing)
         │
         ▼
    Connectivity Layer (Starlink / satellite / cellular / Wi-Fi)
         │
         ▼
    Processing Pipeline (edge or cloud, per deployment option)
         │
         ▼
    Clinician Approval (review, edit, confirm/reject)
         │
         ▼
    EHR Updater (FHIR bundle construction, submission)
         │
         ▼
    EHR System (hospital/regional health record)
```

Priority queuing ensures critical clinical data (e.g., medication administration, emergency interventions) is transmitted first when bandwidth is constrained.

## Edge Compute Requirements

### Model Sizing for Edge Deployment

| Model | VRAM (GPU) | Disk | Use Case |
|-------|------------|------|----------|
| Whisper Large v3 (1.5B params) | ~10 GB | ~3 GB | STT (best accuracy) |
| Whisper Turbo (809M params) | ~6 GB | ~1.6 GB | STT (near-large accuracy, ~8x faster) |
| Whisper Medium (769M params) | ~5 GB | ~1.5 GB | STT (balanced) |
| Whisper Small (244M params) | ~2 GB | ~500 MB | STT (lightweight, suitable for resource-constrained edge) |
| PaddleOCR / Tesseract | <1 GB | <500 MB | Document OCR |
| Phi-4 Mini 3.8B (Q4 quantised) | ~3 GB | ~2.5 GB | Clinical structuring (efficient, good quality) |
| Mistral 7B (Q4 quantised) | ~6 GB | ~4 GB | Clinical structuring (higher quality) |
| Llama 4 Scout 17Bx16E (Q4 quantised) | ~50 GB | ~45 GB | Clinical structuring (best quality, requires dedicated GPU) |

Note: VRAM figures are for GPU inference (as documented in the OpenAI Whisper repository). For CPU-only inference, models load into system RAM with similar memory requirements plus overhead. Whisper Turbo uses the same encoder as Large v3 but with a reduced decoder (4 layers vs 32), achieving near-large accuracy at approximately 8x the inference speed — making it an excellent choice for real-time edge STT where a GPU is available. Turbo does not support translation tasks; use Medium or Large for multilingual translation.

Llama 4 Scout is a Mixture-of-Experts (MoE) model with 17B active parameters per forward pass and 16 experts (~109B total parameters). All expert weights must be resident in memory even though only a subset activate per token, making it unsuitable for typical edge hardware without a high-end GPU (e.g., NVIDIA A100 80 GB or H100). For most edge deployments, Phi-4 Mini or Mistral 7B are the practical choices.

**Minimum edge hardware (Whisper Small + Phi-4 Mini)**: 16 GB RAM, 4-core CPU, 50 GB SSD. This enables real-time STT and clinical structuring on modest hardware.

**Recommended edge hardware (Whisper Medium + Mistral 7B)**: 32 GB RAM, 8-core CPU, 100 GB SSD, optional GPU (NVIDIA Jetson Orin Nano or similar for accelerated inference). A ruggedised mini-PC or medical-grade tablet with attached compute module is typical.

### Sync Patterns

- **Immediate sync**: When connectivity is available, approved documents sync in real-time.
- **Batch sync**: During connectivity windows, all queued documents transmit in priority order.
- **Conflict resolution**: Edge-generated documents carry a monotonic sequence ID and timestamp. The cloud EHR updater uses last-write-wins with clinician arbitration for conflicts.

## Caching and Clinical Audit Trail

All processing artefacts must be retained for clinical and medico-legal audit:

| Cache Layer | Contents | Retention |
|-------------|----------|-----------|
| Blob storage | Original audio files, photographs | Minimum 7 years (jurisdiction-dependent) |
| Transcription cache | Raw STT output, intermediate transcripts | As per blob storage |
| OCR cache | Raw OCR output, bounding boxes, confidence scores | As per blob storage |
| Inference/output cache | LLM-structured outputs, approval decisions, clinician edits | As per blob storage |

Edge storage must be sized for the maximum expected disconnection period. When sync completes, artefacts replicate to cloud object storage (encrypted at rest, access-logged).

## Connectivity Patterns

- **Store-and-forward**: Default mode. All outputs queue locally and forward when connectivity returns. The mobile app and edge compute both maintain independent queues.
- **Opportunistic sync**: Background process continuously probes for connectivity and transmits queued items without user intervention.
- **Priority queuing**: Critical clinical events (cardiac arrest documentation, medication errors, time-critical interventions) are tagged high-priority and transmit first.
- **Conflict resolution**: If a record is modified on both edge and cloud (e.g., a clinician updates a note on a second device), the system flags the conflict for manual clinician resolution rather than auto-merging.

## Medical Device Integration

On-board medical devices (monitors, ventilators, infusion pumps, defibrillators) feed structured data into the pipeline:

```
Medical Device → Device Data Endpoint (serial/BLE/Wi-Fi)
→ HL7 FHIR Resource Mapping → Record Compiler → Clinician Approval → EHR
```

- **Standard**: HL7 FHIR R4 for all structured clinical data exchange.
- **Device protocols**: IEEE 11073 (point-of-care devices), serial protocols (legacy monitors), BLE GATT profiles (wearables).
- **Mapping layer**: Translates device-native formats into FHIR Observation, Procedure, and DiagnosticReport resources.

## Technology Stack

### Open-Source Components

| Function | Options |
|----------|---------|
| Speech-to-Text | OpenAI Whisper (open-source, including Turbo variant), Vosk, Faster Whisper (CTranslate2-optimised) |
| OCR | Tesseract 5, PaddleOCR, EasyOCR |
| LLM (edge) | Mistral 7B (quantised), Phi-4 Mini (3.8B), Llama 3.1 8B (quantised) |
| LLM inference runtime | llama.cpp, Ollama, vLLM |
| FHIR server | HAPI FHIR (Java), IBM FHIR Server (LinuxForHealth, open-source) |
| Message queue | NATS, RabbitMQ, EMQX (for MQTT from devices) |
| Object storage (edge) | MinIO |
| Container orchestration (edge) | K3s, MicroK8s |

### Managed Service Mappings

| Function | Azure | AWS | GCP |
|----------|-------|-----|-----|
| STT | Azure AI Speech (medical model) | Amazon Transcribe Medical | Cloud Speech-to-Text (medical adaptation) |
| OCR | Azure AI Document Intelligence | Amazon Textract | Document AI |
| LLM | Azure OpenAI Service | Amazon Bedrock | Vertex AI |
| Edge runtime | Azure IoT Edge | AWS IoT Greengrass | Google Distributed Cloud |
| Object storage | Azure Blob Storage | Amazon S3 | Cloud Storage |
| FHIR | Azure Health Data Services | AWS HealthLake | Cloud Healthcare API |
| Auth/IAM | Entra ID | Cognito + IAM | Cloud Identity |

## Safety and Regulatory Considerations

### Regulatory Frameworks

- **Australia (TGA)**: Clinical AI that influences treatment decisions may be classified as a Software as a Medical Device (SaMD) under the Therapeutic Goods Act 1989. Risk classification depends on whether the AI output is advisory (Class I/IIa) or autonomous (Class IIb/III). The clinician-in-the-loop pattern keeps most implementations at Class IIa. The TGA's guidance on classifying active medical devices (including software) provides the definitive classification rules.
- **United States (FDA)**: FDA SaMD and Clinical Decision Support guidance may apply. Clinical decision support software that lets a health care professional independently review the basis for a recommendation can fall outside the device definition under Section 520(o)(1)(E) of the FD&C Act, but classification is fact-specific and must be confirmed by regulatory counsel.
- **European Union (CE/MDR)**: Medical Device Regulation (EU) 2017/745 applies. AI documentation assistants with clinician override are typically classified as Class IIa under Rule 11 (software intended to provide information for diagnostic or therapeutic decisions).
- **United States (HIPAA)**: For systems handling US patient data, the HIPAA Security Rule (45 CFR Part 164, Subpart C) mandates administrative, physical, and technical safeguards for electronic protected health information (ePHI). Edge devices storing patient data must implement encryption, access controls, and audit logging per these requirements.

### Data Sovereignty

- Health data must remain within the relevant jurisdiction's boundaries (e.g., Australian health data under the Privacy Act 1988 and the My Health Records Act 2012, which governs the national electronic health record system).
- Edge-first architectures inherently support data sovereignty — data stays local until explicitly released to a cloud endpoint within the same jurisdiction.
- Encryption in transit (TLS 1.3 minimum) and at rest (AES-256) is mandatory.
- In the US context, HIPAA's data breach notification requirements (45 CFR Part 164, Subpart D) apply to any device storing or transmitting ePHI, including edge devices.

### Clinical Override

The architecture must guarantee that a clinician can always:
1. Override, edit, or reject any AI output.
2. Bypass the AI pipeline entirely and enter documentation manually.
3. Access all cached/queued documents regardless of connectivity state.
4. Delete erroneous AI outputs before they reach the EHR.

## Scale Guidance

### Small Deployments (single vehicle/facility, <10 clinicians)

- A single ruggedised mini-PC or medical-grade tablet with compute module suffices for the edge layer.
- Whisper Small is adequate for STT with acceptable accuracy on 16 GB devices; Whisper Medium or Whisper Turbo provide better accuracy on 32 GB devices (accounting for concurrent model loading and system overhead).
- Skip Kubernetes (K3s) — run containers directly with Docker Compose or systemd services.
- Use SQLite on-device for the audit cache; sync to cloud object storage on connectivity.
- Clinician approval UI can be a simple web interface served locally from the edge device.
- The full pipeline (STT + OCR + structuring + approval) can run on a single device with 16 GB RAM using Whisper Small + Phi-4 Mini, or 32 GB RAM using Whisper Medium + Mistral 7B.

### Large Deployments (fleet of vehicles/facilities, 50+ clinicians, multi-site)

- Centralised fleet management for edge device provisioning, model updates, and monitoring.
- K3s or MicroK8s for container orchestration across multiple edge nodes per facility.
- Dedicated GPU accelerators (NVIDIA Jetson Orin) for high-throughput transcription when multiple clinicians operate simultaneously.
- Central model registry with staged rollouts (canary → production) for model updates across the fleet.
- Multi-region cloud backend with geo-routing to nearest EHR integration endpoint.
- Centralised observability (OpenTelemetry) aggregating metrics from all edge devices for fleet-wide quality monitoring.
- Consider Whisper Large v3 (requires ~10 GB VRAM) or Whisper Turbo (~6 GB VRAM, near-large accuracy at ~8x speed) for highest accuracy in noisy clinical environments. Fine-tuned medical speech models may improve domain-specific terminology recognition.

## Decision Matrix

| Factor | Cloud-First | Hybrid | Edge-First |
|--------|-------------|--------|------------|
| Connectivity reliability | >90% uptime required | Intermittent acceptable | Not required |
| Latency tolerance | Seconds acceptable | Sub-second for approval UI | Real-time local |
| Data sensitivity | Standard encryption | Local staging before cloud | Data stays local until approved |
| Model quality | Best (full-size models) | Mixed (edge preprocess, cloud infer) | Constrained (quantised models) |
| Operational complexity | Low | Medium | High |
| Hardware cost | Low (phones/tablets) | Medium (edge server) | High (ruggedised compute) |
| Update frequency | Continuous (cloud deploy) | Periodic (edge + cloud) | Planned (edge firmware updates) |
| Regulatory burden | Standard cloud compliance | Split compliance scope | Simplified data residency |
| Connectivity examples | Urban clinics, telehealth | Rural with satellite windows | Airborne, maritime, disaster |

**Selection guidance**: Start with Option 1 (cloud-first) unless connectivity constraints force otherwise. Move to Option 2 when approval workflows must function offline. Move to Option 3 when the AI pipeline itself must function without any connectivity for extended periods.

## Engineering Readiness Pack

This architecture handles clinical records and possible patient-safety impact. It must be treated as a regulated clinical software design until legal and regulatory owners decide otherwise.

### Evidence and claim ledger

Maintain a `claim-ledger.md` with `claim`, `section`, `source`, `source type`, `last verified`, `confidence`, `owner`, and `recheck trigger`.

| Claim class | Current status | Required handling |
|---|---|---|
| Regulatory classification | Jurisdiction and intended-use dependent | Confirm with regulatory counsel for each deployment. |
| Model memory/performance figures | Engineering estimates | Validate on target hardware, quantisation, concurrency, and clinical audio/image samples. |
| Clinical accuracy | Not established by architecture | Run clinical validation for STT, OCR, structuring, and EHR mapping before production. |
| Retention periods | Jurisdiction-dependent | Replace with local medico-legal retention rules. |
| Device integration feasibility | Device-specific | Validate with each device vendor/protocol and hospital biomedical engineering team. |

### Implementation artefacts

Required before build:

- Intended-use statement and regulatory classification memo for each jurisdiction.
- Hazard analysis and safety case covering incorrect transcription, omitted medication, wrong patient, delayed sync, duplicate record, and failed EHR update.
- C4 context/container/deployment diagrams showing mobile app, edge runtime, AI services, approval UI, local cache, sync service, FHIR mapper, cloud backend, and EHR.
- FHIR implementation guide or profile map for Observation, MedicationAdministration, Procedure, DiagnosticReport, DocumentReference, Encounter, Patient, and Provenance.
- Offline-state model for captured, processed, pending_review, approved, queued_for_sync, synced, failed_sync, rejected, and amended.
- ADRs for cloud/hybrid/edge topology, model choices, audit storage, sync conflict strategy, and EHR write path.

### Anti-hallucination and clinical-safety controls

The system must never silently convert uncertain AI output into a clinical fact.

Controls:

- Field-level confidence and provenance for every generated clinical field: source audio/image/device event, timestamp, model, and clinician approval status.
- Explicit uncertainty markers for low-confidence STT/OCR spans and clinically significant terms such as medication, dose, allergy, procedure, observation, and patient identifiers.
- No autonomous EHR writes; all generated content requires clinician review and approval.
- Clinical terminology validation against approved medication, diagnosis, procedure, and observation vocabularies where available.
- Patient identity matching must be deterministic and human-confirmed when confidence is below threshold.
- Prompt-injection and OCR-instruction tests for photographed forms, labels, and screens.

### Threat model

| Threat | Control |
|---|---|
| Wrong-patient documentation | Patient identity verification, barcode/wristband workflow, and clinician confirmation. |
| Medication/dose transcription error | High-risk term highlighting, confidence thresholds, and manual confirmation before EHR write. |
| Offline data loss | Durable local queue, encrypted backup, sync acknowledgements, and replay-safe IDs. |
| Duplicate or conflicting EHR updates | Idempotency keys, FHIR Provenance, and clinician arbitration for conflicts. |
| Edge device compromise | Full-disk encryption, secure boot where possible, MDM, least-privilege service accounts, and remote wipe. |
| Model update regression | Staged rollout, clinical eval pack, rollback, and model/version provenance on every output. |

### Validation and acceptance gates

Minimum validation pack:

- Representative clinical audio from noisy, accented, urgent, and multi-speaker environments.
- Images of handwritten notes, medication labels, monitor screens, forms, and poor-lighting captures.
- Device feeds with dropped events, duplicate events, out-of-order timestamps, and unit-conversion edge cases.
- Clinical-field accuracy metrics for patient identity, medication, dose, route, time, observations, procedures, and free-text notes.
- Safety tests for low-confidence outputs, missing connectivity, failed sync, wrong patient, and rejected clinician approval.
- Promotion gate: no unreviewed EHR writes, no known high-risk medication/patient-ID failure above the clinical safety threshold, and regulatory/clinical-governance sign-off.

### Operational runbook

Runbooks must cover edge-device loss, local cache corruption, failed sync, EHR API outage, model rollback, urgent manual documentation fallback, privacy breach, audit export, and clinical incident investigation.

## References

- HL7 FHIR R4 Specification — https://hl7.org/fhir/R4/
- OpenAI Whisper — https://github.com/openai/whisper
- Faster Whisper (CTranslate2-optimised Whisper) — https://github.com/SYSTRAN/faster-whisper
- HAPI FHIR Server — https://hapifhir.io/
- TGA Software as a Medical Device guidance — https://www.tga.gov.au/how-we-regulate/manufacturing/manufacturing-medical-devices/software-based-medical-devices
- FDA Software as a Medical Device (SaMD) — https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd
- EU Medical Device Regulation 2017/745 — https://eur-lex.europa.eu/eli/reg/2017/745/oj
- IEEE 11073 Point-of-Care Medical Device Communication (Nomenclature) — https://standards.ieee.org/ieee/11073-10101/10343/
- AWS IoT Greengrass — https://docs.aws.amazon.com/greengrass/
- Azure IoT Edge — https://learn.microsoft.com/en-us/azure/iot-edge/
- Google Distributed Cloud Edge — https://cloud.google.com/distributed-cloud/edge/latest/docs
- PaddleOCR — https://github.com/PaddlePaddle/PaddleOCR
- EasyOCR — https://github.com/JaidedAI/EasyOCR
- Meta Llama 4 Scout (17Bx16E MoE) — https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
- Microsoft Phi-4 Mini — https://huggingface.co/microsoft/Phi-4-mini-instruct
- Vosk Offline Speech Recognition — https://alphacephei.com/vosk/
- K3s Lightweight Kubernetes — https://k3s.io/
- MinIO Object Storage — https://min.io/
- Ollama (local LLM inference) — https://ollama.com/
- IBM FHIR Server (LinuxForHealth) — https://github.com/LinuxForHealth/FHIR
- Australian Privacy Act 1988 — https://www.legislation.gov.au/C2004A03712/latest/text
- My Health Records Act 2012 — https://www.legislation.gov.au/C2012A00063/latest
- TGA Classifying Active Medical Devices (Including Software) — https://www.tga.gov.au/resources/guidance/classifying-active-medical-devices-australia-including-software-based-medical-devices
- FDA Clinical Decision Support Software Guidance — https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- HIPAA Security Rule (45 CFR Part 164) — https://www.hhs.gov/hipaa/for-professionals/security/index.html
- IMDRF SaMD Key Definitions — https://www.imdrf.org/documents/software-medical-device-samd-key-definitions
- NATS Messaging System — https://nats.io/
- NIST AI Risk Management Framework and Generative AI Profile — https://www.nist.gov/itl/ai-risk-management-framework
- OWASP Top 10 for LLM Applications 2025 — https://owasp.org/www-project-top-10-for-large-language-model-applications/
