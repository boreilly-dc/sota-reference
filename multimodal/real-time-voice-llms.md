# Building Real-Time Tool-Using Voice Agents

| Field | Value |
|-------|-------|
| Created | 2026-05-30 |
| Last Updated | 2026-08-14 |
| Version | 3.0 |

---

- [Executive Summary](#executive-summary)
- [Architecture Methods](#architecture-methods)
- [Real-Time Conversation Control](#real-time-conversation-control)
- [Tool-Use Protocol and Interface Contracts](#tool-use-protocol-and-interface-contracts)
- [Open and Source-Available Implementation Stacks](#open-and-source-available-implementation-stacks)
- [Managed Hyperscaler Options](#managed-hyperscaler-options)
- [Deployment Channels](#deployment-channels)
- [Security, Privacy, and Operations](#security-privacy-and-operations)
- [Evaluation](#evaluation)
- [Reference Build and Checklist](#reference-build-and-checklist)
- [Selection Guide](#selection-guide)
- [References](#references)

## Executive Summary

A tool-using real-time voice agent has three main architecture choices. The correct choice depends on the required control, conversational behaviour and reasoning depth. No one architecture is the universal default.

| Architecture | Use it when | Main benefit | Main cost |
|---|---|---|---|
| Cascaded streaming STT–LLM–TTS | The workflow needs durable transcripts, approval steps, deterministic tool gates or detailed audits | Separate, replaceable stages give high transcript visibility and policy control | Serial endpointing, model and synthesis stages add delay |
| Native audio-to-audio | Expressive delivery, acoustic understanding, early audio and conversational overlap are more important than complete intermediate-text control | The model works directly with live audio | Tools and business policy still need external control and observation |
| Native voice plus text planner | The conversation needs natural voice interaction and difficult reasoning or tool orchestration | The voice model can keep the interaction active while the planner handles complex work | The design needs a strict routing boundary and synchronised context |

For a cascaded design, keep streaming speech recognition, textual reasoning and tools, and streaming speech synthesis as separate stages. This structure is an engineering choice for support and approval-heavy workflows where control and traceability have priority over minimum latency. Measure the delay of each serial stage rather than one combined model time.

For a native audio-to-audio design, treat its use as an engineering judgement, not as a universal best choice. Select it when direct acoustic processing, expressive prosody and natural overlap have priority. Keep tool execution and business policy outside the voice model so that the application can validate and observe them.

For a hybrid design, let the real-time voice model manage listening, short responses, fillers and delivery. Route difficult reasoning, policy decisions and tool orchestration to a text planner. This can mask planner delay, but only if the system defines which model owns each decision and keeps both contexts consistent. Long-running tools can otherwise block the conversation.

For open orchestration, LiveKit Agents and Pipecat are strong default candidates in this comparison. This is an engineering judgement based on their documented scope, not a universal ranking. LiveKit documents open-source self-hosting and MCP support. Pipecat documents voice-agent orchestration across transports such as WebSockets and WebRTC. Assess both against the required transport, deployment and tool controls.

Do not merge Full-Duplex-Bench-v3, Audio2Tool and VoiceAgentBench into one leaderboard. They differ in audio source, language, model set, task definition, interaction mode and metric. Use each benchmark to test the behaviour that it covers, and state its limitations when reporting a result.

## Architecture Methods

### Cascaded streaming STT–LLM–TTS

Use a cascaded architecture when the application needs explicit text, deterministic control between stages and a durable audit trail.

```text
Live audio
  → streaming STT
  → transcript and end-of-turn decision
  → text LLM or planner
  → validated tool request and result
  → streaming TTS
  → audio playback
```

Keep STT, textual reasoning and tools, and TTS as separate, replaceable stages. This boundary makes transcripts, policy decisions and tool gates visible. It also permits deterministic logic between stages. The trade-off is serial delay from endpointing, first model output and first synthesised audio.

### Native audio-to-audio

Use a native audio-to-audio architecture when the main requirements are direct acoustic understanding, expressive prosody, early audio output and natural overlap.

```text
Live audio
  → native real-time voice model
  ↔ external policy and validated tools
  → streamed audio
  → audio playback
```

Calling this architecture the best default is an engineering judgement for that requirement set, not a universal fact. The design gives up some intermediate-text control. Keep business policy and tool execution outside the model, and observe those operations separately.

### Native voice plus text planner

Use a hybrid architecture when a fast conversational layer must work with a separate reasoning and policy layer.

```text
Live audio
  → real-time voice model
      ├─ listening, short replies, fillers and delivery
      └─ difficult request
            → text planner, policy and tool orchestration
            → result and context update
            → real-time voice model
  → audio playback
```

Define a strict routing boundary. The real-time model owns the immediate conversation. The text planner owns difficult reasoning, policy and tool orchestration. Synchronise user state, planner results and tool state across both layers. This pattern can hide some planner delay, but tools that take more than a few seconds can still block the conversation if the application does not handle them asynchronously.

### Architecture comparison

| Requirement with highest priority | Starting architecture | Required design check |
|---|---|---|
| Transcript visibility, approval and audit | Cascaded streaming | Account for each serial latency stage |
| Prosody, acoustic context and overlap | Native audio-to-audio | Externalise and observe policy and tools |
| Natural delivery plus difficult tool work | Native voice plus text planner | Define ownership and synchronise context |

These are implementation starting points. They are not universal rankings.

### Model components are not complete orchestration stacks

Do not treat a voice model or inference component as a direct replacement for an orchestration framework. Ultravox takes audio input and emits streaming text. Moshi is a speech-text foundation model and a full-duplex spoken-dialogue framework. A complete agent still needs the applicable transports, business-tool integration, lifecycle management and deployment control plane. Moshi's documented low-latency result is a product-specific report, not a universal system latency budget.

## Real-Time Conversation Control

### Detect the end of a turn

Voice activity detection (VAD) detects speech and silence. It does not reliably show that a person has completed a conversational turn. Combine VAD with semantic evidence and, when available, acoustic and prosodic evidence such as intonation, pitch and rhythm. Keep push-to-talk as an option for explicit workflows and noisy conditions.

Endpointing is a trade-off between response delay and premature cut-off. A silence-only policy can respond quickly, but it can close a turn during a thinking pause. Semantic or acoustic detection can wait when an utterance appears incomplete and commit sooner when completion is clear.

```text
Incoming audio
  → speech-presence signal from VAD
  → semantic and acoustic completion evidence
  → one of:
      continue the user turn
      confirm end of turn
      wait for more evidence
```

### Classify overlap before interruption

A short acknowledgement or incidental sound is not necessarily a request for the floor. Treat backchannels separately from genuine barge-ins. If the system falsely treats a backchannel as an interruption, it needs a recovery path that can continue the response.

### Handle barge-in at three separate layers

Do not use one ambiguous `cancel` action. Playback cancellation, generation cancellation and tool-operation cancellation have different effects.

| Layer | Action after a genuine barge-in | Meaning |
|---|---|---|
| Audio playback | Stop playback quickly | Prevent more generated audio from reaching the user |
| Model or TTS generation | Cancel or supersede speculative work | Prevent invalid text or audio from continuing; remove unheard assistant content from model history |
| Read-only tool operation | Cancel when supported, or discard its result | Discarding a result does not prove that the underlying work stopped |
| Mutating or irreversible tool operation | Apply explicit transaction and confirmation rules | Speech interruption alone must not be treated as operation cancellation |

Barge-in can stop playback or generation, but it cannot cancel a committed or dispatched action. Only the downstream system can confirm operation cancellation. Track whether an operation was requested, committed, dispatched, cancelled by the downstream system, or only ignored by the conversation.

### Make provisional state revision-aware

Self-correction needs incremental state that newer user evidence can invalidate before commit. Put a turn identifier and revision identifier on tentative transcripts, plans, tool arguments and TTS work.

```text
turn T, revision 1
  → tentative transcript
  → tentative plan
  → provisional tool arguments
  → optional generated text or audio

new user evidence
  → turn T, revision 2
  → invalidate uncommitted revision 1 work
  → continue from revision 2
```

Preemptive generation can start after a stable ASR-final segment but before end-of-turn confirmation. Preemptive TTS can start earlier audio work. Both are engineering options, not requirements. Keep preemptive output cancellable because renewed speech or transcript changes can invalidate it.

### Measure latency by stage

Measure user-perceived response latency from the actual end of user speech to the actual start of audible playback. Also record each component separately:

1. speech-end detection delay;
2. transcript finalisation;
3. planner or LLM time to first token;
4. tool latency;
5. TTS time to first audio;
6. network transit;
7. playback buffering.

This split shows whether delay comes from turn detection, generation, a tool, synthesis or delivery.

The following values are illustrative engineering starting targets for a non-tool conversational turn. They are inferred design targets, not provider guarantees or universal latency budgets.

| Measure | Illustrative starting target |
|---|---:|
| Speech end to audible response, p50 | approximately 0.7–1.2 seconds |
| Speech end to audible response, p95 | below about 1.8 seconds |
| Endpointing allocation | roughly 0.1–0.5 seconds |
| First model output allocation | roughly 0.2–0.5 seconds |
| First TTS audio allocation | roughly 0.1–0.3 seconds |
| Remaining allocation | network, buffering and playback |

Tune these targets with measurements from the deployed client, transport and models.

### Measure tool turns separately

A tool turn needs two latency measures:

- time from speech end to an audible acknowledgement or filler;
- time from speech end to a useful tool-backed result.

Acknowledge the request promptly instead of putting the full external-operation delay into conversational silence. A separate planner can handle complex tool work while the real-time voice layer maintains the interaction, but a slow synchronous tool can still block conversation.

## Tool-use protocol and interface contracts

A real-time tool loop must separate what the user might mean from what the system is authorised to do. While speech is active, the agent should keep a provisional intent object. It should create an immutable execution intent only after the end-of-turn decision, resolution of corrections, and any required confirmation. This separation is important because even the leading system in Full-Duplex-Bench-v3 failed more than 40% of its self-correction cases, and 86% of silent cases in that benchmark still caused tool calls. Audio2Tool also tests correction and distractor speech, and reports accuracy below 56% for its most difficult tiers. These benchmarks have limits: Full-Duplex-Bench-v3 does not test network failures, and Audio2Tool uses generated speech.

Read-only or reversible preparation can run from provisional intent. An externally visible or irreversible operation must not run until the intent is committed. Full-Duplex-Bench-v3 shows why: early execution in a correction case used stale parameters.

### Reference design: revision-aware intent contract

The following contract is a reference design. The field names are design choices, not a claim about a specific framework.

```json
{
  "conversation_id": "string",
  "turn_id": "string",
  "intent_id": "string",
  "intent_version": 4,
  "status": "provisional | awaiting_approval | committed | superseded",
  "tool_name": "string",
  "arguments": {},
  "argument_source": "provisional_transcript | final_transcript",
  "end_of_turn_decided": false,
  "corrections_resolved": false,
  "confirmation": {
    "required": true,
    "decision": "pending | approved | rejected"
  }
}
```

A new transcript revision increments `intent_version`. It supersedes uncommitted plans, tool arguments, and generated output from an older version. The commit operation copies the final arguments into an immutable execution intent. It does not mutate the provisional object.

Recommended state transitions are:

```text
speech active
  -> provisional intent
  -> revised provisional intent       when newer user evidence arrives
  -> awaiting_approval                when the action needs confirmation
  -> committed                        after end-of-turn, correction resolution, and approval
  -> dispatched
  -> succeeded | failed | cancelled | unknown
  -> compensated                      when an explicit compensating action succeeds

provisional intent | awaiting_approval
  -> superseded                       when a correction replaces the intent
  -> rejected                         when the user refuses approval
```

This is a reference-design state machine. The action ledger should preserve the operational states `proposed`, `awaiting_approval`, `committed`, `dispatched`, `succeeded`, `failed`, `cancelled`, `unknown`, and `compensated`. The agent must tell the user whether nothing happened, the action succeeded, or the outcome is unknown.

### Reference design: committed tool request

```json
{
  "conversation_id": "string",
  "turn_id": "string",
  "action_id": "string",
  "intent_version": 4,
  "tool_call_id": "string",
  "tool_name": "string",
  "arguments": {},
  "arguments_hash": "string",
  "subject": "authenticated-user-reference",
  "authorisation_scope": ["minimum-required-scope"],
  "confirmation_decision": "approved | not_required",
  "policy_result": "allow | deny",
  "idempotency_key": "non-PII committed-action key",
  "attempt": 1
}
```

For a state-changing request, derive the idempotency key from the committed action identity, not from personal information. A retry must use the same key and the same arguments. A changed intent requires a new action identity and a new key.

The downstream tool should authenticate and authorise the specific user with minimum scopes. Chained calls should preserve that user and those scopes. A high-impact action should require step-up authentication. Spoken confirmation does not replace these controls.

Before approval, the agent must render and speak the exact consequential action and its key parameters. This is an approval control. It is not authentication.

### Inline and asynchronous execution

Run a short read-only tool inline when the answer depends on its result. A tool that takes more than a few seconds can block the conversation, so run longer work asynchronously. Give an immediate acknowledgement, then sparse progress updates. Suppress duplicate requests and provide cancellation support where the downstream operation supports it. Require confirmation and an idempotency key before a state-changing operation.

A real-time conversational model can handle the active dialogue while a separate text supervisor handles complex reasoning and tool calls. The OpenAI Realtime API Agents Demo uses this pattern. This supports a hybrid design in which the voice path stays responsive and the tool planner has a separate control boundary.

### Recommended control: interruption and cancellation semantics

The protocol must use separate events and states for four different operations:

1. **Playback cancellation** stops audio that the user would hear. It does not cancel a tool.
2. **Generation cancellation** stops or supersedes speculative model or speech-generation work for an obsolete revision.
3. **Result suppression** discards a late tool result. The downstream work can continue even when the result is discarded.
4. **Downstream-operation cancellation** sends an explicit cancellation request to the executing system. It succeeds only when that system supports cancellation and confirms the outcome.

An interruption must not be treated as downstream-operation cancellation. LiveKit documentation states that interruption discards a tool result but does not cancel the work. For this reason, an irreversible operation needs explicit transaction and ledger semantics.

Recommended cancellation events are:

```json
{
  "event": "playback.cancel | generation.cancel | result.suppress | operation.cancel.requested | operation.cancelled | operation.cancel.failed",
  "conversation_id": "string",
  "turn_id": "string",
  "action_id": "string-or-null",
  "intent_version": 4,
  "reason": "barge_in | correction | user_request | policy",
  "target_id": "playback-or-generation-or-tool-call-id"
}
```

These event names and fields are part of the reference design. They make the four meanings visible in logs and prevent a barge-in event from falsely reporting that an external action stopped. Define sequence numbers, acknowledgements, maximum frame and queue sizes, backpressure behaviour, and session-resume tokens for the transport contract.

### Result boundary and retry rules

Treat every tool result, retrieved record, and MCP response as untrusted data. Apply a strict schema and field allowlist before the result enters model context. A result must not change policy, credentials, or action authority. Run a deterministic policy check before execution of any follow-on action.

Retry only transient failures, and use bounded backoff. After an ambiguous timeout that occurs after dispatch, first query operation status. If that is not possible, replay the same request with the same idempotency key and arguments. Never send a new unkeyed mutation.

## Open and source-available implementation stacks

Start with an open orchestration stack when you need control of media transport, model choice, tool policy, or deployment location. In the reviewed material, LiveKit Agents and Pipecat document the capabilities needed for general-purpose real-time voice-agent composition. Select between them by interface and deployment needs rather than by a universal ranking. LiveKit is a good fit when built-in MCP integration and self-hosting are priorities. Pipecat is a good fit when provider-neutral pipelines and a choice of WebRTC or WebSocket transports are priorities.

### General-purpose orchestration

**LiveKit Agents.** LiveKit states that the full stack can run on an organisation's own servers. Its reviewed MCP implementation supports stdio, Server-Sent Events (SSE), and streamable HTTP transports. This makes it suitable when an application must connect voice sessions to existing MCP servers. Confirm the required transport, authentication method, and server lifecycle in a deployment test.

**Pipecat.** Pipecat provides composable pipelines for a single agent or a multi-agent system. Its documented transports include WebRTC and WebSockets. Use it when the design must combine different speech, model, transport, and tool providers. The available evidence supports broad composition, but it does not establish that one pipeline is best for every workload.

### Compact and specialised stacks

**Hugging Face speech-to-speech.** This project supplies a modular local cascade of voice activity detection, speech recognition, an LLM, and speech synthesis. It also supplies a tested subset of a Realtime-compatible endpoint, and its packaged client can enable local Python tools. Treat it as a compact local pipeline or compatibility endpoint. Do not treat it as a complete distributed-agent control plane without adding transport, policy, scaling, and operations components.

**Home Assistant and Wyoming.** Home Assistant Assist defines a pipeline for wake-word detection, speech recognition, intent recognition, and speech synthesis. Wyoming provides a peer-to-peer TCP protocol that carries JSON Lines and PCM audio between voice-assistant components. This combination is suitable for local home-control satellites. It is specialised for that domain and is not a general business voice-agent framework.

**OpenVoiceOS.** OpenVoiceOS targets smart speakers and other voice-centred devices. Its persona design composes solver plug-ins and their configuration. Use it when skills, plug-ins, personas, or embedded-device support match the product. The reviewed official material did not establish native MCP support or the same LLM-first real-time tool interfaces documented for LiveKit and Pipecat.

### Restricted and higher-risk choices

**TEN Framework licence restriction.** The TEN Framework root licence restricts deployments that compete with Agora's offerings. Therefore, describe the complete root project as **source-available with a restricted root licence**, not as conventional OSI-approved open source. Review the root licence and the licences of the selected subcomponents before adoption.

**Vocode.** The reviewed repository asked for community maintainers, and the evidence pack records no repository push after November 2024. Treat Vocode as a legacy or maintenance-risk option for a new build. Before use, check current maintenance activity, dependency health, and ownership.

### Models are not orchestration platforms

Ultravox and Moshi can form part of a voice stack, but they do not replace an orchestration framework. Ultravox accepts audio and emits streaming text. Moshi is a speech-text foundation model and full-duplex spoken-dialogue framework; its project reports practical latency as low as 200 ms on an L4 GPU. A production system still needs transports, tool execution, policy controls, session lifecycle management, and deployment operations. Treat the latency figure as a project-specific result, not as a deployment guarantee.

### Conditional selection guide

The following guidance is engineering judgement based on the reviewed capabilities:

- Choose **LiveKit Agents** when self-hosting, real-time media infrastructure, and verified MCP transports are primary requirements.
- Choose **Pipecat** when provider-neutral pipeline composition and transport choice are primary requirements.
- Choose **Hugging Face speech-to-speech** for a compact local cascade or a limited Realtime-compatible interface.
- Choose **Home Assistant with Wyoming** for local home-control satellites.
- Choose **OpenVoiceOS** for skills-based smart-speaker or embedded-device products.
- Consider **TEN Framework** only after legal review accepts its restricted root licence.
- Treat **Vocode** as an existing-system or maintenance evaluation, not as a default for a new build.
- Add **Ultravox** or **Moshi** as model components only when the surrounding orchestration and operations design supplies the missing platform functions.

## Managed hyperscaler options

Use a managed path when cloud integration, managed identity, regional operations, or reduced model hosting work is more important than full stack control. Current managed tool-using voice options in the reviewed material include Amazon Nova 2 Sonic, Azure Voice Live or Azure OpenAI Realtime, and Gemini Live. IBM and Oracle provide cascaded managed alternatives. This is a capability map, not a universal ranking.

**Deployment-time check:** Model lifecycle, preview status, regional availability, quotas, connection limits, and supported transports can change. Recheck these facts in official service documentation before each deployment. The statuses below describe the checked 2026 material.

| Provider | Managed pattern | Tool path | Conditional fit |
|---|---|---|---|
| AWS | Native speech-to-speech with Nova 2 Sonic | Native tool events; AgentCore Gateway for MCP-compatible tools | Select when the system needs AWS-native media runtimes, tool integration, or documented SIP-provider patterns. |
| Azure | Voice Live or Azure OpenAI Realtime | Function calling | Select Voice Live for a managed voice-agent API; select Azure OpenAI Realtime when native SIP is required. |
| GCP | Gemini Live on Vertex AI | Function calling and RAG grounding | Select when a WebSocket bridge fits the client and telephony architecture. |
| IBM | watsonx Assistant with IBM Speech | Assistant REST/OpenAPI extensions | Select for Assistant channels and a cascaded speech-dialogue design. Treat watsonx Orchestrate MCP as a separate integration. |
| Oracle | Application-composed OCI Speech and OCI Generative AI pipeline | Function calling or remote MCP calling through the OCI Responses API | Select when the application can own the real-time pipeline and OCI private networking is required. |

### AWS

For a new AWS deployment, prefer **Amazon Nova 2 Sonic** to the first-generation Nova Sonic model. The checked documentation marks the first-generation model as legacy and gives an end-of-life date of 14 September 2026. The checked Nova 2 Sonic getting-started documentation also states an eight-minute connection limit. Recheck the lifecycle date and connection limit before deployment.

Nova 2 Sonic can emit tool interactions for external tools and APIs. AWS also documents client integration through LiveKit, telephony integration with SIP-based providers, WebRTC through Bedrock AgentCore Runtime, and conversion of APIs, Lambda functions, and services into MCP-compatible tools through AgentCore Gateway. This is a broad composition path, but the evidence does not prove that it is preferable for every workload.

### Azure

**Azure Voice Live** provides a managed voice-agent API with function calling. The checked documentation supports WebSocket sessions. It describes direct web and mobile access through WebRTC, with tool calls managed through the WebSocket signalling channel. That WebRTC path was in preview in the checked documentation and used global-standard deployments. Recheck preview, deployment type, and region support before use.

**Azure OpenAI Realtime** supports voice sessions that can call tools. Its documented session paths include WebSocket for trusted server connections, WebRTC for browser media, and SIP through a SIP trunk provider. Choose this Azure path when native SIP ingress is a firm requirement.

### GCP

**Gemini Live on Vertex AI** provides bidirectional, low-latency voice and video sessions through WebSockets. The checked material describes function calling, RAG grounding, and proactive audio, and it marks the service as generally available. No first-party Gemini Live WebRTC or SIP endpoint was confirmed in the reviewed material. Browser, mobile, and telephony deployments therefore need an application server or communications framework to bridge the native WebSocket session. Recheck service status, model availability, regions, and endpoint support at deployment time.

### IBM

**watsonx Assistant with IBM Speech** is a cascaded channel stack, not a native speech-to-speech model. Assistant manages dialogue and actions. IBM Speech to Text and Text to Speech handle phone audio. Assistant supports web chat, a mobile WebView, phone integration, and custom API clients. Its normal tool extension path imports an OpenAPI document for REST-based actions.

Keep **watsonx Orchestrate MCP** separate from **watsonx Assistant**. IBM documents MCP server imports for watsonx Orchestrate. The reviewed evidence does not establish that MCP is part of the Assistant phone session. Do not describe Orchestrate MCP support as native Assistant voice-session support.

### Oracle

Treat Oracle as an **application-composed pipeline**. OCI Speech provides live transcription. The application must connect it, a separately selected speech-output component, and an OCI-hosted LLM or agent for reasoning and tool execution. The OCI Responses API documents application-executed function calling and direct remote MCP calling. It also documents regional endpoints, VCN integration, and private endpoints. The application must own session flow, interruption handling, and the hand-off between speech, model, and tool stages. Do not present this as a single native real-time voice-agent endpoint. Recheck regional service support at deployment time.

## Deployment channels

A deployment channel determines how audio enters and leaves the agent. It does not determine where the model runs. A browser or phone can act only as a media client while the model, tools, and policy controls run in trusted infrastructure.

### Browser

Use WebRTC media tracks for browser audio when the selected service or orchestration framework supports them. Use a data channel or the provider's signalling path for control events. Keep long-lived credentials, tool execution, policy enforcement, and access to private services in a trusted side-band backend. As an engineering default, use short-lived client credentials when direct media access is necessary. For trusted server-to-server links, WebSockets are a suitable transport.
A practical flow is:

```text
browser microphone
  -> WebRTC media session
  -> voice model or media gateway
  -> validated tool request
  -> trusted policy and tool backend
  -> tool result
  -> voice model or orchestrator
  -> WebRTC audio response
```

Azure Voice Live is one provider-specific example: its checked WebRTC path sends browser or mobile media directly, while tool calls use the WebSocket signalling channel. This path was in preview in the checked documentation. Recheck preview status, region support, and deployment requirements before use.
### Mobile

Use the same trust split for a native mobile client or a WebView. Prefer WebRTC when available because it supplies a real-time media path for browser and mobile applications. Keep consequential tool calls and private-service access in the backend. A mobile application must not hold long-lived service credentials or execute privileged tools only because it owns the microphone. Define how the app handles audio-session interruption, backgrounding, network hand-off, reconnection, and session resumption.
IBM Assistant can be embedded as a WebView in a mobile application. Its usual action path uses REST/OpenAPI extensions. Keep this Assistant path separate from watsonx Orchestrate, where IBM documents MCP server imports. The evidence does not establish MCP support inside an Assistant phone or WebView session.
Gemini Live uses a native WebSocket session in the reviewed material. No first-party Gemini Live WebRTC endpoint was confirmed. A browser or mobile application therefore needs an application server or communications framework to bridge media and the Gemini Live WebSocket session.
### Telephony

A telephony deployment normally terminates SIP or PSTN media at a gateway and bridges the call to the agent's real-time media session. A media gateway is a suitable place to normalise codecs, call events, and disconnect behaviour before the agent processes the stream.
The reviewed managed paths support three patterns:

- **Azure:** Azure OpenAI Realtime accepts native SIP through a SIP trunk provider.
- **AWS:** Nova 2 Sonic documentation gives SIP-provider and media-framework integration patterns.
- **IBM:** watsonx Assistant supplies channel-specific phone integration with IBM Speech services.
- **GCP and Oracle:** the checked official material did not confirm a native telephony endpoint, so use a separate telephony bridge.

These patterns are conditional choices, not a provider ranking.
A practical bridge flow is:

```text
PSTN caller
  -> SIP trunk or telephony provider
  -> SIP/media gateway
  -> real-time agent session
  -> trusted tool backend
  -> agent audio
  -> gateway
  -> caller
```

### Local or private infrastructure

For a private deployment, keep the media gateway, orchestration stack, speech components, model endpoint, and tool executor inside the required trust boundary. This is engineering judgement. Use an open stack such as LiveKit Agents or Pipecat when it meets the transport requirements, and use local speech or model components where data policy requires them. Expose only short-lived session access to clients. Route consequential tools through a backend that validates arguments and user confirmation.

Oracle can support a private application-composed pattern through OCI regional endpoints, VCN integration, and private endpoints. OCI Speech, speech output, and the OCI-hosted model remain separate pipeline stages that the application connects. This is not evidence of a single native real-time voice endpoint.
### Channel selection rules

The following guidance is engineering judgement based on the reviewed channel capabilities:

1. Choose **WebRTC** for browser or mobile media when the provider or framework supports it.
2. Choose **WebSockets** for trusted server-to-server sessions and for services that expose WebSocket-only live APIs.
3. Choose a **SIP or media gateway** for telephony unless the selected managed endpoint accepts SIP directly.
4. Keep **tools, durable credentials, policy, and private-service access** in a trusted backend for every client channel.
5. Treat **lifecycle, preview, regional, quota, and endpoint facts as volatile**. Recheck them in official documentation during deployment.

## Security, privacy, and operations

A voice agent joins uncertain speech input to systems that can act. Its safety boundary must therefore control speaker authority, user identity, action authority, untrusted tool data, retries, and uncertain outcomes.

### Recommended control: authority before action

Ambient speech is an authority problem, not only a speech-recognition problem. Audio2Tool tests whether systems distinguish a primary user from a distractor intent, and it reports accuracy below 56% in its most difficult tiers. Because the benchmark uses generated speech, production testing must also cover the target acoustic conditions. A production system should use enrolled-speaker or interaction-channel attribution, wake-word or push-to-talk boundaries, and a no-action default when the command source is ambiguous. This paragraph is a recommended control.

Downstream tools should authenticate and authorise the specific user with minimum scopes. Each chained call should preserve the authenticated subject and its permitted scope. Require step-up authentication for high-impact actions.

Before a privileged, irreversible, or externally visible action, render and speak the exact action and key parameters. Ask for explicit approval. Confirmation is an approval control, not proof of identity.

### Recommended control: isolate untrusted results

Tool output can return to the model context and enable chained prompt-injection effects. Treat tool results, retrieved records, and MCP responses as untrusted data. Validate them against strict schemas and field allowlists. Do not let returned text change policy, credentials, or action authority. Apply a deterministic policy check before each execution.

A safe boundary has this reference-design flow:

```text
speech and channel attribution
  -> authenticated subject
  -> provisional intent
  -> end-of-turn and correction resolution
  -> exact action review
  -> explicit approval, if required
  -> step-up authentication, if required
  -> deterministic policy check
  -> committed action with minimum scope
  -> downstream tool
  -> schema and field-allowlist validation
  -> untrusted result data for model use
```

This flow is a reference design. When each control is enforced, this ordering helps prevent a tool response from increasing its own authority.

### Recommended control: action identity, retries, and uncertain outcomes

Every state-changing request should carry an idempotency key. Derive it from the committed action identity and do not include personal information. A retry must reuse the same key and arguments. If the user changes the intent, create a new action identity and key.

Retry only transient failures and use bounded backoff. If a request times out after dispatch, the operation can have succeeded even though the agent did not receive the response. Query the operation status first. If status lookup is not possible, replay the same request with the same idempotency key and arguments. Do not send a fresh unkeyed mutation.

Keep an action ledger with these states:

```text
proposed
  -> awaiting_approval
  -> committed
  -> dispatched
  -> succeeded | failed | cancelled | unknown
  -> compensated
```

`Compensated` is not the same as cancelled. It records that a later action addressed an earlier completed action. The agent must tell the user whether nothing happened, the action succeeded, or the outcome is unknown.

### Interruption safety

The following distinctions are recommended controls:

| Control | Effect | What it does not prove |
|---|---|---|
| Playback cancellation | Stops audio output to the user. | It does not stop generation or external work. |
| Generation cancellation | Stops or supersedes model or TTS work. | It does not stop a dispatched tool. |
| Result suppression | Prevents a late result from affecting the conversation. | It does not cancel the downstream operation. |
| Downstream-operation cancellation | Requests cancellation from the executing system and records its response. | A request alone does not prove cancellation. |

A speech interruption must not be reported as cancellation of an external action. The ledger must stay at `dispatched` or `unknown` until the downstream system gives an authoritative result.

### Reference design: operational event contract

A useful event schema should correlate the conversation, turn, and action. It should record audio boundaries, turn decisions, revisions, security decisions, tool execution, latency, model versions, and redaction state. This is an opinion-based reference design, not a standard schema.

```json
{
  "event_name": "voice.turn | voice.intent | voice.security | tool.execution | gen_ai.evaluation.result",
  "event_time": "timestamp",
  "conversation_id": "string",
  "turn_id": "string",
  "action_id": "string-or-null",
  "audio": {
    "start": "timestamp-or-null",
    "end": "timestamp-or-null"
  },
  "vad_decision": "speech | silence | end_of_turn | null",
  "transcript": {
    "state": "provisional | final | redacted | absent",
    "revision": 4
  },
  "intent_version": 4,
  "confirmation_decision": "pending | approved | rejected | not_required",
  "authentication_decision": "passed | failed | step_up_required",
  "policy_result": "allow | deny | not_evaluated",
  "tool_call_id": "string-or-null",
  "arguments_hash": "string-or-null",
  "idempotency_key": "string-or-null",
  "attempt": 1,
  "outcome": "proposed | awaiting_approval | committed | dispatched | succeeded | failed | cancelled | unknown | compensated",
  "error_class": "string-or-null",
  "latency_ms": 0,
  "model_version": "string",
  "prompt_version": "string",
  "redaction_state": "content_free | redacted | content_opt_in"
}
```

OpenTelemetry's generative-AI conventions specify `gen_ai.evaluation.result` as an evaluation event name and warn that input-message attributes can contain sensitive or personal information. Raw audio, transcripts, tool arguments, and complete model messages should therefore be opt-in telemetry. Give each a stated purpose, retention rule, access control, and redaction process. Default operational metrics should contain no conversation content or personal information.

### Operational outcomes

Use content-free events for normal operations. Record the action state, latency, error class, model and prompt version, and whether redaction ran. Keep raw content out of default metrics. When an operation has an ambiguous result, keep the ledger at `unknown`, prevent an automatic fresh mutation, and tell the user that the outcome is unknown.

## Evaluation

Voice-tool evaluation must separate recognition, tool use, conversation behaviour, safety, and operational reliability. A single aggregate score can hide a severe failure in any one of these areas. It can also hide the difference between a correct tool request and a completed user task.

### Read the benchmarks as separate diagnostic instruments

Do not combine Full-Duplex-Bench-v3, Audio2Tool, and VoiceAgentBench into one leaderboard. They use different audio sources, languages, model sets, task definitions, interaction modes, and metrics. A score from one benchmark is not a rank against a score from another benchmark.

Full-Duplex-Bench-v3 tests tool use during disfluent, full-duplex speech. It reported an overall Pass@1 of 0.600 for GPT-Realtime. A separate result reported Gemini Live 3.1 task-completion latency of 4.25 seconds. Pass@1 and task-completion latency have different units and denominators. They must remain separate. Neither result proves that the same system will be reliable or fast in a production network.

The benchmark gives useful evidence about eager commitment. In its self-correction scenarios, the leading result was 0.588 and still failed in more than 40% of those scenarios. In a different silence test, 86% of silent cases still caused tool calls. The first percentage uses self-correction scenarios as its denominator. The second uses silent cases. Neither percentage is an overall benchmark failure rate. The benchmark used 100 recordings, local zero-latency mock APIs, and one server region. It did not test access denial, API timeout, or malformed responses. Use it to test provisional state and commit policy, not to claim a production reliability ranking.

Audio2Tool is a diagnostic robustness suite. Its eight tiers include correction, long context, multi-turn state, and background intent. Reported accuracy fell below 56% in tiers 7–8. This value applies to those tiers and must not be presented as an overall accuracy value. The benchmark uses generated speech and static metrics. It does not measure interactive latency or production execution safety.

VoiceAgentBench tests agentic voice tasks, including dependent tool calls and refusal behaviour. The best reported ASR–LLM pipeline achieved 14.8% PF on sequential-dependent tool calling. In a separate adversarial-hint test, refusal rates fell to 35–40% across the tested models. PF and refusal rate measure different outcomes and use different test sets. The benchmark uses synthetic audio and excludes dynamic real-time tool invocation and background noise. Use it to expose dependent-tool and safety weaknesses, especially across the tested Indic languages. Do not use it to infer live interaction quality.

### Build a production test programme

Use the public benchmarks to find classes of weakness. Then run a workload-specific test suite against the complete deployed path. The suite must use the intended microphones, codecs, transports, regions, tools, permissions, and back-end services. It must include the following test groups.

1. **Tool decision:** Test correct tool selection and correct no-tool decisions. Include ambiguous requests and requests that need clarification.
2. **Argument fidelity:** Test names, dates, quantities, identifiers, locale formats, and required fields. Include mid-utterance corrections. Keep provisional values separate from confirmed values.
3. **Strict whole-task success:** Require every necessary tool step, valid arguments, correct dependency order, and a correct final result. Do not count a partly completed tool chain as success.
4. **Response delivery:** Verify that the agent tells the user the result, uncertainty, or failure after tool execution. A correct hidden tool response is not a completed voice task.
5. **Conversation control:** Test interruption, barge-in, silence, long pauses, self-correction, repeated speech, and background intent. Verify that cancelled or superseded speech cannot commit a stale action.
6. **Safety and trust boundaries:** Test unsafe requests, adversarial hints, malformed tool results, and injected instructions in tool output. Test read-only and consequential tools separately.
7. **Operational faults:** Test access denial, timeout, duplicate delivery, partial-chain failure, retry, and recovery after a dropped connection. Verify idempotency where the tool can change state.

Record each result with its own denominator. For example, report tool-selection accuracy over eligible decision points, argument fidelity over expected argument fields or calls, strict success over complete test episodes, and false activation over silence cases. State the denominator next to every percentage. Do not average unlike measures into one success rate.

Measure latency at distinct boundaries. Keep time-to-first-audio, tool-call latency, and end-to-end task latency separate. Report a distribution for each path and label whether a run includes endpointing, model processing, tool execution, synthesis, network transit, and retries. A low first-audio value can coexist with a slow or failed task.

Set release gates from the risk of the deployed workload. A read-only information agent can permit recovery after some tool errors. An agent that changes bookings, payments, identity data, or physical systems needs stricter confirmation, duplicate suppression, rollback, and whole-task success gates. Treat benchmark results as inputs to these local gates, not as substitutes for them.

## Reference build and checklist

This section is a reference design. It uses an open-source-first component flow and labels implementation choices as recommendations. It does not define one mandatory stack.

### Open-source-first component flow

For a general-purpose build, start with LiveKit Agents or Pipecat as the orchestration layer. LiveKit documents open-source self-hosting and native MCP support. Pipecat supports voice and multi-agent systems and provides low-latency WebSocket and WebRTC transports. In this comparison, these are the initial general-purpose open orchestration candidates.

For a compact local cascade, Hugging Face speech-to-speech provides a modular `VAD -> STT -> LLM -> TTS` pipeline, a tested core subset of a Realtime-compatible interface, and optional local Python tools. Treat it as a compact local cascade and endpoint, not as a complete distributed agent platform.

The recommended open-source-first flow is:

```text
browser, mobile, telephony, or local audio client
  -> streaming transport: WebRTC or WebSocket
  -> VAD and turn detector
  -> streaming STT
  -> revision-aware provisional intent
  -> text planner or LLM
  -> deterministic policy and confirmation gate
  -> function or MCP adapter
  -> downstream tool
  -> strict result-schema boundary
  -> response planner
  -> streaming TTS
  -> audio playback

Shared control plane:
  action ledger + event stream + redaction + model/prompt version records
```

This is a cascaded reference design. It keeps STT, textual reasoning and tools, and streaming TTS as separate replaceable stages. This structure gives transcript visibility, policy control, deterministic tool gating, and auditability. It also adds serial end-of-turn, model, and synthesis delays. A useful partial diagnostic subtotal is end-of-utterance delay plus LLM time to first token plus TTS time to first byte. End-to-end latency must also include applicable network transit, playback buffering, and tool delay.

A hybrid variant can replace STT and TTS with a native audio-to-audio model while keeping the text planner, policy gate, action ledger, and tools outside the voice model. This is a reference-design variant based on the architecture scope in the research brief; it is not a comparative performance claim.

### Turn and revision handling

VAD detects speech presence and silence. It does not by itself prove that the user has completed a conversational turn. Combine VAD with semantic evidence and, when available, acoustic and prosodic evidence. Use push-to-talk for explicit workflows or high-noise conditions.

Pre-emptive generation can start the LLM and optional TTS before the end of the user's turn is confirmed. Therefore, tentative transcripts, plans, tool arguments, and generated speech must carry a turn ID and revision ID. Newer user evidence must invalidate uncommitted work from an older revision.

A recommended revision state is:

```json
{
  "conversation_id": "string",
  "turn_id": "string",
  "revision_id": 7,
  "transcript_state": "provisional | final",
  "intent_state": "provisional | awaiting_approval | committed | superseded",
  "generation_id": "string-or-null",
  "playback_id": "string-or-null",
  "action_id": "string-or-null"
}
```

This is a reference-design contract. Keep provisional intent while speech is active. Create an immutable execution intent only after end-of-turn, correction resolution, and required confirmation. Read-only or reversible preparation can run speculatively. Externally visible or irreversible tools must not execute from provisional intent.

### Barge-in contract

On barge-in, stop playback quickly. Cancel or supersede speculative generation, and remove unheard assistant content from model history. For read-only tool work, request cancellation when the tool supports it; otherwise suppress the obsolete result. Do not treat the speech interruption as cancellation of an irreversible operation. Such an operation needs explicit transaction semantics.

Use separate event types:

```text
playback.cancel            stops audio playback
 generation.cancel          stops or supersedes model or TTS generation
 result.suppress            ignores a result that is no longer relevant
 operation.cancel.requested asks the downstream system to cancel work
 operation.cancelled        records confirmed downstream cancellation
 operation.cancel.failed    records that downstream cancellation failed
```

The first three events do not prove that downstream work stopped. Tool interruption can discard a result without cancelling the work. Mutating calls should not rely on conversational interruption semantics.

### Tool execution contract

Run a short read-only tool inline when the answer depends on it. Run longer work asynchronously so it does not block the conversation. Give an immediate acknowledgement and sparse progress updates. Suppress duplicate submissions. Support explicit downstream cancellation where available. Require confirmation and an idempotency key before a state-changing tool.

For each state-changing request:

```json
{
  "action_id": "committed-action-id",
  "intent_version": 7,
  "tool_call_id": "string",
  "tool_name": "string",
  "arguments": {},
  "arguments_hash": "string",
  "idempotency_key": "derived-from-action-id-without-PII",
  "attempt": 1,
  "policy_result": "allow | deny"
}
```

Retry with the same idempotency key and arguments. A changed intent requires a new key. Treat all results as untrusted data. Validate their schema and fields, and run a deterministic policy check before any follow-on action.

### Event fields

The recommended event contract correlates conversation, turn, and action records. It includes:

- `conversation_id`, `turn_id`, and `action_id`;
- audio start and end;
- VAD and end-of-turn decisions;
- provisional or final transcript state and revision;
- intent version;
- confirmation and authentication decisions;
- policy result;
- tool call ID, arguments hash, idempotency key, and attempt;
- outcome and error class;
- latency;
- model and prompt version; and
- redaction state.

This field set is a reference design. Raw audio, transcripts, tool arguments, and full model messages should be opt-in telemetry because they can contain sensitive or personal information. Default operational metrics should avoid content and personal information.

### Build checklist

The following list is a recommended implementation and release checklist.

#### Architecture and media

- [ ] Select cascaded streaming, native audio-to-audio, or native voice with a separate text planner.
- [ ] For a cascaded build, keep VAD, STT, planner and tools, TTS, and transport replaceable.
- [ ] Select an open orchestration layer first: LiveKit Agents or Pipecat for a general-purpose system, or Hugging Face speech-to-speech for a compact local cascade.
- [ ] Define the client path for browser, mobile, telephony, or local infrastructure.
- [ ] Define frame ordering, acknowledgements, backpressure, reconnection, and session resumption.
- [ ] Define fallback behaviour for STT, model, TTS, transport, overload, and session-expiry failures.
- [ ] Record end-of-utterance delay, LLM time to first token, TTS time to first byte, and end-to-end response latency.

#### Turns, revisions, and output

- [ ] Combine VAD with semantic and, where available, acoustic-prosodic end-of-turn evidence.
- [ ] Provide push-to-talk for explicit or high-noise workflows.
- [ ] Put turn and revision IDs on tentative transcripts, plans, tool arguments, and TTS work.
- [ ] Invalidate uncommitted work when a newer revision arrives.
- [ ] Keep provisional intent separate from immutable execution intent.
- [ ] Do not dispatch an externally visible or irreversible action from provisional intent.

#### Interruptions and cancellation

- [ ] Stop playback on barge-in and remove unheard assistant content from model history.
- [ ] Distinguish playback cancellation, generation cancellation, result suppression, and actual downstream-operation cancellation.
- [ ] Do not report a tool as cancelled until the downstream system confirms cancellation.
- [ ] Do not treat an interruption or discarded result as cancellation of a mutating operation.

#### Tools and policy

- [ ] Run short read-only tools inline only when the answer needs the result.
- [ ] Run long tools asynchronously with acknowledgement, sparse progress, duplicate suppression, and explicit cancellation support where available.
- [ ] Require confirmation and an idempotency key before each state-changing tool.
- [ ] Validate tool results, retrieved records, and MCP responses with strict schemas and field allowlists.
- [ ] Prevent tool data from changing policy, credentials, or action authority.
- [ ] Run a deterministic policy check before execution.
- [ ] Derive each mutation key from committed action identity without personal information.
- [ ] Reuse the same key and arguments for retries; use a new key after changed intent.

#### Observability and privacy

- [ ] Correlate conversation, turn, action, intent revision, generation, playback, and tool-call identifiers.
- [ ] Record turn decisions, confirmation, authentication, policy, outcome, error, latency, versions, and redaction state.
- [ ] Keep raw audio, transcripts, tool arguments, and complete model messages out of default telemetry.

#### Evaluation and failure tests

- [ ] Test tool selection and argument fidelity separately.
- [ ] Test strict whole-task success and response delivery.
- [ ] Test interruption, silence, self-correction, and background intent.
- [ ] Test unsafe requests and adversarial hints.
- [ ] Test malformed and injected tool results.
- [ ] Test access denial, timeout, duplicate delivery, and partial-chain recovery.

Full-Duplex-Bench-v3 covers correction and silence but does not test timeouts, access denials, or malformed responses. Audio2Tool covers correction and distractor intent but uses generated speech. VoiceAgentBench reports only 14.8% process fulfilment for its best ASR–LLM pipeline on sequential-dependent tool calling; adversarial hints reduce refusal rates to 35–40%. It does not study dynamic real-time invocation. These limits support a separate production test set for network faults, target acoustics, asynchronous calls, and live interruption.

## Selection guide

Choose the architecture from the workload constraints. Do not choose it from a model leaderboard alone. First classify the required policy control, audit evidence, interaction style, reasoning difficulty, deployment boundary, and client transport. Treat the guidance below as conditional engineering judgement.

### Choose an architecture family

| If the main requirement is… | Prefer… | Reason and condition |
|---|---|---|
| Durable transcripts, approval steps, deterministic tool gates, or replaceable components | Cascaded streaming STT–LLM–TTS | Keep STT, textual reasoning and tools, and streaming TTS as separate stages. This gives strong transcript visibility, policy control, and auditability. Accept the added serial delays from endpointing, model work, and synthesis.  |
| Expressive prosody, direct acoustic understanding, low first-audio latency, or natural overlap | Native audio-to-audio | Let the voice model handle the live conversation, but keep business policy and tool execution external and observable. Choose this only when natural interaction is more important than complete intermediate-text control.  |
| Natural voice interaction plus difficult reasoning, policy, or multi-tool work | Native voice plus a text planner | Let the realtime model listen, deliver short responses, and manage conversational fillers. Let the text planner own difficult reasoning, policy, and tool orchestration. Define a strict ownership boundary and synchronise context between both models.  |

A hybrid design can hide part of the planner delay behind a short spoken acknowledgement. It can also create two inconsistent sources of state. Use it only if the application can define which component owns each tool decision, correction, cancellation, and final response. Long-running tools can otherwise block the conversation.

### Choose an open orchestration stack

For a general-purpose open stack, start the technical evaluation with LiveKit Agents and Pipecat. This is a shortlist for this comparison, not a universal ranking. Select LiveKit Agents when native MCP support, per-turn observability, and full self-hosting are primary requirements. Select Pipecat when a transport-flexible pipeline over WebRTC or WebSockets is the primary requirement. Validate both against the same workload and operational test suite before selection.

Use Hugging Face speech-to-speech when the requirement is a compact local cascade with replaceable VAD, STT, LLM, and TTS stages. It provides a tested core subset of a Realtime-compatible endpoint and can opt in to local Python tools. Do not treat it as a complete distributed agent platform or as full API equivalence.

Use Home Assistant with Wyoming for local home-control satellites. Its pipeline covers wake word detection, STT, intent recognition, and TTS, while Wyoming supplies a peer-to-peer JSONL and PCM protocol. Do not use this specialisation as evidence that it is a generic business voice-agent framework.

Evaluate TEN Framework only after a licence review. The root licence restricts deployments that compete with Agora offerings. Therefore, do not describe the complete root project as conventional OSI open source without qualification.

### Choose a managed hyperscaler path

| Platform condition | Conditional choice |
|---|---|
| The deployment is on AWS and needs native speech-to-speech | Use Nova 2 Sonic for new work. First-generation Nova Sonic is marked legacy and has an end-of-life date of 14 September 2026. Check whether the documented eight-minute connection limit fits the session design.  |
| The deployment is on Azure and needs a managed voice-agent API | Evaluate Voice Live for WebSocket sessions and function calling. Its checked WebRTC path supported direct browser and mobile interaction, but it was preview and used global-standard deployments. Do not make production commitments that assume preview behaviour will remain unchanged.  |
| The deployment is on GCP and needs a managed bidirectional multimodal service | Evaluate Gemini Live on Vertex AI. The checked service was generally available, used WebSocket sessions, and supported function calling. Confirm workload-specific region, data, and latency requirements during deployment design.  |
| The deployment is on IBM and a cascaded channel stack is acceptable | Compose watsonx Assistant with IBM Speech to Text and Text to Speech. Assistant owns dialogue and actions; the speech services handle phone audio. Do not classify this pattern as a native speech-to-speech model.  |
| The deployment is on Oracle and private OCI integration is important | Compose OCI Speech live transcription and a separately verified speech-output component with an OCI-hosted LLM or agent. Use the OCI Responses API for function or MCP calls where applicable. Its documented VCN integration and private endpoints can support a private deployment boundary.  |

### Apply final decision gates

Use the following gates before a selection becomes a production default:

1. **Policy gate:** If a tool can cause a consequential change, require external validation and explicit commit rules. Prefer a cascade when deterministic approval and complete text evidence are mandatory.
2. **Conversation gate:** If overlap, backchannels, and expressive delivery are core requirements, test a native path. Keep tools outside the acoustic model’s authority.
3. **Reasoning gate:** If tasks need difficult planning or dependent tool chains, test a hybrid path against a cascade. Select the hybrid only if its latency benefit is larger than its state-synchronisation cost.
4. **Deployment gate:** If local operation is mandatory, evaluate the compact Hugging Face cascade or the specialised Home Assistant/Wyoming path before a managed service. For general open orchestration, compare LiveKit Agents and Pipecat on the intended transport and hardware.
5. **Platform gate:** If an existing hyperscaler boundary is mandatory, select only the architecture that the platform actually supplies. AWS, Azure, and GCP offer reviewed native live paths; the reviewed IBM and Oracle patterns are composed cascades.
6. **Lifecycle and licence gate:** Reject a default that depends on a near-end-of-life model, an unsuitable preview feature, or unacceptable field-of-use terms.

Make the final choice only after the candidates pass the same end-to-end tests. A strong native model result does not remove the need for tool validation, observability, and policy controls. A transparent cascade does not remove the need to measure serial latency. A hybrid does not remove the need for one authoritative tool state.

## References

### Architecture, turn-taking, and transport

1. [OpenAI voice-agent architectures](https://developers.openai.com/api/docs/guides/voice-agents.md) — native live-audio and chained STT–LLM–TTS designs.
2. [OpenAI Realtime VAD](https://developers.openai.com/api/docs/guides/realtime-vad.md) — server and semantic VAD controls.
3. [OpenAI Realtime WebRTC](https://developers.openai.com/api/docs/guides/realtime-webrtc.md) and [WebSocket](https://developers.openai.com/api/docs/guides/realtime-websocket.md) — client and server transport patterns.
4. [OpenAI Realtime Agents demo](https://github.com/openai/openai-realtime-agents) — realtime voice model with a separate text supervisor.
5. [LiveKit turn handling](https://docs.livekit.io/agents/logic/turns.md), [turn detector](https://docs.livekit.io/agents/logic/turns/turn-detector.md), and [adaptive interruption handling](https://docs.livekit.io/agents/logic/turns/adaptive-interruption-handling.md) — endpointing, interruption, and backchannel controls.
6. [LiveKit asynchronous tools](https://docs.livekit.io/agents/logic/tools/async.md) and [function tools](https://docs.livekit.io/agents/logic/tools/definition.md) — progress, cancellation, and interruption semantics.
7. [LiveKit observability data](https://docs.livekit.io/deploy/observability/data.md) — per-turn and per-stage latency measures.
8. [Deepgram streaming latency](https://developers.deepgram.com/docs/measuring-streaming-latency.md) and [TTS latency](https://developers.deepgram.com/docs/text-to-speech-latency.md) — end-of-turn latency and TTS time to first byte.
9. [Beyond Turn-Based Interfaces](https://aclanthology.org/2024.emnlp-main.1192/) — synchronous full-duplex dialogue modelling.
10. [Prior Lessons of Incremental Dialogue and Robot Action Management](https://arxiv.org/abs/2501.00953) — revision-aware incremental interpretation and action.

### Open and local stacks

11. [LiveKit Agents](https://github.com/livekit/agents) — open realtime-agent orchestration, tools, MCP, WebRTC, and SIP.
12. [Pipecat](https://github.com/pipecat-ai/pipecat) and [Pipecat MCP client](https://docs.pipecat.ai/api-reference/server/utilities/mcp/mcp) — composable voice pipelines, transports, and MCP tools.
13. [Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech) — modular local VAD–STT–LLM–TTS pipeline.
14. [TEN Framework](https://github.com/TEN-framework/ten-framework) and [root licence](https://github.com/TEN-framework/ten-framework/blob/main/LICENSE) — graph-based realtime orchestration and deployment restrictions.
15. [Home Assistant Assist pipelines](https://developers.home-assistant.io/docs/voice/pipelines/) and [Wyoming protocol](https://github.com/rhasspy/wyoming) — local voice pipeline and peer-to-peer speech-component protocol.
16. [OpenVoiceOS Core](https://github.com/OpenVoiceOS/ovos-core) — skills, personas, and embedded voice-device platform.
17. [Ultravox](https://github.com/fixie-ai/ultravox) and [Moshi](https://github.com/kyutai-labs/moshi) — voice-model components rather than complete tool orchestration platforms.
18. [Vocode](https://github.com/vocodedev/vocode-core) — cascaded voice-agent framework with current maintenance caveats.

### Managed hyperscaler services

19. [Amazon Nova 2 Sonic getting started](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-getting-started.html), [code examples](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-code-examples.html), and [integrations](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-integrations.html) — streaming speech, tools, and channel integration.
20. [Amazon Bedrock AgentCore WebRTC](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-webrtc.html) and [AgentCore Gateway](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html) — managed media runtime and MCP-compatible tools.
21. [Azure Voice Live WebRTC](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live-webrtc) and [function calling](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-voice-live-function-calling) — managed voice sessions and tools.
22. [Azure OpenAI Realtime audio](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/realtime-audio) and [SIP](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/realtime-audio-sip) — WebSocket, WebRTC, and telephony paths.
23. [Gemini Live API](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/live-api), [capabilities](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/live-api/configure-gemini-capabilities), and [WebSocket guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/live-api/get-started-websocket) — bidirectional media and function calling.
24. [IBM watsonx Assistant deployment](https://cloud.ibm.com/docs/watson-assistant?topic=watson-assistant-deploy-assistant), [phone integration](https://cloud.ibm.com/docs/watson-assistant?topic=watson-assistant-deploy-phone), and [custom extensions](https://cloud.ibm.com/docs/watson-assistant?topic=watson-assistant-build-custom-extension) — managed cascaded voice channels and REST actions.
25. [IBM watsonx Orchestrate MCP servers](https://www.ibm.com/docs/en/watsonx/watson-orchestrate/base?topic=tools-mcp-servers) — MCP support in Orchestrate, separate from Assistant voice sessions.
26. [OCI Speech Live Transcribe](https://docs.oracle.com/en-us/iaas/Content/speech/using/using-live-transcribe.htm), [OCI Generative AI](https://docs.oracle.com/en-us/iaas/Content/generative-ai/use-llms.htm), and [OCI Responses API](https://docs.oracle.com/en-us/iaas/Content/generative-ai/responses-api.htm) — components for an application-composed Oracle pipeline.

### Safety, observability, and evaluation

27. [OWASP LLM01: Prompt Injection](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM01_PromptInjection.md), [LLM02: Sensitive Information Disclosure](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM02_SensitiveInformationDisclosure.md), and [LLM03: Excessive Agency](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM03_ExcessiveAgency.md) — tool mediation, least privilege, confirmation, and privacy guidance.
28. [Stripe idempotent requests](https://docs.stripe.com/api/idempotent_requests) — safe retries for state-changing API requests.
29. [OpenTelemetry GenAI agent spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md) and [GenAI events](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md) — agent and evaluation telemetry conventions.
30. [Full-Duplex-Bench-v3 paper](https://arxiv.org/abs/2604.04847) and [repository](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/main/v3) — disfluent realtime voice-tool tasks, correction, silence, and latency.
31. [Audio2Tool paper](https://arxiv.org/abs/2604.22821) and [repository](https://github.com/ramitpahwa/Audio2Tool) — speech tool-use tiers for correction, context, noise, and distractor intent.
32. [VoiceAgentBench paper](https://arxiv.org/abs/2510.07978) and [repository](https://github.com/ola-krutrim/VoiceAgentBench) — dependent tools, multilingual tasks, and refusal behaviour.
