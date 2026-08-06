# Real-Time Voice LLMs for Voice Assistants

| Field | Value |
|-------|-------|
| Created | 2026-05-30 |
| Last Updated | 2026-08-06 |
| Version | 2.0 |

---

- [Executive Summary](#executive-summary)
- [Terms and Architecture Taxonomy](#terms-and-architecture-taxonomy)
- [Open Models and Local Deployment](#open-models-and-local-deployment)
- [Managed Real-Time Voice Models](#managed-real-time-voice-models)
- [Voice-to-Voice Tool Calling on Phone Hardware](#voice-to-voice-tool-calling-on-phone-hardware)
- [Tool-Calling Research and Benchmarks](#tool-calling-research-and-benchmarks)
- [Latency, Turn-Taking, and Voice Quality](#latency-turn-taking-and-voice-quality)
- [Open-Source Orchestration](#open-source-orchestration)
- [Multilingual Support](#multilingual-support)
- [Security and Production Controls](#security-and-production-controls)
- [Decision Framework](#decision-framework)
- [Caveats and Limitations](#caveats-and-limitations)
- [References](#references)

## Executive Summary

Real-time voice systems now use two main designs. A **native speech-to-speech model** accepts audio and emits speech directly. A **cascaded voice agent** connects voice activity detection (VAD), speech recognition, a text LLM, and speech synthesis. Native models usually provide more natural turn-taking and preserve tone. Cascades provide better observability, simpler tool integration, and more control over each component.

The most important change since the May 2026 version of this survey is the arrival of explicit **speech-native action models**. **DuplexSLA** adds a structured action channel to a full-duplex speech model. It can emit tool calls while it continues to speak. However, its checkpoint and inference code were still unreleased on 6 August 2026. It is a research result, not a deployable product. **Full-Duplex-Bench-v3** and **Audio2Tool** also show that tool selection alone is not sufficient. Voice agents must pass correct arguments, handle a user who changes their mind, avoid acting on background speech, and confirm the result in speech.

For production systems, the strongest managed voice-to-voice models with tool calling include **OpenAI Realtime**, **Google Gemini 3.1 Flash Live**, **Amazon Nova 2 Sonic**, and models exposed through **Azure Voice Live**. These run in the cloud. A phone can be the client, but the model does not run on the phone.

A phone can run a fully offline voice agent with tool calling, but the verified practical design is a **cascade**:

```text
on-device VAD/ASR → small on-device tool-calling LLM → allowlisted app tool → on-device TTS
```

On Android, **Gemma 4 E2B or E4B with LiteRT-LM** provides the reasoning and structured tool-call component. Google reports that E2B can use less than 1.5 GB of memory on some devices. On Apple devices that support Apple Intelligence, the **Foundation Models framework** provides guided generation and tool calling. The app must add Apple Speech and speech synthesis around it. These are voice-to-voice systems at the application level, but they are not single native speech-to-speech models.

As of 6 August 2026, this review found **no downloadable, general-purpose native voice-to-voice model with both verified tool calling and verified real-time execution on production phone hardware**. MiniCPM-o provides strong open full-duplex speech, but the current 9B MiniCPM-o 4.5 needs at least 10–11 GB in its quantised builds and its documented local real-time targets are Macs and GPUs. Its documentation does not show a native structured tool channel. DuplexSLA has that channel, but it is a 7B research model without released deployment artefacts.

## Terms and Architecture Taxonomy

The terms in this field are often used inconsistently. This article uses the following definitions.

- **Speech-to-speech (S2S) or voice-to-voice**: audio input and spoken output. The internal model can still use text or hidden representations.
- **Half-duplex**: one side speaks at a time. Barge-in can stop the assistant, but simultaneous listening and speaking are not native model behaviours.
- **Full-duplex**: the system listens while it speaks. It can distinguish an interruption from a short acknowledgement such as “mm-hmm”.
- **Native audio**: audio is represented inside the model rather than first being finalised as an external transcript.
- **Tool calling**: the model selects an executable function and produces structured arguments. The host application executes the function.

### 1. Cascaded voice agent

```text
VAD / turn detector → ASR → text LLM + tools → TTS
```

Each stage is separate. This remains the safest default for enterprise and on-device systems.

- **Strengths**: transcripts and tool calls are inspectable; each component is replaceable; small mobile models are available; policy checks can run before execution.
- **Weaknesses**: ASR errors propagate; text loses some prosody; sequential processing adds latency; turn handling needs external logic.
- **Best fit**: regulated work, phone deployment, offline use, complex tool policy, and systems that need traceable decisions.

### 2. Audio-input LLM with speech output

The model accepts audio directly, reasons through an LLM, and emits text or speech. A separate speech decoder can be attached to the model. Qwen Omni, Ultravox, and some managed live models fit this broad class.

- **Strengths**: fewer ASR boundary errors; access to prosody; tool calls can use a text reasoning layer.
- **Weaknesses**: speech generation and turn control can still be separate; the system is less observable than a cascade.

### 3. Native full-duplex speech model

A single backbone consumes user audio while it produces assistant audio. Moshi, PersonaPlex, MiniCPM-o 4.5, and DuplexSLA are examples of the research direction.

- **Strengths**: natural overlap, backchannels, interruption, and low model-level delay.
- **Weaknesses**: high continuous compute; difficult debugging; limited phone support; most open models do not expose a reliable structured action channel.

### Architecture comparison

| Dimension | Cascaded | Audio-input LLM | Native full-duplex |
|---|---|---|---|
| External transcript | Yes | Optional | Usually no |
| Prosody retained | Limited | Partial to strong | Strong |
| Tool-call integration | Mature | Good when a text channel exists | Emerging |
| Turn-taking | External VAD/semantic detector | Model plus external control | Native |
| Observability | High | Medium | Low |
| Component replacement | Full | Limited | Minimal |
| Phone viability | **Yes** | Limited | Not yet verified for a general tool-using model |

## Open Models and Local Deployment

The table separates released models from research-only systems. “Open” does not mean that a model runs on a phone.

| Model | Architecture | Tool calling | Full-duplex | Local hardware evidence | Licence / maturity |
|---|---|---:|---:|---|---|
| **DuplexSLA** | 7B native Speech–Language–Action model | **Native structured action channel** | Yes | Paper uses mainstream inference accelerators; no phone recipe | MIT repository; weights and code still “coming soon” |
| **MiniCPM-o 4.5** | 9B end-to-end omni model | No documented native tool channel | Yes | 19 GB BF16; 10–11 GB quantised; Mac/GPU demos | Apache 2.0; released |
| **MiniCPM-o 2.6** | 8B omni model | No documented native tool channel | Limited live streaming | Quantised edge builds exist; old “on your phone” title does not prove the complete tool-using S2S stack | Apache 2.0; legacy release |
| **Moshi** | 7.6B codec-based native speech model | No native tools | Yes | CUDA, MLX, and Rust paths; laptop/workstation class | CC-BY 4.0 weights; released |
| **PersonaPlex** | 7B Moshi-derived speech model | No native tools | Yes | Consumer GPU / workstation class | Open research release |
| **Ultravox v0.7** | Audio encoder plus text LLM | Yes | Framework-managed rather than native duplex | Server GPU; size depends on backbone | Open weights; released |
| **GLM-4-Voice** | End-to-end spoken dialogue | No verified general tool channel | No | GPU; int4 builds exist | Open weights |
| **Qwen3-Omni** | Thinker–Talker omni model | Model-dependent text tools | Streaming, not equivalent to native duplex in all releases | 30B-A3B class; workstation/server | Apache 2.0 |
| **Step-Audio 2 mini** | 7B speech model | Tool ability depends on integration | Streaming | Accelerator class | Apache 2.0; base for DuplexSLA |
| **Freeze-Omni / SALMONN-omni / Mini-Omni** | Research speech models | No mature general tool interface | Varies | GPU research deployment | Research releases |

### Important corrections to older guidance

- **Ultravox is not one fixed 355B model.** It is a family that combines an audio encoder with different text backbones. Its deployment cost depends on the selected checkpoint.
- **MiniCPM-o 4.5 is the current MiniCPM omni release**, not 2.6. It provides strong full-duplex capabilities but is not a phone-scale model: the project reports 10–11 GB for quantised variants and at least 16 GB RAM for some Mac half-duplex paths.
- **Moshi and PersonaPlex are conversational speech models, not complete voice agents.** An application needs a separate planner if it must call tools.
- **DuplexSLA is the clearest native tool-calling design**, but it cannot yet be treated as a deployable open model because the repository has not released its checkpoint, inference server, or benchmark data.

### Deployment classes

| Class | Practical options | Main limit |
|---|---|---|
| **Phone** | On-device ASR + 1–4B tool LLM + system or small TTS | RAM, thermal throttling, battery, app model size |
| **Laptop / Apple Silicon** | Hugging Face cascade; Moshi MLX; MiniCPM-o quantised on high-memory Macs | Sustained memory bandwidth and heat |
| **Consumer GPU** | MiniCPM-o, Moshi, PersonaPlex, Ultravox, local cascades | One concurrent full-duplex session can occupy most VRAM |
| **Data centre** | Batch and concurrent serving of open models | Cost, scaling, and audio-session state |
| **Cloud API** | OpenAI, Gemini, Nova Sonic, Azure Voice Live | Network, data governance, recurring cost |

## Managed Real-Time Voice Models

The following services provide audio input, spoken output, and tool use. Exact model names and preview status change quickly, so applications should pin model versions where the provider permits it.

| Hyperscaler | Model or service | Tool path | Notes |
|---|---|---|---|
| **AWS** | **Amazon Nova 2 Sonic** through Bedrock | Native tool-use events in the bidirectional stream | AWS provides an official speech-to-speech tool-use example. |
| **Azure** | **Voice Live API** and Azure OpenAI Realtime | Function calling and remote MCP servers | Voice Live combines model, speech, noise suppression, echo cancellation, and avatar options. |
| **GCP** | **Gemini 3.1 Flash Live Preview** | Function calling and Google Search through the Live API | Low-latency audio-to-audio model; preview status requires production review. |
| **IBM** | watsonx Assistant plus Speech to Text / Text to Speech | Assistant actions and orchestration | A managed cascade, not a frontier native audio-to-audio model. |
| **Oracle** | OCI Speech plus an OCI-hosted LLM | Application-defined orchestration | OCI Speech documentation is transcription-focused; no first-party native S2S tool model was found. |

OpenAI is not a hyperscaler, but its **Realtime API** is a key reference point. It supports streaming speech, function tools, remote MCP servers, server-side controls, and SIP. The service is cloud-hosted even when the client runs on a phone.

Independent results are more useful than vendor latency claims. Full-Duplex-Bench-v3 evaluated GPT-Realtime, Gemini Live 2.5 and 3.1, Grok, Ultravox v0.7, and a Whisper→GPT-4o→TTS cascade with the same LiveKit harness. On this benchmark, GPT-Realtime had the best Pass@1 at 0.600. Gemini Live 3.1 was second at 0.540 and had the fastest measured task-completion latency at 4.25 seconds, but it did not produce a spoken response in 22% of cases. These are full multi-tool task times, not time-to-first-audio measurements.

## Voice-to-Voice Tool Calling on Phone Hardware

### Three different claims

A “voice model on a phone” can mean three different systems. They must not be treated as equivalent.

1. **Native phone-resident model**: one local model accepts speech, produces speech, and emits structured tool calls.
2. **Phone-resident voice-agent cascade**: local ASR, a local tool-calling LLM, local app tools, and local TTS form one voice experience.
3. **Phone client for a cloud model**: the phone captures and plays audio, but a remote service performs inference and tool selection.

Only the first claim proves that a voice-to-voice LLM itself runs on phone hardware. The third is common in product marketing but says nothing about on-device inference.

### Capability matrix

| Candidate | Direct audio input | Spoken output | Tool channel | Full-duplex | Phone execution evidence | August 2026 assessment |
|---|---:|---:|---|---:|---|---|
| **DuplexSLA** | Yes | Yes | **Native, structured and time-aligned** | Yes | None; 7B accelerator-oriented design | Best research match, not deployable on a phone |
| **MiniCPM-o 4.5** | Yes | Yes | No documented native function-call stream | Yes | Current documented minimum is 10–11 GB quantised; local targets are Macs/GPUs | Native voice, but neither phone-ready nor a verified tool caller |
| **MiniCPM-o 2.6** | Yes | Yes | No verified native function-call stream | Partial/live streaming | Project used “on your phone”, but current evidence does not show the complete S2S + tools system on a production phone | Do not count as a verified phone tool-calling model |
| **Gemma 4 E2B/E4B + mobile speech components** | Audio through separate ASR | Through separate TTS | **Native LLM tool calling / structured output** | No; application-managed barge-in | Android and iOS support through LiteRT-LM; E2B below 1.5 GB on some devices | Best open phone reasoning component; complete system is a cascade |
| **Apple Foundation Models + Speech + AVSpeechSynthesizer** | Through Apple Speech | Through system TTS | **Foundation Models `Tool` API** | No; application-managed | Runs on Apple Intelligence-capable devices | Best first-party iPhone cascade |
| **Android ML Kit ASR + Gemma 4 + Android TTS** | Through on-device ASR | Through Android TTS | **Gemma/LiteRT-LM tool calling** | No; application-managed | Supported mobile components; exact performance varies by device | Best documented open Android design |
| **OpenAI / Gemini / Nova / Azure mobile client** | Streamed to cloud | Streamed from cloud | Cloud model tools | Model-dependent | Phone is only a client | Production option, not on-device inference |

### What works now

The practical offline phone design is a cascade. It can still present one continuous voice interface to the user.

```text
microphone
  → wake word or push-to-talk
  → on-device streaming ASR
  → schema-constrained tool LLM
  → policy and confirmation gate
  → allowlisted app function
  → short structured result
  → on-device TTS
```

#### Android

A current open Android stack can use:

- **ASR**: Android or ML Kit on-device speech recognition, or a small Whisper/Parakeet port where the application can accept the extra model size.
- **Reasoning and tools**: **Gemma 4 E2B or E4B** through **LiteRT-LM** or Android AICore. Google states that LiteRT-LM supports constrained decoding and tool calling. It reports less than 1.5 GB memory for E2B on some devices.
- **Speech output**: Android `TextToSpeech`, or a compact local TTS model if the device has sufficient memory.
- **Actions**: application functions, Android intents, local databases, or remote APIs behind a strict allowlist.

This design can operate offline for local tools. A network tool still requires connectivity, but the user’s speech and model inference do not have to leave the phone.

#### iPhone

On Apple Intelligence-capable devices, an application can use:

- **ASR**: Apple Speech APIs, with on-device recognition where the locale and device support it.
- **Reasoning and tools**: the **Foundation Models framework**. Apple documents guided Swift structure generation and a `Tool` protocol for local or online functions.
- **Speech output**: `AVSpeechSynthesizer` or another local TTS component.
- **Actions**: app-scoped Swift tools. The framework can call a tool to gather data or perform side effects.

The Foundation Models language model is text-oriented. Apple does not describe it as one native speech-to-speech model. The application owns endpointing, barge-in, speech playback cancellation, and tool confirmation.

### Why a single native phone model is still difficult

A full-duplex model performs continuous audio encoding and decoding even when little text is generated. It must also keep a conversational KV cache and a speech codec or decoder in memory. A 7–9B model can fit in heavily quantised form on a high-memory phone, but fitting is not enough. It must produce every audio chunk before the playback deadline without exhausting the thermal or battery budget.

Current evidence illustrates the gap:

- DuplexSLA uses a 7B backbone and a 160 ms clock. Its paper discusses mainstream inference accelerators, not phones.
- MiniCPM-o 4.5 uses 19 GB in BF16 and 10–11 GB in quantised variants. Its project recommends at least 16 GB RAM for Mac half-duplex speech and 24 GB for a full-duplex Mac path.
- Gemma 4 E2B is phone-sized, but it is a tool-calling text/audio-understanding component rather than a native speech generator.

### Recommended phone architecture

Use the following controls for a production phone agent:

1. **Use push-to-talk or a local wake word by default.** Continuous capture increases privacy and battery risk.
2. **Expose a small tool allowlist per screen or task.** Do not put every application function in every prompt.
3. **Use schema-constrained output.** Reject unknown tools, extra fields, and invalid enum values before execution.
4. **Separate read and write tools.** Execute read-only tools immediately. Require confirmation for messages, purchases, account changes, deletion, calls, and device-control actions.
5. **Do not commit on partial speech.** Keep tool arguments provisional until endpointing or explicit confirmation. This prevents the “Rome—actually Milan” failure.
6. **Cancel stale generations.** If the user resumes speech, cancel queued TTS and discard tool calls from the superseded turn.
7. **Return compact tool results.** A phone model should not ingest a large API response. Filter and summarise it before the next inference step.
8. **Set memory and thermal limits.** Unload optional TTS voices, shorten context, and fall back to system TTS before the operating system terminates the app.
9. **Offer an explicit cloud fallback.** Use it only with user consent and indicate when audio or text leaves the device.
10. **Keep an action audit record.** Record the final transcript, selected tool, validated arguments, confirmation, result, and spoken acknowledgement.

## Tool-Calling Research and Benchmarks

### DuplexSLA

DuplexSLA is the first model in this survey designed around a synchronised **speech, language, and action** interface. It starts from Step-Audio 2 mini at approximately 7B parameters. Every 160 ms chunk contains:

- causal user-audio features;
- assistant audio tokens;
- up to ten text tokens for delayed transcripts, planning, turn-control labels, or JSON-style tool calls.

The separate action channel lets the model call a tool without stopping its speech. It also emits `interrupt`, `backchannel`, and response control labels from the same internal state that generates speech.

On its own 900-case tool subset, the paper reports:

| System | Average accuracy | Average tool-call delay |
|---|---:|---:|
| ASR + LLM cascade | 91.33% | 2.77 s |
| DuplexSLA | 85.56% | **0.64 s** |

DuplexSLA is about four times faster on this measurement, but the cascade is 5.77 percentage points more accurate. These are author-reported results on a new benchmark with synthetic training and evaluation design choices. The checkpoint and inference server were not released at the time of this review.

### Full-Duplex-Bench-v3

Full-Duplex-Bench-v3 tests real human audio with fillers, pauses, hesitations, false starts, and self-corrections. Its 100 scenarios require chained API calls across travel, finance, housing, and e-commerce. It measures tool selection, argument accuracy, spoken response quality, Pass@1, interruption, turn taking, and latency.

| System | Pass@1 | Tool selection F1 | Argument accuracy | Task completion | Interruption rate |
|---|---:|---:|---:|---:|---:|
| GPT-Realtime | **0.600** | **0.876** | **0.680** | 6.89 s | **13.5%** |
| Gemini Live 3.1 | 0.540 | 0.817 | 0.588 | **4.25 s** | 19.2% |
| Gemini Live 2.5 | 0.490 | 0.786 | 0.593 | 7.26 s | 14.1% |
| Cascaded Whisper→GPT-4o→TTS | 0.450 | 0.803 | 0.562 | 10.12 s | 33.0% |
| Grok | 0.430 | 0.797 | 0.542 | 6.65 s | 25.5% |
| Ultravox v0.7 | 0.410 | 0.794 | 0.513 | 8.40 s | 47.9% |

The benchmark’s central result is not that one model wins. It is that **all systems fail often on self-correction**. GPT-Realtime led that category at 0.588, which still means failure in more than 40% of cases. Early tool execution reduces latency but can lock in an obsolete argument before the user finishes a correction.

### Audio2Tool

Audio2Tool contains approximately 30,000 queries and 152 functions across smart-car, smart-home, and wearable domains. Its eight tiers cover direct commands, parameters, multiple intents, implied intent, long irrelevant context, corrections, multi-turn dialogue, and competing speech from another speaker. It uses cloned voices and added automotive and indoor noise.

The paper evaluates open SpeechLMs and Whisper→LLM cascades. It reports that:

- simple direct commands often exceed 75% tool accuracy for the stronger speech models;
- exact match and argument F1 frequently fall below 35% on multi-intent and implicit tasks;
- long-form, corrective, and multi-turn tasks remain difficult;
- added noise causes substantial degradation;
- end-to-end SpeechLMs do **not** consistently outperform strong ASR→LLM cascades.

Audio2Tool uses synthetic speech, while Full-Duplex-Bench-v3 uses real human recordings. They are complementary rather than directly comparable.

### What to measure

A useful voice-agent evaluation must report these values separately:

1. **Tool selection**: did the model select every required tool and no extra tool?
2. **Argument accuracy**: did it preserve names, dates, identifiers, units, and corrected values?
3. **Commit timing**: did it call the tool before the user finished or confirmed?
4. **Tool latency**: when did the executable call become available?
5. **Task completion**: did all steps complete, including dependencies between calls?
6. **Spoken confirmation**: did the user hear the correct result?
7. **Turn behaviour**: did the assistant interrupt, ignore a backchannel, or stay silent?
8. **Robustness**: what happens with noise, accents, a second speaker, and a failed tool?

## Latency, Turn-Taking, and Voice Quality

### Do not combine different latency measurements

Voice articles often put unrelated values in one table. At least four latency definitions are in use:

- **Model chunk latency**: whether a model produces the next audio unit before its playback deadline.
- **Time to first audio**: the delay from the end of a user turn to the first assistant sound.
- **Tool-call latency**: the delay until a complete executable function call is available.
- **Task-completion latency**: the delay until tools finish and the assistant speaks the requested result.

A 200 ms model-level figure cannot be compared with the 4.25–10.12 second multi-tool task times in Full-Duplex-Bench-v3. Telephone tests also add network, media gateway, codec, jitter-buffer, and PSTN delay.

### Conversation timing

Human conversation has short gaps, overlaps, and acknowledgements. A useful system must support more than VAD:

- **Endpointing** decides when a user turn is complete.
- **Barge-in** stops assistant playback when the user takes the floor.
- **Backchannel detection** distinguishes “right” or “mm-hmm” from a new request.
- **Pause handling** avoids replying during a hesitation.
- **Self-correction handling** updates provisional state before an action is committed.

Native full-duplex models learn some of these behaviours inside the model. Cascades can use semantic endpointing such as Pipecat Smart Turn or LiveKit turn detection. External control remains easier to test and override.

### Voice quality

Speech naturalness, word accuracy, and conversational intelligence are separate qualities. MOS and arena preference results from different test sets are not directly comparable. For component-level TTS choices, see [`audio-generation-ai.md`](../media-generation/audio-generation-ai.md). For speech and general audio understanding, see [`local-audio-language-models.md`](local-audio-language-models.md).

Production evaluation should include:

- pronunciation of names, numbers, abbreviations, and Māori or other local terms;
- stability across long responses;
- interruption recovery;
- consistency between spoken output and the tool result;
- language switching and accent robustness;
- echo cancellation when the assistant listens while its own audio plays.

## Open-Source Orchestration

A model is only one part of a voice agent. These open projects provide the transport, queues, interruption logic, and tool loop.

| Framework | Architecture | Tool integration | Best use |
|---|---|---|---|
| **LiveKit Agents** | Cascaded and realtime-model adapters | Function tools and MCP integrations | WebRTC/SIP production agents |
| **Pipecat** | Frame-based voice pipeline | Application tools and MCP adapters | Flexible provider-neutral pipelines |
| **Hugging Face speech-to-speech** | Modular VAD→STT→LLM→TTS | Streams text and tool calls; OpenAI Realtime-compatible API | Fully open local server and experimentation |
| **Home Assistant Voice / Wyoming** | Local modular pipeline | Home Assistant intents and services | Private smart-home control |
| **OpenVoiceOS** | Modular assistant | Skills and service integrations | General local assistant |
| **TEN Framework** | Graph-based realtime pipeline | Extension-based functions | Visual composition and multimodal agents |

The Hugging Face project is a useful reference implementation. Its current pipeline uses Silero VAD, Parakeet or Whisper for STT, an OpenAI-compatible or local LLM, and Qwen3-TTS, Kokoro, or Pocket TTS. It can point the LLM slot at Gemma 4 through llama.cpp. This is a server/laptop framework, not a proven mobile runtime, but the same separation of concerns applies to phone applications.

### Tool-loop pattern

```text
1. Receive partial speech, but do not execute a side effect.
2. Finalise or confirm the user turn.
3. Ask the LLM for a schema-constrained tool call.
4. Validate the tool and arguments against local policy.
5. Ask for confirmation when the action is consequential.
6. Execute the tool with a timeout and an idempotency key.
7. Feed a compact typed result to the LLM.
8. Speak the result and record the action outcome.
```

For long-running tools, the assistant can speak a short status message. The status message must not claim success before the tool returns.

## Multilingual Support

Language counts from vendors often mix text understanding, speech recognition, and speech generation. A system supports a spoken language only when **input recognition, reasoning, and spoken output** all support it at acceptable quality.

- **Cloud**: Gemini Live and OpenAI Realtime provide the broadest practical multilingual coverage. Check the exact live-model documentation because supported input and output languages can differ.
- **Open native models**: MiniCPM-o 4.5 focuses on high-quality English and Chinese speech. Moshi and PersonaPlex are primarily English. GLM-4-Voice is strongest in Chinese and English.
- **Local cascades**: Whisper-family ASR plus a multilingual small LLM and a matching TTS voice gives the widest open language choice.
- **Phones**: system ASR/TTS coverage varies by operating-system version, downloaded language pack, locale, and device.

For New Zealand deployments, test New Zealand English, Māori names and place names, code-switching, dates, currency, and telephone-quality audio. Do not infer support from an English benchmark alone.

## Security and Production Controls

Voice tools can create physical or financial effects. The input channel also receives background conversations that were not intended as commands.

### Required controls

- **Consent and indication**: show when the microphone is active and when audio leaves the device.
- **Wake-word or interaction gating**: do not treat all ambient speech as a command.
- **Speaker or session binding**: use device unlock, app authentication, or speaker verification for sensitive actions. Voice alone is not strong authentication.
- **Tool allowlists**: expose only the functions needed for the current task.
- **Least privilege**: give each tool the minimum account and device permissions.
- **Confirmation**: require a clear confirmation for irreversible or consequential actions.
- **Argument display**: show the recipient, amount, date, destination, or device before execution.
- **Prompt-injection boundaries**: treat tool results, web pages, messages, and documents as untrusted data.
- **Timeouts and idempotency**: prevent duplicate actions when a user repeats a request or the network retries.
- **Auditability**: retain the approved request and actual tool outcome according to the organisation’s privacy policy.
- **Synthetic-voice disclosure**: make the assistant identity clear. Do not imply that a cloned voice is the real person.

### Full-duplex-specific risk

A full-duplex model can act while the user continues to speak. This creates a speed-versus-correctness problem. Use a two-stage state model:

```text
provisional intent → final/confirmed intent → executable action
```

Read-only prefetch can start from provisional state. A write action must wait for final state. If the user corrects an argument, invalidate all dependent provisional calls.

## Decision Framework

| Requirement | Recommended design | Reason |
|---|---|---|
| Fully offline phone agent with tools | On-device ASR + Gemma 4 or Apple Foundation Models + system TTS | Only practical verified phone design |
| Native full-duplex research with actions | DuplexSLA | Dedicated synchronised action channel, but no released weights yet |
| Open local natural conversation without tools | Moshi or PersonaPlex | Mature native duplex speech behaviour |
| Open local voice agent with tools | Hugging Face speech-to-speech, LiveKit, or Pipecat cascade | Inspectable tool loop and replaceable components |
| High-quality managed voice tools | OpenAI Realtime, Gemini Live, Nova 2 Sonic, or Azure Voice Live | Production APIs and integrated tools |
| Regulated or high-consequence actions | Cascaded system with confirmation and audit controls | Maximum observability and policy control |
| Broad multilingual support | Cloud live model, or Whisper + multilingual LLM + matching TTS | Native open models have narrower speech coverage |
| Smart home | Home Assistant Voice / Wyoming | Local services and constrained tool scope |
| Telephony | LiveKit or Pipecat with SIP plus a managed model or server cascade | Handles media transport, interruption, and call state |

### Recommended choices

- **Phone**: use a cascade. Treat native phone S2S tool calling as an open research target.
- **Private workstation**: use an open cascade when tools matter; use Moshi, PersonaPlex, or MiniCPM-o when natural duplex conversation matters more than actions.
- **Cloud production**: select between OpenAI, Gemini, Nova, and Azure using a task-specific test set. Include corrections, background speakers, tool failures, and long calls.
- **Research**: use DuplexSLA’s action-channel design as the current reference, but wait for released artefacts before making reproducibility claims.

## Caveats and Limitations

- **Snapshot date**: this article reflects sources checked on 6 August 2026. Preview model names and API prices can change.
- **No universal leaderboard**: VoiceBench, Full-Duplex-Bench, Audio2Tool, provider speech arenas, and telephone benchmarks test different properties. Their scores must not be merged.
- **Research artefacts**: DuplexSLA’s paper and repository are public, but its weights, inference code, and benchmark data were still pending.
- **Developer-reported results**: MiniCPM-o and DuplexSLA performance claims come largely from their developers. The article labels these results and does not treat them as independent replication.
- **Phone evidence**: model fit, a mobile client, and realtime on-device speech are different claims. This review requires evidence for the complete local path before calling a native model phone-ready.
- **Synthetic audio**: Audio2Tool uses generated voices and injected noise. Full-Duplex-Bench-v3 uses a much smaller set of real human recordings.
- **English bias**: many tool and duplex benchmarks are English-heavy. Chinese-focused models and low-resource languages need separate evaluation.
- **Tool safety is outside most benchmarks**: correct function selection does not prove authorisation, confirmation, rollback, or resistance to prompt injection.
- **Energy is under-reported**: few projects publish sustained phone battery, thermal, and throttling measurements for continuous voice sessions.

## References

### Native and open voice models

1. [DuplexSLA technical paper](https://arxiv.org/abs/2605.20755) and [repository](https://github.com/hyzhang24/DuplexSLA) — 7B full-duplex Speech–Language–Action model; synchronised action channel; release status.
2. [MiniCPM-o repository](https://github.com/OpenBMB/MiniCPM-o) and [MiniCPM-o 4.5 technical report](https://arxiv.org/abs/2604.27393) — full-duplex omni model, deployment requirements, quantised memory, and Apache 2.0 licence.
3. [Moshi paper](https://arxiv.org/abs/2410.00037) and [repository](https://github.com/kyutai-labs/moshi) — native full-duplex speech and Mimi codec.
4. [NVIDIA PersonaPlex](https://research.nvidia.com/labs/adlr/personaplex/) — Moshi-derived full-duplex voice and role control.
5. [Ultravox repository](https://github.com/fixie-ai/ultravox) — open audio-language family with tool use through text backbones.
6. [Qwen3-Omni repository](https://github.com/QwenLM/Qwen3-Omni) — open Thinker–Talker omni model.
7. [Step-Audio 2 repository](https://github.com/stepfun-ai/Step-Audio2) — open speech model and DuplexSLA base.
8. [GLM-4-Voice repository](https://github.com/THUDM/GLM-4-Voice) — end-to-end spoken dialogue model.

### Phone and edge deployment

9. [Google: Gemma 4 agentic skills on the edge](https://developers.googleblog.com/en/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/) — E2B/E4B, LiteRT-LM, constrained decoding, tool calling, memory, Android and iOS support.
10. [LiteRT-LM overview](https://ai.google.dev/edge/litert-lm/overview) — mobile and edge LLM runtime.
11. [Google on-device function-calling example](https://developers.googleblog.com/google-ai-edge-small-language-models-multimodality-rag-function-calling/) — voice input to local function execution.
12. [Android on-device inference](https://developer.android.com/blog/posts/build-intelligent-android-apps-on-device-inference) — ML Kit on-device speech and GenAI options.
13. [Apple Foundation Models framework](https://developer.apple.com/documentation/foundationmodels) — on-device models, guided generation, tools, and supported-device requirement.
14. [Apple Foundation Models `Tool`](https://developer.apple.com/documentation/foundationmodels/tool) — app-defined data and side-effect tools.
15. [Apple Speech framework](https://developer.apple.com/documentation/speech) and [AVSpeechSynthesizer](https://developer.apple.com/documentation/avfaudio/avspeechsynthesizer) — speech input and output components.

### Voice tool benchmarks

16. [Full-Duplex-Bench-v3 paper](https://arxiv.org/abs/2604.04847), [project page](https://daniellin94144.github.io/FDB-v3-demo/), and [code](https://github.com/DanielLin94144/Full-Duplex-Bench) — real disfluent audio, multi-step tools, accuracy, and latency.
17. [Audio2Tool paper](https://arxiv.org/abs/2604.22821) and [repository](https://github.com/RamitPahwa/Audio2Tool) — approximately 30,000 audio-to-tool queries and eight complexity tiers.
18. [VoiceAgentBench](https://arxiv.org/abs/2510.07978) — agentic voice tasks and multi-tool workflows.
19. [VoiceBench](https://arxiv.org/abs/2410.17196) — broad evaluation for LLM-based voice assistants.
20. [Full-Duplex-Bench original](https://arxiv.org/abs/2503.04721) — turn-taking and overlap evaluation.

### Managed voice and orchestration

21. [OpenAI Realtime guide](https://developers.openai.com/api/docs/guides/realtime) — realtime speech, function tools, MCP, SIP, and server controls.
22. [Gemini Live API](https://ai.google.dev/gemini-api/docs/live-api), [Gemini 3.1 Flash Live Preview](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-live-preview), and [Live API tools](https://ai.google.dev/gemini-api/docs/live-api/tools) — audio-to-audio and function calling.
23. [Amazon Nova 2 Sonic](https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-conversational-speech.html) and [tool-use example](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-code-examples.html) — Bedrock bidirectional speech and tools.
24. [Azure Voice Live function calling](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-voice-live-function-calling) and [MCP server integration](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-voice-live-mcp-server) — managed voice tools.
25. [IBM watsonx Assistant](https://cloud.ibm.com/docs/watson-assistant) and [IBM Text to Speech](https://www.ibm.com/products/text-to-speech) — managed assistant cascade.
26. [OCI Speech](https://docs.oracle.com/en-us/iaas/Content/speech/using/speech.htm) — Oracle transcription service.
27. [LiveKit Agents](https://github.com/livekit/agents) — open WebRTC/SIP voice-agent framework.
28. [Pipecat](https://github.com/pipecat-ai/pipecat) — open realtime voice and multimodal pipeline.
29. [Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech) — modular open voice-agent server with streaming tool calls.
30. [Home Assistant Wyoming protocol](https://www.home-assistant.io/integrations/wyoming/) — local modular voice services.
