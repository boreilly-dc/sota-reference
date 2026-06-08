# Real-Time Voice LLMs for Voice Assistants

| Field | Value |
|-------|-------|
| Created | 2026-05-30 |
| Last Updated | 2026-05-30 |
| Version | 1.0 |

---

- [Architecture Taxonomy](#architecture-taxonomy)
- [Cloud Speech-to-Speech APIs](#cloud-speech-to-speech-apis)
- [Open-Source Models for Local Deployment](#open-source-models-for-local-deployment)
- [Hardware Requirements and Deployment Tiers](#hardware-requirements-and-deployment-tiers)
- [Latency Benchmarks: Local vs Cloud](#latency-benchmarks-local-vs-cloud)
- [Accuracy and Quality](#accuracy-and-quality)
- [Expressiveness and Conversational Dynamics](#expressiveness-and-conversational-dynamics)
- [Orchestration Frameworks](#orchestration-frameworks)
- [Tool Use and Function Calling](#tool-use-and-function-calling)
- [Multilingual Support](#multilingual-support)
- [Decision Framework](#decision-framework)
- [Areas of Uncertainty](#areas-of-uncertainty)
- [References](#references)

## Architecture Taxonomy

Three distinct architectures exist for real-time voice AI in 2026, each with different tradeoffs between latency, flexibility, and observability.

### 1. Chained Pipeline (STT → LLM → TTS)

Three separate models in sequence: speech recognition, language model reasoning, and speech synthesis. Each stage can be independently swapped and evaluated.

- **Typical latency**: 600–2800ms end-to-end (400–800ms with streaming overlap)
- **Strengths**: Full observability (transcripts and traces at each stage), independent component evaluation, widest vendor choice
- **Weaknesses**: Latency compounds across stages; paralinguistic information (tone, emotion) is lost at the STT boundary
- **Best for**: Enterprise deployments requiring auditability, regulated industries, custom LLM logic

### 2. Half-Cascade Speech-to-Speech

Audio input is encoded directly, processed by a text-based LLM for reasoning, then synthesised back to speech. No intermediate text transcript is exposed to the user.

- **Typical latency**: 300–700ms steady-state (0.78–2.98s TTFT depending on provider)
- **Strengths**: Lower latency than cascaded; retains some prosodic cues; tool calling supported
- **Weaknesses**: TTS quality typically lower than specialist TTS models; less observable than cascaded
- **Examples**: OpenAI gpt-realtime-1.5, Google Gemini 3.1 Flash Live, xAI Grok Voice Agent, Ultravox

### 3. Native Audio End-to-End

A single model processes audio input and generates audio output directly, reasoning in audio/latent space without an explicit text stage.

- **Typical latency**: 160–300ms model-level
- **Strengths**: Lowest latency; maintains emotional tone; full-duplex conversation possible; natural interruption handling
- **Weaknesses**: Opaque reasoning (no text trace); harder to debug; limited flexibility for voice customisation
- **Examples**: Moshi (Kyutai), NVIDIA PersonaPlex, Step-Audio R1.1, Amazon Nova 2 Sonic, Kimi-Audio

### Architecture at a Glance

| Dimension | Chained Pipeline | Half-Cascade S2S | Native Audio |
|-----------|-----------------|------------------|--------------|
| End-to-end latency | 600–2800ms | 300–700ms | 160–300ms |
| Paralinguistic awareness | No | Partial | Full |
| Observability | High | Medium | Low |
| Component swappability | Full | Limited | None |
| Turn-taking / interruption | Requires VAD | Built-in | Native full-duplex |
| Enterprise readiness (2026) | Dominant | Growing | Emerging |

## Cloud Speech-to-Speech APIs

Production cloud APIs as of May 2026, measured by Artificial Analysis and independent benchmarks.

### Proprietary Platforms

| Provider | Model | Architecture | TTFT | Big Bench Audio | Pricing | Languages |
|----------|-------|-------------|------|----------------|---------|-----------|
| xAI | Grok Voice Agent | Half-Cascade | ~0.78s | ~93% | ~$0.05/min | 20+ |
| OpenAI | gpt-realtime-1.5 | Half-Cascade | ~0.82s | ~81% | ~$0.06–0.30/min | 50+ |
| Amazon | Nova 2 Sonic | Native Audio | ~1.14s | ~88% | ~$0.02/min | 7 |
| Google | Gemini 3.1 Flash Live | Unified S2S | ~2.98s | ~96% | ~$0.02/min | 90+ |
| Alibaba | Qwen3.5 Omni Flash Realtime | Thinker-Talker MoE | ~0.79s | — | API pricing | 119 written / 10 voice |
| StepFun | Step-Audio R1.1 | Native Audio | ~1.51s | 97% | Community hosted | Multi |
| Hume | EVI 3 | Unified S2S | — | — | Premium tier | Multi |

**Pricing notes**: OpenAI gpt-realtime-1.5 costs $32/1M audio input tokens + $64/1M audio output tokens (~2,200 tokens/minute of audio). Google and Amazon are approximately 7–12x cheaper per minute. OpenAI's realtime-mini tier (early 2026) reduced costs ~5x from original GPT-4o-realtime pricing.

### Real-World vs Vendor-Reported Latency

Vendor-reported latencies (300–500ms) represent optimal conditions. Real-world measurements via automated phone calls (voicebenchmark.ai, May 2026) show significantly higher end-to-end latencies:

| Platform | Current Latency | Median (24h) |
|----------|----------------|--------------|
| Dasha | 1,075ms | 1,079ms |
| Retell AI | 1,354ms | 1,403ms |
| LiveKit | 1,560ms | 1,925ms |
| OpenAI Realtime | 1,587ms | 1,414ms |
| ElevenLabs | 1,692ms | 1,999ms |
| VAPI | 2,647ms | 2,714ms |

The 2–5x gap between vendor claims and phone-call measurements comes from PSTN overhead, network conditions, full conversation context accumulation, and real-world load.

## Open-Source Models for Local Deployment

### Full Speech-to-Speech Models (End-to-End)

| Model | Params | VRAM (FP16) | Latency | Full-Duplex | License | Notes |
|-------|--------|-------------|---------|-------------|---------|-------|
| **Moshi** (Kyutai) | 7.6B | 16–20GB | ~200ms | Yes | CC-BY | Pioneer of full-duplex; PyTorch/MLX/Rust backends |
| **PersonaPlex** (NVIDIA) | 7B | 16–20GB | 170ms turn-taking | Yes | MIT | Zero-shot voice cloning; built on Moshi |
| **GLM-4-Voice** (Zhipu AI) | 9B | ~18GB (bf16) / ~9GB (int4) | Real-time streaming | No | Open | Chinese + English; CosyVoice decoder |
| **Ultravox** (Fixie.ai) | 355B (MoE) | H100/B200 class | <300ms | No | Open | Half-cascade; NOT consumer-hardware viable |
| **Step-Audio R1.1** (StepFun) | Large | Datacenter only | ~1.5s TTFT | — | Apache 2.0 | 97% Big Bench Audio; Dual-Brain Architecture |

### Text-to-Speech Models (for Cascaded Pipelines)

| Model | Params | Size | VRAM | RTF on Consumer GPU | Quality (MOS) | License |
|-------|--------|------|------|-------------------|---------------|---------|
| **Kokoro** (hexgrad) | 82M | 330MB | CPU real-time | 6.5x (RTX 4070 browser) | ~4.5 | Apache 2.0 |
| **Orpheus** (Canopy Labs) | 3B | 3.5GB | ~6GB (FP8) | Real-time (RTX 3090+) | ~4.6 | Apache 2.0 |
| **Sesame CSM** (Sesame AI) | 1B | ~2GB | 2–8GB | Faster-than-RT | ~4.7 | Apache 2.0 |
| **Dia 1.6B** (Nari Labs) | 1.6B | ~3GB | ~10GB (FP16) | 1.0–2.2x (RTX 4090) | High | Apache 2.0 |
| **F5-TTS** (SWivid) | ~1B | 1.6GB | GPU required | Sub-7s all lengths | High | MIT |
| **Piper** (Rhasspy) | 15–65M | <100MB | CPU only | 50x real-time (CPU) | Good (below LLM-TTS) | MIT |
| **Fish Speech 1.5** | ~500M–1B | — | GPU required | Real-time capable | High | CC-BY-NC-SA 4.0 |

### Speech Recognition (ASR) Models

| Model | Params | WER (English) | Latency | Hardware | License |
|-------|--------|---------------|---------|----------|---------|
| **Whisper large-v3** | 1.55B | ~2–3% (LibriSpeech) | 300–600ms | GPU recommended | MIT |
| **Whisper large-v3-turbo** | 809M | Near-identical to v3 | 5–6x faster | GPU / Apple Silicon | MIT |
| **Deepgram Nova-3** | Proprietary | 6.84% (streaming) | Sub-300ms | Cloud API | Commercial |
| **NVIDIA Parakeet TDT 0.6B** | 600M | SOTA (Open ASR Leaderboard) | 50x faster than Whisper v3 | GPU | CC-BY-4.0 |
| **Kimi-Audio** (Moonshot) | 7B+ | 1.28% (LibriSpeech, SOTA) | Streaming | A100+ class | Apache 2.0 |
| **Kyutai STT 2.6B** | 2.6B | Top Open ASR | 500ms streaming | GPU | CC-BY-4.0 |

## Hardware Requirements and Deployment Tiers

### Tier 1: High-End Consumer GPU (RTX 4090, 24GB VRAM)

- **Full S2S**: Moshi single-session at FP16 (16–20GB)
- **Cascaded pipeline**: Whisper small.en + Qwen 14B (Q4) + Orpheus/Kokoro → **1.5–3s end-to-end**
- **TTS options**: Orpheus (FP8), Dia (FP16), Sesame CSM — all real-time
- Can run both LLM and TTS on same GPU with careful memory management

### Tier 2: Mid-Range GPU (RTX 3060 12GB / RTX 4060 8GB)

- **Cascaded pipeline**: Whisper base.en + Qwen 3.5B (Q4) + Kokoro → **2–4s end-to-end**
- **TTS options**: Kokoro (trivial), Sesame CSM (6–8GB), Orpheus (FP8 tight fit at 12GB)
- Cannot run Moshi at full precision; quantised variants may work

### Tier 3: Apple Silicon (M3/M4 Pro, 36GB+ unified memory)

- **Cascaded pipeline**: Whisper Turbo + Llama 8B + Piper/Kokoro → **2–4s end-to-end**
- **Moshi**: MLX backend available; performance data limited
- Advantage: Large unified memory pool avoids VRAM constraints
- Bandwidth (400–614 GB/s) is the constraint, not capacity

### Tier 4: CPU-Only / Raspberry Pi

- **Cascaded (CPU-only PC)**: Whisper tiny.en + Qwen 3B + Piper → **5–10s end-to-end**
- **Raspberry Pi 5**: Whisper tiny.en (~6s per 10s audio) + Piper (50x RT) — simple command patterns only
- **Home Assistant Speech-to-Phrase**: Near-instant on Pi 4 for predefined phrases
- Full conversational AI is not viable on Pi-class hardware

### Tier 5: Datacenter (H100/B200)

- **Concurrent S2S**: Moshi 3–4 sessions per L40S (48GB), 8+ per H100 (80GB)
- **Orpheus production**: 16–25 concurrent real-time streams per H100
- **Ultravox**: Requires B200/H100 class for 355B MoE backbone
- **Cloud APIs**: All proprietary models run on datacenter-class hardware

## Latency Benchmarks: Local vs Cloud

| Deployment | Configuration | End-to-End Latency | Notes |
|-----------|--------------|-------------------|-------|
| Cloud S2S (optimal) | OpenAI/Grok/Gemini | 300–700ms | Vendor-reported, controlled conditions |
| Cloud S2S (real-world phone) | Same via PSTN | 1,000–2,700ms | Includes network + telephony overhead |
| Local Moshi (RTX 4090) | Native audio | ~200ms model-level | Single session, excludes I/O |
| Local cascaded (RTX 4090) | Whisper + Qwen 14B + Kokoro | 1,500–3,000ms | Full pipeline with streaming |
| Local cascaded (RTX 3060) | Whisper base + Qwen 4B + Kokoro | 2,000–4,000ms | |
| Local cascaded (M3 Pro) | Whisper Turbo + Llama 8B + Piper | 2,000–4,000ms | Unified memory advantage |
| Local cascaded (CPU only) | Whisper tiny + Qwen 3B + Piper | 5,000–10,000ms | Not conversational |
| Docker local-voice-ai (GPU) | LiveKit + Nemotron + Qwen + Kokoro | 500–1,500ms | Optimised stack with GPU |
| Human conversational response | — | ~200ms | Target benchmark |

**Key insight**: Local cascaded pipelines on consumer GPUs are 2–5x slower than cloud S2S APIs but provide full privacy. The primary bottlenecks are STT and LLM inference, not TTS. Moshi on RTX 4090 is the only local option approaching cloud-competitive latency.

## Accuracy and Quality

### ASR Accuracy (Word Error Rate)

| Model | LibriSpeech test-clean | Streaming WER | Notes |
|-------|----------------------|---------------|-------|
| Kimi-Audio | 1.28% | — | Current SOTA (May 2026) |
| NVIDIA Parakeet TDT | ~1.6% | — | #1 on Open ASR Leaderboard |
| Ultravox (GLM-4.6) | 2.28% | — | Full speech-to-speech model |
| Whisper large-v3 | ~2–3% | — | Batch processing |
| Deepgram Nova-3 | — | 6.84% | Production streaming |
| AssemblyAI Universal-2 | — | 6.88% | With intelligence features |
| Whisper V3 (avg across langs) | — | 7.4% | 99+ languages |

### TTS Quality (Mean Opinion Score)

MOS scores from different benchmarks are not directly comparable. Within the CodeSOTA leaderboard (April 2026):

| Model | MOS | Notes |
|-------|-----|-------|
| ElevenLabs Turbo v2.5 | ~4.8 | Commercial SOTA |
| Sesame CSM | ~4.7 | Open-source, conversational context |
| Orpheus 3B | ~4.6 | Open-source, emotion tags |
| Kokoro-82M | ~4.5 | Open-source, runs on CPU |
| Piper | ~3.5–4.0 | Lightweight, CPU-optimised |

Open-source TTS models are now in the same quality band as commercial APIs. The gap has effectively closed for English.

### Voice Quality Benchmarks

| Benchmark | What it Measures | Notable Results |
|-----------|-----------------|-----------------|
| Big Bench Audio | Audio reasoning + comprehension | Step-Audio R1.1: 97%, Gemini 3.1: 96% |
| VoiceBench | Multi-task speech model evaluation | Ultravox: 87.05/90.75 |
| Full-Duplex-Bench | Turn-taking, interruption, backchanneling | Moshi: best overall |
| VocalBench | 24k instances, 27 models, 4 dimensions | Comprehensive 2026 benchmark |
| TTS Spaces Arena | Community TTS quality ranking | Kokoro-82M: #1 |

## Expressiveness and Conversational Dynamics

### Emotion and Prosody Control

| Model | Approach | Capabilities |
|-------|---------|-------------|
| **Orpheus** | Explicit emotion tags | 8 tags: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>` |
| **Sesame CSM** | Implicit context-driven | Natural pauses, hesitations ("umms", "uhhs"), mouth sounds from conversational context |
| **Dia** | Audio conditioning | Laughter, coughing, throat clearing via transcript notation |
| **Hume EVI 3** | Multimodal emotional reasoning | 30 distinct emotions; detects sarcasm, adapts tone mid-speech |
| **PersonaPlex** | Native full-duplex | Contextual backchannels ("uh-huh", "yeah", "oh okay") without explicit programming |
| **Moshi** | Full-duplex architecture | Can laugh, sigh, whisper; listens and speaks simultaneously |
| **GPT-4o Voice** | Integrated | Can sing, laugh, cry (since May 2025); responds to user emotions |

### Conversational Dynamics

Full-duplex models (Moshi, PersonaPlex) handle turn-taking and interruption natively:

| Metric | Moshi | PersonaPlex | GPT-4o |
|--------|-------|-------------|--------|
| Turn-taking latency | 112ms | 170ms | — |
| Interruption response | 37ms | 240ms | 620ms |
| Full-duplex | Yes | Yes | Semi (barge-in) |
| Backchanneling | Native | Native | No |

For cascaded pipelines, turn-taking requires external handling:
- **Deepgram Flux**: ~260ms end-of-turn detection with EagerEndOfTurn events for speculative LLM generation
- **LiveKit Agents**: Semantic transformer model for turn detection (86% precision, 100% recall)
- **Krisp**: Dedicated model distinguishing backchannels from intentional interruptions

### Voice Cloning

| Model | Method | Data Required | Notes |
|-------|--------|---------------|-------|
| PersonaPlex | Zero-shot audio prompt | Single reference clip | No fine-tuning needed |
| Sesame CSM | Reference utterances as context | A few example segments | Context-based adaptation |
| Orpheus | Zero-shot + 6 built-in voices | Reference audio | GGUF format available |
| Hume EVI 3 | Natural language description | None (text prompt) | "Speak with a warm, confident tone" |
| Open-source LoRA | Fine-tuning | Minutes of audio | Single 16GB GPU sufficient |

## Orchestration Frameworks

### Open-Source Frameworks for Local Voice Assistants

| Framework | Focus | Key Feature | Transport | License |
|-----------|-------|-------------|-----------|---------|
| **Pipecat** | Voice agents | 40+ service integrations, linear pipeline | WebRTC, WebSocket | Open source |
| **LiveKit Agents** | Real-time comms | Semantic turn detection, SIP telephony, self-hostable server | WebRTC, SIP | Open source |
| **Home Assistant Voice** | Smart home | Wyoming protocol for modular voice components | Wyoming, local | Open source |
| **OpenVoiceOS (OVOS)** | General assistant | Mycroft successor, HiveMind distributed | Various | Open source |
| **local-voice-ai** | Quick start | Docker Compose one-click setup | WebRTC (LiveKit) | Open source |
| **TEN Framework** | Visual builder | Directed graph, drag-and-drop TMAN Designer | Multi-transport | Open source |

### Production Platforms (Cloud)

| Platform | Latency | Cost | Notes |
|----------|---------|------|-------|
| Retell AI | ~620ms E2E | $0.07/min | HIPAA included |
| Vapi | Sub-500ms avg | Varies | 300M+ calls processed |
| ElevenLabs Conversational | v3 quality | $0.10/min base | Premium voice quality |
| Deepgram Voice Agent | Sub-400ms | Bundled | Self-hosted option with Flux STT |

## Tool Use and Function Calling

A critical capability for voice assistants is the ability to call external tools (APIs, databases, calendars, search) mid-conversation. This distinguishes a voice *agent* from a voice *chatbot*.

### How Tool Calling Works in Voice-to-Voice Models

In speech-to-speech models, tool calling follows this pattern:
1. User speaks a request requiring external data
2. Model generates a structured tool-call request (JSON) internally
3. System executes the tool call against an external service
4. Tool result is fed back to the model
5. Model resumes generating spoken audio incorporating the result

The challenge is **latency**: tool execution adds round-trip time (typically 200–2000ms) during a live conversation. Advanced models use conversational preambles ("let me check that for you") to fill silence during tool execution.

### Tool Use Support by Platform

| Platform | Tool Calling | MCP Support | Parallel Calls | Notes |
|----------|-------------|-------------|----------------|-------|
| **OpenAI gpt-realtime-1.5/2** | Yes (GA) | Remote MCP servers | Yes | 128K context; preambles during execution; SIP integration |
| **Google Gemini 3.1 Flash Live** | Yes | Via open-source client | Yes | 90.8% on ComplexFuncBench Audio; Google Search built-in |
| **Amazon Nova 2 Sonic** | Yes | Via Bedrock | Yes | Native tool use on Bedrock runtime |
| **xAI Grok Voice Agent** | Yes | OpenAI-compatible schema | Yes | Follows OpenAI Realtime tool-call spec |
| **Moshi** (open-source) | No | No | — | Pure audio model; no text reasoning layer for tool calls |
| **Ultravox** | Yes (text output) | — | Yes | Outputs text for tool calls; pairs with downstream TTS |
| **ElevenLabs Conversational** | Yes | — | Yes | Function calling + RAG in bundled agent platform |

### MCP (Model Context Protocol) for Voice Agents

MCP is emerging as the standard protocol for connecting voice agents to external tools. Key integrations as of May 2026:

- **OpenAI Realtime API**: Supports remote MCP servers natively (GA since August 2025). Developers register MCP server URLs and the model can invoke tools during live audio sessions.
- **Azure Voice Live SDK**: Direct MCP server connection for real-time tool calling via the VoiceLive SDK.
- **LiveKit Agents**: Native MCP support — agents can discover and call MCP tools during voice sessions.
- **Pipecat**: MCP integration via pipeline services; tools execute as pipeline stages.

### Local Tool Calling

For fully local voice assistants, tool calling requires a text-based LLM in the pipeline (cascaded architecture). The LLM generates structured tool calls which are executed locally:

- **LiveKit local-voice-ai**: Supports tool definitions via the LLM (Qwen3-4B or larger)
- **Home Assistant**: Native Ollama integration handles structured tool calls for smart home control; Qwen 3.5 9B handles these reliably on 8GB GPU
- **Moshi limitation**: As a native audio model without a text reasoning layer, Moshi cannot natively call tools. It must be paired with a text LLM for agentic behaviour.

### Practical Considerations

- **Latency budget**: Tool calls add 200–2000ms. Models like GPT-Realtime-2 use filler speech ("let me look that up") to maintain conversational flow.
- **Parallel tool calls**: GPT-Realtime-2 and Gemini support multiple concurrent tool calls — critical for complex workflows (e.g., checking calendar AND looking up a contact simultaneously).
- **Reasoning effort**: GPT-Realtime-2 offers adjustable reasoning effort (`minimal` to `xhigh`). Lower effort = faster but less accurate tool-call decisions. Default is `low` for latency.
- **Failure recovery**: GPT-Realtime-2 specifically highlights "more natural recovery behaviour when something fails" — important for production voice agents where tool calls can timeout.
- **Open-source gap**: No open-source native-audio model currently supports tool calling. For local deployments needing tool use, a cascaded pipeline with a tool-calling LLM (Qwen, Llama) is required.

## Multilingual Support

| Model/API | Languages | Notes |
|-----------|-----------|-------|
| Gemini 3.1 Flash Live | 90+ (S2S) | Broadest multilingual S2S |
| Qwen3.5 Omni Flash | 119 written / 19 voice comprehension / 10 generation | Most language coverage overall |
| GPT-4o / Realtime | 50+ | Near-human fluency; GPT-Realtime-Translate for live translation |
| Whisper V3 | 99+ | Best open-source multilingual ASR |
| ElevenLabs v3 | 32+ | Commercial TTS leader |
| Sesame CSM | English only | Limited non-English from data contamination |
| Orpheus | English only | — |
| Kokoro | English (US + British) | — |
| Moshi | English primarily | Limited multi-language |
| Piper | 30+ | Wide coverage but lower quality per language |

For English-focused applications, all open-source models perform well. For multilingual, cloud APIs (Gemini, GPT-4o, Qwen) or Whisper + Piper local stacks are required.

## Decision Framework

### When to Use Cloud S2S APIs

- Sub-second response is a hard requirement
- Budget available ($0.02–0.30/min depending on provider)
- Multilingual support needed beyond English
- Production scale with SLAs required
- Full-duplex conversation dynamics needed without local GPU

### When to Use Local Deployment

- Privacy/data sovereignty requirements (healthcare, finance, government)
- No recurring API costs at scale (high-volume deployments)
- Offline operation required
- Custom model fine-tuning needed
- Acceptable latency: 1.5–4s on GPU, or 200ms with Moshi on RTX 4090

### Recommended Local Stacks by Use Case

| Use Case | Stack | Hardware | Latency |
|----------|-------|----------|---------|
| Best local conversational | Moshi (full-duplex) | RTX 4090 (24GB) | ~200ms |
| Balanced quality/latency | Whisper Turbo + Qwen 9B + Orpheus | RTX 4090 or M4 Pro | 1.5–3s |
| Budget GPU | Whisper base + Qwen 4B + Kokoro | RTX 3060 (12GB) | 2–4s |
| Apple Silicon | Whisper Turbo + Llama 8B + Kokoro/Piper | M3/M4 Pro 36GB+ | 2–4s |
| Smart home commands | Speech-to-Phrase / Piper | Raspberry Pi 4/5 | <1s (phrases only) |
| Docker quick-start | local-voice-ai | 12GB RAM + any GPU | 500–1,500ms |

## Areas of Uncertainty

- **Gemini 3.1 Flash Live latency**: Measured 2.98s TTFT by Artificial Analysis but marketed as "real-time" — may include cold-start or measurement artifacts. Other sources show competitive latency under different conditions.
- **Qwen3-Omni consumer deployment**: 30B parameters at Q4 could theoretically fit RTX 4090 (~17–20GB), but weight availability for local download is unconfirmed.
- **Step-Audio R1.1 local hardware**: Tops Big Bench Audio at 97% but no VRAM/GPU requirements published for self-hosting.
- **MOS score comparability**: Scores from different evaluation sets (CodeSOTA, TTS Arena, custom benchmarks) use different test sets and evaluators. Differences below 0.1 MOS are noise.
- **local-voice-ai 500ms claim**: The Docker project claims 500–1500ms on 12GB RAM "no GPU required" but independent benchmarks show CPU-only pipelines at 5–10s. The lower figure likely assumes GPU acceleration despite the "no GPU" wording.

## References

1. [Voice AI Models in 2026: LLM Comparison Guide](https://www.coval.ai/blog/voice-ai-models-2026) — Coval, May 2026
2. [Best Voice AI Models in May 2026](https://futureagi.com/blog/best-voice-ai-may-2026/) — FutureAGI, May 2026
3. [Real-Time vs Turn-Based Voice Agents in 2026](https://softcery.com/lab/ai-voice-agents-real-time-vs-turn-based-tts-stt-architecture) — Softcery, April 2026
4. [Building a Completely Local Voice AI Agent](https://themenonlab.blog/blog/local-voice-ai-complete-guide) — The Menon Lab, February 2026
5. [Pipeline vs Realtime Architecture Voice Bot Latency](https://versatik.net/en/news/pipeline-vs-realtime-architecture-voice-bot-latency) — Versatik, April 2026
6. [Real-Time TTS Streaming with Orpheus on RTX 3090](https://www.bitbasti.com/blog/audio-streaming-with-orpheus) — Bitbasti, April 2025
7. [Kokoro-82M — When smaller means better in TTS](https://unfoldai.com/kokoro-82m/) — UnfoldAI, January 2025
8. [Sesame CSM GitHub Repository](https://github.com/SesameAILabs/csm) — Sesame AI Labs
9. [Moshi: Full-Duplex Speech-to-Speech](https://kyutai.org/2024/09/18/moshi-release.html) — Kyutai Labs, September 2024
10. [NVIDIA PersonaPlex](https://research.nvidia.com/labs/adlr/personaplex/) — NVIDIA Research, January 2026
11. [Dia 1.6B Installation & Deployment](https://deepwiki.com/nari-labs/dia/4-installation-and-deployment) — Nari Labs
12. [Voice AI Leaderboard](https://voicebenchmark.ai/) — Dasha.ai, May 2026
13. [Full-Duplex-Bench](https://full-duplex-bench.github.io/) — arXiv:2503.04721
14. [Real-Time S2S AI on GPU Cloud](https://www.spheron.network/blog/speech-to-speech-gpu-cloud-moshi-sesame-csm-hertz-dev/) — Spheron, 2026
15. [Building a Local Voice Assistant: Latency Benchmarks](https://www.local-llm.net/guides/local-voice-assistant/) — Local-LLM.net
16. [Orpheus TTS GitHub Repository](https://github.com/canopyai/Orpheus-TTS) — Canopy Labs
17. [Pipecat Framework](https://github.com/pipecat-ai/pipecat) — Daily.co
18. [LiveKit Agents](https://github.com/livekit/agents) — LiveKit
19. [Home Assistant Wyoming Protocol](https://www.home-assistant.io/integrations/wyoming/) — Home Assistant
20. [Kokoro WebGPU Benchmarks](https://quick-tts.com/blog/kokoro-webgpu-benchmarks.html) — Quick TTS
21. [Whisper large-v3-turbo Release](https://github.com/openai/whisper/discussions/2363) — OpenAI
22. [VocalBench: Benchmarking Vocal Conversational Abilities](https://arxiv.org/abs/2505.15727) — arXiv, May 2025
23. [Kimi-Audio GitHub Repository](https://github.com/MoonshotAI/Kimi-Audio) — MoonshotAI
24. [GLM-4-Voice Architecture](https://deepwiki.com/zai-org/GLM-4-Voice/2.2-language-model-(9b)) — Zhipu AI
25. [OpenVoiceOS + Home Assistant Integration](https://blog.openvoiceos.org/posts/2025-09-17-ovos_ha_dream_team) — OVOS Blog
