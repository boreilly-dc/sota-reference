# Open-Source Audio Language Models for Local Audio Understanding and Reasoning

| Field | Value |
|-------|-------|
| Created | 2026-06-03 |
| Last Updated | 2026-06-03 |
| Version | 1.0 |

---

- [Executive Summary](#executive-summary)
- [Audio Understanding vs Voice Assistants (Scope)](#audio-understanding-vs-voice-assistants-scope)
- [Which Model Should I Use?](#which-model-should-i-use)
- [Model Landscape (June 2026)](#model-landscape-june-2026)
- [Benchmarks](#benchmarks)
- [Architecture Overview](#architecture-overview)
- [Hardware Requirements and VRAM Guide](#hardware-requirements-and-vram-guide)
- [Inference Frameworks](#inference-frameworks)
- [Quantisation](#quantisation)
- [Specialised Models: Music and Long-Audio](#specialised-models-music-and-long-audio)
- [Limitations and Reliability](#limitations-and-reliability)
- [Licensing](#licensing)
- [Managed Service Equivalents](#managed-service-equivalents)
- [Caveats and Limitations](#caveats-and-limitations)
- [References](#references)

## Executive Summary

Audio language models (audio LLMs, or *large audio-language models* — LALMs) accept audio as input and produce text (and, for "omni" models, speech) as output. Unlike a plain speech-recognition system that only transcribes words, an audio LLM is meant to *understand* what it hears — answering questions about a recording, describing environmental sounds, analysing music, reasoning about who spoke and how they felt, and following spoken instructions. They are the audio counterpart to the vision-language models covered in [`local-multimodal-vision-language-models.md`](local-multimodal-vision-language-models.md).

As of mid-2026, the open-source field is led by a handful of families: **Qwen** audio/omni models (Alibaba), **Audio Flamingo** (NVIDIA), **Kimi-Audio** (Moonshot AI), **Phi-4-multimodal** (Microsoft), **MiniCPM-o** (OpenBMB), **SALMONN** (Tsinghua/ByteDance), and **GAMA**/**DeSTA** from the academic community. A 7–8B open model now runs comfortably on a single 24 GB consumer GPU — and, quantised, on 8–12 GB cards — while delivering audio-understanding quality that, on several benchmarks, approaches the best proprietary APIs.

The frontier for raw audio understanding is still held by **Google Gemini** (native audio, up to ~8.4 hours of context) and **OpenAI GPT‑4o audio**, both proprietary and API-only. But the gap is narrow and, on specific axes, closed: **Audio Flamingo 3** (NVIDIA, 7B, fully open) reports state-of-the-art results across 20+ understanding/reasoning benchmarks, and 2026 entrants such as **Qwen3-Omni** push open models further. The important caveat is that many headline "beats Gemini/GPT‑4o" results are *developer-reported*; independent, peer-reviewed benchmarks (MMAU, MMAU-Pro, SAKURA, MMAR) tell a more sobering story — even the best models sit well below the human ceiling and collapse on temporal and multi-hop reasoning.

The single biggest licensing trap: **NVIDIA's Audio Flamingo model weights are non-commercial only** (the code is MIT, the checkpoints are not). Almost every other leading open model — Qwen, Kimi-Audio, Phi-4-multimodal, MiniCPM-o, Ultravox, IBM Granite-Speech, SALMONN, GAMA, Step-Audio — is permissively licensed (Apache 2.0 or MIT) and usable commercially.

## Audio Understanding vs Voice Assistants (Scope)

This article is about **audio understanding and reasoning**: taking an audio clip and producing an *analysis* of it. It deliberately does **not** re-cover real-time voice assistants — the speech-to-text → LLM → text-to-speech pipelines, full-duplex speech-to-speech models, latency budgets, and conversational turn-taking. Those are covered in the companion article [`real-time-voice-llms.md`](real-time-voice-llms.md).

The boundary is genuinely blurry in 2026, because "omni" models (Qwen2.5-Omni, Qwen3-Omni, MiniCPM-o, Step-Audio 2, Kimi-Audio) do *both*: they understand input audio **and** generate speech. Where such a model appears here, the focus is on its **understanding** capabilities (what it can tell you about a recording), not its speech-generation or low-latency-dialogue behaviour, which belong to the voice article. A useful mental split:

| Capability | This article | `real-time-voice-llms.md` |
|---|---|---|
| "What instruments are in this clip? Is the speaker angry? Transcribe and summarise this 30-minute meeting." | ✅ Audio understanding/reasoning | — |
| "Hold a low-latency spoken conversation; respond in a natural voice." | — | ✅ Voice assistant / S2S |
| Cascaded STT→LLM→TTS, TTS quality (MOS), barge-in, latency | — | ✅ |
| Audio encoders, audio-QA benchmarks (MMAU/MMAR), audio captioning | ✅ | — |

## Which Model Should I Use?

Quick decision matrix for **audio understanding** by hardware and use case (all open-source unless noted):

| Hardware | Best open model | Why |
|----------|-----------------|-----|
| 8 GB VRAM / edge | MiniCPM-o 2.6 (4-bit) or Qwen2.5-Omni-3B (Q4) | Both fit ~8 GB; MiniCPM-o.cpp runs on Jetson-class devices |
| 12–16 GB VRAM | Qwen2.5-Omni-7B (AWQ/GPTQ-Int4) or Phi-4-multimodal | Strong all-round understanding; Phi-4-mm is MIT-licensed and ~11 GB at FP16 |
| 24 GB VRAM | Audio Flamingo 3 7B *(non-commercial)* or Kimi-Audio 7B | AF3 is benchmark-leading for understanding; Kimi-Audio is a permissively-licensed all-rounder |
| 24 GB+, commercial use | Qwen2.5-Omni-7B / Qwen3-Omni or Kimi-Audio | Apache 2.0/MIT; avoids AF3's non-commercial weights |
| Apple Silicon (32 GB+ unified) | Qwen2.5-Omni-3B via LM Studio (MLX) | Audio-LLM MLX support is still maturing; the 3B omni model is the safe choice |
| Best quality, API OK | Google Gemini (native audio) — proprietary | Up to ~8.4 h context; strongest independent reasoning scores |

**Task-specific picks (open-source):**

1. **General audio understanding / reasoning** — Audio Flamingo 3 (non-commercial) for maximum quality; Qwen2.5-Omni-7B or Kimi-Audio for commercial use.
2. **Long audio (tens of minutes)** — Audio Flamingo 3 (up to 10 min natively) and the AF-Next research line (up to 30 min); otherwise chunk and aggregate.
3. **Music understanding** — NVIDIA's Music Flamingo (non-commercial) is purpose-built; Qwen and Kimi-Audio are solid general alternatives.
4. **Pure speech transcription/translation** — IBM Granite-Speech (Apache 2.0) or Whisper; these are ASR/AST tools, not general audio reasoners.

## Model Landscape (June 2026)

### Leading open-source audio LLMs

| Model | Developer | Params | Audio encoder → LLM | Input | Output | Licence |
|-------|-----------|--------|---------------------|-------|--------|---------|
| **Audio Flamingo 3** | NVIDIA | 7B | AF-Whisper → Qwen2.5-7B (MLP) | speech+sound+music | text (+voice) | Code MIT, **weights non-commercial** |
| **Qwen2.5-Omni** | Alibaba | 7B (8.5B) / 3B | Whisper-style → Qwen2.5 (Thinker-Talker) | text/img/audio/video | text + speech | Apache 2.0 |
| **Qwen3-Omni** | Alibaba | (omni) | omni encoder → Qwen3 | text/img/audio/video | text + speech | Apache 2.0 |
| **Qwen2-Audio** | Alibaba | 7B | Whisper-large-v3 → Qwen2 (MLP) | speech+sound+music | text | Apache 2.0 |
| **Kimi-Audio** | Moonshot AI | 7B | 12.5 Hz tokenizer + continuous feats → Qwen2.5-7B | speech+sound+music | text + speech | Apache 2.0 / MIT |
| **Phi-4-multimodal** | Microsoft | 5.6B | speech encoder (LoRA) → Phi-4-mini | speech+vision | text | MIT |
| **MiniCPM-o 2.6** | OpenBMB | 8B | Whisper-medium-300M → Qwen2.5-7B | vision+speech+audio | text + speech | Apache 2.0 |
| **SALMONN** | Tsinghua/ByteDance | 7B / 13B | Whisper + BEATs → Vicuna/Llama (window Q-Former) | speech+sound+music | text | Apache 2.0 |
| **GAMA** | U. Maryland | ~7B | multi-feature → Llama | sound+music+speech | text | Apache 2.0 |
| **DeSTA2 / DeSTA2.5-Audio** | NVIDIA/NTU | 8B | Whisper → Llama (descriptive alignment) | speech (+sound+music) | text | open (unconfirmed type) |
| **Ultravox** | Fixie AI | 1B / 8B / 70B | Whisper-medium → Llama (adapter) | speech | text | MIT |
| **IBM Granite-Speech** | IBM | 2B / 8B | two-pass ASR → Granite LLM | speech | text | Apache 2.0 |
| **Step-Audio 2 (mini)** | StepFun | 8B | — | speech (+audio) | speech + text | Apache 2.0 |
| **LTU / LTU-AS** | MIT | ~7B | AST/Whisper → LLaMA | sound (+speech) | text | open (research) |

Notes:

- **Audio Flamingo 3** (NVIDIA, July 2025, NeurIPS 2025 Spotlight) is the strongest *fully open* understanding model by its own and several third-party accounts, with on-demand chain-of-thought ("thinking"), multi-turn multi-audio chat, and long-audio reasoning to 10 minutes. Its **weights are non-commercial** — the most important caveat in this table for production users.
- **Qwen2.5-Omni** (March 2025, Apache 2.0) is the most practical general-purpose choice: 7B and 3B variants, official quantised builds, broad framework support. **Qwen3-Omni** (September 2025, Apache 2.0; see the Qwen3-Omni repository) is the confirmed successor. A further "Qwen3.5-Omni" (Plus 30B-A3B MoE / Flash / Light) was reported by a commercial host in early 2026 claiming SOTA on 215 audio/audio-visual subtasks; treat this as an unverified vendor claim pending an official release.
- **Kimi-Audio** (Moonshot AI, April 2025) is a permissively-licensed all-rounder trained on 13M+ hours, strong across ASR, audio understanding, and audio QA.
- **MiniCPM-o 2.6** (OpenBMB, January 2025) is the standout *edge* model: GPT‑4o-class multimodal in ~8 GB via 4-bit quantisation, with a dedicated C/C++ engine.
- **IBM Granite-Speech** and **Whisper** are **ASR/translation** systems, not general audio reasoners — Granite uses an explicit two-pass (transcribe-then-LLM) design. Included for completeness because they are often mistaken for audio LLMs.
- **Step-Audio 2** and **Ultravox** straddle the voice-assistant boundary; see [`real-time-voice-llms.md`](real-time-voice-llms.md) for their speech-generation/dialogue behaviour.

### Where open models stand vs the frontier

The best proprietary audio understanding is **Google Gemini** (built multimodal from the ground up; native audio up to ~8.4 hours / 1M tokens per prompt) and **OpenAI GPT‑4o audio**. On the peer-reviewed **MMAR** deep-reasoning benchmark, frontier models still lead (Gemini ahead of GPT‑4o audio), but open **Qwen2.5-Omni** and **Kimi-Audio** are within striking distance, and **Audio Flamingo 3** reports beating Gemini Pro 1.5/2.5 and GPT‑4o-audio on its benchmark suite (developer-reported). The honest summary: **open models have closed most of the gap on short-clip understanding, but everyone — open and closed — is weak on long-audio, temporal, and multi-hop reasoning** (see [Limitations](#limitations-and-reliability)).

## Benchmarks

Audio-understanding evaluation matured rapidly in 2024–2026. The benchmarks below are the ones that matter; most are open-source with public leaderboards.

| Benchmark | Venue | Size | What it measures |
|-----------|-------|------|------------------|
| **MMAU** | ICLR 2025 | 10k clips, 27 tasks | Multi-task audio understanding across speech/sound/music; MCQ accuracy. Human ceiling **82.2%** |
| **MMAU-Pro** | AAAI 2026 | 5,305 instances, 49 skills | Harder MMAU successor; adds spatial audio, multi-audio, long-form (≤10 min), open-ended QA |
| **MMAR** | NeurIPS 2025 | 1,000 items | *Deep* multi-step reasoning over speech/sound/music/mixed, from real-world video |
| **AIR-Bench** | ACL 2024 | 19k foundation + 2k chat | Foundation (single-choice) + chat (open-ended, GPT‑4-judged) across speech/sound/music |
| **AudioBench** | NAACL 2025 | 8 categories, 40+ datasets | Speech, audio-scene, and voice/paralinguistic understanding |
| **Dynamic-SUPERB Phase-2** | ICLR 2025 | 180 tasks | Largest speech/audio task suite; adds regression + sequence generation |
| **SAKURA** | Interspeech 2025 | 4 attributes | Single- vs multi-hop reasoning over speech/audio attributes |
| **TREA** | Interspeech 2025 | 600 Qs | Temporal reasoning: event ordering, duration, counting |
| **MuChoMusic** | ICML 2024 | 1,187 MCQ / 644 tracks | Music knowledge and reasoning |
| **Clotho-AQA** | 2022 | 1,991 clips / 35,838 QA | Environmental-sound audio QA (yes/no + single-word) |
| **HalluAudio** | ACL 2026 | 5k+ QA | Hallucination detection across speech/sound/music |

### Representative scores

**MMAU** (original version, music / sound / speech subset accuracy, from the Kimi-Audio report — note MMAU was revised in May 2025, so these are not directly comparable to current submissions):

| Model | Music | Sound | Speech |
|-------|-------|-------|--------|
| Kimi-Audio | 61.7 | 73.3 | 60.7 |
| Qwen2.5-Omni | 62.2 | 67.6 | 53.9 |
| Qwen2-Audio | 59.0 | 69.1 | 52.6 |
| Step-Audio | 49.4 | 53.8 | 47.8 |
| GLM-4-Voice | 38.9 | 43.5 | 32.4 |

For reference, the original MMAU paper reported **Gemini 1.5 at 66.2%** overall and **Qwen2-Audio at 55.4%**, against a **human expert ceiling of 82.2%**.

**MMAR** deep-reasoning leaderboard (community-aggregated, 2026 — treat as indicative, not authoritative):

| Model | MMAR accuracy |
|-------|---------------|
| Gemini 3.1 Pro *(proprietary)* | ~83.7% |
| Qwen3.5-Omni-Plus *(claimed)* | ~80% |
| Gemini 2.5 Pro *(proprietary)* | ~73.2% |
| Qwen2.5-Omni | ~64.1% |
| GPT‑4o Audio *(proprietary)* | ~63% |
| Kimi-Audio | ~53.5% |
| Qwen2-Audio-Instruct | ~46.5% |
| Audio Flamingo 2 | ~42.2% |

**MMAU-Pro** (peer-reviewed, AAAI 2026) is the most sobering datapoint: even **Gemini 2.5 Flash reaches only 59.2%** and **Audio Flamingo 3 only 51.7%**, with several skill categories near random. The headroom above today's models is large.

A new **Audio Reasoning Challenge** at Interspeech 2026 builds on MMAR with chain-of-thought annotations and a strict "both reasoning path and answer must be correct" criterion — a direct response to how unstable current LALM reasoning is.

## Architecture Overview

Almost every audio LLM follows the same recipe as a vision-language model: an **audio encoder** turns the waveform into a sequence of feature vectors, a **connector/adapter** projects those into the language model's embedding space, and a (usually frozen or LoRA-adapted) **LLM backbone** does the reasoning.

### Audio encoders

The encoder choice determines what the model can "hear" well:

- **Whisper encoder** (large-v2/v3) is the workhorse, used by SALMONN, Qwen-Audio/Qwen2-Audio, MiniCPM-o, Ultravox, and (in a modified form) Audio Flamingo 3. It is excellent for **speech** and produces dense, frame-level features.
- **BEATs** specialises in **non-speech sound events**; SALMONN pairs it with Whisper so the model hears both words and ambient sound.
- **CLAP** (contrastive language–audio pretraining) aligns audio with text in a shared space via contrastive learning. It excels at open-vocabulary **sound/music classification and retrieval**, but produces *global* clip-level embeddings rather than frame-level detail — used in Audio Flamingo 1 and 2.
- **AF-Whisper** (NVIDIA, Audio Flamingo 3) is a *unified* encoder: Whisper-large-v3 retrained with a captioning objective across speech, sound, and music, so a single encoder handles all three modalities with dense features — replacing the earlier Whisper+CLAP/BEATs split.

### Connectors

- **MLP / linear projector** (LLaVA-style) — the simplest and now most common; used by Qwen-Audio and Audio Flamingo 3. The audio frames are downsampled and linearly mapped to LLM tokens.
- **Window-level Q-Former** — SALMONN's approach: the long encoder output is split into temporal windows, and a BLIP-style Q-Former with learnable queries compresses each window into a few tokens. This keeps the LLM token budget manageable for long inputs.
- **Gated cross-attention** — the original Flamingo design (Audio Flamingo 1), injecting audio into LLM layers rather than as input tokens.

### Continuous features vs discrete tokens

A central design axis:

- **Continuous embeddings** (encoder features projected by an adapter) preserve fine acoustic detail and, per a controlled EMNLP 2025 study, **generally outperform discrete tokens for understanding** tasks. This is what understanding-focused models use.
- **Discrete audio tokens** (from RVQ codecs like EnCodec, SpeechTokenizer, or Kyutai's Mimi) can be predicted directly by an autoregressive LLM with no separate decoder, which makes them attractive for **speech generation**. SpeechTokenizer disentangles *semantic* content (first quantiser layer) from *acoustic* detail (later layers).

The practical pattern in 2026: **understanding uses continuous features; omni/generation models add a discrete-token path for the speech-output side.**

### Omni / any-to-any architectures

"Omni" models handle understanding and speech generation together:

- **Qwen2.5-Omni** uses a **Thinker-Talker** split: the *Thinker* is a standard LLM doing text reasoning/understanding; the *Talker* is a dual-track autoregressive head that consumes the Thinker's hidden states and emits audio tokens, decoded by a sliding-window diffusion transformer for streaming speech. **TMRoPE** position encoding time-aligns audio and video.
- **MiniCPM-o 2.6** combines SigLip (vision), Whisper-medium (audio), and a TTS head around a Qwen2.5-7B core for end-to-end streaming multimodal interaction.
- **Moshi** (Kyutai) and **Step-Audio 2** are full-duplex speech-to-speech models — relevant here only for their understanding side; their dialogue behaviour belongs to [`real-time-voice-llms.md`](real-time-voice-llms.md).

### Long audio

Quadratic attention makes long audio expensive. Strategies in use:

- **Fixed-window chunking** — Audio Flamingo 3 processes up to 10 minutes as 20 × 30-second windows; the AF-Next research line extends to 30 minutes.
- **Block-wise / streaming encoders** — Qwen2.5-Omni ingests audio in blocks for streaming.
- **Compression** — methods like CTC-guided compression reduce thousands of encoder frames to a few hundred tokens before the LLM sees them.

Even so, **performance degrades sharply with length** (see Limitations) — long-audio understanding is an open problem, not a solved one.

## Hardware Requirements and VRAM Guide

A general rule: **an audio LLM needs more VRAM than a text LLM of the same parameter count**, because the audio encoder (Whisper/AF-Whisper adds 300M–600M+ parameters) and mel-spectrogram buffers sit on top of the LLM, and the encoder is normally kept near FP16 even when the LLM is quantised.

| Model | Params | FP16 VRAM | 4-bit VRAM | Comfortable on |
|-------|--------|-----------|------------|----------------|
| Qwen2.5-Omni-3B | 4.8B | ~10 GB | ~5–6 GB | 8 GB GPU / Apple Silicon |
| Qwen2.5-Omni-7B | 8.5B | ~18 GB | ~10–12 GB (AWQ/GPTQ-Int4) | 12–16 GB GPU |
| Qwen2-Audio-7B | 7B | ~16 GB | ~6 GB | 12–16 GB GPU |
| Audio Flamingo 3 | 7B | ~14–16 GB | (no public 4-bit) | 24 GB GPU |
| Phi-4-multimodal | 5.6B | ~11 GB | ~4–6 GB | 12 GB GPU |
| Kimi-Audio-7B | 7B | ~24 GB (full stack) | ~8 GB | 24 GB GPU |
| MiniCPM-o 2.6 | 8B | ~16 GB | ~8 GB (MiniCPM-o.cpp) | 8–12 GB GPU / Jetson |
| SALMONN | 7B / 13B | ~16 / ~28 GB | — | 24 GB GPU (7B) |
| Ultravox | 8B | ~16 GB | ~5 GB (GGUF) | 8–12 GB GPU |
| Granite-Speech-4.1 | 2B | ~4 GB | ~2 GB | 8 GB GPU / CPU |
| Granite-Speech-3.3 | 8B | ~16 GB | ~6 GB | 12–16 GB GPU |

Tier mapping:

- **8 GB GPU (RTX 4060/3060) or edge**: Qwen2.5-Omni-3B (Q4), MiniCPM-o 2.6 (4-bit), Ultravox-1B, Granite-Speech-2B.
- **12–16 GB GPU (RTX 4070 Ti/4080)**: Qwen2.5-Omni-7B (AWQ/GPTQ-Int4), Phi-4-multimodal, Ultravox-8B (Q4), Qwen2-Audio-7B.
- **24 GB GPU (RTX 4090/3090)**: any 7B model at FP16/Q8, including Audio Flamingo 3 and Kimi-Audio.
- **Apple Silicon (32–64 GB unified)**: Qwen2.5-Omni-3B via LM Studio (MLX); audio-LLM MLX support beyond this is still thin.

> ⚠️ Kimi-Audio's higher published figures (up to 24–64 GB) reflect the full model with its **audio-generation/codec stack** loaded. Understanding-only use needs considerably less. Likewise, very low "Q4 ≈ 5 GB" figures for omni models typically count the **LLM backbone only**, not the encoder, which stays at higher precision.

## Inference Frameworks

Framework support is the single biggest practical constraint for local audio LLMs — far more limited than for text or even vision models.

| Framework | Audio-LLM support | Notes |
|-----------|-------------------|-------|
| **HuggingFace Transformers** | Broadest, most reliable | Native model classes for Qwen2-Audio, Qwen2.5-Omni, Audio Flamingo 3, Phi-4-multimodal, MiniCPM-o, SALMONN, Granite-Speech, Kimi-Audio. The reference path for understanding. |
| **vLLM** | Broadest for *serving* | Native audio support for Qwen2-Audio, Ultravox, Kimi-Audio, Granite-Speech, Audio Flamingo 3, MiniCPM-o, Phi-4-mm-audio, Whisper. Best for high-throughput/batch. |
| **llama.cpp (GGUF)** | **Experimental, narrow** | Audio via `libmtmd`: Ultravox (most mature), Qwen2-Audio, Qwen2.5-Omni (early/unreliable). Maintainers label audio "highly experimental, reduced quality". **No** AF3, Phi-4-mm, Kimi-Audio, SALMONN, Granite-Speech. |
| **Ollama** | **No native audio input** | Audio-input support is an open feature request; MiniCPM-o on Ollama runs its vision path, not audio. |
| **MiniCPM-o.cpp** | MiniCPM-o only | Dedicated ggml engine; 8 GB 4-bit, streaming audio/video, Jetson-friendly. |
| **MLX / MLX-Audio (Apple)** | Whisper STT/TTS only | Does **not** run understanding audio LLMs directly; Qwen2.5-Omni-3B works via LM Studio's MLX build. |

**Practical guidance:**

- For **maximum model coverage and correctness**, use **HuggingFace Transformers**.
- For **throughput / serving**, use **vLLM** — it has the widest native audio-model list and proper batching.
- For **CPU / tiny-VRAM / edge**, your realistic options are **Ultravox via llama.cpp** (speech only) or **MiniCPM-o via MiniCPM-o.cpp**. Do not assume your favourite text-LLM runner handles audio — most don't yet.

Throughput is encoder-bound at the start: e.g. Ultravox on an A100-40GB shows ~150 ms time-to-first-token and 50–100 tok/s after the audio prefill. Batch (offline) transcription-plus-understanding is far more forgiving than interactive use.

## Quantisation

Audio LLMs quantise much like vision-language models:

- **The LLM backbone quantises well** — Q4_K_M / Q8 (GGUF), AWQ, and GPTQ-Int4 all work. Qwen2.5-Omni ships **official GPTQ-Int4 and AWQ** builds that cut VRAM by 50%+; MiniCPM-o has GGUF and a 4-bit MiniCPM-o.cpp path; Ultravox has community GGUFs (Q4–Q6); Kimi-Audio runs at 4-bit.
- **Keep the audio encoder near FP16.** vLLM's LLM-Compressor documentation explicitly advises *against* quantising encoder parameters — the accuracy cost outweighs the small memory saving. In llama.cpp, the Ultravox `mmproj` (projector) is kept at F16. This mirrors the well-documented quantisation sensitivity of *vision* encoders.

The upshot: a 4-bit *backbone* + FP16 *encoder* is the standard local recipe, and it is why "4-bit" audio-LLM VRAM figures are higher than the same arithmetic would give for a text model.

## Specialised Models: Music and Long-Audio

- **Music understanding** — NVIDIA's **Music Flamingo** (7B, an Audio Flamingo derivative trained with reinforcement learning on music) is the purpose-built open option, but inherits Audio Flamingo's **non-commercial weights**. For commercial use, Qwen2.5-Omni and Kimi-Audio are the strongest permissively-licensed music understanders. The **MuChoMusic** benchmark (knowledge + reasoning over 644 tracks) is the standard music evaluation.
- **Long-audio** — **Audio Flamingo 3** handles 10 minutes natively; the **AF-Next** research line (NVIDIA + University of Maryland, April 2026) introduces *Temporal Audio Chain-of-Thought* — reasoning steps grounded to timestamps — and extends to 30-minute inputs, with three variants (Instruct, Think, Captioner). AF-Next reports beating Gemini 2.5 Pro on long-audio benchmarks (developer-reported). The **ChronosAudio** and **AudioMarathon** benchmarks exist specifically because most evaluations only test short clips and models degrade badly beyond a minute.
- **Sound-event / environmental audio** — Audio Flamingo 3 and **GAMA** are the strongest open options; **Clotho-AQA** is the canonical environmental-sound QA dataset.

## Limitations and Reliability

The benchmark literature is unusually candid about how far audio LLMs still have to go. The recurring failure modes:

1. **Temporal reasoning is weak.** Event *ordering*, *duration* estimation, and *counting* of sound events are consistently below 50% accuracy on the TREA benchmark — well behind humans. "How many times did the dog bark, and was it before or after the doorbell?" is genuinely hard for these models.
2. **Multi-hop reasoning collapses.** On SAKURA, accuracy falls dramatically from single-hop to multi-hop even when the model *can* extract each individual attribute: Qwen2-Audio 81%→49%, GPT‑4o Audio 71%→54%, Gemini-1.5-pro 63%→47%. Models perceive but fail to *chain*.
3. **Long audio degrades sharply.** ChronosAudio/AudioMarathon show severe performance loss as clips lengthen; short-clip skill does not transfer to realistic recordings.
4. **Hallucination is real and under-studied.** HalluAudio (ACL 2026), the first large-scale audio-hallucination benchmark, finds pervasive deficiencies in *acoustic grounding* (claiming sounds that aren't there), temporal reasoning, and music-attribute understanding, especially under adversarial prompts or mixed audio.
5. **Perceptual errors dominate.** MMAU error analysis attributes the majority of mistakes (55–64% across models) to *perception* — mishearing the audio — rather than reasoning.
6. **"Just ASR + LLM?"** A live research question is whether speech-understanding models meaningfully surpass a cascaded transcribe-then-LLM pipeline. For many speech tasks, the margin over a strong ASR+LLM baseline is small — a reminder that "end-to-end audio model" does not automatically mean "deeper understanding".

**Implications for use:** audio LLMs are reliable for *holistic* tasks (summarise this clip, what's the general scene, transcribe and answer a high-level question) and unreliable for *precise* tasks (exact counts, fine temporal order, multi-step deductions, long recordings). Verify any quantitative or ordering claim, and prefer chunk-and-aggregate pipelines for long audio.

## Licensing

| Model | Licence | Commercial use |
|-------|---------|----------------|
| Qwen2-Audio / Qwen2.5-Omni / Qwen3-Omni | Apache 2.0 | ✅ Yes |
| Kimi-Audio | Apache 2.0 (Qwen-derived code) + MIT | ✅ Yes |
| Phi-4-multimodal | MIT | ✅ Yes |
| MiniCPM-o 2.6 | Apache 2.0 | ✅ Yes |
| SALMONN | Apache 2.0 | ✅ Yes |
| GAMA | Apache 2.0 | ✅ Yes |
| Ultravox | MIT | ✅ Yes |
| IBM Granite-Speech | Apache 2.0 | ✅ Yes |
| Step-Audio 2 / Step-Audio-R1 | Apache 2.0 | ✅ Yes |
| DeSTA2 / DeSTA2.5-Audio | Open (exact type unconfirmed) | ⚠️ Verify the repo LICENSE |
| **Audio Flamingo 2 / 3 / Music Flamingo** | **Code MIT; weights NVIDIA OneWay Noncommercial** | ❌ **Non-commercial only** |

**The one trap to remember:** the entire **Audio Flamingo** family (including Music Flamingo and the AF-Next research line) ships its *checkpoints* under NVIDIA's non-commercial licence. The source code is MIT, but you cannot deploy the released weights commercially. For commercial audio understanding, default to **Qwen2.5-Omni / Qwen3-Omni, Kimi-Audio, Phi-4-multimodal, or MiniCPM-o**, all of which are Apache 2.0 or MIT.

## Managed Service Equivalents

If you would rather call an API than self-host, the major hyperscalers offer audio understanding (note: only the major hyperscalers are listed, per this repository's conventions):

| Provider | Service | Audio-understanding capability |
|----------|---------|-------------------------------|
| **GCP** | **Gemini API / Vertex AI** (incl. Live API native audio) | Native audio understanding — transcription, chapterisation, key-event detection, summarisation, audio Q&A; up to ~8.4 h (1M tokens) per prompt. The strongest managed option. |
| **Azure** | **Azure AI Foundry / Azure OpenAI** (GPT‑4o audio, realtime) + **Azure AI Speech** | GPT‑4o audio models for audio-input understanding and real-time speech-to-speech; Azure AI Speech for ASR/translation. |
| **AWS** | **Amazon Bedrock — Nova Sonic / Nova 2 Sonic**; **Amazon Transcribe** | Nova Sonic unifies speech understanding and generation for contextual voice interaction; Transcribe (+ Call Analytics) covers ASR and analytics. |
| **IBM** | **watsonx.ai — Granite-Speech** | Primarily ASR/AST (two-pass transcribe-then-LLM), not general audio reasoning. The open Granite-Speech weights are also self-hostable (Apache 2.0). |
| **Oracle** | **OCI Speech** | ASR/transcription only; no dedicated audio-understanding/reasoning model found. |

For genuine audio *reasoning*, **GCP (Gemini)** and **Azure (GPT‑4o audio)** are the most capable managed offerings; **AWS Nova Sonic** targets conversational voice; **IBM** and **Oracle** are transcription-centric.

## Caveats and Limitations

- **Vendor-reported SOTA.** Many "beats Gemini/GPT‑4o" claims (Audio Flamingo 3, Qwen3.5-Omni, AF-Next) are from the developers' own evaluations. Where possible this article leans on peer-reviewed benchmarks (MMAU, MMAU-Pro, SAKURA, TREA, MMAR) and labels vendor claims as such.
- **A fast-moving, partly unverified frontier.** "Qwen3.5-Omni" (early 2026) rests on a single commercial-host source and is flagged as unverified; "Qwen3-Omni" (September 2025, Apache 2.0) is the confirmed release. 2026 MMAR leaderboard numbers are community-aggregated.
- **Benchmark contamination.** Audio benchmarks built on public datasets (AudioCaps, Clotho) risk training-data overlap; audio-specific contamination study lags the text domain.
- **Scope boundary.** Speech-generation quality, latency, and dialogue dynamics are out of scope here — see [`real-time-voice-llms.md`](real-time-voice-llms.md).
- **Coverage gaps.** No published audio-benchmark scores were found for Phi-4-multimodal; DeSTA2's exact licence is unconfirmed; Apple-Silicon/MLX support for understanding-focused audio LLMs is immature and lightly documented. Several searches hit rate limits, so this is a representative — not exhaustive — survey.

## References

1. [Audio Flamingo 3 — NVIDIA ADLR](https://research.nvidia.com/labs/adlr/AF3/) — fully open 7B LALM; AF-Whisper encoder; NeurIPS 2025 Spotlight.
2. [NVIDIA Audio Flamingo series overview (DeepWiki)](https://deepwiki.com/NVIDIA/audio-flamingo) — AF1→AF3 + Music Flamingo lineage.
3. [Audio Flamingo 2 — NVIDIA ADLR](https://research.nvidia.com/labs/adlr/AF2/) — 3B long-audio model.
4. [Audio Flamingo 3 — HuggingFace Transformers docs](https://huggingface.co/docs/transformers/main/en/model_doc/audioflamingo3.md) — architecture, 10-min windowing, framework support.
5. [Audio Flamingo Next (AF-Next) — GAMMA Lab, U. Maryland](https://gamma.umiacs.umd.edu/media/af_next/) — Temporal Audio Chain-of-Thought; 30-min audio.
6. [NVIDIA Audio Flamingo GitHub (licensing)](https://github.com/NVIDIA/audio-flamingo) — code MIT, weights non-commercial.
7. [SALMONN: Towards Generic Hearing Abilities for LLMs (arXiv 2310.13289, ICLR 2024)](https://arxiv.org/abs/2310.13289) — Whisper+BEATs, window-level Q-Former.
8. [SALMONN GitHub (ByteDance/Tsinghua)](https://github.com/bytedance/SALMONN) — 7B/13B; video-SALMONN line.
9. [Qwen2.5-Omni Technical Report (arXiv 2503.20215)](https://arxiv.org/abs/2503.20215) — Thinker-Talker, TMRoPE, streaming.
10. [Qwen2.5-Omni GitHub](https://github.com/QwenLM/Qwen2.5-Omni) — 7B/3B, VRAM, GPTQ-Int4/AWQ, vLLM.
11. [Qwen2-Audio GitHub](https://github.com/QwenLM/Qwen2-Audio) — Whisper-large-v3 + Qwen2.
12. [Qwen3-Omni Legal & Licensing (DeepWiki)](https://deepwiki.com/QwenLM/Qwen3-Omni/9-legal-and-licensing) and [LICENSE](https://github.com/QwenLM/Qwen3-Omni/blob/main/LICENSE) — Apache 2.0.
13. [Kimi-Audio GitHub (Moonshot AI)](https://github.com/MoonshotAI/Kimi-Audio) and [Technical Report (arXiv 2504.18425)](https://arxiv.org/abs/2504.18425) — 7B, 12.5 Hz tokenizer.
14. [Kimi-Audio-7B model card](https://huggingface.co/moonshotai/Kimi-Audio-7B) — Apache 2.0 / MIT licensing.
15. [MiniCPM-o 2.6 model card (OpenBMB)](https://huggingface.co/openbmb/MiniCPM-o-2_6) — encoder stack and capabilities.
16. [MiniCPM-o.cpp (DeepWiki)](https://deepwiki.com/360CVGroup/MiniCPM-o.cpp) — 8 GB 4-bit edge engine.
17. [Phi-4-multimodal LICENSE (Microsoft, HuggingFace)](https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/LICENSE) — MIT.
18. [Granite Speech — IBM Documentation](https://www.ibm.com/granite/docs/models/speech) and [GitHub](https://github.com/ibm-granite/granite-speech-models) — Apache 2.0, two-pass ASR/AST.
19. [DeSTA2 GitHub (NTU)](https://github.com/kehanlu/desta2) and [DeSTA2.5-Audio (DeepWiki)](https://deepwiki.com/kehanlu/DeSTA2.5-Audio) — descriptive speech-text alignment.
20. [GAMA GitHub (U. Maryland)](https://github.com/Sreyan88/GAMA) — general-purpose audio LLM, Apache 2.0.
21. [Ultravox v0.5 GGUF + llama.cpp notes](https://huggingface.co/steampunque/ultravox-v0_5-llama-3_1-8b-Hybrid-GGUF) — Fixie AI, MIT.
22. [Step-Audio 2 Mini release (MarkTechPost)](https://www.marktechpost.com/2025/08/31/stepfun-ai-releases-step-audio-2-mini-an-open-source-8b-speech-to-speech-ai-model-that-surpasses-gpt-4o-audio/) — Apache 2.0.
23. [MMAU benchmark (GitHub, ICLR 2025)](https://github.com/Sakshi113/mmau) — 10k clips, 27 tasks, human ceiling 82.2%.
24. [MMAU-Pro project page (AAAI 2026)](https://sonalkum.github.io/mmau-pro/) — 5,305 instances, 49 skills.
25. [MMAR: Deep Reasoning Benchmark (arXiv 2505.13032, NeurIPS 2025)](https://arxiv.org/abs/2505.13032) — 1,000 deep-reasoning items.
26. [SAKURA: Multi-hop Reasoning of LALMs (Interspeech 2025)](https://arxiv.org/html/2505.13237v2) — single- vs multi-hop collapse.
27. [TREA: Temporal Reasoning Evaluation of Audio (GitHub, Interspeech 2025)](https://github.com/iiscleap/Audio-LLM-benchmarking-uncertainty) — ordering/duration/counting.
28. [AIR-Bench (GitHub, ACL 2024)](https://github.com/OFA-Sys/AIR-Bench) — foundation + chat dimensions.
29. [AudioBench (GitHub, NAACL 2025)](https://github.com/AudioLLMs/AudioBench) — 8 categories, 40+ datasets.
30. [Dynamic-SUPERB Phase-2 (arXiv 2411.05361, ICLR 2025)](https://arxiv.org/abs/2411.05361) — 180 tasks.
31. [MuChoMusic (ICML 2024)](https://mulab-mir.github.io/muchomusic/) — music understanding benchmark.
32. [Clotho-AQA (arXiv 2204.09634)](https://arxiv.org/abs/2204.09634) — environmental-sound audio QA.
33. [HalluAudio: Hallucination Benchmark for LALMs (arXiv 2604.19300, ACL 2026)](https://arxiv.org/abs/2604.19300) — acoustic grounding deficiencies.
34. [Audio Reasoning Challenge — Interspeech 2026](https://audio-reasoning-challenge.github.io/) — MMAR + chain-of-thought.
35. [Speech Discrete Tokens or Continuous Features? (arXiv 2508.17863, EMNLP 2025)](https://arxiv.org/abs/2508.17863) — continuous features win for understanding.
36. [Moshi: speech-text foundation model (arXiv 2410.00037, Kyutai)](https://arxiv.org/abs/2410.00037) — Mimi codec, discrete tokens.
37. [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/) — native audio-model list.
38. [Audio input support in llama.cpp (discussion)](https://github.com/ggml-org/llama.cpp/discussions/13759) and [multimodal docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md) — experimental audio via libmtmd.
39. [LLM-Compressor multimodal quantisation guidance (vLLM)](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/multimodal_vision/) — keep encoders near FP16.
40. [Audio Understanding — Gemini API docs (Google)](https://ai.google.dev/gemini-api/docs/audio) and [Gemini 2.5 native audio](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-2-5-native-audio/) — up to 8.4 h context.
41. [Amazon Nova Sonic — Bedrock docs (AWS)](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-sonic.html) — unified speech understanding/generation.
42. [GPT‑4o audio models in Azure AI Foundry (Microsoft)](https://devblogs.microsoft.com/foundry/azure-openai-gpt4o-audio-models-developer-guide/) — audio understanding + realtime.
43. [OCI AI Services (Oracle)](https://docs.oracle.com/cd/G30556_01/books/AppsAdmin/c-About-OCI-AI-Services.html) — OCI Speech ASR.
