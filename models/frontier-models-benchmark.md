# Frontier AI Models

| Field | Value |
|-------|-------|
| Created | 2026-03-17 |
| Last Updated | 2026-06-08 |
| Version | 2.9 |

---

- [Overview](#overview)
- [Best Overall](#best-overall)
- [Agentic Coding](#agentic-coding)
- [Tool Use](#tool-use)
- [Visual Understanding](#visual-understanding)
- [Audio Understanding](#audio-understanding)
- [Voice-to-Voice](#voice-to-voice)
- [Open Source](#open-source)
- [Best Models <= 20B Parameters](#best-models--20b-parameters)
- [Throughput (tok/s)](#throughput-toks)
- [References](#references)

## Overview

This article ranks current best-in-class frontier AI models across nine capability domains, synthesised from live benchmark data. The LMArena Elo and the Best Overall / Agentic Coding / Tool Use sections were refreshed on 8 June 2026 (Arena snapshot of 7 June 2026; Claude Opus 4.8 and Gemma 4 12B added); other sections retain their May 2026 pull. Sources include LMArena (Arena AI), SWE-Bench Verified/Pro, Terminal-Bench 2.0/2.1/Hard, Aider Polyglot, BFCL V4, Tau²-bench, MMMU-Pro, Artificial Analysis, BenchLM, and the OpenVLM/Open ASR leaderboards.

Each model entry includes origin, release date, model lineage, licence, parameter count, context window, and throughput data where known.

## Best Overall

Ranked by LMArena (Chatbot Arena) Elo ratings from human blind-preference votes, cross-referenced with multi-benchmark composites.

| # | Model | Origin | Released | Predecessor | Successor | Licence | Params | Context | tok/s | TTFT |
|---|-------|--------|----------|-------------|-----------|---------|--------|---------|-------|------|
| 1 | **Claude Opus 4.6 (thinking)** | Anthropic | 2026-02-05 | Claude Opus 4.5 | Claude Opus 4.7 | Proprietary | Undisclosed | 1M | ~50-60 | ~2-5 s |
| 2 | **Claude Opus 4.7 (thinking)** | Anthropic | 2026-04-16 | Claude Opus 4.6 | Claude Opus 4.8 | Proprietary | Undisclosed | 1M | ~50-60 | ~2-5 s |
| 3 | **Claude Opus 4.6** | Anthropic | 2026-02-05 | Claude Opus 4.5 | Claude Opus 4.7 | Proprietary | Undisclosed | 1M | ~50-60 | ~2-5 s |
| 4 | **Claude Opus 4.7** | Anthropic | 2026-04-16 | Claude Opus 4.6 | Claude Opus 4.8 | Proprietary | Undisclosed | 1M | ~50-60 | ~2-5 s |
| 5 | **Muse Spark** | Meta | 2026-04 | — | Incumbent | Proprietary | Undisclosed | Unknown | Unknown | Unknown |
| 6 | **Gemini 3.1 Pro Preview** | Google DeepMind | 2026-02-19 | Gemini 3 Pro | Incumbent | Proprietary | Undisclosed (MoE) | 1M | ~110 | ~0.9 s |
| 7 | **Gemini 3 Pro** | Google DeepMind | 2026-01 | Gemini 2.5 Pro | Gemini 3.1 Pro | Proprietary | Undisclosed (MoE) | 1M | ~80+ | ~1-3 s |
| 8 | **Claude Opus 4.8 (thinking)** | Anthropic | 2026-05-28 | Claude Opus 4.7 | Incumbent | Proprietary | Undisclosed | 1M | ~50-60 (2.5× fast mode) | ~2-5 s |
| 9 | **GPT-5.5 (high)** | OpenAI | 2026-04-23 | GPT-5.4 | Incumbent | Proprietary | Undisclosed | 1M | Unknown | Unknown |
| 10 | **GPT-5.4 High** | OpenAI | 2026-03-05 | GPT-5.3 Codex | GPT-5.5 | Proprietary | Undisclosed | 1M | ~187 | ~1.2 s |

**Key observations (LMArena Elo refreshed to the 7 June 2026 snapshot):**
- Anthropic holds the top of the Arena. **Claude Opus 4.6 (thinking) leads at Elo 1504**, with Opus 4.7 (thinking) just behind at 1501 — roughly +13 over the nearest non-Anthropic model (Muse Spark at 1489). Note the lineage is *non-monotonic on the text arena*: the newer Opus 4.7 and Opus 4.8 score slightly below Opus 4.6 here, even though they lead on agentic coding and tool-use benchmarks.
- **Claude Opus 4.8** (released 28 May 2026) is the current flagship, entering the Arena at Elo 1482 (thinking) / 1479 (base). It posts the strongest agentic-coding and computer-use results of the Opus line (SWE-Bench Pro 69.2%), with a 2.5× fast mode now ~3× cheaper than on Opus 4.7, at unchanged $5/$25 per 1M pricing.
- **GPT-5.5 "Spud"** (released 23 April 2026) is a full-generation upgrade over GPT-5.4. It leads the Artificial Analysis Intelligence Index at 60, breaking the previous three-way tie at 57. Its best Arena variant (high) sits at Elo 1482. Priced at $5/M input, $30/M output (Standard); $30/$180 (Pro).
- **Muse Spark** (Meta) holds a place in the Arena top 5 at Elo 1489. Details are limited.
- GPT-5.4 Pro still leads BenchLM's overall composite at 92/100 (#1 of 225+ models evaluated); GPT-5.5 follows at 91/100.
- GLM-5.1 (Zhipu AI, Elo 1475) is the highest-ranked open-source model — see the [Open Source](#open-source) section.

## Agentic Coding

Composite ranking across SWE-Bench Verified, SWE-Bench Pro, Terminal-Bench 2.0, Terminal-Bench Hard, and Aider Polyglot — the benchmarks most representative of real-world agentic coding.

| # | Model | Origin | Released | Predecessor | Successor | Licence | Params | Context | tok/s | TTFT |
|---|-------|--------|----------|-------------|-----------|---------|--------|---------|-------|------|
| 1 | **Claude Opus 4.8** | Anthropic | 2026-05-28 | Claude Opus 4.7 | Incumbent | Proprietary | Undisclosed | 1M | ~50-60 (2.5× fast mode) | ~2-5 s |
| 2 | **Claude Opus 4.7** | Anthropic | 2026-04-16 | Claude Opus 4.6 | Claude Opus 4.8 | Proprietary | Undisclosed | 1M | ~50-60 | ~2-5 s |
| 3 | **GPT-5.5** | OpenAI | 2026-04-23 | GPT-5.4 | Incumbent | Proprietary | Undisclosed | 1M | Unknown | Unknown |
| 4 | **GPT-5.4 Pro** | OpenAI | 2026-03-06 | GPT-5.3 Codex | GPT-5.5 | Proprietary | Undisclosed | 1M | ~187 | ~1.2 s |
| 5 | **Gemini 3.1 Pro** | Google DeepMind | 2026-02-19 | Gemini 3 Pro | Incumbent | Proprietary | Undisclosed (MoE) | 1M | ~110 | ~0.9 s |
| 6 | **Claude Opus 4.6** | Anthropic | 2026-02-05 | Claude Opus 4.5 | Claude Opus 4.7 | Proprietary | Undisclosed | 1M | ~50-60 | ~2-5 s |
| 7 | **Claude Sonnet 4.6** | Anthropic | 2026-02-17 | Claude Sonnet 4.5 | Incumbent | Proprietary | Undisclosed | 1M | ~80-100 | ~1-3 s |
| 8 | **Kimi K2.6** | Moonshot AI | 2026-04-20 | Kimi K2.5 | Incumbent | Open weights (Modified MIT) | ~1T / 32B active (MoE) | 256K | Unknown | Unknown |
| 9 | **MiniMax M2.5** | MiniMax | Q1 2026 | MiniMax M2.1 | Incumbent | Open weights | 230B (MoE) | 204K | ~100 | Unknown |
| 10 | **GLM-4.7** | Zhipu AI | Q1 2026 | GLM-4.6 | Incumbent | Open-source | Undisclosed (MoE) | 128K | Unknown | Unknown |

**Benchmark scores (where available):**

| Model | SWE-Bench Verified | SWE-Bench Pro | Terminal-Bench | Terminal-Bench Hard | Aider Polyglot |
|-------|-------------------|--------------|--------------------|--------------------|---------------|
| Claude Opus 4.8 | 88.6% | **69.2%** | 74.6% (TB 2.1) | — | — |
| Claude Opus 4.7 | 87.6% | 64.3% | 66.1% (TB 2.1) | — | — |
| GPT-5.5 | 88.7% | 58.6% | 82.7% (TB 2.0) | — | — |
| GPT-5.4 Pro | 86.0% | — | — | — | — |
| GPT-5.4 | 78.2% | 57.7% | — | 58% | — |
| Gemini 3.1 Pro | 80.6% | 54.2% | 78.4% | 54% | — |
| Claude Opus 4.6 | 80.8% | — | 74.7% (TB 2.0) | 53% | — |
| Claude Sonnet 4.6 | 79.6% | — | — | 49% | — |
| Kimi K2.6 | 80.2% | 58.6% | — | — | — |
| MiniMax M2.5 | 80.2% | — | — | — | — |
| GLM-4.7 | 73.8% | — | — | — | — |

**Key observations:**
- **Claude Opus 4.8 (released 28 May 2026) now leads SWE-Bench Pro at 69.2%** — a ~5-point jump over Opus 4.7 (64.3%) on the hardest agentic coding benchmark. It also edges Opus 4.7 on SWE-Bench Verified (88.6% vs 87.6%) and Terminal-Bench 2.1 (74.6% vs 66.1%).
- On SWE-Bench Verified, **Opus 4.8 (88.6%) and GPT-5.5 (88.7%) are effectively tied** at the top (both vendor-reported). Independent harnesses (vals.ai, mini-swe-agent) run 5-7 points lower across the board.
- **Terminal-Bench is harness-sensitive**: GPT-5.5 reaches 83.4% on OpenAI's Codex CLI harness but ~78% on the public Terminus-2 harness; Opus 4.8's 74.6% is on Terminus-2 (TB 2.1). Compare only within the same harness.
- GPT-5.4 Pro retains the highest SWE-Bench Verified score at 86.0%, though this appears scaffold-dependent — vals.ai lists GPT-5.4 at 78.2% on the same benchmark with a different harness.
- **Kimi K2.6** (Moonshot AI, 20 April 2026) matches GPT-5.5 on SWE-Bench Pro at 58.6% — the strongest open-weights coding result. Supports 12-hour autonomous coding sessions and 300-agent swarms.
- Claude Mythos Preview (gated, ~50 companies via "Project Glasswing") scored 93.9% on SWE-Bench Verified — not publicly available but signals the ceiling for agentic coding.
- The scaffold/harness effect remains large: the same model can swing 22+ points depending on the agent framework used.

## Tool Use

Ranked by the Berkeley Function Calling Leaderboard V4 (BFCL V4, last updated 12 April 2026), supplemented by Tau2-bench for business agent tool use.

| # | Model | Origin | Released | Predecessor | Successor | Licence | Params | Context | tok/s | TTFT |
|---|-------|--------|----------|-------------|-----------|---------|--------|---------|-------|------|
| 1 | **GLM-4.5 (FC)** | Zhipu AI | 2025-07-28 | GLM-4 | GLM-4.7 | Open-source (MIT) | 355B (MoE) | 128K | Unknown | Unknown |
| 2 | **Claude Sonnet 4** | Anthropic | 2025-06 | Claude Sonnet 3.7 | Claude Sonnet 4.5 | Proprietary | Undisclosed | 200K | Unknown | Unknown |
| 3 | **Kimi K2.5** | Moonshot AI | 2026-01-27 | Kimi K2 | Incumbent | Open weights (MIT) | ~1T / 32B active (MoE) | 128K | Unknown | Unknown |
| 4 | **Claude Sonnet 4.5** | Anthropic | 2025-09-29 | Claude Sonnet 4 | Claude Sonnet 4.6 | Proprietary | Undisclosed | 200K | ~80-100 | ~1-3 s |
| 5 | **Llama 3.1 405B** | Meta | 2024-07-23 | Llama 3 70B | Llama 4 | Open-source (Llama) | 405B (dense) | 128K | Provider-dependent | Provider-dependent |
| 6 | **GPT-5 (high)** | OpenAI | 2025-08-07 | GPT-4o | GPT-5.2 | Proprietary | Undisclosed | 256K | ~63 | ~96 s (reasoning) |
| 7 | **Gemini 2.5 Pro** | Google DeepMind | 2025-03 | Gemini 2.0 Pro | Gemini 3 Pro | Proprietary | Undisclosed (MoE) | 1M | ~80+ | ~1-2 s |
| 8 | **Claude Opus 4.6** | Anthropic | 2026-02-05 | Claude Opus 4.5 | Incumbent | Proprietary | Undisclosed | 1M (beta) | ~50-60 | ~2-5 s |

**BFCL V4 — current top entries (overall accuracy):**

| Model | Overall Accuracy | Licence |
|-------|-----------------|---------|
| Qwen3.7 Max | 75.0% | Proprietary (preview) |
| Qwen3.5-397B-A17B | 72.9% | Apache 2.0 |
| Qwen3.5-122B-A10B | 72.2% | Apache 2.0 |
| GLM-4.5 (FC) | 70.9% | Open-source (MIT) |
| Claude Opus 4.1 | 70.4% | Proprietary |
| Claude Sonnet 4 | 70.3% | Proprietary |
| Qwen3.5-27B | 68.5% | Apache 2.0 |

**Tau²-bench — current per-domain leaders:**

| Domain | Leader | Score |
|--------|--------|-------|
| Airline | Claude Sonnet 4.5 | 70.0% |
| Retail | Claude Opus 4.6 | 91.9% |
| Telecom | Claude Opus 4.6 | 99.3% |

**Key observations:**
- **Qwen now dominates structured function calling**: Qwen3.7 Max (preview) leads BFCL V4 at 75.0%, with three more Qwen 3.5 variants in the top eight. **GLM-4.5 (Zhipu AI, MIT) is the best fully-open model at 70.9%.**
- On the multi-turn business-agent axis, Anthropic leads Tau²-bench: Claude Opus 4.6 tops retail (91.9%) and telecom (99.3%), and Claude Sonnet 4.5 tops airline (70.0%).
- **Both leaderboards are stale.** BFCL V4 has not been updated since 12 April 2026 and Tau²-bench lacks the newest releases — neither has evaluated Claude Opus 4.7/4.8, GPT-5.5, Gemini 3.1 Pro, or Kimi K2.6. Anthropic and OpenAI do not publish BFCL/Tau² figures for their latest models, so the strongest agentic-coding models are absent from the public tool-calling boards. Treat this section as a lagging indicator and cross-reference the [Agentic Coding](#agentic-coding) section.

## Visual Understanding

Ranked by MMMU-Pro (college-level multimodal understanding), cross-referenced with Video-MMMU, MathVista, ChartQA, and LMArena Vision.

| # | Model | Origin | Released | Predecessor | Successor | Licence | Params | Context | tok/s | TTFT |
|---|-------|--------|----------|-------------|-----------|---------|--------|---------|-------|------|
| 1 | **GPT-5.4 Pro** | OpenAI | 2026-03-06 | GPT-5.3 Codex | GPT-5.5 | Proprietary | Undisclosed | 1M | ~187 | ~1.2 s |
| 2 | **Gemini 3 Pro** | Google DeepMind | 2026-01 | Gemini 2.5 Pro | Gemini 3.1 Pro | Proprietary | Undisclosed (MoE) | 1M | ~80+ | ~1-3 s |
| 3 | **GPT-5.2** | OpenAI | 2025-12-11 | GPT-5 | GPT-5.4 | Proprietary | Undisclosed | 400K | ~187 | ~1.2 s |
| 4 | **Claude Opus 4.6** | Anthropic | 2026-02-05 | Claude Opus 4.5 | Claude Opus 4.7 | Proprietary | Undisclosed | 1M | ~50-60 | ~2-5 s |
| 5 | **Gemini 2.5 Pro** | Google DeepMind | 2025-03 | Gemini 2.0 Pro | Gemini 3 Pro | Proprietary | Undisclosed (MoE) | 1M | ~80+ | ~1-2 s |
| 6 | **Claude Sonnet 4.6** | Anthropic | 2026-02-17 | Claude Sonnet 4.5 | Incumbent | Proprietary | Undisclosed | 1M | ~80-100 | ~1-3 s |
| 7 | **Grok 4 Heavy** | xAI | 2025-07 | Grok 3 | Grok 4.1 | Proprietary | Undisclosed | 256K | Unknown | Unknown |
| 8 | **Qwen 3.5 VL** | Alibaba | 2026-02-16 | Qwen 2.5-VL-72B | Incumbent | Open-source (Apache 2.0) | 397B / 17B active (MoE) | 128K | Unknown | Unknown |

**Benchmark scores:**

| Model | MMMU-Pro | Video-MMMU | MathVista | ChartQA |
|-------|---------|-----------|-----------|---------|
| GPT-5.4 Pro | 94.0% | — | — | — |
| Gemini 3 Pro | 81.0% | 87.6% | 82.3% | 89.1% |
| GPT-5.2 | 79.5% | — | — | — |
| Claude Opus 4.6 | 76.8% | — | — | — |
| Gemini 2.5 Pro | 75.2% | — | — | — |
| Claude Sonnet 4.6 | 74.5% | — | — | — |
| Grok 4 Heavy | 73.6% | — | — | — |
| Qwen 3.5 VL | 71.5% | — | — | — |

**Key observations:**
- **GPT-5.4 Pro now leads MMMU-Pro at 94.0%** — a 13-point jump over Gemini 3 Pro (81.0%). This is the first time a non-Google model has topped this benchmark.
- Google DeepMind still dominates breadth across vision benchmarks. Gemini 3 Pro leads Video-MMMU (87.6%), MathVista (82.3%), and ChartQA (89.1%).
- Qwen 3.5 VL (397B MoE, Apache 2.0) is the best open-source vision model at 71.5% MMMU-Pro.
- Notable open-source runners-up: GLM-4.5V (106B MoE, MIT — SOTA on 41 open multimodal benchmarks), GLM-4.1V-9B-Thinking (9B, rivals 72B models), InternVL3-78B (strong on OCR and hallucination resistance).

## Audio Understanding

Ranked by accuracy (WER) and breadth of audio capabilities. The field spans dedicated ASR models and multimodal models with native audio.

| # | Model | Origin | Released | Predecessor | Successor | Licence | Params | Speed | Key Capabilities |
|---|-------|--------|----------|-------------|-----------|---------|--------|-------|-----------------|
| 1 | **NVIDIA Canary-Qwen 2.5B** | NVIDIA | Late 2025 | Canary-1B | Incumbent | CC-BY-4.0 | 2.5B | 418x RTFx | Best accuracy (5.63% WER), English only, SALM architecture for QA over audio |
| 2 | **IBM Granite Speech 3.3 8B** | IBM | 2025 | Granite Speech 3.0 | Incumbent | Apache 2.0 | ~9B | — | Enterprise-grade (5.85% WER), noise-robust, multi-language translation |
| 3 | **Gemini 2.5 Flash Native Audio** | Google DeepMind | 2025-12 | Gemini 2.0 Flash | Gemini 3.x | Proprietary | Undisclosed | Real-time | Native audio (no ASR cascade), 24 languages, function calling (71.5%), live translation 70+ languages |
| 4 | **OpenAI Whisper Large V3** | OpenAI | 2023-11 | Whisper Large V2 | Whisper V3 Turbo | MIT | 1.55B | Fast | Multilingual gold standard: 99+ languages, 7.4% WER |
| 5 | **OpenAI Whisper Large V3 Turbo** | OpenAI | 2024-10 | Whisper Large V3 | Incumbent | MIT | 809M | 216x RTFx | 6x faster than V3, 7.75% WER, optimal for speed-first multilingual |
| 6 | **Qwen2-Audio** | Alibaba | 2024 | Qwen-Audio | Incumbent | Open-source | ~8B | — | Universal: speech + environmental sound + music understanding |
| 7 | **SenseVoice** | FunAudioLLM (Alibaba) | 2024-07 | — | Incumbent | Open-source | Small / Large | Very fast | Multi-task: ASR + emotion recognition + audio event detection |
| 8 | **NVIDIA Parakeet TDT 1.1B** | NVIDIA | 2024 | Parakeet CTC | Canary-Qwen | CC-BY-4.0 | 1.1B | >2,000x RTFx | Ultra-fast streaming, English only, ideal for edge |

**Key observations:**
- The field has bifurcated: dedicated ASR models (Canary, Whisper, Parakeet) optimise for accuracy/speed on transcription, while multimodal models (Gemini Native Audio, Qwen2-Audio) understand audio semantically alongside text and vision.
- NVIDIA Canary-Qwen leads accuracy (5.63% WER) by combining a FastConformer encoder with a Qwen3-1.7B LLM decoder — a SALM architecture that enables summarisation and QA over audio, not just transcription.
- Gemini 2.5 Flash Native Audio is the most capable overall: no ASR cascade, native function calling, live translation across 70+ languages, and multimodal context during audio processing.
- 7 of 8 top models are open-source. Whisper V3 (MIT) remains the multilingual standard despite being two years old.

## Voice-to-Voice

Ranked by latency, conversational quality, and full-duplex capability for real-time voice conversation.

| # | Model / System | Origin | Released | Predecessor | Successor | Licence | Params | Latency | Key Capabilities |
|---|---------------|--------|----------|-------------|-----------|---------|--------|---------|-----------------|
| 1 | **OpenAI gpt-realtime** | OpenAI | 2025-08 (GA) | GPT-4o Realtime Preview | Audio-first model (Q1 2026) | Proprietary | Undisclosed | Sub-second | Single-model speech-to-speech, MCP/SIP, barge-in, used by Zillow/T-Mobile |
| 2 | **Gemini 2.5 Flash Native Audio (Live API)** | Google DeepMind | 2025-12 | Gemini 2.0 Flash Live | Gemini 3.x | Proprietary | Undisclosed | Sub-second | Native audio in/out, 30 HD voices, 24 languages, proactive audio mode |
| 3 | **Hume EVI 3** | Hume AI | 2025-05 | EVI 2 | Incumbent | Proprietary | Undisclosed | Conversational | Rated higher than GPT-4o (blind test): empathy, expressiveness, naturalness. 100K+ custom voices, emotion understanding |
| 4 | **NVIDIA PersonaPlex** | NVIDIA | 2026-01 | — | Incumbent | Open-source | Undisclosed | ~170 ms | Full-duplex, customisable voice/role, handles interruptions and backchannels |
| 5 | **Kyutai Moshi** | Kyutai Labs | 2024-07 | — | Incumbent | Apache 2.0 | ~7B | 160-200 ms | Pioneer full-duplex speech-to-speech, parallel streams, emotional cues |
| 6 | **ElevenLabs Conversational AI** | ElevenLabs | 2025-11 | ElevenLabs TTS API | Incumbent | Proprietary | Undisclosed | <300 ms | Enterprise orchestration platform, 70+ languages, tool integration |
| 7 | **Ultravox** | Fixie.ai | 2025 (v0.4.1) | Ultravox v0.3 | Incumbent | MIT | ~8B (Llama 3.1 8B + Whisper) | Low | Speech-native LLM, open-source, self-hostable |
| 8 | **OpenAI Audio-First Model** | OpenAI | Expected Q1 2026 | gpt-realtime | — | Proprietary (expected) | Undisclosed | TBD | Audio-first architecture (not text-first), full barge-in, continuous exchange |

**Key observations:**
- The field has split between **cascade-free native models** (gpt-realtime, Gemini Native Audio, EVI 3, PersonaPlex, Moshi) that process audio end-to-end, and **pipeline systems** (ElevenLabs) that chain STT + LLM + TTS.
- Full-duplex (simultaneous listen + speak) is the current frontier. NVIDIA PersonaPlex and Kyutai Moshi lead open-source; OpenAI's upcoming audio-first model targets this for proprietary.
- Hume EVI 3 is notable for its emotion understanding — the model reads and responds to emotional cues in speech, outperforming GPT-4o in blind comparisons on expressiveness and empathy.

## Open Source

The top open-source models overall, ranked by a composite of Chatbot Arena Elo, Artificial Analysis Intelligence Index, and key benchmark scores.

| # | Model | Origin | Released | Predecessor | Successor | Params (total / active) | Architecture | Context | Licence | Key Scores |
|---|-------|--------|----------|-------------|-----------|------------------------|-------------|---------|---------|------------|
| 1 | **GLM-5.1** | Zhipu AI | 2026-04-08 | GLM-5 | Incumbent | Undisclosed | MoE | 128K | Apache 2.0 | Arena Elo 1475, successor to GLM-5 |
| 2 | **GLM-5** | Zhipu AI | 2026-02-11 | GLM-4.5 (355B) | GLM-5.1 | 744B / ~40B | MoE | 128K | Apache 2.0 | Arena Elo 1456, Quality Index 49.6, MMLU-Pro 89.7%, HumanEval 92.1% |
| 3 | **DeepSeek V3.2** | DeepSeek | 2026-02-15 | DeepSeek V3.1 | Incumbent | 685B / 37B | MoE (DeepSeekMoE + MLA) | 128K | DeepSeek Licence | Quality Index 41.2, LiveCodeBench 86-90%, AIME 92%, SWE-Bench 72.8% |
| 4 | **Kimi K2.6** | Moonshot AI | 2026-04-20 | Kimi K2.5 | Incumbent | ~1T / 32B | MoE | 256K | Open weights (Modified MIT) | Intelligence Index 54, SWE-Bench Pro 58.6%, 300-agent swarm coding |
| 5 | **MiniMax M2.5** | MiniMax | Q1 2026 | MiniMax M2.1 | Incumbent | 230B / undisclosed | MoE | 204K | Open weights | Quality Index 42.0, SWE-Bench Verified 80.2% (#1 open-weights coding) |
| 6 | **MiMo-V2-Flash** | Xiaomi | 2025-12-16 | MiMo-V1 | Incumbent | 309B / 15B | MoE (hybrid attn + MTP) | 128K | Apache 2.0 | LiveCodeBench 87%, AIME 96%, ~150 tok/s |
| 7 | **Qwen3-235B-A22B** | Alibaba | 2025-04-29 | Qwen 2.5-72B | Qwen 3.5 | 235B / 22B | MoE | 128K | Apache 2.0 | Hybrid thinking (fast/deep), 119 languages |
| 8 | **Llama 4 Maverick** | Meta | 2025-04-05 | Llama 3.3 70B | Llama 4 Behemoth (pending) | 400B / 17B (128 experts) | MoE | 10M | Llama Community | Natively multimodal |
| 9 | **Gemma 4 31B** | Google DeepMind | 2026-04-02 | Gemma 3 27B | Incumbent | 31B | Dense | 256K | Apache 2.0 | Arena Elo ~1451 (#4 open), multimodal (text/image/video/audio), native function calling |
| 10 | **Mistral Large 3** | Mistral AI | 2025-12-02 | Mistral Large 2 (123B) | Incumbent | 675B / 41B | MoE | 256K | Apache 2.0 | Multimodal, strong European compliance |

**Key observations:**
- **GLM-5.1** (Zhipu AI, April 2026) takes the open-source crown from GLM-5, reaching Arena Elo 1475. GLM-5 (744B MoE) originally unseated DeepSeek in February 2026. Native support for 14+ Asian languages including Tamil, Hindi, and Telugu.
- **Kimi K2.6** (Moonshot AI, 20 April 2026) is the strongest new open-weights entry. Artificial Analysis Intelligence Index of 54 — the highest for any open-weight model — with SWE-Bench Pro 58.6% matching GPT-5.5. Supports 12-hour autonomous coding sessions and 300-agent coordinated swarms. $0.60/M input, $2.50/M output.
- **Gemma 4** (Google DeepMind, April 2, 2026) is a major open-source entry. The 31B dense model ranks among the top open models on Arena AI's text leaderboard (Elo ~1451; the 26B A4B MoE sits at ~1438). Apache 2.0 licence — a significant shift from Gemma 3's more restrictive Gemma Licence. Originally four sizes (E2B, E4B, 26B A4B MoE, 31B dense); a fifth — the **Gemma 4 12B** dense model with native audio — was added ~3 June 2026 as the 16 GB-laptop sweet spot (see the [Gemma 4 12B overview](gemma-4-12b.md)). The family has passed 150M downloads.
- **GLM-4.7** (Zhipu AI) is a notable coding-focused release with thinking capabilities: AIME 2025 95.7%, SWE-bench Verified 73.8%, HLE 42.8% (+12.4% over GLM-4.6).
- Every top open-source model at the frontier tier uses MoE architecture (Gemma 4 31B dense is the exception, ranking via raw parameter density).
- Apache 2.0 dominates licensing, with Meta's Llama Community Licence the main exception.
- The gap to proprietary models continues to narrow: GLM-5.1 at Arena Elo 1475 vs the Arena leader Claude Opus 4.6 (thinking) at 1504 — a ~29 Elo gap, down from ~200+ a year ago.

## Best Models <= 20B Parameters

Models with 20B or fewer total parameters (or, for MoE models, 20B or fewer active parameters with modest total size enabling single-GPU deployment).

| # | Model | Origin | Released | Predecessor | Successor | Total Params | Active Params | Context | Licence | Standout Capability |
|---|-------|--------|----------|-------------|-----------|-------------|---------------|---------|---------|---------------------|
| 1 | **Qwen3.5-9B (Reasoning)** | Alibaba | Q1 2026 | Qwen3-8B | Incumbent | 9B | 9B (dense) | 128K | Apache 2.0 | Intelligence Index 32 — most intelligent model under 10B |
| 2 | **Gemma 4 12B** | Google DeepMind | 2026-06-03 | Gemma 4 E4B | Incumbent | 11.95B | 11.95B (dense) | 256K | Apache 2.0 | Only mid-sized Gemma 4 with native audio; MMLU-Pro 77.2, AIME 2026 77.5, GPQA 78.8; runs in ~16 GB |
| 3 | **Qwen3-14B** | Alibaba | 2025-04-29 | Qwen 2.5-14B | Qwen3.5-14B | 14.8B | 14.8B (dense) | 131K | Apache 2.0 | 119 languages, hybrid thinking |
| 4 | **Mistral 3 (Ministral) 14B** | Mistral AI | 2025-12-02 | Mistral Small 3.1 | Incumbent | 14B | 14B (dense) | 256K | Apache 2.0 | Multimodal (vision), reasoning variant available |
| 5 | **Gemma 4 26B (MoE)** | Google DeepMind | 2026-04-02 | Gemma 3 27B | Incumbent | 26B (MoE) | 4B | 256K | Apache 2.0 | Multimodal (text/image/video), function calling, only 4B active params |
| 6 | **Llama 4 Scout** | Meta | 2025-04-05 | Llama 3.2-3B | Incumbent | 109B (16 experts) | 17B | 10M | Llama Community | Natively multimodal, fits single H100 |
| 7 | **Phi-4-mini** | Microsoft | 2025-02-27 | Phi-3.5-mini | Phi-4-multimodal | 3.8B | 3.8B (dense) | 128K | MIT | Matches models 2x its size on maths/code, function calling |
| 8 | **Gemma 4 E4B** | Google DeepMind | 2026-04-02 | Gemma 3 4B | Incumbent | 4B | 4B (dense) | 256K | Apache 2.0 | Multimodal (text/image/video/audio), function calling, 256K context in a 4B model |
| 9 | **Qwen3.5-4B (Reasoning)** | Alibaba | Q1 2026 | Qwen3-4B | Incumbent | 4B | 4B (dense) | 32K | Apache 2.0 | Intelligence Index 27 — most intelligent model under 5B, rivals Qwen2.5-72B on some tasks |
| 10 | **Gemma 4 E2B** | Google DeepMind | 2026-04-02 | Gemma 3 2B | Incumbent | 2.3B | 2.3B (dense) | 256K | Apache 2.0 | Smallest Gemma 4, multimodal, 256K context, mobile/edge deployment |

**Key observations:**
- Qwen dominates the small-model tier on reasoning. Qwen3.5-9B achieves Intelligence Index 32, and the 4B variant (Index 27) rivals models 18x its size on select benchmarks.
- **Gemma 4 12B** (released ~3 June 2026) is the newest entry and the family's "16 GB laptop sweet spot": an 11.95B dense model that, per Google's model card, nears the 26B A4B MoE on agentic (Tau² 69.0 vs 68.2), coding and long-context tasks while using under half the memory — and it is the **only mid-sized Gemma 4 with native audio**. See the dedicated [Gemma 4 12B overview](gemma-4-12b.md). It is too new to appear on LMArena or the public tool-calling leaderboards yet.
- **Gemma 4** (April 2026) also brings the 26B MoE (only 4B active — remarkably efficient), the E4B dense, and the E2B dense. All feature 256K context, multimodal capabilities, native function calling, and Apache 2.0 licensing. The MoE variant is particularly notable for fitting frontier-class capabilities into 4B active parameters.
- All top small models are dense (not MoE), with the exception of Gemma 4 26B and Llama 4 Scout.
- All are open-source with permissive licences (Apache 2.0 or MIT).
- Hardware: 7-14B dense models fit in 14-28 GB VRAM at FP16, or 4-8 GB quantised (Q4). A consumer RTX 4090 (24 GB) handles all of these.

## Throughput (tok/s)

Output token generation speed measured by Artificial Analysis (tested every 8 hours on live API endpoints) and the Awesome Agents speed leaderboard, as of April 2026.

### Artificial Analysis Intelligence Index

The Intelligence Index is a composite score across multiple benchmarks. As of early June 2026:

| # | Model | Score | Notes |
|---|-------|-------|-------|
| 1 | **Claude Opus 4.8** | 61.4 | New #1 (released 28 May 2026) |
| 2 | **GPT-5.5** | 60 | Broke the previous three-way tie at 57 |
| 3 | **Claude Opus 4.7** | 57 | |
| 4 | **Gemini 3.1 Pro Preview** | 57 | |
| 5 | **GPT-5.4** | 57 | |
| 6 | **Kimi K2.6** | 54 | Strongest open-weight model |
| 7 | **Claude Opus 4.6** | 53 | |

### Model Speed Rankings (via default API provider)

| # | Model | Output tok/s | TTFT (s) | Provider | Price (output /M tok) |
|---|-------|-------------|----------|----------|-----------------------|
| 1 | **Mercury 2** (diffusion LLM) | 629-1,009 | 0.80 | Inception Labs | $0.75 |
| 2 | **IBM Granite 3.3 8B** | 375 | 0.50 | IBM | $0.04 |
| 3 | **IBM Granite 4.0 H Small** | 355 | 0.50 | IBM | $0.05 |
| 4 | **Gemini 2.5 Flash-Lite** | 341 | 0.42 | Google | $0.30 |
| 5 | **AWS Nova Micro** | 305 | 0.60 | Amazon | $0.14 |
| 6 | **Gemini 2.5 Flash** | 221 | 0.45 | Google | $0.60 |
| 7 | **Gemini 3 Flash Preview** | 199 | 0.50 | Google | $0.80 |
| 8 | **GPT-5.2** | 187 | 1.20 | OpenAI | $10.00 |
| 9 | **Claude 4.5 Haiku** | 135 | 0.60 | Anthropic | $1.25 |
| 10 | **Gemini 3.1 Pro Preview** | 110 | 0.90 | Google | $5.00 |
| 11 | **GPT-5 (high reasoning)** | 58 | 1.50 | OpenAI | $15.00 |
| 12 | **Claude 4.5 Sonnet** | 40 | 1.50 | Anthropic | $15.00 |

### Inference Provider Speed Rankings (Llama 4 Maverick 400B)

| # | Provider | Output tok/s | Hardware |
|---|----------|-------------|----------|
| 1 | **Cerebras** | 2,522 | CS-3 Wafer-Scale Engine |
| 2 | **SambaNova** | 794 | SN40L RDU |
| 3 | **Groq** | 549 | LPU |
| 4 | **Amazon Bedrock** | 290 | NVIDIA GPUs |
| 5 | **Google Cloud** | 125 | TPU v5 |
| 6 | **Microsoft Azure** | 54 | NVIDIA GPUs |

### Time to First Token — Best Performers

| Model / Provider | TTFT (s) |
|-----------------|----------|
| Groq (any model) | <0.40 |
| Gemini 2.5 Flash-Lite | 0.42 |
| Gemini 2.5 Flash | 0.45 |
| IBM Granite 3.3 8B | 0.50 |
| Claude 4.5 Haiku | 0.60 |

**Key observations:**
- **Mercury 2** (Inception Labs, February 2026) is a paradigm shift: a diffusion-based LLM that generates tokens in parallel rather than autoregressively. At 629-1,009 tok/s it is 1.7-2x faster than any traditional LLM, while matching Claude 4.5 Haiku on reasoning (AIME 91.1%, GPQA 73.6%).
- **Cerebras CS-3** is the undisputed hardware speed leader at 2,522 tok/s on a 400B MoE model — roughly 4-6x faster than Groq. The wafer-scale engine holds entire models in SRAM, eliminating memory bandwidth bottlenecks.
- **Speed vs intelligence remains inversely correlated** but the gap is narrowing. Gemini 2.5 Flash (221 tok/s, $0.60/M output) is the current best compromise: fast enough for real-time streaming, competitive on benchmarks, and 25x cheaper than GPT-5.
- For **real-time chat/voice**: prioritise TTFT (<500 ms). Groq and Gemini 2.5 Flash-Lite are best.
- For **agentic workflows**: end-to-end latency per call matters most. Gemini 2.5 Flash is recommended.
- For **batch processing**: raw throughput wins. Cerebras at 2,500+ tok/s is unmatched.

## References

### Leaderboards and Benchmarks
- Arena AI (formerly LMArena / Chatbot Arena) — https://arena.ai/leaderboard
- SWE-Bench Verified / Pro — https://www.swebench.com
- Terminal-Bench 2.0 / Hard — https://www.marc0.dev/en/leaderboard
- Aider Polyglot Leaderboard — https://aider.chat/docs/leaderboards/
- BFCL V4 (Berkeley Function Calling) — https://gorilla.cs.berkeley.edu/leaderboard.html
- MMMU-Pro — https://artificialanalysis.ai/evaluations/mmmu-pro
- OpenVLM Leaderboard — https://huggingface.co/spaces/opencompass/open_vlm_leaderboard
- Open ASR Leaderboard — https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
- Artificial Analysis LLM Leaderboard — https://artificialanalysis.ai/leaderboards/models
- BenchLM — https://benchlm.ai
- Awesome Agents Speed Leaderboard — https://awesomeagents.ai/leaderboards/ai-speed-latency-leaderboard/
- LiveBench — https://livebench.ai

### Model Announcements and Documentation
- GPT-5.5 — https://ofox.ai/blog/gpt-5-5-api-vs-claude-opus-gemini-3-1-flagship-2026/
- Claude Opus 4.7 — https://ofox.ai/blog/claude-opus-4-7-api-review-upgrade-guide-2026/
- Claude Mythos Preview — https://www.infoq.com/news/2026/04/anthropic-claude-mythos/
- Kimi K2.6 — https://huggingface.co/moonshotai/Kimi-K2.6
- GPT-5.4 — https://openai.com/index/introducing-gpt-5-4/
- Claude Opus 4.6 — https://www.anthropic.com/news/claude-opus-4-6
- Claude Sonnet 4.6 — https://www.anthropic.com/news/claude-sonnet-4-6
- Gemini 3.1 Pro — https://deepmind.google/models/model-cards/gemini-3-1-pro/
- Grok 4.20 — https://x.ai/news/grok-4
- GLM-5 — https://z.ai/blog/glm-5
- DeepSeek V3.2 — https://api-docs.deepseek.com/news/news251201
- Kimi K2.5 / K2.6 — https://en.wikipedia.org/wiki/Kimi_(chatbot)
- MiniMax M2.5 — https://www.minimax.io
- MiMo-V2-Flash — https://mimo.xiaomi.com/blog/mimo-v2-flash
- Qwen3 / 3.5 — https://arxiv.org/abs/2505.09388, https://artificialanalysis.ai/articles/qwen3-5-small-models
- Llama 4 — https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- Mistral Large 3 — https://docs.mistral.ai/models/mistral-large-3-25-12
- Phi-4-mini — https://arxiv.org/abs/2503.01743
- Gemma 3 — https://arxiv.org/abs/2503.19786
- Gemma 4 — https://deepmind.google/models/gemma/gemma-4/
- GLM-4.7 — https://z.ai/blog/glm-4.7
- GPT-5.4 Pro — https://benchlm.ai/models/gpt-5-4-pro
- Mercury 2 — https://rits.shanghai.nyu.edu/ai/inception-launches-mercury-2-diffusion-powered-reasoning-at-1000-tokens-per-second/

### Audio and Voice Models
- OpenAI gpt-realtime — https://openai.com/index/introducing-gpt-realtime/
- Gemini Native Audio — https://blog.google/products-and-platforms/products/gemini/gemini-audio-model-updates/
- Hume EVI 3 — https://www.hume.ai/blog/introducing-evi-3
- NVIDIA PersonaPlex — https://research.nvidia.com/labs/adlr/personaplex/
- Kyutai Moshi — https://kyutai.org/Moshi.pdf
- NVIDIA Canary-Qwen — https://huggingface.co/nvidia/canary-qwen-2.5b
- IBM Granite Speech — https://huggingface.co/ibm-granite/granite-speech-3.3-8b
- Whisper V3 / Turbo — https://github.com/openai/whisper
- Qwen2-Audio — https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/qwen2_audio
- Open ASR benchmarks — https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks

### Analysis and Roundups
- OfficeChai Agentic Coding Rankings — https://officechai.com/ai/10-best-agentic-coding-and-terminal-use-models-march-2026/
- Pluralsight Best AI Models 2026 — https://www.pluralsight.com/resources/blog/ai-and-data/best-ai-models-2026-list
- MorphLLM Best AI for Coding — https://www.morphllm.com/best-ai-model-for-coding
- WhatLLM Vision Models — https://whatllm.org/blog/best-vision-models-january-2026
- Roboflow Best Multimodal Models — https://blog.roboflow.com/best-multimodal-models/
- BuildFastWithAI Best AI Models April 2026 — https://www.buildfastwithai.com/blogs/best-ai-models-april-2026
- Grok 4.1 LMArena dominance (April 2026) — https://www.modezone.com/grok-4-1-dominates-the-lmarena-leaderboard
- BenchLM LLM Leaderboard History — https://benchlm.ai/llm-leaderboard-history
- Lushbinary Gemma 4 Developer Guide — https://lushbinary.com/blog/gemma-4-developer-guide-benchmarks-architecture-local-deployment-2026/
- Ofox LLM Leaderboard April 2026 — https://ofox.ai/blog/llm-leaderboard-best-ai-models-ranked-2026/
- GPT-5.5 Intelligence Index — https://artificialanalysis.ai/articles/openai-gpt5-5-is-the-new-leading-AI-model
- GPT-5.5 vs Claude Opus 4.7 vs Gemini 3.1 Pro — https://ofox.ai/blog/gpt-5-5-api-vs-claude-opus-gemini-3-1-flagship-2026/
- Kimi K2.6 Release — https://www.marktechpost.com/2026/04/20/moonshot-ai-releases-kimi-k2-6-with-long-horizon-coding-agent-swarm-scaling-to-300-sub-agents-and-4000-coordinated-steps/
- Claude Mythos Preview — https://llm-stats.com/blog/research/claude-mythos-preview-launch
- LLM Stats Tool Calling Rankings — https://llm-stats.com/leaderboards/best-ai-for-tool-calling
- vals.ai SWE-bench Verified — https://llm-stats.com/benchmarks/swe-bench-verified
