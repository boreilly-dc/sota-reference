# Best Local LLMs for Consumer Hardware (Mid-2026)

| Field | Value |
|-------|-------|
| Created | 2026-05-31 |
| Last Updated | 2026-06-27 |
| Version | 2.0 |

---

- [Overview](#overview)
- [Model Release Timeline](#model-release-timeline)
- [The Landscape: Model Families](#the-landscape-model-families)
- [Best Models by VRAM Tier](#best-models-by-vram-tier)
- [Best Models by Use-Case](#best-models-by-use-case)
- [Benchmark Comparison](#benchmark-comparison)
- [Hardware Requirements](#hardware-requirements)
- [Quantisation Formats](#quantisation-formats)
- [Inference Frameworks](#inference-frameworks)
- [MoE Models on Consumer Hardware](#moe-models-on-consumer-hardware)
- [Apple Silicon and MLX](#apple-silicon-and-mlx)
- [AMD GPUs for Local Inference](#amd-gpus-for-local-inference)
- [What Changed Since May 2026](#what-changed-since-may-2026)
- [Licensing](#licensing)
- [Recommendations](#recommendations)
- [References](#references)

## Overview

The local LLM landscape in late June 2026 has shifted decisively in favour of mid-sized models. Two releases define the month: **Gemma 4 12B** (June 3) — an encoder-free multimodal model that handles text, image, and audio on 16 GB — and **GLM-5.2** (June 13) — Zhipu's MIT-licensed coding powerhouse scoring 62.1% on SWE-bench Pro. Meanwhile, Ollama has jumped from v0.24 to **v0.30.11** in six weeks, Apple's **M5 Pro/Max** chips are shipping with 614 GB/s memory bandwidth, and the **RTX 5080** (16 GB, ~$999) has become the new value king for local inference.

For a consumer with 16–32 GB of VRAM, the practical choices are now: **Gemma 4 12B** (best 16 GB multimodal), **Qwen 3.6-27B** (best coding at 24 GB), and **Gemma 4 31B** (best maths/reasoning at 24 GB). The Qwen 3.6-35B-A3B MoE achieves 120 tok/s on an RTX 4090 — the fastest consumer-hardware model of this quality. Apache 2.0 is the default licence.

## Model Release Timeline

All models relevant to local consumer hardware, ordered by release date:

| Date | Model | Lab | Params (total / active) | Architecture | Licence |
|------|-------|-----|------------------------|--------------|---------|
| 2024-09-19 | Qwen 2.5-Coder (7B/14B/32B) | Alibaba | 7B–32B | Dense | Apache 2.0 |
| 2024-12-26 | DeepSeek V3 | DeepSeek | 671B / 37B | MoE | DeepSeek Licence |
| 2025-01-20 | DeepSeek R1 | DeepSeek | 671B / 37B | MoE | MIT |
| 2025-01-20 | DeepSeek R1-Distill (1.5B–70B) | DeepSeek | 1.5B–70B | Dense | MIT |
| 2025-04-05 | Llama 4 Scout | Meta | 109B / 17B | MoE (16 experts) | Llama 4 Community |
| 2025-04-05 | Llama 4 Maverick | Meta | 400B / 17B | MoE (128 experts) | Llama 4 Community |
| 2025-04-29 | Qwen 3 (0.6B–235B) | Alibaba | 0.6B–235B | Dense + MoE | Apache 2.0 |
| 2025-07 | Qwen 3-2507 (235B-A22B/30B-A3B/4B) | Alibaba | Various | Dense + MoE | Apache 2.0 |
| 2025-09 | Phi-4-reasoning-plus | Microsoft | 14.7B | Dense | MIT |
| 2025-12-02 | Mistral Large 3 | Mistral AI | 675B / 41B | MoE | Apache 2.0 |
| 2026-01 | Falcon H1R 7B | TII | 7B | Hybrid (Mamba+Transformer) | Falcon LLM 1.0 |
| 2026-02-11 | GLM-5 | Zhipu AI | 744B / ~40B | MoE | MIT |
| 2026-02-16 | Qwen 3.5 (397B-A17B, 122B-A10B, 35B-A3B, 27B) | Alibaba | Various | MoE + Dense | Apache 2.0 |
| 2026-03-26 | Voxtral TTS | Mistral AI | 4B | Dense (TTS) | CC-BY-NC |
| 2026-03-30 | Qwen 3.5-Omni | Alibaba | — | MoE | Apache 2.0 |
| 2026-03-31 | Gemma 4 (E2B/E4B/26B-A4B/31B) | Google DeepMind | 2.3B–31B | Dense + MoE | Apache 2.0 |
| 2026-04-08 | GLM-5.1 | Zhipu AI | ~754B / ~40B | MoE | MIT |
| 2026-04-16 | Gemma 4 MTP variants | Google DeepMind | 2.3B–31B | Dense + MoE | Apache 2.0 |
| 2026-04-16 | Qwen 3.6-35B-A3B | Alibaba | 35B / 3B | MoE (256 experts, 8+1 active) | Apache 2.0 |
| 2026-04-20 | Kimi K2.6 | Moonshot AI | ~1T / 32B | MoE | Modified MIT |
| 2026-04-22 | Qwen 3.6-27B (dense) | Alibaba | 27B | Dense | Apache 2.0 |
| 2026-04-24 | DeepSeek V4 (Pro + Flash) | DeepSeek | 1.6T / 49B (Pro) | MoE | MIT |
| 2026-04-29 | Mistral Medium 3.5 | Mistral AI | 128B | Dense | Modified MIT |
| 2026-06-03 | **Gemma 4 12B Unified** | Google DeepMind | 11.95B | Dense (encoder-free) | Apache 2.0 |
| 2026-06-08 | Phi-5 + MAI models | Microsoft | — | — | MIT |
| 2026-06-13 | **GLM-5.2** | Zhipu AI | ~753B / 40B | MoE | MIT |
| 2026-06 | Llama 4.5 (Scout/Maverick refresh) | Meta | ~109B–400B | MoE | Llama Community |
| 2026-06 | Magistral-Small-2506 | Mistral AI | — | — | Apache 2.0 |

## The Landscape: Model Families

### Gemma 4 (Google DeepMind)

Released progressively from March 31 to June 3, 2026, all under **Apache 2.0**. Five sizes:

| Variant | Parameters | Architecture | Context | Modalities | Notes |
|---------|-----------|--------------|---------|------------|-------|
| E2B | 2.3B eff / 5.1B total | Dense | 128K | Text/image/audio | Edge/mobile |
| E4B | 4.5B eff / 8B total | Dense | 128K | Text/image/audio | Edge |
| **12B Unified** | 11.95B | Dense (encoder-free) | 256K | Text/image/audio | New sweet spot for 16 GB |
| 26B-A4B | 26B total / 3.8B active | MoE | 256K | Text/image/video | Efficient inference |
| 31B | 31B | Dense | 256K | Text/image/video | Flagship reasoning/maths |

**Gemma 4 12B** (June 3, 2026) is the standout addition. Its encoder-free unified architecture projects raw image patches and audio waveforms directly into the LLM embedding space through lightweight linear layers (~35M vision module), eliminating the latency and memory overhead of separate encoders. It scores 77.2% MMLU-Pro and 77.5% AIME 2026 — nearly matching the 26B MoE at half the memory.

Multi-Token Prediction (MTP) variants for E2B, E4B, 31B, and 26B-A4B were released April 16, enabling speculative decoding for ~2x generation speed.

### Qwen 3.5 / 3.6 (Alibaba)

The most active model family in 2026, with multiple releases under **Apache 2.0**:

- **Qwen 3.5** (Feb 16, 2026): Flagship at 397B total / 17B active (MoE), plus 122B-A10B, 35B-A3B, and 27B. Scores 88.4% GPQA Diamond, 87.8% MMLU-Pro.
- **Qwen 3.6-35B-A3B** (Apr 16, 2026): 35B total / 3B active MoE (256 experts, 8+1 active), 262K native context (extendable to 1M). Natively multimodal (vision-language). Scores 73.4% SWE-bench, 92.7% AIME 2026. Achieves ~120 tok/s on RTX 4090.
- **Qwen 3.6-27B** (Apr 22, 2026): Dense 27B, 262K native context (extendable to 1M). Natively multimodal (vision-language, early fusion training). Surpasses the 397B flagship on coding — 77.2% SWE-bench Verified, 86.2% MMLU-Pro, 87.8% GPQA Diamond, 94.1% AIME 2026, 83.9% LiveCodeBench v6.
- **Qwen 3.5-Omni** (Mar 30, 2026): Only open model with real-time streaming speech output.
- **Qwen 3.6-Max-Preview** (closed): 60.6% SWE-bench Pro, 92.4% GPQA Diamond — competitive with GPT-5.5 on coding.

The Qwen 3 base family (released April 2025, updated July 2025 as "Qwen3-2507") spans 0.6B to 235B-A22B, all with hybrid thinking mode (/think toggles chain-of-thought on/off).

**Note**: Qwen 3.6 does NOT support Fill-in-the-Middle (FIM). For tab-completion in IDEs, use Qwen 2.5-Coder or the forthcoming Qwen3-Coder series.

### Gemma 4 12B Deep Dive

| Spec | Value |
|------|-------|
| Parameters | 11.95B (dense) |
| Architecture | Decoder-only transformer, encoder-free unified |
| Context | 256K tokens |
| Modalities | Text, image, audio (native); video via frame sampling |
| Vision module | ~35M params (lightweight linear projection) |
| Thinking mode | Configurable (on/off) |
| Licence | Apache 2.0 |
| Release | June 3, 2026 |
| VRAM (Q4) | ~7–8 GB |
| Target hardware | 16 GB laptops and GPUs |

**Benchmarks** (official, from Google model card):

| Benchmark | Score |
|-----------|-------|
| MMLU-Pro | 77.2% |
| AIME 2026 | 77.5% |
| LiveCodeBench v6 | 72.0% |
| GPQA Diamond | 78.8% |
| Codeforces Elo | 1659 |
| DocVQA | 94.9% |
| InfoVQA | 88.4% |
| MMMU-Pro | 69.1% |
| MATH-Vision | 79.7% |

The 12B bridges the gap between the edge E4B and the 26B MoE: it beats Gemma 3 27B (the previous generation's flagship) on GPQA Diamond, MMLU-Pro, and DocVQA while running on half the VRAM.

### GLM-5 Series (Zhipu AI)

All under **MIT licence**:

- **GLM-5** (Feb 2026): 744B MoE / ~40B active. First frontier model on Huawei Ascend hardware.
- **GLM-5.1** (Apr 8, 2026): ~754B / ~40B active. Previously the highest Arena Elo among open models.
- **GLM-5.2** (Jun 13, 2026): ~753B / 40B active, 1M context. Strongest open-source model on coding benchmarks: 62.1% SWE-bench Pro, 81.0 Terminal-Bench 2.1. Improved MTP speculative decoding (+20% acceptance length). Requires ~239 GB at 2-bit quant (Mac Studio with M3 Ultra 256 GB, or 4× RTX 3090).

### Llama 4 / 4.5 (Meta)

Under the **Llama 4 Community Licence** (free below 700M MAU):

- **Scout**: 109B total / 17B active, 16 experts, 10M context
- **Maverick**: 400B total / 17B active, 128 experts, 1M context
- **Llama 4.5** (early June 2026): Mid-cycle refresh with improved agentic tool-use stability [unverified — secondary only]

Scout needs ~55 GB at Q4 (multi-GPU or Mac Studio); Maverick needs ~200 GB (datacenter only). The restrictive licence and hardware requirements make Llama 4 a poor choice for most local users.

### DeepSeek V4 (DeepSeek)

Released April 2026 under **MIT licence**:

- **V4-Pro**: 1.6T / 49B active — strongest open-source coder (80.6% SWE-bench, ~93% LiveCodeBench). Datacenter only.
- **V4-Flash**: 284B / 13B active — 79% SWE-bench. Single H100 but not consumer GPUs.
- **V4.1**: Minor update appearing on leaderboards (June 2026); leads the open-weight Arena Elo slot.

### Other Notable Models

| Model | Lab | Params | Licence | Notes |
|-------|-----|--------|---------|-------|
| Phi-4-reasoning-plus | Microsoft | 14.7B dense | MIT | Best maths per parameter (81.3% AIME 2024) |
| Phi-5 | Microsoft | — | MIT | Released June 8, 2026 alongside 7 MAI models |
| Mistral Medium 3.5 | Mistral AI | 128B dense | Modified MIT | Best non-MoE open model (77.6% SWE-bench) |
| Mistral Small 4 | Mistral AI | 119B MoE | — | Recent MoE offering |
| Magistral-Small-2506 | Mistral AI | Compact | Apache 2.0 | Compact reasoning, June 2026 |
| Kimi K2.6 | Moonshot AI | ~1T / 32B | Modified MIT | Top-tier open-weight; competitive on coding/reasoning |
| MiMo-V2.5-Pro | Xiaomi | 1.02T / 42B | — | Ties Kimi K2.6 on Intelligence Index |
| Falcon H1R 7B | TII | 7B hybrid | Falcon LLM 1.0 | Mamba/Transformer hybrid |

## Best Models by VRAM Tier

| VRAM | Best Models | Notes |
|------|------------|-------|
| **8 GB** | Gemma 4 E4B, Qwen 3 4B (thinking), Phi-4-reasoning-plus Q3 | Thinking mode on Qwen 3 4B scores 73.8% AIME |
| **12 GB** | Phi-4-reasoning-plus Q4, Qwen 3 8B, Gemma 4 12B Q3 | Best maths on 12 GB: Phi-4 (81.3% AIME) |
| **16 GB** | **Gemma 4 12B Q4**, Qwen 3.6-35B-A3B Q4 (with offload), Gemma 4 26B-A4B | Gemma 4 12B: 77.2% MMLU-Pro, multimodal, ~21 tok/s on RTX 4060 |
| **24 GB** (RTX 4090/5080) | Qwen 3.6-27B Q4, Gemma 4 31B Q4, Qwen 3.6-35B-A3B Q4 | 35B-A3B MoE: ~120 tok/s on RTX 4090 |
| **32 GB** (RTX 5090) | Qwen 3.6-27B Q5/Q8, Gemma 4 31B Q5 | Higher quant = better reasoning accuracy |
| **48–64 GB** (Mac Studio M4 Max) | Qwen 3.5 (397B-A17B) Q4, Llama 4 Scout Q3 | MoE flagships become viable |
| **128 GB** (M5 Max / M3 Ultra) | 70B models Q4–Q8, GLM-5.2 IQ2 (with M3 Ultra 256 GB) | M5 Max 128 GB at 614 GB/s |

## Best Models by Use-Case

### Coding

| Tier | Model | Score | VRAM |
|------|-------|-------|------|
| Best (24 GB) | Qwen 3.6-27B dense | 77.2% SWE-bench Verified | ~17 GB Q4 |
| Best speed (24 GB) | Qwen 3.6-35B-A3B MoE | 73.4% SWE-bench, ~120 tok/s | ~22 GB Q4 |
| Best (16 GB) | Gemma 4 12B | 72% LiveCodeBench v6 | ~7–8 GB Q4 |
| Best autocomplete/FIM | Qwen 2.5-Coder 32B | 92.7% HumanEval | ~20 GB Q4 |
| Best (8 GB) | Qwen 2.5-Coder 7B | 88.4% HumanEval | ~5 GB Q4 |

Note: Qwen 3.6 does NOT support Fill-in-the-Middle (FIM). For tab-completion in IDEs, use Qwen 2.5-Coder.

### Reasoning and Maths

| Model | AIME 2026 | GPQA Diamond | VRAM |
|-------|-----------|--------------|------|
| Qwen 3.6-27B (dense) | 94.1% | 87.8% | ~17 GB Q4 |
| Gemma 4 31B (dense) | 89.2% | 84.3% | ~20 GB Q4 |
| Gemma 4 12B | 77.5% | 78.8% | ~7–8 GB Q4 |
| Phi-4-reasoning-plus (14.7B) | 81.3% (2024) | 56.1% | ~10 GB Q4 |
| Qwen 3 4B (/think mode) | 73.8% (2024) | — | ~2.5 GB Q4 |

### Vision and Multimodal

| Model | Key Benchmark | Best For | VRAM |
|-------|--------------|----------|------|
| Gemma 4 31B | MMMU-Pro 76.9% | Image/video understanding | ~20 GB Q4 |
| **Gemma 4 12B** | DocVQA 94.9%, MMMU-Pro 69.1% | Best 16 GB multimodal (text/image/audio) | ~7–8 GB Q4 |
| Gemma 4 26B-A4B | MMMU-Pro ~74% | Fast multimodal MoE | ~18 GB Q4 |
| Qwen 3.6-27B (native vision) | — | General vision + text | ~17 GB Q4 |
| PaddleOCR-VL 0.9B | 92.6% OmniDocBench | Document OCR (CPU) | CPU only |

Gemma 4 12B is the first model to handle text, image, AND audio at 16 GB without separate encoders.

### Embeddings (for RAG)

| Model | nDCG@10 | Dimensions | Licence |
|-------|---------|-----------|---------|
| Qwen3 Embedding 8B | 0.818 | 4096 | Apache 2.0 |
| Qwen3 Embedding 4B | 0.802 | 2560 | Apache 2.0 |
| BAAI/bge-m3 (568M) | 0.753 | 1024 | MIT |
| Nomic Embed Text (137M) | — | 768 | Apache 2.0 |

### General Chat/Assistant

For balanced quality, speed, and personality:
- **24 GB**: Qwen 3.6-27B or Gemma 4 31B (both Apache 2.0)
- **16 GB**: **Gemma 4 12B** (multimodal) or Gemma 4 26B-A4B (MoE, text/image/video)
- **8 GB**: Qwen 3 8B or Gemma 4 E4B

## Benchmark Comparison

Key benchmarks (June 2026). HumanEval and MMLU are saturated and no longer differentiate models:

| Model | MMLU-Pro | SWE-bench Verified | LiveCodeBench v6 | GPQA Diamond | AIME 2026 |
|-------|----------|-------------------|------------------|--------------|-----------|
| Qwen 3.6-27B (dense) | 86.2% | 77.2% | 83.9 | 87.8% | 94.1% |
| Gemma 4 31B (dense) | 85.2% | ~68% | 80.0 | 84.3% | 89.2% |
| **Gemma 4 12B** (dense) | 77.2% | — | 72.0 | 78.8% | 77.5% |
| Qwen 3.6-35B-A3B (MoE) | ~84% | 73.4% | ~80 | 86.0% | 92.7% |
| Qwen 3.5-27B (dense) | ~86% | 72.4% | 80.7 | 85.5% | ~85% |
| Phi-4-reasoning-plus (14B) | ~74% | — | — | 56.1% | 81.3% |
| GLM-5.2 (753B MoE, datacenter) | — | — | — | — | — |

**Summary**: Qwen 3.6-27B leads on coding (SWE-bench 77.2%) and reasoning (AIME 94.1%). Gemma 4 31B leads on multimodal and vision. Gemma 4 12B delivers ~90% of the 31B's reasoning quality at half the VRAM. GLM-5.2 leads on SWE-bench Pro (62.1%) but requires datacenter hardware.

## Hardware Requirements

### VRAM Formula

```
VRAM ≈ (Parameters × Bits per Weight) ÷ 8 + KV Cache Overhead
```

KV cache adds ~25% at 8K context, ~100% at 32K context. Key optimisations in 2026: PagedAttention (vLLM), NVFP4 KV cache quantisation, CPU KV cache offloading, and DeepSeek MLA (multi-latent attention).

### GPU Recommendations

| GPU | VRAM | Price (approx.) | Best Models | tok/s (typical) |
|-----|------|-----------------|-------------|-----------------|
| RTX 4060 Ti | 16 GB | ~$450 | Gemma 4 12B Q4, 14B Q4 | ~40–60 |
| RTX 5080 | 16 GB GDDR7 | ~$999 | Gemma 4 12B Q4, Qwen 3.6-35B-A3B Q3 | ~60–90 |
| RTX 4090 | 24 GB | ~$1,800 | Qwen 3.6-27B Q4, Gemma 4 31B Q4 | ~30–40 (27B dense); ~120 (35B MoE) |
| RTX 5090 | 32 GB GDDR7 | ~$1,999 | 27–32B Q8, 70B Q3 | ~200 |

### Apple Silicon

| Chip | Unified Memory | Bandwidth | Best Models | tok/s |
|------|---------------|-----------|-------------|-------|
| M4 (16–32 GB) | 16–32 GB | 120 GB/s | Gemma 4 12B Q4, 7B–13B Q8 | ~15–30 |
| M4 Pro (24–64 GB) | 24–64 GB | 273 GB/s | 27–30B Q4–Q5 | ~20–30 |
| M4 Max (64–128 GB) | 64–128 GB | 546 GB/s | 70B Q4, MoE flagships | ~15–25 |
| **M5 Pro** (36–64 GB) | 36–64 GB | 307 GB/s | 30B Q4–Q5 | ~25–35 |
| **M5 Max** (64–128 GB) | 64–128 GB | 614 GB/s | 70B Q4–Q8, GLM-5.2 IQ2 (256 GB only) | ~30–40+ |
| M3 Ultra (192–256 GB) | 192–256 GB | 819 GB/s | Anything including GLM-5.2 IQ2 | ~25–35 (70B) |

M5 Pro/Max use Fusion Architecture (two 3nm dies), shipped March 2026. M5 Max delivers over 4x peak GPU compute for AI vs M4 Max. Apple explicitly positions it for "higher token generation for LLMs". M5 Ultra expected late 2026. No M4 Ultra was produced — Apple skipped to M3 Ultra for Mac Studio 2025.

## Quantisation Formats

### Format Comparison

| Format | Hardware | Quality at 4-bit | Speed | Best For |
|--------|----------|-----------------|-------|----------|
| **GGUF** | CPU, GPU, Metal | Baseline | Good | Default choice, cross-platform |
| **NVFP4** | MLX (Apple Silicon) | +0.5% vs Q4_K_M | Fastest on Mac | Apple Silicon (via Ollama MLX engine) |
| **AWQ** | CUDA only | +0.5–1.0% vs GPTQ | Fast | Best quality at 4-bit on NVIDIA |
| **GPTQ** | CUDA only | Slightly behind AWQ | Fast | Legacy, mature ecosystem |
| **EXL2** | CUDA only | Variable (2.0–8.0 bpw) | Fastest single-user | Precise VRAM targeting |
| **MLX** | Apple Silicon only | Comparable to GGUF Q4 | Fastest on Mac | Apple Silicon native |

**NVFP4** (new in 2026): Ollama's MLX engine introduced NVFP4 quantisation which halves quality loss vs Q4_K_M. Gemma 4 12B perplexity: BF16 17.54, NVFP4 17.95, Q4_K_M 18.36 (lower is better).

### Recommended Quantisation Levels

| Use Case | Recommended Quant | Quality Retention |
|----------|------------------|-------------------|
| General chat | Q4_K_M (GGUF) or NVFP4 (MLX) | 90–95% of FP16 |
| Coding / agent tasks | Q4_K_M minimum, Q5_K_M preferred | Reasoning degrades 3x faster than perplexity |
| Mission-critical reasoning | Q5_K_M or Q8 | >97% of FP16 |
| VRAM-constrained (<8 GB) | Q3_K_M or IQ2 (Unsloth) | Acceptable for chat only, not reasoning |
| Small models (1–3B) | Q8_0 or FP16 | Quant overhead minimal at small sizes |

**Unsloth Dynamic v2.0** is the community standard for high-quality GGUF quantisations, using importance-matrix layer-wise quantisation that outperforms standard methods on MMLU, Aider Polyglot, and KL Divergence. Compatible with llama.cpp, Ollama, and all GGUF engines.

**TurboQuant** (ICLR 2026): CPU-optimised quantisation merged in llama.cpp forks — TQ3 gives 4.9x compression vs FP16, TQ4 gives 3.8x, with +30–50% throughput when combined with MTP.

## Inference Frameworks

### Framework Comparison (June 2026)

| Framework | Best For | Platform | Key Feature |
|-----------|----------|----------|-------------|
| **Ollama** (v0.30.11) | Quick setup, single-user | All (MLX on Mac) | One-command model pull, NVFP4, MTP speculative decoding |
| **LM Studio** (v0.4.14) | GUI users, team sharing | All (MLX on Mac) | Visual management, LM Link remote, MTP stable |
| **llama.cpp** | Maximum portability | All | CPU+GPU split, MTP for Qwen 3.6, Vulkan backend |
| **MLX** (v0.31.x) | Apple Silicon native | macOS only | LoRA/QLoRA on-device, M5 Neural Accelerators |
| **vLLM** (v0.21.0) | Multi-user API server | Linux (NVIDIA/AMD) | PagedAttention, TOKENSPEED_MLA, DeepSeek V4 support |
| **SGLang** | RAG-heavy production | Linux (NVIDIA) | 400K+ GPUs, trillions of tokens daily, LMSYS-hosted |

### Ollama (v0.30.11, June 2026)

Ollama has gone from v0.24 (May) to v0.30.11 (June 26, 2026) — an extraordinary release cadence:

- **v0.23.1**: Gemma 4 MTP speculative decoding on Mac via MLX (2x speed on 31B coding)
- **v0.24.0**: Codex App support, reworked MLX sampler
- **v0.25–0.30**: Rapid iteration; NVFP4 quantisation support, expanded model library
- MLX backend on Apple Silicon since v0.19.0 (March 2026): 1.6x faster prefill, 2x decode
- OpenAI-compatible API on port 11434
- Supports Gemma 4 (including 12B), Qwen 3.5/3.6, DeepSeek, GLM-5, and 100+ models

### llama.cpp (May–June 2026)

Building on the April 2026 rewrite:
- **MTP speculative decoding** merged (PR #22673) for Qwen 3.6 — ~2x throughput on 27B dense
- MoE models show no net MTP speedup at batch=1 on consumer GPUs
- Head-contiguous KV cache layout for coalesced reads
- Vulkan backend competitive — can beat CUDA by ~40% on some workloads
- Unified Metal/CUDA/ROCm/Vulkan dispatch
- GGUF remains the canonical cross-platform format

### MLX (v0.31.x)

- M5 Neural Accelerator support (macOS 26.2+): up to 4x faster time-to-first-token
- NVFP4 quantisation: halves quality loss vs Q4_K_M with comparable speed
- Gemma 4 12B on M5 Max via Ollama MLX engine: **55 tok/s** (NVFP4), **46 tok/s** (Q4_K_M)
- Fastest generation for models under 14B (20–87% over llama.cpp)
- Native LoRA/QLoRA fine-tuning on-device
- Slower prefill at long contexts (8K+) — llama.cpp wins there

### LM Studio (v0.4.14)

- mlx-engine v1.8.1
- MTP speculative decoding promoted to stable: 1.5–3x throughput improvement
- LM Link: Tailscale-backed remote access for team sharing
- MCP Host functionality, OpenAI + Anthropic-compatible API on localhost:1234
- Parallel vision predictions

## MoE Models on Consumer Hardware

**Critical rule**: MoE models need VRAM proportional to TOTAL parameters, not active parameters. Speed benefits are real; memory savings are not.

| Model | Total / Active | VRAM at Q4 | Consumer Viable? |
|-------|---------------|-----------|-----------------|
| Gemma 4 26B-A4B | 26B / 3.8B | ~16 GB | Yes (RTX 5080/4080+) |
| Qwen 3.6-35B-A3B | 35B / 3B | ~22 GB | Yes (RTX 4090, ~120 tok/s) |
| Llama 4 Scout | 109B / 17B | ~55 GB | No (multi-GPU/Mac Studio) |
| GLM-5.2 | 753B / 40B | ~239 GB (IQ2) | Extreme (M3 Ultra 256 GB only) |
| Llama 4 Maverick | 400B / 17B | ~200 GB | No (datacenter only) |

**For consumer hardware (<48 GB VRAM), dense models always outperform MoE at comparable VRAM usage.** The only consumer-friendly MoE exceptions are Gemma 4 26B-A4B and Qwen 3.6-35B-A3B. The 35B-A3B's 120 tok/s on an RTX 4090 makes it exceptionally fast for agentic workflows — 3–4x faster generation than the comparable-quality 27B dense model.

## Apple Silicon and MLX

### The 2026 Mac Stack

Both Ollama (v0.30.11) and LM Studio (v0.4.14) use MLX as their Apple Silicon backend. MLX is the only framework targeting M5 Neural Accelerators (requires macOS 26.2+).

**MLX advantages**:
- Fastest generation for models under 14B (20–87% over llama.cpp)
- NVFP4 quantisation: best quality/speed ratio on Apple Silicon
- Gemma 4 12B: 55 tok/s NVFP4, 46 tok/s Q4_K_M (M5 Max)
- Native LoRA/QLoRA fine-tuning on-device
- Swift bindings for iOS development
- M5 Neural Accelerators: up to 4x faster TTFT

**MLX limitations**:
- Mac-only (not portable)
- Slower prefill at long contexts (8K+) — llama.cpp wins there
- Smaller pre-quantised model ecosystem than GGUF
- No CPU+GPU split for oversized models (llama.cpp can do this)

### Recommended Mac Setups

- **MacBook Air/Pro M4 (16–24 GB)**: Gemma 4 12B Q4 or NVFP4 — best multimodal on 16 GB
- **MacBook Pro M5 Pro (36–64 GB)**: Qwen 3.6-27B Q4–Q5 or Gemma 4 31B — full coding/reasoning
- **Mac Mini M4 Pro (36 GB)**: 30B models at Q5 — excellent home server
- **MacBook Pro M5 Max (64–128 GB)**: 70B models, multi-model serving, MoE flagships
- **Mac Studio M4 Max (128 GB)**: Run 70B at Q8 or Qwen 3.5 397B-A17B Q4
- **Mac Studio M3 Ultra (256 GB)**: Run anything including GLM-5.2 at IQ2

## AMD GPUs for Local Inference

ROCm 7.2 (January 2026) officially supports RDNA 4 consumer GPUs (RX 9070 XT, 9070, 9060 XT, 9060) and RDNA 3 (7900 XTX). llama.cpp, Ollama, and vLLM all work on AMD consumer GPUs [unverified — secondary only].

**Current state**:
- 30–40% performance gap vs NVIDIA due to lack of dedicated tensor cores
- RX 9070 XT: 16 GB VRAM on 256-bit bus — same tier as RTX 5080 but ~30% slower for LLM inference
- RX 7900 XTX: 24 GB VRAM — still the best AMD card for local LLMs (matches RTX 4090 VRAM)
- ROCm is Linux-only for serious ML work; Windows support is limited
- Flash Attention requires Triton backend for consumer RDNA
- PyTorch 2.10 has first-class ROCm 7.1 support

**Verdict**: AMD is viable for local inference but requires Linux and tolerance for a 30–40% perf penalty. The 7900 XTX (24 GB, often $650–800 used) remains the best value if you're AMD-committed.

## What Changed Since May 2026

### June 2026
- **Gemma 4 12B Unified** (Jun 3) — encoder-free multimodal at 16 GB; beats Gemma 3 27B
- **GLM-5.2** (Jun 13) — strongest open coder (62.1% SWE-bench Pro, 81.0 Terminal-Bench)
- **Phi-5 + 7 MAI models** (Jun 8) — Microsoft's latest; details emerging
- **Llama 4.5** (early Jun) — mid-cycle refresh of Scout/Maverick [unverified — secondary only]
- **Magistral-Small-2506** — Mistral's compact reasoning model
- **Ollama v0.25–v0.30.11** — rapid iteration; NVFP4, expanded model support
- **Ollama MLX Performance blog** (Jun 11) — NVFP4 benchmarks for Gemma 4 12B
- **25+ open-weight models** released in one week across LLMs, image, audio, video, and 3D

### Key May 2026 events (from v1.0)
- Ollama 0.23–0.24: Codex App support, Gemma 4 MTP on MLX
- vLLM v0.21: DeepSeek V4 on Blackwell, TOKENSPEED_MLA backend
- llama.cpp MTP for Qwen 3.6 (~2x generation on 27B dense)
- MLX 0.31: M5 Neural Accelerator support
- LM Studio 0.4.14: MTP speculative decoding promoted to stable

## Licensing

| Family | Licence | Commercial Use | Key Restrictions |
|--------|---------|---------------|-----------------|
| Gemma 4 | Apache 2.0 | Unrestricted | None |
| Qwen 3.5/3.6 | Apache 2.0 | Unrestricted | None |
| DeepSeek V4 | MIT | Unrestricted | None |
| Phi-4/5 | MIT | Unrestricted | None |
| GLM-5/5.1/5.2 | MIT | Unrestricted | None |
| Mistral Large 3 | Apache 2.0 | Unrestricted | None |
| Mistral Medium 3.5 | Modified MIT | Unrestricted | Minor attribution |
| Magistral-Small-2506 | Apache 2.0 | Unrestricted | None |
| Llama 4/4.5 | Llama Community | Conditional | 700M MAU cap, attribution, acceptable use policy |
| Kimi K2.6 | Modified MIT | Unrestricted | Minor terms |
| Falcon H1R | Falcon LLM 1.0 | Unrestricted | No-litigate clause |

Apache 2.0 and MIT dominate. Meta's Llama licence is now the outlier, not the norm.

## Recommendations

### If you have an RTX 4090 (24 GB) or RTX 5080 (16 GB)

**RTX 4090 (24 GB)**:
- **General use**: Qwen 3.6-27B Q4_K_M (~17 GB) or Gemma 4 31B Q4_K_M (~20 GB)
- **Fastest generation**: Qwen 3.6-35B-A3B Q4_K_M (~22 GB) — ~120 tok/s
- **Coding**: Qwen 3.6-27B for agentic coding (77.2% SWE-bench); Qwen 2.5-Coder 32B for FIM/autocomplete
- **Maths/Reasoning**: Gemma 4 31B (89.2% AIME) or Qwen 3.6-27B (94.1% AIME)

**RTX 5080 (16 GB)**:
- **General use**: Gemma 4 12B Q4 (~7–8 GB) — leaves room for KV cache at long context
- **Multimodal**: Gemma 4 12B — handles text, images, AND audio natively
- **Value play**: RTX 5080 at ~$999 is the new sweet spot; same VRAM as 4060 Ti but much faster

### If you have a Mac with 32–128 GB unified memory

Same models as GPU users — unified memory acts as VRAM. Use **Ollama** (v0.30.11, MLX backend) or **LM Studio** (v0.4.14) for the best experience.

- **16–24 GB**: Gemma 4 12B NVFP4 (55 tok/s on M5 Max)
- **36–64 GB**: Qwen 3.6-27B Q4–Q5 or Gemma 4 31B — full coding/reasoning
- **128 GB (M5 Max)**: 70B Q4–Q8, multi-model serving, MoE flagships
- **256 GB (M3 Ultra)**: Run GLM-5.2 at IQ2 — the strongest open coder locally

### If you have 16 GB (RTX 5080/4060 Ti or Mac base)

**The new sweet spot**: Gemma 4 12B Q4 (~7–8 GB) is the clear winner:
- 77.2% MMLU-Pro, 77.5% AIME, 72% LiveCodeBench
- Multimodal: text + image + audio in one model
- ~21 tok/s on RTX 4060, ~55 tok/s on M5 Max (NVFP4)
- Leaves VRAM headroom for long contexts

**Alternative**: Gemma 4 26B-A4B Q4 (~16 GB) if you need more capability and can sacrifice headroom.

### If you have 8 GB

- **General**: Qwen 3 8B or Gemma 4 E4B
- **Coding**: Qwen 2.5-Coder 7B (88.4% HumanEval)
- **Reasoning**: Qwen 3 4B with /think mode (73.8% AIME at 2.5 GB)

### Framework choice

- **Just getting started**: Ollama (`ollama run gemma4:12b` or `ollama run qwen3.6:27b`)
- **Want a GUI**: LM Studio
- **Need maximum control**: llama.cpp directly
- **Apple Silicon fine-tuning**: MLX
- **Multi-user API server**: vLLM
- **RAG-heavy production**: SGLang

## References

1. [Google AI — Gemma 4 Releases](https://ai.google.dev/gemma/docs/releases) — Official release timeline (confirms June 3, 2026 for 12B)
2. [Google AI — Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4) — Official benchmarks for all variants
3. [Google Blog — Introducing Gemma 4 12B](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/) — Official announcement
4. [Hugging Face — google/gemma-4-12B](https://huggingface.co/google/gemma-4-12B) — Official model card
5. [GitHub — QwenLM/Qwen3.6](https://github.com/QwenLM/Qwen3.6) — Official Qwen 3.6 repository
6. [Hugging Face — Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) — Official model card (262K context, benchmarks)
7. [Qwen Blog — Qwen3.6-27B](https://qwen.ai/blog?id=qwen3.6-27b) — Official release announcement
8. [Qwen Blog — Qwen3.6-35B-A3B](https://qwen.ai/blog?id=qwen3.6-35b-a3b) — Official release announcement
9. [GitHub — zai-org/GLM-5](https://github.com/zai-org/GLM-5) — GLM-5.2 official repository (Jun 13, 2026)
10. [Ollama Blog — MLX Performance](https://ollama.com/blog/mlx-performance) — NVFP4 benchmarks for Gemma 4 12B (Jun 11, 2026)
11. [GitHub — Ollama Releases](https://github.com/ollama/ollama/releases) — v0.30.11 (Jun 26, 2026)
12. [LM Studio Changelog](https://lmstudio.ai/changelog) — v0.4.14
13. [Apple Newsroom — M5 Pro and M5 Max](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/) — Official M5 specs
14. [NVIDIA — RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) — 32 GB GDDR7, $1,999 MSRP
15. [NVIDIA — RTX 5080](https://www.nvidia.com/en-au/geforce/graphics-cards/50-series/rtx-5080/) — 16 GB GDDR7, Blackwell
16. [Unsloth — Dynamic v2.0 GGUFs](https://unsloth.ai/blog/dynamic-v2) — Importance-matrix GGUF quantisation
17. [Codersera — Local AI Runtimes May 2026](https://codersera.com/blog/local-ai-runtimes-may-2026-update/) — Ollama/vLLM/llama.cpp/MLX changelog
18. [GitHub — SGLang](https://github.com/sgl-project/sglang) — 400K+ GPU deployment scale
19. [InsiderLLM — Qwen 3.6 Local AI Guide](https://insiderllm.com/guides/qwen-3-6-local-ai-guide/) — Benchmarks and VRAM tables
20. [TechStartups — Gemma 4 12B Launch](https://techstartups.com/2026/06/03/google-deepmind-launches-gemma-4-12b-bringing-frontier-ai-model-to-everyday-laptops/) — Benchmark summary
21. [Lushbinary — Gemma 4 12B Developer Guide](https://lushbinary.com/blog/gemma-4-12b-developer-guide-benchmarks-multimodal/) — Architecture and benchmark analysis
22. [APXML — Gemma 4 12B Specs](https://apxml.com/models/gemma-4-12b) — Hardware requirements
23. [arXiv:2604.07035 — Gemma 4, Phi-4, and Qwen3 Accuracy-Efficiency Tradeoffs](https://arxiv.org/html/2604.07035v1) — Academic comparison
24. [GitHub — llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) — MTP merge, continuous builds
25. [LMArena Leaderboard](https://arena.ai/leaderboard/text) — Live model rankings
