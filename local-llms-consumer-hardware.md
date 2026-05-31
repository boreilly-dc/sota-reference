# Best Local LLMs for Consumer Hardware (Mid-2026)

| Field | Value |
|-------|-------|
| Created | 2026-05-31 |
| Last Updated | 2026-05-31 |
| Version | 1.0 |

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
- [What Changed in the Last 6 Months](#what-changed-in-the-last-6-months)
- [Licensing](#licensing)
- [Recommendations](#recommendations)
- [References](#references)

## Overview

The local LLM landscape in mid-2026 is unrecognisable from a year ago. The gap between open-weight and proprietary models has compressed to approximately 3 months (per Epoch AI), and six major labs shipped frontier-credible open models in April 2026 alone. MoE (Mixture-of-Experts) is now the dominant architecture at scale, Apache 2.0 has become the standard licence for serious open-weight releases, and inference frameworks have matured dramatically — Ollama now uses MLX on Apple Silicon, llama.cpp received its largest rewrite ever, and quantisation quality has improved to the point where Q4 models retain 90–95% of full-precision quality.

For a consumer with an RTX 4090 (24 GB VRAM) or an Apple Silicon Mac with 32–64 GB unified memory, the practical options are excellent: Gemma 4 31B and Qwen 3.6 27B trade blows as the best all-round local models, while specialised variants dominate coding, reasoning, vision, and embedding tasks.

## Model Release Timeline

All models relevant to local consumer hardware, ordered by release date:

| Date | Model | Lab | Params (total / active) | Architecture | Licence |
|------|-------|-----|------------------------|--------------|---------|
| 2024-07-23 | Llama 3.1 405B | Meta | 405B | Dense | Llama 3.1 Community |
| 2024-09-19 | Qwen 2.5 (0.5B–72B) | Alibaba | 0.5B–72B | Dense | Apache 2.0 |
| 2024-09-19 | Qwen 2.5-Coder (7B/14B/32B) | Alibaba | 7B–32B | Dense | Apache 2.0 |
| 2024-12-26 | DeepSeek V3 | DeepSeek | 671B / 37B | MoE | DeepSeek Licence |
| 2025-01-20 | DeepSeek R1 | DeepSeek | 671B / 37B | MoE | MIT |
| 2025-01-20 | DeepSeek R1-Distill (1.5B–70B) | DeepSeek | 1.5B–70B | Dense | MIT |
| 2025-03-12 | Gemma 3 (1B–27B) | Google DeepMind | 1B–27B | Dense | Gemma Licence |
| 2025-04-05 | Llama 4 Scout | Meta | 109B / 17B | MoE (16 experts) | Llama 4 Community |
| 2025-04-05 | Llama 4 Maverick | Meta | 400B / 17B | MoE (128 experts) | Llama 4 Community |
| 2025-04-29 | Qwen 3 (0.6B–235B) | Alibaba | 0.6B–235B | Dense + MoE | Apache 2.0 |
| 2025-09 | Phi-4-reasoning-plus | Microsoft | 14.7B | Dense | MIT |
| 2025-12-02 | Mistral Large 3 | Mistral AI | 675B / 41B | MoE | Apache 2.0 |
| 2026-01 | Falcon H1R 7B | TII | 7B | Hybrid (Mamba+Transformer) | Falcon LLM 1.0 |
| 2026-02-11 | GLM-5 | Zhipu AI | 744B / ~40B | MoE | MIT |
| 2026-02-16 | Qwen 3.5 (0.8B–397B) | Alibaba | 397B / 17B (flagship) | MoE | Apache 2.0 |
| 2026-03-26 | Voxtral TTS | Mistral AI | 4B | Dense (TTS) | CC-BY-NC |
| 2026-03-30 | Qwen 3.5-Omni | Alibaba | — | MoE | Apache 2.0 |
| 2026-03-31 | Gemma 4 (E2B/E4B/26B-A4B/31B) | Google DeepMind | 2.3B–31B | Dense + MoE | Apache 2.0 |
| 2026-04-02 | Gemma 4 31B (HF release) | Google DeepMind | 31B | Dense | Apache 2.0 |
| 2026-04-08 | GLM-5.1 | Zhipu AI | ~754B / ~40B | MoE | MIT |
| 2026-04-16 | Gemma 4 MTP variants | Google DeepMind | 2.3B–31B | Dense + MoE | Apache 2.0 |
| 2026-04-16 | Qwen 3.6-35B-A3B | Alibaba | 35B / 3B | MoE | Apache 2.0 |
| 2026-04-20 | Kimi K2.6 | Moonshot AI | ~1T / 32B | MoE | Modified MIT |
| 2026-04-22 | Qwen 3.6-27B (dense) | Alibaba | 27B | Dense | Apache 2.0 |
| 2026-04-22 | MiMo-V2.5-Pro | Xiaomi | 1.02T / 42B | MoE | — |
| 2026-04-24 | DeepSeek V4 (Pro + Flash) | DeepSeek | 1.6T / 49B (Pro) | MoE | MIT |
| 2026-04 | Hunyuan Hy3 Preview | Tencent | 295B / 21B | MoE | Tencent Community |
| 2026-04-30 | Mistral Medium 3.5 | Mistral AI | 128B | Dense | Modified MIT |

## The Landscape: Model Families

### Gemma 4 (Google DeepMind)

Released March 31, 2026 under **Apache 2.0** — a significant licence upgrade from Gemma 3's restrictive terms. Four sizes:

| Variant | Parameters | Architecture | Context | Notes |
|---------|-----------|--------------|---------|-------|
| E2B | 2.3B | Dense | 128K | Edge/mobile, multimodal (text/image/audio) |
| E4B | 4.5B | Dense | 128K | Edge, multimodal (text/image/audio) |
| 26B-A4B | 26B total / 3.8B active | MoE | 256K | Efficient inference, multimodal (text/image/video) |
| 31B | 31B | Dense | 256K | Flagship, multimodal (text/image/video) |

Multi-Token Prediction (MTP) variants followed on April 16, 2026, enabling speculative decoding for ~2x generation speed. Arena Elo ~1451 (#4 among open models). Scores 85.2% MMLU-Pro, 89.2% AIME 2026, 80% LiveCodeBench.

### Qwen 3.5 / 3.6 (Alibaba)

The most active model family in 2026, with multiple releases under **Apache 2.0**:

- **Qwen 3.5** (Feb 16, 2026): Flagship at 397B total / 17B active (MoE), 256K context. Scores 88.4% GPQA Diamond, 87.8% MMLU-Pro, 76.4% SWE-bench Verified.
- **Qwen 3.6-35B-A3B** (Apr 16, 2026): 35B total / 3B active MoE, 256K context. Best efficiency model — 73.4% SWE-bench, fits on 16 GB with offloading.
- **Qwen 3.6-27B** (Apr 22, 2026): Dense 27B, 262K context. The new default for 24 GB GPUs — 77.2% SWE-bench Verified, native vision baked in.
- **Qwen 3.5-Omni** (Mar 30, 2026): Only open model with real-time streaming speech output.

The Qwen 3 base family (released 2025) spans 0.6B to 32B dense plus a 235B-A22B MoE, all with hybrid thinking mode (/think toggles chain-of-thought on/off).

### Llama 4 (Meta)

Released April 2025 under the **Llama 4 Community Licence** (free below 700M MAU, requires attribution):

- **Scout**: 109B total / 17B active, 16 experts, 10M token context window
- **Maverick**: 400B total / 17B active, 128 experts, 1M context

Both are MoE with text + image input. Scout needs ~55 GB at Q4 (multi-GPU or Mac Studio); Maverick needs ~200 GB (datacenter only). Maverick scores 80.5% MMLU-Pro but underperforms on coding (43.4% LiveCodeBench v6) despite its size. The restrictive licence and hardware requirements make Llama 4 a poor choice for most local users in 2026.

### Phi-4 (Microsoft)

**Phi-4-reasoning-plus** (14.7B dense, MIT licence, 32K context) is the best math model per parameter: 81.3% AIME 2024, 97.7% MATH-500. Fits on a 12 GB GPU at Q4. Specialised for math/science reasoning — not a general assistant.

### Mistral

- **Mistral Large 3** (Dec 2025): 675B / 41B active MoE, Apache 2.0, 256K context. Underperforms relative to size (~65% SWE-bench, ~78% MMLU-Pro).
- **Mistral Medium 3.5** (Apr 30, 2026): 128B dense, modified MIT, 256K context. Highest-quality non-MoE open model (77.6% SWE-bench Verified). Needs ~70 GB VRAM at Q4.
- **Mistral Small 3.1** (24B): Fits on 16 GB, good balanced general use.

### DeepSeek V4

Released April 2026 under **MIT licence**:

- **V4-Pro**: 1.6T / 49B active — strongest open-source coder (80.6% SWE-bench, ~93% LiveCodeBench). Requires datacenter hardware.
- **V4-Flash**: 284B / 13B active — 79% SWE-bench, $0.14/M input via API. Runs on a single H100 but not consumer GPUs.

### Other Notable Models

| Model | Lab | Params | Licence | Notes |
|-------|-----|--------|---------|-------|
| GLM-5.1 | Zhipu AI | 754B MoE / ~40B active | MIT | Highest Arena Elo among open models (1474) |
| Kimi K2.6 | Moonshot AI | 1T MoE / 32B active | Modified MIT | Leads Intelligence Index (54), ties GPT-5.5 on SWE-Bench Pro |
| MiMo-V2.5-Pro | Xiaomi | 1.02T / 42B active | — | Ties Kimi K2.6 on Intelligence Index |
| Falcon H1R 7B | TII | 7B hybrid | Apache-derived | Mamba/Transformer hybrid, Jan 2026 |
| Hunyuan Hy3 | Tencent | 295B / 21B active | Community | April 2026, conditional commercial use |

## Best Models by VRAM Tier

| VRAM | Best Models | Notes |
|------|------------|-------|
| **8 GB** | Gemma 4 E4B, Qwen 3 4B (thinking), Phi-4 14B Q3 | Thinking mode on Qwen 3 4B scores 73.8% AIME |
| **12 GB** | Phi-4-reasoning-plus Q4, Qwen 3 8B, DeepSeek R1-Distill-7B | Best math on 12 GB: Phi-4 (81.3% AIME) |
| **16 GB** | Qwen 3.6-35B-A3B Q4, Gemma 4 26B-A4B, Mistral Small 3.1 | Qwen 3.6 MoE: 73.4% SWE-bench on 16 GB |
| **24 GB** (RTX 4090) | Qwen 3.6-27B Q4, Gemma 4 31B Q4, DeepSeek R1-Distill-32B | Qwen 3.6-27B: 77.2% SWE-bench, ~25 tok/s |
| **32 GB** (RTX 5090) | Qwen 3.6-27B Q5/Q8, Gemma 4 31B Q5 | Higher quant = better reasoning accuracy |
| **48–64 GB** (Mac Studio) | Qwen 3.5 (397B-A17B) Q4, Llama 4 Scout Q3 | MoE flagships become viable |
| **96–128 GB** (Mac Ultra/M5 Max) | 70B models Q4-Q5, Llama 3.3 70B, Qwen 2.5 72B | First Macs to run 70B comfortably |

## Best Models by Use-Case

### Coding

| Tier | Model | Score | VRAM |
|------|-------|-------|------|
| Best (24 GB) | Qwen 3.6-27B dense | 77.2% SWE-bench Verified | ~17 GB Q4 |
| Best (16 GB) | Qwen 3.6-35B-A3B MoE | 73.4% SWE-bench Verified | ~22 GB Q4 (16 GB with offload) |
| Best autocomplete/FIM | Qwen 2.5-Coder 32B | 92.7% HumanEval | ~20 GB Q4 |
| Best (8 GB) | Qwen 2.5-Coder 7B | 88.4% HumanEval | ~5 GB Q4 |

Note: Qwen 3.x does NOT support Fill-in-the-Middle (FIM). For tab-completion in IDEs, use Qwen 2.5-Coder.

### Reasoning and Maths

| Model | AIME 2024 | MATH-500 | VRAM |
|-------|-----------|----------|------|
| Phi-4-reasoning-plus (14.7B) | 81.3% | 97.7% | ~10 GB Q4 |
| DeepSeek R1-Distill-Qwen-32B | 72.6% | 94.3% | ~18 GB Q4 |
| Qwen 3 32B (/think mode) | 70.0% | 95.2% | ~20 GB Q4 |
| DeepSeek R1-Distill-Qwen-14B | 69.7% | 93.9% | ~6.5 GB Q4 |
| Qwen 3 4B (/think mode) | 73.8% | 91.4% | ~2.5 GB Q4 |

Thinking mode more than doubles math accuracy — Qwen 3 32B drops from 95.2% to 43.6% MATH-500 without it.

### Vision and Multimodal

| Model | MMMU-Pro | Best For | VRAM |
|-------|----------|----------|------|
| Gemma 4 31B | 76.9% | Image/video understanding | ~20 GB Q4 |
| Gemma 4 26B-A4B | 73.8% | Fast multimodal MoE | ~18 GB Q4 |
| Qwen 3.6-27B (native vision) | — | General vision + text | ~17 GB Q4 |
| Qwen3-VL 8B | MathVista 85.8 | Document/chart analysis | ~6 GB Q4 |
| PaddleOCR-VL 0.9B | 92.6% OmniDocBench | Document OCR (CPU) | CPU only |

LLaVA is now legacy — outperformed by Qwen-VL, Gemma, and Phi at every comparable size.

### Embeddings (for RAG)

| Model | nDCG@10 | Dimensions | Latency | Licence |
|-------|---------|-----------|---------|---------|
| Qwen3 Embedding 8B | 0.818 | 4096 | 56ms | Apache 2.0 |
| Qwen3 Embedding 4B | 0.802 | 2560 | 28ms | Apache 2.0 |
| BAAI/bge-m3 (568M) | 0.753 | 1024 | 29ms | MIT |
| Nomic Embed Text (137M) | — | 768 | ~15ms | Apache 2.0 |
| Qwen3 Embedding 0.6B | 0.751 | 1024 | 23ms | Apache 2.0 |

The gap between open-source and proprietary embedding models has effectively closed in 2026.

### General Chat/Assistant

For balanced quality, speed, and personality:
- **24 GB**: Qwen 3.6-27B or Gemma 4 31B (both Apache 2.0)
- **16 GB**: Gemma 4 26B-A4B or Qwen 3.6-35B-A3B
- **8 GB**: Qwen 3 8B or Gemma 4 E4B

## Benchmark Comparison

Key benchmarks in 2026 (HumanEval and MMLU are saturated and no longer differentiate models):

| Model | MMLU-Pro | SWE-bench Verified | LiveCodeBench | GPQA Diamond | AIME |
|-------|----------|-------------------|---------------|--------------|------|
| Qwen 3.6-27B (dense) | ~84% | 77.2% | 83.9 | — | — |
| Gemma 4 31B (dense) | 85.2% | ~68% | 80.0 | 84.3% | 89.2% |
| Qwen 3.5 27B (dense) | ~86% | 72.4% | 80.7 | 85.5% | ~85% |
| Qwen 3.6-35B-A3B (MoE) | ~84% | 73.4% | ~80 | ~82% | 92.7% |
| Gemma 4 26B-A4B (MoE) | — | — | — | — | — |
| Phi-4-reasoning-plus (14B) | ~74% | — | — | 56.1% | 81.3% |
| Llama 4 Maverick (400B MoE) | 80.5% | ~70% | 43.4 | 69.8% | — |

**Summary**: Gemma 4 31B leads on maths (AIME 89.2%) and vision (MMMU-Pro 76.9%). Qwen 3.6-27B leads on coding (SWE-bench 77.2%). Both are within 1–2% on reasoning benchmarks. Llama 4 Maverick lags on coding despite being 13x larger.

## Hardware Requirements

### VRAM Formula

```
VRAM ≈ (Parameters × Bits per Weight) ÷ 8 + KV Cache Overhead
```

KV cache adds ~25% at 8K context, ~100% at 32K context.

### GPU Recommendations

| GPU | VRAM | Price (approx.) | Best Models | tok/s (typical) |
|-----|------|-----------------|-------------|-----------------|
| RTX 4060 Ti | 16 GB | ~$450 | 14B Q4, 8B Q8 | ~50–70 |
| RTX 4070 Ti | 12 GB | ~$600 | 14B Q4 | ~80 |
| RTX 4080 | 16 GB | ~$1,200 | 24–30B Q4 | ~120 |
| RTX 4090 | 24 GB | ~$1,800 | 27–32B Q4-Q5 | ~150 |
| RTX 5090 | 32 GB | ~$2,000 | 32B Q8, 70B Q3 | ~200 |

### Apple Silicon

| Chip | Unified Memory | Best Models | tok/s |
|------|---------------|-------------|-------|
| M1/M2 (16 GB) | 16 GB | 7B–13B Q4 | ~15–25 |
| M3/M4 Pro (36 GB) | 36 GB | 30B Q4–Q5 | ~20–30 |
| M3/M4 Max (64 GB) | 64 GB | 70B Q4 (viable) | ~15–20 |
| M4 Max (128 GB) | 128 GB | 70B Q5–Q8 | ~20–25 |
| M5 Max (128 GB) | 128 GB | 70B Q4–Q8 | ~30+ |

Apple Silicon unified memory acts as VRAM. The M5 (with macOS 26.2+) unlocks Neural Accelerators for up to 4x faster time-to-first-token.

## Quantisation Formats

### Format Comparison

| Format | Hardware | Portability | Quality at 4-bit | Speed | Best For |
|--------|----------|-------------|-----------------|-------|----------|
| **GGUF** | CPU, GPU, Metal | Highest (all platforms) | Baseline | Good | Default choice, cross-platform |
| **AWQ** | CUDA only | GPU-only | +0.5–1.0% vs GPTQ | Fast | Best quality at 4-bit on NVIDIA |
| **GPTQ** | CUDA only | GPU-only | Slightly behind AWQ | Fast | Legacy, mature ecosystem |
| **EXL2** | CUDA only | NVIDIA-only | Variable (2.0–8.0 bpw) | Fastest single-user | Precise VRAM targeting |
| **MLX** | Apple Silicon only | Mac-only | Comparable to GGUF Q4 | Fastest on Mac | Apple Silicon native |

### Recommended Quantisation Levels

| Use Case | Recommended Quant | Quality Retention |
|----------|------------------|-------------------|
| General chat | Q4_K_M (GGUF) | 90–95% of FP16 |
| Coding / agent tasks | Q4_K_M minimum, Q5_K_M preferred | Reasoning degrades 3x faster than perplexity |
| Mission-critical reasoning | Q5_K_M or Q8 | >97% of FP16 |
| VRAM-constrained (<8 GB) | Q3_K_M | Acceptable for chat only, not reasoning |
| Small models (1–3B) | Q8_0 or FP16 | Quant overhead minimal at small sizes |

**Key insight**: Reasoning benchmarks (GSM8K, HumanEval) degrade ~3x faster than perplexity under quantisation. Never go below Q4 for coding or agent workloads.

**Unsloth Dynamic v2.0** is the de facto community standard for high-quality GGUF quantisations, outperforming standard methods on accuracy preservation.

## Inference Frameworks

### Framework Comparison (May 2026)

| Framework | Best For | Platform | Key Feature |
|-----------|----------|----------|-------------|
| **Ollama** | Quick setup, single-user chat | All (MLX on Mac) | One-command model pull, coding agent integration |
| **LM Studio** | GUI users, team sharing | All (MLX on Mac) | Visual model management, LM Link remote access |
| **llama.cpp** | Maximum portability, oversized models | All | CPU+GPU split, widest format support |
| **MLX** | Apple Silicon native, fine-tuning | macOS only | LoRA/QLoRA on-device, fastest for <14B |
| **vLLM** | Multi-user API server, production | Linux (NVIDIA/AMD) | 793 tok/s peak, PagedAttention |
| **SGLang** | RAG-heavy workloads | Linux (NVIDIA) | 29% higher throughput than vLLM on prefix reuse |

### Ollama (v0.24.0, May 2026)

- MLX backend on Apple Silicon since v0.19.0 (March 2026): 1.6x faster prefill, 2x decode
- Supports Gemma 4, Qwen 3.5/3.6, Llama 4, DeepSeek, GLM-5, and 100+ models
- OpenAI-compatible API on port 11434
- New `ollama launch` command for coding agent integration (Claude Code, Codex)
- Desktop app with drag-and-drop multimodal input

### llama.cpp (April 2026 Rewrite)

The largest architectural rewrite in the project's history:
- New kernel generator (replaces 11,000 lines of duplicated intrinsics)
- Head-contiguous KV cache layout for coalesced reads
- Unified Metal/CUDA/ROCm backend dispatch
- **Result**: 2.1x throughput on 70B Q4_K_M (M3 Ultra: 14.1 → 29.6 tok/s)
- MTP speculative decoding merged for Qwen 3.6 (~2x generation on 27B dense)
- Vulkan backend now competitive — can beat CUDA by ~40% on some hardware
- GGUF remains the canonical cross-platform format

### MLX vs llama.cpp on Apple Silicon

| Model Size | MLX Advantage | Notes |
|-----------|---------------|-------|
| <14B | 20–87% faster generation | M4 Max: Qwen3-8B 4-bit = 93 tok/s (MLX) vs 77 tok/s (llama.cpp) |
| 27B+ | Near zero | Memory bandwidth is the bottleneck |
| Long context (8K+) | llama.cpp wins | MLX prefill significantly slower; FlashAttention gap |

MLX supports LoRA/QLoRA fine-tuning on-device (llama.cpp is inference-only). MLX requires its own model format — not interchangeable with GGUF.

### LM Studio (v0.4.14, May 2026)

- Uses MLX on Apple Silicon (mlx-engine v1.8.1)
- MTP speculative decoding: 1.5–3x tok/s improvement
- LM Link: Tailscale-backed remote access for team sharing
- MCP Host functionality, OpenAI + Anthropic-compatible API on localhost:1234
- Headless `llmster` daemon for server deployments

## MoE Models on Consumer Hardware

**Critical rule**: MoE models need VRAM proportional to TOTAL parameters, not active parameters. Speed benefits are real; memory savings are not.

| Model | Total / Active | VRAM at Q4 | Consumer Viable? |
|-------|---------------|-----------|-----------------|
| Gemma 4 26B-A4B | 26B / 3.8B | ~16 GB | Yes (RTX 4080+) |
| Qwen 3.6-35B-A3B | 35B / 3B | ~22 GB | Yes (24 GB with offload) |
| Mixtral 8x7B | 46.7B / 13B | ~26 GB | No (exceeds RTX 4090) |
| Llama 4 Scout | 109B / 17B | ~55 GB | No (multi-GPU/Mac Studio) |
| Llama 4 Maverick | 400B / 17B | ~200 GB | No (datacenter only) |
| DeepSeek V3 | 671B / 37B | ~350 GB | No |

**For consumer hardware (<48 GB VRAM), dense models always outperform MoE at comparable VRAM usage.** The only consumer-friendly MoE exceptions are Gemma 4 26B-A4B and Qwen 3.6-35B-A3B, which have very low active parameter counts.

MoE routing errors cascade under aggressive quantisation — dense models degrade more gracefully below Q4.

## Apple Silicon and MLX

### The 2026 Mac Stack

Both Ollama and LM Studio now use MLX as their Apple Silicon backend. MLX is the only framework targeting M5 Neural Accelerators (requires macOS 26.2+).

**MLX advantages**:
- Fastest generation for models under 14B (20–87% over llama.cpp)
- Native LoRA/QLoRA fine-tuning on-device
- Swift bindings for iOS development
- M5 Neural Accelerators: up to 4x faster TTFT, 30–60% overall improvement

**MLX limitations**:
- Mac-only (not portable)
- Slower prefill at long contexts (8K+) — llama.cpp wins there
- Smaller pre-quantised model ecosystem than GGUF
- No CPU+GPU split for oversized models (llama.cpp can do this)
- Some compatibility issues with newer models (e.g., hybrid attention)

### Recommended Mac Setups

- **MacBook Air/Pro M3 (24 GB)**: Qwen 3.6-27B Q4 or Gemma 4 26B-A4B — good for coding assistance
- **Mac Mini M4 Pro (36 GB)**: 30B models at Q5–Q8 — excellent home server
- **Mac Studio M4 Max (64–128 GB)**: 70B models, multi-model serving
- **Mac Studio M4 Ultra (192 GB)**: Run anything including Llama 4 Scout

## What Changed in the Last 6 Months

### December 2025
- **Mistral Large 3** (675B MoE, Apache 2.0) — first major open MoE at scale

### January 2026
- **Falcon H1R 7B** — hybrid Mamba/Transformer architecture from TII
- **LMArena rebrand** from LMSYS Chatbot Arena — Elo scale shift of 30+ points

### February 2026
- **Qwen 3.5** flagship (397B-A17B) — strongest open scientific reasoner
- **GLM-5** (Zhipu AI) — first frontier model on Huawei Ascend hardware

### March 2026
- **Gemma 4** released under Apache 2.0 (huge licence upgrade from Gemma 3)
- **Ollama 0.19** adds MLX backend — 2x speed on Apple Silicon
- **Voxtral TTS** (Mistral, 4B) — first credible open-weights TTS

### April 2026
- **Gemma 4 MTP variants** — speculative decoding support
- **Qwen 3.6** (35B-A3B and 27B dense) — new consumer sweet spots
- **DeepSeek V4** (1.6T, MIT) — strongest open coder
- **GLM-5.1** — takes open-source Arena Elo crown (1474)
- **Kimi K2.6** — leads Intelligence Index (54)
- **MiMo-V2.5-Pro** (Xiaomi) — surprise frontier-credible entry
- **llama.cpp major rewrite** — 2.1x throughput
- **Mistral Medium 3.5** (128B dense) — best non-MoE open model

### May 2026
- **Ollama 0.24** — Codex App support, MLX sampler improvements
- **vLLM v0.21** — DeepSeek V4 on Blackwell, TOKENSPEED_MLA backend
- **llama.cpp MTP for Qwen 3.6** — ~2x generation on 27B dense
- **MLX 0.31** — M5 Neural Accelerator support, up to 4x faster TTFT
- **LM Studio 0.4.14** — stable MTP speculative decoding

## Licensing

| Family | Licence | Commercial Use | Key Restrictions |
|--------|---------|---------------|-----------------|
| Gemma 4 | Apache 2.0 | Unrestricted | None |
| Qwen 3.5/3.6 | Apache 2.0 | Unrestricted | None |
| DeepSeek V4 | MIT | Unrestricted | None |
| Phi-4 | MIT | Unrestricted | None |
| GLM-5.1 | MIT | Unrestricted | None |
| Mistral Large 3 | Apache 2.0 | Unrestricted | None |
| Mistral Medium 3.5 | Modified MIT | Unrestricted | Minor attribution |
| Llama 4 | Llama Community | Conditional | 700M MAU cap, attribution required, acceptable use policy |
| Kimi K2.6 | Modified MIT | Unrestricted | Minor terms |
| Falcon H1R | Falcon LLM 1.0 | Unrestricted | Apache-derived, no-litigate clause |

**The licensing landscape has shifted decisively toward permissive open source.** Apache 2.0 and MIT dominate. Meta's Llama licence is now the outlier, not the norm.

## Recommendations

### If you have an RTX 4090 (24 GB) or similar

**General use**: Qwen 3.6-27B Q4_K_M (~17 GB) or Gemma 4 31B Q4_K_M (~20 GB)
- Qwen 3.6-27B: Fastest inference (~25–35 tok/s), best SWE-bench, native vision
- Gemma 4 31B: Best maths (AIME 89.2%), best human-preference (Arena #3 open)

**Coding**: Qwen 3.6-27B for agentic coding; Qwen 2.5-Coder 32B for autocomplete/FIM

**Maths/Reasoning**: DeepSeek R1-Distill-Qwen-32B Q4 (~18 GB) or Phi-4-reasoning-plus Q5 (~10 GB)

### If you have a Mac with 32–64 GB unified memory

Same models as above — Apple Silicon unified memory acts as VRAM. Use Ollama (MLX backend) or LM Studio for the best experience. Expect ~15–25 tok/s on M3/M4 Pro/Max for 27–32B models.

### If you have 16 GB (RTX 4070/4080 or Mac M-series base)

**General**: Gemma 4 26B-A4B Q4 (~16 GB) or Qwen 3.6-35B-A3B with offloading
**Coding**: Qwen 2.5-Coder 14B Q4 or DeepSeek Coder V2 16B
**Reasoning**: Phi-4-reasoning-plus Q4 (~10 GB) — comfortably fits with headroom

### If you have 8 GB

**General**: Qwen 3 8B or Gemma 4 E4B (both excellent at this tier)
**Coding**: Qwen 2.5-Coder 7B (88.4% HumanEval)
**Reasoning**: Qwen 3 4B with /think mode (73.8% AIME at 2.5 GB — remarkable)

### Framework choice

- **Just getting started**: Ollama (`ollama run qwen3.5:27b`)
- **Want a GUI**: LM Studio
- **Need maximum control**: llama.cpp directly
- **Apple Silicon fine-tuning**: MLX
- **Multi-user API server**: vLLM
- **RAG-heavy production**: SGLang

## References

1. [Google AI — Gemma Releases](https://ai.google.dev/gemma/docs/releases) — Official release documentation
2. [Google Open Source Blog — Gemma 4 Apache 2.0](https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html)
3. [Botmonster — Gemma 4 vs Qwen 3.5 vs Llama 4](https://botmonster.com/ai/gemma-4-vs-qwen-3-5-vs-llama-4-open-model-comparison-2026/) — Detailed benchmark comparison
4. [Codersera — Open-Source LLMs Landscape May 2026](https://codersera.com/blog/open-source-llms-landscape-2026/)
5. [InsiderLLM — Best Local Coding Models 2026](https://insiderllm.com/guides/best-local-coding-models-2026/)
6. [InsiderLLM — Best Local LLMs for Math & Reasoning](https://insiderllm.com/guides/best-local-llms-math-reasoning/)
7. [InsiderLLM — Vision Models Locally](https://insiderllm.com/guides/vision-models-locally/)
8. [MLSystems Review — llama.cpp 2026 Rewrite](https://mlsystemsreview.com/llama-cpp-2026-rewrite/)
9. [Groundy — MLX vs llama.cpp on Apple Silicon](https://groundy.com/articles/mlx-vs-llamacpp-on-apple-silicon-which-runtime-to-use-for-local-llm-inference/)
10. [Hadidiz Flow — Best Open-Source Embedding Models 2026](https://hadidizflow.com/blog/2026-03-02-best-open-source-embedding-models-for-local-rag/)
11. [WhatLLM — Best Local LLMs 2026](https://whatllm.org/best-local-llm)
12. [PromptQuorum — Local LLM Hardware Guide 2026](https://www.promptquorum.com/local-llms/local-llm-hardware-guide-2026)
13. [Ofox.ai — LLM Leaderboard April 2026](https://ofox.ai/blog/llm-leaderboard-best-ai-models-ranked-2026/)
14. [ByteIota — Ollama MLX 2x Faster](https://byteiota.com/ollama-mlx-2x-faster-local-ai-on-apple-silicon-2026/)
15. [Codersera — Local AI Runtimes May 2026](https://codersera.com/blog/local-ai-runtimes-may-2026-update/)
16. [Codersera — vLLM vs Ollama vs LM Studio 2026](https://codersera.com/blog/vllm-vs-ollama-vs-lm-studio-production-2026/)
17. [FamStack — MLX vs GGUF Apple Silicon](https://famstack.dev/guides/mlx-vs-gguf-apple-silicon/)
18. [Presenc.ai — Quantisation Quality Benchmarks 2026](https://presenc.ai/research/local-llm-quantization-quality-benchmarks-2026)
19. [InsiderLLM — MoE Models Explained](https://insiderllm.com/guides/moe-models-explained/)
20. [Unsloth — Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
21. [arXiv:2511.05502 — Production-Grade Local LLM Inference on Apple Silicon](https://arxiv.org/pdf/2511.05502)
22. [LMArena Leaderboard](https://arena.ai/leaderboard/text)
23. [Qwen3 on Ollama](https://ollama.com/library/qwen3)
24. [GitHub — QwenLM/Qwen3](https://github.com/QwenLM/Qwen3)
