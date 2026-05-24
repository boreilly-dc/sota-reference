# Open-Source Multimodal Vision-Language Models for Local Inference

| Field | Value |
|-------|-------|
| Created | 2026-05-24 |
| Last Updated | 2026-05-25 |
| Version | 1.1 |

---

- [Executive Summary](#executive-summary)
- [Which Model Should I Use?](#which-model-should-i-use)
- [Model Landscape (May 2026)](#model-landscape-may-2026)
- [Independent Benchmark Rankings](#independent-benchmark-rankings)
- [Models Best Suited for Detailed Image Description](#models-best-suited-for-detailed-image-description)
- [Architecture Overview](#architecture-overview)
- [Hardware Requirements and VRAM Guide](#hardware-requirements-and-vram-guide)
- [Inference Frameworks](#inference-frameworks)
- [Quantisation](#quantisation)
- [Apple Silicon Deployment](#apple-silicon-deployment)
- [Specialised Models: OCR and Document Understanding](#specialised-models-ocr-and-document-understanding)
- [Hallucination and Reliability](#hallucination-and-reliability)
- [Licensing](#licensing)
- [Managed Service Equivalents](#managed-service-equivalents)
- [Caveats and Limitations](#caveats-and-limitations)
- [References](#references)

## Executive Summary

Vision-language models (VLMs) accept image input and produce text output — answering questions about images, generating descriptions, reading documents, and interpreting visual scenes. As of mid-2026, open-source VLMs have reached a level where an 8B-parameter model running on consumer hardware can produce detailed, accurate image descriptions that rival proprietary APIs from 18 months prior.

The field is dominated by a few model families: **Qwen-VL** (Alibaba), **Gemma** (Google), **InternVL** (Shanghai AI Lab), **Molmo** (Allen AI), and **LLaVA-OneVision** (NTU/Microsoft). For the specific task of *thorough, detailed image description*, Molmo and larger Qwen models stand out due to training data that emphasises dense captioning.

On the LMArena Vision Arena (May 2026), the top open-weight model (GLM-4.6v) sits just 25 Elo points behind the frontier proprietary leader (Gemini 3 Pro Vision at ~1486). For local deployment specifically, Qwen 2.5-VL at Elo ~1454 is the highest-ranked model that runs comfortably on consumer hardware.

## Which Model Should I Use?

Quick decision matrix based on hardware and use case:

| Hardware | Best Model | Why |
|----------|-----------|-----|
| 4 GB VRAM / CPU | SmolVLM2 2.2B or Moondream 2B | Fits in memory; basic captioning |
| 8 GB VRAM | Qwen3-VL 8B (Q4_K_M) | Best all-round at this tier; MathVista 85.8, DocVQA 95+ |
| 12-16 GB VRAM | Qwen2.5-VL 7B or InternVL3-8B | Excellent document understanding; Qwen2.5-VL DocVQA 95.7 |
| 16-24 GB VRAM | Gemma 4 26B-A4B (MoE) or Phi-4-reasoning-vision 15B | Gemma 4: only 3.8B active params, fast; Phi-4: science/diagrams |
| 24+ GB VRAM | Qwen 3.6-27B or InternVL3-78B (Q4) | Top-tier quality; Qwen 3.6 claimed new local SOTA |
| Apple M-series 32GB+ | Any of the above via Ollama/MLX-VLM | Unified memory advantage; see Apple Silicon section |

**For thorough, detailed description specifically:**

1. **Molmo 2 8B** — Trained on PixMo (dense descriptions averaging hundreds of words per image). Purpose-built for exhaustive captioning. Apache 2.0.
2. **Qwen 3.6-27B** or **Qwen2.5-VL 7B** — Strong general-purpose VLMs with excellent document/scene understanding. Larger context windows produce more detailed output.
3. **Gemma 3/4 27B** — Produces detailed descriptions even at 4B; scales up well.

## Model Landscape (May 2026)

### Tier 1: Frontier Local Models (16+ GB VRAM)

| Model | Params | Active Params | VRAM (Q4_K_M) | Key Strengths |
|-------|--------|---------------|---------------|---------------|
| Qwen 3.6-27B | 27B (dense) | 27B | ~17 GB | Claimed new local SOTA; vision baked into base arch |
| Gemma 4 26B-A4B | 26B (MoE) | 3.8B | ~18 GB | Fastest multimodal MoE; extremely fast decode |
| InternVL3-78B | 78B | 78B | ~45 GB (Q4) | MMMU 72.2 — SOTA among open-source |
| Qwen2.5-VL 72B | 72B | 72B | ~42 GB (Q4) | Rivals GPT-4o on document understanding |

### Tier 2: Consumer Hardware (6-16 GB VRAM)

| Model | Params | VRAM (Q4_K_M) | Open VLM Rank | Avg Score | MMMU | OCRBench |
|-------|--------|---------------|---------------|-----------|------|----------|
| Qwen3-VL 8B | 8B | ~6 GB | — | — | — | — |
| InternVL3-8B | 8B | ~6 GB | #30 / 285 | 73.6 | 62.2 | 884 |
| Qwen2.5-VL 7B | 7B | ~6 GB | #55 / 285 | 70.9 | 58.0 | 888 |
| MiniCPM-o 2.6 | 8.7B | ~6 GB | #60 / 285 | 70.2 | 50.9 | 889 |
| Molmo 2 8B | 8B | ~6 GB | — | — | — | — |
| Gemma 3 12B | 12B | ~8 GB | #121 / 285 | 62.8 | 55.2 | 702 |
| Phi-4-MultiModal | 5.6B | ~4 GB | #102 / 285 | 64.7 | 56.0 | 844 |
| Pixtral 12B | 12B | ~8 GB | #134 / 285 | 61.0 | 51.1 | 685 |
| LLaVA-OneVision 7B | 8B | ~6 GB | #139 / 285 | 60.2 | 47.9 | 622 |
| Molmo-7B-D (v1) | 8B | ~6 GB | #158 / 285 | 57.4 | 49.1 | 656 |

### Tier 3: Edge / Embedded (2-6 GB VRAM)

| Model | Params | VRAM | Open VLM Rank | Avg Score | Best For |
|-------|--------|------|---------------|-----------|----------|
| Gemma 3 4B | 4B | ~3.2 GB | #173 / 285 | 55.4 | Basic captioning; surprisingly detailed |
| InternVL3-2B | 2.1B | ~2 GB | #105 / 285 | 64.5 | Best at this size (OCR: 831) |
| Qwen2.5-VL 3B | 3.8B | ~3 GB | #108 / 285 | 64.5 | Strong OCR (828); general purpose |
| SmolVLM2 2.2B | 2.2B | ~2 GB | #191 / 285 | 52.2 | Lightweight; mobile |
| Moondream 2B | 2B | ~2 GB | #237 / 285 | 43.0 | Fast basic captioning; Apache 2.0 |
| Florence-2 0.7B | 0.7B | <1 GB | N/A (specialised) | — | Structured captioning, detection |
| PaddleOCR-VL 0.9B | 0.9B | CPU-viable | N/A (specialised) | — | OCR/document parsing |
| InternVL3-1B | 0.9B | <1 GB | #163 / 285 | 57.0 | Smallest open VLM with decent OCR (798) |
| SmolVLM2 500M | 0.5B | <1 GB | #250 / 285 | 40.9 | Edge deployment; OCR: 609 |

### LMArena Vision Arena Rankings (May 2026)

For context, these are the top frontier models ranked by human preference on visual reasoning and description tasks:

| Rank | Model | Vision Elo | Type |
|------|-------|-----------|------|
| 1 | Gemini 3 Pro Vision | ~1486 | Proprietary |
| 2 | Claude Opus 4.6 Vision | ~1478 | Proprietary |
| 3 | GPT-5.2 Vision | ~1471 | Proprietary |
| 5 | GLM-4.6v | ~1461 | Open-weight |
| — | Qwen 2.5-VL | ~1454 | Open-weight |

The Vision Arena prompt distribution emphasises visual reasoning and description — making it particularly relevant for the "thorough image description" use case. However, confidence intervals for the top-3 overlap, and the leaderboard under-represents document-extraction workloads.

## Independent Benchmark Rankings

Data from the **HuggingFace Open VLM Leaderboard** (opencompass/open_vlm_leaderboard) — 285 VLMs evaluated across 8 benchmarks (MMBench v1.1, MMStar, MMMU, MathVista, OCRBench, AI2D, HallusionBench, MMVet). Last updated March 2026.

### Top Open-Source Models (All Sizes)

| Rank | Model | Params | Avg Score | MMBench | MMStar | MMMU | OCRBench | HallusBench |
|------|-------|--------|-----------|---------|--------|------|----------|-------------|
| 7 | InternVL3-78B | 78B | 79.1 | 87.7 | 73.4 | 72.2 | 908 | 59.1 |
| 10 | InternVL3-38B | 38B | 77.8 | 86.8 | 72.6 | 69.7 | 886 | 58.4 |
| 17 | Qwen2.5-VL-72B | 73B | 76.1 | 87.8 | 70.5 | 68.2 | 882 | 54.6 |
| 25 | InternVL3-14B | 15B | 75.2 | 83.6 | 68.9 | 64.8 | 877 | 55.9 |
| 26 | Qwen2.5-VL-32B | 34B | 74.8 | 84.0 | 70.3 | 68.9 | 856 | 58.4 |
| 30 | InternVL3-8B | 8B | 73.6 | 82.1 | 68.7 | 62.2 | 884 | 49.0 |
| 55 | Qwen2.5-VL-7B | 8B | 70.9 | 82.2 | 64.1 | 58.0 | 888 | 51.9 |
| 78 | Gemma 3-27B | 27B | 67.4 | 78.9 | 59.6 | 64.8 | 753 | 48.8 |

Context: the top overall model (#1) is SenseNova-V6-5-Pro (proprietary, Avg 82.2). The top proprietary model from a major lab is Gemini-2.5-Pro at #4 (Avg 80.1).

### Models ≤ 8B Params (Consumer GPU-Friendly)

| Rank | Model | Params | Avg | MMStar | MMMU | OCR | Hallus | MMVet |
|------|-------|--------|-----|--------|------|-----|--------|-------|
| 8 | BlueLM-2.6-3B | 3B | 78.4 | 80.1 | 62.4 | 881 | 63.1 | 78.5 |
| 30 | InternVL3-8B | 8B | 73.6 | 68.7 | 62.2 | 884 | 49.0 | 82.8 |
| 47 | Ovis2-8B | 9B | 71.8 | 64.6 | 57.4 | 891 | 56.3 | 65.1 |
| 55 | Qwen2.5-VL-7B | 8B | 70.9 | 64.1 | 58.0 | 888 | 51.9 | 69.7 |
| 60 | MiniCPM-o-2.6 | 9B | 70.2 | 63.3 | 50.9 | 889 | 51.1 | 67.2 |
| 81 | InternVL2.5-4B-MPO | 4B | 67.2 | 61.0 | 51.8 | 879 | 47.5 | 66.0 |
| 102 | Phi-4-MultiModal | 5.6B | 64.7 | 58.9 | 56.0 | 844 | 40.5 | 51.9 |
| 105 | InternVL3-2B | 2.1B | 64.5 | 61.1 | 48.7 | 831 | 41.9 | 67.0 |
| 108 | Qwen2.5-VL-3B | 3.8B | 64.5 | 56.3 | 51.2 | 828 | 46.6 | 60.0 |
| 158 | Molmo-7B-D | 8B | 57.4 | 56.1 | 49.1 | 656 | 46.4 | 41.5 |
| 163 | InternVL3-1B | 0.9B | 57.0 | 52.3 | 43.2 | 798 | 37.2 | 58.7 |
| 191 | SmolVLM2 2.2B | 2.3B | 52.2 | 46.0 | 41.6 | 725 | 40.6 | 34.9 |
| 237 | Moondream2 | 1.9B | 43.0 | 42.1 | 29.3 | 585 | 33.0 | 40.4 |

### Key Observations from Leaderboard Data

1. **InternVL3 dominates across sizes**: InternVL3-8B (#30) is the clear winner for 8B-class models, beating Qwen2.5-VL-7B (#55) by 2.7 points. At 2B, InternVL3-2B (#105) also leads.

2. **Qwen2.5-VL excels at OCR**: Despite lower overall rankings, Qwen2.5-VL models score exceptionally on OCRBench (888 for 7B, 882 for 72B) — near the top of the entire leaderboard regardless of size.

3. **Molmo's benchmark paradox**: Molmo-7B-D ranks #158 overall — far below InternVL3-8B or Qwen2.5-VL-7B. However, Molmo was purpose-built for detailed captioning (PixMo training), and standard benchmarks (MMMU, MathVista) don't test this. CapArena (ACL 2025) confirmed benchmarks are poor predictors of captioning quality.

4. **Gemma 3 underperforms on OCR**: Gemma 3-27B (#78, OCR: 753) and 12B (#121, OCR: 702) fall well behind InternVL3 and Qwen2.5-VL on OCR tasks, suggesting they are better suited to general scene understanding than document processing.

5. **HallusionBench**: No small model scores well on hallucination resistance. The best 8B-class model (InternVL3-8B: 49.0) scores far below frontier models (SenseNova: 66.7). This confirms the hallucination risk warning for detailed description tasks.

6. **BlueLM-2.6-3B anomaly**: Ranks #8 overall at only 3B params (Avg 78.4) — an extraordinary result. Not yet widely available in local inference frameworks; worth watching.

7. **Newer models not yet tested**: Qwen3-VL, Gemma 4, and Molmo 2 are not yet on the leaderboard (last updated March 2026). Their claimed improvements are based on self-reported metrics only.

### Benchmark Definitions

| Benchmark | What It Tests |
|-----------|---------------|
| MMBench v1.1 | General visual perception and reasoning |
| MMStar | Multi-turn visual dialogue |
| MMMU | Multi-discipline multimodal understanding |
| MathVista | Mathematical reasoning with visual input |
| OCRBench | Text recognition accuracy in images |
| AI2D | Science diagram understanding |
| HallusionBench | Resistance to visual hallucination |
| MMVet | Open-ended visual QA grading |

## Models Best Suited for Detailed Image Description

The CapArena benchmark (ACL 2025) — the first large-scale human evaluation specifically for detailed image captioning — found that **standard VLM benchmarks (MMMU, MathVista, etc.) do NOT reliably predict detailed captioning quality**. Traditional metrics like BLEU and CLIPScore also fail entirely for this task. This means model rankings based on standard benchmarks should be treated as indicative, not definitive, for description quality.

### Why Molmo Stands Out

**Molmo 2** (Allen Institute for AI) is specifically designed for dense captioning:

- Trained on **PixMo**: 1 million curated image-text pairs with descriptions averaging **hundreds of words per image**
- Original Molmo-7B-D performed between GPT-4V and GPT-4o on captioning tasks
- Molmo 2 8B achieves SOTA among open-weight models on captioning, pointing, counting, and tracking
- Fully open: weights, training data, and code are all Apache 2.0
- Supported in llama.cpp and MLX-VLM

The key differentiator is training data density. While most VLMs are trained on brief captions (1-2 sentences), Molmo's PixMo dataset contains exhaustive multi-paragraph descriptions covering spatial relationships, object attributes, background context, and scene composition.

### Prompting for Detailed Description

Regardless of model choice, prompting significantly affects description thoroughness:

- **Effective**: "Describe this image in exhaustive detail. Include spatial relationships, colours, textures, any text visible, the apparent setting, lighting conditions, and what appears to be happening."
- **Less effective**: "What's in this image?" (produces brief responses)

Larger models (27B+) produce more detailed descriptions by default due to higher reasoning capacity and longer generation tendencies.

## Architecture Overview

Modern open-source VLMs share a common architecture: **Vision Encoder → Connector → LLM Backbone**.

### Vision Encoders

| Encoder | Used By | Resolution Handling |
|---------|---------|-------------------|
| SigLIP (SO400M) | Gemma 3/4, LLaVA-OneVision, DeepSeek-VL | Tiling or Pan & Scan |
| InternViT (300M / 6B) | InternVL3 series | 448×448 tile grid + pixel unshuffle |
| Custom ViT (window attention) | Qwen2.5-VL, Qwen3-VL | Native Dynamic Resolution (no resizing) |
| CLIP ViT-L/14 | Falcon 2, older models | Fixed resolution |

**SigLIP** has largely superseded EVA-CLIP in 2025-2026 designs due to its sigmoid-based contrastive loss that scales more efficiently.

### Connectors (Vision → Language)

The dominant pattern in 2025-2026 is **simple MLP projectors** (2-layer) rather than cross-attention or perceiver architectures:

- **InternVL3**: MLP projector (4.5M–172M params) + pixel unshuffle (4× token reduction)
- **LLaVA-OneVision**: 2-layer MLP projector
- **Qwen2.5-VL**: Vision-Language Merger (compresses 4 adjacent patch features → 1 token)
- **Gemma 3**: SigLIP maps images to 256 soft tokens

Cross-attention (used by Flamingo/OpenFlamingo) is largely deprecated in favour of lightweight MLP projection, which is simpler to implement and quantise.

### Variable Resolution Strategies

| Model | Strategy | Token Efficiency | Best For |
|-------|----------|-----------------|----------|
| Qwen2.5-VL | Naive Dynamic Resolution (native size, no resizing) | High | Documents, varied aspect ratios |
| InternVL3 | 448×448 tiling + pixel unshuffle (1/4 reduction) | Medium | General purpose |
| LLaVA-OneVision | AnyRes tiling with SigLIP | Medium | General purpose |
| Gemma 3 | Pan & Scan (896×896 input, 256 tokens) | High | Speed, fixed budget |

Qwen2.5-VL's approach is particularly document-friendly as it preserves native resolution information without forced resizing, making it excellent for text-heavy images.

### Position Encoding for Multimodal

- **Qwen2.5-VL**: M-RoPE (Multimodal Rotary Position Embedding) with temporal, height, and width components — enables precise spatial and video-temporal referencing
- **InternVL3**: Variable Visual Position Encoding (V2PE) — uses smaller position increments for visual tokens to improve long-context understanding

## Hardware Requirements and VRAM Guide

### NVIDIA GPUs

| GPU | VRAM | Can Run (Q4_K_M) |
|-----|------|-------------------|
| RTX 3060 / 4060 | 8 GB | Up to 8B models (Qwen3-VL 8B, Molmo 2 8B) |
| RTX 3080 / 4070 Ti | 12 GB | Up to 12B (Gemma 3 12B, Pixtral 12B) |
| RTX 3090 / 4080 | 16 GB | Up to 15B (Phi-4-reasoning-vision) |
| RTX 4090 | 24 GB | Up to 27B (Qwen 3.6-27B, Gemma 3 27B) |
| A100 / H100 | 40-80 GB | 72B+ models at full precision |

### Apple Silicon

| Mac | Unified Memory | Can Run |
|-----|---------------|---------|
| M1/M2 16 GB | 16 GB | Up to 8B (Q4_K_M) comfortably |
| M2/M3/M4 Pro 36 GB | 36 GB | Up to 27B (Q4_K_M) |
| M4 Pro/Max 48 GB | 48 GB | Up to 27B with room; 72B (Q3) tight |
| M4 Max 128 GB | 128 GB | 72B (Q4_K_M) comfortably |
| M4 Ultra 192 GB | 192 GB | 78B+ at high quantisation |

Apple Silicon's advantage is **unified memory**: the GPU and CPU share the same RAM pool, allowing models that would require multi-GPU setups on NVIDIA to run on a single Mac. The trade-off is lower memory bandwidth (~400 GB/s on M4 Max vs ~3 TB/s on H100), resulting in slower tok/s at equivalent model sizes.

### CPU-Only

Models under ~3B can run acceptably on modern CPUs:
- **PaddleOCR-VL 0.9B**: Viable on CPU for OCR tasks
- **Florence-2 0.7B**: CPU-friendly for captioning/detection
- **SmolVLM 256M/500M**: Designed for edge/CPU deployment

## Inference Frameworks

### llama.cpp (Recommended for Most Users)

The primary local inference engine for GGUF-quantised models. Multimodal support via **libmtmd** was merged May 2025.

**Supported models** (as of May 2026): Gemma 3 (4B/12B/27B), Gemma 4 (26B-A4B/31B/E2B/E4B), InternVL 2.5 and 3 (1B/4B/8B), Qwen 2.5 VL (3B/7B/32B), Qwen2.5 Omni (3B/7B), Qwen3 Omni (30B-A3B), Moondream2, SmolVLM (256M/500M/2.2B), SmolVLM2, Pixtral 12B, Mistral Small 3.1 24B.

**Quick start (macOS)**:
```bash
brew install llama.cpp

# CLI with image input
llama-mtmd-cli -hf ggml-org/gemma-3-4b-it-GGUF --image photo.jpg \
  -p "Describe this image in detail"

# OpenAI-compatible server
llama-server -hf ggml-org/gemma-3-4b-it-GGUF
# Then POST to http://localhost:8080/v1/chat/completions with image_url
```

Vision models require **two GGUF files**: the main model and a separate `mmproj` (multimodal projector) file. The `-hf` flag auto-downloads both.

### Ollama

User-friendly wrapper around llama.cpp (switched to MLX backend on Apple Silicon in v0.19.0, March 2026).

```bash
ollama run llama3.2-vision:11b "Describe this image" --image photo.jpg
ollama run qwen2.5-vl:7b "What do you see?" --image photo.jpg
```

**Supported vision models**: Qwen2.5-VL (3B/7B/32B), Gemma 3 (4B/12B/27B), InternVL3, Molmo, Moondream, LLaVA, Pixtral. Note: Qwen 3.6 vision is **not yet supported** as of May 2026.

### MLX-VLM (Apple Silicon Native)

Purpose-built for Apple Silicon using the MLX framework. Supports 50+ VLM architectures with features specific to Mac deployment:

- Vision feature caching (avoids re-encoding the same image)
- TurboQuant KV cache
- Distributed inference across Apple Neural Engine
- LoRA fine-tuning on Mac

```bash
pip install mlx-vlm
python -m mlx_vlm.generate --model Qwen/Qwen2.5-VL-7B-Instruct \
  --image photo.jpg --prompt "Describe this image thoroughly"
```

Supported models include: Qwen2.5-VL, Qwen3-VL, InternVL, Gemma, Phi, Pixtral, Molmo, SmolVLM, Moondream, MiniCPM, Florence, and many more.

### vLLM / SGLang (Server Deployment)

For production serving with batching, continuous batching, and high throughput:

- **vLLM**: Supports Qwen2.5-VL, InternVL, LLaVA, Pixtral. Best for multi-user server deployments.
- **SGLang**: Optimised for structured generation with vision models. RadixAttention for prefix caching.

### LM Studio

Desktop GUI application supporting vision models via GGUF. Drag-and-drop model loading, image input via the chat interface. Good for non-technical users wanting a ChatGPT-like local experience with images.

### Framework Compatibility Matrix

| Model | llama.cpp | Ollama | MLX-VLM | vLLM |
|-------|-----------|--------|---------|------|
| Qwen2.5-VL (3B/7B/32B) | Yes | Yes | Yes | Yes |
| Qwen3-VL 8B | Yes | Yes | Yes | Yes |
| Gemma 3 (4B/12B/27B) | Yes | Yes | Yes | Yes |
| Gemma 4 26B-A4B | Yes | Partial | Yes | Yes |
| InternVL3 (1B/4B/8B) | Yes | Yes | Yes | Yes |
| Molmo 2 (4B/8B) | Yes | Yes | Yes | Partial |
| Moondream 2B | Yes | Yes | Yes | No |
| SmolVLM2 | Yes | Partial | Yes | No |
| Florence-2 | No | No | Yes | Partial |
| Pixtral 12B | Yes | Yes | Yes | Yes |
| Qwen 3.6-27B (vision) | Yes | No* | Partial | Partial |

*Not yet supported as of May 2026.

## Quantisation

### GGUF Quantisation (Recommended Default: Q4_K_M)

| Quant Level | Bits | Quality | Size (8B model) | Use Case |
|-------------|------|---------|-----------------|----------|
| Q2_K | 2-3 | Poor — significant degradation | ~3 GB | Last resort on minimal hardware |
| Q4_K_M | 4-5 | Good — minimal quality loss | ~4.5 GB | **Recommended default** |
| Q5_K_M | 5-6 | Very good | ~5.5 GB | Quality-sensitive applications |
| Q6_K | 6 | Near-lossless | ~6.5 GB | When VRAM permits |
| Q8_0 | 8 | Lossless | ~8.5 GB | Reference quality |
| F16 | 16 | Full precision | ~16 GB | Development/testing |

**VLM-specific VRAM note**: Vision models have overhead beyond the LLM backbone due to the vision encoder and projector. A "7B" VLM typically uses ~6 GB at Q4_K_M (vs ~4.5 GB for a text-only 7B model).

### Critical Finding: Vision Encoder Sensitivity

Research (University of Maryland, Jan 2026) demonstrates that the **vision encoder (ViT) is disproportionately sensitive to quantisation** relative to its parameter count. The LLM backbone tolerates aggressive quantisation much better.

Practical implications:
- The mmproj (multimodal projector) file should be kept at FP16 when possible
- If quality degrades noticeably after quantisation, try keeping the vision encoder at higher precision while quantising only the LLM backbone
- Q4_K_M on the LLM + FP16 on the vision encoder is a strong default

### Alternative Quantisation Methods

| Method | Best For | Framework Support |
|--------|----------|-------------------|
| GGUF (Q4_K_M) | llama.cpp, Ollama, LM Studio | Universal local |
| AWQ | vLLM, TGI server deployment | Fast GPU inference |
| GPTQ | Transformers, vLLM | Accurate; slower to apply |
| bitsandbytes (NF4) | HuggingFace Transformers | Easy; less optimised |
| MLX native quant | MLX-VLM | Apple Silicon optimised |

## Apple Silicon Deployment

### Performance Characteristics

| Metric | M4 Pro (48 GB) | M4 Max (128 GB) | Notes |
|--------|----------------|-----------------|-------|
| Decode speed (8B Q4) | ~45-50 tok/s | ~55-60 tok/s | Ollama/MLX |
| Decode speed (27B Q4) | ~15-20 tok/s | ~25-30 tok/s | Memory bandwidth limited |
| TTFT (8B, image input) | ~1-2 s | ~0.8-1.5 s | Image encoding adds latency |

### MLX vs llama.cpp on Apple Silicon

Since Ollama v0.19.0 (March 2026), the MLX backend is used on Apple Silicon:

- **MLX decode speed**: ~3× faster for MoE models, 1.4-1.8× faster for dense models (vs older llama.cpp backend)
- **MLX prefill (TTFT)**: Actually *slower* than llama.cpp — disadvantage for short prompts and single-image queries
- **Memory**: ~13% savings with MLX
- **mlx-community on HuggingFace**: 4,300+ pre-converted models available

For **image description workloads** (long generation after a single image prompt), MLX's decode speed advantage is significant. For batch processing of many images with short responses, llama.cpp's faster prefill may be preferable.

### Recommended Setup for Mac Users

1. **Easiest**: `brew install ollama` → `ollama run qwen2.5-vl:7b`
2. **More control**: `pip install mlx-vlm` → Use Python API directly
3. **Maximum flexibility**: `brew install llama.cpp` → Use llama-server with OpenAI-compatible API

## Specialised Models: OCR and Document Understanding

For document-heavy workloads (receipts, forms, scientific papers, handwritten text), specialised compact models often outperform much larger general VLMs:

| Model | Params | Strength | OmniDocBench | Notes |
|-------|--------|----------|--------------|-------|
| PaddleOCR-VL | 0.9B | Overall document parsing leader | 92.6 | CPU-viable; Baidu |
| dots.ocr | 1.7B / 3B | 100+ languages; unified pipeline | Beats GPT-4o (3B) | Layout + recognition + reading order |
| GOT-OCR 2.0 | 580M | Markdown/LaTeX output | — | ~4 GB VRAM; structured notation |
| DeepSeek-OCR | 3B | Throughput (7-20× token compression) | — | 200K+ pages/day on single A100 |
| Qwen2.5-VL 7B | 7B | General VLM with excellent OCR | 95.7 DocVQA | Best general-purpose option |

VLMs achieve **3-4× lower character error rate** (CER) than traditional OCR engines (Tesseract, PaddleOCR classic) on noisy scans, receipts, and distorted text. However, traditional engines remain competitive on clean, well-formatted documents.

## Hallucination and Reliability

A critical concern for "thorough image description": models can confidently describe objects, relationships, and details that are **not present in the image**. This is especially problematic for:

- Dense/detailed descriptions (more tokens = more opportunity for fabrication)
- Low-resolution or ambiguous images
- Scenes with many objects or complex spatial relationships

### Known Failure Modes

1. **Object hallucination**: Describing objects not present (e.g., "a cat on the windowsill" when there is no cat)
2. **Attribute misassignment**: Correct objects, wrong attributes (e.g., "the red car" when the car is blue)
3. **Spatial confabulation**: Incorrect spatial relationships between objects
4. **Text hallucination**: "Reading" text from images that is too blurry to actually parse — producing plausible-but-wrong text
5. **Confidence without calibration**: Models rarely express uncertainty; they generate plausible descriptions regardless of input quality

### Mitigation Strategies

- **Use larger models**: Hallucination rates generally decrease with model size
- **Lower temperature**: Reduces creative generation that can drift from visual grounding
- **Structured prompting**: Ask about specific regions or aspects rather than open-ended "describe everything"
- **Cross-validation**: Run the same image through 2-3 models and compare; divergent claims are likely hallucinations
- **Fine-tuning with RL**: Emerging research (May 2025) uses reinforcement learning to reduce captioning hallucinations via the CHAIR metric

### The CapArena Finding

Standard VLM benchmarks (MMMU, MathVista, DocVQA) do **not reliably predict** which model produces the best detailed descriptions. The CapArena benchmark (ACL 2025) found that human preference for detailed captions correlates poorly with these standard metrics. When choosing a model specifically for thorough description, benchmark scores are an imperfect guide — testing with your own images is essential.

## Licensing

| Model | Licence | Commercial Use | Training Data Open |
|-------|---------|----------------|-------------------|
| Molmo 2 | Apache 2.0 | Yes | Yes (PixMo) |
| Qwen2.5-VL 7B+ | Apache 2.0 | Yes | No |
| Qwen2.5-VL 3B | Qwen Licence (restrictive) | Limited | No |
| Qwen3-VL | Apache 2.0 | Yes | No |
| SmolVLM / SmolVLM2 | Apache 2.0 | Yes | Partial |
| Florence-2 | MIT | Yes | No |
| Pixtral 12B | Apache 2.0 | Yes | No |
| Gemma 3 / 4 | Google Terms (permissive) | Yes | No |
| InternVL3 | Code: MIT; Weights: Qwen Licence | Check Qwen terms | No |
| LLaVA-OneVision 1.5 | Apache 2.0 (full pipeline) | Yes | Yes |
| Moondream | Code: Apache 2.0; Photon engine requires API key | Partial | No |

**Fully open** (weights + data + training code): Molmo 2, LLaVA-OneVision 1.5. These are the only options if you need to reproduce or audit the entire training pipeline.

## Managed Service Equivalents

For teams that want VLM capabilities without managing local infrastructure:

| Provider | Service | Models Available |
|----------|---------|-----------------|
| AWS | Amazon Bedrock | Claude (vision), Llama Vision via SageMaker |
| Azure | Azure AI Studio / Azure OpenAI | GPT-4o, Phi-4-vision, Florence-2 via Azure ML |
| GCP | Vertex AI | Gemini 3 Pro/Flash (vision), Gemma via Model Garden |
| IBM | watsonx.ai | Granite Vision (limited); external model hosting |
| Oracle | OCI Generative AI | Llama Vision; limited VLM selection |

These provide the same base models (or close equivalents) with managed scaling, but at significantly higher per-query cost than local inference. Local deployment is preferable when: data privacy is required, latency budget is tight, query volume is high, or fine-tuning for specific description styles is needed.

## Caveats and Limitations

1. **Benchmark ≠ Description Quality**: Standard benchmarks (MMMU, MathVista) are poor predictors of detailed captioning quality (CapArena, ACL 2025). Model rankings in this article are indicative, not definitive, for the description use case.

2. **Single-Source Claims**: The claim that Qwen 3.6-27B is "new local SOTA" comes from a single source (InsiderLLM, credibility 0.65) and could not be independently verified at time of writing. Framework support (Ollama) is not yet available.

3. **Hallucination Risk**: Longer, more detailed descriptions carry higher hallucination risk. No local VLM currently matches GPT-4o-class models on factual accuracy of dense captioning.

4. **Rapidly Evolving Field**: Model rankings shift monthly. Qwen-VL, InternVL, and Gemma families all release new versions frequently. Check framework compatibility before committing to a model.

5. **Vision Arena Limitations**: The LMArena Vision Arena has lower vote counts than the text arena, resulting in wider confidence intervals. The top-3 positions often overlap statistically.

6. **Quantisation Trade-offs**: Vision encoders are more sensitive to quantisation than LLM backbones. Aggressive quantisation (Q2/Q3) may disproportionately harm image understanding even if text generation appears fine.

## References

1. InsiderLLM. "Best Vision Models You Can Run Locally: Every Model, Every GPU Tier." May 2026. https://insiderllm.com/guides/vision-models-locally/
2. OpenGVLab. "InternVL3: Advancing Open-Source Multimodal Models with Native Multimodal Pretraining." April 2025. https://internvl.github.io/blog/2025-04-11-InternVL-3.0/
3. Willison, Simon. "Trying out llama.cpp's new vision support." May 2025. https://simonwillison.net/2025/May/10/llama-cpp-vision/
4. llama.cpp. "Multimodal Documentation." https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
5. Fang et al. "CapArena: Benchmarking and Analyzing Detailed Image Captioning in the LLM Era." ACL 2025. https://arxiv.org/abs/2503.12329
6. Allen Institute for AI. "Molmo 2 Model Family." https://allenai.org/molmo
7. LMMS Lab. "LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal LLMs." https://www.lmms-lab.com/posts/llava_onevision_1_5/
8. MLX-VLM. "Vision Language Model Inference and Fine-Tuning on Apple Silicon." https://github.com/Blaizzy/mlx-vlm
9. "MLX: The Next Inference Engine for Apple Silicon." March 2026. https://yage.ai/share/mlx-apple-silicon-en-20260331.html
10. Yang et al. "Towards Understanding Best Practices for Quantization of Vision-Language Models." January 2026. https://arxiv.org/abs/2601.15287
11. Alibaba/Qwen Team. "Qwen2.5-VL Technical Report." https://arxiv.org/abs/2502.13923
12. Dubrov. "The Definitive Guide to OCR in 2026: From Pipelines to VLMs." March 2026. https://slavadubrov.github.io/blog/2026/03/04/the-definitive-guide-to-ocr-in-2026-from-pipelines-to-vlms/
13. Li et al. "LLaVA-OneVision: Easy Visual Task Transfer." August 2024. https://arxiv.org/abs/2408.03326
14. EvolvingLMMs Lab. "LLaVA-OneVision-2." https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2
15. LMArena Vision Leaderboard. May 2026. https://arena.ai/leaderboard/vision (via https://agileleadershipdayindia.org/blogs/lmsys-chatbot-arena-rankings/vision-multimodal-leaderboard-lmarena.html)
16. SmolVLM. "Redefining small and efficient multimodal models." https://arxiv.org/abs/2504.05299
17. BentoML. "Multimodal AI: The Best Open-Source Vision Language Models in 2026." https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models
18. Park et al. "Mitigating Image Captioning Hallucinations in Vision-Language Models." May 2025. https://arxiv.org/abs/2505.03420
19. OpenCompass. "Open VLM Leaderboard." HuggingFace Spaces. Last updated March 2026. https://huggingface.co/spaces/opencompass/open_vlm_leaderboard
