# Embedding Models: Best Open-Source Local Models and How They Compare with Proprietary Alternatives

| Field | Value |
|-------|-------|
| Created | 2026-03-19 |
| Last Updated | 2026-04-30 |
| Version | 3.0 |

---

- [Executive Summary](#executive-summary)
- [What Changed Since March 2026](#what-changed-since-march-2026)
- [Understanding MTEB Benchmarks](#understanding-mteb-benchmarks)
- [Best Open-Source Embedding Models for Local Deployment](#best-open-source-embedding-models-for-local-deployment)
  - [Tier 1: Large Models (7B+)](#tier-1-large-models-7b)
  - [Tier 2: Medium Models (300M--1.5B)](#tier-2-medium-models-300m15b)
  - [Tier 3: Small/Edge Models (<300M)](#tier-3-smalledge-models-300m)
- [Multimodal Embedding Models](#multimodal-embedding-models)
- [Code Embedding Models](#code-embedding-models)
- [Running Embedding Models on Linux](#running-embedding-models-on-linux)
- [Running Embedding Models on macOS](#running-embedding-models-on-macos)
- [Inference Frameworks](#inference-frameworks)
- [Hardware Requirements](#hardware-requirements)
- [Head-to-Head: Open-Source vs Proprietary](#head-to-head-open-source-vs-proprietary)
- [Proprietary Embedding APIs (Hyperscaler-Available)](#proprietary-embedding-apis-hyperscaler-available)
- [Cost Analysis: Self-Hosting vs API](#cost-analysis-self-hosting-vs-api)
- [Decision Framework](#decision-framework)
- [Hyperscaler Managed Embedding Services](#hyperscaler-managed-embedding-services)
- [Context-Length Performance](#context-length-performance)
- [Matryoshka Embeddings and Quantisation](#matryoshka-embeddings-and-quantisation)
- [Areas of Uncertainty](#areas-of-uncertainty)
- [References](#references)

## Executive Summary

Open-source embedding models have decisively surpassed proprietary APIs on standard benchmarks. Microsoft's **Harrier-OSS-v1** (April 2026, MIT licence) now ranks #1 on multilingual MTEB v2 with a score of ~74.3 for its 27B variant. **Jina v5-text-small** (677M params, Feb 2026) scores 71.7 on MTEB v2 English — higher than any proprietary API. The field has shifted from "can open-source compete?" to "which open-source model fits my use case?"

The biggest development since March 2026 is the arrival of **natively multimodal embedding models**. Google's **Gemini Embedding 2** (March 2026) maps text, images, video, audio, and PDF into a single 3072-dimensional vector space. **Qwen3-VL-Embedding** (open-source, 2B params) beats all closed-source APIs on cross-modal retrieval benchmarks. **Cohere Embed v4** supports interleaved text+image with 128K context.

For most production RAG pipelines, the recommendation has shifted: **Jina v5-text-small** (677M) or **Qwen3-Embedding-0.6B/4B** for general use; **BGE-M3** (568M) when you need hybrid search; **Harrier-OSS-v1 0.6B** for multilingual. On Apple Silicon, **Ollama v0.19** (March 2026) switched to the MLX inference engine, delivering up to 3x faster throughput than llama.cpp.

## What Changed Since March 2026

| Date | Event |
|------|-------|
| Feb 18, 2026 | Jina Embeddings v5 released (v5-text-small 677M, v5-text-nano 239M) |
| Feb 26, 2026 | Perplexity releases pplx-embed with native INT8/binary outputs |
| Mar 10, 2026 | Google releases Gemini Embedding 2 (5-modality multimodal) |
| Mar 30, 2026 | Microsoft releases Harrier-OSS-v1 (#1 MTEB v2 multilingual) |
| Mar 30, 2026 | Ollama v0.19 switches to MLX on Apple Silicon |
| Apr 4, 2026 | Cohere retires older embedding models; Embed v4 is sole offering |
| Apr 2026 | Sentence Transformers v5.4 adds first-class multimodal support |
| Apr 2026 | Gemini Embedding 2 reaches general availability |

## Understanding MTEB Benchmarks

The Massive Text Embedding Benchmark (MTEB) is the standard framework for evaluating embedding models, covering eight task categories: retrieval, classification, clustering, semantic textual similarity (STS), pair classification, reranking, summarisation, and bitext mining.

**Critical caveat: MTEB version differences.** Multiple leaderboards exist with different task sets and **scores are not directly comparable across versions**:

- **MTEB v2 (refreshed English, 2026)**: Newer tasks and scoring methodology. Harrier-OSS-v1 27B leads at ~74.3; Jina v5-text-small scores 71.7.
- **Legacy MTEB (56 tasks)**: The original benchmark. NV-Embed-v2 scores 72.31; BGE-en-ICL scores 71.24.
- **MMTEB (multilingual)**: 500+ tasks across 250+ languages. Qwen3-Embedding-8B scores 70.58.
- **MTEB v2 English (older snapshot)**: Gemini Embedding 001 led at 68.32 before newer models submitted.

**Known MTEB limitations:**

- All scores are **self-reported** by model developers with no independent verification.
- The overall score **blends 8 categories**, hiding task-specific trade-offs. A model tuned for retrieval may underperform on clustering, dragging down its average. For RAG/search, focus on retrieval and STS sub-scores.
- MTEB is **text-only** (v1) and does not test cross-lingual retrieval, MRL dimension truncation, or truly long documents (10K+ tokens).
- **MMEB** (Massive Multimodal Embedding Benchmark) adds multimodal tasks but uses easy distractors, making it hard to differentiate models on fine-grained understanding.
- Leaderboard rankings are **volatile** — new submissions constantly reshuffle the order.
- **Domain-specific models** almost always outperform general-purpose models on their target domains, despite lower MTEB aggregate scores.
- Always **evaluate on your own corpus** using metrics like MRR and NDCG rather than relying solely on MTEB.

## Best Open-Source Embedding Models for Local Deployment

### Tier 1: Large Models (7B+)

These models deliver the highest benchmark scores but require a discrete GPU with 16GB+ VRAM (or 8GB+ with quantisation). Not practical for CPU-only deployment.

#### Microsoft Harrier-OSS-v1 27B --- New #1 on MTEB v2

- **MTEB v2 (multilingual)**: ~74.3 (highest on MTEB v2 as of April 2026)
- **Parameters**: 27B (also available in **0.6B** and **270M** variants)
- **Max tokens**: 32,768
- **Languages**: 100+
- **Licence**: MIT
- **VRAM**: ~54GB (FP16), ~27GB (INT8) for 27B; ~1.2GB for 0.6B
- **Local deployment**: sentence-transformers, HuggingFace TEI
- **Why choose it**: Highest MTEB v2 score of any open-source model. MIT licence allows commercial use. Strong multilingual performance across 131 tasks. The **0.6B variant** is practical for most hardware. Built on Google's Gemma architecture.
- **Limitations**: 27B variant requires significant GPU resources. Released March 30, 2026 with no research paper — limited documentation on architecture and training methodology.

#### Qwen3-Embedding-8B (Alibaba) --- Best Overall Apache 2.0

- **MTEB**: 70.58 (MMTEB multilingual), 80.68 (MTEB Code)
- **Parameters**: 8B (also available in **4B** and **0.6B** variants)
- **Dimensions**: 32 to 4,096+ (Matryoshka, flexible)
- **Max tokens**: 32,768
- **Licence**: Apache 2.0
- **VRAM**: ~16GB (FP16), ~8GB (INT8 quantised)
- **Local deployment**: sentence-transformers, vLLM, Ollama (`ollama pull qwen3-embedding`)
- **Why choose it**: Best combination of performance, commercial licence (Apache 2.0), and flexibility. Excellent multilingual and code performance. The **0.6B variant** fits easily in 2GB of VRAM or runs on CPU.
- **Limitations**: The 8B variant requires a capable GPU at full precision.

#### NV-Embed-v2 (NVIDIA) --- Highest Legacy MTEB

- **MTEB (legacy 56-task)**: 72.31 (highest on legacy leaderboard)
- **Retrieval score**: 62.65
- **Parameters**: 7.85B
- **Dimensions**: 4,096 (fixed)
- **Max tokens**: 32,768
- **Licence**: CC-BY-NC-4.0 (non-commercial only)
- **VRAM**: 16GB+ required
- **Local deployment**: sentence-transformers, vLLM
- **Why choose it**: Highest overall legacy MTEB score. Strong across all task categories.
- **Limitations**: **Non-commercial licence** severely limits production use. Fixed (non-Matryoshka) dimensions.

#### BGE-en-ICL (BAAI) --- Best MIT-Licensed Large Model

- **MTEB (legacy)**: 71.24
- **Parameters**: ~7B
- **Dimensions**: 4,096
- **Max tokens**: 32,768
- **Licence**: MIT
- **VRAM**: ~14GB (FP16), ~7GB (INT8)
- **Local deployment**: sentence-transformers, vLLM
- **Why choose it**: Strong all-round performance with fully permissive MIT licence. In-context learning capability.
- **Limitations**: Large model; requires a discrete GPU.

#### Nomic Embed Code (Nomic AI) --- Best Open-Source Code Embedding

- **Parameters**: 7B
- **Licence**: Apache 2.0
- **Local deployment**: sentence-transformers
- **Why choose it**: Outperforms Voyage Code 3 and OpenAI Embed 3 Large on CodeSearchNet benchmarks. Fully open source — training data, code, and weights all published. Trained on CoRNStack dataset (ICLR 2025 paper). Supports Python, Java, Ruby, PHP, JavaScript, Go.
- **Limitations**: Requires GPU. Benchmark lead is on CodeSearchNet specifically — may not generalise to all code retrieval scenarios.

### Tier 2: Medium Models (300M--1.5B)

The sweet spot for local deployment. These models run on consumer GPUs (4--8GB VRAM) and several work on CPU.

#### Jina Embeddings v5-text-small (Jina AI) --- Best Under 1B

- **MTEB v2 English**: 71.7
- **MMTEB**: 67.7
- **Parameters**: 677M (built on Qwen3-0.6B-Base backbone)
- **Dimensions**: 1024
- **Licence**: Available for self-hosting
- **Local deployment**: sentence-transformers, Jina API
- **Why choose it**: Top MTEB v2 score among models under 1B params. Uses 4 task-specific LoRA adapters (retrieval, similarity, classification, text matching). Distilled from Qwen3-Embedding-4B teacher. Also available as **v5-text-nano** (239M params, MTEB v2: 71.0) — the top MMTEB performer under 500M params.
- **Limitations**: Released February 2026; limited production track record. Licence terms should be checked for commercial use.

#### BGE-M3 (BAAI) --- Best All-Rounder for Hybrid Search

- **MTEB v2 English avg**: ~63.0
- **Parameters**: 568M
- **Dimensions**: 1024
- **Max tokens**: 8,192
- **Licence**: MIT
- **Retrieval modes**: Dense, sparse, and multi-vector (ColBERT-style) --- all from a single model
- **VRAM**: ~2GB (GPU), runs on CPU with ~3GB RAM
- **Local deployment**: sentence-transformers, FastEmbed, Ollama, ONNX
- **Why choose it**: The only major open-source model offering all three retrieval modes (dense, sparse, multi-vector) in one model. MIT licence. Multilingual (100+ languages). Runs comfortably on consumer hardware including Mac. Excellent for hybrid search pipelines.
- **Context-length caveat**: BGE-M3 excels at short contexts but degrades sharply beyond ~2K tokens. Independent benchmarking shows passkey retrieval accuracy of 1.0 at 512 tokens but dropping to 0.32 at 4K — despite the 8192-token maximum. For longer inputs, use Qwen3-Embedding-0.6B or chunk to keep inputs under 2K tokens.
- **Pooling strategy**: Use CLS pooling (not MEAN or LAST). Wrong pooling can cut performance by 50%+.

#### pplx-embed (Perplexity) --- Best for Storage Efficiency

- **MTEB Multilingual v2**: 69.66 nDCG@10 (4B variant)
- **Parameters**: 0.6B and 4B variants
- **Max tokens**: 32,768
- **Licence**: MIT
- **Local deployment**: sentence-transformers, Perplexity API, HuggingFace
- **Why choose it**: Native INT8 and binary embedding outputs reduce storage by 4x and 32x vs FP32. Built on Qwen3 backbones with diffusion-based continued pretraining and quantisation-aware training. No instruction prefixes required.
- **Limitations**: Released February 2026; benchmark claims are vendor-reported and not yet independently verified.

#### Snowflake Arctic Embed (Snowflake Labs) --- Best Retrieval per Parameter

- **Parameters**: Up to 334M (multiple size variants: xs/s/m/l)
- **Dimensions**: 1024
- **Licence**: Apache 2.0
- **VRAM**: ~1.5GB (largest variant), smaller variants run on CPU
- **Local deployment**: sentence-transformers, FastEmbed, Ollama (`ollama pull snowflake-arctic-embed`), NVIDIA NIM
- **Why choose it**: SOTA retrieval performance per size class on MTEB/BEIR. The 334M model competes with models 20x its size on retrieval benchmarks. v2.0 adds multilingual support.
- **Limitations**: Optimised primarily for retrieval; less competitive on classification/clustering.

#### stella_en_1.5B_v5 --- Best Mid-Size English-Only

- **Parameters**: 1.5B
- **Dimensions**: 1024 (Matryoshka)
- **Licence**: MIT
- **VRAM**: ~3GB (GPU), viable on CPU
- **Local deployment**: sentence-transformers, ONNX
- **Why choose it**: Good balance of size and performance for English-only use cases. Matryoshka support allows dimension reduction.

### Tier 3: Small/Edge Models (<300M)

These models run on any machine --- including laptops, Raspberry Pis, and CI/CD servers.

#### Jina v5-text-nano (Jina AI) --- Best Small Model

- **MTEB v2 English**: 71.0
- **MMTEB**: Top performer under 500M params
- **Parameters**: 239M (EuroBERT-210M backbone)
- **Dimensions**: 1024
- **Licence**: Available for self-hosting
- **Why choose it**: Remarkably high MTEB v2 score for its size. Distilled from larger Qwen3 teacher model.

#### Harrier-OSS-v1 270M (Microsoft) --- Best Small Multilingual

- **Parameters**: 270M
- **Languages**: 100+
- **Licence**: MIT
- **Why choose it**: Part of the MTEB v2 #1 model family. Strong multilingual performance in a tiny package. MIT licence.

#### embeddinggemma-300m (Google) --- Lightweight Multilingual

- **Parameters**: 300M (built on Gemma 3 and T5Gemma)
- **Dimensions**: 768
- **Languages**: 100+
- **Licence**: Apache 2.0
- **RAM**: ~1.5GB
- **Local deployment**: sentence-transformers, Ollama
- **Why choose it**: Designed for on-device and edge deployment. Apache 2.0 licensed.

#### Nomic embed-text-v1.5 --- Best Fully Open Model

- **Parameters**: 137M
- **Dimensions**: 768 (Matryoshka: 768/512/256/128/64)
- **MTEB**: ~62
- **Max tokens**: 8,192
- **Licence**: Apache 2.0
- **RAM**: ~1GB
- **Local deployment**: sentence-transformers, Ollama (`ollama pull nomic-embed-text`), FastEmbed, ONNX
- **Why choose it**: Fully open --- code, training data, and weights all published. Runs on CPU at ~500 passages/sec on a c6i.xlarge (~$125/month) — nearly matching OpenAI 3-small quality (MTEB 62 vs 62.3) at zero per-token cost.

#### all-MiniLM-L6-v2 --- Fastest, Smallest

- **Parameters**: 22M
- **Dimensions**: 384
- **MTEB**: ~56.3
- **Max tokens**: 256 (optimal), 512 (max)
- **Licence**: Apache 2.0
- **RAM**: <0.5GB
- **Local deployment**: sentence-transformers (default model), FastEmbed, Ollama, ONNX
- **Why choose it**: Sub-10ms inference on CPU. 5--14K sentences/sec. Ideal for prototyping, edge/IoT/mobile.
- **Limitations**: Significantly lower accuracy. English only. Short context.

## Multimodal Embedding Models

A major 2026 development: embedding models that handle text, images, video, audio, and documents in a single vector space, enabling cross-modal search.

| Model | Modalities | Params | Dims | Open-Source? | Key Strength |
|-------|-----------|--------|------|-------------|-------------|
| **Gemini Embedding 2** | Text, image, video, audio, PDF | Unknown | 3072 | No (API) | Only 5-modality model; best cross-lingual |
| **Qwen3-VL-Embedding** | Text, image, doc images, video | 2B / 8B | 2048 | Yes (Apache 2.0) | Best cross-modal retrieval (R@1=0.945) |
| **Jina Embeddings v4** | Text, image, PDF | 3.8B | 2048 | Self-hostable | ColBERT-style multi-vector + LoRA adapters |
| **NVIDIA Omni-Embed-Nemotron** | Text, image, audio, video | 4.7B | -- | Non-commercial | Only open 4-modality model |
| **Cohere Embed v4** | Text, image (interleaved) | Unknown | 1024 | No (API) | 128K context; Matryoshka + quantisation |
| **Voyage Multimodal 3.5** | Text, image, video | Unknown | 1024 | No (API) | Best MRL compression (0.7% loss at 256d) |

**Independent benchmark findings** (Cheney Zhang, March 2026, 10-model test):

- **Cross-modal retrieval**: Qwen3-VL-2B (0.945) > Gemini (0.928) > Voyage (0.900). The open-source 2B model's smaller modality gap (0.25 vs 0.73) explains its lead.
- **Cross-lingual retrieval**: Gemini (0.997) dominates, perfectly aligning even idiomatic expressions (Chinese "画蛇添足" → English "to gild the lily").
- **Needle-in-a-haystack**: Gemini scored perfectly across the full 4K–32K range. BGE-M3 degraded at 8K (0.920). Sub-335M models collapsed at 4K+.
- **MRL compression**: Voyage (ρ=0.880) and Jina v4 (0.833) led with <1% degradation at 256 dims. Gemini ranked last (0.668).
- **No single model wins every round.** Model selection depends on use case.

## Code Embedding Models

| Model | Params | Licence | Key Benchmark | Notes |
|-------|--------|---------|--------------|-------|
| **Nomic Embed Code** | 7B | Apache 2.0 | Beats Voyage Code 3 on CodeSearchNet | Fully open (data, code, weights); ICLR 2025 |
| **Qwen3-Embedding-8B** | 8B | Apache 2.0 | 80.68 MTEB Code | Best code score among general-purpose models |
| **Voyage Code 3** | Unknown | API | 71.2 on code retrieval, +13.8% vs OpenAI | 300+ languages; Matryoshka; now MongoDB-owned |
| **CodeSage Large V2** | 1.3B | Apache 2.0 | Strong on The Stack V2 | MRL support; available in 130M/356M/1.3B |
| **CodeRankEmbed** | 137M | MIT | Specialised bi-encoder | 8192 context; lightweight |
| **Jina Code V2** | 137M | Apache 2.0 | Competitive lightweight option | 8192 context |

For code search, the open-source Nomic Embed Code 7B now leads CodeSearchNet benchmarks, surpassing the previously top-ranked Voyage Code 3.

## Running Embedding Models on Linux

Linux is the most straightforward platform for local embedding models. All frameworks have first-class Linux support.

### GPU (NVIDIA CUDA)

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cu121
pip3 install sentence-transformers

python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
emb = model.encode(['Hello world'], normalize_embeddings=True)
print(f'Dimensions: {emb.shape[1]}')
"
```

For production GPU serving, use **HuggingFace Text Embeddings Inference (TEI)** via Docker:

```bash
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-m3
```

### GPU (AMD ROCm)

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/rocm6.0
pip3 install sentence-transformers
```

### CPU-only

For models under ~500M parameters, CPU inference is practical:

```bash
pip3 install "sentence-transformers[onnx]"
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3', backend='onnx')
emb = model.encode(['Hello world'])
"
```

### Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull nomic-embed-text
curl http://localhost:11434/api/embeddings \
  -d '{"model": "nomic-embed-text", "prompt": "Hello world"}'
```

Ollama embedding models: `nomic-embed-text`, `mxbai-embed-large`, `snowflake-arctic-embed`, `all-minilm`, `qwen3-embedding`.

## Running Embedding Models on macOS

Apple Silicon Macs (M1–M5) can run embedding models efficiently. **Ollama v0.19** (March 30, 2026) switched to the **MLX** inference engine on Apple Silicon, delivering significant performance improvements:

- **~3x faster** on MoE models vs llama.cpp (~130 tok/s vs 43 tok/s on M4 Pro)
- **13% less memory** than equivalent GGUF models
- Zero-copy tensor operations via Apple's unified memory architecture
- M5 Neural Accelerators provide **4.06x faster TTFT** vs M4

### Ollama (recommended for Mac)

```bash
brew install ollama
ollama serve &
ollama pull nomic-embed-text       # 137M, runs on any Mac
ollama pull snowflake-arctic-embed # 334M, runs on any Mac
ollama pull qwen3-embedding        # 0.6B/4B/8B variants available
```

Ollama v0.19 auto-routes: GGUF files use llama.cpp (Metal), safetensors use MLX.

### Apple Silicon GPU (Metal / MPS)

```bash
pip3 install torch sentence-transformers
python3 -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
emb = model.encode(['Hello world'], normalize_embeddings=True)
"
```

### HuggingFace TEI (production on Mac)

TEI now supports **native Apple Silicon installation via Homebrew**, making it viable for production embedding serving on Mac without Docker.

### What fits on which Mac

| Mac | Unified Memory | Recommended Models |
|-----|---------------|-------------------|
| M1/M2 (8GB) | 8GB | all-MiniLM, bge-small/base, Nomic v1.5, embeddinggemma-300m, Harrier 270M |
| M1/M2/M3 (16GB) | 16GB | Above + BGE-M3, Snowflake Arctic, stella 1.5B, Qwen3-0.6B, Jina v5-nano |
| M3/M4 Pro (32GB) | 32GB | Above + Qwen3-4B, Jina v5-small, INT8 quantised 8B models |
| M4 Max/Ultra (64GB+) | 64GB+ | Any model at full precision including 8B+ FP16 |

### MLX ecosystem

The MLX embedding/inference ecosystem has grown rapidly: 8+ inference servers (mlx-lm, Rapid-MLX, vLLM-MLX, oMLX, LM Studio mlx-engine), with 4,316 pre-converted models on HuggingFace's mlx-community. Key gaps: multi-LoRA serving, deterministic inference, cross-platform consistency.

## Inference Frameworks

| Framework | Best For | GPU | CPU | Mac | Latest Update |
|-----------|---------|-----|-----|-----|--------------|
| **sentence-transformers** | General use, widest model support | CUDA, MPS | Yes | Yes | **v5.4**: multimodal support (text/image/audio/video), Flash Attention 2 |
| **Ollama** | Simplest local setup, Mac-friendly | CUDA, Metal/MLX | Yes | Yes | **v0.19**: MLX engine on Apple Silicon (3x faster) |
| **HuggingFace TEI** | Production GPU/Mac serving | CUDA, ROCm | ARM64 | **Homebrew** | Apple Silicon support, SPLADE pooling, gRPC |
| **vLLM** | High-throughput GPU serving (7B+) | CUDA | No | No | Pooling model framework with auto-conversion |
| **FastEmbed** | CPU-first, lightweight | No | Yes | Yes | ONNX Runtime with quantised weights |
| **ONNX Runtime** | CPU speedup (2--3x over PyTorch) | CUDA | Yes | Yes | `pip3 install "sentence-transformers[onnx]"` |
| **llama.cpp / GGUF** | Quantised CPU inference | CUDA, Metal | Yes | Yes | OpenAI-compatible /v1/embeddings endpoint |
| **MLX** | Apple Silicon native | Metal | Via unified mem | **Native** | 8+ inference servers, 4316 HF models |
| **Model2Vec** | Ultra-fast static embeddings | No | Yes | Yes | 50x size reduction, 500x speed |

## Hardware Requirements

| Model | Parameters | VRAM (FP16) | VRAM (INT8) | CPU-only? | Approx Speed (GPU) | Approx Speed (CPU) |
|-------|-----------|-------------|-------------|-----------|--------------------|--------------------|
| Harrier-OSS-v1 27B | 27B | ~54GB | ~27GB | Impractical | ~30-50 docs/s | N/A |
| Qwen3-Embedding-8B | 8B | ~16GB | ~8GB | Slow | ~50-100 docs/s | ~5-10 docs/s |
| Qwen3-Embedding-4B | 4B | ~8GB | ~4GB | Slow | ~80-150 docs/s | ~10-20 docs/s |
| Jina v5-text-small | 677M | ~2GB | ~1GB | Viable | ~200-400 docs/s | ~50-100 docs/s |
| BGE-M3 | 568M | ~2GB | ~1GB | Viable | ~200-500 docs/s | ~40-80 docs/s |
| Snowflake Arctic (L) | 334M | ~1.5GB | <1GB | Viable | ~300-700 docs/s | ~60-120 docs/s |
| Harrier-OSS-v1 270M | 270M | ~1GB | <1GB | Good | ~300-700 docs/s | ~60-120 docs/s |
| Jina v5-text-nano | 239M | ~1GB | <1GB | Good | ~300-700 docs/s | ~80-150 docs/s |
| Nomic v1.5 | 137M | <1GB | N/A | Good | ~500-1000 docs/s | ~100-200 docs/s |
| all-MiniLM-L6-v2 | 22M | <0.5GB | N/A | Excellent | ~1000-3000 docs/s | ~500-1400 docs/s |

**VRAM sizing formula**: `params_billions × bytes_per_param + KV_cache_overhead`. At Q4_K_M quantisation (~0.5 bytes/param), most embedding models (100M–1B) fit in 1–4GB.

**Practical hardware recommendations**:
- **Any Mac or laptop (8GB+ RAM)**: Nomic v1.5, all-MiniLM, Harrier 270M, Jina v5-nano, embeddinggemma-300m.
- **Mac with 16GB+ unified memory**: Above + BGE-M3, Snowflake Arctic, Jina v5-small, Qwen3-0.6B.
- **Linux with RTX 3060/4060 (12GB VRAM)**: All models up to ~1.5B; INT8-quantised 8B models.
- **Linux with RTX 3090/4090 (24GB VRAM)**: All models including Qwen3-8B at FP16.
- **Linux with A100/H100 (40--80GB)**: Production deployment of any model including Harrier 27B.

## Head-to-Head: Open-Source vs Proprietary

| Model | Type | MTEB Score | Version | Dims | Max Tokens | Params | Licence | Price/M tokens |
|-------|------|-----------|---------|------|------------|--------|---------|---------------|
| **Harrier-OSS-v1 27B** | Open | ~74.3 | v2 | -- | 32,768 | 27B | MIT | Free |
| **Jina v5-text-small** | Open | 71.7 | v2 | 1024 | -- | 677M | Self-host | Free |
| **Jina v5-text-nano** | Open | 71.0 | v2 | 1024 | -- | 239M | Self-host | Free |
| **Qwen3-Embedding-8B** | Open | 70.58 | MMTEB | 32-4096 | 32,768 | 8B | Apache 2.0 | Free |
| pplx-embed-v1-4B | Open | 69.66 | v2 multi | -- | 32,768 | 4B | MIT | Free |
| Voyage 3-large | API | 67.1 | v1 | 1024 | 32,000 | N/A | Proprietary | $0.18 |
| Cohere Embed v4 | API | 66.2 | v1 | 1024 | 128,000 | N/A | Proprietary | $0.10 |
| Jina v3 | API | 65.5 | v1 | 1024 | 8,192 | N/A | CC-BY-NC | $0.02 |
| OpenAI 3-large | API | 64.6 | v2 | 3072 | 8,191 | N/A | Proprietary | $0.13 |
| **BGE-M3** | Open | 63.0 | v2 | 1024 | 8,192 | 568M | MIT | Free |
| Mistral Embed | API | ~63 | v1 | 256-3072 | 8,192 | N/A | Proprietary | $0.10 |
| OpenAI 3-small | API | 62.3 | v2 | 1536 | 8,191 | N/A | Proprietary | $0.02 |
| **Nomic v1.5** | Open | ~62 | v1 | 768 | 8,192 | 137M | Apache 2.0 | Free |
| Gemini Embedding | API | 68.32 | v1 | 3072 | 2,048 | N/A | Proprietary | $0.15 |

**Note**: MTEB scores across versions (v1, v2, MMTEB) are **not directly comparable**. The table shows the version used. Within each version, relative rankings are meaningful.

**Key takeaways**:
- The top 5 models by MTEB v2 score are all open-source.
- Open-source models now definitively surpass proprietary APIs on benchmarks.
- **Jina v5-text-nano** (239M params, MTEB v2: 71.0) scores higher than every proprietary API on MTEB v2, while running on any laptop.

## Proprietary Embedding APIs (Hyperscaler-Available)

All proprietary models listed below are available through at least one major hyperscaler.

| Model | MTEB Score | Dims | Max Tokens | Price/M tokens | Available Via |
|-------|-----------|------|------------|---------------|--------------|
| **Gemini Embedding 2** | -- (multimodal) | 3072 (MRL) | -- | $0.15 (online), $0.12 (batch) | GCP Vertex AI |
| **Gemini Embedding 001** | 68.32 (v1) | 3072 (MRL) | 2,048 | $0.15 | GCP Vertex AI |
| **Voyage 3-large** | 67.1 | 1024 | 32,000 | $0.18 | Azure AI |
| **Voyage 3-lite** | 61.4 | 512 | 32,000 | $0.02 | Azure AI |
| **Cohere Embed v4** | 66.2 | 256--1536 (MRL) | 128,000 | $0.10--0.12 | AWS Bedrock, Azure AI, Oracle OCI |
| **OpenAI 3-large** | 64.6 (v2) | 3072 (MRL) | 8,191 | $0.13 ($0.065 batch) | Azure OpenAI |
| **OpenAI 3-small** | 62.3 (v2) | 1536 | 8,191 | $0.02 ($0.01 batch) | Azure OpenAI |
| **Mistral Embed** | ~63 | 256--3072 (MRL) | 8,192 | $0.10 | Azure AI, AWS Bedrock |
| **Amazon Titan V2** | -- | -- | 8,192 | $0.20 ($0.10 batch) | AWS Bedrock |
| **Google text-embedding-005** | 63.0 | 768 | 2,048 | ~$0.025/1K chars | GCP Vertex AI |
| **IBM Granite Embedding** | -- | -- | -- | $0.10 (flat) | IBM watsonx |

**Notable proprietary features** not yet matched by open-source:
- **Cohere Embed v4**: 128K context window, native multimodal (interleaved text+image), and multiple quantisation formats (float, int8, uint8, binary, ubinary). Supports fine-tuning with as few as 256 examples. Older Cohere models retired April 4, 2026.
- **Gemini Embedding 2**: Only model supporting 5 modalities (text, image, video, audio, PDF) in a single vector space. Best cross-lingual retrieval and long-document performance in independent benchmarks.
- **Voyage AI domain models**: Purpose-built models for finance, law, and code. Voyage Code 3 leads code retrieval at $0.12--$0.18/M tokens. MongoDB acquired Voyage AI for $220M in February 2025.
- **OpenAI batch API**: 50% discount; Cohere retires older models in favour of v4.

## Cost Analysis: Self-Hosting vs API

### API Costs at Scale

| Volume (tokens/month) | OpenAI Small ($0.02) | Mistral ($0.10) | Cohere v4 ($0.10) | OpenAI Large ($0.13) | Gemini ($0.15) | Voyage Large ($0.18) |
|----------------------|---------------------|----------------|-------------------|---------------------|---------------|---------------------|
| 10M | $0.20 | $1.00 | $1.00 | $1.30 | $1.50 | $1.80 |
| 100M | $2.00 | $10.00 | $10.00 | $13.00 | $15.00 | $18.00 |
| 1B | $20.00 | $100.00 | $100.00 | $130.00 | $150.00 | $180.00 |
| 10B | $200.00 | $1,000 | $1,000 | $1,300 | $1,500 | $1,800 |

### Self-Hosting Costs

- **Existing Mac/Linux machine (small models)**: $0 marginal cost. Nomic v1.5, all-MiniLM, Harrier 270M run on hardware you already have.
- **Nomic on CPU (c6i.xlarge)**: ~$125/month, ~500 passages/sec — nearly matching OpenAI 3-small quality at zero per-token cost.
- **Consumer GPU (RTX 4090)**: ~$1,500 one-time + ~$50/month electricity.
- **Cloud GPU (A100 spot)**: ~$1.50/hour = ~$1,080/month. BGE-M3 processes ~8,000 tokens/sec at ~$0.001/M tokens.

### Breakeven Analysis

Self-hosting becomes cheaper than APIs when:

| Comparison | Breakeven Volume | Source |
|-----------|-----------------|--------|
| vs OpenAI 3-large ($0.13/M) | ~3.6M tokens/month (reserved GPU) | Kanopy Labs |
| vs OpenAI 3-small ($0.02/M) | ~23.5M tokens/month | Kanopy Labs |
| vs any API (general) | ~10--15M embeddings/month | PE Collective |

**Hidden costs of self-hosting**: Raw GPU costs represent only 30--40% of true infrastructure investment. Plan a 2.5--3x multiplier for networking, storage, monitoring, redundancy, and engineering labour.

**Hidden cost of APIs**: Re-embedding. When switching models or updating chunking strategy, the entire corpus must be re-embedded. A 10M document corpus costs ~$3,400 to re-embed at OpenAI 3-large rates. Plan for 2--3 re-embedding cycles per year.

**Volume-based guidance:**
- Under 1M tokens/month: Use APIs.
- 1--10M: Use APIs, monitor costs.
- 10--50M: Evaluate hybrid (self-host primary, API burst).
- 50--100M: Self-host primary with API overflow.
- Over 100M: Self-host is mandatory for cost efficiency.

## Decision Framework

### Model selection quick guide

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Best quality (have GPU) | Harrier-OSS-v1 27B | #1 MTEB v2, MIT, 100+ languages |
| Best quality (no large GPU) | Jina v5-text-small | 71.7 MTEB v2, only 677M params |
| General RAG (Apache 2.0) | Qwen3-Embedding-8B/4B/0.6B | Top MTEB, Matryoshka, strong multilingual |
| Hybrid search (dense+sparse) | BGE-M3 | Only model with dense+sparse+ColBERT |
| Retrieval-focused (tiny) | Snowflake Arctic Embed | SOTA retrieval/size, Apache 2.0 |
| Multilingual (small) | Harrier-OSS-v1 0.6B or 270M | #1 family on MTEB v2 multilingual, MIT |
| Code search | Nomic Embed Code 7B | Beats Voyage Code 3, fully open |
| Full-document embedding | GTE-Qwen2-7B-instruct | 128K context, no chunking |
| Multimodal (text+image) | Qwen3-VL-Embedding-2B | Best cross-modal, open-source |
| Multimodal (5 modalities) | Gemini Embedding 2 (API) | Only 5-modality model |
| Edge/IoT/mobile | all-MiniLM-L6-v2 | 22M params, sub-10ms on CPU |
| Prototyping | Nomic v1.5 or Ollama defaults | Easy setup, runs anywhere |
| Maximum speed on CPU | all-MiniLM-L6-v2 + ONNX | 5--14K sentences/sec |
| Storage efficiency | pplx-embed | Native INT8/binary outputs |
| Best API value | Jina v3 ($0.02/M) | MTEB 65.5 at 1/9th Voyage price |

## Hyperscaler Managed Embedding Services

| Provider | Service | Models Available |
|----------|---------|-----------------|
| **AWS** | Amazon Bedrock, SageMaker | Cohere Embed v4, Titan Embeddings V2 ($0.20/M), Mistral Embed |
| **Azure** | Azure OpenAI Service, Azure AI | OpenAI text-embedding-3-*, Cohere Embed v4, Voyage 3.5 |
| **GCP** | Vertex AI | Gemini Embedding 001/2, text-embedding-005 |
| **IBM** | watsonx.ai | Granite Embedding (107M/278M multilingual), Slate (125M/30M), all at $0.10/M flat |
| **Oracle** | OCI Generative AI | Cohere Embed v4 (character-based pricing) |

All hyperscalers also support self-hosting open-weight models on their GPU instances.

## Context-Length Performance

MTEB scores measure average performance but hide how performance changes with input length. Independent benchmarking reveals different degradation curves.

### Passkey retrieval accuracy by context length

| Model | Best pooling | 512 tokens | 2K tokens | 4K tokens | 8K tokens |
|-------|-------------|-----------|----------|----------|----------|
| **Qwen3-Embedding-0.6B** | LAST | 1.00 | 0.94 | 1.00 | 1.00 |
| **BGE-M3** | CLS | 1.00 | 0.80 | 0.32 | 0.34 |
| **Jina-Embeddings-v3** | MEAN | 1.00 | 0.92 | 0.36 | 0.40 |

### Needle-in-a-haystack (Cheney Zhang benchmark)

| Model | 1K | 4K | 8K | 16K | 32K | Overall |
|-------|-----|-----|-----|------|------|---------|
| Gemini Embed 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| OpenAI 3-large | 1.000 | 1.000 | 1.000 | — | — | **1.000** |
| Jina v4 | 1.000 | 1.000 | 1.000 | — | — | **1.000** |
| Cohere v4 | 1.000 | 1.000 | 1.000 | — | — | **1.000** |
| BGE-M3 | 1.000 | 1.000 | **0.920** | — | — | 0.973 |
| mxbai (335M) | 0.980 | **0.600** | **0.400** | — | — | 0.660 |
| nomic (137M) | 1.000 | **0.460** | **0.440** | — | — | 0.633 |

"—" = exceeds context window or untested.

**Pooling strategy matters enormously.** Using the wrong pooling can cut performance by 50%+. Always use the model's recommended pooling: CLS for BGE-M3, LAST for Qwen3, MEAN for Jina v3.

## Matryoshka Embeddings and Quantisation

**Matryoshka Representation Learning (MRL)** is now supported by most major models, enabling 50--90% storage reduction by truncating embedding dimensions at inference time with minimal quality loss.

| Model | Full Dims | MRL Support | Compression Quality (256d) |
|-------|----------|-------------|---------------------------|
| Voyage MM-3.5 | 1024 | Yes | ρ=0.874 (0.7% loss) |
| Jina v4 | 2048 | Yes | ρ=0.828 (0.6% loss) |
| OpenAI 3-large | 3072 | Yes | ρ=0.762 (0.6% loss) |
| Nomic v1.5 | 768 | Yes (768/512/256/128/64) | ρ=0.774 (0.8% loss) |
| Gemini Embedding 2 | 3072 | Yes | ρ=0.689 (-0.8% — last place) |
| Cohere Embed v4 | 1024 | Yes (256--1536) | — |
| Mistral Embed | 1024 | Yes (256--3072) | — |

**Quantisation developments:**
- **Cohere Embed v4** and **Voyage Code 3** support native float/int8/uint8/binary/ubinary at the API level.
- **pplx-embed** (Perplexity, Feb 2026) outputs INT8 and binary embeddings natively, reducing storage 4x and 32x vs FP32, via quantisation-aware training.
- **MatQuant** (Google, Feb 2025, co-authored by Jeff Dean): Extends Matryoshka nesting from dimensions to model weight quantisation. A single INT8 model can be served at INT4 or INT2, with INT2 outperforming standard INT2 quantisation by 4--7%.
- Combining binary quantisation with Matryoshka dimension reduction can cut vector search costs by ~80%.

## Areas of Uncertainty

- **MTEB v1 vs v2 incompatibility**: Scores across versions are not comparable. The existing leaderboard mixes v1 and v2 scores, making model comparisons confusing. Harrier-OSS-v1 27B's 74.3 (v2) and NV-Embed-v2's 72.31 (legacy v1) are on different scales.
- **MTEB score reliability**: All scores are self-reported. No independent verification exists. Rankings are volatile and shift weekly.
- **Harrier-OSS-v1 documentation gap**: Released without a research paper or detailed model card. Architecture details and smaller-variant benchmark scores are sparse.
- **Self-hosting breakeven**: Estimates range from 3.6M to 23.5M tokens/month depending on which API you're replacing and whether you count engineering labour (2.5--3x multiplier on raw GPU costs).
- **Multimodal embedding quality**: Most benchmarks focus on text and image. Audio and video embedding quality is poorly evaluated. MMEB's easy distractors make it hard to differentiate models on fine-grained understanding.
- **Apple Silicon performance**: MLX speed claims (3x over llama.cpp) are from limited benchmarks on MoE models specifically. Embedding-specific throughput comparisons across M1–M5 are sparse.
- **Throughput benchmarks**: No standardised, independently verified throughput benchmarks for embedding models exist. Speeds in this article are approximate.
- **Jina v5 licensing**: The exact licence terms for commercial self-hosting should be verified directly with Jina AI.
- **Real-world vs MTEB performance**: A 5-point MTEB gap translates to roughly 3--8% better recall@10 in practice, but this varies significantly by domain. Domain-specific models routinely outperform higher-MTEB general models on their target domains.

## References

1. [MTEB Leaderboard - HuggingFace](https://huggingface.co/spaces/mteb/leaderboard) - Official MTEB benchmark leaderboard
2. [Microsoft Open-Sources Harrier-OSS-v1 (Bing Blog)](https://blogs.bing.com/search/April-2026/Microsoft-Open-Sources-Industry-Leading-Embedding-Model) - Official Harrier announcement, #1 MTEB v2
3. [Gemini Embedding 2 (Google Blog)](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/) - Official 5-modality multimodal embedding model announcement
4. [jina-embeddings-v5-text-small (HuggingFace)](https://huggingface.co/jinaai/jina-embeddings-v5-text-small) - Jina v5 model card with MTEB v2 scores
5. [Qwen3-VL-Embedding (arXiv 2601.04720)](https://arxiv.org/abs/2601.04720) - Multimodal embedding and reranker technical report
6. [Perplexity pplx-embed Launch (Insights)](https://insights.marvin-42.com/articles/perplexity-launches-pplx-embed-family-for-web-scale-retrieval-with-int8-and-binary-outputs) - Native INT8/binary embedding outputs
7. [Embedding Model Benchmark 2026 (Cheney Zhang)](https://zc277584121.github.io/rag/2026/03/20/embedding-models-benchmark-2026.html) - Independent 10-model cross-modal/cross-lingual/needle benchmark
8. [MLX: The Next Inference Engine for Apple Silicon](https://yage.ai/share/mlx-apple-silicon-en-20260331.html) - Ollama v0.19 MLX switch analysis
9. [Sentence Transformers v5.4 Documentation](https://www.sbert.net/) - Multimodal support, Flash Attention 2
10. [Nomic Embed Code Announcement](https://www.nomic.ai/news/introducing-state-of-the-art-nomic-embed-code) - Open-source code embedding beating Voyage
11. [Cohere Embed v4 Changelog](https://docs.cohere.com/changelog/embed-multimodal-v4) - 128K context, multimodal, quantisation formats
12. [Amazon Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) - Official AWS embedding pricing
13. [GCP Vertex AI Pricing](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing) - Official Google embedding pricing
14. [IBM watsonx.ai Pricing](https://www.ibm.com/products/watsonx-ai/pricing) - Official IBM embedding pricing ($0.10/M flat)
15. [Text Embedding Models Compared 2026 (PE Collective)](https://pecollective.com/tools/text-embedding-models-compared/) - April 2026 pricing comparison
16. [Embedding Models Compared (Kanopy Labs)](https://kanopylabs.com/blog/embedding-models-compared) - Self-hosting breakeven analysis
17. [Top Embedding Models on MTEB (Modal)](https://modal.com/blog/mteb-leaderboard-article) - Guide to reading MTEB scores and task-specific selection
18. [Jina Embeddings V4 (Jina AI)](https://jina.ai/models/jina-embeddings-v4/) - 3.8B multimodal model with ColBERT support
19. [NVIDIA Omni-Embed-Nemotron (HuggingFace)](https://huggingface.co/nvidia/omni-embed-nemotron-3b) - 4.7B four-modality embedding model
20. [Matryoshka Quantization (arXiv 2502.06786)](https://arxiv.org/abs/2502.06786) - Multi-scale quantisation from Jeff Dean et al.
21. [Mistral Embed Pricing (CloudPrice)](https://cloudprice.net/models/mistral%2Fmistral-embed) - $0.10/M tokens, MRL support
22. [MMTEB: Massive Multilingual Text Embedding Benchmark (arXiv 2502.13595)](https://arxiv.org/abs/2502.13595) - 500+ tasks, 250+ languages
23. [Qwen3-Embedding (GitHub)](https://github.com/QwenLM/Qwen3-Embedding) - Official repository, 0.6B/4B/8B variants
24. [Snowflake Arctic Embed (GitHub)](https://github.com/Snowflake-Labs/arctic-embed) - Retrieval-focused embedding suite
25. [Microsoft Harrier-OSS-v1 Analysis (RevolutionInAI)](https://www.revolutioninai.com/2026/04/microsoft-harrier-oss-v1-multilingual-embedding-model-2026.html) - Built on Gemma architecture analysis
26. [Voyage Code 3 (Voyage AI Blog)](https://blog.voyageai.com/2024/12/04/voyage-code-3/) - Code retrieval with Matryoshka embeddings
27. [Comprehensive Embedding Models Evaluation](https://x22x22.github.io/embedding_models_comprehensive_evaluation.html) - Context-length performance comparison
28. [6 Best Code Embedding Models Compared (Modal)](https://modal.com/blog/6-best-code-embedding-models-compared) - Code embedding model comparison
