# Embedding Models: Best Open-Source Local Models and How They Compare with Proprietary Alternatives

| Field | Value |
|-------|-------|
| Created | 2026-03-19 |
| Last Updated | 2026-03-20 |
| Version | 2.1 |

---

- [Executive Summary](#executive-summary)
- [Understanding MTEB Benchmarks](#understanding-mteb-benchmarks)
- [Best Open-Source Embedding Models for Local Deployment](#best-open-source-embedding-models-for-local-deployment)
  - [Tier 1: Large Models (7B+)](#tier-1-large-models-7b)
  - [Tier 2: Medium Models (300M--1.5B)](#tier-2-medium-models-300m15b)
  - [Tier 3: Small/Edge Models (<300M)](#tier-3-smalledge-models-300m)
- [Running Embedding Models on Linux](#running-embedding-models-on-linux)
- [Running Embedding Models on macOS](#running-embedding-models-on-macos)
- [Inference Frameworks](#inference-frameworks)
- [Hardware Requirements](#hardware-requirements)
- [Head-to-Head: Open-Source vs Proprietary](#head-to-head-open-source-vs-proprietary)
- [Proprietary Embedding APIs (Hyperscaler-Available)](#proprietary-embedding-apis-hyperscaler-available)
- [Cost Analysis: Self-Hosting vs API](#cost-analysis-self-hosting-vs-api)
- [Decision Framework](#decision-framework)
- [Hyperscaler Managed Embedding Services](#hyperscaler-managed-embedding-services)
- [Context-Length Performance: BGE-M3 vs Alternatives](#context-length-performance-bge-m3-vs-alternatives)
- [Areas of Uncertainty](#areas-of-uncertainty)
- [References](#references)

## Executive Summary

Open-source embedding models have reached parity with --- and in several benchmark categories surpassed --- proprietary frontier models. The best local models (Qwen3-Embedding-8B, NV-Embed-v2, BGE-en-ICL) score higher than all proprietary APIs on the legacy 56-task MTEB leaderboard, while Google's Gemini Embedding 001 leads the refreshed MTEB v2 English leaderboard. Organisations no longer need to sacrifice quality to run embeddings locally.

For most production RAG pipelines, a mid-sized open-source model like **BGE-M3** (568M params, MIT licence) or **Snowflake Arctic Embed** (334M params, Apache 2.0) delivers excellent retrieval performance while running on a single consumer GPU or even CPU. On Apple Silicon Macs, models up to ~1.5B parameters run comfortably using Metal acceleration, and smaller models run well on CPU alone.

This article focuses on open-source models you can run on a Linux workstation or Mac, with proprietary APIs covered briefly as a baseline for comparison.

## Understanding MTEB Benchmarks

The Massive Text Embedding Benchmark (MTEB) is the standard framework for evaluating embedding models, covering eight task categories: retrieval, classification, clustering, semantic textual similarity (STS), pair classification, reranking, summarisation, and bitext mining.

**Critical caveat: MTEB version differences.** Multiple leaderboards exist with different task sets:

- **MTEB v2 (refreshed English)**: Newer tasks; Gemini Embedding 001 leads at 68.32 average.
- **Legacy MTEB (56 tasks)**: The original benchmark; NV-Embed-v2 scores 72.31, BGE-en-ICL scores 71.24.
- **MMTEB (multilingual)**: 131 tasks across 250+ languages; Qwen3-Embedding-8B scores 70.58.

Scores are **not directly comparable** across versions. All MTEB scores are self-reported by model developers with no independent verification. Focus on task-specific scores (e.g. retrieval) rather than overall averages for your use case.

## Best Open-Source Embedding Models for Local Deployment

### Tier 1: Large Models (7B+)

These models deliver the highest benchmark scores but require a discrete GPU with 16GB+ VRAM (or 8GB+ with quantisation). Not practical for CPU-only deployment.

#### Qwen3-Embedding-8B (Alibaba) --- Best Overall

- **MTEB**: 70.58 (MMTEB multilingual), 80.68 (MTEB Code)
- **Parameters**: 8B (also available in **4B** and **0.6B** variants)
- **Dimensions**: 32 to 4,096+ (Matryoshka, flexible)
- **Max tokens**: 32,768
- **Licence**: Apache 2.0
- **VRAM**: ~16GB (FP16), ~8GB (INT8 quantised)
- **Local deployment**: sentence-transformers, vLLM, Ollama (`ollama pull qwen3-embedding`)
- **Why choose it**: Best overall open-source model. Apache 2.0 allows commercial use. Excellent multilingual and code performance. The **0.6B variant** is a practical option for machines without a large GPU --- it fits easily in 2GB of VRAM or runs on CPU.
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
- **Why choose it**: Highest overall MTEB score. Strong across all task categories.
- **Limitations**: **Non-commercial licence** severely limits production use. Fixed (non-Matryoshka) dimensions. Relatively weak on retrieval vs overall score.

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

#### GTE-Qwen2-7B-instruct (Alibaba) --- Longest Context

- **MTEB (legacy)**: 70.24
- **Parameters**: ~7.6B
- **Dimensions**: Up to 3,584
- **Max tokens**: 128,000
- **Licence**: Apache 2.0
- **VRAM**: ~30.5GB (FP16), ~16GB (quantised)
- **Local deployment**: sentence-transformers, vLLM
- **Why choose it**: 128K context window --- the longest of any open-source embedding model. Embeds full documents without chunking. Strong English and Chinese performance.
- **Limitations**: Very high VRAM requirement at full precision; needs quantisation on consumer GPUs.

#### llama-embed-nemotron-8b (NVIDIA) --- Top Multilingual (Non-Commercial)

- **Parameters**: 8B (fine-tuned from Llama-3.1-8B, released October 2025)
- **Licence**: Non-commercial (NVIDIA community licence)
- **Local deployment**: sentence-transformers, vLLM
- **Why choose it**: Top multilingual MTEB scores.
- **Limitations**: **Non-commercial licence.** Requires significant VRAM.

### Tier 2: Medium Models (300M--1.5B)

The sweet spot for local deployment. These models run on consumer GPUs (4--8GB VRAM) and several work on CPU. They cover the vast majority of production use cases.

#### BGE-M3 (BAAI) --- Best All-Rounder

- **MTEB v2 English avg**: ~63.0
- **Parameters**: 568M
- **Dimensions**: 1024
- **Max tokens**: 8,192
- **Licence**: MIT
- **Retrieval modes**: Dense, sparse, and multi-vector (ColBERT-style) --- all from a single model
- **VRAM**: ~2GB (GPU), runs on CPU with ~3GB RAM
- **Local deployment**: sentence-transformers, FastEmbed, Ollama, ONNX
- **Why choose it**: The only major open-source model offering all three retrieval modes (dense, sparse, multi-vector) in one model. MIT licence. Multilingual (100+ languages). Runs comfortably on consumer hardware including Mac. Excellent for hybrid search pipelines combining keyword and semantic search.
- **Context-length performance**: BGE-M3 excels at short contexts but degrades sharply beyond ~2K tokens. Independent benchmarking shows passkey retrieval scores of 1.0 at 512 tokens, 0.8 at 2K, but dropping to 0.32 at 4K and 0.34 at 8K — despite the 8192-token advertised maximum. For documents longer than ~2K tokens, consider Qwen3-Embedding-0.6B (which maintains ≥94% accuracy up to 8K with LAST pooling) or chunking to keep inputs under 2K tokens.
- **Pooling strategy**: Use CLS pooling with BGE-M3 (not MEAN or LAST). Pooling choice has a dramatic impact — the wrong pooling strategy can cut performance by 50%+.
- **Limitations**: Lower MTEB scores than 7B+ models. Dense-only retrieval is less competitive. Performance degrades significantly on longer inputs.

#### stella_en_1.5B_v5 --- Best Mid-Size English-Only

- **Parameters**: 1.5B
- **Dimensions**: 1024 (Matryoshka)
- **Licence**: MIT
- **VRAM**: ~3GB (GPU), viable on CPU
- **Local deployment**: sentence-transformers, ONNX
- **Why choose it**: Good balance of size and performance for English-only use cases. Matryoshka support allows dimension reduction without retraining.

#### Snowflake Arctic Embed (Snowflake Labs) --- Best Retrieval per Parameter

- **Parameters**: Up to 334M (multiple size variants: xs/s/m/l)
- **Dimensions**: 1024
- **Licence**: Apache 2.0
- **VRAM**: ~1.5GB (largest variant), smaller variants run on CPU
- **Local deployment**: sentence-transformers, FastEmbed, Ollama (`ollama pull snowflake-arctic-embed`), NVIDIA NIM
- **Why choose it**: SOTA retrieval performance per size class on MTEB/BEIR. The 334M model competes with models 20x its size on retrieval benchmarks. v2.0 adds multilingual support. Runs on a Mac without issues.
- **Limitations**: Optimised primarily for retrieval; less competitive on classification/clustering.

#### Jina Embeddings v3 / v4 (Jina AI) --- Best Multimodal Local Option

- **Jina v3 MTEB**: ~62 (text-only, 65.52 per some sources)
- **Jina v4 MTEB retrieval rank**: #5
- **Parameters**: Moderate (smallest among top-5 MTEB retrieval peers)
- **Dimensions**: Flexible (Matryoshka)
- **Licence**: v3 available for self-hosting (CC-BY-NC-4.0); v4 available in GGUF format
- **Local deployment**: sentence-transformers (v3), llama.cpp/GGUF (v4), Jina API
- **Why choose it**: v3 uses task-specific LoRA adapters for per-task optimisation at inference time. v4 is multimodal (text + visual documents), scoring 72.19 on JinaVDR and 84.11 on ViDoRe benchmarks. Compact for its performance class. GGUF format makes v4 easy to run locally via llama.cpp.
- **Limitations**: v3 has a non-commercial licence. v4 benchmark claims are self-reported.

#### NVIDIA Omni-Embed-Nemotron-3B --- Only Four-Modality Model

- **Parameters**: 3B
- **Multimodal**: Text, image, audio, and video encoding
- **Licence**: Non-commercial
- **VRAM**: ~6GB (FP16)
- **Local deployment**: sentence-transformers
- **Why choose it**: The only embedding model supporting four modalities. Designed for multi-modal RAG.
- **Limitations**: **Non-commercial licence.** Relatively new with limited independent benchmarking.

### Tier 3: Small/Edge Models (<300M)

These models run on any machine --- including laptops, Raspberry Pis, and CI/CD servers. They need no GPU and minimal RAM.

#### embeddinggemma-300m (Google) --- Best Small Multilingual

- **Parameters**: 300M (built on Gemma 3 and T5Gemma)
- **Dimensions**: 768
- **Languages**: 100+
- **Licence**: Apache 2.0
- **RAM**: ~1.5GB
- **Local deployment**: sentence-transformers, Ollama
- **Why choose it**: Lightweight with strong multilingual coverage. Designed for on-device and edge deployment. Runs on CPU comfortably. Apache 2.0 licensed.

#### Nomic embed-text-v1.5 --- Best Fully Open Model

- **Parameters**: 137M
- **Dimensions**: 768 (Matryoshka: 768/512/256/128/64)
- **MTEB**: ~62
- **Max tokens**: 8,192
- **Licence**: Apache 2.0
- **RAM**: ~1GB
- **Local deployment**: sentence-transformers, Ollama (`ollama pull nomic-embed-text`), FastEmbed, ONNX
- **Why choose it**: Fully open --- code, training data, and weights all published. Runs on CPU. Matryoshka support for flexible dimensions. Long 8K context window for its size. Excellent Ollama integration makes it trivial to deploy on Mac or Linux.

#### all-MiniLM-L6-v2 --- Fastest, Smallest

- **Parameters**: 22M
- **Dimensions**: 384
- **MTEB**: ~56.3
- **Max tokens**: 256 (optimal), 512 (max)
- **Licence**: Apache 2.0
- **RAM**: <0.5GB
- **Local deployment**: sentence-transformers (default model), FastEmbed, Ollama (`ollama pull all-minilm`), ONNX
- **Why choose it**: Sub-10ms inference on CPU. 5--14K sentences/sec on CPU. Tiny footprint. Ideal for prototyping, edge/IoT/mobile, and pipelines where embedding is not the bottleneck. The default model in sentence-transformers.
- **Limitations**: Significantly lower accuracy than larger models. English only. Short context.

#### bge-small-en-v1.5 / bge-base-en-v1.5 (BAAI)

- **Parameters**: 33M (small) / 110M (base)
- **Dimensions**: 384 (small) / 768 (base)
- **MTEB**: ~62 (small), ~64 (base)
- **Licence**: MIT
- **RAM**: ~0.5GB (small), ~1.2GB (base)
- **Local deployment**: sentence-transformers, FastEmbed (default model is bge-small), ONNX
- **Why choose it**: Higher accuracy than all-MiniLM with only modest size increase. bge-small is FastEmbed's default model. MIT licence.

## Running Embedding Models on Linux

Linux is the most straightforward platform for local embedding models. All frameworks have first-class Linux support.

### GPU (NVIDIA CUDA)

Most embedding frameworks assume NVIDIA GPUs with CUDA on Linux.

```bash
# Install PyTorch with CUDA support
pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# Install sentence-transformers
pip3 install sentence-transformers

# Run any model
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

PyTorch supports AMD GPUs via ROCm on Linux. sentence-transformers works with ROCm-enabled PyTorch:

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/rocm6.0
pip3 install sentence-transformers
```

### CPU-only

For models under ~500M parameters, CPU inference is practical:

```bash
# ONNX backend for 2-3x speedup over default PyTorch
pip3 install "sentence-transformers[onnx]"
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3', backend='onnx')
emb = model.encode(['Hello world'])
"
```

Or use FastEmbed for the simplest CPU-first setup:

```bash
pip3 install fastembed
python3 -c "
from fastembed import TextEmbedding
model = TextEmbedding('BAAI/bge-small-en-v1.5')
emb = list(model.embed(['Hello world']))
"
```

### Ollama (simplest for any model)

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull nomic-embed-text
curl http://localhost:11434/api/embeddings \
  -d '{"model": "nomic-embed-text", "prompt": "Hello world"}'
```

Ollama embedding models available: `nomic-embed-text`, `mxbai-embed-large`, `snowflake-arctic-embed`, `all-minilm`, `qwen3-embedding`.

## Running Embedding Models on macOS

Apple Silicon Macs (M1/M2/M3/M4) can run embedding models efficiently using Metal acceleration for GPU-backed inference, or CPU for smaller models.

### Apple Silicon GPU (Metal / MPS)

PyTorch supports Apple's Metal Performance Shaders (MPS) backend. Models up to ~1.5B parameters fit in unified memory on a 16GB Mac; 8B models require a 32GB+ Mac.

```bash
# Install PyTorch (Metal support included by default on macOS)
pip3 install torch sentence-transformers

# sentence-transformers automatically uses MPS when available
python3 -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
emb = model.encode(['Hello world'], normalize_embeddings=True)
print(f'Dimensions: {emb.shape[1]}')
"
```

### Ollama (recommended for Mac)

Ollama is the easiest way to run embedding models on macOS. It uses Metal acceleration automatically on Apple Silicon.

```bash
brew install ollama
ollama serve &
ollama pull nomic-embed-text       # 137M, runs on any Mac
ollama pull snowflake-arctic-embed # 334M, runs on any Mac
ollama pull qwen3-embedding        # 0.6B/4B/8B variants available
```

### CPU-only (Intel or Apple Silicon)

For models under ~300M parameters, CPU inference is fast enough for most use cases on any Mac:

```bash
pip3 install fastembed
python3 -c "
from fastembed import TextEmbedding
model = TextEmbedding('BAAI/bge-small-en-v1.5')
emb = list(model.embed(['Hello world']))
"
```

### What fits on which Mac

| Mac | Unified Memory | Recommended Models |
|-----|---------------|-------------------|
| M1/M2 (8GB) | 8GB | all-MiniLM, bge-small/base, Nomic v1.5, embeddinggemma-300m |
| M1/M2/M3 (16GB) | 16GB | Above + BGE-M3, Snowflake Arctic, stella_en_1.5B, Qwen3-0.6B |
| M1/M2/M3 Pro/Max (32GB) | 32GB | Above + Qwen3-4B, INT8 quantised 8B models |
| M2/M3/M4 Ultra (64GB+) | 64GB+ | Any model at full precision including 8B FP16 |

## Inference Frameworks

| Framework | Best For | GPU | CPU | Mac | Install |
|-----------|---------|-----|-----|-----|---------|
| **sentence-transformers** | General use, widest model support | CUDA, MPS | Yes | Yes | `pip3 install sentence-transformers` |
| **Ollama** | Simplest local setup, Mac-friendly | CUDA, Metal | Yes | Yes | `brew install ollama` or curl script |
| **FastEmbed** | CPU-first, lightweight | No | Yes | Yes | `pip3 install fastembed` |
| **ONNX Runtime** | CPU speedup (2--3x over PyTorch) | CUDA | Yes | Yes | `pip3 install "sentence-transformers[onnx]"` |
| **HuggingFace TEI** | Production GPU serving | CUDA | No | No | Docker container |
| **vLLM** | High-throughput GPU serving (7B+) | CUDA | No | No | `pip3 install vllm` |
| **llama.cpp / GGUF** | Quantised CPU inference | CUDA, Metal | Yes | Yes | Build from source or use Ollama |
| **Model2Vec** | Ultra-fast static embeddings | No | Yes | Yes | `pip3 install model2vec` |

### sentence-transformers

The go-to library. Supports all HuggingFace models, handles tokenisation/batching/normalisation automatically. Supports PyTorch, ONNX, and OpenVINO backends.

```python
from sentence_transformers import SentenceTransformer

# Default (PyTorch, auto-selects GPU/MPS/CPU)
model = SentenceTransformer("BAAI/bge-m3")

# ONNX backend for 2-3x CPU speedup
model = SentenceTransformer("BAAI/bge-m3", backend="onnx")

embeddings = model.encode(["Your text here"], normalize_embeddings=True)
```

### FastEmbed (Qdrant)

CPU-first embedding library using ONNX Runtime. Default model is `bge-small-en-v1.5`. Significantly faster than default HuggingFace Transformers inference on CPU.

```python
from fastembed import TextEmbedding
model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = list(model.embed(["Your text here"]))
```

### Model2Vec

Converts sentence transformer models into static embedding models with up to 50x size reduction and 500x speed improvement, at the cost of some quality. Useful for extremely latency-sensitive or resource-constrained scenarios.

## Hardware Requirements

| Model | Parameters | VRAM (FP16) | VRAM (INT8) | CPU-only? | Approx Speed (GPU) | Approx Speed (CPU) |
|-------|-----------|-------------|-------------|-----------|--------------------|--------------------|
| Qwen3-Embedding-8B | 8B | ~16GB | ~8GB | Slow | ~50-100 docs/s | ~5-10 docs/s |
| Qwen3-Embedding-4B | 4B | ~8GB | ~4GB | Slow | ~80-150 docs/s | ~10-20 docs/s |
| Qwen3-Embedding-0.6B | 0.6B | ~2GB | ~1GB | Viable | ~200-400 docs/s | ~50-100 docs/s |
| NV-Embed-v2 | 7.85B | 16GB+ | ~8GB | Impractical | ~50-100 docs/s | N/A |
| BGE-en-ICL | ~7B | ~14GB | ~7GB | Impractical | ~50-100 docs/s | N/A |
| GTE-Qwen2-7B | 7.6B | ~30.5GB | ~16GB | Impractical | ~50-100 docs/s | N/A |
| stella_en_1.5B_v5 | 1.5B | ~3GB | ~1.5GB | Viable | ~150-300 docs/s | ~30-60 docs/s |
| BGE-M3 | 568M | ~2GB | ~1GB | Viable | ~200-500 docs/s | ~40-80 docs/s |
| Snowflake Arctic (L) | 334M | ~1.5GB | <1GB | Viable | ~300-700 docs/s | ~60-120 docs/s |
| embeddinggemma-300m | 300M | ~1.5GB | <1GB | Good | ~300-700 docs/s | ~60-120 docs/s |
| Nomic v1.5 | 137M | <1GB | N/A | Good | ~500-1000 docs/s | ~100-200 docs/s |
| bge-base-en-v1.5 | 110M | ~2.1GB | N/A | Good | ~400-800 docs/s | ~80-150 docs/s |
| all-MiniLM-L6-v2 | 22M | <0.5GB | N/A | Excellent | ~1000-3000 docs/s | ~500-1400 docs/s |

**Practical hardware recommendations**:
- **Any Mac or Linux laptop (8GB+ RAM)**: Nomic v1.5, all-MiniLM, bge-small/base, embeddinggemma-300m.
- **Mac with 16GB+ unified memory**: Above + BGE-M3, Snowflake Arctic, stella_en_1.5B, Qwen3-0.6B.
- **Linux with RTX 3060/4060 (12GB VRAM)**: All models up to ~1.5B; INT8-quantised 8B models.
- **Linux with RTX 3090/4090 (24GB VRAM)**: All models including Qwen3-8B and NV-Embed-v2 at FP16.
- **Linux with A100/H100 (40--80GB)**: Production deployment of large models with high-throughput batch processing.

## Head-to-Head: Open-Source vs Proprietary

| Model | Type | MTEB Score | Dims | Max Tokens | Params | Licence | Price/M tokens |
|-------|------|-----------|------|------------|--------|---------|---------------|
| **NV-Embed-v2** | Open | 72.31 (legacy) | 4096 | 32,768 | 7.85B | CC-BY-NC | Free |
| **BGE-en-ICL** | Open | 71.24 (legacy) | 4096 | 32,768 | ~7B | MIT | Free |
| **Qwen3-Embedding-8B** | Open | 70.58 (MMTEB) | 32-4096 | 32,768 | 8B | Apache 2.0 | Free |
| **GTE-Qwen2-7B** | Open | 70.24 (legacy) | 3584 | 128,000 | 7.6B | Apache 2.0 | Free |
| Gemini Embedding 001 | API | 68.32 (v2) | 3072 | 2,048 | N/A | Proprietary | $0.15 |
| Voyage-4 | API | ~67 | 1024 | 32,000 | N/A | Proprietary | $0.06 |
| Cohere Embed v4 | API | 65.2 (v2) | 1024 | 128,000 | N/A | Proprietary | $0.12 |
| OpenAI 3-large | API | 64.60 (v2) | 3072 | 8,191 | N/A | Proprietary | $0.13 |
| **BGE-M3** | Open | 63.0 (v2) | 1024 | 8,192 | 568M | MIT | Free |
| Mistral Embed | API | ~63 | 1024 | 8,192 | N/A | Proprietary | $0.01 |
| **Nomic v1.5** | Open | ~62 | 768 | 8,192 | 137M | Apache 2.0 | Free |
| OpenAI 3-small | API | 62.26 (v2) | 1536 | 8,191 | N/A | Proprietary | $0.02 |
| **Snowflake Arctic** | Open | SOTA/size | 1024 | 512 | 334M | Apache 2.0 | Free |
| **all-MiniLM-L6-v2** | Open | 56.3 | 384 | 512 | 22M | Apache 2.0 | Free |

**Key takeaways**:
- The top 4 models by MTEB score are all open-source.
- **Qwen3-Embedding-8B** offers the best combination of performance, commercial licence (Apache 2.0), and flexibility.
- **BGE-M3** matches or beats OpenAI 3-small and Mistral Embed while running on a laptop --- with no API costs and no data leaving your machine.
- For retrieval-focused use cases, **Snowflake Arctic Embed** delivers remarkable performance per parameter.

## Proprietary Embedding APIs (Hyperscaler-Available)

All proprietary models listed below are available through at least one major hyperscaler (AWS, Azure, GCP, IBM, or Oracle). They serve as baselines for comparison and are relevant when self-hosting is not feasible.

| Model | MTEB Score | Dims | Max Tokens | Price/M tokens | Available Via |
|-------|-----------|------|------------|---------------|--------------|
| **Gemini Embedding 001** | 68.32 (v2) | 3072 (Matryoshka) | 2,048 | $0.15 | GCP Vertex AI |
| **Voyage-4-large** | ~67+ | 2048 | 32,000 | $0.12 | Azure AI |
| **Voyage-4** | ~67 | 1024 | 32,000 | $0.06 | Azure AI |
| **Cohere Embed v4** | 65.2 (v2) | 1024 (Matryoshka) | 128,000 | $0.12 | AWS Bedrock, Azure AI, Oracle OCI |
| **OpenAI 3-large** | 64.60 (v2) | 3072 (Matryoshka) | 8,191 | $0.13 | Azure OpenAI |
| **Mistral Embed** | ~63 | 1024 | 8,192 | $0.01 | Azure AI, AWS Bedrock |
| **OpenAI 3-small** | 62.26 (v2) | 1536 | 8,191 | $0.02 | Azure OpenAI |

**Notable proprietary features** not yet matched by open-source:
- **Cohere Embed v4**: 128K token context window (embeds full documents without chunking) and native multimodal (text + images).
- **Voyage AI domain models**: Purpose-built models for finance, law, and code at $0.12--$0.18/M tokens.
- **Batch API discounts**: OpenAI offers 50% off via batch API; Voyage offers ~33% off.

## Cost Analysis: Self-Hosting vs API

### API Costs at Scale

| Volume (tokens/month) | Mistral ($0.01/M) | OpenAI Small ($0.02/M) | Voyage-4 ($0.06/M) | Cohere v4 ($0.12/M) | OpenAI Large ($0.13/M) | Gemini ($0.15/M) |
|----------------------|-------------------|----------------------|-------------------|-------------------|----------------------|-----------------|
| 10M | $0.10 | $0.20 | $0.60 | $1.20 | $1.30 | $1.50 |
| 100M | $1.00 | $2.00 | $6.00 | $12.00 | $13.00 | $15.00 |
| 1B | $10.00 | $20.00 | $60.00 | $120.00 | $130.00 | $150.00 |
| 10B | $100.00 | $200.00 | $600.00 | $1,200.00 | $1,300.00 | $1,500.00 |

### Self-Hosting Costs

- **Existing Mac/Linux machine (small models)**: $0. Nomic v1.5, all-MiniLM, bge-small run on hardware you already have.
- **Consumer GPU (RTX 4090)**: ~$1,500 one-time + ~$50/month electricity. Runs any model up to 8B params.
- **Cloud GPU (A10G on AWS)**: ~$0.75/hour = ~$540/month (on-demand) or ~$200/month (reserved).

### Breakeven Analysis

Self-hosting becomes cheaper than APIs when:
- Processing **>1B tokens/month** using expensive APIs (OpenAI Large, Gemini)
- Processing **>10B tokens/month** using cheap APIs (OpenAI Small, Mistral)
- **For small models on existing hardware**: Immediately cheaper --- no additional cost beyond electricity.

Running NV-Embed-v2 on a single A100 costs roughly $1--2/hour, translating to ~$0.001/M tokens --- 10--20x cheaper than the cheapest commercial API at full utilisation. Multiple analyses place the full-TCO breakeven (including DevOps overhead and idle time) between 500M and 11B tokens/month.

**Hidden cost: vector storage.** Higher-dimensional embeddings require proportionally more storage. At 100M documents: 3072-d vectors (OpenAI large, Gemini) require ~1.2TB, while 1024-d vectors (BGE-M3, Cohere) require ~400GB --- a 3x difference in vector database costs.

## Decision Framework

### Run locally on your existing Mac/Linux machine when:
- Using a small/medium model (up to ~1.5B params on GPU, ~300M on CPU)
- Data privacy or air-gapped deployment is required
- Latency matters (15--50ms local vs 200--800ms API round-trip)
- You want zero per-token cost
- Prototyping or development use

### Add a dedicated GPU (or cloud GPU) when:
- You need an 8B-class model for top-tier quality
- High throughput is required (>100K embeddings/day)
- Volume justifies hardware cost (>1B tokens/month replaces API spend)

### Use a proprietary API when:
- Volume is low (<100M tokens/month) and operational simplicity matters
- You need 128K context for full-document embedding (Cohere v4)
- You need domain-specific models without fine-tuning (Voyage finance/law/code)
- No GPU infrastructure is available and models >1.5B are required

### Model selection quick guide

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| General RAG (best quality, have GPU) | Qwen3-Embedding-8B | Top MTEB, Apache 2.0, Matryoshka |
| General RAG (no large GPU) | BGE-M3 | Hybrid search, MIT, runs on laptop |
| Retrieval-focused (small footprint) | Snowflake Arctic Embed | SOTA retrieval/size, Apache 2.0 |
| Multilingual | Qwen3-Embedding-8B or BGE-M3 | Both support 100+ languages |
| Code search | Qwen3-Embedding-8B | 80.68 on MTEB Code benchmark |
| Full-document embedding | GTE-Qwen2-7B-instruct | 128K context, no chunking needed |
| Multimodal (text + images) | Jina v4 | GGUF available, strong ViDoRe scores |
| Edge/IoT/mobile | all-MiniLM-L6-v2 | 22M params, sub-10ms on CPU |
| Prototyping | Nomic v1.5 or bge-small | Easy Ollama/FastEmbed setup, runs anywhere |
| Maximum speed on CPU | all-MiniLM-L6-v2 + ONNX | 5--14K sentences/sec |

## Hyperscaler Managed Embedding Services

| Provider | Service | Models Available |
|----------|---------|-----------------|
| **AWS** | Amazon Bedrock, SageMaker | Cohere Embed v4, Titan Embeddings v2, Mistral Embed |
| **Azure** | Azure OpenAI Service, Azure AI | OpenAI text-embedding-3-*, Cohere Embed v4, Voyage 3.5 |
| **GCP** | Vertex AI | Gemini Embedding 001, text-embedding-005 |
| **IBM** | watsonx.ai | slate.125m, granite.embedding |
| **Oracle** | OCI Generative AI | Cohere Embed |

All hyperscalers also support self-hosting open-weight models on their GPU instances (e.g., AWS p4d/p5 with A100/H100, Azure NC-series, GCP A3 with H100).

## Context-Length Performance: BGE-M3 vs Alternatives

MTEB scores measure average performance but hide a critical variable: how performance changes with input length. Independent benchmarking across context lengths reveals that models have very different degradation curves.

### Passkey retrieval accuracy by context length

| Model | Best pooling | 512 tokens | 2K tokens | 4K tokens | 8K tokens |
|-------|-------------|-----------|----------|----------|----------|
| **Qwen3-Embedding-0.6B** | LAST | 1.00 | 0.94 | 1.00 | 1.00 |
| **BGE-M3** | CLS | 1.00 | 0.80 | 0.32 | 0.34 |
| **Jina-Embeddings-v3** | MEAN | 1.00 | 0.92 | 0.36 | 0.40 |
| **E5-Base-4K** | MEAN | 0.70 | 0.86 | 0.72 | 0.72 |
| **Nomic-Embed-Text-v1.5** | MEAN | 0.16 | 0.22 | 0.46 | 0.58 |

Source: [Comprehensive Embedding Models Evaluation](https://x22x22.github.io/embedding_models_comprehensive_evaluation.html)

**Key takeaways:**
- **BGE-M3's sweet spot is ≤2K tokens.** Despite supporting 8192 tokens, it performs best on shorter inputs. If your chunks are under 2K tokens (which is typical for RAG at 256–512 tokens), BGE-M3 is excellent.
- **Qwen3-Embedding-0.6B is the context-length champion** among similarly sized models. It maintains near-perfect accuracy at all tested lengths.
- **Pooling strategy matters enormously.** Using the wrong pooling can cut performance by 50%+. Always use the model's recommended pooling: CLS for BGE-M3, LAST for Qwen3, MEAN for Jina v3.
- **Bigger doesn't always mean better for retrieval.** In an independent 16-model benchmark on product search, e5-small (118M params) achieved 100% Top-5 accuracy at 16ms latency, outperforming models 70x its size. Domain-specific performance varies significantly from MTEB averages.

### When to choose BGE-M3

- You need **hybrid search** (dense + sparse + ColBERT) from a single model — BGE-M3 is the only major model offering all three modes
- Your **chunks are ≤2K tokens** (standard for most RAG pipelines)
- You need **multilingual** support (100+ languages)
- You want a **permissive licence** (MIT) for commercial use
- You're running on **limited hardware** (568M params, ~2GB VRAM)

### When to choose an alternative

| Scenario | Better choice | Why |
|----------|--------------|-----|
| Long chunks (>2K tokens) | Qwen3-Embedding-0.6B | Maintains accuracy at 4K–8K tokens |
| Maximum retrieval quality | Qwen3-Embedding-8B or BGE-en-ICL | Higher MTEB scores across all tasks |
| Fastest inference | all-MiniLM-L6-v2 or e5-small | 16ms latency, 100% Top-5 on some benchmarks |
| Best retrieval per parameter | Snowflake Arctic Embed | SOTA retrieval at 334M params |
| Full-document embedding | GTE-Qwen2-7B-instruct | 128K context, no chunking needed |
| Multimodal (text + images) | Jina v4 | Vision-language retrieval |
| Dense-only retrieval | Qwen3-Embedding-0.6B or Snowflake Arctic | Higher dense scores than BGE-M3 |

## Areas of Uncertainty

- **MTEB score reliability**: All scores are self-reported. No independent verification exists. Models may be optimised for MTEB tasks rather than real-world performance. Focus on task-specific scores (e.g. retrieval) relevant to your use case.
- **Qwen3-Embedding-8B dimensions**: Sources disagree on maximum dimensions (4,096 vs 7,168). The HuggingFace model card should be treated as authoritative.
- **Throughput benchmarks**: No standardised, independently verified throughput benchmarks for embedding models exist. Speeds in this article are approximate and vary by hardware, batch size, sequence length, and framework.
- **TCO breakeven**: Estimates range from 500M to 11B tokens/month depending on assumptions about GPU utilisation, DevOps overhead, and which API is being replaced.
- **Apple Silicon performance**: MPS backend performance varies by model architecture. Some models run slower on MPS than CPU due to incomplete operator support. Test with your specific model.
- **Multilingual quality**: Models claiming "100+ language support" have widely varying quality across languages. Test on your target language before committing.

## References

1. [MTEB Leaderboard - HuggingFace](https://huggingface.co/spaces/mteb/leaderboard) - Official MTEB benchmark leaderboard
2. [Embedding Model Leaderboard: MTEB Rankings March 2026 - Awesome Agents](https://awesomeagents.ai/leaderboards/embedding-model-leaderboard-mteb-march-2026/) - Comprehensive model comparison with pricing
3. [Top embedding models on the MTEB leaderboard - Modal](https://modal.com/blog/mteb-leaderboard-article) - Technical guide to interpreting MTEB and choosing models
4. [Best Embedding Models for RAG (2026) - PremAI](https://blog.premai.io/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/) - 10-model comparison with cost analysis
5. [Gemini Embedding now generally available - Google Developers Blog](https://developers.googleblog.com/en/gemini-embedding-available-gemini-api/) - Official Gemini Embedding announcement
6. [Jina Embeddings V4 - Jina AI](https://jina.ai/models/jina-embeddings-v4/) - Multimodal embedding model details
7. [Snowflake Arctic Embed - GitHub](https://github.com/Snowflake-Labs/arctic-embed) - Retrieval-focused embedding suite
8. [Ollama Embedded Models Guide - ColabNix](https://collabnix.com/ollama-embedded-models-the-complete-technical-guide-to-local-ai-embeddings-in-2025/) - Local deployment via Ollama
9. [sentence-transformers Documentation](https://www.sbert.net/) - Python embedding model library
10. [MTEB GitHub Repository](https://github.com/embeddings-benchmark/mteb) - Benchmark framework source code
11. [Voyage AI Pricing](https://docs.voyageai.com/docs/pricing) - Official Voyage AI pricing and model tiers
12. [Cohere Embed v4 Changelog](https://docs.cohere.com/changelog/embed-multimodal-v4) - Official Cohere v4 announcement
13. [Cohere Embed v4 on AWS Bedrock](https://aws.amazon.com/about-aws/whats-new/2025/10/coheres-embed-v4-multimodal-embeddings-bedrock/) - AWS availability
14. [Embedding Models Pricing Comparison - Awesome Agents](https://awesomeagents.ai/pricing/embedding-models-pricing/) - Normalised pricing comparison
15. [FastEmbed - Qdrant](https://github.com/qdrant/fastembed) - CPU-first ONNX embedding library
16. [Sentence Transformers Efficiency Guide](https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html) - ONNX/OpenVINO backend documentation
17. [Amazon Titan Embeddings Benchmark - Philipp Schmid](https://www.philschmid.de/amazon-titan-embeddings) - Self-hosting vs API cost analysis
18. [Best Open-Source Embedding Models Benchmarked - Supermemory](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/) - Retrieval accuracy and latency benchmarks
19. [NV-Embed-v2 - HuggingFace](https://huggingface.co/nvidia/NV-Embed-v2) - Official NVIDIA model card
20. [Qwen3-Embedding - GitHub](https://github.com/QwenLM/Qwen3-Embedding) - Official Qwen3 embedding repository
21. [Voyage 3.5 on Azure AI](https://ai.azure.com/catalog/models/voyage-3.5-embedding-model) - Azure availability and benchmarks
22. [Scaling PyTorch Inference with ONNX Runtime - Microsoft](https://opensource.microsoft.com/blog/2022/04/19/scaling-up-pytorch-inference-serving-billions-of-daily-nlp-inferences-with-onnx-runtime/) - ONNX Runtime production benchmarks
23. [Comprehensive Embedding Models Evaluation: Native vs Chunked Processing](https://x22x22.github.io/embedding_models_comprehensive_evaluation.html) - Context-length performance comparison across 5 models
24. [Benchmark of 16 Best Open Source Embedding Models for RAG - AIMultiple](https://aimultiple.com/open-source-embedding-models) - Independent 16-model retrieval benchmark on product search
25. [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/abs/2402.03216v3) - Original BGE-M3 paper
