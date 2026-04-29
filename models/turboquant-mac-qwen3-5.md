# TurboQuant and Its Use on Mac Hardware (Especially with Qwen3.5 Models)

| Field | Value |
|-------|-------|
| Created | 2026-03-29 |
| Last Updated | 2026-03-29 |
| Version | 1.1 |

---

- [Executive Summary](#executive-summary)
- [What Is TurboQuant?](#what-is-turboquant)
- [How TurboQuant Works](#how-turboquant-works)
- [TurboQuant vs Other Quantisation Methods](#turboquant-vs-other-quantisation-methods)
- [Community Implementations](#community-implementations)
- [Qwen3.5 Architecture and Its Implications](#qwen35-architecture-and-its-implications)
- [TurboQuant on Mac / Apple Silicon](#turboquant-on-mac--apple-silicon)
- [Practical Guide: Running TurboQuant + Qwen3.5 on Mac](#practical-guide-running-turboquant--qwen35-on-mac)
- [Benchmark Results on Apple Silicon](#benchmark-results-on-apple-silicon)
- [Attention Fidelity and Quality Metrics](#attention-fidelity-and-quality-metrics)
- [Limitations and Caveats](#limitations-and-caveats)
- [Reception and Controversy](#reception-and-controversy)
- [Outlook](#outlook)
- [References](#references)

## Executive Summary

TurboQuant is a two-stage vector quantisation algorithm from Google Research, presented at ICLR 2026. It compresses LLM key-value (KV) caches to 3 bits per value with zero accuracy loss, achieving 6x memory reduction and up to 8x attention-logit speedup on H100 GPUs. Within 24 hours of its release on 24 March 2026, the open-source community ported it to both Apple's MLX framework and llama.cpp, with early benchmarks specifically targeting Qwen3.5 models.

On Apple Silicon, TurboQuant implementations range from hardware-accelerated V2 variants (near-native speed via Metal kernels) to paper-correct V3 variants (Lloyd-Max codebooks, 5.5x compression). Early benchmarks on M4 Max and M5 Max show that 3-bit compression is the practical sweet spot — delivering ~5x compression with 99.5% attention fidelity. However, a community consensus has emerged that QJL residual correction (Algorithm 2 from the paper) does not improve quality in practice when it replaces MSE bits, and the K/V norm disparity in models like Qwen makes uniform bit allocation suboptimal.

Qwen3.5's hybrid Gated DeltaNet architecture presents both an opportunity (massive context windows up to 262K tokens that benefit from KV cache compression) and a constraint (only a fraction of layers use full-attention KV caches).

## What Is TurboQuant?

TurboQuant is a training-free compression algorithm for high-dimensional vectors, targeting two primary use cases:

1. **KV cache compression** in large language models during inference
2. **Vector search** acceleration for similarity lookups at scale

It was developed by Amir Zandieh (Research Scientist) and Vahab Mirrokni (VP and Google Fellow) at Google Research, with collaborators from Google DeepMind, KAIST, and NYU (Praneeth Kacham, Majid Hadian, Insu Han, Majid Daliri, Lars Gottesburen, Rajesh Jayaram).

The paper "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" was presented at ICLR 2026. Two companion algorithms — PolarQuant (AISTATS 2026) and Quantized Johnson-Lindenstrauss (QJL) — form the building blocks.

The core innovation is **eliminating memory overhead** from quantisation. Traditional methods store per-block scale factors and zero-points in full precision, adding 1–2 extra bits per number. TurboQuant uses data-independent, analytically precomputed codebooks, achieving near-optimal distortion with zero overhead.

## How TurboQuant Works

TurboQuant operates in two stages that together produce an unbiased estimator for attention scores (which are inner products between queries and keys).

### Stage 1: PolarQuant (High-Quality Compression)

PolarQuant exploits two properties of multivariate normal random variables:

1. **Random preconditioning produces Gaussians**: Multiplying any vector **x** by a random orthogonal matrix **S** yields a multivariate Gaussian output S·x ~ N(0, ||x||² · I), regardless of the original data distribution.

2. **Norm concentration**: The norm of a Gaussian vector concentrates tightly around √d in high dimensions, following a generalised gamma distribution.

After preconditioning, PolarQuant applies a **recursive polar transformation**:

1. Pair adjacent coordinates (x₁, x₂), convert each pair to polar coordinates (r, θ) via `atan2`. This produces d/2 angles and d/2 radii.
2. At Level 1, angles span [0, 2π) and are quantised with 4 bits (16 buckets).
3. Take the d/2 radii, pair them again, convert to polar. Since radii are always positive, angles are now restricted to [0, π/2].
4. At Level 2+, angles are quantised with just 2 bits (4 buckets), because they concentrate ever more tightly around π/4.
5. Recurse for log₂(d) levels total. At the deepest levels, angles are essentially constant at 45° and can be quantised with practically one bucket.

The key insight: because the distribution at each level is known analytically (derived from the Gaussian structure), **codebooks can be precomputed once** using Lloyd's algorithm on the theoretical PDF — no calibration data, no per-model tuning, no runtime scanning.

For a 128-dimensional FP16 vector (2,048 bits), PolarQuant stores:
- 64 level-1 angles at 4 bits = 256 bits
- 32 level-2 angles at 2 bits = 64 bits
- 16 level-3 angles at 2 bits = 32 bits
- 8 level-4 angles at 2 bits = 16 bits
- 8 leftover radii at 16 bits = 128 bits
- **Total: 496 bits → 4.13x compression**

### Stage 2: QJL (1-Bit Residual Correction)

The Quantised Johnson-Lindenstrauss (QJL) transform is applied to the tiny residual error left from Stage 1. It:

- Uses the Johnson-Lindenstrauss Transform to shrink the residual
- Reduces each resulting number to a single sign bit (+1 or -1)
- Requires zero memory overhead
- Produces an **unbiased estimator** for inner products (attention scores)

The combination of PolarQuant (b-1 bits for the main signal) and QJL (1 bit for residual correction) yields an overall b-bit quantiser with provably unbiased inner-product estimation.

**Practical caveat**: Community implementations have consistently found that QJL works as *additional* information alongside MSE-quantised values, but *not* as a replacement for MSE bits. See [QJL Consensus](#qjl-consensus-mse-only-beats-qjlmse) for details.

## TurboQuant vs Other Quantisation Methods

| Method | Type | Compression | Overhead | Calibration | Conference |
|--------|------|-------------|----------|-------------|------------|
| **TurboQuant** | KV cache (vector quant) | 6x (3-bit) | Zero | None (data-independent) | ICLR 2026 |
| **KIVI** | KV cache (asymmetric) | 2.6x (2-bit) | Per-block scales | Per-model | ICML 2024 |
| **NVIDIA KVTC** | KV cache (transform coding) | Varies | Varies | Required | ICLR 2026 |
| **NVFP4** | Weights + activations | ~4x | Per-block scales | Offline or runtime | — |
| **GGUF Q4_K_M** | Weight quantisation | ~4x | Per-block scales | Per-model | — |
| **AWQ** | Weight quantisation | ~4x | Channel-wise | Calibration set | — |
| **GPTQ** | Weight quantisation | ~4x | Per-group | Calibration set | — |

Key differentiators:

- **vs KIVI**: TurboQuant achieves 6x compression (vs 2.6x) with better quality preservation across LongBench benchmarks. KIVI uses asymmetric 2-bit quantisation with per-channel keys and per-token values. KIVI ships with HuggingFace Transformers integration.
- **vs NVFP4**: TurboQuant eliminates per-block scale factors entirely. NVFP4 uses uniformly spaced buckets and requires calibration data or runtime max-scan. TurboQuant uses distribution-aware buckets precomputed analytically.
- **vs GGUF/AWQ/GPTQ**: These target weight quantisation (compressing the static model). TurboQuant targets the dynamic KV cache that grows with sequence length. They are complementary — a model can use GGUF-quantised weights and TurboQuant-compressed KV caches simultaneously.

### Benchmark Results (Google's Evaluation)

Tested on Gemma, Mistral, and Llama-3.1-8B-Instruct across five long-context benchmarks:

- **LongBench**: Optimal scoring on question answering, code generation, summarisation
- **Needle In A Haystack**: Perfect retrieval scores at 6x compression
- **ZeroSCROLLS, RULER, L-Eval**: Consistent quality preservation
- **Attention logit speedup**: 4-bit TurboQuant achieves up to **8x speedup** over 32-bit unquantised keys on H100 GPUs

For vector search, TurboQuant achieves superior 1@k recall ratios on the GloVe dataset (d=200) compared to Product Quantisation (PQ) and RabbiQ baselines.

## Community Implementations

Within 24 hours of TurboQuant's announcement on 24 March 2026, the open-source community produced multiple implementations:

### MLX (Apple Silicon)

#### Flovflo/turboquant-mlx-qwen35-kv

- **Repository**: [Flovflo/turboquant-mlx-qwen35-kv](https://github.com/Flovflo/turboquant-mlx-qwen35-kv)
- **Target**: `mlx-community/Qwen3.5-35B-A3B-4bit` KV cache
- **Status**: TurboQuant-inspired prototype (uses MLX affine quantisation, not PolarQuant)
- **HuggingFace**: [flovflo/turboquant-mlx-qwen35-kv](https://huggingface.co/flovflo/turboquant-mlx-qwen35-kv)

#### sharpner/turboquant-mlx

- **Repository**: [sharpner/turboquant-mlx](https://github.com/sharpner/turboquant-mlx)
- **Target**: General KV cache compression on Apple Silicon
- **Status**: Most comprehensive MLX implementation, with V2 (hardware-accelerated) and V3 (paper-correct Lloyd-Max) variants
- **Hardware**: Benchmarked on M4 Max 64 GB

The sharpner implementation provides two architectural families:

**V2 Variants** (Affine quantisation, hardware-accelerated via `mx.quantized_matmul`):

| Variant | Rotation | Norm-Baking | QJL | Relative Speed |
|---------|----------|-------------|-----|----------------|
| LEAN | No | No | No | ~100% (fastest) |
| rotated | Yes | Yes | No | ~70% |
| rotated+QJL | Yes | Yes | Yes | ~30% |

**V3 Variants** (Lloyd-Max codebook, paper-correct quality, software dequant):

| Variant | Description | Relative Speed |
|---------|-------------|----------------|
| uniform | All channels at b-bit Lloyd-Max | ~18% |
| mixed | Outlier channels at (b+1)-bit, rest at b-bit; enables fractional rates (2.5, 3.5) | ~16% |

V2 prioritises speed (5–6x faster than V3) via Metal kernel acceleration. V3 prioritises quality via optimal codebooks matching the paper's approach. See [M4 Max Benchmarks](#m4-max-benchmarks-sharpnerturboquant-mlx) for detailed results.

### llama.cpp

- **Discussion**: [ggml-org/llama.cpp#20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
- **Branch**: `feat/turbo-quant`
- **Weight formats**: TBQ3_0 and TBQ4_0 (weight quantisation)
- **Cache types**: `turbo3` (~3.25 bits/val, ~4.9x compression) and `turbo4` (~4.25 bits/val, ~3.8x compression)
- **Implementation**: CPU (C, no dependencies) + CUDA kernels + Metal kernels (TheTom/turboquant_plus fork)
- **Tests**: 18/18 passing, MSE matching paper within 1%

Usage for KV cache compression:

```bash
# Use turbo3 cache type for both K and V
./build/bin/llama-cli -m model.gguf \
  --cache-type-k turbo3 --cache-type-v turbo3
```

**CUDA port** (spiritbuun, RTX 3090 24 GB, Q6 Qwen3.5-27B at 128K context):
- Prefill speed: 99.6% of baseline
- Decode speed: 97.5% of baseline
- Perplexity: -1.17% vs q8_0 (i.e. slightly *better* than q8_0) at 3.5x KV compression

### PyTorch

| Repository | Target | Validation |
|-----------|--------|------------|
| [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) | From-scratch implementation | Validated on Qwen2.5-3B-Instruct (RTX 3060) |
| [AliesTaha/polar_quant](https://github.com/AliesTaha/polar_quant) | CUDA kernels | ~75% cuBLAS throughput |

### Official Release

Google's official implementation is expected around Q2 2026. An open feature request exists on the vLLM project to integrate TurboQuant as a native KV cache quantisation option.

## Qwen3.5 Architecture and Its Implications

Qwen3.5 introduces a **Gated DeltaNet hybrid attention** architecture that is fundamentally different from standard transformers. Understanding this architecture is critical to evaluating TurboQuant's effectiveness.

### Architecture Overview

Qwen3.5 uses a 3:1 ratio of linear DeltaNet layers to full softmax attention layers:

```
8 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))
```

Key implications for TurboQuant:

| Model | Total Layers | Full-Attention Layers | KV Cache Layers | Context Window |
|-------|-------------|----------------------|-----------------|----------------|
| Qwen3.5-4B | 32 | ~8 | ~8 | 262,144 |
| Qwen3.5-27B | 64 | 16 | 16 | 262,144 |
| Qwen3.5-35B-A3B (MoE) | 40 | 10 | 10 | 262,144 |

Because only a fraction of layers use full-attention KV caches (the rest use DeltaNet's `ArraysCache`), TurboQuant's KV cache compression applies to a subset of the model's memory. This limits the headline compression ratio but the full-attention layers are still a significant memory consumer, particularly at long contexts.

### The DeltaNet Problem on Mac

Qwen3.5's DeltaNet architecture is poorly optimised in llama.cpp's C++ inference engine. One practitioner measured a **14x latency regression** when upgrading from Qwen 3 to Qwen 3.5 on llama.cpp (1.5s → 20.7s on M1). Apple's MLX framework, being Metal-native, handles DeltaNet significantly better — reducing latency to 6.9s on the same hardware.

This makes TurboQuant on MLX particularly relevant: the same framework that resolves Qwen3.5's inference performance also provides the best path for TurboQuant KV cache compression on Mac.

### Model Loading Requirements

Qwen 3.5 is natively a **vision-language model** (VLM). Even for text-only inference, the vision tower weights must be present in the architecture. This means:

- **`mlx-lm`** cannot load Qwen 3.5
- **`mlx-vlm`** is required (VLM-aware, handles vision + text)
- Standard GGUF quantisation via llama.cpp works but with the DeltaNet performance penalty

## TurboQuant on Mac / Apple Silicon

### Why TurboQuant Matters for Mac Users

Apple Silicon's unified memory architecture makes it uniquely suited for local LLM inference, but memory is shared between CPU, GPU, and the Neural Engine. KV cache growth at long contexts directly competes with model weights for available memory. On a 32 GB Mac:

- Qwen3.5-35B-A3B-4bit model weights: ~18-20 GB
- Remaining for KV cache + OS: ~12-14 GB
- Without compression, 262K context fills the KV cache rapidly

TurboQuant's 6x KV cache reduction means longer contexts fit in the same memory budget, or the same contexts leave more room for batch processing.

### Current State of MLX Implementations

The most complete MLX implementation is [Flovflo/turboquant-mlx-qwen35-kv](https://github.com/Flovflo/turboquant-mlx-qwen35-kv), targeting `mlx-community/Qwen3.5-35B-A3B-4bit`. It is explicitly labelled **"TurboQuant-inspired"** with several deviations from the paper:

| Aspect | Google Paper | MLX Prototype |
|--------|-------------|---------------|
| Main quantiser | PolarQuant (polar coordinates) | MLX affine quantisation |
| Residual correction | QJL (1-bit unbiased estimator) | Simple packed sign sketch + RMS scaling |
| Hardware target | H100 GPUs | Apple Silicon (Metal) |
| Scope | All KV layers | Only full-attention layers (10 of 40 in Qwen3.5-35B-A3B) |

What the prototype does implement:
- Lightweight structured random transform on keys before quantisation
- Affine MLX quantisation for compressed key representation
- 1-bit residual sign sketch with residual RMS term for score correction
- Optional value quantisation via the same MLX affine path
- Runtime patching of Qwen3.5 attention dispatch without forking `mlx-lm`

### Other MLX Ports

The [sharpner/turboquant-mlx](https://github.com/sharpner/turboquant-mlx) repository provides the most comprehensive proof-of-concept, with both V2 (hardware-accelerated) and V3 (paper-correct Lloyd-Max) variants. See [M4 Max Benchmarks](#m4-max-benchmarks-sharpnerturboquant-mlx) for detailed quality and throughput tables.

@Prince_Canuma was among the first to port TurboQuant to MLX, testing with Qwen3.5-35B within 24 hours of the announcement. A Twitter user also reported running a Qwen3.5-27B variant at full 262K context on an M1 Max with 64 GB using a TurboQuant fork with a 40 GB prompt cache.

## Practical Guide: Running TurboQuant + Qwen3.5 on Mac

### Hardware Requirements

| Configuration | Minimum | Recommended |
|--------------|---------|-------------|
| Qwen3.5-4B (4-bit) | 8 GB unified memory | 16 GB |
| Qwen3.5-35B-A3B (4-bit) | 24 GB unified memory | 32+ GB |
| Qwen3.5-27B (Q4_K_M) | 32 GB unified memory | 64 GB |

### Using the Flovflo MLX Implementation

```bash
# Set up environment
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip setuptools wheel
./.venv/bin/pip install -e '.[dev]'

# Smoke test
./.venv/bin/pytest -q

# Generate text with TurboQuant backend
./.venv/bin/tqkv generate \
  mlx-community/Qwen3.5-35B-A3B-4bit \
  'Hi, what can you help me with?' \
  --backend turboquant \
  --max-tokens 64

# Run benchmark (one backend at a time for RAM safety)
./.venv/bin/tqkv benchmark \
  mlx-community/Qwen3.5-35B-A3B-4bit \
  --backend turboquant \
  --prompt-tokens 2048 \
  --generation-tokens 8 \
  --output benchmarks/turboquant_2048_8.json
```

Three backends are available for comparison:
- `baseline` — standard KV cache
- `mlx_quant` — existing MLX affine KV quantisation
- `turboquant` — TurboQuant-inspired rotated key cache with residual sketch correction

### Using llama.cpp with TurboQuant

The `feat/turbo-quant` branch of llama.cpp introduces both weight formats (TBQ3_0/TBQ4_0) and KV cache types (turbo3/turbo4):

```bash
# Build from the TurboQuant branch
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout feat/turbo-quant
cmake -B build && cmake --build build

# Quantise model weights to TBQ4_0
./build/bin/llama-quantize model-f16.gguf model-tbq4.gguf TBQ4_0

# Or use turbo3 KV cache compression with any model
./build/bin/llama-cli -m model.gguf \
  --cache-type-k turbo3 --cache-type-v turbo3
```

The Metal fork (TheTom/turboquant_plus) adds Apple Silicon support with SET_ROWS, dequantize, and flash attention Metal kernels for end-to-end operation.

### MLX via Standard Tooling (Without TurboQuant)

For Qwen3.5 on Mac without TurboQuant-specific tooling:

```bash
# Install mlx-vlm (required for Qwen3.5, mlx-lm cannot load it)
pip3 install mlx-vlm

# Run inference
python3 -m mlx_vlm.generate \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt "Hello, how are you?"
```

## Benchmark Results on Apple Silicon

### TurboQuant-Inspired MLX vs Baseline (Flovflo Implementation)

Model: `mlx-community/Qwen3.5-35B-A3B-4bit`, ~30 GB Apple Silicon machine.

#### 2048 Prompt / 8 Gen Tokens (3 trials) — Best Case

| Backend | Prompt TPS | Gen TPS | Wall Time (s) | Cache |
|---------|-----------|---------|---------------|-------|
| baseline | 514.34 | 35.67 | 5.67 | 80.12 MB |
| mlx_quant | 516.13 | 38.30 | 5.16 | 44.77 MB |
| **turboquant** | **679.14** | **44.83** | **4.20** | **45.10 MB** |

TurboQuant vs baseline: **+32% prompt**, **+25.7% decode**, **-26% wall time**, **-43.7% cache**

TurboQuant vs mlx_quant: **+31.6% prompt**, **+17.1% decode**, **-18.5% wall time**, cache within +0.73%

#### 1024 Prompt / 8 Gen Tokens (1 trial)

| Backend | Prompt TPS | Gen TPS | Wall Time (s) | Cache |
|---------|-----------|---------|---------------|-------|
| baseline | 378.00 | 28.98 | 6.29 | 59.15 MB |
| mlx_quant | 471.34 | 49.89 | 2.57 | 38.87 MB |
| turboquant | 490.29 | 50.65 | 2.43 | 39.04 MB |

#### 128 Prompt / 8 Gen Tokens (3 trials) — Short Context

| Backend | Prompt TPS | Gen TPS | Wall Time (s) | Cache |
|---------|-----------|---------|---------------|-------|
| baseline | 270.47 | 54.32 | 0.93 | 38.17 MB |
| mlx_quant | 274.53 | 52.20 | 0.87 | 33.71 MB |
| turboquant | 266.08 | 52.65 | 1.09 | 33.73 MB |

At short context, TurboQuant is **primarily a memory optimisation** with no speed advantage.

#### Important Nuances

- The -26% wall time at 2048 tokens includes a warmup effect on run 1. Warm runs show **-15.4%**.
- The -43.7% cache headline is slightly inflated by MLX baseline `KVCache` growing in 256-token blocks. The actual full-attention KV portion is reduced by **71.1%**, with end-to-end used cache down **39.9%**.
- Qwen3.5's mixed architecture means only 10 of 40 layers benefit from KV cache compression. On a fully dense attention model, gains would be larger.

### M4 Max Benchmarks (sharpner/turboquant-mlx)

Hardware: Apple M4 Max (64 GB), unified memory (~400 GB/s bandwidth). Model: Llama 3.2 3B (4-bit weight-quantised from mlx-community).

#### Throughput (tok/s)

| Strategy | T=512 | T=1024 | T=2048 | T=4096 | T=8192 |
|----------|-------|--------|--------|--------|--------|
| fp16 baseline | 208 | 199 | 191 | 175 | 148 |
| V2 4-bit LEAN | 188 | 188 | 184 | 174 | 156 |
| V2 4-bit rotated | 135 | 133 | 131 | 124 | 115 |
| V2 3-bit rot+QJL | 101 | 96 | 84 | 65 | 45 |
| V3 3.5-bit mixed | 82 | 74 | 59 | 42 | 24 |
| V3 3-bit Lloyd-Max | 98 | 86 | 70 | 47 | 27 |
| V3 2.5-bit mixed | 83 | 75 | 59 | 42 | 24 |

**Bandwidth crossover**: V2 compressed cache overtakes fp16 at approximately T~4K tokens, where reduced memory traffic from the smaller cache outweighs the dequantisation overhead.

#### KV Cache Compression at T=8192

| Strategy | Cache Size | Compression |
|----------|-----------|-------------|
| fp16 | 969 MB | 1x |
| V2 4-bit LEAN | 266 MB | 3.6x |
| V3 3.5-bit mixed | 236 MB | 4.1x |
| V3 3-bit Lloyd-Max | 207 MB | 4.7x |
| V3 2.5-bit mixed | 177 MB | **5.5x** |

#### Perplexity (Multi-Model Quality)

| Strategy | bits/dim | Llama 3.2 3B | Llama 3.1 8B | Mistral 7B | Gemma 3 4B (D=256) |
|----------|----------|-------------|-------------|------------|---------------------|
| fp16 baseline | 16 | 12.94 | 9.47 | 6.79 | 12.18 |
| V2 4-bit rotated | 4 | 12.84 (-0.8%) | 9.61 (+1.4%) | 6.89 (+1.4%) | 12.53 (+2.9%) |
| V2 4-bit LEAN | 4 | 13.02 (+0.6%) | 9.85 (+4.0%) | 6.87 (+1.2%) | 12.37 (+1.6%) |
| V2 3-bit rot+QJL | 3 | 13.63 (+5.3%) | 10.21 (+7.8%) | 7.14 (+5.1%) | **12.05 (-1.1%)** |
| V3 3.5-bit mixed | 3.5 | **12.98 (+0.3%)** | 10.10 (+6.7%) | 7.06 (+4.0%) | 12.44 (+2.1%) |
| V3 3-bit Lloyd-Max | 3 | 13.60 (+5.1%) | 10.28 (+8.6%) | 7.27 (+7.0%) | 12.93 (+6.2%) |
| V3 2.5-bit mixed | 2.5 | 16.44 (+27.0%) | 12.80 (+35.2%) | 7.53 (+10.8%) | 13.04 (+7.0%) |

Key findings:
- **V2 3-bit rot+QJL beats fp16 on Gemma 3** (D=256): PPL 12.05 vs 12.18 — the rotation + QJL correction acts as a regulariser at larger head dimensions.
- **V3 3.5-bit mixed is near-lossless**: +0.3% PPL on Llama 3.2 3B at 4.1x compression.
- **Larger head_dim (D=256) dramatically improves quantisation quality**: Gemma 3 at V3 2.5-bit shows only +7% PPL vs +27% for Llama 3B (D=128).
- V2 is 5–6x faster than V3 due to hardware-accelerated Metal kernels vs software dequant.

#### Recommended Strategies

| Use Case | Strategy | Quality (D=128) | Quality (D=256) | Speed |
|----------|----------|-----------------|-----------------|-------|
| Maximum speed | V2 4-bit LEAN | +0.6–4% PPL | +1.6% PPL | ~105% of fp16 at 8K |
| Best quality at 4-bit | V2 4-bit rotated | -0.8 to +1.4% | +2.9% | ~78% of fp16 |
| Best 3-bit (D=256) | V2 3-bit rot+QJL | +5–8% | -1.1% | ~30% of fp16 at 8K |
| Near-lossless compression | V3 3.5-bit mixed | +0.3–7% | +2.1% | ~16% of fp16 |
| Aggressive compression | V3 2.5-bit mixed | +11–35% | +7.0% | ~16% of fp16 |

### M5 Max Benchmarks (llama.cpp Metal Fork)

Hardware: Apple M5 Max. Model: Qwen3.5-35B-A3B (turbo3 KV cache type via TheTom/turboquant_plus).

#### Speed Recovery (turbo3 vs q8_0 baseline)

| Context Length | turbo3/q8_0 Ratio |
|---------------|-------------------|
| 2K | 0.987x |
| 4K | 0.989x |
| 8K | 0.995x |
| 16K | 0.989x |
| 32K | 0.995x |

Speed is flat at **98.7–99.5% of q8_0 through 32K tokens** with no degradation trend, at 4.9x KV cache compression. PPL overhead: +1.1% (5.471 vs 5.414).

On the 35B MoE variant, turbo3 achieved **97% of q8_0 speed with 4.6x smaller cache** after community shader optimisations.

Prompt processing with turbo3 sometimes *exceeded* q8_0 because the smaller KV cache reduces memory bandwidth requirements.

### Qwen3.5 on Mac Without TurboQuant (Reference)

From practitioner benchmarks (Mac M1 16 GB, Q4_K_M):

| Configuration | Mean Latency | Quality Score |
|--------------|-------------|---------------|
| Qwen 3 + llama.cpp | 1.5s | 0.958 |
| Qwen 3.5 + llama.cpp | 20.7s | 0.966 |
| Qwen 3.5 + MLX (mlx-vlm) | 6.9s | 0.988 |

Production Mac Mini M4 Pro hardware is estimated at 2–3x faster than M1, putting Qwen 3.5 + MLX at roughly 3–4s per response.

## Attention Fidelity and Quality Metrics

### PyTorch Validation (tonbistudio/turboquant-pytorch)

Validated on Qwen2.5-3B-Instruct (RTX 3060), averaged across all 36 layers (2 KV heads per layer = 72 checks):

| Config | Context | Cosine Sim | Top-1 Match | Top-5 Match |
|--------|---------|------------|-------------|-------------|
| TQ 4-bit | 2K | 0.9989 | 85% | 96% |
| TQ 4-bit | 4K | 0.9986 | 92% | 94% |
| TQ 4-bit | 8K | 0.9983 | 86% | 96% |
| **TQ 3-bit** | **2K** | **0.9961** | **85%** | **94%** |
| **TQ 3-bit** | **4K** | **0.9955** | **75%** | **88%** |
| **TQ 3-bit** | **8K** | **0.9945** | **86%** | **94%** |
| TQ 2-bit | 2K | 0.9897 | 63% | 83% |
| TQ 2-bit | 4K | 0.9878 | 65% | 85% |
| TQ 2-bit | 8K | 0.9851 | 71% | 89% |

Key observations:
- Cosine similarity is remarkably stable across context lengths (0.998+ at 4-bit regardless of 2K or 8K).
- **3-bit is the practical sweet spot**: ~5x compression with 99.5% attention fidelity.
- 2-bit works but pushes it — 66% top-1 match means the model would sometimes attend to different tokens.
- The paper's "zero accuracy loss" claim at 3.5 bits is plausible given these numbers.

### llama.cpp Metal Fork (M5 Max, Qwen3.5-35B-A3B)

- Cosine similarity ~0.95 at 3.5-bit
- Effective kurtosis normalisation from ~900 to ~2.9 (near-Gaussian after rotation)
- NIAH recall at TQ 3.5-bit (27B dense): 45% vs 82% for uniform 4-bit — long-context retrieval accuracy degrades, suggesting cumulative dequantisation errors in rotation paths

### CUDA Port (RTX 3090, Qwen3.5-27B)

- Perplexity: -1.17% vs q8_0 at 3.5x compression (slightly *better* than baseline)
- Prefill: 99.6% of baseline speed
- Decode: 97.5% of baseline speed

## Limitations and Caveats

### QJL Consensus: MSE-only Beats QJL+MSE

A significant community finding: multiple independent implementers (TheTom, unixsysdev, scos-lab, sharpner) all found that the paper's Algorithm 2 — using (b-1)-bit MSE + 1-bit QJL — produces *worse* results than b-bit MSE-only in practice.

The root cause: reducing MSE from b bits to (b-1) bits halves the number of centroids (e.g. from 8 to 4 at 3-bit), increasing MSE distortion by ~3.5x. This distortion is *exponentially* amplified by softmax. QJL's linear correction cannot compensate for this exponential loss.

The sharpner implementation demonstrated this directly: V3 3-bit "prod" (2-bit MSE + QJL) gives PPL 19.48 vs V3 3-bit MSE-only at 13.60.

QJL *does* help when added as *extra* information alongside full-resolution MSE (V2 3-bit rot+QJL: +5.3% vs +6.6% PPL without QJL). But as a *replacement* for MSE bits — as the paper recommends — it consistently degrades quality.

### K/V Norm Disparity

The paper does not address this, but modern LLMs have dramatically different Key vs Value vector magnitudes. Data from scos-lab's 8-model benchmark:

| Model | K Mean Norm | V Mean Norm | K/V Ratio |
|-------|-------------|-------------|-----------|
| GPT-2 (124M) | 11.8 | — | — |
| Phi-2 (2.8B) | 13.1 | — | — |
| Qwen2.5-3B | 172.1 | — | — |
| Qwen2.5-7B | 274.0 | 2.6 | **106x** |
| Qwen2.5-1.5B | 778.6 | — | **182x** |

Since quantisation error scales with norm squared, **K needs far more bits than V**. The K/V ratio predicts the optimal bit budget:
- K/V < 10x: 3-bit uniform works (GPT-2 family)
- K/V 10–60x: 4.5–5 bit asymmetric (Phi-2, Qwen-3B)
- K/V > 100x: 5.5+ bit or mixed precision (Qwen-1.5B, 7B)

Uniform bit allocation (same bits for K and V) as recommended in the paper may be suboptimal for Qwen-family models.

### Current MLX Implementation Gaps

1. **Not faithful PolarQuant+QJL**: Current MLX prototypes use standard MLX affine quantisation as the main quantiser, not polar-coordinate-based PolarQuant. The residual correction is a simplified sign sketch, not a full QJL estimator.
2. **V3 speed penalty**: Paper-correct Lloyd-Max codebook variants run at ~16–18% of fp16 speed on M4 Max without custom Metal kernels for codebook dequant+matmul.
3. **Decode throughput regression at short contexts**: At 128 tokens, the TurboQuant backend trails baseline decode throughput by up to 19.5% in some runs.
4. **Limited to full-attention layers**: Qwen3.5's DeltaNet layers (the majority) are not targeted, capping the achievable compression ratio.

### Google's 8x Speedup Claim

The 8x attention-logit speedup reported by Google was measured on H100 GPUs at 4-bit precision. This number should not be directly expected on Apple Silicon because:

- H100 has specialised tensor cores optimised for quantised operations
- Apple Silicon's unified memory has different bandwidth characteristics (~400 GB/s vs 2.0 TB/s on A100/H100)
- MLX Metal kernels are not as mature as Google's internal CUDA implementation
- Qwen3.5's hybrid architecture limits the proportion of computation that benefits

### Long-Context Retrieval Degradation

NIAH (Needle In A Haystack) recall at TQ 3.5-bit on 27B dense models drops to 45% (vs 82% for uniform 4-bit quant). This suggests cumulative dequantisation errors accumulate through the rotation path at long contexts (up to 128K tokens). Short-context quality remains solid. The turbo3 perplexity score of 165 (vs baseline ~6.1) was flagged as a bug under investigation in the llama.cpp fork.

### Weight vs KV Cache Quantisation

TurboQuant's primary target is runtime KV cache compression, not weight quantisation. The llama.cpp community has explored TBQ3_0/TBQ4_0 as weight quantisation formats, but this is a secondary application. The two approaches are complementary — a model can have GGUF-quantised weights and TurboQuant-compressed KV caches simultaneously.

## Reception and Controversy

### Paper Criticism (RaBitQ Misrepresentation)

On 27 March 2026, Jianyang Gao (@gaoj0017) posted a detailed critique on X accusing the TurboQuant authors of:

1. Misrepresenting RaBitQ's random rotation core
2. Making incorrect technical claims about prior work
3. Presenting misleading theoretical and experimental comparisons

Gao stated that these issues were **flagged to the authors before submission**, who acknowledged them but did not fully correct them in the accepted version. A public comment on OpenReview elaborates on three specific errors in theory and experiments. The post received over 2,600 likes, amplifying discussions on citation integrity in quantisation research.

While TurboQuant's KV-cache compression gains are acknowledged as valid by critics, concerns remain over the representation of prior work.

### Market Impact

TurboQuant's announcement on 24 March 2026 triggered a 3–6% sell-off in memory chip stocks (Micron, Samsung Electronics, SK Hynix) on 26–27 March. TurboQuant primarily reduces pressure on High Bandwidth Memory (HBM) used for KV cache storage in GPUs, with minimal direct impact on DDR5 or NAND.

Analysts described the sell-off as "overdone" — TurboQuant targets inference-stage KV cache compression (not training), real-world deployment would take months to years, and the **Jevons paradox** suggests efficiency gains may spur greater overall AI usage, potentially increasing total memory demand. Core DRAM shortage drivers remain unchanged; Q1 2026 contract prices surged 80–95% QoQ.

## Outlook

TurboQuant represents a significant step forward in KV cache compression, with particular relevance for memory-constrained local inference on Apple Silicon. Key developments to watch:

1. **Google official release** (expected Q2 2026) — will provide reference CUDA kernels and potentially JAX/XLA integration
2. **vLLM integration** — an open feature request exists to add TurboQuant as a native KV cache quantisation option
3. **Faithful MLX implementation** — porting the actual PolarQuant + QJL algorithms to Metal kernels, rather than using MLX affine quantisation as a proxy; custom Metal kernels for Lloyd-Max codebook dequant could close the V3 speed gap
4. **llama.cpp mainline merge** — the `feat/turbo-quant` branch contains working CPU + CUDA + Metal implementations awaiting review
5. **Asymmetric K/V bit allocation** — the K/V norm disparity finding suggests allocating more bits to keys and fewer to values, which could improve quality at the same compression ratio
6. **Qwen3.5 DeltaNet kernel optimisation** — as both MLX and llama.cpp improve their DeltaNet kernels, the baseline performance will improve, and TurboQuant's relative advantage for KV cache compression will become more clearly measurable
7. **Apple MLX framework native support** — given the rapid community adoption, Apple may integrate TurboQuant-style KV compression directly into MLX

For Mac users running Qwen3.5 models today, the practical recommendation is to use MLX (via `mlx-vlm`) as the inference framework — it handles DeltaNet architecture 3x faster than llama.cpp — and experiment with TurboQuant-inspired KV cache compression at medium-to-long contexts where the memory and throughput benefits materialise. For llama.cpp users, the turbo3 cache type on the Metal fork delivers 98.7–99.5% of q8_0 speed with ~5x compression.

## References

1. Zandieh, A., Mirrokni, V. et al. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." ICLR 2026. [arXiv:2504.19874](https://arxiv.org/html/2504.19874v1)
2. Google Research Blog. "TurboQuant: Redefining AI efficiency with extreme compression." 24 March 2026. [https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
3. Taha, A. "I spent 31 hours on the math behind TurboQuant so you don't have to." Baseten Blog, 27 March 2026. [https://www.baseten.co/blog/i-spent-31-hours-on-the-math-behind-turboquant-so-you-dont-have-to/](https://www.baseten.co/blog/i-spent-31-hours-on-the-math-behind-turboquant-so-you-dont-have-to/)
4. Ars Technica. "Google's TurboQuant AI-compression algorithm can reduce LLM memory usage by 6x." March 2026. [https://arstechnica.com/ai/2026/03/google-says-new-turboquant-compression-can-lower-ai-memory-usage-without-sacrificing-quality/](https://arstechnica.com/ai/2026/03/google-says-new-turboquant-compression-can-lower-ai-memory-usage-without-sacrificing-quality/)
5. VentureBeat. "Google's new TurboQuant algorithm speeds up AI memory 8x, cutting costs by 50%." March 2026. [https://venturebeat.com/infrastructure/googles-new-turboquant-algorithm-speeds-up-ai-memory-8x-cutting-costs-by-50](https://venturebeat.com/infrastructure/googles-new-turboquant-algorithm-speeds-up-ai-memory-8x-cutting-costs-by-50)
6. Flovflo/turboquant-mlx-qwen35-kv. GitHub. [https://github.com/Flovflo/turboquant-mlx-qwen35-kv](https://github.com/Flovflo/turboquant-mlx-qwen35-kv)
7. sharpner/turboquant-mlx. GitHub. [https://github.com/sharpner/turboquant-mlx](https://github.com/sharpner/turboquant-mlx)
8. tonbistudio/turboquant-pytorch. GitHub. [https://github.com/tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)
9. "TurboQuant - Extreme KV Cache Quantization." llama.cpp Discussion #20969. [https://github.com/ggml-org/llama.cpp/discussions/20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
10. Sheriff, A. "From Qwen 3 to Qwen 3.5 on Apple Silicon: A 14x Latency Regression, and How MLX Got Us Back." Medium, 3 March 2026. [https://medium.com/@aejaz.sheriff/from-qwen-3-to-qwen-3-5-on-apple-silicon-a-14x-latency-regression-and-how-mlx-got-us-back-0ed9ed21fa68](https://medium.com/@aejaz.sheriff/from-qwen-3-to-qwen-3-5-on-apple-silicon-a-14x-latency-regression-and-how-mlx-got-us-back-0ed9ed21fa68)
11. BuildFastWithAI. "How Google's TurboQuant Compresses LLM Memory by 6x." [https://www.buildfastwithai.com/blogs/google-turboquant-kv-cache-6x-compression](https://www.buildfastwithai.com/blogs/google-turboquant-kv-cache-6x-compression)
12. aiHola. "Google TurboQuant Compresses LLM KV Cache 6x." [https://aihola.com/article/google-turboquant-kv-cache-compression](https://aihola.com/article/google-turboquant-kv-cache-compression)
13. Grokipedia. "TurboQuant." [https://grokipedia.com/page/TurboQuant](https://grokipedia.com/page/TurboQuant)
14. Qwen Documentation. "MLX LM." [https://qwen.readthedocs.io/en/latest/run_locally/mlx-lm.html](https://qwen.readthedocs.io/en/latest/run_locally/mlx-lm.html)
15. scos-lab. "TurboQuant 8-Model Benchmark." llama.cpp Discussion #20969. [https://github.com/ggml-org/llama.cpp/discussions/20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
16. TheTom/turboquant_plus. Metal fork of llama.cpp with turbo3/turbo4 cache types. [https://github.com/ggml-org/llama.cpp/discussions/20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
17. spiritbuun. CUDA port benchmark results. llama.cpp Discussion #20969. [https://github.com/ggml-org/llama.cpp/discussions/20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
18. Gao, J. (@gaoj0017). "TurboQuant RaBitQ Misrepresentation Critique." X, 27 March 2026.
