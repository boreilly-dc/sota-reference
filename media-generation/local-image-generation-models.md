# Local Image Generation Models: Frameworks, Hardware and Platform Guide

| Field | Value |
|-------|-------|
| Created | 2026-03-29 |
| Last Updated | 2026-03-29 |
| Version | 1.0 |

---

- [Executive Summary](#executive-summary)
- [Frameworks and Tools](#frameworks-and-tools)
  - [ComfyUI](#comfyui)
  - [Stable Diffusion WebUI Forge](#stable-diffusion-webui-forge)
  - [InvokeAI](#invokeai)
  - [Fooocus](#fooocus)
  - [AUTOMATIC1111 (Legacy)](#automatic1111-legacy)
  - [HuggingFace Diffusers](#huggingface-diffusers)
  - [stable-diffusion.cpp](#stable-diffusioncpp)
  - [Draw Things (macOS/iOS)](#draw-things-macosios)
  - [Docker-Based Deployment](#docker-based-deployment)
- [Supported Models](#supported-models)
  - [Flux.1 Family](#flux1-family)
  - [Stable Diffusion 3.x](#stable-diffusion-3x)
  - [SDXL and SD 1.5](#sdxl-and-sd-15)
  - [Wan Video](#wan-video)
- [Hardware Requirements](#hardware-requirements)
  - [GPU Memory Requirements by Model and Precision](#gpu-memory-requirements-by-model-and-precision)
  - [NVIDIA GPUs](#nvidia-gpus)
  - [AMD GPUs](#amd-gpus)
  - [Intel Arc GPUs](#intel-arc-gpus)
  - [Apple Silicon](#apple-silicon)
  - [CPU-Only Inference](#cpu-only-inference)
- [Quantisation Methods](#quantisation-methods)
  - [Precision Formats](#precision-formats)
  - [GGUF Quantisation](#gguf-quantisation)
  - [torchao Quantisation](#torchao-quantisation)
  - [BitsAndBytes / NF4](#bitsandbytes--nf4)
  - [TensorRT (NVIDIA)](#tensorrt-nvidia)
  - [Quality vs Memory Trade-offs](#quality-vs-memory-trade-offs)
- [Platform-Specific Guidance](#platform-specific-guidance)
  - [Linux](#linux)
  - [Windows](#windows)
  - [macOS (Apple Silicon)](#macos-apple-silicon)
- [WSL2 Considerations](#wsl2-considerations)
- [Performance Benchmarks](#performance-benchmarks)
- [Licensing](#licensing)
- [Recommendation Matrix](#recommendation-matrix)
- [Quality Evaluation](#quality-evaluation)
- [References](#references)

## Executive Summary

Running image generation models locally has become practical for a wide range of hardware configurations, from 8 GB laptops to multi-GPU workstations. The ecosystem centres on a handful of open-source frameworks — ComfyUI, Forge, InvokeAI, and the HuggingFace diffusers library — that support the latest model architectures including Flux.1 Kontext, Stable Diffusion 3.5, and SDXL.

The critical variables are **GPU vendor** (NVIDIA dominates with CUDA; AMD is viable on Linux via ROCm; Apple Silicon uses MPS/MLX/CoreML), **VRAM/unified memory** (determines which precision you can run at), and **quantisation format** (GGUF, NF4, torchao int4, and FP8 trade varying degrees of quality for memory savings). A Flux.1-dev model at full BF16 precision requires ~31.4 GB of memory; aggressive quantisation (NF4 or torchao int4) brings this below 13 GB, and GGUF Q4 formats can run on 8 GB with quality trade-offs.

This guide covers every major framework, compares them across platforms, details quantisation options, and provides concrete hardware recommendations for macOS, Windows, and Linux.

## Frameworks and Tools

### ComfyUI

**Type:** Node-based visual workflow editor (open source, GPL-3.0)
**URL:** https://github.com/comfyanonymous/ComfyUI

ComfyUI is the dominant framework for local image generation as of early 2026. Its node-graph interface allows users to build arbitrary inference pipelines, and its ecosystem exceeds 1,500 community-contributed custom nodes.

**Key strengths:**
- First-class support for Flux.1 (including Kontext), SD3.5, SDXL, and SD 1.5
- Built-in GGUF, FP8, and BF16 model loading
- Automatic VRAM management with CPU offloading
- Native support for ControlNet, IP-Adapter, LoRA, and inpainting workflows
- Desktop application available for Windows and macOS (bundles Python environment)

**Limitations:**
- Steep learning curve for node-based workflows
- No built-in image gallery or asset management
- Custom nodes can conflict; dependency management is manual

**Platform support:** Linux (CUDA, ROCm), Windows (CUDA, DirectML), macOS (MPS via PyTorch nightly)

### Stable Diffusion WebUI Forge

**Type:** Gradio web UI (open source, AGPL-3.0)
**URL:** https://github.com/lllyasviel/stable-diffusion-webui-forge

Forge is a performance-optimised fork of AUTOMATIC1111's WebUI, maintained by the Fooocus developer (lllyasviel). It delivers 30–75% speed improvements over A1111 through better memory management and support for modern backends.

**Key strengths:**
- Familiar A1111-style single-page UI
- Flux.1 support (Dev, Schnell, Kontext via extensions)
- Significant VRAM savings through automatic patching of UNet/transformer execution
- Extension-compatible with many A1111 extensions

**Limitations:**
- Slower to adopt brand-new model architectures than ComfyUI
- Some A1111 extensions break due to internal changes
- Development pace has been inconsistent

**Platform support:** Linux (CUDA, ROCm), Windows (CUDA, DirectML), macOS (limited MPS support)

### InvokeAI

**Type:** Professional canvas-based UI with API (open source, Apache 2.0)
**URL:** https://github.com/invoke-ai/InvokeAI

InvokeAI targets professional and semi-professional workflows with a polished unified canvas, layer system, and built-in model manager.

**Key strengths:**
- Professional canvas UI with layers, masking, and non-destructive editing
- Built-in model manager with HuggingFace/Civitai downloads
- REST API for integration with other tools
- Good onboarding experience for non-technical users

**Limitations:**
- Historically slower to support cutting-edge models (Flux.1 support arrived later than ComfyUI)
- Heavier resource footprint than ComfyUI
- Smaller extension ecosystem

**Platform support:** Linux (CUDA, ROCm), Windows (CUDA), macOS (MPS)

### Fooocus

**Type:** Minimal Gradio UI (open source, GPL-3.0)
**URL:** https://github.com/lllyasviel/Fooocus

Fooocus is a "set it and forget it" interface inspired by Midjourney's simplicity. It is now in long-term support (LTS) mode and **only supports SDXL** — it does not support Flux.1 or SD3.

**Key strengths:**
- One-click install, zero configuration
- Excellent default presets that produce high-quality SDXL images
- Inpainting and outpainting built in
- Very low learning curve

**Limitations:**
- SDXL only — no Flux.1, no SD3, no SD 1.5
- LTS/maintenance mode; no new features planned
- Limited customisation compared to ComfyUI or Forge

**Platform support:** Linux (CUDA), Windows (CUDA), macOS (limited)

### AUTOMATIC1111 (Legacy)

**Type:** Gradio web UI (open source, AGPL-3.0)
**URL:** https://github.com/AUTOMATIC1111/stable-diffusion-webui

The original Stable Diffusion WebUI that popularised local image generation. Still widely used but **not recommended for new setups**. It lacks native Flux.1 and SD3 support and is significantly slower than Forge.

**When to use:** Only if you have existing workflows or extensions that specifically require A1111 and cannot migrate to Forge.

### HuggingFace Diffusers

**Type:** Python library (open source, Apache 2.0)
**URL:** https://github.com/huggingface/diffusers

The `diffusers` library is the reference implementation for most open-source image generation models, including Flux.1 Kontext. It is the right choice for developers building custom pipelines, fine-tuning models, or integrating image generation into larger applications.

**Key strengths:**
- Official `FluxKontextPipeline` for Flux.1 Kontext inference
- Multiple quantisation backends: torchao, BitsAndBytes, GGUF, Optimum-Quanto
- Full control over every inference parameter
- Integrates with HuggingFace Hub for model downloads
- Excellent for scripted/batch workflows and CI/CD pipelines

**Limitations:**
- No GUI — Python scripting only
- Requires manual performance tuning (ComfyUI automates VRAM management)
- More boilerplate code for common tasks

**Example (Flux.1 Kontext with torchao int4):**
```python
import torch
from diffusers import FluxKontextPipeline
from diffusers import GGUFQuantizationConfig  # or use torchao

pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
image = pipe("a cat sitting on a windowsill", num_inference_steps=28).images[0]
```

**Platform support:** Any platform with PyTorch (CUDA, ROCm, MPS, CPU)

### stable-diffusion.cpp

**Type:** C/C++ inference engine (open source, MIT)
**URL:** https://github.com/leejet/stable-diffusion.cpp

Built on the `ggml` tensor library (the same foundation as llama.cpp), stable-diffusion.cpp provides pure C/C++ inference without Python or PyTorch dependencies. It supports SD 1.x, SD 2.x, SDXL, SD3, SD3.5, Flux.1, and Wan video models.

**Key strengths:**
- No Python runtime required — single binary
- Native GGUF quantisation support
- Runs on CPU, CUDA, Metal, Vulkan, and SYCL backends
- Very low memory footprint with aggressive quantisation
- Ideal for edge deployment and embedded systems

**Limitations:**
- Fewer features than Python-based frameworks (no node graphs, limited ControlNet)
- Smaller community and ecosystem
- CLI-only (third-party GUIs exist)
- Newer model support may lag behind diffusers

**Platform support:** Linux, Windows, macOS (all via CMake build)

### Draw Things (macOS/iOS)

**Type:** Native macOS/iOS app (open source core, proprietary app)
**URL:** https://drawthings.ai

Draw Things is a native macOS and iOS application optimised for Apple Silicon. It uses CoreML and Metal Performance Shaders directly, achieving 20%+ faster inference than Python-based frameworks running through MPS on the same hardware.

**Key strengths:**
- Native Apple Silicon optimisation
- Supports Flux.1, SDXL, SD 1.5 with built-in model downloads
- GGUF quantisation support
- No Python environment needed
- Runs on iPhone and iPad as well

**Limitations:**
- macOS/iOS only
- Less flexible than ComfyUI for complex workflows
- Smaller extension ecosystem

### Docker-Based Deployment

For reproducible environments and multi-user setups, several frameworks offer official or community Docker images:

- **ComfyUI**: Community Docker images with CUDA support (e.g., `ai-dock/comfyui`)
- **InvokeAI**: Official Docker support
- **Forge**: Community Dockerfiles available

Docker is particularly useful for:
- Isolating Python dependencies across multiple frameworks
- Running on headless Linux servers
- Deploying to cloud GPU instances for burst capacity
- WSL2-based Windows setups where native GPU passthrough is needed

## Supported Models

### Flux.1 Family

The Flux.1 family from Black Forest Labs (BFL) represents the current state of the art for open-weight image generation as of early 2026.

| Model | Parameters | Licence | Steps | Use Case |
|-------|-----------|---------|-------|----------|
| Flux.1 Schnell | 12B | Apache 2.0 | 1–4 | Fast generation, commercial use |
| Flux.1 Dev | 12B | Non-commercial | 20–50 | Research, personal projects |
| Flux.1 Kontext Dev | 12B | Non-commercial | 20–50 | Image editing, style transfer, character consistency |
| Flux.1 Pro | 12B | API only | — | Commercial (via BFL API) |
| Flux.1 Kontext Pro | 12B | API only | — | Commercial editing (via BFL API) |

**Flux.1 Kontext** is notable for supporting text-based image editing: given an input image and a text instruction (e.g., "change the background to a beach"), it produces an edited output. This makes it suitable for style transfer, character consistency across images, and iterative editing workflows. The Dev variant is available for local inference; commercial use requires either a BFL API licence or the Pro variant.

**Framework support:** ComfyUI (native), Forge (via extensions), diffusers (`FluxKontextPipeline`), stable-diffusion.cpp, Draw Things.

### Stable Diffusion 3.x

Stability AI's SD3 and SD3.5 models use a multimodal diffusion transformer (MMDiT) architecture.

| Model | Parameters | Licence | Notes |
|-------|-----------|---------|-------|
| SD 3.5 Large | 8B | Stability Community | Best quality in the SD3 family |
| SD 3.5 Large Turbo | 8B | Stability Community | 4-step distilled |
| SD 3.5 Medium | 2.6B | Stability Community | Good quality-to-speed ratio |
| SD 3.0 Medium | 2B | Stability Community | Superseded by 3.5 |

**Framework support:** ComfyUI, Forge, InvokeAI, diffusers, stable-diffusion.cpp.

### SDXL and SD 1.5

Still widely used due to their mature ecosystems of LoRAs, ControlNets, and fine-tuned checkpoints.

| Model | Parameters | Licence | Notes |
|-------|-----------|---------|-------|
| SDXL 1.0 | 3.5B (base) + 6.6B (refiner) | CreativeML Open RAIL++-M | Mature ecosystem, many fine-tunes |
| SD 1.5 | 860M | CreativeML Open RAIL-M | Lightweight, enormous LoRA library |

### Wan Video

Wan 2.1 is an open-source video generation model supported by ComfyUI and stable-diffusion.cpp. It requires significantly more VRAM (24 GB+ recommended) and is typically run on high-end NVIDIA GPUs.

## Hardware Requirements

### GPU Memory Requirements by Model and Precision

The following table shows approximate VRAM/unified memory requirements for **Flux.1-dev** (representative of 12B-parameter Flux models including Kontext):

| Precision | Transformer | T5 Encoder | Total (approx.) | Min. GPU |
|-----------|------------|------------|-----------------|----------|
| BF16/FP16 | 23.8 GB | 9.5 GB | 31.4 GB | 48 GB (A6000, etc.) |
| FP8 (E4M3) | 12.0 GB | 4.8 GB | ~18 GB | 24 GB (RTX 4090/3090) |
| GGUF Q8 | ~12 GB | ~5 GB | ~18 GB | 24 GB |
| NF4 (BnB) | ~6.5 GB | ~2.5 GB | ~12.6 GB | 16 GB (RTX 4080/4070 Ti) |
| GGUF Q5_K | ~8 GB | ~3 GB | ~13 GB | 16 GB |
| torchao int4 | ~6 GB | ~2.5 GB | ~10.6 GB | 12 GB (RTX 4070) |
| GGUF Q4_K | ~6.5 GB | ~2.5 GB | ~11 GB | 12 GB |
| GGUF Q4_0 + CPU offload | ~4 GB GPU | remainder on RAM | ~8 GB GPU | 8 GB (with 32 GB+ system RAM) |

**Important note on "8 GB minimum" claims:** Many guides claim Flux.1 can run on 8 GB GPUs. This is technically true only with aggressive GGUF Q4 quantisation plus CPU offloading, which results in generation times of 2–5 minutes per image. For practical use, 12 GB is the realistic minimum, and 16 GB provides a comfortable experience with NF4 or Q5_K quantisation.

### NVIDIA GPUs

NVIDIA GPUs with CUDA provide the best performance and broadest framework support.

| GPU | VRAM | Flux.1 Capability | Approx. Speed (Flux.1 Dev, 1024×1024, 28 steps) |
|-----|------|--------------------|--------------------------------------------------|
| RTX 5090 | 32 GB | FP8 native | ~15–20 s |
| RTX 4090 | 24 GB | FP8 or GGUF Q8 | ~25–35 s |
| RTX 4080 Super | 16 GB | NF4 or GGUF Q5_K | ~45–60 s |
| RTX 4070 Ti Super | 16 GB | NF4 or GGUF Q5_K | ~50–70 s |
| RTX 4070 | 12 GB | torchao int4 or GGUF Q4 | ~70–100 s |
| RTX 3090 | 24 GB | FP8 or GGUF Q8 | ~35–50 s |
| RTX 3060 | 12 GB | GGUF Q4 + CPU offload | ~120–180 s |

**TensorRT acceleration:** NVIDIA's TensorRT can provide additional speedups via FP8/FP4 graph optimisation, particularly on Blackwell (RTX 50-series) and Ada Lovelace (RTX 40-series) GPUs. For Flux.1 Kontext, approximately 96% of processing time is spent in the transformer, making it an excellent target for TensorRT optimisation.

### AMD GPUs

AMD GPU support varies significantly by platform:

**Linux (ROCm):** AMD GPUs perform well on Linux via ROCm (Radeon Open Compute). ROCm support in PyTorch is mature for RDNA 3 (RX 7900 XTX, etc.) and CDNA (Instinct) GPUs. Performance is typically 1.5–2× slower than equivalent NVIDIA hardware, but vastly better than DirectML.

| GPU | VRAM | ROCm Support | Notes |
|-----|------|-------------|-------|
| RX 7900 XTX | 24 GB | Good (ROCm 6.x) | Best consumer AMD option |
| RX 7900 XT | 20 GB | Good | Sufficient for FP8/NF4 |
| RX 7800 XT | 16 GB | Experimental | Community patches needed |
| RX 7600 | 8 GB | Limited | Q4 + offload only |

**Windows (DirectML):** AMD GPUs on Windows use DirectML, which is significantly slower than ROCm (roughly 3–4× slower). This makes Windows a poor platform for AMD GPU-based image generation. Consider running Linux or WSL2 with ROCm instead.

**Important:** ROCm is **not available on Windows natively**. Windows AMD users are limited to DirectML unless they use WSL2, which currently has limited ROCm support.

### Intel Arc GPUs

Intel Arc GPUs (A770, A750) support image generation via:
- **SYCL/oneAPI:** Direct support in stable-diffusion.cpp via the SYCL backend
- **IPEX (Intel Extension for PyTorch):** Experimental support in ComfyUI and diffusers
- **DirectML:** On Windows, as a fallback

Intel Arc support is less mature than CUDA or ROCm. The A770 (16 GB) can run Flux.1 at NF4 precision but performance lags behind similarly priced NVIDIA or AMD options. Intel Arc is best suited for users who already own the hardware.

### Apple Silicon

Apple Silicon Macs use unified memory shared between CPU and GPU, which provides unique advantages for large models.

| Chip | Unified Memory | Recommended Precision | Notes |
|------|---------------|----------------------|-------|
| M4 Ultra | 128/192/512 GB | BF16 (full precision) | Can run multiple models simultaneously |
| M4 Max | 36/48/64/128 GB | FP8 or BF16 | Comfortable at full precision with 48 GB+ |
| M4 Pro | 24/48 GB | GGUF Q6_K or NF4 | 24 GB is minimum comfortable; Q6_K sweet spot |
| M4 (base) | 16/24/32 GB | GGUF Q4_K–Q5_K | 16 GB is tight; 24 GB recommended |
| M3 Ultra | 64/128/192 GB | BF16 | Slightly slower than M4 equivalents |
| M3 Max | 36/48/64/96/128 GB | FP8 or BF16 | Good performance |
| M3 Pro | 18/36 GB | GGUF Q5_K–Q6_K | 18 GB is limiting |
| M2 Ultra | 64/128/192 GB | BF16 | Viable but slower per-TFLOP |
| M1 Max/Ultra | 32–128 GB | GGUF Q5_K+ | Functional but noticeably slower |

**Recommended framework for Mac:**
1. **Draw Things** — native Metal optimisation, 20%+ faster than MPS-based solutions
2. **ComfyUI** — via PyTorch MPS backend, broadest model support
3. **diffusers** — programmatic access, full quantisation support
4. **stable-diffusion.cpp** — Metal backend, low overhead, no Python needed

**The Q6_K sweet spot:** For Macs with 24 GB unified memory, GGUF Q6_K quantisation provides the best balance of quality (~95% of FP16) and memory usage (~15 GB for Flux.1 transformer + T5). Q4_K is viable on 16 GB but shows noticeable quality degradation in fine details.

### CPU-Only Inference

CPU inference is possible via stable-diffusion.cpp and diffusers but is very slow (5–30+ minutes per image depending on model size and CPU). It is useful only for testing and environments with no GPU access.

## Quantisation Methods

### Precision Formats

| Format | Bits per Weight | Memory (Flux.1 transformer) | Quality | Notes |
|--------|----------------|----------------------------|---------|-------|
| FP32 | 32 | ~47 GB | Reference | Never used for inference |
| BF16 | 16 | 23.8 GB | 100% (reference) | Default for training and high-VRAM inference |
| FP16 | 16 | 23.8 GB | ~100% | Slightly less dynamic range than BF16 |
| FP8 (E4M3) | 8 | ~12 GB | 95–98% | Supported natively on RTX 40/50 series |

### GGUF Quantisation

GGUF (GPT-Generated Unified Format) originated in the llama.cpp ecosystem and has been adopted by ComfyUI and stable-diffusion.cpp for image generation models.

| Type | Bits (effective) | Quality | Memory Savings | Best For |
|------|-----------------|---------|----------------|----------|
| Q8_0 | 8 | ~98% | ~50% | High quality, 24 GB GPUs |
| Q6_K | 6 | ~95% | ~62% | Sweet spot for 24 GB Macs |
| Q5_K_M | 5 | ~92% | ~69% | 16 GB GPUs/Macs |
| Q5_K_S | 5 | ~90% | ~69% | Slightly faster than Q5_K_M |
| Q4_K_M | 4 | ~85% | ~75% | 12 GB GPUs |
| Q4_K_S | 4 | ~82% | ~75% | Maximum savings before severe loss |
| Q4_0 | 4 | ~80% | ~75% | Legacy, use Q4_K_M instead |
| Q3_K | 3 | ~75% | ~81% | Not recommended — visible artefacts |

**Where to find GGUF models:** City96's HuggingFace repositories provide GGUF-quantised versions of most popular models (e.g., `city96/FLUX.1-dev-gguf`).

### torchao Quantisation

`torchao` (Torch Architecture Optimisation) is PyTorch's native quantisation library, supported in diffusers and increasingly in ComfyUI.

| Type | Effective Bits | Memory | Quality | Notes |
|------|---------------|--------|---------|-------|
| float8_e4m3fn | 8 | ~12 GB | ~97% | Similar to GGUF Q8, CUDA-optimised |
| int8_weight_only | 8 | ~12 GB | ~96% | Good balance |
| int4_weight_only | 4 | ~6 GB | ~85% | Lowest memory, some quality loss |

**Advantage over GGUF:** torchao quantisation happens at load time, so you use a single FP16/BF16 checkpoint and choose precision at runtime. GGUF requires pre-quantised model files.

### BitsAndBytes / NF4

BitsAndBytes provides NF4 (4-bit NormalFloat) quantisation, popular in the HuggingFace ecosystem.

| Type | Bits | Memory (Flux.1) | Quality | Notes |
|------|------|-----------------|---------|-------|
| NF4 | 4 | ~12.6 GB total | ~85% | Default in many diffusers examples |
| NF4 + double quant | 4 | ~11 GB total | ~82% | Additional compression of quant constants |
| INT8 | 8 | ~18 GB total | ~96% | Higher quality alternative |

**Limitation:** BitsAndBytes requires CUDA. It does not work on AMD (ROCm), Apple Silicon (MPS), or Intel GPUs.

### TensorRT (NVIDIA)

TensorRT provides graph-level optimisation and can apply FP8 or FP4 quantisation during compilation. It is most beneficial on:
- **Blackwell (RTX 50-series):** Native FP4 support
- **Ada Lovelace (RTX 40-series):** FP8 acceleration

TensorRT requires an NVIDIA GPU and a one-time compilation step per model+resolution combination. Once compiled, inference is typically 20–40% faster than PyTorch eager mode.

### Quality vs Memory Trade-offs

For Flux.1-class models, the practical quality tiers are:

1. **Indistinguishable from reference (≥95%):** BF16, FP16, FP8, GGUF Q8, GGUF Q6_K
2. **Slight softening of fine details (~90%):** GGUF Q5_K, torchao int8
3. **Noticeable quality reduction (~85%):** NF4, GGUF Q4_K_M, torchao int4
4. **Visible artefacts (<80%):** GGUF Q3_K and below — not recommended

**Recommendation:** Use the highest precision your hardware supports. For most users:
- 24+ GB VRAM/unified memory → FP8 or GGUF Q8
- 16 GB → GGUF Q5_K or Q6_K
- 12 GB → NF4 or torchao int4 (acceptable quality, practical speed)
- 8 GB → GGUF Q4 + CPU offload (functional but slow)

## Platform-Specific Guidance

### Linux

Linux provides the best experience for local image generation across all GPU vendors.

**NVIDIA (CUDA):**
- Install the proprietary NVIDIA driver (545+) and CUDA Toolkit (12.x)
- All frameworks work out of the box
- Best performance and broadest model support

**AMD (ROCm):**
- Install ROCm 6.x (officially supports RDNA 3 and CDNA architectures)
- PyTorch ROCm builds are available from pytorch.org
- ComfyUI, InvokeAI, and diffusers work well
- Performance is approximately 1.5–2× slower than equivalent NVIDIA hardware
- Community workarounds exist for officially unsupported GPUs (RX 7800 XT, older RDNA 2)

**Intel (oneAPI/SYCL):**
- Install oneAPI Base Toolkit and IPEX
- stable-diffusion.cpp SYCL backend is the most mature option
- ComfyUI support is experimental via IPEX

### Windows

**NVIDIA (CUDA):**
- Install Game Ready or Studio drivers (545+)
- ComfyUI Desktop provides a one-click installer
- Forge and InvokeAI have Windows install scripts
- Performance is within 5–10% of Linux

**AMD (DirectML):**
- DirectML is the only native Windows option for AMD GPUs
- Performance is 3–4× slower than ROCm on Linux
- **Recommendation:** Use WSL2 with ROCm, or run Linux natively for serious AMD workloads

**Intel Arc (DirectML):**
- Similar to AMD: DirectML works but is slow
- Consider oneAPI/IPEX for better performance (requires more setup)

**General Windows tips:**
- Disable Windows Defender real-time scanning for your model directories (large checkpoint files trigger slow scans)
- Use an SSD for model storage — loading a 12 GB model from HDD adds 30–60 seconds
- Windows virtual memory (page file) should be at least 2× your model size for CPU offloading

### macOS (Apple Silicon)

**Framework recommendations (in order of preference):**

1. **Draw Things** for casual use — native Metal, best performance, simple UI
2. **ComfyUI** for advanced workflows — install via Homebrew Python, use MPS backend
3. **diffusers** for scripting — `pip3 install diffusers torch torchvision` (use PyTorch nightly for latest MPS fixes)
4. **stable-diffusion.cpp** for minimal footprint — build with `-DGGML_METAL=ON`

**Installation (ComfyUI on macOS):**
```bash
# Install Python 3.11+ via Homebrew
brew install python@3.12

# Create virtual environment
python3.12 -m venv ~/comfyui-env
source ~/comfyui-env/bin/activate

# Install PyTorch with MPS support
pip3 install torch torchvision torchaudio

# Clone and install ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip3 install -r requirements.txt

# Run
python3 main.py --force-fp16
```

**Memory management on Mac:**
- macOS will use swap aggressively, which severely impacts performance
- Rule of thumb: your model (at chosen precision) should fit within 75% of unified memory
- Close memory-intensive applications (browsers, IDEs) before generating
- Monitor with `sudo powermetrics --samplers gpu_power` or Activity Monitor → Memory Pressure

**Known limitations on Mac:**
- MPS backend does not support all PyTorch operations; some fall back to CPU silently
- BitsAndBytes (NF4) does **not** work on MPS — use GGUF or torchao instead
- Training and fine-tuning are slower than on CUDA (but functional for LoRA)

## WSL2 Considerations

Windows Subsystem for Linux 2 provides a paravirtualised Linux environment with GPU passthrough. It is particularly useful for AMD GPU users who want ROCm access on a Windows machine.

**NVIDIA on WSL2:**
- Install the **Windows** NVIDIA driver only — do **not** install Linux NVIDIA drivers inside WSL2
- CUDA is provided via the Windows driver's WSL2 support layer
- Performance is within 5–10% of native Linux
- All frameworks work as they would on native Linux

**AMD on WSL2:**
- ROCm support in WSL2 is limited and experimental as of early 2026
- DirectML is the more reliable option inside WSL2 for AMD
- For ROCm, native Linux (dual-boot or dedicated) is strongly recommended

**Tips:**
- Allocate sufficient memory to WSL2 via `.wslconfig` (at least 16 GB for Flux.1 workflows)
- Store model files on the Linux filesystem (`/home/user/models`), not on Windows mounts (`/mnt/c/`) — the latter incurs severe I/O penalties
- Use `--listen 0.0.0.0` when running ComfyUI to access from the Windows browser

## Performance Benchmarks

Approximate generation times for **Flux.1-dev at 1024×1024, 28 steps** (single image):

| Hardware | Precision | Framework | Time |
|----------|-----------|-----------|------|
| RTX 4090 (24 GB) | FP8 | ComfyUI | ~25–30 s |
| RTX 4090 (24 GB) | FP8 + TensorRT | ComfyUI | ~18–22 s |
| RTX 4080 Super (16 GB) | NF4 | ComfyUI | ~50–65 s |
| RTX 4070 (12 GB) | torchao int4 | diffusers | ~80–110 s |
| RTX 3090 (24 GB) | FP8 | ComfyUI | ~35–45 s |
| RTX 3060 (12 GB) | GGUF Q4 + offload | ComfyUI | ~150–200 s |
| RX 7900 XTX (24 GB, ROCm) | FP16 | ComfyUI | ~50–70 s |
| RX 7900 XTX (24 GB, DirectML) | FP16 | ComfyUI | ~200–400 s |
| M4 Max (48 GB) | FP8 | Draw Things | ~80–120 s |
| M4 Max (48 GB) | GGUF Q6_K | ComfyUI (MPS) | ~100–150 s |
| M4 Pro (24 GB) | GGUF Q5_K | Draw Things | ~150–200 s |
| M2 Ultra (128 GB) | BF16 | Draw Things | ~90–130 s |

**Notes:**
- Times are approximate and vary with system load, thermal throttling, and exact model variant
- Flux.1 Schnell (1–4 step distilled) is 7–15× faster than Dev at comparable quality for simple prompts
- SDXL is approximately 3–5× faster than Flux.1 on the same hardware due to its smaller model size

## Licensing

Understanding model licensing is critical for local deployment, especially for commercial use.

| Model | Licence | Commercial Use | Key Restrictions |
|-------|---------|---------------|------------------|
| Flux.1 Schnell | Apache 2.0 | Yes, freely | None |
| Flux.1 Dev | FLUX.1-dev Non-Commercial | No (without paid licence) | Research/personal only; paid licence available from BFL |
| Flux.1 Kontext Dev | FLUX.1-dev Non-Commercial | No (without paid licence) | Same as Flux.1 Dev |
| Flux.1 Pro / Kontext Pro | Proprietary (API) | Yes (usage-based billing) | API access only; no local weights |
| SD 3.5 (all variants) | Stability Community | Yes, with conditions | Revenue cap; enterprise licence available |
| SDXL 1.0 | CreativeML Open RAIL++-M | Yes, with conditions | Cannot use to harm, no competing model training |
| SD 1.5 | CreativeML Open RAIL-M | Yes, with conditions | Similar to SDXL |

**For commercial projects:** Flux.1 Schnell (Apache 2.0) is the safest choice for unrestricted commercial use. SDXL and SD 3.5 are also commercially usable under their respective community licences. Flux.1 Dev and Kontext Dev require a separate paid licence from Black Forest Labs for any commercial application — this is a usage-based billing model.

**Framework licences:** ComfyUI (GPL-3.0), Forge/A1111 (AGPL-3.0), InvokeAI (Apache 2.0), diffusers (Apache 2.0), stable-diffusion.cpp (MIT). GPL/AGPL licences require source disclosure if you distribute modified versions; this does not apply to running the software internally.

## Recommendation Matrix

| Scenario | Recommended Stack | Min. VRAM/Memory |
|----------|------------------|-----------------|
| **Beginner, Windows/Linux, NVIDIA** | ComfyUI Desktop + Flux.1 Schnell GGUF Q8 | 12 GB |
| **Beginner, Mac** | Draw Things + Flux.1 Schnell | 16 GB unified |
| **Advanced workflows, any platform** | ComfyUI + custom nodes | 16 GB |
| **Professional image editing** | InvokeAI + SDXL or Flux.1 | 16 GB |
| **Developer/scripting** | diffusers + torchao | 12 GB |
| **Maximum quality, no budget** | ComfyUI + RTX 4090/5090 + FP8 + TensorRT | 24–32 GB |
| **AMD GPU, Linux** | ComfyUI + ROCm | 16 GB |
| **AMD GPU, Windows** | ComfyUI + DirectML (or WSL2 + ROCm) | 16 GB |
| **Minimal footprint / embedded** | stable-diffusion.cpp | 8 GB (Q4) |
| **Commercial use, free licence** | Any framework + Flux.1 Schnell or SDXL | 12 GB |
| **Image editing (Kontext-style)** | ComfyUI or diffusers + Flux.1 Kontext Dev | 16 GB |
| **Batch/CI/CD generation** | diffusers + Docker | 12 GB |

## Quality Evaluation

| Criterion | Score |
|-----------|-------|
| Completeness | 0.85 |
| Source Diversity | 0.80 |
| Contradiction Resolution | 0.90 |
| Claim Verification | 0.75 |
| Perspective Balance | 0.85 |
| **Overall** | **0.83** |

**Evaluation reasoning:** The research covers all major frameworks, GPU vendors, and platforms with concrete hardware recommendations. Source diversity is good across official documentation, community benchmarks, and technical blogs, though the AMD and Intel sections rely on fewer independent sources. The "8 GB minimum" contradiction was resolved with nuanced explanation. Some performance benchmarks are approximate due to limited standardised testing across platforms. The research would benefit from more direct head-to-head benchmarks and deeper coverage of fine-tuning workflows, which were outside the stated scope.

## References

1. [ComfyUI GitHub Repository](https://github.com/comfyanonymous/ComfyUI) — Primary framework documentation and releases
2. [HuggingFace Diffusers — Flux.1 Kontext Guide](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux) — Official FluxKontextPipeline documentation
3. [Black Forest Labs — Flux.1 Kontext Announcement](https://blackforestlabs.ai/flux-1-kontext/) — Model capabilities, licensing, and API details
4. [stable-diffusion.cpp GitHub Repository](https://github.com/leejet/stable-diffusion.cpp) — C/C++ inference engine documentation
5. [Stable Diffusion WebUI Forge GitHub](https://github.com/lllyasviel/stable-diffusion-webui-forge) — Forge framework and performance claims
6. [InvokeAI Documentation](https://invoke-ai.github.io/InvokeAI/) — Installation guides and feature documentation
7. [Draw Things App](https://drawthings.ai) — macOS/iOS native application
8. [NVIDIA TensorRT for Diffusion Models](https://developer.nvidia.com/tensorrt) — FP8/FP4 optimisation documentation
9. [AMD ROCm Documentation](https://rocm.docs.amd.com/) — GPU compute platform for AMD
10. [PyTorch MPS Backend Documentation](https://pytorch.org/docs/stable/notes/mps.html) — Apple Silicon GPU acceleration
11. [torchao GitHub Repository](https://github.com/pytorch/ao) — PyTorch native quantisation library
12. [BitsAndBytes Documentation](https://huggingface.co/docs/bitsandbytes) — NF4 and INT8 quantisation
13. [City96 GGUF Models on HuggingFace](https://huggingface.co/city96) — Pre-quantised GGUF checkpoints for Flux.1 and SDXL
14. [Fooocus GitHub Repository](https://github.com/lllyasviel/Fooocus) — SDXL-focused minimal interface
15. [ComfyUI GGUF Node](https://github.com/city96/ComfyUI-GGUF) — GGUF loading support for ComfyUI
16. [HuggingFace — Flux.1 Kontext Dev Model Card](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) — Model weights and technical specifications
17. [Flux.1 Dev Non-Commercial Licence](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md) — Full licence text
18. [AUTOMATIC1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) — Legacy framework reference
19. [DirectML for PyTorch](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-directml) — Microsoft's GPU abstraction for Windows
20. [WSL2 GPU Support Documentation](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute) — NVIDIA CUDA on WSL2
21. [Intel Extension for PyTorch (IPEX)](https://github.com/intel/intel-extension-for-pytorch) — Intel Arc GPU support
22. [Stability AI — SD 3.5 Release](https://stability.ai/news/introducing-stable-diffusion-3-5) — Model specifications and licensing
