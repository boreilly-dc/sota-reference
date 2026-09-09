# AI Kernel Programming Across Hardware and Languages

| Field | Value |
|-------|-------|
| Created | 2026-09-09 |
| Last Updated | 2026-09-09 |
| Version | 1.0 |

---

- [Executive summary](#executive-summary)
- [What an AI kernel is](#what-an-ai-kernel-is)
- [The stack: from framework to silicon](#the-stack-from-framework-to-silicon)
- [Hardware and kernel stack matrix](#hardware-and-kernel-stack-matrix)
- [NVIDIA GPUs](#nvidia-gpus)
- [AMD GPUs](#amd-gpus)
- [Intel CPUs, GPUs and NPUs](#intel-cpus-gpus-and-npus)
- [Apple Silicon](#apple-silicon)
- [Google TPUs](#google-tpus)
- [AWS Trainium and Inferentia](#aws-trainium-and-inferentia)
- [Other data-centre AI accelerators](#other-data-centre-ai-accelerators)
- [Mobile, embedded and browser accelerators](#mobile-embedded-and-browser-accelerators)
- [Programming language matrix](#programming-language-matrix)
- [Framework matrix](#framework-matrix)
- [Kernel libraries by operation](#kernel-libraries-by-operation)
- [Portability and performance](#portability-and-performance)
- [Decision guide](#decision-guide)
- [Recommendations](#recommendations)
- [References](#references)

## Executive summary

**CUDA and PyTorch are not alternatives at the same layer.** CUDA is NVIDIA's GPU programming platform. PyTorch is a model and tensor framework that selects a hardware backend and calls generated or prebuilt kernels. A typical PyTorch operation on an NVIDIA GPU follows this path:

`Python/PyTorch -> dispatcher and compiler -> ATen or Triton kernel -> cuBLAS/cuDNN/CUTLASS or generated CUDA code -> CUDA runtime and driver -> NVIDIA GPU`

The equivalent path can use ROCm on AMD, XPU and oneAPI on Intel, Metal on Apple, XLA on Google TPU, or Neuron on AWS accelerators. The correct comparison is therefore:

- **Frameworks:** PyTorch, JAX, TensorFlow/Keras and ONNX Runtime.
- **Graph compilers and intermediate representations:** TorchInductor, XLA/StableHLO, MLIR, IREE and Apache TVM.
- **Kernel languages:** CUDA C++, HIP C++, Triton, CuTe DSL, Pallas, NKI, SYCL, Metal Shading Language, Ascend C and vendor-specific languages.
- **Kernel libraries:** cuBLAS, cuDNN, CUTLASS, rocBLAS, MIOpen, Composable Kernel, oneDNN, Metal Performance Shaders and similar libraries.
- **Hardware runtimes:** CUDA, ROCm/HIP, Level Zero, Metal, TPU runtime, AWS Neuron, Qualcomm QNN and other device runtimes.

The main findings are:

1. **NVIDIA has the broadest and most mature kernel ecosystem.** CUDA C++, CUDA Python, Triton, CuTe DSL, Numba-CUDA, CUTLASS and extensive tuned libraries support both high-level users and specialist kernel engineers.
2. **AMD is the strongest second ecosystem for data-centre GPUs.** HIP offers a CUDA-like C++ path. Triton supports NVIDIA and AMD GPUs. Composable Kernel, AITER, rocBLAS, hipBLASLt and MIOpen provide tuned AI operations.
3. **Python is now a real kernel-authoring language, but only through compiled DSLs.** Triton, CuTe DSL, JAX Pallas, AWS NKI, Numba-CUDA, Helion and TileLang trace or compile restricted Python. They do not execute the Python interpreter inside a GPU or AI accelerator kernel.
4. **PyTorch has the widest practical hardware reach, but coverage is not uniform.** Its mature paths are CUDA, ROCm, CPU, Apple MPS and Intel XPU. Vendor extensions add TPU, Trainium, Gaudi, IPU, Ascend, Tenstorrent and IBM Spyre support. Each extension has different operator and custom-kernel coverage.
5. **Portable graph execution is much easier than portable hand-written kernels.** ONNX Runtime, OpenXLA, IREE, TVM, Vulkan, WebGPU and SYCL can reduce application lock-in. They cannot guarantee that one kernel source reaches peak performance on every architecture.
6. **NPUs are usually compiler targets, not open kernel targets.** Apple Neural Engine, Windows NPUs, Samsung NPUs and many mobile accelerators normally accept model graphs through Core ML, ONNX Runtime, LiteRT or a vendor SDK. Public arbitrary-kernel interfaces are limited or absent.
7. **Use prebuilt kernels first.** Write a custom kernel only when profiling shows a material unsupported operation, a fusion opportunity, a layout conversion, or a quantised primitive that graph compilation cannot solve.

### Practical default choices

| Requirement | Default choice | Reason |
|---|---|---|
| NVIDIA-only training or inference | PyTorch + `torch.compile`; Triton for custom kernels; CUDA/CUTLASS for the last performance tier | Best ecosystem depth and direct hardware control |
| NVIDIA and AMD servers | PyTorch + Triton, with vendor libraries under each backend | One high-level kernel language covers both main GPU families, subject to target tuning |
| AMD-only performance work | PyTorch ROCm + AITER/Composable Kernel; HIP C++ for low-level gaps | Uses AMD-native libraries and exposes architecture-specific features |
| Google TPU | JAX + XLA; Pallas when a custom kernel is necessary | Pallas is the supported Python kernel route for TPU |
| AWS Trainium or Inferentia | PyTorch NeuronX; NKI for custom kernels | NKI exposes the accelerator through a Python kernel DSL |
| Apple application or local inference | Core ML for deployment; MLX for research; Metal for custom GPU kernels | Core ML can select CPU, GPU and Neural Engine; MLX and Metal provide GPU control |
| Intel deployment | OpenVINO or ONNX Runtime; oneDNN for CPU/GPU primitives; PyTorch XPU for model work | Strong graph optimisation and CPU support, with a SYCL route for custom GPU code |
| Cross-vendor edge deployment | ONNX Runtime or IREE; Vulkan/WebGPU where required | Broad device coverage and smaller vendor-specific application surface |
| Browser inference | ONNX Runtime Web or a WebGPU-native framework; WGSL only for true kernel gaps | WebGPU is the standard browser compute layer |
| C++ product with portable GPU kernels | SYCL, Vulkan compute, or a backend abstraction with CUDA/HIP specialisations | Better application portability than a single-vendor API |
| Rust-native product | Burn/CubeCL or `wgpu`/WGSL; use CUDA bindings only for NVIDIA-specific hot paths | Keeps most code in Rust while retaining an escape hatch |

## What an AI kernel is

A **kernel** is a compiled function that runs on an accelerator or on a parallel CPU execution unit. In AI systems, kernels implement operations such as matrix multiplication, convolution, normalisation, attention, activation, reduction, sampling, quantisation and collective communication.

The word *kernel* is also used for several related items. These are not interchangeable:

| Term | Meaning | Examples |
|---|---|---|
| Device kernel | Code dispatched to many GPU threads, TPU tiles, NPU units or CPU vector lanes | CUDA `__global__` function, Triton `@jit` function, Metal compute function, NKI kernel |
| Kernel programming model | Rules for work partitioning, memory, synchronisation and dispatch | CUDA, HIP, SYCL, OpenCL, Metal, Pallas, NKI |
| Kernel DSL | A restricted language that generates device code | Triton, CuTe DSL, Pallas, Helion, TileLang, Mojo |
| Kernel library | A collection of tuned implementations | cuBLAS, cuDNN, CUTLASS, MIOpen, Composable Kernel, oneDNN, KleidiAI |
| Graph compiler | A compiler that selects, fuses or generates kernels from a model graph | TorchInductor, XLA, IREE, TVM, TensorRT, OpenVINO |
| AI framework | User-facing tensor, differentiation and model system | PyTorch, JAX, TensorFlow, Keras |
| Runtime or driver API | Software that loads code, allocates memory and submits work | CUDA Runtime/Driver, HIP, Level Zero, Metal, Neuron runtime |
| Operating-system kernel | The privileged core of an operating system | Linux, XNU, Windows NT; not the subject of this article |

### Prebuilt, generated and hand-written kernels

AI software uses three main sources of kernels:

1. **Prebuilt library kernels.** The framework calls a tuned GEMM, convolution or collective from a vendor or open-source library. This is usually the first and best option.
2. **Compiler-generated kernels.** A graph compiler fuses framework operations and emits a new kernel. Examples include TorchInductor emitting Triton or C++ and XLA lowering StableHLO through target-specific code generation.
3. **Hand-written custom kernels.** A developer writes CUDA, HIP, Triton, Pallas, Metal or another kernel language for a specialised operation or a performance-critical fusion.

A framework call such as `torch.matmul(a, b)` is not itself a kernel definition. The framework chooses an implementation based on device, data type, layout, size and compiler state. The selected implementation can be a vendor library call, a generated kernel, or a custom operator.

## The stack: from framework to silicon

The following model prevents category errors:

| Layer | Main responsibility | Representative technologies |
|---|---|---|
| Model and tensor API | Express models, tensors, automatic differentiation and training | PyTorch, JAX, TensorFlow/Keras, MLX |
| Capture and graph IR | Record operations in a portable or framework-specific form | FX/Export, StableHLO, HLO, ONNX, MLIR, TOSA |
| Graph and kernel compiler | Fuse operations, select schedules, generate target code | TorchInductor, XLA, IREE, TVM, TensorRT, OpenVINO, TT-Forge |
| Operator and kernel library | Supply tuned implementations for common operations | cuDNN, CUTLASS, Composable Kernel, oneDNN, MPSGraph |
| Kernel language | Express a custom parallel operation | CUDA C++, HIP C++, Triton, CuTe DSL, Pallas, SYCL, MSL, NKI |
| Runtime and driver | Manage devices, memory, queues, streams and executable loading | CUDA, HIP, Level Zero, Metal, Neuron, QNN |
| Instruction set and matrix engine | Execute scalar, vector, tensor or matrix instructions | NVIDIA Tensor Cores, AMD Matrix Cores, Intel XMX/AMX, TPU MXU, Arm SME2 |

Two consequences follow:

- A framework can support hardware without exposing its native kernel language. For example, Core ML can use Apple Neural Engine although Apple does not publish a general-purpose ANE kernel API.
- A portable kernel language still needs per-target schedules. A Triton program can compile for NVIDIA and AMD, but tile sizes, warp or wave behaviour, memory instructions and optimal pipelines can differ.

## Hardware and kernel stack matrix

This table shows the principal supported routes. **Native custom kernels** means that developers can write device-level code with a documented toolchain. **Graph only** means that the common public route is model conversion or graph compilation, not arbitrary device code.

| Hardware | Native custom-kernel routes | Important kernel libraries | Main framework routes | Typical host languages | Access level |
|---|---|---|---|---|---|
| NVIDIA data-centre and consumer GPUs | CUDA C++, CUDA Python tile kernels, CuTe DSL, Triton/Gluon, Numba-CUDA, Mojo, TileLang | cuBLAS/cuBLASLt, cuDNN, CUTLASS, cuSPARSELt, Transformer Engine, FlashAttention, FlashInfer, xFormers | PyTorch, JAX, TensorFlow, ONNX Runtime, TensorRT, MLX CUDA backend | Python, C++, C, Julia, Rust bindings | Full |
| AMD Instinct and supported Radeon GPUs | HIP C++, Triton/Gluon, Composable Kernel/CK Tile, AITER, Mojo, TileLang | rocBLAS, hipBLASLt, MIOpen, rocWMMA, rocFFT, hipSPARSE, RCCL | PyTorch ROCm, JAX ROCm, TensorFlow ROCm, ONNX Runtime MIGraphX, vLLM/SGLang | Python, C++, C, Julia | Full on supported targets |
| Intel Xe, Arc, Data Centre GPU Max and Core Ultra GPU | SYCL/DPC++, Level Zero/OpenCL, Intel GPU kernel toolchains; Triton-derived support through the PyTorch XPU stack | oneDNN, oneMKL, XeTLA and OpenVINO kernels | PyTorch XPU, TensorFlow, JAX plug-in, OpenVINO, ONNX Runtime | Python, C++, C, SYCL, Julia | Full GPU path; NPU differs |
| x86 CPUs | C/C++, intrinsics, assembly, ISPC, OpenMP; generated C++/LLVM kernels | oneDNN, BLAS libraries, XNNPACK, FBGEMM, QNNPACK | All major frameworks and runtimes | Nearly all languages | Full |
| Arm CPUs | C/C++, Neon/SVE/SME intrinsics and assembly | Arm Compute Library, KleidiAI, XNNPACK, Kleidi integrations in llama.cpp | PyTorch, TensorFlow/LiteRT, ONNX Runtime, ExecuTorch | C, C++, Python, Rust, Swift/Kotlin through frameworks | Full |
| Apple GPU | Metal Shading Language, MLX custom Metal kernels, Metal C++ host API | Metal Performance Shaders, MPSGraph, Accelerate | Core ML, MLX, PyTorch MPS, TensorFlow Metal, JAX Metal plug-in | Swift, Objective-C, C++, Python | Full GPU path |
| Apple Neural Engine | No public arbitrary device-kernel language | Apple compiler-selected Core ML operations | Core ML and compatible conversion tools | Swift, Objective-C, Python for conversion | Graph only |
| Google TPU | Pallas with Mosaic TPU; XLA custom calls at lower integration levels | XLA-generated TPU kernels and TPU libraries | JAX, TensorFlow, PyTorch/XLA | Python; C++ for compiler/runtime integration | Full through Pallas, but experimental and constrained |
| AWS Trainium and Inferentia | Neuron Kernel Interface (NKI) | Neuron compiler/runtime libraries, Neuron collectives | PyTorch NeuronX; JAX NeuronX; TensorFlow support varies by generation | Python for models and NKI; C++ at integration layers | Full through NKI |
| Intel Gaudi | TPC-C through the TPC SDK | SynapseAI kernel libraries | PyTorch with Intel Gaudi software | Python host, TPC-C/C++ extensions | Full, vendor-specific |
| Huawei Ascend | Ascend C, a C/C++-style operator language | CANN operator libraries, ACLNN | MindSpore, PyTorch adapter, TensorFlow adapter paths | Python host, Ascend C/C++ | Full, vendor-specific |
| Tenstorrent | TT-Metalium low-level kernels | TT-NN operator library | TT-NN, TT-Forge routes from PyTorch/JAX/ONNX | C++ kernels, Python and C++ host APIs | Full, vendor-specific |
| Graphcore IPU | Poplar C++ vertex and graph programming | PopLibs | PopTorch, TensorFlow for IPU, PopART/ONNX | Python model code, C++ custom operations | Full, vendor-specific |
| Cerebras WSE | Cerebras Software Language (CSL) | Cerebras SDK libraries | Cerebras PyTorch stack; model-zoo workflows | CSL device code, Python host code | Full, vendor-specific |
| IBM Spyre | Public framework backend; low-level kernel toolchain is not generally public | Spyre software-stack kernels | `torch-spyre`, vLLM plug-ins | Python, C++ framework integration | Framework/compiler route |
| Qualcomm Adreno GPU, Hexagon DSP and HTP/NPU | OpenCL/Vulkan for GPU; QNN custom operation packages where SDK access permits | Qualcomm AI Engine Direct/QNN libraries | ONNX Runtime QNN, LiteRT/QNN delegate, vendor tools | C/C++, Python conversion tools, Java/Kotlin app layer | Mixed; NPU is mostly graph/operator-package based |
| Other Android NPUs | Usually vendor compiler and delegate, sometimes restricted custom operators | Samsung Neural/Exynos AI, MediaTek NeuroPilot and SoC-specific libraries | LiteRT, ONNX Runtime where an execution provider exists, vendor SDKs | Kotlin/Java/C++ app code, Python conversion | Mainly graph only |
| Cross-vendor desktop/mobile GPUs | Vulkan compute, OpenCL C, SYCL; WebGPU/WGSL | Vendor drivers and portable runtime libraries | IREE, TVM, ONNX Runtime, llama.cpp, MLC-based stacks | C++, C, Rust, JavaScript/TypeScript | Full GPU compute, lower AI-specific abstraction |
| Browser GPU | WGSL compute shaders through WebGPU | Browser and framework-generated shader libraries | ONNX Runtime Web, TensorFlow.js, TVM web runtimes, WebLLM-style stacks | JavaScript/TypeScript, Rust or C++ through WebAssembly | Full WebGPU compute within browser limits |

## NVIDIA GPUs

NVIDIA exposes the deepest set of kernel-development layers. A team can stay entirely in a framework, write Python DSL kernels, write CUDA C++ directly, or combine these levels.

### Native programming routes

- **CUDA C++** gives direct control over SIMT threads, blocks, clusters, memory spaces, streams, graphs and Tensor Core instructions. It is the reference route for new hardware features and precise tuning.
- **CUDA Python** includes low-level Python bindings to CUDA host APIs. NVIDIA's current CUDA programming guide also documents Python kernel programming, including tile kernels. Python code is compiled or controls compiled device code; it is not interpreted on each GPU thread.
- **CuTe C++ and CUTLASS** provide composable abstractions and templates for GEMM, convolution and related tensor operations. They expose layouts, tensors, thread-value mappings and architecture-specific matrix instructions.
- **CuTe DSL** is CUTLASS's Python-based kernel DSL. It offers JIT compilation, explicit thread and memory control, DLPack integration, and lower-level access to GPU backends or specialised intermediate representations.
- **Triton** is a Python-based language and MLIR compiler. It expresses work in blocked tensor programs rather than one scalar program per CUDA thread. It is usually easier than CUDA C++ for fused AI operators.
- **Gluon** is a lower-level programming model in the Triton project. It exposes layouts and target-specific NVIDIA and AMD intrinsics. It trades some Triton automation for more schedule control.
- **Numba-CUDA** compiles a restricted subset of Python into CUDA kernels. NVIDIA now maintains the target package. It is useful for scientific and custom array kernels, but the AI kernel ecosystem is smaller than Triton's.
- **Mojo and TileLang** are emerging routes. Mojo targets CPU and GPU code with a systems language that uses Python-like syntax. TileLang is an open-source, TVM-based, tile-oriented Python DSL. Its September 2026 support matrix lists CUDA as primary, ROCm and Apple Metal as supported, and CPU/WebGPU as experimental. Both require more ecosystem evaluation than CUDA, CUTLASS or Triton.

### Main tuned libraries

| Library | Primary scope | Typical caller |
|---|---|---|
| cuBLAS and cuBLASLt | Dense matrix multiplication and linear algebra | PyTorch, JAX, TensorFlow, TensorRT, custom C++/Python applications |
| cuDNN | Deep-neural-network operations and fused execution plans | PyTorch, TensorFlow, JAX/XLA |
| CUTLASS | Source-level and generated GEMM, convolution and tensor kernels | Framework extensions and specialist kernel code |
| cuSPARSELt | Structured sparse matrix operations | Sparse training and inference stacks |
| cuFFT, cuSPARSE and cuSOLVER | FFT, sparse algebra and solvers | Scientific AI and framework operations |
| NCCL | Multi-GPU collectives rather than single-device compute | Distributed training and inference |
| Transformer Engine | Low-precision transformer operations, including FP8-class paths | PyTorch and JAX transformer training |
| FlashAttention, FlashInfer and xFormers | Attention and serving primitives | PyTorch, vLLM, SGLang and model servers |

### PyTorch on NVIDIA

PyTorch can use prebuilt ATen/CUDA operators and NVIDIA libraries. `torch.compile` uses TorchInductor to fuse and generate code. GPU code generation often uses Triton, but this is an implementation choice rather than the definition of PyTorch.

Developers can add:

- a C++/CUDA custom operator through `torch.utils.cpp_extension` and `torch.library`;
- a user-defined Triton kernel called from compiled PyTorch code;
- a structured `torch.library.triton_op`, which allows `torch.compile` to trace into its implementation and supports PyTorch subsystem registration; or
- a library kernel through a C, C++ or Python binding.

Use CUDA C++ or CuTe when the operation needs architecture-specific instructions, persistent scheduling, precise register/shared-memory control or the newest Tensor Core path. Use Triton when development speed, fusion and integration with `torch.compile` are more important than complete hardware control.

## AMD GPUs

ROCm is AMD's compute platform. HIP is its C++ runtime API and kernel programming language. HIP deliberately aligns much of its source model and API with CUDA, and HIPIFY tools can translate a significant subset of CUDA source. Source similarity does not remove the need to retune for AMD's wavefronts, memory hierarchy and matrix instructions.

### Native programming routes

- **HIP C++** is the general-purpose low-level route. It supports runtime compilation, device intrinsics, cooperative groups and the usual grid/block kernel model.
- **Triton** has production use on supported AMD GPUs. A kernel can share high-level source with an NVIDIA target, but target-specific configurations and intrinsics can still be necessary.
- **Gluon** exposes AMD CDNA and RDNA intrinsics as well as NVIDIA intrinsics.
- **Composable Kernel and CK Tile** are HIP C++ libraries and programming abstractions for performance-critical AI kernels. CK Tile gives explicit tensor, tile-distribution and scheduling concepts.
- **AITER** supplies AMD-optimised operators and kernels for transformer training and inference. It combines tuned kernel implementations and integration paths used by serving frameworks.
- **rocWMMA** exposes wave-level matrix multiply-accumulate operations.
- **HIPIFY** helps port CUDA source to HIP. It is a migration aid, not a performance-portability guarantee.

### Main tuned libraries

| Library | Primary scope | NVIDIA analogue |
|---|---|---|
| rocBLAS and hipBLASLt | Dense BLAS and tunable GEMM | cuBLAS and cuBLASLt |
| MIOpen | Deep-learning primitives | cuDNN |
| Composable Kernel | Source-level AI kernel templates and tile abstractions | CUTLASS |
| rocWMMA | Matrix-core intrinsics and fragments | WMMA/CUTLASS primitives |
| rocFFT and hipSPARSE | FFT and sparse operations | cuFFT and cuSPARSE |
| RCCL | Multi-GPU collectives | NCCL |
| AITER | Transformer and serving operations | Mixed analogue: Transformer Engine, FlashInfer and custom CUDA libraries |

### Framework support

PyTorch ROCm deliberately reuses the `torch.cuda` Python API in many places. This improves source compatibility but can confuse terminology: a tensor with `device="cuda"` in a ROCm build runs on AMD hardware. JAX and TensorFlow also have ROCm distributions or plug-ins. ONNX Runtime's current AMD direction centres on MIGraphX rather than assuming that an older ROCm execution-provider description remains current.

The main practical portability route between NVIDIA and AMD is **PyTorch plus Triton**, with separate tuning configurations and vendor-library fallbacks. For maximum AMD performance, use AITER, Composable Kernel or HIP for operations that do not map well to portable Triton.

## Intel CPUs, GPUs and NPUs

Intel has three distinct AI targets. They must not be treated as one backend:

1. **x86 CPUs**, including AMX matrix instructions;
2. **Intel GPUs**, including Xe, Arc and data-centre products with XMX matrix engines; and
3. **Intel NPUs**, which are mainly compiler-driven inference targets.

### CPU kernels

The dominant low-level languages are C, C++, assembly, intrinsics, OpenMP and compiler-generated LLVM/C++ code. The main AI libraries include:

- **oneDNN** for neural-network primitives and graph execution on Intel and compatible CPUs and GPUs;
- **oneMKL** for mathematical kernels;
- **FBGEMM**, **XNNPACK** and **QNNPACK** in framework and edge paths; and
- framework-generated CPU kernels from TorchInductor, XLA, TVM or IREE.

Python frameworks call these kernels. Python is not the device language. Numba and similar JIT systems can generate CPU machine code for restricted Python functions.

### Intel GPU kernels

- **SYCL/DPC++** is Intel's main open, standards-based C++ route. It can target CPUs and GPUs through oneAPI runtimes.
- **Level Zero** is a lower-level host API for Intel devices. It is not normally the best starting point for an AI kernel.
- **OpenCL C** remains available in parts of the ecosystem.
- **oneDNN and oneMKL** supply tuned operations.
- **XeTLA and related Intel templates** expose lower-level GPU tensor operations for specialist code.
- **PyTorch XPU** exposes Intel GPU execution from Python. Current PyTorch releases integrate XPU support and `torch.compile` paths. Use the exact supported device and release matrix because client and data-centre GPU coverage differs.
- **OpenVINO** is usually the best model-level route for optimised Intel inference.

Triton support for Intel has changed through external and upstream integration work. Treat it as release-specific. Confirm the exact PyTorch, Triton, compiler and GPU combination before selecting it as a portable source target. SYCL is the safer documented low-level route when Intel GPU kernel ownership is a requirement.

### Intel NPU

The normal public path is OpenVINO or ONNX Runtime with an NPU execution provider. The compiler partitions supported graph operations to the NPU and uses CPU or GPU fallbacks where necessary. This is not equivalent to writing a general SYCL kernel for the NPU. Teams that require an arbitrary NPU kernel must validate access to a device-specific SDK before choosing the hardware.

## Apple Silicon

Apple Silicon combines Arm CPU cores, an Apple GPU and Apple Neural Engine (ANE) in a unified-memory system. Each target has a different programming route.

### Apple GPU

- **Metal Shading Language (MSL)** is the native C++-like language for compute shaders and GPU kernels.
- **Metal Performance Shaders (MPS) and MPSGraph** provide tuned primitives and graph execution.
- **MLX** is an open-source array framework designed for Apple Silicon. Its custom Metal-kernel API accepts Metal source and integrates the kernel with MLX arrays. MLX 0.32 also provides Linux CPU and CUDA packages, but Apple CPU/GPU and unified memory remain its defining design.
- **PyTorch MPS** maps PyTorch operations to MPS and Metal.
- **TensorFlow Metal** is a PluggableDevice that accelerates TensorFlow training on Mac GPUs.
- **JAX Metal** uses an OpenXLA and PjRT plug-in for Apple GPU acceleration.
- **Core ML** can schedule supported model operations across CPU, GPU and ANE.

Swift, Objective-C and C++ are common application-side languages. Python is common for MLX, PyTorch, JAX and model conversion. MSL is the direct device language.

### Apple Neural Engine

Apple does not provide a public general-purpose ANE kernel language comparable to CUDA, Metal or NKI. Developers submit a supported Core ML graph. Apple's compiler and runtime decide whether an operation runs on CPU, GPU or ANE. A custom Metal operation targets the GPU, not ANE. This distinction is important when an application requires deterministic accelerator placement or a novel operator.

## Google TPUs

The default TPU software path is JAX or TensorFlow through XLA. PyTorch uses PyTorch/XLA. XLA lowers framework operations to target-specific executable code and usually removes the need to write a kernel.

**Pallas** is JAX's Python kernel language for custom GPU and TPU kernels. It gives explicit control over memory movement, block specifications and pipelining when XLA cannot produce the required fusion or schedule. The TPU backend lowers through Mosaic. Google documents the TPU path as experimental and supports a constrained subset of JAX NumPy. A portable Pallas function still needs target-specific kernel design because GPU and TPU execution and memory models differ.

PyTorch/XLA can call Pallas kernels to provide a custom-kernel route for TPU programs. This is useful for operations such as attention variants and quantised kernels that are not adequately expressed by standard PyTorch operators.

Use Pallas only after XLA profiling identifies a real kernel-level problem. Pallas exposes much more control, but it also exposes TPU memory and scheduling constraints that graph-level users can otherwise avoid.

## AWS Trainium and Inferentia

AWS Neuron is the compiler, runtime and framework integration stack for Trainium and Inferentia. The primary model-development route is PyTorch NeuronX. JAX integrations also exist in the Neuron ecosystem; availability and feature parity depend on accelerator generation and SDK release.

**Neuron Kernel Interface (NKI)** is a Python-based programming environment for custom kernels. It exposes compute and data-movement operations for NeuronCore hardware. NKI source is compiled; it is not interpreted Python on the accelerator. It is the nearest architectural equivalent to Triton or Pallas in the AWS accelerator stack.

Use NKI when:

- the framework lacks an operation;
- a model needs a specialised fusion;
- an operation has inefficient data movement; or
- a team must optimise a critical primitive for a known Trainium or Inferentia generation.

The trade-off is direct AWS hardware lock-in. A useful design is to place NKI behind a PyTorch custom operation and keep a framework implementation as the correctness reference and fallback.

## Other data-centre AI accelerators

### Intel Gaudi

Intel Gaudi uses the SynapseAI software suite. PyTorch is the main model-level interface. The suite supplies a large library of Tensor Processing Core operations and graph compilation.

The **TPC SDK** is the low-level custom-kernel route. Its TPC-C language is a C-derived device language compiled by an LLVM-based compiler. The SDK includes a simulator and debugger. This path is suitable for unsupported operators and target-specific tuning. It is not source-compatible with CUDA or HIP.

### Huawei Ascend

Huawei's CANN stack supports **Ascend C**, a C/C++-style operator-development language. Kernel functions operate on global and local tensor abstractions and use CANN APIs for data movement and computation. MindSpore exposes custom Ascend C operator integration, while PyTorch and TensorFlow deployments usually use vendor adapters or graph conversion.

Ascend C gives direct operator control, but it is specific to Ascend hardware and CANN releases. Teams must keep a portable reference operator because framework and accelerator support can differ by region, product and software release.

### Tenstorrent

Tenstorrent publishes an open-source software stack with three main levels:

- **TT-Metalium** is the low-level C++ programming model for compute and data-movement kernels on Tensix cores.
- **TT-NN** is a higher-level Python and C++ operator library.
- **TT-Forge** is an MLIR-based compiler path designed to accept models from PyTorch, JAX, ONNX and related ecosystems.

TT-Metalium gives extensive hardware control, including local SRAM and network-on-chip movement. The model differs substantially from CUDA SIMT, so kernel source is not portable even when the framework model is.

### Graphcore IPU

Graphcore's **Poplar SDK** exposes a C++ graph-programming API and C++ vertex code that runs on IPU tiles. PopLibs supplies common operations. PopTorch and TensorFlow integrations let Python model code call the stack. Custom operations normally combine Python registration with C++ Poplar graph and vertex implementations.

The programming model maps computation and data to many IPU tiles. It therefore requires different optimisation decisions from GPU thread-block programming.

### Cerebras WSE

The Cerebras SDK supports custom wafer-scale programs. Developers write device code in the **Cerebras Software Language (CSL)**, a C-like language, and use Python for host orchestration. The compiler maps processing elements and communication across the wafer-scale fabric. Cerebras framework software also supports higher-level PyTorch workflows.

CSL is a genuine low-level kernel route, but it targets a spatial dataflow architecture. Its placement and communication model is not directly portable to GPU DSLs.

### IBM Spyre

The open `torch-spyre` project integrates IBM Spyre as a PyTorch device and provides a custom TorchInductor backend. It also supports emerging vLLM integration. Public material focuses on framework and compiler integration. A broadly available low-level custom-kernel SDK is not yet comparable to CUDA, HIP, NKI or CSL. Treat custom-kernel access as a vendor-specific capability that must be confirmed for the deployment environment.

## Mobile, embedded and browser accelerators

### Qualcomm

Qualcomm AI Engine Direct, also called **QNN**, is the low-level graph API for CPU, Adreno GPU, DSP and Hexagon Tensor Processor backends. It accepts converted models from PyTorch, TensorFlow and ONNX workflows. ONNX Runtime's QNN execution provider constructs and executes QNN graphs.

Custom operation packages exist in the SDK, but availability, supported backends and tooling can depend on the product programme. The most open custom-GPU routes on Qualcomm hardware are Vulkan compute and OpenCL. HTP/NPU work is usually graph and operator-package development rather than a public general-purpose kernel language.

### Samsung and MediaTek

Samsung Neural SDK and Exynos AI Studio, and MediaTek NeuroPilot, focus on model conversion, quantisation, compilation and execution on SoC accelerators. LiteRT delegates and vendor runtimes are the usual application route. Public low-level NPU kernel access is more restricted than the CUDA, HIP, Metal, TPU Pallas or NKI models.

For these devices, design around supported graph operations. Keep unsupported operations on CPU or GPU, or obtain vendor-specific custom-operator access before fixing the architecture.

### Arm CPU and GPU

**KleidiAI** provides open-source micro-kernels for AI workloads on Arm CPUs. It includes hand-optimised routines for architecture features such as dot-product, I8MM, SVE and SME2. It is designed for integration into C and C++ frameworks. **Arm Compute Library** supplies broader CPU and Mali GPU functions for machine learning and computer vision.

Developers can also write:

- C/C++ with Neon, SVE or SME intrinsics;
- assembly for the narrowest hot paths;
- OpenCL kernels for compatible Arm GPUs; or
- Vulkan compute shaders for cross-vendor mobile GPU execution.

Use KleidiAI or an established framework backend before writing assembly. Micro-kernel work has a high testing burden across instruction-set versions, core designs, data types and edge cases.

### Vulkan, OpenCL and SYCL

These open standards are important portability layers:

- **Vulkan compute** uses SPIR-V shaders and runs across many NVIDIA, AMD, Intel, Arm and Qualcomm GPUs. It is strong for deployment and graphics-compute integration. It has less AI-specific library depth than CUDA or ROCm.
- **OpenCL** uses OpenCL C or SPIR-V and supports heterogeneous processors. Driver quality and version support vary widely.
- **SYCL** is a single-source C++ programming model. Intel DPC++ is the largest implementation, while other implementations target additional vendors. Source portability does not guarantee equal library coverage or performance.

IREE and TVM can lower model or tensor representations to these targets. This is usually more maintainable than writing every model operation as a raw shader.

### WebGPU

WebGPU exposes portable GPU compute in browsers and native implementations. Kernels are written in **WGSL**. JavaScript and TypeScript are the usual host languages; Rust and C++ can reach WebGPU natively or through WebAssembly tooling.

ONNX Runtime Web can execute supported graph partitions through WebGPU. Frameworks can generate WGSL so most application developers do not need to hand-write shaders. Direct WGSL remains useful for missing operators, data-layout transformations and fused browser-specific operations. Browser limits on memory, shader features and long-running work still apply.

## Programming language matrix

This matrix answers the inverse question: given a programming language, which hardware and kernel routes are practical?

| Language | Direct or DSL kernel routes | Principal hardware | Framework and binding routes | Assessment |
|---|---|---|---|---|
| Python | Triton/Gluon, CuTe DSL, Numba-CUDA, Pallas, NKI, Helion, TileLang, Taichi | NVIDIA, AMD, TPU, Trainium/Inferentia; target coverage varies by project | PyTorch, JAX, TensorFlow, MLX, ONNX Runtime | Best productivity; compiled subset semantics; target tuning remains necessary |
| CUDA C++ | CUDA kernels, CuTe C++, CUTLASS | NVIDIA GPUs | PyTorch/JAX/TensorFlow custom ops, TensorRT plug-ins | Highest NVIDIA control and maturity; strongest lock-in |
| HIP C++ | HIP kernels, CK Tile, Composable Kernel, rocWMMA | AMD GPUs; some HIP modes can target NVIDIA, but AMD is the practical focus | PyTorch ROCm and C++ extensions | Best low-level AMD route; CUDA-like source model eases migration |
| Standard C++ | SYCL, OpenCL host code, Vulkan host APIs, Metal C++, CPU intrinsics, vendor SDKs | CPU and most accelerator families | LibTorch, XLA clients, ONNX Runtime, OpenVINO, TVM, IREE | Broadest systems-language reach; backend abstraction still required |
| C | OpenCL C, vendor C APIs, CPU intrinsics and micro-kernels | CPUs, GPUs and embedded accelerators | ONNX Runtime C API and lightweight runtimes | Stable ABI and embedded fit; lower productivity |
| SYCL C++ | Single-source kernels and host code | Intel first; cross-vendor coverage depends on implementation | oneAPI libraries, custom framework extensions | Strong standards route; validate compiler and library maturity per target |
| Metal Shading Language | Metal compute kernels | Apple GPUs | MLX custom kernels, PyTorch/TensorFlow custom integrations, native apps | Best direct Apple GPU route; no ANE target |
| WGSL | WebGPU compute shaders | Browser-visible and native WebGPU GPUs | ONNX Runtime Web and web frameworks | Best browser kernel route; constrained feature set and debugging |
| Swift / Objective-C | Host APIs for Metal, MPS and Core ML; Metal kernels remain MSL | Apple CPU, GPU and ANE through Core ML | Core ML, native Metal, Swift model libraries | Best Apple application integration; not a separate device-kernel language |
| Rust | `wgpu`/WGSL, CubeCL, Rust-CUDA experiments, CUDA bindings such as `cudarc` | CubeCL targets NVIDIA, AMD, Apple, Vulkan/WebGPU and CPU; other routes vary | Burn, `tch-rs`, ONNX Runtime bindings | Strong safety and deployment story; CubeCL is alpha and the tuned-kernel ecosystem is smaller |
| Julia | CUDA.jl, AMDGPU.jl, oneAPI.jl | NVIDIA, AMD and Intel GPUs | Julia ML and array packages | Productive technical-computing kernels; smaller production AI ecosystem |
| Mojo | Native compiled CPU and GPU kernels, MAX custom operations | CPU, NVIDIA and AMD coverage evolves by release | MAX graphs and PyTorch custom-kernel integration | Promising unified language; newer ecosystem and toolchain |
| TPC-C | Intel Gaudi custom kernels | Gaudi | SynapseAI and PyTorch | Full target control; vendor-specific |
| Ascend C | C/C++-style custom operators | Huawei Ascend | CANN, MindSpore and adapters | Full target control; vendor-specific |
| CSL | Spatial device programs | Cerebras WSE | Cerebras SDK and Python host | Full wafer-scale control; architecture-specific |
| TT-Metalium C++ | Compute and data-movement kernels | Tenstorrent | TT-NN and TT-Forge | Full target control; architecture-specific |

### What "Python kernel" means

Python kernel systems use one of these techniques:

- trace a restricted Python function into an intermediate representation;
- JIT-compile decorated functions;
- stage Python control flow at compile time; or
- use Python only to build a lower-level kernel representation.

Normal Python features such as arbitrary objects, dynamic allocation, exceptions and unrestricted runtime control flow are generally unavailable inside device code. Each DSL has its own type, memory, synchronisation and control-flow rules.

## Framework matrix

A framework's hardware support and its custom-kernel support are separate properties. "Yes" below means a documented, practical route exists. It does not mean complete operator parity across all devices.

| Framework or compiler | NVIDIA | AMD | Intel GPU | Apple GPU | TPU | Trainium/Inferentia | Mobile/edge NPU | Custom-kernel integration |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PyTorch eager | Mature | Mature through ROCm | Supported through XPU | Supported through MPS | Through PyTorch/XLA | Through NeuronX | Through ExecuTorch, vendor backends or export | C++/CUDA/HIP extensions, `torch.library`, vendor extensions |
| `torch.compile` / TorchInductor | Mature | Mature on supported ROCm targets | Supported, release-specific | More limited than CUDA/ROCm | PyTorch/XLA compilation path | Neuron compiler path | Export and backend-specific lowering | Triton/Helion, C++ extensions, custom backends |
| JAX/XLA | Mature | Supported through ROCm plug-in/distribution | Experimental plug-in path | Metal plug-in | Native primary target | NeuronX path | Limited; export/compiler route | Pallas, XLA custom calls, FFI |
| TensorFlow/Keras | Mature CUDA path | ROCm distribution | oneDNN and plug-in paths | TensorFlow Metal | Native XLA path | SDK-dependent | LiteRT and delegates | C++ custom ops, device kernels, XLA custom calls |
| MLX | Linux CUDA backend available | No documented native target | No documented native target | Native Apple CPU/GPU | No | No | No ANE execution route | Custom Metal kernels, CUDA kernel APIs and C++ extensions |
| ONNX Runtime | CUDA and TensorRT providers | MIGraphX path | OpenVINO and other providers | Core ML provider | No general TPU provider | Neuron integration is workload-dependent | QNN, Core ML and vendor providers | Custom operators and execution providers, usually C/C++ |
| OpenVINO | CPU/GPU through supported plug-ins | Limited compared with Intel targets | Primary | CPU route; not the native Apple stack | No | No | Intel NPU primary | Graph transformations and extension APIs |
| IREE | CUDA | ROCm | Vulkan/other backends | Metal and Vulkan paths | Compiler work is target-dependent | No standard route | Vulkan and platform integrations | MLIR dialects, compiler code generation and runtime modules |
| Apache TVM | CUDA | ROCm | OpenCL/Vulkan and target backends | Metal | Community/target-dependent | Community/target-dependent | Broad compiler and runtime target set | TensorIR/TIR schedules, code generation and external libraries |
| Core ML | No | No | No | Apple GPU | No | No | Apple ANE | Model graph and custom CPU/GPU operations; no public ANE kernel language |
| TensorRT | Primary | No | No | No | No | No | NVIDIA DLA on supported Jetson systems | Plug-ins in C++ and CUDA |

### PyTorch-specific interpretation

PyTorch is both a Python API and a C++ framework. It contains a dispatcher, ATen operator library, autograd, graph capture, export and compilers. It can therefore reach kernels in several ways:

1. Dispatch an existing operator for the current device.
2. Call a vendor library such as cuDNN, MIOpen, oneDNN or MPS.
3. Let TorchInductor generate and fuse kernels.
4. Register an opaque or traceable custom operator.
5. Use an out-of-tree device backend through PyTorch's backend integration interfaces.

This means "PyTorch supports hardware X" is incomplete. A useful assessment must also ask:

- Does eager mode support the needed operators?
- Does training and autograd work?
- Does `torch.compile` work and produce efficient kernels?
- Can the model use distributed collectives?
- Can custom operators access the hardware?
- Is the toolchain upstream, vendor-maintained or community-maintained?

### ONNX Runtime and execution providers

ONNX Runtime partitions a model graph among execution providers. CUDA, TensorRT, OpenVINO, Core ML, QNN, DirectML, WebGPU and other providers do not expose identical operator sets. Unsupported nodes can fall back to CPU, which can add copies and synchronisation. Always inspect graph partitioning and profile the resulting execution.

DirectML remains a low-level DirectX 12 ML abstraction, but Microsoft's open-source DirectML repository is in maintenance mode. For new Windows NPU deployments, follow the current Windows AI and ONNX Runtime provider guidance rather than assuming that the older `torch-directml` path provides modern NPU coverage.

## Kernel libraries by operation

### Dense matrix multiplication and linear layers

| Hardware | Preferred prebuilt or template libraries | Custom route when necessary |
|---|---|---|
| NVIDIA | cuBLASLt, CUTLASS, Transformer Engine | CuTe DSL/C++, CUDA C++, Triton |
| AMD | hipBLASLt, rocBLAS, Composable Kernel | CK Tile, HIP C++, Triton, AITER |
| Intel CPU/GPU | oneDNN, oneMKL, XeTLA | SYCL/DPC++, C++ intrinsics |
| Apple GPU | MPSGraph and Metal Performance Shaders | MSL or MLX custom Metal kernel |
| Google TPU | XLA-generated operations | Pallas/Mosaic |
| AWS accelerators | Neuron compiler libraries | NKI |
| Arm CPU | KleidiAI, Arm Compute Library, XNNPACK | Intrinsics or assembly |

### Convolution

Use cuDNN on NVIDIA, MIOpen on AMD, oneDNN on Intel, MPSGraph/MPS on Apple, Arm Compute Library on Arm and the accelerator compiler on TPU or Trainium. Hand-written convolution is rarely the first useful custom kernel because layout selection and algorithm search are complex.

### Attention and transformer-specific operations

The ecosystem includes:

- FlashAttention and xFormers for PyTorch GPU workloads;
- FlashInfer for LLM serving primitives;
- NVIDIA Transformer Engine for low-precision transformer training;
- AMD AITER and Composable Kernel implementations;
- Triton implementations embedded in PyTorch, vLLM, SGLang and model repositories; and
- XLA, Pallas and NKI implementations for TPU and AWS accelerators.

A library name alone does not prove portability. Inspect the repository's exact backend, architecture and data-type support. Many "PyTorch" packages contain CUDA-only kernels and need a separate AMD, CPU or accelerator implementation.

### Quantised and low-bit operations

Low-bit kernels have especially strong hardware coupling. Relevant mechanisms include:

- NVIDIA Tensor Core data types and CUTLASS/cuBLASLt implementations;
- AMD matrix instructions through hipBLASLt, rocWMMA, CK and AITER;
- Intel AMX/XMX and oneDNN;
- Arm dot-product, I8MM, SVE and SME2 through KleidiAI;
- TPU and Neuron target-specific compiler or custom-kernel paths; and
- shader-based implementations in Metal, Vulkan and WebGPU.

The declared format is not sufficient. Confirm packing layout, scale granularity, accumulation type, zero-point semantics and whether the operation is dequantise-then-multiply or a true low-bit matrix instruction.

### Distributed communication

Communication kernels and libraries are adjacent to compute kernels:

- NCCL for NVIDIA;
- RCCL for AMD;
- oneCCL for Intel and portable CPU/GPU use;
- TPU collectives through XLA;
- Neuron collectives for AWS accelerators; and
- vendor fabric APIs for IPU, Cerebras and Tenstorrent.

A custom GEMM can be fast while the end-to-end model remains slow because collectives, host scheduling or memory movement dominate. Profile the full step or request.

## Portability and performance

### Portability tiers

| Tier | Example | Source portability | Performance portability | Typical effort |
|---|---|---:|---:|---|
| Framework graph | PyTorch, JAX or ONNX model | High | Medium to high when operators are supported | Lowest |
| Portable compiler IR | StableHLO/MLIR, ONNX, IREE, TVM | High at graph level | Medium; depends on backend and schedules | Low to medium |
| Cross-vendor kernel DSL | Triton for NVIDIA and AMD; TileLang, Mojo or CubeCL where supported | Medium to high | Medium; retuning is usually required | Medium |
| Standards-based device language | SYCL, Vulkan, OpenCL, WebGPU | Medium to high | Low to medium without target libraries | Medium to high |
| Similar vendor languages | CUDA and HIP | Medium after translation | Low without architecture-specific tuning | High |
| Vendor-native kernel language | CUDA, HIP, MSL, Pallas/TPU, NKI, TPC-C, CSL | Low | Highest attainable on its target | Highest |

**Performance portability is not binary.** A kernel can compile on two devices and still use one device poorly. Important differences include:

- warp size versus wavefront size;
- shared/local memory size and bank layout;
- cache hierarchy;
- matrix-instruction shapes and supported data types;
- asynchronous-copy and tensor-memory engines;
- register limits and occupancy rules;
- launch overhead and persistent-kernel support; and
- interconnect and collective topology.

### Compiler-generated kernels versus libraries

Graph compilers are good at elementwise fusion, layout propagation, constant folding and removal of intermediate tensors. Tuned libraries are usually stronger for large GEMM and convolution families. Modern stacks combine both:

1. compile the surrounding graph;
2. call a vendor library for a large structured operation; and
3. generate fused kernels for the operations around it.

A custom kernel is justified when this hybrid path leaves a measurable gap. Typical examples are unusual attention, sparse routing, fused quantisation, custom reduction, irregular indexing or a new data type.

### Open source and proprietary boundaries

Open-source components exist at every major level:

- frameworks: PyTorch, JAX, TensorFlow and MLX;
- DSLs: Triton, Pallas, TileLang, Helion, Mojo core components and CubeCL;
- compilers: MLIR, OpenXLA, IREE and TVM;
- libraries: CUTLASS, Composable Kernel, AITER, oneDNN, KleidiAI and Arm Compute Library; and
- runtimes: substantial parts of ROCm, Intel oneAPI components, Vulkan/WebGPU implementations and vendor SDK samples.

However, drivers, firmware, profilers, binary libraries or deployment services can remain proprietary. "Open-source kernel" does not mean that the complete hardware stack is open.

## Decision guide

### Step 1: Decide whether a custom kernel is necessary

Start with this order:

1. Use the framework operation.
2. Enable graph compilation, such as `torch.compile` or JAX `jit`.
3. Use the best supported library operation.
4. Use an established open-source specialist library.
5. Write a portable DSL kernel.
6. Write a vendor-native kernel only if the previous routes do not meet the measured requirement.

This order minimises maintenance, numerical-correctness and hardware-version work.

### Step 2: Select the abstraction by ownership need

| Need | Suitable level |
|---|---|
| Run an existing model | Framework plus vendor backend or ONNX Runtime execution provider |
| Fuse common tensor operations | Graph compiler, then Triton or another generated-kernel path |
| Add one PyTorch operation on NVIDIA and AMD | `torch.library.triton_op` plus tuned Triton configurations and a framework fallback |
| Use the newest NVIDIA matrix feature | CuTe DSL/CUTLASS or CUDA C++ |
| Port CUDA source to AMD | HIPIFY, HIP C++, then profile and replace hot operations with AMD libraries or CK/AITER |
| Add a TPU operation | Pallas/Mosaic with a JAX reference implementation |
| Add a Trainium/Inferentia operation | NKI with a PyTorch reference implementation |
| Add an Apple GPU operation | MLX custom Metal kernel or MSL custom operator |
| Use Apple Neural Engine | Express a supported Core ML graph; redesign unsupported operations |
| Build portable embedded inference | ONNX Runtime, IREE or TVM with Vulkan, CPU and vendor NPU backends |
| Ship browser inference | ONNX Runtime Web/WebGPU first; direct WGSL only for missing hot operations |

### Step 3: Validate the full support matrix

Before committing to a stack, test:

- target architecture and operating system;
- framework, compiler, driver and firmware versions;
- training versus inference;
- forward and backward operations;
- dynamic shapes and variable sequence lengths;
- required precision and quantisation format;
- multi-device collectives;
- graph breaks and CPU fallbacks;
- cold compilation and cache behaviour;
- numerical tolerances against a reference implementation; and
- packaging and deployment without developer toolchains.

### Step 4: Keep a portable reference

Every custom kernel should have a plain framework or CPU reference implementation. Use it for:

- correctness tests;
- unsupported devices;
- unusual shapes or data types;
- debugging and bisecting regressions; and
- future ports.

Test empty tensors, non-contiguous layouts, alignment, tails, large indices, NaN and infinity behaviour, deterministic requirements, mixed precision, gradient correctness and concurrency.

## Recommendations

1. **Use a layered architecture.** Keep model logic in PyTorch, JAX, TensorFlow or ONNX-compatible graphs. Isolate each custom kernel behind a documented operator boundary.
2. **Choose Triton as the first custom GPU DSL for Python teams that target NVIDIA and AMD.** Maintain target-specific tuning and a reference fallback. Do not claim write-once peak performance.
3. **Use CUDA/CuTe/CUTLASS for NVIDIA-only peak work and HIP/CK/AITER for AMD-only peak work.** These stacks expose hardware features sooner and more completely.
4. **Use Pallas for TPU and NKI for AWS accelerators.** They are purpose-built Python kernel routes and map better to those architectures than GPU programming models do. Current AWS NKI documentation marks its direct bare-metal path as relevant to Trainium2 and Trainium3; check the specific Neuron generation before implementation.
5. **Treat Apple GPU and Apple Neural Engine as separate targets.** Metal supports GPU kernels. Core ML controls ANE graph deployment.
6. **Use OpenVINO and oneDNN for Intel inference first.** Use SYCL only when the project must own an Intel or cross-vendor custom GPU kernel.
7. **Use ONNX Runtime, IREE or TVM for broad deployment, not as proof of equal performance.** Inspect graph partitioning and benchmark each target.
8. **Treat mobile NPUs as graph compilers unless a documented custom-operator SDK is available to the project.** Do not design around an assumed low-level NPU API.
9. **Profile end to end.** Kernel duration alone can hide graph breaks, copies, compilation, launch overhead and collective costs.
10. **Track software versions with benchmark results.** Kernel behaviour changes with compiler, library, driver, firmware and architecture releases.

## References

### Cross-framework and compiler foundations

- PyTorch, "Using User-Defined Triton Kernels with `torch.compile`": https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html
- PyTorch, "Custom C++ and CUDA Operators": https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
- Triton documentation: https://triton-lang.org/main/index.html
- OpenXLA, XLA: https://openxla.org/xla
- OpenXLA, StableHLO: https://openxla.org/stablehlo
- MLIR project: https://mlir.llvm.org/
- IREE documentation: https://iree.dev/
- Apache TVM documentation: https://tvm.apache.org/docs/
- ONNX Runtime execution providers: https://onnxruntime.ai/docs/execution-providers/

### NVIDIA

- NVIDIA, CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-programming-guide/
- NVIDIA, CUDA Python: https://developer.nvidia.com/cuda/python
- NVIDIA, CUDA Python bindings: https://nvidia.github.io/cuda-python/cuda-bindings/latest/index.html
- NVIDIA, cuDNN: https://docs.nvidia.com/deeplearning/cudnn/latest/
- NVIDIA, cuBLAS: https://docs.nvidia.com/cuda/cublas/
- NVIDIA, CUTLASS: https://github.com/NVIDIA/cutlass
- NVIDIA, CuTe DSL introduction: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html
- NVIDIA, TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html
- NVIDIA, Numba-CUDA: https://nvidia.github.io/numba-cuda/
- NVIDIA, Transformer Engine: https://github.com/NVIDIA/TransformerEngine
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- FlashInfer: https://github.com/flashinfer-ai/flashinfer
- xFormers: https://github.com/facebookresearch/xformers

### AMD

- AMD, HIP documentation: https://rocm.docs.amd.com/projects/HIP/en/latest/
- AMD, MIOpen: https://rocm.docs.amd.com/projects/MIOpen/en/latest/
- AMD, Composable Kernel: https://rocm.docs.amd.com/projects/composable_kernel/en/latest/
- AMD, AITER: https://github.com/ROCm/aiter
- AMD, rocBLAS: https://rocm.docs.amd.com/projects/rocBLAS/en/latest/
- AMD, hipBLASLt: https://github.com/ROCm/hipBLASLt
- AMD, rocWMMA: https://rocm.docs.amd.com/projects/rocWMMA/en/latest/
- AMD, HIPIFY: https://github.com/ROCm/HIPIFY
- AMD, RCCL: https://rocm.docs.amd.com/projects/rccl/en/latest/

### Intel and CPUs

- Intel, oneAPI DPC++/C++ Compiler: https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html
- Intel, oneDNN: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html
- Intel, OpenVINO: https://docs.openvino.ai/
- Khronos, SYCL: https://www.khronos.org/sycl/
- Intel Gaudi, SynapseAI Software Suite: https://docs.habana.ai/en/latest/Gaudi_Overview/SynapseAI_Software_Suite.html
- Intel Gaudi, TPC Programming: https://docs.habana.ai/en/latest/TPC/index.html
- Arm, KleidiAI: https://github.com/ARM-software/kleidiai
- Arm Compute Library: https://www.arm.com/products/development-tools/embedded-and-software/compute-library

### Apple

- Apple, Metal Shading Language specification: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- Apple, PyTorch on Metal: https://developer.apple.com/metal/pytorch/
- Apple, TensorFlow Metal plug-in: https://developer.apple.com/metal/tensorflow-plugin/
- Apple, JAX Metal plug-in: https://developer.apple.com/metal/jax/
- Apple, Core ML: https://developer.apple.com/machine-learning/core-ml/
- MLX project: https://github.com/ml-explore/mlx
- MLX custom Metal kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- MLX repository and current installation targets: https://github.com/ml-explore/mlx

### TPU and AWS accelerators

- JAX, Pallas: https://docs.jax.dev/en/latest/pallas/index.html
- JAX, writing TPU kernels with Pallas: https://docs.jax.dev/en/latest/pallas/tpu/details.html
- AWS, Neuron Kernel Interface: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html

### Other accelerators and edge targets

- Qualcomm AI Engine Direct SDK: https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
- ONNX Runtime QNN execution provider: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- Microsoft, DirectML: https://learn.microsoft.com/en-us/windows/ai/directml/dml
- Huawei, Ascend C kernel function model: https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0014.html
- Tenstorrent, TT-Metalium: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html
- Tenstorrent, TT-NN: https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html
- Graphcore, Poplar SDK overview: https://docs.graphcore.ai/projects/sdk-overview/en/latest/overview.html
- Graphcore, custom IPU operations: https://docs.graphcore.ai/projects/custom-ops/en/latest/custom-ops.html
- Cerebras SDK: https://sdk.cerebras.ai/
- Cerebras CSL language reference: https://sdk.cerebras.ai/csl/language_index
- IBM, `torch-spyre`: https://github.com/torch-spyre/torch-spyre
- Khronos, Vulkan Compute for ML: https://docs.vulkan.org/tutorial/latest/ML_Inference/Vulkan_Compute_for_ML/01_introduction.html
- Khronos, OpenCL: https://www.khronos.org/opencl/
- WebGPU project: https://webgpu.org/
- ONNX Runtime WebGPU: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html

### Emerging language and kernel projects

- PyTorch Helion: https://docs.pytorch.org/helion/
- Modular Mojo/MAX GPU programming: https://docs.modular.com/stable/gpu/intro-tutorial/
- Modular MAX custom operations: https://docs.modular.com/max/develop/custom-ops
- TileLang: https://github.com/tile-ai/tilelang
- Taichi: https://docs.taichi-lang.org/docs/overview
- Burn: https://github.com/tracel-ai/burn
- CubeCL: https://github.com/tracel-ai/cubecl
- ThunderKittens: https://github.com/HazyResearch/ThunderKittens
- CUDA.jl: https://github.com/JuliaGPU/CUDA.jl
- AMDGPU.jl: https://amdgpu.juliagpu.org/stable/
- oneAPI.jl: https://github.com/JuliaGPU/oneAPI.jl
