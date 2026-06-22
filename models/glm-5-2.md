# GLM-5.2

| Field | Value |
|-------|-------|
| Created | 2026-06-23 |
| Last Updated | 2026-06-23 |
| Version | 1.0 |

---

- [Overview](#overview)
- [The GLM-5 lineage](#the-glm-5-lineage)
- [Architecture](#architecture)
- [Benchmark performance](#benchmark-performance)
- [LMArena standing](#lmarena-standing)
- [Pricing and access](#pricing-and-access)
- [Running GLM-5.2](#running-glm-52)
- [How GLM-5.2 compares](#how-glm-52-compares)
- [Limitations and caveats](#limitations-and-caveats)
- [References](#references)

## Overview

GLM-5.2 is the open-weight flagship from **Z.ai** (formerly Zhipu AI), released on **13 June 2026** under the permissive **MIT licence**. It is a large mixture-of-experts (MoE) model — roughly **753 billion total parameters with ~40 billion active per token** — with a **1-million-token context window**, engineered explicitly for **long-horizon, autonomous coding agents**.

The headline claim from Z.ai is that GLM-5.2 edges past OpenAI's GPT-5.5 on several multi-step engineering benchmarks while costing roughly **one-sixth** as much to run. Independent trackers are more conservative — they place the GLM-5 line competitively within the open field but below the top open models (Kimi K2.6, DeepSeek V4-Pro) on the hardest *general* reasoning composites. The pattern is consistent: GLM-5.2 is a **coding- and agent-specialised** model that punches well above its weight on software-engineering tasks, less so on broad knowledge benchmarks.

Crucially for open-source users, the weights are genuinely free to download and self-host (Hugging Face, ModelScope), with FP8 and GGUF quantisations available from day one. Z.ai monetises through its hosted API and a flat-rate GLM Coding Plan rather than by withholding the model.

## The GLM-5 lineage

GLM-5.2 is the third release in a fast cadence, each sharpening Z.ai's "agentic engineering" focus:

| Release | Date | Highlights |
|---------|------|-----------|
| **GLM-5** | 11 Feb 2026 | 744B MoE, ~200K context; first open model to reach 50 on the Artificial Analysis Intelligence Index. Reportedly trained on Chinese accelerators (Huawei Ascend) rather than Nvidia. |
| **GLM-5.1** | 7–8 Apr 2026 | ~754B MoE, 200K context; long-horizon agentic-engineering tuning, SWE-bench Pro 58.4 (vendor). Still the highest-ranked open model on the *text* LMArena (Elo 1475). |
| **GLM-5.2** | 13 Jun 2026 | ~753B MoE, **1M context**; tops several open-weight coding benchmarks (vendor), SWE-bench Pro 62.1. |

All three are open-weight and MIT-licensed. The jump to a **1M-token context** is GLM-5.2's headline architectural change, alongside higher coding-benchmark scores. The series sits in the Chinese open-weight cluster alongside DeepSeek V4-Pro, Moonshot's Kimi K2.6, and MiniMax M3.

## Architecture

GLM-5.2 is a sparse **mixture-of-experts** transformer — about **753B total / ~40B active** parameters per token. Notable design points:

- **Long-context sparse attention.** Z.ai's "IndexShare" reuses indexers across sparse-attention layers to cut per-token compute at long context, which is what makes the 1M-token window practical to serve.
- **Multi-token prediction / speculative decoding.** A built-in speculative-decoding path reportedly raises acceptance length by up to ~20%, improving throughput.
- **Configurable thinking effort.** Two reasoning levels — **High** and **Max** — trade capability against latency and compute. The `max` tier is the strongest-performing variant (and the one that appears on LMArena).
- **Agent-native features.** Native tool calling, MCP integration, and structured JSON output, with up to **128K (131,072) output tokens**.

It is **text-in, text-out** and multilingual; despite a multimodal architecture class in some model metadata, there is no verified vision/image input. The model was reportedly trained substantially on Chinese accelerators (Huawei Ascend and others).

## Benchmark performance

Z.ai's vendor figures are strong on coding and agentic tasks; independent indices are more conservative. The numbers below are **vendor-reported** unless noted, and run higher than third-party reproductions — treat them as upper bounds.

| Benchmark | GLM-5.2 | Notes |
|-----------|---------|-------|
| SWE-bench Pro | 62.1 | Vendor; Z.ai claims it leads several closed models. Up from 58.4 on GLM-5.1. |
| Terminal-Bench 2.1 | 81.0 | Vendor agentic terminal coding |
| GPQA Diamond | 91.2 | Vendor-reported |
| AIME 2026 | 99.2 | Vendor-reported |
| HLE (Humanity's Last Exam) | 40.5 (54.7 with tools) | Vendor-reported |
| Artificial Analysis Intelligence Index | ~40 (GLM-5.1 measured) | Independent; placed GLM-5.1 mid-pack among open models |

The cross-model "GLM ahead of DeepSeek, behind Kimi" SWE-bench Pro ranking comes from third-party aggregators, **not** from NIST's CAISI evaluation (which assessed only DeepSeek V4-Pro). Confirm against current leaderboards before relying on these standings.

## LMArena standing

On the **text** LMArena (Chatbot Arena) leaderboard, GLM-5.2's best variant (`glm-5.2 max`) sits at **Elo ~1471** in the 22 June 2026 snapshot — making it one of the strongest open-weight models on the board, but **marginally below its own predecessor GLM-5.1 (1475)**.

This is not a data error: both values are live in the same snapshot. It mirrors a pattern now visible across several frontier families (e.g. Anthropic's Opus 4.6 → 4.7 → 4.8 also descends on the text arena). LMArena measures **human blind-preference on open-ended chat**, which does not always track capability on **agentic coding** — exactly the axis GLM-5.2 was tuned for. GLM-5.2 leads GLM-5.1 on coding and long-horizon benchmarks while losing a few points of general-chat preference. For an Elo-vs-time view of where GLM-5.2 sits relative to the frontier, see the [Frontier AI Models benchmark](frontier-models-benchmark.md).

## Pricing and access

The **open weights are free** to download, self-host, fine-tune, and use commercially under the MIT licence, from Hugging Face (`zai-org/GLM-5.2`) and ModelScope, including FP8 and GGUF quantisations.

Hosted access (API model ID `glm-5.2`, served from the Z.ai / BigModel platform):

| Channel | Input | Output | Notes |
|---------|-------|--------|-------|
| OpenRouter | ~$1.20 / MTok | ~$4.10 / MTok | VentureBeat estimated ~one-sixth of GPT-5.5's cost for comparable long-horizon coding |
| Z.ai API (GLM-5.1 reference) | ~$1.40 / MTok | ~$4.40 / MTok | Cached input ~$0.26 / MTok |
| GLM Coding Plan | Flat-rate subscription | — | Plugs into agentic IDE/CLI coding tools |

One access caveat: hosted use routes through Chinese infrastructure, and Z.ai is on the **US Entity List** — relevant for data-residency, compliance, and government procurement. Self-hosting the open weights avoids the hosted-API concerns entirely.

## Running GLM-5.2

**Open-source / self-hosted (recommended for sensitive or regulated workloads):**

- **vLLM** and **SGLang** — production serving with tensor/expert parallelism; FP8 weights cut the GPU footprint substantially for a ~753B MoE.
- **Hugging Face Transformers** — reference implementation for experimentation and fine-tuning.
- **llama.cpp / Ollama / LM Studio** — GGUF quantisations for local and workstation use (quantised; expect quality trade-offs versus FP8/BF16).

A ~753B-parameter MoE is a multi-GPU deployment even at FP8 — plan for a multi-accelerator node (e.g. 8× high-memory GPUs) for full-precision-class serving, or use aggressive GGUF quantisation for smaller setups at reduced quality.

**Managed routes on the major hyperscalers** (deploying the open weights yourself, since GLM-5.2 is not offered as a first-party managed model):

- **AWS** — Amazon SageMaker (bring-your-own-container / LMI with vLLM) on multi-GPU instances.
- **Azure** — Azure Machine Learning managed online endpoints, or Azure Kubernetes Service with vLLM/SGLang.
- **GCP** — Vertex AI custom containers or GKE with GPU node pools.
- **Oracle** — OCI Data Science / OCI Kubernetes Engine on bare-metal GPU shapes.
- **IBM** — watsonx.ai custom foundation-model deployment.

## How GLM-5.2 compares

- **Versus other open models.** Within the Chinese open-weight cluster (DeepSeek V4-Pro, Kimi K2.6, Qwen, MiniMax M3), GLM-5.2's draws are its long-horizon agentic-coding tuning, the 1M context, and a developer-friendly coding subscription. On the independent Artificial Analysis Intelligence Index it trails Kimi K2.6 and DeepSeek V4-Pro on the hardest *general* tasks, while leading much of the open field on *coding*.
- **Versus the proprietary frontier.** For agentic coding, the best-in-class proprietary models remain **Claude Opus 4.8** and **Claude Fable 5** (and GPT-5.5 / Gemini 3.x), which still lead on the hardest general reasoning and on independent composites. GLM-5.2's pitch is reaching a large fraction of that capability on coding at roughly a sixth of the cost — and being fully self-hostable.
- **Within the GLM-5 series.** GLM-5.2 adds the 1M context and higher coding scores over GLM-5.1 (200K). On the text arena GLM-5.1 still edges it on general-chat preference (see [LMArena standing](#lmarena-standing)).

## Limitations and caveats

- **Below the closed frontier** on the hardest general-reasoning tasks, and behind the top open models on independent general composites.
- **Vendor-heavy benchmarks.** The headline "beats GPT-5.5 / Opus" claims are Z.ai-reported and need independent corroboration; the cross-model SWE-bench Pro ranking comes from aggregators, not CAISI.
- **China-hosted API and Entity List.** Hosted use carries data-residency, compliance, and procurement considerations; self-hosting the MIT weights mitigates this.
- **Text-only.** No verified vision/image input despite a multimodal architecture class in the model metadata.
- **Large to self-host.** A ~753B MoE needs multi-GPU infrastructure for full-quality serving.

## References

- Z.ai — GLM-5.2 guide and API docs: https://docs.z.ai/guides/llm/glm-5.2
- Hugging Face — `zai-org/GLM-5.2` model card: https://huggingface.co/zai-org/GLM-5.2
- VentureBeat — "Z.ai's open-weights GLM-5.2 beats GPT-5.5 on multiple long-horizon coding benchmarks for 1/6th the cost": https://venturebeat.com/technology/z-ais-open-weights-glm-5-2-beats-gpt-5-5-on-multiple-long-horizon-coding-benchmarks-for-1-6th-the-cost
- The AI Rankings — "GLM-5.2: Benchmarks, Pricing & Review": https://theairankings.com/zhipu/glm-5/
- LLM Stats — GLM-5.2 benchmarks, pricing, context window: https://llm-stats.com/models/glm-5.2
- Artificial Analysis — GLM-5.1 model page: https://artificialanalysis.ai/models/glm-5-1
- OpenRouter — `z-ai/glm-5.2`: https://openrouter.ai/z-ai/glm-5.2
- Reuters (via AOL) — "China's AI startup Zhipu releases GLM-5": https://www.aol.com/articles/chinas-ai-startup-zhipu-releases-132222546.html
- LMArena (Arena AI) text leaderboard: https://lmarena.ai
- Codersera — "GLM 5.2 Release — 1M Context, Coding-First (June 2026)": https://codersera.com/blog/glm-5-2-release-1m-context-coding-2026/

