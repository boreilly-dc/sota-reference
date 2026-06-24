# GLM-5.2

| Field | Value |
|-------|-------|
| Created | 2026-06-23 |
| Last Updated | 2026-06-24 |
| Version | 2.0 |

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

GLM-5.2 is the open-weight flagship from **Z.ai** (formerly Zhipu AI), released on **13 June 2026** under the permissive **MIT licence**. It is a large mixture-of-experts (MoE) model — Z.ai's own repository describes it as **744B total parameters with ~40B active per token** (744B-A40B) — with a **1-million-token (1,048,576) context window**, engineered explicitly for **long-horizon, autonomous coding agents**. (The "753B" total seen in some write-ups is the Hugging Face safetensors parameter count, which includes embedding/output-head weights outside the MoE budget; both numbers describe the same model.)

The headline claim from Z.ai is that GLM-5.2 edges past OpenAI's GPT-5.5 on several multi-step engineering benchmarks at a fraction of the hosted cost (see [Pricing and access](#pricing-and-access)). That comparison comes from **Z.ai's own cross-model table** (surfaced by VentureBeat), not an independent evaluation. What independent measurement *does* now show is striking: as of 16 June 2026, **Artificial Analysis ranks GLM-5.2 as the leading open-weights model on its Intelligence Index v4.1 (score 51)**, ahead of DeepSeek V4-Pro and Kimi K2.6 — a reversal of the GLM-5.1-era picture, where the line sat mid-pack. The pattern is consistent: GLM-5.2 is a **coding- and agent-specialised** model that now also leads the open field on a broad independent composite, while the very hardest *general-reasoning* crown still belongs to the closed frontier.

Crucially for open-source users, the weights are genuinely free to download and self-host (Hugging Face, ModelScope), with an **official FP8** build available from day one and **community GGUF** quantisations (Unsloth) shortly after. Z.ai monetises through its hosted API and a flat-rate GLM Coding Plan rather than by withholding the model.

## The GLM-5 lineage

GLM-5.2 is the third release in a fast cadence, each sharpening Z.ai's "agentic engineering" focus:

| Release | Date | Highlights |
|---------|------|-----------|
| **GLM-5** | 11 Feb 2026 | 744B-A40B MoE, ~200K context; first of the line, pitched at "agentic engineering". Reportedly trained on Chinese accelerators (Huawei Ascend) rather than Nvidia. |
| **GLM-5.1** | 7–8 Apr 2026 | 744B-A40B MoE, 200K context; long-horizon agentic-engineering tuning, SWE-bench Pro 58.4 (vendor). AA Intelligence Index ~40. |
| **GLM-5.2** | 13 Jun 2026 | 744B-A40B MoE, **1M context**; SWE-bench Pro 62.1 (vendor); **AA Intelligence Index 51 — top open-weights model**. |

All three are open-weight and MIT-licensed. The jump to a **1M-token context** is GLM-5.2's headline architectural change, alongside an 11-point leap on the AA Intelligence Index over GLM-5.1 (40 → 51). The series sits in the Chinese open-weight cluster alongside DeepSeek V4-Pro, Moonshot's Kimi K2.6, and MiniMax M3.

## Architecture

GLM-5.2 is a sparse **mixture-of-experts** transformer — **744B total / ~40B active** parameters per token (≈5.4% activation), pre-trained on a reported **28.5 trillion tokens**. Notable design points:

- **Long-context sparse attention.** Z.ai's "IndexShare" reuses the same indexer across every four sparse-attention layers, cutting per-token FLOPs by ~2.9× at 1M context — which is what makes the 1M-token window practical to serve. (Secondary write-ups describe a DeepSeek-style sparse-attention scheme in the upper layers with dense attention in the first few; the per-layer split is not in Z.ai's own docs.)
- **Multi-token prediction / speculative decoding.** A built-in MTP path raises speculative-decoding acceptance length by up to ~20%, improving throughput.
- **Configurable thinking effort.** Two reasoning levels — **`high`** and **`max`** (the default) — trade capability against latency and compute; thinking can be disabled with `enable_thinking=false`. The `max` tier is the strongest-performing variant.
- **Agent-native features.** Native tool/function calling, MCP integration, structured JSON output, context caching, and web search, with up to **131,072 (128K) output tokens** (note that some hosted routes cap default output lower — e.g. OpenRouter at 32K).

It is **text-in, text-out** and multilingual; vision is handled by separate models (GLM-5V-Turbo). The Hugging Face card uses an `AutoModelForMultimodalLM` class name, but that is HF infrastructure naming — the `pipeline_tag` is `text-generation` and there is no verified image/audio/video input. The model was **reportedly** trained substantially on Huawei Ascend accelerators; this is widely reported in the press but is **not stated in Z.ai's own model card** (which only lists Ascend as a supported *deployment* target).

## Benchmark performance

Z.ai's vendor figures are strong on coding and agentic tasks; independent indices run a few points lower. The "GLM-5.2" column below is **vendor-reported** unless the row says otherwise — treat vendor numbers as upper bounds.

| Benchmark | GLM-5.2 (vendor) | Independent (Artificial Analysis) | Notes |
|-----------|---------|-----------------------------------|-------|
| SWE-bench Pro | 62.1 | not independently reproduced | Up from 58.4 on GLM-5.1. No Scale SEAL / standardised-harness entry yet. |
| Terminal-Bench 2.1 | 81.0 (HF card 82.7) | **78** | Independent score ~3 pts below vendor. GLM-5.1 was ~62–63. |
| GPQA Diamond | 91.2 | **89** | Independent ~2 pts below vendor. |
| AIME 2026 | 99.2 | — | High contamination exposure; treat with caution. |
| HLE (Humanity's Last Exam) | 40.5 (54.7 with tools) | **~40** | Independent corroborates the no-tools figure. |
| Artificial Analysis Intelligence Index v4.1 | — | **51 (top open-weights)** | Ahead of MiniMax-M3 (44), DeepSeek V4-Pro (44), Kimi K2.6 (43). |

Two caveats survive verification: the cross-model "GLM ahead of GPT-5.5 / DeepSeek" rankings come from **Z.ai's own comparison table**, not from a single standardised harness — different scaffolds and retry logic shift SWE-bench-style scores by several points. And there is **no NIST CAISI evaluation of GLM-5.2**: CAISI assessed only DeepSeek V4-Pro (May 2026). Where Artificial Analysis ran the model independently, GLM-5.2's numbers held up well — a few points below the vendor figures, but enough to make it the **top open-weights model on AA's composite**.

## LMArena standing

On the **text** LMArena (Chatbot Arena) leaderboard, GLM-5.2 is among the strongest open-weight models, but its **exact Elo and its ordering relative to GLM-5.1 are snapshot- and mirror-dependent**. A 22 June snapshot had GLM-5.2's `max` variant marginally *below* GLM-5.1; an OpenLM.ai mirror checked on 24 June shows the reverse — GLM-5.2 at ~1488 Arena Elo versus GLM-5.1 at ~1467, with GLM-5.2 also ahead on the Coding arena (~1525). Because Arena Elo is a live, relative rating that shifts as models enter and votes accumulate, the safe statement is: **GLM-5.2 sits at or near the top of the open-weight field on the text arena, within a few Elo points of GLM-5.1, and clearly leads it on the coding arena** — not that either definitively outranks the other on general chat. LMArena measures human blind-preference on open-ended chat, which does not always track agentic-coding capability — the axis GLM-5.2 was tuned for. For an Elo-vs-time view, see the [Frontier AI Models benchmark](frontier-models-benchmark.md).

## Pricing and access

The **open weights are free** to download, self-host, fine-tune, and use commercially under the MIT licence, from Hugging Face (`zai-org/GLM-5.2`, plus `zai-org/GLM-5.2-FP8`) and ModelScope.

Hosted access (API model ID `glm-5.2`), verified 24 June 2026:

| Channel | Input | Output | Notes |
|---------|-------|--------|-------|
| Z.ai API (official) | $1.40 / MTok | $4.40 / MTok | Cached input $0.26 / MTok; a **"Limited-time Free"** promotion is currently listed on Z.ai's pricing page. |
| OpenRouter (z-ai first-party route) | $0.98 / MTok | $3.08 / MTok | Cached input ~$0.18 / MTok; ~20 providers, cheapest ~$0.95 / $3.00. |
| GLM Coding Plan | Flat-rate subscription (from ~$18–30/mo) | — | Plugs into agentic IDE/CLI coding tools (Claude Code, Cline, etc.). |

(The earlier ~$1.20 / $4.10 OpenRouter estimate that circulated at launch has been superseded by the live marketplace rate above.) One access caveat: hosted use routes through Chinese infrastructure, and Z.ai's parent **Zhipu AI has been on the US Entity List since January 2025** (Footnote 4 designation) — relevant for data-residency, compliance, and government procurement. Note the Entity List restricts *exports to* Zhipu; it does **not** prohibit downstream use of the already-published MIT-licensed weights. Self-hosting the open weights avoids the hosted-API data-routing concerns entirely.

## Running GLM-5.2

**Open-source / self-hosted (recommended for sensitive or regulated workloads):**

- **vLLM** (v0.23+) and **SGLang** (v0.5.13+) — production serving with tensor/expert parallelism; an **official FP8** build (`zai-org/GLM-5.2-FP8`, E4M3) roughly halves the footprint of the ~744B MoE. Hugging Face Transformers, KTransformers, and Unsloth are also supported, plus Huawei **Ascend NPU** via vLLM-Ascend / xLLM / SGLang.
- **llama.cpp / LM Studio** — **community GGUF** quantisations (Unsloth) for local and workstation use; Z.ai does not publish an official GGUF, and Ollama currently exposes only a `:cloud` tag (no local variant yet).

Indicative hardware (community-measured, treat as approximate):

| Precision | Footprint | Example hardware |
|-----------|-----------|------------------|
| BF16 (full) | ~1.5 TB | 16× H100 80GB or 8× H200 141GB |
| FP8 (official) | ~0.75 TB | 8× H200 (comfortable), 8× H100 (tight); needs Hopper+ for E4M3 |
| Q4_K_M GGUF | ~376 GB | 4× H100 / 2× H200 / 512 GB DDR5 workstation |
| ~2-bit GGUF | ~239 GB | Mac Studio M3 Ultra 256 GB (≈3–9 tok/s) |

The 1M context adds roughly 80–100 GB of KV-cache VRAM at FP8 on top of the weights.

**Managed routes on the major hyperscalers.** The **predecessor GLM-5** is offered as a first-party managed model on **AWS Bedrock** (`zai.glm-5`, 200K context, 11 regions, in-region only) and **Azure AI Foundry** (serverless, via Fireworks). **GLM-5.2 itself is not yet listed** on any hyperscaler as of 24 June 2026; no GLM managed offering was found on GCP Vertex AI, OCI, or IBM watsonx. Until GLM-5.2 appears in a catalogue, the hyperscaler path is to deploy the open weights yourself:

- **AWS** — SageMaker (bring-your-own-container / LMI with vLLM) on multi-GPU instances.
- **Azure** — Azure ML managed online endpoints, or AKS with vLLM/SGLang.
- **GCP** — Vertex AI custom containers or GKE GPU node pools.
- **Oracle** — OCI Data Science / OKE on bare-metal GPU shapes.
- **IBM** — watsonx.ai custom foundation-model deployment.

## How GLM-5.2 compares

- **Versus other open models.** On the independent Artificial Analysis Intelligence Index v4.1, GLM-5.2 (51) now **leads** the open-weight field — ahead of MiniMax-M3 (44), DeepSeek V4-Pro (44) and Kimi K2.6 (43); a WhatLLM composite tells the same story. Its draws are long-horizon agentic-coding tuning, the 1M context, and a developer-friendly coding subscription. The picture is not a clean sweep, though: **DeepSeek V4-Pro still leads on SWE-bench *Verified* (~80–84%, vendor) and competitive-programming benchmarks (LiveCodeBench, Codeforces)**, while GLM-5.2 leads on SWE-bench *Pro* and HLE-with-tools (both from Z.ai's own table).
- **Versus the proprietary frontier.** For agentic coding, the best-in-class proprietary models remain **Claude Opus 4.8** and **Claude Fable 5** (and GPT-5.5 / Gemini 3.x), which still lead on the very hardest general reasoning. On agentic-coding benchmarks GLM-5.2 trails Opus 4.8 by roughly 1–13 points depending on the test (e.g. Terminal-Bench 2.1 81.0 vs 85.0, vendor table). GLM-5.2's pitch is reaching most of that coding capability — e.g. Terminal-Bench 2.1 81.0 vs Opus 4.8's 85.0 (vendor table) — at well under half the per-token hosted cost, and being fully self-hostable.
- **Within the GLM-5 series.** GLM-5.2 adds the 1M context (from 200K) and clear coding gains over GLM-5.1, and jumps from ~40 to 51 on the AA Index. On the text arena the two are within a few Elo points and snapshot-dependent (see [LMArena standing](#lmarena-standing)).

## Limitations and caveats

- **Hardest general reasoning still belongs to the closed frontier.** GLM-5.2 leads the *open* field on AA's composite but does not top the overall (closed-inclusive) leaderboards.
- **Vendor-heavy benchmarks.** The headline "beats GPT-5.5 / near-Opus" claims are Z.ai-reported, from a self-built cross-model table; the cross-model SWE-bench Pro ranking is not from a standardised harness or from CAISI. Where Artificial Analysis re-ran the model, vendor figures came in ~2–3 points high.
- **No independent SWE-bench Pro reproduction yet**, and AIME 2026 (99.2) carries high contamination exposure.
- **China-hosted API and Entity List.** Hosted use carries data-residency, compliance, and procurement considerations (Zhipu on the US Entity List since Jan 2025); self-hosting the MIT weights mitigates this. The Entity List does not bar use of the published weights.
- **Text-only.** No verified vision/image input; vision is a separate model (GLM-5V-Turbo).
- **Large to self-host.** A ~744B MoE needs multi-GPU infrastructure (8× H200-class at FP8) for full-quality serving; aggressive GGUF quantisation trades quality for a single-node or Mac-Studio footprint.
- **Not yet a managed hyperscaler model.** Only the predecessor GLM-5 is offered managed (AWS Bedrock, Azure Foundry); GLM-5.2 must currently be self-deployed on the hyperscalers.

## References

- Z.ai — GLM-5.2 guide and API docs: https://docs.z.ai/guides/llm/glm-5.2
- Z.ai — official pricing: https://docs.z.ai/guides/overview/pricing
- Z.ai — "GLM-5.2: Built for Long-Horizon Tasks" (blog): https://z.ai/blog/glm-5.2
- Hugging Face — `zai-org/GLM-5.2` model card: https://huggingface.co/zai-org/GLM-5.2
- GitHub — `zai-org/GLM-5` (official repository, GLM-5/5.1/5.2): https://github.com/zai-org/GLM-5
- Artificial Analysis — "GLM-5.2 is the new leading open-weights model on the Artificial Analysis Intelligence Index": https://artificialanalysis.ai/articles/glm-5-2-is-the-new-leading-open-weights-model-on-the-artificial-analysis-intelligence-index
- OpenRouter — `z-ai/glm-5.2` (pricing, context): https://openrouter.ai/z-ai/glm-5.2
- Chatbot Arena (OpenLM.ai mirror) — text/coding leaderboard: https://openlm.ai/chatbot-arena/
- NIST — CAISI Evaluation of DeepSeek V4 Pro: https://www.nist.gov/news-events/news/2026/05/caisi-evaluation-deepseek-v4-pro
- US BIS — Entity List press release (Zhipu AI added): https://www.bis.gov/press-release/commerce-further-restricts-chinas-artificial-intelligence-advanced-computing-capabilities
- VentureBeat — "Z.ai's open-weights GLM-5.2 beats GPT-5.5 ... for 1/6th the cost": https://venturebeat.com/technology/z-ais-open-weights-glm-5-2-beats-gpt-5-5-on-multiple-long-horizon-coding-benchmarks-for-1-6th-the-cost
- Groundy — "GLM-5.2 Benchmarks: What 62.1% SWE-bench Pro and 99.2% AIME Actually Mean": https://groundy.com/articles/glm-5-2-benchmarks-what-62-1-swe-bench-pro-and-99-2-aime-actually-mean/
- Unsloth — GLM-5.2 local-run documentation (GGUF): https://unsloth.ai/docs/models/glm-5.2
- CodingFleet — "GLM-5.2 vs DeepSeek V4 Pro": https://codingfleet.com/blog/glm-5-2-vs-deepseek-v4-pro/
