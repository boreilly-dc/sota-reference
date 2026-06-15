# Open Models for Coding Agents: Benchmarks and Performance vs Frontier Closed Models

| Field | Value |
|-------|-------|
| Created | 2026-05-31 |
| Last Updated | 2026-06-16 |
| Version | 1.1 |

---

- [Executive Summary](#executive-summary)
- [Benchmark Landscape in 2026](#benchmark-landscape-in-2026)
- [Frontier Closed Models: The Ceiling](#frontier-closed-models-the-ceiling)
- [Large Open Models (Cloud/Multi-GPU)](#large-open-models-cloudmulti-gpu)
- [Consumer-Hardware Open Models (16–24 GB VRAM)](#consumer-hardware-open-models-1624-gb-vram)
- [Head-to-Head Comparison Tables](#head-to-head-comparison-tables)
- [The Open–Closed Gap: Benchmark by Benchmark](#the-openclosed-gap-benchmark-by-benchmark)
- [Agentic Coding: Models Designed for Tool Use](#agentic-coding-models-designed-for-tool-use)
- [Quantisation and Local Inference](#quantisation-and-local-inference)
- [Agent Frameworks for Open Models](#agent-frameworks-for-open-models)
- [Cost Analysis](#cost-analysis)
- [Limitations of Open Models](#limitations-of-open-models)
- [Recommendations](#recommendations)
- [Areas of Uncertainty](#areas-of-uncertainty)
- [References](#references)

## Executive Summary

As of June 2026, the gap between open and closed models for coding agents has narrowed dramatically on some benchmarks while remaining substantial on others. The picture depends heavily on which benchmark you trust:

- **LiveCodeBench** (algorithmic coding): open models are within 2–3 points of the closed ceiling (DeepSeek V4 Pro at 87.5% vs Claude Fable 5 at 89.8% on the Vals.ai standardised harness).
- **SWE-bench Verified** (repository-level bug fixes): open models reach 72–81% vs closed at 79–95%. **OpenAI retired this benchmark in February 2026** citing ~59% flawed tests; treat all Verified numbers as inflated and increasingly deprecated.
- **SWE-bench Pro** (contamination-resistant): now the headline benchmark, but split into two regimes. **Vendor-reported**: Claude Fable 5 at 80.3% vs the best open model (MiniMax M3) at 59.0%. **Scale SEAL standardised** (much harsher): GPT-5.4 at 59.1% vs the only open entrant, Qwen3-Coder-480B, at 38.7%. The two regimes differ by 15–30 points for the same model.
- **Aider Polyglot** (multi-language editing): a ~14-point gap persists — GPT-5 at 88.0% vs DeepSeek-V3.2-Exp Reasoner at 74.2% — **but this leaderboard is now stale** (not updated with mid-2026 models).
- **Terminal-Bench 2.0** (hard real-world tasks): the gap remains large — closed agents at 75–85% vs open at 24–52% (best open: GLM-5 at 52.4% via Terminus 2).

The best large open models (DeepSeek V4 Pro, Kimi K2.6, MiniMax M3, GLM-5/5.1) now rival Claude Opus 4.5/4.6 on SWE-bench Verified while costing 20–60× less. June 2026 brought a wave of new open releases: **MiniMax M3** (1 Jun — 59.0% SWE-bench Pro vendor, beating GPT-5.5; 1M context, multimodal), **NVIDIA Nemotron 3 Ultra** (4 Jun — the first competitive *Western* open coding model, 71.9% Verified, free on OpenRouter), and **GLM-5.2** (13 Jun — 1M context, weights release pending). On the closed side, **Claude Fable 5** (9 Jun) reset the ceiling to 95.0% Verified / 80.3% SWE-bench Pro. For consumer hardware, Qwen 3.6-27B (≈18.9 GB at Q4_K_M) and Qwen 3.6-35B-A3B (MoE, 101 tok/s on RTX 3090) remain the standout single-GPU picks, claiming 73–77% SWE-bench Verified — though these scores are still vendor-reported and not independently reproduced.

However, on the hardest agentic benchmarks (Terminal-Bench 2.0, Scale SEAL SWE-bench Pro) and in real-world multi-step coding chains, closed models maintain a significant lead. The agent scaffold matters as much as the model: the same model can vary by 15–20+ percentage points depending on the framework used.

## Benchmark Landscape in 2026

**HumanEval** is effectively saturated — all frontier models (open and closed) score 90–95%. It no longer differentiates. The benchmarks that matter in 2026 are:

| Benchmark | What It Tests | Contamination Risk | Gap Indicator |
|-----------|--------------|-------------------|---------------|
| **SWE-bench Verified** | Fix real GitHub issues (500 instances) | High — **retired by OpenAI (Feb 2026)** | Narrow but unreliable |
| **SWE-bench Pro** | Decontaminated repo tasks | Low | Medium (two regimes — see below) |
| **LiveCodeBench** | Fresh competitive-programming problems | Low (rolling updates) | Minimal (2–3 pp) |
| **Aider Polyglot** | Multi-language code editing (225 exercises) | Low | Significant (14 pp), **but stale** |
| **Terminal-Bench 2.0** | Complex real-world terminal tasks (89 tasks) | Very low | Large (25–60 pp), harness-dependent |
| **BigCodeBench Hard** | Complex function-level coding | Low | Stale (not updated for 2026 models) |

**SWE-bench Verified is now effectively deprecated.** OpenAI withdrew it in February 2026 after analysis found roughly 59% of its tests were flawed (under-specified, broken, or solvable by trivial means). Vendor-reported Verified numbers are typically 5–7 points higher than standardised evaluations (e.g., Vals.ai) and are increasingly omitted from serious comparisons.

**SWE-bench Pro now has two distinct score regimes that must not be conflated:**

1. **Vendor-reported** — each lab uses its own agent scaffold. Generous: e.g. Claude Fable 5 at 80.3%, MiniMax M3 at 59.0%.
2. **Scale SEAL standardised** — a fixed, decontaminated scaffold applied uniformly. Far harsher: the top score is GPT-5.4 at 59.1%, and the *only* open model with a public SEAL entry is Qwen3-Coder-480B at 38.7%. SEAL's commercial (private-code) split drops scores another ~10 points.

Vendor and SEAL numbers for the same model can differ by 15–30 points. **Aider Polyglot has not been refreshed** with mid-2026 models (no GPT-5.2+, Claude Fable 5, DeepSeek V4, or MiniMax M3), so its rankings are now lagging. SWE-bench Pro (SEAL) and Terminal-Bench 2.0 remain the most reliable indicators of genuine agentic capability.

## Frontier Closed Models: The Ceiling

These represent the performance ceiling that open models are measured against.

Scores are vendor-reported SWE-bench Pro unless noted. SWE-bench Verified is shown for continuity but is deprecated.

| Model | Released | SWE-bench Verified | SWE-bench Pro (vendor) | LiveCodeBench | Terminal-Bench 2.0 (best harness) |
|-------|----------|-------------------|------------------------|---------------|-------------------|
| **Claude Fable 5** | 9 Jun 2026 | 95.0% | 80.3% | 89.8% | — |
| Claude Mythos (preview) | Jun 2026 | 93.9% | 77.8% | — | — |
| Claude Opus 4.8 | 27 May 2026 | 88.6% | 69.2% | 87.8% | — |
| Claude Opus 4.7 | 16 Apr 2026 | 87.6% | 64.3% | 85.1% | 80.2% (WOZCODE) |
| GPT-5.5 | 23 Apr 2026 | 88.7% / 82.6%† | 58.6% | 85.3% | 84.7% (NexAU-AHE) |
| GPT-5.3 Codex | — | 85.0% | 56.8% | 87.3% | 78.4% (SageAgent) |
| Gemini 3.1 Pro | 19 Feb 2026 | 80.6% / 78.8%† | 54.2% | 88.5% | 80.2% (TongAgents) |
| Claude Opus 4.6 | — | 80.8% | 51.9% | — | 76.4% (Meta-Harness) |

†Second figure is the Vals.ai standardised harness (mini-SWE-agent), ~5–7 points below vendor-reported maxes.

**Vals.ai standardised harness** (fair cross-model comparison): GPT-5.5 82.6%, Claude Opus 4.7 82.0%, Gemini 3.1 Pro 78.8%. **Claude Fable 5** (9 Jun) is the new frontier ceiling on most coding metrics; the contemporaneous "Mythos" preview is a research-tier model not generally available.

## Large Open Models (Cloud/Multi-GPU)

These models require multi-GPU setups or cloud API access but approach frontier closed-model performance at a fraction of the cost.

### Tier 1: Frontier-Competitive (>75% SWE-bench Verified)

SWE-bench Pro figures here are **vendor-reported** (each lab's own scaffold) — directly comparable to the closed-model vendor column above, but 15–30 points above Scale SEAL standardised numbers. LiveCodeBench is the Vals.ai standardised score where available.

| Model | Released | Params (Total/Active) | Arch | SWE-bench Verified | SWE-bench Pro (vendor) | LiveCodeBench | Licence | API $/M (in / out) |
|-------|----------|----------------------|------|-------------------|------------------------|---------------|---------|--------------------|
| **DeepSeek V4 Pro** | 24 Apr | 1.6T / 49B | MoE | 80.6% | 55.4% | 87.5 | MIT | $0.44 / $0.87 |
| **MiniMax M3** | 1 Jun | undisclosed | MoE (MSA) | 80.5% | **59.0%** | 82.2 | Open-weight‡ | $0.60 / $2.40 |
| **Kimi K2.6** (Moonshot) | 20 Apr | 1T / 32B | MoE | 80.2% | 58.6% | 86.8 | Modified MIT | — / $4.00 |
| **GLM-5.1** (Z.ai/Zhipu) | 1 Apr | 754B / 40B | MoE | ~78% | 58.4% | 81.4 | MIT | — / $3.20 |
| **MiMo V2.5 Pro** (Xiaomi) | 27 Apr | 1.0T / 42B | MoE | 78.0% | 57.2% | — | MIT | — / $0.87 |
| **GLM-5** (Z.ai/Zhipu) | 11 Feb | 744B / 40B | MoE | 77.8% | — | 81.9 | MIT | — / $3.20 |
| **Kimi K2.5** (Moonshot) | — | 1T / 32B | MoE | 76.8% | — | 83.9 | Modified MIT | — / $2.00 |
| **DeepSeek V4 Flash** | 24 Apr | 284B / 13B | MoE | 79.0% | 52.6% | — | MIT | $0.14 / $0.28 |
| **Nemotron 3 Ultra** (NVIDIA) | 4 Jun | 550B / 55B | MoE | 71.9% | — | 86.0 | OpenMDW-1.1 | Free (OpenRouter) |

‡ MiniMax M3 was announced open-weight with the weights release promised within ~10 days of the 1 Jun launch; confirm availability before relying on self-hosting.

**Newest entrants (June 2026):**

- **MiniMax M3** (1 Jun) is the standout: its **vendor-reported 59.0% on SWE-bench Pro edges out GPT-5.5 (58.6%)**, with a 1M-token context via MiniMax Sparse Attention (MSA), native multimodal input, and ~15.6× faster decoding than M2 at 1M context — at $0.60/$2.40 per M tokens.
- **NVIDIA Nemotron 3 Ultra** (4 Jun) is the first genuinely competitive **Western** open coding model in this tier (550B/55B MoE, 71.9% Verified, 86.0 LiveCodeBench), and is free to use on OpenRouter. It partially breaks the Chinese-lab monopoly noted under Limitations.
- **GLM-5.2** (13 Jun) extends the GLM-5 line to a 1M-token context (up from 200K) with a coding-first focus and works inside Claude Code with a config change. Weights were slated for release the week of 13 Jun; **no independent benchmarks were available at the time of writing**, so it is not yet scored here.
- **DeepSeek V4 Flash** (284B/13B) delivers near-frontier coding at the cheapest frontier-class price on the market ($0.28/M output). Note that "**DeepSeek V4 Pro Max**" is not a separate model — it refers to V4 Pro running in *Think Max* reasoning mode.

### Tier 2: Strong (60–75% SWE-bench Verified)

| Model | Params (Total/Active) | Architecture | SWE-bench Verified | LiveCodeBench | Aider Polyglot | Licence |
|-------|----------------------|--------------|-------------------|---------------|----------------|---------|
| **DeepSeek-V3.2-Exp Reasoner** | 671B / 37B | MoE + CoT | 60.0% | — | 74.2% | MIT |
| **Devstral 2** (Mistral) | 123B / 123B | Dense | 72.2% | — | — | Modified MIT |
| **Qwen3-Coder-480B** | 480B / 35B | MoE (160 experts) | 69.6% (OpenHands) | 70.7 | — | Apache 2.0 |
| **DeepSeek-V3.2 Chat** | 671B / 37B | MoE | 59.0% | — | 70.2% | MIT |
| **Llama 4 Maverick** (Meta) | 400B / 17B | MoE (128 experts) | ~63% | — | 15.6% | Llama licence |

### Hardware Requirements for Large Open Models

| Model | FP16 VRAM | Q4 VRAM | Minimum Setup | Self-Hosting Cost |
|-------|-----------|---------|---------------|-------------------|
| DeepSeek V4 Pro (1.6T) | ~3.2 TB | ~800 GB | 16×H100 cluster | ~$70K/month |
| DeepSeek V3/R1 (671B) | ~1.3 TB | ~400 GB | 8×H100 | ~$36K/month |
| Qwen3-Coder-480B | ~960 GB | ~256 GB | 8×H100 | ~$36K/month |
| Llama 4 Maverick (400B) | ~900 GB | ~257 GB | 8×H100 | ~$36K/month |
| Devstral 2 (123B) | ~246 GB | ~75 GB | 2×H100 or 4×A100 | ~$8K/month |

For most users, these models are accessed via API rather than self-hosted. The DeepSeek official API ($0.27–$0.44/M input tokens) is 20–35× cheaper than frontier closed models. Third-party providers (Together AI, Fireworks, OpenRouter) add a 30–100% markup but offer additional features.

### The MoE Advantage for Coding

Mixture-of-Experts models dominate the open-model leaderboards because they offer:

1. **Lower per-token compute**: DeepSeek V3 activates only 37B of 671B parameters per token — same compute as a 37B dense model with knowledge distributed across 671B.
2. **Better throughput at API scale**: 5.76× throughput improvement and 93.3% KV cache reduction vs dense equivalents.
3. **Training efficiency**: DeepSeek V3 trained in 2.79M H800 GPU-hours — less than 1/10 the cost of Llama 3.1 405B.

The trade-off: all parameters must reside in VRAM even though only a fraction activate. This makes MoE models expensive to self-host but cheap to serve at API scale.

## Consumer-Hardware Open Models (16–24 GB VRAM)

These models run on a single consumer GPU (RTX 3090/4090/5090) or Apple Silicon Mac with 32–64 GB unified memory, using quantisation.

### Top Picks by VRAM Budget

| Model | Type | VRAM (Q4_K_M) | SWE-bench Verified | HumanEval | Speed (RTX 4090) | Best For |
|-------|------|--------------|-------------------|-----------|-------------------|----------|
| **Qwen 3.6-27B** | Dense | ~18.9 GB (Q6_K ≈22.5 GB) | 77.2%† | 92.1% | ~35 tok/s (≈78 with DFlash on 3090) | Best all-round coding |
| **Qwen 3.6-35B-A3B** | MoE (3B active) | 22.1 GB (Q4) / 16.6 GB (Q3) | 73.4%† | — | ~85–101 tok/s (RTX 3090) | Speed + quality balance |
| **Devstral Small 2** (Mistral) | Dense 24B | 14.5 GB | 68.0% | 90.1% | ~40 tok/s | Agentic multi-file refactoring |
| **GLM-4.7** (Zhipu) | Dense 9B | ~6 GB | not reported | 94.2% | ~90 tok/s | Punches far above its size on code-gen |
| **Qwen 3 Coder 30B-A3B** | MoE (3B active) | 17 GB | 60.4% (EntroPO) | ~90% | ~85 tok/s | Fast agentic loops |
| **DeepSeek R1 Distill 14B** | Dense 14B | 8 GB | ~18% | ~80% | ~60 tok/s | Reasoning/debugging (8GB cards) |
| **Codestral 25.12** (Mistral) | Dense 22B | 16 GB | ~42% | 89.7% | — | Inline completion/autocomplete |
| **Gemma 4 26B-A4B** (Google) | MoE (3.8B active) | ~15–17 GB | ~38.6% | 84.9% | ~149 tok/s (up to 600 with vLLM batching) | Maximum speed |

†**Verification status (June 2026):** Qwen 3.6-27B's 77.2% and 3.6-35B-A3B's 73.4% SWE-bench Verified scores remain **vendor-reported** using Qwen's own agent scaffold. Multiple independent reviewers report the numbers "line up directionally," but no full third-party SWE-bench reproduction outside Qwen's scaffold has been published, and real-world users note tool-use drift in long agent loops (repeating failed actions). Treat as an upper bound. A purported **"Qwen3-Coder-Next" (80B/3B active, ~70.6% SWE-bench)** appears in some aggregator tables and an arXiv preprint, but two of three independent searches could not confirm it as a released, consumer-deployable model — it has been removed from this table pending confirmation (see Areas of Uncertainty).

### Inference Speed by Hardware

| Hardware | Qwen 3.6-27B (Q4_K_M) | Qwen 3 Coder 30B-A3B (Q4) | DeepSeek R1 Distill 14B (Q4) |
|----------|----------------------|---------------------------|------------------------------|
| RTX 3090 (24 GB) | ~25 tok/s | ~65 tok/s | ~50 tok/s |
| RTX 4090 (24 GB) | ~35 tok/s | ~85 tok/s | ~60 tok/s |
| RTX 5090 (32 GB) | ~50 tok/s (Q6_K) | ~135 tok/s | ~100 tok/s |
| Apple M4 Max (48 GB) | ~42 tok/s | ~40 tok/s | ~30 tok/s |
| Apple M5 Pro (64 GB) | ~48 tok/s | — | — |

MoE models (Qwen 3 Coder 30B-A3B) are faster than dense models of similar quality because only 3B parameters activate per token despite having access to 30B of learned knowledge.

### The Small-Model Surprise: GLM-4.7 (9B)

GLM-4.7 with only 9B parameters achieves 84.9 on LiveCodeBench and 94.2% on HumanEval — rivalling models 50× larger on code generation tasks. This suggests that parameter-efficient training on code data can compress coding capability into remarkably small models. However, its SWE-bench agentic performance is not reported, and small models typically struggle with the multi-step reasoning required for real-world code agent tasks.

## Head-to-Head Comparison Tables

### LiveCodeBench (Vals.ai Standardised, June 2026)

The open–closed gap on pure algorithmic coding is now negligible — open models sit within ~2 points of the closed ceiling.

| Model | Score | Type |
|-------|-------|------|
| Claude Fable 5 | 89.78% | Closed |
| Gemini 3.1 Pro Preview | 88.48% | Closed |
| GPT-5.2 Codex | 87.99% | Closed |
| Claude Opus 4.8 | 87.82% | Closed |
| Gemini 3.5 Flash | 87.60% | Closed |
| **DeepSeek V4 Pro** | **87.48%** | **Open** |
| GPT-5.3 Codex | 87.31% | Closed |
| **Qwen3.7 Max** | **87.06%** | Closed (proprietary) |
| **Kimi K2.6** | **86.77%** | **Open** |
| **Nemotron 3 Ultra** | **85.98%** | **Open** |
| **Qwen3.6 Plus** | **85.95%** | **Open** |
| **GLM-4.7** | **82.23%** | **Open** |
| **MiniMax M3** | **82.15%** | **Open** |
| **GLM-5 / 5.1** | **81.4–81.9%** | **Open** |

### SWE-bench Pro — Vendor-Reported (June 2026)

Each lab's own scaffold. Comparable across vendors but inflated relative to Scale SEAL.

| Model | Score | Type |
|-------|-------|------|
| Claude Fable 5 | 80.3% | Closed |
| Claude Mythos (preview) | 77.8% | Closed |
| Claude Opus 4.8 | 69.2% | Closed |
| Claude Opus 4.7 | 64.3% | Closed |
| Qwen3.7 Max | 60.6% | Closed (proprietary) |
| **MiniMax M3** | **59.0%** | **Open** |
| GPT-5.5 / **Kimi K2.6** | **58.6%** | Closed / **Open** |
| **GLM-5.1** | **58.4%** | **Open** |
| **MiMo V2.5 Pro** | **57.2%** | **Open** |
| **MiniMax M2.7** | **56.2%** | **Open** |
| **DeepSeek V4 Pro** | **55.4%** | **Open** |
| Gemini 3.1 Pro | 54.2% | Closed |
| **DeepSeek V4 Flash** | **52.6%** | **Open** |

### SWE-bench Pro — Scale SEAL Standardised (June 2026)

A fixed, decontaminated scaffold applied uniformly. Far harsher; **only one open model has a public entry.**

| Model | Score | Type |
|-------|-------|------|
| GPT-5.4 (xHigh) | 59.1% | Closed |
| Claude Opus 4.6 (thinking) | 51.9% | Closed |
| Gemini 3.1 Pro (thinking) | 46.1% | Closed |
| Claude Opus 4.5 | 45.9% | Closed |
| GPT-5 (High) | 41.8% | Closed |
| GPT-5.2 Codex | 41.0% | Closed |
| **Qwen3-Coder-480B** | **38.7%** | **Open** |

### Terminal-Bench 2.0 (Official Leaderboard, June 2026)

Harness-dependent; the best harness per model is shown. The open–closed gap is the largest of any benchmark here.

| Model (harness) | Score | Type |
|-----------------|-------|------|
| GPT-5.5 (NexAU-AHE) | 84.7% | Closed |
| GPT-5.5 (Codex CLI) | 82.2% | Closed |
| Claude Opus 4.7 (WOZCODE) | 80.2% | Closed |
| Gemini 3.1 Pro (TongAgents) | 80.2% | Closed |
| **GLM-5 (Terminus 2)** | **52.4%** | **Open** |
| **MiniMax M2.7 (IndusAGI)** | **45.1%** | **Open** |
| **Kimi K2.5 (Terminus 2)** | **43.2%** | **Open** |
| **DeepSeek V3.2 (Terminus 2)** | **39.6%** | **Open** |
| **Qwen3-Coder-480B (Terminus 2)** | **23.9%** | **Open** |

Note: MiniMax M3 (66.0% on Terminal-Bench 2.1), Kimi K2.6 (66.7%) and Qwen 3.7 Plus (70.3%) report much higher *vendor* Terminal-Bench figures than the official-harness open scores above — another illustration of the 15–25-point scaffold gap.

### Aider Polyglot Leaderboard (stale — last refreshed early 2026)

This leaderboard has **not** been updated with mid-2026 models (no GPT-5.2+, Claude Fable 5, DeepSeek V4, MiniMax M3, or Qwen 3.6/3.7). Retained for historical reference only.

| Rank | Model | Score | Type |
|------|-------|-------|------|
| 1 | GPT-5 (high) | 88.0% | Closed |
| 4 | Gemini 2.5 Pro (32k think) | 83.1% | Closed |
| — | **DeepSeek-V3.2-Exp Reasoner** | **74.2%** | **Open** |
| — | Claude Opus 4 (32k thinking) | 72.0% | Closed |
| — | **DeepSeek R1 (0528)** | **71.4%** | **Open** |
| — | **DeepSeek-V3.2-Exp Chat** | **70.2%** | **Open** |
| — | **Qwen3 235B** | **59.6%** | **Open** |
| — | **Llama 4 Maverick** | **15.6%** | **Open** |

## The Open–Closed Gap: Benchmark by Benchmark

| Benchmark | Best Closed | Best Open | Gap | Trend |
|-----------|------------|-----------|-----|-------|
| HumanEval | ~95% | ~95% | **0 pp** | Saturated |
| LiveCodeBench (Vals.ai) | 89.8% (Fable 5) | 87.5% (DeepSeek V4 Pro) | **2 pp** | Effectively closed |
| SWE-bench Verified (vendor) | 95.0% (Fable 5) | 80.6% (DeepSeek V4 Pro) | **14 pp** | Benchmark deprecated; gap widened by Fable 5 |
| SWE-bench Pro (vendor) | 80.3% (Fable 5) | 59.0% (MiniMax M3) | **21 pp** | Open caught GPT-5.5 but not Claude |
| SWE-bench Pro (Scale SEAL) | 59.1% (GPT-5.4) | 38.7% (Qwen3-Coder-480B) | **20 pp** | Few open submissions; gap real |
| Aider Polyglot (stale) | 88.0% | 74.2% | **14 pp** | Leaderboard no longer updated |
| Terminal-Bench 2.0 | 84.7% (GPT-5.5) | 52.4% (GLM-5) | **32 pp** | Large gap persists |

The headline shift since May is that **Claude Fable 5 (9 Jun) pushed the closed ceiling well above the open frontier on the curated benchmarks** (SWE-bench Verified/Pro vendor), even as open models converged with GPT-5.5 and Gemini 3.1 Pro. On standardised harnesses (LiveCodeBench Vals.ai, SWE-bench Pro SEAL) the picture is more sober: open models match closed on algorithmic coding but trail by ~20 points on decontaminated repository tasks, and by ~30 on hard terminal tasks.

**Key insight**: The gap depends on task difficulty. On pure code generation (LiveCodeBench), open models have essentially caught up. On multi-step agentic tasks requiring planning, tool use, and error recovery (Terminal-Bench 2.0), closed models maintain a commanding lead. SWE-bench Verified is somewhere in between but is likely inflated for all models due to contamination.

## Agentic Coding: Models Designed for Tool Use

Several open models are now specifically trained for agentic coding rather than just code completion:

### Purpose-Built Agentic Models

| Model | Approach | SWE-bench Verified | Designed For |
|-------|----------|-------------------|-------------|
| **Devstral 2 / Small 2** (Mistral) | Fine-tuned from Mistral for code-agent tasks | 68.0% (Small 2) – 72.2% (Devstral 2) | Mistral Vibe CLI, OpenHands, general agents |
| **MiniMax M3** | MoE (MSA), agentically trained, 1M context | 80.5% (59.0% Pro vendor) | OpenHands, Cline, computer use |
| **DeepSeek V4 Pro / Flash** | MoE, improved tool-call reliability over V3 | 80.6% / 79.0% | Cloud-scale agentic coding |
| **OpenHands LM 32B** | RL fine-tuned (SWE-Gym) on agent trajectories | 37.2% | OpenHands/SWE-agent (now superseded) |
| **Skywork-SWE-32B** | RL-trained specifically for SWE-bench | 38–47% | SWE-agent workflows (now superseded) |

The older 32B agentic fine-tunes (OpenHands LM, Skywork-SWE) — both based on Qwen2.5-Coder — are now effectively obsolete: Qwen 3.6-27B reaches 77.2% Verified on a single GPU, far exceeding them. A purported **"Qwen3-Coder-Next" (80B/3B active, ~70.6%)** was previously listed here but could not be confirmed as a released model (see Areas of Uncertainty). `mini-SWE-agent` notably reaches ~74% SWE-bench Verified in ~100 lines of Python, underscoring how much the scaffold matters.

### Key Differences from Code Completion Models

Agentic coding models are trained on:
- Multi-turn tool-use trajectories (not just single-turn completion)
- Error recovery and retry patterns
- File navigation and repository understanding
- Test generation and execution feedback loops
- Environment interaction via shell commands

The gap between "code completion" and "agentic coding" performance is large. Models that score 90%+ on HumanEval may drop to 15–40% on SWE-bench when used as agents, because the tasks require planning, tool use, and multi-step reasoning.

## Quantisation and Local Inference

### Recommended Quantisation Formats

| Format | Quality Retention | Speed | Platform | Best For |
|--------|------------------|-------|----------|----------|
| **GGUF Q4_K_M** | ~92% | Moderate | CPU + GPU + Apple Silicon | Universal default |
| **GGUF Q6_K** | ~96% | Slower | CPU + GPU + Apple Silicon | Quality-sensitive tasks |
| **AWQ** (4-bit) | ~95% | 741 tok/s (vLLM Marlin) | NVIDIA GPU only | Maximum GPU throughput |
| **GPTQ** (4-bit) | ~90% | 712 tok/s (vLLM Marlin) | NVIDIA GPU only | Legacy; prefer AWQ |
| **EXL2** (mixed-bit) | Variable | Fastest single-user | NVIDIA GPU only | ExLlamaV2/TabbyAPI |
| **FP8** | ~99% | Fast | NVIDIA Ampere+ | Near-lossless if VRAM allows |
| **NVFP4** | ~92% | Fastest on Blackwell | RTX 5090 only | Blackwell-native acceleration |

**Recommendations**:
- **Apple Silicon**: GGUF is the only option. Use Q4_K_M for most models, Q6_K if RAM allows.
- **NVIDIA GPU (single card)**: AWQ for batch/server use (vLLM), GGUF Q4_K_M for single-user (llama.cpp/Ollama).
- **NVIDIA GPU (speed priority)**: EXL2 via ExLlamaV2 or TabbyAPI for single-user inference.

### Speculative Decoding and Quant Tooling (updated June 2026)

- **Block-diffusion (DFlash)** gives a measured ~2.0–2.56× mean speedup on Qwen 3.6-27B Q4_K_M on a single RTX 3090 — 78 tok/s on HumanEval-style tasks (2.24×), ~70 tok/s on Math500. The fully-trained Qwen 3.5 DFlash draft reaches 3.43× on HumanEval, roughly the ceiling once 3.6 training lands. NVIDIA-only (no Mac/AMD).
- **Multi-Token Prediction (MTP)** for Qwen 3.6 landed in **mainline llama.cpp (PR #22673, 4 May 2026)**, enabling native speculative decoding without a separate draft model. Three competing backends now exist: ik_llama (MTP), BeeLlama (DFlash), and mainline llama.cpp.
- **Unsloth Dynamic quants** (UD-Q4_K_M, UD-Q4_K_XL) are now recommended over plain Q4_K_M for coding *agents* — they preserve tool-call formatting and instruction-following noticeably better at the same bit-width.
- **TurboQuant 3-bit KV-cache compression** (ICLR 2026) lets Gemma 4 26B-A4B run its full 262K context on a 24 GB RTX 4090 (≈22.3 GB, ~129 tok/s).
- **Caveat:** CUDA 13.2 produces gibberish with low-bit Qwen 3.6 quants — use CUDA 13.1 or 13.3.

## Agent Frameworks for Open Models

No major framework is exclusively optimised for open models — all are model-agnostic. The best options:

GitHub star counts verified via the GitHub API on 16 Jun 2026.

| Framework | Stars | Open-Model Support | Best Open Model Pairing |
|-----------|-------|-------------------|------------------------|
| **OpenHands** | ~77.2K | Excellent (model-agnostic, sandboxed) | DeepSeek V4 Pro / MiniMax M3 |
| **Cline** | ~63.3K | Excellent (IDE/CLI, local models) | Qwen 3.6-27B (local) |
| **Aider** | ~46.3K | Excellent (any OpenAI-compatible API) | DeepSeek-V3.2-Exp (74.2%) |
| **Roo Code** | ~24.2K (**repo archived**) | Good (VS Code, multi-mode) | Devstral Small 2 |
| **SWE-agent** | ~19.5K | Good (research-focused; `mini-SWE-agent` ~74% in 100 LOC) | DeepSeek V3.2 |

The scaffold quality has become as important as model capability — the same model varies 15–25 points across harnesses on Terminal-Bench 2.0. `mini-SWE-agent` reaching ~74% SWE-bench Verified in ~100 lines of Python is the clearest demonstration: open-source scaffolds with the right model often outperform far more complex setups. Open-source scaffolds paired with closed models still top most leaderboards. (Note: Roo Code's GitHub repository is now archived; Cline, from which it forked, remains active.)

## Cost Analysis

Cost ratios are against Claude Opus 4.8 output ($25/M) as baseline. Prices are list output $/M tokens, June 2026.

| Solution | SWE-bench (Verified / Pro vendor) | Input $/M | Output $/M | Cost Ratio (output) |
|----------|-----------------------------------|-----------|------------|---------------------|
| Claude Fable 5 (Anthropic API) | 95.0% / 80.3% | $10.00 | $50.00 | 2× |
| Claude Opus 4.8 (Anthropic API) | 88.6% / 69.2% | $5.00 | $25.00 | 1× (baseline) |
| Claude Opus 4.7 (Anthropic API) | 87.6% / 64.3% | $5.00 | $25.00 | 1× |
| GPT-5.5 (OpenAI API) | 82.6%‡ / 58.6% | — | ~$30.00 | 1.2× |
| Gemini 3.1 Pro (Google) | 78.8% / 54.2% | — | ~$12.00 | 0.48× |
| **Kimi K2.6** (Moonshot) | 80.2% / 58.6% | — | $4.00 | **0.16×** |
| **GLM-5.1** (Z.ai) | ~78% / 58.4% | — | $3.20 | **0.128×** |
| **MiniMax M3** | 80.5% / 59.0% | $0.60 | $2.40 | **0.096×** |
| **Qwen 3.6-Flash (35B-A3B)** | 73.4% / 49.5% | — | $0.90 | **0.036×** |
| **DeepSeek V4 Pro** (official API) | 80.6% / 55.4% | $0.44 | $0.87 | **0.035×** |
| **DeepSeek V4 Flash** (official API) | 79.0% / 52.6% | $0.14 | $0.28 | **0.011×** |
| **Nemotron 3 Ultra** (OpenRouter) | 71.9% / — | free | free | **~0×** |
| **Qwen 3.6-27B** (local, RTX 4090) | 77.2%† / 53.5% | — | ~$0 (electricity) | **~0×** |
| **Devstral Small 2** (local) | 68.0% / — | — | ~$0 | **~0×** |

†Vendor-reported, not independently verified. ‡Vals.ai standardised (vendor max ~88.7%).

DeepSeek V4 Pro delivers ~91% of Claude Opus 4.8's vendor SWE-bench-Pro performance at **3.5% of the output cost**, and MiniMax M3 actually **edges out GPT-5.5 on vendor SWE-bench Pro (59.0% vs 58.6%) at under a tenth of the price**. DeepSeek V4 Flash ($0.28/M) is the cheapest frontier-class option on the market, and NVIDIA Nemotron 3 Ultra is free on OpenRouter. For organisations processing millions of tokens, the open-model cost advantage is now 10–60×, with the best open models within ~1–2 points of GPT-5.5/Gemini 3.1 Pro on most metrics — though still well behind Claude Fable 5/Opus 4.8 on the curated benchmarks.

## Limitations of Open Models

Despite rapid progress, open models still trail closed models in several areas:

1. **Multi-step planning depth**: Open models (especially those <100B active params) lose coherence beyond 3–4 step agentic chains. GLM 5.1 and DeepSeek V3.2 degrade noticeably on tasks requiring 10+ sequential reasoning steps.

2. **Tool-call reliability**: DeepSeek V4 improved "substantially" over V3 but frontier closed models still handle complex tool chains more reliably — fewer malformed calls, better error recovery.

3. **Effective context utilisation**: Most open models have shorter effective context windows than Claude (200K) or GPT-5.5 (400K). Qwen 3.6 Plus supports 1M tokens but long-context performance degrades in practice.

4. **Terminal-Bench gap**: The 50-point gap on Terminal-Bench 2.0 reveals that open models struggle with hard, real-world terminal tasks requiring system administration, debugging, and complex environment setup.

5. **Instruction following under failure**: When a task fails partway, closed models are better at adapting strategy. Open models tend to repeat failed approaches (higher step-repetition rate).

6. **Structured harness dependency**: All models perform better in structured agent frameworks, but open models show a larger gap between "raw chat" and "properly scaffolded" performance.

7. **Geographic concentration (now slightly less extreme)**: The leading open coding models remain overwhelmingly from Chinese labs (DeepSeek, Qwen/Alibaba, Kimi/Moonshot, GLM/Zhipu, MiniMax, MiMo/Xiaomi). The notable June 2026 change is **NVIDIA Nemotron 3 Ultra** (4 Jun) — the first genuinely competitive *Western* open coding model (71.9% Verified, 86.0 LiveCodeBench), joining Mistral's Devstral as a Western exception. Meta's Llama 4 Maverick still scores poorly on coding (15.6% Aider Polyglot), and Llama 4 Behemoth remains unreleased.

## Recommendations

### For Production Agentic Coding (Maximum Quality)
Use **closed models** — **Claude Fable 5** (9 Jun) is the new ceiling (95.0% Verified, 80.3% SWE-bench Pro), with Claude Opus 4.8 and GPT-5.5 close behind. Use these when correctness matters and cost is secondary. The Terminal-Bench and Scale SEAL SWE-bench Pro gaps remain real.

### For Cost-Sensitive Production
Use **DeepSeek V4 Pro** via API — 80.6% Verified / 55.4% SWE-bench Pro at ~1/29th the output cost of Claude Opus 4.8 — or **MiniMax M3** if you want the strongest open SWE-bench Pro vendor score (59.0%, edging GPT-5.5) plus 1M context and multimodal. **DeepSeek V4 Flash** ($0.28/M) is the cheapest frontier-class option; **Nemotron 3 Ultra** is free on OpenRouter. Pair with OpenHands or Aider.

### For Local Development (24 GB GPU)
**Qwen 3.6-27B** (Q6_K, ~22.5 GB) remains the leader if vendor benchmarks hold. **Qwen 3.6-35B-A3B** for speed-sensitive use (101 tok/s on RTX 3090). **Devstral Small 2** (24B, 68.0% Verified) as the best-verified agentic alternative. Use **Unsloth Dynamic quants** (UD-Q4_K_M) for better tool-call reliability, and enable DFlash/MTP speculative decoding for ~2× throughput.

### For Budget Hardware (8–16 GB GPU)
**Qwen 3.6-35B-A3B** at UD-Q3_K_M (16.6 GB) fits 16 GB cards at full speed. **GLM-4.7 (9B)** (~6 GB) punches far above its size on code generation (94.2% HumanEval). **DeepSeek R1 Distill 14B** (8 GB) for reasoning/debugging; **Gemma 4 26B-A4B** for maximum speed.

### For Agentic Coding Specifically
**Devstral Small 2** (24B, agentically tuned, runs locally) for single-GPU agent loops, or **DeepSeek V4 Pro / MiniMax M3** via API for cloud-scale agentic coding. (The previously-recommended "Qwen3-Coder-Next" could not be confirmed as a released model — see Areas of Uncertainty.)

## Areas of Uncertainty

- **Qwen 3.6-27B's SWE-bench score** (77.2%) and 3.6-35B-A3B's (73.4%) remain vendor-reported using Qwen's own scaffold. As of June 2026, multiple independent reviewers report the numbers "line up directionally," but **no full third-party SWE-bench reproduction outside Qwen's scaffold has been published**, and real-world users report tool-use drift in long agent loops. Treat as upper bounds.
- **"Qwen3-Coder-Next" (80B/3B active)**: This model — previously listed in this article at 70.6% SWE-bench — could **not be confirmed** as a released, consumer-deployable model by two of three independent searches in June 2026. It appears in some aggregator tables and an arXiv preprint (2603.00729) but not in Qwen's own current release lineup (Qwen3.6-27B, 35B-A3B, Qwen3-Coder-480B). Removed from the headline tables pending confirmation.
- **"DeepSeek V4 Pro Max" is not a separate model** — it refers to DeepSeek V4 Pro running in *Think Max* reasoning mode. The released models are V4 Pro (1.6T/49B) and V4 Flash (284B/13B).
- **SWE-bench Verified is deprecated** (OpenAI withdrew it Feb 2026 over ~59% flawed tests). Vendor-reported numbers stay in this article for continuity but should not be the basis for model selection. Prefer SWE-bench Pro (SEAL) and Terminal-Bench 2.0.
- **SWE-bench Pro vendor vs Scale SEAL gap (15–30 pp)**: Almost all open-model SWE-bench Pro figures here are vendor-reported. Only Qwen3-Coder-480B (38.7%) has a public Scale SEAL entry; DeepSeek V4, MiniMax M3, GLM-5.1, and Kimi K2.6 have no SEAL submissions, so their decontaminated standing is unverified.
- **MiniMax M3 / GLM-5.2 weight releases**: M3 (1 Jun) and GLM-5.2 (13 Jun) were announced open-weight with weights release pending shortly after launch — confirm availability before relying on self-hosting. GLM-5.2 had no independent benchmarks at the time of writing.
- **Terminal-Bench 2.0 representation**: Open models may score low partly because they lack investment in advanced agent scaffolds (most use basic Terminus 2), not solely due to model capability — vendor TB figures (MiniMax M3 66%, Kimi K2.6 66.7%, Qwen 3.7 Plus 70.3%) are far higher than official-harness open scores.
- **Long-context coding performance**: No standardised benchmark exists for 100K+ token repository-level coding tasks. Several June models claim 1M-token contexts (DeepSeek V4 Pro, MiniMax M3, GLM-5.2), but real-world behaviour at these scales is largely anecdotal.
- **Llama 4 Behemoth**: Still unreleased as of June 2026 — no verified coding benchmark results.

## References

1. [Aider Polyglot Coding Leaderboard](https://aider.chat/docs/leaderboards/) — Official Aider benchmark (fetched 2026-05-31)
2. [SWE-bench Official Leaderboard](https://www.swebench.com) — Princeton SWE-bench Verified (fetched 2026-05-31)
3. [Vals.ai SWE-bench Verified](https://www.vals.ai/benchmarks/swebench) — Standardised evaluation harness
4. [Vals.ai LiveCodeBench](https://www.vals.ai/benchmarks/lcb) — Standardised LiveCodeBench scores
5. [Terminal-Bench 2.0 Leaderboard](https://www.tbench.ai/leaderboard/terminal-bench/2.0) — Stanford/Laude Labs
6. [Terminal-Bench 2.0 Paper](https://arxiv.org/abs/2601.11868) — Merrill et al., Jan 2026
7. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — Official architecture details
8. [DeepSeek-V3 GitHub](https://github.com/deepseek-ai/DeepSeek-V3) — MIT licence, 671B/37B MoE
9. [Qwen3-Coder GitHub](https://github.com/QwenLM/Qwen3-Coder) — 480B/35B and Next (80B/3B) variants
10. [OpenHands LM 32B](https://www.openhands.dev/blog/introducing-openhands-lm-32b----a-strong-open-coding-agent-model) — RL-trained agentic model
11. [Devstral 2 Announcement](https://mistral.ai/news/devstral-2-vibe-cli/) — 123B dense, 72.2% SWE-bench
12. [Devstral Small 1.1](https://mistral.ai/news/devstral-2507) — Apache 2.0, 24B
13. [DeepSeek V4 Pro Pricing](https://www.explainx.ai/blog/deepseek-v4-pro-permanent-api-pricing-discount) — $0.435/M input
14. [Local LLM Inference Guide 2026](https://blog.starmorph.com/blog/local-llm-inference-tools-guide) — Quantisation format comparison
15. [Best Local Coding Models 2026](https://insiderllm.com/guides/best-local-coding-models-2026/) — VRAM tier rankings
16. [Qwen 3 Coder vs DeepSeek R1 Distill](https://willitrunai.com/blog/qwen-3-coder-vs-deepseek-coding) — MoE vs dense comparison
17. [AI Agent Benchmark Roundup May 2026](https://codersera.com/blog/ai-agent-benchmarks-state-of-leaderboard-may-2026/) — Cross-benchmark analysis
18. [Marc0.dev SWE-Bench Leaderboard](https://www.marc0.dev/en/leaderboard) — Multi-benchmark aggregator
19. [Beyond Synthetic Benchmarks](https://arxiv.org/abs/2510.26130) — Real-world vs synthetic coding performance gap
20. [AGENTIF: Benchmarking Instruction Following in Agentic Scenarios](https://arxiv.org/abs/2505.16944) — NeurIPS 2025
21. [Open-Source Coding Agents 2026](https://agentmarketcap.ai/blog/2026/04/10/open-source-coding-agents-2026-openhands-swe-agent-aider-vs-claude-code-codex) — Agent framework comparison
22. [MoE Architecture Explained](https://ninadpathak.com/blog/mixture-of-experts-explained/) — DeepSeek V3 MoE analysis
23. [ArkForge Benchmark Snapshot](https://ark-forge.github.io/genesis/benchmark.html) — 44-model code benchmark compilation (April 2026)
24. [Morph SWE-bench Pro Leaderboard](https://www.morphllm.com/swe-bench-pro) — Vendor + Scale SEAL SWE-bench Pro/Verified (fetched 2026-06-16)
25. [CodingFleet SWE-bench Pro Leaderboard 2026](https://codingfleet.com/blog/swe-bench-pro-leaderboard-2026/) — Cross-model SWE-bench Pro + pricing (June 2026)
26. [MiniMax M3 announcement](https://www.minimax.io/blog/minimax-m3) — Open-weight, MSA, 1M context (1 Jun 2026)
27. [MiniMax M3 guide](https://www.aimadetools.com/blog/minimax-m3-complete-guide/) — Specs, benchmarks, pricing
28. [NVIDIA Nemotron 3 Ultra model card](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b/modelcard) — 550B/55B MoE, OpenMDW-1.1 (4 Jun 2026)
29. [GLM-5.2 open-source release](https://awesomeagents.ai/news/zhipu-glm-5-2-open-source/) — 1M context, coding-first (13 Jun 2026)
30. [DeepSeek V4 Pro guide](https://www.aimadetools.com/blog/deepseek-v4-pro-complete-guide/) — Architecture, Think Max mode, pricing
31. [DeepSeek V4 (Morph)](https://www.morphllm.com/deepseek-v4) — V4 Pro / V4 Flash specs and pricing
32. [Kimi K2.6 guide](https://codersera.com/blog/kimi-k2-6-complete-guide-2026/) — 1T/32B MoE, Agent Swarm
33. [Qwen 3.6 local AI guide](https://insiderllm.com/guides/qwen-3-6-local-ai-guide/) — 27B/35B-A3B VRAM, speeds, benchmark caveats
34. [Best local coding models 2026](https://insiderllm.com/guides/best-local-coding-models-2026/) — VRAM tiers, DFlash/MTP, quant tooling (June 2026)
35. [Devstral Small 2 hardware guide](https://runaihome.com/blog/devstral-small-2-local-ai-hardware-guide-2026/) — VRAM, speeds, 68.0% SWE-bench
36. [Vals.ai LiveCodeBench](https://www.vals.ai/benchmarks/lcb) — Standardised LiveCodeBench, June 2026 snapshot
37. [Terminal-Bench 2.0 Leaderboard](https://www.tbench.ai/leaderboard/terminal-bench/2.0) — Stanford/Laude Labs (fetched 2026-06-16)
38. [OpenHands GitHub](https://github.com/OpenHands/OpenHands), [Cline GitHub](https://github.com/cline/cline), [Aider GitHub](https://github.com/Aider-AI/aider), [SWE-agent GitHub](https://github.com/SWE-agent/SWE-agent) — Framework star counts (fetched 2026-06-16)
