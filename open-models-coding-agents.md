# Open Models for Coding Agents: Benchmarks and Performance vs Frontier Closed Models

| Field | Value |
|-------|-------|
| Created | 2026-05-31 |
| Last Updated | 2026-05-31 |
| Version | 1.0 |

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

As of May 2026, the gap between open and closed models for coding agents has narrowed dramatically on some benchmarks while remaining substantial on others. The picture depends heavily on which benchmark you trust:

- **LiveCodeBench** (algorithmic coding): open models are within 1–3 points of closed (DeepSeek V4 Pro at 87.5% vs Gemini 3.1 Pro at 88.5%).
- **SWE-bench Verified** (repository-level bug fixes): open models reach 71–80% vs closed at 79–89%, but this benchmark has known contamination issues.
- **SWE-bench Pro** (contamination-resistant): the gap widens — Claude Opus 4.7 at 64.3% vs Qwen3.7 Max at 60.6%.
- **Aider Polyglot** (multi-language editing): a 14-point gap persists — GPT-5 at 88.0% vs DeepSeek-V3.2-Exp Reasoner at 74.2%.
- **Terminal-Bench 2.0** (hard real-world tasks): the gap is enormous — closed agents at 60–90% vs open at 25–40%.

The best large open models (DeepSeek V4 Pro Max, Kimi K2.6, GLM-5) now rival Claude Opus 4.5/4.6 on SWE-bench Verified while costing 20–35× less. For consumer hardware, Qwen 3.6-27B (16.8 GB at Q4_K_M) and Qwen 3.6-35B-A3B (MoE, 101 tok/s on RTX 3090) represent a breakthrough — claiming 73–77% SWE-bench scores from a single GPU.

However, on the hardest agentic benchmarks (Terminal-Bench 2.0, SWE-bench Pro) and in real-world multi-step coding chains, closed models maintain a significant lead. The agent scaffold matters as much as the model: the same model can vary by 20+ percentage points depending on the framework used.

## Benchmark Landscape in 2026

**HumanEval** is effectively saturated — all frontier models (open and closed) score 90–95%. It no longer differentiates. The benchmarks that matter in 2026 are:

| Benchmark | What It Tests | Contamination Risk | Gap Indicator |
|-----------|--------------|-------------------|---------------|
| **SWE-bench Verified** | Fix real GitHub issues (500 instances) | High (known contamination) | Narrow (8 pp) |
| **SWE-bench Pro** | Same tasks, contamination-resistant | Low | Medium (4–15 pp) |
| **LiveCodeBench** | Fresh competitive-programming problems | Low (rolling updates) | Minimal (1–3 pp) |
| **Aider Polyglot** | Multi-language code editing (225 exercises) | Low | Significant (14 pp) |
| **Terminal-Bench 2.0** | Complex real-world terminal tasks (89 tasks) | Very low | Massive (40–50 pp) |
| **BigCodeBench Hard** | Complex function-level coding | Low | Stale (not updated for 2026 models) |

SWE-bench Verified scores are the most widely reported but also the most inflated. Vendor-reported numbers are typically 5–7 points higher than standardised evaluations (e.g., Vals.ai). SWE-bench Pro and Terminal-Bench 2.0 are more reliable indicators of genuine agentic capability.

## Frontier Closed Models: The Ceiling

These represent the performance ceiling that open models are measured against.

| Model | SWE-bench Verified | SWE-bench Pro | LiveCodeBench | Aider Polyglot | Terminal-Bench 2.0 |
|-------|-------------------|---------------|---------------|----------------|-------------------|
| GPT-5.5 | 88.7% | — | 85.3% | 88.0% (high) | 82.2% (Codex CLI) |
| Claude Opus 4.7 | 87.6% | 64.3% | 87.8% | 72.0% (32k think) | 90.2% (vix agent) |
| Gemini 3.1 Pro | 80.6% | 54.2% | 88.5% | 83.1% (06-05) | 80.2% (TongAgents) |
| Claude Opus 4.6 | 80.8% | 51.9% | — | — | 79.8% (ForgeCode) |
| GPT-5.3 Codex | 85.0% | 56.8% | 87.3% | — | — |
| Claude Sonnet 4.6 | 79.6% | — | — | 61.3% (32k think) | — |

**Vals.ai standardised harness** (fair cross-model comparison using mini-SWE-agent): GPT-5.5: 82.6%, Claude Opus 4.7: 82.0%, Gemini 3.1 Pro: 78.8%. These are ~5–7 points below vendor-reported maxes.

## Large Open Models (Cloud/Multi-GPU)

These models require multi-GPU setups or cloud API access but approach frontier closed-model performance at a fraction of the cost.

### Tier 1: Frontier-Competitive (>75% SWE-bench Verified)

| Model | Params (Total/Active) | Architecture | SWE-bench Verified | LiveCodeBench | Aider Polyglot | Licence | API Cost (input/output per M tokens) |
|-------|----------------------|--------------|-------------------|---------------|----------------|---------|--------------------------------------|
| **DeepSeek V4 Pro Max** | 1.6T / 49B | MoE | 80.6% | 93.5 | — | MIT | $0.44 / $0.87 |
| **Kimi K2.6** (Moonshot) | >1T / — | — | 80.2% | 86.8 | — | Open-weight | — |
| **MiniMax M2.5** | — | — | 80.2% | — | — | Open-weight | Free tier available |
| **MiMo-V2-Pro** (Xiaomi) | 1T / — | — | 78.0% | — | — | Open-weight | — |
| **GLM-5** (Zhipu AI) | 744B / — | — | 77.8% | — | — | MIT | — |
| **Kimi K2.5** (Moonshot) | >1T / — | — | 76.8% | 85.0 | — | Open-weight | — |
| **Qwen3.6 Plus** (Alibaba) | — | — | ~78.8% | 86.0 | — | Apache 2.0 | — |

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

| Model | Type | VRAM (Q4_K_M) | SWE-bench | HumanEval | Speed (RTX 4090) | Best For |
|-------|------|--------------|-----------|-----------|-------------------|----------|
| **Qwen 3.6-27B** | Dense | 16.8 GB | 77.2%† | 92.1% | ~35 tok/s | Best all-round coding |
| **Qwen 3.6-35B-A3B** | MoE (3B active) | 22 GB (Q4) / 17 GB (Q3) | 73.4%† | — | ~85–101 tok/s | Speed + quality balance |
| **Qwen3-Coder-Next** | MoE (80B/3B active) | ~24 GB (Q4) | 70.6% | ~94% | ~18–22 tok/s | Agentic coding |
| **Devstral Small 2** (Mistral) | Dense 24B | 14 GB | 56.4%–68.0% | 90.1% | ~40 tok/s | Multi-file refactoring |
| **Qwen 3 Coder 30B-A3B** | MoE (3B active) | 17 GB | 60.4% (EntroPO) | ~90% | ~85 tok/s | Fast agentic loops |
| **DeepSeek R1 Distill 14B** | Dense 14B | 8 GB | ~18% | ~80% | ~60 tok/s | Reasoning/debugging (8GB cards) |
| **Codestral 25.12** (Mistral) | Dense 22B | 16 GB | ~42% | 89.7% | — | Inline completion/autocomplete |
| **Gemma 4 26B-A4B** (Google) | MoE (3.8B active) | 14 GB | ~38.6% | 84.9% | ~600 tok/s (vLLM) | Maximum speed |

†Qwen 3.6-27B's 77.2% SWE-bench claim is vendor-reported and not yet independently verified on the official SWE-bench leaderboard. Treat as an upper bound.

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

### SWE-bench Verified (Official Leaderboard, May 2026)

| Rank | Model | Score | Type |
|------|-------|-------|------|
| 1 | Sonar Foundation + Claude 4.5 Opus | 79.2% | Closed |
| 2 | TRAE + Doubao-Seed-Code | 78.8% | Closed |
| 3 | Live-SWE-agent + Gemini 3 Pro | 77.4% | Closed |
| 6 | Lingxi v1.5 × Kimi K2 | 71.2% | **Open** |
| 7 | GLM-5 | 69.7% | **Open** |
| 8 | OpenHands + Qwen3-Coder-480B | 69.6% | **Open** |
| 9 | Minimax 2.5 | 68.3% | **Open** |
| 13 | Qwen3-Coder-30B-A3B (EntroPO) | 60.4% | **Open** |
| 18 | Devstral Small (24B) | 56.4% | **Open** |

### Aider Polyglot Leaderboard (May 2026)

| Rank | Model | Score | Type |
|------|-------|-------|------|
| 1 | GPT-5 (high) | 88.0% | Closed |
| 2 | GPT-5 (medium) | 86.7% | Closed |
| 4 | Gemini 2.5 Pro (32k think) | 83.1% | Closed |
| 12 | **DeepSeek-V3.2-Exp Reasoner** | **74.2%** | **Open** |
| 16 | **DeepSeek R1 (0528)** | **71.4%** | **Open** |
| 18 | **DeepSeek-V3.2-Exp Chat** | **70.2%** | **Open** |
| 14 | Claude Opus 4 (32k thinking) | 72.0% | Closed |
| 25 | **Qwen3 235B** | **59.6%** | **Open** |
| 30 | **DeepSeek V3 (0324)** | **55.1%** | **Open** |
| 44 | **Qwen3 32B** | **40.0%** | **Open** |
| 61 | **Llama 4 Maverick** | **15.6%** | **Open** |

### LiveCodeBench (Vals.ai Standardised, May 2026)

| Model | Score | Type |
|-------|-------|------|
| Gemini 3.1 Pro Preview | 88.49% | Closed |
| GPT 5.2 Codex | 87.99% | Closed |
| Claude Opus 4.8 | 87.82% | Closed |
| **DeepSeek V4 Pro** | **87.48%** | **Open** |
| GPT 5.3 Codex | 87.31% | Closed |
| **Kimi K2.6** | **86.77%** | **Open** |
| **Qwen3.6 Plus** | **85.95%** | **Open** |

## The Open–Closed Gap: Benchmark by Benchmark

| Benchmark | Best Closed | Best Open | Gap | Trend |
|-----------|------------|-----------|-----|-------|
| HumanEval | ~95% | ~95% | **0 pp** | Saturated |
| LiveCodeBench | 88.5% | 87.5% | **1 pp** | Effectively closed |
| SWE-bench Verified (vendor) | 88.7% | 80.6% | **8 pp** | Narrowing fast |
| SWE-bench Verified (standardised) | 82.6% | — | — | Open models not yet tested on Vals.ai |
| SWE-bench Pro | 64.3% | 60.6% | **4 pp** | Surprisingly close at top |
| Aider Polyglot | 88.0% | 74.2% | **14 pp** | Persistent |
| Terminal-Bench 2.0 | 90.2% | 39.6% | **50 pp** | Massive gap persists |

**Key insight**: The gap depends on task difficulty. On pure code generation (LiveCodeBench), open models have essentially caught up. On multi-step agentic tasks requiring planning, tool use, and error recovery (Terminal-Bench 2.0), closed models maintain a commanding lead. SWE-bench Verified is somewhere in between but is likely inflated for all models due to contamination.

## Agentic Coding: Models Designed for Tool Use

Several open models are now specifically trained for agentic coding rather than just code completion:

### Purpose-Built Agentic Models

| Model | Approach | SWE-bench | Designed For |
|-------|----------|-----------|-------------|
| **Qwen3-Coder-Next** (80B/3B active) | Agentically trained on executable task synthesis + RL | 70.6% | Cline, Claude Code, Qwen Code |
| **Devstral** family (Mistral) | Fine-tuned from Mistral-Small for code agent tasks | 53.6–72.2% | Mistral Vibe CLI, general agents |
| **OpenHands LM 32B** | RL fine-tuned (SWE-Gym) on successful agent trajectories | 37.2% | OpenHands/SWE-agent |
| **Skywork-SWE-32B** | RL-trained specifically for SWE-bench | 38–47% | SWE-agent workflows |

Qwen3-Coder-Next is notable for achieving 70.6% on SWE-bench Verified with only 3B active parameters — it is explicitly designed for local agentic coding and supports integration with major agent platforms.

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

### Speculative Decoding

Block-diffusion (DFlash) provides a measured 2.56× speedup on Qwen 3.6-27B Q4_K_M on RTX 3090. llama.cpp added Multi-Token Prediction (MTP) support for Qwen 3.6 in May 2026, enabling native speculative decoding without a draft model.

## Agent Frameworks for Open Models

No major framework is exclusively optimised for open models — all are model-agnostic. The best options:

| Framework | Stars | Open-Model Support | Best Open Model Pairing |
|-----------|-------|-------------------|------------------------|
| **OpenHands** | 60K+ | Excellent (model-agnostic) | Qwen3-Coder-480B (69.6%) |
| **Aider** | 35K+ | Excellent (any OpenAI-compatible API) | DeepSeek-V3.2-Exp (74.2%) |
| **SWE-agent** | — | Good (research-focused) | DeepSeek V3.2 (59%) |
| **Live-SWE-agent** | — | Scaffold-first approach | Any (79.2% with Claude 4.5) |
| **Cline** | — | Explicitly supported by Qwen3-Coder | Qwen3-Coder-Next (70.6%) |

The scaffold quality has become as important as model capability. Live-SWE-agent achieves 79.2% on SWE-bench Verified with Claude 4.5 — outperforming many other closed-model + scaffold combinations. Open-source scaffolds with closed models often outperform closed scaffolds + open models.

## Cost Analysis

| Solution | SWE-bench Verified | Cost per 1M tokens (output) | Cost Ratio |
|----------|-------------------|----------------------------|------------|
| Claude Opus 4.7 (Anthropic API) | 87.6% | $75.00 | 1× (baseline) |
| GPT-5.5 (OpenAI API) | 88.7% | ~$15.00 | 0.2× |
| DeepSeek V4-Pro (official API) | 80.6% | $0.87 | **0.012×** |
| DeepSeek V3.1 (official API) | 66% | $1.10 | 0.015× |
| DeepSeek V3.1 (Together AI) | 66% | $1.70 | 0.023× |
| Qwen 3.6-27B (local, RTX 4090) | 77.2%† | ~$0 (electricity only) | **~0×** |
| Devstral Small 2 (local) | 56.4–68% | ~$0 | **~0×** |

†Vendor-reported, not independently verified.

DeepSeek V4 Pro delivers ~91% of Claude Opus 4.7's SWE-bench performance at ~1.2% of the cost. For organisations processing millions of tokens, this represents transformational savings.

## Limitations of Open Models

Despite rapid progress, open models still trail closed models in several areas:

1. **Multi-step planning depth**: Open models (especially those <100B active params) lose coherence beyond 3–4 step agentic chains. GLM 5.1 and DeepSeek V3.2 degrade noticeably on tasks requiring 10+ sequential reasoning steps.

2. **Tool-call reliability**: DeepSeek V4 improved "substantially" over V3 but frontier closed models still handle complex tool chains more reliably — fewer malformed calls, better error recovery.

3. **Effective context utilisation**: Most open models have shorter effective context windows than Claude (200K) or GPT-5.5 (400K). Qwen 3.6 Plus supports 1M tokens but long-context performance degrades in practice.

4. **Terminal-Bench gap**: The 50-point gap on Terminal-Bench 2.0 reveals that open models struggle with hard, real-world terminal tasks requiring system administration, debugging, and complex environment setup.

5. **Instruction following under failure**: When a task fails partway, closed models are better at adapting strategy. Open models tend to repeat failed approaches (higher step-repetition rate).

6. **Structured harness dependency**: All models perform better in structured agent frameworks, but open models show a larger gap between "raw chat" and "properly scaffolded" performance.

7. **Geographic concentration**: The leading open coding models are almost exclusively from Chinese labs (DeepSeek, Qwen/Alibaba, Kimi/Moonshot, GLM/Zhipu, MiniMax). Meta's Llama 4 Maverick scores poorly on coding benchmarks (15.6% Aider Polyglot). Mistral's Devstral is the main Western exception.

## Recommendations

### For Production Agentic Coding (Maximum Quality)
Use **closed models** (Claude Opus 4.7, GPT-5.5) when correctness matters and cost is secondary. The Terminal-Bench and SWE-bench Pro gaps are real.

### For Cost-Sensitive Production
Use **DeepSeek V4 Pro** via API — 80.6% SWE-bench at 1/86th the cost of Claude Opus. Pair with OpenHands or Aider for best results.

### For Local Development (24 GB GPU)
**Qwen 3.6-27B** (Q4_K_M, 16.8 GB) is the clear leader if vendor benchmarks hold. **Qwen 3.6-35B-A3B** for speed-sensitive use (101 tok/s). **Devstral Small 2** (24B) as a well-verified alternative.

### For Budget Hardware (8–16 GB GPU)
**DeepSeek R1 Distill 14B** (8 GB Q4_K_M) for reasoning/debugging. **Gemma 4 26B-A4B** (14 GB) for maximum speed on light tasks.

### For Agentic Coding Specifically
**Qwen3-Coder-Next** (80B/3B active) — purpose-built for coding agents with explicit Cline/Claude Code support and 70.6% SWE-bench.

## Areas of Uncertainty

- **Qwen 3.6-27B's SWE-bench score** (77.2%) is vendor-reported and not independently verified on the official leaderboard. The official SWE-bench site only shows the 480B variant at 69.6%.
- **DeepSeek V4 Pro Max's 80.6% SWE-bench** is from aggregator sites; not yet on the official SWE-bench leaderboard at time of writing.
- **SWE-bench Verified contamination**: Models trained on GitHub data (which includes SWE-bench issues) may score artificially high. SWE-bench Pro addresses this but has fewer model submissions.
- **Terminal-Bench 2.0 representation**: Open models may score low partly because they lack investment in advanced agent scaffolds (most use basic Terminus 2), not solely due to model capability.
- **Long-context coding performance**: No standardised benchmark exists for 100K+ token repository-level coding tasks. Real-world behaviour at these scales is largely anecdotal.
- **StarCoder 3**: No technical paper found; performance claims (85.4% HumanEval, 15B params) come from a single source.
- **Llama 4 Behemoth**: Status unclear — no verified coding benchmark results found despite being announced.

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
