# Current Best Frontier LLMs

| Field | Value |
|-------|-------|
| Created | 2026-05-21 |
| Last Updated | 2026-08-03 |
| Version | 1.1 |

---

- [Anthropic](#anthropic)
- [OpenAI](#openai)
- [Google](#google)
- [References](#references)

## Anthropic

- **Claude Opus 4.7**: most capable model — 1M token context, extended thinking, strongest agentic stamina
- **Claude Sonnet 4.6**: best general-purpose model — ~95% of Opus quality at 1/5 the cost, default in Claude.ai and Claude Code
- **Claude Haiku 4.5**: best fast/cheap model — classification, extraction, real-time UX at scale

## OpenAI

OpenAI released the three-tier GPT-5.6 family on 9 July 2026:

- **GPT-5.6 Sol**: most capable tier — use for difficult reasoning, high-value coding, and long-horizon work; US$5 input, US$0.50 cached input, and US$30 output per million tokens
- **GPT-5.6 Terra**: balanced tier — use when Luna does not meet the task-quality threshold; US$2.50 input, US$0.25 cached input, and US$15 output per million tokens
- **GPT-5.6 Luna**: fastest and lowest-cost tier — use for high-volume agents, routine coding assistance, extraction, browsing, and interactive workloads; US$1 input, US$0.10 cached input, and US$6 output per million tokens

These recommendations use OpenAI's tier positioning, published prices, and separate workload observations. They do not average unrelated benchmark scores into one universal ranking. BenchmarkList records 97 Luna results and links each observation to its source. Representative direct-model results include 50.3% on Agents' Last Exam, 69.6% on AutomationBench, 74.6 on the Artificial Analysis Coding Agent Index, and 92.5% F1 on Graphwalks BFS at 256K context. These figures are OpenAI-reported launch results and must be validated against the target workload. For self-hosting or provider independence, use an open-weight alternative such as GLM-5.2.

## Google

- **Gemini 3.1 Pro**: most capable general model — 2M token context, strong on multimodal and document-scale work
- **Gemini 3.5 Flash**: best fast model — launched May 19, 2026 at I/O; outperforms 3.1 Pro on coding and agentic benchmarks, 4x faster output than comparable frontier models
- **Gemini 3.1 Flash-Lite**: cheapest/lightest model — ~1/8 the price of 3.1 Pro

*Note: Gemini 3.5 Pro was announced at I/O 2026 and is expected June 2026.*

## References

- https://kindatechnical.com/claude-ai/the-claude-model-family-in-2026-opus-sonnet-and-haiku-explained.html
- https://freeacademy.ai/blog/which-chatgpt-model-should-you-use-2026
- https://www.aipricing.guru/openai-pricing/
- https://teamai.com/blog/large-language-models-llms/gemini-models-explained-the-complete-2026-guide/
- https://www.buildfastwithai.com/blogs/google-io-2026-gemini-3-5-flash-announcements
- https://openai.com/index/introducing-o3-and-o4-mini/
- https://openai.com/index/gpt-5-6/
- https://deploymentsafety.openai.com/gpt-5-6-preview
- https://benchmarklist.com/models/openai-gpt-5.6-luna/
- https://benchmarklist.com/api/v1/manifest.json
- https://benchmarklist.com/agents/
