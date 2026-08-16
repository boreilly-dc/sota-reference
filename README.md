# SOTA Reference

Research articles on state-of-the-art topics in AI and software engineering.

## Benchmark and leaderboard sites

Use these live source sites for current benchmark results and leaderboard snapshots. Scores from different benchmarks, harnesses, or versions are not directly comparable and must not be combined into a universal ranking.

- **General model leaderboards:** [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/leaderboards/models), [Vellum LLM Leaderboard](https://www.vellum.ai/llm-leaderboard), [Vellum Open LLM Leaderboard](https://www.vellum.ai/open-llm-leaderboard), and [Arena AI](https://arena.ai/leaderboard)
- **Provenance and benchmark index:** [BenchmarkList](https://benchmarklist.com/)
- **Coding and agent benchmarks:** [Aider Polyglot](https://aider.chat/docs/leaderboards/), [Vals.ai SWE-bench](https://www.vals.ai/benchmarks/swebench), [Vals.ai LiveCodeBench](https://www.vals.ai/benchmarks/lcb), and [Terminal-Bench](https://www.tbench.ai/leaderboard/terminal-bench/2.0)
- **Tool use and function calling:** [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html), [Toolathlon](https://toolathlon.xyz/docs/leaderboard), and [τ-bench](https://github.com/sierra-research/tau2-bench)
- **Tool use, retrieval, and document understanding:** [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html), [MTEB](https://leaderboard.mteb.org/), and [OCRBench](https://huggingface.co/spaces/echo840/ocrbench-leaderboard)
- **Media and voice:** [Artificial Analysis Speech Arena](https://artificialanalysis.ai/text-to-speech/leaderboard), [Artificial Analysis Text-to-Video](https://artificialanalysis.ai/video/leaderboard/text-to-video), [VBench](https://benchmarklist.com/benchmarks/vbench_video_generation/), and [VoiceBenchmark](https://voicebenchmark.ai/)

## Benchmark plots

Tracked plot snapshots are generated with `python3 .scripts/plot_benchmarks.py --all`. The plots keep cost, tool use, and coding benchmarks separate; they do not form a universal model ranking.

- [Cost versus intelligence](images/cost-per-intelligence.png) — Artificial Analysis Intelligence Index against a defined 1M-input + 1M-output workload cost
- [Tool-use benchmarks](images/tool-use-benchmarks.png) — BFCL V4, τ-bench domains, and Toolathlon-Verified panels
- [Agentic coding benchmarks](images/agentic-coding-benchmarks.png) — SWE-bench Pro, Terminal-Bench 2.1, and LiveCodeBench source observations

## Benchmark provenance

Benchmark-heavy articles use [BenchmarkList](https://benchmarklist.com/) as a discovery and provenance index where it has suitable verified coverage. The repository keeps the underlying paper, model card, benchmark, or independent leaderboard as a source because BenchmarkList usually indexes published observations rather than independently rerunning them. Imported records must have a verified review state, dated snapshot, source URL, metric direction, and explicit subject type. Scores from unrelated benchmarks, harnesses, or versions are not combined into a universal ranking.

Selections are declared in [`.data/benchmarklist_manifest.json`](.data/benchmarklist_manifest.json), generated per-article records live in [`.data/benchmarklist/`](.data/benchmarklist/), and [`.data/benchmarklist_coverage.json`](.data/benchmarklist_coverage.json) records migrated, blocked, primary-source-only, not-applicable, and later-wave articles. Pull requests validate these files offline; live API refreshes remain explicit, reviewable maintenance actions.

Articles are organised into topic folders. Survey-style articles compare options across a field; canonical blueprints live under [`reference-designs/`](reference-designs/).

## Models

- [Frontier AI Models Benchmark](models/frontier-models-benchmark.md) — Rankings across overall performance, agentic coding, tool use, vision, audio, voice, open source, small models, and throughput
- [Current Best Frontier LLMs](models/current-best-frontier-llms.md) — Quick-reference list of the best models from Anthropic, OpenAI, and Google as of May 2026
- [Best Local LLMs for Consumer Hardware](models/local-llms-consumer-hardware.md) — Open-source models (Gemma 4, Qwen 3.5/3.6, Llama 4, Phi-4, Mistral) for local inference on GPUs and Apple Silicon with benchmarks, VRAM tiers, quantisation, and framework comparison
- [Gemma 4 12B Overview](models/gemma-4-12b.md) — Google's encoder-free multimodal 12B (native audio, 256K context, 16 GB laptop-ready): architecture, full Gemma 4 family benchmark comparison, and how it sits between E4B and the 26B MoE
- [Molmo2 Capabilities Overview](models/molmo2.md) — Ai2's fully-open vision-language family (4B/8B/7B-O): image captioning, OCR/scene-text and document text extraction, video understanding and dense captioning, and best-in-class spatio-temporal grounding (pointing/tracking)
- [TurboQuant on Mac with Qwen3.5](models/turboquant-mac-qwen3-5.md) — Google Research's two-stage KV-cache quantiser (ICLR 2026): 3-bit/6× memory reduction, and the MLX / llama.cpp ports targeting Qwen3.5 on Apple Silicon
- [GLM-5.2 Overview](models/glm-5-2.md) — Z.ai's MIT-licensed open-weight coding flagship (744B-A40B MoE, 1M context): architecture, the GLM-5 lineage, vendor vs independent benchmarks (now the top open-weight model on the Artificial Analysis Intelligence Index), live pricing, US Entity List status, and self-hosted/hyperscaler deployment

## RAG & retrieval

- [RAG & Context Engineering](rag/rag-and-context-engineering.md) — Retrieval-augmented generation patterns, chunking strategies, and managed services
- [Benchmarks for RAG Chatbots](rag/rag-chatbot-benchmarks.md) — Benchmarks for testing RAG-powered chatbots
- [Large Document LLM Methods](rag/large-document-llm-methods.md) — Methods for processing large documents with LLM-based chatbots
- [Embedding Models](rag/embedding-models.md) — Best open-source local embedding models and how they compare with proprietary alternatives
- [Embedding Pre-Screening for Topic Relevance](rag/embedding-pre-screening-chatbot-topic-relevance.md) — Embedding-based pre-screening for chatbot topic relevance
- [turbovec — TurboQuant Vector Quantisation](rag/turbovec-vector-quantization.md) — Rust ANN index (+Python bindings) implementing Google's data-oblivious TurboQuant quantiser: 2/4-bit compression, online ingest, in-kernel filtered search, SIMD (NEON/AVX-512) scoring, and recall/speed claims vs FAISS

## Agents & tool use

- [Agentic Coding: Claude Code vs OpenAI Codex](agents/agentic-coding-claude-vs-openai.md) — Best-in-class models, benchmark comparison, architecture differences, and consistency analysis for Claude Code and OpenAI Codex
- [Open Models for Coding Agents](agents/open-models-coding-agents.md) — Best open models for coding agents vs frontier closed models across SWE-bench, Aider, LiveCodeBench, and Terminal-Bench, with consumer-hardware and large-model tiers
- [Research Agent Frameworks](agents/frameworks-research-agents.md) — Frameworks for building autonomous research agents
- [Grounded Search for LLMs](agents/grounded-search-for-llms.md) — Build-your-own grounded web search: reference architecture, retrieval/attribution/verification methods, RL-trained agentic search, open-source frameworks, and hyperscaler grounding
- [Agentic Harnesses](agents/agentic-harnesses.md) — Survey of frameworks for building agentic systems across Python, JS/TS, Go, and C#/.NET: agent application frameworks, vendor agent SDKs, coding-agent harnesses, durable-execution engines, hyperscaler managed runtimes, and interoperability protocols (MCP, A2A, AGNTCY), with a decision guide and anti-patterns
- [Alternatives to MCP: Tool-Calling Strategies](agents/mcp-alternatives-and-tool-calling-strategies.md) — MCP's rise as the dominant tool-integration standard and the alternative tool-calling strategies for connecting LLMs to external tools and data

## Media generation

- [Audio Generation AI](media-generation/audio-generation-ai.md) — Voice, music, and sound/Foley synthesis (proprietary + open) in 2026: TTS leaders (ElevenLabs, Cartesia, Kokoro, Chatterbox, F5-TTS), music (Suno/Udio, ACE-Step, YuE), SFX/video-to-audio (ElevenLabs SFX, Veo V2A, TangoFlux, MMAudio), the codec/tokeniser/embedding foundation (EnCodec, SNAC, Mimi, CLAP, T5), hardware tiers, licensing traps, and hyperscaler services
- [Video Generation AI](media-generation/video-generation-ai.md) — Text-to-video, image-to-video, and world models (proprietary + open) in 2026: frontier models (Sora 2, Veo 3.1, Kling 3.0, Seedance 2.0, Runway, Hailuo, Luma Ray3, Firefly), open leaders (Wan 2.2, HunyuanVideo 1.5, LTX-Video, Mochi, CogVideoX, Open-Sora), the VAE/DiT/tokeniser foundation and autoregressive LLM connection, video embeddings, evaluation (VBench, FVD/JEDi), world models (Genie 3, Cosmos, Oasis, Marble), hardware/VRAM tiers, hyperscaler services, and safety/provenance/regulation
- [Local Image Generation Models](media-generation/local-image-generation-models.md) — Open-source image-generation models, frameworks (e.g. InvokeAI), hardware requirements, and platform guidance for running diffusion locally
- [Creating Deepfakes: Models, Hardware & Accessibility](media-generation/deepfake-creation-models-and-accessibility.md) — The offence-and-accessibility counterpart to the detection article: how face-swap/talking-head/voice-clone/real-time pipelines work, the open-source toolkit (FaceFusion, Deep-Live-Cam, F5-TTS, etc.) and no-skill SaaS path, hardware/cost tiers, a calibrated threat model, real-world incidents, and the process controls that defeat executive impersonation

## Multimodal understanding

- [Local Multimodal Vision-Language Models](multimodal/local-multimodal-vision-language-models.md) — Open-source VLMs for image identification, interpretation, and detailed description running on local hardware
- [OCR in 2026](multimodal/ocr-models.md) — Multimodal LLMs vs specialist and traditional OCR: benchmarks, when non-LLM solutions win, reliability, hybrid pipelines, cost, and hyperscaler services
- [Local Audio Language Models](multimodal/local-audio-language-models.md) — Open-source audio LLMs (Qwen-Omni, Audio Flamingo, Kimi-Audio, Phi-4-mm, MiniCPM-o) for audio understanding and reasoning: model landscape, architecture, benchmarks (MMAU/MMAR), VRAM/inference, quantisation, licensing, and hyperscaler services
- [Building Real-Time Tool-Using Voice Agents](multimodal/real-time-voice-llms.md) — Cascaded, native, and hybrid architectures; open and managed stacks; revision-aware tool protocols; deployment channels; safety controls; and evaluation
- [Facial Recognition: Deepfake & Impersonation Detection](multimodal/facial-recognition-deepfake-impersonation-detection.md) — Modern face recognition plus the trust layers that defend it: PAD/liveness, injection detection, deepfake detection (foundation-model SOTA & the generalisation problem), morphing attack detection (NIST FATE MORPH), provenance (C2PA), demographic bias, hyperscaler liveness services, and standards/regulation (ISO 30107, FIDO, EU AI Act, BIPA)

## Evaluation & observability

- [Chatbot Evaluation: LLM-as-a-Judge](evaluation/chatbot-evaluation-llm-as-judge.md) — Modern methods and best practices for evaluating chatbots using LLMs as judges
- [Preventing Topic Hijacking](evaluation/chatbot-topic-hijacking-prevention.md) — Preventing topic hijacking and prompt injection in domain-specific chatbots
- [Modern AI Observability](evaluation/ai-observability.md) — LLM observability concepts, OpenTelemetry GenAI conventions, agentic/swarm tracing, open-source tools, and hyperscaler services

## Developer best practices

- [Prompting Best Practices](dev-best-practices/prompting.md) — Prompting techniques, prompt storage patterns, CI/CD testing, and multi-cloud management for professional services
- [Azure AI Development Best Practices](dev-best-practices/azure-ai-development.md) — Platform architecture, RAG, agents, security, cost management, and evaluation for building AI systems on Azure in 2026
- [AWS AI Development Best Practices](dev-best-practices/aws-ai-development.md) — Bedrock platform, RAG (Knowledge Bases + S3 Vectors), AgentCore, Guardrails, cost management, and evaluation for building AI systems on AWS in 2026
- [GCP AI Development Best Practices](dev-best-practices/gcp-ai-development.md) — Gemini Enterprise Agent Platform, ADK, Agent Runtime, RAG Engine / search / AlloyDB AI, security, cost management, and evaluation for building AI systems on Google Cloud in 2026
- [DAG Workflows](dev-best-practices/dag-workflows.md) — DAG orchestration tools (Airflow, Prefect, Dagster, Temporal, Flyte, Argo, Kestra), design patterns, anti-patterns, testing, lineage, and managed services

## Reference designs

- [RAG Knowledge Base for Mixed Document Sizes](reference-designs/rag-knowledge-base-mixed-document-sizes.md) — Production RAG pipeline for collections spanning 1- to 600-page documents (hybrid retrieval + RRF + reranking on pgvector)
- [Edge-First AI Clinical Documentation](reference-designs/edge-first-clinical-documentation.md) — Edge-compute AI processing for clinical documentation in disconnected environments with clinician-in-the-loop approval and EHR integration
- [Copilot MCP Integration (Azure)](reference-designs/copilot-mcp-integration-azure.md) — Microsoft 365 Copilot enterprise backend integration via Azure AI Foundry, MCP servers, and API Management with OBO/ACL/gateway patterns
- [Conversational AI with Tiered Semantic Routing](reference-designs/conversational-ai-semantic-routing.md) — Tiered routing architecture that bypasses the full agentic planner for high-confidence single-intent requests using embedding similarity and slot guards
- [RAG Compliance Assistant (Small-Scale Azure)](reference-designs/rag-compliance-assistant-azure.md) — Serverless RAG architecture for 100-500 document compliance/advisory Q&A with cited copy-pasteable answers and DeepEval CI/CD quality gates
- [AI-Assisted OIA/FOI Processing Pipeline](reference-designs/oia-foi-processing-pipeline.md) — Eight-stage AI-assisted pipeline for Official Information Act and Freedom of Information request processing with mandatory human-in-the-loop at every decision point
- [Automated Multi-Step AI Research Pipeline](reference-designs/ai-investment-research-pipeline.md) — Workflow engine for chaining LLM calls across structured analytical processes with state management, context budgeting, and human checkpoints
- [AI-Assisted Change Impact Assessment for Mega-Projects](reference-designs/ai-change-impact-assessment-megaproject.md) — Agentic AI for assessing change request impacts against regulatory and compliance document baselines on $1B+ infrastructure projects
