# SOTA Reference

Research articles on state-of-the-art topics in AI and software engineering.

Articles are organised into topic folders. Survey-style articles compare options across a field; canonical blueprints live under [`reference-designs/`](reference-designs/).

## Models

- [Frontier AI Models Benchmark](models/frontier-models-benchmark.md) — Rankings across overall performance, agentic coding, tool use, vision, audio, voice, open source, small models, and throughput
- [Current Best Frontier LLMs](models/current-best-frontier-llms.md) — Quick-reference list of the best models from Anthropic, OpenAI, and Google as of May 2026
- [Best Local LLMs for Consumer Hardware](models/local-llms-consumer-hardware.md) — Open-source models (Gemma 4, Qwen 3.5/3.6, Llama 4, Phi-4, Mistral) for local inference on GPUs and Apple Silicon with benchmarks, VRAM tiers, quantisation, and framework comparison
- [Gemma 4 12B Overview](models/gemma-4-12b.md) — Google's encoder-free multimodal 12B (native audio, 256K context, 16 GB laptop-ready): architecture, full Gemma 4 family benchmark comparison, and how it sits between E4B and the 26B MoE
- [Molmo2 Capabilities Overview](models/molmo2.md) — Ai2's fully-open vision-language family (4B/8B/7B-O): image captioning, OCR/scene-text and document text extraction, video understanding and dense captioning, and best-in-class spatio-temporal grounding (pointing/tracking)
- [TurboQuant on Mac with Qwen3.5](models/turboquant-mac-qwen3-5.md) — Google Research's two-stage KV-cache quantiser (ICLR 2026): 3-bit/6× memory reduction, and the MLX / llama.cpp ports targeting Qwen3.5 on Apple Silicon

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
- [Real-Time Voice LLMs](multimodal/real-time-voice-llms.md) — Voice-to-voice models for assistants: architectures, local vs cloud deployment, latency, expressiveness, tool use, and open-source options
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

## Model Elo Timeline

LMArena (Chatbot Arena) Elo ratings for frontier AI models over time, coloured by lab with family lines connecting models of the same class.

### Last 6 Months

![Model Elo Timeline - Last 6 Months](images/model-elo-timeline-6m.png)

### Last 2 Years

![Model Elo Timeline - Last 2 Years](images/model-elo-timeline.png)

## Open-Source Model Elo Timeline

LMArena Elo ratings for open-weight models over time, showing the progression of each model family.

![Open-Source Model Elo Timeline](images/local-model-elo-timeline.png)

## Tool Use Benchmarks

Scores across BFCL V4 (structured function calling) and Tau²-bench domains (airline, retail, telecom agent tool use).

![Tool Use Benchmarks](images/tool-use-benchmarks.png)

### Local Models (≤ 30B params)

Tool use performance for models that can run locally, with frontier model reference lines. Covers BFCL V4, Docker's practical tool calling eval, and Tau²-bench Retail.

![Tool Use - Local Models](images/tool-use-local-models.png)

## Gemma 4 Family Benchmarks

Official Google model-card scores across the Gemma 4 family, including the new **12B** (released ~3 June 2026). The 12B nears the 26B A4B MoE on reasoning, coding and agentic (Tau²) tasks at under half the memory. See the [Gemma 4 12B overview](models/gemma-4-12b.md).

![Gemma 4 Family Benchmarks](images/gemma4-family-benchmarks.png)
