# SOTA Reference

Research articles on state-of-the-art topics in AI and software engineering.

## Articles

- [Frontier AI Models Benchmark](frontier-models-benchmark.md) — Rankings across overall performance, agentic coding, tool use, vision, audio, voice, open source, small models, and throughput
- [RAG & Context Engineering](rag-and-context-engineering.md) — Retrieval-augmented generation patterns, chunking strategies, and managed services
- [Embedding Models](embedding-models.md) — Best open-source local embedding models and how they compare with proprietary alternatives
- [Research Agent Frameworks](frameworks-research-agents.md) — Frameworks for building autonomous research agents
- [Chatbot Evaluation: LLM-as-a-Judge](chatbot-evaluation-llm-as-judge.md) — Modern methods and best practices for evaluating chatbots using LLMs as judges
- [Benchmarks for RAG Chatbots](rag-chatbot-benchmarks.md) — Benchmarks for testing RAG-powered chatbots
- [Large Document LLM Methods](large-document-llm-methods.md) — Methods for processing large documents with LLM-based chatbots
- [Preventing Topic Hijacking](chatbot-topic-hijacking-prevention.md) — Preventing topic hijacking and prompt injection in domain-specific chatbots
- [Embedding Pre-Screening for Topic Relevance](embedding-pre-screening-chatbot-topic-relevance.md) — Embedding-based pre-screening for chatbot topic relevance
- [Agentic Coding: Claude Code vs OpenAI Codex](agentic-coding-claude-vs-openai.md) — Best-in-class models, benchmark comparison, architecture differences, and consistency analysis for Claude Code and OpenAI Codex
- [Current Best Frontier LLMs](current-best-frontier-llms.md) — Quick-reference list of the best models from Anthropic, OpenAI, and Google as of May 2026
- [Local Multimodal Vision-Language Models](local-multimodal-vision-language-models.md) — Open-source VLMs for image identification, interpretation, and detailed description running on local hardware
- [Prompting Best Practices](dev-best-practices/prompting.md) — Prompting techniques, prompt storage patterns, CI/CD testing, and multi-cloud management for professional services
- [Azure AI Development Best Practices](dev-best-practices/azure-ai-development.md) — Platform architecture, RAG, agents, security, cost management, and evaluation for building AI systems on Azure in 2026

## Reference designs

- [RAG Knowledge Base for Mixed Document Sizes](reference-designs/rag-knowledge-base-mixed-document-sizes.md) — Production RAG pipeline for collections spanning 1- to 600-page documents (hybrid retrieval + RRF + reranking on pgvector)

## Model Elo Timeline

LMArena (Chatbot Arena) Elo ratings for frontier AI models over time, coloured by lab with family lines connecting models of the same class.

### Last 6 Months

![Model Elo Timeline - Last 6 Months](images/model-elo-timeline-6m.png)

### Last 2 Years

![Model Elo Timeline - Last 2 Years](images/model-elo-timeline.png)

## Tool Use Benchmarks

Scores across BFCL V4 (structured function calling) and Tau²-bench domains (airline, retail, telecom agent tool use).

![Tool Use Benchmarks](images/tool-use-benchmarks.png)

### Local Models (≤ 30B params)

Tool use performance for models that can run locally, with frontier model reference lines. Covers BFCL V4, Docker's practical tool calling eval, and Tau²-bench Retail.

![Tool Use - Local Models](images/tool-use-local-models.png)
