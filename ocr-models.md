# OCR in 2026: Multimodal LLMs, Specialised Models, and When Non-LLM Solutions Still Win

| Field | Value |
|-------|-------|
| Created | 2026-06-02 |
| Last Updated | 2026-06-02 |
| Version | 1.0 |

---

- [Executive Summary](#executive-summary)
- [How to Choose: Quick Decision Guide](#how-to-choose-quick-decision-guide)
- [The 2026 Landscape: Four Tiers](#the-2026-landscape-four-tiers)
- [Benchmarks and How to Read Them](#benchmarks-and-how-to-read-them)
- [Best Multimodal LLMs for OCR](#best-multimodal-llms-for-ocr)
  - [Frontier Proprietary VLMs and Dedicated OCR APIs](#frontier-proprietary-vlms-and-dedicated-ocr-apis)
  - [Open-Source Specialist Document VLMs](#open-source-specialist-document-vlms)
  - [Open-Source General-Purpose VLMs](#open-source-general-purpose-vlms)
- [When a Non-LLM Solution Is the Better Choice](#when-a-non-llm-solution-is-the-better-choice)
- [Reliability and Safety of VLM-Based OCR](#reliability-and-safety-of-vlm-based-ocr)
- [Hybrid Pipelines and Tiered Routing](#hybrid-pipelines-and-tiered-routing)
- [Cost Economics](#cost-economics)
- [Hyperscaler Managed Services](#hyperscaler-managed-services)
- [Use-Case Recommendations](#use-case-recommendations)
- [What Changed Since January 2026](#what-changed-since-january-2026)
- [Caveats and Areas of Uncertainty](#caveats-and-areas-of-uncertainty)
- [References](#references)

---

## Executive Summary

By mid-2026, "OCR" no longer means a single tool. It spans four overlapping tiers — traditional engines, specialist document vision-language models (VLMs), open general VLMs, and frontier proprietary VLMs — and the right choice depends almost entirely on the document, the volume, and the tolerance for error.

The headline finding is a genuine split between two ways of measuring quality:

- **On automated document-parsing benchmarks (OmniDocBench), small specialist models win.** Sub-1.5B-parameter open models — **MinerU2.5-Pro (1.2B, 95.69 on OmniDocBench v1.6)**, **GLM-OCR (0.9B, 95.22)** and **PaddleOCR-VL-1.5/1.6 (0.9B, 94.93–96.3)** — now beat the best frontier general models (Gemini 3 Pro ≈ 90.3, Qwen3-VL-235B ≈ 89.2, GPT-5.2 ≈ 85.4) by roughly **5–10 points**, at a fraction of the cost. [1][7][8][9][12]
- **On human-preference evaluation (OCR Arena) and complex reasoning, frontier VLMs win.** Gemini 3 Flash/Pro, Claude Opus 4.6+, and GPT-5.x lead when documents demand layout judgement, instruction-following, chart reasoning, or structured JSON extraction from messy real-world inputs. [2][6]

For the user's core question — *is there ever a better non-LLM solution?* — the answer is **yes, frequently**, and it is one of the most important practical lessons of 2026:

- **Cost and throughput**: self-hosted specialist/traditional OCR runs at roughly **$0.09–0.70 per 1,000 pages** versus **$1.50–70 per 1,000 pages** for cloud APIs and frontier-VLM calls — a **10–167× gap**. [9][12][13]
- **Latency and determinism**: traditional engines process a page in **0.5–3 s on CPU**; frontier VLMs take **5–15 s via API** and are non-deterministic. [13]
- **Localisation and reliability**: a peer-reviewed CVPR 2026 result shows a **5M-parameter PP-OCRv5** rivalling billion-parameter VLMs on standard OCR while offering **superior text localisation and far fewer hallucinations**. [10]
- **Low-resource scripts**: for languages such as Sinhala and Tamil, specialist engines (**Surya**, **Google Document AI**) still lead frontier models. [11]
- **VLMs hallucinate**: they can produce fluent, internally consistent, *fabricated* text that passes spell-checks and totals — a failure mode traditional OCR does not have. [10][13][14]

The practical 2026 consensus is therefore **not "VLMs replace OCR"** but **tiered, hybrid pipelines**: cheap deterministic engines handle the ~80% of clean pages, specialist VLMs handle structured/complex pages, and frontier VLMs are reserved for the hardest documents — often with a traditional-OCR cross-check to catch hallucinations. [13][15]

> Related articles in this repository: [Local Multimodal Vision-Language Models](local-multimodal-vision-language-models.md) (running VLMs locally), [Large Document LLM Methods](large-document-llm-methods.md) (processing long documents), and [RAG & Context Engineering](rag-and-context-engineering.md) (OCR as a pipeline stage).

---

## How to Choose: Quick Decision Guide

| If you need… | Best non-LLM / specialist option | Best multimodal LLM option |
|---|---|---|
| **Maximum document-parsing accuracy, self-hosted** | MinerU2.5-Pro, GLM-OCR, PaddleOCR-VL-1.6 (open, sub-1.5B) | — (specialists win here) |
| **Best results on the hardest, messiest documents** | — | Gemini 3 Pro / Claude Opus 4.6+ / GPT-5.x |
| **Lowest cost at high volume** | Self-hosted PP-OCRv5 / DeepSeek-OCR2 (~$0.09–0.20/1k pages) | — |
| **Lowest latency / real-time** | PP-OCRv5, Tesseract, Surya (CPU/GPU, <1–3 s) | — |
| **Clean printed text / scanned archives** | Tesseract, PP-OCRv5 (98–99% on clean 300 dpi) | — (overkill) |
| **Complex tables, forms, nested structure → JSON** | PaddleOCR-VL, Granite Docling | Claude Opus 4.6+, Gemini 3 Pro |
| **Math / LaTeX** | GOT-OCR 2.0, dots.ocr | Gemini 3 Flash, GPT-5.x |
| **Handwriting** | (mixed; specialist ICR) | Gemini 3.1 Pro, Mistral OCR 3 |
| **Low-resource / non-Latin scripts** | Surya, Google Document AI | Qwen3-VL, Gemini 3 Ultra |
| **Edge / mobile / CPU-only** | PP-OCRv5 mobile (3.5 MB), Granite Docling (258M) | — |
| **Turn-key managed API** | Mistral OCR 3 ($1–2/1k) | AWS Textract / Azure DI / Google Document AI |
| **Regulated / compliance (SOC2, HIPAA, FedRAMP)** | — | Hyperscaler managed services |

---

## The 2026 Landscape: Four Tiers

OCR in 2026 is best understood as a progression from cheap-and-narrow to expensive-and-general. The field describes this as the shift from **"OCR-1.0"** (modular *detect → recognise → post-process* pipelines) to **"OCR-2.0"** (end-to-end VLMs that emit structured Markdown/JSON directly). [2]

| Tier | What it is | Examples | Speed | Cost/1k pages | Best at |
|---|---|---|---|---|---|
| **1. Traditional engines** | Detector + recogniser, no LLM | Tesseract, PP-OCRv5, EasyOCR, Surya, docTR | 0.5–3 s/page (CPU) | ~$0.09–0.20 (self-host) | Clean printed text, high volume, determinism, edge |
| **2. Specialist document VLMs** | Small VLMs fine-tuned for document parsing | GLM-OCR, PaddleOCR-VL, MinerU2.5, DeepSeek-OCR2, dots.ocr, olmOCR 2, GOT-OCR 2.0 | 3–8 s/page (GPU) | ~$0.09–0.70 (self-host) | PDF→Markdown, tables, formulas, reading order |
| **3. Open general VLMs** | Large general multimodal models | Qwen3-VL, InternVL3, Nemotron Nano V2 VL | 3–10 s/page | self-host / API | Documents needing reasoning + OCR together |
| **4. Frontier proprietary VLMs** | Closed flagship multimodal models + dedicated OCR APIs | Gemini 3.x, GPT-5.x, Claude Opus 4.6–4.8, Mistral OCR 3 | 5–15 s/page (API) | $1.50–70 | The messiest, hardest, judgement-heavy documents |

The crucial 2026 insight: **moving up a tier does not monotonically increase accuracy.** A 0.9B Tier-2 specialist now *out-scores* a 235B Tier-3/4 generalist on pure document parsing, because document parsing rewards precise localisation and faithful transcription rather than broad world knowledge. [7][10] You move up a tier for *judgement* (ambiguous layouts, reasoning, instruction-following), not for raw transcription accuracy.

---

## Benchmarks and How to Read Them

Benchmark selection drives the conclusion, so read leaderboards with three rules in mind:

1. **Name *and* version matter.** OmniDocBench v1.0, v1.5 and v1.6 are **not score-comparable** (v1.6 typically runs ~0.5–1 point higher for the same model). OCRBench v1 (scored out of 1.0) and OCRBench v2 (out of 100) are completely different scales. [9][3]
2. **Automated vs human-preference disagree.** Automated parsing benchmarks reward faithful transcription (specialists win); human-preference arenas reward usable, well-structured output (frontier VLMs win). Cite both lenses, not one. [2][7]
3. **Watch for conflicts of interest.** Some leaderboards are operated by vendors who also rank their own models at the top. [4]

### The major benchmarks (mid-2026)

| Benchmark | What it measures | Current top entries | Notes |
|---|---|---|---|
| **OmniDocBench v1.6** | Full-page parsing: text, formulas, tables, reading order | MinerU2.5-Pro **95.69**, GLM-OCR **95.22**, PaddleOCR-VL-1.5 **94.93** | Specialists dominate; Gemini 3 Pro **90.33**, Qwen3-VL-235B **89.15**, GPT-5.2 **85.4** [7][8] |
| **OmniDocBench v1.5** | Earlier version of the above | GLM-OCR **94.62**, PaddleOCR-VL **94.50**, DeepSeek-OCR2 **91.09**, Gemini 3 Flash **≈90.1** | Not comparable to v1.6 [1][9][12][3] |
| **OCRBench v2** | 10,000 human-verified QA pairs, 31 scenarios, bilingual | Qwen2.5-VL-72B **63.7/100** (Overall-CN) | Hard benchmark; most models score <50/100 [3][5] |
| **olmOCR-bench** | PDF-linearisation quality (1,403 pages) | LightOnOCR-2-1B **83.2**, Chandra **83.1**, olmOCR v0.4 **82.4** | #1 disputed across evaluators (see below) [4][6] |
| **OCR Arena** | Live human-preference voting | Gemini 3 Flash, then Gemini 3 Pro, Claude Opus 4.6, GPT-5.2 | Frontier VLMs lead human preference [2] |
| **Real5-OmniDocBench** | Real-world distortion stress test | PaddleOCR-VL-1.5 **92.05** | New 2026 robustness benchmark [9] |

**Contested leaderboard:** the olmOCR-bench "#1" claim varies by evaluator and version — LightOnOCR-2-1B at 83.2 (VoidSource), Nanonets OCR-3 at 87.4 (the Nanonets-operated IDP Leaderboard), Unsiloed at 88.0, Interfaze at 85.7. Treat any single "global #1" claim with caution. [4][6]

**Where VLMs measurably underperform specialists/traditional OCR on benchmarks:**

- **Tables** remain the hardest subtask for everyone (top scorers ~53–89% depending on the test); some general VLMs collapse here (e.g. PaddleOCR-VL scores ~85.7 on the ArXiv subtask but only ~37.8 on Tables in one evaluation). [4]
- **ArXiv/scientific text**: some frontier flash-tier models scored poorly on the ArXiv subtask of olmOCR-bench versus dedicated parsers. [4]
- **Reading order** on complex multi-column layouts needed a dedicated architectural fix — DeepSeek-OCR 2's *Visual Causal Flow* cut reading-order errors ~33%. [9]

---

## Best Multimodal LLMs for OCR

### Frontier Proprietary VLMs and Dedicated OCR APIs

These lead on the hardest, judgement-heavy documents and on human-preference evaluation. They are the most capable generalists but the most expensive and slowest, and they hallucinate (see [reliability](#reliability-and-safety-of-vlm-based-ocr)).

| Model | Released | Context | OCR positioning | Indicative cost |
|---|---|---|---|---|
| **Gemini 3 Pro / 3.1 Pro** | Dec 2025 / 2026 | 1M tokens | Best frontier all-rounder for volume + handwriting; ~90.3 OmniDocBench v1.5 | ~$1.25/M input tokens [1] |
| **Gemini 3 Flash / 3.5 Flash** | 2026 (I/O May 2026) | 1M tokens | Frontier-grade OCR at flash latency/price; tops OCR Arena | ~$0.50–1.50/M input [2][6] |
| **Gemini 3.1 Ultra** | Mar 2026 | 2M tokens | Largest context; multilingual/non-English OCR | not confirmed |
| **GPT-5.4 / GPT-5.5** | Mar / ~May 2026 | ~1–1.1M tokens | Single-pass dense scans, handwritten forms, diagrams, charts; ~85.8 OmniDocBench | from ~$2.50/M input [6] |
| **Claude Opus 4.6 → 4.7 → 4.8** | Feb 5 / Apr 16 / May 28 2026 | 200K tokens | Best at complex *structured extraction* (nested tables, forms, legal/medical) and instruction-following | ~$15/M input; ~$0.025–0.035/page vision [1][6] |
| **Mistral OCR 3** (`mistral-ocr-2512`) | Dec 2025 | — | Dedicated OCR API; 74% win-rate over OCR 2; ~96.6% tables, ~88.9% handwriting | **$2/1k pages ($1 batch)** [6][16][17] |

Notes:
- **Mistral OCR 3** is the standout *dedicated* (non-chat) OCR API: cheap, fast, strong on tables/handwriting, and the natural managed default when you don't need a hyperscaler's compliance story. It is SaaS-only with a self-hosting option for data residency, but lacks published SOC2/HIPAA/FedRAMP documentation. No newer Mistral OCR version was found as of June 2026. [16][17][13]
- Claims of dramatic vision jumps in the newest Claude releases (e.g. "98.5% visual acuity" for Opus 4.7) come from analytical blogs using **non-standard metrics** and should be read as directional, not benchmark-grade. [6]
- OCR-specific benchmarks for the very newest releases (Claude Opus 4.8, GPT-5.5) were **not yet published** at the time of writing.

### Open-Source Specialist Document VLMs

This is where open source is strongest in 2026: small, cheap, self-hostable models that **top the automated document-parsing leaderboards**. Open-source-first deployments should start here.

| Model | Params | Licence | Released | Strengths |
|---|---|---|---|---|
| **MinerU2.5-Pro** | 1.2B | Custom (Apache-2.0-based) | Apr–May 2026 | **#1 OmniDocBench v1.6 (95.69)**; PDF/Office→Markdown/JSON; data-centric [7][8] |
| **GLM-OCR** (Z.ai) | 0.9B | MIT (model) | Feb 2026 | 95.22 v1.6 / 94.62 v1.5; document understanding [1][12] |
| **PaddleOCR-VL-1.5 / 1.6** | 0.9B | Apache-2.0-based | Jan / May 28 2026 | 94.5 → 96.3; polygonal detection for warped text; tables/formulas [9][18] |
| **DeepSeek-OCR / OCR2** | 3B | MIT | Oct 2025 / Jan 2026 | Contextual optical compression (7–20× fewer vision tokens); **200k+ pages/day on one A100**; Visual Causal Flow reading order [2][19][9] |
| **dots.ocr / dots.mocr** | 1.7B | MIT code + custom weights | Jul 2025 / Mar 2026 | Unified layout+text+tables+formulas+reading order; **100+ languages** [2][20] |
| **olmOCR 2** (AllenAI) | 7B (Qwen2.5-VL) | Apache-2.0 | Jul 2025 | Fully open (weights+data+code); **<$200/M pages**; PDF linearisation [21] |
| **GOT-OCR 2.0** | 580M | Research licence | Sep 2024 | Runs on ~4 GB VRAM; Markdown/LaTeX/structured notation [22] |
| **Nanonets-OCR-3 / -s** | 3B | Open weights | 2025–26 | Markdown with LaTeX, tables, signatures; ships confidence scores + bounding boxes [4] |
| **NVIDIA Nemotron Nano (V2) VL** | 8B / 12B | NVIDIA open licence | Jun / Nov 2025 | Topped OCRBench v2 at release; hybrid Mamba-Transformer; single-GPU [23][24] |
| **IBM Granite Docling** | 258M | Open source | 2025–26 | Compact; ~97.9% table accuracy; the open core of IBM's Docling stack [25] |

### Open-Source General-Purpose VLMs

General multimodal models that do OCR well as one capability among many — choose these when the task needs **reasoning *and* OCR together** (e.g. "read this chart and explain the trend"). See the companion article [Local Multimodal Vision-Language Models](local-multimodal-vision-language-models.md) for local-deployment detail.

- **Qwen3-VL** (2B–235B, Apache-2.0): the leading open general VLM family for OCR; native 256K context, 32 languages, robust in low light/blur. Newer Qwen 3.5/3.6-VL generations exist but lacked published OCR-specific cards at writing. [2][26]
- **InternVL3 / InternVL3.5** (1B–241B, MIT): strong native-multimodal generalists, competitive on document tasks. [27]
- Hardware: GOT-OCR 2.0 ~4 GB VRAM; 3B specialists ~6–8 GB at Q4; Qwen3-VL-8B Q4 ~5 GB; 72B-class needs an A100/multi-GPU; Granite Docling (258M) runs on CPU. [2][26]

---

## When a Non-LLM Solution Is the Better Choice

This is the section to read if your instinct is "just send everything to a frontier VLM." In 2026 that is often the *wrong* default. There are five situations where a non-LLM or specialist tool is clearly better.

**1. High volume and tight budgets.** The cost gap is enormous and decisive. Self-hosted traditional/specialist OCR runs at roughly **$0.09–0.70 per 1,000 pages**; frontier-VLM API calls run **$1.50–15+** and structured cloud extraction up to **$70**. At 10M pages/month, that is **~$200–1,000 self-hosted vs $20,000–25,000+ on cloud APIs** — a 40–125× difference. [9][13][28]

**2. Latency-sensitive or real-time work.** Traditional engines return a page in **0.5–3 s on CPU**; specialist VLMs take **3–8 s on a GPU**; frontier VLMs take **5–15 s via API**, with variable tail latency. If a human is waiting or you are in an interactive loop, the deterministic engine wins. [13]

**3. Clean, high-volume printed text.** **Tesseract** still hits **98–99% accuracy on clean 300 dpi printed documents** across 100+ languages at zero inference cost on CPU. A frontier VLM adds cost, latency, and hallucination risk for no accuracy gain. [13]

**4. Precise localisation, determinism, and edge deployment.** A peer-reviewed CVPR 2026 result is the clearest statement of the case: a **5M-parameter PP-OCRv5** rivals billion-parameter VLMs on standard OCR benchmarks while offering **superior bounding-box localisation and demonstrably fewer hallucinations**, and compresses to a **3.5 MB mobile build**. VLMs, by design, forgo the precise text localisation that pipeline users rely on. [10]

**5. Low-resource and non-Latin scripts.** For Sinhala, **Surya** leads (WER 2.61%); for Tamil, **Google Document AI** leads (CER 0.78%); Tesseract and EasyOCR are significantly weaker on these scripts. Latin-script OCR is essentially solved, but low-resource scripts remain an open problem where specialist engines beat frontier generalists. [11]

### The leading non-LLM / specialist engines

| Engine | Type | Strengths | Notes |
|---|---|---|---|
| **PP-OCRv5 (PaddleOCR)** | Detector + recogniser | 106+ languages, 3.5 MB mobile build, top accuracy among traditional engines, GPU-optional | Best traditional default; CVPR 2026 5M variant rivals VLMs [10][13] |
| **Surya** | Detection + layout + recognition | 90+ languages, layout analysis, reading order, table recognition, LaTeX; the go-to layout stage in hybrid pipelines | Open source [11][29] |
| **Tesseract** | Recogniser | 100+ languages, CPU-only, zero inference cost, fully deterministic | Weak on handwriting/complex layouts [13] |
| **docTR** | Det+rec toolkit | Modular DBNet/CRNN etc., ONNX deployment, production-friendly | MIT |
| **EasyOCR** | Recogniser | Quick Python integration | Lower accuracy than PaddleOCR |

### A note on Mathpix and math OCR

Math was historically a niche where the specialist tool (**Mathpix**) was unbeatable. That moat has eroded: an independent 2026 benchmark found **Gemini 3 Flash both cheaper (~$0.004 vs $0.025/page) and more accurate** than Mathpix on multi-column mathematical content, with Mathpix making critical semantic errors. The same benchmark, however, is a cautionary tale about VLMs: **Grok 4.1 Fast completely fabricated content — only 1–4% similarity to the source page** while producing plausible-looking LaTeX. The lesson is not "VLMs always win math" but "evaluate per model, and verify output." [14]

---

## Reliability and Safety of VLM-Based OCR

The single biggest reason *not* to default to a VLM is reliability. Traditional OCR fails *visibly* (garbled characters, low confidence scores); VLM OCR can fail *invisibly*.

- **Hallucination / fabrication.** VLMs can emit contextually plausible but factually wrong text. The dangerous variant is *internally consistent* fabrication — e.g. altering a line item *and* adjusting the total to match — which passes spell-checks and arithmetic checks. Traditional OCR errors are statistically random and easy to flag; VLM errors are coherent and self-consistent, so they are harder to detect. [10][13]
- **Complete fabrication under stress.** On degraded or unusual inputs, some models generate output with near-zero relationship to the source (the Grok 4.1 Fast case: 1–4% similarity). [14]
- **Silent omission.** VLMs can skip content (a paragraph, a column, a page) without any error signal. Formal measurement of omission rates is still sparse as of mid-2026. [13]
- **Non-determinism.** The same page can yield different transcriptions across runs at temperature > 0 — a problem for audit and reproducibility.
- **Visual prompt injection.** Because a VLM *reads and may act on* text inside an image, instructions embedded in a document (a "screenshot jailbreak") can bypass text-only safety filters. This is an attack surface that traditional OCR simply does not have, and it matters for any pipeline that processes untrusted documents. [30] See also [Preventing Topic Hijacking](chatbot-topic-hijacking-prevention.md).

**Mitigation patterns (2026 best practice):**

1. **Traditional cross-check.** Run a cheap deterministic engine alongside the VLM; agreement between two independent methods is a strong correctness signal because their error modes differ. [13][15]
2. **Confidence routing.** Use models/engines that emit confidence scores and bounding boxes (e.g. PaddleOCR, Nanonets-OCR-3) and escalate only low-confidence regions.
3. **Sequence alignment.** Align VLM output against traditional-OCR output (e.g. Needleman–Wunsch) to localise and correct divergences. [15]
4. **Treat document text as untrusted data** for injection-hardening; never let extracted text become executable instructions to a downstream agent. [30]

---

## Hybrid Pipelines and Tiered Routing

The dominant production architecture in 2026 is a **confidence-based tiered fallback**, not a single model. A representative pipeline: [13][15]

- **Tier 0 — Embedded text extraction.** If the PDF already has a text layer, extract it directly (free, perfect). 
- **Tier 1 — Traditional engine** (PaddleOCR / Tesseract) handles the ~80% of clean pages at lowest cost and latency.
- **Tier 2 — Specialist document VLM** (dots.ocr / PaddleOCR-VL / Qwen3-VL) for pages where Tier-1 confidence is moderate (~0.70–0.90) or structure is complex.
- **Tier 3 — Frontier VLM** (Gemini 3 / GPT-5.x / Claude Opus 4.6+) only for the hardest, lowest-confidence pages.

A common open-source variant pairs **Surya for layout/reading-order detection** with a **local VLM (olmOCR / GLM-OCR / Qwen3-VL) for recognition**, plus a **sequence-alignment correction step** against traditional OCR — giving fully offline, hallucination-checked, searchable output on consumer hardware. [15]

---

## Cost Economics

Cost per 1,000 pages spans nearly three orders of magnitude. This is the dominant driver of architecture decisions at scale.

| Solution | Cost / 1,000 pages | Notes |
|---|---|---|
| Self-hosted traditional (CPU) | **~$0.09–0.20** | Cheapest; clean text [13][28] |
| Self-hosted open VLM (A100 spot) | **~$0.10** | ~$1,000/mo for 10M pages [28] |
| Self-hosted open VLM (H100) | **~$0.14–0.70** | olmOCR ~$0.19; vs GPT-4o-class API ~$12.48 [12][13] |
| **Mistral OCR 3** | **$1–2** | $1 batch; cheapest managed API [16][17] |
| AWS Textract — basic text | **$1.50** ($0.60 >1M) | [13][28] |
| Azure DI — Read (basic OCR) | **~$1.50** | [31] |
| Google Document AI — Enterprise OCR | **$1.50** ($0.60 >5M) | [32] |
| Google Document AI — Form Parser / Custom | **$30** ($20 >1M) | generative custom extractor priced same [32] |
| AWS Textract — Forms+Tables+Queries | **$70** ($55 >1M) | structured extraction [33] |

**Self-host vs cloud break-even** (community estimate): roughly **50,000 pages/month** for basic OCR, but only **~5,000 pages/month** for tables/forms extraction — because cloud structured-extraction pricing ($30–70/1k) is so much higher than basic OCR. Self-hosting estimates assume GPU spot pricing and exclude engineering/maintenance overhead, so true TCO is higher. [28]

The market has visibly **bifurcated**: a *commodity* tier (Mistral OCR 3 at $1–2/1k; open-source at $0.09–0.70/1k) competing purely on price, and an *enterprise* tier (hyperscalers at $1.50–70/1k) competing on compliance, SLAs, custom training, and ecosystem integration. [17]

---

## Hyperscaler Managed Services

Per this repository's conventions, managed services are noted only for the five major hyperscalers (AWS, Azure, GCP, IBM, Oracle). For most teams the **open-source-first** path (self-hosted specialist VLM, or Mistral OCR 3 as a managed API) is cheaper; the hyperscaler case rests on **compliance (SOC2/HIPAA/FedRAMP), SLAs, data residency, custom model training, and native cloud integration**.

| Cloud | Service | OCR / basic | Structured extraction | 2026 notes |
|---|---|---|---|---|
| **AWS** | Textract | $1.50/1k ($0.60 >1M) | Forms+Tables+Queries $70/1k; Expense $10/1k | Custom Queries; S3/Lambda-native [33] |
| **Azure** | AI Document Intelligence (ex–Form Recognizer) | Read ~$1.50/1k | Prebuilt ~$10–15/1k; Custom | New **custom *generative* extraction** tier; on-prem containers [31] |
| **GCP** | Document AI | Enterprise OCR $1.50/1k ($0.60 >5M) | Form Parser / Custom $30/1k; Layout Parser $10/1k | Gemini-powered custom extractor priced same as custom model [32] |
| **IBM** | **Docling** + watsonx Orchestrate | (no per-page API) | self-hosted workflow | IBM offers **open-source Docling** (Granite Docling model) on Code Engine, not a per-page OCR API [25][34] |
| **Oracle** | OCI Document Understanding | per-transaction; 5,000 free/mo | OCR, Document Extraction, Custom Extraction, Custom Training | exact per-1k rates only via dynamic calculator [35] |

Two structural points worth noting:

- **IBM is the open-source-first hyperscaler here**: rather than a metered OCR endpoint, IBM ships the open **Docling** library and the compact **Granite Docling** model, deployed on your own infrastructure — which doubles as an excellent self-hostable option in any cloud. [25][34]
- All three of AWS/Azure/GCP now offer **LLM/generative-AI-powered custom extraction** at the same price as their classic custom models — the frontier-VLM capability folded into the managed IDP stack. The broader 2026 trend is **"agentic IDP"**: document pipelines that combine extraction, reasoning, validation, routing, and action rather than just returning text. [32][36]

---

## Use-Case Recommendations

**By document type:**

| Document type | Recommended approach |
|---|---|
| Clean printed text / scanned archives | PP-OCRv5 or Tesseract (self-hosted) |
| Scientific papers / math / LaTeX | GOT-OCR 2.0 or dots.ocr (open); Gemini 3 Flash / GPT-5.x (frontier) |
| Complex tables & forms → JSON | PaddleOCR-VL or Granite Docling (open); Claude Opus 4.6+ / Gemini 3 Pro (hardest) |
| Invoices / receipts (IDP) | AWS Textract / Azure DI / Google Document AI; or Mistral OCR 3 + extraction |
| Legal / medical (compliance) | Hyperscaler managed service; frontier VLM behind a compliant boundary |
| Handwriting | Gemini 3.1 Pro or Mistral OCR 3; verify output |
| Low-resource / non-Latin scripts | Surya or Google Document AI; Qwen3-VL / Gemini 3 Ultra |
| Mixed/messy real-world documents | Frontier VLM (Gemini 3 / Claude Opus / GPT-5.x), with traditional cross-check |

**By deployment constraint:**

| Constraint | Recommended approach |
|---|---|
| Lowest cost at scale | Self-hosted PP-OCRv5 / DeepSeek-OCR2 |
| Air-gapped / on-prem | PaddleOCR-VL, MinerU, Granite Docling, Azure DI containers |
| Edge / mobile / CPU | PP-OCRv5 mobile (3.5 MB), Granite Docling (258M) |
| Consumer GPU self-host | dots.ocr, GLM-OCR, Qwen3-VL-8B, olmOCR 2 |
| Turn-key managed | Mistral OCR 3 (commodity) or a hyperscaler (enterprise) |

---

## What Changed Since January 2026

This article supersedes an internal January 2026 OCR survey. The five-month delta is substantial:

- **Specialists overtook frontier models on document parsing.** GLM-OCR (0.9B, Feb 2026), PaddleOCR-VL-1.5 (Jan) → 1.6 (May), and MinerU2.5-Pro (1.2B, Apr–May) now top OmniDocBench, beating 235B generalists. [1][7][9][12][18]
- **Specialist refreshes:** DeepSeek-OCR2 (Jan, Visual Causal Flow reading-order fix), dots.mocr (Mar), MinerU v3.1.0 (Apr). [9][19][20]
- **Frontier cadence:** Claude Opus 4.6 (Feb 5) → 4.7 (Apr 16) → 4.8 (May 28); GPT-5.4 (Mar 5, ~1M context, native vision) → GPT-5.5 (~May); Gemini 3.1 Ultra (Mar, 2M context) and Gemini 3.5 Flash (May). [6]
- **Dedicated OCR API:** Mistral OCR 3 (`mistral-ocr-2512`, Dec 2025) remains the latest; its $1–2/1k pricing reframed the managed market. [16][17]
- **Benchmarks matured:** OmniDocBench v1.6 and Real5-OmniDocBench added; OCR Arena (human preference) gained prominence; the automated-vs-human-preference divergence became the key interpretive lesson. [3][9]
- **Architecture consensus:** tiered/hybrid pipelines with traditional cross-checks displaced "send everything to a VLM" as best practice. [13][15]

---

## Caveats and Areas of Uncertainty

- **Benchmark versions are not comparable.** OmniDocBench v1.0/v1.5/v1.6 and OCRBench v1/v2 use different scales and datasets. All figures here are labelled by version; do not compare across them.
- **Some proprietary OCR figures are provisional.** Vision metrics for the newest closed models (Claude Opus 4.8, GPT-5.5) lacked published OCR-specific benchmarks at writing; some cited percentages (e.g. "98.5% visual acuity") come from analytical blogs using non-standard metrics. [6]
- **Leaderboard conflicts of interest.** The Nanonets-operated IDP Leaderboard ranks Nanonets OCR-3 #1; this is contested by other evaluators. [4]
- **Hallucination is under-quantified.** The risk is well-documented qualitatively, but formal fabrication/omission-rate studies remain sparse as of mid-2026. [10][13]
- **Pricing is approximate and changes.** Hyperscaler figures are from official pages where extractable; some (Oracle, parts of Azure) are behind dynamic calculators and are reported structurally. Self-hosting costs assume spot GPU pricing and exclude engineering overhead.
- **Geographic/recency limits.** Research was English-language and current to 2026-06-02; the field moves monthly.

---

## References

1. [Best LLM for OCR 2026: 7 Models Ranked](https://ofox.ai/blog/best-ai-model-for-ocr-2026/) — Ofox.ai (commercial aggregator; benchmark figures cross-referenced). Credibility: 6/10
2. [The Definitive Guide to OCR in 2026: From Pipelines to VLMs](https://slavadubrov.github.io/blog/2026/03/04/the-definitive-guide-to-ocr-in-2026-from-pipelines-to-vlms/) — V. Dubrov (independent engineering blog). Credibility: 8/10
3. [OCRBench v2 (official repo, NeurIPS 2025)](https://github.com/Yuliang-Liu/MultimodalOCR) — primary benchmark source. Credibility: 9.5/10
4. [olmOCR-Bench / IDP Leaderboard](https://www.idp-leaderboard.org/benchmarks/olmocr/) — Nanonets-operated (note COI). Credibility: 8/10
5. [OCRBench v2 Leaderboard](https://www.codesota.com/benchmark/ocrbench-v2) — aggregator. Credibility: 8.5/10
6. [Gemini 3.5 Flash vs Claude Opus 4.7 vs GPT-5.5](https://www.aimadetools.com/blog/gemini-3-5-flash-vs-claude-opus-4-7-vs-gpt-5-5/) — comparison blog (affiliate). Credibility: 6.5/10
7. [MinerU2.5-Pro: Pushing the Limits of Data-Centric Document Parsing at Scale](https://arxiv.org/abs/2604.04771) — arXiv (OpenDataLab). Credibility: 9/10
8. [OmniDocBench 2026: Specialists Outperform Frontier](https://www.bestaiweb.ai/mineru-2-5-glm-ocr-and-gemini-3-pro-the-2026-omnidocbench-race-for-document-parsing-supremacy/) — analysis (cites primary sources). Credibility: 7.5/10
9. [OmniDocBench v1.5 Leaderboard](https://www.idp-leaderboard.org/benchmarks/omnidocbench) and [DeepSeek-OCR 2: Visual Causal Flow](https://arxiv.org/abs/2601.20552) — IDP Leaderboard; arXiv. Credibility: 8–9/10
10. [PP-OCRv5: A Specialized 5M-Parameter Model Rivaling Billion-Parameter VLMs](https://arxiv.org/abs/2603.24373) — CVPR 2026 (Baidu). Credibility: 9.5/10
11. [Zero-shot OCR Accuracy of Low-Resourced Languages: Sinhala and Tamil](https://arxiv.org/abs/2507.18264) — RANLP 2025. Credibility: 9/10
12. [GLM-OCR: Z.ai's 0.9B Model Tops Document Benchmarks](https://rits.shanghai.nyu.edu/ai/glm-ocr-z-ais-0-9b-model-takes-the-top-spot-on-document-understanding-benchmarks/) — NYU Shanghai. Credibility: 8.5/10
13. [olmOCR-Bench Leaderboard 2026](https://voidsource.dev/en/ai/benchmarks/olmocr-bench) and self-hosting cost analysis — VoidSource; cost figures cross-referenced. Credibility: 8/10
14. [Math PDF OCR Benchmark: Gemini Flash vs Mathpix](https://igorrivin.github.io/blog/ocr-benchmark/) — independent (small sample). Credibility: 7/10
15. [Building a Local LLM-Powered Hybrid OCR Engine](https://www.ahnafnafee.dev/blog/local-llm-pdf-ocr) — developer blog with working code. Credibility: 6.5/10
16. [Introducing Mistral OCR 3](https://mistral.ai/news/mistral-ocr-3/) — Mistral AI (official). Credibility: 9.5/10
17. [Mistral OCR 3: $2/1000 Pages Cuts Document AI Costs 97%](https://byteiota.com/mistral-ocr-3-2-1000-pages-cuts-document-ai-costs-97/) — analysis (cross-checked vs official). Credibility: 7/10
18. [PaddleOCR-VL-1.5 / 1.6 documentation](https://www.paddleocr.ai/main/en/index.html) — PaddlePaddle (official). Credibility: 8.5/10
19. [DeepSeek-OCR (official repo)](https://github.com/deepseek-ai/DeepSeek-OCR) — DeepSeek-AI. Credibility: 9.5/10
20. [dots.ocr (official repo)](https://github.com/rednote-hilab/dots.ocr) — rednote-hilab. Credibility: 9/10
21. [olmOCR (official repo)](https://github.com/allenai/olmocr) — AllenAI. Credibility: 9.5/10
22. [GOT-OCR 2.0 (official repo)](https://github.com/Ucas-HaoranWei/GOT-OCR2.0) — UCAS. Credibility: 9/10
23. [NVIDIA Llama Nemotron Nano VL tops OCRBench](https://developer.nvidia.com/blog/new-nvidia-llama-nemotron-nano-vision-language-model-tops-ocr-benchmark-for-accuracy/) — NVIDIA (official). Credibility: 9/10
24. [NVIDIA Nemotron Nano V2 VL](https://arxiv.org/abs/2511.03929) — arXiv (NVIDIA). Credibility: 9/10
25. [IBM Granite Docling](https://www.ibm.com/granite/docs/models/docling) — IBM (official). Credibility: 9/10
26. [Qwen3-VL GPU/VRAM guides](https://willitrunai.com/blog/qwen-3-gpu-requirements) — hardware sizing. Credibility: 7.5/10
27. [InternVL (official repo)](https://github.com/OpenGVLab/InternVL) — OpenGVLab. Credibility: 9/10
28. [Awesome OCR 2026: Costs, Latency and Scaling](https://github.com/WalidHadri-Iron/awesome-ocr-2026/blob/master/docs/10-costs-latency-and-scaling.md) — community. Credibility: 6.5/10
29. [Surya OCR (official repo)](https://github.com/datalab-to/surya) — Datalab. Credibility: 8/10
30. [Visual Prompt Injection (OCR Attacks)](https://genbounty.com/academy/adversarial-attacks/visual-prompt-injection-ocr) — security advisory. Credibility: 6/10
31. [Azure AI Document Intelligence Pricing](https://azure.microsoft.com/en-us/pricing/details/document-intelligence/) — Microsoft (official). Credibility: 8/10
32. [Google Document AI Pricing](https://cloud.google.com/document-ai/pricing) — Google Cloud (official). Credibility: 9.5/10
33. [AWS Textract Pricing](https://aws.amazon.com/textract/pricing/) — AWS (official). Credibility: 9.5/10
34. [IBM Docling (IBM Developer)](https://developer.ibm.com/components/docling/) — IBM (official). Credibility: 8.5/10
35. [Oracle OCI Document Understanding](https://www.oracle.com/artificial-intelligence/document-understanding/) — Oracle (official). Credibility: 8/10
36. [Intelligent Document Processing: The Future of Automation (2026)](https://highpeaksw.com/intelligent-document-processing/) — consultancy analysis. Credibility: 7/10
