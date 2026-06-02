# Molmo2: Capabilities Overview

| Field | Value |
|-------|-------|
| Created | 2026-06-03 |
| Last Updated | 2026-06-03 |
| Version | 1.0 |

---

## Table of contents

- [At a glance](#at-a-glance)
- [Executive summary](#executive-summary)
- [Model family and variants](#model-family-and-variants)
- [Architecture](#architecture)
- [Training data and recipe](#training-data-and-recipe)
- [Capability matrix](#capability-matrix)
- [Image captioning](#image-captioning)
- [OCR and text extraction from images](#ocr-and-text-extraction-from-images)
- [Video understanding](#video-understanding)
- [Video captioning](#video-captioning)
- [Reading text in video: on-screen text and subtitles](#reading-text-in-video-on-screen-text-and-subtitles)
- [Grounding: pointing, counting, and tracking](#grounding-pointing-counting-and-tracking)
- [Benchmark comparison summary](#benchmark-comparison-summary)
- [Deployment and how to run](#deployment-and-how-to-run)
- [MolmoPoint](#molmopoint)
- [Limitations](#limitations)
- [How Molmo2 fits against managed services](#how-molmo2-fits-against-managed-services)
- [Bottom line and recommendations](#bottom-line-and-recommendations)
- [Caveats and uncertainties](#caveats-and-uncertainties)
- [References](#references)

---

## At a glance

**Molmo2** (stylised *Molmo 2*; "Multimodal Open Language Model") is a family of fully-open vision-language models (VLMs) from the **Allen Institute for AI (Ai2)**, released **16 December 2025**. It is the successor to the 2024 Molmo family and is documented in the CVPR 2026 paper *"Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding"* (arXiv 2601.10611).

| Property | Value |
|----------|-------|
| Developer | Allen Institute for AI (Ai2) |
| Released | 16 December 2025 |
| Variants | Molmo2-4B, Molmo2-8B, Molmo2-O-7B |
| Vision encoder | SigLIP 2 So400m/14 @ 384px (all variants) |
| LLM backbone | Qwen3 (4B, 8B); OLMo 3 (7B-O, fully open) |
| Inputs | Single image, multi-image sets, **video** |
| Outputs | Free-form text **and** grounded outputs (points, object tracks, grounded chain-of-thought) |
| Licence | Apache 2.0 (weights); training data carries third-party constraints |
| Openness | Open weights + open training **data** + open training **code** (no distillation from closed VLMs) |
| Headline strength | Short-video understanding, captioning, counting, and **spatio-temporal grounding** |
| Relative weakness | Long (10 min+) video; dense-document OCR; multimodal reasoning (MMMU/MathVista) |

## Executive summary

Molmo2 is the first **fully open** (weights + data + code, no distillation from proprietary models) VLM family to bring **grounding** — pointing and object tracking in pixels — into the **video** domain, not just still images. Its flagship 8B model is *state-of-the-art among open models* on short-video understanding, video captioning, and counting, and it is competitive with — and on grounding tasks ahead of — much larger proprietary systems such as Gemini 3 Pro and Gemini 2.5 Pro.

For the three capabilities of primary interest here:

- **Image captioning** — A genuine strength. Molmo2 inherits the dense, length-controllable captioning behaviour of Molmo v1 (PixMo-Cap) and is trained to emit highly detailed descriptions on demand.
- **OCR / text extraction from images** — Strong but not the category leader. Molmo2-8B is excellent at *scene/natural-image* text (TextVQA 85.7, beating GPT-5, Gemini 3 Pro and Qwen3-VL-8B) and very good on documents (DocVQA 93.2). On the most OCR-dense document/chart/infographic benchmarks it trails the best open-weight model (Qwen3-VL). It has no dedicated raw-transcription "OCR mode" and does not report OCRBench.
- **Video understanding and captioning** — The headline capability. Molmo2 leads open models on short-video QA and dense video captioning, and dramatically outperforms even proprietary models on video *pointing* and *tracking*. On-screen/scene text in video is supported (the model trains on news-video and road-sign text-reading data) but is not separately benchmarked.

The remainder of this article details each capability with concrete benchmark numbers, then covers deployment, the MolmoPoint extension, limitations, and how Molmo2 relates to managed cloud services.

## Model family and variants

All three variants share the same vision encoder and connector design; they differ only in the language-model backbone.

| Variant | Total params | LLM backbone | Max context | "Fully open"? | Notes |
|---------|-------------|--------------|-------------|---------------|-------|
| **Molmo2-4B** | ~4.85B | Qwen3-4B-Instruct-2507 | 36,864 tokens | Weights/data/code open; backbone is Qwen3 | Compact "workstation" model; surprisingly close to the 8B |
| **Molmo2-8B** | ~8.66B | Qwen3-8B | 36,864 tokens | Weights/data/code open; backbone is Qwen3 | Best overall; the flagship |
| **Molmo2-O-7B** | ~7.76B | OLMo 3 7B Instruct | 65,536 tokens | **Yes** — every component (LLM, encoder choice, data, code) is open | Demonstrates a fully-open end-to-end stack on Ai2's own LLM |

The "**-O**" suffix denotes the OLMo-backed, end-to-end-open build: because the language backbone is Ai2's own OLMo 3, every layer from the LLM to the training checkpoints can be inspected and modified. The 4B and 8B use Qwen3 backbones, so while Ai2's added weights, data and code are open, the base LLM carries Alibaba's separate Qwen3 licence.

Ai2 also released **pointing/tracking-specialised** builds and the separate **MolmoPoint** family (see [MolmoPoint](#molmopoint)).

## Architecture

Molmo2 follows the now-standard VLM design — a pre-trained vision transformer (ViT) connected to a pre-trained LLM via a lightweight connector — extended to handle multi-image and video input with grounding.

- **Vision encoder:** SigLIP 2 **So400m/14 at 384×384** (≈380M parameters, 27 layers, patch size 14 → 729 patches per crop). The connector draws features from the 3rd-to-last and 9th-from-last ViT layers, following Molmo v1. Each frame is encoded **independently** — there is no temporal convolution or cross-frame ViT attention; all temporal mixing happens in the LLM. (DINOv2 is *not* used.)
- **Connector:** A multi-headed **attentional pooling** layer (the mean of patches serves as the query) followed by a shared MLP projection. Images pool over **2×2** patch windows; video frames pool over **3×3** windows to cut token count (≈81 vision tokens per video frame). The same connector parameters are shared across image and video.
- **Image handling:** Overlapping multi-crop tiling (as in Molmo v1) — up to ~24 crops at inference. Unlike Molmo v1, crops are resized to 378 px rather than black-padded, matching how SigLIP 2 was trained.
- **Video handling:** Frames are sampled at **S = 2 fps** as single crops, with a maximum of **F = 128 frames** (or **384** during long-context training and inference). For longer videos, F frames are sampled uniformly; the **last frame is always included** (players freeze on it, so it often matters to the user). Visual tokens are interleaved with **text timestamps**; multi-image inputs use image-index tokens.
- **Bi-directional vision attention (PrefixLM):** Vision tokens — even across different frames/images — are allowed to forward-attend to one another, while text remains causal. Ai2 reports this yields notable gains. (This requires SDPA-style attention rather than FlashAttention-2.)
- **Grounded output format:** Points are emitted as a compact HTML-like string carrying frame timestamps, **object indices** (which enable counting and identity-consistent tracking), and normalised x/y coordinates — letting one autoregressive decoder produce text, points, counts, and tracks in a unified way.

## Training data and recipe

Molmo2's central contribution is **data**: a suite of **9 new datasets** (7 video, 2 multi-image), all built **without distilling from closed VLMs** — closed *text-only* LLMs are used to help generate questions, but no proprietary vision model is in the loop.

Key new datasets:

| Dataset | Type | Scale | Purpose |
|---------|------|-------|---------|
| **Molmo2-Cap** | human | 104k video-level + 431k clip-level captions | Dense video captioning (avg **924 words/video** — the densest such corpus to date) |
| **Molmo2-AskModelAnything** | human | 140k QA pairs | Free-form, fine-grained video QA (text, actions, temporal relations) |
| **Molmo2-CapQA** | synthetic | ~1M QA (200k videos) | Large-scale video QA from model-generated captions |
| **Molmo2-SubtitleQA** | synthetic | 300k QA (100k videos) | QA requiring **audio-transcript + visual** reasoning (Whisper-1 transcripts) |
| **Molmo2-VideoPoint** | human | 650k+ pointing queries (280k videos) | Open-vocabulary spatio-temporal pointing |
| **Molmo2-VideoTrack** | human | part of ~520k grounding instances | Object tracking with complex referring queries |
| **Multi-image QA / pointing** | human + synthetic | 45k image sets, 72k QA | Multi-image understanding and document pointing |

Captioning data is collected by having annotators **speak** detailed descriptions (transcribed with Whisper-1, then rewritten by a text LLM for coherence) and merging in frame-level details from Molmo v1 — the same PixMo-Cap methodology that gave the original Molmo its dense-captioning edge.

**Three-stage training:** (1) image-captioning + image-pointing pre-training (using PixMo-Cap, PixMo-Points, PixMo-Count, CoSyn-Point, plus filtered Tulu text data; mix ≈ 60% captioning / 30% pointing / 10% text); (2) joint supervised fine-tuning over the integrated image + multi-image + video mixture; (3) a short long-context stage (sequence length 36,864, up to 384 frames). Training innovations include a **token-weighting** scheme (down-weighting very long caption/pointing outputs so they don't swamp short-answer tasks), **sequence packing**, and a **message-tree** schedule for throughput. For OCR-style image QA, Molmo2 uses **CoSyn** synthetic data instead of Molmo v1's PixMo-Docs.

## Capability matrix

| Capability | Supported? | Strength | Evidence |
|------------|-----------|----------|----------|
| Single-image understanding / VQA | ✅ | SoTA-for-open on VQA v2, RealWorldQA | Table 6 |
| Image **captioning** (dense, length-controllable) | ✅ | Strong (inherited PixMo-Cap behaviour) | §Image captioning |
| **OCR / text in images** (scene text) | ✅ | Strong — beats GPT-5/Gemini 3 on TextVQA | TextVQA 85.7 |
| **OCR / documents & charts** | ✅ | Good, but trails Qwen3-VL | DocVQA 93.2, ChartQA 86.0 |
| Multi-image reasoning | ✅ | Best-in-class open | MuirBench, MMIU, Blink |
| **Video** QA (short, <~3 min) | ✅ | SoTA-for-open | MVBench 75.9, PerceptionTest 82.1 |
| Long video (10 min+) | ⚠️ | Competitive but trails proprietary | Video-MME 69.9, MLVU 60.2 |
| **Dense video captioning** | ✅ | Best open; beats Gemini 3 Pro on F1 | Caption F1 43.2 |
| Video **pointing** (spatio-temporal) | ✅ | **Far ahead of everyone**, incl. Gemini | F1 38.4 vs 20.0 |
| Video **object tracking** | ✅ | Beats proprietary + specialist models | 56.2 vs 41.1 J&F |
| Counting (image & video) | ✅ | A standout strength | PixMoCount 88.5; video 35.5 |
| On-screen text / subtitles in video | ◐ | Trained for it; not separately benchmarked | NewsVideoQA, RoadTextVQA, SubtitleQA |
| Multimodal reasoning (math/exam) | ⚠️ | Weakest area | MMMU 53.0, MathVista 58.9 |
| Audio understanding | ❌ | No audio input; uses *text* transcripts only | — |

## Image captioning

Captioning is a **core strength** and one of Molmo's defining features since v1. Molmo2's image captioning behaviour comes from **PixMo-Cap** — the human-spoken, densely detailed image caption dataset introduced with the original Molmo — used in the pre-training stage with **length conditioning** (the model can be prompted for short or very long captions) and transcript prediction.

In practice this means Molmo2 will produce **dense, paragraph-length descriptions** that enumerate fine-grained visual detail (objects, attributes, spatial relations, text on signs, etc.) rather than the terse one-liners typical of caption-only models. Because the same pooled vision features feed a capable LLM, captions are fluent and well-organised.

Two important caveats:

1. The **Molmo2 paper does not re-run classic image-captioning benchmarks** (e.g. COCO Captions, NoCaps). Image captioning quality is inherited from the well-validated Molmo v1 / PixMo-Cap lineage; the paper's *captioning* metrics are all about **video**. For published image-caption metrics you would reference the original Molmo work.
2. Captioning is **length-controllable and grounded**: combined with pointing, the model can produce *grounded* captions where described objects are localised with points — useful for verification and downstream automation.

## OCR and text extraction from images

This is the capability most often misunderstood, so the nuance matters.

Molmo2 is a **general VLM, not a dedicated OCR engine**. It does not have a "transcribe this page verbatim" mode that is separately benchmarked, and it does **not** report OCRBench. Its text-reading ability is measured indirectly through visual-question-answering benchmarks that require reading text in the image. On those, Molmo2 is **strong — and on natural-scene text, class-leading** — but on the densest document/chart/infographic tasks it sits just behind the best open-weight specialist (Qwen3-VL).

**Image text-reading and document benchmarks (test/val split, higher = better):**

| Model | AI2D (diagrams) | ChartQA | DocVQA | InfoVQA | TextVQA (scene) | VQA v2 | RealWorldQA |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Molmo2-8B** | 95.8 | 86.0 | 93.2 | 80.1 | **85.7** | **87.0** | **77.6** |
| **Molmo2-4B** | 95.6 | 86.1 | 87.8 | 78.6 | 85.0 | 86.6 | 75.4 |
| **Molmo2-O-7B** | 93.7 | 84.9 | 90.4 | 77.9 | 84.7 | 86.6 | 73.6 |
| Qwen3-VL-8B (open) | 85.7 | **89.6** | **96.1** | **83.1** | 82.8 | 82.3 | 71.5 |
| InternVL3.5-8B (open) | 84.0 | 86.7 | 92.3 | 79.1 | 78.2 | 79.5 | 67.5 |
| GPT-5 (proprietary) | 97.1 | 89.6 | 88.9 | 83.0 | 78.7 | 79.7 | 80.8 |
| Gemini 3 Pro (proprietary) | 98.7 | 93.7 | 87.1 | 86.9 | 74.1 | 74.1 | 73.6 |

*(Source: Molmo2 paper Table 6. Bold marks where Molmo2-8B leads its comparison set.)*

How to read this for "extract text from arbitrary images":

- **Scene / natural-image text (TextVQA): a Molmo2 strength.** At 85.7, Molmo2-8B beats Qwen3-VL-8B (82.8), GPT-5 (78.7) and Gemini 3 Pro (74.1). If your images are photos, product shots, signage, screenshots-in-the-wild, Molmo2 reads embedded text very well.
- **Documents (DocVQA 93.2) and diagrams (AI2D 95.8): very good.** Molmo2-8B beats both GPT-5 and Gemini 3 Pro on DocVQA and is near the top on AI2D. It comfortably handles forms, receipts, and structured documents at a Q&A level.
- **The densest document/chart/infographic tasks: Molmo2 trails the leader.** On DocVQA (96.1), ChartQA (89.6) and InfoVQA (83.1) the open-weight **Qwen3-VL-8B is ahead.** The Molmo2 authors explicitly acknowledge being "a bit behind the best open-weight model on OCR-heavy benchmarks such as DocVQA or InfoQA." Plausible reasons: SigLIP 2 at a fixed 384 px (fewer effective pixels for tiny text than Qwen3-VL's native-resolution encoder) and the switch to CoSyn synthetic doc data.

**Practical guidance.** For *understanding and querying* text within arbitrary images — "what does this sign say?", "what's the total on this receipt?", "summarise this slide" — Molmo2 is excellent. For *bulk, verbatim transcription* of dense multi-page documents, tables, or small-font scans, a specialist is a better fit. Open-source options to pair with or prefer for that job include **Qwen3-VL** (strongest general open VLM on document OCR here), **GOT-OCR 2.0**, **dots.ocr**, and **PaddleOCR/Florence-2** for layout-aware extraction. (See [How Molmo2 fits against managed services](#how-molmo2-fits-against-managed-services) for hyperscaler OCR equivalents.)

No multilingual-OCR or handwriting figures are reported; the QA training data is filtered to English, so non-Latin scripts and handwriting should be treated as **unverified** and tested before relying on them.

## Video understanding

Video is what Molmo2 was built for, and it is the strongest *open* model across most short-video tasks. It accepts a video (sampled at 2 fps, up to 128 frames standard / 384 in long-context mode), interleaves frames with text timestamps, and answers questions or produces grounded outputs.

**Video understanding benchmarks (higher = better):**

| Model | MVBench | Perception&shy;Test | NextQA | Video&#8209;MME | LongVideo&shy;Bench | MLVU | EgoSchema | Human-pref **Elo** (rank) |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Molmo2-8B** | **75.9** | 82.1 | 86.2 | 69.9 | 67.5 | 60.2 | 62.0 | **1057 (#5)** |
| **Molmo2-4B** | 75.1 | 81.3 | 85.5 | 69.6 | 68.0 | 63.0 | 61.2 | 1041 (#8) |
| **Molmo2-O-7B** | 74.8 | 79.6 | 84.3 | 64.9 | 63.7 | 55.2 | 56.8 | 1033 (#9) |
| Qwen3-VL-8B (open) | 68.7 | 72.7 | 83.4 | 71.4 | 62.4 | 57.6 | 69.8 | 1054 (#6) |
| Eagle2.5-8B (open) | 74.8 | 81.0 | 85.0 | 72.4 | 66.4 | 60.4 | 72.2 | 1019 (#11) |
| GPT-5 (proprietary) | 74.1 | 79.4 | 86.3 | 83.3 | 72.6 | 77.7 | 75.6 | 1031 (#10) |
| Gemini 3 Pro (proprietary) | 70.4 | 77.6 | 84.3 | 88.6 | 75.9 | 75.7 | 68.9 | 1082 (#3) |
| Gemini 2.5 Pro (proprietary) | 70.6 | 78.4 | 85.3 | 87.8 | 76.8 | 81.5 | 72.2 | **1096 (#1)** |

*(Source: Molmo2 paper Table 2. The "Elo" column is from a 105k-rating human-preference study over ~450 video questions, ranked across all 21 models in the table.)*

Takeaways:

- **Short-video understanding is SoTA-for-open.** Molmo2-8B tops or near-tops MVBench (75.9), PerceptionTest (82.1) and NextQA (86.2), beating Gemini 3 Pro and matching GPT-5 on several. The compact **4B is barely behind the 8B** — a notable efficiency result.
- **Human preference puts Molmo2-8B 5th overall** (Elo 1057), *above* GPT-5 (#10) and Claude Sonnet 4.5 (#12) and ahead of every open-weight competitor — only the three Gemini models and GPT-5 mini rank higher. This is the strongest evidence that the model is genuinely useful, not just benchmark-tuned.
- **Long video is the weak spot.** On Video-MME (69.9), LongVideoBench (67.5), MLVU (60.2) and LVBench (52.8), Molmo2 trails the Gemini models substantially (Video-MME ~88, MLVU ~75–82). Ai2 attributes this to a lack of *open* long (10 min+) training data and the compute cost of ultra-long-context training — a deliberate, acknowledged trade-off, not a bug.

## Video captioning

Dense video captioning is a flagship Molmo2 capability and the basis for much of its training data. The model can watch a clip and produce a long, detailed, temporally-ordered description of what happens.

This is backed by **Molmo2-Cap**, the densest open video-caption corpus to date: 104k video-level captions averaging **924 words per video**, versus 547 for LLaVA-Video-178K and 280 for ShareGPT4Video. Captions were produced by annotators *speaking* descriptions (richer than typing), transcribed and enriched with frame-level detail — and crucially **without** generating captions from a closed VLM.

Captioning quality is evaluated on **Molmo2-CapTest** (693 Creative-Commons videos, each with ≥4 human reference captions), scoring precision/recall/**F1** of the statements in the model's caption against the human references using an LLM judge:

| Model | Video caption **F1** |
|-------|:---:|
| GPT-5 mini (proprietary) | 56.6 |
| GPT-5 (proprietary) | 50.1 |
| **Molmo2-8B** | **43.2** (best open) |
| Gemini 2.5 Pro (proprietary) | 42.1 |
| **Molmo2-O-7B** | 40.1 |
| **Molmo2-4B** | 39.9 |
| Gemini 3 Pro (proprietary) | 36.0 |
| Qwen3-VL-8B (open) | 26.7 |

*(Source: Table 2. Metric is on Ai2's own CapTest set, so treat absolute numbers as Molmo-defined; the relative ranking is the signal.)*

Molmo2-8B is the **best open model** on this metric and beats both Gemini 3 Pro and Gemini 2.5 Pro, trailing only the GPT-5 family. The combination of dense captioning *and* grounding means Molmo2 can also produce **grounded video captions** — describing events while pointing to where/when they occur — which even most proprietary models cannot do.

## Reading text in video: on-screen text and subtitles

For "extract text from video", the honest answer is: **Molmo2 is trained to read on-screen and scene text in video, but there is no dedicated video-OCR benchmark score to quote.** The evidence is indirect:

- **On-screen / scene text training data.** Molmo2's video QA mixture explicitly includes **NewsVideoQA** (reading text overlays/chyrons in news video) and **RoadTextVQA** (reading text on road signs and storefronts in driving video). Training on these teaches the model to read text that appears *within the visual frames* of a video.
- **Subtitles via audio transcript, not visual OCR.** The **Molmo2-SubtitleQA** dataset (300k QA over 100k videos) is built by transcribing the **audio** with Whisper-1 and appending the transcript as text after the visual tokens. This teaches the model to *reason jointly over spoken language and visuals* — but the "subtitle" content comes from the **audio track, not from reading burned-in caption pixels.** Molmo2 itself takes **no audio input** at inference; if you want it to use spoken dialogue, you must supply a transcript as text alongside the video.
- **Frame-level OCR is achievable but unmeasured.** Because the model reads text in still images well (TextVQA 85.7) and processes video as frames, asking "what text appears in this clip?" will work for clear on-screen text. Ai2 does not publish a frame-text-extraction accuracy figure, so treat heavy video-OCR workloads as **needs-testing**.

Practical pattern for robust video text extraction: pair Molmo2's visual reading with an explicit **ASR transcript** (e.g. Whisper) supplied as text, exactly as the SubtitleQA recipe does — giving the model both the on-screen text (visual) and the spoken text (transcript).

## Grounding: pointing, counting, and tracking

Grounding is Molmo2's signature differentiator and where it most clearly **beats proprietary models**. "Grounding" means localising things in pixels — and Molmo2 extends the 2D image-pointing paradigm of Molmo v1 into **space and time** for video.

- **Video pointing** ("click the moment and location where X happens"): Molmo2-4B **39.9 F1**, Molmo2-8B **38.4 F1**, versus **Gemini 3 Pro 20.0**, **Gemini 2.5 Pro 13.0**, **GPT-5 4.1**, and **Qwen3-VL-8B ~1.5**. Proprietary models can barely do this task; they must be coaxed into emitting bounding boxes whose centres are then used as points.
- **Video counting** (count events/objects over time): Molmo2-8B **35.5** accuracy vs Qwen3-VL **29.6**; on the BURST-VideoCount "close" metric Molmo2 even edges GPT-5. Image counting is also a standout (PixMoCount 88.5, far above all baselines).
- **Video object tracking** (continuously locate an object as it moves/occludes): Molmo2 **outperforms all baselines, including specialist video-segmentation models**, across MeViS, Ref-YouTube-VOS, Ref-DAVIS, ReasonVOS and Ai2's new **Molmo2-Track** benchmark — e.g. **56.2 vs Gemini 3 Pro 41.1** on the J&F segmentation-quality metric (points are converted to masks via SAM 2 for scoring). It excels especially on reasoning-heavy and occlusion-heavy cases.

These grounded outputs are produced by the *same* autoregressive decoder as text, using the object-indexed point format — so a single prompt can ask the model to describe, count, point, and track in one pass.

## Benchmark comparison summary

Where Molmo2-8B sits relative to its main peers (✅ leads / ◐ competitive / ⚠️ trails):

| Task area | vs best open-weight (Qwen3-VL) | vs proprietary (GPT-5 / Gemini 3 Pro) |
|-----------|:---:|:---:|
| Scene-text reading (TextVQA) | ✅ leads | ✅ leads |
| Document/chart OCR (DocVQA/ChartQA/InfoVQA) | ⚠️ trails Qwen3-VL | ◐ beats on DocVQA, mixed elsewhere |
| General image VQA (VQA v2, RWQA) | ✅ leads | ◐ competitive |
| Multimodal reasoning (MMMU, MathVista) | ⚠️ trails | ⚠️ trails |
| Short-video QA (MVBench, PerceptionTest) | ✅ leads | ◐ matches/beats |
| Long-video QA (Video-MME, MLVU) | ◐ mixed | ⚠️ trails Gemini clearly |
| Dense video captioning (F1) | ✅ leads | ◐ beats Gemini, trails GPT-5 |
| Video pointing / counting / tracking | ✅ leads decisively | ✅ **beats Gemini 3 Pro decisively** |
| Human-preference video Elo | ✅ #5 overall, top open | ◐ above GPT-5, below Gemini |

The one-line synthesis: **Molmo2-8B is the best open model for short-video understanding, captioning, counting and grounding, the only model (open or closed) with strong video pointing/tracking, a strong general image and scene-text model, and a deliberate non-leader on long video and dense-document OCR.**

## Deployment and how to run

- **Weights & code:** Model cards at `huggingface.co/allenai/Molmo2-8B` (and `-4B`, `-O-7B`); code at `github.com/allenai/molmo2`; paper arXiv 2601.10611.
- **Licence:** Model weights are **Apache 2.0**. ⚠️ The *training data* bundles third-party datasets "for academic and non-commercial research use only," and the 4B/8B inherit the **Qwen3** backbone licence — review provenance before commercial deployment. The **Molmo2-O-7B (OLMo 3)** build is the cleanest choice for fully-open, end-to-end-inspectable use.
- **Runtimes:** HuggingFace **Transformers** (`AutoModelForImageTextToText` + `AutoProcessor`) and **vLLM ≥ 0.15.0** (with a Gradio demo). Context parallelism is provided for long-video / low-VRAM setups. Note the model needs SDPA-style attention (bi-directional vision), not FlashAttention-2.
- **Public demo:** `playground.allenai.org`.
- **Approximate VRAM (weights only — video KV-cache adds substantially):**

| Variant | FP16 | Q8 | Q4 | Fits on |
|---------|:---:|:---:|:---:|---------|
| Molmo2-4B | ~10.8 GB | ~5.4 GB | ~2.7 GB | A single 24 GB consumer GPU (e.g. RTX 4090) comfortably; smaller cards when quantised |
| Molmo2-8B | ~19.4 GB | ~9.7 GB | ~4.8 GB | A 24 GB card at FP16 for images; long video benefits from 40–80 GB (A100/H100) |
| Molmo2-O-7B | ~16 GB | ~8 GB | ~4 GB | Similar to 8B |

*(VRAM figures are third-party estimates for weights only; real video inference with up to ~31k vision tokens at 384 frames needs materially more headroom.)*

## MolmoPoint

Alongside Molmo2, Ai2 released **MolmoPoint**, a separate model family with a redesigned **pointing architecture**. Instead of emitting coordinates as text, MolmoPoint uses a **coarse-to-fine grounding** mechanism with three special tokens — `<PATCH>` → `<SUBPATCH>` → `<LOCATION>` — attending over visual tokens to pick a coarse patch, refining with lower-level ViT features, then predicting an exact location. It uses rotary embeddings to encode spatial proximity and a "no-more-points" class to stop cleanly, cutting each point from ~8 tokens to **3 tokens** and improving counting and pointing accuracy (it wins human preference ~59% of the time over the baseline).

Variants: **MolmoPoint-8B** (general), **MolmoPoint-GUI-8B** (software-interface / agentic UI pointing, trained on 36k high-res screenshots with 2M+ annotated points), and **MolmoPoint-Vid-4B** (video). If your use case is primarily UI automation or high-precision pointing rather than general VQA/captioning, MolmoPoint is the more specialised tool; for captioning, OCR and video understanding, the main Molmo2 models are the right choice.

## Limitations

Drawn from the paper's own limitations discussion and benchmark gaps:

1. **Long-video** (10 min+) understanding trails proprietary and some open-weight models — a function of scarce open long-video training data and compute limits.
2. **Grounding is effectively limited to clips under ~3 minutes**, because point/track annotations were collected at 2 fps and longer clips would force sub-2-fps sampling that misaligns with the labels.
3. **Dense-document OCR** trails the best open-weight model (Qwen3-VL); the fixed 384 px SigLIP 2 input limits effective resolution for very small text.
4. **Multimodal reasoning** (MMMU 53.0, MathVista 58.9 for the 8B) is the weakest area, attributed to a lack of multimodal reasoning training data.
5. **Repeating-text degeneration** can occur when generating *very* long video captions with greedy decoding (after thousands of tokens) — mitigated by sampling (top-p 0.95, temperature 0.7) for long generations.
6. **No audio input.** "Subtitle"/dialogue reasoning relies on an externally supplied transcript; the model does not ingest audio.
7. **Not fully open at the encoder** for the headline metrics: the SigLIP 2 vision encoder is open-weights but trained on closed data. (No closed *VLM* was used to generate training data, which is the stronger openness claim.)
8. **English-centric**: QA data is filtered to English; multilingual and handwriting performance is unmeasured.

## How Molmo2 fits against managed services

Molmo2 is open-source-first by design, so the natural comparison is self-hosting Molmo2 versus a managed cloud API. For teams that prefer managed equivalents, the relevant offerings from the major hyperscalers are:

- **Image OCR / document extraction:** AWS **Textract**, Azure **AI Document Intelligence** (formerly Form Recognizer), Google Cloud **Document AI**, IBM **Watson Document Understanding / Datacap**, Oracle Cloud **AI Document Understanding**. For dense, verbatim document OCR these specialist services typically beat any general VLM, including Molmo2 — use them when transcription fidelity and layout structure matter most.
- **General image understanding / captioning:** AWS **Rekognition**, Azure **AI Vision** (image captioning/dense captions), Google Cloud **Vision API**, Oracle **Vision**. Molmo2 offers far richer, instructable, dense captioning than these fixed-API services.
- **Video understanding:** AWS **Rekognition Video**, Azure **AI Video Indexer**, Google Cloud **Video Intelligence API**. These do label/shot/face/text detection well but offer nothing like Molmo2's open-vocabulary **pointing and tracking** or instructable dense captioning.
- **Hosted frontier VLMs:** Gemini (Google Cloud **Vertex AI**), Claude and others via AWS **Bedrock** / Azure **AI Foundry** / Oracle **OCI Generative AI**, give stronger long-video and reasoning performance but cannot do video grounding and are closed.

**Open-source alternatives** worth pairing with or substituting for Molmo2 by task: **Qwen3-VL** (strongest general open VLM here, and the document-OCR leader), **InternVL 3.5**, **GLM-4.1V**, **MiniCPM-V 4.5** for general VLM work; **GOT-OCR 2.0**, **dots.ocr**, **PaddleOCR**, **Florence-2** for specialist/verbatim OCR and layout; and **Molmo2 itself** as the open leader for video grounding, counting, and dense video captioning.

## Bottom line and recommendations

- **Use Molmo2 when** you need an open, inspectable, self-hostable model for: dense and grounded **image/video captioning**; **short-video** Q&A; **counting**; and especially **spatio-temporal grounding** (pointing/tracking) — where it is the best available option, open or closed.
- **Reach for a specialist when** you need verbatim OCR of dense/multi-page/small-font documents (Qwen3-VL or a dedicated OCR model / managed Document AI), long-form (10 min+) video reasoning (a Gemini-class API), or multimodal exam-style reasoning.
- **For "extract text from arbitrary images":** Molmo2 is a strong choice for *reading and answering about* text in photos, signage, screenshots, slides and standard documents (class-leading on scene text). For *bulk verbatim transcription*, layer a specialist OCR model on top.
- **For "caption things":** Molmo2 is one of the best open options for both images and video, with the unique bonus of grounded captions.
- **For "extract text from video":** workable for clear on-screen text; pair it with an ASR transcript (the SubtitleQA pattern) for dialogue, and test on your footage since there is no published video-OCR score.
- **Variant choice:** **Molmo2-8B** for best quality; **Molmo2-4B** when VRAM/throughput matters (it's remarkably close); **Molmo2-O-7B** when end-to-end openness or a clean licence chain is the priority.

## Caveats and uncertainties

- Several headline benchmarks (Molmo2-VideoPoint, Molmo2-Track, Molmo2-CapTest, PixMoCount) are **Ai2's own evaluation sets**. The relative rankings are credible, but absolute numbers are not directly comparable to externally-defined benchmarks.
- A widely-repeated stat that Molmo2-8B trained on **9.19M videos** (vs 72.5M for Meta's PerceptionLM) comes from a **low-credibility secondary blog** and could not be confirmed in the primary paper — treat as **unverified**.
- The **CVPR 2026 poster abstract** quotes "32.9% vs 17%" for video pointing, whereas the latest arXiv v4 paper gives **38.4 vs 20.0 (Gemini 3 Pro) / 13.0 (Gemini 2.5 Pro)**. The v4 numbers are used here as authoritative; the qualitative conclusion (Molmo2 is ~2× ahead) is unchanged.
- Image-captioning quality is asserted from the inherited PixMo-Cap lineage; the Molmo2 paper does **not** re-benchmark classic image-caption metrics.
- VRAM figures are third-party weight-only estimates; budget more for video.

## References

1. [Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding](https://arxiv.org/abs/2601.10611) — arXiv 2601.10611 (CVPR 2026). Primary technical paper (Ai2). HTML: https://arxiv.org/html/2601.10611v4
2. [Molmo 2: State-of-the-art video understanding, pointing, and tracking](https://allenai.org/blog/molmo2) — Ai2 official blog.
3. [Molmo | Ai2](https://allenai.org/molmo) — Ai2 product page (variant descriptions).
4. [allenai/molmo2](https://github.com/allenai/molmo2) — Official code repository (training + Transformers/vLLM inference).
5. [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B), [Molmo2-4B](https://huggingface.co/allenai/Molmo2-4B), [Molmo2-O-7B](https://huggingface.co/allenai/Molmo2-O-7B) — Official HuggingFace model cards.
6. [MolmoPoint: Better pointing architecture for vision-language models](https://allenai.org/blog/molmopoint) — Ai2 blog (MolmoPoint extension).
7. [Ai2 Releases Molmo 2 (press release)](https://www.businesswire.com/news/home/20251216910167/en/Ai2-Releases-Molmo-2-State-of-the-Art-Open-Multimodal-Family-for-Video-and-Multi-Image-Understanding) — BusinessWire, 16 Dec 2025.
8. [Ai2's Molmo 2 shows open-source models can rival proprietary giants in video](https://venturebeat.com/infrastructure/ai2s-molmo-2-shows-open-source-models-can-rival-proprietary-giants-in-video) — VentureBeat.
9. [Allen Institute for AI introduces Molmo 2](https://siliconangle.com/2025/12/16/allen-institute-ai-introduces-molmo-2-bringing-open-video-understanding-ai-systems/) — SiliconAngle.
10. [Molmo2 CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/36933) — CVPR 2026 (note: poster abstract quotes an earlier pointing aggregation).
11. [Molmo 2 coverage (training-data volume claim — unverified)](https://abit.ee/en/artificial-intelligence/molmo-2-allen-institute-ai2-multimodal-model-video-ai-open-source-2026-en) — abit.ee (low credibility; cited only for the unverified 9.19M-video figure).
