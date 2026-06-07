# Creating Deepfakes in 2026: Models, Hardware and How Accessible the Technology Has Become

| Field | Value |
|-------|-------|
| Created | 2026-06-08 |
| Last Updated | 2026-06-08 |
| Version | 1.0 |

---

## Table of contents

- [Executive summary](#executive-summary)
- [Scope and how to read this article](#scope-and-how-to-read-this-article)
- [The threat in one page: a calibrated picture](#the-threat-in-one-page-a-calibrated-picture)
- [How deepfakes are made: the four pipeline classes](#how-deepfakes-are-made-the-four-pipeline-classes)
  - [1. Face-swap](#1-face-swap)
  - [2. Talking-head, reenactment and lip-sync](#2-talking-head-reenactment-and-lip-sync)
  - [3. Voice cloning](#3-voice-cloning)
  - [4. Real-time / live deepfakes](#4-real-time--live-deepfakes)
- [The open-source toolkit](#the-open-source-toolkit)
- [The no-skill path: commercial and hyperscaler services](#the-no-skill-path-commercial-and-hyperscaler-services)
- [Hardware and cost: what it actually takes](#hardware-and-cost-what-it-actually-takes)
- [How easy is it really? Barriers and the steel-man](#how-easy-is-it-really-barriers-and-the-steel-man)
- [Anatomy of an executive-impersonation attack](#anatomy-of-an-executive-impersonation-attack)
- [Documented incidents](#documented-incidents)
- [The numbers, handled honestly](#the-numbers-handled-honestly)
- [Defences and process controls](#defences-and-process-controls)
- [A defensive checklist for protecting an executive team](#a-defensive-checklist-for-protecting-an-executive-team)
- [Key findings and confidence](#key-findings-and-confidence)
- [Areas of uncertainty, limitations and caveats](#areas-of-uncertainty-limitations-and-caveats)
- [References](#references)

---

## Executive summary

In 2026, the question is no longer *whether* a convincing deepfake of one of your executives can be made — it can, cheaply and quickly — but *which kind of deepfake is a realistic operational threat to your organisation today, and which is still hard.* Getting that distinction right is the difference between a calibrated defence and either complacency or panic. This article is written for two readers at once: the security researcher who needs the model names, architectures and hardware specifics, and the executive who simply wants to know how exposed their leadership team is and what to do about it.

The honest, evidence-led picture has three tiers:

1. **Voice cloning is solved, cheap and proven.** A convincing clone of an executive's voice can be built from as little as **3–30 seconds** of the audio that is already public — earnings calls, conference talks, YouTube interviews, podcasts — using free open-source models ([F5-TTS](https://arxiv.org/abs/2410.06885), [the VALL-E lineage](https://arxiv.org/abs/2301.02111), Chatterbox, XTTS) or a consumer subscription such as ElevenLabs (from ~US$5/month). Real-time voice *conversion* (speak in your own voice, come out sounding like the target) runs at **sub-20-millisecond latency on a laptop CPU**. Voice-only "CEO fraud" is the most mature and most frequently confirmed attack — it has been documented since the [first case in 2019](https://www.wsj.com/articles/fraudsters-use-ai-to-mimic-ceos-voice-in-unusual-cybercrime-case-11567157402).

2. **Pre-recorded / one-way deepfake video is widespread.** Fake "endorsement" videos of executives and celebrities promoting fraudulent investment platforms drove the single largest slice — **US$632M** — of the FBI's 2025 AI-fraud tally. These don't need to survive a live conversation, so quality is the only barrier, and quality is now excellent.

3. **Interactive, real-time video face-swap on a live two-way call is the hardest tier — and the evidence base is more nuanced than the headlines suggest.** It is *confirmed in the wild only for romance scams* ([WIRED's reporting on the "Yahoo Boys"](https://www.wired.com/story/yahoo-boys-real-time-deepfake-scams/)), where the scammer controls the conversation. The famous [Arup US$25.6M Hong Kong loss](https://www.weforum.org/stories/2025/02/deepfake-ai-cybercrime-arup/) is the closest corporate case — but **no primary source has confirmed whether the fake "executives" on that call were interactive or pre-rendered clips**, and Arup's own CIO frames it as "technology-enhanced social engineering." Real-time face-swap on a consumer GPU still tops out around **20 FPS** and breaks on profile turns, hand-over-face occlusion and fast motion.

The technology is genuinely accessible: the leading tools are free, open-source, GUI-driven, installable in one click ([FaceFusion](https://github.com/facefusion/facefusion) + [Pinokio](https://pinokio.co/), [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)), and supported by a vast YouTube/Discord how-to ecosystem. Arup's CIO built a working real-time deepfake of himself in **about 45 minutes**. The expertise-heavy era is over: [DeepFaceLab](https://github.com/iperov/DeepFaceLab), the old train-it-yourself standard, was archived in November 2024 because single-photo, no-training tools superseded it.

The defensive conclusion is liberating rather than frightening: because the strongest, most-proven attacks are voice and one-way video, **the controls that defeat them are process controls, not detection technology** — out-of-band callback verification, dual authorisation on payments, and pre-agreed challenge phrases. These already defeated the [WPP](https://www.theguardian.com/technology/article/2024/may/10/ceo-wpp-deepfake-scam), [Ferrari](https://fortune.com/2024/07/27/ferrari-deepfake-attempt-scammer-security-question-ceo-benedetto-vigna-cybersecurity-ai/) and [LastPass](https://blog.lastpass.com/posts/attempted-audio-deepfake-call-targets-lastpass-employee) attempts. Passive deepfake *detection* is structurally losing the arms race; provenance and process are where durable trust lives.

## Scope and how to read this article

- **Goal.** Explain how deepfakes are created and run in 2026 — the generation pipelines, the open-source and commercial models, the hardware required — and give a calibrated answer to "how easy and accessible is this, really?", grounded in real incidents and realistic attacker workflows, for an audience worried about executive impersonation.
- **In scope.** Generative architectures (autoencoder face-swap, single-shot GAN, diffusion, neural rendering, flow-matching/codec-LM voice); open-source toolchains; commercial and hyperscaler avatar/voice services; consumer-GPU vs cloud hardware; cost/time/skill barriers; real-time video-call and voice-phishing workflows; documented incidents; defences and process controls.
- **Out of scope — and where to find it.** The deep technical treatment of **deepfake *detection*** (liveness, presentation/injection attack detection, passive detectors, morphing, provenance internals, hyperscaler liveness services) lives in the companion article, [**Facial Recognition: Deepfake & Impersonation Detection**](facial-recognition-deepfake-impersonation-detection.md). This article is the *offence-and-accessibility* counterpart; it touches detection only where it bears on defence.
- **Conventions.** Open-source options are presented first in every category; managed-service mentions are limited to the major hyperscalers (AWS, Azure, GCP, IBM, Oracle) per house style. Each major finding carries a confidence level. **Vendor-reported statistics are flagged explicitly** — a large share of the alarming numbers in this field come from companies that sell detection or anti-fraud products, and we treat those as upper bounds, not facts. Dates and "current" judgements are anchored to June 2026.

> **A note for the worried executive (read this first).** The instinct after reading a deepfake headline is to buy detection software. That is usually the wrong first move. The single most effective thing you can do this week costs nothing: institute an **out-of-band callback rule** for any payment or sensitive request, no matter how convincingly it appears to come from you. Every confirmed attack that *failed* did so because a human verified through a second channel — not because software caught a fake.

## The threat in one page: a calibrated picture

Most coverage collapses three very different capabilities into one scary word. Separating them is the most useful thing this article can do. The table below is the lens for everything that follows.

| Attack mode | Maturity in the wild (2026) | What it needs | Realistic against your execs? |
|---|---|---|---|
| **Voice clone — pre-recorded or real-time** (vishing call, voicemail, "confirm this wire") | **Proven & common.** Confirmed since 2019; layered into business-email-compromise (BEC) chains today. | 3–30 s of public audio; free OSS or ~US$5/mo SaaS; a phone | **Yes — this is the primary threat.** High likelihood, high impact. |
| **Pre-recorded / one-way video** (fake "endorsement" clip, a video message, a one-way "leave-a-recording" ask) | **Widespread.** ~US$632M of the FBI's 2025 AI-fraud losses were celebrity/exec endorsement deepfakes for fake investment platforms. | Some reference footage; one-shot tools; no live constraint | **Yes**, especially for brand/investment fraud and to "prove" a vishing pretext. |
| **Interactive real-time video face-swap on a live two-way call** (the "everyone on the Zoom was fake" scenario) | **Confirmed only in romance scams** (WIRED/"Yahoo Boys"); **not** forensically confirmed in any corporate case. Arup is the closest candidate but its mechanism is unconfirmed. | Reference imagery, a capable GPU (~20 FPS on an RTX 4090), virtual-camera plumbing, *and* the operator skill to hold a live conversation while managing the rig | **Emerging.** Lower likelihood today, very high impact. Breaks on profile turns / hand occlusion / fast motion. Rapidly improving. |

**Why the distinction matters for defence.** The two proven tiers (voice, one-way video) are defeated by *process*, not by spotting artefacts. The hard tier (interactive video) is the one where "ask them to turn sideways" still works — for now. Anchoring your whole programme on the dramatic interactive-video scenario risks buying detection plugins while leaving the wide-open voice/BEC door unlocked.

> **Confidence: High** on the bifurcation itself; **Medium** on the precise mechanism of the Arup case (the public record is genuinely ambiguous — see [Documented incidents](#documented-incidents) and [Areas of uncertainty](#areas-of-uncertainty-limitations-and-caveats)).

## How deepfakes are made: the four pipeline classes

Almost every deepfake in 2026 is one of four things — a **face-swap**, a **talking-head/reenactment**, a **voice clone**, or a **real-time** combination of these fed into a live call. Understanding the pipelines explains both why they are so accessible and where their tells come from.

### 1. Face-swap

Face-swap replaces the face in a target video/photo with someone else's identity while keeping the target's pose, expression and lighting. Three architectural families coexist:

- **Autoencoder (the DeepFaceLab/FaceSwap lineage).** A *shared encoder* learns a pose/expression/lighting latent; *separate decoders* are trained per identity. To put person A onto person B's body, you encode B's frame and decode it through A's decoder. This is the original "deepfake" recipe — **highest fidelity for a specific identity pair, but it must be trained per-identity** (hours to days on a GPU) and needs lots of source imagery. It is now the *minority* approach.
- **Single-shot GAN (the dominant accessible method).** Models such as **InsightFace's `inswapper`**, **SimSwap** and **GHOST** extract a fixed **identity embedding** (an ArcFace-style vector) from *one* source photo and inject it into the target frame in a single forward pass — **no per-identity training at all.** This is what makes "drop in one photo and go" tools possible.
- **Diffusion-based face-swap (the quality frontier).** Iterative denoising with identity conditioning (cross-attention / IP-Adapter / LoRA) gives the highest quality and stable training, at higher inference cost — increasingly mitigated by few-step distillation.

The **pipeline stages** are consistent across families and worth knowing because each is a potential tell:

1. **Face detection** (RetinaFace/SCRFD) finds faces per frame.
2. **Landmark alignment** warps the face to a canonical pose.
3. **Identity encoding** (ArcFace embedding) — or a decoder pass, for autoencoders.
4. **The swap** — forward pass (GAN), decoder (autoencoder) or denoising (diffusion).
5. **Blending & colour correction** (Poisson/seamless blending, histogram matching) to hide the seam.
6. **Face restoration / super-resolution** (**GFPGAN**, **CodeFormer**) to sharpen detail.

> **Confidence: High.** The migration from train-per-identity autoencoders to single-shot embedding injection is the single biggest reason the skill barrier collapsed: identity is now a *vector you extract from a LinkedIn headshot*, not a model you train.

### 2. Talking-head, reenactment and lip-sync

These animate a *still image* (or re-drive an existing video) so a person appears to speak/move — the engine behind one-way "video message" deepfakes.

- **Implicit-keypoint warping — [LivePortrait](https://github.com/KlingAIResearch/LivePortrait)** (Kling AI): extracts implicit keypoints and warps a single portrait to follow a driving video or audio. Efficient enough for near-real-time; adopted by major video platforms.
- **3DMM-coefficient — [SadTalker](https://sadtalker.github.io/)** (CVPR 2023): predicts 3D morphable-model head-pose and expression coefficients from audio (separate ExpNet/PoseVAE), then renders. Natural head motion from one photo.
- **Audio-to-lip — Wav2Lip, LatentSync, [MuseTalk](https://github.com/TMElyralab/MuseTalk)**: synchronise mouth movements to an audio track. Wav2Lip is the accuracy benchmark (but soft/blurry mouth); MuseTalk does latent-space inpainting at **30 fps+** on a datacentre GPU.
- **Diffusion talking-heads — the realism frontier:** **[EMO/EMO2](https://humanaigc.github.io/emote-portrait-alive/)** (Alibaba, ECCV 2024) turns one photo + audio into arbitrarily long, expressive talking-head video via Reference-Attention (identity) + Audio-Attention (motion); **[Hallo2](https://github.com/fudan-generative-vision/hallo2)** (Fudan, ICLR 2025) reaches 4K and tens-of-minutes duration; **[OmniHuman-1.5](https://omnihuman-lab.github.io/v1_5/)** (ByteDance) bridges a multimodal LLM with a diffusion transformer for >1-minute, multi-person, camera-moving animation; **EchoMimic** (Ant Group, AAAI 2025) and **AniPortrait** (Tencent) round out the field.

> **Confidence: High** on capabilities; these are published, peer-reviewed systems with public code. **Caveat:** the diffusion talking-heads are largely *offline* generators — superb for a pre-recorded clip, not yet for a low-latency live call.

### 3. Voice cloning

Modern voice cloning is **zero-shot**: a single model clones an unseen voice at inference time from a short reference clip, no per-voice training.

- **Neural-codec language models** — the **[VALL-E](https://arxiv.org/abs/2301.02111)** lineage (Microsoft) treats speech as next-token prediction over discrete audio-codec tokens; it demonstrated **zero-shot cloning from a 3-second clip**, and VALL-E 2 reported "human parity" in 2024.
- **Flow-matching / diffusion-transformer TTS** — **[F5-TTS](https://arxiv.org/abs/2410.06885)** (MIT-licensed) clones from **1–5 seconds** of reference audio and runs on a consumer GPU or Apple Silicon; Alibaba's **CosyVoice** uses an LLM→flow-matching→vocoder stack with cross-lingual cloning.
- **Quality/data reality:** ~3 s yields a usable clone; **~10–30 s yields one that is difficult for a human to distinguish** from the real voice. The reference audio every executive has already published (earnings calls, keynotes, podcasts) is more than enough.

> **Confidence: High.** Zero-shot voice cloning from seconds of public audio is the most reliably weaponisable deepfake capability in 2026.

### 4. Real-time / live deepfakes

"Live" deepfakes combine the above into a stream that can be fed into a video call or phone line.

- **Live face-swap** — **[Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)** and **DeepFaceLive** apply single-shot face-swap to a webcam feed and expose it as a **virtual camera** selectable in Zoom/Teams. Throughput is the binding constraint: ~**20 FPS on an RTX 4090** (not a smooth 30), less on mid-range cards, with a processing delay that must be compensated by delaying the audio. **InsightFace's `inswapper-512-live`** pushes 512×512 swaps onto an **iPhone** (iOS app, 2025) and Mac.
- **Live voice conversion** — **RVC** (training-based) plus low-latency research systems (**LLVC** at <20 ms on CPU, **Google StreamVC** at ~20 ms on mobile) make speak-as-the-target *audio* genuinely real-time on commodity hardware; converted audio is routed into call software via a virtual audio cable.
- **The honest limitation:** real-time *video* still shows the tells (profile turns, hand-over-face occlusion, fast motion, glasses glare, blink artefacts) and demands the operator improvise a live conversation while babysitting the pipeline. Real-time *audio* has essentially no such ceiling.

> **Confidence: High** that real-time audio impersonation is practical on a laptop; **Medium** that real-time interactive *video* is reliable enough to survive a sceptical, interactive corporate call today.

## The open-source toolkit

The defining feature of 2026 is that the best tools are **free, open-source, and built for non-programmers.** The expertise-heavy era ended when [DeepFaceLab was archived in November 2024](https://github.com/iperov/DeepFaceLab) — single-photo, no-training tools had made it obsolete for most uses.

**Face-swap (video):**

| Tool | What it does | Skill / interface | Notes |
|---|---|---|---|
| [**FaceFusion**](https://github.com/facefusion/facefusion) | Face swap + lip-sync + face enhancement | **Low** — WebUI, one-click [Pinokio](https://pinokio.co/) installer | The current leading tool (~25k stars), actively maintained; has content-moderation features that community forks strip |
| [**Deep-Live-Cam**](https://github.com/hacksider/Deep-Live-Cam) | **Real-time** webcam face-swap from one photo | **Low** — GUI | Feeds Zoom/Teams via virtual camera; AGPL-3.0 |
| **DeepFaceLab** *(archived)* | Train-per-identity high-fidelity swaps | **High** — CLI/notebooks | Archived 11/2024; superseded |
| **Rope / ReActor** | Roop-derived GUI swaps; ReActor is a Stable-Diffusion/ComfyUI extension | Low–Medium | **ReActor explicitly ships with no NSFW filter** and moved off GitHub to Codeberg — illustrative of the "guardrails are optional" reality of the ecosystem |

**Talking-head / lip-sync:** [LivePortrait](https://github.com/KlingAIResearch/LivePortrait), [SadTalker](https://sadtalker.github.io/), Wav2Lip, LatentSync, [MuseTalk](https://github.com/TMElyralab/MuseTalk), [Hallo2](https://github.com/fudan-generative-vision/hallo2), EchoMimic, AniPortrait, and the diffusion engines [EMO2](https://humanaigc.github.io/emote-portrait-alive/) / [OmniHuman-1.5](https://omnihuman-lab.github.io/v1_5/) (the latter two via demos/limited release).

**Voice cloning (open-source):**

| Model | Reference audio | Real-time? | Licence | Notes |
|---|---|---|---|---|
| [**F5-TTS**](https://arxiv.org/abs/2410.06885) | 1–5 s | Near | CC-BY-NC | Flow-matching DiT; consumer-GPU/Apple-Silicon friendly |
| **VALL-E 2** | 3 s | No (batch) | Research | "Human parity" (2024); codec-LM |
| **Chatterbox (Resemble)** | 5–10 s | Yes (~6× real-time) | MIT | Strong quality; *vendor-run* blind test claimed wins over ElevenLabs |
| **XTTS-v2 (Coqui)** | 6 s | Near | MPL-2.0 | 17 languages; Coqui shut down 2024, community-maintained |
| **OpenVoice v2 / Fish-Speech / CosyVoice2** | sec–tens of sec | Varies | MIT / CC-BY-NC | Multilingual cloning |
| **RVC** | (trains on target) | **Yes** | MIT | Voice *conversion* (speak→target), not TTS |
| Kokoro / MeloTTS | — | Yes | Apache/MIT | High quality but **preset voices only — no arbitrary cloning** |

**The accessibility multiplier.** Beyond the tools themselves, a thick support layer removes the remaining friction: **one-click installers** (Pinokio), **Hugging Face Spaces** and **Google Colab** notebooks (run it in a browser, no local GPU), **ComfyUI** workflows, and an enormous **YouTube/Discord** tutorial ecosystem. This is why Arup's CIO could stand up a working real-time deepfake of himself in ~45 minutes.

> **Confidence: High.** Open-source-first is not a stylistic choice here — the open tools *are* the state of the practice for misuse, precisely because they are unconstrained and free.

## The no-skill path: commercial and hyperscaler services

For an attacker unwilling to touch code at all, commercial services produce broadcast-quality voice and avatars — but, importantly, the reputable ones have **consent guardrails** that specifically frustrate non-consensual impersonation. This is a meaningful (if imperfect) friction point.

**Voice (SaaS):** **ElevenLabs** is the quality leader. Its *Instant Voice Cloning* (1–5 minutes of audio, from ~US$5/month) requires documented consent to clone someone else and bars public figures; its higher-fidelity *Professional Voice Cloning* is **locked to your own voice via a voice-captcha**. These guardrails mean the open-source models, not ElevenLabs, are the realistic tool for cloning a *non-consenting* executive.

**Video avatars (SaaS):** **HeyGen** (from ~US$24/month) and **Synthesia** (from ~US$18/month) create talking avatars. Critically, [HeyGen requires an **on-camera consent video**](https://help.heygen.com/en/articles/12092609-recording-your-consent-video) — you must record yourself speaking an authorisation statement — which blocks "clone the CEO from his YouTube keynote." Both are SOC 2-certified and aimed at legitimate corporate video.

**Hyperscaler services (consent-gated by design):**

- **Azure AI Speech — Custom Neural Voice** is **limited-access**: you must apply, and you must submit a **recorded consent statement from the voice talent** plus 300+ training sentences. (Per house style, this is the relevant Microsoft managed offering.)
- **AWS — Amazon Polly** offers high-quality TTS but **no arbitrary voice-cloning** service — there is nothing to misuse for impersonation.
- **GCP — Cloud Text-to-Speech Custom Voice** exists but is **gated behind an application/onboarding** process with consent requirements.
- **IBM / Oracle** provide neural TTS in watsonx/OCI respectively, with preset/curated voices rather than open consumer voice-cloning.

> **Confidence: High.** The reputable commercial and hyperscaler services have deliberately raised friction (consent recordings, limited access, voice-captcha). The accessibility problem lives squarely in the **unconstrained open-source ecosystem**, which is exactly why it dominates misuse.

## Hardware and cost: what it actually takes

The headline for an executive: **the hardware bar is a gaming PC, and the cost bar is dollars, not thousands.**

| Capability | Hardware | Time | Indicative cost |
|---|---|---|---|
| Voice clone (open-source, zero-shot) | Any modern laptop; a free Colab GPU | Minutes | **~Free** (or ~US$5/mo SaaS) |
| Real-time voice conversion | Laptop CPU (LLVC) / any GPU | Live | ~Free |
| One-shot face-swap photo/video | 4–8 GB GPU, or free Colab T4, or a **phone** (`inswapper-512-live`) | Seconds–minutes | ~Free |
| Real-time live face-swap into a call | RTX 3060→4090 (≈20 FPS at the top end) | Live | Cost of a gaming PC |
| Train-per-identity (DeepFaceLab) high-fidelity | 12 GB+ VRAM (RTX 3060/4090) | **12–48+ hours** | Electricity / cloud GPU rental |
| Polished avatar + voice via SaaS | None (browser) | Minutes | ~US$5–64/mo |

Two structural points:

- **The cost asymmetry is the whole story.** Analyst and vendor commentary converges on roughly **tens of dollars and ~30 seconds of public audio** to clone an executive's voice — against which organisations spend heavily on detection. The attacker's marginal cost is trivial; the defender's is not.
- **Training is no longer required for the accessible attacks.** The expensive 12–48-hour DeepFaceLab path still gives the best *bespoke* video, but the *single-photo* tools that need no training are what made the capability mass-accessible.

> **Confidence: High** on the hardware/cost tiers (corroborated across tool docs, GitHub discussions and tutorials). **Medium** on the exact "~US$50" figure for end-to-end executive voice cloning — it originates in vendor/analysis commentary, though it is consistent with known SaaS pricing.

## How easy is it really? Barriers and the steel-man

It is tempting — and partly true — to say "anyone can do this now." But honesty about the *remaining* barriers is what lets an executive respond with calibration rather than fear. The barriers differ sharply by attack mode.

**What is genuinely easy today (low/no barrier):**

- **Cloning a voice** from public audio — free, fast, runs anywhere.
- **Making a pre-recorded fake video/voicemail** — no live-conversation constraint, so quality is the only bar, and quality is high.
- **A real-time *voice* call** as the target — sub-20 ms conversion on a laptop.

**What is still hard — the steel-man for "interactive video exec fraud is not yet a commodity":**

A successful *interactive* video-call impersonation of your CFO requires the attacker to get **all** of the following right *simultaneously, in real time, in one attempt*:

1. **Source material** — enough clean, frontal footage of the target. Many executives have abundant *audio* but limited clean frontal *video* (earnings calls are often audio-led; conference footage has awkward angles/lighting).
2. **A capable rig** — a GPU holding ~20 FPS, plus virtual-camera and audio-delay plumbing that stays in sync.
3. **Live operation under pressure** — improvising a convincing conversation *while* babysitting the pipeline and avoiding the tells.
4. **Surviving challenges** — a profile turn, a hand waved across the face, fast motion, or an unexpected personal question can all break the illusion ([the Ferrari attempt died on a single personal question](https://fortune.com/2024/07/27/ferrari-deepfake-attempt-scammer-security-question-ceo-benedetto-vigna-cybersecurity-ai/)).
5. **Defender's home advantage** — staff *know* their executives' mannerisms, speech patterns and what they would plausibly ask for; the attacker must model all of it.

This is why the only **confirmed** interactive real-time video face-swaps in the wild are **romance scams** ([WIRED's "Yahoo Boys" reporting](https://www.wired.com/story/yahoo-boys-real-time-deepfake-scams/)), where the scammer *leads* a low-stakes, emotionally-charged conversation — not adversarial finance calls where the victim probes. The economic-rationality point reinforces this: why run a fragile real-time video rig when a **voice call plus a spoofed email** is cheaper, more reliable, and already nets billions in BEC fraud?

**But the barriers are falling, and the trajectory is the warning.** Single-photo tools keep cutting the source-material requirement; the diffusion talking-heads are beginning to handle occlusion that today's tools fail; and the live challenge tricks (turn sideways, wave a hand) have an estimated **1–2 year shelf life**. Plan for interactive video to migrate from "emerging" to "common" within the planning horizon.

> **Confidence: High** that voice/one-way video are commodity and interactive video is still operationally hard; **Medium** on the precise pace at which interactive video becomes commodity-grade.

## Anatomy of an executive-impersonation attack

A realistic kill-chain, assembled from the documented cases and threat reporting — useful for tabletop exercises:

1. **Target selection & reconnaissance.** Executives are *uniquely exposed*: their face and voice are abundant in earnings calls, keynotes, YouTube, LinkedIn video, podcasts and press. Rank-and-file staff are not — which is exactly why leadership is the target. The attacker also maps *who reports to whom* and *who can move money* (finance staff, EAs).
2. **Asset creation.** Scrape **~30 s+ of clean audio** → zero-shot voice clone. Optionally build a face/talking-head model from public imagery. Cost: roughly tens of dollars and an afternoon.
3. **Pretext & staging.** Craft urgency + authority + confidentiality: a "time-sensitive, confidential acquisition," a "supplier payment that must clear today," a "quick favour, don't tell anyone yet." Often a **first-contact via WhatsApp or a spoofed email**, then escalation to a voice call or video meeting for "proof."
4. **The ask.** A wire transfer to a new account, a change of supplier banking details, gift cards, or credential/MFA resets.
5. **Cash-out & laundering.** Funds split across mule accounts (Arup: 15 transfers to 5 accounts in a day) and moved quickly across borders.

The variable that determines which *tier* of deepfake is used is **how much interactivity the pretext demands**: a voicemail or one-way clip needs only the easy tier; a live, probing finance call needs the hard tier.

> **Confidence: High** on the workflow shape (consistent across FBI/FinCEN guidance and the case record).

## Documented incidents

| Case | Date | Mode | Outcome | What we actually know |
|---|---|---|---|---|
| **[Arup](https://www.weforum.org/stories/2025/02/deepfake-ai-cybercrime-arup/)** (Hong Kong) | Jan 2024 | Multi-participant deepfake **video call** | **US$25.6M lost** (15 transfers, 5 accounts) | Confirmed by HK police and Arup's CIO. **Mechanism unconfirmed** — whether the fake execs were *interactive* or *pre-rendered clips* is not stated by any primary source. CIO frames it as "technology-enhanced social engineering." |
| **[UK/German energy firm](https://www.wsj.com/articles/fraudsters-use-ai-to-mimic-ceos-voice-in-unusual-cybercrime-case-11567157402)** | Mar 2019 | **Voice clone** call | **€220k (~US$243k) lost** | The first documented AI-voice corporate fraud; reported by insurer Euler Hermes via WSJ. |
| **[WPP](https://www.theguardian.com/technology/article/2024/may/10/ceo-wpp-deepfake-scam)** | 2024 | Voice clone + YouTube footage on Teams; fake WhatsApp | **Failed** | Targeted exec recognised the attempt; CEO Mark Read warned staff. |
| **[Ferrari](https://fortune.com/2024/07/27/ferrari-deepfake-attempt-scammer-security-question-ceo-benedetto-vigna-cybersecurity-ai/)** | Jul 2024 | **Voice clone** (CEO's accent) on WhatsApp | **Failed** | Foiled by a **personal verification question** (a book recommendation) the impersonator couldn't answer. |
| **[LastPass](https://blog.lastpass.com/posts/attempted-audio-deepfake-call-targets-lastpass-employee)** | Apr 2024 | **Voice clone** of CEO via WhatsApp | **Failed** | Employee flagged the unusual channel + urgency and reported it. |
| **[Yahoo Boys](https://www.wired.com/story/yahoo-boys-real-time-deepfake-scams/)** (WIRED) | 2023–24 | **Interactive real-time** face-swap on live calls | Ongoing | The one **confirmed** interactive real-time video deepfake use — but **romance scams**, not corporate fraud. |

Two patterns stand out. First, **the confirmed corporate cases are overwhelmingly voice**, and the failures all failed for the *same* reason: a human verified through a second channel or asked something only the real person could answer. Second, there is a **survivorship bias** — the failed attempts are public precisely because they failed; successful frauds are under-reported for reputational reasons, so the Arup case is unusual in being both successful *and* disclosed.

> **Confidence: High** on the incident facts; **Medium** specifically on the Arup interaction mechanism.

## The numbers, handled honestly

The deepfake-fraud statistics that circulate are a minefield, because **most of the dramatic ones come from companies that sell detection or anti-fraud products.** We separate them by provenance.

**Government / primary (treat as reliable, but read the scope):**

- **FBI IC3 2025:** **US$893M** in losses across **22,364** AI-tagged complaints — the first dedicated AI section in IC3's history. **Crucial scope caveat:** this is *all* AI-enabled fraud, not executive deepfakes. The largest slice, **~US$632M, is investment-scam deepfakes** (fake celebrity/exec *endorsement* videos for fraudulent platforms) — *one-way* content, not live impersonation. Voice cloning has *no separate line* (it cuts across romance, elder fraud, BEC). The FBI stresses the figure is a **floor**, since victims rarely recognise AI involvement.
- **[FinCEN Alert FIN-2024-Alert004](https://www.fincen.gov/sites/default/files/shared/FinCEN-Alert-DeepFakes-Alert508FINAL.pdf)** (Nov 2024): US financial institutions' suspicious-activity reports citing deepfake media rose through 2023–24, mostly to defeat identity-verification/onboarding controls.

**Consultancy projection (model-based, not observed):**

- **Deloitte** projects US gen-AI-enabled fraud losses could reach **US$40B by 2027** (aggressive) or ~US$22B (conservative), from US$12.3B in 2023. A forecast, not a measurement.

**Vendor-reported (flagged — directional only, low baselines, self-platform data):**

- **Sumsub** +393% YoY deepfake fraud; **Pindrop** +1,300% at contact centres; **iProov** +2,665% in injection attacks; **Surfshark** US$2.19B cumulative reported losses; "humans detect high-quality deepfake video only **~24.5%** of the time." These are *unaudited*, measured only on each vendor's own platform, and percentages are amplified by very low baselines. Useful as a direction of travel; not as fact.

> **Confidence: High** that the *direction* is sharply up; **Low** on any specific vendor percentage. The single most-cited figure (FBI's $893M) is routinely mis-framed: most of it is *not* live impersonation.

## Defences and process controls

The central defensive insight: **because the proven attacks are voice and one-way video, the durable defences are process and provenance — not artefact detection.** [NSA/FBI/CISA's joint guidance](https://media.defense.gov/2023/Sep/12/2003298925/-1/-1/0/CSI-DEEPFAKE-THREATS.PDF) is explicit that passive detection is insufficient alone and must be paired with verification process and provenance.

**1. Process controls (the highest-leverage layer — defeat *any* media quality):**

- **Out-of-band callback verification.** For any payment, banking-detail change, or sensitive request, verify by calling back on a **directory-known number** (never one the caller supplied), regardless of how convincing the request looks. This is the control that defeated WPP and is the FBI/ABA and national-CERT top recommendation. The Arup loss is, at root, the *absence* of this control.
- **Dual / multi-person authorisation with cooling-off windows** on high-value or unusual transfers and supplier-bank changes. Removes the single point of human failure that urgency exploits.
- **Pre-agreed challenge phrases / "safe words"** within the executive and finance teams — an authentication factor a deepfake cannot supply. The Ferrari attempt died on exactly this.
- **A "no urgent secret transfers" cultural norm** and "voicemail/WhatsApp approvals are never valid" rule — these neutralise the urgency-and-secrecy pressure that *every* one of these scams relies on.
- **Live-call challenges (advisory, limited shelf-life):** asking the caller to **turn fully side-on**, **wave a spread hand across the face**, or perform an unexpected action breaks today's real-time face-swaps. Useful as a tripwire, but treat it as *advisory only* — its 1–2 year shelf life means it must never be the primary control.

**2. Technical detection (a routing aid, not a gate):**

Real-time detection plugins exist — **Reality Defender** (Zoom/Teams), **Pindrop** (voice), **Intel FakeCatcher** — but independent validation is sparse and these are largely vendor-claimed. Use them to *route* suspicious events to a human, **never as the sole decision-maker** for a payment. Passive detection is **structurally losing** the arms race: detectors overfit to known generators, generalise poorly to new ones, and degrade under call-codec compression. (The detection field is covered in depth in the [companion detection article](facial-recognition-deepfake-impersonation-detection.md).)

**3. Provenance and standards (durable, but asymmetric):**

- **C2PA Content Credentials (ISO 22144)** cryptographically sign and carry the edit history of media. Powerful for *your own outbound* media (press, brand assets, exec communications) so recipients can verify authenticity — but it **cannot authenticate an incoming fake**, because an attacker simply won't sign theirs. Provenance proves what's *real*, rather than chasing what's fake.
- **[NIST AI 600-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)** (Generative AI Profile, 2024) — voluntary US framework; names synthetic-content/"information integrity" risk and recommends provenance, red-teaming and field-testing.
- **[EU AI Act Article 50](https://www.freshfields.com/en/our-thinking/blogs/technology-quotient/eu-ai-act-unpacked-8-new-rules-on-deepfakes-102jb19)** — providers must machine-readably mark synthetic content; deployers must label deepfakes "clearly and distinguishably"; penalties up to €35M/7% of turnover. The honest caveat: this binds **lawful** deployers, not criminals — its value is normalising labelling and provenance, not stopping attackers directly.

## A defensive checklist for protecting an executive team

A concrete, prioritised programme for the worried executive:

1. **Institute the out-of-band callback rule today** — written policy, directory-known numbers only, for all payments and sensitive asks. (Free, highest impact.)
2. **Mandate dual authorisation + a cooling-off window** on high-value and unusual transfers and any change to supplier banking details.
3. **Agree challenge phrases / safe words** within the exec, finance and EA circles; refresh periodically.
4. **Make "urgent + secret + bypass-process" the #1 red flag** in security-awareness training; brief finance and EAs specifically on voice-clone vishing and the Arup/Ferrari cases.
5. **Reduce the attack surface where reasonable** — recognise that public exec audio/video is the raw material; you can't (and shouldn't) eliminate it, but you can avoid gratuitously publishing long, clean, frontal footage and can brief execs that their public media *is* the attacker's training set.
6. **Adopt C2PA Content Credentials for official outbound media** so stakeholders can verify genuine executive communications.
7. **Pilot real-time detection on video-conferencing as a tripwire** that escalates to human verification — not as a payment gate.
8. **Tabletop the scenario** using the [attack workflow](#anatomy-of-an-executive-impersonation-attack) above, including a live, probing challenge to a "CFO on a video call."

> **Confidence: High.** This ordering reflects the evidence that *process* — not detection — is what has actually stopped these attacks.

## Key findings and confidence

- **The threat is real but must be tiered: voice (proven/common) and one-way video (widespread) vs interactive real-time video (emerging, still hard).** *Confidence: High.*
- **Voice cloning from seconds of public audio is the most weaponisable capability, and executives are uniquely exposed by their public media footprint.** *Confidence: High.*
- **The skill barrier has collapsed** — free, one-click, single-photo open-source tools (FaceFusion, Deep-Live-Cam) plus a huge tutorial ecosystem; a working real-time deepfake in ~45 minutes. *Confidence: High.*
- **The hardware bar is a gaming PC; the cost bar is dollars** (or a free Colab/phone). *Confidence: High.*
- **Reputable commercial/hyperscaler services have consent guardrails; the accessibility problem is the unconstrained open-source ecosystem.** *Confidence: High.*
- **Process controls — not detection — are what defeat these attacks**, evidenced by every failed attempt on record. *Confidence: High.*
- **The Arup mechanism (interactive vs pre-rendered) is unconfirmed**, and confirmed interactive real-time video face-swap exists only in romance scams. *Confidence: Medium.*
- **Most circulating fraud statistics are vendor-sourced and over-stated; the FBI's $893M is mostly *not* live impersonation.** *Confidence: High on the caution; Low on any specific vendor figure.*

## Areas of uncertainty, limitations and caveats

- **The Arup case mechanism is genuinely ambiguous.** The balance of secondary commentary leans "interactive/real-time," but no primary police/forensic source confirms whether the fake executives interacted live or were pre-rendered clips played into the meeting. We have deliberately *not* asserted it as proof of commodity interactive-video fraud.
- **Vendor-statistic bias.** The threat-scale numbers disproportionately originate with detection/anti-fraud vendors; percentages from low baselines, measured on single platforms, are not market-representative. Treated as directional only.
- **Scope of the FBI figure.** The US$893M is all AI-enabled fraud; ~$632M is one-way investment-endorsement deepfakes. It should not anchor an estimate of *live executive-impersonation* losses.
- **"Technically demonstrated" ≠ "reliable in an adversarial call."** Lab/demo capability (e.g. real-time face-swap FPS, "human detection 24.5%") does not transfer cleanly to a sceptical, probing corporate video call. We have tried not to conflate the two.
- **Shelf-life of defences and of attacks both move.** Live-call challenge tricks have a ~1–2 year window; equally, provenance adoption and platform detection are improving. The arms race cuts both ways.
- **Recency/coverage.** Anchored to June 2026; the open-source tool landscape rotates quickly (tools archived/forked/renamed) even as the underlying architectures are stable. Coverage is English-language-weighted.

## References

1. [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching (ACL 2025)](https://arxiv.org/abs/2410.06885) — Credibility 0.95 — Primary paper; zero-shot voice cloning from 1–5 s.
2. [VALL-E: Neural Codec Language Models are Zero-Shot TTS (Microsoft, 2023)](https://arxiv.org/abs/2301.02111) — 0.95 — Foundational codec-LM TTS; 3-second cloning; VALL-E 2 "human parity."
3. [EMO: Emote Portrait Alive (Alibaba, ECCV 2024)](https://humanaigc.github.io/emote-portrait-alive/) — 0.9 — Audio-driven diffusion talking-head; single image + audio.
4. [OmniHuman-1.5 (ByteDance, 2025)](https://omnihuman-lab.github.io/v1_5/) — 0.9 — MLLM + diffusion-transformer avatar generation.
5. [SadTalker (CVPR 2023)](https://sadtalker.github.io/) — 0.9 — 3DMM-coefficient audio-driven talking-head.
6. [LivePortrait (Kling AI)](https://github.com/KlingAIResearch/LivePortrait) — 0.85 — Implicit-keypoint portrait animation.
7. [Hallo2 (Fudan, ICLR 2025)](https://github.com/fudan-generative-vision/hallo2) — 0.9 — 4K, long-duration portrait animation.
8. [MuseTalk (Tencent Music)](https://github.com/TMElyralab/MuseTalk) — 0.9 — Real-time latent-space lip-sync (30 fps+).
9. [FaceFusion (GitHub)](https://github.com/facefusion/facefusion) — 0.9 — Leading open-source face-swap/lip-sync platform; Pinokio one-click installer.
10. [Deep-Live-Cam (GitHub)](https://github.com/hacksider/Deep-Live-Cam) — 0.9 — Real-time single-photo webcam face-swap into video calls.
11. [Deep-Live-Cam FPS discussion (GitHub)](https://github.com/hacksider/Deep-Live-Cam/discussions/727) — 0.8 — First-hand ~20 FPS on RTX 4090.
12. [DeepFaceLab (archived Nov 2024)](https://github.com/iperov/DeepFaceLab) — 0.9 — The superseded train-per-identity tool.
13. [Pinokio one-click AI installer](https://pinokio.co/) — 0.9 — Lowers the skill barrier for non-programmers.
14. [HeyGen — Recording your Consent Video (official help)](https://help.heygen.com/en/articles/12092609-recording-your-consent-video) — 0.95 — Consent-video guardrail against third-party cloning.
15. [Azure AI Speech — Custom Neural Voice (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/custom-neural-voice) — 0.98 — Limited-access, consent-gated voice cloning.
16. [WEF — Lessons from the $25M Arup deepfake attack (CIO interview)](https://www.weforum.org/stories/2025/02/deepfake-ai-cybercrime-arup/) — 0.92 — First-party confirmation; "technology-enhanced social engineering"; ~45-min self-deepfake.
17. [Wall Street Journal — Fraudsters Use AI to Mimic CEO's Voice (2019)](https://www.wsj.com/articles/fraudsters-use-ai-to-mimic-ceos-voice-in-unusual-cybercrime-case-11567157402) — 0.95 — First documented AI-voice corporate fraud.
18. [The Guardian — WPP CEO deepfake scam (2024)](https://www.theguardian.com/technology/article/2024/may/10/ceo-wpp-deepfake-scam) — 0.92 — Failed attempt; recognised by target.
19. [Fortune — Ferrari exec foils deepfake with a personal question (2024)](https://fortune.com/2024/07/27/ferrari-deepfake-attempt-scammer-security-question-ceo-benedetto-vigna-cybersecurity-ai/) — 0.9 — Challenge-question defence in action.
20. [LastPass — Attempted Audio Deepfake Call (2024)](https://blog.lastpass.com/posts/attempted-audio-deepfake-call-targets-lastpass-employee) — 0.95 — First-party disclosure; failed voice-clone attempt.
21. [WIRED — The Real-Time Deepfake Romance Scams Have Arrived](https://www.wired.com/story/yahoo-boys-real-time-deepfake-scams/) — 0.9 — Confirmed interactive real-time face-swap (romance scams).
22. [FBI IC3 2025 Annual Report — AI-enabled fraud ($893M)](https://www.ic3.gov/AnnualReport/Reports/2025_IC3Report.pdf) — 0.9 — Primary government data; scope is all AI fraud (~$632M investment-endorsement deepfakes).
23. [FinCEN Alert FIN-2024-Alert004 — Fraud Schemes Using Deepfake Media (2024)](https://www.fincen.gov/sites/default/files/shared/FinCEN-Alert-DeepFakes-Alert508FINAL.pdf) — 0.98 — Primary US Treasury alert.
24. [Deloitte — Deepfake banking and AI fraud risk](https://www.deloitte.com/us/en/insights/industry/financial-services/deepfake-banking-fraud-risk-on-the-rise.html) — 0.8 — $40B-by-2027 projection (model-based).
25. [NSA/FBI/CISA — Contextualizing Deepfake Threats to Organizations (2023)](https://media.defense.gov/2023/Sep/12/2003298925/-1/-1/0/CSI-DEEPFAKE-THREATS.PDF) — 0.95 — Joint advisory; process + provenance over passive detection.
26. [PCMag — Spot a Real-Time Deepfake: Tell the Person to Turn Sideways](https://www.pcmag.com/news/need-to-spot-a-real-time-deepfake-tell-the-person-to-turn-sideways) — 0.8 — Metaphysic.ai demo of the profile-turn tell.
27. [Freshfields — EU AI Act: New Rules on Deepfakes (Article 50)](https://www.freshfields.com/en/our-thinking/blogs/technology-quotient/eu-ai-act-unpacked-8-new-rules-on-deepfakes-102jb19) — 0.88 — Transparency/labelling obligations.
28. [NIST AI 600-1 — Generative AI Profile (2024)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) — 0.95 — US voluntary GenAI risk framework.
29. [ACFE Fraud Magazine — The flawless fraud of real-time deepfakes (2024)](https://www.acfe.com/fraud-magazine/all-issues/issue/article?s=2024-julyaug-real-time-deepfakes) — 0.85 — Practitioner analysis; pre-recorded vs real-time distinction.
30. *Companion article:* [Facial Recognition: Deepfake & Impersonation Detection](facial-recognition-deepfake-impersonation-detection.md) — the detection-side counterpart to this article.
