# Audio Generation AI: Voice, Sound, and Music Synthesis (2026)

| Field | Value |
|-------|-------|
| Created | 2026-06-08 |
| Last Updated | 2026-06-08 |
| Version | 1.0 |

---

- [Executive Summary](#executive-summary)
- [Scope and How to Read This Article](#scope-and-how-to-read-this-article)
- [The Shared Technical Foundation: Codecs, Tokenisers, Embeddings, and the LLM Connection](#the-shared-technical-foundation-codecs-tokenisers-embeddings-and-the-llm-connection)
  - [Neural Audio Codecs and Tokenisers](#neural-audio-codecs-and-tokenisers)
  - [Semantic vs Acoustic Tokens](#semantic-vs-acoustic-tokens)
  - [The Three Generation Paradigms](#the-three-generation-paradigms)
  - [Audio Embedding and Text-Conditioning Models](#audio-embedding-and-text-conditioning-models)
- [Voice Synthesis (Text-to-Speech)](#voice-synthesis-text-to-speech)
  - [Proprietary TTS Leaders](#proprietary-tts-leaders)
  - [Open-Source TTS Leaders](#open-source-tts-leaders)
  - [TTS Benchmarks and Leaderboards](#tts-benchmarks-and-leaderboards)
- [Music Synthesis](#music-synthesis)
  - [Proprietary Music Models](#proprietary-music-models)
  - [Open-Source Music Models](#open-source-music-models)
  - [The Copyright and Training-Data Situation](#the-copyright-and-training-data-situation)
- [General Sound, Sound Effects, and Foley](#general-sound-sound-effects-and-foley)
  - [Proprietary Sound and Video-to-Audio](#proprietary-sound-and-video-to-audio)
  - [Open-Source Text-to-Audio and Foley](#open-source-text-to-audio-and-foley)
  - [Evaluation Metrics for Sound Generation](#evaluation-metrics-for-sound-generation)
- [Hardware: What It Takes to Run Them](#hardware-what-it-takes-to-run-them)
- [Managed Service Equivalents (Hyperscalers)](#managed-service-equivalents-hyperscalers)
- [Decision Guide](#decision-guide)
- [Areas of Uncertainty](#areas-of-uncertainty)
- [Caveats and Limitations](#caveats-and-limitations)
- [References](#references)

---

## Executive Summary

Audio generation AI in 2026 has matured into three overlapping but distinct domains — **voice synthesis** (text-to-speech, TTS), **music synthesis** (songs and instrumentals), and **general sound synthesis** (sound effects, Foley, ambient audio) — all increasingly unified by a common technical substrate borrowed directly from large language models: **discrete audio tokens** produced by neural codecs, **transformer/diffusion generators** that predict or denoise those tokens, and **contrastive embedding models** (CLAP, MuLan, T5) that bind text prompts to audio.

The headline findings:

- **Voice (TTS)** is the most commoditised. Proprietary leaders — **ElevenLabs v3**, **Cartesia Sonic**, **Google Gemini 2.5 TTS / Chirp 3 HD**, **Hume Octave 2**, **OpenAI gpt-4o-mini-tts** — compete on expressiveness, latency (~40–200 ms time-to-first-audio) and emotion control. The open-source gap has closed to roughly **0.2 MOS** in naturalness: **Kokoro** (82M params, runs in <1 GB VRAM, Apache 2.0) is the efficiency champion; **Chatterbox**, **F5-TTS**, **Fish Speech**, **Dia2**, **Orpheus**, **XTTS-v2** and **Qwen3-TTS** cover voice cloning and expressivity. Self-hosting is **50–200× cheaper** than premium APIs at scale.
- **Music** is led commercially by **Suno (v5.5)** and **Udio**, with **Google Lyria 2**, **Stable Audio**, and **ElevenLabs Music v2**. The open side leapt forward in 2026 with **ACE-Step 1.5** (3.5B, MIT, full songs with vocals on 4 GB VRAM in seconds) and **YuE** (7B, lyrics-to-song). The domain is shadowed by **active copyright litigation** (RIAA labels vs Suno/Udio), though Warner and Universal have begun settling and licensing.
- **General sound / Foley** is led proprietarily by **ElevenLabs Sound Effects**, **Adobe Firefly** audio, and Google's **V2A** (now baked into **Veo 3.1** for native video soundtracks). Open-source SOTA includes **TangoFlux** (flow matching, 30 s in 3.7 s), **Stable Audio Open**, **AudioLDM 2**, **AudioX**, and **MMAudio** (the leading open video-to-audio/Foley model).
- **The LLM/embedding connection** is the real through-line. Neural codecs (**EnCodec**, **DAC**, **SNAC**, **Mimi**, **WavTokenizer**) turn audio into discrete tokens so a transformer can "speak audio" the way it speaks text; embedding models (**CLAP**, **MuLan**, **w2v-BERT/HuBERT**, **T5/FLAN-T5**) provide the text↔audio bridge for conditioning and the metrics (FAD, CLAP score) for evaluation. Three architectural paradigms now coexist: **autoregressive token prediction**, **latent diffusion**, and **flow matching**.

A consistent caution runs through this article: **benchmark scores are not cross-comparable** between leaderboards, several "beats the commercial models" claims are **vendor self-reported**, and many models marketed as "open" carry **non-commercial or research-only licences**. Each is flagged in context.

## Scope and How to Read This Article

This article covers **generation** — producing new audio from text, reference audio, video, or other conditioning. It is the counterpart to two companion articles that cover the *input* side of audio AI:

- [`local-audio-language-models.md`](local-audio-language-models.md) — audio *understanding* (taking audio in, reasoning about it).
- [`real-time-voice-llms.md`](real-time-voice-llms.md) — full-duplex *voice assistants* and speech-to-speech conversational pipelines.

There is deliberate overlap at the edges: a conversational speech model like Sesame CSM or Kyutai Moshi both *understands* and *generates*, and an "omni" LLM does everything. Where a model is primarily a voice assistant, this article references it but defers depth to the companion articles.

The structure is intentional. The **technical foundation** section comes first because the same codecs, tokenisers, and embedding models recur across voice, music, and sound — understanding them once explains all three domains and directly answers the "in the context of LLMs and embedding models" part of the brief. The three domain sections (voice, music, sound) follow, then consolidated **hardware** guidance and **managed-service** equivalents.

Throughout, every category presents an **open-source option** alongside the best-in-class proprietary one, and managed-service mentions are limited to the major hyperscalers (AWS, Azure, GCP, IBM, Oracle).

## The Shared Technical Foundation: Codecs, Tokenisers, Embeddings, and the LLM Connection

The single most important idea in modern audio generation is that **audio can be treated as a language**. A raw waveform is millions of continuous samples per minute — impossible for a transformer to model directly. A **neural audio codec** compresses that waveform into a short sequence of **discrete tokens** drawn from a learned codebook. Once audio is a sequence of integers, a transformer can predict the next token exactly as a language model predicts the next word — and the same architectural toolkit (attention, scaling laws, instruction tuning) transfers over. This is why TTS, music, and sound generation all increasingly look like LLM problems.

### Neural Audio Codecs and Tokenisers

A neural codec is an encoder → quantiser → decoder trained end-to-end. Almost all use **Residual Vector Quantisation (RVQ)**: the first codebook quantises the latent vector, the next codebook quantises the *residual error*, and so on. More codebooks = higher fidelity but longer token sequences (the central tension for LLM-style generation, which wants *short* sequences).

| Codec | Origin | Frame rate | Token rate (approx.) | Codebooks | Bitrate | Notable users |
|---|---|---|---|---|---|---|
| **SoundStream** | Google (2021) | 50 Hz | ~500/s | RVQ (variable) | variable | AudioLM, MusicLM (acoustic) |
| **EnCodec** | Meta (2022) | ~75 Hz | ~600/s (8 cb) | up to 8 (24 kHz) / stereo (48 kHz) | 1.5–24 kbps | MusicGen, Bark, VALL-E |
| **DAC** (Descript) | Descript (2023) | 89 Hz | ~801/s | 9 | — | high-fidelity reconstruction |
| **SNAC** | Papla/ETH (2024) | multi-scale 14/29/57/115 Hz | ~150/s (24 kHz) | 3–4 | 2.6 kbps | **Orpheus-TTS** |
| **Mimi** | Kyutai (2024) | 12 Hz | ~150/s | 32 (split RVQ) | 1.1 kbps | **Moshi, Sesame CSM** |
| **WavTokenizer** | (2024) | — | 40–75/s | 1 | very low | long-audio LMs |
| **X-Codec2** | (2025) | — | ~50/s | 1 (vocab 65 536) | — | Llasa TTS |

The 2024–2025 trend is **fewer tokens per second** — SNAC matches DAC's perceptual quality at 3–6× lower token rate by giving coarse codebooks a lower frame rate than fine ones; Mimi pushes to 12 Hz. Lower token rates make autoregressive generation faster and let a model "hear" more context within a fixed sequence length (sources: [SNAC paper](https://arxiv.org/abs/2410.14411); [codec comparison](https://aadonis-ai.github.io/notebook/neural-audio-codecs/codec-comparison/); [Mimi docs](https://huggingface.co/docs/transformers/model_doc/mimi)).

### Semantic vs Acoustic Tokens

A key distinction underpins LLM-based audio:

- **Semantic tokens** come from *self-supervised speech/audio models* (**w2v-BERT**, **HuBERT**) at low rates (~25 Hz). They capture *content and long-term structure* — what is being said or the musical phrase — but not fine audio detail.
- **Acoustic tokens** come from *codecs* (SoundStream, EnCodec, SNAC) and capture *fidelity* — timbre, speaker identity, recording texture.

The classic recipe, established by **AudioLM** (Google, 2022), is hierarchical: generate semantic tokens first for coherence, then condition acoustic-token generation on them for fidelity. **Mimi's** 2024 innovation collapses this into a *single* codec by distilling the first RVQ level to reproduce w2v-BERT features — so one tokeniser emits both semantic and acoustic streams (sources: [AudioLM](https://arxiv.org/abs/2209.03143); [Moshi/Mimi](https://arxiv.org/html/2410.00037v2)).

### The Three Generation Paradigms

By 2026 three architectures coexist, often within the same product family:

1. **Autoregressive token prediction** — a transformer predicts codec tokens one at a time, exactly like text generation. Examples: **AudioLM**, **VALL-E** (TTS as language modelling over EnCodec codes), **MusicGen**, **Bark**, **Orpheus** (a fine-tuned Llama-3.2-3B emitting SNAC tokens), **YuE**. Strengths: streaming, in-context voice cloning from a short prompt, lyric alignment. Weakness: slower (sequential).
2. **Latent diffusion** — denoise a continuous latent (then decode to waveform), conditioned on a text embedding. Examples: **AudioLDM / AudioLDM 2**, **Stable Audio (Open)**, **Tango**. Strengths: high fidelity, strong text adherence. Weakness: less natural for streaming/real-time speech.
3. **Flow matching** — learn an optimal-transport path from noise to the target representation; non-autoregressive and fast. Examples: **F5-TTS** (flow matching with a Diffusion Transformer on mel-spectrograms), **TangoFlux**. Strengths: very fast, faithful, no explicit duration model.

The boundaries blur — **ACE-Step** combines diffusion with a deep-compression autoencoder and a linear transformer; **AudioX** is a unified diffusion transformer that ingests text, video, image, *and* audio (sources: [VALL-E](https://arxiv.org/abs/2301.02111); [F5-TTS](https://arxiv.org/abs/2410.06885); [TangoFlux](https://arxiv.org/abs/2412.21037); [ACE-Step](https://arxiv.org/abs/2506.00045)).

### Audio Embedding and Text-Conditioning Models

Embedding models are the **text↔audio bridge**. They do three jobs: condition generation, enable retrieval, and provide evaluation metrics.

- **CLAP** (Contrastive Language-Audio Pretraining, LAION / Microsoft) — a dual encoder (audio: HTS-AT/CNN14/Wav2Vec2; text: RoBERTa/GPT-2) trained with InfoNCE to share a 512–1024-dim space. It conditions **AudioLDM** and provides the **CLAP score** evaluation metric. ([CLAP overview](https://www.emergentmind.com/topics/contrastive-language-audio-pretraining-clap))
- **MuLan** (Google) — joint music-text embedding used to condition **MusicLM**; at inference the text embedding substitutes for the audio embedding used in training.
- **T5 / FLAN-T5** — instruction-tuned text encoders. **Tango** showed FLAN-T5 conditioning beats a non-instruction-tuned encoder; **Stable Audio**, **MusicGen**, and **AudioLDM 2** use T5/FLAN-T5 (AudioLDM 2 combines CLAP *and* FLAN-T5). ([Tango](https://arxiv.org/abs/2304.13731); [AudioLDM 2](https://huggingface.co/docs/diffusers/en/api/pipelines/audioldm2))
- **w2v-BERT / HuBERT** — self-supervised speech encoders supplying *semantic tokens* (above).
- **MERT** — music-understanding embedding model (analysis/retrieval/eval); **AudioMAE** — masked-autoencoder audio features used inside AudioLDM 2; **Whisper encoder** — sometimes repurposed as an audio embedding. (Coverage of these three was thinner in this review — see [Areas of Uncertainty](#areas-of-uncertainty).)

## Voice Synthesis (Text-to-Speech)

TTS is the most mature and competitive audio-generation market. The differentiators in 2026 are **expressiveness/emotion control**, **latency** (time-to-first-audio, TTFA), **zero-shot voice cloning** from a few seconds of reference, and **language coverage**. Raw naturalness is largely solved at the top — the open-source-to-proprietary gap has narrowed to roughly 0.2 MOS.

### Proprietary TTS Leaders

| Model | Vendor | Stand-out strength | Languages | Latency (TTFA) | Voice cloning | Indicative price (per 1M chars) |
|---|---|---|---|---|---|---|
| **Eleven v3** | ElevenLabs | Most expressive; huge voice library | 70+ | ~75 ms (Flash v2.5) | from 3 min audio | ~$100 (v3); ~$50 (Flash) |
| **Sonic 4** | Cartesia | Fastest; State-Space-Model arch; on-prem | 40+ | **~40 ms** | 3 s instant | ~$15 |
| **Gemini 2.5 TTS / Chirp 3 HD** | Google | NL-prompt style/emotion control; 30 speakers, 80+ locales | 50+ | low | via GCP | ~$30 (Chirp 3 HD); $4 WaveNet |
| **Octave 2** | Hume | Strongest emotional fidelity (EVI) | 30+ | low | yes | ~$50 |
| **gpt-4o-mini-tts / gpt-realtime** | OpenAI | Instructable persona; GPT-ecosystem | multi | realtime | no (standard) | ~$9 (mini) to ~$160 (realtime) |
| **Aura-2** | Deepgram | Enterprise on-prem/VPC; flat-rate | EN-centric | <200 ms (→~90 ms) | no (standard) | ~$30 |
| **Dialog 3.0** | PlayAI (PlayHT) | Conversational; IVR/audiobooks | multi | low | instant | ~$39 |
| **Polly Generative / Azure Neural HD** | AWS / Microsoft | Deep cloud integration; on-prem containers | many | low | Custom Neural Voice | ~$16–30 |

On the crowdsourced **Artificial Analysis Speech Arena** (blind A/B votes, ~1000-centred Elo, June 2026), the top entries were Inworld's *Fun-Realtime-TTS*, *Gemini 3.1 Flash TTS*, *xAI TTS*, and *Cartesia Sonic 3.5* — i.e. the proprietary realtime models cluster at the top. ([Artificial Analysis](https://artificialanalysis.ai/text-to-speech/leaderboard); [provider comparison](https://futureagi.com/blog/best-text-to-speech-providers-2026/); [pricing](https://pintoedai.com/tools/ai-tts-api-pricing))

### Open-Source TTS Leaders

The open field is rich and moves monthly. **The licence matters as much as the quality** — several popular "open" models are non-commercial. Truly permissive (Apache 2.0 / MIT) models are marked ✅.

| Model | Params | Licence | Commercial? | Voice cloning | Languages | VRAM / footprint | Notes |
|---|---|---|---|---|---|---|---|
| **Kokoro** | 82M | Apache 2.0 | ✅ | No (54 presets) | ~9 (EN-centric) | <1–3 GB; CPU-viable | Efficiency king; ~200× realtime on RTX 4090; UTMOS ~4.48 |
| **Chatterbox / Turbo** | 350M | MIT | ✅ | Yes (5–10 s) | EN (multiling. variant) | 4–8 GB | Emotion control, paralinguistic tags; output watermarked; strong blind-test results |
| **F5-TTS** | ~330M | CC-BY-NC 4.0 | ⚠️ NC | Yes (3–15 s) | multi | ~4 GB | Flow-matching; leading zero-shot cloning; commercial exception may apply |
| **Fish Speech / OpenAudio** | ~500M (open) | Apache 2.0 (base) | ✅ (base) | Yes (10–30 s) | 80+ | ~4 GB | S2 Pro (4.4B) is hosted/paid-commercial |
| **XTTS-v2** | ~467M | Coqui CPML | ⚠️ NC | Yes (6 s) | 17 | ~4 GB | Most-downloaded; non-commercial licence |
| **Dia2** | 1–2B | Apache 2.0 | ✅ | Yes | EN | ~5 GB | Dialogue-first, multi-speaker `[S1]/[S2]`, streaming |
| **Orpheus-TTS** | ~3B | Apache 2.0 | ✅ | Yes | EN | ~8–10 GB (est.) | Llama-3.2-3B + SNAC; very expressive; ~200 ms streaming |
| **Qwen3-TTS** | 600M | Apache 2.0 | ✅ | Yes (3 s) | 10 | ~4 GB | 97 ms streaming |
| **Sesame CSM-1B** | 1B | Apache 2.0 | ✅ | context-based | EN | ~4–6 GB | Conversational; Llama backbone + Mimi |
| **VibeVoice** | 1.5B (0.5B RT) | research | ❌ | — | EN/ZH | ~6 GB | Up to 90 min multi-speaker; research-only |
| **Piper** | 6–60M | MIT | ✅ | No | 30+ | <100 MB; CPU | Edge/Raspberry-Pi; real-time on CPU |
| **Parler-TTS** | 880M | Apache 2.0 | ✅ | No (text-described voice) | EN | ~4 GB | Describe the voice in words |
| **StyleTTS 2 / OpenVoice v2** | ~150M / — | MIT | ✅ | Yes (OpenVoice) | multi (OpenVoice) | low | OpenVoice strong multilingual cloning |

**Practical reading:** for permissive commercial use with cloning, **Chatterbox**, **Fish Speech (base)**, **Dia2**, **Orpheus**, **Qwen3-TTS**, and **OpenVoice v2** are the safe picks; **Kokoro** and **Piper** win on pure efficiency (no cloning); **F5-TTS** and **XTTS-v2** are excellent but **non-commercial** by default (sources: [open-source TTS comparison](https://www.tryspeakeasy.io/blog/open-source-text-to-speech-2026); [open-weight survey](https://presenc.ai/research/best-open-weight-text-to-speech-models-2026); [CodeSOTA](https://www.codesota.com/guides/tts-models)).

### TTS Benchmarks and Leaderboards

Two families of benchmark dominate, and **their scores are not comparable to each other**:

- **Artificial Analysis Speech Arena** — production APIs, blind human preference, ~1000-centred Elo, 74+ models.
- **Hugging Face TTS Arena v2** — community models + APIs, same blind A/B method.
- **CodeSOTA** — an independent registry using ~1500-centred Elo; useful for relative ranking but with **low vote counts** (wide confidence intervals). It places Chatterbox Turbo and ElevenLabs v3 at the top of its vendor track and Kokoro/XTTS at the top of its open-weight track.

All of these measure **perceived quality/naturalness, not word accuracy**, and MOS figures from different papers are not directly comparable. Treat any single ranking as indicative, not definitive. ([benchmark overview](https://www.marktechpost.com/2026/05/30/best-text-to-speech-tts-models-in-2026-a-benchmark-based-comparison/))

## Music Synthesis

Music generation is the most commercially visible and the most legally contested domain. The state of the art produces **full songs with vocals and lyrics**, multi-minute structure, stems for editing, and increasingly fine control. Unlike TTS, **there is no widely-accepted LMArena-style public leaderboard** for music quality — rankings rest on community blind tests and each paper's own FAD/CLAP metrics.

### Proprietary Music Models

| Model | Vendor | Vocals + lyrics | Length / quality | Editing & control | Commercial use | Notes |
|---|---|---|---|---|---|---|
| **Suno v5.5** | Suno | ✅ (incl. voice cloning) | up to ~8 min, 44.1 kHz | Studio DAW, up to 12-stem separation, MIDI export, personas | Yes on paid tiers | Market leader: ~2M paid subs, ~$300M ARR; API via Enterprise |
| **Udio v1.5** | Udio | ✅ | 48 kHz stereo, extendable to ~15 min | Inpainting, Sessions timeline | Yes (UMG settlement adds download rules) | Strong instrumental fidelity; ex-DeepMind founders |
| **ElevenLabs Music v2** | ElevenLabs | ✅ multilingual | section-by-section, long-form | Inpainting, mid-song genre transitions, embedded SFX | Yes (licensed-data trained) | Launched May 2026; part of voice ecosystem; API price cut ~50% |
| **Lyria 2** | Google DeepMind | ✅ | multi-minute | Music AI Sandbox; YouTube Dream Track | Enterprise/limited | Powers Google/YouTube music features; not a direct consumer rival |
| **Stable Audio 2.5 / 3.0** | Stability AI | ❌ (instrumental/sound design) | up to ~3–6 min, 44.1 kHz stereo | per-tier commercial licence | Yes (cleanest legal posture) | Trained on licensed AudioSparx data; 3.0 ships open weights |
| **AIVA** | AIVA | limited | orchestral/cinematic | MIDI export | Full ownership on Pro | Niche: classical/score composition |

Suno raised a $250M Series C (Menlo Ventures + Nvidia) at a ~$2.45B valuation in late 2025; the generative-music market is estimated at roughly **$2B in 2026** (single-source figure — treat as indicative). ([Suno vs Udio](https://neuronad.com/suno-vs-udio/); [2026 music-AI comparison](https://gudz.ai/posts/ai-music-generation-2026); [ElevenLabs Music v2](https://www.buildfastwithai.com/blogs/elevenlabs-music-v2-review-2026); [Lyria/Music AI Sandbox](https://deepmind.google/blog/music-ai-sandbox-now-with-new-features-and-broader-access/))

### Open-Source Music Models

| Model | Params | Licence | Full song + vocals | Length | Hardware / speed | Notes |
|---|---|---|---|---|---|---|
| **ACE-Step 1.5** | 3.5B | **MIT** ✅ | ✅ | full songs | **4 GB VRAM**; <2 s/song on A100, <10 s on RTX 3090 | Diffusion + deep-compression AE + linear transformer; LoRA fine-tuning; the standout open release of 2026 |
| **YuE** | 7B | open | ✅ (lyrics-to-song) | up to 5 min | ~24 GB+ VRAM (LLM-based, slower) | Autoregressive foundation model; strong lyric alignment |
| **MusicGen** (AudioCraft) | 300M–1.5B | code MIT / weights **CC-BY-NC** ⚠️ | ❌ (instrumental) | ~30 s typical | 8–16 GB | 2023-era; still useful for instrumentals; non-commercial weights |
| **Stable Audio Open 1.0** | 1B | Stability Community ⚠️ | ❌ (instrumental/sound) | up to 47 s, 44.1 kHz stereo | 8–16 GB | Trained on cleared CC data; commercial needs separate licence |

For an open, commercially-usable, full-song-with-vocals model that runs on modest hardware, **ACE-Step 1.5 (MIT)** is the clear 2026 recommendation; **YuE** is the choice when lyric alignment and length matter and a 24 GB+ GPU is available. Note ACE-Step's "quality beyond most commercial models" claim is from its own paper/community sources and is **not independently verified** (sources: [ACE-Step repo](https://github.com/ace-step/ACE-Step); [ACE-Step 1.5](https://ace-step.github.io/ace-step-v1.5.github.io/); [YuE](https://openreview.net/forum?id=hZy6YG2Ij8); [GPU deployment guide](https://www.spheron.network/blog/deploy-open-source-ai-music-generation-gpu-cloud-2026/)).

### The Copyright and Training-Data Situation

Music generation is uniquely shadowed by litigation, and this materially affects which tools are safe for commercial work:

- **June 2024:** the RIAA (on behalf of Universal, Sony, Warner) sued **Suno** and **Udio** for training on copyrighted recordings.
- **Late 2025:** **Warner** settled with *both* Suno and Udio (licensing partnerships). **Universal** settled with **Udio** (October 2025) and announced a jointly-licensed platform for 2026.
- **Mid-2026:** **Sony** cases remain **active against both** companies. A US fair-use ruling was expected around summer 2026, and Germany's **GEMA** had a ruling against Suno scheduled for 12 June 2026.

**Implication:** **Stable Audio** (licensed AudioSparx data), **Adobe Firefly** audio, and **ElevenLabs Music** (licensed-data trained) advertise the cleanest legal posture and explicit commercial clearance, which is why they are often preferred for client/commercial work despite Suno/Udio's stronger raw output. Open models like **ACE-Step** shift the legal responsibility to whoever trained/deployed them — the licence permits use, but training-data provenance is not always documented (sources: [lawsuit tracker](https://www.chartlex.com/blog/business/music-industry-ai-lawsuits-tracker-2026); [Sony continuing litigation](https://www.digitalmusicnews.com/2025/12/18/sony-music-udio-suno-lawsuit-updates/); [musicians' explainer](https://weraveyou.com/2026/05/suno-udio-umg-copyright-lawsuit-musicians-2026/)).

## General Sound, Sound Effects, and Foley

This domain covers **text-to-audio (TTA)** for non-speech, non-music sound — sound effects, ambient textures, Foley — plus the rapidly-growing **video-to-audio (V2A)** task of generating synchronised soundtracks for silent video. The lines blur: AudioLDM 2 generates speech, music, *and* sound; AudioX is a single model for "anything-to-audio"; ElevenLabs Music v2 embeds SFX inside songs.

### Proprietary Sound and Video-to-Audio

| Tool | Vendor | Input | Output | Control | Commercial posture |
|---|---|---|---|---|---|
| **Sound Effects (v2)** | ElevenLabs | text | up to 60 s; 4 variations; loopable | `prompt_influence`, `duration_seconds` | Royalty-free for paid subs |
| **Firefly Sound Effects** | Adobe | text + voice/audio hint | WAV (audio) / MP4 (video) | timeline placement, layering, 750-char prompt | Trained on licensed + public-domain; "commercially safe" |
| **V2A (in Veo 3.1)** | Google DeepMind | video pixels + text | native dialogue + SFX + ambient at 48 kHz | integrated into video gen | Not a standalone API — bundled in Veo |

**Google Veo 3.1** (October 2025) is notable as the first major video model with **native synchronised audio** — dialogue, sound effects, and ambience generated in one pass with the video (8-second clips, chainable to ~140 s, $0.10–0.60/second). The underlying **V2A** diffusion technology is not sold separately ([Veo overview](https://deeka.ai/blog/veo-ai-in-2026-google-s-video-generator-explained-features-pricing-how-to-use-it); [DeepMind V2A](https://deepmind.google/blog/generating-audio-for-video/); [ElevenLabs SFX docs](https://elevenlabs.io/docs/overview/capabilities/sound-effects); [Adobe Firefly SFX](https://www.adobe.com/products/firefly/features/sound-effect-generator.html)).

### Open-Source Text-to-Audio and Foley

| Model | Params | Licence | Task | Length / quality | Hardware / speed | Notes |
|---|---|---|---|---|---|---|
| **TangoFlux** | 515M | open (code+models); paper CC-BY-SA | text→audio | up to 30 s, 44.1 kHz | 3.7 s on one A40; ~8–16 GB inference | Flow matching (FluxTransformer) + CRPO; fast & faithful; ICLR 2026 |
| **Stable Audio Open 1.0** | 1B | Stability Community ⚠️ | text→audio/sound | up to 47 s, 44.1 kHz stereo | 8–16 GB | DiT + T5; cleared training data; Small variant = 11 s |
| **AudioLDM 2** | 1.1B | open | text→audio/music/speech | variable | 8–16 GB | Latent diffusion + CLAP + FLAN-T5 + AudioMAE |
| **AudioGen** (AudioCraft) | 0.3–1.5B | code MIT / weights **CC-BY-NC** ⚠️ | text→sound | ~10 s, 16 kHz mono | 8–16 GB | 2023-era; non-commercial weights; limited recent updates |
| **AudioX** | — | open | anything→audio (text/video/image/audio) | — | GPU | Unified DiT; ICLR 2026; strong on TTA + TTM |
| **MMAudio** | from 157M | open | **video→audio / Foley** + text→audio | 8 s in ~1.23 s (smallest) | modest GPU | **Leading open V2A/Foley model**; CVPR 2025 (Sony AI/UIUC) |

For open **text-to-sound**, **TangoFlux** offers the best speed/fidelity balance and a genuinely open release; for **video-to-audio/Foley**, **MMAudio** is the open SOTA and the practical answer to Google's proprietary V2A (sources: [TangoFlux](https://arxiv.org/abs/2412.21037); [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0); [AudioLDM 2](https://github.com/haoheliu/audioldm2); [AudioX](https://arxiv.org/abs/2503.10522); [MMAudio](https://github.com/hkchengrex/MMAudio)).

### Evaluation Metrics for Sound Generation

Unlike TTS (human preference) and music (largely informal), TTA has **established objective metrics**:

- **FAD / FDopenl3** (Fréchet Audio Distance) — distance between generated and real audio feature distributions; lower is better.
- **CLAP score** — cosine similarity between generated audio and the text prompt in CLAP's shared embedding space; measures text adherence.
- **KL divergence (KLpasst)** — label-distribution match against a reference, via an audio classifier.
- **AudioCaps** is the standard benchmark dataset; **TTA-Bench** (AAAI 2026) extends evaluation to seven dimensions (functional performance, reliability, social responsibility) with 118,000+ human annotations — a move beyond pure perceptual quality (sources: [TTA-Bench](https://arxiv.org/html/2509.02398v1); [Stable Audio metrics](https://deepwiki.com/Stability-AI/stable-audio-metrics/5.1-full-evaluation-mode)).

These metrics reuse the **embedding models** from the foundation section — another illustration that the embedding layer is simultaneously the conditioning mechanism *and* the yardstick.

## Hardware: What It Takes to Run Them

A defining feature of 2026 audio generation is how *cheap* most of it has become to run locally. Audio models are far smaller than frontier LLMs — the largest open music/voice models are single-digit billions of parameters, and many useful ones are well under 1B. The table groups open models by deployment tier. VRAM figures are approximate working numbers for fp16 inference; quantisation (int8/int4) lowers them further, and the benchmark GPU for a paper (e.g. TangoFlux on a 48 GB A40) is usually far larger than the minimum needed.

| Tier | Hardware | What runs here | Examples |
|---|---|---|---|
| **CPU / edge** | Modern CPU, Raspberry Pi, <1 GB | Lightweight TTS only | **Piper**, **Kokoro** (CPU-viable), Kitten TTS |
| **Entry GPU** | 4–8 GB VRAM (GTX 1650 / RTX 3050, laptop GPUs) | Most TTS with cloning; full-song music; short SFX | **Kokoro**, **Chatterbox**, **F5-TTS**, **Fish Speech**, **Qwen3-TTS**, **Dia2**, **ACE-Step 1.5** (4 GB), **Stable Audio Open**, **TangoFlux** |
| **Mid GPU** | 12–16 GB (RTX 4070/4080, T4) | Larger TTS, AudioLDM 2, comfortable SFX/music | **Orpheus** (3B), AudioLDM 2, MMAudio, AudioX |
| **High-end consumer** | 24 GB (RTX 3090/4090) | LLM-based long-form music; batch/serving | **YuE** (7B), VibeVoice, fast batched serving of smaller models |
| **Data-centre** | 40–80 GB (A100/H100/H200) | Training, fine-tuning, high-throughput serving | any of the above at scale; Fish Audio S2 Pro |

**Cost economics.** Self-hosting an open TTS model such as Kokoro costs on the order of **~$0.50 per 1M characters** in compute versus **~$50–160 per 1M** for premium APIs — a 50–200× reduction. The break-even versus a service like ElevenLabs is roughly **5M characters/month**; below that, managed APIs are usually cheaper once engineering and ops time are counted ([open-weight economics](https://presenc.ai/research/best-open-weight-text-to-speech-models-2026)). For music and SFX, self-hosting ACE-Step or TangoFlux on a single consumer GPU is essentially free per generation after hardware, which is decisive for high-volume or privacy-sensitive workloads.

**Real-time factor (RTF).** RTF < 1 means faster than real-time. Kokoro reaches RTF ≈ 0.03 (≈30× real-time) on an RTX 4090; most 300–600M TTS models sit at RTF 0.1–0.2 on a mid GPU; ACE-Step renders a full song in under 10 s on an RTX 3090. Streaming TTS models (Qwen3-TTS ~97 ms, Orpheus ~200 ms) are fast enough for conversational use.

## Managed Service Equivalents (Hyperscalers)

For teams that prefer not to self-host, the major hyperscalers offer managed audio generation — strongest for **voice**, thinner for music/sound.

- **AWS** — **Amazon Polly** (Neural ~$16/1M chars; **Generative** voices ~$30/1M) for TTS, with Speech Marks and a 12-month free tier. Third-party voice/music/sound models (e.g. via **Amazon Bedrock** Marketplace and SageMaker JumpStart) can host open models like Stable Audio or TTS checkpoints on managed GPU.
- **Azure** — **Azure AI Speech**: Neural TTS (~$16/1M), **HD V2 Neural** (~$30/1M) with context-aware emotion, **Custom Neural Voice**, and **on-prem/Kubernetes containers** for regulated deployments (129+ voices, 54 locales). OpenAI TTS models are also available via **Azure OpenAI**.
- **GCP** — **Google Cloud Text-to-Speech**: **Chirp 3 HD** (~$30/1M, 1M free/month), classic **WaveNet** (~$4/1M), and **Gemini 2.5 TTS** (Flash/Pro) with natural-language style control; **Lyria** music via Vertex/Music AI Sandbox (enterprise/limited); **Veo** for native video-with-audio.
- **IBM** — **watsonx / Watson Text to Speech**: enterprise neural TTS with on-prem/Cloud Pak deployment; positioned for regulated/enterprise rather than cutting-edge expressivity.
- **Oracle** — **OCI Speech / AI Services**: TTS within OCI; also a popular venue for **renting raw GPU** (A10/A100/H100) to self-host any open model with full data control.

For self-hosting open models, all five clouds rent the GPUs in the [hardware tiers](#hardware-what-it-takes-to-run-them) above; the entry/mid tiers (one 8–24 GB GPU) cover the large majority of audio-generation workloads.

## Decision Guide

| If you need… | Proprietary best | Open-source best | Notes |
|---|---|---|---|
| Expressive TTS, many voices | ElevenLabs v3 | Orpheus / Chatterbox | Chatterbox (MIT) is commercially safe |
| Lowest-latency realtime voice | Cartesia Sonic 4 (~40 ms) | Qwen3-TTS / Orpheus | See [`real-time-voice-llms.md`](real-time-voice-llms.md) |
| Cheapest high-volume TTS | hyperscaler WaveNet ($4/1M) | **Kokoro** (~$0.50/1M self-hosted) | Break-even ~5M chars/month |
| Edge / offline TTS | — | **Piper** / Kokoro (CPU) | <100 MB to <1 GB |
| Voice cloning, permissive licence | ElevenLabs / Cartesia | Fish Speech (base), Dia2, OpenVoice v2 | Avoid F5-TTS/XTTS for commercial (NC) |
| Full songs with vocals | **Suno v5.5** | **ACE-Step 1.5** (MIT, 4 GB) | Suno strongest; ACE-Step best open |
| Commercially-clean music | Stable Audio / ElevenLabs Music | ACE-Step (provenance caveat) | Licensed-data training matters legally |
| Sound effects / Foley (text) | ElevenLabs SFX / Adobe Firefly | **TangoFlux** | Firefly = "commercially safe" |
| Video-to-audio / Foley | Google V2A (in Veo 3.1) | **MMAudio** | MMAudio is the open answer |
| Build your own audio LLM | — | EnCodec/SNAC/Mimi + transformer | Codec choice drives token budget |

## Areas of Uncertainty

- **Benchmark incomparability.** TTS leaderboards use different Elo scales (Artificial Analysis ~1000-centred; CodeSOTA ~1500-centred) and CodeSOTA's vote counts are low — rankings are provisional. There is **no standard public leaderboard for music generation** at all.
- **Self-reported superiority.** Claims that open models "beat" commercial ones — ACE-Step "beyond most commercial models", Chatterbox "63.75% preferred over ElevenLabs" — originate from the model makers and lack independent replication.
- **VRAM precision.** Exact minimum VRAM for Orpheus (3B), YuE (7B), and Fish Audio S2 Pro was not confirmed from primary sources; the tiers above are informed estimates.
- **Thin embedding coverage.** MERT, AudioMAE, and Whisper-encoder usage in generation/conditioning were less thoroughly sourced than CLAP/T5/w2v-BERT.
- **Fast-moving products.** ElevenLabs Music v2 (May 2026), ACE-Step 1.5 (Feb 2026), and Stable Audio 3.0 are recent enough that mature independent evaluation is limited. Pricing changes monthly.
- **Music legal trajectory.** Settlement/litigation status (especially Sony vs Udio, and the US fair-use and GEMA rulings due mid-2026) was unresolved at the time of writing and will shift which tools are "commercially safe".

## Caveats and Limitations

- **Source mix.** Primary sources (arXiv papers, official model cards, vendor docs) anchor the technical and architectural claims. Market/pricing/ranking figures lean on comparison blogs (CodeSOTA, presenc.ai, SpeakEasy, FutureAGI, gudz.ai, neuronad) that have commercial interests; these were cross-checked against primary sources where possible and should be reverified before procurement decisions.
- **"Open" is not always free-to-use.** Several widely-used models carry **non-commercial** (F5-TTS CC-BY-NC, XTTS CPML, MusicGen/AudioGen weights CC-BY-NC), **community/tiered** (Stable Audio), or **research-only** (VibeVoice) licences. Verify the exact licence — and training-data provenance — before commercial deployment.
- **Watermarking & provenance.** Some open models (Chatterbox, VibeVoice) watermark output; proprietary music/voice tools increasingly embed provenance signals. This matters for both detection and authenticity claims — see [`deepfake-creation-models-and-accessibility.md`](deepfake-creation-models-and-accessibility.md) and [`facial-recognition-deepfake-impersonation-detection.md`](facial-recognition-deepfake-impersonation-detection.md) for the misuse and detection angle of synthetic voice.
- **Recency.** All figures reflect a June 2026 snapshot. The web searches behind the SFX and codec sections hit some rate-limiting, so a few models named in the brief (Make-An-Audio, Higgs Audio, IndexTTS, MeloTTS, Zonos) are under-covered here rather than absent from the field.
- **Geographic/language coverage.** Most quality and latency claims are English-centric; multilingual performance varies and is generally weaker for low-resource languages.

## References

### Technical foundation — codecs, tokenisers, paradigms, embeddings
1. [AudioLM: a Language Modeling Approach to Audio Generation](https://arxiv.org/abs/2209.03143) — Google, IEEE TASLP 2023. Credibility 0.95.
2. [VALL-E: Neural Codec Language Models are Zero-Shot TTS](https://arxiv.org/abs/2301.02111) — Microsoft. Credibility 0.95.
3. [SNAC: Multi-Scale Neural Audio Codec](https://arxiv.org/abs/2410.14411) — NeurIPS 2024 Workshop. Credibility 0.90.
4. [Moshi / Mimi: speech-text foundation model for real-time dialogue](https://arxiv.org/html/2410.00037v2) — Kyutai. Credibility 0.92.
5. [EnCodec repository](https://github.com/facebookresearch/encodec) — Meta. Credibility 0.95.
6. [Mimi model documentation](https://huggingface.co/docs/transformers/model_doc/mimi) — Hugging Face. Credibility 0.92.
7. [Neural audio codec comparison (DAC/SNAC/WavTokenizer/X-Codec2)](https://aadonis-ai.github.io/notebook/neural-audio-codecs/codec-comparison/) — researcher blog. Credibility 0.75.
8. [F5-TTS: Flow Matching with Diffusion Transformer](https://arxiv.org/abs/2410.06885) — ACL 2025. Credibility 0.90.
9. [Tango: Text-to-Audio with Instruction-Tuned LLM + Latent Diffusion](https://arxiv.org/abs/2304.13731). Credibility 0.90.
10. [AudioLDM 2 (Diffusers docs)](https://huggingface.co/docs/diffusers/en/api/pipelines/audioldm2) — Hugging Face. Credibility 0.92.
11. [CLAP overview](https://www.emergentmind.com/topics/contrastive-language-audio-pretraining-clap). Credibility 0.80.
12. [MusicLM architecture analysis](https://zhangtemplar.github.io/musiclm/). Credibility 0.80.
13. [Bark architecture](https://openlaboratory.com/models/bark/). Credibility 0.75.

### Voice synthesis (TTS)
14. [ElevenLabs models documentation](https://elevenlabs.io/docs/overview/models) — official. Credibility 0.95.
15. [Google Cloud TTS release notes (Gemini 2.5 TTS, Chirp 3 HD)](https://docs.cloud.google.com/text-to-speech/docs/release-notes) — official. Credibility 0.95.
16. [Sesame CSM-1B model card](https://huggingface.co/sesame/csm-1b) — official. Credibility 0.92.
17. [Artificial Analysis TTS Leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard) — independent. Credibility 0.88.
18. [CodeSOTA: Best TTS Models 2026](https://www.codesota.com/guides/tts-models) — independent registry. Credibility 0.85.
19. [AI TTS API pricing tracker](https://pintoedai.com/tools/ai-tts-api-pricing). Credibility 0.82.
20. [Best TTS APIs 2026 (provider comparison)](https://futureagi.com/blog/best-text-to-speech-providers-2026/). Credibility 0.75.
21. [Best open-source TTS 2026](https://www.tryspeakeasy.io/blog/open-source-text-to-speech-2026). Credibility 0.72.
22. [Best open-weight TTS 2026](https://presenc.ai/research/best-open-weight-text-to-speech-models-2026). Credibility 0.75.
23. [TTS benchmark comparison](https://www.marktechpost.com/2026/05/30/best-text-to-speech-tts-models-in-2026-a-benchmark-based-comparison/) — MarkTechPost. Credibility 0.78.

### Music synthesis
24. [ACE-Step: A Step Towards Music Generation Foundation Model](https://arxiv.org/abs/2506.00045) — arXiv. Credibility 0.90.
25. [ACE-Step 1.5 project page](https://ace-step.github.io/ace-step-v1.5.github.io/). Credibility 0.85.
26. [ACE-Step repository](https://github.com/ace-step/ACE-Step). Credibility 0.95.
27. [YuE: Scaling Open Foundation Models for Long-Form Music Generation](https://openreview.net/forum?id=hZy6YG2Ij8) — peer-reviewed. Credibility 0.90.
28. [GPU deployment guide: YuE/ACE-Step/MusicGen/Stable Audio Open](https://www.spheron.network/blog/deploy-open-source-ai-music-generation-gpu-cloud-2026/). Credibility 0.70.
29. [Suno vs Udio (2026)](https://neuronad.com/suno-vs-udio/). Credibility 0.75.
30. [AI music generation 2026 comparison](https://gudz.ai/posts/ai-music-generation-2026). Credibility 0.65.
31. [Suno pricing](https://suno.com/pricing) — official. Credibility 0.95.
32. [ElevenLabs Music v2 review](https://www.buildfastwithai.com/blogs/elevenlabs-music-v2-review-2026). Credibility 0.70.
33. [Music AI Sandbox / Lyria](https://deepmind.google/blog/music-ai-sandbox-now-with-new-features-and-broader-access/) — Google DeepMind. Credibility 0.90.
34. [Music industry AI lawsuits tracker 2026](https://www.chartlex.com/blog/business/music-industry-ai-lawsuits-tracker-2026). Credibility 0.75.
35. [Sony Music continuing litigation](https://www.digitalmusicnews.com/2025/12/18/sony-music-udio-suno-lawsuit-updates/) — Digital Music News. Credibility 0.85.
36. [Suno/Udio copyright explainer](https://weraveyou.com/2026/05/suno-udio-umg-copyright-lawsuit-musicians-2026/). Credibility 0.80.

### General sound, SFX, Foley
37. [TangoFlux: Fast and Faithful Text-to-Audio with Flow Matching + CRPO](https://arxiv.org/abs/2412.21037) — ICLR 2026. Credibility 0.95.
38. [TangoFlux repository](https://github.com/declare-lab/Tangoflux). Credibility 0.95.
39. [MMAudio: Multimodal Joint Training for Video-to-Audio](https://github.com/hkchengrex/MMAudio) — CVPR 2025. Credibility 0.95.
40. [Stable Audio Open 1.0 model card](https://huggingface.co/stabilityai/stable-audio-open-1.0) — official. Credibility 0.95.
41. [Stable Audio 3.0 announcement](https://stability.ai/news-updates/meet-stable-audio-3-the-model-family-built-for-artistic-experimentation-with-open-weight-models) — official. Credibility 0.95.
42. [AudioLDM 2 repository](https://github.com/haoheliu/audioldm2). Credibility 0.95.
43. [AudioX: A Unified Framework for Anything-to-Audio](https://arxiv.org/abs/2503.10522) — ICLR 2026. Credibility 0.92.
44. [TTA-Bench: Comprehensive Benchmark for Text-to-Audio](https://arxiv.org/html/2509.02398v1) — AAAI 2026. Credibility 0.92.
45. [Stable Audio metrics (FAD/CLAP/KL)](https://deepwiki.com/Stability-AI/stable-audio-metrics/5.1-full-evaluation-mode). Credibility 0.90.
46. [ElevenLabs Sound Effects documentation](https://elevenlabs.io/docs/overview/capabilities/sound-effects) — official. Credibility 0.95.
47. [Adobe Firefly sound effect generator](https://www.adobe.com/products/firefly/features/sound-effect-generator.html) — official. Credibility 0.95.
48. [Google DeepMind: Generating audio for video (V2A)](https://deepmind.google/blog/generating-audio-for-video/) — official. Credibility 0.95.
49. [Google Veo 3.1 overview](https://deeka.ai/blog/veo-ai-in-2026-google-s-video-generator-explained-features-pricing-how-to-use-it). Credibility 0.75.
