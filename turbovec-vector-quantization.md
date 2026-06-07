# turbovec — TurboQuant vector quantisation for ANN search

| Field | Value |
|-------|-------|
| Created | 2026-06-08 |
| Last Updated | 2026-06-08 |
| Version | 1.0 |

---

- [What it is](#what-it-is)
- [How the algorithm works](#how-the-algorithm-works)
- [Key features](#key-features)
- [Repo layout](#repo-layout)
- [Claimed performance](#claimed-performance-100k-vectors-k64-median-of-5-runs)
- [Maturity](#maturity)
- [Build](#build)
- [References](#references)

**Source:** https://github.com/RyanCodrai/turbovec (cloned to `~/experiments/turbovec/repo/`, 2026-06-08)
**Crate:** crates.io `turbovec` v0.8.0 · **PyPI:** `turbovec` v0.7.0 · MIT licence
**Algorithm paper:** [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026)

## What it is

A **Rust vector-search index with Python bindings** implementing Google Research's **TurboQuant** algorithm — a data-oblivious quantiser for compressing embedding vectors for approximate nearest-neighbour (ANN) search. Pitch: a 10M-document corpus needing 31 GB as float32 fits in ~4 GB compressed and searches faster than FAISS.

**Target use case:** local / air-gapped RAG where memory, privacy, or latency matter — no managed service, no data leaving the machine.

## How the algorithm works

Each vector is treated as a direction on a hypersphere, then:

1. **Normalise** — strip and store the length as one float.
2. **Random rotation** — one shared orthogonal matrix makes every coordinate follow a known Beta→Gaussian distribution, independent of the input data (the "data-oblivious" property — no training/codebook fitting).
3. **TQ+ per-coordinate calibration** — fits a shift+scale per coordinate on the *first* add to correct finite-dimension drift, then frozen (no rebuilds). Recall gain up to +1.4pp @1 on the cells that drift most (e.g. GloVe at 2-bit).
4. **Lloyd-Max scalar quantisation** — optimal buckets computed from the maths, not the data (4 buckets at 2-bit, 16 at 4-bit). Distortion within ~2.7× of the Shannon distortion-rate lower bound.
5. **Bit-pack** — e.g. a 1536-dim vector goes 6,144 B → 384 B (16× at 2-bit).
6. **Length-renormalised scoring** — a per-vector scalar (adapted from RaBitQ, SIGMOD 2024) removes the inner-product bias quantisation introduces, at zero query-time cost.

**Search:** rotates the query once into the same domain and scores directly against codebook values using hand-written SIMD kernels (NEON on ARM, AVX-512BW with AVX2 fallback on x86), with nibble-split lookup tables. The x86 kernel adapts FAISS FastScan's pack layout, nibble-LUT scoring, and u16 accumulator strategy.

## Key features

- **Online ingest** — add vectors and they're indexed; no separate train phase, no parameter tuning, no rebuilds as the corpus grows.
- **Filtered / hybrid search** — pass an id allowlist (or slot bitmask); filtering happens *inside* the SIMD kernel at 32-vector block granularity, so selective filters skip work rather than over-fetch. Output length is `min(k, len(allowed))`.
- **`IdMapIndex`** — stable uint64 external ids that survive O(1) deletes.
- **Persistence** — `write()`/`load()` with versioned IO (`.tq`, `.tvim`, `.tv`).
- **Framework integrations** — drop-in vector stores for LangChain (`InMemoryVectorStore`), LlamaIndex (`SimpleVectorStore`), Haystack (`InMemoryDocumentStore`), and Agno (`LanceDb`).

## Repo layout

- **`turbovec/`** — Rust core (~4,100 LOC). Notable: `search.rs` (1,821 — SIMD scoring kernels), `lib.rs` (852 — index API), `encode.rs` (395), `id_map.rs`, `io.rs`, `codebook.rs`, `pack.rs`, `rotation.rs`. Deps: `ndarray` (+BLAS via Accelerate on macOS / OpenBLAS on Linux), `rayon`, `faer`, `statrs`, `rand`. ~15 integration test files (kernel correctness, concurrency, filtering, IO versioning, calibration, etc.).
- **`turbovec-python/`** — PyO3 / maturin bindings (377 LOC) exposing `TurboQuantIndex` / `IdMapIndex`, plus Python integration tests.
- **`benchmarks/`** — recall/speed/compression suites vs FAISS, saved JSON results, diagram generation, and a `rabitq_poc/` exploration folder.
- **`docs/`** — API reference, integration guides, benchmark SVGs.
- Workspace `Cargo.toml` with tuned release profile (`lto`, `codegen-units=1`, `opt-level=3`); x86 builds target `x86-64-v3` (AVX2 baseline, Haswell 2013+).

## Claimed performance (100K vectors, k=64, median of 5 runs)

- **Recall** vs FAISS `IndexPQ` (LUT256, nbits=8): beats it by 0.4–3.4 pts at R@1 on OpenAI d=1536/d=3072 across 2-bit and 4-bit, both converging to 1.0 by k=4. On low-dim GloVe d=200, ties at 4-bit (+0.3) and trails by 1.2 pts at 2-bit at R@1 (the harder regime where the asymptotic Beta assumption is looser).
- **Speed** vs FAISS IndexPQFastScan: 12–20% faster on ARM (Apple M3 Max) across every config; on x86 (Intel Sapphire Rapids) wins every 4-bit config by 1–6% and runs within ~1% on 2-bit ST, slightly behind (2–4%) only on 2-bit MT.

## Maturity

MIT-licensed, published on both crates.io and PyPI. Active development (release #84, CI on every PR) but flagged "Development Status :: 3 - Alpha". Supports Linux + macOS, Python 3.9–3.14. Includes a `.claude/` directory — appears developed with Claude Code; `CONTRIBUTING.md` states PRs are by invitation with CODEOWNERS gating `main`.

## Build

- **Python:** `cd turbovec-python && maturin build --release && pip install target/wheels/*.whl`
- **Rust:** `cargo build --release`

## References

- [TurboQuant paper](https://arxiv.org/abs/2504.19874) (ICLR 2026) — the algorithm implemented.
- [RaBitQ paper](https://arxiv.org/abs/2405.12497) (SIGMOD 2024) — source of the per-vector length-renormalisation correction.
- [FAISS FastScan](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)) — pack layout / nibble-LUT scoring the x86 kernel adapts.
