#!/usr/bin/env python3
"""Plot the Gemma 4 family benchmark comparison (official HF model-card scores).

Grouped bar chart: one group per benchmark, one bar per family member, with the
12B highlighted. Matches the repo's dark chart style.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA_FILE = Path(__file__).resolve().parent.parent / ".data" / "gemma4_family_benchmarks.json"
IMG_DIR = Path(__file__).resolve().parent.parent / "images"


def main():
    with open(DATA_FILE) as f:
        data = json.load(f)

    models = data["models"]
    benchmarks = data["benchmarks"]
    names = [m["name"] for m in models]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    n_models = len(models)
    n_bench = len(benchmarks)
    group_width = 0.82
    bar_w = group_width / n_models
    x = np.arange(n_bench)

    for i, m in enumerate(models):
        offsets = x - group_width / 2 + bar_w * (i + 0.5)
        scores = [b["scores"].get(m["name"]) for b in benchmarks]
        is_hl = m.get("highlight")
        is_ref = m.get("reference")
        bars = ax.bar(
            offsets, scores, bar_w * 0.92,
            color=m["colour"],
            edgecolor="#e6edf3" if is_hl else m["colour"],
            linewidth=2.0 if is_hl else 0.6,
            alpha=0.55 if is_ref else 0.95,
            hatch="//" if is_ref else None,
            zorder=3, label=m["short"],
        )
        for off, s in zip(offsets, scores):
            if s is None:
                continue
            ax.text(off, s + 0.8, f"{s:.0f}", ha="center", va="bottom",
                    fontsize=7.0, color="#e6edf3" if is_hl else "#8b949e",
                    fontweight="bold" if is_hl else "normal", rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([b["name"] for b in benchmarks], fontsize=11, color="#e6edf3")
    ax.set_ylabel("Score (higher is better)", fontsize=12, color="#8b949e")
    ax.set_ylim(0, 100)

    today = datetime.now().strftime("%-d %B %Y")
    ax.set_title(
        f"Gemma 4 Family — Official Benchmark Comparison (Google model card) [{today}]",
        fontsize=16, color="#e6edf3", fontweight="bold", pad=18,
    )

    ax.grid(True, axis="y", alpha=0.15, color="#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#30363d")
    ax.spines["bottom"].set_color("#30363d")
    ax.tick_params(colors="#8b949e", which="both", labelsize=10)

    handles = [mpatches.Patch(facecolor=m["colour"],
                              edgecolor="#e6edf3" if m.get("highlight") else m["colour"],
                              linewidth=2.0 if m.get("highlight") else 0.6,
                              hatch="//" if m.get("reference") else None,
                              label=(m["short"] + "  (12B — new)" if m.get("highlight") else m["short"]))
               for m in models]
    ax.legend(handles=handles, loc="upper left", fontsize=10, ncol=2,
              framealpha=0.3, edgecolor="#30363d", facecolor="#161b22",
              labelcolor="#e6edf3", title="Gemma 4 sizes (hatched = Gemma 3 ref.)",
              title_fontsize=9)

    fig.text(
        0.5, 0.01,
        "Source: google/gemma-4-12B Hugging Face model card  |  Vendor-reported (Google) figures, compared within-family  |  "
        "12B released ~3 Jun 2026; others Apr 2026  |  Gemma 4 12B is not yet on LMArena or public tool-calling boards",
        ha="center", fontsize=8, color="#6e7681",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    IMG_DIR.mkdir(exist_ok=True)
    output_path = IMG_DIR / "gemma4-family-benchmarks.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
