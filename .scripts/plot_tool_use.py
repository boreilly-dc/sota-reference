#!/usr/bin/env python3
"""Plot tool use benchmark scores as a 2x2 subplot grid (BFCL V4 + Tau2 domains)."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA_FILE = Path(__file__).resolve().parent.parent / ".data" / "tool_use_benchmarks.json"
IMG_DIR = Path(__file__).resolve().parent.parent / "images"


def plot_subplot(ax, benchmark, labs, *, title, x_min=None):
    """Render a single horizontal bar chart for one benchmark."""
    models = benchmark["models"]
    names = [m["name"] for m in models]
    scores = [m["score"] for m in models]
    colours = [labs[m["lab"]]["colour"] for m in models]
    edge_colours = ["#FFD700" if m.get("open_source") else c
                    for m, c in zip(models, colours)]
    hatches = ["///" if m.get("open_source") else "" for m in models]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, scores, color=colours, edgecolor=edge_colours,
                   linewidth=1.5, height=0.6, zorder=3)

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9, color="#e6edf3")
    ax.invert_yaxis()

    for i, (score, name) in enumerate(zip(scores, names)):
        ax.text(score + 0.5, i, f"{score:.1f}%", va="center", fontsize=8,
                color="#e6edf3", fontweight="bold")

    if x_min is not None:
        ax.set_xlim(left=x_min)
    ax.set_xlabel("Accuracy (%)", fontsize=9, color="#8b949e")
    ax.set_title(title, fontsize=11, color="#e6edf3", fontweight="bold", pad=10)

    ax.set_facecolor("#0d1117")
    ax.grid(True, axis="x", alpha=0.15, color="#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#30363d")
    ax.spines["bottom"].set_color("#30363d")
    ax.tick_params(colors="#8b949e", which="both", labelsize=8)


def main():
    with open(DATA_FILE) as f:
        data = json.load(f)

    labs = data["labs"]
    benchmarks = [
        (data["bfcl_v4"], "BFCL V4 — Overall Accuracy", 45),
        (data["tau2_airline"], "Tau²-Bench — Airline", 50),
        (data["tau2_retail"], "Tau²-Bench — Retail", 70),
        (data["tau2_telecom"], "Tau²-Bench — Telecom", 80),
    ]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")

    today = datetime.now().strftime("%-d %B %Y")
    fig.suptitle(
        f"Tool Use Benchmark Scores [{today}]",
        fontsize=18, color="#e6edf3", fontweight="bold", y=0.97,
    )

    for ax, (bench, title, x_min) in zip(axes.flat, benchmarks):
        plot_subplot(ax, bench, labs, title=title, x_min=x_min)

    # Build legend from all labs present
    used_labs = set()
    for bench, _, _ in benchmarks:
        for m in bench["models"]:
            used_labs.add(m["lab"])

    legend_handles = []
    for lab_name in sorted(used_labs):
        colour = labs[lab_name]["colour"]
        legend_handles.append(
            mpatches.Patch(facecolor=colour, edgecolor=colour, label=lab_name))
    legend_handles.append(
        mpatches.Patch(facecolor="#555555", edgecolor="#FFD700",
                       hatch="///", label="Open source"))

    fig.legend(
        handles=legend_handles, loc="lower center", ncol=min(len(legend_handles), 6),
        fontsize=9, framealpha=0.3, edgecolor="#30363d", facecolor="#161b22",
        labelcolor="#e6edf3", bbox_to_anchor=(0.5, 0.01),
    )

    fig.text(
        0.5, 0.04,
        "Sources: BFCL V4 (gorilla.cs.berkeley.edu, Apr 2026), BenchLM, "
        "Tau²-bench (Sierra Research, airank.dev)  |  Hatched = open source",
        ha="center", fontsize=8, color="#6e7681",
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    IMG_DIR.mkdir(exist_ok=True)
    output_path = IMG_DIR / "tool-use-benchmarks.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
