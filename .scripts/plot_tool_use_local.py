#!/usr/bin/env python3
"""Plot tool use benchmark scores for local models (<= 30B params), 2x2 subplots."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA_FILE = Path(__file__).resolve().parent.parent / ".data" / "tool_use_local_models.json"
IMG_DIR = Path(__file__).resolve().parent.parent / "images"


def plot_bar_subplot(ax, benchmark, labs, *, title, x_min=None):
    """Render a horizontal bar chart with a frontier reference line."""
    models = benchmark["models"]
    names = [f"{m['name']}  ({m['params']})" for m in models]
    scores = [m["score"] for m in models]
    colours = [labs.get(m["lab"], {"colour": "#888888"})["colour"] for m in models]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, scores, color=colours, edgecolor=[c for c in colours],
                   linewidth=1.2, height=0.55, zorder=3, alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8.5, color="#e6edf3")
    ax.invert_yaxis()

    for i, score in enumerate(scores):
        ax.text(score + 0.4, i, f"{score:.1f}%", va="center", fontsize=8,
                color="#e6edf3", fontweight="bold")

    # Frontier reference line
    if "frontier_ref" in benchmark:
        ref = benchmark["frontier_ref"]
        ax.axvline(x=ref["score"], color="#FF4444", linestyle="--", linewidth=1.5,
                   alpha=0.7, zorder=4)
        ax.text(ref["score"], -0.6, f"  {ref['name']} ({ref['score']}%)",
                fontsize=7.5, color="#FF4444", alpha=0.9, va="bottom",
                fontweight="bold")

    if x_min is not None:
        ax.set_xlim(left=x_min)
    ax.set_xlabel("Score (%)", fontsize=9, color="#8b949e")
    ax.set_title(title, fontsize=11, color="#e6edf3", fontweight="bold", pad=10)

    ax.set_facecolor("#0d1117")
    ax.grid(True, axis="x", alpha=0.15, color="#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#30363d")
    ax.spines["bottom"].set_color("#30363d")
    ax.tick_params(colors="#8b949e", which="both", labelsize=8)


def plot_gap_subplot(ax, comparison, labs):
    """Render the local-vs-frontier gap comparison as a grouped bar chart."""
    benchmarks = comparison["benchmarks"]
    names = [b["name"] for b in benchmarks]
    local_scores = [b["best_local"]["score"] for b in benchmarks]
    frontier_scores = [b["best_frontier"]["score"] for b in benchmarks]
    gaps = [b["gap"] for b in benchmarks]

    x = np.arange(len(names))
    width = 0.35

    bars_frontier = ax.bar(x - width / 2, frontier_scores, width,
                           label="Best Frontier", color="#FF4444", alpha=0.85,
                           edgecolor="#FF4444", zorder=3)
    bars_local = ax.bar(x + width / 2, local_scores, width,
                        label="Best Local (≤30B)", color="#00C49A", alpha=0.85,
                        edgecolor="#00C49A", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, color="#e6edf3")
    ax.set_ylabel("Score (%)", fontsize=9, color="#8b949e")
    ax.set_title("Best Local vs Frontier — Tool Use Gap",
                 fontsize=11, color="#e6edf3", fontweight="bold", pad=10)

    # Annotate gaps
    for i, (ls, fs, gap) in enumerate(zip(local_scores, frontier_scores, gaps)):
        mid = (ls + fs) / 2
        ax.annotate(f"Δ{gap:.1f}",
                    xy=(i, mid), fontsize=8.5, color="#FFD700",
                    ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#161b22",
                              edgecolor="#FFD700", alpha=0.8))

    # Annotate model names on bars
    for i, b in enumerate(benchmarks):
        ax.text(i + width / 2, local_scores[i] + 0.8,
                b["best_local"]["name"], fontsize=6.5, color="#00C49A",
                ha="center", va="bottom", rotation=0)

    ax.legend(fontsize=8, loc="lower right", framealpha=0.3,
              edgecolor="#30363d", facecolor="#161b22", labelcolor="#e6edf3")

    ax.set_facecolor("#0d1117")
    ax.grid(True, axis="y", alpha=0.15, color="#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#30363d")
    ax.spines["bottom"].set_color("#30363d")
    ax.tick_params(colors="#8b949e", which="both", labelsize=8)

    y_min = min(min(local_scores), min(frontier_scores)) - 10
    ax.set_ylim(bottom=max(0, y_min))


def main():
    with open(DATA_FILE) as f:
        data = json.load(f)

    labs = data["labs"]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")

    today = datetime.now().strftime("%-d %B %Y")
    fig.suptitle(
        f"Tool Use Benchmarks — Local Models (≤ 30B params) [{today}]",
        fontsize=18, color="#e6edf3", fontweight="bold", y=0.97,
    )

    # Top-left: BFCL V4
    plot_bar_subplot(axes[0, 0], data["bfcl_v4"], labs,
                     title="BFCL V4 — Overall Accuracy", x_min=40)

    # Top-right: Docker F1
    plot_bar_subplot(axes[0, 1], data["docker_f1"], labs,
                     title="Docker Tool Calling F1", x_min=55)

    # Bottom-left: Tau2-Retail
    plot_bar_subplot(axes[1, 0], data["tau2_retail"], labs,
                     title="Tau²-Bench — Retail Domain", x_min=0)

    # Bottom-right: Local vs Frontier comparison
    plot_gap_subplot(axes[1, 1], data["local_vs_frontier"], labs)

    # Build legend for bar charts
    used_labs = set()
    for key in ["bfcl_v4", "docker_f1", "tau2_retail"]:
        for m in data[key]["models"]:
            used_labs.add(m["lab"])

    legend_handles = []
    for lab_name in sorted(used_labs):
        colour = labs.get(lab_name, {"colour": "#888"})["colour"]
        legend_handles.append(
            mpatches.Patch(facecolor=colour, edgecolor=colour, label=lab_name))
    legend_handles.append(
        plt.Line2D([0], [0], color="#FF4444", linestyle="--", linewidth=1.5,
                   label="Best frontier (reference)"))

    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=min(len(legend_handles), 7),
        fontsize=9, framealpha=0.3, edgecolor="#30363d", facecolor="#161b22",
        labelcolor="#e6edf3", bbox_to_anchor=(0.5, 0.01),
    )

    fig.text(
        0.5, 0.04,
        "Sources: BFCL V4 (gorilla.cs.berkeley.edu), Docker practical eval, "
        "Tau²-bench (Sierra Research), BenchLM, haimaker.ai  |  "
        "Dashed red line = best frontier model for reference  |  * = exceeds 30B",
        ha="center", fontsize=8, color="#6e7681",
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    IMG_DIR.mkdir(exist_ok=True)
    output_path = IMG_DIR / "tool-use-local-models.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
