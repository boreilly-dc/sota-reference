#!/usr/bin/env python3
"""Render an apples-for-apples OCR score versus parameter count chart."""

import argparse
import logging
import math
import sys
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_common import DATA_DIR, IMG_DIR, INK, MUTED, configure_figure, load_json, style_axis

DATA_FILE = DATA_DIR / "ocr_performance_vs_parameters.json"
OUTPUT_FILE = IMG_DIR / "ocr-performance-vs-parameter-count.png"
LOG_FILE = Path("/tmp/sota-plot-ocr-performance.log")
SPECIALIST = "#2A78D6"
GENERALIST = "#EB6834"
MAX_PARAMS = 50
BENCHMARK_ID = "idp_omnidocbench_v1_5_overall"


def validate(data):
    benchmarks = {row["benchmark_id"]: row for row in data.get("benchmarks", [])}
    if set(benchmarks) != {BENCHMARK_ID}:
        raise ValueError("The apples-for-apples chart must contain one benchmark and metric")
    if not benchmarks[BENCHMARK_ID].get("higher_is_better"):
        raise ValueError("Expected an overall score where higher is better")

    identities = set()
    for row in data.get("observations", []):
        identity = (row.get("model"), row.get("benchmark_id"))
        if identity in identities:
            raise ValueError(f"Duplicate observation: {identity}")
        identities.add(identity)
        if row.get("benchmark_id") != BENCHMARK_ID:
            raise ValueError(f"Incompatible benchmark for {row.get('model')}")
        if row.get("openness") != "open_weights":
            raise ValueError(f"Non-open-weight model: {row.get('model')}")
        params = row.get("parameters_billions")
        if not isinstance(params, (int, float)) or not 0 < params < MAX_PARAMS:
            raise ValueError(f"Parameter limit failure: {row.get('model')} ({params})")
        value = row.get("value")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Invalid score: {row.get('model')}")
        if not row.get("licence"):
            raise ValueError(f"Missing licence: {row.get('model')}")
        for field in ("source_url", "weights_url"):
            parsed = urlparse(row.get(field, ""))
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError(f"Invalid {field}: {row.get('model')}")
    if not any(row["model_type"] == "ocr_specialist" for row in data["observations"]):
        raise ValueError("Missing OCR specialist")
    if not any(row["family"] == "Gemma 4" for row in data["observations"]):
        raise ValueError("Missing Gemma 4 model")
    print(f"OCR plot data validation passed: {len(identities)} comparable observations")


def parameter_label(row):
    total = f'{row["parameters_billions"]:.2f}B total'
    effective = row.get("effective_parameters_billions")
    architecture = row.get("published_architecture_billions")
    if effective is not None:
        return f"{effective:g}B effective · {total}"
    if architecture is not None:
        return f"{architecture:g}B architecture · {total}"
    return total


def plot(data):
    rows = sorted(data["observations"], key=lambda row: row["parameters_billions"])
    fig = configure_figure(
        "OCR versus Gemma 4: one benchmark, one metric",
        "Snapshot: 9 September 2026 | Open weights, <50B total checkpoint parameters | One third-party OmniDocBench v1.5 leaderboard",
    )
    fig.set_size_inches(15, 9)
    ax = fig.add_subplot(111)
    style_axis(ax)

    offsets = {
        "GLM-OCR": (14, 11),
        "Gemma 4 E2B IT": (14, -23),
        "Gemma 4 E4B IT": (-14, 12),
    }
    aligns = {"Gemma 4 E4B IT": "right"}

    for row in rows:
        specialist = row["model_type"] == "ocr_specialist"
        colour = SPECIALIST if specialist else GENERALIST
        marker = "o" if specialist else "D"
        ax.scatter(
            row["parameters_billions"], row["value"], s=230, marker=marker,
            color=colour, edgecolor="#0d1117", linewidth=1.5, zorder=3,
        )
        ax.annotate(
            f'{row["model"]}\n{parameter_label(row)} · score {row["value"]:.2f}',
            (row["parameters_billions"], row["value"]),
            xytext=offsets[row["model"]], textcoords="offset points",
            ha=aligns.get(row["model"], "left"), va="center",
            color=INK, fontsize=10.2, fontweight="bold", linespacing=1.35,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
        )

    ax.set_xscale("log")
    ax.set_xlim(0.95, 10)
    ax.set_ylim(38, 74)
    ax.set_xticks([1, 2, 3, 5, 8, 10], labels=["1B", "2B", "3B", "5B", "8B", "10B"])
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax.set_yticks([40, 45, 50, 55, 60, 65, 70])
    ax.set_xlabel("Total checkpoint parameters · logarithmic scale", color=MUTED, labelpad=14)
    ax.set_ylabel("OmniDocBench v1.5 overall score · higher is better", color=MUTED, labelpad=14)
    ax.grid(True, color="#30363d", alpha=0.22)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SPECIALIST,
               markeredgecolor="#0d1117", markersize=10, label="OCR specialist"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=GENERALIST,
               markeredgecolor="#0d1117", markersize=9, label="General-purpose VLM"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=False, labelcolor=INK, fontsize=10)

    fig.text(
        0.08, 0.885,
        "All points use the same third-party OmniDocBench v1.5 overall-score table",
        color=MUTED, fontsize=11,
    )
    fig.text(
        0.08, 0.115,
        "Only Gemma 4 E2B IT and E4B IT have published overall scores in this common evaluation.\n"
        "Google reports a different aggregate edit-distance metric for 12B, 26B-A4B, and 31B; those values are excluded.",
        color=MUTED, fontsize=9.4, linespacing=1.5,
    )
    fig.text(
        0.92, 0.115,
        "GLM-OCR leads E4B by 9.53 points",
        ha="right", color=SPECIALIST, fontsize=10.5, fontweight="bold",
    )

    IMG_DIR.mkdir(exist_ok=True)
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.22, top=0.80)
    fig.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {OUTPUT_FILE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="Validate data without rendering")
    args = parser.parse_args()
    data = load_json(DATA_FILE)
    validate(data)
    if not args.validate:
        plot(data)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.basicConfig(filename=LOG_FILE, level=logging.ERROR)
        logging.exception("OCR performance plot failed")
        raise
