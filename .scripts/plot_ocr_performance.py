#!/usr/bin/env python3
"""Render open-weight OCR performance against published parameter count."""

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


def validate(data):
    benchmarks = {row["benchmark_id"]: row for row in data.get("benchmarks", [])}
    if len(benchmarks) != len(data.get("benchmarks", [])):
        raise ValueError("Duplicate benchmark IDs")

    identities = set()
    for row in data.get("observations", []):
        identity = (row.get("model"), row.get("benchmark_id"))
        if identity in identities:
            raise ValueError(f"Duplicate observation: {identity}")
        identities.add(identity)
        if row.get("benchmark_id") not in benchmarks:
            raise ValueError(f"Unknown benchmark for {row.get('model')}")
        if row.get("openness") != "open_weights":
            raise ValueError(f"Non-open-weight model: {row.get('model')}")
        params = row.get("parameters_billions")
        if not isinstance(params, (int, float)) or not 0 < params < MAX_PARAMS:
            raise ValueError(f"Parameter limit failure: {row.get('model')} ({params})")
        if not isinstance(row.get("value"), (int, float)) or not math.isfinite(row["value"]):
            raise ValueError(f"Invalid score: {row.get('model')}")
        if not row.get("licence"):
            raise ValueError(f"Missing licence: {row.get('model')}")
        for field in ("source_url", "weights_url"):
            parsed = urlparse(row.get(field, ""))
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError(f"Invalid {field}: {row.get('model')}")

    required = {"omnidocbench_v1_6_overall", "omnidocbench_v1_5_ned"}
    if set(benchmarks) != required:
        raise ValueError(f"Expected benchmark panels {sorted(required)}")
    print(f"OCR plot data validation passed: {len(identities)} observations")


def label_text(row):
    params = f'{row["parameters_billions"]:g}B'
    if "active_parameters_billions" in row:
        params += f' total / {row["active_parameters_billions"]:g}B active'
    elif "effective_parameters_billions" in row:
        params += f' total / {row["effective_parameters_billions"]:g}B effective'
    return f'{row["model"]}\n{params} · {row["value"]:.3f}'


def plot(data):
    by_benchmark = {
        benchmark["benchmark_id"]: [
            row for row in data["observations"]
            if row["benchmark_id"] == benchmark["benchmark_id"]
        ]
        for benchmark in data["benchmarks"]
    }

    fig = configure_figure(
        "Open-weight OCR: performance versus parameter count",
        "Snapshot: 31 August 2026 | Open weights, <50B total parameters | Primary-source papers and model cards",
    )
    fig.set_size_inches(18, 9.5)
    axes = fig.subplots(1, 2, gridspec_kw={"wspace": 0.24})

    left = axes[0]
    style_axis(left)
    specialist_rows = by_benchmark["omnidocbench_v1_6_overall"]
    offsets = {
        "NaviDC-OCR": (8, 8),
        "OvisOCR2": (-8, 15),
        "PaddleOCR-VL-1.6": (8, -18),
        "MinerU2.5-Pro": (8, -30),
        "GLM-OCR": (-8, -16),
        "HunyuanOCR-1.5": (8, -24),
        "Ovis2.6-30B-A3B": (-8, 12),
    }
    aligns = {"OvisOCR2": "right", "GLM-OCR": "right", "Ovis2.6-30B-A3B": "right"}
    for row in specialist_rows:
        specialist = row["model_type"] == "ocr_specialist"
        left.scatter(
            row["parameters_billions"], row["value"], s=130, marker="o" if specialist else "D",
            color=SPECIALIST if specialist else GENERALIST, edgecolor="#0d1117", linewidth=1.2, zorder=3,
        )
        left.annotate(
            label_text(row) if not specialist else f'{row["model"]}\n{row["parameters_billions"]:g}B · {row["value"]:.2f}',
            (row["parameters_billions"], row["value"]),
            xytext=offsets[row["model"]], textcoords="offset points",
            ha=aligns.get(row["model"], "left"), va="center",
            color=INK, fontsize=8.3, linespacing=1.25,
        )
    left.set_xscale("log")
    left.set_xlim(0.65, 42)
    left.set_ylim(93.1, 97.15)
    left.set_xticks([0.8, 1, 2, 5, 10, 30], labels=["0.8B", "1B", "2B", "5B", "10B", "30B"])
    left.get_xaxis().set_minor_formatter(plt.NullFormatter())
    left.set_title("Current open-weight field · OmniDocBench v1.6", color=INK, fontsize=13, fontweight="bold")
    left.text(0.5, 1.01, "Overall score · higher is better", transform=left.transAxes, ha="center", color=MUTED, fontsize=9)
    left.set_xlabel("Published parameters (log scale)", color=MUTED)
    left.set_ylabel("Overall score", color=MUTED)
    left.grid(True, color="#30363d", alpha=0.22)

    right = axes[1]
    style_axis(right)
    gemma_rows = sorted(by_benchmark["omnidocbench_v1_5_ned"], key=lambda row: row["parameters_billions"])
    right.plot(
        [row["parameters_billions"] for row in gemma_rows],
        [row["value"] for row in gemma_rows],
        color=GENERALIST, linewidth=2, alpha=0.75, zorder=2,
    )
    offsets = {
        "Gemma 4 E2B": (8, 9),
        "Gemma 4 E4B": (8, 10),
        "Gemma 4 12B": (8, 8),
        "Gemma 4 26B-A4B": (-8, -25),
        "Gemma 4 31B": (-8, 12),
    }
    aligns = {"Gemma 4 26B-A4B": "right", "Gemma 4 31B": "right"}
    for row in gemma_rows:
        right.scatter(
            row["parameters_billions"], row["value"], s=125, marker="D",
            color=GENERALIST, edgecolor="#0d1117", linewidth=1.2, zorder=3,
        )
        right.annotate(
            label_text(row), (row["parameters_billions"], row["value"]),
            xytext=offsets[row["model"]], textcoords="offset points",
            ha=aligns.get(row["model"], "left"), va="center",
            color=INK, fontsize=8.2, linespacing=1.25,
        )
    right.set_xscale("log")
    right.set_xlim(4.2, 38)
    right.set_ylim(0.31, 0.115)
    right.set_xticks([5, 8, 12, 20, 30], labels=["5B", "8B", "12B", "20B", "30B"])
    right.get_xaxis().set_minor_formatter(plt.NullFormatter())
    right.set_title("Gemma 4 · OmniDocBench v1.5", color=INK, fontsize=13, fontweight="bold")
    right.text(0.5, 1.01, "Normalised edit distance · lower is better", transform=right.transAxes, ha="center", color=MUTED, fontsize=9)
    right.set_xlabel("Published total parameters (log scale)", color=MUTED)
    right.set_ylabel("Normalised edit distance", color=MUTED)
    right.grid(True, color="#30363d", alpha=0.22)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SPECIALIST,
               markeredgecolor="#0d1117", markersize=9, label="OCR specialist"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=GENERALIST,
               markeredgecolor="#0d1117", markersize=8, label="General-purpose VLM"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2,
               frameon=False, labelcolor=INK, fontsize=9.5)
    fig.text(
        0.5, 0.085,
        "Do not compare panel heights: v1.6 uses a composite overall score; Gemma 4 reports v1.5 normalised edit distance.",
        ha="center", color="#E8B32A", fontsize=9.5, fontweight="bold",
    )
    IMG_DIR.mkdir(exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.16, top=0.78, wspace=0.24)
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
