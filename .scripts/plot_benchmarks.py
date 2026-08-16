#!/usr/bin/env python3
"""Render cost, tool-use, and agentic-coding benchmark plots."""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_common import (DATA_DIR, IMG_DIR, INK, MUTED, colour_for, configure_figure,
                         load_json, save_figure, style_axis)

COST_FILE = DATA_DIR / "cost_per_intelligence.json"
TOOL_FILE = DATA_DIR / "tool_use_current.json"
CODING_FILES = [
    DATA_DIR / "benchmarklist" / "open-models-coding-agents.json",
    DATA_DIR / "benchmarklist" / "agentic-coding-claude-vs-openai.json",
    DATA_DIR / "benchmarklist" / "glm-5-2.json",
]


def validate_cost(data):
    required = {"model_id", "name", "developer", "intelligence_index", "input_usd_per_million", "output_usd_per_million", "source_url", "sampled_at"}
    for model in data["models"]:
        missing = required - model.keys()
        if missing:
            raise ValueError(f"Cost row {model.get('name')} missing {sorted(missing)}")
        if not 0 < model["intelligence_index"] <= 100:
            raise ValueError(f"Invalid intelligence score for {model['name']}")
        if model["input_usd_per_million"] < 0 or model["output_usd_per_million"] < 0:
            raise ValueError(f"Invalid price for {model['name']}")


def validate_tool(data):
    ids = {item["benchmark_id"] for item in data["benchmarks"]}
    if len(ids) != len(data["benchmarks"]):
        raise ValueError("Duplicate tool benchmark IDs")
    for benchmark in data["benchmarks"]:
        if not benchmark.get("source_url") or not benchmark.get("sampled_at"):
            raise ValueError(f"Missing provenance for {benchmark['benchmark_id']}")
    for row in data["observations"]:
        if row["benchmark_id"] not in ids or not math.isfinite(row["value"]):
            raise ValueError(f"Invalid tool observation: {row}")
        if not 0 <= row["value"] <= 100:
            raise ValueError(f"Tool score outside percentage range: {row}")


def metric_value(observation):
    metric = observation.get("metric")
    if isinstance(metric, dict) and isinstance(metric.get("value"), (int, float)):
        return metric["value"]
    if isinstance(observation.get("score"), (int, float)):
        return observation["score"]
    return None


def coding_rows():
    rows = {}
    model_metadata = {}
    for path in CODING_FILES:
        data = load_json(path)
        for model in data.get("models", []):
            model_metadata[model.get("model_id")] = model
        for observation in data.get("observations", []):
            if observation.get("benchmark_id") not in {"swe_bench_pro", "vals_terminal_bench_2_1", "livecodebench_official"}:
                continue
            value = metric_value(observation)
            if value is None:
                continue
            key = (observation["benchmark_id"], observation.get("subject_id"), observation.get("sampled_at"), observation.get("qualifiers", {}).get("harness"))
            model = model_metadata.get(observation.get("model_id"), {})
            rows[key] = observation | {"plot_value": value, "developer": model.get("developer", "")}
    return list(rows.values())


def plot_cost():
    data = load_json(COST_FILE)
    validate_cost(data)
    fig = configure_figure("Cost versus Intelligence", "Source: Artificial Analysis snapshot, 17 August 2026 | Cost = 1M input + 1M output tokens at published list prices | Up and left is better")
    ax = fig.add_subplot(111)
    style_axis(ax)
    for model in data["models"]:
        cost = model["input_usd_per_million"] + model["output_usd_per_million"]
        marker = "D" if model.get("openness") == "open_weights" else "o"
        ax.scatter(cost, model["intelligence_index"], s=90, marker=marker, color=colour_for(model["developer"]), edgecolor=INK, linewidth=0.6, zorder=3)
        ax.annotate(model["name"], (cost, model["intelligence_index"]), xytext=(7, 5), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Estimated cost (USD per 1M input + 1M output tokens)", color=MUTED)
    ax.set_ylabel("Artificial Analysis Intelligence Index", color=MUTED)
    ax.set_ylim(bottom=45, top=65)
    ax.grid(True, which="both", axis="x", alpha=0.12, color="#30363d")
    save_figure(fig, IMG_DIR / "cost-per-intelligence.png")


def plot_tool():
    data = load_json(TOOL_FILE)
    validate_tool(data)
    benchmarks = data["benchmarks"]
    fig = configure_figure("Tool-use benchmark scores", "Sources: BFCL V4, τ-bench, and Toolathlon-Verified | Panels are separate benchmark/version snapshots; scores are not combined")
    axes = fig.subplots(2, 3).flat
    for ax, benchmark in zip(axes, benchmarks):
        rows = [row for row in data["observations"] if row["benchmark_id"] == benchmark["benchmark_id"]]
        rows.sort(key=lambda row: row["value"])
        names = [row["name"] for row in rows]
        values = [row["value"] for row in rows]
        colours = [colour_for(row["developer"]) for row in rows]
        ax.barh(np.arange(len(rows)), values, color=colours, height=0.58)
        ax.set_yticks(np.arange(len(rows)), labels=names, fontsize=8)
        ax.set_xlabel(f"{benchmark['metric_label']} ({benchmark['unit']})", color=MUTED, fontsize=8)
        ax.set_title(f"{benchmark['display_name']} ({benchmark['version']})", color=INK, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 100)
        style_axis(ax)
        for index, value in enumerate(values):
            ax.text(value + 1, index, f"{value:.1f}", va="center", fontsize=8, color=INK)
    for ax in list(axes)[len(benchmarks):]:
        ax.set_visible(False)
    save_figure(fig, IMG_DIR / "tool-use-benchmarks.png")


def plot_coding():
    rows = coding_rows()
    if not rows:
        raise ValueError("No coding observations available")
    panels = [("swe_bench_pro", "SWE-bench Pro"), ("vals_terminal_bench_2_1", "Terminal-Bench 2.1"), ("livecodebench_official", "LiveCodeBench")]
    fig = configure_figure("Agentic coding benchmarks", "Source-linked BenchmarkList observations | Vendor and harness results remain separate; LiveCodeBench is code generation, not repository-level agency")
    axes = fig.subplots(1, 3).flat
    for ax, (benchmark_id, title) in zip(axes, panels):
        panel = [row for row in rows if row["benchmark_id"] == benchmark_id]
        panel.sort(key=lambda row: row["plot_value"])
        names = [row.get("display_name", row.get("subject_id", "Unknown")) for row in panel]
        values = [row["plot_value"] for row in panel]
        colours = [colour_for(row.get("developer", "")) for row in panel]
        ax.barh(np.arange(len(panel)), values, color=colours, height=0.58)
        ax.set_yticks(np.arange(len(panel)), labels=names, fontsize=8)
        ax.set_xlabel("Score (%)", color=MUTED, fontsize=8)
        ax.set_title(title, color=INK, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 100)
        style_axis(ax)
        for index, value in enumerate(values):
            ax.text(value + 1, index, f"{value:.1f}", va="center", fontsize=8, color=INK)
    save_figure(fig, IMG_DIR / "agentic-coding-benchmarks.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", choices=["cost-intelligence", "tool-use", "agentic-coding"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate or args.all or args.plot == "cost-intelligence":
        validate_cost(load_json(COST_FILE))
    if args.validate or args.all or args.plot == "tool-use":
        validate_tool(load_json(TOOL_FILE))
    if args.validate:
        print("Plot data validation passed")
    if args.all or args.plot == "cost-intelligence":
        plot_cost()
    if args.all or args.plot == "tool-use":
        plot_tool()
    if args.all or args.plot == "agentic-coding":
        plot_coding()


if __name__ == "__main__":
    main()
