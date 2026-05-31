#!/usr/bin/env python3
"""Plot AI model Elo ratings over time, coloured by lab, with family connections."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_FILE = Path(__file__).resolve().parent.parent / ".data" / "model_elo_history.json"
BASE_DIR = Path(__file__).resolve().parent.parent
IMG_DIR = BASE_DIR / "images"


def plot_chart(models, labs, families, *, output_path, title_suffix,
               month_interval, font_scale=1.0):
    """Render a single Elo-vs-date chart."""

    fs = font_scale  # shorthand

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # -- Family connection lines --
    for family_name, member_names in families.items():
        members = [m for m in models if m["name"] in member_names]
        members.sort(key=lambda m: m["date"])
        if len(members) < 2:
            continue
        lab = members[0]["lab"]
        colour = labs[lab]["colour"]
        dates = [m["date"] for m in members]
        elos = [m["elo"] for m in members]
        ax.plot(dates, elos, color=colour, alpha=0.2, linewidth=1.5, zorder=1)

    # -- Scatter points --
    plotted_labs = set()
    for m in models:
        lab = m["lab"]
        colour = labs[lab]["colour"]
        edge = "#555555" if lab == "xAI" else colour
        marker = "D" if m.get("open_source") else "o"
        label = lab if lab not in plotted_labs else None
        plotted_labs.add(lab)
        ax.scatter(
            m["date"], m["elo"],
            c=colour, edgecolors=edge, s=90 * fs, zorder=3,
            marker=marker, linewidths=0.8, label=label,
        )

    # -- Annotations with collision avoidance --
    annotations = [(m["date"], m["elo"], m["name"], labs[m["lab"]]["colour"])
                   for m in models]
    annotations.sort(key=lambda a: (a[0], a[1]))

    placed_boxes = []
    for date, elo, name, colour in annotations:
        x_num = mdates.date2num(date)
        best_offset_x, best_offset_y = 8, 10
        for cy in [10, -18, 24, -32, 38, -46, 52, -60, 66]:
            for cx in [8, -60, 8, -80]:
                test_x = x_num + cx * 0.3
                test_y = elo + cy
                if not any(abs(test_x - px) < 12 and abs(test_y - py) < 16
                           for px, py in placed_boxes):
                    best_offset_x, best_offset_y = cx, cy
                    break
            else:
                continue
            break
        placed_boxes.append((x_num + best_offset_x * 0.3, elo + best_offset_y))

        short = name.replace("Preview", "Prev.").replace("Beta", "B.")
        ax.annotate(
            short, (date, elo),
            textcoords="offset points", xytext=(best_offset_x, best_offset_y),
            fontsize=6.5 * fs, color=colour, alpha=0.85,
            arrowprops=dict(arrowstyle="-", color=colour, alpha=0.25, lw=0.5),
        )

    # -- Axes --
    ax.set_xlabel("Release date", fontsize=12 * fs, color="#8b949e")
    ax.set_ylabel("LMArena Elo", fontsize=12 * fs, color="#8b949e")
    today = datetime.now().strftime("%-d %B %Y")
    ax.set_title(
        f"Frontier AI Model Performance (LMArena Elo) \u2014 {title_suffix} [{today}]",
        fontsize=16 * fs, color="#e6edf3", pad=20, fontweight="bold",
    )

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9 * fs)
    ax.tick_params(colors="#8b949e", which="both", labelsize=9 * fs)

    ax.grid(True, alpha=0.15, color="#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#30363d")
    ax.spines["bottom"].set_color("#30363d")

    # -- Legend --
    legend_handles = []
    for lab_name in sorted(labs.keys()):
        colour = labs[lab_name]["colour"]
        if any(m["lab"] == lab_name for m in models):
            legend_handles.append(
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colour,
                           markersize=8 * fs, linestyle="None", label=lab_name))
    legend_handles.append(
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#8b949e",
                   markersize=7 * fs, linestyle="None", label="Open source"))
    legend_handles.append(
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#8b949e",
                   markersize=7 * fs, linestyle="None", label="Proprietary"))

    ax.legend(
        handles=legend_handles, loc="upper left", fontsize=9 * fs,
        framealpha=0.3, edgecolor="#30363d", facecolor="#161b22",
        labelcolor="#e6edf3",
    )

    fig.text(
        0.5, 0.01,
        "Sources: LMArena (lmarena.ai), Artificial Analysis, frontier-models-benchmark.md  |  "
        "Diamonds = open source, circles = proprietary  |  Lines connect models in the same family",
        ha="center", fontsize=8 * fs, color="#6e7681",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    with open(DATA_FILE) as f:
        data = json.load(f)

    labs = data["labs"]
    families = data["families"]
    all_models = [m for m in data["models"] if m.get("plot", True)]

    for m in all_models:
        m["date"] = datetime.strptime(m["release_date"], "%Y-%m-%d")

    IMG_DIR.mkdir(exist_ok=True)

    # Chart 1: last 2 years
    cutoff_2y = datetime(2024, 5, 7)
    models_2y = [m for m in all_models if m["date"] >= cutoff_2y]
    plot_chart(
        models_2y, labs, families,
        output_path=IMG_DIR / "model-elo-timeline.png",
        title_suffix="Last 2 Years",
        month_interval=2,
    )

    # Chart 2: last 6 months, larger font
    cutoff_6m = datetime(2025, 11, 7)
    models_6m = [m for m in all_models if m["date"] >= cutoff_6m]
    plot_chart(
        models_6m, labs, families,
        output_path=IMG_DIR / "model-elo-timeline-6m.png",
        title_suffix="Last 6 Months",
        month_interval=1,
        font_scale=1.3,
    )


if __name__ == "__main__":
    main()
