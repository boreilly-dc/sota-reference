#!/usr/bin/env python3
"""Plot Elo timeline for open-source/open-weight models."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_FILE = Path(__file__).resolve().parent.parent / ".data" / "local_model_elo_history.json"
IMG_DIR = Path(__file__).resolve().parent.parent / "images"


def main():
    with open(DATA_FILE) as f:
        data = json.load(f)

    labs = data["labs"]
    families = data["families"]
    models = data["models"]

    for m in models:
        m["date"] = datetime.strptime(m["release_date"], "%Y-%m-%d")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Draw family connection lines
    for family_name, member_names in families.items():
        members = [m for m in models if m["name"] in member_names]
        members.sort(key=lambda m: m["date"])
        if len(members) < 2:
            continue
        dates = [m["date"] for m in members]
        elos = [m["elo"] for m in members]
        colour = labs[members[0]["lab"]]["colour"]
        ax.plot(dates, elos, color=colour, alpha=0.25, linewidth=1.8, zorder=1)

    # Plot individual model points
    plotted_labs = set()
    for m in models:
        lab = m["lab"]
        colour = labs[lab]["colour"]
        label = lab if lab not in plotted_labs else None
        plotted_labs.add(lab)
        marker = "D"  # diamond for open-source
        ax.scatter(
            m["date"], m["elo"],
            c=colour, s=90, marker=marker, zorder=5,
            edgecolors="white", linewidths=0.5, label=label, alpha=0.9,
        )

    # Annotate each point
    placed_boxes = []
    for m in models:
        x_num = mdates.date2num(m["date"])
        elo = m["elo"]
        colour = labs[m["lab"]]["colour"]

        # Simple label placement — try offsets
        offsets = [(8, 6), (8, -12), (-60, 6), (-60, -12), (8, 14), (8, -20)]
        best_offset = offsets[0]
        for ox, oy in offsets:
            test_x = x_num + ox * 0.3
            test_y = elo + oy
            conflict = False
            for px, py in placed_boxes:
                if abs(test_x - px) < 15 and abs(test_y - py) < 12:
                    conflict = True
                    break
            if not conflict:
                best_offset = (ox, oy)
                break

        placed_boxes.append((x_num + best_offset[0] * 0.3, elo + best_offset[1]))

        short = m["name"]
        if len(short) > 20:
            short = short[:18] + "…"

        ax.annotate(
            short, (m["date"], elo),
            xytext=best_offset, textcoords="offset points",
            fontsize=7.5, color=colour, alpha=0.9,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=colour, alpha=0.3, lw=0.5),
        )

    ax.set_xlabel("Release Date", fontsize=11, color="#8b949e")
    ax.set_ylabel("Arena Elo", fontsize=11, color="#8b949e")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax.grid(True, alpha=0.1, color="#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#30363d")
    ax.spines["bottom"].set_color("#30363d")
    ax.tick_params(colors="#8b949e", which="both")

    # Legend
    legend = ax.legend(
        loc="upper left", fontsize=9, framealpha=0.3,
        edgecolor="#30363d", facecolor="#161b22", labelcolor="#e6edf3",
    )

    today = datetime.now().strftime("%-d %B %Y")
    ax.set_title(
        f"Open-Source / Open-Weight Model Elo Timeline [{today}]",
        fontsize=16, color="#e6edf3", fontweight="bold", pad=15,
    )

    fig.text(
        0.5, 0.02,
        "Sources: LMArena (arena.ai)  |  All models are open-weight  |  "
        "Diamond markers = open source  |  Lines connect models in the same family",
        ha="center", fontsize=8, color="#6e7681",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    IMG_DIR.mkdir(exist_ok=True)
    output_path = IMG_DIR / "local-model-elo-timeline.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
