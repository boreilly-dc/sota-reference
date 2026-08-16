#!/usr/bin/env python3
"""Shared plotting helpers for tracked benchmark snapshots."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / ".data"
IMG_DIR = BASE_DIR / "images"
SURFACE = "#0d1117"
INK = "#e6edf3"
MUTED = "#8b949e"
GRID = "#30363d"
LAB_COLOURS = {
    "Anthropic": "#D97757", "OpenAI": "#10A37F", "Google DeepMind": "#4285F4",
    "DeepSeek": "#A78BFA", "Alibaba": "#E8B32A", "Moonshot AI": "#F472B6",
    "Z.ai": "#00C49A", "Mistral": "#FF7000", "Meta": "#60A5FA",
}


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", alpha=0.16, color=GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def colour_for(developer):
    return LAB_COLOURS.get(developer, "#94A3B8")


def configure_figure(title, footer):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 12), facecolor=SURFACE)
    fig.suptitle(title, fontsize=18, color=INK, fontweight="bold", y=0.97)
    fig.text(0.5, 0.02, footer, ha="center", fontsize=8, color="#6e7681")
    return fig


def save_figure(fig, path):
    IMG_DIR.mkdir(exist_ok=True)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {path}")
