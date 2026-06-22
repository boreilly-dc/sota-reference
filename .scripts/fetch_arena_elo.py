#!/usr/bin/env python3
"""Fetch latest LMArena Elo ratings and update model_elo_history.json.

Data source: github.com/oolong-tea-2026/arena-ai-leaderboards (daily snapshots).
Dry-run by default — use --apply to write changes.
"""

import argparse
import json
import statistics
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

REPO_BASE = "https://raw.githubusercontent.com/oolong-tea-2026/arena-ai-leaderboards/main/data"
DATA_FILE = Path(__file__).resolve().parent.parent / ".data" / "model_elo_history.json"

# Arena slug → our display name.
# For models with variants (thinking, high, etc.), map every variant to the
# same display name; the script takes the max score across variants.
SLUG_MAP = {
    # OpenAI
    "gpt-4":                          "GPT-4",
    "gpt-4-turbo":                    "GPT-4 Turbo",
    "gpt-4o":                         "GPT-4o",
    "gpt-5":                          "GPT-5",
    "gpt-5.2":                        "GPT-5.2",
    "gpt-5.2-pro":                    "GPT-5.2 Pro",
    "gpt-5.4":                        "GPT-5.4",
    "gpt-5.4-high":                   "GPT-5.4",
    "gpt-5.4-pro":                    "GPT-5.4 Pro",
    "gpt-5.5":                        "GPT-5.5",
    "gpt-5.5-high":                   "GPT-5.5",
    "gpt-5.5-instant":                None,  # skip — separate variant we don't track
    "gpt-5.1-high":                   None,  # GPT-5.1 not tracked
    "gpt-5.3-chat-latest":            None,  # GPT-5.3 not tracked
    "gpt-5.4-mini-high":              None,  # mini variant not tracked

    # Anthropic
    "claude-3-opus":                  "Claude 3 Opus",
    "claude-3.5-sonnet":              "Claude 3.5 Sonnet",
    "claude-3.5-sonnet-v2":           "Claude 3.5 Sonnet v2",
    "claude-sonnet-4":                "Claude Sonnet 4",
    "claude-opus-4":                  "Claude Opus 4",
    "claude-sonnet-4-5":              "Claude Sonnet 4.5",
    "claude-sonnet-4-6":              "Claude Sonnet 4.6",
    "claude-opus-4-5":                "Claude Opus 4.5",
    "claude-opus-4-6":                "Claude Opus 4.6",
    "claude-opus-4-6-thinking":       "Claude Opus 4.6",
    "claude-opus-4-7":                "Claude Opus 4.7",
    "claude-opus-4-7-thinking":       "Claude Opus 4.7",
    "claude-opus-4-8":                "Claude Opus 4.8",
    "claude-opus-4-8-thinking":       "Claude Opus 4.8",
    "claude-fable-5":                 "Claude Fable 5",
    "claude-fable-5-thinking":        "Claude Fable 5",
    "claude-mythos-5":                None,  # restricted-access sibling, not on public arena

    # Google DeepMind
    "gemini-1.0-pro":                 "Gemini 1.0 Pro",
    "gemini-1.5-pro":                 "Gemini 1.5 Pro",
    "gemini-2.0-flash":               "Gemini 2.0 Flash",
    "gemini-2.5-pro":                 "Gemini 2.5 Pro",
    "gemini-3-pro":                   "Gemini 3 Pro",
    "gemini-3.1-pro-preview":         "Gemini 3.1 Pro Preview",
    "gemini-3-flash":                 None,  # we don't track this yet
    "gemma-3-27b":                    "Gemma 3 27B",
    "gemma-4-31b":                    "Gemma 4 31B",

    # xAI
    "grok-2":                         "Grok 2",
    "grok-3":                         "Grok 3",
    "grok-4.1":                       "Grok 4.1",
    "grok-4.20-beta1":                "Grok 4.20 Beta",

    # Meta
    "llama-3-70b":                    "Llama 3 70B",
    "llama-3.1-405b":                 "Llama 3.1 405B",
    "llama-4-maverick":               "Llama 4 Maverick",
    "muse-spark":                     "Muse Spark",

    # DeepSeek
    "deepseek-v2":                    "DeepSeek V2",
    "deepseek-v3":                    "DeepSeek V3",
    "deepseek-r1":                    "DeepSeek R1",
    "deepseek-v3.2-exp":              "DeepSeek V3.2-Exp",
    "deepseek-v4-pro":                "DeepSeek V4 Pro",

    # Mistral
    "mistral-large":                  "Mistral Large",
    "mistral-large-2":                "Mistral Large 2",
    "mistral-large-3":                "Mistral Large 3",

    # Alibaba
    "qwen-2.5-72b":                   "Qwen 2.5-72B",
    "qwen3-235b":                     "Qwen3-235B",

    # Zhipu AI
    "glm-4":                          "GLM-4",
    "glm-4.5":                        "GLM-4.5",
    "glm-4.7":                        "GLM-4.7",
    "glm-5":                          "GLM-5",
    "glm-5.1":                        "GLM-5.1",
    "glm-5.2 (max)":                  None,  # GLM-5.2 not tracked yet; arena score sits below GLM-5.1

    # Baidu — not tracked
    "ernie-5.1":                      None,
    "ernie-5.1-preview":              None,
    "ernie-5.0-preview-1203":         None,

    # Moonshot — not tracked in elo history
    "kimi-k2.6":                      None,
    "kimi-k2.5-thinking":             None,

    # Xiaomi — not tracked in elo history
    "mimo-v2.5-pro":                  None,
    "mimo-v2-pro":                    None,

    # Alibaba preview — not tracked
    "qwen3.5-max-preview":            None,
    "qwen3.6-max-preview":            None,
    "qwen3.7-max-preview":            None,

    # Google — Gemini 3.5 Flash not tracked yet
    "gemini-3.5-flash":               None,

    # Bytedance — not tracked
    "dola-seed-2.0-pro":              None,

    # MiniMax — not tracked
    "minimax-m3":                     None,
}

# Patterns that match arena slugs not in the exact map above.
# Checked with startswith() — for versioned/dated variants.
SLUG_PREFIX_MAP = {
    "gpt-5.2-chat-latest":            "GPT-5.2",  # updated production checkpoint
    "claude-opus-4-1-":               "Claude Opus 4",
    "grok-4.20-beta-":                "Grok 4.20 Beta",
    "grok-4.20-multi-agent":          "Grok 4.20 Beta",
    "grok-4.1-thinking":              "Grok 4.1",
    "deepseek-v4-pro-thinking":       "DeepSeek V4 Pro",
    "gemini-3-flash":                 None,
    "claude-opus-4-5-":               "Claude Opus 4.5",
    "claude-sonnet-4-5-":             "Claude Sonnet 4.5",
    "claude-sonnet-4-6-":             "Claude Sonnet 4.6",
}


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "fetch-arena-elo/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def resolve_slug(slug: str) -> str | None:
    """Map an arena slug to our display name, or None to skip."""
    if slug in SLUG_MAP:
        return SLUG_MAP[slug]
    for prefix, name in SLUG_PREFIX_MAP.items():
        if slug.startswith(prefix):
            return name
    return "__UNKNOWN__"


def fetch_snapshot(date: str) -> dict:
    url = f"{REPO_BASE}/{date}/text.json"
    data = fetch_json(url)
    best: dict[str, int] = {}
    unmapped: list[str] = []
    for entry in data["models"]:
        slug = entry["model"]
        score = entry["score"]
        display = resolve_slug(slug)
        if display == "__UNKNOWN__":
            unmapped.append(slug)
            continue
        if display is None:
            continue
        if display not in best or score > best[display]:
            best[display] = score
    return best, unmapped


def estimate_drift(latest_date: str, days_back: int = 7) -> int:
    """Compare two snapshots to estimate overall Elo scale drift."""
    dt = datetime.strptime(latest_date, "%Y-%m-%d")
    earlier_date = (dt - timedelta(days=days_back)).strftime("%Y-%m-%d")
    try:
        earlier, _ = fetch_snapshot(earlier_date)
    except Exception:
        return 0
    latest, _ = fetch_snapshot(latest_date)
    shifts = []
    for name in set(earlier) & set(latest):
        shifts.append(latest[name] - earlier[name])
    if not shifts:
        return 0
    return round(statistics.median(shifts))


def check_monotonicity(models: list[dict], families: dict) -> list[str]:
    model_map = {m["name"]: m for m in models}
    warnings = []
    for family_name, members in families.items():
        prev_elo = None
        prev_name = None
        for name in members:
            if name not in model_map:
                continue
            elo = model_map[name]["elo"]
            if prev_elo is not None and elo <= prev_elo:
                warnings.append(
                    f"  {family_name}: {prev_name} ({prev_elo}) >= {name} ({elo})"
                )
            prev_elo = elo
            prev_name = name
    return warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    parser.add_argument("--date", help="Use a specific snapshot date (YYYY-MM-DD) instead of latest")
    args = parser.parse_args()

    with open(DATA_FILE) as f:
        data = json.load(f)

    if args.date:
        snap_date = args.date
    else:
        latest = fetch_json(f"{REPO_BASE}/latest.json")
        snap_date = latest["date"]

    print(f"Fetching arena snapshot for {snap_date}...")
    arena_scores, unmapped = fetch_snapshot(snap_date)

    if unmapped:
        print(f"\n  Unknown arena slugs (add to SLUG_MAP):")
        for s in unmapped:
            print(f"    {s}")

    drift = estimate_drift(snap_date)
    if drift:
        print(f"\n  Estimated scale drift over last 7 days: {drift:+d}")

    changes = []
    unmatched = []
    for model in data["models"]:
        name = model["name"]
        old_elo = model["elo"]
        if name in arena_scores:
            new_elo = arena_scores[name]
            if new_elo != old_elo:
                changes.append((name, old_elo, new_elo, "arena"))
                model["elo"] = new_elo
        elif drift:
            new_elo = old_elo + drift
            if new_elo != old_elo:
                changes.append((name, old_elo, new_elo, "drift"))
                model["elo"] = new_elo
        else:
            unmatched.append(name)

    print(f"\n{'DRY RUN' if not args.apply else 'APPLYING'} — {len(changes)} changes:\n")
    for name, old, new, source in changes:
        tag = "" if source == "arena" else " (drift)"
        print(f"  {name:30s}  {old:5d} → {new:5d}{tag}")

    if unmatched:
        print(f"\n  Unmatched (no arena data, no drift): {len(unmatched)}")
        for n in unmatched:
            print(f"    {n}")

    mono_warnings = check_monotonicity(data["models"], data["families"])
    if mono_warnings:
        print(f"\n  Monotonicity warnings:")
        for w in mono_warnings:
            print(w)

    if args.apply and changes:
        today = datetime.now().strftime("%Y-%m-%d")
        data["metadata"]["last_updated"] = today
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\n  Written to {DATA_FILE}")
        print(f"  last_updated set to {today}")
    elif not args.apply and changes:
        print(f"\n  (use --apply to write changes)")


if __name__ == "__main__":
    main()
