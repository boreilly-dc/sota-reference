#!/usr/bin/env python3
"""Fetch latest LMArena Elo ratings and update model_elo_history.json.

Data source: github.com/oolong-tea-2026/arena-ai-leaderboards (daily snapshots).
Dry-run by default — use --apply to write changes.
Auto-discovers new models not yet tracked (use --no-discover to disable).
"""

import argparse
import json
import re
import statistics
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

REPO_BASE = "https://raw.githubusercontent.com/oolong-tea-2026/arena-ai-leaderboards/main/data"
DATA_FILE = Path(__file__).resolve().parent.parent / ".data" / "model_elo_history.json"
DISCOVERED_FILE = Path(__file__).resolve().parent.parent / ".data" / "slug_map_discovered.json"

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
    "claude-3-5-sonnet-20240620":     "Claude 3.5 Sonnet",
    "claude-3-5-sonnet-20241022":     "Claude 3.5 Sonnet v2",
    "claude-sonnet-4":                "Claude Sonnet 4",
    "claude-sonnet-4-20250514":       "Claude Sonnet 4",
    "claude-sonnet-4-20250514-thinking-32k": "Claude Sonnet 4",
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
    "grok-3-preview-02-24":           "Grok 3",
    "grok-3-mini-high":               "Grok 3",
    "grok-3-mini-beta":               "Grok 3",
    "grok-4.1":                       "Grok 4.1",
    "grok-4.20-beta1":                "Grok 4.20 Beta",

    # Meta
    "llama-3-70b":                    "Llama 3 70B",
    "llama-3.1-405b":                 "Llama 3.1 405B",
    "llama-4-maverick":               "Llama 4 Maverick",
    "llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick",
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
    "glm-5.2":                        "GLM-5.2",
    "glm-5.2 (max)":                  "GLM-5.2",  # "max" thinking tier is the best-performing GLM-5.2 variant

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
    "qwen3-235b":                     "Qwen3-235B",  # matches dated/instruct variants e.g. qwen3-235b-a22b-instruct-2507
}


# --- Discovery infrastructure ---

# Slug prefixes for labs/model lines we never want to track.
SKIP_PREFIXES = [
    "ernie-",       # Baidu
    "kimi-",        # Moonshot
    "mimo-",        # Xiaomi
    "dola-",        # Bytedance
    "minimax-",     # MiniMax
    "yi-",          # 01.AI — not tracked
    "internlm-",   # Shanghai AI Lab — not tracked
    "chatglm-",    # Legacy ChatGLM — superseded by GLM
    "command-r",    # Cohere — not tracked
    "reka-",        # Reka — not tracked
    "jamba-",       # AI21 — not tracked
]

# Suffixes that indicate a variant of an existing model, not a new model line.
VARIANT_SUFFIXES = [
    "-high",
    "-thinking",
    "-xhigh",
    "-chat-latest",
    "-instant",
    "-mini",
    "-latest",
    "-turbo",
    "-online",
]

# Map slug prefixes to lab names for inference.
LAB_PREFIXES = [
    ("gpt-", "OpenAI"),
    ("o1-", "OpenAI"),
    ("o3-", "OpenAI"),
    ("o4-", "OpenAI"),
    ("claude-", "Anthropic"),
    ("gemini-", "Google DeepMind"),
    ("gemma-", "Google DeepMind"),
    ("grok-", "xAI"),
    ("llama-", "Meta"),
    ("muse-", "Meta"),
    ("deepseek-", "DeepSeek"),
    ("mistral-", "Mistral"),
    ("qwen", "Alibaba"),
    ("glm-", "Zhipu AI"),
]

# Map lab → whether models are open-source by default.
LAB_OPEN_SOURCE = {
    "Meta": True,
    "DeepSeek": True,
    "Alibaba": True,
    "Mistral": False,   # Mistral Large is not open
    "Google DeepMind": False,
}


def load_discovered_slugs() -> dict:
    """Load previously discovered slug mappings."""
    if DISCOVERED_FILE.exists():
        with open(DISCOVERED_FILE) as f:
            return json.load(f)
    return {}


def save_discovered_slugs(discovered: dict) -> None:
    """Persist discovered slug mappings."""
    with open(DISCOVERED_FILE, "w") as f:
        json.dump(discovered, f, indent=2, ensure_ascii=False)
        f.write("\n")


PROPER_NAMES = {
    "gpt": "GPT",
    "glm": "GLM",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "llama": "Llama",
    "gemini": "Gemini",
    "gemma": "Gemma",
    "grok": "Grok",
    "claude": "Claude",
    "mistral": "Mistral",
    "muse": "Muse",
    "opus": "Opus",
    "sonnet": "Sonnet",
    "fable": "Fable",
    "haiku": "Haiku",
    "pro": "Pro",
    "flash": "Flash",
    "maverick": "Maverick",
    "spark": "Spark",
    "large": "Large",
    "ultra": "Ultra",
    "exp": "Exp",
    "beta": "Beta",
    "preview": "Preview",
}

# Model name prefixes where the convention is acronym-version (e.g. GPT-6, GLM-5).
HYPHENATED_PREFIXES = {"gpt", "glm"}


def slug_to_display_name(slug: str) -> str:
    """Convert an arena slug to a human-readable display name.

    Examples:
        gpt-6 → GPT-6
        claude-sonnet-5 → Claude Sonnet 5
        deepseek-v5 → DeepSeek V5
        qwen4-120b → Qwen4-120B
        gemma-4-31b → Gemma 4 31B
    """
    clean = slug
    for suffix in VARIANT_SUFFIXES:
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
            break

    parts = clean.split("-")
    result = []
    for i, part in enumerate(parts):
        if re.match(r"^\d+b$", part, re.IGNORECASE):
            result.append(part.upper())
        elif re.match(r"^v\d", part):
            result.append(part[0].upper() + part[1:])
        elif re.match(r"^\d+(\.\d+)?$", part):
            # If previous part is a hyphenated prefix (GPT, GLM), join with hyphen
            if result and result[-1].lower() in HYPHENATED_PREFIXES:
                result[-1] = result[-1] + "-" + part
            else:
                result.append(part)
        elif part.lower() in PROPER_NAMES:
            result.append(PROPER_NAMES[part.lower()])
        else:
            result.append(part.capitalize())
    return " ".join(result)


def infer_lab(slug: str) -> str | None:
    """Infer lab name from slug prefix."""
    for prefix, lab in LAB_PREFIXES:
        if slug.startswith(prefix):
            return lab
    return None


def infer_family(display_name: str, lab: str, existing_families: dict) -> str | None:
    """Try to slot a new model into an existing family or suggest a new one."""
    best_match = None
    best_shared_len = 0

    for family_name, members in existing_families.items():
        if not members:
            continue
        for member in members:
            member_words = member.split()
            new_words = display_name.split()
            shared = []
            for mw, nw in zip(member_words, new_words):
                if mw == nw:
                    shared.append(mw)
                else:
                    break
            if len(shared) >= 2 and len(shared) > best_shared_len:
                best_shared_len = len(shared)
                best_match = family_name

    if best_match:
        return best_match

    # Couldn't match existing family — derive from the model name.
    # Strip trailing version numbers to get the "line" name.
    # e.g. "GPT-6" → "GPT-6" (whole thing is the family for a new major version)
    # "DeepSeek V5" → "DeepSeek V" (matches existing pattern)
    words = display_name.split()
    if len(words) >= 2 and re.match(r"^[Vv]?\d", words[-1]):
        # Check if stripping the last word matches an existing family
        candidate = " ".join(words[:-1])
        for family_name in existing_families:
            if family_name.startswith(candidate):
                return family_name
        return candidate
    return display_name


def should_skip_slug(slug: str) -> bool:
    """Return True if this slug belongs to a lab/line we never track."""
    for prefix in SKIP_PREFIXES:
        if slug.startswith(prefix):
            return True
    return False


def is_variant_of_existing(slug: str, existing_names: set) -> str | None:
    """If this slug is a variant of an already-tracked model, return its display name."""
    for suffix in VARIANT_SUFFIXES:
        if slug.endswith(suffix):
            base_slug = slug[: -len(suffix)]
            # Check if the base slug is in SLUG_MAP
            if base_slug in SLUG_MAP:
                return SLUG_MAP[base_slug]
            # Check prefix map
            for prefix, name in SLUG_PREFIX_MAP.items():
                if base_slug.startswith(prefix) and name is not None:
                    return name
    return None


def discover_model(slug: str, score: int, data: dict) -> dict | None:
    """Attempt to classify an unknown slug and return a new model entry, or None to skip."""
    if should_skip_slug(slug):
        return None

    # Check if it's a variant of an existing tracked model
    existing_names = {m["name"] for m in data["models"]}
    variant_of = is_variant_of_existing(slug, existing_names)
    if variant_of:
        # It's a variant — don't add a new model, just note the mapping
        return {"_variant_of": variant_of, "_slug": slug}

    lab = infer_lab(slug)
    if lab is None:
        return None  # Can't determine lab — skip

    # Check if lab exists in data
    if lab not in data["labs"]:
        return None  # Unknown lab not in our dataset — skip

    display_name = slug_to_display_name(slug)

    # Don't add if display name already exists
    if display_name in existing_names:
        return {"_variant_of": display_name, "_slug": slug}

    family = infer_family(display_name, lab, data["families"])
    is_open = LAB_OPEN_SOURCE.get(lab, False)
    if slug.startswith("muse-"):
        is_open = False

    return {
        "name": display_name,
        "lab": lab,
        "family": family or lab,
        "release_date": datetime.now().strftime("%Y-%m-%d"),
        "elo": score,
        "open_source": is_open,
        "plot": True,
        "_slug": slug,
    }


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "fetch-arena-elo/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def resolve_slug(slug: str, discovered: dict | None = None) -> str | None:
    """Map an arena slug to our display name, or None to skip."""
    if slug in SLUG_MAP:
        return SLUG_MAP[slug]
    if discovered and slug in discovered:
        return discovered[slug]
    for prefix, name in SLUG_PREFIX_MAP.items():
        if slug.startswith(prefix):
            return name
    return "__UNKNOWN__"


def fetch_snapshot(date: str, discovered: dict | None = None) -> tuple:
    url = f"{REPO_BASE}/{date}/text.json"
    raw = fetch_json(url)
    best: dict[str, int] = {}
    unmapped: list[tuple[str, int]] = []
    for entry in raw["models"]:
        slug = entry["model"]
        score = round(entry["score"])
        display = resolve_slug(slug, discovered)
        if display == "__UNKNOWN__":
            unmapped.append((slug, score))
            continue
        if display is None:
            continue
        if display not in best or score > best[display]:
            best[display] = score
    return best, unmapped


def estimate_drift(latest_date: str, discovered: dict | None = None, days_back: int = 7) -> int:
    """Compare two snapshots to estimate overall Elo scale drift."""
    dt = datetime.strptime(latest_date, "%Y-%m-%d")
    earlier_date = (dt - timedelta(days=days_back)).strftime("%Y-%m-%d")
    try:
        earlier, _ = fetch_snapshot(earlier_date, discovered)
    except Exception:
        return 0
    latest, _ = fetch_snapshot(latest_date, discovered)
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
    parser.add_argument("--no-discover", action="store_true", help="Disable new model discovery")
    args = parser.parse_args()

    with open(DATA_FILE) as f:
        data = json.load(f)

    discovered = load_discovered_slugs()

    if args.date:
        snap_date = args.date
    else:
        latest = fetch_json(f"{REPO_BASE}/latest.json")
        snap_date = latest["date"]

    print(f"Fetching arena snapshot for {snap_date}...")
    arena_scores, unmapped = fetch_snapshot(snap_date, discovered)

    # --- Discovery ---
    new_models = []
    new_variants = []
    still_unknown = []

    if not args.no_discover and unmapped:
        for slug, score in unmapped:
            result = discover_model(slug, score, data)
            if result is None:
                # Explicitly skipped (skip-prefix lab or unknown lab)
                continue
            elif "_variant_of" in result:
                new_variants.append(result)
            else:
                new_models.append(result)
                # Also record the score under the display name for the Elo update pass
                display = result["name"]
                if display not in arena_scores or score > arena_scores[display]:
                    arena_scores[display] = score
        # Anything in unmapped that wasn't handled by discover is truly unknown
        handled_slugs = {m["_slug"] for m in new_models + new_variants}
        still_unknown = [(s, sc) for s, sc in unmapped if s not in handled_slugs and not should_skip_slug(s)]
    else:
        still_unknown = unmapped

    if new_models:
        print(f"\n  Discovered {len(new_models)} new model(s):")
        for m in new_models:
            print(f"    + {m['name']:30s}  Elo {m['elo']:5d}  ({m['lab']}, family: {m['family']})")

    if new_variants:
        print(f"\n  Identified {len(new_variants)} new variant mapping(s):")
        for v in new_variants:
            print(f"    ~ {v['_slug']:40s} → {v['_variant_of']}")

    if still_unknown:
        print(f"\n  Still unknown arena slugs ({len(still_unknown)}):")
        for s, sc in still_unknown:
            print(f"    ? {s} (Elo {sc})")

    drift = estimate_drift(snap_date, discovered)
    if drift:
        print(f"\n  Estimated scale drift over last 7 days: {drift:+d}")

    # --- Update existing models ---
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

    print(f"\n{'DRY RUN' if not args.apply else 'APPLYING'} — {len(changes)} Elo updates:\n")
    for name, old, new, source in changes:
        tag = "" if source == "arena" else " (drift)"
        print(f"  {name:30s}  {old:5d} → {new:5d}{tag}")

    if unmatched:
        print(f"\n  Unmatched (no arena data, no drift): {len(unmatched)}")
        for n in unmatched:
            print(f"    {n}")

    # --- Apply new models to data ---
    if new_models:
        for m in new_models:
            entry = {k: v for k, v in m.items() if not k.startswith("_")}
            data["models"].append(entry)
            # Add to appropriate family
            family_name = m["family"]
            if family_name in data["families"]:
                if m["name"] not in data["families"][family_name]:
                    data["families"][family_name].append(m["name"])
            else:
                data["families"][family_name] = [m["name"]]

    mono_warnings = check_monotonicity(data["models"], data["families"])
    if mono_warnings:
        print(f"\n  Monotonicity warnings:")
        for w in mono_warnings:
            print(w)

    if args.apply and (changes or new_models):
        today = datetime.now().strftime("%Y-%m-%d")
        data["metadata"]["last_updated"] = today
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\n  Written to {DATA_FILE}")
        print(f"  last_updated set to {today}")

        # Persist discovered slug mappings
        new_discovered = dict(discovered)
        for m in new_models:
            new_discovered[m["_slug"]] = m["name"]
        for v in new_variants:
            new_discovered[v["_slug"]] = v["_variant_of"]
        if new_discovered != discovered:
            save_discovered_slugs(new_discovered)
            print(f"  Updated {DISCOVERED_FILE.name} with {len(new_discovered) - len(discovered)} new mapping(s)")
    elif not args.apply and (changes or new_models):
        print(f"\n  (use --apply to write changes)")


if __name__ == "__main__":
    main()
