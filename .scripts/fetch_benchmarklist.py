#!/usr/bin/env python3
"""Fetch focused GPT-5.6 metadata and benchmark observations from BenchmarkList.

Dry-run by default. Use --apply to update the tracked JSON file.
"""

import argparse
import difflib
import json
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

API_BASE = "https://benchmarklist.com/api/v1"
DATA_FILE = Path(__file__).resolve().parent.parent / ".data" / "benchmarklist_frontier_models.json"
MODEL_IDS = (
    "openai-gpt-5.6-luna",
    "openai-gpt-5.6-terra",
    "openai-gpt-5.6-sol",
)
BENCHMARK_IDS = (
    "agents_last_exam",
    "automationbench",
    "artificial_analysis_coding_agent_index",
    "toolathlon",
    "graphwalks_bfs_256k_f1",
    "openai_mrcr_v2_8_needle_512k_1m",
    "osworld2_0_benchmarking_computer_use_agents_on_long_horizon_real_world_tasks",
)
LUNA_ID = MODEL_IDS[0]


def fetch_json(url: str, attempts: int = 5) -> dict:
    """Fetch JSON with bounded retries for transient rate limits."""
    request = urllib.request.Request(url, headers={"User-Agent": "sota-reference-benchmark-fetch/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Unable to fetch {url}")


def metric_definitions(benchmark: dict) -> list[dict]:
    """Retain the fields needed to interpret benchmark metrics."""
    return [
        {
            key: metric.get(key)
            for key in ("key", "label", "unit", "higher_is_better")
            if metric.get(key) is not None
        }
        for metric in benchmark.get("metrics", [])
    ]


def find_luna_row(snapshot: dict) -> dict | None:
    """Find a direct Luna model row, excluding variants, agents, and systems."""
    for row in snapshot.get("results", []):
        if row.get("subject_type") != "model":
            continue
        if row.get("model_id") == LUNA_ID or row.get("subject_id") in (LUNA_ID, "gpt-5.6-luna"):
            return row
    return None


def select_observation(result_resource: dict) -> tuple[dict, dict]:
    """Select the newest direct-model observation with canonical ID preferred."""
    candidates = []
    for snapshot in result_resource.get("snapshots", []):
        row = find_luna_row(snapshot)
        if row:
            canonical = row.get("model_id") == LUNA_ID
            candidates.append((canonical, snapshot.get("sampled_at") or "", snapshot, row))
    if not candidates:
        raise ValueError(f"No direct {LUNA_ID} row for {result_resource['benchmark_id']}")
    _, _, snapshot, row = max(candidates, key=lambda item: (item[0], item[1]))
    return snapshot, row


def clean_model(model: dict) -> dict:
    """Retain stable model identity, pricing, links, and release provenance."""
    fields = (
        "model_id", "name", "developer", "developer_id", "family",
        "isReasoningModel", "aliases", "release_date", "cost", "openness",
        "links", "source_provenance", "notes", "model_type", "urls",
    )
    return {key: model[key] for key in fields if key in model}


def clean_observation(benchmark: dict, snapshot: dict, row: dict) -> dict:
    """Normalise one source observation without combining benchmark metrics."""
    metadata = row.get("metadata", {})
    qualifiers = {
        key: metadata[key]
        for key in (
            "analysis_method", "reasoning_effort", "source_benchmark_label",
            "source_model_label", "source_table", "self_reported",
        )
        if key in metadata
    }
    observation = {
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_name": benchmark["name"],
        "category": benchmark.get("category"),
        "primary_metric": benchmark.get("primary_metric"),
        "metric_definitions": metric_definitions(benchmark),
        "score": row.get("score"),
        "rank": row.get("rank"),
        "raw_score": row.get("raw_score"),
        "metrics": row.get("metrics", {}),
        "model_id": row.get("model_id", LUNA_ID),
        "subject_id": row.get("subject_id"),
        "subject_type": row.get("subject_type"),
        "display_name": row.get("display_name"),
        "qualifiers": qualifiers,
        "sampled_at": snapshot.get("sampled_at"),
        "source_url": row.get("source_url") or snapshot.get("source_url"),
        "source_type": row.get("source_type") or snapshot.get("source_type"),
        "self_reported": row.get("self_reported"),
        "verified": row.get("verified"),
        "benchmark_urls": benchmark.get("urls", {}),
    }
    return {key: value for key, value in observation.items() if value is not None and value != {}}


def build_data() -> dict:
    """Fetch source collections and build the focused tracked dataset."""
    models_resource = fetch_json(f"{API_BASE}/models.json")
    benchmarks_resource = fetch_json(f"{API_BASE}/benchmarks.json")
    model_map = {model["model_id"]: model for model in models_resource["models"]}
    benchmark_map = {
        benchmark["benchmark_id"]: benchmark
        for benchmark in benchmarks_resource["benchmarks"]
    }

    missing_models = sorted(set(MODEL_IDS) - set(model_map))
    missing_benchmarks = sorted(set(BENCHMARK_IDS) - set(benchmark_map))
    if missing_models or missing_benchmarks:
        raise ValueError(
            f"Missing API records: models={missing_models}, benchmarks={missing_benchmarks}"
        )

    observations = []
    for benchmark_id in BENCHMARK_IDS:
        benchmark = benchmark_map[benchmark_id]
        result_resource = fetch_json(f"{API_BASE}/results/{benchmark_id}.json")
        snapshot, row = select_observation(result_resource)
        observations.append(clean_observation(benchmark, snapshot, row))

    return {
        "metadata": {
            "description": "Focused BenchmarkList metadata and source observations for the OpenAI GPT-5.6 family",
            "last_updated": date.today().isoformat(),
            "api_version": "1",
            "sources": [
                f"{API_BASE}/manifest.json",
                f"{API_BASE}/models.json",
                f"{API_BASE}/benchmarks.json",
                "https://benchmarklist.com/models/openai-gpt-5.6-luna/",
            ],
            "model_ids": list(MODEL_IDS),
            "benchmark_ids": list(BENCHMARK_IDS),
            "notes": "Each benchmark remains a separate source observation. Raw scores are not combined across benchmarks. Direct model rows are selected; model variants, agents, and systems are excluded.",
        },
        "models": [clean_model(model_map[model_id]) for model_id in MODEL_IDS],
        "observations": observations,
    }


def render(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    args = parser.parse_args()

    new_text = render(build_data())
    old_text = DATA_FILE.read_text() if DATA_FILE.exists() else ""
    if new_text == old_text:
        print(f"No changes: {DATA_FILE}")
        return

    diff = difflib.unified_diff(
        old_text.splitlines(), new_text.splitlines(),
        fromfile=str(DATA_FILE), tofile=str(DATA_FILE), lineterm="",
    )
    print("\n".join(diff))
    if args.apply:
        DATA_FILE.write_text(new_text)
        print(f"\nWritten to {DATA_FILE}")
    else:
        print("\nDry run only; use --apply to write changes.")


if __name__ == "__main__":
    main()
