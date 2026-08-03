#!/usr/bin/env python3
"""Fetch manifest-selected benchmark observations from BenchmarkList.

Dry-run by default. Use --apply to write generated data, or --audit to print
repository coverage and quality-gate results.
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
BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_FILE = BASE_DIR / ".data" / "benchmarklist_manifest.json"
OUTPUT_DIR = BASE_DIR / ".data" / "benchmarklist"
LEGACY_FILE = BASE_DIR / ".data" / "benchmarklist_frontier_models.json"
COVERAGE_FILE = BASE_DIR / ".data" / "benchmarklist_coverage.json"


def fetch_json(url: str, attempts: int = 5) -> dict:
    """Fetch JSON with bounded retries for transient errors."""
    request = urllib.request.Request(url, headers={"User-Agent": "sota-reference-benchmark-fetch/2.0"})
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
    """Retain fields needed to interpret benchmark metrics."""
    return [
        {
            key: metric.get(key)
            for key in ("key", "label", "unit", "higher_is_better")
            if metric.get(key) is not None
        }
        for metric in (benchmark.get("metrics") or [])
    ]


def source_url(snapshot: dict, row: dict) -> str | None:
    """Return the observation's underlying source URL."""
    return row.get("source_url") or snapshot.get("source_url")


def row_identity(row: dict) -> set[str]:
    """Return stable identities that can match a manifest selection."""
    return {value for value in (row.get("model_id"), row.get("subject_id")) if value}


def select_rows(result: dict, selection: dict) -> list[tuple[dict, dict]]:
    """Select the newest eligible row for each requested subject identity."""
    selected = []
    for subject_id in selection["subject_ids"]:
        candidates = []
        for snapshot in result.get("snapshots", []):
            for row in snapshot.get("results", []):
                if subject_id not in row_identity(row):
                    continue
                if row.get("subject_type") not in selection["subject_types"]:
                    continue
                metric = selection["metric"]
                if metric not in row.get("metrics", {}) and row.get("score") is None:
                    continue
                canonical = row.get("model_id") == subject_id
                candidates.append((canonical, snapshot.get("sampled_at") or "", snapshot, row))
        if not candidates:
            raise ValueError(f"No eligible {subject_id} row for {result['benchmark_id']}")
        _, _, snapshot, row = max(candidates, key=lambda item: (item[0], item[1]))
        selected.append((snapshot, row))
    return selected


def validate_observation(benchmark: dict, snapshot: dict, row: dict, selection: dict) -> list[str]:
    """Apply manifest quality gates and return failure messages."""
    failures = []
    if benchmark.get("review_state") != "verified":
        failures.append(f"review_state={benchmark.get('review_state')!r}")
    if not snapshot.get("sampled_at"):
        failures.append("missing sampled_at")
    if not source_url(snapshot, row):
        failures.append("missing source_url")
    definitions = {item.get("key"): item for item in benchmark.get("metrics") or []}
    definition = definitions.get(selection["metric"])
    if not definition or "higher_is_better" not in definition:
        failures.append(f"metric {selection['metric']!r} lacks direction")
    if row.get("subject_type") not in selection["subject_types"]:
        failures.append(f"disallowed subject_type={row.get('subject_type')!r}")
    return failures


def clean_model(model: dict) -> dict:
    """Retain stable model identity, pricing, links, and provenance."""
    fields = (
        "model_id", "name", "developer", "developer_id", "family",
        "isReasoningModel", "aliases", "release_date", "cost", "openness",
        "links", "source_provenance", "notes", "model_type", "urls",
    )
    return {key: model[key] for key in fields if key in model}


def clean_observation(benchmark: dict, snapshot: dict, row: dict, selection: dict) -> dict:
    """Normalise one observation without combining benchmark metrics."""
    metadata = row.get("metadata", {})
    qualifiers = {
        key: metadata[key]
        for key in (
            "analysis_method", "reasoning_effort", "harness", "agent",
            "source_benchmark_label", "source_model_label", "source_table",
            "self_reported",
        )
        if key in metadata
    }
    metric = selection["metric"]
    metric_value = row.get("metrics", {}).get(metric)
    observation = {
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_name": benchmark["name"],
        "category": benchmark.get("category"),
        "review_state": benchmark.get("review_state"),
        "selected_metric": metric,
        "metric_definitions": metric_definitions(benchmark),
        "metric": metric_value,
        "score": row.get("score"),
        "rank": row.get("rank"),
        "raw_score": row.get("raw_score"),
        "model_id": row.get("model_id"),
        "subject_id": row.get("subject_id"),
        "subject_type": row.get("subject_type"),
        "display_name": row.get("display_name"),
        "qualifiers": qualifiers,
        "sampled_at": snapshot.get("sampled_at"),
        "source_url": source_url(snapshot, row),
        "source_type": row.get("source_type") or snapshot.get("source_type"),
        "self_reported": row.get("self_reported"),
        "verified": row.get("verified"),
        "benchmark_links": benchmark.get("links", {}),
        "benchmark_urls": benchmark.get("urls", {}),
    }
    return {key: value for key, value in observation.items() if value is not None and value != {}}


def discover_benchmark_articles() -> list[str]:
    """Find articles that contain benchmark, leaderboard, or evaluation claims."""
    topic_dirs = ("models", "rag", "agents", "multimodal", "media-generation", "evaluation")
    terms = ("benchmark", "leaderboard", " eval", "accuracy", " elo", "score")
    articles = []
    for topic_dir in topic_dirs:
        for path in sorted((BASE_DIR / topic_dir).glob("*.md")):
            text = path.read_text().lower()
            if any(term in text for term in terms):
                articles.append(str(path.relative_to(BASE_DIR)))
    return articles


def build_outputs(manifest: dict) -> tuple[dict[Path, dict], dict]:
    """Build generated article datasets and the coverage report."""
    models_resource = fetch_json(f"{API_BASE}/models.json")
    benchmarks_resource = fetch_json(f"{API_BASE}/benchmarks.json")
    model_map = {model["model_id"]: model for model in models_resource["models"]}
    benchmark_map = {item["benchmark_id"]: item for item in benchmarks_resource["benchmarks"]}
    result_cache = {}
    outputs = {}
    coverage_articles = {}

    for article, config in manifest["articles"].items():
        status = config["status"]
        if status != "migrated":
            coverage_articles[article] = {"status": status, "reason": config.get("reason")}
            continue

        observations = []
        quality_failures = []
        for selection in config["observations"]:
            benchmark_id = selection["benchmark_id"]
            if benchmark_id not in benchmark_map:
                raise ValueError(f"Unknown benchmark_id: {benchmark_id}")
            benchmark = benchmark_map[benchmark_id]
            if benchmark_id not in result_cache:
                result_cache[benchmark_id] = fetch_json(f"{API_BASE}/results/{benchmark_id}.json")
            for snapshot, row in select_rows(result_cache[benchmark_id], selection):
                failures = validate_observation(benchmark, snapshot, row, selection)
                if failures:
                    quality_failures.append({
                        "benchmark_id": benchmark_id,
                        "subject_id": row.get("subject_id"),
                        "failures": failures,
                    })
                    continue
                observations.append(clean_observation(benchmark, snapshot, row, selection))

        if quality_failures:
            raise ValueError(f"Quality gate failures for {article}: {quality_failures}")

        models = []
        for model_id in config.get("model_ids", []):
            if model_id not in model_map:
                raise ValueError(f"Unknown model_id: {model_id}")
            models.append(clean_model(model_map[model_id]))

        output = {
            "metadata": {
                "article": article,
                "description": "BenchmarkList observations and source provenance selected for this article",
                "last_updated": date.today().isoformat(),
                "api_version": "1",
                "quality_policy": manifest["metadata"]["quality_policy"],
                "notes": "BenchmarkList indexes source observations; it does not independently rerun every evaluation. Do not combine unrelated raw metrics.",
            },
            "models": models,
            "observations": sorted(
                observations,
                key=lambda item: (item["benchmark_id"], item.get("subject_id", "")),
            ),
        }
        output_path = OUTPUT_DIR / config["output"]
        outputs[output_path] = output
        coverage_articles[article] = {
            "status": "migrated",
            "output": str(output_path.relative_to(BASE_DIR)),
            "observation_count": len(observations),
            "benchmark_count": len({item["benchmark_id"] for item in observations}),
            "quality_failures": 0,
        }

    discovered_articles = discover_benchmark_articles()
    known_articles = set(coverage_articles) | set(manifest.get("later_wave", []))
    unclassified = sorted(set(discovered_articles) - known_articles)
    coverage = {
        "metadata": {
            "description": "Repository BenchmarkList migration and quality-gate coverage",
            "last_updated": date.today().isoformat(),
        },
        "articles": coverage_articles,
        "later_wave": manifest.get("later_wave", []),
        "unclassified_benchmark_articles": unclassified,
        "summary": {
            "benchmark_articles_discovered": len(discovered_articles),
            "migrated": sum(item["status"] == "migrated" for item in coverage_articles.values()),
            "blocked": sum(item["status"] == "blocked" for item in coverage_articles.values()),
            "later_wave": len(manifest.get("later_wave", [])),
            "unclassified": len(unclassified),
        },
    }
    outputs[COVERAGE_FILE] = coverage
    return outputs, coverage


def render(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def show_diff(path: Path, new_text: str) -> bool:
    """Print a unified diff and return whether the file changed."""
    old_text = path.read_text() if path.exists() else ""
    if new_text == old_text:
        print(f"No changes: {path}")
        return False
    diff = difflib.unified_diff(
        old_text.splitlines(), new_text.splitlines(),
        fromfile=str(path), tofile=str(path), lineterm="",
    )
    print("\n".join(diff))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    parser.add_argument("--audit", action="store_true", help="Print coverage and quality summary")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_FILE.read_text())
    outputs, coverage = build_outputs(manifest)
    changed = []
    for path, data in outputs.items():
        text = render(data)
        if show_diff(path, text):
            changed.append((path, text))

    if args.audit:
        print("\nCoverage summary:")
        print(json.dumps(coverage["summary"], indent=2))
        for article, item in coverage["articles"].items():
            print(f"  {item['status']:8s} {article}")

    if args.apply:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for path, text in changed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"Written to {path}")
        if LEGACY_FILE.exists():
            LEGACY_FILE.unlink()
            print(f"Removed legacy file {LEGACY_FILE}")
    elif changed:
        print("\nDry run only; use --apply to write changes.")


if __name__ == "__main__":
    main()
