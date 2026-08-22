"""Pool independent PropertyBench seeds without hiding a weak seed.

One deterministic seed landing just below a gate is not evidence that a
mechanism is broken; selecting a friendlier seed is not evidence that it is
sound.  This reporter keeps every constituent score visible and pools the
row-level held-out classifications before applying the precommitted gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmarks.common.manifest import code_revision, read_manifest


def aggregate(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two independent PropertyBench runs are required")
    runs = []
    rows: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for path in paths:
        directory = path if path.is_dir() else path.parent
        summary_path = directory / "summary.json" if path.is_dir() else path
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = read_manifest(directory)
        if manifest and manifest.get("benchmark") != "propertybench":
            raise ValueError(f"{directory}: not a PropertyBench manifest")
        seed = int(summary["seed"])
        if seed in seen_seeds:
            raise ValueError(f"duplicate seed {seed}")
        seen_seeds.add(seed)
        lane = summary["future_process_volatility"]
        lane_rows = lane.get("rows")
        if not isinstance(lane_rows, list) or not lane_rows:
            raise ValueError(f"{summary_path}: future-process rows are missing")
        rows.extend(lane_rows)
        runs.append({
            "path": str(directory),
            "seed": seed,
            "code_revision": manifest.get("code_revision"),
            "replicates": manifest.get("replicates"),
            "cases": lane["cases"],
            "balanced_accuracy": lane["balanced_accuracy"],
            "graduated": summary["graduated"],
        })

    revisions = {run["code_revision"] for run in runs if run["code_revision"]}
    if len(revisions) > 1:
        raise ValueError(f"runs use different code revisions: {sorted(revisions)}")
    labels = ("increased", "decreased", "stable")
    recalls = {
        label: statistics.mean(bool(row["correct"]) for row in rows
                               if row["expected"] == label)
        for label in labels
    }
    balanced = statistics.mean(recalls.values())
    return {
        "schema_version": "0.1",
        "benchmark": "propertybench",
        "lane": "future_process_volatility",
        "constituent_runs": runs,
        "independent_seeds": len(seen_seeds),
        "cases": len(rows),
        "class_recall": recalls,
        "balanced_accuracy": balanced,
        "gate": {
            "name": "balanced_accuracy_at_least_55pct",
            "threshold": .55,
            "passed": balanced >= .55,
        },
        "evaluated_code_revision": next(iter(revisions), None),
        "summarized_by_code_revision": code_revision(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = aggregate(args.runs)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
