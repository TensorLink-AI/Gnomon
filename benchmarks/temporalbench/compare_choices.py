"""Paired field-level comparison of TemporalBench choice answers.

The row summaries carry only counts.  Pairing those counts would pretend we
know which individual questions each arm fixed.  Detail receipts retain the
field-level verdicts, so this reporter joins exactly those fields and applies
an exact McNemar test overall and per tier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.common.manifest import incompatibilities, read_manifest
from benchmarks.report import mcnemar


def _fields(run: Path) -> dict[str, dict[str, bool]]:
    result = {}
    for path in sorted((run / "details").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        choice = (record.get("verdict") or {}).get("choice") or {}
        fields = choice.get("fields")
        if fields is None:
            fields = choice.get("per_question")
        if isinstance(fields, list):
            fields = {str(index): value for index, value in enumerate(fields)}
        if isinstance(fields, dict):
            result[path.stem] = {
                str(name): bool(value) for name, value in fields.items()
                if isinstance(value, bool)
            }
    return result


def compare(baseline_dir: Path, treatment_dir: Path) -> dict[str, Any]:
    left_manifest, right_manifest = (
        read_manifest(baseline_dir), read_manifest(treatment_dir))
    problems = incompatibilities(left_manifest, right_manifest)
    if problems:
        raise ValueError("; ".join(problems))
    left, right = _fields(baseline_dir), _fields(treatment_dir)
    shared_tasks = sorted(set(left) & set(right))
    if not shared_tasks:
        raise ValueError("no choice-bearing task ids in common")

    by_tier: dict[str, dict[str, Any]] = {}
    all_left: dict[str, bool] = {}
    all_right: dict[str, bool] = {}
    for tier in ("T1", "T2", "T3", "T4"):
        tier_left, tier_right = {}, {}
        for task_id in shared_tasks:
            if not task_id.endswith(f"_{tier}"):
                continue
            for field in sorted(set(left[task_id]) & set(right[task_id])):
                key = f"{task_id}:{field}"
                tier_left[key] = left[task_id][field]
                tier_right[key] = right[task_id][field]
        if tier_left:
            by_tier[tier] = _summary(tier_left, tier_right)
            all_left.update(tier_left)
            all_right.update(tier_right)
    if not all_left:
        raise ValueError("shared tasks contain no shared choice fields")
    return {
        "schema_version": "0.1",
        "baseline": str(baseline_dir),
        "treatment": str(treatment_dir),
        "baseline_code_revision": left_manifest.get("code_revision"),
        "treatment_code_revision": right_manifest.get("code_revision"),
        "matched_tasks": len(shared_tasks),
        "overall": _summary(all_left, all_right),
        "by_tier": by_tier,
    }


def _summary(left: dict[str, bool], right: dict[str, bool]) -> dict[str, Any]:
    n = len(left)
    return {
        "questions": n,
        "baseline_correct": sum(left.values()),
        "treatment_correct": sum(right.values()),
        "baseline_accuracy": sum(left.values()) / n,
        "treatment_accuracy": sum(right.values()) / n,
        "paired_test": mcnemar(left, right),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--treatment", required=True, type=Path)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare(args.baseline, args.treatment)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
