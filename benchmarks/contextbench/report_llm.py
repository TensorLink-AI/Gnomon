"""Compare matched raw-LLM and compiled-context ContextBench runs."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    keyed = {str(row["case_id"]): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"duplicate case IDs in {path}")
    return keyed


def _exact_sign_p(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(positive, negative) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _bootstrap_ci(values: list[float], *, seed: int = 20260829,
                  draws: int = 10_000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choices(values, k=len(values)))
                       for _ in range(draws))
    return [estimates[int(0.025 * (draws - 1))],
            estimates[int(0.975 * (draws - 1))]]


def _paired_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["raw_context_smape"])
              - float(row["compiled_context_smape"]) for row in rows]
    positive = sum(value > 1e-12 for value in deltas)
    negative = sum(value < -1e-12 for value in deltas)
    return {
        "cases": len(rows),
        "raw_mean_context_smape": mean(
            float(row["raw_context_smape"]) for row in rows),
        "compiled_mean_context_smape": mean(
            float(row["compiled_context_smape"]) for row in rows),
        "mean_raw_minus_compiled_smape": mean(deltas),
        "median_raw_minus_compiled_smape": median(deltas),
        "mean_delta_95ci": _bootstrap_ci(deltas),
        "compiled_wins": positive,
        "ties": len(deltas) - positive - negative,
        "compiled_losses": negative,
        "exact_sign_p": _exact_sign_p(positive, negative),
        "raw_context_harm_rate": mean(
            float(row["raw_context_smape"]) >
            float(row["raw_history_smape"]) + 1e-12 for row in rows),
        "compiled_context_harm_rate": mean(
            float(row["compiled_context_smape"]) >
            float(row["compiled_history_smape"]) + 1e-12 for row in rows),
    }


def compare(raw_dir: str | Path, compiled_dir: str | Path) -> dict[str, Any]:
    raw_dir, compiled_dir = Path(raw_dir), Path(compiled_dir)
    raw_summary = json.loads((raw_dir / "summary.json").read_text())
    compiled_summary = json.loads((compiled_dir / "summary.json").read_text())
    if raw_summary.get("condition") != "raw-llm":
        raise ValueError("raw directory is not a raw-llm run")
    if compiled_summary.get("condition") != "compiled-context":
        raise ValueError("compiled directory is not a compiled-context run")
    corpus_hash = raw_summary.get("corpus_manifest_sha256")
    if not corpus_hash or compiled_summary.get("corpus_manifest_sha256") != corpus_hash:
        raise ValueError("runs do not use the same corpus manifest")
    raw = _read_rows(raw_dir / "observations.jsonl")
    compiled = _read_rows(compiled_dir / "observations.jsonl")
    if set(raw) != set(compiled):
        raise ValueError("runs do not contain the same case IDs")
    paired = []
    for case_id in sorted(raw):
        left, right = raw[case_id], compiled[case_id]
        if left.get("status") != "answered" or right.get("status") != "answered":
            raise ValueError(f"case {case_id} is not answered in both runs")
        if left.get("family") != right.get("family"):
            raise ValueError(f"case {case_id} family mismatch")
        paired.append({
            "case_id": case_id, "family": left["family"],
            "raw_context_smape": left["context_smape"],
            "raw_history_smape": left["history_smape"],
            "compiled_context_smape": right["context_smape"],
            "compiled_history_smape": right["history_smape"],
        })
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        by_family[str(row["family"])].append(row)
    return {
        "benchmark": "contextbench-llm-comparison",
        "schema_version": 1,
        "corpus_manifest_sha256": corpus_hash,
        "model": raw_summary.get("llm_usage", {}).get("model"),
        "reasoning_effort": raw_summary.get("reasoning_effort"),
        "narrative_style": raw_summary.get("narrative_style"),
        "overall": _paired_metrics(paired),
        "families": {name: _paired_metrics(rows)
                     for name, rows in sorted(by_family.items())},
        "compiler": {
            "calls": compiled_summary.get("compiler_calls"),
            "event_precision": compiled_summary.get("compiler_event_precision"),
            "event_recall": compiled_summary.get("compiler_event_recall"),
            "false_events": compiled_summary.get("compiler_false_events"),
        },
        "usage": {
            "raw": raw_summary.get("llm_usage_observations"),
            "compiled": compiled_summary.get("llm_usage_observations"),
        },
        "paired_rows": paired,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--compiled-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare(args.raw_dir, args.compiled_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "benchmark", "model", "overall", "families", "compiler", "usage")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
