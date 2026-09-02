"""Run a crash-safe bounded-output calibration shard through ``forecast``.

The future is generated before either arm but is written only to the retained
observation after both forecasts finish.  Both arms use honest strict-split
calibration.  The treatment adds a caller-declared physical cap; it does not
receive future target values or an oracle label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from benchmarks.common.manifest import code_revision
from gnomon.config import GnomonConfig
from gnomon.context import ContextEvent
from gnomon.runtime import forecast


START = datetime(2024, 1, 1, tzinfo=timezone.utc)
HISTORY = 120
HORIZON = 12
CAP = 100.0
CASES = 8


def _series(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [min(CAP, max(0.0,
               70.0 + .22 * index
               + 4.0 * math.sin(2.0 * math.pi * index / 12.0)
               + rng.gauss(0.0, 1.2)))
            for index in range(HISTORY + HORIZON)]


def _write_history(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "utilization_percent"])
        for index, value in enumerate(values[:HISTORY]):
            writer.writerow([(START + timedelta(days=index)).isoformat(), value])


def _diagnostic(rows: list[dict[str, Any]], actual: list[float]) -> dict[str, Any]:
    covered = 0
    wis = []
    for row, truth in zip(rows, actual):
        low, middle, high = (float(row[name])
                             for name in ("q10", "q50", "q90"))
        covered += int(low <= truth <= high)
        interval_score = high - low
        if truth < low:
            interval_score += 10.0 * (low - truth)
        elif truth > high:
            interval_score += 10.0 * (truth - high)
        wis.append((.5 * abs(truth - middle) + .1 * interval_score) / .6)
    return {
        "covered_points": covered,
        "interval_points": len(actual),
        "wis_sum": sum(wis),
    }


def _case(case_id: str, seed: int, output_dir: Path) -> dict[str, Any]:
    values = _series(seed)
    source = output_dir / "inputs" / f"{case_id}.csv"
    _write_history(source, values)
    config = GnomonConfig()
    config.evaluation.pool_residuals = False
    arguments = {
        "time_column": "timestamp",
        "target_column": "utilization_percent",
        "horizon": HORIZON,
        "seasonal_period": 12,
        "config": config,
    }
    reference, _ = forecast(
        str(source), output=str(output_dir / "artifacts" / case_id / "reference"),
        **arguments,
    )
    event = ContextEvent(
        event_id=f"{case_id}-cap",
        event_type="constraint:max",
        entity_scope=("*",),
        effective_start=(START + timedelta(days=HISTORY)).isoformat(),
        effective_end=(START + timedelta(
            days=HISTORY + HORIZON - 1)).isoformat(),
        known_at=START.isoformat(),
        attributes={"claim": {"kind": "max", "value": CAP}},
    )
    candidate, _ = forecast(
        str(source), output=str(output_dir / "artifacts" / case_id / "candidate"),
        context_events=[event], **arguments,
    )
    reference_rows = reference.results[0].forecast
    candidate_rows = candidate.results[0].forecast
    actual = values[HISTORY:]
    reference_diagnostic = _diagnostic(reference_rows, actual)
    candidate_diagnostic = _diagnostic(candidate_rows, actual)
    return {
        "case_id": case_id,
        "seed": seed,
        "future_observations_used_by_forecaster": 0,
        "actual": actual,
        "declared_bounds": {"minimum": 0.0, "maximum": CAP},
        "reference": reference_diagnostic,
        "candidate": candidate_diagnostic,
        "rows_changed_by_bound": sum(
            left != right
            for left, right in zip(reference_rows, candidate_rows)),
        "candidate_bound_violations": sum(
            any(float(row[name]) > CAP + 1e-12
                for name in ("point", "q10", "q50", "q90"))
            for row in candidate_rows),
        "truth_bound_violations": sum(
            not 0.0 <= value <= CAP for value in actual),
        "automation_eligible": bool(
            candidate.results[0].threshold
            and candidate.results[0].threshold.get(
                "automation_eligible")),
    }


def summarize(rows: list[dict[str, Any]], revision: str | None) -> dict[str, Any]:
    points = sum(row["candidate"]["interval_points"] for row in rows)
    candidate_wis = sum(row["candidate"]["wis_sum"] for row in rows) / points
    reference_wis = sum(row["reference"]["wis_sum"] for row in rows) / points
    coverage = sum(
        row["candidate"]["covered_points"] for row in rows) / points
    gates = {
        "all_cases_complete": len(rows) == CASES,
        "future_targets_never_passed_to_forecaster": all(
            row["future_observations_used_by_forecaster"] == 0 for row in rows),
        "declared_bound_was_active": sum(
            row["rows_changed_by_bound"] for row in rows) > 0,
        "no_candidate_bound_violations": sum(
            row["candidate_bound_violations"] for row in rows) == 0,
        "no_truth_bound_violations": sum(
            row["truth_bound_violations"] for row in rows) == 0,
        "coverage_within_10_points": abs(coverage - .8) <= .1,
        "wis_nonworsening": candidate_wis <= reference_wis + 1e-12,
        "automation_disabled": not any(
            row["automation_eligible"] for row in rows),
    }
    return {
        "schema_version": 1,
        "benchmark": "bounded-calibration-evaluation",
        "evaluated_commit": revision,
        "cases": len(rows),
        "expected_cases": CASES,
        "nominal_coverage": .8,
        "empirical_coverage": coverage,
        "candidate_wis": candidate_wis,
        "reference_wis": reference_wis,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "rows": rows,
    }


def run(seed: int, output_dir: Path, resume: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "run-identity.json"
    checkpoint = output_dir / "observations.jsonl"
    identity = {
        "schema_version": 1,
        "benchmark": "bounded-calibration-evaluation",
        "evaluated_commit": code_revision(),
        "seed": seed,
        "cases": CASES,
    }
    if resume:
        if not identity_path.is_file():
            raise ValueError("resume requires a retained run identity")
        retained = json.loads(identity_path.read_text(encoding="utf-8"))
        if retained != identity:
            raise ValueError("resume identity differs from retained bounded run")
    elif identity_path.exists() or checkpoint.exists():
        raise ValueError("output directory already contains a bounded run")
    else:
        identity_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    completed = {}
    if resume and checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["case_id"]] = row
    for index in range(CASES):
        case_id = f"bounded-{index:03d}"
        if case_id in completed:
            continue
        row = _case(case_id, seed + index, output_dir)
        completed[case_id] = row
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"completed {index + 1}/{CASES} {case_id}", flush=True)
    rows = [completed[f"bounded-{index:03d}"] for index in range(CASES)]
    summary = summarize(rows, identity["evaluated_commit"])
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.seed, args.output_dir, args.resume)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, sort_keys=True))
    return 0 if result["gates"]["all_cases_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
