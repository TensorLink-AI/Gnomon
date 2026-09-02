"""Run the frozen categorical-state calibration matrix.

The future target is generated with the history but is not passed to the
candidate.  A second revision can be compared with ``--reference-summary``;
the comparison refuses a different seed or case matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from benchmarks.common.manifest import code_revision
from gnomon.context_intelligence import fit_categorical_state_candidate


FAMILIES = (
    "balanced_effect",
    "state_dependent_noise",
    "seasonal_interaction",
    "irrelevant_states",
    "sparse_future_state",
    "unseen_future_state",
)
CASES_PER_FAMILY = 30
HISTORY = 96
HORIZON = 16
PERIOD = 8


def _states(rng: random.Random, family: str) -> tuple[list[str], list[str]]:
    if family == "sparse_future_state":
        history = ["rare" if index in {8, 24, 40, 56, 72, 88} else "usual"
                   for index in range(HISTORY)]
        return history, ["rare"] * HORIZON
    if family == "unseen_future_state":
        return [rng.choice(("a", "b")) for _ in range(HISTORY)], ["new"] * HORIZON
    history = [rng.choice(("a", "b")) for _ in range(HISTORY)]
    if family == "state_dependent_noise":
        future = ["a" if index % 4 else "b" for index in range(HORIZON)]
    else:
        future = [rng.choice(("a", "b")) for _ in range(HORIZON)]
    return history, future


def _target(rng: random.Random, family: str,
            states: list[str]) -> list[float]:
    values = []
    for index, state in enumerate(states):
        phase = 12.0 * math.sin(2.0 * math.pi * index / PERIOD)
        if family == "balanced_effect":
            value = 50.0 + (8.0 if state == "a" else -8.0) + rng.gauss(0, 1.5)
        elif family == "state_dependent_noise":
            value = (50.0 + (10.0 if state == "b" else 0.0)
                     + rng.gauss(0, 8.0 if state == "b" else 1.0))
        elif family == "seasonal_interaction":
            active = 10.0 if state == "a" and index % PERIOD in {1, 2, 3} else 0.0
            value = 50.0 + phase + active + rng.gauss(0, 1.5)
        elif family == "irrelevant_states":
            value = 50.0 + phase + rng.gauss(0, 2.0)
        elif family == "sparse_future_state":
            value = 50.0 + (6.0 if state == "rare" else 0.0) + rng.gauss(0, 2.0)
        else:
            value = 50.0 + phase + rng.gauss(0, 2.0)
        values.append(value)
    return values


def _wis(actual: float, low: float, middle: float, high: float) -> float:
    interval = high - low
    if actual < low:
        interval += 10.0 * (low - actual)
    elif actual > high:
        interval += 10.0 * (actual - high)
    return (.5 * abs(actual - middle) + .1 * interval) / .6


def _case(family: str, case_index: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    history_states, future_states = _states(rng, family)
    all_states = [*history_states, *future_states]
    complete = _target(rng, family, all_states)
    history, actual = complete[:HISTORY], complete[HISTORY:]
    primary = [{"timestamp": f"future-{index:02d}"}
               for index in range(HORIZON)]
    candidate = fit_categorical_state_candidate(
        history, history_states, future_states, primary=primary,
        claim_ids=["known-state-schedule"],
        hypothesis_id=f"{family}-{case_index:03d}",
        replay_origin_eligible=[True] * HISTORY,
        seasonal_period=(PERIOD if family in {
            "seasonal_interaction", "irrelevant_states",
            "unseen_future_state"} else None),
    )
    rows = candidate["forecast"]
    ordered = all(
        all(math.isfinite(float(row[key])) for key in ("q10", "q50", "q90"))
        and float(row["q10"]) <= float(row["q50"]) <= float(row["q90"])
        for row in rows)
    scores = [_wis(truth, float(row["q10"]), float(row["q50"]),
                   float(row["q90"]))
              for truth, row in zip(actual, rows)]
    covered = sum(float(row["q10"]) <= truth <= float(row["q90"])
                  for truth, row in zip(actual, rows))
    return {
        "case_id": f"{family}-{case_index:03d}",
        "family": family,
        "seed": seed,
        "future_observations_used_by_forecaster": 0,
        "ordered_quantiles": ordered,
        "mean_wis": statistics.mean(scores),
        "covered_points": covered,
        "interval_points": HORIZON,
        "numeric_rows": [[float(row[key]) for key in ("q10", "q50", "q90")]
                         for row in rows],
        "contract": {
            "support": candidate["support"],
            "selection_eligible": candidate["selection_eligible"],
            "human_selection_eligible": candidate["human_selection_eligible"],
            "automation_eligible": candidate["automation_eligible"],
            "primary_forecast_unchanged": candidate["primary_forecast_unchanged"],
        },
        "interval_calibration": candidate["validation"]["interval_calibration"],
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    points = sum(int(row["interval_points"]) for row in rows)
    return {
        "cases": len(rows),
        "coverage": sum(int(row["covered_points"]) for row in rows) / points,
        "mean_wis": statistics.mean(float(row["mean_wis"]) for row in rows),
        "median_wis": statistics.median(float(row["mean_wis"]) for row in rows),
    }


def summarize(rows: list[dict[str, Any]], identity: dict[str, Any],
              reference: dict[str, Any] | None = None) -> dict[str, Any]:
    by_family = {family: _aggregate(
        [row for row in rows if row["family"] == family]) for family in FAMILIES}
    overall = _aggregate(rows)
    base_gates = {
        "all_cases_complete": len(rows) == len(FAMILIES) * CASES_PER_FAMILY,
        "future_targets_never_passed_to_forecaster": all(
            row["future_observations_used_by_forecaster"] == 0 for row in rows),
        "quantiles_finite_and_ordered": all(row["ordered_quantiles"] for row in rows),
    }
    comparison: dict[str, Any] | None = None
    if reference is not None:
        expected = reference.get("run_identity") or {}
        for key in ("seed", "families", "cases_per_family", "history", "horizon"):
            if expected.get(key) != identity.get(key):
                raise ValueError(f"reference identity differs on {key}")
        reference_rows = {row["case_id"]: row for row in reference.get("rows") or []}
        if set(reference_rows) != {row["case_id"] for row in rows}:
            raise ValueError("reference case matrix differs")
        reference_family = reference["by_family"]
        family_ratio = {
            family: ((by_family[family]["median_wis"]
                      - reference_family[family]["median_wis"])
                     / max(reference_family[family]["median_wis"], 1e-12))
            for family in FAMILIES
        }
        invariant_contract = all(
            row["contract"] == reference_rows[row["case_id"]]["contract"]
            for row in rows)
        sparse_numeric_equal = all(
            row["numeric_rows"] == reference_rows[row["case_id"]]["numeric_rows"]
            for row in rows if row["family"] in {
                "sparse_future_state", "unseen_future_state"})
        gates = {
            **base_gates,
            "contract_decisions_unchanged": invariant_contract,
            "sparse_and_unseen_numerically_unchanged": sparse_numeric_equal,
            "aggregate_wis_nonworsening": overall["mean_wis"]
                <= float(reference["overall"]["mean_wis"]) + 1e-12,
            "family_median_regression_within_2pct": max(family_ratio.values()) <= .02,
            "state_dependent_noise_median_wis_improves_5pct":
                family_ratio["state_dependent_noise"] <= -.05,
            "coverage_within_10_points": abs(overall["coverage"] - .8) <= .1,
        }
        comparison = {"family_median_wis_relative_change": family_ratio}
    else:
        gates = base_gates
    return {
        "schema_version": 1,
        "benchmark": "categorical-state-calibration",
        "evaluated_commit": identity["evaluated_commit"],
        "run_identity": identity,
        "overall": overall,
        "by_family": by_family,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        **({"comparison": comparison} if comparison is not None else {}),
        "rows": rows,
    }


def run(seed: int, output_dir: Path, *, resume: bool = False,
        reference_summary: Path | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "run-identity.json"
    checkpoint = output_dir / "observations.jsonl"
    identity = {
        "schema_version": 1,
        "benchmark": "categorical-state-calibration",
        "evaluated_commit": code_revision(),
        "seed": seed,
        "families": list(FAMILIES),
        "cases_per_family": CASES_PER_FAMILY,
        "history": HISTORY,
        "horizon": HORIZON,
    }
    if resume:
        if not identity_path.is_file():
            raise ValueError("resume requires a retained run identity")
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("resume identity differs from retained run")
    elif identity_path.exists() or checkpoint.exists():
        raise ValueError("output directory already contains a run")
    else:
        identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    completed: dict[str, dict[str, Any]] = {}
    if resume and checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["case_id"]] = row
    ordinal = 0
    for family_index, family in enumerate(FAMILIES):
        for case_index in range(CASES_PER_FAMILY):
            case_id = f"{family}-{case_index:03d}"
            ordinal += 1
            if case_id in completed:
                continue
            row = _case(family, case_index,
                        seed + family_index * 100_000 + case_index)
            completed[case_id] = row
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
            print(f"completed {ordinal}/{len(FAMILIES) * CASES_PER_FAMILY} {case_id}",
                  flush=True)
    rows = [completed[f"{family}-{index:03d}"]
            for family in FAMILIES for index in range(CASES_PER_FAMILY)]
    reference = (json.loads(reference_summary.read_text(encoding="utf-8"))
                 if reference_summary is not None else None)
    result = summarize(rows, identity, reference)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-summary", type=Path)
    args = parser.parse_args()
    result = run(args.seed, args.output_dir, resume=args.resume,
                 reference_summary=args.reference_summary)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     indent=2, sort_keys=True))
    return 0 if result["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
