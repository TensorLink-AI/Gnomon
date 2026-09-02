"""Run the frozen graduated-horizon publication matrix.

The matrix asks the production forecast surface for a horizon longer than the
history can evaluate at full rigor.  Future target values are generated first
but never written to the input file or passed to Gnomon.  They are used only
after publication to score whether an evaluated prefix plus a labelled
best-effort tail is more useful than a flat degraded full-horizon baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from benchmarks.common.manifest import code_revision
from gnomon.toolspec import runner_for


FAMILIES = ("trend", "seasonal", "level", "random_walk")
HISTORY = 48
HORIZON = 24
PERIOD = 24
CASES_PER_FAMILY = 20


def _wis(actual: float, low: float, middle: float, high: float) -> float:
    interval = high - low
    if actual < low:
        interval += 10.0 * (low - actual)
    elif actual > high:
        interval += 10.0 * (actual - high)
    return (.5 * abs(actual - middle) + .1 * interval) / .6


def _series(family: str, seed: int) -> tuple[list[float], list[float], str]:
    rng = random.Random(seed)
    total = HISTORY + HORIZON
    if family == "trend":
        slope = rng.uniform(.25, .75)
        values = [40.0 + slope * index + rng.gauss(0.0, .7)
                  for index in range(total)]
        return values[:HISTORY], values[HISTORY:], "D"
    if family == "seasonal":
        amplitude = rng.uniform(7.0, 14.0)
        phase = rng.uniform(-.25, .25)
        values = [
            30.0 + amplitude * math.sin(
                2.0 * math.pi * (index / PERIOD + phase))
            + rng.gauss(0.0, .7)
            for index in range(total)
        ]
        return values[:HISTORY], values[HISTORY:], "h"
    if family == "level":
        level = rng.uniform(30.0, 70.0)
        values = [level + rng.gauss(0.0, 1.5) for _ in range(total)]
        return values[:HISTORY], values[HISTORY:], "D"
    if family == "random_walk":
        values = [rng.uniform(30.0, 70.0)]
        for _ in range(1, total):
            values.append(values[-1] + rng.gauss(0.0, 1.0))
        return values[:HISTORY], values[HISTORY:], "D"
    raise ValueError(f"unknown family: {family}")


def _case(family: str, case_index: int, seed: int) -> dict[str, Any]:
    history, future, frequency = _series(family, seed)
    start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory(prefix="gnomon-graduated-") as raw:
        root = Path(raw)
        source = root / "history.csv"
        step = timedelta(hours=1) if frequency == "h" else timedelta(days=1)
        source.write_text("timestamp,value\n" + "\n".join(
            f"{(start + step * index).isoformat()},{value}"
            for index, value in enumerate(history)) + "\n", encoding="utf-8")
        arguments: dict[str, Any] = {
            "input": str(source), "time": "timestamp", "target": "value",
            "frequency": frequency, "horizon": HORIZON,
            "output_dir": str(root / "output"),
        }
        if family == "seasonal":
            arguments["seasonal_period"] = PERIOD
        payload = runner_for("gnomon_forecast")(arguments)
        artifact = json.loads((Path(payload["artifact_path"]) /
                               "artifact.json").read_text(encoding="utf-8"))
    result = artifact["results"][0]
    rows = result["forecast"]
    reasons = [str(item.get("code")) for item in
               (result.get("support_assessment") or {}).get("reasons") or []]
    numeric = [[float(row[key]) for key in ("q10", "q50", "q90")]
               for row in rows]
    scores = [_wis(actual, *quantiles)
              for actual, quantiles in zip(future, numeric)]
    points = [row[1] for row in numeric]
    tiers = [str(row.get("tier")) for row in rows]
    prefix = 0
    for tier in tiers:
        if tier == "best_effort":
            break
        prefix += 1
    return {
        "case_id": f"{family}-{case_index:03d}",
        "family": family,
        "seed": seed,
        "future_observations_used_by_forecaster": 0,
        "support": result.get("support"),
        "tier_floor": payload.get("tier_floor"),
        "selected_model": result.get("selected_model"),
        "horizon_split": "horizon_split" in reasons,
        "evaluated_prefix_steps": prefix,
        "best_effort_tail_steps": sum(tier == "best_effort" for tier in tiers),
        "rows": len(rows),
        "points": points,
        "quantiles": numeric,
        "tiers": tiers,
        "ordered_quantiles": all(
            low <= middle <= high
            and all(math.isfinite(value) for value in (low, middle, high))
            for low, middle, high in numeric),
        "flat_point_path": len(set(round(value, 12) for value in points)) == 1,
        "mean_absolute_error": statistics.mean(
            abs(actual - point) for actual, point in zip(future, points)),
        "mean_wis": statistics.mean(scores),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
        "mean_wis": statistics.mean(float(row["mean_wis"]) for row in rows),
        "median_wis": statistics.median(float(row["mean_wis"]) for row in rows),
        "mean_absolute_error": statistics.mean(
            float(row["mean_absolute_error"]) for row in rows),
        "split_rate": statistics.mean(bool(row["horizon_split"]) for row in rows),
    }


def summarize(rows: list[dict[str, Any]], identity: dict[str, Any],
              reference: dict[str, Any] | None = None) -> dict[str, Any]:
    by_family = {family: _aggregate(
        [row for row in rows if row["family"] == family])
        for family in FAMILIES}
    overall = _aggregate(rows)
    base_gates = {
        "all_cases_complete": len(rows) == (
            len(FAMILIES) * int(identity["cases_per_family"])),
        "future_targets_never_passed_to_forecaster": all(
            row["future_observations_used_by_forecaster"] == 0 for row in rows),
        "all_horizons_complete": all(row["rows"] == HORIZON for row in rows),
        "quantiles_finite_and_ordered": all(
            row["ordered_quantiles"] for row in rows),
    }
    comparison: dict[str, Any] | None = None
    if reference is None:
        gates = base_gates
    else:
        expected = reference.get("run_identity") or {}
        for key in ("seed", "families", "cases_per_family", "history", "horizon"):
            if expected.get(key) != identity.get(key):
                raise ValueError(f"reference identity differs on {key}")
        reference_rows = {row["case_id"]: row
                          for row in reference.get("rows") or []}
        current_rows = {row["case_id"]: row for row in rows}
        if set(reference_rows) != set(current_rows):
            raise ValueError("reference case matrix differs")
        eligible_ids = [case_id for case_id, row in reference_rows.items()
                        if row["support"] == "degraded"
                        and row["selected_model"] == "last_value"
                        and row["flat_point_path"]
                        and not row["horizon_split"]]
        eligible = [current_rows[case_id] for case_id in eligible_ids]
        ref_eligible = [reference_rows[case_id] for case_id in eligible_ids]
        signal_ids = [case_id for case_id in eligible_ids
                      if reference_rows[case_id]["family"] in {"trend", "seasonal"}]
        signal = [current_rows[case_id] for case_id in signal_ids]
        ref_signal = [reference_rows[case_id] for case_id in signal_ids]
        safety_ids = [case_id for case_id in eligible_ids
                      if reference_rows[case_id]["family"] in {"level", "random_walk"}]
        safety = [current_rows[case_id] for case_id in safety_ids]
        ref_safety = [reference_rows[case_id] for case_id in safety_ids]

        def mean(items: list[dict[str, Any]], field: str) -> float:
            return statistics.mean(float(item[field]) for item in items)

        family_changes = {
            family: ((by_family[family]["median_wis"]
                      - reference["by_family"][family]["median_wis"])
                     / max(float(reference["by_family"][family]["median_wis"]),
                           1e-12))
            for family in FAMILIES
        }
        gates = {
            **base_gates,
            "reference_contains_all_control_families": all(
                any(reference_rows[case_id]["family"] == family
                    for case_id in eligible_ids) for family in FAMILIES),
            "graduated_split_applies_to_80pct": bool(eligible) and
                statistics.mean(bool(row["horizon_split"])
                                for row in eligible) >= .8,
            "split_has_contiguous_evaluated_prefix_and_tail": all(
                row["horizon_split"]
                and 0 < row["evaluated_prefix_steps"] < HORIZON
                and row["best_effort_tail_steps"] ==
                    HORIZON - row["evaluated_prefix_steps"]
                and row["tiers"] ==
                    row["tiers"][:row["evaluated_prefix_steps"]]
                    + ["best_effort"] * row["best_effort_tail_steps"]
                for row in eligible if row["horizon_split"]),
            "split_floor_is_best_effort": all(
                row["tier_floor"] == "best_effort"
                for row in eligible if row["horizon_split"]),
            "aggregate_wis_nonworsening": mean(eligible, "mean_wis")
                <= mean(ref_eligible, "mean_wis") + 1e-12,
            "aggregate_mae_improves_2pct": mean(eligible, "mean_absolute_error")
                <= mean(ref_eligible, "mean_absolute_error") * .98,
            "signal_wis_improves_10pct": mean(signal, "mean_wis")
                <= mean(ref_signal, "mean_wis") * .9,
            "safety_control_wis_regression_within_2pct": mean(
                safety, "mean_wis") <= mean(ref_safety, "mean_wis") * 1.02,
            "family_median_wis_regression_within_3pct": max(
                family_changes.values()) <= .03,
        }
        comparison = {
            "reference_eligible_cases": len(eligible_ids),
            "treatment_split_cases": sum(
                bool(row["horizon_split"]) for row in eligible),
            "eligible_mean_wis_relative_change": (
                mean(eligible, "mean_wis") / mean(ref_eligible, "mean_wis") - 1),
            "eligible_mae_relative_change": (
                mean(eligible, "mean_absolute_error") /
                mean(ref_eligible, "mean_absolute_error") - 1),
            "signal_wis_relative_change": (
                mean(signal, "mean_wis") / mean(ref_signal, "mean_wis") - 1),
            "safety_wis_relative_change": (
                mean(safety, "mean_wis") / mean(ref_safety, "mean_wis") - 1),
            "family_median_wis_relative_change": family_changes,
        }
    return {
        "schema_version": 2,
        "benchmark": "graduated-horizon-publication",
        "evaluated_commit": identity["evaluated_commit"],
        "run_identity": identity,
        "overall": overall,
        "by_family": by_family,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        **({"comparison": comparison} if comparison is not None else {}),
        "rows": rows,
    }


def run(seed: int, cases_per_family: int, output_dir: Path, *,
        resume: bool = False, reference_summary: Path | None = None) \
        -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "run-identity.json"
    checkpoint = output_dir / "observations.jsonl"
    identity = {
        "schema_version": 2,
        "benchmark": "graduated-horizon-publication",
        "evaluated_commit": code_revision(),
        "seed": seed,
        "families": list(FAMILIES),
        "cases_per_family": cases_per_family,
        "history": HISTORY,
        "horizon": HORIZON,
        "seasonal_period": PERIOD,
    }
    if resume:
        if not identity_path.is_file():
            raise ValueError("resume requires a retained run identity")
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("resume identity differs from retained run")
    elif identity_path.exists() or checkpoint.exists():
        raise ValueError("output directory already contains a run")
    else:
        identity_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    completed: dict[str, dict[str, Any]] = {}
    if resume and checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["case_id"]] = row
    total = len(FAMILIES) * cases_per_family
    ordinal = 0
    for family_index, family in enumerate(FAMILIES):
        for case_index in range(cases_per_family):
            ordinal += 1
            case_id = f"{family}-{case_index:03d}"
            if case_id in completed:
                continue
            row = _case(family, case_index,
                        seed + family_index * 100_000 + case_index)
            completed[case_id] = row
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
            print(f"completed {ordinal}/{total} {case_id}", flush=True)
    rows = [completed[f"{family}-{index:03d}"]
            for family in FAMILIES for index in range(cases_per_family)]
    reference = (json.loads(reference_summary.read_text(encoding="utf-8"))
                 if reference_summary is not None else None)
    result = summarize(rows, identity, reference)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cases-per-family", type=int,
                        default=CASES_PER_FAMILY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-summary", type=Path)
    args = parser.parse_args()
    result = run(args.seed, args.cases_per_family, args.output_dir,
                 resume=args.resume,
                 reference_summary=args.reference_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
