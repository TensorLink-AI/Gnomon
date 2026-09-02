"""Run the frozen fold-starved structural-admission matrix.

Only the training prefix reaches :func:`gnomon.evaluation.evaluate`.  The
future suffix is retained by this harness until after publication.  A later
revision can be compared with ``--reference-summary``; comparison refuses a
different seed or case matrix.
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
from benchmarks.modelbench.run_production_selector import _published_points
from gnomon.config import GnomonConfig
from gnomon.evaluation import evaluate
from gnomon.models import last_value


FAMILIES = (
    "stable_trend",
    "stable_level",
    "random_walk",
    "recent_level_shift",
    "recent_trend_reversal",
    "intermittent",
)
HORIZONS = (3, 6)
CASES_PER_FAMILY_HORIZON = 30
HISTORY = 18
SAFETY_FAMILIES = (
    "random_walk", "recent_level_shift", "recent_trend_reversal",
    "intermittent",
)


def _complete_series(rng: random.Random, family: str,
                     horizon: int) -> list[float]:
    length = HISTORY + horizon
    level = rng.uniform(30.0, 90.0)
    if family == "stable_trend":
        slope = rng.choice((-1.0, 1.0)) * rng.uniform(0.6, 1.4)
        return [level + slope * index + rng.gauss(0.0, 0.65)
                for index in range(length)]
    if family == "stable_level":
        return [level + rng.gauss(0.0, 1.5) for _ in range(length)]
    if family == "random_walk":
        values = []
        current = level
        for _ in range(length):
            current += rng.gauss(0.0, 1.5)
            values.append(current)
        return values
    if family == "recent_level_shift":
        shift = rng.choice((-1.0, 1.0)) * rng.uniform(6.0, 12.0)
        return [level + (shift if index >= HISTORY - 5 else 0.0)
                + rng.gauss(0.0, 0.8) for index in range(length)]
    if family == "recent_trend_reversal":
        slope = rng.choice((-1.0, 1.0)) * rng.uniform(0.7, 1.3)
        pivot = HISTORY - 6
        return [level + (slope * index if index <= pivot else
                         slope * pivot - slope * (index - pivot))
                + rng.gauss(0.0, 0.55) for index in range(length)]
    return [rng.uniform(5.0, 15.0) if rng.random() < 0.22 else 0.0
            for _ in range(length)]


def _mae(actual: list[float], predicted: list[float]) -> float:
    return statistics.mean(abs(a - p) for a, p in zip(actual, predicted))


def _case(family: str, horizon: int, case_index: int,
          seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    complete = _complete_series(rng, family, horizon)
    history, actual = complete[:HISTORY], complete[HISTORY:]
    config = GnomonConfig()
    config.models.statsforecast_enabled = False
    assessment = evaluate(
        history, horizon, 1, 0.02, frequency="synthetic",
        tsfm_names=[], strict_abstention=False, config=config,
    )
    points, selected, support, fallback_disclosed = _published_points(
        assessment, history, horizon, 1)
    baseline = last_value(history, horizon, 1)
    baseline_loss = _mae(actual, baseline)
    selected_loss = _mae(actual, points)
    departed = any(not math.isclose(left, right, rel_tol=0.0,
                                    abs_tol=1e-12)
                   for left, right in zip(points, baseline))
    return {
        "case_id": f"{family}-h{horizon}-{case_index:03d}",
        "family": family,
        "horizon": horizon,
        "seed": seed,
        "history_length": HISTORY,
        "future_observations_used_by_forecaster": 0,
        "selected": selected,
        "support": support,
        "engine_degraded": assessment.degraded,
        "selection_fold_count": assessment.selection_fold_count,
        "selection_guardrail_applied": assessment.selection_guardrail_applied,
        "fallback_disclosed": fallback_disclosed,
        "departed_last_value": departed,
        "baseline_loss": baseline_loss,
        "selected_loss": selected_loss,
        "relative_gain": ((baseline_loss - selected_loss) /
                          max(baseline_loss, 1e-12)),
        "harmful_departure": bool(departed and selected_loss > baseline_loss),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    departures = [row for row in rows if row["departed_last_value"]]
    return {
        "cases": len(rows),
        "departure_rate": len(departures) / len(rows),
        "harmful_departures": sum(row["harmful_departure"] for row in rows),
        "median_baseline_loss": statistics.median(
            float(row["baseline_loss"]) for row in rows),
        "median_selected_loss": statistics.median(
            float(row["selected_loss"]) for row in rows),
        "median_relative_gain": statistics.median(
            float(row["relative_gain"]) for row in rows),
        "mean_selected_loss": statistics.mean(
            float(row["selected_loss"]) for row in rows),
    }


def summarize(rows: list[dict[str, Any]], identity: dict[str, Any],
              reference: dict[str, Any] | None = None) -> dict[str, Any]:
    if reference is not None:
        expected = reference.get("run_identity") or {}
        for key in ("seed", "families", "horizons",
                    "cases_per_family_horizon", "history"):
            if expected.get(key) != identity.get(key):
                raise ValueError(f"reference identity differs on {key}")
    by_family = {family: _aggregate(
        [row for row in rows if row["family"] == family])
        for family in FAMILIES}
    overall = _aggregate(rows)
    base_gates = {
        "all_cases_complete": len(rows) == (
            len(FAMILIES) * len(HORIZONS) * CASES_PER_FAMILY_HORIZON),
        "future_targets_never_passed_to_forecaster": all(
            row["future_observations_used_by_forecaster"] == 0 for row in rows),
        "all_results_remain_degraded": all(row["engine_degraded"] for row in rows),
        "all_losses_finite": all(math.isfinite(float(row[key]))
                                  for row in rows
                                  for key in ("baseline_loss", "selected_loss")),
    }
    comparison: dict[str, Any] | None = None
    if reference is None:
        gates = base_gates
    else:
        reference_rows = {row["case_id"]: row
                          for row in reference.get("rows") or []}
        if set(reference_rows) != {row["case_id"] for row in rows}:
            raise ValueError("reference case matrix differs")
        reference_family = reference["by_family"]
        family_loss_ratio = {
            family: ((by_family[family]["median_selected_loss"]
                      - reference_family[family]["median_selected_loss"])
                     / max(reference_family[family]["median_selected_loss"],
                           1e-12))
            for family in FAMILIES
        }
        departure_gain = {
            family: (by_family[family]["departure_rate"]
                     - reference_family[family]["departure_rate"])
            for family in FAMILIES
        }
        gates = {
            **base_gates,
            "aggregate_mean_loss_nonworsening":
                overall["mean_selected_loss"]
                <= float(reference["overall"]["mean_selected_loss"]) + 1e-12,
            "family_median_loss_regression_within_2pct":
                max(family_loss_ratio.values()) <= 0.02,
            "safety_family_harmful_departures_do_not_increase": all(
                by_family[family]["harmful_departures"]
                <= reference_family[family]["harmful_departures"]
                for family in SAFETY_FAMILIES),
            "stable_trend_departure_rate_improves_25_points":
                departure_gain["stable_trend"] >= 0.25,
            "stable_trend_median_loss_improves_10pct":
                family_loss_ratio["stable_trend"] <= -0.10,
            "stable_level_departure_rate_improves_15_points":
                departure_gain["stable_level"] >= 0.15,
            "stable_level_median_loss_improves_2pct":
                family_loss_ratio["stable_level"] <= -0.02,
        }
        comparison = {
            "family_median_selected_loss_relative_change": family_loss_ratio,
            "family_departure_rate_change": departure_gain,
        }
    return {
        "schema_version": 1,
        "benchmark": "degraded-structural-admission",
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
        "benchmark": "degraded-structural-admission",
        "evaluated_commit": code_revision(),
        "seed": seed,
        "families": list(FAMILIES),
        "horizons": list(HORIZONS),
        "cases_per_family_horizon": CASES_PER_FAMILY_HORIZON,
        "history": HISTORY,
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
    ordinal = 0
    total = len(FAMILIES) * len(HORIZONS) * CASES_PER_FAMILY_HORIZON
    for family_index, family in enumerate(FAMILIES):
        for horizon_index, horizon in enumerate(HORIZONS):
            for case_index in range(CASES_PER_FAMILY_HORIZON):
                case_id = f"{family}-h{horizon}-{case_index:03d}"
                ordinal += 1
                if case_id in completed:
                    continue
                seed_i = (seed + family_index * 100_000
                          + horizon_index * 10_000 + case_index)
                row = _case(family, horizon, case_index, seed_i)
                completed[case_id] = row
                with checkpoint.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                print(f"completed {ordinal}/{total} {case_id}", flush=True)
    rows = [completed[f"{family}-h{horizon}-{index:03d}"]
            for family in FAMILIES for horizon in HORIZONS
            for index in range(CASES_PER_FAMILY_HORIZON)]
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
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, sort_keys=True))
    return 0 if result["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
