"""Prospective, resumable calibration/action evaluation for the v0.8 A1 gate."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from gnomon.config import GnomonConfig
from gnomon.evaluation import conformal_spreads, evaluate, interval_from_spread


FAMILIES = (
    "level", "trend", "seasonal", "intermittent", "heteroskedastic",
    "regime_shift",
)


def _series(seed: int, family: str, length: int, season: int) -> list[float]:
    rng = random.Random(seed)
    level = rng.uniform(40.0, 90.0)
    slope = rng.choice((-1.0, 1.0)) * rng.uniform(.08, .35)
    amplitude = rng.uniform(3.0, 9.0)
    values: list[float] = []
    for index in range(length):
        if family == "level":
            value = level + rng.gauss(0, 1.2)
        elif family == "trend":
            value = level + slope * index + rng.gauss(0, 1.0)
        elif family == "seasonal":
            value = (level + amplitude * math.sin(
                2 * math.pi * index / season) + rng.gauss(0, .9))
        elif family == "intermittent":
            value = rng.uniform(4.0, 18.0) if rng.random() < .24 else 0.0
        elif family == "heteroskedastic":
            scale = .4 + 2.8 * index / max(1, length - 1)
            value = level + slope * index * .2 + rng.gauss(0, scale)
        else:
            shift = 0.0 if index < int(length * .62) else amplitude
            value = level + shift + rng.gauss(0, 1.0)
        values.append(value)
    return values


def _interval_score(actual: float, low: float, high: float) -> float:
    score = high - low
    if actual < low:
        score += 10.0 * (low - actual)
    elif actual > high:
        score += 10.0 * (actual - high)
    return score


def _pinball(actual: float, estimate: float, quantile: float) -> float:
    error = actual - estimate
    return max(quantile * error, (quantile - 1.0) * error)


def _arm(history: list[float], actual: list[float], horizon: int,
         season: int, frequency: str, *, pooled: bool) -> dict[str, Any]:
    config = GnomonConfig()
    config.evaluation.pool_residuals = pooled
    assessment = evaluate(
        history, horizon, season, .02, frequency=frequency, tsfm_names=[],
        strict_abstention=False, config=config,
    )
    if (not assessment.supported or assessment.final_candidate is None
            or not assessment.residuals):
        return {
            "executed": True, "complete": False,
            "supported": assessment.supported,
            "selected_model": assessment.selected_model,
            "measured_prior_coverage": assessment.coverage,
            "residual_fold_count": assessment.residual_fold_count,
            "residuals_pooled_across_selection":
                assessment.residuals_pooled_across_selection,
        }
    points = assessment.final_candidate.fit(history, season).predict(horizon)
    spreads = conformal_spreads(
        assessment.residuals_by_lead, horizon, assessment.residuals,
        recentre=not assessment.degraded,
    )
    intervals = [interval_from_spread(points[index], spreads[index + 1])
                 for index in range(horizon)]
    covered = [low <= observed <= high
               for observed, (low, _, high) in zip(actual, intervals)]
    interval_scores = [_interval_score(observed, low, high)
                       for observed, (low, _, high) in zip(actual, intervals)]
    pinball = [statistics.mean((
        _pinball(observed, low, .1),
        _pinball(observed, middle, .5),
        _pinball(observed, high, .9),
    )) for observed, (low, middle, high) in zip(actual, intervals)]
    last = history[-1]
    final_low, _, final_high = intervals[-1]
    signal = ("increase" if final_low > last else
              "decrease" if final_high < last else None)
    truth = ("increase" if actual[-1] > last else
             "decrease" if actual[-1] < last else "stable")
    action_correct = None if signal is None else signal == truth
    prior_coverage = assessment.coverage
    policy_eligible = bool(
        not pooled and horizon >= 10 and assessment.residual_fold_count >= 1
        and prior_coverage is not None and .65 <= prior_coverage <= .95)
    return {
        "executed": True, "complete": True, "supported": True,
        "selected_model": assessment.selected_model,
        "points": points, "coverage": sum(covered) / len(covered),
        "covered_points": sum(covered), "interval_points": len(covered),
        "mean_interval_score": statistics.mean(interval_scores),
        "mean_pinball": statistics.mean(pinball),
        "mean_width": statistics.mean(high - low
                                      for low, _, high in intervals),
        "measured_prior_coverage": prior_coverage,
        "residual_fold_count": assessment.residual_fold_count,
        "residuals_pooled_across_selection":
            assessment.residuals_pooled_across_selection,
        "policy_eligible": policy_eligible,
        "action_signal": signal, "action_truth": truth,
        "action_correct": action_correct,
        "false_action_cost": 1.0 if action_correct is False else 0.0,
        "selective_utility": (1.0 if action_correct is True else
                              -1.0 if action_correct is False else 0.0),
    }


def _cases(seed: int, cases_per_family: int) -> list[dict[str, Any]]:
    cases = []
    for family_index, family in enumerate(FAMILIES):
        for index in range(cases_per_family):
            horizon = 3 if index % 2 == 0 else 12
            frequency = "D" if index % 4 < 2 else "h"
            season = 7 if family == "seasonal" and frequency == "D" else \
                24 if family == "seasonal" else 1
            case_seed = seed + family_index * 100_000 + index
            cases.append({
                "case_id": f"{family}-{index:03d}", "family": family,
                "seed": case_seed, "horizon": horizon,
                "frequency": frequency, "season": season,
            })
    return cases


def _aggregate(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    values = [row[arm] for row in rows if row[arm].get("complete")]
    points = sum(int(item["interval_points"]) for item in values)
    actions = [item for item in values
               if item.get("policy_eligible") and item.get("action_signal")]
    return {
        "completed": len(values),
        "coverage": (sum(int(item["covered_points"]) for item in values)
                     / points if points else None),
        "interval_points": points,
        "mean_interval_score": (statistics.mean(
            float(item["mean_interval_score"]) for item in values)
            if values else None),
        "mean_pinball": (statistics.mean(
            float(item["mean_pinball"]) for item in values)
            if values else None),
        "mean_width": (statistics.mean(float(item["mean_width"])
                                       for item in values)
                       if values else None),
        "policy_eligible_cases": sum(
            bool(item.get("policy_eligible")) for item in values),
        "actions": len(actions),
        "false_action_cost": sum(
            float(item["false_action_cost"]) for item in actions),
        "selective_utility": sum(
            float(item["selective_utility"]) for item in actions),
    }


def summarize(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    pooled = _aggregate(rows, "pooled")
    strict = _aggregate(rows, "strict")
    by_family = {
        family: _aggregate([row for row in rows if row["family"] == family],
                           "strict")
        for family in FAMILIES
    }
    family_coverage = [item["coverage"] for item in by_family.values()
                       if item["coverage"] is not None]
    point_parity = all(
        row["pooled"].get("points") == row["strict"].get("points")
        for row in rows if row["pooled"].get("complete")
        and row["strict"].get("complete"))
    gates = {
        "all_cases_complete": len(rows) == expected and all(
            row[arm].get("executed") is True
            for row in rows for arm in ("pooled", "strict")),
        "point_forecasts_unchanged": point_parity,
        "strict_split_provenance": all(
            row["strict"].get("residuals_pooled_across_selection") is False
            for row in rows if row["strict"].get("complete")),
        "pooled_never_policy_eligible": pooled["policy_eligible_cases"] == 0,
        "overall_coverage_within_10_points": bool(
            strict["coverage"] is not None
            and abs(float(strict["coverage"]) - .8) <= .1),
        "family_coverage_within_15_points": bool(
            len(family_coverage) == len(FAMILIES)
            and all(abs(float(value) - .8) <= .15
                    for value in family_coverage)),
        "interval_score_nonworsening": bool(
            strict["mean_interval_score"] is not None
            and pooled["mean_interval_score"] is not None
            and float(strict["mean_interval_score"])
            <= float(pooled["mean_interval_score"]) + 1e-12),
        "proper_score_nonworsening": bool(
            strict["mean_pinball"] is not None
            and pooled["mean_pinball"] is not None
            and float(strict["mean_pinball"])
            <= float(pooled["mean_pinball"]) + 1e-12),
        "false_action_cost_nonincreasing": (
            strict["false_action_cost"] <= pooled["false_action_cost"]),
        "positive_selective_utility": strict["actions"] > 0
            and strict["selective_utility"] > 0,
    }
    return {
        "schema_version": 1, "benchmark": "calibration-action-evaluation",
        "cases": len(rows), "expected_cases": expected,
        "arms": {"pooled": pooled, "strict": strict},
        "strict_by_family": by_family, "gates": gates,
        "all_promotion_gates_passed": all(gates.values()),
        "rows": rows,
    }


def run(seed: int, cases_per_family: int, output_dir: Path,
        resume: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "observations.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if resume and checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["case_id"]] = row
    cases = _cases(seed, cases_per_family)
    for index, case in enumerate(cases, 1):
        if case["case_id"] in completed:
            continue
        complete = _series(
            case["seed"], case["family"], 120 + case["horizon"],
            case["season"])
        history, actual = complete[:-case["horizon"]], complete[-case["horizon"]:]
        row = {
            **case,
            "future_observations_used_by_forecaster": 0,
            "pooled": _arm(history, actual, case["horizon"], case["season"],
                           case["frequency"], pooled=True),
            "strict": _arm(history, actual, case["horizon"], case["season"],
                           case["frequency"], pooled=False),
        }
        completed[case["case_id"]] = row
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"completed {index}/{len(cases)} {case['case_id']}", flush=True)
    rows = [completed[case["case_id"]] for case in cases]
    summary = summarize(rows, len(cases))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cases-per-family", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run(args.seed, args.cases_per_family, args.output_dir, args.resume)
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "rows"}, indent=2, sort_keys=True))
    return 0 if summary["gates"]["all_cases_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
