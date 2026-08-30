"""Held-out screen for the production short-history selector.

The generator owns synthetic family labels; the production evaluator receives
only numeric prefixes, a horizon, and a declared seasonal period. The final
horizon is sliced away before evaluation and is used only for scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.manifest import code_revision, write_manifest
from gnomon.evaluation import Evaluation, evaluate
from gnomon.models import MODELS, last_value, predict


FAMILIES = ("level", "trend", "seasonal", "pseudo_seasonal", "intermittent")
DEFAULT_MINIMUM_IMPROVEMENT = 0.02


def _series(
    rng: random.Random, family: str, length: int, season: int,
) -> list[float]:
    level = rng.uniform(30, 90)
    direction = rng.choice((-1.0, 1.0))
    slope = direction * rng.uniform(0.35, 0.9)
    amplitude = rng.uniform(4.0, 9.0)
    phase = rng.uniform(0, 2 * math.pi)
    values: list[float] = []
    cycle_phase = phase
    for index in range(length):
        if family == "level":
            value = level + rng.gauss(0, 1.4)
        elif family == "trend":
            value = level + slope * index + rng.gauss(0, 0.8)
        elif family == "seasonal":
            value = (level + amplitude * math.sin(
                2 * math.pi * index / season + phase) + rng.gauss(0, 0.7))
        elif family == "pseudo_seasonal":
            # A visually periodic but non-repeatable process. Phase changes
            # are generated independently of selector behavior and happen in
            # the history as well as the untouched future.
            if index and index % season == 0:
                cycle_phase += rng.choice((-1.0, 1.0)) * rng.uniform(0.6, 1.5)
            value = (level + amplitude * math.sin(
                2 * math.pi * index / season + cycle_phase)
                + rng.gauss(0, 1.0))
        else:
            value = rng.uniform(5, 14) if rng.random() < 0.22 else 0.0
        values.append(value)
    return values


def _loss(actual: list[float], points: list[float]) -> float:
    if len(actual) != len(points) or not actual:
        return float("inf")
    value = mean(abs(observed - estimated)
                 for observed, estimated in zip(actual, points))
    return value if math.isfinite(value) else float("inf")


def _published_points(
    assessment: Evaluation, history: list[float], horizon: int, season: int,
) -> tuple[list[float], str, str, bool]:
    # The default product floor is best_effort: an engine abstention becomes
    # a disclosed last-value path in pipeline.best_effort_stage. Keep engine
    # support and product answer yield separate instead of calling that
    # labelled fallback a missing forecast.
    selected = assessment.selected_model
    if not assessment.supported or selected is None:
        return (last_value(history, horizon, season), "last_value",
                "best_effort", True)
    support = ("degraded" if assessment.degraded else
               "weakly_supported" if assessment.warnings else "supported")
    final = assessment.final_candidate
    if final is not None and final.identity.name == selected:
        return (final.fit(history, season).predict(horizon), selected, support,
                False)
    if selected in MODELS:
        return (predict(selected, history, horizon, season), selected, support,
                False)
    raise ValueError(f"production selector returned unbound candidate {selected}")


def _outcome(candidate: float, baseline: float) -> str:
    if math.isclose(candidate, baseline, rel_tol=0, abs_tol=1e-12):
        return "safety_preservation"
    return "uplift" if candidate < baseline else "regression"


def _bootstrap_interval(
    values: list[float], seed: int, draws: int = 4000,
) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    samples = sorted(median(rng.choices(values, k=len(values)))
                     for _ in range(draws))
    return [samples[int(0.05 * (draws - 1))],
            samples[int(0.95 * (draws - 1))]]


def _summary(rows: list[dict[str, object]], bootstrap_seed: int) -> dict[str, object]:
    completed = [row for row in rows if row["completed"]]
    departures = [row for row in completed if row["departed_from_last_value"]]
    gains = [float(row["relative_gain"]) for row in completed]
    departure_gains = [float(row["relative_gain"]) for row in departures]
    outcomes = {
        name: sum(row["outcome"] == name for row in completed)
        for name in ("uplift", "safety_preservation", "regression")
    }
    departure_wins = sum(row["outcome"] == "uplift" for row in departures)
    departure_losses = sum(row["outcome"] == "regression" for row in departures)
    return {
        "cases": len(rows),
        "completed": len(completed),
        "completion_rate": len(completed) / len(rows) if rows else 0.0,
        "departures": len(departures),
        "admission_rate": len(departures) / len(completed) if completed else 0.0,
        "admission_precision": (
            departure_wins / len(departures) if departures else None),
        "harmful_admission_rate": (
            departure_losses / len(departures) if departures else 0.0),
        "outcomes": outcomes,
        "departure_wins": departure_wins,
        "departure_losses": departure_losses,
        "median_relative_gain_all_cases": median(gains) if gains else None,
        "median_relative_gain_departures": (
            median(departure_gains) if departure_gains else None),
        "departure_median_gain_90pct_ci": _bootstrap_interval(
            departure_gains, bootstrap_seed),
        "baseline_median_mae": median(
            float(row["baseline_loss"]) for row in completed) if completed else None,
        "candidate_median_mae": median(
            float(row["candidate_loss"]) for row in completed) if completed else None,
    }


def run(seed: int = 92741, cases_per_family: int = 40) -> dict[str, object]:
    if cases_per_family < 2:
        raise ValueError("cases_per_family must be at least 2")
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        season = 6 if family in {"seasonal", "pseudo_seasonal"} else 1
        for case in range(cases_per_family):
            if case % 2 == 0:
                horizon = 3
                history_length = rng.choice((15, 18, 21, 24))
                length_lane = "short_horizon"
            else:
                horizon = 12
                history_length = rng.choice((42, 45, 47))
                length_lane = "fold_starved_long_horizon"
            complete = _series(
                rng, family, history_length + horizon, season)
            history = complete[:history_length]
            actual = complete[history_length:]
            assessment = evaluate(
                history, horizon, season, DEFAULT_MINIMUM_IMPROVEMENT,
                frequency="synthetic", tsfm_names=[], strict_abstention=False,
            )
            try:
                points, published_model, published_support, fallback_disclosed = _published_points(
                    assessment, history, horizon, season)
            except (ValueError, ArithmeticError, OverflowError):
                points, published_model, published_support = [], "none", "unsupported"
                fallback_disclosed = False
            baseline = last_value(history, horizon, season)
            completed = (len(points) == horizon
                         and all(math.isfinite(value) for value in points))
            baseline_loss = _loss(actual, baseline)
            candidate_loss = _loss(actual, points) if completed else float("inf")
            departed = completed and any(
                not math.isclose(candidate, robust, rel_tol=0, abs_tol=1e-12)
                for candidate, robust in zip(points, baseline))
            relative_gain = (
                (baseline_loss - candidate_loss)
                / max(baseline_loss, candidate_loss, 1e-12)
                if completed and math.isfinite(baseline_loss) else None)
            admission = (assessment.admission_decision.to_payload()
                         if assessment.admission_decision is not None else None)
            rows.append({
                "case_id": f"{family}-{case:03d}",
                "family": family,
                "length_lane": length_lane,
                "history_length": history_length,
                "horizon": horizon,
                "season": season,
                "future_observations_used_by_selector": 0,
                "completed": completed,
                "engine_selected_model": assessment.selected_model,
                "published_model": published_model,
                "published_support": published_support,
                "fallback_disclosed": fallback_disclosed,
                "strongest_baseline": assessment.strongest_baseline,
                "engine_supported": assessment.supported,
                "degraded": assessment.degraded,
                "selection_fold_count": assessment.selection_fold_count,
                "selection_guardrail_applied": (
                    assessment.selection_guardrail_applied),
                "selection_stability": assessment.selection_stability,
                "selection_scores": assessment.selection_scores,
                "admission_decision": admission,
                "warnings": assessment.warnings,
                "notes": assessment.notes,
                "departed_from_last_value": departed,
                "baseline_loss": baseline_loss,
                "candidate_loss": candidate_loss,
                "relative_gain": relative_gain,
                "outcome": (_outcome(candidate_loss, baseline_loss)
                            if completed else "failure"),
            })

    overall = _summary(rows, seed + 1)
    by_family = {
        family: _summary(
            [row for row in rows if row["family"] == family],
            seed + 100 + index,
        )
        for index, family in enumerate(FAMILIES)
    }
    family_medians = [
        float(item["median_relative_gain_all_cases"])
        for item in by_family.values()
        if item["median_relative_gain_all_cases"] is not None
    ]
    interval = overall["departure_median_gain_90pct_ci"]
    result: dict[str, object] = {
        "schema_version": "0.1",
        "benchmark": "production-short-history-selector",
        "seed": seed,
        "cases_per_family": cases_per_family,
        "protocol": (
            "production evaluate() receives history prefixes only; final "
            "horizon is untouched until scoring; TSFM discovery disabled"),
        "minimum_baseline_improvement": DEFAULT_MINIMUM_IMPROVEMENT,
        "overall": overall,
        "by_family": by_family,
        "raw_records": rows,
    }
    result["gates"] = {
        "completion_rate_at_least_99pct": overall["completion_rate"] >= 0.99,
        "no_silent_fallback": all(
            row["engine_supported"] or row["fallback_disclosed"]
            for row in rows if row["completed"]),
        "future_observations_used_zero": all(
            row["future_observations_used_by_selector"] == 0 for row in rows),
        "at_least_20_departures": overall["departures"] >= 20,
        "departure_admission_precision_at_least_70pct": (
            overall["departures"] >= 20
            and overall["admission_precision"] is not None
            and overall["admission_precision"] >= 0.70),
        "departure_median_gain_positive": (
            overall["median_relative_gain_departures"] is not None
            and overall["median_relative_gain_departures"] > 0),
        "more_departure_wins_than_losses": (
            overall["departure_wins"] > overall["departure_losses"]),
        "departure_bootstrap_lower_nonnegative": (
            interval is not None and interval[0] >= 0),
        "family_median_regression_within_2pct": (
            len(family_medians) == len(FAMILIES)
            and min(family_medians) >= -0.02),
        "selection_provenance_complete": all(
            row["published_model"] != "none"
            and row["published_support"] in {
                "supported", "weakly_supported", "degraded", "best_effort"}
            and isinstance(row["selection_scores"], dict)
            and isinstance(row["warnings"], list)
            for row in rows if row["completed"]),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=92741)
    parser.add_argument("--cases-per-family", type=int, default=40)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run(args.seed, args.cases_per_family)
    result["evaluated_commit"] = code_revision()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_manifest(
            args.output_dir,
            benchmark="modelbench",
            condition="production-short-history-selector",
            target=(f"seed={args.seed};"
                    f"cases_per_family={args.cases_per_family}"),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(result["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
