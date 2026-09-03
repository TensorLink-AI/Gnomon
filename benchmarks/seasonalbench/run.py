"""Run the frozen v0.7 Q1 seasonal-period admission reproduction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from statistics import mean, median
from typing import Any

from benchmarks.common.checkpoint import prepare_run_identity
from benchmarks.common.manifest import code_revision, write_manifest
from gnomon.evaluation import Evaluation, evaluate
from gnomon.models import (
    MODELS, historical_mean, last_value, predict, seasonal_naive,
)
from gnomon.support import (
    assess_forecast_support, disclose_seasonal_period_override,
    forecast_headline,
)
from gnomon.temporal import detect_season


FAMILIES = ("stable_seasonal", "phase_unstable", "level", "trend")
SEEDS = (6101, 6102, 6103, 6104)
LANES = {"sufficient": 72, "fold_starved": 24}
CONDITIONS: tuple[tuple[str, int | None], ...] = (
    ("auto", None), ("correct_6", 6), ("neighbor_5", 5), ("neighbor_7", 7),
)
HORIZON = 6
TRUE_PERIOD = 6
MINIMUM_IMPROVEMENT = 0.02


def _series(family: str, seed: int, length: int = 78) -> list[float]:
    rng = random.Random(seed + 10_000 * FAMILIES.index(family))
    level = rng.uniform(40.0, 90.0)
    amplitude = rng.uniform(5.0, 10.0)
    phase = rng.uniform(0.0, 2 * math.pi)
    slope = rng.choice((-1.0, 1.0)) * rng.uniform(0.3, 0.8)
    cycle_phase = phase
    values: list[float] = []
    for index in range(length):
        if family == "stable_seasonal":
            value = level + amplitude * math.sin(
                2 * math.pi * index / TRUE_PERIOD + phase) + rng.gauss(0, 0.7)
        elif family == "phase_unstable":
            if index and index % TRUE_PERIOD == 0:
                cycle_phase += rng.choice((-1.0, 1.0)) * rng.uniform(0.6, 1.5)
            value = level + amplitude * math.sin(
                2 * math.pi * index / TRUE_PERIOD + cycle_phase) + rng.gauss(0, 1.0)
        elif family == "level":
            value = level + rng.gauss(0, 1.4)
        else:
            value = level + slope * index + rng.gauss(0, 0.8)
        values.append(value)
    return values


def _loss(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        return float("inf")
    result = mean(abs(left - right) for left, right in zip(actual, predicted))
    return result if math.isfinite(result) else float("inf")


def _published(
    assessment: Evaluation, history: list[float], season: int,
) -> tuple[list[float], str, str, bool]:
    selected = assessment.selected_model
    if not assessment.supported or selected is None:
        return (
            last_value(history, HORIZON, season), "last_value", "best_effort", True,
        )
    support = (
        "degraded" if assessment.degraded else
        "weakly_supported" if assessment.warnings else "supported"
    )
    final = assessment.final_candidate
    if final is not None and final.identity.name == selected:
        return final.fit(history, season).predict(HORIZON), selected, support, False
    if selected in MODELS:
        return predict(selected, history, HORIZON, season), selected, support, False
    raise ValueError(f"unbound production candidate {selected}")


def _headline(
    support: str, assessment: Evaluation, *, override: int | None,
    detected_period: int, detected_strength: float, detected_basis: str,
) -> tuple[str, str]:
    public = assess_forecast_support(support, assessment.warnings, assessment)
    disclose_seasonal_period_override(
        public, override=override, detected_period=detected_period,
        detected_strength=detected_strength, detected_basis=detected_basis,
    )
    row_tier = (
        "conditionally_supported"
        if public.status == "conditionally_supported" else
        "best_effort" if public.status == "inconclusive" else "supported"
    )
    rows = [{"timestamp": "held-out", "tier": row_tier}] * HORIZON
    return forecast_headline(support, public.to_dict(), rows), public.status


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    complete = _series(case["family"], case["seed"])
    history_length = LANES[case["lane"]]
    history = complete[:history_length]
    actual = complete[history_length:history_length + HORIZON]
    detected_period, detected_strength, detected_basis = detect_season(
        history, "synthetic")
    used_period = case["override"] or detected_period
    period_basis = "override" if case["override"] is not None else detected_basis
    assessment = evaluate(
        history, HORIZON, used_period, MINIMUM_IMPROVEMENT,
        frequency="synthetic", tsfm_names=[], strict_abstention=False,
    )
    try:
        points, model, support, fallback = _published(
            assessment, history, used_period)
    except (ValueError, ArithmeticError, OverflowError):
        points, model, support, fallback = [], "none", "unsupported", False
    product_complete = (
        len(points) == HORIZON and all(math.isfinite(value) for value in points)
    )
    references = {
        "last_value": _loss(actual, last_value(history, HORIZON, TRUE_PERIOD)),
        "historical_mean": _loss(
            actual, historical_mean(history, HORIZON, TRUE_PERIOD)),
        "true_period_seasonal_naive": _loss(
            actual, seasonal_naive(history, HORIZON, TRUE_PERIOD)),
    }
    reference_name = min(references, key=references.get)
    reference_loss = references[reference_name]
    candidate_loss = _loss(actual, points) if product_complete else float("inf")
    gain = (
        (reference_loss - candidate_loss)
        / max(reference_loss, candidate_loss, 1e-12)
        if product_complete else None
    )
    robust = last_value(history, HORIZON, TRUE_PERIOD)
    departed = product_complete and any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(points, robust)
    )
    headline, support_status = _headline(
        support, assessment, override=case["override"],
        detected_period=detected_period, detected_strength=detected_strength,
        detected_basis=detected_basis,
    )
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "seed": case["seed"],
        "lane": case["lane"],
        "condition": case["condition"],
        "history_length": history_length,
        "horizon": HORIZON,
        "true_period": TRUE_PERIOD,
        "override": case["override"],
        "detected_period": detected_period,
        "detected_strength": detected_strength,
        "detected_basis": detected_basis,
        "used_period": used_period,
        "period_basis": period_basis,
        "visible_cycles_for_used_period": history_length / used_period,
        "future_observations_used_by_selector": 0,
        "engine_supported": assessment.supported,
        "product_complete": product_complete,
        "fallback_disclosed": fallback,
        "engine_selected_model": assessment.selected_model,
        "published_model": model,
        "published_support": support,
        "support_status": support_status,
        "baseline_improvement": assessment.improvement,
        "selection_fold_count": assessment.selection_fold_count,
        "selection_guardrail_applied": assessment.selection_guardrail_applied,
        "selection_scores": assessment.selection_scores,
        "selection_stability": assessment.selection_stability,
        "warnings": assessment.warnings,
        "notes": assessment.notes,
        "headline": headline,
        "headline_high_confidence": headline.startswith("High-confidence"),
        "departed_from_last_value": departed,
        "reference_losses": references,
        "strongest_reference": reference_name,
        "reference_loss": reference_loss,
        "candidate_loss": candidate_loss,
        "relative_gain": gain,
        "paired_gain_vs_correct": None,
    }


def _cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": f"{family}-{seed}-{lane}-{condition}",
            "family": family, "seed": seed, "lane": lane,
            "condition": condition, "override": override,
        }
        for family in FAMILIES
        for seed in SEEDS
        for lane in LANES
        for condition, override in CONDITIONS
    ]


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    departures = [row for row in rows if row["departed_from_last_value"]]
    gains = [float(row["relative_gain"]) for row in rows
             if row["relative_gain"] is not None]
    dep_gains = [float(row["relative_gain"]) for row in departures
                 if row["relative_gain"] is not None]
    return {
        "cases": len(rows),
        "product_complete": sum(row["product_complete"] for row in rows),
        "engine_supported": sum(row["engine_supported"] for row in rows),
        "departures": len(departures),
        "departure_wins": sum(value > 0 for value in dep_gains),
        "departure_losses": sum(value < 0 for value in dep_gains),
        "departure_precision": (
            sum(value > 0 for value in dep_gains) / len(dep_gains)
            if dep_gains else None),
        "median_relative_gain_all": median(gains) if gains else None,
        "median_relative_gain_departures": median(dep_gains) if dep_gains else None,
        "supported_harmful_departures": sum(
            row["departed_from_last_value"]
            and float(row["relative_gain"] or 0.0) < 0
            and row["support_status"] == "supported"
            for row in rows),
        "supported_harmful_nonbaseline_departures": sum(
            row["departed_from_last_value"]
            and float(row["relative_gain"] or 0.0) < 0
            and row["support_status"] == "supported"
            and row["published_model"] not in {
                "last_value", "seasonal_naive", "historical_mean"}
            for row in rows),
    }


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = {
        (row["family"], row["seed"], row["lane"]): row
        for row in rows if row["condition"] == "correct_6"
    }
    for row in rows:
        if row["condition"].startswith("neighbor_"):
            paired = correct[(row["family"], row["seed"], row["lane"])]
            left, right = float(paired["candidate_loss"]), float(row["candidate_loss"])
            row["paired_gain_vs_correct"] = (
                (left - right) / max(left, right, 1e-12)
                if math.isfinite(left) and math.isfinite(right) else None
            )
    by_cell = {
        f"{family}/{lane}/{condition}": _group([
            row for row in rows
            if row["family"] == family and row["lane"] == lane
            and row["condition"] == condition
        ])
        for family in FAMILIES for lane in LANES
        for condition, _ in CONDITIONS
    }
    stable_auto = [
        row for row in rows if row["family"] == "stable_seasonal"
        and row["lane"] == "sufficient" and row["condition"] == "auto"
    ]
    stable_evidenced = [
        row for row in rows if row["family"] == "stable_seasonal"
        and row["lane"] == "sufficient"
        and row["condition"] in {"auto", "correct_6"}
        and row["departed_from_last_value"]
    ]
    stable_gains = [float(row["relative_gain"]) for row in stable_evidenced]
    neighbor_cells: dict[str, float | None] = {}
    for family in FAMILIES:
        for lane in LANES:
            for condition in ("neighbor_5", "neighbor_7"):
                values = [
                    float(row["paired_gain_vs_correct"])
                    for row in rows if row["family"] == family
                    and row["lane"] == lane and row["condition"] == condition
                    and row["paired_gain_vs_correct"] is not None
                ]
                neighbor_cells[f"{family}/{lane}/{condition}"] = (
                    median(values) if values else None)
    ambiguous = [row for row in rows if row["family"] != "stable_seasonal"]
    ambiguous_departures = [
        row for row in ambiguous if row["departed_from_last_value"]
        and row["published_model"] not in {
            "last_value", "seasonal_naive", "historical_mean"}
    ]
    ambiguous_wins = sum(float(row["relative_gain"] or 0.0) > 0
                         for row in ambiguous_departures)
    auto_cycle_honesty = all(
        row["detected_basis"] != "autocorrelation"
        or row["history_length"] >= 2 * row["detected_period"]
        for row in rows if row["condition"] == "auto"
    )
    # The actual publication condition: a model other than the strongest
    # production baseline was published. Keep it separate from fallback
    # identity so a disclosed last-value fallback is not called admission.
    fold_starved_nonbaseline = [
        row for row in rows if row["lane"] == "fold_starved"
        and row["engine_supported"]
        and row["published_model"] not in {"last_value", "seasonal_naive",
                                           "historical_mean"}
    ]
    gates = {
        "all_128_product_cases_complete": len(rows) == 128
            and all(row["product_complete"] for row in rows),
        "engine_product_and_fallback_explicit": all(
            isinstance(row["engine_supported"], bool)
            and isinstance(row["product_complete"], bool)
            and isinstance(row["fallback_disclosed"], bool) for row in rows),
        "future_observations_used_zero": all(
            row["future_observations_used_by_selector"] == 0 for row in rows),
        "stable_auto_period6_at_least_75pct": (
            sum(row["used_period"] == TRUE_PERIOD for row in stable_auto)
            / len(stable_auto) >= 0.75),
        "auto_inference_requires_two_visible_cycles": auto_cycle_honesty,
        "stable_evidenced_departure_median_positive": bool(stable_gains)
            and median(stable_gains) > 0,
        "stable_evidenced_departure_precision_at_least_70pct":
            bool(stable_gains)
            and sum(value > 0 for value in stable_gains) / len(stable_gains) >= 0.70,
        "neighbor_period_cell_medians_within_2pct": all(
            value is not None and value >= -0.02
            for value in neighbor_cells.values()),
        "ambiguous_departures_precise_or_not_supported_harm": (
            not ambiguous_departures
            or ambiguous_wins / len(ambiguous_departures) >= 0.70
            or all(not (
                float(row["relative_gain"] or 0.0) < 0
                and row["support_status"] == "supported")
                for row in ambiguous_departures)),
        "fold_starved_nonbaseline_is_caveated_with_fold_count": all(
            row["support_status"] != "supported"
            and row["selection_fold_count"] > 0
            and not row["headline_high_confidence"]
            for row in fold_starved_nonbaseline),
    }
    return {
        "schema_version": "0.1",
        "benchmark": "seasonal-period-admission",
        "evaluated_commit": code_revision(),
        "scope": "full",
        "overall": _group(rows),
        "by_cell": by_cell,
        "neighbor_paired_median_gain": neighbor_cells,
        "gates": gates,
        "passed": all(gates.values()),
        "raw_records": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "cases.jsonl"
    cases = _cases()
    prepare_run_identity(
        args.output_dir,
        {
            "schema_version": 1,
            "benchmark": "seasonal-period-admission",
            "code_revision": code_revision(),
            "case_ids": [case["case_id"] for case in cases],
            "protocol": "frozen-v0.7-q1",
        },
        resume=args.resume,
        state_paths=[checkpoint, args.output_dir / "summary.json"],
    )
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["case_id"]] = row
    for case in cases:
        if case["case_id"] in completed:
            continue
        row = _run_case(case)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        completed[row["case_id"]] = row
    rows = [completed[case["case_id"]] for case in cases]
    result = _summarise(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(
        args.output_dir, benchmark="seasonalbench",
        condition="paired-period-and-fold-starvation",
        target="128 frozen history-prefix cases; horizon=6; true_period=6",
    )
    print(json.dumps({
        key: result[key] for key in (
            "benchmark", "evaluated_commit", "scope", "overall", "gates",
            "passed")
    }, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
