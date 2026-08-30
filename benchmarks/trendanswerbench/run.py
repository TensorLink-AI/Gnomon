"""Run the frozen v0.7 Q2 seasonal-trend answer reproduction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

from benchmarks.common.manifest import code_revision, write_manifest
from gnomon.temporal import detect_season
from gnomon.temporal_executables import _innovation_scale
from gnomon.temporal_question import compile_temporal_question
from gnomon.temporal_reasoning import answer_scoped_question


SEEDS = tuple(range(7301, 7309))
PERIOD = 24
LONG_HISTORY = 240
LONG_HORIZON = 48
SHORT_HISTORY = 30
SHORT_HORIZON = 12

FAMILIES: tuple[dict[str, Any], ...] = (
    {"name": "additive_up", "mechanism": "additive", "slope": .08},
    {"name": "additive_down", "mechanism": "additive", "slope": -.08},
    {"name": "additive_zero", "mechanism": "additive", "slope": 0.0},
    {"name": "multiplicative_up", "mechanism": "multiplicative", "slope": .08},
    {"name": "multiplicative_down", "mechanism": "multiplicative", "slope": -.08},
    {"name": "multiplicative_zero", "mechanism": "multiplicative", "slope": 0.0},
    {"name": "noisy_up", "mechanism": "noisy", "slope": .08},
    {"name": "noisy_down", "mechanism": "noisy", "slope": -.08},
    {"name": "noisy_zero", "mechanism": "noisy", "slope": 0.0},
    {"name": "plateau_after_up", "mechanism": "plateau", "slope": .08},
    {"name": "plateau_after_down", "mechanism": "plateau", "slope": -.08},
    {"name": "changepoint_up", "mechanism": "changepoint", "shift": 12.0},
    {"name": "changepoint_down", "mechanism": "changepoint", "shift": -12.0},
    {"name": "unadmitted_season_up", "mechanism": "unadmitted", "slope": .08},
    {"name": "unadmitted_season_down", "mechanism": "unadmitted", "slope": -.08},
    {"name": "unadmitted_season_zero", "mechanism": "unadmitted", "slope": 0.0},
    {"name": "insufficient_cycle_up", "mechanism": "insufficient", "slope": .08},
    {"name": "insufficient_cycle_down", "mechanism": "insufficient", "slope": -.08},
    {"name": "insufficient_cycle_zero", "mechanism": "insufficient", "slope": 0.0},
)


def _case_seed(family_index: int, seed: int) -> int:
    return seed + 10_000 * family_index


def _generate(case: dict[str, Any]) -> tuple[list[float], list[float]]:
    family_index = next(
        index for index, item in enumerate(FAMILIES)
        if item["name"] == case["family"])
    rng = random.Random(_case_seed(family_index, int(case["seed"])))
    mechanism = str(case["mechanism"])
    history_length = int(case["history_length"])
    horizon = int(case["horizon"])
    total = history_length + horizon
    phase = rng.uniform(0.0, 2 * math.pi)
    values: list[float] = []
    for index in range(total):
        noise_sd = 1.2 if mechanism == "noisy" else .6
        noise = rng.gauss(0.0, noise_sd)
        if mechanism in {"additive", "unadmitted", "insufficient"}:
            level = 100.0 + float(case["structural_slope"]) * index
            value = level + 8.0 * math.sin(
                2 * math.pi * index / PERIOD + phase) + noise
        elif mechanism == "multiplicative":
            level = 100.0 + float(case["structural_slope"]) * index
            value = level * (1.0 + .08 * math.sin(
                2 * math.pi * index / PERIOD + phase)) + noise
        elif mechanism == "noisy":
            value = 100.0 + float(case["structural_slope"]) * index + noise
        elif mechanism == "plateau":
            plateau_at = history_length - 72
            effective_index = min(index, plateau_at)
            value = 100.0 + float(case["pre_plateau_slope"]) * effective_index + noise
        elif mechanism == "changepoint":
            shift_at = history_length - 72
            value = 100.0 + (float(case["level_shift"])
                             if index >= shift_at else 0.0) + noise
        else:  # pragma: no cover - the frozen table is exhaustive
            raise ValueError(mechanism)
        values.append(value)
    return values[:history_length], values[history_length:]


def _cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for family in FAMILIES:
        mechanism = str(family["mechanism"])
        short = mechanism == "insufficient"
        for seed in SEEDS:
            structural_slope = (
                0.0 if mechanism in {"plateau", "changepoint"}
                else float(family.get("slope", 0.0)))
            cases.append({
                "case_id": f"{family['name']}-{seed}",
                "family": family["name"],
                "mechanism": mechanism,
                "seed": seed,
                "history_length": SHORT_HISTORY if short else LONG_HISTORY,
                "horizon": SHORT_HORIZON if short else LONG_HORIZON,
                "declared_period": 1 if mechanism == "unadmitted" else (
                    PERIOD if mechanism in {"additive", "multiplicative",
                                             "insufficient"} else 1),
                "oracle_period": PERIOD if mechanism in {
                    "additive", "multiplicative", "unadmitted", "insufficient"
                } else 1,
                "structural_slope": structural_slope,
                **({"pre_plateau_slope": float(family["slope"])}
                   if mechanism == "plateau" else {}),
                **({"level_shift": float(family["shift"])}
                   if mechanism == "changepoint" else {}),
                "identifiable": not short,
            })
    return cases


def _phase_fixed_slope(values: list[float], period: int) -> float | None:
    if len(values) < 2:
        return None
    if period <= 1:
        centre = (len(values) - 1) / 2
        denominator = sum((index - centre) ** 2
                          for index in range(len(values)))
        return (sum((index - centre) * value
                    for index, value in enumerate(values)) / denominator
                if denominator else None)
    if len(values) < 2 * period:
        return None
    phase_means = {
        phase: statistics.mean(values[phase::period])
        for phase in range(period)
    }
    phase_times = {
        phase: statistics.mean(range(phase, len(values), period))
        for phase in range(period)
    }
    numerator = denominator = 0.0
    for index, value in enumerate(values):
        phase = index % period
        centred_time = index - phase_times[phase]
        numerator += centred_time * (value - phase_means[phase])
        denominator += centred_time * centred_time
    return numerator / denominator if denominator else None


def _direction(value: float | None) -> str | None:
    if value is None:
        return None
    return "upward" if value > .02 else "downward" if value < -.02 else "constant"


def _direction_consistent(direction: str, estimate: float) -> bool:
    return _direction(estimate) == direction


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    history, future = _generate(case)
    history_before = list(history)
    question = compile_temporal_question({
        "id": "future_trend",
        "verb": "predict",
        "property": "trend",
        "target": "value",
        "measure": "slope",
        "horizon": case["horizon"],
    }, available_targets=["value"])
    engine = answer_scoped_question(
        question,
        reports={"value": {"frequency": "synthetic"}},
        execution_inputs={"value": (history, int(case["declared_period"]))},
    )
    best = engine.get("best_estimate") or {}
    answer = engine.get("answer") or {}
    support = engine.get("support") or {}
    interval = answer.get("interval") or {}
    direction = best.get("value")
    estimate = answer.get("estimate")
    state = support.get("state")
    automation = support.get("automation_eligible")
    scale = _innovation_scale(history, int(case["oracle_period"]))
    realized_slope = _phase_fixed_slope(future, int(case["oracle_period"]))
    realized = (realized_slope / scale
                if realized_slope is not None and scale > 0 else None)
    actual_direction = _direction(realized)
    numeric_complete = (
        direction in {"upward", "downward", "constant"}
        and state in {"supported", "weak"}
        and isinstance(estimate, (int, float))
        and math.isfinite(float(estimate))
    )
    abstention_complete = (
        direction in {None, "uncertain"}
        and state == "abstained"
        and estimate is None
        and interval in ({}, {"lower": None, "upper": None})
    )
    complete = (
        isinstance(automation, bool)
        and (numeric_complete or abstention_complete)
    )
    correct = (direction == actual_direction
               if actual_direction is not None and complete else None)
    if not complete:
        utility = -2.0
    elif actual_direction is None:
        utility = 0.0 if state == "abstained" else (
            -2.0 if state == "supported" else -.25)
    elif state == "supported":
        utility = 1.0 if correct else -2.0
    elif state == "weak":
        utility = .25 if correct else -.25
    else:
        utility = 0.0
    lower = interval.get("lower")
    upper = interval.get("upper")
    covered = (float(lower) <= realized <= float(upper)
               if realized is not None
               and isinstance(lower, (int, float))
               and isinstance(upper, (int, float)) else None)
    detected_period, detected_strength, detected_basis = detect_season(
        history, "synthetic")
    reasoning = answer.get("reasoning") or {}
    immutable = (
        history == history_before
        and (engine.get("synthesis_policy") or {}).get(
            "primary_forecast_unchanged") is True
        and reasoning.get("primary_forecast_unchanged") is True
    )
    interval_consistent = True
    if state == "supported" and isinstance(lower, (int, float)) \
            and isinstance(upper, (int, float)):
        interval_consistent = (
            _direction(float(lower)) == direction == _direction(float(upper)))
    return {
        **case,
        "evaluated_commit": code_revision(),
        "future_observations_used_by_engine": 0,
        "product_complete": complete,
        "inputs_unchanged": history == history_before,
        "primary_forecast_unchanged": immutable,
        "detected_period_diagnostic": detected_period,
        "detected_strength_diagnostic": detected_strength,
        "detected_basis_diagnostic": detected_basis,
        "direction": direction,
        "estimate": estimate,
        "interval": interval,
        "support": state,
        "automation_eligible": automation,
        "answer_support": answer.get("support"),
        "answer_automation_eligible": answer.get("automation_eligible"),
        "oracle_adjusted_slope": realized,
        "actual_direction": actual_direction,
        "direction_correct": correct,
        "interval_covered": covered,
        "direction_slope_consistent": (
            direction in {None, "uncertain"} and estimate is None
            if state == "abstained" else
            _direction_consistent(str(direction), float(estimate))
            if numeric_complete else False),
        "supported_interval_direction_consistent": interval_consistent,
        "authority_fields_agree": (
            automation == (state == "supported")
            and answer.get("support") == state
            and answer.get("automation_eligible") == automation
            and best.get("support") == state
            and best.get("automation_eligible") == automation),
        "selective_utility": utility,
        "headline": engine.get("headline"),
        "limitations": engine.get("limitations"),
        "engine_answer": engine,
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [bool(row[key]) for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    identifiable = [row for row in rows if row["actual_direction"] is not None]
    supported = [row for row in identifiable if row["support"] == "supported"]
    return {
        "cases": len(rows),
        "product_complete": sum(bool(row["product_complete"]) for row in rows),
        "identifiable_cases": len(identifiable),
        "direction_accuracy": _rate(identifiable, "direction_correct"),
        "supported_cases": len(supported),
        "supported_direction_accuracy": _rate(supported, "direction_correct"),
        "supported_interval_coverage": _rate(supported, "interval_covered"),
        "all_interval_coverage": _rate(identifiable, "interval_covered"),
        "abstained_cases": sum(row["support"] == "abstained" for row in rows),
        "weak_cases": sum(row["support"] == "weak" for row in rows),
        "automation_eligible_cases": sum(
            row["automation_eligible"] is True for row in rows),
        "mean_selective_utility": statistics.mean(
            float(row["selective_utility"]) for row in rows) if rows else None,
    }


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {
        str(family["name"]): _group([
            row for row in rows if row["family"] == family["name"]])
        for family in FAMILIES
    }
    supported = [row for row in rows if row["support"] == "supported"
                 and row["actual_direction"] is not None]
    unsafe = [row for row in rows if row["mechanism"] in {
        "unadmitted", "insufficient"} and row["automation_eligible"] is True]
    insufficient = [row for row in rows
                    if row["mechanism"] == "insufficient"]
    gates = {
        "all_152_product_cases_complete": (
            len(rows) == 152 and all(row["product_complete"] for row in rows)),
        "future_observations_used_zero": all(
            row["future_observations_used_by_engine"] == 0 for row in rows),
        "primary_forecast_and_inputs_immutable": all(
            row["primary_forecast_unchanged"] and row["inputs_unchanged"]
            for row in rows),
        "support_and_automation_fields_agree": all(
            row["authority_fields_agree"] for row in rows),
        "supported_direction_accuracy_at_least_90pct": (
            bool(supported) and _rate(supported, "direction_correct") >= .90),
        "supported_interval_coverage_at_least_80pct": (
            bool(supported) and _rate(supported, "interval_covered") >= .80),
        "unadmitted_and_insufficient_automation_zero": not unsafe,
        "insufficient_cycle_answers_abstain": all(
            row["support"] == "abstained" for row in insufficient),
        "direction_slope_interval_internally_consistent": all(
            row["direction_slope_consistent"]
            and row["supported_interval_direction_consistent"] for row in rows),
    }
    return {
        "schema_version": "0.1",
        "benchmark": "seasonal-trend-typed-answers",
        "evaluated_commit": code_revision(),
        "scope": "full",
        "overall": _group(rows),
        "by_family": by_family,
        "unsafe_automation_case_ids": [row["case_id"] for row in unsafe],
        "gates": gates,
        "passed": all(gates.values()),
        "raw_records": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "cases.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["case_id"]] = row
    for case in _cases():
        if case["case_id"] in completed:
            continue
        row = _run_case(case)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        completed[row["case_id"]] = row
    rows = [completed[case["case_id"]] for case in _cases()]
    summary = _summarise(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    write_manifest(
        args.output_dir,
        benchmark="trendanswerbench",
        condition="frozen-seasonal-trend-typed-answer-reproduction",
        target="152 property-stratified history-prefix cases",
        protocol="docs/v0.7-q2-seasonal-trend-protocol.md",
    )
    print(json.dumps({
        key: summary[key] for key in (
            "benchmark", "evaluated_commit", "overall", "gates", "passed")
    }, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
