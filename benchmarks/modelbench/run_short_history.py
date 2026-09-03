"""Independent short-history benchmark for classical and panel candidates.

Cases are generated without benchmark/task/channel labels entering production
code.  Model-family selection uses only a trailing slice of the training
history; the final horizon is untouched until scoring.  Panel candidates run
their production LOCO admission method and borrow only fold-bounded prefixes.
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
from benchmarks.modelbench.run_production_selector import _published_points
from gnomon.config import GnomonConfig
from gnomon.evaluation import evaluate
from gnomon.models import MODELS, last_value, predict
from gnomon.panel_pooling import PanelTrendCandidate


FAMILIES = ("level", "trend", "seasonal", "intermittent", "multiplicative")
GFR_SHORT_CASE_IDS = (
    "short:seasonal:one-cycle",
    "short:seasonal:two-cycles",
    "short:trend:two-horizons",
    "short:level:two-horizons",
    "short:intermittent:two-cycles",
    "short:noise:two-cycles",
)


def _series(rng: random.Random, family: str, length: int, direction: float = 1) -> list[float]:
    level = rng.uniform(20, 80)
    values: list[float] = []
    for index in range(length):
        if family == "level":
            value = level + rng.gauss(0, 1.2)
        elif family == "trend":
            value = level + direction * .9 * index + rng.gauss(0, .7)
        elif family == "seasonal":
            value = level + 6 * math.sin(2 * math.pi * index / 6) + rng.gauss(0, .7)
        elif family == "intermittent":
            value = rng.uniform(4, 12) if rng.random() < .22 else 0.0
        else:
            value = level * (1.025 ** index) * math.exp(rng.gauss(0, .025))
        values.append(value)
    return values


def _panel_series(rng: random.Random, length: int, slope: float,
                  noise: float) -> list[float]:
    level = rng.uniform(20, 80)
    return [level + slope * index + rng.gauss(0, noise)
            for index in range(length)]


def _loss(actual: list[float], points: list[float]) -> float:
    if len(actual) != len(points) or not actual:
        return float("inf")
    value = mean(abs(observed - predicted)
                 for observed, predicted in zip(actual, points))
    return value if math.isfinite(value) else float("inf")


def _classical_forecast(history: list[float], horizon: int, season: int) -> tuple[str, list[float]]:
    """Choose one production family on training-internal validation."""
    holdout = min(horizon, max(2, len(history) // 5))
    train, validation = history[:-holdout], history[-holdout:]
    losses: list[tuple[float, str]] = []
    for name in MODELS:
        try:
            loss = _loss(validation, predict(name, train, holdout, season))
            if math.isfinite(loss):
                losses.append((loss, name))
        except (ValueError, ArithmeticError, OverflowError):
            continue
    selected = min(losses)[1] if losses else "last_value"
    try:
        points = predict(selected, history, horizon, season)
    except (ValueError, ArithmeticError, OverflowError):
        selected, points = "last_value", last_value(history, horizon, season)
    if any(not math.isfinite(value) for value in points):
        selected, points = "last_value", last_value(history, horizon, season)
    return selected, points


def _outcome(candidate: float, baseline: float) -> str:
    if math.isclose(candidate, baseline, rel_tol=0, abs_tol=1e-12):
        return "safety_preservation"
    return "uplift" if candidate < baseline else "regression"


def _gfr_short_history_cases(seed: int) -> list[dict[str, object]]:
    """Run the production selector on six untouched final horizons."""
    rng = random.Random(seed ^ 0x6F726163)
    config = GnomonConfig()
    config.models.statsforecast_enabled = False

    def one(case_id: str, family: str, train_length: int, horizon: int,
            season: int, *, expected: str,
            noise_only: bool = False) -> dict[str, object]:
        if noise_only:
            level = rng.uniform(20, 80)
            complete = [level + rng.gauss(0, 2.0)
                        for _ in range(train_length + horizon)]
        else:
            complete = _series(rng, family, train_length + horizon)
        history, actual = complete[:train_length], complete[train_length:]
        baseline_points = last_value(history, horizon, season)
        assessment = evaluate(
            history, horizon, season, .02, frequency="synthetic",
            tsfm_names=[], strict_abstention=False, config=config,
        )
        candidate_points, selected, support, fallback_disclosed = (
            _published_points(assessment, history, horizon, season))
        baseline_loss = _loss(actual, baseline_points)
        selected_loss = _loss(actual, candidate_points)
        oracle_losses = []
        for name in MODELS:
            try:
                loss = _loss(actual, predict(name, history, horizon, season))
            except (ValueError, ArithmeticError, OverflowError):
                continue
            if math.isfinite(loss):
                oracle_losses.append((loss, name))
        oracle_loss, oracle_model = min(
            oracle_losses, default=(baseline_loss, "last_value"))
        expected_action = (
            "publish_candidate"
            if expected == "oracle" and baseline_loss > 0
            and oracle_loss <= .98 * baseline_loss
            else "retain_baseline")
        departed = any(not math.isclose(
            candidate, baseline, rel_tol=0, abs_tol=1e-12)
                       for candidate, baseline in zip(
                           candidate_points, baseline_points))
        return {
            "case_id": case_id, "family": family,
            "history_length": train_length, "horizon": horizon,
            "seasonal_period": season, "selected": selected,
            "published_support": support,
            "fallback_disclosed": fallback_disclosed,
            "engine_supported": assessment.supported,
            "selection_fold_count": assessment.selection_fold_count,
            "selection_guardrail_applied": (
                assessment.selection_guardrail_applied),
            "oracle_best_model": oracle_model,
            "oracle_best_loss": oracle_loss,
            "oracle_used_by_selector": False,
            "baseline_loss": baseline_loss, "selected_loss": selected_loss,
            "expected_action": expected_action,
            "actual_action": ("publish_candidate" if departed
                              else "retain_baseline"),
            "outcome": _outcome(selected_loss, baseline_loss),
            "selection_input_ends_before_scored_horizon": True,
        }

    cases = [
        one("short:seasonal:one-cycle", "seasonal", 6, 3, 6,
            expected="retain"),
        one("short:seasonal:two-cycles", "seasonal", 12, 3, 6,
            expected="oracle"),
        one("short:intermittent:two-cycles", "intermittent", 12, 3, 6,
            expected="retain"),
        one("short:noise:two-cycles", "noise", 12, 3, 6,
            expected="retain", noise_only=True),
    ]
    for case_id, family in (
            ("short:trend:two-horizons", "trend"),
            ("short:level:two-horizons", "level")):
        components = [one(case_id, family, 18, horizon, 1,
                          expected="oracle")
                      for horizon in (3, 6)]
        baseline_loss = mean(float(item["baseline_loss"])
                             for item in components)
        selected_loss = mean(float(item["selected_loss"])
                             for item in components)
        oracle_loss = mean(float(item["oracle_best_loss"])
                           for item in components)
        cases.append({
            "case_id": case_id, "family": family,
            "history_length": 18, "horizons": [3, 6],
            "selected": [item["selected"] for item in components],
            "oracle_best_model": [item["oracle_best_model"]
                                  for item in components],
            "oracle_best_loss": oracle_loss,
            "oracle_used_by_selector": False,
            "baseline_loss": baseline_loss, "selected_loss": selected_loss,
            "expected_action": ("publish_candidate"
                                if baseline_loss > 0
                                and oracle_loss <= .98 * baseline_loss
                                else "retain_baseline"),
            "actual_action": ("retain_baseline" if all(
                item["selected"] == "last_value" for item in components)
                else "publish_candidate"),
            "outcome": _outcome(selected_loss, baseline_loss),
            "selection_input_ends_before_scored_horizon": True,
            "components": components,
        })
    by_id = {str(item["case_id"]): item for item in cases}
    return [by_id[case_id] for case_id in GFR_SHORT_CASE_IDS]


def run(seed: int = 82631, cases_per_family: int = 40) -> dict[str, object]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        season = 6 if family == "seasonal" else 1
        for case in range(cases_per_family):
            horizon, train_length = 3, rng.choice((15, 18, 21, 24))
            complete = _series(rng, family, train_length + horizon)
            history, actual = complete[:train_length], complete[train_length:]
            baseline = last_value(history, horizon, season)
            selected, enhanced = _classical_forecast(history, horizon, season)
            base_loss, enhanced_loss = _loss(actual, baseline), _loss(actual, enhanced)
            rows.append({
                "case_id": f"{family}-{case}", "lane": "classical",
                "family": family, "history_length": train_length,
                "selected": selected, "baseline_loss": base_loss,
                "candidate_loss": enhanced_loss,
                "outcome": _outcome(enhanced_loss, base_loss),
            })

    panel_regimes = {
        "comparable_strong": (.9, .7, False),
        "comparable_marginal": (.25, 1.2, False),
        "null": (0.0, 1.2, False),
        "mixed_direction": (.9, .7, True),
    }
    for regime, (slope, noise, mixed_direction) in panel_regimes.items():
        for case in range(cases_per_family):
            horizon, train_length = 3, rng.choice((15, 18, 21, 24))
            full_panel: dict[str, list[float]] = {}
            for channel in range(5):
                direction = -1 if mixed_direction and channel % 2 else 1
                full_panel[f"series_{channel}"] = _panel_series(
                    rng, train_length + horizon, direction * slope, noise)
            train_panel = {name: values[:train_length]
                           for name, values in full_panel.items()}
            candidate = PanelTrendCandidate("series_0", train_panel)
            evidence = candidate.lightweight_evidence(horizon, 1, .02)
            history = train_panel["series_0"]
            actual = full_panel["series_0"][train_length:]
            baseline = last_value(history, horizon, 1)
            points = candidate(train_length, horizon) if evidence else baseline
            base_loss, published_loss = _loss(actual, baseline), _loss(actual, points)
            rows.append({
                "case_id": f"panel-{regime}-{case}",
                "lane": "pooling", "family": regime,
                "history_length": train_length, "admitted": evidence is not None,
                "baseline_loss": base_loss, "candidate_loss": published_loss,
                "outcome": _outcome(published_loss, base_loss),
            })

    def summary(items: list[dict[str, object]]) -> dict[str, object]:
        gains = [(float(row["baseline_loss"]) - float(row["candidate_loss"]))
                 / max(float(row["baseline_loss"]), 1e-12) for row in items]
        return {
            "cases": len(items),
            "median_relative_gain": median(gains),
            # Means of ratios are deliberately omitted: near-zero intermittent
            # horizons make them arbitrarily large despite negligible MAE.
            "baseline_median_mae": median(
                float(row["baseline_loss"]) for row in items),
            "candidate_median_mae": median(
                float(row["candidate_loss"]) for row in items),
            "outcomes": {name: sum(row["outcome"] == name for row in items)
                         for name in ("uplift", "safety_preservation", "regression")},
        }

    classical = [row for row in rows if row["lane"] == "classical"]
    pooling = [row for row in rows if row["lane"] == "pooling"]
    by_regime = {regime: [row for row in pooling if row["family"] == regime]
                 for regime in panel_regimes}

    def pooling_summary(items: list[dict[str, object]]) -> dict[str, object]:
        result = summary(items)
        admitted = [row for row in items if row["admitted"]]
        harmful = sum(row["outcome"] == "regression" for row in admitted)
        return {**result, "admitted": len(admitted),
                "admission_rate": len(admitted) / len(items),
                "admission_precision": (
                    sum(row["outcome"] == "uplift" for row in admitted)
                    / len(admitted) if admitted else None),
                "harmful_admission_rate": harmful / len(admitted)
                if admitted else 0.0}
    result = {
        "schema_version": "0.1", "seed": seed,
        "protocol": "training-internal selection; untouched final horizon",
        "classical": summary(classical),
        "classical_by_family": {family: summary(
            [row for row in classical if row["family"] == family])
            for family in FAMILIES},
        "pooling": {regime: pooling_summary(items)
                    for regime, items in by_regime.items()},
        "precision_gate_min_admissions": 20,
        "raw_records": rows,
        "gfr_cases": _gfr_short_history_cases(seed),
    }
    result["gates"] = {
        "classical_median_gain_positive": result["classical"]["median_relative_gain"] > 0,
        "strong_pooling_admission_precision_at_least_70pct": (
            result["pooling"]["comparable_strong"]["admitted"] < 20
            or result["pooling"]["comparable_strong"]["admission_precision"] >= .70),
        "null_admission_rate_below_05": (
            result["pooling"]["null"]["admission_rate"] < .05),
        "marginal_admission_rate_below_05": (
            result["pooling"]["comparable_marginal"]["admission_rate"] < .05),
        "marginal_harmful_admissions_zero": (
            result["pooling"]["comparable_marginal"]["outcomes"]["regression"] == 0),
        "mixed_direction_admission_rate_below_05": (
            result["pooling"]["mixed_direction"]["admission_rate"] < .05),
        "raw_records_retained": len(rows) == cases_per_family * 9,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=82631)
    parser.add_argument("--cases-per-family", type=int, default=40)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run(args.seed, args.cases_per_family)
    result["evaluated_commit"] = code_revision()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_manifest(args.output_dir, benchmark="modelbench",
                       condition="short-history-classical-and-pooling",
                       target=f"seed={args.seed};cases_per_family={args.cases_per_family}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(result["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
