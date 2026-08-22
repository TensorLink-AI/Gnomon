"""Deterministic matched benchmark for fitted volatility executables."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import statistics
import sys
import argparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gnomon.volatility import fit_volatility_executable, residuals  # noqa: E402


def _case(seed: int, regime: str, scale: float) -> tuple[list[float], float, str]:
    rng = random.Random(seed)
    values, level = [], 100.0
    for index in range(360):
        fraction = index / 359
        sigma = scale
        if regime == "increase": sigma *= 1 + 2 * fraction
        elif regime == "decrease": sigma *= 3 - 2 * fraction
        elif regime == "abrupt": sigma *= 1 if index < 240 else 2.5
        elif regime == "seasonal": sigma *= 1.8 if index % 24 < 8 else .7
        noise = rng.gauss(0, sigma)
        if regime == "heavy_tail" and rng.random() < .04:
            noise *= 6
        level += .02
        values.append(level + noise)
    season = 24 if regime == "seasonal" else 1
    prefix_count = len(residuals(values, season))
    realised_scales = []
    for draw in range(128):
        future_rng = random.Random(seed * 1000 + draw)
        future_values, future_level = [], level
        for index in range(16):
            fraction = (360 + index) / 359
            sigma = scale
            if regime == "increase": sigma *= 1 + 2 * fraction
            elif regime == "decrease": sigma *= max(.5, 3 - 2 * fraction)
            elif regime == "abrupt": sigma *= 2.5
            elif regime == "seasonal": sigma *= 1.8 if (360 + index) % 24 < 8 else .7
            noise = future_rng.gauss(0, sigma)
            if regime == "heavy_tail" and future_rng.random() < .04:
                noise *= 6
            future_level += .02
            future_values.append(future_level + noise)
        held_out = residuals(values + future_values, season)[prefix_count:]
        centre = statistics.median(held_out)
        realised_scales.append(1.4826 * statistics.median(
            abs(item - centre) for item in held_out))
    oracle = statistics.median(realised_scales)
    recent = residuals(values, season)[-48:]
    recent_scale = 1.4826 * statistics.median(
        abs(item - statistics.median(recent)) for item in recent)
    direction = ("increased" if oracle / recent_scale > 1.25 else
                 "decreased" if oracle / recent_scale < .8 else "stable")
    return values, max(oracle, 1e-12), direction


def _balanced_case(seed: int, direction: str, shape: str,
                   scale: float) -> tuple[list[float], float, str]:
    """Predictable but non-trivial direction case with a sealed oracle.

    Direction is balanced by construction, while the mechanism varies. The
    prefix contains only an evolving variance signature; no label or future
    parameter reaches the executable. ``step`` and ``heavy_tail`` are treated
    as held-out generator families in the report.
    """
    rng = random.Random(seed)
    values, level = [], 100.0
    length, horizon = 320, 16
    sign = 1 if direction == "increased" else -1 if direction == "decreased" else 0

    def prefix_sigma(index: int) -> float:
        progress = max(0.0, (index - (length - 96)) / 95)
        if shape == "ramp":
            factor = ((.5 + 1.1 * progress) if sign > 0 else
                      (2.0 - 1.4 * progress) if sign < 0 else 1.0)
        elif shape == "step":
            factor = ((1.8 if index >= length - 24 else 1.0) if sign > 0 else
                      (.45 if index >= length - 24 else 1.0) if sign < 0 else 1.0)
        elif shape == "seasonal":
            phase = .25 * math.sin(2 * math.pi * index / 48)
            factor = max(.2, 1 + phase + sign * .8 * progress)
        else:  # heavy_tail: a robust trend plus contamination.
            factor = ((.55 + .95 * progress) if sign > 0 else
                      (1.8 - 1.2 * progress) if sign < 0 else 1.0)
        return max(.2, scale * factor)

    for index in range(length):
        noise = rng.gauss(0, prefix_sigma(index))
        if shape == "heavy_tail" and rng.random() < .035:
            noise *= 5
        level += .015
        values.append(level + noise)

    prefix_residual_count = len(residuals(values))
    realised_scales = []
    # The transition visible near the end of the prefix continues, rather
    # than a magical cutoff-only jump. Stable cases retain the current scale.
    end_scale = prefix_sigma(length - 1)
    future_factor = 2.2 if sign > 0 else .30 if sign < 0 else 1.0
    for draw in range(128):
        future_rng = random.Random(seed * 10_000 + draw)
        future, future_level = [], level
        for offset in range(horizon):
            progress = (offset + 1) / horizon
            sigma = end_scale * (1 + (future_factor - 1) * progress)
            noise = future_rng.gauss(0, max(.05, sigma))
            if shape == "heavy_tail" and future_rng.random() < .035:
                noise *= 5
            future_level += .015
            future.append(future_level + noise)
        held_out = residuals(values + future)[prefix_residual_count:]
        realised_scales.append(_robust_scale(held_out))
    oracle = statistics.median(realised_scales)
    recent = residuals(values)[-48:]
    recent_scale = max(_robust_scale(recent), 1e-12)
    observed_direction = ("increased" if oracle / recent_scale > 1.25 else
                          "decreased" if oracle / recent_scale < .8 else "stable")
    return values, max(oracle, 1e-12), observed_direction


def _robust_scale(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    centre = statistics.median(values)
    mad = 1.4826 * statistics.median(abs(item - centre) for item in values)
    return mad if mad > 1e-12 else statistics.stdev(values)


def _short_history_case(seed: int, regime: str, length: int,
                        scale: float) -> tuple[list[float], float, str]:
    """Short-history case with a sealed 128-draw oracle.

    These prefixes are too short for rolling-origin folds, so the fitted
    executable runs on its short-history fallback path.  The oracle is the
    realized future residual scale relative to the prefix's own reference
    window — the exact quantity the executable publishes.
    """
    rng = random.Random(seed)
    values, level = [], 100.0
    horizon = 8

    def sigma(index: int) -> float:
        fraction = index / max(length - 1, 1)
        if regime == "ramp_up":
            return scale * (1 + 2 * fraction)
        if regime == "ramp_down":
            return scale * (3 - 2 * fraction)
        if regime == "late_jump":
            return scale * (2.5 if index >= length - length // 4 else 1.0)
        if regime == "late_calm":
            return scale * (1.0 if index >= length - length // 4 else 2.5)
        return scale

    for index in range(length):
        level += .03
        values.append(level + rng.gauss(0, sigma(index)))
    prefix_count = len(residuals(values))
    end_sigma = sigma(length - 1)
    realised_scales = []
    for draw in range(128):
        future_rng = random.Random(seed * 10_000 + draw)
        future, future_level = [], level
        for offset in range(horizon):
            step = end_sigma
            if regime in {"ramp_up", "ramp_down"}:
                step = sigma(length - 1 + offset + 1) if regime == "ramp_up" \
                    else max(.3 * scale, sigma(length - 1) - 2 * scale
                             * (offset + 1) / max(length - 1, 1))
            future_level += .03
            future.append(future_level + future_rng.gauss(0, max(.05, step)))
        held_out = residuals(values + future)[prefix_count:]
        if len(held_out) >= 2:
            realised_scales.append(_robust_scale(held_out))
    oracle = statistics.median(realised_scales)
    errors = residuals(values)
    window = max(8, min(48, len(errors) // 2))
    reference = max(_robust_scale(errors[-window:]), 1e-12)
    ratio = oracle / reference
    direction = ("increased" if ratio > 1.25 else
                 "decreased" if ratio < .8 else "stable")
    return values, max(oracle, 1e-12), direction


def _short_history_suite() -> dict[str, object]:
    """Measure the short-history fallback against honest baselines.

    ``abstain_all`` is the pre-change behaviour (every short history answered
    "uncertain"); ``always_stable`` is the persistence-null baseline. The
    fallback graduates only if its best-estimate accuracy beats both.
    """
    rows: list[dict[str, object]] = []
    regimes = ("flat", "ramp_up", "ramp_down", "late_jump", "late_calm")
    for regime_index, regime in enumerate(regimes):
        for length_index, length in enumerate((28, 40, 60)):
            for scale_index, scale in enumerate((.5, 2.0)):
                for replicate in range(8):
                    seed = (500_000 + regime_index * 10_000
                            + length_index * 1_000 + scale_index * 100
                            + replicate)
                    values, oracle, direction = _short_history_case(
                        seed, regime, length, scale)
                    fitted = fit_volatility_executable(values, horizon=8)
                    estimate = max(fitted.scale, 1e-12)
                    rows.append({
                        "regime": regime, "length": length, "scale": scale,
                        "seed": seed, "candidate": fitted.candidate,
                        "direction": fitted.direction,
                        "oracle_direction": direction,
                        "fallback_used": bool(fitted.diagnostics.get(
                            "short_history_point_estimate")),
                        "correct": fitted.direction == direction,
                        "absolute_log_scale_error": abs(
                            math.log(estimate / oracle)),
                    })
    labels = ("increased", "stable", "decreased")
    class_counts = {label: sum(row["oracle_direction"] == label
                               for row in rows) for label in labels}
    recalls = {}
    for label in labels:
        members = [row for row in rows if row["oracle_direction"] == label]
        recalls[label] = (statistics.mean(row["correct"] for row in members)
                         if members else None)
    accuracy = statistics.mean(row["correct"] for row in rows)
    always_stable = statistics.mean(
        row["oracle_direction"] == "stable" for row in rows)
    return {
        "cases": len(rows),
        "fallback_rate": statistics.mean(row["fallback_used"] for row in rows),
        "uncertain_rate": statistics.mean(
            row["direction"] == "uncertain" for row in rows),
        "best_estimate_direction_accuracy": accuracy,
        "class_counts": class_counts,
        "class_recall": recalls,
        "balanced_direction_accuracy": statistics.mean(
            value for value in recalls.values() if value is not None),
        "abstain_all_baseline_accuracy": 0.0,
        "always_stable_baseline_accuracy": always_stable,
        "beats_always_stable": accuracy > always_stable,
        "mean_absolute_log_scale_error": statistics.mean(
            row["absolute_log_scale_error"] for row in rows),
        "rows": rows,
    }


def _direction_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    labels = ("increased", "stable", "decreased")
    recalls = {}
    for label in labels:
        members = [row for row in rows if row["oracle_direction"] == label]
        recalls[label] = (statistics.mean(
            row["best_estimate_direction_correct"] for row in members)
            if members else None)
    balanced = statistics.mean(value for value in recalls.values()
                               if value is not None)
    mean_brier = statistics.mean(row["brier"] for row in rows)
    uniform_brier = 2 / 3
    return {
        "cases": len(rows),
        "class_counts": {label: sum(
            row["oracle_direction"] == label for row in rows) for label in labels},
        "class_recall": recalls,
        "balanced_direction_accuracy": balanced,
        "mean_brier": mean_brier,
        "uniform_brier": uniform_brier,
        "brier_skill_vs_uniform": (uniform_brier - mean_brier) / uniform_brier,
        "claim_rate": statistics.mean(
            row["direction_support"] == "supported" for row in rows),
        "claimed_accuracy": (statistics.mean(
            row["best_estimate_direction_correct"] for row in rows
            if row["direction_support"] == "supported")
            if any(row["direction_support"] == "supported" for row in rows)
            else None),
    }


def _balanced_subset(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Deterministically equalize oracle classes after sealing futures."""
    labels = ("increased", "stable", "decreased")
    groups = {label: sorted(
        (row for row in rows if row["oracle_direction"] == label),
        key=lambda row: int(row["seed"])) for label in labels}
    count = min((len(group) for group in groups.values()), default=0)
    return [row for label in labels for row in groups[label][:count]]


def run() -> dict[str, object]:
    rows = []
    regimes = ("constant", "increase", "decrease", "abrupt", "seasonal",
               "heavy_tail")
    scales = (.5, 1.5, 4.0)
    for regime_index, regime in enumerate(regimes):
        for scale_index, scale in enumerate(scales):
            for replicate in range(10):
                seed = 1000 + regime_index * 10_000 + scale_index * 100 + replicate
                values, oracle, direction = _case(seed, regime, scale)
                fitted = fit_volatility_executable(
                    values, horizon=16, season=24 if regime == "seasonal" else 1)
                baseline = fit_volatility_executable(
                    values, horizon=16,
                    season=24 if regime == "seasonal" else 1,
                    minimum_improvement=float("inf"))
                estimate = max(fitted.scale, 1e-12)
                baseline_error = abs(math.log(max(baseline.scale, 1e-12) / oracle))
                variance_ratio = estimate * estimate / (oracle * oracle)
                rows.append({
                    "regime": regime, "scale": scale,
                    "seed": seed,
                    "candidate": fitted.candidate, "support": fitted.support,
                    "direction": fitted.direction,
                    "direction_support": fitted.direction_support,
                    "direction_probability": max(
                        fitted.direction_probabilities.values()),
                    "oracle_direction": direction,
                    "absolute_log_scale_error": abs(math.log(estimate / oracle)),
                    "constant_baseline_log_scale_error": baseline_error,
                    "beats_constant": abs(math.log(estimate / oracle)) < baseline_error,
                    "qlike": variance_ratio - math.log(variance_ratio) - 1,
                    "covered": fitted.lower <= oracle <= fitted.upper,
                    "best_estimate_direction_correct": fitted.direction == direction,
                    "direction_correct": (fitted.direction == direction)
                    if fitted.direction_support == "supported" else None,
                })
    supported = [row for row in rows if row["support"] == "supported"]
    direction_claims = [row for row in rows
                        if row["direction_correct"] is not None
                        and row["direction_support"] == "supported"]
    direction_counts = {
        label: sum(row["oracle_direction"] == label for row in rows)
        for label in ("increased", "stable", "decreased")
    }
    majority_direction_accuracy = max(direction_counts.values()) / len(rows)
    best_estimate_direction_accuracy = statistics.mean(
        row["best_estimate_direction_correct"] for row in rows)
    balanced_rows = []
    for shape_index, shape in enumerate(("ramp", "seasonal", "step", "heavy_tail")):
        for direction_index, direction in enumerate(
                ("increased", "stable", "decreased")):
            for scale_index, scale in enumerate((.5, 2.0, 5.0)):
                for replicate in range(8):
                    seed = (100_000 + shape_index * 10_000
                            + direction_index * 1_000 + scale_index * 100 + replicate)
                    values, oracle, oracle_direction = _balanced_case(
                        seed, direction, shape, scale)
                    fitted = fit_volatility_executable(values, horizon=16)
                    baseline = fit_volatility_executable(
                        values, horizon=16, minimum_improvement=float("inf"))
                    estimate = max(fitted.scale, 1e-12)
                    probabilities = fitted.direction_probabilities
                    brier = sum((probabilities[label]
                                 - (label == oracle_direction)) ** 2
                                for label in ("increased", "stable", "decreased"))
                    balanced_rows.append({
                        "shape": shape, "requested_direction": direction,
                        "oracle_direction": oracle_direction, "scale": scale,
                        "seed": seed, "candidate": fitted.candidate,
                        "direction_candidate": fitted.direction_candidate,
                        "direction": fitted.direction,
                        "direction_support": fitted.direction_support,
                        "best_estimate_direction_correct": (
                            fitted.direction == oracle_direction),
                        "absolute_log_scale_error": abs(math.log(estimate / oracle)),
                        "constant_baseline_log_scale_error": abs(math.log(
                            max(baseline.scale, 1e-12) / oracle)),
                        "brier": brier,
                    })
    balanced_selected = _balanced_subset(balanced_rows)
    development = _balanced_subset([row for row in balanced_rows
                                    if row["shape"] in {"ramp", "seasonal"}])
    held_out = _balanced_subset([row for row in balanced_rows
                                 if row["shape"] in {"step", "heavy_tail"}])
    short_history = _short_history_suite()
    short_history_rows = short_history.pop("rows")
    return {"schema_version": "0.4", "cases": len(rows),
            "short_history_suite": short_history,
            "short_history_rows": short_history_rows,
            "independent_seeds": len({row["seed"] for row in rows}),
            "supported": len(supported),
            "mean_absolute_log_scale_error": statistics.mean(
                row["absolute_log_scale_error"] for row in rows),
            "constant_baseline_mean_log_scale_error": statistics.mean(
                row["constant_baseline_log_scale_error"] for row in rows),
            "win_rate_vs_constant": statistics.mean(
                row["beats_constant"] for row in rows),
            "supported_win_rate_vs_constant": statistics.mean(
                row["beats_constant"] for row in supported) if supported else None,
            "mean_qlike": statistics.mean(row["qlike"] for row in rows),
            "interval_coverage": statistics.mean(row["covered"] for row in rows),
            "direction_claim_rate": len(direction_claims) / len(rows),
            "claimed_direction_accuracy": (statistics.mean(
                row["direction_correct"] for row in direction_claims)
                if direction_claims else None),
            "best_estimate_direction_accuracy": best_estimate_direction_accuracy,
            "majority_direction_baseline_accuracy": majority_direction_accuracy,
            "best_estimate_beats_majority_direction_baseline": (
                best_estimate_direction_accuracy > majority_direction_accuracy),
            "balanced_suite": {
                "all": _direction_metrics(balanced_selected),
                "development_families": _direction_metrics(development),
                "held_out_families": _direction_metrics(held_out),
                "by_shape": {
                    shape: _direction_metrics([
                        row for row in balanced_selected
                        if row["shape"] == shape])
                    for shape in sorted({str(row["shape"])
                                         for row in balanced_selected})
                },
                "held_out_shapes": ["step", "heavy_tail"],
                "mean_absolute_log_scale_error": statistics.mean(
                    row["absolute_log_scale_error"] for row in balanced_selected),
                "constant_baseline_mean_log_scale_error": statistics.mean(
                    row["constant_baseline_log_scale_error"]
                    for row in balanced_selected),
            },
            "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = run()
    if args.output_dir:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"rows", "short_history_rows"}},
                     indent=2, sort_keys=True))
    # Volatility direction is intentionally a diagnostic until its independent
    # balanced suite graduates; benchmark execution itself remains successful.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
