"""Property-specific, held-out synthetic benchmark with graduation gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gnomon.temporal_executables import (  # noqa: E402
    _property_value, fit_dependence_executable, fit_temporal_executable,
)
from gnomon.temporal_reasoning import (  # noqa: E402
    _forecast_volatility_alignment, _seasonality_alignment,
)
from gnomon.temporal_evidence import (  # noqa: E402
    aggregate_evidence, compare_windows,
)


def _paths(seed: int, regime: str, *, n: int = 420,
           horizon: int = 24) -> tuple[list[float], list[float], int]:
    rng = random.Random(seed)
    season = 48 if regime == "season_shift" else 24 if "season" in regime else 1
    values: list[float] = []
    for index in range(n + horizon):
        slope = .04 if regime in {"up", "season_up"} else -.04 if regime == "down" else 0
        level = 100 + slope * index
        if regime == "shift" and index >= n:
            level += 8
        seasonal = (5 * math.sin(2 * math.pi * index / 24)
                    if "season" in regime else 0)
        if regime == "season_shift":
            level += 8 if ((index + 12) // 24) % 2 == 0 else 0
        noise = rng.gauss(0, .6)
        if regime == "extreme" and index >= n and index == n + 8:
            noise += 9
        if regime == "season_extreme" and index % 24 == 8:
            noise += 9
        if regime == "heavy_extreme" and rng.random() < .1:
            noise += rng.choice((-1, 1)) * 7
        values.append(level + seasonal + noise)
    return values[:n], values[n:], season


def run(*, seed: int = 9100, replicates: int = 12) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    regimes = {
        "level": ("flat", "up", "down", "shift"),
        "trend": ("flat", "up", "down", "season_up"),
        "seasonality": ("flat", "season", "season_up"),
        "regime": ("flat", "up", "shift", "season_shift"),
        "extreme": ("flat", "extreme", "season", "season_extreme",
                    "heavy_extreme"),
    }
    for prop, names in regimes.items():
        for regime_index, regime in enumerate(names):
            for replicate in range(replicates):
                case_seed = seed + 10_000 * list(regimes).index(prop) + 100 * regime_index + replicate
                history, future, season = _paths(case_seed, regime)
                fitted = fit_temporal_executable(
                    history, property=prop, horizon=len(future), season=season)
                actual = _property_value(prop, history, future, season)
                from gnomon.temporal_executables import _label
                actual_label = _label(prop, actual)
                rows.append({
                    "property": prop, "regime": regime, "seed": case_seed,
                    "estimate": fitted.estimate, "actual": actual,
                    "absolute_error": abs(fitted.estimate - actual),
                    "claimed": fitted.support == "supported",
                    "direction": fitted.direction, "actual_direction": actual_label,
                    "best_estimate_direction_correct": fitted.direction == actual_label,
                    "direction_correct": (fitted.direction == actual_label
                                          if fitted.support == "supported" else None),
                    "covered": fitted.lower <= actual <= fitted.upper,
                    "primary_forecast_unchanged": fitted.diagnostics[
                        "primary_forecast_unchanged"],
                })

    # Dependence is generated independently and scored in the same currency.
    for relation, coefficient in (("positive", .9), ("weak", 0), ("negative", -.9)):
        for replicate in range(replicates * 2):
            case_seed = seed + 90_000 + int((coefficient + 1) * 1000) + replicate
            rng = random.Random(case_seed)
            left = [0.0]
            right = [0.0]
            shocks: list[tuple[float, float]] = []
            for _ in range(444):
                x = rng.gauss(0, 1)
                y = coefficient * x + rng.gauss(0, .25 if coefficient else 1)
                left.append(left[-1] + x)
                right.append(right[-1] + y)
                shocks.append((x, y))
            fitted = fit_dependence_executable(left[:421], right[:421], horizon=24)
            future = shocks[420:444]
            from gnomon.temporal_executables import _correlation
            actual = _correlation([x for x, _ in future], [y for _, y in future])
            actual_label = ("positive" if actual is not None and actual >= .3 else
                            "negative" if actual is not None and actual <= -.3 else "weak")
            rows.append({
                "property": "dependence", "regime": relation, "seed": case_seed,
                "estimate": fitted.estimate, "actual": actual,
                "absolute_error": abs((fitted.estimate or 0) - (actual or 0)),
                "claimed": fitted.support == "supported",
                "direction": fitted.direction, "actual_direction": actual_label,
                "best_estimate_direction_correct": fitted.direction == actual_label,
                "direction_correct": (fitted.direction == actual_label
                                      if fitted.support == "supported" else None),
                "covered": (fitted.lower is not None and actual is not None
                            and fitted.lower <= actual <= fitted.upper),
                "primary_forecast_unchanged": fitted.diagnostics[
                    "primary_forecast_unchanged"],
            })

    by_property = {}
    for prop in sorted({str(row["property"]) for row in rows}):
        subset = [row for row in rows if row["property"] == prop]
        claims = [row for row in subset if row["claimed"]]
        by_property[prop] = {
            "cases": len(subset), "claim_rate": len(claims) / len(subset),
            "claimed_direction_accuracy": (statistics.mean(
                bool(row["direction_correct"]) for row in claims) if claims else None),
            "best_estimate_direction_accuracy": statistics.mean(
                bool(row["best_estimate_direction_correct"]) for row in subset),
            "interval_coverage": (statistics.mean(bool(row["covered"]) for row in claims)
                                  if claims else None),
            "mean_absolute_error": statistics.mean(float(row["absolute_error"]) for row in subset),
        }
    claims = [row for row in rows if row["claimed"]]
    alignment_rows = []
    for period in (12, 24, 48):
        for expected in ("continued", "shifting", "absent"):
            for replicate in range(replicates):
                case_seed = seed + 200_000 + period * 100 + replicate
                rng = random.Random(case_seed)
                length = 8 * period
                history = [10 + 4 * math.sin(2 * math.pi * index / period)
                           + rng.gauss(0, .35) for index in range(length)]
                if expected == "continued":
                    shift = 0
                    future = [10 + 4 * math.sin(
                        2 * math.pi * (length + index - shift) / period)
                              + rng.gauss(0, .35) for index in range(2 * period)]
                elif expected == "shifting":
                    shift = max(3, period // 4)
                    future = [10 + 4 * math.sin(
                        2 * math.pi * (length + index - shift) / period)
                              + rng.gauss(0, .35) for index in range(2 * period)]
                else:
                    future = [10 + rng.gauss(0, .35)
                              for _ in range(2 * period)]
                observed = _seasonality_alignment(history, future, period)
                alignment_rows.append({
                    "period": period, "expected": expected,
                    "observed": observed["direction"], "seed": case_seed,
                    "correct": observed["direction"] == expected,
                })
    alignment_recalls = {
        label: statistics.mean(bool(row["correct"]) for row in alignment_rows
                               if row["expected"] == label)
        for label in ("continued", "shifting", "absent")
    }
    # Mechanism-held-out alignment stress. The original sinusoid-only lane is
    # intentionally easy; production sees harmonics, asymmetric cycles,
    # amplitude drift and signals close to the noise floor. These cases share
    # no TemporalBench labels or series and report each family separately so
    # one convenient waveform cannot hide a failure mode.
    alignment_stress_rows = []
    families = ("harmonic", "asymmetric", "amplitude_drift", "low_snr")
    for family_index, family in enumerate(families):
        for expected in ("continued", "shifting", "absent"):
            for replicate in range(replicates * 2):
                period = (12, 24, 48)[replicate % 3]
                case_seed = (seed + 250_000 + family_index * 10_000
                             + replicate + 100 * ("continued", "shifting",
                                                  "absent").index(expected))
                rng = random.Random(case_seed)
                length = 7 * period
                noise = 1.5 if family == "low_snr" else .55

                def wave(index: int, *, shift: float = 0.0,
                         future: bool = False) -> float:
                    phase = 2 * math.pi * (index - shift) / period
                    if family == "harmonic":
                        shape = math.sin(phase) + .45 * math.sin(2 * phase + .4)
                    elif family == "asymmetric":
                        shape = (1.0 if math.sin(phase) >= .35 else
                                 -.55 if math.sin(phase) <= -.65 else
                                 math.sin(phase))
                    else:
                        shape = math.sin(phase)
                    amplitude = 3.0
                    if family == "amplitude_drift":
                        amplitude *= (.65 + .7 * index /
                                      max(length + 2 * period - 1, 1))
                    return 10 + amplitude * shape + rng.gauss(0, noise)

                history = [wave(index) for index in range(length)]
                if expected == "absent":
                    future = [10 + rng.gauss(0, noise)
                              for _ in range(2 * period)]
                else:
                    phase_shift = 0 if expected == "continued" else period // 4
                    future = [wave(length + index, shift=phase_shift,
                                   future=True)
                              for index in range(2 * period)]
                observed = _seasonality_alignment(history, future, period)
                alignment_stress_rows.append({
                    "family": family, "period": period, "expected": expected,
                    "observed": observed["direction"], "seed": case_seed,
                    "correct": observed["direction"] == expected,
                })
    stress_by_family = {}
    for family in families:
        subset = [row for row in alignment_stress_rows
                  if row["family"] == family]
        recalls = {
            label: statistics.mean(bool(row["correct"]) for row in subset
                                   if row["expected"] == label)
            for label in ("continued", "shifting", "absent")
        }
        stress_by_family[family] = {
            "class_recall": recalls,
            "balanced_accuracy": statistics.mean(recalls.values()),
        }
    stress_balanced = statistics.mean(
        row["balanced_accuracy"] for row in stress_by_family.values())
    # Independent panel cases exercise the distinction between temporal folds
    # and breadth across related signals.  Magnitude, noise, and the number of
    # disagreeing channels vary; no TemporalBench labels or channel names are
    # used here.
    panel_rows = []
    for expected, multiplier in (("increased", 2.5), ("decreased", .35),
                                 ("stable", 1.0)):
        for replicate in range(replicates * 3):
            evidence = {}
            for member in range(7):
                case_seed = seed + 300_000 + replicate * 100 + member
                rng = random.Random(case_seed)
                reference = [rng.gauss(0, 1) for _ in range(48)]
                # One dissenting channel prevents a synthetic unanimity test.
                factor = (1.0 if member == replicate % 7 else multiplier)
                recent = [rng.gauss(0, factor) for _ in range(48)]
                evidence[str(member)] = compare_windows(
                    reference + recent, window=48)
            result = aggregate_evidence(evidence, property="volatility")
            panel_rows.append({
                "expected": expected, "observed": result["direction"],
                "support": result["support"],
                "effective_series": result["effective_series"],
                "correct": result["direction"] == expected,
            })
    panel_recalls = {
        label: statistics.mean(bool(row["correct"]) for row in panel_rows
                               if row["expected"] == label)
        for label in ("increased", "decreased", "stable")
    }
    # A published forecast is already an immutable computed path. Questions
    # about its dispersion must describe that path rather than launch a second
    # history-only volatility prediction. These fresh paths vary level and
    # trend so the check is specifically about residual dispersion.
    forecast_volatility_rows = []
    for expected, multiplier in (("increased", 2.2), ("decreased", .35),
                                 ("stable", 1.0)):
        for replicate in range(replicates * 3):
            case_seed = seed + 400_000 + replicate
            rng = random.Random(case_seed)
            history = [100 + .03 * index + rng.gauss(0, 1)
                       for index in range(180)]
            anchor = history[-1]
            future = [anchor + .03 * (index + 1)
                      + rng.gauss(0, multiplier)
                      for index in range(36)]
            result = _forecast_volatility_alignment(history, future, 1)
            forecast_volatility_rows.append({
                "expected": expected, "observed": result["direction"],
                "ratio": result["future_to_reference_ratio"],
                "correct": result["direction"] == expected,
                "seed": case_seed,
            })
    forecast_volatility_recalls = {
        label: statistics.mean(bool(row["correct"])
                               for row in forecast_volatility_rows
                               if row["expected"] == label)
        for label in ("increased", "decreased", "stable")
    }
    gates = {
        "complete": len(rows) == sum(len(v) * replicates for v in regimes.values()) + 6 * replicates,
        "claimed_direction_accuracy_at_least_70pct": bool(claims) and statistics.mean(
            bool(row["direction_correct"]) for row in claims) >= .7,
        "supported_interval_coverage_at_least_70pct": statistics.mean(
            bool(row["covered"]) for row in claims) >= .7,
        "claim_rate_at_least_10pct": len(claims) / len(rows) >= .1,
        "best_estimate_direction_accuracy_at_least_60pct": statistics.mean(
            bool(row["best_estimate_direction_correct"]) for row in rows) >= .6,
        "primary_immutable": all(bool(row["primary_forecast_unchanged"]) for row in rows),
        "seasonality_alignment_balanced_accuracy_at_least_80pct": (
            statistics.mean(alignment_recalls.values()) >= .8),
        "seasonality_alignment_stress_balanced_accuracy_at_least_65pct": (
            stress_balanced >= .65),
        "panel_volatility_balanced_accuracy_at_least_75pct": (
            statistics.mean(panel_recalls.values()) >= .75),
        "forecast_path_volatility_balanced_accuracy_at_least_80pct": (
            statistics.mean(forecast_volatility_recalls.values()) >= .8),
    }
    property_gates = {
        prop: {
            "claimed_accuracy_at_least_70pct": (
                metrics["claimed_direction_accuracy"] is None
                or metrics["claimed_direction_accuracy"] >= .7),
            "supported_coverage_at_least_70pct": (
                metrics["interval_coverage"] is None
                or metrics["interval_coverage"] >= .7),
            "best_estimate_accuracy_at_least_50pct": (
                metrics["best_estimate_direction_accuracy"] >= .5),
        }
        for prop, metrics in by_property.items()
    }
    gates["every_property_claim_gate"] = all(
        all(values.values()) for values in property_gates.values())
    return {"schema_version": "0.2", "seed": seed, "cases": len(rows),
            "by_property": by_property, "gates": gates,
            "property_gates": property_gates,
            "seasonality_alignment": {
                "cases": len(alignment_rows), "class_recall": alignment_recalls,
                "balanced_accuracy": statistics.mean(alignment_recalls.values()),
                "rows": alignment_rows,
            },
            "seasonality_alignment_stress": {
                "cases": len(alignment_stress_rows),
                "by_family": stress_by_family,
                "balanced_accuracy": stress_balanced,
                "rows": alignment_stress_rows,
            },
            "panel_volatility": {
                "cases": len(panel_rows), "class_recall": panel_recalls,
                "balanced_accuracy": statistics.mean(panel_recalls.values()),
                "rows": panel_rows,
            },
            "forecast_path_volatility": {
                "cases": len(forecast_volatility_rows),
                "class_recall": forecast_volatility_recalls,
                "balanced_accuracy": statistics.mean(
                    forecast_volatility_recalls.values()),
                "rows": forecast_volatility_rows,
            },
            "graduated": all(gates.values()), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = run(seed=args.seed, replicates=args.replicates)
    if args.output_dir:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     indent=2, sort_keys=True))
    return 0 if result["graduated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
