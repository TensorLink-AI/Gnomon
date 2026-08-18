"""Canonical, deterministic temporal properties for agents and artifacts.

The profile deliberately separates movement of the expected path, dispersion
of observations around that path, and uncertainty in the forecast.  Those
quantities are often conflated by language models but answer different
operational questions.
"""

from __future__ import annotations

import math
import statistics
from typing import Any


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _robust_scale(values: list[float]) -> float:
    if not values:
        return 0.0
    centre = _median(values)
    mad = _median([abs(value - centre) for value in values])
    if mad > 1e-12:
        return 1.4826 * mad
    if len(values) > 1:
        return float(statistics.stdev(values))
    return 0.0


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    centre = (len(values) - 1) / 2
    denominator = sum((index - centre) ** 2 for index in range(len(values)))
    return (sum((index - centre) * value
                for index, value in enumerate(values)) / denominator
            if denominator else 0.0)


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    lm, rm = statistics.mean(left), statistics.mean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - lm) ** 2 for a in left)
                            * sum((b - rm) ** 2 for b in right))
    return numerator / denominator if denominator > 1e-12 else None


def _residuals(values: list[float], season: int) -> list[float]:
    """Remove a line and, when supported, phase means without leakage."""
    if season > 1 and len(values) >= 2 * season:
        cycle_means = [statistics.mean(values[index:index + season])
                       for index in range(0, len(values) - season + 1, season)]
        slope = _slope(cycle_means) / season
    else:
        slope = _slope(values)
    centre = (len(values) - 1) / 2
    detrended = [value - slope * (index - centre)
                 for index, value in enumerate(values)]
    if season <= 1 or len(values) < 2 * season:
        baseline = _median(detrended)
        return [value - baseline for value in detrended]
    phase = {
        offset: statistics.mean(detrended[offset::season])
        for offset in range(season)
    }
    return [value - phase[index % season]
            for index, value in enumerate(detrended)]


def _direction(ratio: float | None) -> str:
    if ratio is None:
        return "uncertain"
    if ratio < 0.8:
        return "decreased"
    if ratio > 1.25:
        return "increased"
    return "stable"


def temporal_profile(
    values: list[float], *, season: int = 1,
    forecast_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a compact profile from observations and optional forecast rows.

    Predictive volatility is a conservative extrapolation of repeatable
    residual-scale movement, not a reading of point-path smoothness or interval
    width.  It is marked uncertain when the history cannot support that claim.
    """
    if not values:
        raise ValueError("temporal_profile requires observations")
    numeric = [float(value) for value in values if math.isfinite(float(value))]
    if not numeric:
        raise ValueError("temporal_profile requires finite observations")
    residuals = _residuals(numeric, max(1, int(season)))
    n = len(numeric)
    trend_values = numeric
    trend_step = 1
    if season > 1 and n >= 2 * season:
        trend_values = [statistics.mean(numeric[index:index + season])
                        for index in range(0, n - season + 1, season)]
        trend_step = season
    window = min(n // 2, max(8, min(max(2 * season, n // 4), 48)))
    enough = n >= max(24, 2 * window)
    recent = residuals[-window:] if window else residuals
    reference = residuals[-2 * window:-window] if enough else []
    recent_scale = _robust_scale(recent)
    reference_scale = _robust_scale(reference)
    ratio = (recent_scale / reference_scale
             if reference_scale > 1e-12 else None)
    observed = _direction(ratio) if enough else "uncertain"

    # Require the direction to appear across consecutive blocks before
    # projecting it. This avoids turning one outlier or regime boundary into a
    # volatility forecast.
    # Equal thirds make "persistent" literal. A capped trailing window can
    # straddle one abrupt change and manufacture two successively smaller
    # mixed blocks from a single regime transition.
    block = n // 3
    scales = ([_robust_scale(residuals[-3 * block:-2 * block]),
               _robust_scale(residuals[-2 * block:-block]),
               _robust_scale(residuals[-block:])]
              if block >= 8 and n >= 3 * block else [])
    predictive = "uncertain"
    if scales and min(scales) > 1e-12:
        first_ratio = scales[1] / scales[0]
        second_ratio = scales[2] / scales[1]
        if first_ratio < 0.9 and second_ratio < 0.9:
            predictive = "decreased"
        elif first_ratio > 1.1 and second_ratio > 1.1:
            predictive = "increased"
        elif all(0.8 <= item <= 1.25
                 for item in (first_ratio, second_ratio)):
            predictive = "stable"

    split = None
    if n >= 24:
        candidates = []
        for index in range(8, n - 7):
            left = _robust_scale(residuals[:index])
            right = _robust_scale(residuals[index:])
            if min(left, right) > 1e-12:
                candidates.append((max(left / right, right / left), index,
                                   right / left))
        if candidates:
            strength, index, change_ratio = max(candidates)
            if strength >= 1.5:
                split = {"type": "variance_change", "index": index,
                         "scale_ratio_after_before": change_ratio,
                         "direction": _direction(change_ratio)}

    trend_half = len(trend_values) // 2
    early_slope = (_slope(trend_values[:trend_half]) / trend_step
                   if trend_half >= 2 else 0.0)
    recent_slope = (_slope(trend_values[trend_half:]) / trend_step
                    if len(trend_values) - trend_half >= 2 else 0.0)
    overall_slope = _slope(trend_values) / trend_step
    scale = _robust_scale(residuals)
    material = max(scale / max(n, 1), 1e-12)
    if abs(recent_slope) <= material:
        persistence = "flat_or_decayed"
    elif early_slope * recent_slope < 0:
        persistence = "reversed"
    elif abs(early_slope) > material and abs(recent_slope) < 0.5 * abs(early_slope):
        persistence = "decaying"
    elif early_slope * recent_slope > 0:
        persistence = "persistent"
    else:
        persistence = "emerging"

    seasonal_stability = "not_estimated"
    half = n // 2
    if season > 1 and half >= 2 * season and n - half >= 2 * season:
        early_acf = _correlation(numeric[:half - season], numeric[season:half])
        late = numeric[half:]
        late_acf = _correlation(late[:-season], late[season:])
        if early_acf is not None and late_acf is not None:
            seasonal_stability = (
                "stable" if abs(late_acf - early_acf) <= 0.2
                else "strengthening" if late_acf > early_acf
                else "weakening")

    meaningful_scale = scale > max(1e-12, 1e-9 * max(abs(value) for value in numeric))
    lag1 = (_correlation(residuals[:-1], residuals[1:])
            if meaningful_scale else None)
    extreme_count = (sum(abs(value - _median(residuals)) >= 3.5 * scale
                         for value in residuals)
                     if scale > 1e-12 else 0)
    points = [float(row["point"]) for row in (forecast_rows or [])
              if row.get("point") is not None]
    widths = [float(row["q90"]) - float(row["q10"])
              for row in (forecast_rows or [])
              if row.get("q90") is not None and row.get("q10") is not None]
    point_range = max(points) - min(points) if points else None
    historical_marginal_mad = _median(
        [abs(value - _median(numeric)) for value in numeric])
    point_mad = (_median([abs(value - _median(points)) for value in points])
                 if points else None)
    # For an 80% normal interval q90-q10 = 2*1.28155*sigma and
    # MAD = .67449*sigma. This is explicitly an approximation; the published
    # quantiles remain the authoritative uncertainty contract.
    implied_residual_mad = (
        _median(widths) * 0.6744897501960817 / (2 * 1.2815515655446004)
        if widths else None)
    predicted_marginal_mad = (
        math.sqrt(point_mad ** 2 + implied_residual_mad ** 2)
        if point_mad is not None and implied_residual_mad is not None else None)
    marginal_ratio = (
        predicted_marginal_mad / historical_marginal_mad
        if predicted_marginal_mad is not None
        and historical_marginal_mad > 1e-12 else None)
    return {
        "schema_version": "0.1",
        "level": {"latest": numeric[-1], "historical_median": _median(numeric)},
        "trend": {"slope_per_step": overall_slope,
                  "recent_slope_per_step": recent_slope,
                  "persistence": persistence},
        "seasonality": {"period_steps": int(season),
                        "stability": seasonal_stability},
        "volatility": {
            "quantity": "observation_residual_scale",
            "reference_scale": reference_scale if enough else None,
            "recent_scale": recent_scale,
            "recent_to_reference_ratio": ratio,
            "observed_direction": observed,
            "predictive_direction": predictive,
            "support": "supported" if predictive != "uncertain" else "inconclusive",
            "window_steps": window,
            "method": "detrended_seasonally_adjusted_robust_scale",
        },
        "marginal_variability": {
            "quantity": "forecast_horizon_values_vs_raw_history",
            "historical_mad": historical_marginal_mad,
            "predicted_horizon_mad": predicted_marginal_mad,
            "predicted_to_historical_ratio": marginal_ratio,
            "predictive_direction": _direction(marginal_ratio),
            "support": ("conditionally_supported"
                        if marginal_ratio is not None else "inconclusive"),
            "method": "forecast_point_mad_plus_interval_implied_residual_mad",
            "assumptions": [
                "q10/q90 width approximated as a symmetric 80% distribution",
                "this is marginal horizon variability, not residual volatility",
            ] if marginal_ratio is not None else [],
        },
        "regimes": {"strongest_variance_change": split},
        "extremes": {"robust_3_5_scale_count": extreme_count},
        "dependence": {"residual_lag1_correlation": lag1,
                       "interpretation": (
                           "persistent" if lag1 is not None and lag1 >= 0.3
                           else "mean_reverting" if lag1 is not None and lag1 <= -0.3
                           else "weak" if lag1 is not None else "not_estimated")},
        "forecast_behavior": {
            "expected_path_range": point_range,
            "median_prediction_interval_width": _median(widths) if widths else None,
            "expected_path_variation_is_observation_volatility": False,
            "prediction_interval_width_is_observation_volatility": False,
            "volatility_terms_are_not_interchangeable": True,
        },
        "source": "computed_from_observations",
    }


def compact_temporal_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """High-traffic projection that cannot crowd forecast rows off MCP.

    The complete profile is available to typed temporal executables without
    changing the frozen forecast artifact. The response keeps
    the distinctions an agent needs for routing and claims; methods, windows,
    scales, and assumptions remain available from the artifact.
    """
    volatility = profile["volatility"]
    marginal = profile["marginal_variability"]
    return {
        "volatility": {
            "recent_to_reference_ratio": volatility["recent_to_reference_ratio"],
            "observed_direction": volatility["observed_direction"],
            "predictive_direction": volatility["predictive_direction"],
            "support": volatility["support"],
        },
        "marginal_variability": {
            "predicted_to_historical_ratio": marginal["predicted_to_historical_ratio"],
            "predictive_direction": marginal["predictive_direction"],
            "support": marginal["support"],
        },
    }
