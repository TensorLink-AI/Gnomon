from __future__ import annotations

import math

from gnomon.temporal_profile import temporal_profile


def _wave(scales: list[float], block: int = 40) -> list[float]:
    pattern = (-1.4, -0.6, 0.2, 1.0, 0.5, -0.3, 0.8, -0.2)
    values = []
    for scale in scales:
        for index in range(block):
            values.append(100 + 0.05 * len(values)
                          + scale * pattern[index % len(pattern)])
    return values


def test_profile_detects_persistent_decreasing_observation_volatility() -> None:
    profile = temporal_profile(_wave([4.0, 2.5, 1.2]), season=1)
    volatility = profile["volatility"]
    assert volatility["observed_direction"] == "decreased"
    assert volatility["predictive_direction"] == "decreased"
    assert volatility["recent_to_reference_ratio"] < 0.8


def test_profile_detects_persistent_increasing_observation_volatility() -> None:
    profile = temporal_profile(_wave([1.0, 2.0, 4.0]), season=1)
    assert profile["volatility"]["observed_direction"] == "increased"
    assert profile["volatility"]["predictive_direction"] == "increased"
    assert profile["regimes"]["strongest_variance_change"] is not None


def test_profile_does_not_call_a_smooth_path_low_observation_volatility() -> None:
    rows = [
        {"point": 10.0, "q10": 0.0 - step, "q90": 20.0 + step}
        for step in range(8)
    ]
    profile = temporal_profile(_wave([3.0, 3.0, 3.0]), forecast_rows=rows)
    behavior = profile["forecast_behavior"]
    assert behavior["expected_path_range"] == 0.0
    assert behavior["median_prediction_interval_width"] > 20
    assert behavior["expected_path_variation_is_observation_volatility"] is False
    assert behavior["prediction_interval_width_is_observation_volatility"] is False
    assert profile["volatility"]["predictive_direction"] == "stable"
    assert profile["marginal_variability"]["predicted_horizon_mad"] is not None
    assert profile["marginal_variability"]["quantity"] == \
        "forecast_horizon_values_vs_raw_history"


def test_profile_is_conservative_on_short_or_single_change_history() -> None:
    short = temporal_profile([float(index % 3) for index in range(12)])
    assert short["volatility"]["observed_direction"] == "uncertain"
    assert short["volatility"]["predictive_direction"] == "uncertain"

    one_change = temporal_profile(_wave([4.0, 4.0, 1.0]))
    assert one_change["volatility"]["predictive_direction"] == "uncertain"


def test_profile_values_are_json_safe_finite_or_none() -> None:
    profile = temporal_profile([5.0] * 40, season=7)
    stack = [profile]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, float):
            assert math.isfinite(value)


def test_profile_covers_trend_seasonality_dependence_and_extremes() -> None:
    trend = [index + 0.1 * ((index % 5) - 2) for index in range(120)]
    assert temporal_profile(trend)["trend"]["persistence"] == "persistent"

    seasonal = [10 + 3 * math.sin(2 * math.pi * index / 12)
                for index in range(120)]
    seasonal_profile = temporal_profile(seasonal, season=12)
    assert seasonal_profile["seasonality"]["stability"] == "stable"
    assert abs(seasonal_profile["trend"]["slope_per_step"]) < 1e-10
    assert seasonal_profile["dependence"]["interpretation"] == "not_estimated"

    alternating = [float((-1) ** index) for index in range(120)]
    assert temporal_profile(alternating)["dependence"]["interpretation"] == \
        "mean_reverting"

    with_extreme = [0.1 * ((index % 5) - 2) for index in range(119)] + [10.0]
    assert temporal_profile(with_extreme)["extremes"][
        "robust_3_5_scale_count"] >= 1
