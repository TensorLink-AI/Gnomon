from __future__ import annotations

import math
import random

from gnomon.volatility import fit_volatility_executable


def _series(seed: int, scales: list[float], block: int = 80) -> list[float]:
    rng = random.Random(seed)
    values: list[float] = []
    level = 100.0
    for scale in scales:
        for _ in range(block):
            level += .03
            values.append(level + rng.gauss(0, scale))
    return values


def test_fitted_executable_is_deterministic_and_self_identifying() -> None:
    values = _series(7, [1, 1.5, 2.3, 3.4])
    first = fit_volatility_executable(values, horizon=8)
    second = fit_volatility_executable(values, horizon=8)
    assert first == second
    published = first.execute()
    assert published["estimate"] == first.scale
    assert published["executable"]["candidate"] == first.candidate
    assert first.diagnostics["folds"] >= 3
    assert set(first.diagnostics["qlike_scores"]) == {
        "constant", "recent", "ewma_slow", "ewma", "ewma_fast",
        "scale_trend", "robust_scale_trend", "regime_mixture",
        "scale_momentum", "scale_momentum_long"}
    assert published["property_distribution"]["quantity"] == \
        "future_to_reference_residual_scale_ratio"
    assert abs(sum(published["direction_probabilities"].values()) - 1) < 1e-9
    assert published["decision"]["automation_eligible"] == \
        published["automation_eligible"]


def test_short_history_publishes_weak_baseline_without_direction_guess() -> None:
    fitted = fit_volatility_executable(_series(3, [1], block=25), horizon=7)
    assert fitted.support == "weak"
    assert fitted.candidate == "constant"
    assert fitted.direction == "uncertain"
    assert fitted.property_distribution.folds < 3
    assert fitted.direction_support == "weak"
    assert fitted.execute()["automation_eligible"] is False


def test_no_residual_history_still_abstains() -> None:
    fitted = fit_volatility_executable([], horizon=7)
    assert fitted.support == "abstained"
    assert fitted.direction_support == "abstained"
    assert fitted.direction == "uncertain"
    assert math.isfinite(fitted.scale)


def test_long_horizon_uses_disclosed_proxy_without_automation() -> None:
    fitted = fit_volatility_executable(_series(9, [1], block=50), horizon=69)
    assert fitted.diagnostics["proxy_horizon_calibration"] is True
    assert fitted.diagnostics["target_span"] < 69
    assert fitted.execute()["automation_eligible"] is False
