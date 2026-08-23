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


def test_short_history_publishes_weak_best_estimate_direction() -> None:
    """A short flat history earns a weak point state, never an abstention.

    Abstention ("uncertain") is reserved for histories where no reference
    ratio is computable at all; a computable estimate is published with weak
    support and explicit automation ineligibility instead.
    """
    fitted = fit_volatility_executable(_series(3, [1], block=25), horizon=7)
    assert fitted.support == "weak"
    assert fitted.candidate == "reference_tail_persistence"
    assert fitted.direction == "stable"
    assert fitted.property_distribution.folds < 3
    assert fitted.direction_support == "weak"
    assert fitted.diagnostics["short_history_point_estimate"] is True
    assert fitted.execute()["automation_eligible"] is False


def test_short_history_onset_transition_directions_are_consistent() -> None:
    """An in-progress scale transition yields the persistence direction.

    The published candidate, scale, ratio and direction must agree: the
    fallback estimator owns all of them, so a warm-up-biased whole-history
    scale cannot contradict the published point state.
    """
    calm_then_wild = _series(1, [1], block=26) + _series(51, [6.0], block=6)
    jump = fit_volatility_executable(calm_then_wild, horizon=7)
    assert jump.direction == "increased"
    assert jump.direction_support == "weak"
    assert jump.ratio is not None and jump.ratio > 1.25
    assert abs(jump.scale / jump.reference_scale - jump.ratio) < 1e-9
    assert jump.property_distribution.estimate == jump.ratio
    wild_then_calm = _series(1, [6.0], block=26) + _series(51, [0.5], block=6)
    calm = fit_volatility_executable(wild_then_calm, horizon=7)
    assert calm.direction == "decreased"
    assert calm.execute()["automation_eligible"] is False


def test_degenerate_history_without_reference_ratio_stays_uncertain() -> None:
    fitted = fit_volatility_executable([5.0] * 4, horizon=7)
    assert fitted.direction == "uncertain"
    assert fitted.ratio is None


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
