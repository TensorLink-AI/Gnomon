from __future__ import annotations

import random

import pytest

from gnomon.breach import estimate_horizon_breach


def _rows(horizon: int):
    return [
        {"point": 0.0, "q10": -2.0, "q50": 0.0, "q90": 2.0}
        for _ in range(horizon)
    ]


def _by_lead(paths: list[list[float]]) -> dict[int, list[float]]:
    return {step + 1: [path[step] for path in paths]
            for step in range(len(paths[0]))}


def _ar_paths(seed: int, *, shifted: bool) -> list[list[float]]:
    rng = random.Random(seed)
    paths = []
    for origin in range(20):
        value = 0.0
        path = []
        for _ in range(8):
            value = 0.65 * value + rng.gauss(0.0, 1.0)
            path.append(value + (5.0 if shifted and origin >= 10 else 0.0))
        paths.append(path)
    return paths


def test_regime_gate_accepts_stationary_origins_across_seeds() -> None:
    accepted = 0
    for seed in range(30):
        risk = estimate_horizon_breach(
            _rows(8), 2.5, _by_lead(_ar_paths(seed, shifted=False)),
            measured_interval_coverage=0.8,
            calibration_is_verifiable=True,
        )
        accepted += risk["support"] == "supported"
    # A gross-change guard is allowed to be conservative, but it must not
    # turn ordinary stationary noise into a refusal engine.
    assert accepted >= 27


def test_regime_gate_rejects_large_unseen_level_shifts_across_seeds() -> None:
    rejected = 0
    for seed in range(30):
        risk = estimate_horizon_breach(
            _rows(8), 2.5, _by_lead(_ar_paths(seed, shifted=True)),
            measured_interval_coverage=0.8,
            calibration_is_verifiable=True,
        )
        # The estimate survives at best-effort; what a level shift must
        # cost is governed authority.
        rejected += risk["support"] != "supported"
    assert rejected >= 27


def test_cumulative_distribution_is_positive_affine_equivariant() -> None:
    horizon = 8
    scale, shift = 2.5, 17.0
    paths = _ar_paths(41, shifted=False)
    original = estimate_horizon_breach(
        _rows(horizon), 2.5, _by_lead(paths),
        measured_interval_coverage=0.8,
        calibration_is_verifiable=True,
    )["cumulative_horizon"]
    rows = [{key: scale * value + shift for key, value in row.items()}
            for row in _rows(horizon)]
    transformed = estimate_horizon_breach(
        rows, scale * 2.5 + shift,
        _by_lead([[scale * value for value in path] for path in paths]),
        measured_interval_coverage=0.8,
        calibration_is_verifiable=True,
    )["cumulative_horizon"]
    assert transformed["point_total"] == \
        scale * original["point_total"] + horizon * shift
    assert transformed["median_total"] == pytest.approx(
        scale * original["median_total"] + horizon * shift, abs=2e-6)
    width = (original["total_interval_80"]["upper"]
             - original["total_interval_80"]["lower"])
    transformed_width = (transformed["total_interval_80"]["upper"]
                         - transformed["total_interval_80"]["lower"])
    assert transformed_width == pytest.approx(scale * width, abs=2e-6)
