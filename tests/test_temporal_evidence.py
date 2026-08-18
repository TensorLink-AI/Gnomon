from __future__ import annotations

import random

from gnomon.temporal_evidence import (
    aggregate_evidence, compare_windows, window_evidence,
)


def _changed(seed: int, ratio: float) -> list[float]:
    rng = random.Random(seed)
    first = [rng.gauss(0, 1) for _ in range(48)]
    second = [rng.gauss(0, ratio) for _ in range(48)]
    return first + second


def test_window_comparison_separates_level_trend_and_residual_scale() -> None:
    evidence = compare_windows(_changed(7, 3.0), window=48)
    assert evidence["identifiable"] is True
    assert evidence["properties"]["volatility"]["direction"] == "increased"
    assert evidence["provenance"]["uses_future_observations"] is False


def test_panel_aggregation_uses_breadth_without_calling_series_folds() -> None:
    rows = {f"series-{seed}": compare_windows(_changed(seed, 3.0), window=48)
            for seed in range(8)}
    result = aggregate_evidence(rows, property="volatility")
    assert result["direction"] == "increased"
    assert result["effective_series"] >= 5
    assert result["support"] == "supported"
    assert result["provenance"]["independence_claimed"] is False


def test_panel_disagreement_is_uncertain_not_a_forced_majority() -> None:
    rows = {
        "up-a": compare_windows(_changed(1, 3.0), window=48),
        "up-b": compare_windows(_changed(2, 3.0), window=48),
        "down-a": compare_windows(_changed(3, .2), window=48),
        "down-b": compare_windows(_changed(4, .2), window=48),
    }
    result = aggregate_evidence(rows, property="volatility")
    assert result["direction"] == "uncertain"
    assert result["support"] == "abstained"
    assert result["reason"] == "no_cross_series_consensus"


def test_typed_evidence_preserves_mode_source_and_provenance() -> None:
    evidence = window_evidence(
        _changed(17, 3.0), property="volatility", window=48)
    payload = evidence.to_dict()
    assert payload["mode"] == "observed"
    assert payload["source"] == "adjacent_observed_windows"
    assert payload["direction"] == "increased"
    assert payload["provenance"]["uses_future_observations"] is False


def test_observed_seasonal_phase_transition_is_measured() -> None:
    import math
    period = 24
    first = [10 + 4 * math.sin(2 * math.pi * index / period)
             for index in range(4 * period)]
    second = [10 + 4 * math.sin(2 * math.pi * (index - 7) / period)
              for index in range(4 * period)]
    evidence = window_evidence(
        first + second, property="seasonality", season=period,
        window=4 * period)
    assert evidence.direction == "phase_shifted"
    assert evidence.identifiable is True


def test_short_history_marks_seasonal_transition_unidentifiable() -> None:
    evidence = window_evidence(
        [1, 2, 1, 2, 1, 2], property="seasonality", season=4)
    assert evidence.identifiable is False
    assert evidence.support == "abstained"


def test_marginal_volatility_is_identifiable_but_weak() -> None:
    reference = [(-1.0 if index % 2 else 1.0) for index in range(96)]
    recent = [1.28 * value for value in reference]
    evidence = window_evidence(
        reference + recent, property="volatility", window=96)
    assert evidence.identifiable is True
    assert evidence.direction == "increased"
    assert evidence.support == "weak"


def test_stable_volatility_needs_and_carries_equivalence_interval() -> None:
    rng = random.Random(91)
    reference = [rng.gauss(0, 1) for _ in range(192)]
    recent = list(reference)
    evidence = window_evidence(
        reference + recent, property="volatility", window=192)
    comparison = compare_windows(reference + recent, window=192)
    item = comparison["properties"]["volatility"]
    assert item["interval_level"] == .90
    assert len(item["interval"]) == 2
    assert evidence.direction == "stable"
    assert evidence.support == "supported"
    assert item["interval"][0] >= .8
    assert item["interval"][1] <= 1.25
