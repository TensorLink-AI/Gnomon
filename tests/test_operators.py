"""Operator substrate: correctness on planted signals, honest abstention on
trap inputs."""

import pytest

from gnomon.operators import (
    anomaly_score,
    changepoint_detection,
    cross_correlation,
    difference,
    evaluate_actions,
    evaluate_threshold_risk,
    event_study,
    regime_detection,
    resample_aggregate,
    seasonality_analysis,
    select_window,
    simulate_scenario,
)


def _ts(n):
    return [f"2026-01-{day:02d}" if day <= 31 else f"2026-02-{day-31:02d}" for day in range(1, n + 1)]


NOISE = [0.5, -0.3, 0.2, -0.4, 0.1, 0.3, -0.2, -0.1, 0.4, -0.5]


def _noisy(base, n, offset=0):
    return [base + NOISE[(i + offset) % len(NOISE)] for i in range(n)]


# -- changepoints / regimes ----------------------------------------------

def test_changepoint_finds_planted_shift():
    values = _noisy(100, 20) + _noisy(115, 20)
    result = changepoint_detection(_ts(40), values)
    assert result["support"]["status"] == "supported"
    assert len(result["changepoints"]) == 1
    change = result["changepoints"][0]
    assert abs(change["index"] - 20) <= 1
    assert change["shift"] == pytest.approx(15, abs=2)


def test_changepoint_no_change_is_supported_conclusion():
    result = changepoint_detection(_ts(30), _noisy(100, 30))
    assert result["changepoints"] == []
    assert result["support"]["status"] == "supported"
    assert result["support"]["reasons"][0]["code"] == "no_change_detected"


def test_changepoint_short_history_is_inconclusive():
    result = changepoint_detection(_ts(8), _noisy(100, 8))
    assert result["support"]["status"] == "inconclusive"


def test_regime_shift_vs_transient():
    sustained = _noisy(100, 20) + _noisy(120, 15)
    result = regime_detection(_ts(35), sustained)
    assert result["classification"] == "regime_shift"

    transient = _noisy(100, 18) + [140, 142, 141, 143] + _noisy(100, 13, offset=3)
    result = regime_detection(_ts(35), transient)
    assert result["classification"] == "transient_anomaly"
    # The return edge is interpreted jointly with the opening edge; treating
    # it alone would falsely label restoration of the baseline as a new regime.
    classifications = [regime["classification"] for regime in result["regimes"]]
    assert "transient_anomaly" in classifications
    assert result["regimes"][-1]["support"]["sensitivity"][
        "paired_reversal"] is True


def test_regime_short_post_window_is_inconclusive():
    values = _noisy(100, 20) + [130.0, 130.5]
    result = regime_detection(_ts(22), values)
    if result["changepoints"]:
        assert result["regimes"][-1]["classification"] == "undetermined"
        assert result["regimes"][-1]["support"]["status"] == "inconclusive"


# -- anomalies / seasonality ---------------------------------------------

def test_anomaly_score_flags_spike():
    values = _noisy(50, 30)
    values[17] = 90.0
    result = anomaly_score(_ts(30), values)
    flagged = [item["timestamp"] for item in result["anomalies"]]
    assert _ts(30)[17] in flagged


def test_anomaly_short_history_abstains():
    result = anomaly_score(_ts(5), [1, 2, 3, 4, 5])
    assert result["support"]["status"] == "inconclusive"


def test_seasonality_detects_weekly_cycle():
    weekly = [10, 14, 13, 9, 4, 1, 3]
    values = [100 + weekly[i % 7] for i in range(42)]
    result = seasonality_analysis(values, "D")
    assert result["period"] == 7
    assert result["support"]["status"] == "supported"


# -- correlation / events -------------------------------------------------

def test_cross_correlation_finds_lead():
    driver = [i % 9 * 2.0 + (i % 4) for i in range(40)]
    follower = [0.0, 0.0, 0.0] + [value * 1.5 for value in driver[:-3]]
    result = cross_correlation(driver, follower)
    assert result["support"]["status"] == "supported"
    assert result["best"]["lag"] == 3
    assert "associational" in result["support"]["assumptions"][0]


def test_cross_correlation_short_series_abstains():
    result = cross_correlation([1, 2, 3], [3, 2, 1])
    assert result["support"]["status"] == "inconclusive"


def test_event_study_measures_contrast():
    values = _noisy(100, 14) + _noisy(112, 14)
    timestamps = _ts(28)
    result = event_study(timestamps, values, [timestamps[14]], window=7)
    study = result["studies"][0]
    assert study["effect"] == pytest.approx(12, abs=1.5)
    assert study["support"]["status"] == "supported"


def test_event_study_out_of_range_abstains():
    timestamps = _ts(10)
    result = event_study(timestamps, _noisy(100, 10), [timestamps[1]], window=7)
    assert result["studies"][0]["support"]["status"] == "inconclusive"


# -- scenarios / threshold / actions -------------------------------------

def test_simulate_scenario_shift_and_scale():
    rows = [{"timestamp": "t1", "point": 100.0, "q10": 90.0, "q50": 100.0, "q90": 110.0}]
    result = simulate_scenario(rows, [{"type": "shift", "value": 10}])
    assert result["rows"][0]["point"] == 110.0
    assert result["rows"][0]["q90"] == 120.0
    result = simulate_scenario(rows, [{"type": "scale", "value": 2.0}])
    assert result["rows"][0]["q10"] == 180.0
    assert result["support"]["status"] == "conditionally_supported"


def test_evaluate_threshold_risk_matches_the_rows_it_reads():
    """The operator previously passed probability-keyed quantiles where
    step-keyed spreads were expected and raised KeyError on every
    invocation. Now the spreads come from the rows themselves, so the
    probabilities must agree with the quantiles printed beside them:
    small above a row's q90, large above its q10."""
    residuals = [-5.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0]
    rows = [
        {"timestamp": f"t{i}", "point": p, "q10": p - 4.0, "q50": p, "q90": p + 4.0}
        for i, p in enumerate([100.0, 101.0], 1)
    ]
    points = [row["point"] for row in rows]

    result = evaluate_threshold_risk(rows, points, residuals, 104.0)
    assert result["support"]["status"] == "supported"
    probabilities = result["risk"]["probability_above"]
    assert len(probabilities) == 2
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert probabilities[0] <= 0.3  # threshold sits at row 1's q90
    assert result["peak_step_probability"] == max(probabilities)

    below = evaluate_threshold_risk(rows, points, residuals, 96.0)
    assert below["risk"]["probability_above"][0] >= 0.7  # at row 1's q10

    empty = evaluate_threshold_risk(rows, points, [], 104.0)
    assert empty["risk"] is None
    assert empty["support"]["status"] == "inconclusive"


def test_evaluate_actions_degraded_without_utilities():
    result = evaluate_actions(
        [{"name": "scale_up"}, {"name": "wait"}],
        {"exceed": 0.4, "no_exceed": 0.6},
    )
    assert result["selected"] is None
    assert result["support"]["status"] == "conditionally_supported"
    assert result["support"]["reasons"][0]["code"] == "missing_utility_inputs"
    # The comparison is still returned — degraded, not failed.
    assert len(result["evaluations"]) == 2


def test_evaluate_actions_chooses_with_utilities():
    result = evaluate_actions(
        [{"name": "scale_up"}, {"name": "wait"}],
        {"exceed": 0.6, "no_exceed": 0.4},
        utilities={
            "scale_up": {"exceed": 100, "no_exceed": -20},
            "wait": {"exceed": -200, "no_exceed": 10},
        },
    )
    assert result["selected"] == "scale_up"
    assert result["support"]["status"] == "supported"
    assert result["decision_rule"] == "maximum expected utility over feasible actions"


def test_evaluate_actions_constraint_infeasibility():
    result = evaluate_actions(
        [{"name": "risky", "residual_risk": 0.9}],
        {"exceed": 0.5, "no_exceed": 0.5},
        max_acceptable_risk=0.2,
    )
    assert result["selected"] is None
    assert result["support"]["status"] == "unsupported"
    assert result["support"]["reasons"][0]["code"] == "no_feasible_action"


def test_evaluate_actions_invalid_probabilities():
    result = evaluate_actions(
        [{"name": "a"}], {"exceed": 0.9, "no_exceed": 0.9},
    )
    assert result["support"]["status"] == "invalid"


# -- window / resample / difference --------------------------------------

def test_window_resample_difference():
    timestamps = _ts(10)
    window = select_window(timestamps, list(range(10)), start=timestamps[2], end=timestamps[5])
    assert window["values"] == [2, 3, 4, 5]
    aggregated = resample_aggregate(timestamps, [1.0] * 10, factor=2, statistic="sum")
    assert aggregated["values"] == [2.0] * 5
    assert difference([1.0, 4.0, 9.0], lag=1) == [3.0, 5.0]
