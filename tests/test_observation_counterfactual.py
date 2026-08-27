from gnomon.observation_counterfactual import fit_observation_counterfactual
from datetime import datetime, timedelta, timezone
import math
import random


def _future():
    return ["2026-04-01T00:00:00+00:00",
            "2026-04-02T00:00:00+00:00"]


def test_replay_admits_filter_when_disruptions_poison_raw_last_value():
    rng = random.Random(0)
    history = []
    mask = []
    for index in range(120):
        disrupted = index % 5 in {0, 1}
        mask.append(disrupted)
        history.append((-100.0 + rng.gauss(0, 3)) if disrupted
                       else (20.0 + rng.gauss(0, 1)))

    candidate, evidence = fit_observation_counterfactual(
        history, mask, _future())

    assert candidate is not None
    assert evidence["status"] == "admitted"
    assert evidence["selection_eligible"] is True
    assert evidence["origins"] >= 12
    assert evidence["candidate_mae"] < evidence["strongest_raw_mae"]
    assert (evidence["candidate_probabilistic_loss"]
            < evidence["strongest_probabilistic_loss"])
    assert evidence["strongest_raw_comparator"] in {
        "last_value", "window_average", "drift", "linear_trend", "theta", "ets"}
    assert evidence["chronological_block_wins"] >= 2
    assert candidate["conditional_replay"] == evidence


def test_point_uplift_cannot_admit_a_worse_predictive_distribution():
    history = []
    mask = []
    for index in range(90):
        disrupted = index % 6 in {0, 1, 2}
        mask.append(disrupted)
        history.append(-8.0 if disrupted else 20.0 + (index % 3 - 1))

    _, evidence = fit_observation_counterfactual(history, mask, _future())

    assert evidence["candidate_mae"] < evidence["strongest_raw_mae"]
    assert (evidence["candidate_probabilistic_loss"]
            > evidence["strongest_probabilistic_loss"])
    assert evidence["status"] == "not_admitted"


def test_replay_refuses_filter_that_does_not_improve_conditional_targets():
    history = [float(index) for index in range(1, 91)]
    mask = [index % 6 in {0, 1, 2} for index in range(90)]

    candidate, evidence = fit_observation_counterfactual(
        history, mask, _future())

    assert candidate is not None
    assert evidence["status"] == "not_admitted"
    assert evidence["selection_eligible"] is False


def test_short_replay_remains_visible_but_cannot_lead():
    history = [10.0, -1.0, 11.0, -1.0, 12.0, 13.0]
    mask = [False, True, False, True, False, False]

    candidate, evidence = fit_observation_counterfactual(
        history, mask, _future())

    assert candidate is not None
    assert evidence["status"] == "insufficient_replay"
    assert evidence["selection_eligible"] is False
    assert len(candidate["quantiles"]) == 2


def test_input_is_not_mutated():
    history = [20.0 if index % 4 else -5.0 for index in range(60)]
    mask = [index % 4 == 0 for index in range(60)]
    before_history, before_mask = list(history), list(mask)

    fit_observation_counterfactual(history, mask, _future())

    assert history == before_history
    assert mask == before_mask


def test_daily_phase_family_is_fold_safe_and_beats_raw_comparators():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timestamps = []
    history = []
    mask = []
    for index in range(24 * 21):
        observed = start + timedelta(hours=index)
        phase = index % 24
        disrupted = phase in {20, 21, 22, 23}
        timestamps.append(observed.isoformat())
        mask.append(disrupted)
        history.append(0.0 if disrupted else (
            30.0 + 8.0 * math.sin(2.0 * math.pi * phase / 24.0)))
    future = [(start + timedelta(hours=len(history) + step)).isoformat()
              for step in range(24)]

    candidate, evidence = fit_observation_counterfactual(
        history, mask, future, history_timestamps=timestamps)

    assert candidate is not None
    assert evidence["family"] == "seasonal_phase_median"
    assert evidence["daily_period_steps"] == 24
    assert evidence["scheme"] == \
        "expanding_origin_phase_holdout_reconstruction"
    assert evidence["targeted_reconstruction_replay"] is True
    assert evidence["simulated_contamination_phase_count"] == 4
    assert evidence["status"] == "admitted"
    assert evidence["candidate_mae"] < evidence["strongest_raw_mae"]
    assert (evidence["candidate_probabilistic_loss"]
            < evidence["strongest_probabilistic_loss"])
    assert evidence["chronological_block_wins"] >= 2
    medians = [row["q50"] for row in candidate["quantiles"]]
    assert max(medians) - min(medians) > 10.0


def test_daily_phase_family_requires_regular_timestamp_evidence():
    history = [20.0 + float(index % 24) for index in range(120)]
    mask = [index % 24 in {20, 21, 22, 23} for index in range(120)]

    _, evidence = fit_observation_counterfactual(history, mask, _future())

    assert "seasonal_phase_median" not in evidence.get("families_compared", [])
