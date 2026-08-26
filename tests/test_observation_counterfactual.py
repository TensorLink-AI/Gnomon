from gnomon.observation_counterfactual import fit_observation_counterfactual


def _future():
    return ["2026-04-01T00:00:00+00:00",
            "2026-04-02T00:00:00+00:00"]


def test_replay_admits_filter_when_disruptions_poison_raw_last_value():
    history = []
    mask = []
    for index in range(90):
        disrupted = index % 6 in {0, 1, 2}
        mask.append(disrupted)
        history.append(-8.0 if disrupted else 20.0 + (index % 3 - 1))

    candidate, evidence = fit_observation_counterfactual(
        history, mask, _future())

    assert candidate is not None
    assert evidence["status"] == "admitted"
    assert evidence["selection_eligible"] is True
    assert evidence["origins"] >= 12
    assert evidence["candidate_mae"] < evidence["strongest_raw_mae"]
    assert evidence["strongest_raw_comparator"] in {
        "last_value", "window_average", "drift", "linear_trend", "theta", "ets"}
    assert evidence["chronological_block_wins"] >= 2
    assert candidate["conditional_replay"] == evidence


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
