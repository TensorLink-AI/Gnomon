from __future__ import annotations

from gnomon.discrimination import discriminate


def _weights(payload: dict) -> dict[str, float]:
    return {row["value"]: row["relative_weight"]
            for row in payload["hypotheses"]}


def test_a_clear_trend_is_discriminated_from_flatness_on_held_out_points() -> None:
    values = [float(2 * step) for step in range(30)]
    payload = discriminate(values, property="trend")
    assert payload["identifiable"] is True
    assert payload["best"] == "upward"
    assert payload["separation"] == "clear"
    weights = _weights(payload)
    assert weights["upward"] > weights["constant"]
    assert weights["downward"] == 0.0
    assert payload["weights_are_fit_evidence_not_probabilities"] is True
    assert payload["provenance"]["uses_future_observations"] is False


def test_a_flat_series_favours_the_simpler_constant_surrogate() -> None:
    values = [5.0 + .1 * (-1) ** step for step in range(30)]
    payload = discriminate(values, property="trend")
    assert payload["best"] == "constant"


def test_a_persistent_level_shift_beats_the_reference_median() -> None:
    values = [10.0] * 20 + [20.0] * 10
    payload = discriminate(values, property="level")
    assert payload["best"] == "higher"
    assert payload["separation"] == "clear"
    assert _weights(payload)["lower"] == 0.0


def test_an_unshifted_level_stays_similar() -> None:
    values = [10.0 + .2 * (-1) ** step for step in range(30)]
    payload = discriminate(values, property="level")
    assert payload["best"] == "similar"


def test_a_volatility_increase_is_measured_not_asserted() -> None:
    values = ([.1 * (-1) ** step for step in range(26)]
              + [2.0 * (-1) ** step for step in range(16)])
    payload = discriminate(values, property="volatility")
    assert payload["best"] == "increased"
    assert payload["scale_ratio"] > 1
    assert _weights(payload)["decreased"] == 0.0


def test_a_spike_is_distinguished_from_a_level_shift_by_what_followed() -> None:
    spike = [10.0] * 24 + [30.0] + [10.0] * 5
    shift = [10.0] * 24 + [20.0] * 6
    spike_payload = discriminate(spike, property="disturbance")
    shift_payload = discriminate(shift, property="disturbance")
    assert spike_payload["best"] == "sudden_spike"
    assert shift_payload["best"] == "level_shift"


def test_a_quiet_series_reports_stable_without_inventing_a_disturbance() -> None:
    values = [10.0 + .1 * (-1) ** step for step in range(30)]
    payload = discriminate(values, property="disturbance")
    assert payload["best"] == "stable"
    assert _weights(payload)["level_shift"] == 0.0


def test_short_histories_abstain_with_a_typed_reason() -> None:
    payload = discriminate([1.0, 2.0, 3.0], property="trend")
    assert payload["identifiable"] is False
    assert payload["reason"] == "insufficient_history_for_held_out_split"


def test_unregistered_properties_return_none() -> None:
    assert discriminate([1.0] * 30, property="seasonality") is None


def test_discrimination_is_deterministic() -> None:
    values = [float(step % 7) + step * .3 for step in range(40)]
    assert discriminate(values, property="trend") == discriminate(
        values, property="trend")


def test_the_packet_carries_the_measured_discrimination() -> None:
    from gnomon.temporal_planner import build_evidence_plan
    from gnomon.temporal_question import TemporalQuestion

    question = TemporalQuestion("q", "predict", "x", "trend", horizon=6)
    result = {
        "best_estimate": {"value": "constant", "support": "weak"},
        "answer": {"direction": "constant", "support": "weak",
                   "estimate": .0, "executable": {
                       "kind": "published_forecast_projection"}},
    }
    discrimination = discriminate(
        [float(2 * step) for step in range(30)], property="trend")
    plan = build_evidence_plan(question, result,
                               discrimination=discrimination)
    assert plan["discrimination"]["best"] == "upward"
    packet = plan["packet"]
    assert packet["discriminating_evidence"]["separation"] == "clear"
    assert packet["evidence_sufficiency"]["separation"] == "clear"
    values = {row["value"]: row for row in packet["interpretations"]}
    # The surrogate-only interpretation entered the packet as selectable,
    # citing the measured fit; the canonical answer is untouched.
    assert values["upward"]["compatible"] is True
    assert "held_out_hypothesis_fit" in values["upward"]["supporting"]
    assert values["upward"]["held_out_fit"] > .8
    assert plan["packet"]["selection_contract"]["canonical"]["value"] == \
        "constant"
