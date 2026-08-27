import pytest

from gnomon.agent_context import (
    build_temporal_decision_reconciliation,
    build_sampled_context_prior_prompt,
    candidate_from_sampled_paths,
    recommended_sample_count,
    sample_path_stability,
    seal_temporal_decision_prior,
    verify_temporal_decision_prior,
)


def test_provider_neutral_prior_prompt_keeps_host_owned_regular_grid_compact():
    history = [f"2026-01-01T0{hour}:00:00+00:00" for hour in range(3)]
    future = [f"2026-01-01T0{hour}:00:00+00:00" for hour in range(3, 6)]

    prompt = build_sampled_context_prior_prompt(
        timestamps=history, values=[1, 2, 3], future_timestamps=future,
        context="A planned event may increase demand.")

    assert prompt.count("step_seconds=3600") == 2
    assert "[1,2,3]" in prompt
    assert '"forecast_path"' in prompt
    assert "Do not echo timestamps" in prompt


def test_provider_neutral_prior_parser_retains_valid_paths_independently():
    future = ["2026-01-02T00:00:00+00:00",
              "2026-01-03T00:00:00+00:00"]
    candidate, diagnostics = candidate_from_sampled_paths([
        'prose {"forecast_path":{"values":[2,4],"rationale":"a"}}',
        '{"forecast_path":{"values":[4,8]}}',
        '{"forecast_path":{"values":[5]}}',
    ], future, history_values=[1, 2, 3])

    assert diagnostics["requested"] == 3
    assert diagnostics["accepted"] == 2
    assert candidate is not None
    assert candidate["_validated_sample_paths"] == [[2.0, 4.0], [4.0, 8.0]]
    assert [row["q50"] for row in candidate["quantiles"]] == [3.0, 6.0]
    assert diagnostics["stability"]["interpretation"] == \
        "stability_not_historical_skill"


def test_provider_neutral_prior_rejects_nonfinite_and_wrong_grid_paths():
    candidate, diagnostics = candidate_from_sampled_paths([
        '{"forecast_path":{"values":[1]}}',
        '{"forecast_path":{"values":[1,"NaN"]}}',
    ], ["2026-01-02T00:00:00+00:00",
        "2026-01-03T00:00:00+00:00"])

    assert candidate is None
    assert diagnostics["accepted"] == 0
    assert diagnostics["rejected"] == 2


def test_sample_count_policy_is_bounded_and_requires_a_distribution():
    assert recommended_sample_count(4) == 5
    assert recommended_sample_count(95) == 5
    assert recommended_sample_count(96) == 4
    with pytest.raises(ValueError, match="positive"):
        recommended_sample_count(0)


def test_stability_rejects_empty_paths():
    with pytest.raises(ValueError, match="non-empty"):
        sample_path_stability([], [1, 2])


def _prior(**updates):
    answer = {
        "breach_expected": True, "breach_probability": .7,
        "first_breach_step": 3, "action": "act",
    }
    answer.update(updates)
    return seal_temporal_decision_prior(
        answer, question_sha256="a" * 64,
        proposer_id="host:test", model="test-model")


def test_decision_prior_is_host_sealed_non_authoritative_and_tamper_evident():
    receipt = _prior()
    assert verify_temporal_decision_prior(receipt)
    assert receipt["support"] == "prior_assisted"
    assert receipt["automation_eligible"] is False
    changed = {**receipt, "action": "monitor"}
    assert not verify_temporal_decision_prior(changed)


def test_reconciliation_surfaces_conflict_without_mutating_primary():
    packet = {
        "support": "best_effort",
        "threshold_analysis": {"horizon_event": {
            "probability_any_breach": .25}},
        "governed_decision": {
            "advisory_action": "monitor",
            "human_action_authority": "advisory",
            "automation_eligible": False,
        },
    }
    before = repr(packet)
    result = build_temporal_decision_reconciliation(
        packet, _prior(), question_sha256="a" * 64)
    assert repr(packet) == before
    assert result["conflict"] == {
        "prediction": True, "action": True, "probability_delta": .45}
    assert result["primary_forecast_unchanged"] is True
    assert result["selection_policy"]["human_may_select"] is True
    assert result["selection_policy"]["automation_eligible"] is False


def test_reconciliation_rejects_cross_question_and_model_claimed_order():
    packet = {"threshold_analysis": {}, "governed_decision": {}}
    with pytest.raises(ValueError, match="another question"):
        build_temporal_decision_reconciliation(
            packet, _prior(), question_sha256="b" * 64)
    with pytest.raises(ValueError, match="breach_expected"):
        seal_temporal_decision_prior(
            {"capture": {"host_attested": True}, "action": "act"},
            question_sha256="a" * 64, proposer_id="host", model="x")
