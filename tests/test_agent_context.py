import pytest

from gnomon.agent_context import (
    build_temporal_decision_reconciliation,
    build_sampled_context_prior_prompt,
    candidate_from_sampled_paths,
    decision_selection_synthesis_payload,
    recommended_initial_sample_count,
    recommended_sample_count,
    sampled_prior_sufficiency,
    sample_path_stability,
    seal_temporal_decision_selection,
    seal_temporal_decision_prior,
    verify_temporal_decision_reconciliation,
    verify_temporal_decision_selection,
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
    assert recommended_initial_sample_count(4) == 3
    assert recommended_initial_sample_count(96) == 3
    assert recommended_sample_count(4) == 5
    assert recommended_sample_count(95) == 5
    assert recommended_sample_count(96) == 4
    with pytest.raises(ValueError, match="positive"):
        recommended_sample_count(0)


def test_stability_rejects_empty_paths():
    with pytest.raises(ValueError, match="non-empty"):
        sample_path_stability([], [1, 2])


def test_sampled_prior_sufficiency_accepts_valid_coherent_elicitation():
    stability = sample_path_stability(
        [[10, 11, 12], [10.2, 11.1, 12.1], [9.8, 10.9, 11.9]],
        [7, 8, 9, 10],
    )

    result = sampled_prior_sufficiency({
        "requested": 3, "accepted": 3, "stability": stability})

    assert result["eligible_for_human_recommendation"] is True
    assert result["reason_codes"] == []
    assert result["historical_skill_evidence"] is False
    assert result["automation_eligible"] is False


def test_sampled_prior_sufficiency_demotes_malformed_dispersed_paths():
    stability = {
        "version": "0.1",
        "interpretation": "stability_not_historical_skill",
        "scale_basis": "median_nonzero_history_increment",
        "path_count": 3, "horizon": 95,
        "median_pointwise_q80_width_scaled": 1.17,
        "p90_pointwise_q80_width_scaled": 13.38,
        "median_pairwise_mae_scaled": 3.92,
        "max_pairwise_mae_scaled": 4.53,
        "mean_direction_agreement": .82,
        "unanimous_direction_fraction": .47,
    }

    result = sampled_prior_sufficiency({
        "requested": 5, "accepted": 3, "stability": stability})

    assert result["eligible_for_human_recommendation"] is False
    assert "low_valid_path_fraction" in result["reason_codes"]
    assert set(result["reason_codes"]) & {
        "directionally_unstable_paths", "dispersed_sampled_paths"}


def test_sampled_prior_sufficiency_is_invariant_to_units_and_level():
    paths = [[10, 11, 13], [10.1, 11.2, 13.1], [9.9, 10.8, 12.9]]
    transformed = [[1000 + 25 * value for value in path] for path in paths]
    base = sampled_prior_sufficiency({
        "requested": 3, "accepted": 3,
        "stability": sample_path_stability(paths, [7, 8, 9, 10]),
    })
    rescaled = sampled_prior_sufficiency({
        "requested": 3, "accepted": 3,
        "stability": sample_path_stability(
            transformed, [1175, 1200, 1225, 1250]),
    })

    assert rescaled["eligible_for_human_recommendation"] == base[
        "eligible_for_human_recommendation"]
    assert rescaled["reason_codes"] == base["reason_codes"]
    assert rescaled["observed_direction_agreement"] == pytest.approx(
        base["observed_direction_agreement"])
    assert rescaled["observed_pairwise_to_pointwise_ratio"] == pytest.approx(
        base["observed_pairwise_to_pointwise_ratio"])


def test_sampled_prior_sufficiency_fails_closed_on_invalid_diagnostics():
    result = sampled_prior_sufficiency({
        "requested": 3, "accepted": 3,
        "stability": {
            "interpretation": "stability_not_historical_skill",
            "path_count": 2,
            "mean_direction_agreement": 1,
            "median_pairwise_mae_scaled": -1,
            "median_pointwise_q80_width_scaled": 0,
        },
    })

    assert result["eligible_for_human_recommendation"] is False
    assert "invalid_stability_diagnostics" in result["reason_codes"]


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
    assert verify_temporal_decision_reconciliation(result)
    selection = seal_temporal_decision_selection(result, {
        "selected_source": "synthesis",
        "counterevidence_source": "immutable_primary",
        "action": "act", "confidence": "low",
        "what_would_change": "Two additional non-breach origins.",
        "automation_action": "withhold",
    })
    assert selection["support"] == "prior_assisted"
    assert selection["primary_forecast_unchanged"] is True
    assert selection["automation_eligible"] is False
    assert verify_temporal_decision_selection(selection)
    synthesis = decision_selection_synthesis_payload(result, selection)
    assert synthesis["label"] == "hypothesis_ranking"
    assert synthesis["proposer_id"] == "host:test"
    assert synthesis["automation_eligible"] is False


def test_reconciliation_rejects_cross_question_and_model_claimed_order():
    packet = {"threshold_analysis": {}, "governed_decision": {}}
    with pytest.raises(ValueError, match="another question"):
        build_temporal_decision_reconciliation(
            packet, _prior(), question_sha256="b" * 64)
    with pytest.raises(ValueError, match="breach_expected"):
        seal_temporal_decision_prior(
            {"capture": {"host_attested": True}, "action": "act"},
            question_sha256="a" * 64, proposer_id="host", model="x")


def test_selection_requires_counterevidence_and_withholds_automation():
    packet = {
        "threshold_analysis": {"horizon_event": {
            "probability_any_breach": .25}},
        "governed_decision": {"advisory_action": "monitor",
                              "automation_eligible": False},
    }
    reconciliation = build_temporal_decision_reconciliation(
        packet, _prior(), question_sha256="a" * 64)
    base = {
        "selected_source": "independent_prior", "action": "act",
        "confidence": "medium", "what_would_change": "More replay origins.",
        "automation_action": "withhold",
    }
    with pytest.raises(ValueError, match="counterevidence"):
        seal_temporal_decision_selection(reconciliation, base)
    with pytest.raises(ValueError, match="authorize automation"):
        seal_temporal_decision_selection(reconciliation, {
            **base, "counterevidence_source": "immutable_primary",
            "automation_action": "act"})


def test_reconciliation_accepts_only_cutoff_safe_matching_skill():
    packet = {
        "threshold_analysis": {"horizon_event": {
            "probability_any_breach": .25}},
        "governed_decision": {"advisory_action": "monitor",
                              "automation_eligible": False},
    }
    skill = {
        "proposer_id": "host:test", "resolved": 30,
        "graduated_for_human_prior": True,
        "support_upgrade_allowed": False,
        "automation_upgrade_allowed": False,
        "rule": "paired_categorical_sign_test_and_shrunk_net_v1",
        "known_at": "2026-01-02T00:00:00+00:00",
    }
    result = build_temporal_decision_reconciliation(
        packet, _prior(), question_sha256="a" * 64,
        proposer_skill=skill,
        decision_cutoff="2026-01-03T00:00:00+00:00")
    assert result["selection_policy"]["prior_has_outcome_skill"] is True
    assert result["proposer_skill"]["resolved"] == 30
    with pytest.raises(ValueError, match="not known"):
        build_temporal_decision_reconciliation(
            packet, _prior(), question_sha256="a" * 64,
            proposer_skill=skill,
            decision_cutoff="2026-01-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="another proposer"):
        build_temporal_decision_reconciliation(
            packet, _prior(), question_sha256="a" * 64,
            proposer_skill={**skill, "proposer_id": "other"},
            decision_cutoff="2026-01-03T00:00:00+00:00")
