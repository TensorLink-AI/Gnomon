from datetime import datetime, timedelta, timezone
import json

import pytest

from gnomon.llm_dossier import validate_temporal_dossier
from gnomon.publication import publish_result, verify_publication

from gnomon.agent_context import (
    build_relationship_prior_prompt,
    build_temporal_decision_reconciliation,
    build_sampled_context_prior_prompt,
    candidate_from_sampled_paths,
    candidate_from_relationship_prior_specs,
    decision_selection_synthesis_payload,
    recommended_initial_sample_count,
    recommended_sample_count,
    sampled_path_anchor_indices,
    sampled_prior_sufficiency,
    sample_path_stability,
    seal_temporal_decision_selection,
    seal_temporal_decision_prior,
    verify_temporal_decision_reconciliation,
    verify_temporal_decision_selection,
    verify_temporal_decision_prior,
)


def test_relationship_prior_prompt_requests_only_safe_declarative_form():
    prompt = build_relationship_prior_prompt(
        context="Pressure follows the affinity laws.",
        target_name="pressure", driver_name="speed")
    assert '"family":"linear|power"' in prompt
    assert "Do not output coefficients" in prompt
    assert "code" in prompt


def test_relationship_prior_executes_stable_power_specs_host_side():
    outputs = [
        '{"relationship_prior":{"family":"power","exponent":2,'
        '"rationale":"affinity law"}}' for _ in range(5)]
    driver = [10.0 + index for index in range(12)]
    target = [3.0 * value ** 2 for value in driver]
    candidate, diagnostics = candidate_from_relationship_prior_specs(
        outputs, target_history=target, driver_history=driver,
        future_driver=[22.0, 23.0], future_timestamps=["a", "b"],
        claim_ids=["law", "transition"])

    assert candidate is not None
    assert candidate["forecast"][0]["q50"] == pytest.approx(3 * 22 ** 2)
    assert candidate["provenance_class"] == "model_authored_relationship_prior"
    assert candidate["automation_eligible"] is False
    assert diagnostics["historical_skill_evidence"] is False


def test_relationship_prior_allows_signed_target_scale():
    outputs = [
        '{"relationship_prior":{"family":"power","exponent":2}}'
        for _ in range(4)]
    candidate, diagnostics = candidate_from_relationship_prior_specs(
        outputs, target_history=[-2, -8, -18, -32],
        driver_history=[1, 2, 3, 4], future_driver=[5],
        future_timestamps=["a"], claim_ids=["law"])
    assert candidate is not None
    assert candidate["forecast"][0]["q50"] == pytest.approx(-50)
    assert diagnostics["eligible_for_human_recommendation"] is True


def test_sealed_relationship_prior_survives_large_contextual_jump_as_scenario():
    outputs = [
        '{"relationship_prior":{"family":"power","exponent":2}}'
        for _ in range(4)]
    candidate, _ = candidate_from_relationship_prior_specs(
        outputs, target_history=[.01, .04, .09, .16],
        driver_history=[1, 2, 3, 4], future_driver=[100],
        future_timestamps=["2026-01-02T00:00:00+00:00"],
        claim_ids=["claim-1", "claim-2"])
    context = (
        "Output is estimated from speed using a named power law. "
        "At 00:00 speed changes to 100.")
    dossier, reasons = validate_temporal_dossier({"claims": [{
        "source_span": "Output is estimated from speed using a named power law.",
        "relation": "unknown", "confidence": 1.0,
        "effective_start": None, "effective_end": None,
        "timing_status": "atemporal_context",
    }, {
        "source_span": "At 00:00 speed changes to 100.",
        "relation": "unknown", "confidence": 1.0,
        "effective_start": "2026-01-02T00:00:00+00:00",
        "effective_end": "2026-01-02T00:00:00+00:00",
        "timing_status": "resolved",
    }]}, context_text=context, cutoff="2026-01-01T00:00:00+00:00",
        future_timestamps=["2026-01-02T00:00:00+00:00"],
        history=[.01, .04, .09, .16], compiler_model="host-model",
        governed_candidate=candidate)
    assert not reasons
    assert dossier["forecast_candidate"] is not None
    assert dossier["candidate_support"] == "prior_assisted"
    assert dossier["automation_eligible"] is False
    payload = publish_result({
        "support": "best_effort",
        "forecast": [{"timestamp": "2026-01-02T00:00:00+00:00",
                      "point": .16, "q10": .1, "q50": .16, "q90": .2}],
    }, mode="scenario", dossiers=[dossier])
    assert len(payload["candidate_portfolio"]) == 2
    assert payload["primary_forecast_unchanged"] is True
    assert payload["automation"]["eligible"] is False
    assert verify_publication(payload)


def test_relationship_prior_rejects_disagreement_and_unsafe_specs():
    outputs = [
        '{"relationship_prior":{"family":"power","exponent":2}}',
        '{"relationship_prior":{"family":"linear","exponent":1}}',
        '{"relationship_prior":{"family":"power","exponent":9}}',
        '{"relationship_prior":null}',
    ]
    candidate, diagnostics = candidate_from_relationship_prior_specs(
        outputs, target_history=[2, 3, 4, 5],
        driver_history=[1, 2, 3, 4], future_driver=[5],
        future_timestamps=["a"], claim_ids=["law"])
    assert candidate is None
    assert diagnostics["eligible_for_human_recommendation"] is False


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


def test_long_prior_prompt_uses_host_owned_sparse_anchor_contract():
    history = [f"2026-01-01T{hour:02d}:00:00+00:00" for hour in range(4)]
    future = [f"2026-01-{2 + index // 24:02d}T{index % 24:02d}:00:00+00:00"
              for index in range(40)]

    prompt = build_sampled_context_prior_prompt(
        timestamps=history, values=[1, 2, 3, 4], future_timestamps=future,
        context="A comparable site supplies one reference point.")

    anchors = sampled_path_anchor_indices(40)
    assert len(anchors) == 32
    assert anchors[0] == 0 and anchors[-1] == 39
    assert "Return exactly 32 finite values" in prompt
    assert str(anchors) in prompt
    assert "linearly interpolate" in prompt


def test_sparse_anchor_paths_expand_deterministically_on_host_time_grid():
    future = [f"2026-01-{2 + index // 24:02d}T{index % 24:02d}:00:00+00:00"
              for index in range(40)]
    anchors = sampled_path_anchor_indices(len(future))
    values = [2.0 * index for index in anchors]

    candidate, diagnostics = candidate_from_sampled_paths([
        '{"forecast_path":{"values":' + str(values) + '}}',
    ], future, history_values=[-2, -1, 0])

    assert candidate is not None
    assert candidate["_validated_sample_paths"][0] == [
        2.0 * index for index in range(40)]
    shape = diagnostics["response_shapes"][0]
    assert shape["path_representation"] == (
        "host_anchor_linear_interpolation")
    assert shape["anchor_count"] == 32


def test_sparse_anchor_interpolation_uses_time_not_row_distance():
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    elapsed_hours = [0]
    for index in range(1, 40):
        elapsed_hours.append(elapsed_hours[-1] + (3 if index % 7 == 0 else 1))
    future = [(epoch + timedelta(hours=value)).isoformat()
              for value in elapsed_hours]
    anchors = sampled_path_anchor_indices(len(future))
    values = [2.0 * elapsed_hours[index] for index in anchors]

    candidate, diagnostics = candidate_from_sampled_paths([
        '{"forecast_path":{"values":' + str(values) + '}}',
    ], future, history_values=[-2, -1, 0])

    assert diagnostics["accepted"] == 1
    assert candidate is not None
    assert candidate["_validated_sample_paths"][0] == pytest.approx([
        2.0 * value for value in elapsed_hours])


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
    assert [item["status"] for item in diagnostics["response_shapes"]] == [
        "accepted", "accepted", "rejected_wrong_shape"]
    assert len(diagnostics["response_shapes"][0]["sha256"]) == 64
    assert "output" not in diagnostics["response_shapes"][0]


def test_analogue_paths_require_grounded_majority_consistent_reference():
    future = ["2026-01-02T00:00:00+00:00",
              "2026-01-03T00:00:00+00:00"]
    outputs = [json.dumps({"forecast_path": {
        "values": values, "claim_ids": claim_ids}})
        for values, claim_ids in [
            ([10, 12], ["target", "coastal"]),
            ([11, 13], ["target", "coastal"]),
            ([1, 1], ["target", "inland"]),
            ([9, 9], ["target", "coastal", "inland"]),
            ([8, 8], ["invented", "coastal"]),
        ]]

    candidate, diagnostics = candidate_from_sampled_paths(
        outputs, future, history_values=[0, 1, 2],
        allowed_claim_ids={"target", "coastal", "inland"},
        required_claim_groups=[{"target"}, {"coastal", "inland"}],
        single_choice_claim_ids={"coastal", "inland"})

    assert candidate is not None
    assert candidate["_validated_sample_paths"] == [
        [10.0, 12.0], [11.0, 13.0]]
    assert candidate["_selected_claim_ids"] == ["coastal", "target"]
    assert diagnostics["reference_selection"] == {
        "counts": {"coastal": 2, "inland": 1},
        "required_majority": 2,
        "selected_claim_id": "coastal",
        "interpretation": "majority_consistent_comparable_not_skill",
    }
    assert diagnostics["accepted_after_reference_consensus"] == 2
    assert diagnostics["accepted"] == 2
    assert diagnostics["rejected"] == 3
    assert "paths citing minority comparables were excluded" in (
        diagnostics["rejection_reasons"])
    assert [item["status"] for item in diagnostics["response_shapes"]][-2:] == [
        "rejected_ambiguous_reference_choice", "rejected_unknown_claim_ids"]


def test_analogue_paths_with_tied_reference_choice_are_withheld():
    future = ["2026-01-02T00:00:00+00:00"]
    outputs = [json.dumps({"forecast_path": {
        "values": [value], "claim_ids": ["target", choice]}})
        for value, choice in [(10, "a"), (1, "b")]]

    candidate, diagnostics = candidate_from_sampled_paths(
        outputs, future, allowed_claim_ids={"target", "a", "b"},
        required_claim_groups=[{"target"}, {"a", "b"}],
        single_choice_claim_ids={"a", "b"})

    assert candidate is None
    assert diagnostics["reference_selection"]["selected_claim_id"] is None
    assert diagnostics["accepted"] == 0
    assert diagnostics["rejected"] == 2
    assert "sampled paths did not agree on one comparable" in (
        diagnostics["rejection_reasons"])


def test_external_analogue_selection_requires_a_disclosed_assumption():
    future = ["2026-01-02T00:00:00+00:00"]
    candidate, diagnostics = candidate_from_sampled_paths([
        '{"forecast_path":{"values":[3],"claim_ids":["target","peer"]}}',
    ], future, allowed_claim_ids={"target", "peer"},
        required_claim_groups=[{"target"}, {"peer"}],
        single_choice_claim_ids={"peer"}, require_rationale=True)

    assert candidate is None
    assert diagnostics["response_shapes"][0]["status"] == \
        "rejected_missing_external_assumption"

    prompt = build_sampled_context_prior_prompt(
        timestamps=["2026-01-01T00:00:00+00:00"], values=[1],
        future_timestamps=future, context="Reference rows omit attributes.",
        claim_catalog={"target": "coastal", "peer": "North [1, 4]"},
        single_choice_claim_ids={"peer"},
        external_matching_assumption_required=True)
    assert "external matching assumption" in prompt
    assert "not source-grounded or automation-safe" in prompt


def test_sampled_driver_paths_are_transformed_by_governed_math():
    future = ["2026-01-02T00:00:00+00:00",
              "2026-01-03T00:00:00+00:00"]
    candidate, diagnostics = candidate_from_sampled_paths([
        '{"forecast_path":{"values":[10,20]}}',
        '{"forecast_path":{"values":[20,30]}}',
    ], future, history_values=[1, 2, 3],
        path_transform=lambda path: [(value / 10) ** 2 for value in path])

    assert diagnostics["accepted"] == 2
    assert candidate is not None
    assert candidate["_validated_sample_paths"] == [[1.0, 4.0], [4.0, 9.0]]
    assert [row["q50"] for row in candidate["quantiles"]] == [2.5, 6.5]


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


def test_zero_marginal_width_does_not_waive_path_coherence():
    result = sampled_prior_sufficiency({
        "requested": 3,
        "accepted": 3,
        "stability": {
            "version": "0.1",
            "interpretation": "stability_not_historical_skill",
            "scale_basis": "level_floor",
            "path_count": 3,
            "horizon": 6,
            "median_pointwise_q80_width_scaled": 0.0,
            "p90_pointwise_q80_width_scaled": 4.0,
            "median_pairwise_mae_scaled": 1.8,
            "max_pairwise_mae_scaled": 3.0,
            "mean_direction_agreement": 1.0,
            "unanimous_direction_fraction": 1.0,
        },
    })

    assert result["eligible_for_human_recommendation"] is False
    assert "dispersed_sampled_paths" in result["reason_codes"]


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
