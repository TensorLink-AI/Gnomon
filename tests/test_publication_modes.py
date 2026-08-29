from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gnomon.llm_dossier import (attach_host_candidate_elicitation,
                                validate_temporal_dossier)
from gnomon.publication import (build_scenario_catalog, publish_result,
                                best_effort_prior_selection,
                                dominant_scenario_id,
                                scenario_selection_contract, select_publication,
                                validate_scenario_selection,
                                verify_publication, write_publication)
from gnomon.publication import record_publication
from gnomon.tracking import TrackingStore
from gnomon.artifacts import verify_artifact_integrity
from gnomon.toolspec import runner_for
from gnomon.toolspec import enforce_response_budget


TIMES = ["2026-01-03T00:00:00+00:00", "2026-01-04T00:00:00+00:00"]


def _stable_sampling(path_count: int) -> dict:
    return {
        "version": "0.2",
        "interpretation": "stability_not_historical_skill",
        "scale_basis": "median_nonzero_history_increment",
        "path_count": path_count,
        "horizon": len(TIMES),
        "median_pointwise_q80_width_scaled": .5,
        "mean_pointwise_q80_width_scaled": .5,
        "p90_pointwise_q80_width_scaled": .7,
        "median_pairwise_mae_scaled": .4,
        "max_pairwise_mae_scaled": .8,
        "mean_direction_agreement": .9,
        "unanimous_direction_fraction": .5,
    }


def _result():
    return {
        "support": "supported",
        "forecast": [
            {"timestamp": TIMES[0], "point": 10.0, "q10": 9, "q50": 10, "q90": 11},
            {"timestamp": TIMES[1], "point": 10.0, "q10": 9, "q50": 10, "q90": 11},
        ],
    }


def test_failed_alternatives_do_not_trigger_a_model_selection_call():
    scenarios = [
        {"scenario_id": "primary", "role": "immutable_primary",
         "selection_eligible": True, "human_selection_eligible": True},
        {"scenario_id": "state", "role": "governed_categorical_state_mapping",
         "selection_eligible": False, "human_selection_eligible": False},
    ]
    assert dominant_scenario_id(scenarios) == "primary"


def test_replayed_reference_law_answers_best_effort_conditional_question():
    scenarios = [
        {"scenario_id": "primary", "role": "immutable_primary",
         "selection_eligible": True, "human_selection_eligible": True},
        {"scenario_id": "model-assisted", "role": "model_assisted",
         "selection_eligible": True, "human_selection_eligible": True,
         "support": "conditionally_supported"},
        {"scenario_id": "law", "role": "governed_reference_law_mapping",
         "selection_eligible": True, "human_selection_eligible": True,
         "support": "prior_assisted", "effect": {"validation": {
             "best_effort_human_gate_passed": True}}},
    ]

    assert dominant_scenario_id(scenarios) == "law"


def _dossier():
    raw = {
        "claims": [{"source_span": "promotion begins tomorrow",
                    "relation": "supports_increase",
                    "effective_start": TIMES[0], "effective_end": TIMES[1],
                    "confidence": .8}],
        "forecast_candidate": {"quantiles": [
            {"timestamp": TIMES[0], "q10": 10, "q50": 11, "q90": 12},
            {"timestamp": TIMES[1], "q10": 11, "q50": 12, "q90": 13},
        ], "rationale": "promotion"},
    }
    return validate_temporal_dossier(
        raw, context_text="promotion begins tomorrow",
        cutoff="2026-01-02T00:00:00+00:00", future_timestamps=TIMES,
        history=[8, 9, 10], compiler_model="test-model")[0]


def test_strict_never_promotes_prior_assisted_candidate():
    payload = publish_result(_result(), mode="strict", dossiers=[_dossier()])
    assert payload["recommended_scenario_id"] == "primary"
    assert payload["recommended_forecast"] == _result()["forecast"]
    assert payload["automation"]["eligible"] is False
    assert verify_publication(payload)


def test_candidate_distance_from_primary_is_sealed_and_non_authoritative():
    scenarios, _ = build_scenario_catalog(_result(), dossiers=[_dossier()])
    candidate = next(item for item in scenarios
                     if item["role"] == "model_authored")
    disagreement = candidate["effect"]["primary_disagreement"]
    assert disagreement == {
        "available": True,
        "interpretation": "difference_not_skill",
        "scale_basis": "median_primary_q80_width",
        "median_absolute_difference_scaled": .75,
        "max_absolute_difference_scaled": 1.0,
        "direction_disagreement_fraction": 1.0,
    }
    assert candidate["support"] == "prior_assisted"
    assert candidate["automation_eligible"] is False
    assert candidate["effect"]["distribution"] == {
        "kind": "sealed_model_quantiles",
        "horizon": 2,
        "quantile_levels": [0.1, 0.5, 0.9],
        "source": "sealed_context_receipt",
        "probabilistic_consumers_should_use": "quantiles",
        "stability_evidence": "not_measured",
        "historical_skill_evidence": False,
        "compact_human_summary": "recommended_forecast",
        "automation_eligible": False,
    }
    assert candidate["scenario_seal_sha256"]


def test_claims_only_context_is_retained_without_claiming_numeric_use():
    span = "A comparable site reached 120 last summer."
    dossier, reasons = validate_temporal_dossier({
        "claims": [{
            "source_span": span, "relation": "unknown",
            "effective_start": TIMES[0], "effective_end": TIMES[1],
            "mechanism": "Weak external analogue", "confidence": .3,
        }],
        "hypotheses": [{
            "kind": "historical_analogue", "claim_ids": ["claim-1"],
            "target_series": ["*"], "known_at":
                "2026-01-02T00:00:00+00:00", "lag_steps": 0,
            "direction": "unknown", "rationale": "Different site and season.",
        }],
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")
    assert not reasons
    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])

    assert payload["recommended_scenario_id"] == "primary"
    assert payload["context_summary"]["status"] == "scenario_only"
    assert payload["context_summary"]["counts"] == {
        "used": 0, "scenario": 1, "rejected": 0}
    disposition = payload["context_dispositions"][0]
    assert disposition["reason_code"] == \
        "interpretation_only_no_numeric_path"
    assert "did not alter" in disposition["reason"]


def test_same_claim_in_multiple_sealed_dossiers_is_presented_once():
    span = "A comparable site reached 120 last summer."
    dossier, reasons = validate_temporal_dossier({
        "claims": [{
            "source_span": span, "relation": "unknown",
            "effective_start": TIMES[0], "effective_end": TIMES[1],
            "mechanism": "Weak external analogue", "confidence": .3,
        }],
        "hypotheses": [{
            "kind": "historical_analogue", "claim_ids": ["claim-1"],
            "target_series": ["*"], "known_at":
                "2026-01-02T00:00:00+00:00", "lag_steps": 0,
            "direction": "unknown", "rationale": "Comparable site.",
        }],
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")
    assert not reasons

    payload = publish_result(
        _result(), mode="best_effort",
        dossiers=[dossier, deepcopy(dossier)])

    assert payload["context_summary"]["counts"] == {
        "used": 0, "scenario": 1, "rejected": 0}
    assert len(payload["context_dispositions"]) == 1
    disposition = payload["context_dispositions"][0]
    assert disposition["representation_count"] == 2
    assert disposition["source_context_ids"] == [
        "dossier-1:claim-1", "dossier-2:claim-1"]
    assert disposition["cited_fact"]["source_span"] == span
    assert verify_publication(payload)


def test_valid_covariate_input_is_distinct_from_forecast_admission():
    result = _result()
    result["covariates"] = {
        "considered": False, "admitted": False, "retained": [],
        "rejected": [{"reason": "base evaluation is unsupported"}],
    }
    payload = publish_result(result, mode="best_effort")
    assert payload["context_input_evaluation"]["status"] == (
        "received_not_evaluable")
    assert payload["context_summary"]["status"] == "received_not_evaluable"
    assert "passed ingestion" in payload["context_summary"]["message"]
    assert payload["context_summary"]["follow_up_required_for_current_recommendation"]
    assert verify_publication(payload)


def test_unresolved_trigger_claim_returns_dated_recovery_not_generic_rejection():
    span = "Demand typically falls during public holidays."
    dossier, reasons = validate_temporal_dossier({
        "claims": [{
            "source_span": span, "relation": "supports_decrease",
            "effective_start": None, "effective_end": None,
            "timing_status": "unresolved_trigger",
            "mechanism": "Holiday demand effect", "confidence": .7,
        }],
        "hypotheses": [{
            "kind": "unsupported", "claim_ids": ["claim-1"],
            "target_series": ["*"], "predictor_series": None,
            "known_at": "2026-01-02T00:00:00+00:00", "lag_steps": 0,
            "direction": "decrease", "rationale": "Trigger date missing.",
        }],
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")
    assert not reasons

    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])

    disposition = next(item for item in payload["context_dispositions"]
                       if item.get("claim_id") == "claim-1")
    assert disposition["disposition"] == "scenario"
    assert disposition["reason_code"] == "trigger_timing_unresolved"
    assert disposition["recovery_action"]["code"] == "provide_dated_trigger"
    assert disposition["recovery_action"]["automation_eligible"] is False
    assert payload["recommended_scenario_id"] == "primary"
    assert payload["automation"]["eligible"] is False


def test_atemporal_claim_requests_applicability_not_a_trigger_date():
    span = "On average, the service receives 12 incidents per year."
    dossier, reasons = validate_temporal_dossier({
        "claims": [{
            "source_span": span, "relation": "supports_stability",
            "effective_start": None, "effective_end": None,
            "timing_status": "atemporal_context", "confidence": .7,
        }],
        "hypotheses": [{
            "kind": "historical_analogue", "claim_ids": ["claim-1"],
            "target_series": ["*"], "predictor_series": None,
            "known_at": "2026-01-02T00:00:00+00:00", "lag_steps": 0,
            "direction": "unknown", "rationale": "Historical background.",
        }],
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")
    assert not reasons

    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])

    disposition = next(item for item in payload["context_dispositions"]
                       if item.get("claim_id") == "claim-1")
    assert disposition["reason_code"] == "background_context_not_conditioned"
    assert disposition["recovery_action"]["code"] \
        == "provide_applicability_evidence"
    assert disposition["cited_fact"] == {
        "source_span": span,
        "relation": "supports_stability",
        "confidence": .7,
    }
    assert "date" not in disposition["recovery_action"]["message"]
    assert payload["recommended_scenario_id"] == "primary"
    assert payload["automation"]["eligible"] is False


def test_unmatched_comparable_returns_specific_attribute_recovery():
    span = "North (pop. 90): [11, 28] orders"
    dossier, reasons = validate_temporal_dossier({
        "claims": [{
            "source_span": span, "relation": "supports_stability",
            "effective_start": None, "effective_end": None,
            "timing_status": "atemporal_context",
            "mechanism": (
                "source-stated comparable-entity numeric range without "
                "stated matching attributes"),
            "confidence": 1.0,
        }],
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")
    assert not reasons

    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])

    disposition = payload["context_dispositions"][0]
    recovery = disposition["recovery_action"]
    assert recovery["code"] == "provide_comparable_attributes"
    assert "matching rule" in recovery["message"]
    assert recovery["automation_eligible"] is False
    assert payload["recommended_scenario_id"] == "primary"
    assert payload["context_summary"][
        "additional_evidence_could_change_recommendation"] is True


def test_selected_external_analogue_admits_that_more_evidence_can_help():
    dossier = _dossier()
    dossier = attach_host_candidate_elicitation(
        dossier, requested_paths=3, accepted_paths=3,
        aggregation="linear_empirical_marginal_q10_q50_q90",
        stability=_stable_sampling(3),
        temperature=1.0, sample_paths=[
            [10.8, 11.8], [11.0, 12.0], [11.2, 12.2]])
    dossier["claims"][0].update({
        "effective_start": None, "effective_end": None,
        "timing_status": "atemporal_context",
        "mechanism": (
            "source-stated comparable-entity numeric range without "
            "stated matching attributes"),
    })
    import hashlib, json
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])

    assert payload["recommended_scenario_id"] == "primary"
    scenarios = payload["candidate_portfolio"]
    contract = scenario_selection_contract(
        scenarios=scenarios, dossiers=[dossier],
        allow_prior_assisted_choice=True)
    assert contract["selection_required"] is True
    selection = validate_scenario_selection({
        "selected_scenario_id": "prior-assisted-1",
        "ranking": ["prior-assisted-1", "primary"],
        "cited_claim_ids": ["claim-1"],
        "counterevidence_claim_ids": [],
        "counterevidence_hypothesis_ids": [],
        "confidence": .4,
        "rationale": "The external coastal assumption is useful but weak.",
        "what_would_change_selection": "Source-stated peer attributes.",
    }, scenarios=scenarios, dossiers=[dossier])
    assert selection is not None
    payload = publish_result(
        _result(), mode="best_effort", dossiers=[dossier],
        scenario_selection=selection)
    assert payload["recommended_scenario_id"] == "prior-assisted-1"
    summary = payload["context_summary"]
    assert summary["follow_up_required_for_current_recommendation"] is False
    assert summary["additional_evidence_could_change_recommendation"] is True
    assert summary[
        "further_calls_add_nothing_for_current_recommendation"] is False
    assert verify_publication(payload)


def test_atemporal_peer_bound_preserves_fact_and_requests_a_reference_path():
    span = "A comparable site's maximum was 25.83 at 21:10:00."
    dossier, reasons = validate_temporal_dossier({
        "claims": [{
            "source_span": span, "relation": "constrains_range",
            "effective_start": None, "effective_end": None,
            "timing_status": "atemporal_context", "confidence": .5,
        }],
        "hypotheses": [{
            "kind": "bound", "claim_ids": ["claim-1"],
            "target_series": ["*"], "predictor_series": None,
            "known_at": "2026-01-02T00:00:00+00:00", "lag_steps": 0,
            "direction": "unknown", "rationale": "Peer upper bound only.",
        }],
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")
    assert not reasons

    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])

    disposition = payload["context_dispositions"][0]
    assert disposition["cited_fact"]["source_span"] == span
    assert disposition["cited_fact"]["relation"] == "constrains_range"
    assert disposition["recovery_action"]["required_evidence"][:2] == [
        "reference observations over the forecast grid",
        "target-to-reference scale or historical overlap",
    ]
    assert payload["recommended_scenario_id"] == "primary"
    assert payload["automation"]["eligible"] is False


def test_exact_cited_scenario_is_human_facing_but_never_automatable():
    span = "In this case demand will be only 5 times the usual level."
    dossier, reasons = validate_temporal_dossier({
        "claims": [{
            "source_span": span, "relation": "supports_increase",
            "effective_start": TIMES[0], "effective_end": TIMES[0],
            "confidence": .8,
        }],
        "effect_proposal": {
            "shape": "temporary_pulse", "unit": "fraction_of_level",
            "location": 4, "lower": 4, "upper": 4, "confidence": .8,
            "delay_steps": 0, "duration_steps": 1,
            "scope": {"kind": "single_series", "series": ["*"]},
            "claim_ids": ["claim-1"], "composition": "scenario_only",
        },
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
        future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")
    assert not reasons

    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])
    assert payload["recommended_scenario_id"] == "effect-composed-1"
    assert payload["recommended_support"] == "hypothetical_sensitivity"
    assert payload["selection_contract"]["selection_required"] is False
    assert payload["selection_contract"]["deterministic_scenario_id"] == \
        "effect-composed-1"
    assert payload["automation"]["eligible"] is False
    assert payload["primary_forecast"] == _result()["forecast"]
    assert verify_publication(payload)


def test_approximate_cited_scenario_cannot_be_silently_demoted():
    span = "In this case demand will be about 5 times the usual level."
    dossier, reasons = validate_temporal_dossier({
        "claims": [{
            "source_span": span, "relation": "supports_increase",
            "effective_start": TIMES[0], "effective_end": TIMES[0],
            "confidence": .7,
        }],
        "effect_proposal": {
            "shape": "temporary_pulse", "unit": "fraction_of_level",
            "location": 4, "lower": 2, "upper": 4, "confidence": .7,
            "delay_steps": 0, "duration_steps": 1,
            "scope": {"kind": "single_series", "series": ["*"]},
            "claim_ids": ["claim-1"], "composition": "scenario_only",
        },
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
        future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")
    assert not reasons

    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])
    assert payload["recommended_scenario_id"] == "effect-composed-1"
    assert payload["selection_contract"]["selection_required"] is False
    assert payload["selection_contract"]["deterministic_scenario_id"] == \
        "effect-composed-1"
    with pytest.raises(ValueError, match="evidence-dominant"):
        select_publication(payload, {
            "selected_scenario_id": "primary",
            "ranking": ["primary", "effect-composed-1"],
            "cited_claim_ids": ["claim-1"],
            "counterevidence_claim_ids": [],
            "confidence": .6,
            "rationale": "Prefer the context-free path.",
            "what_would_change_selection": "More observations.",
        })


def test_best_effort_keeps_unselected_model_candidate_visible_only():
    payload = publish_result(
        _result(), mode="best_effort", dossiers=[_dossier()],
        automation_policy={"authorize": True})
    assert payload["recommended_scenario_id"] == "primary"
    assert any(item["scenario_id"] == "prior-assisted-1"
               for item in payload["candidate_portfolio"])
    assert payload["primary_forecast"] == _result()["forecast"]
    assert payload["automation"]["eligible"] is False
    authority = payload["recommendation_authority"]
    assert authority["selection_method"] == "immutable_primary_default"
    assert authority["independent_selection_performed"] is False
    assert authority["historically_admitted"] is False
    assert authority["prior_assisted"] is False
    assert authority["human_review_required"] is False
    assert verify_publication(payload)


def test_automation_policy_is_actionable_and_can_authorize_supported_primary():
    incomplete = publish_result(
        _result(), mode="strict",
        automation_policy={"authorize": True})
    assert incomplete["automation"] == {
        "eligible": False,
        "explicit_policy_supplied": True,
        "policy_complete": False,
        "requested": True,
        "authority_source": "host_controlled",
        "reason_code": "incomplete_policy",
        "reason": ("Automation requires policy_id and minimum_support set to "
                   "supported or context_trusted."),
        "required_fields": ["authorize", "policy_id", "minimum_support"],
        "missing_fields": ["policy_id", "minimum_support"],
    }

    candidate = publish_result(
        _result(), mode="strict",
        automation_policy={"authorize": True, "policy_id": "ops-v1",
                           "minimum_support": "supported"})
    assert candidate["automation"]["eligible"] is True
    assert candidate["automation"]["reason_code"] == "authorized"
    assert candidate["automation"]["missing_fields"] == []


def test_automation_policy_rejects_unknown_fields_and_wrong_types():
    with pytest.raises(ValueError, match="unknown fields"):
        publish_result(_result(), automation_policy={"authorize": True,
                       "policy_id": "ops", "minimum_support": "supported",
                       "force": True})
    with pytest.raises(ValueError, match="authorize must be boolean"):
        publish_result(_result(), automation_policy={"authorize": "yes"})
    legacy = publish_result(
        _result(), automation_policy={"allow": True, "policy_id": "ops",
                                      "minimum_support": "supported"})
    assert legacy["automation"]["eligible"] is True
    assert legacy["automation"]["normalization"] == "allow->authorize"


def test_advisory_channel_cannot_authorize_supported_primary():
    payload = publish_result(
        _result(), automation_policy={
            "authorize": True, "policy_id": "ops",
            "minimum_support": "supported"},
        automation_authority=False)
    assert payload["automation"]["eligible"] is False
    assert payload["automation"]["requested"] is True
    assert payload["automation"]["authority_source"] == "mcp_advisory"
    assert payload["automation"]["reason_code"] == \
        "untrusted_authorization_channel"


def test_external_prediction_rejection_teaches_candidate_lane():
    result = {**_result(), "context_rejections": [{
        "context_id": "vendor", "disposition": "rejected",
        "reason_code": "external_prediction_not_constraint",
        "reason": "A prediction is not a binding constraint.",
    }]}
    payload = publish_result(result, mode="best_effort")
    disposition = payload["context_dispositions"][0]
    assert disposition["recovery_action"]["code"] == \
        "submit_governed_forecast_candidate"
    assert disposition["recovery_action"]["automation_eligible"] is False


def test_partial_model_anchors_use_primary_outside_supplied_window():
    timestamps = [f"2026-01-0{day}T00:00:00+00:00" for day in range(3, 8)]
    result = {"support": "supported", "forecast": [
        {"timestamp": stamp, "point": 10.0, "q10": 9, "q50": 10, "q90": 11}
        for stamp in timestamps]}
    span = "A comparable operation had a temporary middle-period peak."
    dossier, reasons = validate_temporal_dossier({
        "claims": [{"source_span": span, "relation": "unknown",
                    "effective_start": "2025-01-01T00:00:00+00:00",
                    "effective_end": "2025-01-02T00:00:00+00:00",
                    "confidence": .5}],
        "forecast_candidate": {"quantile_anchors": [
            {"timestamp": timestamps[2], "q10": 18, "q50": 20, "q90": 22},
        ], "rationale": "model-prior conditional peak"},
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
        future_timestamps=timestamps, history=[8, 9, 10, 11],
        compiler_model="test")
    assert not reasons

    payload = publish_result(result, mode="best_effort", dossiers=[dossier])
    scenario = next(item for item in payload["candidate_portfolio"]
                    if item["role"] == "model_authored")

    assert scenario["forecast"][0]["q50"] == 10
    assert scenario["forecast"][2]["q50"] == 20
    assert scenario["forecast"][-1]["q50"] == 10
    assert payload["primary_forecast"] == result["forecast"]
    assert scenario["automation_eligible"] is False
    assert verify_publication(payload)


def test_unadmitted_observation_sensitivity_needs_explicit_selection():
    dossier = _dossier()
    dossier["candidate_critique"]["candidate_origin"] = \
        "observation_interpretation_counterfactual"
    dossier["forecast_candidate"]["conditional_replay"] = {
        "status": "scenario_only_outcome_inferred_mask",
        "selection_eligible": False,
    }
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    import hashlib, json
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])

    assert payload["recommended_scenario_id"] == "primary"
    scenario = next(item for item in payload["candidate_portfolio"]
                    if item["role"] == "observation_counterfactual")
    assert scenario["selection_eligible"] is True
    assert scenario["effect"]["conditional_replay"][
        "selection_eligible"] is False
    contract_scenario = next(
        item for item in payload["selection_contract"]["scenarios"]
        if item["scenario_id"] == scenario["scenario_id"])
    assert contract_scenario["derivation"][
        "conditional_replay_status"] == \
        "scenario_only_outcome_inferred_mask"
    assert contract_scenario["derivation"]["historically_admitted"] is False


def test_selection_contract_compares_contaminated_primary_with_conditional_path():
    dossier = _dossier()
    dossier["observation_interpretations"] = [{
        "interpretation_id": "observation-interpretation-1",
        "kind": "historical_contamination",
        "claim_ids": ["claim-1"],
        "excluded_observations": 2,
        "retained_observations": 166,
        "input_mutated": False,
        "conditional_replay": {
            "status": "insufficient_replay_origins",
            "origins": 4,
            "minimum_origins": 12,
            "selection_eligible": False,
        },
    }]
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    import hashlib, json
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])
    contract = payload["selection_contract"]

    assert contract["observation_evidence"] == [{
        "interpretation_id": "observation-interpretation-1",
        "claim_ids": ["claim-1"],
        "kind": "historical_contamination",
        "excluded_observations": 2,
        "retained_observations": 166,
        "input_mutated": False,
        "conditional_replay": {
            "status": "insufficient_replay_origins",
            "origins": 4,
            "minimum_origins": 12,
            "selection_eligible": False,
        },
    }]
    primary = next(item for item in contract["scenarios"]
                   if item["role"] == "immutable_primary")
    candidate = next(item for item in contract["scenarios"]
                     if item["role"] == "model_authored")
    assert primary["derivation"][
        "primary_retains_claimed_contamination"] is True
    assert primary["derivation"][
        "conditional_path_addresses_claimed_contamination"] is False
    assert candidate["derivation"][
        "primary_retains_claimed_contamination"] is False
    assert candidate["derivation"][
        "conditional_path_addresses_claimed_contamination"] is True


def test_insufficient_observation_replay_allows_only_human_prior_selection():
    dossier = _dossier()
    dossier["candidate_critique"] = {
        "status": "accepted", "reasons": [],
        "selection_eligible": False,
        "selection_reason": (
            "Observation replay is insufficient; the model path cannot "
            "upgrade evidential support."),
        "candidate_origin": "model_authored",
    }
    dossier["observation_interpretations"] = [{
        "interpretation_id": "observation-interpretation-1",
        "kind": "historical_contamination", "claim_ids": ["claim-1"],
        "excluded_observations": 2, "retained_observations": 20,
        "input_mutated": False,
        "conditional_replay": {
            "status": "insufficient_replay", "origins": 4,
            "minimum_origins": 12, "selection_eligible": False,
        },
    }]
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    import hashlib, json
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    scenarios = build_scenario_catalog(_result(), dossiers=[dossier])[0]
    candidate = next(item for item in scenarios
                     if item["role"] == "model_authored")
    assert candidate["selection_eligible"] is False
    assert candidate["human_selection_eligible"] is True
    assert candidate["automation_eligible"] is False

    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier],
                             scenario_selection={
        "selected_scenario_id": candidate["scenario_id"],
        "ranking": [candidate["scenario_id"], "primary"],
        "cited_claim_ids": ["claim-1"],
        "counterevidence_claim_ids": [], "confidence": .55,
        "rationale": "The primary retains the cited contaminated readings.",
        "what_would_change_selection": "Sufficient replay evidence.",
    })
    assert payload["recommended_scenario_id"] == candidate["scenario_id"]
    assert payload["recommended_support"] == "prior_assisted"
    assert payload["automation"]["eligible"] is False
    assert payload["primary_forecast"] == _result()["forecast"]


def test_admitted_observation_counterfactual_is_not_shadowed_by_its_claim():
    dossier = _dossier()
    dossier["claims"][0]["source_span"] = (
        "The sensor was offline, which resulted in zero readings.")
    dossier["candidate_critique"]["candidate_origin"] = \
        "observation_interpretation_counterfactual"
    dossier["forecast_candidate"]["conditional_replay"] = {
        "status": "admitted", "selection_eligible": True,
        "human_recommendation_eligible": True,
    }
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    import hashlib, json
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])
    scenario = next(item for item in payload["candidate_portfolio"]
                    if item["role"] == "observation_counterfactual")
    assert scenario["selection_eligible"] is True
    assert scenario["effect"]["distribution"] == {
        "kind": "sealed_governed_quantiles",
        "candidate_origin": "observation_interpretation_counterfactual",
        "horizon": 2,
        "quantile_levels": [0.1, 0.5, 0.9],
        "source": "sealed_context_receipt",
        "probabilistic_consumers_should_use": "quantiles",
        "historical_skill_evidence": True,
        "validation_status": "admitted",
        "compact_human_summary": "recommended_forecast",
        "automation_eligible": False,
    }
    assert payload["recommended_scenario_id"] == scenario["scenario_id"]
    assert payload["recommendation_authority"][
        "conditional_replay_admitted"] is True


def test_best_effort_may_use_positive_replay_below_strict_margin():
    dossier = _dossier()
    dossier["candidate_critique"]["candidate_origin"] = \
        "observation_interpretation_counterfactual"
    dossier["forecast_candidate"]["conditional_replay"] = {
        "status": "not_admitted",
        "selection_eligible": False,
        "human_recommendation_eligible": True,
    }
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    import hashlib, json
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    strict = publish_result(_result(), mode="strict", dossiers=[dossier])
    best = publish_result(_result(), mode="best_effort", dossiers=[dossier])

    assert strict["recommended_scenario_id"] == "primary"
    assert best["recommended_scenario_id"] == "prior-assisted-1"
    assert best["recommendation_authority"]["selection_method"] == \
        "conditional_replay_best_effort"
    assert best["recommended_support"] == "prior_assisted"
    assert best["automation"]["eligible"] is False
    assert best["primary_forecast"] == strict["primary_forecast"]
    candidate_contract = next(
        item for item in best["selection_contract"]["scenarios"]
        if item["scenario_id"] == "prior-assisted-1")
    assert candidate_contract["derivation"][
        "human_recommendation_eligible"] is True


def test_replay_admitted_observation_counterfactual_has_truthful_authority():
    import random
    rng = random.Random(0)
    start = datetime(2025, 10, 5, tzinfo=timezone.utc)
    history_times = [(start + timedelta(days=index)).isoformat()
                     for index in range(120)]
    history = []
    for index in range(120):
        disrupted = index % 5 in {0, 1}
        history.append((-100.0 + rng.gauss(0, 3)) if disrupted
                       else (20.0 + rng.gauss(0, 1)))
    claim = (
        "Maintenance lasted for 2 days every 5 days starting from "
        "2025-10-05 00:00:00, resulting in no requests recorded.")
    context = claim + " There will be no future maintenance."
    dossier, reasons = validate_temporal_dossier(
        {"claims": [{"source_span": claim, "relation": "unknown",
                     "effective_start": history_times[0],
                     "effective_end": history_times[-1], "confidence": 1}]},
        context_text=context, cutoff=history_times[-1],
        future_timestamps=TIMES, history=history,
        history_timestamps=history_times, compiler_model="test")
    assert not reasons
    assert dossier["candidate_critique"]["selection_eligible"] is True

    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])

    assert payload["recommended_scenario_id"] == "prior-assisted-1"
    candidate = next(item for item in payload["candidate_portfolio"]
                     if item["scenario_id"] == "prior-assisted-1")
    assert candidate["role"] == "observation_counterfactual"
    assert candidate["support"] == "conditionally_supported"
    assert candidate["effect"]["conditional_replay"][
        "selection_eligible"] is True
    authority = payload["recommendation_authority"]
    assert authority["selection_method"] == "conditional_replay_evidence"
    assert authority["conditional_replay_admitted"] is True
    assert authority["historically_admitted"] is False
    assert authority["human_review_required"] is True
    assert payload["automation"]["eligible"] is False
    assert verify_publication(payload)
    assert payload["selection_contract"]["selection_required"] is False
    assert payload["selection_contract"]["deterministic_scenario_id"] == (
        payload["recommended_scenario_id"])


def test_governed_selection_repoints_sealed_path_without_reforecasting():
    original = publish_result(
        _result(), mode="best_effort", dossiers=[_dossier()])
    original_seals = [item["scenario_seal_sha256"]
                      for item in original["candidate_portfolio"]]
    selected = select_publication(original, {
        "selected_scenario_id": "prior-assisted-1",
        "ranking": ["prior-assisted-1", "primary"],
        "cited_claim_ids": ["claim-1"],
        "counterevidence_claim_ids": [],
        "confidence": .6,
        "rationale": "The conditional claim is not independently validated.",
        "what_would_change_selection": "A resolved outcome history.",
    })
    assert original["recommended_scenario_id"] == "primary"
    assert selected["recommended_scenario_id"] == "prior-assisted-1"
    assert [item["scenario_seal_sha256"]
            for item in selected["candidate_portfolio"]] == original_seals
    assert selected["primary_forecast"] == original["primary_forecast"]
    assert selected["primary_forecast_unchanged"] is True
    assert selected["recommended_support"] == "prior_assisted"
    assert selected["automation"]["eligible"] is False
    assert selected["recommendation_authority"][
        "selection_pass_performed"] is True
    assert selected["recommendation_authority"][
        "selector_independence"] == "not_attested"
    assert selected["recommendation_authority"][
        "independent_selection_performed"] is False
    assert selected["supersedes_publication_seal_sha256"] == original[
        "publication_seal_sha256"]
    assert verify_publication(original)
    assert verify_publication(selected)


def test_governed_selection_refuses_strict_or_tampered_publication():
    strict = publish_result(_result(), mode="strict")
    with pytest.raises(ValueError, match="strict"):
        select_publication(strict, {})
    broken = deepcopy(publish_result(
        _result(), mode="scenario", dossiers=[_dossier()]))
    broken["primary_forecast"][0]["point"] = -999
    with pytest.raises(ValueError, match="invalid publication"):
        select_publication(broken, {})


def test_mcp_selector_persists_new_sidecar_without_reforecasting(tmp_path):
    original = publish_result(
        _result(), mode="best_effort", dossiers=[_dossier()], artifact_id="f1")
    original_path = write_publication(tmp_path / "forecast_f1", original)
    payload = runner_for("gnomon_select_scenario")({
        "publication_path": str(original_path),
        "scenario_selection": {
            "selected_scenario_id": "primary",
            "ranking": ["primary", "prior-assisted-1"],
            "cited_claim_ids": [],
            "counterevidence_claim_ids": ["claim-1"],
            "confidence": .6,
            "rationale": "Prefer the governed primary pending outcomes.",
            "what_would_change_selection": "Resolved candidate outcomes.",
        },
    })
    assert payload["recommended_scenario_id"] == "primary"
    assert payload["artifact_id"] == "f1"
    assert payload["primary_forecast_unchanged"] is True
    assert payload["automation"]["eligible"] is False
    assert payload["reasoning"]["canonical_source"] == "/headline"
    assert payload["reasoning"]["sufficiency"] == {
        "sufficient_for": [
            "select_scenario:canonical_answer", "explain_support"],
        "further_calls_add_nothing_for": [
            "select_scenario:canonical_answer", "explain_support"],
        "requires_follow_up": False,
    }
    assert payload["reasoning"]["resolution"]["kind"] == "complete"
    selected_path = Path(payload["publication_path"])
    assert selected_path.is_file()
    assert selected_path != original_path
    assert original_path.read_text(encoding="utf-8") == (
        __import__("json").dumps(original, indent=2, sort_keys=True) + "\n")


def test_model_authored_path_cannot_bypass_governed_transform_authority():
    result = _result()
    result["transformation_candidates"] = [{
        "transformation_id": "equation-1",
        "forecast": [{**row, "point": 20, "q10": 19, "q50": 20,
                      "q90": 21} for row in result["forecast"]],
        "lane": "historically_testable", "claim_ids": ["claim-1"],
        "known_at": "2026-01-02T00:00:00+00:00",
        "output_unit": "value", "source_seal_sha256": "sealed",
        "primary_forecast_unchanged": True, "automation_eligible": False,
        "validation": {"recurrence_plausibility_passed": True,
                       "recurrence_replay_admitted": False,
                       "recurrence_replay_reason": "did_not_beat_last_value"},
    }]
    payload = publish_result(result, mode="best_effort", dossiers=[_dossier()])
    assert payload["recommended_scenario_id"] == "primary"
    by_id = {item["scenario_id"]: item
             for item in payload["candidate_portfolio"]}
    assert by_id["transformation-1"]["selection_eligible"] is False
    assert by_id["prior-assisted-1"]["selection_eligible"] is False
    assert by_id["prior-assisted-1"]["human_selection_eligible"] is False
    assert "owns recommendation authority" in " ".join(
        by_id["prior-assisted-1"]["assumptions"])

    contract = scenario_selection_contract(
        scenarios=payload["candidate_portfolio"], dossiers=[_dossier()])
    assert all(item["scenario_id"] != "prior-assisted-1"
               or item["human_selection_eligible"] is False
               for item in contract["scenarios"])
    assert "every ineligible scenario must rank below every eligible" in (
        contract["instruction"])


def test_single_validated_declarative_transform_is_evidence_dominant():
    result = _result()
    result["transformation_candidates"] = [{
        "transformation_id": "equation-1",
        "forecast": [{**row, "point": 20, "q10": 19, "q50": 20,
                      "q90": 21} for row in result["forecast"]],
        "lane": "prior_assisted", "claim_ids": ["claim-1"],
        "known_at": "2026-01-02T00:00:00+00:00",
        "output_unit": "value", "source_seal_sha256": "sealed",
        "primary_forecast_unchanged": True, "automation_eligible": False,
        "validation": {
            "approved_ast": True, "constants_entailed": True,
            "known_at_cutoff": True, "units_checked": True,
        },
    }]
    payload = publish_result(result, mode="best_effort", dossiers=[_dossier()])

    assert payload["recommended_scenario_id"] == "transformation-1"
    assert payload["recommended_support"] == "prior_assisted"
    assert payload["automation"]["eligible"] is False
    contract = scenario_selection_contract(
        scenarios=payload["candidate_portfolio"], dossiers=[_dossier()])
    assert contract["selection_required"] is False
    assert contract["deterministic_scenario_id"] == "transformation-1"


def test_partial_holdout_model_assistance_stays_visible_but_not_selectable():
    result = _result()
    result["model_assisted"] = {
        "support": "prior_assisted", "selected_model": "seasonal_naive",
        "points": [12.0 for _ in result["forecast"]],
        "validation": {"basis": "single_trailing_holdout",
                       "out_of_sample_steps": 2},
        "plausibility": {"valid": True},
        "automation_eligible": False, "primary_forecast_unchanged": True,
    }
    scenarios, _ = build_scenario_catalog(result)
    assisted = next(item for item in scenarios
                    if item["scenario_id"] == "model-assisted")
    assert assisted["human_selection_eligible"] is False
    assert assisted["selection_eligible"] is False
    assert assisted["automation_eligible"] is False
    assert assisted["effect"]["selected_model"] == "seasonal_naive"
    assert assisted["effect"]["interval_basis"] == (
        "immutable_primary_offsets")
    assert all(row["q50"] == 12.0 for row in assisted["forecast"])
    assert all(row["q90"] - row["q10"] == 2.0
               for row in assisted["forecast"])


def test_full_cycle_seasonal_evidence_is_deterministic_in_best_effort():
    result = _result()
    result["model_assisted"] = {
        "support": "prior_assisted", "selected_model": "seasonal_naive",
        "points": [12.0 for _ in result["forecast"]],
        "validation": {"basis": "full_cycle_prequential",
                       "complete_phase_coverage": True,
                       "relative_improvement": .7,
                       "phase_block_wins": 4},
        "plausibility": {"valid": True},
        "automation_eligible": False, "primary_forecast_unchanged": True,
    }
    scenarios, _ = build_scenario_catalog(result)
    assisted = next(item for item in scenarios
                    if item["scenario_id"] == "model-assisted")
    assert assisted["human_selection_eligible"] is True
    assert dominant_scenario_id(scenarios) == "model-assisted"


def test_best_effort_sampled_prior_policy_is_human_only_and_mode_specific():
    dossier = _dossier()
    for row in dossier["forecast_candidate"]["quantiles"]:
        row["q10"] = row["q50"]
        row["q90"] = row["q50"]
    dossier["forecast_candidate"]["elicitation"] = {
        "kind": "sampled_point_paths", "requested_paths": 5,
        "accepted_paths": 4,
        "aggregation": "linear_empirical_marginal_q10_q50_q90",
        "temperature": 1.0, "host_observed": True,
        "historical_skill_evidence": False, "automation_eligible": False,
        "stability": _stable_sampling(4),
    }
    dossier["forecast_candidate"]["sample_paths"] = [
        [10.0 + draw, 11.0 + draw] for draw in range(4)]
    import hashlib, json
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    scenarios, _ = build_scenario_catalog(_result(), dossiers=[dossier])
    sampled = next(item for item in scenarios
                   if item["scenario_id"] == "prior-assisted-1")
    assert [row["q50"] for row in sampled["forecast"]] == [11.0, 12.0]
    assert all(row["q90"] - row["q10"] == 2.0
               for row in sampled["forecast"])
    normalization = sampled["effect"]["uncertainty_normalization"]
    assert normalization["rows_adjusted"] == 2
    assert normalization["candidate_centre_unchanged"] is True
    assert normalization["primary_forecast_unchanged"] is True
    distribution = sampled["effect"]["distribution"]
    assert distribution["kind"] == (
        "calibrated_quantile_offsets_around_model_median")
    assert distribution["probabilistic_consumers_should_use"] == "quantiles"
    assert distribution["raw_model_paths_are"] == (
        "elicitation_stability_only")


def test_bounded_sampled_prior_preserves_primary_outside_context_window():
    dossier = attach_host_candidate_elicitation(
        _dossier(), requested_paths=3, accepted_paths=3,
        aggregation="linear_empirical_marginal_q10_q50_q90",
        temperature=1.0, sample_paths=[[11.0, 12.0]] * 3,
        conditional_windows=[{"start": TIMES[0], "end": TIMES[0]}])

    scenarios, _ = build_scenario_catalog(_result(), dossiers=[dossier])
    sampled = next(item for item in scenarios
                   if item["scenario_id"] == "prior-assisted-1")

    assert [row["q50"] for row in sampled["forecast"]] == [11.0, 10.0]
    assert sampled["effect"]["conditional_scope"] == {
        "applied": True,
        "basis": "host_validated_context_windows",
        "conditioned_rows": 1,
        "fallback_rows_preserved": 1,
        "outside_window_source": "immutable_primary",
        "primary_forecast_unchanged": True,
    }
    assert sampled["automation_eligible"] is False


def test_bounded_context_composes_with_visible_calendar_prior():
    dossier = attach_host_candidate_elicitation(
        _dossier(), requested_paths=3, accepted_paths=3,
        aggregation="linear_empirical_marginal_q10_q50_q90",
        temperature=1.0, sample_paths=[[11.0, 12.0]] * 3,
        conditional_windows=[{"start": TIMES[0], "end": TIMES[0]}])
    result = _result()
    result["model_assisted"] = {
        "support": "prior_assisted",
        "selected_model": "calendar_seasonal_naive",
        "points": [20.0, 30.0],
        "validation": {
            "basis": "single_observed_calendar_cycle_prior",
            "complete_cycle_observed": True,
            "historical_skill_evidence": False,
        },
        "automation_eligible": False,
        "primary_forecast_unchanged": True,
    }

    scenarios, _ = build_scenario_catalog(result, dossiers=[dossier])
    sampled = next(item for item in scenarios
                   if item["scenario_id"] == "prior-assisted-1")

    assert [row["q50"] for row in sampled["forecast"]] == [11.0, 30.0]
    assert sampled["effect"]["conditional_scope"][
        "outside_window_source"] == "model_assisted_prior"
    assert sampled["effect"]["conditional_scope"][
        "fallback_rows_preserved"] == 1
    assert sampled["automation_eligible"] is False


def test_sampled_prior_does_not_auto_override_competing_sensitivity():
    dossier = attach_host_candidate_elicitation(
        _dossier(), requested_paths=3, accepted_paths=3,
        aggregation="linear_empirical_marginal_q10_q50_q90",
        temperature=1.0, sample_paths=[[11.0, 12.0]] * 3)
    result = _result()
    result["sensitivity_scenarios"] = [{
        "forecast": [{**row, "q50": 15.0, "point": 15.0}
                     for row in result["forecast"]],
        "support": "hypothetical_sensitivity",
        "assumptions": ["bounded alternative interpretation"],
    }]
    scenarios, _ = build_scenario_catalog(result, dossiers=[dossier])

    assert best_effort_prior_selection(
        scenarios=scenarios, dossiers=[dossier]) is None
    publication = publish_result(result, mode="best_effort", dossiers=[dossier])
    assert publication["recommended_scenario_id"] == "primary"


def test_under_sampled_prior_remains_visible_but_not_human_selectable():
    dossier = _dossier()
    dossier["forecast_candidate"]["elicitation"] = {
        "kind": "sampled_point_paths", "requested_paths": 3,
        "accepted_paths": 2,
        "aggregation": "linear_empirical_marginal_q10_q50_q90",
        "temperature": 1.0, "host_observed": True,
        "historical_skill_evidence": False, "automation_eligible": False,
    }
    import hashlib, json
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    scenarios, _ = build_scenario_catalog(_result(), dossiers=[dossier])

    sampled = next(item for item in scenarios
                   if item["scenario_id"] == "prior-assisted-1")
    assert sampled["human_selection_eligible"] is False
    assert sampled["automation_eligible"] is False
    assert any("Only 2 valid paths survived" in assumption
               for assumption in sampled["assumptions"])
    assert best_effort_prior_selection(
        scenarios=scenarios, dossiers=[dossier]) is None


def test_unstable_sampled_prior_stays_visible_without_displacing_primary():
    dossier = _dossier()
    dossier["forecast_candidate"]["elicitation"] = {
        "kind": "sampled_point_paths", "requested_paths": 5,
        "accepted_paths": 3,
        "aggregation": "linear_empirical_marginal_q10_q50_q90",
        "temperature": 1.0, "host_observed": True,
        "historical_skill_evidence": False, "automation_eligible": False,
        "stability": {
            **_stable_sampling(3),
            "median_pointwise_q80_width_scaled": 1.17,
            "p90_pointwise_q80_width_scaled": 13.38,
            "median_pairwise_mae_scaled": 3.92,
            "max_pairwise_mae_scaled": 4.53,
            "mean_direction_agreement": .82,
            "unanimous_direction_fraction": .47,
        },
    }
    import hashlib, json
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    scenarios, _ = build_scenario_catalog(_result(), dossiers=[dossier])
    sampled = next(item for item in scenarios
                   if item["scenario_id"] == "prior-assisted-1")
    assessment = sampled["effect"]["elicitation_sufficiency"]

    assert sampled["human_selection_eligible"] is False
    assert assessment["eligible_for_human_recommendation"] is False
    assert assessment["reason_codes"] == [
        "low_valid_path_fraction", "dispersed_sampled_paths"]
    assert best_effort_prior_selection(
        scenarios=scenarios, dossiers=[dossier]) is None
    publication = publish_result(
        _result(), mode="best_effort", dossiers=[dossier])
    assert publication["recommended_scenario_id"] == "primary"
    assert any(item["scenario_id"] == "prior-assisted-1"
               for item in publication["candidate_portfolio"])


def test_sampled_outliers_remain_diagnostics_not_published_tail_width():
    dossier = _dossier()
    for row in dossier["forecast_candidate"]["quantiles"]:
        row["q10"] = row["q50"] - 1000
        row["q90"] = row["q50"] + 1000
    dossier["forecast_candidate"]["elicitation"] = {
        "kind": "sampled_point_paths", "requested_paths": 5,
        "accepted_paths": 5,
        "aggregation": "linear_empirical_marginal_q10_q50_q90",
        "temperature": 1.0, "request_mode":
        "concurrent_single_sample_requests", "host_observed": True,
        "historical_skill_evidence": False, "automation_eligible": False,
        "stability": _stable_sampling(5),
    }
    import hashlib, json
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    scenarios, _ = build_scenario_catalog(_result(), dossiers=[dossier])
    sampled = next(item for item in scenarios
                   if item["scenario_id"] == "prior-assisted-1")
    assert [row["q50"] for row in sampled["forecast"]] == [11.0, 12.0]
    assert all(row["q90"] - row["q10"] == 2.0
               for row in sampled["forecast"])
    assert sampled["effect"]["uncertainty_normalization"]["basis"] == (
        "immutable_primary_offsets_around_sampled_median")
    selection = best_effort_prior_selection(
        scenarios=scenarios, dossiers=[dossier])
    assert selection is not None
    assert selection["selected_scenario_id"] == "prior-assisted-1"
    assert selection["channel"] == "best_effort_sampled_prior_policy"
    best = publish_result(_result(), mode="best_effort", dossiers=[dossier])
    strict = publish_result(_result(), mode="strict", dossiers=[dossier])
    assert best["recommended_scenario_id"] == "prior-assisted-1"
    assert best["automation"]["eligible"] is False
    assert best["primary_forecast_unchanged"] is True
    assert best["recommendation_authority"]["selection_method"] == (
        "best_effort_sampled_prior_policy")
    assert "host-aggregated prior distribution" in best[
        "recommendation_authority"]["reason"]
    assert "consensus" not in best["recommendation_authority"]["reason"]
    assert best["recommendation_authority"][
        "independent_selection_performed"] is False
    selected_disposition = next(item for item in best[
        "context_dispositions"] if item.get("disposition") == "used")
    assert selected_disposition["selection_reason_code"] == \
        "selected_human_facing_scenario"
    assert selected_disposition["reason_code"]
    assert "did not alter the selected numeric forecast" not in \
        selected_disposition["reason"]
    assert "grounds the separately sealed scenario" in \
        selected_disposition["reason"]
    assert "does not upgrade support" in selected_disposition[
        "selection_reason"]
    assert strict["recommended_scenario_id"] == "primary"


def test_failed_categorical_mapping_fallback_requires_bounded_selection():
    dossier = attach_host_candidate_elicitation(
        _dossier(), requested_paths=3, accepted_paths=3,
        aggregation="linear_empirical_marginal_q10_q50_q90",
        temperature=1.0, stability=_stable_sampling(3),
        sample_paths=[[10.8, 11.8], [11.0, 12.0], [11.2, 12.2]],
        governed_fallback="categorical_state_mapping_not_admitted")
    scenarios, _ = build_scenario_catalog(_result(), dossiers=[dossier])

    assert best_effort_prior_selection(
        scenarios=scenarios, dossiers=[dossier]) is None
    publication = publish_result(
        _result(), mode="best_effort", dossiers=[dossier])
    assert publication["recommended_scenario_id"] == "primary"
    contract = scenario_selection_contract(
        scenarios=scenarios, dossiers=[dossier],
        allow_prior_assisted_choice=True)
    assert contract["selection_required"] is True


def test_sampled_prior_does_not_auto_override_competing_eligible_candidate():
    dossier = attach_host_candidate_elicitation(
        _dossier(), requested_paths=3, accepted_paths=3,
        aggregation="linear_empirical_marginal_q10_q50_q90",
        temperature=1.0, stability=_stable_sampling(3),
        sample_paths=[[10.8, 11.8], [11.0, 12.0], [11.2, 12.2]])
    scenarios, _ = build_scenario_catalog(_result(), dossiers=[dossier])
    scenarios.append({
        "scenario_id": "context-executable", "role": "model_assisted",
        "human_selection_eligible": True, "automation_eligible": False,
        "support": "prior_assisted", "claim_ids": [], "forecast": [],
        "assumptions": [], "limitations": [], "effect": {},
    })

    assert best_effort_prior_selection(
        scenarios=scenarios, dossiers=[dossier]) is None


def test_selected_prior_marks_only_its_cited_claims_used():
    dossier = _dossier()
    dossier["claims"].append({
        **dossier["claims"][0],
        "claim_id": "claim-2",
        "source_span": "A competing peer reports [0, 1] units.",
    })
    dossier["forecast_candidate"]["claim_ids"] = ["claim-1"]
    dossier["forecast_candidate"]["elicitation"] = {
        "kind": "sampled_point_paths", "requested_paths": 3,
        "accepted_paths": 3,
        "aggregation": "linear_empirical_marginal_q10_q50_q90",
        "temperature": 1.0, "host_observed": True,
        "historical_skill_evidence": False, "automation_eligible": False,
        "stability": _stable_sampling(3),
    }
    dossier["forecast_candidate"]["sample_paths"] = [
        [10.0 + draw, 11.0 + draw] for draw in range(3)]
    import hashlib, json
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])

    by_claim = {item.get("claim_id"): item
                for item in payload["context_dispositions"]}
    assert by_claim["claim-1"]["disposition"] == "used"
    assert by_claim["claim-1"]["selection_reason_code"] == (
        "selected_human_facing_scenario")
    assert by_claim["claim-2"]["disposition"] == "scenario"
    assert by_claim["claim-2"].get("selection_reason_code") is None


def test_atemporal_disposition_exposes_claim_id_without_parsing_context_id():
    dossier = _dossier()
    dossier["claims"][0]["timing_status"] = "atemporal_context"
    import hashlib, json
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    payload = publish_result(_result(), mode="strict", dossiers=[dossier])

    assert payload["context_dispositions"][0]["claim_id"] == "claim-1"


def test_unapplied_numeric_bound_does_not_own_the_full_recommendation():
    dossier = _dossier()
    dossier["claims"][0].update({
        "relation": "constrains_range",
        "source_span": "Demand will remain below 100 units.",
        "mechanism": "upper bound only",
    })
    candidate = dossier["forecast_candidate"]
    candidate["claim_ids"] = ["claim-1"]
    candidate["quantiles"] = [
        {"timestamp": row["timestamp"], "q10": 10, "q50": 20, "q90": 30}
        for row in _result()["forecast"]]
    dossier.pop("seal_sha256", None)
    import hashlib, json
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        dossier, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])

    candidate_scenario = next(item for item in payload["candidate_portfolio"]
                              if item["role"] == "model_authored")
    assert candidate_scenario["human_selection_eligible"] is True
    assert candidate_scenario["automation_eligible"] is False


def test_selector_contract_exposes_interior_scenario_shape_compactly():
    primary = [{"timestamp": f"2026-01-0{index + 3}T00:00:00+00:00",
                "point": 10, "q10": 9, "q50": 10, "q90": 11}
               for index in range(4)]
    temporary = [{**row, "point": (0 if 1 <= index <= 2 else 10),
                  "q10": (0 if 1 <= index <= 2 else 9),
                  "q50": (0 if 1 <= index <= 2 else 10),
                  "q90": (0 if 1 <= index <= 2 else 11)}
                 for index, row in enumerate(primary)]
    scenarios = [
        {"scenario_id": "primary", "role": "immutable_primary",
         "support": "supported", "claim_ids": [], "forecast": primary,
         "scenario_seal_sha256": "p", "selection_eligible": True},
        {"scenario_id": "closure", "role": "model_authored",
         "support": "prior_assisted", "claim_ids": ["claim-1"],
         "forecast": temporary, "scenario_seal_sha256": "c",
         "selection_eligible": True},
    ]

    contract = scenario_selection_contract(scenarios=scenarios)
    summary = next(item["summary"] for item in contract["scenarios"]
                   if item["scenario_id"] == "closure")

    assert summary["first_q50"] == summary["last_q50"]
    assert summary["minimum_q50"] == 0
    assert summary["turning_points"] == 1
    assert summary["largest_primary_deviation"] < 0
    assert "forecast" not in contract["scenarios"][1]
    assert "conditionally_supported, degraded, best_effort" in contract[
        "instruction"]
    assert "human-review recommendation and never automation" in contract[
        "instruction"]
    assert contract["selection_facts"]["primary_support"] == "supported"
    assert contract["selection_facts"]["primary_is_fully_supported"] is True
    conditional = deepcopy(scenarios)
    conditional[0]["support"] = "conditionally_supported"
    conditional_contract = scenario_selection_contract(scenarios=conditional)
    assert conditional_contract["selection_facts"]["primary_is_fully_supported"] \
        is False
    assert conditional_contract["selection_facts"][
        "fully_supported_primary_presumption_applies"] is False
    assert conditional_contract["selection_facts"][
        "eligible_prior_scenario_ids"] == ["closure"]


def test_relationship_history_rejection_teaches_collection_not_recompilation():
    result = _result()
    result["transformation_rejections"] = [{
        "transformation_id": "fit-relationship",
        "reason_code": "INSUFFICIENT_RELATIONSHIP_HISTORY",
        "reason": "Needs at least 29 aligned observations; 16 are available.",
    }]
    payload = publish_result(result, mode="best_effort")
    rejection = payload["context_dispositions"][0]
    assert rejection["recovery_action"]["code"] == \
        "collect_relationship_history"
    assert "Rerun the same sealed lag structure" in rejection[
        "recovery_action"]["message"]
    assert payload["recommended_scenario_id"] == "primary"


def test_weak_fitted_relationship_retains_primary_without_blaming_context():
    result = _result()
    result["transformation_candidates"] = [{
        "transformation_id": "fit-relationship", "source_seal_sha256": "seal",
        "lane": "historically_testable", "claim_ids": ["claim-1"],
        "primary_forecast_unchanged": True,
        "forecast": [{**row, "point": row["point"] + 2,
                      "q10": row["q10"] + 2, "q50": row["q50"] + 2,
                      "q90": row["q90"] + 2}
                     for row in result["forecast"]],
        "validation": {"validation_points": 24, "skill": -.1,
                       "beats_baseline": False,
                       "specification_known_at_each_origin": False},
    }]
    payload = publish_result(result, mode="best_effort")
    disposition = next(item for item in payload["context_dispositions"]
                       if item["context_id"] == "fit-relationship")
    assert disposition["reason_code"] == \
        "historical_relationship_did_not_beat_baseline"
    assert "no context correction is required" in disposition["reason"]
    assert disposition.get("recovery_action") is None
    assert payload["recommended_scenario_id"] == "primary"


def test_scenario_selection_can_rank_but_not_authorize_or_edit():
    selection = {
        "selected_scenario_id": "prior-assisted-1",
        "ranking": ["prior-assisted-1", "primary"],
        "cited_claim_ids": ["claim-1"], "counterevidence_claim_ids": [],
        "confidence": .7, "rationale": "dated promotion supports uplift",
        "what_would_change_selection": "promotion cancellation",
        "automation_authorized": True,
        "forecast": [{"point": 999}],
    }
    payload = publish_result(_result(), mode="scenario", dossiers=[_dossier()],
                             scenario_selection=selection)
    assert payload["scenario_selection"]["automation_authorized"] is False
    assert "forecast" not in payload["scenario_selection"]
    assert payload["recommended_forecast"][0]["q50"] == 11
    assert verify_publication(payload)


def test_selector_cannot_attach_unrelated_context_to_primary():
    scenarios, _ = build_scenario_catalog(_result(), dossiers=[_dossier()])
    with pytest.raises(ValueError, match="selected sealed scenario"):
        validate_scenario_selection({
            "selected_scenario_id": "primary",
            "ranking": ["primary", "prior-assisted-1"],
            "cited_claim_ids": ["claim-1"],
            "counterevidence_claim_ids": [],
            "confidence": .6,
            "rationale": "The context confirms the primary.",
            "what_would_change_selection": "More evidence.",
        }, scenarios=scenarios, dossiers=[_dossier()])


def test_selector_can_treat_unattached_context_as_counterevidence():
    scenarios, _ = build_scenario_catalog(_result(), dossiers=[_dossier()])
    selection = validate_scenario_selection({
        "selected_scenario_id": "primary",
        "ranking": ["primary", "prior-assisted-1"],
        "cited_claim_ids": [],
        "counterevidence_claim_ids": ["claim-1"],
        "confidence": .6,
        "rationale": "The context supports the alternative but lacks replay.",
        "what_would_change_selection": "Successful historical replay.",
    }, scenarios=scenarios, dossiers=[_dossier()])
    assert selection["cited_claim_ids"] == []
    assert selection["counterevidence_claim_ids"] == ["claim-1"]


def test_validated_context_event_is_citable_without_a_dossier_claim():
    result = _result()
    result["primary_forecast"] = [
        {**row, "point": 9.0, "q10": 8, "q50": 9, "q90": 10}
        for row in result["forecast"]]
    result["context_outcome"] = {
        "status": "applied", "admission_basis": "future_context_contract",
        "events": ["event-1"],
    }
    selection = {
        "selected_scenario_id": "context_conditioned",
        "ranking": ["context_conditioned", "primary"],
        "cited_claim_ids": ["event-1"], "counterevidence_claim_ids": [],
        "confidence": .8, "rationale": "validated event applies",
        "what_would_change_selection": "event cancellation",
    }
    payload = publish_result(result, mode="best_effort",
                             scenario_selection=selection)
    assert payload["scenario_selection"]["cited_claim_ids"] == ["event-1"]
    authority = payload["recommendation_authority"]
    assert authority["selection_method"] == "governed_scenario_selection"
    assert authority["selection_pass_performed"] is True
    assert authority["selector_independence"] == "not_attested"
    assert authority["independent_selection_performed"] is False
    assert payload["context_summary"]["status"] == "used"
    assert verify_publication(payload)


def test_publication_summarizes_mixed_context_lanes_without_contradiction():
    result = _result()
    result["context_outcome"] = {
        "status": "rejected", "events": ["event-parser"],
        "reasons": [{"code": "wrong_representation"}],
    }
    result["transformation_candidates"] = [{
        "transformation_id": "lag-equation", "forecast": result["forecast"],
        "source_seal_sha256": "sealed", "primary_forecast_unchanged": True,
        "lane": "historically_testable", "claim_ids": ["claim-1"],
        "validation": {
            "validation_points": 40, "skill": .9, "beats_baseline": True,
            "recurrence_plausibility_passed": True,
            "recurrence_replay_admitted": True,
            "recurrence_replay_points": 40,
            "recurrence_candidate_mae": .1,
            "recurrence_baseline_mae": 1.0,
        },
    }]
    payload = publish_result(result, mode="best_effort")
    assert payload["recommended_scenario_id"] == "transformation-1"
    assert payload["context_summary"] == {
        "status": "partially_used",
        "summary_is_canonical": True,
        "authoritative_for_publication": True,
        "context_evidence_authority": "historically_admitted",
        "context_changed_human_recommendation": True,
        "context_can_authorize_automation": False,
        "counts": {"used": 1, "scenario": 0, "rejected": 1},
        "follow_up_required_for_current_recommendation": False,
        "additional_evidence_could_change_recommendation": False,
        "further_calls_add_nothing_for_current_recommendation": True,
        "message": (
            "At least one governed context lane affected the human-facing "
            "recommendation; other representations were rejected. See typed "
            "per-lane dispositions."),
    }
    assert verify_publication(payload)
    rejected = next(item for item in payload["context_dispositions"]
                    if item["disposition"] == "rejected")
    assert rejected["recovery_action"][
        "required_for_current_recommendation"] is False
    assert rejected["recovery_action"]["scope"] == \
        "optional_rejected_lane_only"
    contract = scenario_selection_contract(
        scenarios=payload["candidate_portfolio"])
    assert contract["selection_required"] is False
    assert contract["deterministic_scenario_id"] == "transformation-1"


def test_selector_cannot_displace_context_trusted_path_with_weaker_primary():
    result = _result()
    result["support"] = "context_trusted"
    result["primary_forecast"] = [
        {**row, "point": 9.0, "q10": 8, "q50": 9, "q90": 10}
        for row in result["forecast"]]
    result["context_outcome"] = {
        "status": "applied", "admission_basis": "future_context_contract",
        "events": ["event-1"],
    }
    # Force the primary's own path tier below the deterministically validated
    # context path, matching a best-effort primary plus a literal future rule.
    for row in result["primary_forecast"]:
        row["tier"] = "conditionally_supported"
    with pytest.raises(ValueError, match="evidence-dominant path"):
        publish_result(result, mode="best_effort", scenario_selection={
            "selected_scenario_id": "primary",
            "ranking": ["primary", "context_conditioned"],
            "cited_claim_ids": ["event-1"], "counterevidence_claim_ids": [],
            "confidence": .7, "rationale": "prefer history",
            "what_would_change_selection": "more context",
        })
    scenarios, _ = build_scenario_catalog(result)
    contract = scenario_selection_contract(scenarios=scenarios)
    assert contract["selection_required"] is False
    assert contract["deterministic_scenario_id"] == "context_conditioned"


def test_unknown_citations_and_tampering_fail_loudly():
    selection = {
        "selected_scenario_id": "primary",
        "ranking": ["primary", "prior-assisted-1"],
        "cited_claim_ids": ["made-up"], "counterevidence_claim_ids": [],
        "confidence": .5, "rationale": "x",
        "what_would_change_selection": "y",
    }
    with pytest.raises(ValueError, match="unknown claim"):
        publish_result(_result(), mode="scenario", dossiers=[_dossier()],
                       scenario_selection=selection)
    payload = publish_result(_result(), mode="strict")
    damaged = deepcopy(payload)
    damaged["recommended_forecast"][0]["point"] = 999
    assert not verify_publication(damaged)


def test_host_sampled_prior_policy_is_not_mislabeled_as_model_selection():
    dossier = _dossier()
    dossier["forecast_candidate"]["elicitation"] = {
        "kind": "sampled_point_paths", "host_observed": True,
        "requested_paths": 5, "accepted_paths": 5,
        "aggregation": "linear_empirical_marginal_q10_q50_q90",
        "request_mode": "concurrent_single_sample_requests",
        "historical_skill_evidence": False, "automation_eligible": False,
        "stability": _stable_sampling(5),
    }
    import hashlib
    import json
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload = publish_result(_result(), mode="scenario", dossiers=[dossier])
    selection = best_effort_prior_selection(
        scenarios=payload["candidate_portfolio"], dossiers=[dossier])
    assert selection is not None

    selected = select_publication(
        payload, selection,
        selection_channel="best_effort_sampled_prior_policy")

    authority = selected["recommendation_authority"]
    assert authority["selection_method"] == \
        "best_effort_sampled_prior_policy"
    assert authority["selection_pass_performed"] is False
    assert authority["selector_independence"] == "not_applicable"
    assert selected["scenario_selection"]["channel"] == \
        "best_effort_sampled_prior_policy"
    assert verify_publication(selected)


@pytest.mark.parametrize("mutation", [
    "missing_recovery", "dangling_scenario", "false_summary", "wrong_count",
])
def test_resealed_publication_cannot_bypass_context_contract(mutation):
    result = _result()
    result["context_rejections"] = [{
        "context_id": "bad-context", "reason_code": "context_unresolved",
        "reason": "No executable fact was grounded.",
    }]
    payload = publish_result(result, mode="scenario")
    damaged = deepcopy(payload)
    if mutation == "missing_recovery":
        damaged["context_dispositions"][0].pop("recovery_action")
    elif mutation == "dangling_scenario":
        damaged["context_dispositions"][0]["scenario_ids"] = ["missing"]
    elif mutation == "false_summary":
        damaged["context_summary"]["status"] = "used"
    else:
        damaged["scenario_count"] += 1
    import hashlib
    import json
    body = {key: value for key, value in damaged.items()
            if key != "publication_seal_sha256"}
    damaged["publication_seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        default=str).encode()).hexdigest()
    assert not verify_publication(damaged)


def test_prior_assisted_selection_must_receive_and_cite_counter_hypothesis():
    dossier = _dossier()
    dossier["hypotheses"] = [{
        "hypothesis_id": "hyp-counter-1", "kind": "unsupported",
        "claim_ids": ["claim-1"], "direction": "unknown",
        "rationale": "The analogue is from a different season and location.",
        "validation": {"grounded": True, "known_at_cutoff": True,
                       "series_resolved": True},
    }]
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    import hashlib, json
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    scenarios, _ = build_scenario_catalog(_result(), dossiers=[dossier])
    contract = scenario_selection_contract(
        scenarios=scenarios, dossiers=[dossier])
    assert "empty primary claim_ids list as normal" in contract[
        "instruction"]
    assert "never as forecast skill or accuracy" in contract[
        "instruction"]
    exported = next(item for item in contract["claims"]
                    if item["claim_id"] == "hyp-counter-1")
    assert exported["relation"] == "counterevidence"
    assert "different season" in exported["mechanism"]

    selection = {
        "selected_scenario_id": "prior-assisted-1",
        "ranking": ["prior-assisted-1", "primary"],
        "cited_claim_ids": ["claim-1"],
        "counterevidence_claim_ids": [],
        "confidence": .5, "rationale": "the analogue may help",
        "what_would_change_selection": "matched local evidence",
    }
    with pytest.raises(ValueError, match="must cite compiled counterevidence"):
        publish_result(_result(), mode="best_effort", dossiers=[dossier],
                       scenario_selection=selection)
    selection["counterevidence_claim_ids"] = ["hyp-counter-1"]
    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier],
                             scenario_selection=selection)
    assert payload["recommended_scenario_id"] == "prior-assisted-1"
    assert payload["scenario_selection"]["counterevidence_claim_ids"] == []
    assert payload["scenario_selection"][
        "counterevidence_hypothesis_ids"] == ["hyp-counter-1"]
    sealed = publish_result(_result(), mode="best_effort", dossiers=[dossier])
    reranked = select_publication(sealed, selection)
    assert reranked["recommended_scenario_id"] == "prior-assisted-1"


def test_sealed_rerank_preserves_nonrequired_typed_hypothesis_identity():
    dossier = _dossier()
    dossier["hypotheses"] = [{
        "hypothesis_id": "hyp-relationship", "kind": "relationship",
        "claim_ids": ["claim-1"], "direction": "increase",
        "rationale": "A named relationship may apply prospectively.",
        "validation": {"grounded": True, "known_at_cutoff": True,
                       "series_resolved": True},
    }]
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    import hashlib, json
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])
    reranked = select_publication(payload, {
        "selected_scenario_id": "prior-assisted-1",
        "ranking": ["prior-assisted-1", "primary"],
        "cited_claim_ids": ["claim-1"],
        "counterevidence_claim_ids": [],
        "counterevidence_hypothesis_ids": ["hyp-relationship"],
        "confidence": .55,
        "rationale": "Use the sealed prior while acknowledging uncertainty.",
        "what_would_change_selection": "Historical replay rejects the law.",
    })
    assert reranked["recommended_scenario_id"] == "prior-assisted-1"
    assert reranked["scenario_selection"][
        "counterevidence_hypothesis_ids"] == ["hyp-relationship"]


def test_invalid_context_is_typed_rejection_not_silent_drop():
    broken = _dossier()
    broken["compiler_model"] = "tampered"
    payload = publish_result(_result(), mode="scenario", dossiers=[broken])
    assert len(payload["context_dispositions"]) == 1
    rejection = payload["context_dispositions"][0]
    assert rejection["context_id"] == "dossier-1"
    assert rejection["disposition"] == "rejected"
    assert rejection["reason_code"] == "invalid_candidate_seal"
    assert rejection["reason"] == \
        "The dossier seal does not authenticate its body."
    assert rejection["recovery_action"]["code"] == "recompile_from_source"
    assert rejection["recovery_action"]["automation_eligible"] is False


def test_candidate_constraint_failure_is_typed_and_actionable():
    raw = {
        "claims": [{"source_span": "values stay between 4.79 and 9.13",
                    "relation": "constrains_range",
                    "effective_start": TIMES[0], "effective_end": TIMES[1],
                    "confidence": .9}],
        "forecast_candidate": {"quantiles": [
            {"timestamp": timestamp, "q10": 0, "q50": 0, "q90": 0}
            for timestamp in TIMES], "rationale": "bounded path"},
    }
    dossier, _ = validate_temporal_dossier(
        raw, context_text="values stay between 4.79 and 9.13",
        cutoff="2026-01-02T00:00:00+00:00", future_timestamps=TIMES,
        history=[8, 9, 10], compiler_model="test-model")
    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])
    rejection = next(item for item in payload["context_dispositions"]
                     if item["reason_code"] == "forecast_candidate_rejected")
    assert rejection["disposition"] == "rejected"
    assert "violates cited lower bound" in rejection["reason"]
    assert rejection["recovery_action"]
    assert payload["recommended_scenario_id"] == "primary"
    assert verify_publication(payload)


def test_publication_reuses_synthesis_tracking_and_scores_numeric_uplift(tmp_path):
    store = TrackingStore(tmp_path / "tracking.db")
    payload = select_publication(
        publish_result(_result(), mode="best_effort", dossiers=[_dossier()]),
        {
            "selected_scenario_id": "prior-assisted-1",
            "ranking": ["prior-assisted-1", "primary"],
            "cited_claim_ids": ["claim-1"],
            "counterevidence_claim_ids": [], "confidence": .6,
            "rationale": "The cited promotion supports this conditional path.",
            "what_would_change_selection": "Resolved outcomes contradict it.",
        })
    synthesis_id = record_publication(
        store, project="p", forecast_id="f", series="x", payload=payload)
    score = store.resolve_temporal_synthesis(
        project="p", forecast_id="f", series="x", question_id="publication",
        synthesis_id=synthesis_id, outcome={"points": [11.0, 12.0]})
    assert score["rule"] == "numeric_path_wape_v1"
    assert score["synthesis_won"] is True
    assert score["synthesis_delta"] > 0
    summary = store.candidate_outcome_summary("p")
    assert summary[0]["scenario_role"] == "model_authored"
    assert summary[0]["resolved"] == 1
    assert summary[0]["graduated_for_human_prior"] is False
    assert summary[0]["support_upgrade_allowed"] is False


def test_candidate_outcomes_require_repeated_lower_bound_uplift(tmp_path):
    store = TrackingStore(tmp_path / "tracking.db")
    payload = select_publication(
        publish_result(_result(), mode="best_effort", dossiers=[_dossier()]),
        {
            "selected_scenario_id": "prior-assisted-1",
            "ranking": ["prior-assisted-1", "primary"],
            "cited_claim_ids": ["claim-1"],
            "counterevidence_claim_ids": [], "confidence": .6,
            "rationale": "The cited promotion supports this path.",
            "what_would_change_selection": "Resolved outcomes contradict it.",
        })
    for index in range(8):
        forecast_id = f"f-{index}"
        synthesis_id = record_publication(
            store, project="p", forecast_id=forecast_id,
            series="x", payload=payload)
        store.resolve_temporal_synthesis(
            project="p", forecast_id=forecast_id, series="x",
            question_id="publication", synthesis_id=synthesis_id,
            outcome={"points": [11.0, 12.0]})

    summary, = store.candidate_outcome_summary("p")
    assert summary["resolved"] == 8
    assert summary["wins"] == 8
    assert summary["win_rate_wilson_95_lower"] > .5
    assert summary["mean_uplift_vs_primary"] > 0
    assert summary["graduated_for_human_prior"] is True
    assert summary["support_upgrade_allowed"] is False
    assert summary["automation_upgrade_allowed"] is False


def test_candidate_outcome_skill_is_series_local_and_as_of_safe(tmp_path):
    store = TrackingStore(tmp_path / "tracking.db")
    payload = select_publication(
        publish_result(_result(), mode="best_effort", dossiers=[_dossier()]),
        {
            "selected_scenario_id": "prior-assisted-1",
            "ranking": ["prior-assisted-1", "primary"],
            "cited_claim_ids": ["claim-1"],
            "counterevidence_claim_ids": [], "confidence": .6,
            "rationale": "The cited promotion supports this path.",
            "what_would_change_selection": "Resolved outcomes contradict it.",
        })

    def resolved(index, *, series, resolved_at):
        forecast_id = f"{series}-{index}"
        synthesis_id = record_publication(
            store, project="p", forecast_id=forecast_id,
            series=series, payload=payload)
        store.resolve_temporal_synthesis(
            project="p", forecast_id=forecast_id, series=series,
            question_id="publication", synthesis_id=synthesis_id,
            outcome={"points": [11.0, 12.0]}, resolved_at=resolved_at)

    for index in range(8):
        resolved(index, series="target", resolved_at="2026-01-01T00:00:00Z")
    for index in range(8, 10):
        resolved(index, series="target", resolved_at="2026-03-01T00:00:00Z")
    for index in range(3):
        resolved(index, series="unrelated", resolved_at="2026-01-01T00:00:00Z")

    local, = store.candidate_outcome_summary(
        "p", series="target", resolved_before="2026-02-01T00:00:00Z")
    pooled, = store.candidate_outcome_summary(
        "p", resolved_before="2026-02-01T00:00:00Z")
    all_local, = store.candidate_outcome_summary("p", series="target")

    assert local["resolved"] == 8
    assert local["scope"] == {
        "project": "p", "series": "target",
        "resolved_before": "2026-02-01T00:00:00Z"}
    assert pooled["resolved"] == 11
    assert all_local["resolved"] == 10
    with pytest.raises(ValueError, match="timezone-aware"):
        store.candidate_outcome_summary(
            "p", series="target", resolved_before="2026-02-01")


def test_resolved_same_series_candidate_skill_can_guide_only_human_prior():
    evidence = [{
        "scenario_role": "model_authored",
        "candidate_origin": "model_authored",
        "resolved": 12, "wins": 10, "win_rate": 10 / 12,
        "win_rate_wilson_95_lower": .55,
        "mean_uplift_vs_primary": .2,
        "graduated_for_human_prior": True,
        "support_upgrade_allowed": False,
        "automation_upgrade_allowed": False,
        "scope": {
            "project": "p", "series": "target",
            "resolved_before": "2026-02-01T00:00:00Z"},
    }]

    best_effort = publish_result(
        _result(), mode="best_effort", dossiers=[_dossier()],
        candidate_outcome_evidence=evidence)
    strict = publish_result(
        _result(), mode="strict", dossiers=[_dossier()],
        candidate_outcome_evidence=evidence)

    assert best_effort["recommended_scenario_id"] == "prior-assisted-1"
    assert best_effort["recommended_support"] == "prior_assisted"
    assert best_effort["recommendation_authority"]["selection_method"] == \
        "resolved_outcome_human_prior_policy"
    assert best_effort["automation"]["eligible"] is False
    assert best_effort["primary_forecast"] == _result()["forecast"]
    assert best_effort["scenario_selection"]["outcome_skill"] == evidence[0]
    assert verify_publication(best_effort)
    assert strict["recommended_scenario_id"] == "primary"
    assert strict["scenario_selection"] is None


def test_unscoped_candidate_skill_cannot_change_recommendation():
    evidence = [{
        "scenario_role": "model_authored",
        "candidate_origin": "model_authored",
        "resolved": 100, "wins": 100, "win_rate": 1.0,
        "graduated_for_human_prior": True,
        "support_upgrade_allowed": False,
        "automation_upgrade_allowed": False,
        "scope": {"project": "p", "series": None,
                  "resolved_before": None},
    }]
    payload = publish_result(
        _result(), mode="best_effort", dossiers=[_dossier()],
        candidate_outcome_evidence=evidence)

    assert payload["recommended_scenario_id"] == "primary"


def test_mode_invariants_hold_across_varied_bounded_paths():
    for offset in (-2.0, -0.25, 0.0, 0.25, 2.0):
        # Rebuild rather than tamper with the seal.
        raw = {
            "claims": [{"source_span": "promotion begins tomorrow",
                        "relation": "supports_increase",
                        "effective_start": TIMES[0], "effective_end": TIMES[1],
                        "confidence": .6}],
            "forecast_candidate": {"quantiles": [
                {"timestamp": stamp, "q10": 9 + offset + i,
                 "q50": 10 + offset + i, "q90": 11 + offset + i}
                for i, stamp in enumerate(TIMES)]},
        }
        dossier = validate_temporal_dossier(
            raw, context_text="promotion begins tomorrow",
            cutoff="2026-01-02T00:00:00+00:00", future_timestamps=TIMES,
            history=[8, 9, 10], compiler_model="property-test")[0]
        strict = publish_result(_result(), mode="strict", dossiers=[dossier])
        best = publish_result(_result(), mode="best_effort", dossiers=[dossier])
        assert strict["recommended_forecast"] == strict["primary_forecast"]
        assert best["primary_forecast"] == strict["primary_forecast"]
        assert best["recommended_support"] == "supported"
        assert any(item["support"] == "prior_assisted"
                   for item in best["candidate_portfolio"])
        assert best["automation"]["eligible"] is False
        assert verify_publication(strict) and verify_publication(best)


def test_mcp_forecast_persists_verified_sidecar_without_mutating_artifact(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{100 + i}" for i in range(40)) + "\n")
    future = [f"2026-02-{day:02d}T00:00:00+00:00" for day in (10, 11)]
    dossier = validate_temporal_dossier({
        "claims": [{"source_span": "promotion begins tomorrow",
                    "relation": "supports_increase",
                    "effective_start": future[0], "effective_end": future[-1],
                    "confidence": .7}],
        "forecast_candidate": {"quantiles": [
            {"timestamp": future[0], "q10": 139, "q50": 140, "q90": 141},
            {"timestamp": future[1], "q10": 140, "q50": 141, "q90": 142},
        ]}}, context_text="promotion begins tomorrow",
        cutoff="2026-02-09T00:00:00+00:00", future_timestamps=future,
        history=list(range(100, 140)), compiler_model="test")[0]
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 2,
        "output_dir": str(tmp_path / "out"),
        "publication_mode": "best_effort", "temporal_dossiers": [dossier],
    })
    assert payload["publication"]["recommended_scenario_id"] == "primary"
    assert payload["publication"]["projection"] == "compact"
    assert "recommended_forecast" not in payload["publication"]
    assert "candidate_portfolio" not in payload["publication"]
    assert payload["publication"]["selection_contract"]
    assert payload["publication"]["receipt_is_complete_and_sealed"] is True
    receipt = __import__("json").loads(
        Path(payload["publication_path"]).read_text(encoding="utf-8"))
    assert verify_publication(receipt)
    assert Path(payload["publication_path"]).is_file()
    assert verify_artifact_integrity(payload["artifact_path"])


def test_mcp_best_effort_uses_only_prior_same_series_candidate_outcomes(
        tmp_path, monkeypatch):
    from datetime import date, timedelta
    registry = tmp_path / "registry.db"
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(registry))
    store = TrackingStore(registry)
    historical = select_publication(
        publish_result(_result(), mode="best_effort", dossiers=[_dossier()]),
        {
            "selected_scenario_id": "prior-assisted-1",
            "ranking": ["prior-assisted-1", "primary"],
            "cited_claim_ids": ["claim-1"],
            "counterevidence_claim_ids": [], "confidence": .6,
            "rationale": "The cited promotion supports this path.",
            "what_would_change_selection": "Resolved outcomes contradict it.",
        })
    for index in range(8):
        forecast_id = f"historical-{index}"
        synthesis_id = record_publication(
            store, project="p", forecast_id=forecast_id,
            series="__default__", payload=historical)
        store.resolve_temporal_synthesis(
            project="p", forecast_id=forecast_id, series="__default__",
            question_id="publication", synthesis_id=synthesis_id,
            outcome={"points": [11.0, 12.0]},
            resolved_at="2026-01-01T00:00:00Z")

    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{100 + i}" for i in range(40)) + "\n")
    future = [f"2026-02-{day:02d}T00:00:00+00:00" for day in (10, 11)]
    dossier = validate_temporal_dossier({
        "claims": [{"source_span": "promotion begins tomorrow",
                    "relation": "supports_increase",
                    "effective_start": future[0], "effective_end": future[-1],
                    "confidence": .7}],
        "forecast_candidate": {"quantiles": [
            {"timestamp": future[0], "q10": 139, "q50": 140, "q90": 141},
            {"timestamp": future[1], "q10": 140, "q50": 141, "q90": 142},
        ]}}, context_text="promotion begins tomorrow",
        cutoff="2026-02-09T00:00:00+00:00", future_timestamps=future,
        history=list(range(100, 140)), compiler_model="test")[0]

    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 2, "format": "full",
        "output_dir": str(tmp_path / "out"), "project": "p",
        "publication_mode": "best_effort", "temporal_dossiers": [dossier],
    })

    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "prior-assisted-1"
    assert publication["recommendation_authority"]["selection_method"] == \
        "resolved_outcome_human_prior_policy"
    assert publication["candidate_outcome_evidence"][0]["scope"]["series"] == \
        "__default__"
    assert publication["automation"]["eligible"] is False
    assert publication["primary_forecast_unchanged"] is True
    assert verify_publication(publication)


def test_full_format_explicitly_returns_complete_signed_publication(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{100 + i}" for i in range(40)) + "\n")
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 12, "format": "full",
        "output_dir": str(tmp_path / "out"), "publication_mode": "scenario",
    })
    assert payload["publication"]["candidate_portfolio"]
    assert payload["publication"]["recommended_forecast"]
    assert "projection" not in payload["publication"]
    assert verify_publication(payload["publication"])


def test_scenario_overflow_is_bounded_with_typed_dispositions():
    result = _result()
    result["sensitivity_scenarios"] = [{
        "forecast": [{**row, "q50": 10 + index, "point": 10 + index}
                     for row in result["forecast"]],
        "support": "hypothetical_sensitivity", "assumptions": [f"case {index}"],
    } for index in range(12)]
    payload = publish_result(result, mode="scenario", dossiers=[_dossier()])
    assert len(payload["candidate_portfolio"]) == 8
    assert payload["candidate_portfolio"][0]["scenario_id"] == "primary"
    assert any(item["reason_code"] == "bounded_portfolio_overflow"
               for item in payload["context_dispositions"])
    assert verify_publication(payload)


def test_mcp_context_transformation_rejection_is_typed_and_primary_is_intact(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{100 + i}" for i in range(40)) + "\n")
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 2,
        "output_dir": str(tmp_path / "out"),
        "format": "full",
        "publication_mode": "scenario",
        "context_submission": {
            "known_at": "2026-02-09T00:00:00+00:00",
            "transformations": [{
                "known_at": "2026-02-09T00:00:00+00:00",
                "claim_ids": ["invented"], "lane": "prior_assisted",
                "expression": {"op": "python", "code": "raise SystemExit"},
            }],
        },
    })
    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "primary"
    assert publication["primary_forecast"] == publication["recommended_forecast"]
    rejection = next(item for item in publication["context_dispositions"]
                     if item["reason_code"] == "transformation_validation_failed")
    assert rejection["violations"][0]["code"] == "UNVERIFIED_CLAIMS"
    assert publication["automation"]["eligible"] is False
    assert verify_publication(publication)


def test_mcp_model_candidate_is_grid_bound_sealed_and_human_only(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{100 + i}" for i in range(40)) + "\n")
    context = "The signed sales plan expects a temporary campaign uplift."
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 2,
        "output_dir": str(tmp_path / "out-model-candidate"),
        "format": "full", "publication_mode": "best_effort",
        "context_submission": {
            "text": context, "known_at": "2026-02-09T00:00:00+00:00",
            "compiler": "host-agent",
            "model_candidate": {
                "source_spans": [context],
                "sample_paths": [[141, 142], [142, 143], [143, 144]],
                "rationale": "Conditional campaign trajectory.",
                "temperature": 0.7,
            },
        },
    })

    publication = payload["publication"]
    candidate = next(item for item in publication["candidate_portfolio"]
                     if item["role"] == "model_authored")
    assert publication["recommended_scenario_id"] == candidate["scenario_id"]
    assert candidate["support"] == "prior_assisted"
    assert candidate["human_selection_eligible"] is True
    assert candidate["automation_eligible"] is False
    assert candidate["effect"]["distribution"]["sample_count"] == 3
    assert publication["primary_forecast_unchanged"] is True
    assert publication["automation"]["eligible"] is False
    assert verify_publication(publication)


def test_dossier_candidate_plausibility_uses_observed_not_future_boundary():
    from gnomon.publication import compile_dossier_for_result

    context = "The signed plan expects a temporary uplift."
    result = {
        "forecast": [
            {"timestamp": TIMES[0], "q10": 990, "q50": 1000, "q90": 1010},
            {"timestamp": TIMES[1], "q10": 991, "q50": 1001, "q90": 1011},
        ]
    }
    raw = {
        "events": [],
        "claims": [{"source_span": context, "relation": "unknown",
                    "effective_start": None, "effective_end": None,
                    "timing_status": "atemporal_context",
                    "mechanism": "model-authored forecast prior",
                    "confidence": 0.5}],
        "hypotheses": [], "effect_proposal": None,
        "forecast_candidate": {
            "quantiles": [
                {"timestamp": TIMES[0], "q10": 100, "q50": 101, "q90": 102},
                {"timestamp": TIMES[1], "q10": 101, "q50": 102, "q90": 103},
            ], "claim_ids": ["claim-1"], "rationale": "Plan trajectory."},
        "covariate_tables": [], "transformations": [],
        "observation_interpretations": [],
    }

    dossier, rejections = compile_dossier_for_result(
        raw, context_text=context, known_at="2026-01-01T00:00:00Z",
        result=result, compiler_model="test", history=[98, 99, 100])

    assert dossier.get("forecast_candidate") is not None
    assert "forecast_candidate failed boundary-jump plausibility" not in rejections


def test_implausible_optional_model_candidate_preserves_primary(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{100 + i}" for i in range(40)) + "\n")
    context = "A qualitative note expects a change."
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 2,
        "output_dir": str(tmp_path / "out-rejected-candidate"),
        "format": "full", "publication_mode": "best_effort",
        "context_submission": {
            "text": context, "known_at": "2026-02-09T00:00:00+00:00",
            "compiler": "host-agent",
            "model_candidate": {
                "source_spans": [context],
                "sample_paths": [[10000, 10001], [10001, 10002], [9999, 10000]],
                "rationale": "Unbounded trajectory.",
            },
        },
    })

    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "primary"
    assert publication["recommended_forecast"] == publication["primary_forecast"]
    rejection = next(item for item in publication["context_dispositions"]
                     if item.get("reason_code") ==
                     "model_candidate_validation_failed")
    assert rejection["disposition"] == "rejected"
    assert publication["automation"]["eligible"] is False
    assert verify_publication(publication)


def test_mcp_model_candidate_rejects_uncited_or_wrong_grid_paths(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{100 + i}" for i in range(40)) + "\n")
    base = {
        "input": str(source), "horizon": 2,
        "output_dir": str(tmp_path / "out-invalid-candidate"),
        "publication_mode": "best_effort",
        "context_submission": {
            "text": "The plan expects an uplift.",
            "known_at": "2026-02-09T00:00:00+00:00",
            "model_candidate": {
                "source_spans": ["invented quote"],
                "sample_paths": [[141, 142], [142, 143], [143, 144]],
            },
        },
    }
    with pytest.raises(Exception, match="exact source_spans"):
        runner_for("gnomon_forecast")(base)
    base["context_submission"]["model_candidate"]["source_spans"] = [
        "The plan expects an uplift."]
    base["context_submission"]["model_candidate"]["sample_paths"] = [
        [141], [142], [143]]
    with pytest.raises(Exception, match="host-grid-bound"):
        runner_for("gnomon_forecast")(base)


def test_mcp_compiler_rejection_is_visible_in_publication(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{100 + i}" for i in range(40)) + "\n")
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 2,
        "output_dir": str(tmp_path / "out"),
        "format": "full",
        "publication_mode": "best_effort",
        "context_submission": {
            "known_at": "2026-02-09T00:00:00+00:00",
            "rejections": [
                "context_unresolved: no grounded numeric relationship was found"],
        },
    })
    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "primary"
    rejection = next(item for item in publication["context_dispositions"]
                     if item["reason_code"] == "context_unresolved")
    assert rejection["disposition"] == "rejected"
    assert rejection["reason"] == "no grounded numeric relationship was found"
    assert rejection["recovery_action"]["code"] == "provide_grounded_context"
    assert "effective dates" in rejection["recovery_action"]["required_evidence"]
    assert publication["primary_forecast_unchanged"] is True
    assert verify_publication(publication)


def test_every_rejected_context_disposition_has_bounded_recovery():
    result = _result()
    result["transformation_rejections"] = [{
        "transformation_id": "bad-transform",
        "reason_code": "transformation_validation_failed",
        "reason": "unknown series",
    }]
    result["context_rejections"] = [{
        "context_id": "unresolved-document",
        "reason_code": "context_unresolved",
        "reason": "no executable facts",
    }]
    payload = publish_result(result, mode="scenario", dossiers=[{
        **_dossier(), "compiler_model": "tampered"}])
    rejected = [item for item in payload["context_dispositions"]
                if item["disposition"] == "rejected"]
    assert len(rejected) == 3
    for item in rejected:
        action = item["recovery_action"]
        assert action["code"]
        assert action["message"]
        assert action["required_evidence"]
        assert action["automation_eligible"] is False
    assert verify_publication(payload)


def test_duplicate_transformation_preflight_summary_is_not_second_rejection():
    result = _result()
    result["transformation_rejections"] = [{
        "transformation_id": "bad-transform",
        "reason_code": "HORIZON_MISMATCH",
        "reason": "Every future series must match the primary horizon.",
        "violations": [{"code": "HORIZON_MISMATCH",
                        "field": "series_values"}],
    }]
    result["context_rejections"] = [{
        "context_id": "context-submission-1",
        "reason_code": "transformation_preflight_rejected",
        "reason": ('[{"index":1,"violations":[{"code":'
                   '"HORIZON_MISMATCH","field":"series_values"}]}]'),
    }]

    payload = publish_result(result, mode="scenario")

    assert payload["context_summary"]["counts"]["rejected"] == 1
    duplicate = next(item for item in payload["context_dispositions"]
                     if item["context_id"] == "context-submission-1")
    assert duplicate["disposition"] == "superseded"
    assert duplicate["reason_code"] == \
        "duplicate_transformation_preflight_summary"
    assert duplicate["represented_violation_codes"] == ["HORIZON_MISMATCH"]
    assert verify_publication(payload)


def test_typed_wildcard_rejection_teaches_target_binding(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{100 + i}" for i in range(40)) + "\n")
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 2,
        "output_dir": str(tmp_path / "out-typed"), "format": "full",
        "publication_mode": "best_effort",
        "context_submission": {"rejections": [{
            "context_id": "event-proposal-1",
            "reason_code": "unsafe_wildcard_numeric_event",
            "reason": "numeric event did not name the active target",
        }]},
    })

    rejection = next(item for item in payload["publication"][
        "context_dispositions"] if item["context_id"] == "event-proposal-1")
    assert rejection["reason_code"] == "unsafe_wildcard_numeric_event"
    assert rejection["recovery_action"]["code"] == \
        "bind_numeric_event_target"
    assert rejection["recovery_action"]["automation_eligible"] is False


def test_rejected_candidate_normalizes_legacy_string_recovery():
    dossier = _dossier()
    dossier["forecast_candidate"] = None
    dossier["candidate_critique"] = {
        "status": "rejected", "reasons": ["candidate is incomplete"],
        "recovery_action": "try again",
    }
    import hashlib
    import json
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        default=str).encode()).hexdigest()
    payload = publish_result(_result(), mode="scenario", dossiers=[dossier])
    rejected = next(item for item in payload["context_dispositions"]
                    if item["reason_code"] == "forecast_candidate_rejected")
    assert rejected["recovery_action"]["code"] == "correct_context_proposal"
    assert rejected["recovery_action"]["automation_eligible"] is False


def test_mcp_recursive_transformation_binds_history_but_requires_replay_skill(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "wide.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,campaign,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{i + 1},{100 + i}" for i in range(40))
        + "\n")
    formula = "value[t] = 0.5 value[t-1] + 2 campaign[t-1]"
    schedule = "future campaign values are 41 then 42"
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "target_column": "value", "horizon": 2,
        "output_dir": str(tmp_path / "out"),
        "format": "full",
        "publication_mode": "best_effort",
        "context_submission": {
            "text": formula + ". " + schedule,
            "known_at": "2026-02-09T00:00:00+00:00",
            "compiler": "test-agent",
            "proposal": {"claims": [
                {"source_span": formula, "relation": "supports_increase",
                 "effective_start": "2026-02-10T00:00:00+00:00",
                 "effective_end": "2026-02-11T00:00:00+00:00"},
                {"source_span": schedule, "relation": "supports_increase",
                 "effective_start": "2026-02-10T00:00:00+00:00",
                 "effective_end": "2026-02-11T00:00:00+00:00"},
            ]},
            "transformations": [{
                "transformation": {
                    "known_at": "2026-02-09T00:00:00+00:00",
                    "claim_ids": ["claim-1", "claim-2"],
                    "lane": "prior_assisted", "output_unit": "value",
                    "expression": {
                        "op": "recursive_linear", "output_unit": "value",
                        "autoregressive_terms": [{"lag": 1, "coefficient": .5}],
                        "driver_terms": [{"series": "campaign", "lag": 1,
                                          "coefficient": 2}],
                    },
                },
                "units": {"primary": "value", "campaign": "campaign"},
                "series_values": {"campaign": {
                    "values": [41, 42],
                    "known_at": "2026-02-09T00:00:00+00:00",
                    "source_claim_ids": ["claim-2"]}},
            }],
        },
    })
    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "primary"
    scenario = next(item for item in publication["candidate_portfolio"]
                    if item["scenario_id"] == "transformation-1")
    assert [row["q50"] for row in scenario["forecast"]] == [149.5, 156.75]
    assert scenario["selection_eligible"] is False
    assert scenario["effect"]["validation"][
        "recurrence_replay_reason"] == "did_not_beat_last_value"
    assert publication["primary_forecast_unchanged"] is True
    assert publication["automation"]["eligible"] is False
    assert verify_publication(publication)


def test_mcp_fits_source_stated_lag_structure_without_model_coefficients(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "fitted-relationship.csv"
    start = date(2026, 1, 1)
    driver = [float((index * 5) % 13) for index in range(90)]
    target = [4.0, 5.0]
    for index in range(2, 90):
        target.append(2 + .25 * target[index - 1] + 1.5 * driver[index - 1])
    source.write_text("timestamp,driver,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{driver[i]},{target[i]}"
        for i in range(90)) + "\n")
    relationship = "driver and value are parents of value at lag 1"
    schedule = "future driver values are 3 then 8"
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "target_column": "value", "horizon": 2,
        "output_dir": str(tmp_path / "out"), "format": "full",
        "publication_mode": "best_effort",
        "context_submission": {
            "text": relationship + ". " + schedule,
            "known_at": "2026-03-31T00:00:00+00:00", "compiler": "test",
            "proposal": {"claims": [
                {"source_span": relationship, "relation": "unknown",
                 "effective_start": "2026-04-01T00:00:00+00:00",
                 "effective_end": "2026-04-02T00:00:00+00:00"},
                {"source_span": schedule, "relation": "unknown",
                 "effective_start": "2026-04-01T00:00:00+00:00",
                 "effective_end": "2026-04-02T00:00:00+00:00"}]},
            "transformations": [{
                "transformation": {
                    "known_at": "2026-03-31T00:00:00+00:00",
                    "claim_ids": ["claim-1", "claim-2"],
                    "lane": "historically_testable", "output_unit": "value",
                    "expression": {"op": "fit_recursive_linear",
                        "output_unit": "value", "autoregressive_lags": [1],
                        "driver_lags": [{"series": "driver", "lags": [1]}]}},
                "units": {"primary": "value", "driver": "driver"},
                "series_values": {"driver": {"values": [3, 8],
                    "known_at": "2026-03-31T00:00:00+00:00",
                    "source_claim_ids": ["claim-2"]}},
            }],
        },
    })
    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "transformation-1"
    scenario = next(item for item in publication["candidate_portfolio"]
                    if item["scenario_id"] == "transformation-1")
    validation = scenario["effect"]["validation"]
    assert validation["beats_baseline"] is True
    assert validation["specification_known_at_each_origin"] is False
    assert publication["recommendation_authority"]["selected_role"] == \
        "retrospectively_validated"
    assert publication["recommendation_authority"][
        "retrospectively_validated"] is True
    assert publication["context_summary"]["context_evidence_authority"] == \
        "retrospectively_validated"
    assert publication["context_summary"][
        "authoritative_for_publication"] is False
    assert publication["primary_forecast_unchanged"] is True
    assert publication["automation"]["eligible"] is False
    assert not any(item["context_id"] in {
        "dossier-1:claim-1", "dossier-1:claim-2"}
        for item in publication["context_dispositions"])
    assert verify_publication(publication)


def test_mcp_deterministically_compiles_complete_relationship_text(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "deterministic-relationship.csv"
    start = date(2026, 1, 1)
    driver = [1.0] * 30 + [4.0] * 30 + [2.0] * 30
    target = [3.0]
    for index in range(1, 90):
        target.append(.5 * target[index - 1] + 2 * driver[index - 1])
    source.write_text("timestamp,driver,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{driver[i]},{target[i]}"
        for i in range(90)) + "\n")
    context = (
        "driver is 1 from 2026-01-01 to 2026-01-30, 4 from 2026-01-31 "
        "to 2026-03-01, 2 from 2026-03-02 to 2026-03-31, and 5 from "
        "2026-04-01 to 2026-04-02.\n"
        "value^{t} = 0.5 * value^{t-1} + 2 * driver^{t-1}")
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "target_column": "value", "horizon": 2,
        "output_dir": str(tmp_path / "out"), "format": "full",
        "publication_mode": "best_effort",
        "context_submission": {
            "text": context, "known_at": "2026-03-31T00:00:00+00:00",
            "compile": "deterministic_linear",
        },
    })
    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "transformation-1"
    assert publication["primary_forecast_unchanged"] is True
    assert publication["recommendation_authority"]["selected_role"] == \
        "retrospectively_validated"
    assert publication["automation"]["eligible"] is False
    assert not any(item["disposition"] == "rejected"
                   for item in publication["context_dispositions"])
    assert verify_publication(publication)


def test_mcp_deterministic_relationship_refuses_partial_text(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "partial-relationship.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,driver,value\n" + "\n".join(
        f"{start + timedelta(days=i)},1,{100 + i}" for i in range(40)) + "\n")
    context = (
        "driver is 1 from 2026-02-10 to 2026-02-11.\n"
        "value^{t} = 0.5 * value^{t-1} + an unspecified driver effect")
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "target_column": "value", "horizon": 2,
        "output_dir": str(tmp_path / "out"), "format": "full",
        "publication_mode": "best_effort",
        "context_submission": {
            "text": context, "known_at": "2026-02-09T00:00:00+00:00",
            "compile": "deterministic_linear",
        },
    })
    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "primary"
    rejection = next(item for item in publication["context_dispositions"]
                     if item["reason_code"] ==
                     "DETERMINISTIC_RELATIONSHIP_UNRESOLVED")
    assert "No partial arithmetic" in rejection["reason"]
    assert publication["primary_forecast_unchanged"] is True
    assert publication["automation"]["eligible"] is False


def test_missing_recursive_driver_rejects_scenario_not_primary(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{100 + i}" for i in range(40)) + "\n")
    formula = "value[t] = 2 missing_driver[t-1]"
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "target_column": "value", "horizon": 2,
        "output_dir": str(tmp_path / "out"), "publication_mode": "scenario",
        "context_submission": {
            "text": formula, "known_at": "2026-02-09T00:00:00+00:00",
            "compiler": "test", "proposal": {"claims": [{
                "source_span": formula, "relation": "supports_increase",
                "effective_start": "2026-02-10T00:00:00+00:00",
                "effective_end": "2026-02-11T00:00:00+00:00"}]},
            "transformations": [{
                "transformation": {
                    "known_at": "2026-02-09T00:00:00+00:00",
                    "claim_ids": ["claim-1"], "lane": "prior_assisted",
                    "output_unit": "value", "expression": {
                        "op": "recursive_linear", "output_unit": "value",
                        "autoregressive_terms": [],
                        "driver_terms": [{"series": "missing_driver", "lag": 1,
                                          "coefficient": 2}]},
                },
                "units": {"primary": "value", "missing_driver": "driver"},
                "series_values": {"missing_driver": {
                    "values": [1, 1],
                    "known_at": "2026-02-09T00:00:00+00:00",
                    "source_claim_ids": ["claim-1"]}},
            }],
        },
    })
    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "primary"
    assert publication["primary_forecast_unchanged"] is True
    rejection = next(item for item in publication["context_dispositions"]
                     if item["reason_code"] == "MISSING_COLUMNS")
    assert "missing_driver" in rejection["reason"]


def test_documented_history_can_replay_encoded_driver_without_guessing_scale(
        tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "encoded.csv"
    start = date(2026, 1, 1)
    target = [10.0]
    for _ in range(1, 40):
        target.append(.5 * target[-1] + 2 * 2.0)
    source.write_text("timestamp,campaign,value\n" + "\n".join(
        f"{start + timedelta(days=i)},999,{target[i]}" for i in range(40))
        + "\n")
    formula = "value[t] = 0.5 value[t-1] + 2 campaign[t-1]"
    history = "campaign is 2 from 2026-01-01 to 2026-02-09"
    schedule = "future campaign values are 2 then 2"
    claims = [
        {"source_span": text, "relation": "unknown",
         "effective_start": "2026-02-10T00:00:00+00:00",
         "effective_end": "2026-02-11T00:00:00+00:00"}
        for text in (formula, history, schedule)]
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "target_column": "value", "horizon": 2,
        "output_dir": str(tmp_path / "out"),
        "format": "full",
        "publication_mode": "best_effort",
        "context_submission": {
            "text": ". ".join((formula, history, schedule)),
            "known_at": "2026-02-09T00:00:00+00:00",
            "compiler": "test", "proposal": {"claims": claims},
            "transformations": [{
                "transformation": {
                    "known_at": "2026-02-09T00:00:00+00:00",
                    "claim_ids": ["claim-1", "claim-2", "claim-3"],
                    "lane": "historically_testable", "output_unit": "value",
                    "expression": {"op": "recursive_linear",
                        "output_unit": "value", "intercept": 0,
                        "autoregressive_terms": [{"lag": 1, "coefficient": .5}],
                        "driver_terms": [{"series": "campaign", "lag": 1,
                                          "coefficient": 2}]},
                },
                "units": {"primary": "value", "campaign": "campaign"},
                "series_values": {"campaign": {"values": [2, 2],
                    "known_at": "2026-02-09T00:00:00+00:00",
                    "source_claim_ids": ["claim-3"]}},
                "historical_series_segments": {"campaign": [{
                    "start": "2026-01-01", "end": "2026-02-09", "value": 2,
                    "source_claim_ids": ["claim-2"]}]},
            }],
        },
    })
    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "transformation-1"
    scenario = next(item for item in publication["candidate_portfolio"]
                    if item["scenario_id"] == "transformation-1")
    validation = scenario["effect"]["validation"]
    assert validation["recurrence_replay_admitted"] is True
    assert validation["recurrence_replay_candidate_mae"] == pytest.approx(0)
    assert validation["recurrence_history_source"] == "document_cited_segments"
    assert publication["automation"]["eligible"] is False


def test_response_budget_never_breaks_a_publication_seal():
    result = _result()
    result["forecast"] = [
        {"timestamp": f"2026-01-{day:02d}T00:00:00+00:00",
         "point": float(day), "q10": day - 1, "q50": day, "q90": day + 1}
        for day in range(3, 28)]
    publication = publish_result(result, mode="strict")
    trimmed = enforce_response_budget({
        "artifact_path": "/tmp/artifact", "publication": publication,
        "bulk": list(range(1000)),
    }, budget_bytes=1000)
    assert trimmed["truncated"] is True
    assert trimmed["publication"] == publication
    assert verify_publication(trimmed["publication"])
