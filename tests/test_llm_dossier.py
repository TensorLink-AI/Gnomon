from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import pytest

from gnomon.llm_dossier import (
    attach_host_candidate_elicitation,
    deterministic_associational_claims,
    deterministic_explicit_confounding_claims,
    deterministic_historical_observation_claim,
    deterministic_quantitative_background_claims,
    validate_temporal_dossier,
    verify_temporal_dossier_seal,
)


def test_deterministic_quantitative_background_claims_copy_only_statistics():
    text = (
        "Demand averages 85 incidents per year. "
        "A launch in 2027 will eliminate demand. "
        "The busiest month historically recorded 19 incidents.")

    claims = deterministic_quantitative_background_claims(text)

    assert [claim["source_span"] for claim in claims] == [
        "Demand averages 85 incidents per year.",
        "The busiest month historically recorded 19 incidents."]
    assert all(claim["timing_status"] == "atemporal_context"
               for claim in claims)


def test_deterministic_explicit_confounding_is_negative_authority_only():
    text = (
        "Trash fires and field fires tend to co-occur. "
        "This association does not imply causation and is likely due to a "
        "common cause. The city will spray trash piles next month.")

    claims = deterministic_explicit_confounding_claims(text)

    assert [item["source_span"] for item in claims] == [
        "Trash fires and field fires tend to co-occur.",
        "This association does not imply causation and is likely due to a common cause.",
    ]
    assert all(item["relation"] == "unknown" for item in claims)
    assert all(item["timing_status"] == "atemporal_context" for item in claims)
    assert deterministic_explicit_confounding_claims(
        "The campaign caused demand to rise.") == []


def test_deterministic_association_needs_no_disclaimer_and_never_infers_cause():
    claims = deterministic_associational_claims(
        "Trash fires and field fires tend to co-occur. "
        "When field fires increase, trash fires tend to increase. "
        "The collection campaign caused complaints to fall.")

    assert [item["source_span"] for item in claims] == [
        "Trash fires and field fires tend to co-occur.",
        "When field fires increase, trash fires tend to increase."]
    assert all(item["relation"] == "unknown" for item in claims)
    assert deterministic_associational_claims(
        "The campaign caused demand to rise.") == []

    dossier, reasons = validate_temporal_dossier(
        {"claims": claims}, context_text=(
            "Trash fires and field fires tend to co-occur. "
            "When field fires increase, trash fires tend to increase. "
            "The collection campaign caused complaints to fall."),
        cutoff="2026-01-01T00:00:00+00:00",
        future_timestamps=["2026-01-02T00:00:00+00:00"],
        history=[1.0, 2.0], compiler_model="deterministic")
    assert reasons == []
    assert all(item["relationship_authority"] == "associational_only"
               for item in dossier["claims"])
    assert all(item["causal_authority"] is False
               for item in dossier["claims"])


def test_only_host_can_attach_bounded_sampling_provenance():
    span = "The promotion may increase demand."
    raw = {
        "claims": [{"source_span": span, "relation": "supports_increase",
                    "effective_start": "2026-01-02T00:00:00+00:00",
                    "effective_end": "2026-01-02T00:00:00+00:00",
                    "confidence": 1}],
        "forecast_candidate": {"quantiles": [
            {"timestamp": "2026-01-02T00:00:00+00:00",
             "q10": 9, "q50": 10, "q90": 11}],
             "rationale": "bounded prior"},
    }
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff="2026-01-01T00:00:00+00:00",
        future_timestamps=["2026-01-02T00:00:00+00:00"],
        history=[9.0, 10.0], compiler_model="test")
    assert not reasons
    updated = attach_host_candidate_elicitation(
        dossier, requested_paths=5, accepted_paths=4,
        aggregation="linear_empirical_marginal_q10_q50_q90",
        temperature=1, sample_paths=[[10.0] for _ in range(4)])
    assert verify_temporal_dossier_seal(updated)
    assert updated["forecast_candidate"]["elicitation"] == {
        "kind": "sampled_point_paths", "requested_paths": 5,
        "accepted_paths": 4,
        "aggregation": "linear_empirical_marginal_q10_q50_q90",
        "temperature": 1.0, "request_mode": "batch_request",
        "host_observed": True,
        "historical_skill_evidence": False, "automation_eligible": False,
    }
    assert dossier["forecast_candidate"].get("elicitation") is None
    assert updated["forecast_candidate"]["sample_paths"] == [[10.0]] * 4
    with pytest.raises(ValueError, match="path counts"):
        attach_host_candidate_elicitation(
            dossier, requested_paths=3, accepted_paths=4,
            aggregation="linear_empirical_marginal_q10_q50_q90",
            temperature=1)
    with pytest.raises(ValueError, match="horizon"):
        attach_host_candidate_elicitation(
            dossier, requested_paths=2, accepted_paths=2,
            aggregation="linear_empirical_marginal_q10_q50_q90",
            temperature=1, sample_paths=[[1.0], [1.0, 2.0]])


def test_candidate_preserves_exact_cited_claim_subset():
    safe = "Demand averages 10 units per day."
    unresolved = "A future launch will immediately eliminate demand."
    raw = {
        "claims": [
            {"source_span": safe, "relation": "supports_stability",
             "effective_start": None, "effective_end": None,
             "timing_status": "atemporal_context", "confidence": .8},
            {"source_span": unresolved, "relation": "supports_decrease",
             "effective_start": None, "effective_end": None,
             "timing_status": "unresolved_trigger", "confidence": .2},
        ],
        "forecast_candidate": {
            "quantiles": [{"timestamp": "2026-01-02T00:00:00+00:00",
                           "q10": 9, "q50": 10, "q90": 11}],
            "claim_ids": ["claim-1"], "rationale": "safe numeric prior"},
    }

    dossier, reasons = validate_temporal_dossier(
        raw, context_text=f"{safe} {unresolved}",
        cutoff="2026-01-01T00:00:00+00:00",
        future_timestamps=["2026-01-02T00:00:00+00:00"],
        history=[9.0, 10.0], compiler_model="test")

    assert dossier["forecast_candidate"]["claim_ids"] == ["claim-1"]
    assert not any("unresolved trigger" in reason for reason in reasons)


def test_host_sampling_stability_is_sealed_but_cannot_upgrade_support():
    span = "The promotion may increase demand."
    raw = {
        "claims": [{"source_span": span, "relation": "supports_increase",
                    "effective_start": "2026-01-02T00:00:00+00:00",
                    "effective_end": "2026-01-02T00:00:00+00:00",
                    "confidence": 1}],
        "forecast_candidate": {"quantiles": [
            {"timestamp": "2026-01-02T00:00:00+00:00",
             "q10": 9, "q50": 10, "q90": 11}],
             "rationale": "bounded prior"},
    }
    dossier, _ = validate_temporal_dossier(
        raw, context_text=span, cutoff="2026-01-01T00:00:00+00:00",
        future_timestamps=["2026-01-02T00:00:00+00:00"],
        history=[9.0, 10.0], compiler_model="test")
    stability = {
        "version": "0.1", "interpretation": "stability_not_historical_skill",
        "scale_basis": "median_nonzero_history_increment", "path_count": 5,
        "horizon": 1, "median_pointwise_q80_width_scaled": .2,
        "p90_pointwise_q80_width_scaled": .2,
        "median_pairwise_mae_scaled": .1, "max_pairwise_mae_scaled": .3,
        "mean_direction_agreement": 1.0,
        "unanimous_direction_fraction": 1.0,
    }
    updated = attach_host_candidate_elicitation(
        dossier, requested_paths=5, accepted_paths=5,
        aggregation="linear_empirical_marginal_q10_q50_q90",
        temperature=1, stability=stability)
    assert verify_temporal_dossier_seal(updated)
    assert updated["forecast_candidate"]["elicitation"]["stability"] == stability
    assert updated["candidate_support"] == "prior_assisted"
    assert updated["automation_eligible"] is False
    malformed = {**stability, "mean_direction_agreement": 1.1}
    with pytest.raises(ValueError, match="agreement"):
        attach_host_candidate_elicitation(
            dossier, requested_paths=5, accepted_paths=5,
            aggregation="linear_empirical_marginal_q10_q50_q90",
            temperature=1, stability=malformed)


def test_replay_positive_observation_executable_outranks_model_proposal():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timestamps = [(start + timedelta(hours=index)).isoformat()
                  for index in range(24 * 21)]
    history = []
    for index in range(len(timestamps)):
        phase = index % 24
        history.append(0.0 if phase in {20, 21, 22, 23} else
                       30.0 + 8.0 * math.sin(2 * math.pi * phase / 24))
    future = [(start + timedelta(hours=len(history) + index)).isoformat()
              for index in range(24)]
    span = ("The sensor was offline for maintenance every day between 20:00 "
            "and 00:00, which resulted in zero readings.")
    raw = {
        "claims": [{"source_span": span, "relation": "unknown",
                    "effective_start": timestamps[0],
                    "effective_end": timestamps[-1], "confidence": 1}],
        "forecast_candidate": {
            "constant_quantiles": {"q10": 5, "q50": 6, "q90": 7},
            "rationale": "model proposal"},
    }

    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span + " Assume maintenance will not recur.",
        cutoff=timestamps[-1], future_timestamps=future, history=history,
        history_timestamps=timestamps, compiler_model="test")

    assert not reasons
    assert dossier["candidate_critique"]["candidate_origin"] == \
        "observation_interpretation_counterfactual"
    replay = dossier["forecast_candidate"]["conditional_replay"]
    assert replay["human_recommendation_eligible"] is True
    assert dossier["forecast_candidate"]["rationale"] != "model proposal"


def test_exact_nonrecurring_sensor_spike_becomes_historical_interpretation():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timestamps = [(start + timedelta(hours=index)).isoformat()
                  for index in range(24 * 10)]
    history = [20.0 + 3.0 * math.sin(2 * math.pi * (index % 24) / 24)
               for index in range(len(timestamps))]
    spike_start = 96
    history[spike_start:spike_start + 2] = [200.0, 180.0]
    future = [(start + timedelta(hours=len(history) + index)).isoformat()
              for index in range(24)]
    span = (
        "The sensor experienced an unexpected glitch resulting in a spike "
        f"starting from {start + timedelta(hours=spike_start):%Y-%m-%d %H:%M:%S} "
        "for 2 hours.")
    raw = {"claims": [{
        "source_span": span, "relation": "supports_higher_variance",
        "effective_start": timestamps[spike_start],
        "effective_end": timestamps[spike_start + 2], "confidence": .9,
    }]}
    original = list(history)

    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span +
        " Assume that the sensor will not have this glitch in the future.",
        cutoff=timestamps[-1], future_timestamps=future, history=history,
        history_timestamps=timestamps, compiler_model="test")

    assert not reasons
    interpretation = dossier["observation_interpretations"][0]
    assert interpretation["predicate"]["op"] == "timestamp_window"
    assert interpretation["excluded_observations"] == 2
    assert interpretation["input_mutated"] is False
    assert dossier["forecast_candidate"] is not None
    assert history == original


def test_literal_historical_observation_fallback_is_narrow_and_verbatim():
    context = ("Background sentence. Maintenance resulted in no sales recorded "
               "starting 2026-01-02. There will be no future maintenance.")
    claim = deterministic_historical_observation_claim(
        context, history_start="2026-01-01T00:00:00+00:00",
        cutoff="2026-01-10T00:00:00+00:00")
    assert claim is not None
    assert claim["source_span"] == \
        "Maintenance resulted in no sales recorded starting 2026-01-02."
    assert claim["effective_start"] == "2026-01-02T00:00:00+00:00"
    assert claim["compiler_binding"] == "deterministic_literal_fallback"

    assert deterministic_historical_observation_claim(
        "The shop was closed and may close again.",
        history_start="2026-01-01T00:00:00+00:00",
        cutoff="2026-01-10T00:00:00+00:00") is None


def _raw(span: str, rows=None):
    return {
        "claims": [{
            "source_span": span,
            "relation": "supports_decrease",
            "effective_start": "2026-01-05T00:00:00+00:00",
            "effective_end": "2026-01-06T00:00:00+00:00",
            "mechanism": "closure",
            "confidence": 0.9,
        }],
        "forecast_candidate": {"quantiles": rows or [
            {"q10": 8, "q50": 9, "q90": 10},
            {"q10": 7, "q50": 8, "q90": 9},
        ]},
    }


def test_valid_dossier_is_cited_sealed_and_non_automatable():
    span = "The site will be closed on Monday."
    dossier, reasons = validate_temporal_dossier(
        _raw(span), context_text=span,
        cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=["2026-01-05T00:00:00+00:00",
                           "2026-01-06T00:00:00+00:00"],
        history=[8, 9, 10, 11], compiler_model="model-x")
    assert not reasons
    assert dossier["forecast_candidate"] is not None
    assert dossier["candidate_support"] == "prior_assisted"
    assert dossier["automation_eligible"] is False
    assert dossier["primary_forecast_unchanged"] is True
    assert len(dossier["seal_sha256"]) == 64
    assert verify_temporal_dossier_seal(dossier)
    dossier["claims"][0]["confidence"] = 0.1
    assert not verify_temporal_dossier_seal(dossier)


def test_past_tense_yearless_reference_binds_to_latest_past_occurrence():
    span = "As reference, the maximum on June 20th was 25.83."
    raw = {"claims": [{
        "source_span": span, "relation": "unknown",
        "effective_start": "June 20th", "effective_end": "June 20th",
        "confidence": 0.5,
    }], "hypotheses": [{
        "kind": "historical_analogue", "claim_ids": ["claim-1"],
        "target_series": ["*"], "known_at": "2026-09-09T00:00:00+00:00",
        "lag_steps": 0, "direction": "unknown",
    }]}

    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff="2026-09-09T00:00:00+00:00",
        future_timestamps=["2026-09-10T00:00:00+00:00"],
        history=[1, 2, 3], compiler_model="test")

    assert not reasons
    claim = dossier["claims"][0]
    assert claim["effective_start"] == "2026-06-20T00:00:00+00:00"
    assert claim["effective_window_binding"]["kind"] \
        == "most_recent_historical_month_day"
    assert dossier["hypotheses"][0]["kind"] == "historical_analogue"


def test_future_yearless_date_is_not_backdated_as_history():
    span = "The next June 20th will reach 25.83."
    dossier, reasons = validate_temporal_dossier(
        {"claims": [{"source_span": span, "relation": "unknown",
                     "effective_start": "June 20th",
                     "effective_end": "June 20th", "confidence": 0.5}]},
        context_text=span, cutoff="2026-09-09T00:00:00+00:00",
        future_timestamps=["2026-09-10T00:00:00+00:00"],
        history=[1, 2, 3], compiler_model="test")

    assert not dossier["claims"]
    assert "invalid effective window" in reasons[0]


def test_percentage_scale_claim_confidence_is_normalized_and_disclosed():
    span = "The site will be closed on Monday."
    raw = _raw(span)
    raw["claims"][0]["confidence"] = 95
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span,
        cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=["2026-01-05T00:00:00+00:00",
                           "2026-01-06T00:00:00+00:00"],
        history=[8, 9, 10, 11], compiler_model="model-x")
    assert not reasons
    claim = dossier["claims"][0]
    assert claim["confidence"] == 0.95
    assert claim["confidence_normalization"] == {
        "kind": "percent_to_unit_interval", "supplied": 95,
        "normalized": 0.95,
    }
    assert verify_temporal_dossier_seal(dossier)


def test_qualitative_claim_confidence_is_conservative_and_non_authoritative():
    span = "The site will be closed on Monday."
    raw = _raw(span)
    raw["claims"][0]["confidence"] = "high"
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span,
        cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=["2026-01-05T00:00:00+00:00",
                           "2026-01-06T00:00:00+00:00"],
        history=[8, 9, 10, 11], compiler_model="model-x")
    assert not reasons
    claim = dossier["claims"][0]
    assert claim["confidence"] == 0.75
    assert claim["confidence_normalization"] == {
        "kind": "qualitative_to_conservative_unit_interval",
        "supplied": "high", "normalized": 0.75,
        "authority_effect": "none",
    }
    assert dossier["automation_eligible"] is False

    raw = _raw(span)
    raw["claims"][0]["confidence"] = "high confidence"
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span,
        cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=["2026-01-05T00:00:00+00:00",
                           "2026-01-06T00:00:00+00:00"],
        history=[8, 9, 10, 11], compiler_model="model-x")
    assert not reasons
    assert dossier["claims"][0]["confidence"] == 0.75


def test_confidence_range_uses_conservative_endpoint_without_authority():
    span = "A comparable site peaked at 25.83."
    raw = _raw(span)
    raw["claims"][0]["confidence"] = "0.6-0.8"
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span,
        cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=["2026-01-05T00:00:00+00:00",
                           "2026-01-06T00:00:00+00:00"],
        history=[8, 9, 10, 11], compiler_model="model-x")

    assert not reasons
    claim = dossier["claims"][0]
    assert claim["confidence"] == 0.6
    assert claim["confidence_normalization"] == {
        "kind": "numeric_range_to_conservative_unit_interval",
        "supplied": "0.6-0.8", "normalized": 0.6,
        "authority_effect": "none",
    }


def test_unparseable_claim_confidence_retains_grounded_claim_without_authority():
    span = "A comparable site peaked at 25.83."
    raw = _raw(span)
    raw["claims"][0]["confidence"] = {"label": "fairly plausible"}
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span,
        cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=["2026-01-05T00:00:00+00:00",
                           "2026-01-06T00:00:00+00:00"],
        history=[8, 9, 10, 11], compiler_model="model-x")

    assert not reasons
    claim = dossier["claims"][0]
    assert claim["confidence"] == 0.25
    assert claim["confidence_normalization"] == {
        "kind": "unparseable_to_conservative_unit_interval",
        "supplied": "{'label': 'fairly plausible'}",
        "normalized": 0.25,
        "authority_effect": "none",
    }


def test_out_of_range_numeric_claim_confidence_is_still_rejected():
    span = "A comparable site peaked at 25.83."
    raw = _raw(span)
    raw["claims"][0]["confidence"] = 101
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span,
        cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=["2026-01-05T00:00:00+00:00",
                           "2026-01-06T00:00:00+00:00"],
        history=[8, 9, 10, 11], compiler_model="model-x")

    assert dossier["claims"] == []
    assert "claim 1 has invalid confidence" in reasons


def test_historical_zero_contamination_derives_sealed_counterfactual():
    span = ("Historical maintenance caused no withdrawals recorded, and the "
            "maintenance has ended.")
    raw = {
        "claims": [{
            "source_span": span, "relation": "unknown",
            "effective_start": "2026-01-01T00:00:00+00:00",
            "effective_end": "2026-01-05T00:00:00+00:00",
            "confidence": .9,
        }],
        "observation_interpretations": [{
            "kind": "historical_contamination", "claim_ids": ["claim-1"],
            "predicate": {"op": "equals", "value": 0},
            "window": "all_observed_history", "rationale": "availability zeros",
        }],
        "forecast_candidate": None,
    }
    history_times = [f"2026-01-0{day}T00:00:00+00:00" for day in range(1, 6)]
    future = ["2026-01-06T00:00:00+00:00",
              "2026-01-07T00:00:00+00:00"]
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff=history_times[-1],
        future_timestamps=future, history=[10, 0, 12, 0, 14],
        history_timestamps=history_times, compiler_model="test")

    assert not reasons
    interpretation = dossier["observation_interpretations"][0]
    assert interpretation["excluded_observations"] == 2
    assert interpretation["retained_observations"] == 3
    assert interpretation["input_mutated"] is False
    assert dossier["forecast_candidate"]["quantiles"][0]["q50"] == 12
    assert dossier["candidate_critique"]["selection_eligible"] is False
    assert "did not earn the human recommendation gate" in dossier[
        "candidate_critique"]["selection_reason"]
    assert dossier["forecast_candidate"]["conditional_replay"][
        "selection_eligible"] is False
    assert dossier["candidate_critique"]["candidate_origin"] == \
        "observation_interpretation_counterfactual"
    assert dossier["candidate_support"] == "prior_assisted"
    assert dossier["automation_eligible"] is False
    assert dossier["primary_forecast_unchanged"] is True
    assert verify_temporal_dossier_seal(dossier)


def test_qualitative_closure_cannot_invent_zero_contamination():
    span = "The site was historically closed for maintenance."
    raw = {
        "claims": [{
            "source_span": span, "relation": "unknown",
            "effective_start": "2026-01-01T00:00:00+00:00",
            "effective_end": "2026-01-05T00:00:00+00:00",
            "confidence": .8,
        }],
        "observation_interpretations": [{
            "kind": "historical_contamination", "claim_ids": ["claim-1"],
            "predicate": {"op": "equals", "value": 0},
            "window": "all_observed_history", "rationale": "closure",
        }],
    }
    history_times = [f"2026-01-0{day}T00:00:00+00:00" for day in range(1, 6)]
    dossier, _ = validate_temporal_dossier(
        raw, context_text=span, cutoff=history_times[-1],
        future_timestamps=["2026-01-06T00:00:00+00:00"],
        history=[10, 0, 12, 0, 14], history_timestamps=history_times,
        compiler_model="test")

    assert dossier["observation_interpretations"] == []
    assert dossier["forecast_candidate"] is None
    assert dossier["observation_interpretation_critique"]["rejected"][0][
        "code"] == "PREDICATE_NOT_ENTAILED"


def test_verified_absence_claim_auto_binds_repeated_transformed_floor():
    span = ("The ATM was under maintenance, resulting in no withdrawals "
            "recorded. Assume the ATM will not be in maintenance in the future.")
    raw = {
        "claims": [{
            "source_span": ("The ATM was under maintenance, resulting in no "
                            "withdrawals recorded."),
            "relation": "unknown", "effective_start": "unknown",
            "effective_end": "unknown", "confidence": "high confidence",
        }],
    }
    history_times = [f"2026-01-0{day}T00:00:00+00:00" for day in range(1, 7)]
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff=history_times[-1],
        future_timestamps=["2026-01-07T00:00:00+00:00"],
        history=[8, -1.25, 12, -1.25, 14, 10],
        history_timestamps=history_times, compiler_model="test")

    assert not reasons
    interpretation = dossier["observation_interpretations"][0]
    assert interpretation["predicate"] == {"op": "equals", "value": -1.25}
    assert interpretation["predicate_normalization"]["kind"] == \
        "semantic_zero_to_repeated_observed_floor"
    assert interpretation["excluded_observations"] == 2
    assert dossier["forecast_candidate"] is not None
    assert dossier["primary_forecast_unchanged"] is True
    assert dossier["automation_eligible"] is False


def test_verified_absence_claim_exposes_separated_noisy_zero_sensitivity():
    span = ("The ATM was under maintenance for various periods, resulting in "
            "no withdrawals recorded. Assume the ATM will not be in "
            "maintenance in the future.")
    history = [
        24, 27, -.8, .3, 26, 31, -1.1, .7,
        29, 25, -.4, .1, 33, 28, -.6, .9,
        30, 26, -.2, .5,
    ]
    history_times = [f"2026-01-{day:02d}T00:00:00+00:00"
                     for day in range(1, 21)]
    dossier, reasons = validate_temporal_dossier(
        {"claims": [{
            "source_span": ("The ATM was under maintenance for various "
                            "periods, resulting in no withdrawals recorded."),
            "relation": "unknown", "effective_start": "unknown",
            "effective_end": "unknown", "confidence": 1,
        }]}, context_text=span, cutoff=history_times[-1],
        future_timestamps=["2026-01-21T00:00:00+00:00"], history=history,
        history_timestamps=history_times, compiler_model="test")

    assert not reasons
    interpretation = dossier["observation_interpretations"][0]
    assert interpretation["predicate_normalization"]["kind"] == \
        "semantic_zero_to_separated_near_zero_cluster"
    assert interpretation["excluded_observations"] == 10
    replay = dossier["forecast_candidate"]["conditional_replay"]
    assert replay["status"] == "scenario_only_outcome_inferred_mask"
    assert replay["selection_eligible"] is False
    assert dossier["candidate_critique"]["selection_eligible"] is True
    assert dossier["automation_eligible"] is False


def test_noisy_zero_cluster_survives_broad_normal_activity_variance():
    span = ("Maintenance resulted in no withdrawals recorded. The maintenance "
            "has ended.")
    low = [-.7, -.5, -.4, -.3, -.2, -.1, -.05, .1, .2, .3, .4, .5]
    # The normal regime is intentionally broad; that variability should not
    # erase an otherwise separated, source-entailed near-zero component.
    normal = [5, 7, 9, 12, 14, 17, 19, 22, 25, 29, 33, 38]
    history = [value for pair in zip(low, normal) for value in pair]
    history_times = [f"2026-01-{day:02d}T00:00:00+00:00"
                     for day in range(1, 25)]

    dossier, reasons = validate_temporal_dossier(
        {"claims": [{
            "source_span": "Maintenance resulted in no withdrawals recorded.",
            "relation": "unknown", "effective_start": history_times[0],
            "effective_end": history_times[-1], "confidence": 1,
        }]}, context_text=span, cutoff=history_times[-1],
        future_timestamps=["2026-01-25T00:00:00+00:00"], history=history,
        history_timestamps=history_times, compiler_model="test")

    assert not reasons
    interpretation = dossier["observation_interpretations"][0]
    assert interpretation["excluded_observations"] == len(low)
    assert interpretation["predicate_normalization"]["kind"] == \
        "semantic_zero_to_separated_near_zero_cluster"
    assert dossier["forecast_candidate"]["conditional_replay"]["status"] == \
        "scenario_only_outcome_inferred_mask"
    assert dossier["automation_eligible"] is False


def test_absence_claim_does_not_split_one_broad_unimodal_history():
    span = ("The store had reporting failures resulting in no sales recorded. "
            "The reporting failure has ended.")
    history = [float(10 + index) for index in range(24)]
    history_times = [f"2026-01-{day:02d}T00:00:00+00:00"
                     for day in range(1, 25)]
    dossier, _ = validate_temporal_dossier(
        {"claims": [{"source_span": (
            "The store had reporting failures resulting in no sales recorded."),
            "relation": "unknown", "effective_start": history_times[0],
            "effective_end": history_times[-1], "confidence": 1}]},
        context_text=span, cutoff=history_times[-1],
        future_timestamps=["2026-01-25T00:00:00+00:00"], history=history,
        history_timestamps=history_times, compiler_model="test")

    assert dossier["observation_interpretations"] == []
    assert dossier["forecast_candidate"] is None


def test_absence_claim_does_not_turn_isolated_low_outliers_into_a_regime():
    span = ("The store had reporting failures resulting in no sales recorded. "
            "The reporting failure has ended.")
    history = [25.0 + (index % 5) for index in range(60)]
    history[3], history[41] = .1, -.2
    history_times = [f"2026-01-01T00:{minute:02d}:00+00:00"
                     for minute in range(60)]
    dossier, _ = validate_temporal_dossier(
        {"claims": [{"source_span": (
            "The store had reporting failures resulting in no sales recorded."),
            "relation": "unknown", "effective_start": history_times[0],
            "effective_end": history_times[-1], "confidence": 1}]},
        context_text=span, cutoff=history_times[-1],
        future_timestamps=["2026-01-01T01:00:00+00:00"], history=history,
        history_timestamps=history_times, compiler_model="test")

    assert dossier["observation_interpretations"] == []
    assert dossier["forecast_candidate"] is None


def test_cited_recurring_disruption_excludes_by_time_not_observed_value():
    claim = ("The ATM was under maintenance for 2 days, periodically every 4 "
             "days, starting from 2026-01-02 00:00:00, resulting in no "
             "withdrawals recorded.")
    context = claim + " Assume the ATM will not be in maintenance in the future."
    stale_interpretation = {
        "kind": "historical_contamination", "claim_ids": ["claim-99"],
        "predicate": {"op": "recurring_window",
                      "start": "2026-01-02 00:00:00",
                      "duration_steps": 2, "period_steps": 4},
        "window": "cited_window", "rationale": "stale model citation",
    }
    raw = {"claims": [{
        "source_span": claim, "relation": "unknown",
        "effective_start": "2026-01-02T00:00:00+00:00",
        "effective_end": "2026-01-09T00:00:00+00:00", "confidence": .9,
    }], "observation_interpretations": [dict(stale_interpretation)
                                         for _ in range(4)]}
    history_times = [f"2026-01-0{day}T00:00:00+00:00" for day in range(1, 10)]
    history = [11, -1.2, -1.0, 12, 13, -.8, -1.1, 14, 15]
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=context, cutoff=history_times[-1],
        future_timestamps=["2026-01-10T00:00:00+00:00"], history=history,
        history_timestamps=history_times, compiler_model="test")

    assert not reasons
    interpretation = dossier["observation_interpretations"][0]
    assert interpretation["predicate"]["op"] == "recurring_window"
    assert interpretation["excluded_observations"] == 4
    assert interpretation["retained_observations"] == 5
    assert dossier["observation_interpretation_critique"]["rejected"][0][
        "code"] == "UNVERIFIED_CLAIMS"
    assert dossier["forecast_candidate"] is not None
    assert dossier["primary_forecast_unchanged"] is True


def test_cited_daily_clock_outage_excludes_exact_half_open_windows():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history_times = [(start + timedelta(hours=index)).isoformat()
                     for index in range(24 * 7)]
    history = []
    expected = []
    for timestamp in history_times:
        hour = datetime.fromisoformat(timestamp).hour
        offline = hour >= 20
        expected.append(offline)
        history.append(0.0 if offline else 20.0 + hour / 10)
    claim = ("The sensor was offline for maintenance every day between "
             "20:00 and 00:00, which resulted in zero readings.")
    context = claim + " The sensor will not be in maintenance in the future."

    dossier, reasons = validate_temporal_dossier(
        {"claims": [{"source_span": claim, "relation": "unknown",
                     "effective_start": history_times[0],
                     "effective_end": history_times[-1], "confidence": 1}]},
        context_text=context, cutoff=history_times[-1],
        future_timestamps=[(start + timedelta(hours=24 * 7)).isoformat()],
        history=history, history_timestamps=history_times,
        compiler_model="test")

    assert not reasons
    interpretation = dossier["observation_interpretations"][0]
    assert interpretation["predicate"] == {
        "op": "recurring_clock_window", "start_time": "20:00",
        "end_time": "00:00", "interval": "half_open",
        "timezone_basis": "history_timestamp_timezone",
    }
    assert interpretation["excluded_observations"] == sum(expected) == 28
    assert interpretation["retained_observations"] == 140
    assert dossier["forecast_candidate"] is not None
    assert dossier["primary_forecast_unchanged"] is True


def test_clock_window_end_is_included_only_when_history_resolves_it():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history_times = [(start + timedelta(hours=index)).isoformat()
                     for index in range(24 * 7)]
    history = [
        0.0 if datetime.fromisoformat(timestamp).hour in {16, 17, 18}
        else 20.0
        for timestamp in history_times]
    claim = ("The sensor was offline for maintenance every day between "
             "16:00 and 18:00, "
             "which resulted in zero readings.")
    dossier, reasons = validate_temporal_dossier(
        {"claims": [{"source_span": claim, "relation": "unknown",
                     "effective_start": history_times[0],
                     "effective_end": history_times[-1], "confidence": 1}]},
        context_text=claim + " Assume that the sensor will not be in "
        "maintenance in the future.",
        cutoff=history_times[-1],
        future_timestamps=[(start + timedelta(hours=24 * 7)).isoformat()],
        history=history, history_timestamps=history_times,
        compiler_model="test")
    assert not reasons
    interpretation = dossier["observation_interpretations"][0]
    assert interpretation["excluded_observations"] == 21
    assert interpretation["predicate"]["interval"] == \
        "closed_end_empirically_resolved"
    resolution = interpretation["predicate"]["boundary_resolution"]
    assert resolution["matching_observations"] == \
        resolution["observations"] == 7
    assert resolution["uses_future_observations"] is False


def test_model_candidate_cannot_bypass_failed_observation_replay():
    start = datetime(2025, 10, 5, tzinfo=timezone.utc)
    history_times = [(start + timedelta(days=index)).isoformat()
                     for index in range(90)]
    history = [float(index + 1) for index in range(90)]
    future = [(start + timedelta(days=90 + index)).isoformat()
              for index in range(2)]
    claim = ("Maintenance lasted for 3 days every 6 days starting from "
             "2025-10-05 00:00:00, resulting in no requests recorded.")
    context = claim + " There will be no future maintenance."
    raw = {
        "claims": [{"source_span": claim, "relation": "unknown",
                    "effective_start": history_times[0],
                    "effective_end": history_times[-1], "confidence": 1}],
        "forecast_candidate": {"quantiles": [
            {"timestamp": timestamp, "q10": 89 + index,
             "q50": 91 + index, "q90": 93 + index}
            for index, timestamp in enumerate(future)],
            "rationale": "model extrapolation under the supplied context"},
    }
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=context, cutoff=history_times[-1],
        future_timestamps=future, history=history,
        history_timestamps=history_times, compiler_model="test")

    assert not reasons
    assert dossier["observation_interpretations"]
    assert dossier["observation_interpretation_critique"][
        "conditional_replay"]["selection_eligible"] is False
    assert dossier["candidate_critique"]["candidate_origin"] == "model_authored"
    assert dossier["candidate_critique"]["selection_eligible"] is False
    assert "cannot bypass" in dossier["candidate_critique"]["selection_reason"]


def test_uncited_claim_cannot_author_a_candidate():
    dossier, reasons = validate_temporal_dossier(
        _raw("invented"), context_text="The site remains open.",
        cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=["2026-01-05T00:00:00+00:00",
                           "2026-01-06T00:00:00+00:00"],
        history=[8, 9, 10, 11], compiler_model="model-x")
    assert dossier["forecast_candidate"] is None
    assert any("verbatim source_span" in reason for reason in reasons)
    assert any("requires a verified cited claim" in reason for reason in reasons)


def test_bad_quantile_order_and_implausible_jump_are_rejected():
    span = "The site will be closed on Monday."
    bad_order = [{"q10": 10, "q50": 9, "q90": 8}] * 2
    dossier, reasons = validate_temporal_dossier(
        _raw(span, bad_order), context_text=span,
        cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=["2026-01-05T00:00:00+00:00",
                           "2026-01-06T00:00:00+00:00"],
        history=[8, 9, 10, 11], compiler_model="model-x")
    assert dossier["forecast_candidate"] is None
    assert any("quantiles are invalid" in reason for reason in reasons)

    huge = [{"q10": 999, "q50": 1000, "q90": 1001}] * 2
    dossier, reasons = validate_temporal_dossier(
        _raw(span, huge), context_text=span,
        cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=["2026-01-05T00:00:00+00:00",
                           "2026-01-06T00:00:00+00:00"],
        history=[8, 9, 10, 11], compiler_model="model-x")
    assert dossier["forecast_candidate"] is None
    assert any("boundary-jump" in reason for reason in reasons)


def test_cited_bounded_regime_jump_is_retained_as_warned_prior_only():
    future = ["2026-01-05T00:00:00+00:00",
              "2026-01-06T00:00:00+00:00"]
    raw = _raw("A policy doubles demand, which remains below 120.", rows=[
        {"timestamp": stamp, "q10": 88, "q50": 100, "q90": 112}
        for stamp in future])
    raw["claims"] = [
        {"source_span": "A policy doubles demand",
         "relation": "supports_increase", "effective_start": future[0],
         "effective_end": future[-1], "confidence": .8},
        {"source_span": "demand, which remains below 120",
         "relation": "constrains_range", "effective_start": future[0],
         "effective_end": future[-1], "confidence": 1},
    ]
    dossier, reasons = validate_temporal_dossier(
        raw, context_text="A policy doubles demand, which remains below 120.",
        cutoff="2026-01-04T00:00:00+00:00", future_timestamps=future,
        history=[5, 5.1, 5.0, 5.1], compiler_model="test")

    assert not reasons
    candidate = dossier["forecast_candidate"]
    assert candidate is not None
    assert candidate["plausibility"]["warnings"]
    assert dossier["candidate_support"] == "prior_assisted"
    assert dossier["automation_eligible"] is False


def test_direction_and_bound_do_not_justify_large_candidate_jump():
    future = ["2026-01-05T00:00:00+00:00",
              "2026-01-06T00:00:00+00:00"]
    raw = _raw("A policy raises demand, which remains below 120.", rows=[
        {"timestamp": stamp, "q10": 88, "q50": 100, "q90": 112}
        for stamp in future])
    raw["claims"] = [
        {"source_span": "A policy raises demand",
         "relation": "supports_increase", "effective_start": future[0],
         "effective_end": future[-1], "confidence": .8},
        {"source_span": "demand, which remains below 120",
         "relation": "constrains_range", "effective_start": future[0],
         "effective_end": future[-1], "confidence": 1},
    ]
    dossier, reasons = validate_temporal_dossier(
        raw, context_text="A policy raises demand, which remains below 120.",
        cutoff="2026-01-04T00:00:00+00:00", future_timestamps=future,
        history=[5, 5.1, 5.0, 5.1], compiler_model="test")

    assert dossier["forecast_candidate"] is None
    assert "forecast_candidate failed boundary-jump plausibility" in reasons


def test_placeholder_is_rejected_and_degenerate_candidate_is_widened():
    span = "The site will be closed on Monday."
    future = ["2026-01-05T00:00:00+00:00",
              "2026-01-06T00:00:00+00:00"]
    placeholder = _raw(span, [
        {"timestamp": stamp, "q10": 0, "q50": 0, "q90": 0}
        for stamp in future])
    placeholder["forecast_candidate"]["rationale"] = (
        "Placeholder values; Gnomon must apply the relationship.")
    dossier, reasons = validate_temporal_dossier(
        placeholder, context_text=span,
        cutoff="2026-01-04T00:00:00+00:00", future_timestamps=future,
        history=[8, 9, 10, 11], compiler_model="test")
    assert dossier["forecast_candidate"] is None
    assert "forecast_candidate declares itself incomplete" in reasons

    degenerate = _raw(span, [
        {"timestamp": stamp, "q10": 9, "q50": 9, "q90": 9}
        for stamp in future])
    dossier, reasons = validate_temporal_dossier(
        degenerate, context_text=span,
        cutoff="2026-01-04T00:00:00+00:00", future_timestamps=future,
        history=[8, 9, 10, 11], compiler_model="test")
    assert not reasons
    candidate = dossier["forecast_candidate"]
    assert candidate is not None
    assert all(row["q10"] < row["q50"] < row["q90"]
               for row in candidate["quantiles"])
    assert candidate["plausibility"]["uncertainty_normalization"]["code"] == \
        "ROBUST_HISTORY_UNCERTAINTY_FLOOR"


def test_compact_constant_candidate_expands_only_onto_host_grid():
    span = "The historical outage ended and demand should stabilize."
    future = [f"2026-01-{day:02d}T00:00:00+00:00" for day in range(5, 15)]
    raw = {
        "claims": [{"source_span": span, "relation": "supports_stability",
                    "effective_start": future[0], "effective_end": future[-1],
                    "confidence": .7}],
        "forecast_candidate": {
            "constant_quantiles": {"q10": 8, "q50": 10, "q90": 12},
            "rationale": "stable post-outage distribution"},
    }
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=future, history=[8, 9, 10, 11],
        compiler_model="test")

    assert not reasons
    candidate = dossier["forecast_candidate"]
    assert [row["timestamp"] for row in candidate["quantiles"]] == future
    assert all(row["q50"] == 10 for row in candidate["quantiles"])
    assert candidate["path_normalization"] == {
        "kind": "constant_quantiles_expanded_to_host_grid", "steps": 10}


def test_compact_quantile_anchors_interpolate_only_on_host_grid():
    span = "A comparable site had a midday peak and then declined."
    future = [f"2026-01-05T0{hour}:00:00+00:00" for hour in range(5)]
    raw = {
        "claims": [{"source_span": span, "relation": "unknown",
                    "effective_start": "2025-01-05T00:00:00+00:00",
                    "effective_end": "2025-01-05T23:59:59+00:00",
                    "confidence": .5}],
        "forecast_candidate": {"quantile_anchors": [
            {"timestamp": future[0], "q10": 8, "q50": 10, "q90": 12},
            {"timestamp": future[2], "q10": 10, "q50": 12, "q90": 14},
            {"timestamp": future[4], "q10": 6, "q50": 8, "q90": 10},
        ], "rationale": "prior-assisted analogue shape"},
    }
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=future, history=[8, 9, 10, 11],
        compiler_model="test")

    assert not reasons
    candidate = dossier["forecast_candidate"]
    assert [row["q50"] for row in candidate["quantiles"]] == [10, 11, 12, 10, 8]
    assert candidate["path_normalization"] == {
        "kind": "quantile_anchors_linearly_interpolated_on_host_grid",
        "anchors": 3, "steps": 5, "source_field": "quantile_anchors"}


def test_sparse_quantile_rows_use_the_same_strict_anchor_contract():
    span = "A comparable site had a smooth bounded daily path."
    future = [f"2026-01-05T0{hour}:00:00+00:00" for hour in range(5)]
    raw = {
        "claims": [{"source_span": span, "relation": "unknown",
                    "effective_start": "2025-01-05T00:00:00+00:00",
                    "effective_end": "2025-01-05T23:59:59+00:00",
                    "confidence": .5}],
        "forecast_candidate": {"quantiles": [
            {"timestamp": future[0], "q10": 8, "q50": 10, "q90": 12},
            {"timestamp": future[4], "q10": 4, "q50": 6, "q90": 8},
        ], "rationale": "sparse prior-assisted path"},
    }
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=future, history=[8, 9, 10, 11],
        compiler_model="test")

    assert not reasons
    candidate = dossier["forecast_candidate"]
    assert [row["q50"] for row in candidate["quantiles"]] == [10, 9, 8, 7, 6]
    assert candidate["path_normalization"]["source_field"] \
        == "sparse_quantiles_alias"


def test_partial_quantile_anchors_require_immutable_primary_completion():
    span = "A comparable site had a midday peak."
    future = [f"2026-01-05T0{hour}:00:00+00:00" for hour in range(3)]
    raw = {
        "claims": [{"source_span": span, "relation": "unknown",
                    "effective_start": "2025-01-05T00:00:00+00:00",
                    "effective_end": "2025-01-05T23:59:59+00:00",
                    "confidence": .5}],
        "forecast_candidate": {"quantile_anchors": [
            {"timestamp": future[0], "q10": 8, "q50": 10, "q90": 12},
            {"timestamp": future[1], "q10": 9, "q50": 11, "q90": 13},
        ], "rationale": "incomplete anchors"},
    }
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=future, history=[8, 9, 10, 11],
        compiler_model="test")

    assert not reasons
    candidate = dossier["forecast_candidate"]
    assert candidate["requires_primary_completion"] is True
    assert candidate["path_normalization"]["unanchored_edges"] \
        == "immutable_primary"


def test_candidate_must_obey_its_own_cited_numeric_bounds():
    future = ["2026-01-05T00:00:00+00:00",
              "2026-01-06T00:00:00+00:00"]
    raw = _raw("values are bounded below by 4.79", rows=[
        {"timestamp": stamp, "q10": 0, "q50": 0, "q90": 0}
        for stamp in future])
    raw["claims"] = [
        {"source_span": "values are bounded below by 4.79",
         "relation": "constrains_range", "effective_start": future[0],
         "effective_end": future[-1], "confidence": 1},
        {"source_span": "values are bounded above by 9.13",
         "relation": "constrains_range", "effective_start": future[0],
         "effective_end": future[-1], "confidence": 1},
    ]
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=("values are bounded below by 4.79 and values are "
                           "bounded above by 9.13"),
        cutoff="2026-01-04T00:00:00+00:00", future_timestamps=future,
        history=[5, 6, 7, 6], compiler_model="test")
    assert dossier["forecast_candidate"] is None
    assert "forecast_candidate violates cited lower bound" in reasons
    assert dossier["candidate_critique"]["status"] == "rejected"
    assert dossier["candidate_critique"]["recovery_action"]


def test_explicit_relative_past_fact_is_background_not_future_trigger():
    span = (
        "Three months ago, the city inaugurated a new borough. "
        "This borough is adjacent to the river.")
    future = ["2026-04-01T00:00:00+00:00"]
    raw = {
        "claims": [{
            "source_span": span,
            "relation": "supports_increase",
            "effective_start": None,
            "effective_end": None,
            "mechanism": "A previously inaugurated riverfront borough.",
            "confidence": .5,
            "timing_status": "unresolved_trigger",
        }],
        "hypotheses": [{
            "kind": "historical_analogue", "claim_ids": ["claim-1"],
            "target_series": ["*"], "predictor_series": None,
            "known_at": "2026-03-01T00:00:00+00:00", "lag_steps": 0,
            "direction": "increase", "rationale": "A bounded analogue.",
        }],
    }

    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff="2026-03-01T00:00:00+00:00",
        future_timestamps=future, history=[0, 0, 0], compiler_model="test")

    assert not reasons
    claim = dossier["claims"][0]
    assert claim["timing_status"] == "atemporal_context"
    assert claim["effective_window_binding"]["kind"] \
        == "explicit_past_background_reconciled"
    assert claim["effective_window_binding"]["numeric_authority"] is False
    assert claim["effective_window_binding"]["automation_eligible"] is False


def test_cited_historical_distribution_keeps_cold_start_prior_as_scenario():
    span = "Comparable stores recorded monthly orders [9, 30]."
    future = ["2026-04-01T00:00:00+00:00",
              "2026-05-01T00:00:00+00:00"]
    raw = {
        "claims": [{
            "source_span": span, "relation": "supports_increase",
            "effective_start": None, "effective_end": None,
            "mechanism": "A bounded historical comparison.",
            "confidence": .5, "timing_status": "atemporal_context",
        }],
        "forecast_candidate": {"quantiles": [
            {"timestamp": future[0], "q10": 5, "q50": 10, "q90": 20},
            {"timestamp": future[1], "q10": 10, "q50": 25, "q90": 30},
        ], "rationale": "A cold-start prior from the cited comparator."},
    }

    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff="2026-03-01T00:00:00+00:00",
        future_timestamps=future, history=[0, 0, 0], compiler_model="test")

    assert not reasons
    assert dossier["forecast_candidate"] is not None
    assert dossier["automation_eligible"] is False
    warnings = dossier["forecast_candidate"]["plausibility"]["warnings"]
    assert any("cold-start extrapolation" in warning for warning in warnings)


def test_uncited_cold_start_explosion_remains_rejected():
    span = "A new store opened recently."
    future = ["2026-04-01T00:00:00+00:00",
              "2026-05-01T00:00:00+00:00"]
    raw = {
        "claims": [{
            "source_span": span, "relation": "supports_increase",
            "effective_start": None, "effective_end": None,
            "mechanism": "No numeric reference.", "confidence": .5,
            "timing_status": "atemporal_context",
        }],
        "forecast_candidate": {"quantiles": [
            {"timestamp": future[0], "q10": 5, "q50": 10, "q90": 20},
            {"timestamp": future[1], "q10": 10, "q50": 25, "q90": 30},
        ], "rationale": "An unsupported jump."},
    }

    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff="2026-03-01T00:00:00+00:00",
        future_timestamps=future, history=[0, 0, 0], compiler_model="test")

    assert dossier["forecast_candidate"] is None
    assert any("plausibility" in reason for reason in reasons)
