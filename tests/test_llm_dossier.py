from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gnomon.llm_dossier import (
    deterministic_historical_observation_claim,
    validate_temporal_dossier,
    verify_temporal_dossier_seal,
)


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
