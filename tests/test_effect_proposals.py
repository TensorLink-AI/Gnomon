import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gnomon.effect_proposals import (assess_composed_effect, compose_effect,
                                     validate_effect_proposal)
from gnomon.llm_dossier import (
    deterministic_dated_multiplier_dossier, deterministic_events_from_claims,
    validate_temporal_dossier,
)
from gnomon.publication import publish_result, verify_publication
from gnomon.toolspec import runner_for


TIMES = ["2026-01-03T00:00:00+00:00", "2026-01-04T00:00:00+00:00"]
PRIMARY = [
    {"timestamp": TIMES[0], "point": 10.0, "q10": 9.0, "q50": 10.0, "q90": 11.0},
    {"timestamp": TIMES[1], "point": 10.0, "q10": 9.0, "q50": 10.0, "q90": 11.0},
]


def test_explicit_dated_multiplier_has_deterministic_failover_dossier():
    text = (
        "A heatwave began on 2026-01-03 00:00:00 and lasted for "
        "approximately 2 days. Sales reached approximately 4 times the "
        "typical usage.")
    raw = deterministic_dated_multiplier_dossier(
        text, cutoff="2026-01-02T00:00:00+00:00",
        future_timestamps=TIMES, target_name="sales")

    assert raw is not None
    assert raw["effect_proposal"]["location"] == 3.0
    assert raw["effect_proposal"]["delay_steps"] == 0
    assert raw["effect_proposal"]["duration_steps"] == 2
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=text, cutoff="2026-01-02T00:00:00+00:00",
        future_timestamps=TIMES, history=[8, 9, 10],
        history_timestamps=[
            "2025-12-30T00:00:00+00:00",
            "2025-12-31T00:00:00+00:00",
            "2026-01-02T00:00:00+00:00"], compiler_model="deterministic")
    assert not reasons
    assert dossier["effect_proposal"] is not None
    codes = {item["code"] for item in dossier["effect_proposal"].get(
        "semantic_normalizations") or []}
    assert "APPROXIMATE_CITED_LEVEL_MULTIPLIER" in codes


@pytest.mark.parametrize("text", [
    "Sales may increase during the promotion.",
    "Sales reached 4 times typical usage, but no start time was supplied.",
    ("Sales reached 4 times typical usage starting 2026-01-03 00:00:00, "
     "but no duration was supplied."),
    ("Temperatures reached 4 times normal starting 2026-01-03 00:00:00 "
     "for 2 days."),
])
def test_dated_multiplier_failover_refuses_missing_or_wrong_target_facts(text):
    assert deterministic_dated_multiplier_dossier(
        text, cutoff="2026-01-02T00:00:00+00:00",
        future_timestamps=TIMES, target_name="sales") is None


def _proposal(**updates):
    value = {
        "shape": "level_shift", "unit": "target_units",
        "location": 2.0, "lower": 1.0, "upper": 3.0,
        "confidence": .7, "delay_steps": 0, "duration_steps": None,
        "scope": {"kind": "single_series", "series": ["sales"]},
        "claim_ids": ["claim-1"], "rationale": "promotion",
        "uncertainty_basis": "operator estimate",
    }
    value.update(updates)
    return value


def test_typed_effect_is_composed_by_engine_not_model():
    proposal, critique = validate_effect_proposal(
        _proposal(), claim_ids={"claim-1"})
    rows = compose_effect(PRIMARY, proposal)
    assert critique["status"] == "accepted"
    assert [row["q50"] for row in rows] == [12.0, 12.0]
    assert [row["q10"] for row in rows] == [10.0, 10.0]
    assert PRIMARY[0]["q50"] == 10.0
    assert proposal["provenance_class"] == "model_authored_prior"
    assert proposal["uncertainty_basis"] == (
        "model-authored prior; not calibrated against supplied historical "
        "outcomes")
    assert "operator estimate" not in proposal["uncertainty_basis"]


def test_qualitative_citation_cannot_claim_invented_historical_provenance():
    proposal, critique = validate_effect_proposal(
        _proposal(
            location=-2, lower=-4, upper=-.5,
            rationale="estimated from historical holiday patterns",
            uncertainty_basis="similar historical holidays"),
        claim_ids={"claim-1"},
        claim_spans={"claim-1": "traffic typically reduces on holidays"})
    assert critique["status"] == "accepted"
    assert proposal["provenance_class"] == "model_authored_prior"
    assert "historical holiday" not in proposal["rationale"]
    assert "historical holiday" not in proposal["uncertainty_basis"]


def test_cited_numeric_distribution_has_source_stated_provenance():
    proposal, critique = validate_effect_proposal(
        _proposal(location=2, lower=1, upper=3),
        claim_ids={"claim-1"},
        claim_spans={"claim-1": "Use 2 units, with a range from 1 to 3."})
    assert critique["status"] == "accepted"
    assert proposal["provenance_class"] == "source_stated_distribution"
    assert proposal["uncertainty_basis"] == (
        "source-stated values; not calibrated against observed outcomes")


def test_effect_confidence_metadata_cannot_poison_valid_distribution():
    proposal, critique = validate_effect_proposal(
        _proposal(location=4, lower=4, upper=8, confidence="tentative"),
        claim_ids={"claim-1"})
    assert critique["status"] == "accepted"
    assert (proposal["lower"], proposal["location"], proposal["upper"]) == \
        (4, 4, 8)
    assert proposal["confidence"] == 0.25
    assert proposal["confidence_normalization"] == {
        "kind": "qualitative_to_conservative_unit_interval",
        "supplied": "tentative", "normalized": 0.25,
        "authority_effect": "none",
    }

    rejected, critique = validate_effect_proposal(
        _proposal(location=4, lower=4, upper=8, confidence="probably"),
        claim_ids={"claim-1"})
    assert rejected is None
    codes = [item["code"] for item in critique["attempts"][0]["violations"]]
    assert codes == ["INVALID_EFFECT_CONFIDENCE"]


def test_unique_operative_multiplier_resolves_across_separate_claims():
    proposal, critique = validate_effect_proposal(
        _proposal(shape="temporary_pulse", unit="fraction_of_level",
                  location=4, lower=4, upper=8, confidence="tentative",
                  claim_ids=["claim-1", "claim-2"]),
        claim_ids={"claim-1", "claim-2"},
        claim_spans={
            "claim-1": "typically 9 times the usual electricity",
            "claim-2": "only 5 times the usual electricity",
        })
    assert critique["status"] == "accepted"
    assert proposal["lower"] == proposal["location"] == proposal["upper"] == 4
    assert any(item["code"] ==
               "OPERATIVE_MULTIPLIER_SELECTED_ACROSS_CLAIMS"
               for item in proposal["semantic_normalizations"])
    assert proposal["confidence"] == 0.25

    rejected, critique = validate_effect_proposal(
        _proposal(unit="fraction_of_level", location=4, lower=4, upper=4,
                  claim_ids=["claim-1", "claim-2"]),
        claim_ids={"claim-1", "claim-2"}, claim_spans={
            "claim-1": "9 times the usual electricity",
            "claim-2": "5 times the usual electricity",
        })
    assert rejected is None
    assert critique["attempts"][0]["violations"][0]["code"] == \
        "CONFLICTING_CITED_MULTIPLIERS"


def test_cited_level_multiplier_is_normalized_to_additive_fraction():
    proposal, critique = validate_effect_proposal(
        _proposal(shape="temporary_pulse", unit="fraction_of_level",
                  location=4.0, lower=3.5, upper=4.5),
        claim_ids={"claim-1"},
        claim_spans={"claim-1": "demand will be 4 times the usual level"})
    assert critique["status"] == "accepted"
    assert (proposal["lower"], proposal["location"], proposal["upper"]) == \
        (3.0, 3.0, 3.0)
    assert proposal["semantic_normalizations"] == [{
        "code": "MULTIPLIER_TO_ADDITIVE_FRACTION",
        "stated_level_multiplier": 4.0,
        "applied_additive_fraction": 3.0,
        "parameterization_shift": -1.0,
        "basis": "verified cited source span",
        }, {
            "code": "UNSTATED_EFFECT_RANGE_REMOVED",
            "applied_value": 3.0,
            "basis": "citation states one exact multiplier; primary path retains forecast uncertainty",
        }, {
            "code": "EXACT_CITED_LEVEL_MULTIPLIER",
            "stated_level_multiplier": 4.0,
            "applied_additive_fraction": 3.0,
            "basis": "verified cited source span",
        }]
    assert compose_effect(PRIMARY, proposal)[0]["q50"] == 40.0
    assert compose_effect(PRIMARY, proposal)[0]["q10"] == 36.0
    assert compose_effect(PRIMARY, proposal)[0]["q90"] == 44.0


def test_temporary_multiplier_is_plateau_for_stated_duration():
    primary = [
        {"timestamp": f"2026-01-0{index + 1}T00:00:00+00:00",
         "point": 10, "q10": 9, "q50": 10, "q90": 11}
        for index in range(4)]
    proposal, _ = validate_effect_proposal(
        _proposal(shape="temporary_pulse", unit="fraction_of_level",
                  location=2, lower=2, upper=2, duration_steps=3),
        claim_ids={"claim-1"})

    rows = compose_effect(primary, proposal)

    assert [row["q50"] for row in rows] == [30, 30, 30, 10]
    assert [row["q10"] for row in rows] == [27, 27, 27, 9]
    assert [row["q90"] for row in rows] == [33, 33, 33, 11]


def test_exact_bounded_multiplier_repairs_vague_custom_shape():
    raw = _proposal(
        shape="custom_scenario", unit="fraction_of_level",
        location=-0.7, lower=-0.8, upper=-0.6, duration_steps=2)
    proposal, critique = validate_effect_proposal(
        raw, claim_ids={"claim-1"},
        claim_spans={"claim-1": "Traffic will be 30% of usual for 2 hours."})

    assert critique["status"] == "accepted"
    assert proposal["shape"] == "temporary_pulse"
    assert proposal["location"] == proposal["lower"] == proposal["upper"] == -0.7
    assert any(item["code"] == "EXACT_MULTIPLIER_TO_EXECUTABLE_SHAPE"
               for item in proposal["semantic_normalizations"])
    assert [row["q50"] for row in compose_effect(PRIMARY, proposal)] \
        == pytest.approx([3.0, 3.0])


def test_conflicting_cited_multipliers_fail_closed():
    proposal, critique = validate_effect_proposal(
        _proposal(unit="fraction_of_level", claim_ids=["claim-1", "claim-2"]),
        claim_ids={"claim-1", "claim-2"},
        claim_spans={"claim-1": "2 times the usual level",
                     "claim-2": "3 times the usual level"})
    assert proposal is None
    assert critique["attempts"][0]["violations"][0]["code"] == \
        "CONFLICTING_CITED_MULTIPLIERS"


def test_cited_calendar_onset_controls_relative_effect_delay():
    future = [f"2026-01-03T{hour:02d}:00:00+00:00" for hour in range(1, 5)]
    dossier, _ = validate_temporal_dossier({
        "claims": [{"source_span": "demand becomes 2 times the usual level at 2026-01-03 03:00:00",
                    "relation": "supports_increase",
                    "effective_start": future[2], "effective_end": future[3]}],
        "effect_proposal": _proposal(
            shape="temporary_pulse", unit="fraction_of_level",
            location=2.0, lower=2.0, upper=2.0, delay_steps=0,
            duration_steps=1),
    }, context_text="demand becomes 2 times the usual level at 2026-01-03 03:00:00",
       cutoff="2026-01-03T00:00:00+00:00", future_timestamps=future,
       history=[8, 9, 10], compiler_model="test")
    proposal = dossier["effect_proposal"]
    assert proposal["delay_steps"] == 2
    assert proposal["location"] == 1.0
    assert [item["code"] for item in proposal["semantic_normalizations"]] == [
        "MULTIPLIER_TO_ADDITIVE_FRACTION", "EXACT_CITED_LEVEL_MULTIPLIER",
        "CLAIM_ONSET_TO_HORIZON_DELAY"]


def test_uncited_model_authored_onset_cannot_realign_effect():
    future = [f"2026-01-03T{hour:02d}:00:00+00:00" for hour in range(1, 5)]
    dossier, _ = validate_temporal_dossier({
        "claims": [{"source_span": "demand becomes 2 times the usual level",
                    "relation": "supports_increase",
                    "effective_start": future[2], "effective_end": future[3]}],
        "effect_proposal": _proposal(
            shape="temporary_pulse", unit="fraction_of_level",
            location=2.0, lower=2.0, upper=2.0, delay_steps=0,
            duration_steps=1),
    }, context_text="demand becomes 2 times the usual level",
       cutoff="2026-01-03T00:00:00+00:00", future_timestamps=future,
       history=[8, 9, 10], compiler_model="test")
    assert dossier["effect_proposal"]["delay_steps"] == 0
    assert [item["code"] for item in
            dossier["effect_proposal"]["semantic_normalizations"]] == [
                "MULTIPLIER_TO_ADDITIVE_FRACTION",
                "EXACT_CITED_LEVEL_MULTIPLIER"]


def test_separate_cited_timing_and_magnitude_claims_can_align_effect():
    future = [f"2026-01-03T{hour:02d}:00:00+00:00" for hour in range(1, 5)]
    context = ("Event starts at 2026-01-03 03:00:00. "
               "Demand becomes 2 times the usual level.")
    dossier, _ = validate_temporal_dossier({
        "claims": [
            {"source_span": "Event starts at 2026-01-03 03:00:00",
             "relation": "unknown", "effective_start": future[2],
             "effective_end": future[3]},
            {"source_span": "Demand becomes 2 times the usual level",
             "relation": "supports_increase", "effective_start": future[2],
             "effective_end": future[3]},
        ],
        "effect_proposal": _proposal(
            shape="temporary_pulse", unit="fraction_of_level",
            location=2.0, lower=2.0, upper=2.0, delay_steps=0,
            duration_steps=1, claim_ids=["claim-1", "claim-2"]),
    }, context_text=context, cutoff="2026-01-03T00:00:00+00:00",
       future_timestamps=future, history=[8, 9, 10], compiler_model="test")
    proposal = dossier["effect_proposal"]
    assert proposal["location"] == 1.0
    assert proposal["delay_steps"] == 2


def test_single_validated_event_can_supply_missing_effect_timing():
    from gnomon.context import ContextEvent

    future = [f"2026-01-03T{hour:02d}:00:00+00:00" for hour in range(1, 5)]
    context = ("Event starts at 2026-01-03 03:00:00. "
               "Demand becomes approximately 2 times the usual level.")
    event = ContextEvent(
        event_id="event-1", event_type="promotion", entity_scope=("*",),
        effective_start=future[2], effective_end=future[3],
        known_at="2026-01-03T00:00:00+00:00",
        attributes={"evidence_quote": "Event starts at 2026-01-03 03:00:00"})
    dossier, _ = validate_temporal_dossier({
        "claims": [{
            "source_span": "Demand becomes approximately 2 times the usual level",
            "relation": "supports_increase", "effective_start": future[2],
            "effective_end": future[3]}],
        "effect_proposal": _proposal(
            shape="temporary_pulse", unit="fraction_of_level",
            location=1.0, lower=.5, upper=1.5, delay_steps=0,
            duration_steps=1),
    }, context_text=context, cutoff="2026-01-03T00:00:00+00:00",
       future_timestamps=future, history=[8, 9, 10], compiler_model="test",
       validated_events=[event])
    proposal = dossier["effect_proposal"]
    assert proposal["delay_steps"] == 2
    assert proposal["semantic_normalizations"][-1]["basis"] == \
        "single validated context event and forecast grid"


def test_repair_is_bounded_and_typed():
    accepted, critique = validate_effect_proposal(
        _proposal(shape="magic"), claim_ids={"claim-1"}, repair=_proposal())
    assert accepted["shape"] == "level_shift"
    assert critique["status"] == "accepted_after_repair"
    assert critique["attempts_used"] == 2
    rejected, critique = validate_effect_proposal(
        _proposal(shape="magic"), claim_ids={"claim-1"},
        repair=_proposal(shape="still_magic"))
    assert rejected is None and critique["attempts_remaining"] == 0
    assert critique["attempts"][0]["violations"][0]["code"] == "UNKNOWN_EFFECT_SHAPE"


def test_unknown_claim_and_bad_scope_fail_closed():
    proposal, critique = validate_effect_proposal(
        _proposal(claim_ids=["invented"], scope={"kind": "galaxy", "series": []}),
        claim_ids={"claim-1"})
    assert proposal is None
    assert {item["code"] for item in critique["attempts"][0]["violations"]} == {
        "UNVERIFIED_EFFECT_CLAIMS", "INVALID_EFFECT_SCOPE"}


def test_rejected_proposal_is_visible_in_publication_dispositions():
    dossier, _ = validate_temporal_dossier({
        "claims": [{"source_span": "promotion begins tomorrow",
                    "relation": "supports_increase",
                    "effective_start": TIMES[0], "effective_end": TIMES[-1]}],
        "effect_proposal": _proposal(shape="magic"),
    }, context_text="promotion begins tomorrow",
       cutoff="2026-01-02T00:00:00+00:00", future_timestamps=TIMES,
       history=[8, 9, 10], compiler_model="test")
    payload = publish_result({"support": "supported", "forecast": PRIMARY},
                             mode="scenario", dossiers=[dossier])
    rejected = [item for item in payload["context_dispositions"]
                if item["disposition"] == "rejected"]
    assert rejected[0]["reason_code"] == "effect_proposal_rejected"
    assert rejected[0]["violations"][0]["code"] == "UNKNOWN_EFFECT_SHAPE"


def test_single_claim_is_bound_after_gnomon_assigns_its_id():
    raw = _proposal()
    raw.pop("claim_ids")
    dossier, _ = validate_temporal_dossier({
        "claims": [{"source_span": "promotion begins tomorrow",
                    "relation": "supports_increase",
                    "effective_start": TIMES[0], "effective_end": TIMES[-1]}],
        "effect_proposal": raw,
    }, context_text="promotion begins tomorrow",
       cutoff="2026-01-02T00:00:00+00:00", future_timestamps=TIMES,
       history=[8, 9, 10], compiler_model="test")
    assert dossier["effect_proposal"]["claim_ids"] == ["claim-1"]
    assert dossier["effect_proposal"]["citation_binding"] == "single_verified_claim"


def test_only_literal_absolute_claim_becomes_deterministic_override():
    def dossier(span):
        return validate_temporal_dossier({
            "claims": [{"source_span": span, "relation": "supports_decrease",
                        "effective_start": TIMES[0], "effective_end": TIMES[-1]}],
        }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
           future_timestamps=TIMES, history=[8, 9, 10],
           compiler_model="test")[0]
    exact = deterministic_events_from_claims(
        dossier("output drops to zero during the shutdown"))
    assert exact[0]["event_type"] == "override:stated_absolute_value"
    assert exact[0]["deterministic_value_parsed"] == 0.0
    assert deterministic_events_from_claims(
        dossier("output will probably decline during the promotion")) == []


def test_literal_absolute_event_does_not_require_duplicate_claim():
    span = (
        "the meter will be offline for maintenance between 2026-01-03 "
        "00:00:00 and 2026-01-04 00:00:00, which results in zero readings")
    source_event = {
            "event_type": "maintenance outage", "entity_scope": ["*"],
            "effective_start": TIMES[0], "effective_end": TIMES[-1],
            "confidence": 1, "status": "confirmed",
            "evidence_quote": span,
            "event_id": "event_llm_00",
    }

    events = deterministic_events_from_claims(
        {"claims": [], "events": [source_event]},
        target_name="occupancy_rate",
        target_verified_spans={span})

    assert events[0]["event_type"] == "override:stated_absolute_value"
    assert events[0]["deterministic_value_parsed"] == 0.0
    assert events[0]["derived_from_event_id"] == "event_llm_00"


def test_literal_event_cannot_invent_its_effective_window():
    span = "the meter will be offline tomorrow, which results in zero readings"
    events = deterministic_events_from_claims({
        "claims": [],
        "events": [{
            "event_id": "event_llm_00", "event_type": "maintenance outage",
            "entity_scope": ["*"], "effective_start": TIMES[0],
            "effective_end": TIMES[-1], "evidence_quote": span,
        }],
    }, target_name="occupancy_rate", target_verified_spans={span})

    assert events == []


def test_literal_event_can_own_value_while_duplicate_claim_awaits_binding():
    span = (
        "the meter will be offline for maintenance between 2026-01-03 "
        "00:00:00 and 2026-01-04 00:00:00, which results in zero readings")
    dossier = {
        "claims": [{
            "claim_id": "claim-1", "source_span": span,
            "relation": "supports_decrease", "effective_start": TIMES[0],
            "effective_end": TIMES[-1],
            "effective_window_binding": {
                "kind": "model_supplied_unverified",
                "numeric_authority": False,
            },
        }],
        "events": [{
            "event_id": "event_llm_00", "event_type": "maintenance outage",
            "entity_scope": ["*"], "effective_start": TIMES[0],
            "effective_end": TIMES[-1], "evidence_quote": span,
        }],
    }

    events = deterministic_events_from_claims(
        dossier, target_name="occupancy_rate",
        target_verified_spans={span})

    assert len(events) == 1
    assert events[0]["deterministic_value_parsed"] == 0.0
    assert events[0]["derived_from_event_id"] == "event_llm_00"


def test_driver_absolute_value_cannot_override_a_different_target():
    span = (
        "The speed starts at 285.3. At 05:27:09, it rapidly and smoothly "
        "changes to 1593.0.")
    dossier = validate_temporal_dossier({
        "claims": [{"source_span": span, "relation": "supports_increase",
                    "effective_start": TIMES[0],
                    "effective_end": TIMES[-1]}],
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")[0]

    assert deterministic_events_from_claims(
        dossier, target_name="pressure_gap") == []


def test_symbolic_target_does_not_make_driver_schedule_a_target_override():
    span = (
        "X_0 is a covariate and X_1 is the variable to forecast. "
        "The value of X_0 is 0.2 from 2026-01-03 to 2026-01-04. "
        "Parents for X_1 at lag 1 are X_0 and X_1.")
    dossier = validate_temporal_dossier({
        "claims": [{"source_span": span, "relation": "unknown",
                    "effective_start": TIMES[0],
                    "effective_end": TIMES[-1]}],
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")[0]

    assert deterministic_events_from_claims(dossier, target_name="X_1") == []


def test_symbolic_target_literal_is_still_extracted():
    span = "X_0 is a covariate. X_1 takes a value of 7 in the forecast window."
    dossier = validate_temporal_dossier({
        "claims": [{"source_span": span, "relation": "supports_decrease",
                    "effective_start": TIMES[0],
                    "effective_end": TIMES[-1]}],
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")[0]

    events = deterministic_events_from_claims(dossier, target_name="X_1")
    assert events[0]["deterministic_value_parsed"] == 7.0
    assert "X_0" not in events[0]["deterministic_parse_span"]


def test_target_clause_isolated_before_deterministic_bound_parsing():
    span = "The maximal fan speed is 3000 rpm and maximal pressure is 37.5 Pa."
    dossier = validate_temporal_dossier({
        "claims": [{"source_span": span, "relation": "constrains_range",
                    "effective_start": TIMES[0],
                    "effective_end": TIMES[-1]}],
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")[0]

    events = deterministic_events_from_claims(
        dossier, target_name="pressure_gap")
    assert events[0]["deterministic_bound_parsed"] == {
        "min": None, "max": 37.5}
    assert "3000" not in events[0]["deterministic_parse_span"]


def test_unresolved_trigger_rule_is_retained_but_cannot_change_numbers():
    span = "Demand typically falls during public holidays."
    raw = {
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
            "direction": "decrease",
            "rationale": "No dated holiday trigger was supplied.",
        }],
        "effect_proposal": _proposal(claim_ids=["claim-1"]),
        "forecast_candidate": {
            "claim_ids": ["claim-1"],
            "constant_quantiles": {"q10": 1, "q50": 2, "q90": 3},
            "rationale": "Assume the horizon contains a holiday.",
        },
    }

    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
        future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")

    claim = dossier["claims"][0]
    assert claim["timing_status"] == "unresolved_trigger"
    assert claim["effective_window_binding"]["numeric_authority"] is False
    assert dossier["effect_proposal"] is None
    assert dossier["effect_proposal_critique"]["attempts"][0][
        "violations"][0]["code"] == "UNRESOLVED_TRIGGER_TIMING"
    assert dossier["forecast_candidate"] is None
    assert any("unresolved trigger timing" in reason for reason in reasons)
    assert deterministic_events_from_claims(dossier) == []


def test_explicit_cited_onset_corrects_model_unresolved_timing_label():
    span = "The new operating policy starts on 2026-01-03."
    raw = {"claims": [{
        "source_span": span, "relation": "supports_decrease",
        "effective_start": "2026-01-03T00:00:00+00:00",
        "effective_end": "2026-01-04T00:00:00+00:00",
        "timing_status": "unresolved_trigger", "confidence": .7,
    }]}

    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
        future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")

    assert not reasons
    claim = dossier["claims"][0]
    assert claim["timing_status"] == "resolved"
    assert claim["effective_window_binding"] == {
        "kind": "explicit_source_timing_reconciled",
        "basis": (
            "verbatim cited onset matches the supplied effective start; "
            "model-authored unresolved label was corrected"),
        "supplied_timing_status": "unresolved_trigger",
        "numeric_authority": False,
        "automation_eligible": False,
    }


def test_exact_clock_onset_is_reconciled_against_host_forecast_grid():
    span = (
        "The speed starts at 308.0. At 05:14:23, it rapidly and smoothly "
        "changes to 1592.4.")
    raw = {"claims": [{
        "source_span": span, "relation": "supports_increase",
        "effective_start": "1970-01-01T05:14:23+00:00",
        "effective_end": "1970-01-01T05:14:48+00:00",
        "timing_status": "unresolved_trigger", "confidence": 1.0,
    }]}

    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff="1970-01-01T05:14:22+00:00",
        future_timestamps=["1970-01-01T05:14:23+00:00"],
        history=[1, 2, 3], compiler_model="test")

    assert not reasons
    claim = dossier["claims"][0]
    assert claim["timing_status"] == "resolved"
    assert claim["effective_window_binding"]["kind"] == (
        "explicit_source_timing_reconciled")


def test_bare_clock_time_does_not_resolve_trigger_timing():
    span = "The report was issued at 05:14:23. Demand may fall on holidays."
    raw = {"claims": [{
        "source_span": span, "relation": "supports_decrease",
        "effective_start": "1970-01-01T05:14:23+00:00",
        "effective_end": "1970-01-01T05:14:48+00:00",
        "timing_status": "unresolved_trigger", "confidence": .7,
    }]}

    dossier, _ = validate_temporal_dossier(
        raw, context_text=span, cutoff="1970-01-01T05:14:22+00:00",
        future_timestamps=["1970-01-01T05:14:23+00:00"],
        history=[1, 2, 3], compiler_model="test")

    assert dossier["claims"][0]["timing_status"] == "unresolved_trigger"


def test_clock_onset_at_cutoff_governs_first_future_step():
    span = "At 05:14:56, the fan speed changes to 661.1."
    raw = {"claims": [{
        "source_span": span, "relation": "supports_decrease",
        "effective_start": "1970-01-01T05:14:57+00:00",
        "effective_end": "1970-01-01T05:15:11+00:00",
        "timing_status": "unresolved_trigger", "confidence": 1.0,
    }]}

    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff="1970-01-01T05:14:56+00:00",
        future_timestamps=["1970-01-01T05:14:57+00:00"],
        history=[1, 2, 3], compiler_model="test")

    assert not reasons
    assert dossier["claims"][0]["timing_status"] == "resolved"


def test_reconciled_source_timing_does_not_block_separate_literal_override():
    span = "Readings drop to zero starting 2026-01-03."
    dossier, reasons = validate_temporal_dossier(
        {"claims": [{
            "source_span": span, "relation": "supports_decrease",
            "effective_start": "2026-01-03T00:00:00+00:00",
            "effective_end": "2026-01-04T00:00:00+00:00",
            "timing_status": "unresolved_trigger", "confidence": 1.0,
        }]},
        context_text=span, cutoff="2026-01-02T00:00:00+00:00",
        future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")

    assert not reasons
    assert dossier["claims"][0]["effective_window_binding"]["kind"] == \
        "explicit_source_timing_reconciled"
    events = deterministic_events_from_claims(dossier)
    assert len(events) == 1
    assert events[0]["deterministic_value_parsed"] == 0


def test_single_target_event_binding_can_resolve_elliptical_claim_subject():
    span = "The building containing the machine is closed tomorrow."
    dossier, _ = validate_temporal_dossier(
        {"claims": [{
            "source_span": span, "relation": "supports_decrease",
            "effective_start": TIMES[0], "effective_end": TIMES[-1],
            "timing_status": "resolved", "confidence": 1.0,
        }]}, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
        future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")

    assert deterministic_events_from_claims(
        dossier, target_name="withdrawals") == []
    events = deterministic_events_from_claims(
        dossier, target_name="withdrawals", target_verified_spans={span})
    assert events[0]["deterministic_value_parsed"] == 0


def test_unresolved_trigger_is_not_reconciled_from_unrelated_iso_date():
    span = ("The report was issued on 2026-01-03. Demand typically falls "
            "during public holidays.")
    dossier, _ = validate_temporal_dossier(
        {"claims": [{
            "source_span": span, "relation": "supports_decrease",
            "effective_start": "2026-01-03T00:00:00+00:00",
            "effective_end": "2026-01-04T00:00:00+00:00",
            "timing_status": "unresolved_trigger", "confidence": .7,
        }]},
        context_text=span, cutoff="2026-01-02T00:00:00+00:00",
        future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")

    assert dossier["claims"][0]["timing_status"] == "unresolved_trigger"


def test_atemporal_background_is_visible_but_not_a_deterministic_effect():
    span = "On average, the service receives 12 incidents per year."
    raw = {
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
        "effect_proposal": _proposal(claim_ids=["claim-1"]),
    }

    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
        future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")

    assert not reasons
    claim = dossier["claims"][0]
    assert claim["timing_status"] == "atemporal_context"
    assert claim["effective_window_binding"]["kind"] \
        == "forecast_question_scope_atemporal_context"
    assert claim["effective_window_binding"]["numeric_authority"] is False
    assert dossier["hypotheses"][0]["kind"] == "historical_analogue"
    assert dossier["effect_proposal"] is None
    assert dossier["effect_proposal_critique"]["attempts"][0][
        "violations"][0]["code"] \
        == "ATEMPORAL_CONTEXT_NO_NUMERIC_AUTHORITY"
    assert deterministic_events_from_claims(dossier) == []


def test_association_only_claim_cannot_select_model_authored_path():
    span = "Demand and support tickets tend to co-occur and are correlated."
    raw = {
        "claims": [{
            "source_span": span, "relation": "supports_increase",
            "effective_start": None, "effective_end": None,
            "timing_status": "atemporal_context", "confidence": .7,
        }],
        "hypotheses": [{
            "kind": "relationship", "claim_ids": ["claim-1"],
            "target_series": ["*"], "predictor_series": "tickets",
            "known_at": "2026-01-02T00:00:00+00:00", "lag_steps": 0,
            "direction": "increase", "rationale": "Association only.",
        }],
        "series": ["tickets"],
        "forecast_candidate": {
            "claim_ids": ["claim-1"],
            "constant_quantiles": {"q10": 8, "q50": 9, "q90": 10},
            "rationale": "Assume ticket demand changes the target.",
        },
    }

    dossier, reasons = validate_temporal_dossier(
        raw, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
        future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")

    assert not reasons
    assert dossier["claims"][0]["relationship_authority"] \
        == "associational_only"
    assert dossier["claims"][0]["causal_authority"] is False
    assert dossier["forecast_candidate"] is not None
    critique = dossier["candidate_critique"]
    assert critique["selection_eligible"] is False
    assert "Correlation" in critique["selection_reason"]


def test_explicit_causal_claim_is_not_demoted_as_association_only():
    span = "The shutdown causes demand to fall during the stated window."
    dossier, reasons = validate_temporal_dossier({
        "claims": [{
            "source_span": span, "relation": "supports_decrease",
            "effective_start": TIMES[0], "effective_end": TIMES[-1],
            "timing_status": "resolved", "confidence": .7,
        }],
        "forecast_candidate": {
            "claim_ids": ["claim-1"],
            "constant_quantiles": {"q10": 6, "q50": 7, "q90": 8},
        },
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")

    assert not reasons
    assert "relationship_authority" not in dossier["claims"][0]
    assert dossier["candidate_critique"]["selection_eligible"] is True


def test_atemporal_claim_derives_stable_non_numeric_hypothesis():
    span = "On average, the service receives 12 incidents per year."
    dossier, reasons = validate_temporal_dossier({
        "claims": [{
            "source_span": span, "relation": "supports_stability",
            "effective_start": None, "effective_end": None,
            "timing_status": "atemporal_context", "confidence": .7,
        }],
        "hypotheses": [{"kind": "relationship", "claim_ids": ["missing"]}],
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")

    assert not reasons
    assert len(dossier["hypotheses"]) == 1
    assert dossier["hypotheses"][0]["kind"] == "historical_analogue"
    critique = dossier["hypothesis_critique"]
    assert critique["status"] == "accepted_after_deterministic_fallback"
    assert critique["deterministic_fallback"] is True
    assert critique["fallback_basis"] == "verified_atemporal_claims"
    assert critique["rejected"]


def test_association_derives_unsupported_not_causal_hypothesis():
    span = "Demand and bicycle incidents tend to co-occur."
    dossier, _ = validate_temporal_dossier({
        "claims": [{
            "source_span": span, "relation": "supports_increase",
            "effective_start": None, "effective_end": None,
            "timing_status": "atemporal_context", "confidence": .7,
        }],
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")

    hypothesis = dossier["hypotheses"][0]
    assert hypothesis["kind"] == "unsupported"
    assert "without causal or numeric authority" in hypothesis["rationale"]


def test_verbatim_scale_joins_unique_validated_event_timing():
    times = [
        "2026-01-03T00:00:00+00:00", "2026-01-03T01:00:00+00:00",
        "2026-01-03T02:00:00+00:00", "2026-01-03T03:00:00+00:00",
    ]
    magnitude = "Demand will be 4 times the usual level"
    quote = (magnitude + " from 2026-01-03 01:00:00 for 2 hours.")
    event = SimpleNamespace(
        effective_start="2026-01-03T01:00:00+00:00",
        effective_end="2026-01-03T03:00:00+00:00",
        attributes={"evidence_quote": quote})
    dossier, reasons = validate_temporal_dossier({
        "claims": [{
            "source_span": magnitude, "relation": "supports_increase",
            "effective_start": None, "effective_end": None,
            "timing_status": "unresolved_trigger", "confidence": .9,
        }],
    }, context_text=quote, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=times, history=[8, 9, 10], compiler_model="test",
       validated_events=[event])

    assert not reasons
    claim = dossier["claims"][0]
    assert claim["timing_status"] == "resolved"
    assert claim["effective_start"] == "2026-01-03T01:00:00+00:00"
    assert claim["effective_window_binding"]["kind"] \
        == "validated_event_context_join"
    effect = dossier["effect_proposal"]
    assert effect["unit"] == "fraction_of_level"
    assert effect["location"] == 3.0
    assert effect["delay_steps"] == 1
    assert effect["duration_steps"] == 2
    assert effect["compiler_binding"] \
        == "validated_event_plus_verbatim_scale"


def test_event_timing_join_requires_one_unambiguous_containing_event():
    magnitude = "Demand will be 4 times the usual level"
    events = [SimpleNamespace(
        effective_start="2026-01-03T00:00:00+00:00",
        effective_end="2026-01-04T00:00:00+00:00",
        attributes={"evidence_quote": (
            magnitude + " from 2026-01-03 00:00:00.")}) for _ in range(2)]
    dossier, reasons = validate_temporal_dossier({
        "claims": [{
            "source_span": magnitude, "relation": "supports_increase",
            "effective_start": None, "effective_end": None,
            "timing_status": "unresolved_trigger", "confidence": .9,
        }],
    }, context_text=events[0].attributes["evidence_quote"],
       cutoff="2026-01-02T00:00:00+00:00", future_timestamps=TIMES,
       history=[8, 9, 10], compiler_model="test", validated_events=events)

    assert not reasons
    assert dossier["claims"][0]["timing_status"] == "unresolved_trigger"
    assert dossier["effect_proposal"] is None


def test_adjacent_same_paragraph_scale_joins_one_validated_event():
    event_sentence = (
        "A heatwave began on 2026-01-03 01:00:00 and lasted for 2 hours.")
    magnitude = "Demand reached approximately 4 times the usual level."
    context = event_sentence + " " + magnitude
    event = SimpleNamespace(
        effective_start="2026-01-03T01:00:00+00:00",
        effective_end="2026-01-03T03:00:00+00:00",
        attributes={"evidence_quote": event_sentence,
                    "soft_context": {"direction": "increase"}})
    times = [
        "2026-01-03T00:00:00+00:00", "2026-01-03T01:00:00+00:00",
        "2026-01-03T02:00:00+00:00", "2026-01-03T03:00:00+00:00",
    ]
    dossier, reasons = validate_temporal_dossier({
        "claims": [{
            "source_span": magnitude, "relation": "supports_increase",
            "effective_start": None, "effective_end": None,
            "timing_status": "unresolved_trigger", "confidence": .7,
        }],
    }, context_text=context, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=times, history=[8, 9, 10], compiler_model="test",
       validated_events=[event])

    assert not reasons
    assert dossier["claims"][0]["timing_status"] == "resolved"
    effect = dossier["effect_proposal"]
    assert effect["location"] == effect["lower"] == effect["upper"] == 3.0
    codes = {item["code"] for item in effect["semantic_normalizations"]}
    assert "APPROXIMATE_CITED_LEVEL_MULTIPLIER" in codes


def test_adjacent_join_refuses_cross_paragraph_and_opposite_direction():
    event_sentence = (
        "A shutdown began on 2026-01-03 01:00:00 and lasted for 2 hours.")
    magnitude = "Demand reached 4 times the usual level."
    for context, direction in (
            (event_sentence + "\n\n" + magnitude, "increase"),
            (event_sentence + " " + magnitude, "decrease")):
        event = SimpleNamespace(
            effective_start="2026-01-03T01:00:00+00:00",
            effective_end="2026-01-03T03:00:00+00:00",
            attributes={"evidence_quote": event_sentence,
                        "soft_context": {"direction": direction}})
        dossier, _ = validate_temporal_dossier({
            "claims": [{
                "source_span": magnitude, "relation": "supports_increase",
                "effective_start": None, "effective_end": None,
                "timing_status": "unresolved_trigger", "confidence": .7,
            }],
        }, context_text=context, cutoff="2026-01-02T00:00:00+00:00",
           future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test",
           validated_events=[event])
        assert dossier["claims"][0]["timing_status"] == "unresolved_trigger"


def test_approximate_cited_multiplier_removes_unstated_range_and_can_compose():
    proposal, critique = validate_effect_proposal({
        "shape": "temporary_pulse", "unit": "fraction_of_level",
        "location": 3, "lower": 2, "upper": 4, "confidence": .6,
        "delay_steps": 0, "duration_steps": 1,
        "scope": {"kind": "single_series", "series": ["*"]},
        "claim_ids": ["claim-1"],
    }, claim_ids={"claim-1"}, claim_spans={
        "claim-1": "Demand reached approximately 4 times the usual level."})

    assert critique["status"] == "accepted"
    assert proposal["location"] == proposal["lower"] == proposal["upper"] == 3
    assert proposal["provenance_class"] == "source_stated_distribution"
    codes = {item["code"] for item in proposal["semantic_normalizations"]}
    assert codes == {"UNSTATED_APPROXIMATE_RANGE_REMOVED",
                     "APPROXIMATE_CITED_LEVEL_MULTIPLIER"}
    assessment = assess_composed_effect(PRIMARY, proposal)
    assert assessment["accepted"] is True
    assert assessment["scale_guard_disposition"] == \
        "exact_cited_scenario_allowed"


def test_null_optional_bounds_mean_a_point_effect():
    proposal, critique = validate_effect_proposal({
        "shape": "temporary_pulse", "unit": "fraction_of_level",
        "location": 3, "lower": None, "upper": None, "confidence": .6,
        "delay_steps": 0, "duration_steps": 1,
        "scope": {"kind": "single_series", "series": ["*"]},
        "claim_ids": ["claim-1"],
    }, claim_ids={"claim-1"}, claim_spans={
        "claim-1": "Demand reached approximately 4 times the usual level."})

    assert critique["status"] == "accepted"
    assert proposal["lower"] == proposal["location"] == proposal["upper"] == 3


def test_cited_percentage_repairs_invalid_model_distribution():
    proposal, critique = validate_effect_proposal({
        "shape": "temporary_pulse", "unit": "fraction_of_level",
        "location": -.6, "lower": -.4, "upper": -.8, "confidence": .8,
        "delay_steps": 0, "duration_steps": 2,
        "scope": {"kind": "single_series", "series": ["*"]},
        "claim_ids": ["claim-1"],
    }, claim_ids={"claim-1"}, claim_spans={
        "claim-1": "Traffic will be 40% of the usual level for two hours."})

    assert critique["status"] == "accepted"
    assert proposal["lower"] == proposal["location"] == proposal["upper"] == -.6
    codes = {item["code"] for item in proposal["semantic_normalizations"]}
    assert "SOURCE_SCALE_REPLACED_MODEL_DISTRIBUTION" in codes
    assert "EXACT_CITED_LEVEL_MULTIPLIER" in codes


def test_effect_onset_uses_locally_cited_date_not_model_delay():
    future = [f"2026-01-03T0{hour}:00:00+00:00" for hour in range(5)]
    timing = "The scheduled event begins on 2026-01-03 03:00:00."
    magnitude = "Demand will be approximately 4 times the usual level."
    dossier, reasons = validate_temporal_dossier({
        "claims": [{
            "source_span": magnitude, "relation": "supports_increase",
            "effective_start": future[3], "effective_end": future[4],
            "confidence": .7,
        }],
        "effect_proposal": _proposal(
            shape="temporary_pulse", unit="fraction_of_level",
            location=3, lower=3, upper=3, delay_steps=0,
            duration_steps=2),
    }, context_text=f"{timing} {magnitude}",
       cutoff="2026-01-02T00:00:00+00:00", future_timestamps=future,
       history=[8, 9, 10], compiler_model="test")

    assert reasons == []
    proposal = dossier["effect_proposal"]
    assert proposal["delay_steps"] == 3
    assert proposal["semantic_normalizations"][-1]["code"] == \
        "CLAIM_ONSET_TO_HORIZON_DELAY"


def test_effect_onset_does_not_cross_source_paragraphs():
    future = [f"2026-01-03T0{hour}:00:00+00:00" for hour in range(5)]
    timing = "An unrelated event begins on 2026-01-03 03:00:00."
    magnitude = "Demand will be approximately 4 times the usual level."
    dossier, reasons = validate_temporal_dossier({
        "claims": [{
            "source_span": magnitude, "relation": "supports_increase",
            "effective_start": future[3], "effective_end": future[4],
            "confidence": .7,
        }],
        "effect_proposal": _proposal(
            shape="temporary_pulse", unit="fraction_of_level",
            location=3, lower=3, upper=3, delay_steps=0,
            duration_steps=2),
    }, context_text=f"{timing}\n\n{magnitude}",
       cutoff="2026-01-02T00:00:00+00:00", future_timestamps=future,
       history=[8, 9, 10], compiler_model="test")

    assert reasons == []
    assert dossier["effect_proposal"]["delay_steps"] == 0


def test_literal_range_claim_becomes_deterministic_constraint():
    span = "values are bounded above by 10.00 and bounded below by 5.82"
    dossier = validate_temporal_dossier({
        "claims": [{"source_span": span, "relation": "constrains_range",
                    "effective_start": TIMES[0], "effective_end": TIMES[-1]}],
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")[0]
    events = deterministic_events_from_claims(dossier)
    assert events[0]["event_type"] == "constraint:stated_range"
    assert events[0]["deterministic_bound_parsed"] == {"min": 5.82, "max": 10.0}


def test_publication_prefers_effect_composition_and_retains_portfolio():
    dossier, reasons = validate_temporal_dossier({
        "claims": [{"source_span": "promotion begins tomorrow",
                    "relation": "supports_increase",
                    "effective_start": TIMES[0], "effective_end": TIMES[-1],
                    "confidence": .8}],
        "effect_proposal": _proposal(),
        # A full path is accepted for compatibility but must not outrank the
        # safer engine-composed representation of the same context.
        "forecast_candidate": {"quantiles": [
            {"timestamp": stamp, "q10": 12 + i, "q50": 13 + i, "q90": 14 + i}
            for i, stamp in enumerate(TIMES)]},
    }, context_text="promotion begins tomorrow",
       cutoff="2026-01-02T00:00:00+00:00", future_timestamps=TIMES,
       history=[8, 9, 10], compiler_model="test")
    assert dossier["effect_proposal_critique"]["status"] == "accepted"
    payload = publish_result({"support": "supported", "forecast": PRIMARY},
                             mode="best_effort", dossiers=[dossier])
    assert payload["recommended_scenario_id"] == "effect-composed-1"
    assert payload["recommended_forecast"][0]["q50"] == 12
    assert payload["primary_forecast"][0]["q50"] == 10
    assert payload["automation"]["eligible"] is False
    assert {item["role"] for item in payload["candidate_portfolio"]} >= {
        "effect_composed", "model_authored"}
    effect = next(item for item in payload["candidate_portfolio"]
                  if item["role"] == "effect_composed")
    assert effect["effect"]["provenance_class"] == "model_authored_prior"
    assert effect["assumptions"] == [
        "Model-authored conditional effect estimate composed over verified "
        "qualitative context.",
        "model-authored prior; not calibrated against supplied historical "
        "outcomes",
    ]
    assert payload["temporal_state"]["trend"]["direction"] == "flat_or_unknown"
    assert verify_publication(payload)
    disposition = next(item for item in payload["context_dispositions"]
                       if item["reason_code"] == "effect_proposal_composed")
    assert disposition["disposition"] == "used"
    assert disposition["selection_role"] == "human_facing_recommendation"


def test_model_candidate_empirical_story_is_retained_only_as_unverified():
    context = "traffic typically reduces on holidays"
    dossier, reasons = validate_temporal_dossier({
        "claims": [{"source_span": context, "relation": "supports_decrease",
                    "effective_start": TIMES[0], "effective_end": TIMES[-1]}],
        "forecast_candidate": {
            "constant_quantiles": {"q10": 7, "q50": 8, "q90": 9},
            "rationale": "calibrated from many similar historical holidays",
        },
    }, context_text=context, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")
    assert reasons == []
    candidate = dossier["forecast_candidate"]
    assert candidate["provenance_class"] == "model_authored_prior"
    assert "historical holidays" not in candidate["rationale"]
    assert candidate["model_rationale_unverified"] == (
        "calibrated from many similar historical holidays")
    payload = publish_result({"support": "supported", "forecast": PRIMARY},
                             mode="best_effort", dossiers=[dossier])
    scenario = next(item for item in payload["candidate_portfolio"]
                    if item["role"] == "model_authored")
    assert "historical holidays" not in " ".join(scenario["assumptions"])


def test_absolute_zero_claim_cannot_create_additive_zero_scenario():
    span = "The meter is offline tomorrow, which results in zero readings."
    dossier, reasons = validate_temporal_dossier({
        "claims": [{"source_span": span, "relation": "supports_decrease",
                    "effective_start": TIMES[0], "effective_end": TIMES[-1]}],
        "effect_proposal": _proposal(
            location=0, lower=0, upper=0, duration_steps=2),
        "forecast_candidate": {
            "constant_quantiles": {"q10": 8, "q50": 9, "q90": 10},
            "rationale": "zero during maintenance",
        },
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")
    assert reasons == []
    payload = publish_result({"support": "supported", "forecast": PRIMARY},
                             mode="scenario", dossiers=[dossier])
    assert all(item["role"] != "effect_composed"
               for item in payload["candidate_portfolio"])
    shadow = next(item for item in payload["candidate_portfolio"]
                  if item["role"] == "model_authored")
    assert shadow["selection_eligible"] is False
    assert any("deterministic absolute/range" in assumption
               for assumption in shadow["assumptions"])
    superseded = next(item for item in payload["context_dispositions"]
                      if item.get("reason_code") ==
                      "superseded_by_deterministic_context_contract")
    assert superseded["disposition"] == "superseded"
    assert payload["context_summary"]["status"] == "scenario_only"


def test_exact_context_use_is_not_downgraded_by_superseded_effect_lane():
    span = "The ATM has no cash tomorrow, resulting in zero withdrawals."
    dossier, _ = validate_temporal_dossier({
        "claims": [{"source_span": span, "relation": "supports_decrease",
                    "effective_start": TIMES[0], "effective_end": TIMES[-1]}],
        "effect_proposal": _proposal(
            location=0, lower=0, upper=0, duration_steps=2),
    }, context_text=span, cutoff="2026-01-02T00:00:00+00:00",
       future_timestamps=TIMES, history=[8, 9, 10], compiler_model="test")
    result = {
        "support": "context_trusted", "forecast": PRIMARY,
        "primary_forecast": PRIMARY,
        "context_outcome": {
            "status": "applied", "events": ["exact-zero"],
            "admission_basis": "future_context_contract",
        },
    }
    payload = publish_result(result, mode="best_effort", dossiers=[dossier])

    assert payload["context_summary"]["status"] == "used"
    assert payload["context_summary"]["counts"] == {
        "used": 2, "scenario": 0, "rejected": 0}
    assert any(item["disposition"] == "superseded"
               for item in payload["context_dispositions"])


def test_validated_context_path_precedes_weaker_model_effect():
    dossier, _ = validate_temporal_dossier({
        "claims": [{"source_span": "promotion begins tomorrow",
                    "relation": "supports_increase",
                    "effective_start": TIMES[0], "effective_end": TIMES[-1]}],
        "effect_proposal": _proposal(),
    }, context_text="promotion begins tomorrow",
       cutoff="2026-01-02T00:00:00+00:00", future_timestamps=TIMES,
       history=[8, 9, 10], compiler_model="test")
    conditioned = [{**row, "q10": 20, "q50": 21, "q90": 22, "point": 21}
                   for row in PRIMARY]
    result = {
        "support": "context_trusted", "forecast": conditioned,
        "primary_forecast": PRIMARY,
        "context_outcome": {"status": "applied",
                            "admission_basis": "future_context_contract"},
    }
    payload = publish_result(result, mode="best_effort", dossiers=[dossier])
    assert payload["recommended_scenario_id"] == "context_conditioned"
    assert payload["recommended_forecast"] == conditioned
    selected = next(item for item in payload["candidate_portfolio"]
                    if item["scenario_id"] == "context_conditioned")
    assert selected["claim_ids"] == ["claim-1"]
    assert any(item["role"] == "effect_composed"
               for item in payload["candidate_portfolio"])


def test_mcp_one_call_validates_and_composes_raw_context(tmp_path):
    source = tmp_path / "series.csv"
    source.write_text("timestamp,value\n" + "\n".join(
        f"2026-01-{day:02d},{100 + day}" for day in range(1, 21)) + "\n")
    proposal = {
        "claims": [{"source_span": "capacity doubles next week",
                    "relation": "supports_increase",
                    "effective_start": "2026-01-21T00:00:00+00:00",
                    "effective_end": "2026-01-22T00:00:00+00:00",
                    "confidence": .8}],
        "effect_proposal": {
            **_proposal(scope={"kind": "single_series", "series": ["value"]}),
        },
    }
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "target_column": "value", "horizon": 2,
        "output_dir": str(tmp_path / "out"),
        "publication_mode": "best_effort",
        "context_submission": {
            "text": "capacity doubles next week",
            "known_at": "2026-01-20T12:00:00+00:00",
            "compiler": "test-agent", "proposal": proposal,
        },
    })
    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "effect-composed-1"
    assert publication["primary_forecast_unchanged"] is True
    # The compact wire projection carries temporal state once at publication
    # level; the complete receipt retains the selection-contract copy.
    assert publication["temporal_state"]
    assert publication["context_summary"]["status"] == "used"
    assert publication["projection"] == "compact"
    receipt = json.loads(Path(payload["publication_path"]).read_text(
        encoding="utf-8"))
    assert verify_publication(receipt)


@pytest.mark.parametrize("shape", [
    "temporary_pulse", "level_shift", "trend_change", "variance_change",
    "ramp_recovery", "seasonal_amplitude", "seasonal_phase",
    "cross_series_relationship", "saturation_bound",
])
def test_all_composable_shapes_preserve_horizon_and_quantile_order(shape):
    updates = {"shape": shape, "duration_steps": 2}
    if shape == "variance_change":
        updates["unit"] = "fraction_of_level"
    if shape == "cross_series_relationship":
        updates["scope"] = {"kind": "shared", "series": ["sales", "traffic"]}
    if shape.startswith("seasonal_"):
        updates["period_steps"] = 2
    proposal, _ = validate_effect_proposal(
        _proposal(**updates), claim_ids={"claim-1"})
    rows = compose_effect(PRIMARY, proposal)
    assert len(rows) == len(PRIMARY)
    assert all(row["q10"] <= row["q50"] <= row["q90"] for row in rows)


def test_composed_effect_rejects_scale_explosion_and_fractional_negative_base():
    huge, _ = validate_effect_proposal(
        _proposal(location=100, lower=90, upper=110), claim_ids={"claim-1"})
    assessment = assess_composed_effect(PRIMARY, huge)
    assert assessment["accepted"] is False
    assert assessment["violations"][0]["code"] == "IMPLAUSIBLE_COMPOSED_DISPLACEMENT"
    negative = [{**row, "point": -3, "q10": -4, "q50": -3, "q90": -2}
                for row in PRIMARY]
    fractional, _ = validate_effect_proposal(
        _proposal(unit="fraction_of_level", location=.2, lower=.1, upper=.3),
        claim_ids={"claim-1"})
    assessment = assess_composed_effect(negative, fractional)
    assert any(item["code"] == "NONPOSITIVE_FRACTIONAL_BASE"
               for item in assessment["violations"])


def test_exact_cited_multiplier_may_exceed_history_scale_as_scenario_only():
    span = "Electricity will be 5 times the usual level for one hour."
    proposal, critique = validate_effect_proposal(
        _proposal(unit="fraction_of_level", location=5, lower=3, upper=6,
                  duration_steps=1),
        claim_ids={"claim-1"}, claim_spans={"claim-1": span})
    assert critique["status"] == "accepted"
    assert proposal["location"] == proposal["lower"] == proposal["upper"] == 4
    assert any(item["code"] == "EXACT_CITED_LEVEL_MULTIPLIER"
               for item in proposal["semantic_normalizations"])
    narrow_primary = [
        {"point": 100, "q10": 99.9, "q50": 100, "q90": 100.1},
        {"point": 100, "q10": 99.9, "q50": 100, "q90": 100.1},
    ]
    assessment = assess_composed_effect(narrow_primary, proposal)
    assert assessment["accepted"] is True
    assert assessment["maximum_displacement_scales"] > 20
    assert assessment["scale_guard_disposition"] == \
        "exact_cited_scenario_allowed"


def test_approximate_cited_multiplier_can_exceed_historical_scale_guard():
    approximate, _ = validate_effect_proposal(
        _proposal(unit="fraction_of_level", location=4, lower=3, upper=5),
        claim_ids={"claim-1"},
        claim_spans={"claim-1": "about 5 times the usual level"})
    assessment = assess_composed_effect(PRIMARY, approximate)
    assert assessment["accepted"] is True
    assert assessment["scale_guard_disposition"] == \
        "exact_cited_scenario_allowed"


def test_uncited_large_multiplier_still_hits_scale_guard():
    uncited, _ = validate_effect_proposal(
        _proposal(unit="fraction_of_level", location=4, lower=3, upper=5),
        claim_ids={"claim-1"},
        claim_spans={"claim-1": "Demand is expected to rise."})
    assessment = assess_composed_effect(PRIMARY, uncited)
    assert assessment["accepted"] is False
    assert assessment["scale_guard_disposition"] == "rejected"
