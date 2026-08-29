from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gnomon.context import ContextEvent, ContextSource, validate_context_event
from gnomon.context_eval import ContextAssessment
from gnomon.soft_context import (
    context_outcome,
    make_context_receipt,
    write_context_receipt,
)


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _event(event_type: str = "medication_adjustment") -> ContextEvent:
    return ContextEvent(
        event_id="event-1", event_type=event_type,
        entity_scope=("heart_rate",),
        effective_start=START.isoformat(),
        effective_end=(START + timedelta(hours=1)).isoformat(),
        known_at=START.isoformat(),
        source=ContextSource("planning_file", "plan.md"),
        attributes={"soft_context": {
            "effect_family": "level_shift", "direction": "decrease",
            "duration": "temporary",
        }},
    )


def test_grounded_unestimated_event_is_scenario_only() -> None:
    outcome = context_outcome([_event()], "heart_rate")
    assert outcome["status"] == "scenario_only"
    assert outcome["primary_forecast_changed"] is False
    assert outcome["automation_eligible"] is False
    assert outcome["hypotheses"][0]["magnitude"] is None
    assert outcome["recovery_actions"]


def test_out_of_scope_context_cannot_imply_automation_authority() -> None:
    outcome = context_outcome([_event()], "blood_pressure")
    assert outcome == {
        "status": "not_considered",
        "primary_forecast_changed": False,
        "canonical_primary_preserved": True,
        "automation_eligible": False,
        "events": [],
    }


def test_point_supported_interval_weak_context_is_explicitly_non_automatable() -> None:
    assessment = ContextAssessment(
        considered=True, admitted=False,
        reasons=["interval coverage failed"],
        events_used=["event-1"], point_candidate=[101.0, 102.0],
        point_support="point_supported_interval_weak",
        gate_checks=[
            {"code": "point_improvement", "passed": True},
            {"code": "coverage_not_degraded", "passed": False},
        ],
    )

    outcome = context_outcome(
        [_event()], "heart_rate", context_assessment=assessment,
        sensitivity_scenarios=[{
            "events": ["event-1"],
            "support": "point_supported_interval_weak",
        }],
    )

    assert outcome["status"] == "scenario_only"
    assert outcome["selected_output_role"] == "interval_weak_context_scenario"
    assert outcome["scenario_support"] == "point_supported_interval_weak"
    assert outcome["automation_eligible"] is False
    assert outcome["primary_forecast_changed"] is False
    assert outcome["canonical_primary_preserved"] is True
    assert outcome["failed_gate_codes"] == ["coverage_not_degraded"]
    assert "intervals failed" in outcome["basis"]


def test_failed_deterministic_claim_is_rejected_not_scenario() -> None:
    outcome = context_outcome([_event("constraint:capacity")], "heart_rate")
    assert outcome["status"] == "rejected"
    assert outcome["primary_forecast_changed"] is False
    assert outcome["canonical_primary_preserved"] is True
    assert outcome["automation_eligible"] is False


def test_admitted_history_without_horizon_effect_does_not_claim_change() -> None:
    assessment = ContextAssessment(
        considered=True, admitted=True, reasons=[], events_used=["event-1"],
        points=[10.0], residuals=[0.0],
    )
    outcome = context_outcome(
        [_event()], "heart_rate", context_assessment=assessment,
        primary_forecast_changed=False)

    assert outcome["status"] == "applied"
    assert outcome["primary_forecast_changed"] is False
    assert outcome["selected_output_role"] == \
        "primary_forecast_no_numeric_context_change"
    assert "no admitted effect changed this horizon" in outcome["basis"]


def test_structural_scenario_reports_its_decisive_future_gate() -> None:
    assessment = ContextAssessment(
        considered=True, admitted=False,
        reasons=["history evaluation had fewer than four folds"],
        gate_checks=[{"code": "separated_folds_available", "passed": False}],
    )
    event = _event("structural:trend_ceases")
    outcome = context_outcome(
        [event], "heart_rate", context_assessment=assessment,
        future_context={"checks": [{
            "event_id": "event-1", "event_class": "structural",
            "code": "separated_model_folds_available", "passed": False,
            "detail": "four transformation-specific evaluations are required",
        }]},
        sensitivity_scenarios=[{
            "events": ["event-1"], "support": "prior_assisted_structural",
        }],
    )

    assert outcome["status"] == "partially_represented"
    assert outcome["failed_gate_codes"] == [
        "separated_model_folds_available"]
    assert outcome["gate_reasons"] == [
        "four transformation-specific evaluations are required"]


def test_flat_primary_makes_trend_ceases_a_typed_noop_not_an_opaque_rejection():
    event = _event("structural:trend_ceases")
    outcome = context_outcome(
        [event], "heart_rate",
        future_context={"checks": [{
            "event_id": "event-1", "event_class": "structural",
            "code": "emitted_trend_is_directionally_stable", "passed": False,
            "detail": "the emitted path has no stable continuing trend",
        }], "rejected": [{
            "event_id": "event-1",
            "code": "emitted_trend_is_directionally_stable",
        }]},
    )

    assert outcome["status"] == "rejected"
    assert outcome["relationship_to_primary"] == "no_distinct_numeric_path"
    assert outcome["selected_output_role"] == \
        "primary_forecast_already_noncontinuing"
    assert outcome["primary_forecast_changed"] is False
    assert outcome["automation_eligible"] is False
    assert "would not create a defensibly distinct" in outcome["basis"]
    assert outcome["recovery_actions"]


def test_mixed_context_preserves_scenario_and_rejected_dispositions() -> None:
    generic = _event()
    rejected = ContextEvent(**{
        **_event("override:literal_exact").__dict__,
        "event_id": "literal-1",
    })

    outcome = context_outcome([generic, rejected], "heart_rate")

    assert outcome["status"] == "partially_represented"
    assert outcome["events"] == ["event-1", "literal-1"]
    assert outcome["dispositions"] == [
        {"context_id": "event-1", "disposition": "scenario"},
        {"context_id": "literal-1", "disposition": "rejected"},
    ]


def test_soft_context_contract_rejects_numeric_effect_smuggling() -> None:
    event = _event()
    bad = ContextEvent(**{
        **event.__dict__,
        "attributes": {"soft_context": {
            "effect_family": "level_shift", "direction": "decrease",
            "duration": "temporary", "magnitude": 20,
        }},
    })
    assert any("cannot supply a magnitude" in problem
               for problem in validate_context_event(bad))


def test_context_receipt_persistence_is_content_addressed_and_immutable(tmp_path) -> None:
    kwargs = {
        "documents": [{"content_fingerprint": "sha256:abc"}],
        "events": [], "hypotheses": [], "rejected": [],
        "rejected_hypotheses": [], "proposer": {"model": "compiler"},
    }
    receipt = make_context_receipt(**kwargs)
    first = write_context_receipt(receipt, tmp_path)
    second = write_context_receipt(receipt, tmp_path)
    assert first == second
    tampered = {**receipt, "events": [{"event_id": "invented"}]}
    with pytest.raises(ValueError, match="does not match"):
        write_context_receipt(tampered, tmp_path)
