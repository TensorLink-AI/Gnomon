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
    assert outcome["hypotheses"][0]["magnitude"] is None
    assert outcome["recovery_actions"]


def test_point_supported_interval_weak_context_is_explicitly_non_automatable() -> None:
    assessment = ContextAssessment(
        considered=True, admitted=False,
        reasons=["interval coverage failed"],
        events_used=["event-1"], point_candidate=[101.0, 102.0],
        point_support="point_supported_interval_weak",
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
    assert "intervals failed" in outcome["basis"]


def test_failed_deterministic_claim_is_rejected_not_scenario() -> None:
    outcome = context_outcome([_event("constraint:capacity")], "heart_rate")
    assert outcome["status"] == "rejected"
    assert outcome["primary_forecast_changed"] is False


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
