from __future__ import annotations

import pytest

from gnomon.context import (
    ContextEvent,
    ContextSource,
    backtest_admissible,
    events_from_list,
    event_to_dict,
    validate_context_event,
)
from gnomon.llm import LLMAdapter, LLMUnavailable, NullLLMAdapter


def _event(**overrides) -> ContextEvent:
    base = dict(
        event_id="event_01",
        event_type="customer_launch",
        entity_scope=("api-prod",),
        effective_start="2026-08-14T00:00:00+10:00",
        effective_end="2026-08-20T23:59:59+10:00",
        known_at="2026-07-22T09:30:00+10:00",
        source=ContextSource(type="planning_file", reference="launches.md#enterprise-a"),
        created_by="hermes_agent",
    )
    base.update(overrides)
    return ContextEvent(**base)


def test_valid_event_has_no_problems() -> None:
    assert validate_context_event(_event()) == []


def test_naive_timestamps_are_rejected() -> None:
    problems = validate_context_event(_event(known_at="2026-07-22T09:30:00"))
    assert any("timezone" in problem for problem in problems)


def test_effective_range_must_be_ordered() -> None:
    problems = validate_context_event(_event(effective_end="2026-08-01T00:00:00+10:00"))
    assert any("precedes" in problem for problem in problems)


def test_event_status_is_a_closed_contract() -> None:
    assert validate_context_event(_event(status="tentative")) == []
    assert validate_context_event(_event(status="cancelled")) == []
    problems = validate_context_event(_event(status="superseded-ish"))
    assert any("status must be one of" in problem for problem in problems)


def test_verifiable_source_is_backtest_admissible() -> None:
    assert backtest_admissible(_event()) is True


def test_unsourced_event_is_never_backtest_admissible() -> None:
    assert backtest_admissible(_event(source=None)) is False


def test_llm_assertion_source_is_never_backtest_admissible() -> None:
    event = _event(source=ContextSource(type="assertion", reference="model claim"))
    assert backtest_admissible(event) is False


def test_inline_event_cannot_self_attest_historical_known_at() -> None:
    raw = event_to_dict(_event(created_by="user"))
    parsed = events_from_list([raw])[0]
    assert parsed.created_by == "unverified_external"
    assert backtest_admissible(parsed) is False


def test_null_adapter_degrades_deterministically() -> None:
    adapter = NullLLMAdapter()
    assert isinstance(adapter, LLMAdapter)
    with pytest.raises(LLMUnavailable) as excinfo:
        adapter.complete("prompt", {"type": "object"})
    assert excinfo.value.code == "LLM_UNAVAILABLE"
