"""Typed scenario effects become an auditable, queryable outcome memory."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from gnomon.effects import EffectDistribution, EffectProvenance, effect_contract
from gnomon.effect_registry import load_effect_registry, resolve_effect_prior
from gnomon.tracking import ForecastRecord, TrackingStore


def _prior(identifier: str, mean: float, error: float, **changes):
    value = {
        "prior_id": identifier, "event_type": "deploy", "target": "latency",
        "domain": "operations", "population": "api", "unit": "ms",
        "effect_family": "temporary_pulse", "mean": mean,
        "standard_error": error, "delay_steps": [0, 1],
        "duration_steps": [1, 4], "sample_size": 20,
        "source": "validated-experiment", "version": "2026-08",
        "expires_at": "2027-01-01T00:00:00+00:00",
    }
    value.update(changes)
    return value


def test_external_registry_is_versioned_validated_and_precision_pooled(tmp_path):
    path = tmp_path / "effects.json"
    path.write_text(json.dumps({"schema_version": "0.1", "priors": [
        _prior("a", 10.0, 2.0), _prior("b", 20.0, 1.0),
    ]}), encoding="utf-8")
    priors = load_effect_registry(
        path, now=datetime(2026, 8, 14, tzinfo=timezone.utc))
    result = resolve_effect_prior(
        priors, event_type="deploy", target="latency",
        domain="operations", population="api", unit="ms")
    assert result["status"] == "scenario_prior"
    assert result["mean"] == pytest.approx(18.0)
    assert result["standard_error"] == pytest.approx((1 / 1.25) ** 0.5)
    assert result["may_affect_primary_forecast"] is False


def test_external_registry_prefers_specific_evidence_and_rejects_expiry(tmp_path):
    path = tmp_path / "effects.json"
    path.write_text(json.dumps({"schema_version": "0.1", "priors": [
        _prior("generic", 99.0, 1.0, target="*", domain="*", population="*", unit="*"),
        _prior("specific", 7.0, 1.0),
    ]}), encoding="utf-8")
    priors = load_effect_registry(
        path, now=datetime(2026, 8, 14, tzinfo=timezone.utc))
    assert resolve_effect_prior(
        priors, event_type="deploy", target="latency", domain="operations",
        population="api", unit="ms")["mean"] == 7.0
    path.write_text(json.dumps({"schema_version": "0.1", "priors": [
        _prior("expired", 1.0, 1.0, expires_at="2025-01-01T00:00:00+00:00")
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="expired"):
        load_effect_registry(
            path, now=datetime(2026, 8, 14, tzinfo=timezone.utc))


def _record() -> ForecastRecord:
    return ForecastRecord(
        "forecast-1", "capacity", "api", "2026-08-01T00:00:00+00:00",
        4, "D", "seasonal_naive", "supported", None, None, 1.0, "",
        "2026-08-01T00:00:00+00:00",
    )


def _scenario(event: str = "deploy-1", offset: float = 5.0):
    contract = effect_contract(
        EffectDistribution("normal", offset, 1.0, offset - 1.28,
                           offset + 1.28, 0.8, 8),
        EffectProvenance(
            "same_event_same_series", True,
            "2026-08-01T00:00:00+00:00", "deployments.csv", 1.0, 0.8,
            "detrended active-minus-inactive mean",
        ),
        shape="temporary_pulse",
    )
    return {
        "events": [event], "effect": contract,
        "forecast": [
            {"timestamp": f"2026-08-0{day}T00:00:00+00:00",
             "point": 10.0 + (offset if day in (3, 4) else 0.0)}
            for day in range(2, 6)
        ],
    }


class Event:
    event_id = "deploy-1"
    event_type = "deployment"


def test_effect_distribution_rejects_invalid_uncertainty():
    with pytest.raises(ValueError, match="bounds/scale"):
        EffectDistribution("normal", 2.0, -1.0, 1.0, 3.0, 0.8, 3)
    with pytest.raises(ValueError, match="cannot claim probability"):
        EffectDistribution("assumption", 2.0, 0.0, 2.0, 2.0, 0.8, 0)
    with pytest.raises(ValueError, match="timezone"):
        EffectProvenance(
            "human_assumption", False, "2026-08-01T00:00:00",
            None, None, None, "operator assumption",
        )


def test_registry_resolves_against_frozen_primary(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    primary = [
        {"timestamp": f"2026-08-0{day}T00:00:00+00:00", "point": 10.0}
        for day in range(2, 6)
    ]
    assert store.record_effect_scenarios(
        "capacity", "forecast-1", "api", primary, [_scenario()], [Event()]
    ) == 1
    unresolved = store.event_effects("capacity")
    assert unresolved[0]["estimate"] is None
    assert unresolved[0]["effect_distribution"]["location"] == 5.0
    assert unresolved[0]["provenance_class"] == "same_event_same_series"

    # The event is active at steps 1 and 2. Actual-minus-frozen-primary is
    # therefore [4, 6], yielding a realised estimate of 5 in target units.
    assert store._resolve_effect_scenarios(
        _record(), [10.0, 14.0, 16.0, 10.0],
        outcome_known_at="2026-08-06T02:00:00+00:00",
    ) == 1
    row = store.event_effects("capacity", include_unresolved=False)[0]
    assert row["estimate"] == pytest.approx(5.0)
    assert row["baseline_bias"] == 0.0
    assert row["counterfactual_quality"] == "inactive_horizon_bias_adjusted"
    assert row["sample_count"] == 2
    assert row["standard_error"] == pytest.approx(1.0)
    assert row["missing_counterfactual"] is False
    assert row["confounding_status"] == "none_detected"
    assert row["onset_steps"] == 0 and row["duration_steps"] == 2
    assert row["outcome_known_at"] == "2026-08-06T02:00:00+00:00"
    assert row["expected_effect_shape"] == "temporary_pulse"
    assert row["realised_effect_shape"] == "unknown"  # only two active steps


def test_overlapping_scenarios_are_marked_confounded(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    primary = [
        {"timestamp": f"2026-08-0{day}T00:00:00+00:00", "point": 10.0}
        for day in range(2, 6)
    ]
    other = type("Event", (), {"event_id": "deploy-2",
                                "event_type": "deployment"})()
    store.record_effect_scenarios(
        "capacity", "forecast-1", "api", primary,
        [_scenario(), _scenario("deploy-2", 3.0)], [Event(), other],
    )
    store._resolve_effect_scenarios(_record(), [10.0, 14.0, 16.0, 10.0])
    rows = store.event_effects("capacity", include_unresolved=False)
    assert len(rows) == 2
    assert {row["confounding_status"] for row in rows} == {"overlapping_scenarios"}


def test_query_filters_event_type_and_series(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    primary = [
        {"timestamp": f"2026-08-0{day}T00:00:00+00:00", "point": 10.0}
        for day in range(2, 6)
    ]
    store.record_effect_scenarios(
        "capacity", "forecast-1", "api", primary, [_scenario()], [Event()]
    )
    assert len(store.event_effects(
        "capacity", event_type="deployment", series="api")) == 1
    assert store.event_effects("capacity", event_type="promotion") == []


def test_effect_id_is_idempotent_and_content_addressed(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    primary = [
        {"timestamp": f"2026-08-0{day}T00:00:00+00:00", "point": 10.0}
        for day in range(2, 6)
    ]
    for _ in range(2):
        store.record_effect_scenarios(
            "capacity", "forecast-1", "api", primary, [_scenario()], [Event()]
        )
    rows = store.event_effects("capacity")
    assert len(rows) == 1
    json.dumps(rows)  # public query result stays JSON serialisable


def test_realised_effect_removes_bias_seen_outside_event_window(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    primary = [
        {"timestamp": f"2026-08-0{day}T00:00:00+00:00", "point": 10.0}
        for day in range(2, 6)
    ]
    store.record_effect_scenarios(
        "capacity", "forecast-1", "api", primary, [_scenario()], [Event()]
    )
    # Baseline is two units low everywhere; the event adds five more units.
    store._resolve_effect_scenarios(_record(), [12.0, 17.0, 17.0, 12.0])
    row = store.event_effects("capacity", include_unresolved=False)[0]
    assert row["baseline_bias"] == 2.0
    assert row["estimate"] == 5.0


def test_same_artifact_can_be_registered_in_two_projects(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    primary = [
        {"timestamp": f"2026-08-0{day}T00:00:00+00:00", "point": 10.0}
        for day in range(2, 6)
    ]
    for project in ("capacity-a", "capacity-b"):
        store.record_effect_scenarios(
            project, "forecast-1", "api", primary, [_scenario()], [Event()]
        )
    assert len(store.event_effects("capacity-a")) == 1
    assert len(store.event_effects("capacity-b")) == 1


def test_occurrence_must_be_confirmed_before_effect_is_learnable(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    primary = [
        {"timestamp": f"2026-08-0{day}T00:00:00+00:00", "point": 10.0}
        for day in range(2, 6)
    ]
    store.record_effect_scenarios(
        "capacity", "forecast-1", "api", primary, [_scenario()], [Event()]
    )
    store._resolve_effect_scenarios(_record(), [10.0, 14.0, 16.0, 10.0])
    row = store.event_effects("capacity")[0]
    assert row["occurrence_status"] == "unverified"
    assert row["eligible_for_learning"] is False

    store.record_effect_occurrence(
        row["effect_id"], "confirmed",
        known_at="2026-08-06T03:00:00+00:00", note="deployment audit log",
    )
    confirmed = store.event_effects("capacity")[0]
    assert confirmed["occurrence_status"] == "confirmed"
    assert confirmed["occurrence_note"] == "deployment audit log"
    assert confirmed["eligible_for_learning"] is True


def test_cancelled_or_confounded_effect_is_never_learnable(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    primary = [
        {"timestamp": f"2026-08-0{day}T00:00:00+00:00", "point": 10.0}
        for day in range(2, 6)
    ]
    store.record_effect_scenarios(
        "capacity", "forecast-1", "api", primary, [_scenario()], [Event()]
    )
    effect_id = store.event_effects("capacity")[0]["effect_id"]
    store.record_effect_occurrence(
        effect_id, "cancelled", known_at="2026-08-03T00:00:00+00:00")
    store._resolve_effect_scenarios(_record(), [10.0, 14.0, 16.0, 10.0])
    assert store.event_effects("capacity")[0]["eligible_for_learning"] is False


def test_occurrence_rejects_ambiguous_time_and_unknown_effect(tmp_path):
    from gnomon.contracts import GnomonError

    store = TrackingStore(tmp_path / "registry.db")
    with pytest.raises(GnomonError, match="timezone"):
        store.record_effect_occurrence(
            "missing", "confirmed", known_at="2026-08-01T00:00:00")
    with pytest.raises(GnomonError, match="No tracked effect"):
        store.record_effect_occurrence(
            "missing", "confirmed", known_at="2026-08-01T00:00:00+00:00")


def test_hierarchical_pool_is_leave_one_event_out_validated_and_as_of_safe(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    primary = [
        {"timestamp": f"2026-08-0{day}T00:00:00+00:00", "point": 10.0}
        for day in range(2, 6)
    ]
    for index, realised in enumerate((4.0, 5.0, 6.0, 5.0, 4.5, 5.5), 1):
        forecast_id = f"forecast-{index}"
        event_id = f"deploy-{index}"
        event = type("Event", (), {"event_id": event_id,
                                    "event_type": "deployment"})()
        scenario = _scenario(event_id, 5.0)
        store.record_effect_scenarios(
            "capacity", forecast_id, "api" if index <= 4 else "worker",
            primary, [scenario], [event],
        )
        record = ForecastRecord(
            forecast_id, "capacity", "api" if index <= 4 else "worker",
            "2026-08-01T00:00:00+00:00", 4, "D", "seasonal_naive",
            "supported", None, None, 1.0, "",
            "2026-08-01T00:00:00+00:00",
        )
        store._resolve_effect_scenarios(
            record, [10.0, 10.0 + realised, 10.0 + realised, 10.0],
            outcome_known_at=f"2026-08-{10 + index:02d}T00:00:00+00:00",
        )
        effect_id = next(
            row["effect_id"] for row in store.event_effects(
                "capacity", series=record.series)
            if row["forecast_id"] == forecast_id
        )
        store.record_effect_occurrence(
            effect_id, "confirmed",
            known_at=f"2026-08-{10 + index:02d}T00:00:00+00:00",
        )

    prior = store.pooled_effect_prior(
        "capacity", "deployment", "api", as_of="2026-08-31T00:00:00+00:00")
    assert prior["status"] == "validated"
    assert prior["eligible_for_scenario"] is True
    assert prior["validation"]["episodes"] == 6
    assert prior["validation"]["beats_zero_effect"] is True
    assert prior["pooling"]["local_sample_count"] == 4
    assert 4.0 < prior["distribution"]["location"] < 6.0

    early = store.pooled_effect_prior(
        "capacity", "deployment", "api", as_of="2026-08-12T00:00:00+00:00")
    assert early["status"] == "insufficient_evidence"
