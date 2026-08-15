from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from gnomon.effect_registry import load_effect_registry, resolve_effect_prior


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


def test_registry_is_versioned_validated_and_precision_pooled(tmp_path):
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


def test_registry_prefers_specific_evidence_and_rejects_expiry(tmp_path):
    path = tmp_path / "effects.json"
    path.write_text(json.dumps({"schema_version": "0.1", "priors": [
        _prior("generic", 99.0, 1.0, target="*", domain="*", population="*", unit="*"),
        _prior("specific", 7.0, 1.0),
    ]}), encoding="utf-8")
    priors = load_effect_registry(path, now=datetime(2026, 8, 14, tzinfo=timezone.utc))
    assert resolve_effect_prior(
        priors, event_type="deploy", target="latency", domain="operations",
        population="api", unit="ms")["mean"] == 7.0
    path.write_text(json.dumps({"schema_version": "0.1", "priors": [
        _prior("expired", 1.0, 1.0, expires_at="2025-01-01T00:00:00+00:00")
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="expired"):
        load_effect_registry(path, now=datetime(2026, 8, 14, tzinfo=timezone.utc))
