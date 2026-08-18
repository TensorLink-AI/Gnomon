from __future__ import annotations

from datetime import datetime, timezone

from gnomon.effect_registry import prior_from_dict
from gnomon.effect_resolution import resolve_effect_evidence
from gnomon.tracking import TrackingStore


def _external(mean: float = 7.0, *, known_at: str | None = None):
    return prior_from_dict({
        "prior_id": "external-1", "event_type": "deploy",
        "target": "latency", "domain": "operations", "population": "api",
        "unit": "ms", "effect_family": "temporary_pulse", "mean": mean,
        "standard_error": 1.0, "delay_steps": [0, 1],
        "duration_steps": [1, 3], "sample_size": 40,
        "source": "governed-study", "version": "2026-08",
        **({"known_at": known_at} if known_at else {}),
    })


def test_external_prior_requires_knowledge_time_for_governed_resolution(tmp_path):
    result = resolve_effect_evidence(
        TrackingStore(tmp_path / "registry.db"), project="p",
        event_type="deploy", series="api",
        as_of="2026-08-20T00:00:00+00:00", external_priors=[_external()],
        target="latency", domain="operations", population="api", unit="ms",
    )
    assert result["status"] == "insufficient_evidence"


def test_external_prior_beats_human_assumption_without_blending(tmp_path):
    result = resolve_effect_evidence(
        TrackingStore(tmp_path / "registry.db"), project="p",
        event_type="deploy", series="api",
        as_of="2026-08-20T00:00:00+00:00",
        external_priors=[_external(7.0, known_at="2026-08-10T00:00:00+00:00")],
        target="latency", domain="operations", population="api", unit="ms",
        human_assumption={
            "location": -20.0, "known_at": "2026-08-15T00:00:00+00:00",
            "unit": "ms", "source": "operator", "shape": "temporary_pulse",
        },
    )
    assert result["selected_lane"] == "external_prior"
    assert result["effect"]["distribution"]["location"] == 7.0
    assert result["effect"]["provenance"]["provenance_class"] == "external_prior"
    assert result["conflicts"][0]["conflicts_with"] == "human_assumption"


def test_human_assumption_is_explicitly_non_probabilistic(tmp_path):
    result = resolve_effect_evidence(
        TrackingStore(tmp_path / "registry.db"), project="p",
        event_type="novel", series="api",
        as_of="2026-08-20T00:00:00+00:00",
        human_assumption={
            "location": 3.0, "known_at": "2026-08-15T00:00:00+00:00",
            "unit": "requests", "source": "planning meeting",
        },
    )
    assert result["selected_lane"] == "human_assumption"
    assert result["effect"]["distribution"]["distribution"] == "assumption"
    assert result["effect"]["distribution"]["interval_probability"] is None
    assert result["effect"]["provenance"]["observed"] is False
