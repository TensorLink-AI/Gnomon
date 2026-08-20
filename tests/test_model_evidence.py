import json

import pytest

from gnomon.model_evidence import (
    EvidenceRegistryError, ModelEvidenceRegistry, TemporalRegime,
    describe_regime,
)


def _regime():
    return TemporalRegime("subdaily", "1_to_2", "moderate", "low", "low", "stable")


def test_registry_requires_exact_revision_and_regime(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({
        "version": "forge-7",
        "priors": [{
            "model": "toto2_4m", "revision": "abc",
            "regime": dict(_regime().items()), "comparisons": 300,
            "mean_relative_gain": .12, "standard_error": .03,
            "source_ids": ["leave-domain-out-1"], "overlap_risk": "low",
        }],
    }))
    registry = ModelEvidenceRegistry.load(path)
    assert registry.lookup("toto2_4m", "abc", _regime()).registry_version == "forge-7"
    assert registry.lookup("toto2_4m", "different", _regime()) is None


def test_registry_refuses_unversioned_document(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text('{"priors": []}')
    with pytest.raises(EvidenceRegistryError, match="version"):
        ModelEvidenceRegistry.load(path)


def test_regime_projection_uses_only_series_shape_not_domain_identity():
    regime = describe_regime(
        [10 + index * .1 for index in range(60)], 24, 24, "h")
    assert regime.frequency_class == "subdaily"
    assert regime.history_horizon_ratio == "2_to_4"
    assert not hasattr(regime, "channel")


def test_explicit_global_registry_row_is_a_fallback(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({
        "version": "r1", "priors": [{
            "model": "m", "revision": "m@1",
            "regime": {key: "*" for key, _ in _regime().items()},
            "comparisons": 100, "mean_relative_gain": .1,
            "standard_error": .02, "source_ids": ["held-out"],
            "overlap_risk": "low",
        }]}))
    match = ModelEvidenceRegistry.load(path).lookup("m", "m@1", _regime())
    assert match
    assert match.relevance == .1


def test_registry_reports_fraction_of_exact_regime_dimensions(tmp_path):
    regime = dict(_regime().items())
    for key in ("seasonality_strength", "trend_strength",
                "scale_stability"):
        regime[key] = "*"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({
        "version": "r2", "priors": [{
            "model": "m", "revision": "m@2", "regime": regime,
            "comparisons": 100, "mean_relative_gain": .1,
            "standard_error": .02, "source_ids": ["held-out"],
            "overlap_risk": "low",
        }]}))
    match = ModelEvidenceRegistry.load(path).lookup("m", "m@2", _regime())
    assert match is not None
    assert match.relevance == .5
