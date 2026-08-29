import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.scorecard import validate_payload


def _evidence(tmp_path: Path) -> list[dict[str, str]]:
    path = tmp_path / "evidence.json"
    path.write_text('{"retained": true}\n', encoding="utf-8")
    return [{
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }]


def _scorecard(tmp_path: Path) -> dict:
    evidence = _evidence(tmp_path)
    layers = {}
    for name in ("output", "reasoning", "topology"):
        layers[name] = {
            "status": "pass",
            "evaluated_commit": "abc123",
            "dataset_identity": f"{name}-fixture-v1",
            "configuration_identity": "fixture-config-v1",
            "accounting": {
                "expected": 2,
                "completed": 2,
                "answered": 1,
                "abstained": 1,
                "failed": 0,
            },
            "metrics": [{
                "name": f"{name}_rate",
                "value": 1.0,
                "unit": "rate",
                "denominator": 2,
                "gate": {"operator": "gte", "threshold": 1.0,
                         "passed": True},
            }],
            "evidence": evidence,
        }
    return {
        "schema_version": "0.1",
        "scorecard_id": "fixture",
        "scorecard_commit": "abc123",
        "dirty_tree": True,
        "scope": "smoke",
        "layers": layers,
        "invariants": {
            "no_temporal_leakage": {"status": "pass", "evidence": evidence},
        },
        "decision": {"status": "continue", "reason": "Smoke validation."},
    }


def test_valid_scorecard_keeps_layers_separate(tmp_path: Path) -> None:
    result = validate_payload(_scorecard(tmp_path), root=tmp_path)
    assert result["decision"] == "continue"
    assert result["layers"] == {
        "output": {"complete": True, "passed": True},
        "reasoning": {"complete": True, "passed": True},
        "topology": {"complete": True, "passed": True},
    }


def test_missing_layer_fails_closed(tmp_path: Path) -> None:
    payload = _scorecard(tmp_path)
    del payload["layers"]["reasoning"]
    with pytest.raises(ValueError, match="exactly output, reasoning, topology"):
        validate_payload(payload, root=tmp_path)


def test_metric_requires_visible_denominator(tmp_path: Path) -> None:
    payload = _scorecard(tmp_path)
    del payload["layers"]["output"]["metrics"][0]["denominator"]
    with pytest.raises(ValueError, match="denominator must be an integer"):
        validate_payload(payload, root=tmp_path)


def test_metric_denominator_cannot_exceed_layer_population(
        tmp_path: Path) -> None:
    payload = _scorecard(tmp_path)
    payload["layers"]["output"]["metrics"][0]["denominator"] = 3
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_payload(payload, root=tmp_path)


def test_gate_cannot_claim_pass_against_its_value(tmp_path: Path) -> None:
    payload = _scorecard(tmp_path)
    payload["layers"]["reasoning"]["metrics"][0]["value"] = 0.5
    with pytest.raises(ValueError, match="contradicts value and threshold"):
        validate_payload(payload, root=tmp_path)


def test_accounting_keeps_abstentions_and_failures_visible(tmp_path: Path) -> None:
    payload = _scorecard(tmp_path)
    payload["layers"]["topology"]["accounting"]["completed"] = 1
    with pytest.raises(ValueError, match=r"answered \+ abstained \+ failed"):
        validate_payload(payload, root=tmp_path)


def test_evidence_digest_is_verified(tmp_path: Path) -> None:
    payload = _scorecard(tmp_path)
    payload["layers"]["output"]["evidence"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="evidence digest mismatch"):
        validate_payload(payload, root=tmp_path)


def test_evidence_path_cannot_escape_retained_root(tmp_path: Path) -> None:
    payload = _scorecard(tmp_path)
    for layer in payload["layers"].values():
        layer["evidence"][0]["path"] = "../evidence.json"
    payload["invariants"]["no_temporal_leakage"]["evidence"][0]["path"] = (
        "../evidence.json")
    with pytest.raises(ValueError, match="within the evidence root"):
        validate_payload(payload, root=tmp_path)


def test_smoke_scorecard_cannot_promote(tmp_path: Path) -> None:
    payload = _scorecard(tmp_path)
    payload["decision"] = {"status": "promote", "reason": "Not enough."}
    with pytest.raises(ValueError, match="only a full scorecard"):
        validate_payload(payload, root=tmp_path)


def test_promotion_requires_clean_tree_and_passing_invariants(
        tmp_path: Path) -> None:
    payload = _scorecard(tmp_path)
    payload["scope"] = "full"
    payload["decision"] = {"status": "promote", "reason": "All gates pass."}
    with pytest.raises(ValueError, match="dirty-tree"):
        validate_payload(payload, root=tmp_path)
    payload["dirty_tree"] = False
    payload["invariants"]["no_temporal_leakage"]["status"] = "fail"
    with pytest.raises(ValueError, match="failing invariant"):
        validate_payload(payload, root=tmp_path)
