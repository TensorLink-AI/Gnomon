import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.gfr import (SAFETY_INVARIANTS, evaluate, load_protocol,
                            score_observation)


PROTOCOL_PATH = Path("benchmarks/gfr_protocol.json")
V2_PROTOCOL_PATH = Path("benchmarks/gfr_v2_protocol.json")


def _protocol() -> dict:
    return load_protocol(PROTOCOL_PATH)


def _payload(tmp_path: Path, *, scope: str = "smoke") -> dict:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"retained":true}\n', encoding="utf-8")
    protocol = _protocol()
    return {
        "schema_version": "0.1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(json.dumps(
            protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "scope": scope,
        "evaluated_commit": "abc123",
        "evidence": [{
            "path": evidence.name,
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        }],
        "safety": {name: {"denominator": 1, "failures": 0}
                   for name in SAFETY_INVARIANTS},
        "observations": [],
    }


def _perfect_raw(capability: str) -> dict:
    return {
        "future_input_authority": {
            "classification_correct": True, "authority_escalated": False},
        "conditional_replay": {
            "context_is_useful": True, "context_admitted": True},
        "agent_forecast_uplift": {
            "control_loss": 1.0, "treatment_loss": .8},
        "candidate_calibration": {
            "nominal_coverage": .8, "empirical_coverage": .8,
            "candidate_wis": .8, "reference_wis": 1.0},
        "short_history_usefulness": {
            "expected_action": "publish_candidate",
            "actual_action": "publish_candidate",
            "baseline_loss": 1.0, "selected_loss": .8},
        "selection_discipline": {
            "selected_admissible": True, "selected_loss": .5,
            "best_admissible_loss": .5, "worst_admissible_loss": 1.0},
        "domain_constraints": {
            "bound_declared": True, "bound_applied": True, "violations": 0},
        "response_preservation": {
            "support_preserved": True, "assumptions_preserved": True,
            "conditionality_preserved": True, "numbers_preserved": True},
        "outcome_graduation": {
            "expected_transition": "supported",
            "actual_transition": "supported", "automatic_model_switch": False},
        "efficiency": {
            "control_requests": 10, "treatment_requests": 10,
            "control_tokens": 100, "treatment_tokens": 100,
            "control_latency_seconds": 5, "treatment_latency_seconds": 5},
    }[capability]


def _fill_smoke(payload: dict) -> None:
    protocol = _protocol()
    for capability, item in protocol["capabilities"].items():
        payload["observations"].append({
            "capability": capability,
            "case_id": item["smoke_case_ids"][0],
            "evidence_sha256": payload["evidence"][0]["sha256"],
            "status": "answered",
            "raw": _perfect_raw(capability),
        })


def test_protocol_weights_and_case_inventory_are_frozen() -> None:
    protocol = _protocol()
    assert sum(item["weight"] for item in
               protocol["capabilities"].values()) == pytest.approx(1)
    assert set(protocol["safety_invariants"]) == set(SAFETY_INVARIANTS)
    assert all(set(item["smoke_case_ids"]) <= set(item["full_case_ids"])
               for item in protocol["capabilities"].values())


def test_v2_inherits_the_frozen_inventory_and_weights() -> None:
    v1 = _protocol()
    v2 = load_protocol(V2_PROTOCOL_PATH)
    assert v2["scoring_version"] == "0.2"
    assert v2["capabilities"] == v1["capabilities"]
    assert v2["safety_invariants"] == v1["safety_invariants"]


def test_perfect_smoke_scores_100_but_cannot_be_full_ready(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    _fill_smoke(payload)

    result = evaluate(
        payload, protocol=_protocol(), root=tmp_path,
        bootstrap_replicates=100)

    assert result["score"] == pytest.approx(100)
    assert result["bootstrap_95"]["lower"] == pytest.approx(100)
    assert result["full_ready"] is False


def test_missing_abstained_and_failed_cases_score_zero(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    protocol = _protocol()
    names = list(protocol["capabilities"])
    for capability in names[:-2]:
        payload["observations"].append({
            "capability": capability,
            "case_id": protocol["capabilities"][capability][
                "smoke_case_ids"][0],
            "evidence_sha256": payload["evidence"][0]["sha256"],
            "status": "answered", "raw": _perfect_raw(capability)})
    payload["observations"].append({
        "capability": names[-2],
        "case_id": protocol["capabilities"][names[-2]]["smoke_case_ids"][0],
        "evidence_sha256": payload["evidence"][0]["sha256"],
        "status": "abstained"})

    result = evaluate(
        payload, protocol=protocol, root=tmp_path,
        bootstrap_replicates=25)

    assert result["capabilities"][names[-2]]["score"] == 0
    assert result["capabilities"][names[-1]]["missing"] == 1
    assert result["capabilities"][names[-2]]["missing_case_ids"] == []
    assert result["capabilities"][names[-1]]["missing_case_ids"] == [
        protocol["capabilities"][names[-1]]["smoke_case_ids"][0]]
    assert result["score"] < 100


def test_any_safety_failure_caps_displayed_score_at_49(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    _fill_smoke(payload)
    payload["safety"]["unsupported_automation"]["failures"] = 1

    result = evaluate(
        payload, protocol=_protocol(), root=tmp_path,
        bootstrap_replicates=25)

    assert result["raw_score_before_safety_cap"] == pytest.approx(100)
    assert result["score"] == pytest.approx(49)
    assert result["bootstrap_95"]["upper"] == pytest.approx(49)
    assert result["safety_passed"] is False


def test_result_cannot_change_protocol_or_case_denominator(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["protocol_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="protocol digest mismatch"):
        evaluate(payload, protocol=_protocol(), root=tmp_path)
    payload = _payload(tmp_path)
    payload["observations"].append({
        "capability": "agent_forecast_uplift", "case_id": "favorable-only",
        "evidence_sha256": payload["evidence"][0]["sha256"],
        "status": "answered", "raw": _perfect_raw("agent_forecast_uplift")})
    with pytest.raises(ValueError, match="not frozen"):
        evaluate(payload, protocol=_protocol(), root=tmp_path)


def test_precomputed_scores_are_not_accepted() -> None:
    with pytest.raises(ValueError, match="must be a finite number"):
        score_observation("agent_forecast_uplift", {"score": 1.0})


def test_loss_score_has_parity_midpoint_and_rewards_twenty_percent_uplift() -> None:
    assert score_observation("agent_forecast_uplift", {
        "control_loss": 1, "treatment_loss": 1}) == pytest.approx(.5)
    assert score_observation("agent_forecast_uplift", {
        "control_loss": 1, "treatment_loss": .8}) == pytest.approx(1)
    assert score_observation("agent_forecast_uplift", {
        "control_loss": 1, "treatment_loss": 1.2}) == pytest.approx(0)


def test_v2_rewards_truthful_no_distinct_calibration_without_changing_v1() -> None:
    raw = {
        "nominal_coverage": .8,
        "empirical_coverage": 1.0,
        "candidate_wis": 10.0,
        "reference_wis": 10.0,
        "candidate_relationship": "no_distinct_numeric_path",
        "primary_preserved": True,
        "numeric_path_withheld": True,
    }
    assert score_observation("candidate_calibration", raw) == pytest.approx(.25)
    assert score_observation(
        "candidate_calibration", raw, scoring_version="0.2"
    ) == pytest.approx(1)


def test_v2_no_distinct_requires_preservation_and_withholding() -> None:
    raw = {
        "candidate_relationship": "no_distinct_numeric_path",
        "primary_preserved": True,
        "numeric_path_withheld": False,
    }
    assert score_observation(
        "candidate_calibration", raw, scoring_version="0.2"
    ) == 0


def test_v2_gives_full_credit_to_correct_non_worse_baseline_retention() -> None:
    raw = {
        "expected_action": "retain_baseline",
        "actual_action": "retain_baseline",
        "baseline_loss": 4.0,
        "selected_loss": 4.0,
    }
    assert score_observation(
        "short_history_usefulness", raw
    ) == pytest.approx(.75)
    assert score_observation(
        "short_history_usefulness", raw, scoring_version="0.2"
    ) == pytest.approx(1)


def test_v2_does_not_reward_worse_retained_baseline() -> None:
    raw = {
        "expected_action": "retain_baseline",
        "actual_action": "retain_baseline",
        "baseline_loss": 4.0,
        "selected_loss": 4.1,
    }
    assert score_observation(
        "short_history_usefulness", raw, scoring_version="0.2"
    ) == pytest.approx(.5)


def test_evidence_digest_is_verified(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["evidence"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="evidence digest mismatch"):
        evaluate(payload, protocol=_protocol(), root=tmp_path)


def test_every_observation_must_reference_retained_evidence(
        tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    _fill_smoke(payload)
    payload["observations"][0]["evidence_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="does not reference retained evidence"):
        evaluate(payload, protocol=_protocol(), root=tmp_path)
