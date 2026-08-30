from gnomon.agent_response import (
    build_agent_response_contract,
    verify_agent_response_contract,
)


def _payload():
    return {
        "forecast_id": "forecast:test",
        "artifact_path": "/tmp/artifact",
        "results": [{
            "series": "value",
            "support": "supported",
            "warnings": [{
                "message": "Measured interval coverage is indicative, not a calibration guarantee."
            }],
            "context_outcome": {
                "status": "rejected",
                "canonical_primary_preserved": True,
                "automation_eligible": False,
                "failed_gate_codes": ["separated_model_folds_available"],
                "relationship_to_primary": "no_distinct_numeric_path",
                "selected_output_role": "primary_forecast_already_noncontinuing",
                "context_evidence": [{
                    "source": {"reference": "calendar:change-17"}
                }],
            },
            "sensitivity_scenarios": [{
                "consequence_summary": "Conditional path remains distinct."
            }],
        }],
        "publication": {
            "publication_seal_sha256": "abc",
            "primary_forecast_unchanged": True,
            "context_summary": {"context_can_authorize_automation": False},
            "context_dispositions": [],
        },
    }


def test_response_contract_collects_exact_agent_obligations():
    payload = _payload()
    contract = build_agent_response_contract(payload)
    assert contract is not None
    series = contract["series"][0]
    assert series["canonical_primary_preserved"] is True
    assert series["context_automation_eligible"] is False
    assert series["failed_gate_codes"] == ["separated_model_folds_available"]
    assert series["source_references"] == ["calendar:change-17"]
    assert series["relationship_to_primary"] == "no_distinct_numeric_path"
    assert series["scenario_consequence_count"] == 1
    assert series["required_facts"] == [
        "primary_preservation", "context_automation_limit",
        "failed_gate_codes", "source_references", "scenario_consequences",
        "typed_primary_relationship", "interval_limitations"]
    assert verify_agent_response_contract(payload, contract)


def test_response_contract_tampering_fails_verification():
    payload = _payload()
    contract = build_agent_response_contract(payload)
    assert contract is not None
    contract["series"][0]["context_automation_eligible"] = True
    assert not verify_agent_response_contract(payload, contract)


def test_response_contract_deduplicates_publication_obligations():
    payload = _payload()
    payload["publication"]["context_dispositions"] = [{
        "disposition": "scenario",
        "reason_code": "separated_model_folds_available",
        "source_evidence": {"source": {"reference": "calendar:change-17"}},
    }]
    contract = build_agent_response_contract(payload)
    assert contract is not None
    assert contract["series"][0]["failed_gate_codes"] == [
        "separated_model_folds_available"]
    assert contract["series"][0]["source_references"] == [
        "calendar:change-17"]
