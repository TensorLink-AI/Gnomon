from gnomon.agent_response import (
    build_agent_response_contract,
    verify_agent_decision_selection,
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
    assert series["required"] == "all_emitted_fields"
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


def _decision_payload(support="weak", value="upward"):
    return {
        "artifact_path": "/sealed/forecast",
        "answer_receipt": "/sealed/forecast/temporal_answers.json",
        "results": [{"series": "value", "support": support}],
        "answers": [{
            "artifact_id": "forecast:sealed",
            "question": {"id": "trend", "verb": "predict",
                         "property": "trend", "target": "value",
                         "horizon": 12},
            "best_estimate": {
                "value": value, "support": support,
                "automation_eligible": False,
            },
            "answer": {
                "support": support, "automation_eligible": False,
                "interval": [-0.2, 0.4] if value is not None else None,
                "reasoning": {
                    "primary_forecast_unchanged": True,
                    "packet": {
                        "sufficiency": ("insufficient" if support == "abstained"
                                        else "mixed"),
                        "selector": ("gnomon_canonical"
                                     if support == "supported" else "model"),
                        "interpretations": ([{
                            "value": "upward", "support": support,
                            "compatible": True,
                            "decision_eligible": True,
                            "supporting": ["fitted_executable"],
                            "conflicting": ["observed_transition"],
                        }, {
                            "value": "downward", "support": "supported",
                            "compatible": True,
                            "decision_eligible": True,
                            "supporting": ["observed_transition"],
                            "conflicting": ["fitted_executable"],
                        }] if value is not None else []),
                    },
                },
            },
            "calibration_status": {
                "available": value is not None,
                "applicable": value is not None,
            },
            "context_assessment": {
                "relationship_to_primary": "no_distinct_numeric_path",
                "canonical_primary_preserved": True,
            },
        }],
    }


def test_decision_contract_is_complete_and_sealed():
    payload = _decision_payload()
    contract = build_agent_response_contract(payload)
    decision = contract["decisions"][0]
    assert decision["conclusion"] == "upward"
    assert decision["support"] == "weak"
    assert decision["authority"] == "advisory"
    assert decision["decision_eligible"] is True
    assert decision["automation_eligible"] is False
    assert decision["relationship_to_primary"] == "no_distinct_numeric_path"
    assert decision["uncertainty"]["interval_status"] == "present"
    assert decision["provenance"] == {
        "artifact_id": "forecast:sealed", "question_id": "trend",
        "answer_receipt": "/sealed/forecast/temporal_answers.json",
    }
    assert verify_agent_response_contract(payload, contract)


def test_agent_response_contract_exposes_weakest_row_tier():
    payload = {
        "results": [{
            "series": "requests",
            "support": "degraded",
            "support_assessment": {"status": "conditionally_supported"},
            "forecast": [
                {"timestamp": "2026-01-01", "tier": "conditionally_supported"},
                {"timestamp": "2026-01-02", "tier": "best_effort"},
            ],
            "context_outcome": {"status": "not_used"},
        }],
    }

    contract = build_agent_response_contract(payload)
    assert contract is not None
    assert contract["series"][0]["support"] == "degraded"
    assert contract["series"][0]["tier_floor"] == "best_effort"


def test_agent_response_contract_separates_gates_from_dispositions():
    payload = {
        "results": [{
            "series": "api",
            "support": "weakly_supported",
            "forecast": [{"tier": "conditionally_supported"}],
            "context_outcome": {
                "status": "rejected",
                "failed_gate_codes": [
                    "emitted_trend_is_directionally_stable"],
                "canonical_primary_preserved": True,
                "automation_eligible": False,
            },
        }],
        "publication": {
            "context_dispositions": [{
                "disposition": "rejected",
                "reason_code": "rejected",
            }],
        },
    }

    contract = build_agent_response_contract(payload)

    assert contract is not None
    [series] = contract["series"]
    assert series["failed_gate_codes"] == [
        "emitted_trend_is_directionally_stable"]
    assert series["disposition_reason_codes"] == ["rejected"]


def test_decision_verifier_requires_support_for_a_weak_override():
    contract = build_agent_response_contract(_decision_payload())
    assert verify_agent_decision_selection(
        contract, "trend", {"value": "upward"}) == []
    assert verify_agent_decision_selection(
        contract, "trend", {"value": "downward"})[0]["code"] == \
        "DECISION_EVIDENCE_REQUIRED"
    assert verify_agent_decision_selection(
        contract, "trend", {"value": "downward",
                             "cited_evidence": ["fitted_executable"]}
    )[0]["code"] == "DECISION_EVIDENCE_UNSUPPORTED"
    assert verify_agent_decision_selection(
        contract, "trend", {"value": "downward",
                             "cited_evidence": ["observed_transition"]}) == []


def test_decision_verifier_preserves_binding_and_abstention():
    binding = build_agent_response_contract(
        _decision_payload(support="supported"))
    assert verify_agent_decision_selection(
        binding, "trend", {"value": "downward",
                            "cited_evidence": ["observed_transition"]}
    )[0]["code"] == "DECISION_OVERRIDES_BINDING"

    abstained = build_agent_response_contract(
        _decision_payload(support="abstained", value=None))
    decision = abstained["decisions"][0]
    assert decision["value_status"] == "abstention"
    assert decision["decision_eligible"] is False
    assert verify_agent_decision_selection(
        abstained, "trend", {"value": "Uncertain"}) == []
    assert verify_agent_decision_selection(
        abstained, "trend", {"value": "upward"}
    )[0]["code"] == "DECISION_OVERRIDES_ABSTENTION"


def test_inference_scoped_nonbinding_authority_survives_projection():
    payload = _decision_payload(support="supported")
    reasoning = payload["answers"][0]["answer"]["reasoning"]
    packet = reasoning["packet"]
    packet.pop("selector")
    packet["selection_contract"] = {
        "selector": "model",
        "canonical": {
            "value": "upward", "support": "supported",
            "role": "default_not_command",
        },
        "inference_authority": {
            "mode": "predictive",
            "requirements_satisfied": False,
            "missing_evidence": ["rolling_origin_property_fit"],
        },
    }

    contract = build_agent_response_contract(payload)
    decision = contract["decisions"][0]
    assert decision["support"] == "supported"
    assert decision["authority"] == "advisory"
    assert decision["selector"] == "model"
    assert decision["inference_mode"] == "predictive"
    assert decision["inference_requirements_satisfied"] is False
    assert verify_agent_decision_selection(
        contract, "trend", {
            "value": "downward",
            "cited_evidence": ["observed_transition"],
        }) == []
