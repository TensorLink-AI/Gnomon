from __future__ import annotations

import pytest

from gnomon.contracts import GnomonError
from gnomon.temporal_intent import compile_temporal_text, compile_temporal_text_receipt


class Adapter:
    def __init__(self, response):
        self.response = response
        self.prompt = None

    def complete(self, prompt, response_schema):
        self.prompt = prompt
        assert response_schema["required"] == ["status", "questions"]
        return self.response


def test_llm_can_propose_but_deterministic_compiler_binds_scope() -> None:
    adapter = Adapter({"status": "compiled", "questions": [{
        "id": "fleet", "verb": "predict", "property": "volatility",
        "target": {"kind": "aggregate", "members": ["cpu", "mem"],
                   "aggregation": "median_normalized_scale_ratio"},
        "horizon": 12}]})
    result = compile_temporal_text(
        "Will the fleet become more volatile?", available_targets=["cpu", "mem"],
        adapter=adapter)
    assert result[0].scope == "aggregate"
    assert "benchmark" not in adapter.prompt.lower()
    assert "one volatility-change answer" in adapter.prompt
    assert "comparison windows, never members of pair scope" in adapter.prompt


def test_host_resolved_horizon_is_part_of_compiler_instruction() -> None:
    adapter = Adapter({"status": "compiled", "questions": [{
        "id": "v", "verb": "predict", "property": "volatility",
        "target": "cpu", "horizon": 29}]})
    result = compile_temporal_text(
        "Volatility over the forecast horizon?", available_targets=["cpu"],
        adapter=adapter, default_horizon=29)
    assert result[0].horizon == 29
    assert "host-resolved forecast horizon is 29 periods" in adapter.prompt


def test_descriptive_zero_horizon_is_canonicalized_to_unspecified() -> None:
    adapter = Adapter({"status": "compiled", "questions": [{
        "id": "level", "verb": "describe", "property": "level",
        "target": "cpu", "measure": "point", "horizon": 0}]})
    result = compile_temporal_text(
        "What is CPU at now?", available_targets=["cpu"], adapter=adapter)
    assert result[0].target == "cpu"
    assert result[0].horizon is None


def test_predictive_zero_horizon_remains_invalid() -> None:
    adapter = Adapter({"status": "compiled", "questions": [{
        "id": "future", "verb": "predict", "property": "volatility",
        "target": "cpu", "horizon": 0}]})
    with pytest.raises(GnomonError):
        compile_temporal_text(
            "Will CPU become more volatile?", available_targets=["cpu"],
            adapter=adapter)


def test_compiled_status_cannot_silently_contain_no_question() -> None:
    adapter = Adapter({"status": "compiled", "questions": []})
    with pytest.raises(GnomonError) as raised:
        compile_temporal_text(
            "What is CPU doing?", available_targets=["cpu"], adapter=adapter)
    assert raised.value.details["compiler_status"] == "malformed"


def test_llm_proposal_cannot_invent_target_or_aggregation() -> None:
    adapter = Adapter({"status": "compiled", "questions": [{
        "property": "volatility", "target": {
            "kind": "aggregate", "members": ["cpu", "invented"],
            "aggregation": "raw_average"}}]})
    with pytest.raises(GnomonError):
        compile_temporal_text("Is the fleet noisy?", available_targets=["cpu"],
                              adapter=adapter)


def test_receipt_preserves_valid_sibling_when_one_question_is_invalid() -> None:
    adapter = Adapter({"status": "compiled", "questions": [
        {"id": "bad", "verb": "compare", "property": "level",
         "target": {"kind": "aggregate", "members": ["cpu", "mem"],
                    "aggregation": "median_normalized_scale_ratio"}},
        {"id": "good", "verb": "predict", "property": "volatility",
         "target": {"kind": "aggregate", "members": ["cpu", "mem"]}},
    ]})
    receipt = compile_temporal_text_receipt(
        "level and volatility", available_targets=["cpu", "mem"],
        adapter=adapter)
    assert [item.id for item in receipt["accepted"]] == ["good"]
    assert receipt["rejected"][0]["proposal"]["id"] == "bad"


def test_receipt_normalizes_json_encoded_questions_before_sibling_validation() -> None:
    adapter = Adapter({
        "status": "compiled",
        "questions": '[{"id":"v","verb":"predict",'
                     '"property":"volatility","target":"cpu"}]',
    })
    receipt = compile_temporal_text_receipt(
        "Will CPU become more volatile?", available_targets=["cpu"],
        adapter=adapter)
    assert [item.id for item in receipt["accepted"]] == ["v"]
    assert receipt["rejected"] == []


def test_omitted_target_inherits_nearest_explicit_discourse_focus() -> None:
    adapter = Adapter({"status": "compiled", "questions": [
        {"id": "q1", "verb": "compare", "property": "level",
         "target": "heart_rate"},
        {"id": "q2", "verb": "compare", "property": "volatility",
         "target": {"kind": "aggregate", "members": ["heart_rate", "spo2"]}},
        {"id": "q3", "verb": "compare", "property": "seasonality",
         "target": {"kind": "aggregate", "members": ["heart_rate", "spo2"]}},
    ]})
    receipt = compile_temporal_text_receipt(
        "Median heart rate change?\nVolatility change?\nSeasonality alignment?",
        available_targets=["heart_rate", "spo2"], adapter=adapter)
    assert receipt["accepted"][0].target == "heart_rate"
    assert receipt["accepted"][1].scope == "series"
    assert receipt["accepted"][2].scope == "series"
    assert receipt["accepted"][1].target == "heart_rate"
    assert receipt["accepted"][2].target == "heart_rate"
    # The receipt retains what the model actually proposed.
    assert isinstance(receipt["proposed"]["questions"][1]["target"], dict)


def test_explicit_property_router_repairs_property_drift_and_missing_slots() -> None:
    adapter = Adapter({"status": "compiled", "questions": [
        {"id": "q1", "verb": "compare", "property": "volatility",
         "target": "heart_rate"},
        {"id": "q2", "verb": "compare", "property": "seasonality",
         "target": {"kind": "aggregate", "members": ["heart_rate"]}},
    ]})
    receipt = compile_temporal_text_receipt(
        "Median heart rate change?\nVolatility change?\nSeasonality alignment?",
        available_targets=["heart_rate", "spo2"], adapter=adapter,
        default_verb="predict", default_horizon=12)

    assert [item.property for item in receipt["accepted"]] == [
        "level", "volatility", "seasonality"]
    assert receipt["accepted"][0].target == "heart_rate"
    assert all(item.scope == "series" for item in receipt["accepted"][1:])
    assert all(item.target == "heart_rate" for item in receipt["accepted"][1:])
    assert receipt["rejected"] == []


def test_explicit_collective_question_does_not_inherit_series_focus() -> None:
    adapter = Adapter({"status": "compiled", "questions": [
        {"id": "q1", "verb": "compare", "property": "level",
         "target": "cpu"},
        {"id": "q2", "verb": "compare", "property": "volatility",
         "target": {"kind": "aggregate", "members": ["cpu", "mem"]}},
    ]})
    result = compile_temporal_text(
        "CPU level change?\nVolatility across all metrics?",
        available_targets=["cpu", "mem"], adapter=adapter)
    assert result[1].scope == "aggregate"


def test_explicit_each_and_dependence_scope_override_semantic_drift() -> None:
    each = compile_temporal_text(
        "Compare each metric's volatility over the next 12 periods.",
        available_targets=["cpu", "mem"], adapter=Adapter({
            "status": "compiled", "questions": [{
                "id": "q1", "verb": "compare", "property": "volatility",
                "target": {"kind": "aggregate", "members": ["cpu", "mem"]},
            }]}), default_horizon=12)
    related = compile_temporal_text(
        "Are mem and cpu related?", available_targets=["cpu", "mem"],
        adapter=Adapter({"status": "compiled", "questions": [{
            "id": "q1", "verb": "regress", "property": "regression",
            "target": "cpu", "explanatory_variables": ["mem"],
        }]}))

    assert each[0].scope == "each"
    assert each[0].members == ("cpu", "mem")
    assert each[0].horizon == 12
    assert related[0].property == "dependence"
    assert related[0].scope == "pair"
    assert related[0].members == ("mem", "cpu")
    assert related[0].verb == "compare"
    assert related[0].explanatory_variables == ()


def test_explicit_noisier_request_cannot_drift_to_level() -> None:
    result = compile_temporal_text(
        "Will error_rate become noisier over the next 6 periods?",
        available_targets=["error_rate"], adapter=Adapter({
            "status": "compiled", "questions": [{
                "id": "q1", "verb": "predict", "property": "level",
                "target": "error_rate", "measure": "point",
            }]}))

    assert result[0].property == "volatility"
    assert result[0].target == "error_rate"
    assert result[0].horizon == 6
    assert result[0].measure is None
