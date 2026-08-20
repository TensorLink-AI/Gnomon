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
    assert [item.target for item in receipt["accepted"]] == [
        "heart_rate", "heart_rate", "heart_rate"]
    # The receipt retains what the model actually proposed.
    assert isinstance(receipt["proposed"]["questions"][1]["target"], dict)


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
