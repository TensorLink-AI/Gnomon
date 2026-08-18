from __future__ import annotations

import pytest

from gnomon.contracts import GnomonError
from gnomon.temporal_question import (
    compile_temporal_question, compile_temporal_questions,
)


def test_compiler_binds_only_exact_target_and_echoes_alias() -> None:
    question = compile_temporal_question(
        {"id": "v", "verb": "predict", "target": "requests",
         "property": "variance", "horizon": 7},
        available_targets=["requests", "errors"],
    )
    assert question.property == "volatility"
    assert question.target == "requests"
    assert question.to_dict()["horizon"] == 7


def test_compiler_refuses_ambiguous_target_with_repair_call() -> None:
    with pytest.raises(GnomonError) as caught:
        compile_temporal_question(
            {"id": "v", "property": "volatility"},
            available_targets=["requests", "errors"],
        )
    assert caught.value.code == "INVALID_TEMPORAL_QUESTION"
    repair = caught.value.details["repair_call"]
    assert repair["arguments"]["questions"][0]["target"] == "<target>"


def test_compiler_rejects_unknown_measure_instead_of_defaulting() -> None:
    with pytest.raises(GnomonError):
        compile_temporal_question(
            {"property": "volatility", "measure": "vibes"},
            available_targets=["requests"],
        )


def test_compiler_accepts_explicit_auditable_aggregate_scope() -> None:
    question = compile_temporal_question({
        "id": "fleet-v", "verb": "predict", "property": "volatility",
        "target": {"kind": "aggregate", "members": ["cpu", "mem"],
                   "aggregation": "median_normalized_scale_ratio"},
        "horizon": 12,
    }, available_targets=["cpu", "mem", "disk"])
    assert question.scope == "aggregate"
    assert question.members == ("cpu", "mem")
    assert question.to_dict()["target"]["aggregation"] == \
        "median_normalized_scale_ratio"


def test_compiler_rejects_implicit_or_unknown_aggregation() -> None:
    with pytest.raises(GnomonError):
        compile_temporal_question({
            "property": "volatility",
            "target": {"kind": "aggregate", "members": ["cpu", "mem"],
                       "aggregation": "average_raw_mad"},
        }, available_targets=["cpu", "mem"])


def test_compiler_represents_dependence_as_an_exact_pair() -> None:
    question = compile_temporal_question({
        "property": "dependence", "measure": "correlation",
        "target": {"kind": "pair", "members": ["cpu", "mem"]},
    }, available_targets=["cpu", "mem", "disk"])
    assert question.scope == "pair"
    assert question.members == ("cpu", "mem")


def test_compiler_rejects_pair_with_duplicate_member() -> None:
    with pytest.raises(GnomonError) as caught:
        compile_temporal_question({
            "property": "dependence", "measure": "correlation",
            "target": {"kind": "pair", "members": ["cpu", "cpu"]},
        }, available_targets=["cpu", "mem"])
    assert "distinct" in caught.value.details["fields"]["target.members"]


def test_compiler_normalizes_json_encoded_question_array() -> None:
    questions = compile_temporal_questions(
        '[{"id":"v","property":"volatility","target":"cpu"}]',
        available_targets=["cpu"], default_verb="predict")
    assert [question.id for question in questions] == ["v"]


def test_compiler_accepts_named_seasonality_aggregate() -> None:
    question = compile_temporal_question({
        "property": "seasonality",
        "target": {"kind": "aggregate", "members": ["cpu", "mem"]},
    }, available_targets=["cpu", "mem"])
    assert question.aggregation == "median_alignment"


def test_compiler_rejects_malformed_answer_vocabulary() -> None:
    with pytest.raises(GnomonError):
        compile_temporal_question({
            "property": "volatility", "target": "cpu",
            "answer_vocabulary": {"stable": 7},
        }, available_targets=["cpu"])
