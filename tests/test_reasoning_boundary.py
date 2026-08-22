from gnomon.reasoning_boundary import (
    apply_reasoning_boundary, measure_redundant_calls, verify_fact_sources,
)
from gnomon.toolspec import apply_response_contract


def test_every_success_gets_source_addressed_argument_and_sufficiency():
    result = apply_response_contract({
        "headline": "Requests are expected to rise.",
        "artifact_id": "forecast-1",
        "support": {"state": "supported"},
        "results": [{"series": "requests", "forecast": [{"q50": 12.0}]}],
    })
    frame = result["reasoning"]
    assert frame["canonical_immutable"] is True
    assert frame["canonical_source"] == "/headline"
    assert frame["sufficiency"]["further_calls_add_nothing_for"]
    assert verify_fact_sources(result) == []


def test_error_gets_one_terminal_actionable_rejection():
    result = apply_response_contract({
        "status": "error", "error": {"code": "AMBIGUOUS_SCHEMA",
        "message": "Choose a target.", "details": {"candidates": ["a", "b"]},
        "repair_options": [{"tool": "gnomon_forecast",
                            "arguments": {"target_column": "a"}}]},
    })
    assert result["rejection"]["terminal"] is True
    assert result["rejection"]["code"] == "AMBIGUOUS_SCHEMA"
    assert result["rejection"]["admissibility_path"][0]["arguments"] == {
        "target_column": "a"}


def test_dangling_fact_source_fails_loudly():
    payload = {"reasoning": {"facts": [{"name": "x", "source": "/missing"}]}}
    assert verify_fact_sources(payload)[0]["code"] == "FACT_SOURCE_MISSING"


def test_existing_reasoning_is_projected_without_changing_answer():
    payload = {"best_estimate": {"value": "increased"},
               "reasoning": {"because": [{"evidence": "folds"}],
                             "against": [{"evidence": "recent"}],
                             "unknown": ["regime persistence"]}}
    result = apply_reasoning_boundary(payload)
    assert result["best_estimate"] == payload["best_estimate"]
    assert result["reasoning"]["because"] == [{"evidence": "folds"}]


def test_redundant_calls_are_attributed_to_host_after_sufficiency():
    sufficient = apply_response_contract({"headline": "Done", "artifact_id": "f1"})
    metric = measure_redundant_calls([
        {"tool": "gnomon_forecast", "result": sufficient},
        {"tool": "gnomon_get_artifact", "result": {"rows": []}},
    ])
    assert metric == {
        "observed_calls": 2, "surface_required_calls": 1, "redundant_calls": 1,
        "redundant": [{"index": 1, "tool": "gnomon_get_artifact",
                       "reason": "prior response declared task sufficient"}],
    }
