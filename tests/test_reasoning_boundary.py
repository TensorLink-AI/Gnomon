from gnomon.reasoning_boundary import (
    apply_reasoning_boundary, measure_redundant_calls, verify_fact_sources,
)
from gnomon.toolspec import apply_response_contract
import random


def test_every_success_gets_source_addressed_argument_and_sufficiency():
    result = apply_response_contract({
        "headline": "Requests are expected to rise.",
        "artifact_id": "forecast-1",
        "task": {"task_type": "forecast"},
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
    sufficient = apply_response_contract({
        "headline": "Done", "artifact_id": "f1",
        "task": {"task_type": "forecast"},
    })
    metric = measure_redundant_calls([
        {"tool": "gnomon_forecast", "result": sufficient},
        {"tool": "gnomon_get_artifact", "result": {"rows": []}},
    ])
    assert metric == {
        "observed_calls": 2, "surface_required_calls": 1, "redundant_calls": 1,
        "redundant": [{"index": 1, "tool": "gnomon_get_artifact",
                       "reason": "prior response declared task sufficient"}],
    }


def test_malformed_collections_become_typed_contract_errors_not_crashes():
    result = apply_response_contract({
        "headline": "Unsafe shape", "limitations": 1,
        "task": {"task_type": "forecast"},
    })
    assert result["status"] == "error"
    assert result["error"]["code"] == "INVALID_RESPONSE_CONTRACT"
    assert "limitations must be a list" in result["error"]["message"]


def test_headline_alone_does_not_claim_question_sufficiency():
    result = apply_response_contract({"headline": "A generic status line."})
    assert result["reasoning"]["sufficiency"]["sufficient_for"] == []
    assert result["reasoning"]["sufficiency"]["requires_follow_up"] is True
    assert result["reasoning"]["resolution"]["kind"] == "terminal"


def test_abstention_with_recovery_ends_in_parameterized_action():
    action = {"tool": "gnomon_forecast", "arguments": {"horizon": 3}}
    result = apply_response_contract({
        "headline": "No forecast published.",
        "task": {"task_type": "forecast"},
        "support": "unsupported", "recovery_actions": [action],
    })
    assert result["reasoning"]["resolution"] == {
        "kind": "recovery", "action": action,
    }
    assert result["reasoning"]["recovery_plan_ref"] == "/recovery_plan/0"
    assert result["recovery_plan"][0]["execution"] == {
        "mode": "tool", "requires_user_input": False,
        "tool": "gnomon_forecast", "argument_patch": {"horizon": 3},
    }


def test_support_recovery_plan_is_exact_but_does_not_upgrade_authority():
    result = apply_response_contract({
        "headline": "No forecast published.",
        "task": {"task_type": "forecast"},
        "support": "unsupported",
        "recovery_actions": [{
            "code": "reduce_horizon",
            "message": "Retry with horizon 4 or less: already supportable.",
        }, {
            "code": "provide_more_history",
            "message": "Supply more observations.",
        }],
    })
    exact, external = result["recovery_plan"]
    assert exact["execution"] == {
        "mode": "tool", "requires_user_input": False,
        "tool": "gnomon_forecast",
        "argument_patch": {"horizon": 4, "minimum_support": "best_effort"},
    }
    assert "evidence" in exact["authority_limit"]
    assert external["execution"] == {
        "mode": "user_input", "requires_user_input": True,
    }
    assert external["source"] == "/recovery_actions/1"


def test_optional_improvement_does_not_override_complete_resolution():
    result = apply_response_contract({
        "headline": "Forecast published.", "verb": "forecast",
        "tier_floor": "conditionally_supported",
        "recovery_actions": [{
            "code": "provide_more_history", "message": "Improve evaluation.",
        }],
    })
    assert result["reasoning"]["sufficiency"]["requires_follow_up"] is False
    assert result["reasoning"]["resolution"]["kind"] == "complete"
    assert result["reasoning"]["recovery_plan_ref"] == "/recovery_plan/0"
    assert result["recovery_plan"][0]["execution"]["requires_user_input"] is True


def test_best_effort_with_recovery_requires_follow_up() -> None:
    result = apply_response_contract({
        "headline": "Orientation-only fallback.", "verb": "forecast",
        "tier_floor": "best_effort",
        "recovery_actions": [{
            "code": "reduce_horizon", "message": "Retry with horizon 4.",
        }],
    })
    assert result["reasoning"]["sufficiency"]["requires_follow_up"] is True
    assert result["reasoning"]["resolution"]["kind"] == "recovery"


def test_inconclusive_tier_floor_keeps_recovery_resolution():
    result = apply_response_contract({
        "headline": "No forecast published.", "verb": "forecast",
        "tier_floor": "inconclusive",
        "recovery_actions": [{
            "code": "reduce_horizon", "message": "Retry with horizon 4.",
        }],
    })
    assert result["reasoning"]["sufficiency"]["requires_follow_up"] is True
    assert result["reasoning"]["resolution"]["kind"] == "recovery"
    assert result["reasoning"]["recovery_plan_ref"] == "/recovery_plan/0"


def test_error_repair_plan_never_guesses_missing_frequency():
    result = apply_response_contract({
        "status": "error", "error": {
            "code": "AMBIGUOUS_FREQUENCY", "message": "Choose a grid.",
            "details": {"supported": ["D", "W"]},
            "repair_options": [{
                "action": "set_frequency",
                "description": "Pass frequency explicitly.",
            }],
        },
    })
    [repair] = result["recovery_plan"]
    assert repair["execution"] == {
        "mode": "user_input", "requires_user_input": True,
    }
    assert result["rejection"]["recovery_plan_ref"] == "/recovery_plan/0"


def test_triage_fact_uses_public_triage_pointer():
    result = apply_response_contract({
        "headline": "Panel complete.", "task": {"task_type": "forecast"},
        "triage": {"ranking_rule": "threshold then movement"},
    })
    fact = next(item for item in result["reasoning"]["facts"]
                if item["name"] == "series_triage")
    assert fact["source"] == "/triage"


def test_adversarial_payload_shapes_never_escape_the_public_boundary():
    rng = random.Random(20260823)
    shapes = [None, 1, "text", {}, [1], ["x"], [{"source": "/headline"}]]
    for _ in range(250):
        payload = {
            "headline": "Computed answer.",
            "task": {"task_type": "forecast"},
            "limitations": rng.choice(shapes),
            "recovery_actions": rng.choice(shapes),
            "reasoning": rng.choice(shapes),
        }
        result = apply_response_contract(payload)
        assert isinstance(result, dict)
        if result.get("status") == "error":
            assert (result.get("error") or {}).get("code") in {
                "INVALID_RESPONSE_CONTRACT", "UNTRACEABLE_RESPONSE_FACT",
            }
        else:
            assert verify_fact_sources(result) == []
