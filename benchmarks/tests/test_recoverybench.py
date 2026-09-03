from __future__ import annotations

from benchmarks.recoverybench.run import _normalise_identity, _plan_checks, cases
from benchmarks.recoverybench.run_agent import _control_patch, _relay, summarize


def test_frozen_recovery_cases_have_expected_denominators(tmp_path) -> None:
    frozen = cases(tmp_path)
    assert len(frozen) == 6
    assert sum(case["class"] == "automatic" for case in frozen) == 3
    assert sum(case["class"] == "external_choice" for case in frozen) == 3
    assert all(case["tool"].startswith("gnomon_") for case in frozen)


def test_additive_fields_are_excluded_from_canonical_comparison() -> None:
    payload = {
        "status": "complete", "headline": "unchanged",
        "artifact_id": "volatile", "recovery_plan": [{"rank": 1}],
        "reasoning": {"resolution": {"kind": "complete"},
                      "recovery_plan_ref": "/recovery_plan/0"},
    }
    assert _normalise_identity(payload) == {
        "status": "complete", "headline": "unchanged",
        "reasoning": {"resolution": {"kind": "complete"}},
    }


def test_canonical_comparison_normalises_the_declared_run_root(tmp_path) -> None:
    first = tmp_path / "arbitrary-baseline-name"
    second = tmp_path / "unrelated-candidate-name"
    payload = {"error": {"message": f"missing {first}/absent.csv"}}
    replay = {"error": {"message": f"missing {second}/absent.csv"}}
    assert _normalise_identity(payload, (first,)) == \
        _normalise_identity(replay, (second,))


def test_plan_checks_reject_guessed_external_patch() -> None:
    payload = {"recovery_plan": [{
        "rank": 1, "recommended": True,
        "source": "/error/repair_options/0",
        "execution": {"mode": "tool", "argument_patch": {"frequency": "D"}},
    }]}
    checks = _plan_checks(payload, 1, "external_choice")
    assert checks["no_fake_automation"] is False


def test_agent_efficiency_requires_matched_preserved_recovery() -> None:
    common = {
        "single_retry_succeeded": True,
        "patch_matches_recommendation": True,
        "support_preserved": True,
        "automation_withheld": True,
    }
    control = {
        **common, "latency_seconds": 3.0,
        "llm_usage": {
            "requests": 2, "prompt_tokens": 100, "completion_tokens": 20,
            "sample_cache_accounting_complete": True,
        },
    }
    treatment = {
        **common, "latency_seconds": 1.0,
        "llm_usage": {
            "requests": 1, "prompt_tokens": 30, "completion_tokens": 10,
            "sample_cache_accounting_complete": True,
        },
    }

    summary = summarize(control, treatment, "revision")

    assert summary["all_gates_passed"] is True
    assert summary["gfr_raw"] == {
        "control_requests": 2.0,
        "control_tokens": 120.0,
        "control_latency_seconds": 3.0,
        "treatment_requests": 1.0,
        "treatment_tokens": 40.0,
        "treatment_latency_seconds": 1.0,
    }


def test_agent_efficiency_requests_use_configured_transport_retries() -> None:
    class Client:
        def __init__(self, response: str) -> None:
            self.response = response
            self.transport_retries = []

        def completions(self, *args, **kwargs):
            self.transport_retries.append(kwargs.get("transport_retries"))
            return [self.response]

    case = {"tool": "gnomon_forecast", "arguments": {"horizon": 1}}
    response = {
        "tier_floor": "best_effort", "headline": "Recovered forecast",
        "recovery_actions": [], "reasoning": {},
    }
    planner = Client(
        '{"tool":"gnomon_forecast","argument_patch":{"horizon":1}}')
    assert _control_patch(planner, case, response, 10) == {"horizon": 1}
    assert planner.transport_retries == [None]

    relay = Client(
        '{"headline":"Recovered forecast","tier_floor":"best_effort",'
        '"forecast_rows":1,"automation_eligible":false}')
    recovered = {
        "headline": "Recovered forecast", "tier_floor": "best_effort",
        "results": [{"forecast_rows": 1}], "automation_eligible": False,
    }
    assert _relay(relay, recovered, 10)["forecast_rows"] == 1
    assert relay.transport_retries == [None]
