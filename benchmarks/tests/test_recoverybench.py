from __future__ import annotations

from benchmarks.recoverybench.run import _normalise_identity, _plan_checks, cases


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


def test_plan_checks_reject_guessed_external_patch() -> None:
    payload = {"recovery_plan": [{
        "rank": 1, "recommended": True,
        "source": "/error/repair_options/0",
        "execution": {"mode": "tool", "argument_patch": {"frequency": "D"}},
    }]}
    checks = _plan_checks(payload, 1, "external_choice")
    assert checks["no_fake_automation"] is False
