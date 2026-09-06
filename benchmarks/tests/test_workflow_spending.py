"""Reported-cost stops are not prepaid/provider-enforced spending guarantees."""

from dataclasses import replace
import json
from pathlib import Path
import shlex
import sys

import pytest

from benchmarks.tests.test_bounded_agent import Backend, execute, call, response, submit
from benchmarks.tests.test_workflow_driver import experiment, model_server  # noqa: F401
from benchmarks.workflow.accounting import AttemptJournal, receipt, reported_cost_limit
from benchmarks.workflow.bounded_agent import ToolReply
from benchmarks.workflow.matched import prepare, normalized_rows
from benchmarks.workflow.run_workflow import run_command
from benchmarks.workflow.schema import Observation
from benchmarks.workflow.scoring import score_run


@pytest.mark.parametrize("value", [None, True, False, 0, -1, "1", float("nan"), float("inf"), 10**1000])
def test_invalid_reported_cost_limit_is_rejected(value):
    with pytest.raises(ValueError, match="finite and positive"):
        reported_cost_limit({"max_reported_cost_usd": value})


def test_cost_stop_prevents_dispatch_after_model_charge_reaches_threshold(monkeypatch):
    result, requests, backend = execute(monkeypatch, [response(call("first"))], budget={"max_reported_cost_usd": 0.01})
    assert len(requests) == 1 and not backend.calls
    assert result.status == "error" and result.metadata["termination"] == "cap:cost"
    assert result.cost_usd == 0.01 and result.metadata["reported_cost_overrun_usd"] == 0


def test_cost_stop_discloses_one_response_overshoot_even_for_final_submission(monkeypatch):
    result, requests, _ = execute(monkeypatch, [response(submit())], budget={"max_reported_cost_usd": 0.005})
    assert len(requests) == 1 and result.status == "error"
    assert result.cost_usd == 0.01 and result.metadata["reported_cost_overrun_usd"] == pytest.approx(0.005)
    assert result.metadata["budget_exceeded"] is True


def test_final_answer_exactly_at_cost_threshold_is_delivered(monkeypatch):
    result, requests, _ = execute(monkeypatch, [response(submit())], budget={"max_reported_cost_usd": 0.01})
    assert len(requests) == 1 and result.status == "answered"
    assert result.metadata["budget_exceeded"] is False


def test_unknown_startup_cost_stops_before_model_request(monkeypatch):
    backend = Backend()
    backend.startup_cost_usd = None
    result, requests, _ = execute(monkeypatch, [], backend, budget={"max_reported_cost_usd": 1})
    assert requests == [] and result.metadata["termination"] == "cost_usage_unmeasured"
    assert result.cost_usd is None


@pytest.mark.parametrize("charge", [None, 0.02])
def test_cost_checks_between_batched_tools_prevent_second_dispatch(monkeypatch, charge):
    class Charged(Backend):
        def call(self, name, arguments, **kwargs):
            super().call(name, arguments, **kwargs)
            return ToolReply({}, charge)
    result, requests, backend = execute(monkeypatch, [response(call("first", identifier="a"), call("second", identifier="b"))],
                                        Charged(), budget={"max_reported_cost_usd": 0.02})
    assert len(requests) == len(backend.calls) == 1
    assert result.metadata["termination"] == ("cost_usage_unmeasured" if charge is None else "cap:cost")


@pytest.mark.parametrize("final", [False, True])
def test_missing_model_cost_never_authorizes_additional_work(monkeypatch, final):
    value = response(submit() if final else call("first"))
    value["usage"].pop("cost")
    result, requests, backend = execute(monkeypatch, [value], budget={"max_reported_cost_usd": 1})
    assert len(requests) == 1 and not backend.calls and result.cost_usd is None
    assert result.status == ("answered" if final else "error")
    assert result.metadata["budget_exceeded"] is None


def configured(experiment, tmp_path, limit=0.025):
    case, _, _, _, requests = experiment
    path = tmp_path / "experiment.json"
    spec = json.loads(path.read_text())
    spec["common"]["budget"]["max_reported_cost_usd"] = limit
    path.write_text(json.dumps(spec))
    cases = [replace(case, id=str(index)) for index in range(3)]
    command = shlex.join(spec["command"])
    identity = prepare(path, cases, command=command, arm="lean", timeout=5, jobs=1, retries=0)
    return cases, command, identity, requests


def test_real_driver_uses_remaining_arm_allowance_and_resume_does_not_reset_spend(experiment, tmp_path):
    cases, command, identity, requests = configured(experiment, tmp_path)
    path = tmp_path / "observations.jsonl"
    rows = run_command(cases, command, 5, checkpoint_path=path, experiment=identity)
    assert len(requests) == 3  # First case uses2 requests; second overruns on its first; third never starts.
    assert rows[0].status == "answered" and rows[0].cost_usd == 0.02
    assert rows[1].status == "error" and rows[1].cost_usd == 0.01
    assert rows[1].metadata["limits"]["max_reported_cost_usd"] == pytest.approx(0.005)
    assert rows[2].metadata["not_dispatched"] and rows[2].cost_usd == 0
    assert "_spending_allowance_usd" not in json.dumps(requests)
    assert len(normalized_rows(score_run(cases, rows))) == 3  # No survivor-only comparison.
    resumed = run_command(cases, command, 5, checkpoint_path=path, prior=rows, experiment=identity)
    assert len(requests) == 3 and sum(row.cost_usd or 0 for row in resumed) == pytest.approx(0.03)
    journal = AttemptJournal(path.with_suffix(".attempts.sqlite3"))
    try:
        assert journal.reported_spend()["total"] == pytest.approx(0.03)
    finally:
        journal.close()


def test_unfinished_attempt_from_another_case_stops_the_whole_arm(experiment, tmp_path):
    cases, command, identity, requests = configured(experiment, tmp_path)
    path = tmp_path / "observations.jsonl"
    journal = AttemptJournal(path.with_suffix(".attempts.sqlite3"))
    journal.start("prior-case-outside-current-batch", "initial")
    journal.close()
    rows = run_command(cases, command, 5, checkpoint_path=path, experiment=identity)
    assert requests == [] and all(row.metadata["error"] == "spending_usage_unmeasured" for row in rows)
    assert all(row.metadata["not_dispatched"] for row in rows)


def test_failed_finished_attempt_cost_is_included_in_arm_stop(experiment, tmp_path):
    cases, command, identity, requests = configured(experiment, tmp_path)
    path = tmp_path / "observations.jsonl"
    journal = AttemptJournal(path.with_suffix(".attempts.sqlite3"))
    identifier = journal.start("earlier-failure", "initial")
    journal.finish(identifier, receipt(Observation(case_id="earlier-failure", status="error", support="abstained",
        cost_usd=0.03, metadata={"resource_fields": ["cost_usd"]}), "initial"))
    journal.close()
    rows = run_command(cases, command, 5, checkpoint_path=path, experiment=identity)
    assert requests == [] and all(row.metadata["error"] == "cap:cost" for row in rows)


@pytest.mark.parametrize("jobs,retries,persistent", [(2, 0, True), (1, 1, True), (1, 0, False)])
def test_lower_level_runner_refuses_unsafe_spending_modes(experiment, tmp_path, jobs, retries, persistent):
    cases, command, identity, requests = configured(experiment, tmp_path)
    with pytest.raises(ValueError, match="reported-cost control"):
        run_command(cases, command, 5, jobs=jobs, retries=retries, experiment=identity,
                    checkpoint_path=tmp_path / "observations.jsonl" if persistent else None)
    assert requests == []


def test_pinned_experiment_refuses_other_driver_with_reported_cost_control(experiment, tmp_path):
    cases, _, _, _ = configured(experiment, tmp_path)
    path = tmp_path / "experiment.json"
    spec = json.loads(path.read_text())
    spec["command"][1] = str(Path(__file__))
    spec["driver_files"] = [str(Path(__file__))]
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="built-in driver"):
        prepare(path, cases, command=shlex.join(spec["command"]), arm="lean", timeout=5, jobs=1, retries=0)


@pytest.mark.parametrize("limit", [0.01, 0.015])
def test_episode_budget_cannot_reset_between_committed_phases(tmp_path, limit):
    from benchmarks.tests.test_workflow_episodes import execute as execute_episode
    result, records, requests = execute_episode(tmp_path, budget={"max_reported_cost_usd": limit})
    assert result.status == "error" and result.metadata["termination"] == "cap:cost"
    assert len(records) == 1 and records[0]["answer"]["numbers"] == {"forecast": 7}
    assert len(requests) == (1 if limit == 0.01 else 2)


def test_unknown_cost_at_commit_prevents_environment_reveal(tmp_path):
    from benchmarks.tests.test_workflow_episodes import execute as execute_episode, Backend as EpisodeBackend
    class Unreached(EpisodeBackend):
        def reveal(self, revealed, timeout):
            pytest.fail("unknown cost must not authorize more environment work")
    result, records, requests = execute_episode(tmp_path, backend=Unreached(), budget={"max_reported_cost_usd": 1},
                                               usage={"prompt_tokens": 10, "completion_tokens": 5})
    assert len(records) == len(requests) == 1
    assert result.metadata["termination"] == "cost_usage_unmeasured"


@pytest.mark.parametrize("allowance", [None, 0, -1, 2, float("nan")])
def test_driver_rejects_missing_invalid_or_raised_allowance(experiment, tmp_path, allowance):
    from benchmarks.workflow.driver import run
    from benchmarks.workflow.matched import public_context
    from benchmarks.workflow.run_workflow import case_payload
    cases, _, identity, requests = configured(experiment, tmp_path, limit=1)
    payload = {**case_payload(cases[0]), "experiment": public_context(identity)}
    if allowance is not None:
        payload["_spending_allowance_usd"] = allowance
    with pytest.raises(ValueError):
        run(payload)
    assert requests == []


def test_filled_template_pins_same_controls_and_declared_actual_backends(tmp_path):
    from benchmarks.workflow.matched import ARMS, ROOT
    from benchmarks.workflow.schema import load_cases
    source = ROOT / "benchmarks/workflow/experiment"
    spec = json.loads((source / "experiment.example.json").read_text())
    provider = json.loads((source / "providers.example.json").read_text())
    spec["command"] = [sys.executable, str(ROOT / "benchmarks/workflow/driver.py")]
    spec["driver_files"] = [spec["command"][1]]
    spec["common"]["model"]["id"] = "fixture/not-a-paid-model"
    spec["common"]["budget"]["max_reported_cost_usd"] = 0.1
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(spec))
    (tmp_path / "providers.json").write_text(json.dumps(provider))
    (tmp_path / "prompt.txt").write_text((source / "prompt.txt").read_text())
    cases = load_cases(ROOT / "benchmarks/workflow/cases/agent_episodes.jsonl")
    identities = [prepare(path, cases, command=shlex.join(spec["command"]), arm=arm,
                          timeout=120, jobs=1, retries=0) for arm in ARMS]
    assert len({identity["experiment_id"] for identity in identities}) == 1
    assert identities[0]["pinned"]["common"]["budget"]["max_reported_cost_usd"] == 0.1
    assert "--allow-model-requests" not in spec["command"]
    assert {key: value["factory"] for key, value in provider["backends"].items()} == {
        "ordinary": "benchmarks.workflow.software_backend:SoftwareBackend",
        "lean": "benchmarks.workflow.service_backend:lean", "full": "benchmarks.workflow.service_backend:full"}
    assert provider["backends"]["lean"]["options"]["execution_options"] == {"ledger": True, "temporal": True}
