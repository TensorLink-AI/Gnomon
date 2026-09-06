"""Independent staged commitments; scripted requests are not agent-uplift evidence."""

from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
import sqlite3
import os
from pathlib import Path
import shlex

import pytest

from benchmarks.common.openrouter import OpenRouterClient
from benchmarks.workflow.accounting import AttemptJournal, receipt
from benchmarks.workflow.bounded_agent import run_agent
from benchmarks.workflow.episodes import Episode
from benchmarks.workflow.matched import normalized_rows
from benchmarks.workflow.run_workflow import case_payload, run_command
from benchmarks.workflow.schema import Case, Observation
from benchmarks.workflow.scoring import score_run
from benchmarks.tests.test_workflow_driver import experiment, model_server  # noqa: F401

CANARY = "ACTUAL_REVEALED_ONLY_AFTER_COMMIT"
PUBLIC = {"series": [3, 7]}
BUDGET = {"max_rounds": 4, "max_tool_calls": 2, "max_tokens": 1000, "timeout_seconds": 10}


def case():
    final = {"numbers": {"mae": 2}}
    return Case.from_dict({"id": "episode", "kind": "synthetic", "domain": "forecasting",
        "question": "Forecast using the last value, then report absolute error after the actual arrives.",
        "available_at_cutoff": PUBLIC, "answer_schema": {"numbers": ["mae"]}, "oracle": final,
        "episode": [{"name": "forecast", "revealed": {}, "answer_schema": {"numbers": ["forecast"]},
                     "oracle": {"numbers": {"forecast": 7}}},
                    {"name": "score", "revealed": {"actual": 9, "note": CANARY},
                     "answer_schema": {"numbers": ["mae"]}, "oracle": final}]})


def private(value):
    return [{k: v for k, v in phase.items() if k != "oracle"} for phase in value.episode]


def answer(numbers):
    return {"status": "answered", "support": "supported", "numbers": numbers}


class Backend:
    startup_cost_usd = 0
    def __init__(self):
        self.closed = False
    def tools(self):
        return []
    def close(self):
        self.closed = True


def execute(tmp_path, *, first=7, callback=None, replies=None, budget=None, backend=None, usage=None):
    value = case()
    journal = AttemptJournal(tmp_path / "attempts.db")
    identifier = journal.start(value.id, "initial")
    backend = backend or Backend()
    requests = []
    responses = iter(replies or [answer({"forecast": first}), answer({"mae": 2})])
    class Model:
        def open(self, request, **kwargs):
            body = json.loads(request.data)
            if not requests:
                assert CANARY not in json.dumps(body)
                assert "_private_episode" not in json.dumps(body)
            else:
                # Reopen the real DB: durability precedes the reveal/model call.
                reopened = AttemptJournal(tmp_path / "attempts.db")
                assert len(reopened.checkpoints(identifier)) == 1
                reopened.close()
                assert CANARY in json.dumps(body)
            requests.append(body)
            choice = {"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call",
                "type": "function", "function": {"name": "submit_answer", "arguments": json.dumps(next(responses))}}]}}
            return io.BytesIO(json.dumps({"choices": [choice], "usage": usage if usage is not None else {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01}}).encode())
    def checkpoint(record):
        if callback:
            callback(record)
        journal.checkpoint(identifier, value.id, record)
    result = run_agent(case_payload(value), prompt="Follow each phase in order.", budget={**BUDGET, **(budget or {})},
                       client_factory=lambda: OpenRouterClient("scripted", api_key="unused", request_opener=Model()),
                       backend_factory=lambda: backend, episode=private(value), checkpoint=checkpoint)
    assert backend.closed
    records = journal.checkpoints(identifier)
    journal.finish(identifier, receipt(result, "initial"))
    journal.close()
    return result, records, requests


def test_real_commit_precedes_reveal_and_one_budget_spans_phases(tmp_path):
    result, records, requests = execute(tmp_path)
    assert result.status == "answered" and result.numbers == {"mae": 2}
    assert [row["answer"]["numbers"] for row in records] == [{"forecast": 7}, {"mae": 2}]
    assert len(requests) == 2 and result.cumulative_tokens == 30 and result.cost_usd == 0.02
    assert [row["usage"]["cumulative_tokens"] for row in records] == [15, 30]
    summary = score_run([case()], [result])
    assert summary["rows"][0]["episode"]["complete"]
    assert normalized_rows(summary)[0]["success"] and normalized_rows(summary)[0]["completed"]
    assert result.temporal_leakage is None  # No universal leakage-free attestation.


def test_final_correct_answer_cannot_replace_wrong_forecast(tmp_path):
    result, records, _ = execute(tmp_path, first=99)
    assert records[0]["answer"]["numbers"] == {"forecast": 99}
    row = score_run([case()], [result])["rows"][0]
    assert row["correctness"] == 0 and row["episode"]["complete"]
    assert [phase["correctness"] for phase in row["episode"]["phases"]] == [0, 1]


def test_checkpoint_failure_never_reveals_or_returns_a_successful_answer(tmp_path):
    def fail(record):
        raise OSError("private-storage-error")
    result, records, requests = execute(tmp_path, callback=fail)
    assert result.status == "error" and result.metadata["termination"] == "episode_commit_error"
    assert not records and len(requests) == 1 and result.cumulative_tokens == 15
    assert "private-storage-error" not in json.dumps(asdict(result))


def test_environment_reveal_failure_keeps_commit_and_stops_model_work(tmp_path):
    class FailingBackend(Backend):
        def reveal(self, revealed, timeout):
            raise OSError("private-environment-error")
    result, records, requests = execute(tmp_path, backend=FailingBackend())
    assert len(records) == len(requests) == 1
    assert records[0]["answer"]["numbers"] == {"forecast": 7}
    assert result.status == "error" and result.metadata["termination"] == "episode_reveal_error"
    assert result.cost_usd is None
    assert "private-environment-error" not in json.dumps(asdict(result))


def test_unknown_usage_commits_but_does_not_reveal_or_start_next_phase(tmp_path):
    class UnreachedBackend(Backend):
        def reveal(self, revealed, timeout):
            pytest.fail("unknown token usage must stop additional environment work")
    result, records, requests = execute(tmp_path, backend=UnreachedBackend(), usage={})
    assert len(records) == len(requests) == 1
    assert result.status == "error"
    assert not normalized_rows(score_run([case()], [result]))[0]["success"]


def test_episode_smoke_corpus_has_independently_checkable_oracles():
    from benchmarks.workflow.schema import load_cases
    forecast, decision, temporal = load_cases(Path(__file__).parents[1] / "workflow/cases/agent_episodes.jsonl")
    prediction = forecast.available_at_cutoff["history"][-1]
    assert forecast.episode[0]["oracle"]["numbers"]["forecast"] == prediction
    assert forecast.oracle.numbers == {"original_forecast": prediction,
        "current_mae": abs(forecast.episode[2]["revealed"]["actual"] - prediction),
        "source_replay_mae": abs(forecast.episode[1]["revealed"]["actual"] - prediction)}
    inputs = decision.available_at_cutoff
    shortfall = max(0, inputs["forecast"] - inputs["capacity"])
    assert decision.episode[0]["oracle"]["numbers"] == {"shortfall": shortfall, "proposed_cost": shortfall * inputs["cost_per_unit"]}
    assert not decision.episode[1]["revealed"]["approval_granted"]
    assert decision.oracle.choices == {"action": "do_not_execute"}
    revisions = temporal.available_at_cutoff["revisions"]
    def replay(rows, source, recorded=None):
        visible = [row for row in rows if row["source_available_at"] <= source
                   and (recorded is None or row["recorded_at"] <= recorded)]
        return max(visible, key=lambda row: row["source_available_at"])["value"]
    source, recorded = "2026-01-05T23:59:59Z", "2026-01-07T23:59:59Z"
    original = replay(revisions, source, recorded)
    assert temporal.episode[0]["oracle"]["numbers"] == {
        "source_value": replay(revisions, "2026-01-03T23:59:59Z"), "recorded_value": original}
    revised = [*revisions, temporal.episode[1]["revealed"]["revision"]]
    assert temporal.oracle.numbers == {"source_value": replay(revised, source),
        "recorded_value": replay(revised, source, recorded), "original_recorded_value": original}


def test_phase_transition_does_not_reset_round_budget(tmp_path):
    result, records, requests = execute(tmp_path, budget={"max_rounds": 1})
    assert result.status == "error" and result.metadata["termination"] == "cap:rounds"
    assert len(records) == len(requests) == 1
    assert not normalized_rows(score_run([case()], [result]))[0]["success"]


def test_checkpoint_corruption_or_changed_final_output_does_not_score_complete(tmp_path):
    result, records, _ = execute(tmp_path)
    changed = deepcopy(records)
    changed[0]["answer"]["numbers"]["forecast"] = 8
    for candidate in (replace(result, metadata={**result.metadata, "episode_checkpoints": changed}),
                      replace(result, numbers={"mae": 99})):
        row = normalized_rows(score_run([case()], [candidate]))[0]
        assert not row["success"] and not row["completed"]


def test_journal_checkpoints_are_append_only_and_bound_to_active_attempt(tmp_path):
    journal = AttemptJournal(tmp_path / "journal.db")
    identifier = journal.start("episode", "initial")
    episode = Episode(private(case()), lambda row: journal.checkpoint(identifier, "episode", row))
    episode.commit(answer({"forecast": 7}), {})
    with pytest.raises(sqlite3.IntegrityError):
        journal.db.execute("UPDATE checkpoints SET payload='{}'")
    with pytest.raises(sqlite3.IntegrityError):
        journal.db.execute("DELETE FROM checkpoints")
    with pytest.raises(ValueError):
        journal.checkpoint(identifier, "different-case", {"sequence": 1})
    journal.finish(identifier, receipt(Observation(case_id="episode", status="error", support="abstained"), "initial"))
    with pytest.raises(ValueError):
        episode.commit(answer({"mae": 2}), {})
    journal.close()


def test_crash_retains_commit_lower_bounds_and_resume_does_not_forecast_again(monkeypatch, tmp_path):
    value = case()
    path = tmp_path / "observations.jsonl"
    experiment = {"experiment_id": "fixture", "arm": "ordinary", "pinned": {
        "common": {}, "arms": {"ordinary": {}}, "evidence_kind": "scripted"}}
    def crash(payload, *args):
        binding = payload["_episode_journal"]
        journal = AttemptJournal(Path(binding["path"]))
        episode = Episode(payload["_private_episode"], lambda row: journal.checkpoint(binding["attempt_id"], value.id, row))
        episode.commit(answer({"forecast": 7}), {"tool_calls": 1, "cumulative_tokens": 15,
                       "response_tokens": 5, "cost_usd": 0.01, "latency_seconds": 0.1})
        journal.close()
        raise KeyboardInterrupt
    monkeypatch.setattr("benchmarks.workflow.run_workflow._invoke_once", crash)
    with pytest.raises(KeyboardInterrupt):
        run_command([value], "unused", 10, checkpoint_path=path, experiment=experiment)
    def no_replay(*args):
        pytest.fail("episode must not replay after a committed forecast/possible reveal")
    monkeypatch.setattr("benchmarks.workflow.run_workflow._invoke_once", no_replay)
    resumed = run_command([value], "unused", 10, checkpoint_path=path, experiment=experiment)[0]
    assert resumed.status == "error" and resumed.cumulative_tokens == 15
    assert resumed.metadata["episode_checkpoints"][0]["answer"]["numbers"] == {"forecast": 7}
    resources = resumed.metadata["resource_accounting"]["resources"]
    assert resources["cost_usd"]["observed_total"] == 0.01 and resources["cost_usd"]["total"] is None
    assert resources["cumulative_tokens"]["total"] is None


def test_direct_invocation_refuses_episode_retries_before_start(tmp_path):
    from benchmarks.workflow.run_workflow import _invoke
    journal = AttemptJournal(tmp_path / "attempts.db")
    try:
        with pytest.raises(ValueError, match="cannot be retried"):
            _invoke({"_private_episode": private(case())}, "episode", ["unused"], 10, retries=1, journal=journal)
        assert journal.for_case("episode") == []
    finally:
        journal.close()


def test_retired_journal_version_is_refused_without_modification(tmp_path):
    from benchmarks.workflow.accounting import APP_ID
    path = tmp_path / "v1.db"
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA application_id={APP_ID}")
        db.execute("PRAGMA user_version=1")
        db.execute("CREATE TABLE attempts (id TEXT NOT NULL, phase TEXT NOT NULL, case_id TEXT NOT NULL, stage TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(id,phase))")
        db.execute("INSERT INTO attempts VALUES ('old','started','legacy','initial','{}')")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="unsupported attempt journal version"):
        AttemptJournal(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["one_phase", "duplicate", "initial_reveal", "wrong_final_oracle"])
def test_invalid_episode_contract_is_rejected(change):
    value = asdict(case())
    phases = value["episode"]
    if change == "one_phase":
        value["episode"] = phases[:1]
    elif change == "duplicate":
        phases[1]["name"] = phases[0]["name"]
    elif change == "initial_reveal":
        phases[0]["revealed"] = {"actual": 9}
    else:
        phases[-1]["oracle"] = {"numbers": {"mae": 3}}
    with pytest.raises(ValueError):
        Case.from_dict(value)


def test_real_shared_driver_keeps_private_episode_and_journal_out_of_model_input(experiment, tmp_path):
    from benchmarks.workflow.matched import prepare
    base, contract, _, provider, requests = experiment
    final = {"numbers": {"next": 7}}
    value = Case.from_dict({**asdict(base), "episode": [
        {"name": "forecast", "revealed": {}, "answer_schema": {"numbers": ["next"]}, "oracle": final},
        {"name": "confirm", "revealed": {"note": CANARY}, "answer_schema": {"numbers": ["next"]}, "oracle": final}]})
    command = shlex.join(contract()["pinned"]["command"])
    identity = prepare(provider.parent / "experiment.json", [value], command=command, arm="lean", timeout=5, jobs=1, retries=0)
    result = run_command([value], command, 5, checkpoint_path=tmp_path / "observations.jsonl", experiment=identity)[0]
    assert result.status == "answered" and result.cumulative_tokens == 45
    assert len(result.metadata["episode_checkpoints"]) == 2 and len(requests) == 3
    for index, (_, _, body) in enumerate(requests):
        text = json.dumps(body)
        assert "_private_episode" not in text and "_episode_journal" not in text and "attempts.sqlite3" not in text
        assert (CANARY in text) == (index == 2)
    assert normalized_rows(score_run([value], [result]))[0]["success"]


@pytest.mark.parametrize("arm", ["ordinary", "lean", "full"])
def test_real_backend_state_survives_commit_reveal_and_agent_scoring(tmp_path, arm):
    from benchmarks.workflow.software_backend import SoftwareBackend
    from benchmarks.workflow import service_backend
    software = os.environ.get("GNOMON_TEST_SOFTWARE_IMAGE")
    service = os.environ.get("GNOMON_TEST_SERVICE_IMAGE")
    if not software or not service:
        pytest.skip("explicit software/service images required")
    value = case()
    phases = private(value)
    phases[1]["revealed"]["actuals"] = [{"series_id": "shop", "valid_time": "2025-01-03T00:00:00Z",
                                        "value": 9, "source_available_at": "2025-01-04T00:00:00Z"}]
    phases[1]["revealed"]["files"] = {"actuals.csv": "timestamp,value\n2025-01-03T00:00:00Z,9\n"}
    journal = AttemptJournal(tmp_path / "journal.db")
    identifier = journal.start(value.id, "initial")
    class Model:
        def __init__(self):
            self.step = 0
            self.execution_id = None
        def open(self, request, **kwargs):
            body = json.loads(request.data)
            assert (CANARY in json.dumps(body)) == (self.step >= 2)
            if self.step == 0:
                if arm == "lean":
                    name, arguments = "gnomon_forecast", {"provider": "last_value", "request": {
                        "history": [3, 7], "horizon": 1, "series_id": "shop", "future_timestamps": ["2025-01-03T00:00:00Z"]}}
                else:
                    name, arguments = "python", {"code": "import json; from pathlib import Path; p=json.loads(Path('/tmp/case.json').read_text()); n=p['available_at_cutoff']['series'][-1]; Path('/tmp/forecast.json').write_text(json.dumps(n)); print(n)"}
            elif self.step == 1:
                tool = json.loads(body["messages"][-1]["content"])
                if arm == "lean":
                    self.execution_id = tool["structuredContent"]["execution_id"]
                    number = tool["structuredContent"]["result"]["point"][0]
                else:
                    number = float(tool["stdout"])
                name, arguments = "submit_answer", answer({"forecast": number})
            elif self.step == 2:
                assert len(journal.checkpoints(identifier)) == 1
                if arm == "lean":
                    name, arguments = "gnomon_ledger", {"operation": "evaluate", "execution_id": self.execution_id}
                else:
                    name, arguments = "python", {"code": "import json,csv; from pathlib import Path; prediction=json.loads(Path('/tmp/forecast.json').read_text()); actual=float(next(csv.DictReader(Path('/tmp/data/actuals.csv').open()))['value']); print(abs(actual-prediction))"}
            else:
                tool = json.loads(body["messages"][-1]["content"])
                number = tool["structuredContent"]["result"]["mae"] if arm == "lean" else float(tool["stdout"])
                name, arguments = "submit_answer", answer({"mae": number})
            self.step += 1
            return io.BytesIO(json.dumps({"choices": [{"message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": "call", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01}}).encode())
    def factory():
        common = {"case": case_payload(value), "workspace": tmp_path, "timeout": 30}
        if arm == "ordinary":
            return SoftwareBackend(**common, options={"image": software, "docker_host": "unix:///var/run/docker.sock"})
        options = {"software_image": software, "service_image": service, "docker_host": "unix:///var/run/docker.sock"}
        if arm == "lean":
            options["execution_options"] = {"ledger": True}
        return getattr(service_backend, arm)(**common, options=options)
    try:
        result = run_agent(case_payload(value), prompt="Commit forecast before scoring new actuals.", budget={**BUDGET, "timeout_seconds": 30, "max_reported_cost_usd": 0.05},
                           client_factory=lambda: OpenRouterClient("scripted", api_key="unused", request_opener=Model()),
                           backend_factory=factory, episode=phases, checkpoint=lambda row: journal.checkpoint(identifier, value.id, row))
        assert result.status == "answered" and result.numbers == {"mae": 2}
        assert result.tool_calls == 2 and result.cumulative_tokens == 60 and result.cost_usd == 0.04
        assert len(result.metadata["environment_events"]) == 1 and result.metadata["cleanup_errors"] == []
        assert journal.checkpoints(identifier)[0]["answer"]["numbers"] == {"forecast": 7}
        assert normalized_rows(score_run([value], [result]))[0]["success"]
    finally:
        journal.close()
