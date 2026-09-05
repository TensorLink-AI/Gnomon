import json
import shlex
import sqlite3
import sys
import subprocess
from dataclasses import replace

import pytest

from benchmarks.workflow.accounting import AttemptJournal, FIELDS, receipt, summarize
from benchmarks.workflow.run_workflow import _invoke, run_command, DEFAULT_CASES
from benchmarks.workflow.schema import Observation, load_cases, load_observations
from benchmarks.workflow.scoring import score_run
from benchmarks.workflow.compare import compare


def observation(case_id, *, status="answered", calls=1, tokens=10, cost=1):
    return Observation.from_dict({"case_id": case_id, "status": status,
        "support": "abstained" if status == "error" else "supported",
        "tool_calls": calls, "cumulative_tokens": tokens, "response_tokens": tokens,
        "latency_seconds": 0.5, "cost_usd": cost,
        "metadata": {"error": "provider_error"} if status == "error" else {}})


def test_retry_preserves_every_attempt_and_never_changes_first_receipt(monkeypatch, tmp_path):
    responses = iter([observation("a", status="error", calls=3, tokens=100, cost=2), observation("a")])
    monkeypatch.setattr("benchmarks.workflow.run_workflow._invoke_once", lambda *args: next(responses))
    journal = AttemptJournal(tmp_path / "attempts.db")
    try:
        result = _invoke({}, "a", [], 1, retries=1, journal=journal)
        assert result.tool_calls == 4 and result.cumulative_tokens == 110
        assert result.cost_usd == 3 and result.latency_seconds == 1
        rows = journal.for_case("a")
        assert [row["status"] for row in rows] == ["error", "answered"]
        assert [row["resources"]["cumulative_tokens"] for row in rows] == [100, 10]
        assert result.metadata["resource_accounting"]["budget_accounting_complete"] is True
        assert result.metadata["resource_accounting"]["failed_attempts"] == 1
        assert rows[0]["error_code"] == "provider_error"
        with pytest.raises(sqlite3.IntegrityError):
            journal.db.execute("UPDATE attempts SET stage='changed'")
        with pytest.raises(sqlite3.IntegrityError):
            journal.db.execute("DELETE FROM attempts")
    finally:
        journal.close()


def test_resume_preserves_prior_attempts_without_duplicate_charging(monkeypatch, tmp_path):
    case = replace(load_cases(DEFAULT_CASES)[0], stages=())
    checkpoint = tmp_path / "observations.jsonl"
    monkeypatch.setattr("benchmarks.workflow.run_workflow._invoke_once", lambda *args:
                        observation(case.id, status="error", calls=3, tokens=100, cost=2))
    failed = run_command([case], "unused", 1, retries=0, checkpoint_path=checkpoint)
    assert failed[0].cumulative_tokens == 100
    previous = load_observations(checkpoint)
    monkeypatch.setattr("benchmarks.workflow.run_workflow._invoke_once", lambda *args: observation(case.id))
    recovered = run_command([case], "unused", 1, prior=previous, checkpoint_path=checkpoint)
    assert recovered[0].cumulative_tokens == 110 and recovered[0].tool_calls == 4
    assert recovered[0].cost_usd == 3
    again = run_command([case], "unused", 1, prior=recovered, checkpoint_path=checkpoint)
    assert again[0].cumulative_tokens == 110
    assert again[0].metadata["resource_accounting"]["attempts"] == 2


def test_durable_start_survives_interruption_before_any_checkpoint(monkeypatch, tmp_path):
    case = replace(load_cases(DEFAULT_CASES)[0], stages=())
    checkpoint = tmp_path / "observations.jsonl"

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr("benchmarks.workflow.run_workflow._invoke_once", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_command([case], "unused", 1, checkpoint_path=checkpoint)
    assert not checkpoint.exists()
    journal = AttemptJournal(checkpoint.with_suffix(".attempts.sqlite3"))
    assert journal.for_case(case.id)[0]["status"] == "unfinished"
    journal.close()
    monkeypatch.setattr("benchmarks.workflow.run_workflow._invoke_once", lambda *args: observation(case.id))
    recovered = run_command([case], "unused", 1, checkpoint_path=checkpoint)[0]
    resources = recovered.metadata["resource_accounting"]["resources"]
    assert resources["cumulative_tokens"]["observed_total"] == 10
    assert resources["cumulative_tokens"]["total"] is None
    assert recovered.cost_usd is None
    assert recovered.metadata["resource_accounting"]["unfinished"] == 1


def test_stage_failure_usage_survives_resume_of_the_whole_case(monkeypatch, tmp_path):
    source = load_cases(DEFAULT_CASES)[0]
    case = replace(source, oracle=replace(source.oracle, requires_repair=True), stages=({"name": "repair"},))
    checkpoint = tmp_path / "observations.jsonl"
    responses = iter([observation(case.id), observation(case.id, status="error", calls=3, tokens=100)])
    monkeypatch.setattr("benchmarks.workflow.run_workflow._invoke_once", lambda *args: next(responses))
    first = run_command([case], "unused", 1, checkpoint_path=checkpoint)
    assert first[0].cumulative_tokens == 110
    assert first[0].metadata["stage_infrastructure_failures"]
    responses = iter([observation(case.id), observation(case.id)])
    second = run_command([case], "unused", 1, prior=load_observations(checkpoint), checkpoint_path=checkpoint)[0]
    assert second.cumulative_tokens == 130 and second.tool_calls == 6
    assert [item["stage"] for item in second.metadata["attempt_receipts"]] == ["initial", "repair", "initial", "repair"]
    assert not second.metadata["stage_infrastructure_failures"]


def test_legacy_resume_keeps_observed_usage_without_attesting_lost_costs(monkeypatch, tmp_path):
    case = replace(load_cases(DEFAULT_CASES)[0], stages=())
    old = observation(case.id, status="error", calls=3, tokens=100)
    monkeypatch.setattr("benchmarks.workflow.run_workflow._invoke_once", lambda *args: observation(case.id))
    recovered = run_command([case], "unused", 1, prior=[old], checkpoint_path=tmp_path / "observations.jsonl")[0]
    assert recovered.cumulative_tokens == 110
    assert recovered.metadata["resource_accounting"]["resources"]["cumulative_tokens"]["total"] is None


def test_real_process_timeout_retains_unknown_spend_and_measured_wall_time():
    result = _invoke({}, "a", [sys.executable, "-c", "import time; time.sleep(2)"], 0.03, retries=1)
    accounting = result.metadata["resource_accounting"]
    assert accounting["attempts"] == 2
    assert accounting["budget_exceeded"] is True
    assert all(accounting["resources"][key]["total"] is None for key in FIELDS)
    assert all(item["harness_wall_seconds"] >= 0.025 for item in result.metadata["attempt_receipts"])


def test_parallel_real_process_journal_and_normalized_totals(tmp_path):
    cases = [replace(case, stages=()) for case in load_cases(DEFAULT_CASES)[:3]]
    code = "import json,sys; p=json.loads(sys.stdin.readline()); print(json.dumps(dict(case_id=p['id'],status='answered',support='supported',tool_calls=1,cumulative_tokens=10,response_tokens=2,latency_seconds=0.1,cost_usd=0.5)))"
    results = run_command(cases, shlex.join([sys.executable, "-c", code]), 10, jobs=3,
                          checkpoint_path=tmp_path / "observations.jsonl")
    summary = score_run(cases, results)
    assert summary["resource_accounting"]["attempts"] == 3
    assert summary["resource_accounting"]["resources"]["cost_usd"]["total"] == 1.5
    assert summary["resource_accounting"]["budget_accounting_complete"] is True


def test_missing_resource_fields_survive_serialization_as_unknown(tmp_path):
    value = Observation.from_dict({"case_id": "a", "status": "answered", "support": "supported"})
    assert summarize([receipt(value)])["resources"]["tool_calls"]["total"] is None
    from benchmarks.workflow.provenance import write_observations
    path = tmp_path / "observations.jsonl"
    write_observations(path, [value])
    assert load_observations(path)[0].metadata["resource_fields"] == []
    legacy = json.loads(path.read_text())
    legacy["metadata"] = {}
    path.write_text(json.dumps(legacy) + "\n")
    assert load_observations(path)[0].metadata["resource_fields"] == []


@pytest.mark.parametrize("field,value", [("tool_calls", -1), ("tool_calls", True), ("tool_calls", 1.5),
    ("cumulative_tokens", float("inf")), ("latency_seconds", float("nan")), ("cost_usd", -1)])
def test_invalid_resource_values_are_not_clamped_to_zero(field, value):
    with pytest.raises(ValueError):
        Observation.from_dict({"case_id": "a", "status": "answered", "support": "supported", field: value})


def test_unknown_resources_cannot_pass_workflow_budget_gate():
    cases = load_cases(DEFAULT_CASES)
    values = [Observation.from_dict({"case_id": case.id, "status": "answered", "support": "supported"}) for case in cases]
    scored = score_run(cases, values, "unknown")
    assert compare([scored], "unknown")["arms"][0]["gates"]["resource_accounting_complete"] is False


def test_real_cli_export_does_not_turn_missing_measurements_into_free_runs(tmp_path):
    code = "import json,sys; p=json.loads(sys.stdin.readline()); print(json.dumps(dict(case_id=p['id'],status='answered',support='supported')))"
    completed = subprocess.run([sys.executable, "-m", "benchmarks.workflow.run_workflow",
        "--arm", "unmeasured", "--output-dir", str(tmp_path), "--infrastructure-retries", "0",
        "--arm-command", shlex.join([sys.executable, "-c", code])], capture_output=True, text=True, timeout=30)
    assert completed.returncode in (0, 2), completed.stderr
    rows = [json.loads(line) for line in (tmp_path / "gnomonbench.jsonl").read_text().splitlines()]
    assert rows
    assert all(row["tool_calls"] is None and row["cost_usd"] is None and row["run_tokens"] is None for row in rows)
    assert all(row["resource_accounting"]["budget_accounting_complete"] is False for row in rows)
    assert (tmp_path / "observations.attempts.sqlite3").is_file()


def test_journal_rejects_other_databases_and_conflicting_receipts(tmp_path):
    path = tmp_path / "unrelated.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE existing_data(value)")
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="not a Workflow"):
        AttemptJournal(path)
    db = sqlite3.connect(path)
    assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("existing_data",)]
    db.close()
    journal = AttemptJournal()
    try:
        item = receipt(observation("a"))
        journal.import_receipt(item)
        journal.import_receipt(item)
        assert len(journal.for_case("a")) == 1
        with pytest.raises(ValueError, match="conflicting"):
            journal.import_receipt({**item, "status": "error"})
    finally:
        journal.close()


def test_cached_provider_history_is_not_double_charged_by_arm_normalization(tmp_path):
    import time
    from benchmarks.common.openrouter import OpenRouterClient
    from benchmarks.workflow.agent_adapter import _normalize
    from benchmarks.workflow.run_workflow import case_payload
    original = OpenRouterClient("test/model", api_key="unused", sample_cache_dir=tmp_path)
    original._persist_request_record("case", successful=True,
        usage={"prompt_tokens": 60, "completion_tokens": 40, "cost": 2}, transport_attempts=1)
    resumed = OpenRouterClient("test/model", api_key="unused", sample_cache_dir=tmp_path)
    resumed.total_transport_attempts += 1
    resumed._account({"usage": {"prompt_tokens": 6, "completion_tokens": 4, "cost": 0.2}})
    case = load_cases(DEFAULT_CASES)[0]
    normalized = _normalize(case_payload(case), {"status": "answered"}, calls=1,
                            client=resumed, started=time.time(), tool_names=[])
    assert resumed.total_prompt_tokens + resumed.total_completion_tokens == 110
    assert normalized["cumulative_tokens"] == 10
    assert normalized["cost_usd"] == 0.2
    assert normalized["metadata"]["cached_history_not_matched_to_attempt_journal"] is True
    assert "cumulative_tokens" not in normalized["metadata"]["resource_fields"]
    assert "cost_usd" not in normalized["metadata"]["resource_fields"]


@pytest.mark.parametrize("earlier,expected", [(True, True), (None, None), (False, False)])
def test_recovery_cannot_erase_a_prior_leak_or_unmeasured_safety(monkeypatch, earlier, expected):
    responses = iter([replace(observation("a", status="error"), temporal_leakage=earlier),
                      replace(observation("a"), temporal_leakage=False)])
    monkeypatch.setattr("benchmarks.workflow.run_workflow._invoke_once", lambda *args: next(responses))
    result = _invoke({}, "a", [], 1, retries=1)
    assert result.temporal_leakage is expected


@pytest.mark.parametrize("earlier,expected", [(True, True), (None, None), (False, False)])
def test_budget_outcomes_survive_resume_and_normalized_export(monkeypatch, tmp_path, earlier, expected):
    from benchmarks.workflow.matched import normalized_rows
    case = replace(load_cases(DEFAULT_CASES)[0], stages=())
    checkpoint = tmp_path / "observations.jsonl"
    failed = observation(case.id, status="error")
    failed = replace(failed, metadata={**failed.metadata, "budget_exceeded": earlier})
    monkeypatch.setattr("benchmarks.workflow.run_workflow._invoke_once", lambda *args: failed)
    run_command([case], "unused", 1, checkpoint_path=checkpoint)
    recovered = observation(case.id)
    recovered = replace(recovered, metadata={**recovered.metadata, "budget_exceeded": False})
    monkeypatch.setattr("benchmarks.workflow.run_workflow._invoke_once", lambda *args: recovered)
    result = run_command([case], "unused", 1, prior=load_observations(checkpoint), checkpoint_path=checkpoint)
    rows = normalized_rows(score_run([case], result))
    assert rows[0]["budget_exceeded"] is expected
    assert rows[0]["resource_accounting"]["attempts"] == 2


def test_budget_receipts_preserve_historical_uncertainty_and_reject_nonbooleans():
    value = observation("a")
    item = receipt(value)
    item.pop("budget_exceeded")  # Pre-field persisted receipt remains readable, not false.
    assert summarize([item])["budget_exceeded"] is None
    for flag in (True, False):
        candidate = replace(value, metadata={**value.metadata, "budget_exceeded": flag})
        assert receipt(candidate, historical=True)["budget_exceeded"] is (True if flag else None)
    with pytest.raises(ValueError, match="budget measurement"):
        summarize([{**item, "budget_exceeded": 0}])
    with pytest.raises(ValueError, match="budget measurement"):
        receipt(replace(value, metadata={**value.metadata, "budget_exceeded": "false"}))
