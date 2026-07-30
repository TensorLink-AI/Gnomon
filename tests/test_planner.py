"""Phase 5: TemporalPlan IR, validator, executor, bounded repair, and the
gated low-level surface."""

from datetime import datetime, timezone
from pathlib import Path

import json
import pytest

from aion.contracts import AionError
from aion.execution import execute_plan
from aion.ids import FixedClock
from aion.macros import investigate_change
from aion.plan import (
    MAX_REPAIR_ROUNDS,
    PlanStep,
    TemporalPlan,
    compile_task,
    plan_from_dict,
    repair_plan,
    validate_plan,
)

REPO = Path(__file__).resolve().parent.parent
CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))
NOISE = [0.5, -0.3, 0.2, -0.4, 0.1, 0.3, -0.2, -0.1, 0.4, -0.5]


def _shift_csv(tmp_path: Path) -> Path:
    from datetime import date, timedelta
    start = date(2026, 3, 1)
    values = [100 + NOISE[i % 10] for i in range(20)] + [118 + NOISE[i % 10] for i in range(15)]
    rows = [f"{(start + timedelta(days=i)).isoformat()},{value}" for i, value in enumerate(values)]
    path = tmp_path / "shift.csv"
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _investigate_steps(source: Path) -> list[dict]:
    return [
        {"step_id": "load1", "operator": "load",
         "inputs": {"input": str(source), "time_column": "timestamp", "target_column": "value"},
         "produces": "loaded_series"},
        {"step_id": "regimes", "operator": "regime_detection",
         "inputs": {"timestamps": {"$ref": "load1.series.__default__.timestamps"},
                    "values": {"$ref": "load1.series.__default__.values"}},
         "produces": "regimes"},
        {"step_id": "anomalies", "operator": "anomaly_score",
         "inputs": {"timestamps": {"$ref": "load1.series.__default__.timestamps"},
                    "values": {"$ref": "load1.series.__default__.values"}},
         "produces": "anomalies"},
    ]


# -- Validator ------------------------------------------------------------

def test_validator_accepts_well_formed_plan(tmp_path):
    plan = plan_from_dict({"task": {"outputs": ["regimes"]},
                           "steps": _investigate_steps(_shift_csv(tmp_path))})
    assert validate_plan(plan) == []


def test_validator_catches_unknown_operator_and_forward_ref():
    plan = plan_from_dict({"task": {}, "steps": [
        {"step_id": "a", "operator": "granger_test", "inputs": {}},
        {"step_id": "b", "operator": "difference",
         "inputs": {"values": {"$ref": "c"}, "lag": 1}},
    ]})
    codes = {violation["code"] for violation in validate_plan(plan)}
    assert "UNKNOWN_OPERATOR" in codes
    assert "FORWARD_OR_MISSING_REFERENCE" in codes


def test_validator_catches_leakage_and_missing_inputs():
    plan = plan_from_dict({"task": {}, "steps": [
        {"step_id": "a", "operator": "anomaly_score",
         "inputs": {"timestamps": "/etc/data.csv"}},
    ]})
    codes = {violation["code"] for violation in validate_plan(plan)}
    assert "TEMPORAL_LEAKAGE" in codes
    assert "MISSING_INPUT" in codes


def test_validator_catches_duplicates_budget_and_unproduced_output(tmp_path):
    steps = _investigate_steps(_shift_csv(tmp_path))
    steps.append({**steps[1], "step_id": "regimes2"})  # exact duplicate call
    plan = plan_from_dict({
        "task": {"outputs": ["forecast"], "budget": {"max_steps": 2}},
        "steps": steps,
    })
    codes = {violation["code"] for violation in validate_plan(plan)}
    assert {"DUPLICATE_EQUIVALENT_CALL", "BUDGET_EXCEEDED",
            "REQUESTED_OUTPUT_NOT_PRODUCED"} <= codes


def test_validator_rejects_causal_task():
    plan = plan_from_dict({"task": {"claim_class": "causal"}, "steps": [
        {"step_id": "a", "operator": "cross_correlation",
         "inputs": {"values_a": [1.0], "values_b": [2.0]}},
    ]})
    codes = [violation["code"] for violation in validate_plan(plan)]
    assert codes.count("CLAIM_CLASS_INFEASIBLE") == 2  # step-level and task-level


# -- Executor -------------------------------------------------------------

def test_executor_runs_dag_with_provenance(tmp_path):
    source = _shift_csv(tmp_path)
    plan = plan_from_dict({"task": {"outputs": ["regimes"]},
                           "steps": _investigate_steps(source)})
    payload, run_dir = execute_plan(plan, output=str(tmp_path / "out"), clock=CLOCK)
    assert payload["status"] == "complete"
    assert [record["status"] for record in payload["steps"]] == ["complete"] * 3
    assert payload["outputs"]["regimes"]["classification"] == "regime_shift"
    assert (run_dir / "artifact.json").is_file()


def test_executor_deterministic_replay(tmp_path):
    source = _shift_csv(tmp_path)
    plan = plan_from_dict({"task": {"outputs": ["regimes"]},
                           "steps": _investigate_steps(source)})
    first, first_dir = execute_plan(plan, output=str(tmp_path / "out"), clock=CLOCK)
    second, second_dir = execute_plan(plan, output=str(tmp_path / "out"), clock=CLOCK)
    assert first_dir == second_dir
    assert second["replayed"] is True
    assert second["outputs"] == first["outputs"]


def test_executor_step_cache_shared_across_plans(tmp_path):
    source = _shift_csv(tmp_path)
    base_steps = _investigate_steps(source)
    first = plan_from_dict({"task": {}, "steps": base_steps[:2]})
    execute_plan(first, output=str(tmp_path / "out"), clock=CLOCK)
    # A different plan reusing the same calls hits the content-addressed cache.
    second = plan_from_dict({"task": {}, "steps": base_steps})
    payload, _ = execute_plan(second, output=str(tmp_path / "out"), clock=CLOCK)
    cached_flags = {record["step_id"]: record["cached"] for record in payload["steps"]}
    assert cached_flags["load1"] and cached_flags["regimes"]
    assert cached_flags["anomalies"] is False


def test_executor_isolates_failures_and_skips_downstream(tmp_path):
    plan = plan_from_dict({"task": {}, "steps": [
        {"step_id": "bad", "operator": "difference",
         "inputs": {"values": [1.0], "lag": 5}, "failure_policy": "skip"},
        {"step_id": "dependent", "operator": "difference",
         "inputs": {"values": {"$ref": "bad"}, "lag": 1}},
        {"step_id": "independent", "operator": "difference",
         "inputs": {"values": [1.0, 2.0, 4.0], "lag": 1}},
    ]})
    payload, _ = execute_plan(plan, output=str(tmp_path / "out"), clock=CLOCK)
    status = {record["step_id"]: record["status"] for record in payload["steps"]}
    assert status == {"bad": "failed", "dependent": "skipped", "independent": "complete"}
    assert payload["status"] == "partial"


def test_executor_rejects_invalid_plan(tmp_path):
    plan = plan_from_dict({"task": {}, "steps": []})
    with pytest.raises(AionError) as caught:
        execute_plan(plan, output=str(tmp_path / "out"), clock=CLOCK)
    assert caught.value.code == "INVALID_PLAN"


# -- Macros through the executor -----------------------------------------

def test_all_four_macros_run_through_executor(tmp_path):
    source = _shift_csv(tmp_path)
    common = {"input": str(source), "time_column": "timestamp", "target_column": "value"}
    cases = {
        "investigate_change": common,
        "forecast": {**common, "horizon": 5},
        "decide": {**common, "horizon": 5, "threshold": 130.0,
                   "actions": [{"name": "act"}, {"name": "wait"}]},
        "monitor": {**common, "horizon": 5, "threshold": 130.0,
                    "alert_cost": 1.0, "miss_cost": 9.0},
    }
    for task_type, params in cases.items():
        plan = compile_task(task_type, params)
        assert validate_plan(plan) == []
        payload, _ = execute_plan(plan, output=str(tmp_path / "out"), clock=CLOCK)
        assert payload["status"] == "complete", task_type

    # Identical outputs to the direct macro path.
    direct, _ = investigate_change(
        str(source), time_column="timestamp", target_column="value",
        output=str(tmp_path / "direct"), clock=CLOCK,
    )
    plan = compile_task("investigate_change", common)
    executed, _ = execute_plan(plan, output=str(tmp_path / "out2"), clock=CLOCK)
    via_plan = executed["outputs"]["change_investigation"]
    assert via_plan["results"] == direct["results"]
    assert via_plan["investigation_id"] == direct["investigation_id"]


# -- Bounded repair -------------------------------------------------------

def test_repair_is_bounded_and_feedback_driven(tmp_path):
    source = _shift_csv(tmp_path)
    broken = {"task": {}, "steps": [
        {"step_id": "a", "operator": "not_an_operator", "inputs": {}},
    ]}
    seen_feedback = []

    def propose_fix(payload, violations):
        seen_feedback.append([violation["code"] for violation in violations])
        return {"task": {"outputs": ["loaded_series"]},
                "steps": _investigate_steps(source)[:1]}

    plan, violations, rounds = repair_plan(broken, propose_fix)
    assert rounds == 1
    assert violations == []
    assert seen_feedback == [["UNKNOWN_OPERATOR"]]

    def never_fixes(payload, violations):
        return payload

    _, violations, rounds = repair_plan(broken, never_fixes)
    assert rounds == MAX_REPAIR_ROUNDS
    assert violations


# -- Gated surface --------------------------------------------------------

def test_planner_tools_are_gated(monkeypatch):
    from aion.toolspec import runner_for, visible_tools
    monkeypatch.delenv("AION_EXPERIMENTAL_PLANNER", raising=False)
    names = {tool["name"] for tool in visible_tools()}
    assert "aion_execute_plan" not in names
    assert runner_for("aion_execute_plan") is None

    monkeypatch.setenv("AION_EXPERIMENTAL_PLANNER", "1")
    names = {tool["name"] for tool in visible_tools()}
    assert {"aion_compile_task", "aion_validate_plan",
            "aion_execute_plan", "aion_get_run"} <= names


def test_cli_plan_roundtrip(tmp_path, capsys):
    from aion.cli import main
    source = _shift_csv(tmp_path)
    params = json.dumps({"input": str(source), "time_column": "timestamp",
                         "target_column": "value", "horizon": 5})
    assert main(["plan", "compile", "--task-type", "forecast", "--params", params]) == 0
    compiled = json.loads(capsys.readouterr().out)
    plan_json = json.dumps(compiled["plan"])
    assert main(["plan", "validate", "--plan", plan_json]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert main(["plan", "execute", "--plan", plan_json,
                 "--output", str(tmp_path / "out")]) == 0
    executed = json.loads(capsys.readouterr().out)
    assert executed["status"] == "complete"
    assert executed["outputs"]["forecast"]["results"][0]["forecast_rows"] == 5
