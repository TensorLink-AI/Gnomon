import json

import pytest

from gnomon.agent_eval import compare_runs


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_compare_runs_reports_agent_uplift(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    _write(baseline, [
        {"task_id": "inventory", "success": False, "invented_number": True},
        {"task_id": "capacity", "success": True},
    ])
    _write(treatment, [
        {"task_id": "inventory", "success": True},
        {"task_id": "capacity", "success": True},
    ])

    result = compare_runs(str(baseline), str(treatment))

    assert result["absolute_success_uplift"] == 0.5
    assert result["relative_error_reduction"] == 1.0
    assert result["safety_delta"]["invented_number"] == -0.5


def test_compare_runs_requires_same_tasks(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    _write(baseline, [{"task_id": "a", "success": True}])
    _write(treatment, [{"task_id": "b", "success": True}])

    with pytest.raises(ValueError, match="identical task_id"):
        compare_runs(str(baseline), str(treatment))


def test_harness_voided_rows_are_excluded_pairwise(tmp_path):
    # Task "capped" was ended by the harness in the treatment arm
    # (row_abstained). It is not a wrong answer, so it must leave both
    # arms' rates — including the baseline's success on the same task.
    baseline = tmp_path / "baseline.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    _write(baseline, [
        {"task_id": "a", "success": False},
        {"task_id": "b", "success": True},
        {"task_id": "capped", "success": True},
    ])
    _write(treatment, [
        {"task_id": "a", "success": True},
        {"task_id": "b", "success": True},
        {"task_id": "capped", "success": False,
         "row_abstained": "cap:tokens exceeded"},
    ])

    result = compare_runs(str(baseline), str(treatment))

    assert result["tasks_total"] == 3
    assert result["tasks_voided_by_harness"] == 1
    assert result["baseline"]["task_success"] == 0.5
    assert result["treatment"]["task_success"] == 1.0
    assert result["absolute_success_uplift"] == 0.5


def test_all_tasks_voided_yields_no_uplift_claim(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    _write(baseline, [{"task_id": "a", "success": True}])
    _write(treatment, [{"task_id": "a", "success": False,
                        "row_abstained": "cap:rounds"}])

    result = compare_runs(str(baseline), str(treatment))

    assert result["tasks_voided_by_harness"] == 1
    assert result["absolute_success_uplift"] is None
    assert "voided" in result["interpretation"]
