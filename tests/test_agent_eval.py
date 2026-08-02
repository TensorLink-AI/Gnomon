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
