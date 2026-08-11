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
    # The per-arm counters describe the arm's file, not the pre-filtered
    # rows the rates were computed over: the baseline ran 3 tasks, none
    # voided by its own harness, and 2 survived the pairwise exclusion.
    assert result["baseline"]["runs"] == 3
    assert result["baseline"]["runs_voided_by_harness"] == 0
    assert result["baseline"]["runs_graded"] == 2
    assert result["treatment"]["runs"] == 3
    assert result["treatment"]["runs_voided_by_harness"] == 1
    assert result["treatment"]["runs_graded"] == 2


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


def test_duplicate_task_ids_are_rejected(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    _write(baseline, [
        {"task_id": "a", "success": True},
        {"task_id": "a", "success": False},
    ])
    _write(treatment, [{"task_id": "a", "success": True}])

    with pytest.raises(ValueError, match="duplicate task_id"):
        compare_runs(str(baseline), str(treatment))


def test_unmeasured_safety_fields_are_none_not_zero(tmp_path):
    # No row in either file carries a safety field: the deltas must be
    # unmeasured, not a reassuring 0.0.
    baseline = tmp_path / "baseline.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    _write(baseline, [{"task_id": "a", "success": False}])
    _write(treatment, [{"task_id": "a", "success": True}])

    result = compare_runs(str(baseline), str(treatment))

    assert result["baseline"]["temporal_leakage"] is None
    assert result["safety_delta"]["invented_number"] is None
    assert "unmeasured" in result["safety_note"]
    assert "temporal_leakage" in result["safety_note"]


def test_noise_level_uplift_is_not_declared_an_improvement(tmp_path):
    # One discordant pair out of four: uplift +0.25, exact McNemar p=1.0.
    baseline = tmp_path / "baseline.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    _write(baseline, [
        {"task_id": "a", "success": False},
        {"task_id": "b", "success": True},
        {"task_id": "c", "success": True},
        {"task_id": "d", "success": True},
    ])
    _write(treatment, [
        {"task_id": "a", "success": True},
        {"task_id": "b", "success": True},
        {"task_id": "c", "success": True},
        {"task_id": "d", "success": True},
    ])

    result = compare_runs(str(baseline), str(treatment))

    assert result["absolute_success_uplift"] == 0.25
    assert result["success_test"]["p_value"] == 1.0
    assert "not statistically distinguishable" in result["interpretation"]


def test_rows_missing_latency_do_not_average_in_as_zero(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    _write(baseline, [
        {"task_id": "a", "success": True, "latency_seconds": 4.0},
        {"task_id": "b", "success": True},
    ])
    _write(treatment, [
        {"task_id": "a", "success": True, "latency_seconds": 2.0},
        {"task_id": "b", "success": True, "latency_seconds": 2.0},
    ])

    result = compare_runs(str(baseline), str(treatment))

    assert result["baseline"]["average_latency_seconds"] == 4.0
    assert result["treatment"]["average_latency_seconds"] == 2.0
    assert result["baseline"]["average_cost_usd"] is None


def test_null_latency_is_unmeasured_not_a_measured_zero(tmp_path):
    # An explicit JSON null passes a key-presence check; `float(None or 0)`
    # then averaged it in as a measured 0.0 — the same deflation the
    # missing-key fix removed, just spelled differently.
    baseline = tmp_path / "baseline.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    _write(baseline, [
        {"task_id": "a", "success": True, "latency_seconds": 4.0},
        {"task_id": "b", "success": True, "latency_seconds": None},
    ])
    _write(treatment, [
        {"task_id": "a", "success": True, "cost_usd": None},
        {"task_id": "b", "success": True, "cost_usd": None},
    ])

    result = compare_runs(str(baseline), str(treatment))

    assert result["baseline"]["average_latency_seconds"] == 4.0
    assert result["treatment"]["average_cost_usd"] is None
