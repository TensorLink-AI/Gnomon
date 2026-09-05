import json
import math

import pytest

from gnomon.agent_eval import compare_rows, load_runs, summarize_rows, _two_sided_binomial
from gnomon.cli import main


def row(task_id, success=False, **fields):
    return {"task_id": task_id, "success": success, **fields}


def test_capped_hard_tasks_cannot_manufacture_uplift_or_hide_spend():
    baseline = [row("easy", True, completed=True, cost_usd=1), row("hard", True, completed=True, cost_usd=2)]
    treatment = [row("easy", True, completed=True, cost_usd=1),
                 row("hard", False, row_abstained="cap:tokens exceeded", cost_usd=20, tool_calls=9)]
    comparison = compare_rows(baseline, treatment)
    assert comparison["absolute_success_uplift"] == -0.5
    assert comparison["treatment"]["average_cost_usd"] == 10.5
    assert comparison["treatment"]["resources"]["cost_usd"]["total"] == 21
    assert comparison["treatment"]["measurement_coverage"]["tool_calls"]["task_ids"] == ["hard"]
    assert comparison["conditional_completed_pairs"]["treatment_success"] == 1
    assert comparison["conditional_completed_pairs"]["task_ids"] == ["easy"]
    assert comparison["success_test"]["treatment_broke"] == 1
    assert comparison["treatment"]["measurement_coverage"]["budget_exceeded"]["positive"] == 1
    assert comparison["treatment"]["measurement_coverage"]["budget_exceeded"]["unmeasured"] == 1


def test_unknown_completion_and_sparse_safety_have_explicit_denominators():
    rows = [row("correct", True, temporal_leakage=False),
            row("wrong_or_unfinished", False, temporal_leakage=None),
            row("wrong", False, completed=True)]
    summary = summarize_rows(rows)
    assert summary["task_success"] == 1 / 3
    assert summary["completion_rate"] == 1  # Only two known-completed rows, not 3/3.
    assert summary["completion_bounds"] == {"lower": 2 / 3, "upper": 1}
    assert summary["measurement_coverage"]["completed"]["unmeasured"] == 1
    assert summary["temporal_leakage"] == 0
    assert summary["measurement_coverage"]["temporal_leakage"]["measured"] == 1
    assert summary["measurement_coverage"]["temporal_leakage"]["unmeasured"] == 2
    assert summary["error"] is None
    assert summary["appropriate_abstention"] is None


def test_safety_deltas_use_exact_paired_measurements_not_different_cohorts():
    baseline = [row("a", True, invented_number=True), row("b", True, invented_number=False)]
    treatment = [row("b", True, invented_number=True), row("a", True)]
    comparison = compare_rows(baseline, treatment)
    assert comparison["safety_delta"]["invented_number"] == 1
    assert comparison["safety_pairs"]["invented_number"]["task_ids"] == ["b"]
    assert comparison["safety_pairs"]["warning_omission"]["measured_pairs"] == 0


def test_errors_and_abstention_are_not_conflated_with_grading():
    summary = summarize_rows([
        row("refusal", False, status="abstained", appropriate_abstention=True),
        row("error", True, status="error", error=False, completed=True, cost_usd=5),
        row("cap", True, row_abstained="cap:wall_clock", budget_exceeded=False),
        row("wrong", False, completed=True, accuracy=0.25),
    ])
    assert summary["task_success"] == 0  # Contradictory success flags cannot authorize delivery.
    assert summary["completion_bounds"] == {"lower": 0.5, "upper": 0.5}
    assert summary["measurement_coverage"]["error"]["positive"] == 1
    assert summary["measurement_coverage"]["budget_exceeded"]["positive"] == 1
    assert summary["appropriate_abstention"] == 1
    assert summary["conditional_accuracy"] == 0.25


def test_null_numeric_values_are_unknown_and_all_attempt_totals_need_full_coverage():
    summary = summarize_rows([row("a", True, cost_usd=4), row("b", cost_usd=None), row("c")])
    assert summary["average_cost_usd"] == 4
    assert summary["resources"]["cost_usd"]["observed_total"] == 4
    assert summary["resources"]["cost_usd"]["total"] is None
    assert summary["measurement_coverage"]["cost_usd"]["unmeasured"] == 2
    assert summary["calibration"]["brier_score"] is None


def test_numeric_overflow_is_disclosed_without_nonfinite_json():
    summary = summarize_rows([row("a", True, cost_usd=1e308), row("b", cost_usd=1e308)])
    assert summary["average_cost_usd"] == 1e308
    assert summary["resources"]["cost_usd"]["total_overflow"] is True
    assert summary["resources"]["cost_usd"]["observed_total"] is None
    json.dumps(summary, allow_nan=False)


def test_probability_bins_and_brier_include_failed_delivery_without_calibration_claim():
    summary = summarize_rows([
        row("zero", False, success_probability=0),
        row("uncertain", True, success_probability=0.5),
        row("cap", False, success_probability=1, row_abstained="cap:rounds"),
        row("unmeasured", True),
    ])
    calibration = summary["calibration"]
    assert calibration["brier_score"] == pytest.approx(1.25 / 3)
    assert calibration["unmeasured"] == 1
    assert [b["lower"] for b in calibration["bins"]] == [0, 0.5, 0.9]
    assert calibration["bins"][-1]["observed_success"] == 0
    assert calibration["claim"] == "descriptive_only_not_proof_of_calibration"


@pytest.mark.parametrize("k,n,expected", [(0, 0, 1), (0, 1, 1), (0, 6, 0.03125), (1, 6, 0.21875), (3, 6, 1)])
def test_exact_paired_binomial_oracles(k, n, expected):
    assert _two_sided_binomial(k, n) == expected


def test_many_discordant_pairs_do_not_overflow_or_depend_on_input_order():
    baseline = [row(str(i), False, completed=True) for i in range(1500)]
    treatment = [row(str(i), True) for i in reversed(range(1500))]
    comparison = compare_rows(baseline, treatment)
    assert comparison["success_test"]["treatment_fixed"] == 1500
    assert math.isfinite(comparison["success_test"]["p_value"])
    assert comparison["absolute_success_uplift"] == 1


@pytest.mark.parametrize("invalid", [
    [], {}, [row("a", "false")], [row(1)], [row("")], [row("a"), row("a")],
    [row("a", cost_usd=float("nan"))], [row("a", cost_usd=float("inf"))],
    [row("a", cost_usd=-1)], [row("a", cost_usd="1")], [row("a", cost_usd=True)],
    [row("a", cost_usd=10**1000)], [row("a", tool_calls=1.0)], [row("a", run_tokens=-1)],
    [row("a", success_probability=1.01)], [row("a", accuracy=-1)],
    [row("a", temporal_leakage="false")], [row("a", completed=0)],
    [row("a", row_abstained={})], [row("a", error=[])], [row("a", status=[])],
])
def test_invalid_rows_fail_before_reporting(invalid):
    with pytest.raises(ValueError):
        summarize_rows(invalid)


def test_json_import_rejects_nonfinite_nested_metadata_and_oversized_rows(tmp_path):
    path = tmp_path / "bad.jsonl"
    for value in ("NaN", "Infinity", "1e999"):
        path.write_text('{"task_id":"a","success":false,"metadata":{"value":' + value + '}}\n')
        with pytest.raises(ValueError, match="finite"):
            load_runs(str(path))
    path.write_bytes(b" " * 1_048_577)
    with pytest.raises(ValueError, match="byte limit"):
        load_runs(str(path))
    path.write_text('{"task_id":"a","success":false,"success":true}\n')
    with pytest.raises(ValueError, match="duplicate fields"):
        load_runs(str(path))
    path.write_text('{"task_id":"a","success":false,"null":null,"metadata":' + '[' * 2000 + '0' + ']' * 2000 + '}')
    with pytest.raises(ValueError, match="nesting"):
        load_runs(str(path))


def test_incompatible_grading_basis_is_refused():
    with pytest.raises(ValueError, match="success_basis"):
        compare_rows([row("a", True, success_basis="completion")], [row("a", True, success_basis="accuracy")])


def test_jsonl_row_limit_is_enforced_and_null_before_deep_metadata_cannot_bypass_depth(tmp_path, monkeypatch):
    from gnomon import agent_eval
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(row("a")) + "\n" + json.dumps(row("b")) + "\n")
    monkeypatch.setattr(agent_eval, "MAX_RUNS", 1)
    with pytest.raises(ValueError, match="row limit"):
        load_runs(str(path))
    path.write_text('{"task_id":"a","success":false,"null":null,"deep":' + '[' * 130 + 'null' + ']' * 130 + '}')
    with pytest.raises(ValueError, match="nesting"):
        load_runs(str(path))


def test_real_cli_outputs_all_task_metrics(tmp_path, capsys):
    baseline, treatment = tmp_path / "baseline.jsonl", tmp_path / "treatment.jsonl"
    baseline.write_text(json.dumps(row("a", True, cost_usd=1)) + "\n")
    treatment.write_text(json.dumps(row("a", row_abstained="cap:tokens", cost_usd=8)) + "\n")
    assert main(["eval", "compare", "--baseline", str(baseline), "--treatment", str(treatment)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["absolute_success_uplift"] == -1
    assert comparison["treatment"]["resources"]["cost_usd"]["total"] == 8
