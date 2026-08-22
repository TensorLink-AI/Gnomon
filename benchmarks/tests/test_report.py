"""Tests for the comparison report: matching, refusal, and paired tests."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.manifest import (  # noqa: E402
    incompatibilities,
    read_manifest,
    write_manifest,
)
from benchmarks.report import (  # noqa: E402
    comparison_costs,
    compare,
    derived_metrics,
    load_run,
    mcnemar,
    normalise_task_id,
    sign_test,
)


def _write_run(root, name, rows, manifest=None):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    with (run_dir / "gnomonbench.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    if manifest is not None:
        write_manifest(run_dir, **manifest)
    return run_dir


def test_task_ids_normalise_across_layouts():
    assert normalise_task_id("shard#0007") == "shard_0007"
    assert normalise_task_id("shard_0007.json") == "shard_0007"
    assert normalise_task_id("shard#0007") == normalise_task_id("shard_0007.json")


def test_comparison_costs_excludes_unrelated_runs():
    runs = {
        "control": {"summary": {},
                    "manifest": {"llm_usage": {"requests": 2}}},
        "treatment": {"summary": {"llm_usage": {"requests": 3}}},
        "unrelated": {"summary": {"llm_usage": {"requests": 999}}},
    }
    costs = comparison_costs(runs, [["control", "treatment"]])
    assert set(costs) == {"control", "treatment"}
    assert costs["control"]["requests"] == 2
    assert costs["treatment"]["requests"] == 3


def test_manifest_round_trips_and_flags_target_mismatch(tmp_path):
    left = tmp_path / "a"
    right = tmp_path / "b"
    write_manifest(left, benchmark="mtbench", target="time")
    write_manifest(right, benchmark="mtbench", target="macd")
    assert read_manifest(left)["benchmark"] == "mtbench"
    problems = incompatibilities(read_manifest(left), read_manifest(right))
    assert problems and "target differs" in problems[0]


def test_unknown_field_is_not_treated_as_agreement():
    # A missing manifest field must not read as "same"; it is reported as
    # unverified instead.
    assert incompatibilities({"benchmark": "mtbench"}, {}) == []


def test_comparison_refused_when_targets_differ(tmp_path):
    rows = [{"task_id": f"t{i}", "success": True} for i in range(5)]
    _write_run(tmp_path, "ctrl", rows, {"benchmark": "mtbench", "target": "macd"})
    _write_run(tmp_path, "treat", rows, {"benchmark": "mtbench", "target": "time"})
    result = compare(load_run(tmp_path / "ctrl"), load_run(tmp_path / "treat"))
    assert result["comparable"] is False
    assert "target differs" in result["reason"]


def test_comparison_refused_when_no_tasks_are_shared(tmp_path):
    _write_run(tmp_path, "ctrl", [{"task_id": "a", "success": True}],
               {"benchmark": "x", "target": "y"})
    _write_run(tmp_path, "treat", [{"task_id": "b", "success": True}],
               {"benchmark": "x", "target": "y"})
    result = compare(load_run(tmp_path / "ctrl"), load_run(tmp_path / "treat"))
    assert result["comparable"] is False
    assert "no task ids in common" in result["reason"]


def test_means_use_only_the_matched_subset(tmp_path):
    # The treatment also answered an extra, very easy task; it must not
    # improve its reported mean.
    _write_run(tmp_path, "ctrl",
               [{"task_id": "a", "mse": 10.0}, {"task_id": "b", "mse": 20.0}],
               {"benchmark": "x", "target": "y"})
    _write_run(tmp_path, "treat",
               [{"task_id": "a", "mse": 12.0}, {"task_id": "b", "mse": 18.0},
                {"task_id": "easy", "mse": 0.001}],
               {"benchmark": "x", "target": "y"})
    result = compare(load_run(tmp_path / "ctrl"), load_run(tmp_path / "treat"),
                     metric="mse")
    assert result["matched_tasks"] == 2
    assert result["treatment_only"] == 1
    assert result["metrics"]["mse"]["treatment_mean"] == 15.0


def test_missing_manifest_warns_rather_than_silently_comparing(tmp_path):
    rows = [{"task_id": "a", "success": True}, {"task_id": "b", "success": False}]
    _write_run(tmp_path, "ctrl", rows)
    _write_run(tmp_path, "treat", rows)
    result = compare(load_run(tmp_path / "ctrl"), load_run(tmp_path / "treat"))
    assert result["comparable"] is True
    assert "could not be verified" in result["warning"]


def test_mcnemar_counts_discordant_pairs():
    baseline = {"a": False, "b": True, "c": True, "d": False}
    treatment = {"a": True, "b": False, "c": True, "d": False}
    result = mcnemar(baseline, treatment)
    assert result["treatment_fixed"] == 1
    assert result["treatment_broke"] == 1
    assert result["p_value"] == 1.0


def test_sign_test_respects_metric_direction():
    baseline = {f"t{i}": 10.0 for i in range(8)}
    treatment = {f"t{i}": 5.0 for i in range(8)}
    lower = sign_test(baseline, treatment, lower_is_better=True)
    higher = sign_test(baseline, treatment, lower_is_better=False)
    assert lower["treatment_wins"] == 8
    assert higher["treatment_wins"] == 0
    assert lower["p_value"] < 0.01


def test_derived_metrics_from_a_raw_trajectory():
    metrics = derived_metrics({"ground_truth": [1.0, 2.0], "predict": [2.0, 4.0]})
    assert metrics["mse"] == 2.5
    assert metrics["mae"] == 1.5
    # |1-2|/1 and |2-4|/2 are both 100% errors.
    assert round(metrics["mape"], 2) == 100.0


def test_derived_metrics_absent_without_a_trajectory():
    assert derived_metrics({"task_id": "a", "success": True}) == {}


def test_penalized_mean_charges_abstentions_at_the_baseline_score(tmp_path):
    # The treatment answers the easy task and abstains on the hard one.
    # A scored-only mean would reward that; the penalized mean must not.
    _write_run(tmp_path, "ctrl",
               [{"task_id": "easy", "mse": 1.0}, {"task_id": "hard", "mse": 9.0}],
               {"benchmark": "x", "target": "y"})
    _write_run(tmp_path, "treat", [{"task_id": "easy", "mse": 2.0}],
               {"benchmark": "x", "target": "y"})
    result = compare(load_run(tmp_path / "ctrl"), load_run(tmp_path / "treat"),
                     metric="mse")
    penalized = result["metrics"]["mse"]["penalized"]
    assert penalized["abstentions_imputed"] == 1
    assert penalized["baseline_mean"] == 5.0        # (1 + 9) / 2
    assert penalized["treatment_mean"] == 5.5       # (2 + 9) / 2, hard imputed


def test_no_penalized_mean_when_nothing_was_abstained(tmp_path):
    rows = [{"task_id": "a", "mse": 1.0}, {"task_id": "b", "mse": 2.0}]
    _write_run(tmp_path, "ctrl", rows, {"benchmark": "x", "target": "y"})
    _write_run(tmp_path, "treat", rows, {"benchmark": "x", "target": "y"})
    result = compare(load_run(tmp_path / "ctrl"), load_run(tmp_path / "treat"),
                     metric="mse")
    assert "penalized" not in result["metrics"]["mse"]


def test_score_is_lower_is_better(tmp_path):
    # LeakTrap's `score` is a WAPE. The old higher-is-better default
    # inverted the sign test: a treatment with uniformly LOWER error was
    # reported with zero wins.
    _write_run(tmp_path, "ctrl",
               [{"task_id": f"t{i}", "score": 10.0} for i in range(6)],
               {"benchmark": "leakage-trap", "target": "y"})
    _write_run(tmp_path, "treat",
               [{"task_id": f"t{i}", "score": 5.0} for i in range(6)],
               {"benchmark": "leakage-trap", "target": "y"})
    result = compare(load_run(tmp_path / "ctrl"), load_run(tmp_path / "treat"),
                     metric="score")
    entry = result["metrics"]["score"]
    assert entry["lower_is_better"] is True
    assert entry["direction_recognised"] is True
    assert entry["test"]["treatment_wins"] == 6


def test_unrecognised_metric_direction_is_flagged_not_assumed_silently(tmp_path):
    _write_run(tmp_path, "ctrl",
               [{"task_id": f"t{i}", "wibble": 10.0} for i in range(4)],
               {"benchmark": "x", "target": "y"})
    _write_run(tmp_path, "treat",
               [{"task_id": f"t{i}", "wibble": 5.0} for i in range(4)],
               {"benchmark": "x", "target": "y"})
    result = compare(load_run(tmp_path / "ctrl"), load_run(tmp_path / "treat"),
                     metric="wibble")
    entry = result["metrics"]["wibble"]
    assert entry["direction_recognised"] is False
    assert "inverted" in entry["direction_warning"]


def test_harness_voided_rows_stay_out_of_the_success_test(tmp_path):
    # Task b was voided by the harness in the treatment arm (a breached
    # cap). Counting it as a failure reported a harness cap as a model
    # failure; it is excluded pairwise and counted instead.
    _write_run(tmp_path, "ctrl",
               [{"task_id": "a", "success": True},
                {"task_id": "b", "success": True}],
               {"benchmark": "x", "target": "y"})
    _write_run(tmp_path, "treat",
               [{"task_id": "a", "success": True},
                {"task_id": "b", "success": False,
                 "row_abstained": "cap:tokens exceeded"}],
               {"benchmark": "x", "target": "y"})
    result = compare(load_run(tmp_path / "ctrl"), load_run(tmp_path / "treat"))
    assert result["success_voided_excluded"] == {
        "pairs": 1, "baseline_voided": 0, "treatment_voided": 1,
        "basis": result["success_voided_excluded"]["basis"],
    }
    assert result["success_rate"] == {"baseline": 1.0, "treatment": 1.0}
    assert result["success_test"]["n"] == 1


def test_success_rate_is_split_by_basis_when_bases_mix(tmp_path):
    # TemporalBench T2/T4 success = completion; T1/T3 = all choices
    # correct. A pooled rate blends the two, so the split is reported.
    rows_ctrl = [
        {"task_id": "f1", "success": True, "success_basis": "completion"},
        {"task_id": "f2", "success": False, "success_basis": "completion"},
        {"task_id": "q1", "success": False,
         "success_basis": "all_choices_correct"},
    ]
    rows_treat = [
        {"task_id": "f1", "success": True, "success_basis": "completion"},
        {"task_id": "f2", "success": True, "success_basis": "completion"},
        {"task_id": "q1", "success": False,
         "success_basis": "all_choices_correct"},
    ]
    _write_run(tmp_path, "ctrl", rows_ctrl, {"benchmark": "x", "target": "y"})
    _write_run(tmp_path, "treat", rows_treat, {"benchmark": "x", "target": "y"})
    result = compare(load_run(tmp_path / "ctrl"), load_run(tmp_path / "treat"))
    split = result["success_by_basis"]
    assert split["completion"] == {"n": 2, "baseline": 0.5, "treatment": 1.0}
    assert split["all_choices_correct"] == {
        "n": 1, "baseline": 0.0, "treatment": 0.0,
    }


def test_bookkeeping_fields_are_not_swept_as_metrics():
    from benchmarks.report import available_metrics

    tasks = {"t1": {"task_id": "t1", "score": 0.4, "shock": 0.25,
                    "no_leak_ceiling": 0.5, "leak_advantage": 0.2,
                    "choice_total": 3, "tool_calls": 5,
                    "latency_seconds": 2.0}}
    assert available_metrics(tasks) == ["score"]


def test_derived_metrics_refuse_a_wrong_length_prediction():
    # zip would score the overlap and understate the error; the adapters'
    # own scorers refuse the same condition, so the join must too.
    assert derived_metrics(
        {"ground_truth": [1.0, 2.0, 3.0], "predict": [1.0, 2.0]}
    ) == {}
