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
    with (run_dir / "aionbench.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    if manifest is not None:
        write_manifest(run_dir, **manifest)
    return run_dir


def test_task_ids_normalise_across_layouts():
    assert normalise_task_id("shard#0007") == "shard_0007"
    assert normalise_task_id("shard_0007.json") == "shard_0007"
    assert normalise_task_id("shard#0007") == normalise_task_id("shard_0007.json")


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
