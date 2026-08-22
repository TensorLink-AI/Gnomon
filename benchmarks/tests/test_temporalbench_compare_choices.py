"""Field-level paired TemporalBench choice reporting."""

import json

import pytest

from benchmarks.common.manifest import write_manifest
from benchmarks.temporalbench.compare_choices import compare


def _run(root, name, revision, fields):
    directory = root / name
    details = directory / "details"
    details.mkdir(parents=True)
    for task_id, values in fields.items():
        (details / f"{task_id}.json").write_text(json.dumps({
            "verdict": {"choice": {"fields": values}},
        }), encoding="utf-8")
    write_manifest(directory, benchmark="temporalbench",
                   target="official-all-tiers", condition=name,
                   code_revision=revision)
    return directory


def test_choice_report_pairs_fields_not_row_counts(tmp_path):
    baseline = _run(tmp_path, "base", "aaa", {
        "case_T1": {"trend": False, "volatility": True},
        "case_T3": {"trend": False},
    })
    treatment = _run(tmp_path, "treat", "bbb", {
        "case_T1": {"trend": True, "volatility": True},
        "case_T3": {"trend": True},
    })
    result = compare(baseline, treatment)
    assert result["overall"]["questions"] == 3
    assert result["overall"]["baseline_correct"] == 1
    assert result["overall"]["treatment_correct"] == 3
    assert result["overall"]["paired_test"]["treatment_fixed"] == 2
    assert result["by_tier"]["T1"]["questions"] == 2


def test_choice_report_refuses_mismatched_targets(tmp_path):
    baseline = _run(tmp_path, "base", "aaa", {
        "case_T1": {"trend": False},
    })
    treatment = _run(tmp_path, "treat", "bbb", {
        "case_T1": {"trend": True},
    })
    write_manifest(treatment, benchmark="temporalbench", target="other")
    with pytest.raises(ValueError, match="target differs"):
        compare(baseline, treatment)
