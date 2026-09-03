from __future__ import annotations

import json

import pytest

from benchmarks.degradedadmissionbench import run as bench


def test_generator_never_passes_future_to_forecaster(tmp_path, monkeypatch):
    observed_lengths: list[int] = []
    original = bench.evaluate

    def guarded(values, *args, **kwargs):
        observed_lengths.append(len(values))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(bench, "evaluate", guarded)
    row = bench._case("stable_trend", 3, 0, 123)
    assert observed_lengths == [bench.HISTORY]
    assert row["future_observations_used_by_forecaster"] == 0


def test_reference_comparison_refuses_different_matrix(tmp_path):
    identity = {
        "evaluated_commit": "new", "seed": 2,
        "families": list(bench.FAMILIES), "horizons": list(bench.HORIZONS),
        "cases_per_family_horizon": bench.CASES_PER_FAMILY_HORIZON,
        "history": bench.HISTORY,
    }
    reference = {
        "run_identity": {**identity, "evaluated_commit": "old", "seed": 1},
        "rows": [], "by_family": {}, "overall": {},
    }
    with pytest.raises(ValueError, match="seed"):
        bench.summarize([], identity, reference)


def test_resume_refuses_revision_mismatch(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    identity = {
        "schema_version": 1,
        "benchmark": "degraded-structural-admission",
        "evaluated_commit": "old",
        "seed": 7,
        "families": list(bench.FAMILIES),
        "horizons": list(bench.HORIZONS),
        "cases_per_family_horizon": bench.CASES_PER_FAMILY_HORIZON,
        "history": bench.HISTORY,
    }
    (output / "run-identity.json").write_text(json.dumps(identity))
    monkeypatch.setattr(bench, "code_revision", lambda: "new")
    with pytest.raises(ValueError, match="identity differs"):
        bench.run(7, output, resume=True)

