import json

import pytest

from benchmarks.calibrationactionbench.evaluate import run as run_evaluation
from benchmarks.calibrationactionbench.run import run as run_matrix


def test_calibration_action_matrix_is_exact_and_sealed():
    result = run_matrix()

    assert result["cases"] == 12
    assert all(result["gates"].values())


def test_calibration_action_evaluation_is_resumable_and_preserves_points(
        tmp_path):
    first = run_evaluation(2026083701, 1, tmp_path, False)
    resumed = run_evaluation(2026083701, 1, tmp_path, True)

    assert first == resumed
    assert first["cases"] == 6
    assert first["gates"]["all_cases_complete"] is True
    assert first["gates"]["point_forecasts_unchanged"] is True
    assert first["gates"]["pooled_never_policy_eligible"] is True
    assert first["arms"]["strict"]["mean_wis"] is not None


def test_calibration_action_resume_rejects_mismatched_identity(tmp_path):
    run_evaluation(2026083701, 1, tmp_path, False)
    identity_path = tmp_path / "run-identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["seed"] += 1
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(ValueError, match="resume identity differs"):
        run_evaluation(2026083701, 1, tmp_path, True)


def test_calibration_action_fresh_run_refuses_existing_checkpoint(tmp_path):
    run_evaluation(2026083701, 1, tmp_path, False)

    with pytest.raises(ValueError, match="already contains a run"):
        run_evaluation(2026083701, 1, tmp_path, False)
