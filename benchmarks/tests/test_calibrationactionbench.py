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
