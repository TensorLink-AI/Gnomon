import json

import pytest

from benchmarks.categoricalcalibrationbench.run import run


def test_categorical_calibration_matrix_is_complete_and_resumable(tmp_path):
    first = run(2026090203, tmp_path)
    resumed = run(2026090203, tmp_path, resume=True)

    assert first == resumed
    assert first["overall"]["cases"] == 180
    assert first["all_gates_passed"] is True


def test_categorical_calibration_rejects_resume_identity_drift(tmp_path):
    run(2026090203, tmp_path)
    identity_path = tmp_path / "run-identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["seed"] += 1
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(ValueError, match="resume identity differs"):
        run(2026090203, tmp_path, resume=True)
