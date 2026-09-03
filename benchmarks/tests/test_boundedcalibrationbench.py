import json

import pytest

from benchmarks.boundedcalibrationbench.run import run


def test_bounded_calibration_runs_real_product_path_and_resumes(tmp_path):
    first = run(2026090201, tmp_path)
    resumed = run(2026090201, tmp_path, resume=True)

    assert first == resumed
    assert first["cases"] == 8
    assert first["all_gates_passed"] is True
    assert first["candidate_wis"] <= first["reference_wis"]


def test_bounded_calibration_rejects_resume_identity_drift(tmp_path):
    run(2026090201, tmp_path)
    identity_path = tmp_path / "run-identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["seed"] += 1
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(ValueError, match="resume identity differs"):
        run(2026090201, tmp_path, resume=True)
