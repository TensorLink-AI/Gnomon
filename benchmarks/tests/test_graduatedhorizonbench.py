import json

import pytest

from benchmarks.graduatedhorizonbench.run import run


def test_reference_is_deterministic_and_future_blind(tmp_path) -> None:
    first = run(920260903, 1, tmp_path / "first")
    second = run(920260903, 1, tmp_path / "second")

    assert first["all_gates_passed"] is True
    assert second["all_gates_passed"] is True
    assert first["rows"] == second["rows"]
    assert all(row["future_observations_used_by_forecaster"] == 0
               for row in first["rows"])


def test_reference_identity_mismatch_fails_closed(tmp_path) -> None:
    reference = run(920260903, 1, tmp_path / "reference")
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")

    with pytest.raises(ValueError, match="reference identity differs on seed"):
        run(920260904, 1, tmp_path / "treatment",
            reference_summary=path)

