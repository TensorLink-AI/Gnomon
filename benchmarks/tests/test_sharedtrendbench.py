from benchmarks.sharedtrendbench.run import run


def test_shared_trend_is_rejected_while_real_driver_is_admitted(tmp_path):
    result = run(2026090201, tmp_path / "summary.json")

    assert result["all_gates_passed"] is True
    assert result["useful_driver_recall"] >= .85
    assert result["shared_trend_false_admission_rate"] <= .10
    assert result["context_is_useful"] is False
    assert result["context_admitted"] is False
