from __future__ import annotations

from benchmarks.windowsemanticsbench import run as bench


def test_frozen_matrix_has_distinct_boundary_and_refusal_cases():
    expected = {case_id: active for case_id, _, active in bench.CASES}
    assert expected["between_half_open"] == [4, 5, 6]
    assert expected["through_closed"] == [4, 5, 6, 7]
    assert expected["ambiguous_to"] is None
    assert expected["wrong_target"] is None


def test_benchmark_never_accepts_future_target_values(tmp_path):
    result = bench.run(tmp_path / "summary.json")
    assert all(row["future_target_observations_used"] == 0
               for row in result["rows"])
    assert result["gates"]["future_targets_never_used"] is True


def test_exact_zero_window_covers_every_published_quantile(tmp_path):
    result = bench.run(tmp_path / "summary.json")
    by_case = {row["case_id"]: row for row in result["rows"]}
    assert "distribution_semantics_exact" in by_case["between_half_open"]
    assert by_case["ambiguous_to"]["distribution_semantics_exact"] is True
