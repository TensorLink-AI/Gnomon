from benchmarks.trendanswerbench.run import (
    FAMILIES, SEEDS, _cases, _generate, _phase_fixed_slope, _run_case,
    _summarise,
)


def test_frozen_trendanswerbench_matrix_is_complete_and_deterministic() -> None:
    cases = _cases()
    assert len(cases) == len(FAMILIES) * len(SEEDS) == 152
    assert len({case["case_id"] for case in cases}) == 152
    for case in (cases[0], cases[48], cases[-1]):
        first = _generate(case)
        second = _generate(case)
        assert first == second
        assert len(first[0]) == case["history_length"]
        assert len(first[1]) == case["horizon"]


def test_phase_fixed_slope_removes_a_repeating_seasonal_shape() -> None:
    period = 4
    values = [10 + .5 * index + (3, -2, 4, -5)[index % period]
              for index in range(3 * period)]
    assert _phase_fixed_slope(values, period) == .5
    assert _phase_fixed_slope(values[:period], period) is None


def test_product_row_keeps_typed_authority_and_inputs_immutable() -> None:
    row = _run_case(_cases()[0])
    assert row["product_complete"] is True
    assert row["inputs_unchanged"] is True
    assert row["primary_forecast_unchanged"] is True
    assert row["authority_fields_agree"] is True
    assert row["future_observations_used_by_engine"] == 0


def test_summary_keeps_failed_rows_and_unsafe_authority_visible() -> None:
    rows = [_run_case(case) for case in _cases()[:2]]
    summary = _summarise(rows)
    assert summary["overall"]["cases"] == 2
    assert summary["gates"]["all_152_product_cases_complete"] is False
    assert "supported_direction_accuracy_at_least_90pct" in summary["gates"]
