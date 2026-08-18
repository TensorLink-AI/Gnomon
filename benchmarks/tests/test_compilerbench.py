from __future__ import annotations

from benchmarks.compilerbench.generate import cases


def test_cases_are_deterministic_independent_and_cover_refusal_scopes() -> None:
    first, second = cases(), cases()
    assert first == second
    assert len(first) == len({row["id"] for row in first}) == 80
    assert {row["expected"].get("scope") for row in first} >= {
        "series", "each", "aggregate"}
    assert any(row["expected"].get("refusal") for row in first)
    assert any(row["case_kind"] == "time_window_statistic" for row in first)
    assert any(row["case_kind"] == "target_inheritance" for row in first)
    assert all("label" not in row and "options" not in row for row in first)
