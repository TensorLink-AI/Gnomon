from __future__ import annotations

from benchmarks.compilerbench.generate import cases
from benchmarks.compilerbench.run_compilerbench import _sum_usage


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


def test_compiler_usage_includes_retried_attempts() -> None:
    usage = _sum_usage([
        {"model": "m", "base_url": "https://provider.test/v1",
         "requests": 0, "transport_attempts": 3, "prompt_tokens": 0,
         "completion_tokens": 0, "cost_usd": 0,
         "truncation_escalations": 0},
        {"model": "m", "base_url": "https://provider.test/v1",
         "requests": 1, "transport_attempts": 1, "prompt_tokens": 50,
         "completion_tokens": 10, "cost_usd": .01,
         "truncation_escalations": 0},
    ])
    assert usage["requests"] == 1
    assert usage["transport_attempts"] == 4
    assert usage["prompt_tokens"] == 50
