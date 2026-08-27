import pytest

from gnomon.agent_context import (
    build_sampled_context_prior_prompt,
    candidate_from_sampled_paths,
    recommended_sample_count,
    sample_path_stability,
)


def test_provider_neutral_prior_prompt_keeps_host_owned_regular_grid_compact():
    history = [f"2026-01-01T0{hour}:00:00+00:00" for hour in range(3)]
    future = [f"2026-01-01T0{hour}:00:00+00:00" for hour in range(3, 6)]

    prompt = build_sampled_context_prior_prompt(
        timestamps=history, values=[1, 2, 3], future_timestamps=future,
        context="A planned event may increase demand.")

    assert prompt.count("step_seconds=3600") == 2
    assert "[1,2,3]" in prompt
    assert '"forecast_path"' in prompt
    assert "Do not echo timestamps" in prompt


def test_provider_neutral_prior_parser_retains_valid_paths_independently():
    future = ["2026-01-02T00:00:00+00:00",
              "2026-01-03T00:00:00+00:00"]
    candidate, diagnostics = candidate_from_sampled_paths([
        'prose {"forecast_path":{"values":[2,4],"rationale":"a"}}',
        '{"forecast_path":{"values":[4,8]}}',
        '{"forecast_path":{"values":[5]}}',
    ], future, history_values=[1, 2, 3])

    assert diagnostics["requested"] == 3
    assert diagnostics["accepted"] == 2
    assert candidate is not None
    assert candidate["_validated_sample_paths"] == [[2.0, 4.0], [4.0, 8.0]]
    assert [row["q50"] for row in candidate["quantiles"]] == [3.0, 6.0]
    assert diagnostics["stability"]["interpretation"] == \
        "stability_not_historical_skill"


def test_provider_neutral_prior_rejects_nonfinite_and_wrong_grid_paths():
    candidate, diagnostics = candidate_from_sampled_paths([
        '{"forecast_path":{"values":[1]}}',
        '{"forecast_path":{"values":[1,"NaN"]}}',
    ], ["2026-01-02T00:00:00+00:00",
        "2026-01-03T00:00:00+00:00"])

    assert candidate is None
    assert diagnostics["accepted"] == 0
    assert diagnostics["rejected"] == 2


def test_sample_count_policy_is_bounded_and_requires_a_distribution():
    assert recommended_sample_count(4) == 5
    assert recommended_sample_count(95) == 5
    assert recommended_sample_count(96) == 3
    with pytest.raises(ValueError, match="positive"):
        recommended_sample_count(0)


def test_stability_rejects_empty_paths():
    with pytest.raises(ValueError, match="non-empty"):
        sample_path_stability([], [1, 2])
