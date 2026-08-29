import json
import sys
from types import SimpleNamespace

import pytest

from benchmarks.cik.run_cik import (
    _counterfactual_candidate_scores, _load_checkpoint,
    _summarize_selection_diagnostics, _task_information_profile, build_parser,
)


def test_information_profile_flags_only_identical_constant_past_and_future():
    class Series:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    class Frame:
        columns = ["x"]

        def __init__(self, values):
            self._values = values

        def __getitem__(self, column):
            assert column == "x"
            return Series(self._values)

    degenerate = SimpleNamespace(
        past_time=Frame([0.0, 0.0]), future_time=Frame([0.0]))
    shifted = SimpleNamespace(
        past_time=Frame([0.0, 0.0]), future_time=Frame([1.0]))

    profile = _task_information_profile(degenerate)
    assert profile["degenerate_same_constant_case"] is True
    assert profile["passed_to_forecaster"] is False
    assert _task_information_profile(shifted)[
        "degenerate_same_constant_case"] is False


def test_candidate_scores_are_post_forecast_diagnostics_only(monkeypatch):
    class Array:
        def __init__(self, values):
            self.values = values
            self.shape = (len(values), len(values[0]), 1)

        def __getitem__(self, key):
            assert isinstance(key, tuple)
            return self

    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(
        asarray=lambda values, dtype: Array(values)))

    class Task:
        def evaluate(self, samples):
            assert samples.shape == (5, 2, 1)
            return {"metric": float(samples.values[0][0])}

    def rows(value):
        return [
            {"timestamp": "t1", "q10": value - 1, "q50": value,
             "q90": value + 1},
            {"timestamp": "t2", "q10": value, "q50": value + 1,
             "q90": value + 2},
        ]
    scores = _counterfactual_candidate_scores(Task(), {"publication": {
        "recommended_scenario_id": "primary",
        "candidate_portfolio": [
            {"scenario_id": "primary", "role": "immutable_primary",
             "forecast": rows(10), "human_selection_eligible": True},
            {"scenario_id": "prior-1", "role": "model_authored",
             "forecast": rows(20), "human_selection_eligible": False},
        ],
    }}, 5)

    assert [item["selected"] for item in scores] == [True, False]
    assert scores[1]["score"] > scores[0]["score"]
    assert all(item["computed_after_forecast"] is True for item in scores)
    assert all(item["passed_to_forecaster"] is False for item in scores)
    assert scores[0]["human_selection_eligible"] is True
    assert scores[1]["human_selection_eligible"] is False


def test_selection_summary_separates_uplift_from_hindsight_regret():
    summary = _summarize_selection_diagnostics([
        {"selected_score": .4, "primary_score": .7,
         "best_candidate_score": .2, "best_eligible_candidate_score": .4,
         "selected_primary": False,
         "selected_hindsight_best": False,
         "selected_best_eligible": True,
         "primary_forecast_unchanged": True,
         "automation_eligible": False},
        {"selected_score": .3, "primary_score": .3,
         "best_candidate_score": .3, "best_eligible_candidate_score": .3,
         "selected_primary": True,
         "selected_hindsight_best": True,
         "selected_best_eligible": True,
         "primary_forecast_unchanged": True,
         "automation_eligible": False},
    ])

    assert summary["mean_uplift_vs_primary_rcrps"] == pytest.approx(.15)
    assert summary["mean_oracle_headroom_rcrps"] == pytest.approx(.1)
    assert summary["mean_selector_regret_among_eligible_rcrps"] == 0
    assert summary["selected_hindsight_best_cases"] == 1
    assert summary["selected_best_eligible_cases"] == 2
    assert summary["primary_immutability_failures"] == 0
    assert summary["automation_eligible_cases"] == 0
    assert summary["passed_to_forecaster"] is False


def test_resume_retries_provider_and_process_failures_but_keeps_model_results(tmp_path):
    payload = {
        "valid": {"name": "Task", "row": {"seed": 1, "score": 0.2}},
        "provider": {"name": "Task", "row": {
            "seed": 2, "error": "OpenRouter returned HTTP 403: daily limit"}},
        "timeout": {"name": "Task", "row": {
            "seed": 3, "error": "case_timeout_after_900s"}},
        "model": {"name": "Task", "row": {
            "seed": 4, "error": "could not parse any valid forecast"}},
    }
    (tmp_path / "case-checkpoint.json").write_text(json.dumps(payload))
    loaded = _load_checkpoint(tmp_path)
    assert set(loaded) == {"valid", "model"}


def test_held_out_seed_range_is_explicit_in_cli():
    args = build_parser().parse_args([
        "--method", "gnomon-pure", "--seed-start", "6", "--seeds", "2",
        "--output-dir", "/tmp/out",
    ])
    assert list(range(args.seed_start, args.seed_start + args.seeds)) == [6, 7]


def test_direct_control_has_explicit_cache_identified_reasoning_mode():
    pytest.importorskip("cik_benchmark")
    from benchmarks.cik.openrouter_direct_prompt import OpenRouterDirectPrompt

    control = OpenRouterDirectPrompt(
        "provider/model", api_key="test", base_url="https://example.test/v1")

    assert control.reasoning_effort == "none"
    assert control._client.reasoning_effort == "none"
    assert "reasoning=none" in control.cache_name
