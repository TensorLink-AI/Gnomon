import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.cik.run_cik import (
    _candidate_interval_diagnostics, _checkpoint_identity,
    _counterfactual_candidate_scores,
    _load_attempt_checkpoint, _load_checkpoint, _prepare_checkpoint_identity,
    _sample_cache_dir, _summarize_selection_diagnostics,
    _task_information_profile, build_parser,
    _write_attempt_checkpoint, select_tasks, write_outputs,
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
        future_time = None

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
    class Series:
        def tolist(self):
            return [10.0, 11.0]

    class Future:
        columns = ["target"]

        def __getitem__(self, name):
            assert name == "target"
            return Series()

    task = Task()
    task.future_time = Future()
    scores = _counterfactual_candidate_scores(task, {"publication": {
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
    assert scores[0]["nominal_coverage"] == .8
    assert scores[0]["empirical_coverage"] == 1
    assert abs(scores[0]["wis"] - 1 / 3) < 1e-12
    assert scores[1]["wis"] > scores[0]["wis"]


def test_candidate_interval_diagnostics_reject_crossed_quantiles():
    class Future:
        columns = ["target"]

        def __getitem__(self, _name):
            return [10.0]

    with pytest.raises(ValueError, match="crossed"):
        _candidate_interval_diagnostics(
            SimpleNamespace(future_time=Future()),
            [{"q10": 11, "q50": 10, "q90": 12}])


def test_selection_summary_separates_uplift_from_hindsight_regret():
    summary = _summarize_selection_diagnostics([
        {"selected_score": .4, "primary_score": .7,
         "selected_role": "model_authored",
         "best_candidate_score": .2, "best_eligible_candidate_score": .4,
         "selected_primary": False,
         "selected_hindsight_best": False,
         "selected_best_eligible": True,
         "primary_forecast_unchanged": True,
         "automation_eligible": False},
        {"selected_score": .3, "primary_score": .3,
         "selected_role": "immutable_primary",
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
    assert summary["selected_role_counts"] == {
        "immutable_primary": 1, "model_authored": 1}
    assert summary["primary_immutability_failures"] == 0
    assert summary["automation_eligible_cases"] == 0
    assert summary["passed_to_forecaster"] is False


def test_write_outputs_retains_versioned_raw_selection_diagnostics(
        tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "Task" / "seed-7"
    run_dir.mkdir(parents=True)
    (run_dir / "extra_info.json").write_text("{}", encoding="utf-8")
    extra = {
        "benchmark_counterfactual_candidate_scores": [
            {"scenario_id": "primary", "role": "immutable_primary",
             "score": .4, "selected": True,
             "human_selection_eligible": True},
            {"scenario_id": "prior", "role": "model_authored",
             "score": .3, "selected": False,
             "human_selection_eligible": True},
        ],
        "primary_forecast_unchanged": True,
        "automation_eligible": False,
        "total_time": 2.0,
    }
    monkeypatch.setattr(
        "benchmarks.cik.run_cik.load_run_extra_info",
        lambda *_args: extra)
    args = SimpleNamespace(
        method="gnomon-mcp", model="provider/model", seeds=1,
        seed_start=7)
    method = SimpleNamespace(cache_name="method-contract-238")

    write_outputs(
        {"Task": [{"seed": 7, "score": .4,
                   "cumulative_active_seconds": 9.5}]},
        method, args, tmp_path)

    rows = [json.loads(line) for line in (
        tmp_path / "selection-diagnostics.jsonl").read_text(
            encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["benchmark_method"] == "method-contract-238"
    assert rows[0]["diagnostic_schema_version"] == "1"
    assert rows[0]["selected_role"] == "immutable_primary"
    assert rows[0]["computed_after_forecast"] is True
    assert rows[0]["passed_to_forecaster"] is False
    assert rows[0]["candidates"][1]["scenario_id"] == "prior"
    [run_row] = [json.loads(line) for line in (
        tmp_path / "gnomonbench.jsonl").read_text(
            encoding="utf-8").splitlines()]
    assert run_row["latency_seconds"] == 9.5


def test_attempt_checkpoint_retains_cumulative_active_work(tmp_path):
    attempts = {
        "Task::seed=7": [
            {"active_seconds": 3.5, "completed": False,
             "error": "case_timeout_after_3s", "peak_rss_mb": 700},
            {"active_seconds": 1.25, "completed": True,
             "error": None, "peak_rss_mb": 710},
        ],
    }

    _write_attempt_checkpoint(tmp_path, attempts)

    assert _load_attempt_checkpoint(tmp_path) == attempts


def test_resume_retries_provider_and_process_failures_but_keeps_model_results(tmp_path):
    payload = {
        "valid": {"name": "Task", "row": {"seed": 1, "score": 0.2}},
        "provider": {"name": "Task", "row": {
            "seed": 2, "error": "OpenRouter returned HTTP 403: daily limit"}},
        "provider_error_spelling": {"name": "Task", "row": {
            "seed": 5, "error": (
                "OpenRouter request failed after 6 attempts: "
                "HTTP Error 429: Too Many Requests")}},
        "timeout": {"name": "Task", "row": {
            "seed": 3, "error": "case_timeout_after_900s"}},
        "model": {"name": "Task", "row": {
            "seed": 4, "error": "could not parse any valid forecast"}},
    }
    (tmp_path / "case-checkpoint.json").write_text(json.dumps(payload))
    loaded = _load_checkpoint(tmp_path)
    assert set(loaded) == {"valid", "model"}


def test_cik_checkpoint_identity_covers_request_and_corpus_scope(tmp_path):
    class AlphaTask:
        pass

    args = build_parser().parse_args([
        "--method", "gnomon-pure", "--seed-start", "6", "--seeds", "2",
        "--output-dir", str(tmp_path),
    ])
    identity = _checkpoint_identity(args, [AlphaTask], 20, "abc")
    _prepare_checkpoint_identity(tmp_path, identity, fresh=False)
    changed = {**identity, "base_url": "https://different.test/v1"}
    with pytest.raises(SystemExit, match="resume identity mismatch"):
        _prepare_checkpoint_identity(tmp_path, changed, fresh=False)

    changed = {**identity, "sample_parallelism": 2}
    with pytest.raises(SystemExit, match="resume identity mismatch"):
        _prepare_checkpoint_identity(tmp_path, changed, fresh=False)

    changed = {**identity, "mcp_allow_prior_compromise": True}
    with pytest.raises(SystemExit, match="resume identity mismatch"):
        _prepare_checkpoint_identity(tmp_path, changed, fresh=False)


def test_prior_compromise_consent_is_explicit_and_defaults_off():
    parser = build_parser()
    default = parser.parse_args([
        "--method", "gnomon-mcp", "--model", "test/model",
        "--output-dir", "/tmp/cik-default",
    ])
    consented = parser.parse_args([
        "--method", "gnomon-mcp", "--model", "test/model",
        "--mcp-profile", "evidence",
        "--mcp-output-role", "publication_best_effort",
        "--mcp-allow-prior-compromise",
        "--output-dir", "/tmp/cik-consented",
    ])

    assert default.mcp_allow_prior_compromise is False
    assert consented.mcp_allow_prior_compromise is True
    identity = _checkpoint_identity(consented, [], 20, "abc")
    assert identity["mcp_allow_prior_compromise"] is True


def test_shared_sample_cache_is_condition_scoped_and_recorded(tmp_path):
    shared = tmp_path / "shared"
    retained = shared / "control" / "Task-seed7" / "choice.json"
    retained.parent.mkdir(parents=True)
    retained.write_text("{}")
    args = build_parser().parse_args([
        "--method", "control", "--model", "test/model",
        "--sample-cache-root", str(shared),
        "--output-dir", str(tmp_path / "run"),
    ])
    args._sample_cache_case = "Task-seed7"

    assert _sample_cache_dir(args) == (
        shared.resolve() / "control" / "Task-seed7")
    identity = _checkpoint_identity(args, [], 20, "abc")
    assert identity["sample_cache_root"] == str(shared.resolve())
    Path(args.output_dir).mkdir()
    _prepare_checkpoint_identity(
        Path(args.output_dir), identity, fresh=True)
    assert retained.read_text() == "{}"

    treatment = build_parser().parse_args([
        "--method", "gnomon-mcp", "--model", "test/model",
        "--sample-cache-root", str(shared),
        "--output-dir", str(tmp_path / "other"),
    ])
    treatment._sample_cache_case = "Task-seed7"
    assert _sample_cache_dir(treatment) == (
        shared.resolve() / "gnomon-mcp" / "Task-seed7")


def test_cik_checkpoint_refuses_legacy_state_without_identity(tmp_path):
    (tmp_path / "case-checkpoint.json").write_text("{}")
    with pytest.raises(SystemExit, match="without run_identity"):
        _prepare_checkpoint_identity(
            tmp_path, {"schema_version": 1}, fresh=False)


def test_fresh_cik_run_clears_only_its_sample_cache(tmp_path):
    cache_file = tmp_path / "sample-cache" / "key" / "choice-old.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("{}")
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep")

    _prepare_checkpoint_identity(
        tmp_path, {"schema_version": 1}, fresh=True)

    assert not (tmp_path / "sample-cache").exists()
    assert unrelated.read_text() == "keep"


def test_held_out_seed_range_is_explicit_in_cli():
    args = build_parser().parse_args([
        "--method", "gnomon-pure", "--seed-start", "6", "--seeds", "2",
        "--output-dir", "/tmp/out",
    ])
    assert list(range(args.seed_start, args.seed_start + args.seeds)) == [6, 7]


def test_exact_task_shard_preserves_requested_order_and_requires_known_names():
    class AlphaTask:
        pass

    class BetaTask:
        pass

    selected = select_tasks(
        [AlphaTask, BetaTask], task_names=["BetaTask", "AlphaTask"])
    assert selected == [BetaTask, AlphaTask]

    with pytest.raises(SystemExit, match="Unknown CiK task name"):
        select_tasks([AlphaTask], task_names=["MissingTask"])
    with pytest.raises(SystemExit, match="may not repeat"):
        select_tasks([AlphaTask], task_names=["AlphaTask", "AlphaTask"])
    with pytest.raises(SystemExit, match="mutually exclusive"):
        select_tasks([AlphaTask], task_names=["AlphaTask"], task_filter="A")


def test_exact_task_name_cli_is_repeatable():
    args = build_parser().parse_args([
        "--method", "gnomon-pure", "--task-name", "AlphaTask",
        "--task-name", "BetaTask", "--output-dir", "/tmp/out",
    ])
    assert args.task_name == ["AlphaTask", "BetaTask"]


def test_direct_control_has_explicit_cache_identified_reasoning_mode():
    pytest.importorskip("cik_benchmark")
    from benchmarks.cik.openrouter_direct_prompt import OpenRouterDirectPrompt

    control = OpenRouterDirectPrompt(
        "provider/model", api_key="test", base_url="https://example.test/v1")

    assert control.reasoning_effort == "none"
    assert control._client.reasoning_effort == "none"
    assert "reasoning=none" in control.cache_name
    assert "sample_parallelism=4" in control.cache_name


def test_direct_control_retains_provider_usage(monkeypatch):
    pytest.importorskip("cik_benchmark")
    from cik_benchmark.baselines.direct_prompt import DirectPrompt

    from benchmarks.cik.openrouter_direct_prompt import OpenRouterDirectPrompt

    monkeypatch.setattr(
        DirectPrompt, "__call__",
        lambda self, task, samples: ("paths", {"total_time": 2.0}))
    control = OpenRouterDirectPrompt.__new__(OpenRouterDirectPrompt)
    control._client = SimpleNamespace(usage_summary={
        "requests": 2, "prompt_tokens": 30, "completion_tokens": 40})

    samples, extra = control(object(), 5)

    assert samples == "paths"
    assert extra["llm_usage"] == {
        "requests": 2, "prompt_tokens": 30, "completion_tokens": 40}
