import json
from types import SimpleNamespace

from benchmarks.cik.run_cik import (
    _load_checkpoint, _task_information_profile, build_parser,
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
