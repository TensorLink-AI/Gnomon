from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.recallbench.run_recallbench import (  # noqa: E402
    ARMS,
    HORIZON,
    arm_future,
    arm_values,
    generate_cases,
    mase,
    parse_forecast,
    run,
    seasonal_naive,
)


def _args(tmp_path: Path, cases: int = 6,
          resume: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        seed=20260827, cases=cases,
        output_dir=str(tmp_path / "out"), resume=resume, concurrency=2,
        model="scripted-test-model", reasoning_effort="none")


class ScriptedClient:
    """Offline stand-in: forecasts by repeating the last value, so its
    MASE is identical across arms — a model with zero memorization."""

    def completions(self, messages, *, n=1):
        text = messages[-1]["content"]
        values = json.loads(re.search(r"oldest first:\n(\[.*?\])",
                                      text, re.DOTALL).group(1))
        return [json.dumps([values[-1]] * HORIZON)]


class UsageClient(ScriptedClient):
    def __init__(self, usage):
        self.usage_summary = usage


def test_cases_pair_identical_windows_across_arms() -> None:
    first, provenance, futures = generate_cases(11, 10)
    again, _, _ = generate_cases(11, 10)
    assert first == again
    for case in first:
        raw = arm_values(case, "raw")
        anon = arm_values(case, "anon")
        assert len(raw) == len(anon)
        # The anon window is the same window under the case's affine map.
        for r, a in zip(raw, anon):
            assert abs(case.scale_a * r + case.shift_b - a) < 1e-3
        assert len(futures[case.case_id]) == HORIZON
    assert provenance["anonymization"].startswith(
        "per_case_seeded_positive_affine_transform")
    # Yearly and quarterly series cannot run at their true cadence and
    # must be excluded, not mislabelled daily.
    assert any("yearly" in name or "quarterly" in name
               for name in provenance["excluded_unsupported_cadence"])


def test_mase_is_affine_invariant() -> None:
    history = tuple(float(v) for v in range(1, 49))
    actual = [50.0, 51.0, 49.0, 52.0]
    forecast = [49.0, 50.0, 50.0, 50.0]
    base = mase(forecast, actual, history, 7)
    a, b = 2.3, 417.0
    scaled = mase([a * f + b for f in forecast],
                  [a * y + b for y in actual],
                  tuple(a * h + b for h in history), 7)
    assert abs(base - scaled) < 1e-9


def test_seasonal_naive_repeats_the_last_cycle() -> None:
    history = tuple(float(v) for v in [1, 2, 3, 1, 2, 3, 1, 2, 3])
    assert seasonal_naive(history, 3, 5) == [1.0, 2.0, 3.0, 1.0, 2.0]


def test_forecast_parsing_rejects_garbage_without_crashing() -> None:
    assert parse_forecast("no array here", 12) is None
    assert parse_forecast("[1, 2, 3]", 12) is None
    assert parse_forecast('[1, "two", 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]',
                          12) is None
    assert parse_forecast("[1, NaN, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]",
                          12) is None
    parsed = parse_forecast(
        "Here you go:\n[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]\nDone.", 12)
    assert parsed == [float(v) for v in range(1, 13)]


def test_a_matched_offline_run_separates_recall_from_skill(
        tmp_path) -> None:
    summary = run(_args(tmp_path), client=ScriptedClient())
    verdicts = summary["verdicts"]
    # A last-value forecaster has no memory: its raw and anon MASE agree
    # up to shown-value rounding, so the memorization delta is ~0.
    memorization = verdicts["memorization_delta"]
    assert memorization["pairs"] == 6
    assert abs(memorization["mean_delta"]) < 0.02
    assert set(verdicts) >= {"memorization_delta",
                             "skill_vs_gnomon_anonymized",
                             "model_vs_gnomon_raw_reference"}
    for arm in ARMS:
        entry = summary["metrics"][arm]
        assert entry["model_scored"] == 6
        assert entry["seasonal_naive_mean_mase"] is not None
    assert summary["provenance"]["naive_mase_max_cross_arm_drift"] < 0.01
    assert summary["design"]["held_out_future_absent_from_prompts_verified"]
    rows = [json.loads(line) for line in
            (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 6 * len(ARMS)
    for row in rows:
        assert row["dataset"] == summary["provenance"]["dataset_identity"]
        assert row["reasoning_effort"] == "none"


def test_futures_transform_with_the_arm() -> None:
    cases, _, futures = generate_cases(11, 4)
    case = cases[0]
    raw_future = arm_future(case, "raw", futures[case.case_id])
    anon_future = arm_future(case, "anon", futures[case.case_id])
    for r, a in zip(raw_future, anon_future):
        assert abs(case.scale_a * r + case.shift_b - a) < 1e-6


def test_resume_preserves_cumulative_usage_and_nonresume_replaces_rows(
        tmp_path) -> None:
    args = _args(tmp_path, cases=2)
    first_usage = {
        "model": args.model, "base_url": "test", "requests": 4,
        "transport_attempts": 4, "prompt_tokens": 100,
        "completion_tokens": 20, "truncation_escalations": 0,
        "cost_usd": .1,
    }
    run(args, client=UsageClient(first_usage))
    rows_path = tmp_path / "out" / "rows.jsonl"
    assert len(rows_path.read_text().splitlines()) == 4

    resumed = _args(tmp_path, cases=2, resume=True)
    summary = run(resumed, client=UsageClient({
        "model": args.model, "base_url": "test", "requests": 0,
        "transport_attempts": 0, "prompt_tokens": 0,
        "completion_tokens": 0, "truncation_escalations": 0,
        "cost_usd": 0,
    }))
    assert summary["usage"]["requests"] == 4
    assert summary["usage"]["prompt_tokens"] == 100
    assert summary["usage"]["accounting"] == \
        "cumulative_across_matching_resume_invocations"

    run(args, client=UsageClient(first_usage))
    assert len(rows_path.read_text().splitlines()) == 4
