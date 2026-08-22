"""MTBench adapter: patched send-function shapes, tool schema parity,
and the official metric block's edge cases."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.gnomon_forecaster import events_from_proposals
from benchmarks.mtbench.gnomon_forecaster import official_mape, sample_metrics
from benchmarks.mtbench.openrouter_patch import build_send_functions
from benchmarks.mtbench.run_mtbench import _script_argument
from benchmarks.mtbench.tool_agent import TOOL_SPECS


class _StubClient:
    def completions(self, messages, *, n=1):
        return [f"echo:{messages[-1]['content']}"] * n


def test_patched_openai_branches_return_message_objects():
    # Upstream send_to_openai_chatgpt / send_to_openai_o1 return
    # response.choices[0].message; their scripts read `.content` off it.
    functions = build_send_functions(_StubClient())
    for name in ("send_to_openai_chatgpt", "send_to_openai_o1"):
        result = functions[name]("hello", model="x", max_tokens=100,
                                 temperature=0.5, timeout=30)
        assert result.content == "echo:hello", name


def test_patched_string_branches_return_plain_strings():
    functions = build_send_functions(_StubClient())
    for name in ("send_to_anthropic_claude", "send_to_google_gemini",
                 "send_to_deepseek", "send_to_deepseek_r1"):
        result = functions[name]("hello")
        assert result == "echo:hello", name


def test_tool_spec_event_schema_matches_validator():
    # A proposal built from exactly the advertised property names must
    # survive events_from_proposals — a key mismatch here silently
    # rejects every event and degenerates tools mode to history-only.
    forecast_spec = next(
        spec["function"] for spec in TOOL_SPECS
        if spec["function"]["name"] == "gnomon_forecast"
    )
    items = forecast_spec["parameters"]["properties"]["context_events"]["items"]
    proposal = {
        "event_type": "earnings_report",
        "effective_start": "2020-01-10T00:00:00+00:00",
        "effective_end": "2020-01-12T00:00:00+00:00",
        "confidence": 0.8,
        "rationale": "stated in the article",
    }
    assert set(proposal) == set(items["properties"])
    assert set(items["required"]) <= set(proposal)
    events, notes = events_from_proposals(
        [proposal], task_name="spec-parity",
        known_at="2020-01-01T00:00:00+00:00",
        window_start="2020-01-01T00:00:00+00:00",
        window_end="2020-02-01T00:00:00+00:00",
    )
    assert len(events) == 1, notes
    assert events[0].event_type == "earnings_report"


def test_mape_fallback_is_nan_on_all_zero_truth():
    # Upstream calculate_mape masks zeros and means the rest; an
    # all-zero truth leaves an empty mask, whose mean is NaN, not 0.
    assert math.isnan(official_mape([0.0, 0.0], [1.0, 2.0]))


def test_sample_metrics_rejects_length_mismatch():
    try:
        sample_metrics([1.0, 2.0, 3.0], [1.0, 2.0])
    except ValueError as error:
        assert "horizon" in str(error)
    else:
        raise AssertionError("short prediction must not be silently truncated")


def test_sample_metrics_matches_official_math():
    metrics = sample_metrics([2.0, 4.0], [1.0, 6.0])
    assert metrics["mse"] == ((1.0) ** 2 + (2.0) ** 2) / 2
    assert metrics["mae"] == (1.0 + 2.0) / 2
    assert metrics["rmse"] == metrics["mse"] ** 0.5
    assert metrics["mape"] == 100.0 * (0.5 + 0.5) / 2


def test_official_script_argument_supports_both_argparse_forms():
    args = ["--dataset_folder=/tmp/data", "--save_path", "/tmp/out"]
    assert _script_argument(args, "dataset_folder") == "/tmp/data"
    assert _script_argument(args, "save_path") == "/tmp/out"
    assert _script_argument(args, "indicator") is None
