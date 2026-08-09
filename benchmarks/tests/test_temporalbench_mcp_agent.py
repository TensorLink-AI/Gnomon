"""TemporalBench's gnomon-mcp arm: real MCP server in-process, scripted model.

Follows ``test_cik_mcp_agent.py``: the chat client is a script; the MCP
session is the real server code run in-process (CiK's
``InProcessMcpSession``), so the artifact-route tests exercise an actual
``gnomon_forecast`` through the real tool surface. Under test: the
per-channel exits and route taxonomy, the artifact-channel binding
(an artifact cannot be submitted for a channel it did not forecast),
the engine-abstention story through the tool surface (unsupported
artifact rejected with the best_effort recovery named, and a
model-driven ``best_effort: true`` retry accepted with its label), the
path jail, and caps abstaining the whole row.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.mcp_agent import InProcessMcpSession
from benchmarks.temporalbench.mcp_agent import MAX_ROUNDS, run_row


# -- fixtures ---------------------------------------------------------------

def _row(horizon: int = 4, sparse_temp: bool = True) -> dict:
    n = 96
    hr = [70.0 + 5.0 * math.sin(2 * math.pi * k / 24) for k in range(n)]
    spo2 = [97.0 + 0.2 * (k % 7) for k in range(n)]
    # MIMIC's temperature_c pattern: a handful of readings, most nulls.
    temp = [36.5 + 0.1 * (k % 3) if k < 5 else None for k in range(n)]
    channels = {"hr": hr, "spo2": spo2}
    if sparse_temp:
        channels["temperature_c"] = temp
    return {
        "id": "tb-mcp-test", "tier": "T2", "source_dataset": "MIMIC",
        "prompt": "Forecast the next steps of each channel and answer "
                  "the questions. Input (JSON): {...official prompt...}",
        "input": {"history": channels},
        "meta": {"main_key": "hr", "n_horizon": horizon},
        "ground_truth": {key: [0.0] * horizon for key in channels},
        "mcq": {"q1": {"options": ["Higher", "Lower", "Uncertain"],
                       "label": "Higher"}},
    }


class ScriptedClient:
    """A chat client that plays a fixed script instead of a model."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(self, messages, *, n=1, tools=None, tool_choice=None):
        assert self.steps, "model script exhausted before submission"
        step = self.steps.pop(0)
        action = step(messages) if callable(step) else step
        self.total_prompt_tokens += 100
        self.total_completion_tokens += 50 + int(action.get("bump_tokens", 0))
        calls = [
            SimpleNamespace(
                id=f"call{i}", type="function",
                function=SimpleNamespace(name=name,
                                         arguments=json.dumps(args)),
            )
            for i, (name, args) in enumerate(action.get("tool_calls", []))
        ]
        message = SimpleNamespace(content=action.get("content"),
                                  tool_calls=calls or None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")])


def _csv_path(messages) -> str:
    match = re.search(r"(/\S*?history\.csv)", messages[0]["content"])
    assert match, "system prompt does not name the history file"
    return match.group(1)


def _last_tool_payload(messages) -> dict:
    for message in reversed(messages):
        if message.get("role") == "tool":
            return json.loads(message["content"])
    raise AssertionError("no tool result in the conversation yet")


def _forecast_call(messages, channel: str, horizon: int = 4, **extra):
    csv = _csv_path(messages)
    return ("gnomon_forecast", {
        "input": csv, "time_column": "timestamp", "target_column": channel,
        "horizon": horizon, "frequency": "h",
        "output_dir": str(Path(csv).parent / f"out-{channel}"),
        **extra,
    })


VALUES = [97.0, 97.1, 97.2, 97.3]


def _run(row, steps, tmp_path, sessions=None):
    def factory(cwd):
        session = InProcessMcpSession(cwd)
        if sessions is not None:
            sessions.append(session)
        return session

    return run_row(row, ScriptedClient(steps),
                   session_factory=factory, work_dir=str(tmp_path))


# -- exits and routes -------------------------------------------------------

def test_artifact_and_values_exits_with_routes_and_labels(tmp_path):
    seen = {}

    def call_forecast(messages):
        return {"tool_calls": [_forecast_call(messages, "hr")]}

    def submit(messages):
        payload = _last_tool_payload(messages)
        assert payload.get("artifact_path"), payload
        seen["artifact_path"] = payload["artifact_path"]
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"artifact_path": payload["artifact_path"]},
                         "spo2": {"values": VALUES},
                         "temperature_c": {"abstain": True}},
            "mcq": {"q1": "Higher"},
        })]}

    outcome = _run(_row(), [call_forecast, submit], tmp_path)
    assert outcome["channel_route"] == {"hr": "gnomon",
                                        "spo2": "informed-direct",
                                        "temperature_c": "abstain"}
    assert outcome["channel_support"]["hr"] == "supported"
    assert outcome["channel_support"]["spo2"] == "model"
    assert outcome["abstained"] == ["temperature_c: abstained in submission"]
    assert outcome["answer"]["mcq"] == {"q1": "Higher"}
    assert outcome["answer"]["forecast"]["spo2"] == VALUES

    from gnomon.artifacts import read_artifact

    rows = read_artifact(seen["artifact_path"])["results"][0]["forecast"]
    assert outcome["answer"]["forecast"]["hr"] \
        == [float(row["q50"]) for row in rows]  # verbatim, not an edit


def test_values_only_with_no_tool_use_routes_direct(tmp_path):
    outcome = _run(_row(sparse_temp=False), [
        {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"values": VALUES},
                         "spo2": {"values": VALUES}},
            "mcq": {"q1": "Uncertain"},
        })]},
    ], tmp_path)
    assert outcome["channel_route"] == {"hr": "direct", "spo2": "direct"}
    assert outcome["channel_support"] == {"hr": "model", "spo2": "model"}


def test_omitted_channel_is_a_recorded_abstention(tmp_path):
    outcome = _run(_row(sparse_temp=False), [
        {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"values": VALUES}}, "mcq": {},
        })]},
    ], tmp_path)
    assert outcome["channel_route"]["spo2"] == "abstain"
    assert outcome["abstained"] == ["spo2: abstained in submission"]
    assert "spo2" not in outcome["answer"]["forecast"]


# -- the artifact-channel binding ------------------------------------------

def test_artifact_for_the_wrong_channel_is_rejected(tmp_path):
    def call_forecast(messages):
        return {"tool_calls": [_forecast_call(messages, "hr")]}

    def mislabel(messages):
        payload = _last_tool_payload(messages)
        return {"tool_calls": [("submit_answer", {
            "forecast": {"spo2": {"artifact_path": payload["artifact_path"]}},
        })]}

    def recover(messages):
        payload = _last_tool_payload(messages)
        assert payload["accepted"] is False
        assert any("forecasts 'hr', not 'spo2'" in p
                   for p in payload["problems"])
        return {"tool_calls": [("submit_answer", {
            "forecast": {"spo2": {"values": VALUES}}, "mcq": {},
        })]}

    outcome = _run(_row(sparse_temp=False),
                   [call_forecast, mislabel, recover], tmp_path)
    assert outcome["channel_route"]["spo2"] == "informed-direct"


def test_unknown_artifact_is_rejected(tmp_path):
    def bad(messages):
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"artifact_path": "/no/such/artifact"}},
        })]}

    def recover(messages):
        payload = _last_tool_payload(messages)
        assert payload["accepted"] is False
        assert any("not produced" in p for p in payload["problems"])
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"values": VALUES}}, "mcq": {},
        })]}

    outcome = _run(_row(sparse_temp=False), [bad, recover], tmp_path)
    assert outcome["channel_route"]["hr"] == "direct"


# -- the engine abstention story through the tool surface -------------------

def test_unsupported_artifact_rejected_then_best_effort_retry_labeled(tmp_path):
    """The realistic sparse-channel path: the engine abstains, the
    harness names the honest options, the MODEL chooses the labeled
    fallback by retrying with best_effort=true — and the label sticks."""
    def forecast_sparse(messages):
        return {"tool_calls": [_forecast_call(messages, "temperature_c")]}

    def submit_unsupported(messages):
        payload = _last_tool_payload(messages)
        assert payload.get("artifact_path")
        return {"tool_calls": [("submit_answer", {
            "forecast": {"temperature_c":
                         {"artifact_path": payload["artifact_path"]}},
        })]}

    def retry_best_effort(messages):
        payload = _last_tool_payload(messages)
        assert payload["accepted"] is False
        problem, = payload["problems"]
        assert "abstained" in problem and "best_effort" in problem
        return {"tool_calls": [_forecast_call(messages, "temperature_c",
                                              best_effort=True)]}

    def submit_fallback(messages):
        payload = _last_tool_payload(messages)
        return {"tool_calls": [("submit_answer", {
            "forecast": {"temperature_c":
                         {"artifact_path": payload["artifact_path"]},
                         "hr": {"abstain": True},
                         "spo2": {"abstain": True}},
            "mcq": {"q1": "Uncertain"},
        })]}

    outcome = _run(_row(), [forecast_sparse, submit_unsupported,
                            retry_best_effort, submit_fallback], tmp_path)
    assert outcome["channel_route"]["temperature_c"] == "gnomon"
    # The disclosed-fallback label survives into the outcome: a
    # best_effort row can never pass as a supported forecast.
    assert outcome["channel_support"]["temperature_c"] == "best_effort"
    assert len(outcome["answer"]["forecast"]["temperature_c"]) == 4


# -- the path jail ----------------------------------------------------------

def test_jail_blocks_outside_paths_before_the_server(tmp_path):
    outside = tmp_path / "cached-benchmark-data.csv"
    outside.write_text("timestamp,value\n", encoding="utf-8")
    sessions = []

    def escape(messages):
        return {"tool_calls": [("gnomon_forecast", {
            "input": str(outside), "time_column": "timestamp",
            "target_column": "hr", "horizon": 4,
        })]}

    def answer(messages):
        payload = _last_tool_payload(messages)
        assert payload["code"] == "PATH_JAIL"
        assert payload["authored_by"] == "harness"
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"values": VALUES},
                         "spo2": {"values": VALUES}}, "mcq": {},
        })]}

    _run(_row(sparse_temp=False), [escape, answer], tmp_path,
         sessions=sessions)
    assert sessions[0].calls == []  # the call never reached the server


# -- caps abstain the whole row, never fall back ----------------------------

def test_rounds_cap_abstains_every_channel(tmp_path):
    steps = [{"content": "Working on it."}, {"content": "Still thinking."}]
    outcome = _run(_row(sparse_temp=False), steps, tmp_path)
    assert outcome["answer"]["forecast"] == {}
    assert set(outcome["channel_route"].values()) == {"abstain"}
    assert all("no submission" in reason for reason in outcome["abstained"])


def test_token_cap_abstains(tmp_path):
    steps = [{"content": "hmm", "bump_tokens": 300_000},
             {"content": "more"}]
    outcome = _run(_row(sparse_temp=False), steps, tmp_path)
    assert any("cap:tokens" in reason for reason in outcome["abstained"])


def test_wrong_length_values_rejection_names_the_length(tmp_path):
    def recover(messages):
        payload = _last_tool_payload(messages)
        assert payload["accepted"] is False
        assert any("got 2" in p for p in payload["problems"])
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"values": VALUES},
                         "spo2": {"values": VALUES}}, "mcq": {},
        })]}

    outcome = _run(_row(sparse_temp=False), [
        {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"values": VALUES[:2]}}})]},
        recover,
    ], tmp_path)
    assert outcome["answer"]["forecast"]["hr"] == VALUES


# -- run_temporalbench wiring ----------------------------------------------

def test_answer_row_dispatches_and_restricts_tiers():
    import pytest

    from benchmarks.temporalbench.run_temporalbench import answer_row

    with pytest.raises(ValueError, match="T2 and T4"):
        answer_row({"tier": "T1", "prompt": "x"}, "gnomon-mcp", None)


def test_rounds_constant_matches_cik_posture():
    assert MAX_ROUNDS == 10
