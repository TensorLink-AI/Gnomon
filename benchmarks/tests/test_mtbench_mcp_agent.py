"""MTBench's mcp mode: real MCP server in-process, scripted model.

Follows ``test_cik_mcp_agent.py``: the chat client is a script; the MCP
session is the real server code run in-process (CiK's
``InProcessMcpSession``), so the artifact-route test exercises an actual
``gnomon_forecast`` through the real tool surface. Under test: the
three exits and the route taxonomy, the verbatim guarantee on a
submitted artifact, the engine-abstention story through the tool
surface (unsupported artifact rejected with the best_effort recovery
named, a model-driven ``best_effort: true`` retry accepted with its
label, ``engine_abstentions`` counted), the path jail, caps abstaining
the sample, and repairable rejections that name what was received.
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
from benchmarks.mtbench.mcp_agent import run_sample_mcp


# -- fixtures ---------------------------------------------------------------

def _sample(n: int = 72, horizon: int = 4) -> dict:
    values = [100.0 + 0.4 * i + 3.0 * math.sin(2 * math.pi * i / 7)
              for i in range(n)]
    truth = [100.0 + 0.4 * (n + k) for k in range(horizon)]
    return {
        "filename": "finance#0001",
        "input_window": values,
        "output_window": truth,
        "text": "Quarterly earnings beat expectations on strong demand.",
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


def _forecast_call(messages, horizon: int = 4, **extra):
    csv = _csv_path(messages)
    return ("gnomon_forecast", {
        "input": csv, "time_column": "timestamp", "target_column": "value",
        "horizon": horizon, "frequency": "D",
        "output_dir": str(Path(csv).parent / "gnomon-output"),
        **extra,
    })


VALUES = [101.0, 101.5, 102.0, 102.5]


def _run(sample, steps, tmp_path, sessions=None):
    def factory(cwd):
        session = InProcessMcpSession(cwd)
        if sessions is not None:
            sessions.append(session)
        return session

    return run_sample_mcp(sample, ScriptedClient(steps),
                          session_factory=factory, work_dir=str(tmp_path))


# -- exits and routes -------------------------------------------------------

def test_artifact_exit_uses_the_trajectory_verbatim(tmp_path):
    seen = {}

    def call_forecast(messages):
        return {"tool_calls": [_forecast_call(messages)]}

    def submit(messages):
        payload = _last_tool_payload(messages)
        assert payload.get("artifact_path"), payload
        seen["artifact_path"] = payload["artifact_path"]
        return {"tool_calls": [("submit_forecast",
                                {"artifact_path": payload["artifact_path"]})]}

    outcome = _run(_sample(), [call_forecast, submit], tmp_path)
    assert outcome["abstained"] is False
    assert outcome["route"] == "gnomon"
    assert outcome["support"] is not None
    assert outcome["engine_abstentions"] == 0
    assert outcome["artifact_path"] == seen["artifact_path"]

    from gnomon.artifacts import read_artifact

    rows = read_artifact(seen["artifact_path"])["results"][0]["forecast"]
    assert outcome["prediction"] == [float(row["q50"]) for row in rows]


def test_own_values_route_by_tool_use(tmp_path):
    outcome = _run(_sample(), [
        {"tool_calls": [("submit_forecast", {"values": VALUES})]},
    ], tmp_path)
    assert outcome["route"] == "direct"
    assert outcome["prediction"] == VALUES
    assert outcome["support"] is None

    outcome = _run(_sample(), [
        {"tool_calls": [("gnomon_capabilities", {})]},
        {"tool_calls": [("submit_forecast", {"values": VALUES})]},
    ], tmp_path)
    assert outcome["route"] == "informed-direct"


def test_abstain_exit_is_recorded_with_its_route(tmp_path):
    outcome = _run(_sample(), [
        {"tool_calls": [("submit_forecast",
                         {"abstain": True,
                          "reasoning": "nothing defensible"})]},
    ], tmp_path)
    assert outcome["abstained"] is True
    assert outcome["route"] == "abstain"
    assert "model abstained after tool use" in outcome["reasons"]
    assert outcome["submit_reasoning"] == "nothing defensible"


# -- the engine abstention story through the tool surface -------------------

def test_unsupported_artifact_rejected_then_best_effort_labeled(tmp_path):
    """The sparse-history path: the engine abstains, the harness names
    the honest options, the MODEL chooses the labeled fallback — and
    the engine abstention it saw is counted, never hidden."""
    def call_forecast(messages):
        # The engine's default floor now answers with labelled rows; the
        # abstention this scenario exercises needs the strict floor.
        return {"tool_calls": [_forecast_call(
            messages, minimum_support="conditionally_supported")]}

    def submit_unsupported(messages):
        payload = _last_tool_payload(messages)
        assert payload.get("artifact_path")
        return {"tool_calls": [("submit_forecast",
                                {"artifact_path": payload["artifact_path"]})]}

    def retry_best_effort(messages):
        payload = _last_tool_payload(messages)
        assert payload["accepted"] is False
        assert "abstained" in payload["message"]
        assert "best_effort" in payload["message"]
        return {"tool_calls": [_forecast_call(messages, best_effort=True)]}

    def submit_fallback(messages):
        payload = _last_tool_payload(messages)
        return {"tool_calls": [("submit_forecast",
                                {"artifact_path": payload["artifact_path"]})]}

    outcome = _run(_sample(n=5), [call_forecast, submit_unsupported,
                                  retry_best_effort, submit_fallback],
                   tmp_path)
    assert outcome["abstained"] is False
    assert outcome["route"] == "gnomon"
    # The disclosed-fallback label survives; the refusal stays counted.
    assert outcome["support"] == "best_effort"
    assert outcome["engine_abstentions"] == 1
    assert len(outcome["prediction"]) == 4


def test_engine_abstention_answered_with_own_values_is_disclosed(tmp_path):
    def call_forecast(messages):
        return {"tool_calls": [_forecast_call(
            messages, minimum_support="conditionally_supported")]}

    def answer_past_refusal(messages):
        return {"tool_calls": [("submit_forecast", {"values": VALUES})]}

    outcome = _run(_sample(n=5), [call_forecast, answer_past_refusal],
                   tmp_path)
    assert outcome["route"] == "informed-direct"
    assert outcome["engine_abstentions"] == 1


# -- repairable rejections --------------------------------------------------

def test_unknown_artifact_is_repairable(tmp_path):
    def bad(messages):
        return {"tool_calls": [("submit_forecast",
                                {"artifact_path": "/no/such/artifact"})]}

    def recover(messages):
        payload = _last_tool_payload(messages)
        assert payload["accepted"] is False
        assert "not produced" in payload["message"]
        return {"tool_calls": [("submit_forecast", {"values": VALUES})]}

    outcome = _run(_sample(), [bad, recover], tmp_path)
    assert outcome["route"] == "direct"


def test_wrong_length_values_rejection_names_the_length(tmp_path):
    def recover(messages):
        payload = _last_tool_payload(messages)
        assert "got 2" in payload["message"]
        return {"tool_calls": [("submit_forecast", {"values": VALUES})]}

    outcome = _run(_sample(), [
        {"tool_calls": [("submit_forecast", {"values": VALUES[:2]})]},
        recover,
    ], tmp_path)
    assert outcome["abstained"] is False


def test_multiple_exits_at_once_are_rejected(tmp_path):
    def recover(messages):
        payload = _last_tool_payload(messages)
        assert "exactly one" in payload["message"]
        return {"tool_calls": [("submit_forecast", {"values": VALUES})]}

    outcome = _run(_sample(), [
        {"tool_calls": [("submit_forecast",
                         {"values": VALUES, "abstain": True})]},
        recover,
    ], tmp_path)
    assert outcome["abstained"] is False


# -- the path jail ----------------------------------------------------------

def test_jail_blocks_outside_paths_before_the_server(tmp_path):
    outside = tmp_path / "cached-benchmark-data.csv"
    outside.write_text("timestamp,value\n", encoding="utf-8")
    sessions = []

    def escape(messages):
        return {"tool_calls": [("gnomon_forecast", {
            "input": str(outside), "time_column": "timestamp",
            "target_column": "value", "horizon": 4,
        })]}

    def answer(messages):
        payload = _last_tool_payload(messages)
        assert payload["code"] == "PATH_JAIL"
        assert payload["authored_by"] == "harness"
        return {"tool_calls": [("submit_forecast", {"values": VALUES})]}

    _run(_sample(), [escape, answer], tmp_path, sessions=sessions)
    assert sessions[0].calls == []  # the call never reached the server


# -- caps abstain, never fall back -----------------------------------------

def test_no_submission_abstains(tmp_path):
    outcome = _run(_sample(), [
        {"content": "The forecast is about 101."},
        {"content": "As I said, about 101."},
    ], tmp_path)
    assert outcome["abstained"] is True
    assert any("no submission" in reason for reason in outcome["reasons"])


def test_token_cap_abstains(tmp_path):
    outcome = _run(_sample(), [
        {"content": "hmm", "bump_tokens": 300_000},
        {"content": "more"},
    ], tmp_path)
    assert outcome["abstained"] is True
    assert any("cap:tokens" in reason for reason in outcome["reasons"])


# -- run_mtbench wiring -----------------------------------------------------

def test_forecast_sample_dispatches_mcp_mode(monkeypatch, tmp_path):
    from benchmarks.mtbench import gnomon_forecaster

    seen = {}

    def fake_run(sample, client, *, work_dir=None, **kwargs):
        seen["sample"] = sample["filename"]
        return {"abstained": True, "reasons": ["stub"]}

    monkeypatch.setattr("benchmarks.mtbench.mcp_agent.run_sample_mcp",
                        fake_run)
    outcome = gnomon_forecaster.forecast_sample(
        _sample(), mode="mcp", client=None, work_dir=str(tmp_path))
    assert outcome == {"abstained": True, "reasons": ["stub"]}
    assert seen["sample"] == "finance#0001"
