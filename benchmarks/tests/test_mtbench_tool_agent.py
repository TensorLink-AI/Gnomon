"""The MTBench tool arm's two honest exits: a scripted model, real Gnomon.

Follows ``test_cik_mcp_agent.py``: the chat client is a script; the
toolbox runs the real ``gnomon`` code in-process, so the ref-exit test
exercises an actual ``gnomon_forecast`` end to end. Under test: both
exits of ``submit_forecast`` and the route taxonomy (``gnomon`` /
``informed-direct`` / ``direct``), the verbatim guarantee on a
submitted ref, honest abstention via ``forecast_ref: "none"``, the
engine-abstention disclosure, and repairable rejections that name what
was received.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.mtbench.tool_agent import TOOL_SPECS, run_sample


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
    """A chat client that plays a fixed script instead of a model.

    Each step is a dict — ``{"content": ...}`` or ``{"tool_calls":
    [(name, args), ...]}`` — or a callable receiving the running message
    list (so a step can read a previous tool result, exactly as a model
    would).
    """

    def __init__(self, steps):
        self.steps = list(steps)

    def chat(self, messages, *, n=1, tools=None, tool_choice=None):
        assert self.steps, "model script exhausted before submission"
        step = self.steps.pop(0)
        action = step(messages) if callable(step) else step
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


def _last_tool_payload(messages) -> dict:
    for message in reversed(messages):
        if message.get("role") == "tool":
            return json.loads(message["content"])
    raise AssertionError("no tool result in the conversation yet")


VALUES = [101.0, 101.5, 102.0, 102.5]


# -- the tool surface -------------------------------------------------------

def test_submit_spec_offers_both_exits_and_forces_neither():
    submit = next(spec["function"] for spec in TOOL_SPECS
                  if spec["function"]["name"] == "submit_forecast")
    properties = submit["parameters"]["properties"]
    assert "forecast_ref" in properties
    assert "values" in properties
    # Neither exit is schema-mandatory: the exactly-one-of rule lives in
    # the handler, where a violation is a repairable error.
    assert "required" not in submit["parameters"]


# -- exits and routes -------------------------------------------------------

def test_ref_exit_uses_the_gnomon_trajectory_verbatim(tmp_path):
    seen = {}

    def submit_ref(messages):
        payload = _last_tool_payload(messages)
        assert payload["abstained"] is False
        seen["forecast"] = payload["forecast"]
        return {"tool_calls": [("submit_forecast",
                                {"forecast_ref": payload["forecast_ref"]})]}

    result = run_sample(
        _sample(), ScriptedClient([
            {"tool_calls": [("gnomon_forecast", {})]},
            submit_ref,
        ]), work_dir=str(tmp_path))
    assert result["abstained"] is False
    assert result["route"] == "gnomon"
    assert result["forecast_ref"] == "fc1"
    assert result["support"] is not None
    assert result["engine_abstentions"] == 0
    # Byte-for-byte the run the model selected, not an edit of it.
    assert [round(v, 4) for v in result["prediction"]] == seen["forecast"]


def test_own_values_with_no_tool_use_route_direct(tmp_path):
    result = run_sample(
        _sample(), ScriptedClient([
            {"tool_calls": [("submit_forecast",
                             {"values": VALUES, "reasoning": "my own"})]},
        ]), work_dir=str(tmp_path))
    assert result["abstained"] is False
    assert result["route"] == "direct"
    assert result["prediction"] == VALUES
    assert result["support"] is None and result["selected_model"] is None
    assert result["submit_reasoning"] == "my own"


def test_own_values_after_tools_route_informed_direct(tmp_path):
    result = run_sample(
        _sample(), ScriptedClient([
            {"tool_calls": [("series_stats", {})]},
            {"tool_calls": [("submit_forecast", {"values": VALUES})]},
        ]), work_dir=str(tmp_path))
    assert result["route"] == "informed-direct"
    assert result["prediction"] == VALUES


def test_engine_abstention_can_be_answered_and_is_disclosed(tmp_path):
    # 5 bars cannot carry a 4-step backtested forecast: the engine
    # abstains. The model may still answer — labeled informed-direct,
    # with the abstention it saw counted, never hidden.
    def answer_past_refusal(messages):
        payload = _last_tool_payload(messages)
        assert payload["abstained"] is True
        return {"tool_calls": [("submit_forecast", {"values": VALUES})]}

    result = run_sample(
        _sample(n=5), ScriptedClient([
            {"tool_calls": [("gnomon_forecast", {})]},
            answer_past_refusal,
        ]), work_dir=str(tmp_path))
    assert result["abstained"] is False
    assert result["route"] == "informed-direct"
    assert result["engine_abstentions"] == 1


def test_none_still_abstains_honestly(tmp_path):
    result = run_sample(
        _sample(), ScriptedClient([
            {"tool_calls": [("gnomon_forecast", {})]},
            {"tool_calls": [("submit_forecast",
                             {"forecast_ref": "none",
                              "reasoning": "nothing defensible"})]},
        ]), work_dir=str(tmp_path))
    assert result["abstained"] is True
    assert "model abstained after tool use" in result["reasons"]


def test_no_submission_is_an_abstention(tmp_path):
    result = run_sample(
        _sample(), ScriptedClient([
            {"content": "The forecast is about 101."},
            {"content": "As I said, about 101."},
        ] + [{"content": "..."}] * 8), work_dir=str(tmp_path))
    assert result["abstained"] is True
    assert "model submitted no forecast" in result["reasons"]


# -- repairable rejections --------------------------------------------------

def test_wrong_length_values_rejection_names_the_length(tmp_path):
    def recover(messages):
        payload = _last_tool_payload(messages)
        assert "got 2" in payload["error"]
        return {"tool_calls": [("submit_forecast", {"values": VALUES})]}

    result = run_sample(
        _sample(), ScriptedClient([
            {"tool_calls": [("submit_forecast", {"values": VALUES[:2]})]},
            recover,
        ]), work_dir=str(tmp_path))
    assert result["abstained"] is False
    assert result["route"] == "direct"


def test_non_finite_values_are_rejected(tmp_path):
    def recover(messages):
        payload = _last_tool_payload(messages)
        assert "values[1]" in payload["error"]
        return {"tool_calls": [("submit_forecast", {"values": VALUES})]}

    result = run_sample(
        _sample(), ScriptedClient([
            {"tool_calls": [("submit_forecast",
                             {"values": [1.0, None, 2.0, 3.0]})]},
            recover,
        ]), work_dir=str(tmp_path))
    assert result["abstained"] is False


def test_both_exits_at_once_is_rejected(tmp_path):
    def recover(messages):
        payload = _last_tool_payload(messages)
        assert "exactly one" in payload["error"]
        return {"tool_calls": [("submit_forecast", {"values": VALUES})]}

    result = run_sample(
        _sample(), ScriptedClient([
            {"tool_calls": [("submit_forecast",
                             {"forecast_ref": "fc1", "values": VALUES})]},
            recover,
        ]), work_dir=str(tmp_path))
    assert result["abstained"] is False


def test_unknown_ref_is_repairable(tmp_path):
    def recover(messages):
        payload = _last_tool_payload(messages)
        assert "unknown forecast_ref" in payload["error"]
        return {"tool_calls": [("submit_forecast", {"values": VALUES})]}

    result = run_sample(
        _sample(), ScriptedClient([
            {"tool_calls": [("submit_forecast", {"forecast_ref": "fc9"})]},
            recover,
        ]), work_dir=str(tmp_path))
    assert result["abstained"] is False
    assert result["route"] == "direct"


def test_zero_tool_calls_is_recorded_as_zero_not_one(monkeypatch, tmp_path):
    # tools/mcp outcomes carry their real call count, and 0 is a real
    # count: the model abstained without ever invoking a tool.
    # `int(outcome.get("tool_calls") or 1)` mapped that 0 to 1 on exactly
    # the rows where the measurement matters most.
    import json

    from benchmarks.mtbench import gnomon_forecaster as gf

    sample = {"filename": "finance#0001", "output_window": [1.0, 2.0]}
    monkeypatch.setattr(gf, "load_samples", lambda folder: [sample])
    monkeypatch.setattr(
        gf, "forecast_sample",
        lambda s, *, mode, client, work_dir: {
            "abstained": True, "reasons": ["declined"],
            "route": "abstain", "tool_calls": 0,
        })
    gf.run(tmp_path, tmp_path / "out", mode="pure",
           openrouter_model=None, mtbench_root=None)
    (row_line,) = [
        line for line in (tmp_path / "out" / "gnomonbench.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    row = json.loads(row_line)
    assert row["tool_calls"] == 0
    assert row["appropriate_abstention"] is True
