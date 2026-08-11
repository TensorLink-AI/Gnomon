"""The integrated CiK arm: real MCP tools, a scripted model, no network.

The chat client is a script; the MCP session is the real server code
run in-process (``InProcessMcpSession`` calls ``gnomon.mcp_server``'s
handler directly), so the gnomon-route test exercises an actual
``gnomon_forecast`` end to end. Under test: the verbatim tool surface,
both honest exits and the route taxonomy, the path jail, the repair
loop through ``submit_forecast``, and every cap abstaining instead of
falling back.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.gnomon_forecaster import GnomonAbstained
from benchmarks.cik.mcp_agent import (
    MAX_MCP_CALLS,
    MAX_ROUNDS,
    InProcessMcpSession,
    McpAgentForecaster,
    jail_violations,
    openai_tool_specs,
)


# -- fixtures ---------------------------------------------------------------

def _task(horizon: int = 4, n: int = 72):
    from datetime import datetime, timedelta, timezone

    epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
    day = timedelta(days=1)
    values = [100.0 + 0.4 * i + 3.0 * math.sin(2 * math.pi * i / 7)
              for i in range(n)]
    return SimpleNamespace(
        past_time=[((epoch + i * day).isoformat(), values[i])
                   for i in range(n)],
        future_time=[(epoch + (n + k) * day).isoformat()
                     for k in range(horizon)],
        background="Telemetry from one site.",
        constraints=None,
        scenario="Values will stay in their usual range.",
        name="FakeTask",
        seed=1,
    )


class ScriptedClient:
    """A chat client that plays a fixed script instead of a model.

    Each step is a dict — ``{"content": ...}`` or ``{"tool_calls":
    [(name, args), ...]}`` — or a callable receiving the running message
    list (so a step can read the jail path or a previous tool result,
    exactly as a model would).
    """

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

    @property
    def usage_summary(self):
        return {"model": "scripted", "requests": 0,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens}


def _forecaster(steps, tmp_path, sessions=None):
    def factory(cwd):
        session = InProcessMcpSession(cwd)
        if sessions is not None:
            sessions.append(session)
        return session

    return McpAgentForecaster(
        "x/y", client=ScriptedClient(steps), session_factory=factory,
        work_dir=str(tmp_path), trace_dir=tmp_path / "traces",
    )


def _csv_path(messages) -> str:
    match = re.search(r"(/\S*?history\.csv)", messages[0]["content"])
    assert match, "system prompt does not name the history file"
    return match.group(1)


def _last_tool_payload(messages) -> dict:
    for message in reversed(messages):
        if message.get("role") == "tool":
            return json.loads(message["content"])
    raise AssertionError("no tool result in the conversation yet")


QUANTILES = [{"q10": 90.0 + i, "q50": 100.0 + i, "q90": 110.0 + i}
             for i in range(4)]


# -- the verbatim tool surface ---------------------------------------------

def test_tool_specs_are_verbatim_plus_submit():
    from gnomon.toolspec import visible_tools

    tools = visible_tools()
    specs = openai_tool_specs(tools)
    assert [s["function"]["name"] for s in specs[:-1]] \
        == [t["name"] for t in tools]
    for spec, tool in zip(specs, tools):
        assert spec["function"]["description"] == tool["description"]
        assert spec["function"]["parameters"] == tool["inputSchema"]
    assert specs[-1]["function"]["name"] == "submit_forecast"


# -- exits and routes -------------------------------------------------------

def test_direct_exit_zero_calls_routes_direct(tmp_path):
    forecaster = _forecaster(
        [{"tool_calls": [("submit_forecast", {"quantiles": QUANTILES,
                                              "reasoning": "my own"})]}],
        tmp_path,
    )
    samples, extra = forecaster(_task(), 1)
    assert len(samples) == 1 and len(samples[0]) == 4
    # n_samples=1 puts the single path at probability 0.5 == q50.
    assert [row[0] for row in samples[0]] == [100.0, 101.0, 102.0, 103.0]
    assert extra["route"] == "direct"
    assert extra["mcp_calls"] == 0
    assert extra["submit_reasoning"] == "my own"


def test_informed_direct_when_tools_were_consulted(tmp_path):
    forecaster = _forecaster(
        [{"tool_calls": [("gnomon_capabilities", {})]},
         {"tool_calls": [("submit_forecast", {"quantiles": QUANTILES})]}],
        tmp_path,
    )
    _, extra = forecaster(_task(), 1)
    assert extra["route"] == "informed-direct"
    assert extra["mcp_calls"] == 1
    assert extra["tool_sequence"][0] == {"tool": "gnomon_capabilities",
                                         "is_error": False}


def test_gnomon_exit_uses_the_artifact_verbatim(tmp_path):
    def call_forecast(messages):
        csv = _csv_path(messages)
        return {"tool_calls": [("gnomon_forecast", {
            "input": csv, "time_column": "timestamp",
            "target_column": "value", "horizon": 4, "frequency": "D",
            "output_dir": str(Path(csv).parent / "gnomon-output"),
        })]}

    def submit(messages):
        payload = _last_tool_payload(messages)
        assert payload.get("artifact_path"), payload
        return {"tool_calls": [("submit_forecast",
                                {"artifact_path": payload["artifact_path"]})]}

    forecaster = _forecaster([call_forecast, submit], tmp_path)
    samples, extra = forecaster(_task(), 1)
    assert extra["route"] == "gnomon"
    assert extra["mcp_calls"] == 1
    assert extra["support"] is not None

    from gnomon.artifacts import read_artifact

    rows = read_artifact(extra["artifact_path"])["results"][0]["forecast"]
    assert [row[0] for row in samples[0]] \
        == [float(row["q50"]) for row in rows]


def test_submitting_an_unknown_artifact_is_repairable(tmp_path):
    def bad_submit(messages):
        return {"tool_calls": [("submit_forecast",
                                {"artifact_path": "/no/such/artifact"})]}

    def recover(messages):
        payload = _last_tool_payload(messages)
        assert payload["accepted"] is False
        assert "not produced" in payload["message"]
        return {"tool_calls": [("submit_forecast", {"quantiles": QUANTILES})]}

    forecaster = _forecaster([bad_submit, recover], tmp_path)
    _, extra = forecaster(_task(), 1)
    assert extra["route"] == "direct"


def test_malformed_quantiles_are_repairable(tmp_path):
    def recover(messages):
        payload = _last_tool_payload(messages)
        assert payload["accepted"] is False
        # The rejection names the received length, not just the rule.
        assert "got 2" in payload["message"]
        return {"tool_calls": [("submit_forecast", {"quantiles": QUANTILES})]}

    forecaster = _forecaster(
        [{"tool_calls": [("submit_forecast",
                          {"quantiles": QUANTILES[:2]})]},  # wrong length
         recover],
        tmp_path,
    )
    _, extra = forecaster(_task(), 1)
    assert extra["route"] == "direct"


def test_dict_of_arrays_quantiles_rejection_names_the_shape(tmp_path):
    # The natural wrong shape: parallel per-quantile arrays. The
    # rejection must say what arrived and show the accepted form, or a
    # model resends it until the rounds cap becomes an abstention.
    columnar = {
        "q10": [row["q10"] for row in QUANTILES],
        "q50": [row["q50"] for row in QUANTILES],
        "q90": [row["q90"] for row in QUANTILES],
    }

    def recover(messages):
        payload = _last_tool_payload(messages)
        assert payload["accepted"] is False
        assert "keys [q10, q50, q90]" in payload["message"]
        assert "one entry per step" in payload["message"]
        return {"tool_calls": [("submit_forecast", {"quantiles": QUANTILES})]}

    forecaster = _forecaster(
        [{"tool_calls": [("submit_forecast", {"quantiles": columnar})]},
         recover],
        tmp_path,
    )
    _, extra = forecaster(_task(), 1)
    assert extra["route"] == "direct"


# -- the path jail ----------------------------------------------------------

def test_jail_blocks_path_arguments_before_the_server(tmp_path):
    outside = tmp_path / "cached-benchmark-data.csv"
    outside.write_text("timestamp,value\n", encoding="utf-8")
    sessions = []

    def escape(messages):
        return {"tool_calls": [("gnomon_forecast", {
            "input": str(outside), "time_column": "timestamp",
            "target_column": "value", "horizon": 4,
        })]}

    def read_refusal_and_answer(messages):
        payload = _last_tool_payload(messages)
        assert payload["code"] == "PATH_JAIL"
        assert payload["authored_by"] == "harness"
        return {"tool_calls": [("submit_forecast", {"quantiles": QUANTILES})]}

    forecaster = _forecaster([escape, read_refusal_and_answer], tmp_path,
                             sessions=sessions)
    _, extra = forecaster(_task(), 1)
    assert sessions[0].calls == []  # the call never reached the server
    assert extra["tool_sequence"][0].get("jail_violations")


def test_jail_violation_rules():
    jail = Path("/tmp/jail-nonexistent-xyz")
    # Path-named arguments are jailed unconditionally.
    assert jail_violations({"input": "/etc/passwd"}, jail)
    assert not jail_violations({"input": "history.csv"}, jail)
    assert not jail_violations({"input": "store:mydata"}, jail)
    # Free text with "/" is fine unless it names an existing outside path.
    assert not jail_violations(
        {"attributes": {"source_span": "the speed is 30 km/h"}}, jail)
    assert jail_violations({"note": "/etc/passwd"}, jail)


# -- caps abstain, never fall back -----------------------------------------

def test_no_submission_after_nudge_abstains(tmp_path):
    forecaster = _forecaster(
        [{"content": "The forecast is about 100."},
         {"content": "As I said, about 100."}],
        tmp_path,
    )
    with pytest.raises(GnomonAbstained, match="no submission"):
        forecaster(_task(), 1)
    # The trace survives an abstention — it is the arm's measurement.
    assert list((tmp_path / "traces").glob("*.json"))


def test_round_cap_abstains(tmp_path):
    steps = [{"tool_calls": [("gnomon_capabilities", {})]}] * MAX_ROUNDS
    forecaster = _forecaster(steps, tmp_path)
    with pytest.raises(GnomonAbstained, match="cap:rounds"):
        forecaster(_task(), 1)


def test_tool_call_cap_abstains(tmp_path):
    per_round = 3
    rounds = MAX_MCP_CALLS // per_round + 1
    steps = [{"tool_calls": [("gnomon_capabilities", {})] * per_round}
             ] * rounds
    forecaster = _forecaster(steps, tmp_path)
    with pytest.raises(GnomonAbstained, match="cap:tool_calls"):
        forecaster(_task(), 1)


def test_token_cap_abstains(tmp_path):
    forecaster = _forecaster(
        [{"tool_calls": [("gnomon_capabilities", {})],
          "bump_tokens": 300_000},
         {"tool_calls": [("submit_forecast", {"quantiles": QUANTILES})]}],
        tmp_path,
    )
    with pytest.raises(GnomonAbstained, match="cap:tokens"):
        forecaster(_task(), 1)


# -- run_cik wiring ---------------------------------------------------------

def test_run_cik_accepts_the_method_and_rejects_lane_flags(tmp_path):
    from benchmarks.cik.run_cik import build_method, build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--method", "gnomon-mcp", "--model", "x/y",
        "--output-dir", str(tmp_path)])
    method = build_method(args)
    from benchmarks.cik.mcp_agent import MCP_CONTRACT_VERSION

    assert method.cache_name.startswith("McpAgentForecaster_model=x-y")
    assert f"contract={MCP_CONTRACT_VERSION}" in method.cache_name

    flagged = parser.parse_args([
        "--method", "gnomon-mcp", "--model", "x/y", "--future-context",
        "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        build_method(flagged)


def test_cache_name_carries_temperature_and_contract_version():
    """The official cache reuses results by this name; without the
    temperature and the arm's contract version, a rerun at different
    settings silently returned the old runs' results."""
    from benchmarks.cik.mcp_agent import MCP_CONTRACT_VERSION, McpAgentForecaster

    forecaster = McpAgentForecaster(
        "org/model", temperature=0.2, client=object(),
    )
    assert "temperature=0.2" in forecaster.cache_name
    assert f"contract={MCP_CONTRACT_VERSION}" in forecaster.cache_name
    hotter = McpAgentForecaster("org/model", temperature=1.0, client=object())
    assert hotter.cache_name != forecaster.cache_name


def test_stdio_session_kills_a_hung_server_instead_of_blocking_forever():
    import sys as _sys

    from benchmarks.cik.mcp_agent import StdioMcpSession

    session = StdioMcpSession(
        ".", command=[_sys.executable, "-c", "import time; time.sleep(30)"],
        call_timeout=0.3,
    )
    try:
        with pytest.raises(RuntimeError, match="did not answer"):
            session._rpc("initialize", {})
    finally:
        session.close()
