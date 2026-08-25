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

    def __init__(self, steps, compiler_output="[]"):
        self.steps = list(steps)
        self.compiler_output = compiler_output
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

    def completions(self, messages, *, n=1):
        self.total_prompt_tokens += 100
        self.total_completion_tokens += 25
        return [self.compiler_output for _ in range(n)]

    @property
    def usage_summary(self):
        return {"model": "scripted", "requests": 0,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens}


def _forecaster(steps, tmp_path, sessions=None, profile=None,
                compiler_output="[]"):
    def factory(cwd):
        session = InProcessMcpSession(cwd)
        if sessions is not None:
            sessions.append(session)
        return session

    return McpAgentForecaster(
        "x/y", client=ScriptedClient(steps, compiler_output), session_factory=factory,
        work_dir=str(tmp_path), trace_dir=tmp_path / "traces", profile=profile,
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


def test_informed_direct_when_tools_were_consulted(tmp_path, monkeypatch):
    monkeypatch.setenv("GNOMON_MCP_PROFILE", "full")
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


def test_evidence_host_binds_first_valid_forecast_artifact(tmp_path):
    """A governed agent chooses the verb; the host owns publication."""
    def call_forecast(messages):
        csv = _csv_path(messages)
        return {"tool_calls": [("gnomon_forecast", {
            "input": csv, "time_column": "timestamp",
            "target_column": "value", "horizon": 4, "frequency": "D",
            "output_dir": str(Path(csv).parent / "gnomon-output"),
        })]}

    client = ScriptedClient([call_forecast])

    def factory(cwd):
        return InProcessMcpSession(cwd)

    forecaster = McpAgentForecaster(
        "x/y", client=client, session_factory=factory,
        work_dir=str(tmp_path), trace_dir=tmp_path / "traces",
        profile="evidence",
    )
    samples, extra = forecaster(_task(), 1)
    assert extra["route"] == "gnomon"
    assert extra["mcp_calls"] == 1
    assert len(samples[0]) == 4
    trace = json.loads(next((tmp_path / "traces").glob("*.json")).read_text())
    assert trace["trace"][0]["host_bound_submission"] == {
        "accepted": True, "route": "gnomon"}


def test_evidence_compiles_and_host_binds_context(tmp_path):
    task = _task()
    span = "Values will not exceed 120 during the forecast window."
    task.scenario = span
    proposal = json.dumps([{
        "event_type": "constraint:announced_cap",
        "effective_start": task.future_time[0],
        "effective_end": task.future_time[-1],
        "confidence": 1.0,
        "source_span": span,
        "rationale": "The task states a future cap.",
    }])
    sessions = []

    def call_forecast(messages):
        assert "already compiled" in messages[0]["content"]
        return {"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}

    forecaster = _forecaster(
        [call_forecast], tmp_path, sessions=sessions, profile="evidence",
        compiler_output=proposal)
    _, extra = forecaster(task, 1)

    assert extra["route"] == "gnomon"
    assert extra["context_compilation"]["event_count"] == 1
    assert extra["context_compilation"]["future_observations_exposed"] is False
    call_name, arguments = sessions[0].calls[0]
    assert call_name == "gnomon_forecast"
    assert arguments["input"].endswith("history.csv")
    assert arguments["horizon"] == 4
    assert arguments["future_events"] is True
    assert arguments["structural_events"] is True
    assert arguments["context_events"][0]["event_type"] \
        == "constraint:announced_cap"
    assert arguments["context_events"][0]["known_at"] \
        == task.past_time[-1][0]
    receipt = json.loads(Path(
        extra["context_compilation"]["receipt_path"]).read_text())
    assert receipt["future_observations_exposed"] is False
    assert receipt["source"]["sha256"] \
        == extra["context_compilation"]["source_sha256"]


def test_evidence_exposes_only_the_task_required_forecast_tool(tmp_path):
    class InspectingClient(ScriptedClient):
        def chat(self, messages, *, n=1, tools=None, tool_choice=None):
            assert [item["function"]["name"] for item in tools] == [
                "gnomon_forecast", "submit_forecast"]
            return super().chat(messages, n=n, tools=tools,
                                tool_choice=tool_choice)

    client = InspectingClient([{
        "tool_calls": [("gnomon_forecast", {"frequency": "D"})],
    }])
    forecaster = McpAgentForecaster(
        "x/y", client=client,
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence")
    _, extra = forecaster(_task(), 1)
    assert extra["mcp_calls"] == 1


def test_evidence_rejects_model_authored_quantiles(tmp_path):
    def call_forecast(messages):
        refusal = _last_tool_payload(messages)
        assert refusal["accepted"] is False
        assert "model-authored quantiles are disabled" in refusal["message"]
        csv = _csv_path(messages)
        return {"tool_calls": [("gnomon_forecast", {
            "input": csv, "time_column": "timestamp",
            "target_column": "value", "horizon": 4, "frequency": "D",
            "output_dir": str(Path(csv).parent / "gnomon-output"),
        })]}

    forecaster = _forecaster([
        {"tool_calls": [("submit_forecast", {"quantiles": QUANTILES})]},
        call_forecast,
    ], tmp_path, profile="evidence")
    _, extra = forecaster(_task(), 1)
    assert extra["route"] == "gnomon"


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

    evidence_args = parser.parse_args([
        "--method", "gnomon-mcp", "--model", "x/y",
        "--mcp-profile", "evidence", "--output-dir", str(tmp_path)])
    evidence = build_method(evidence_args)
    assert evidence.profile == "evidence"
    assert "profile=evidence" in evidence.cache_name

    flagged = parser.parse_args([
        "--method", "gnomon-mcp", "--model", "x/y", "--future-context",
        "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        build_method(flagged)


def test_run_cik_conditional_is_an_explicit_immutable_primary_arm(
        tmp_path, monkeypatch):
    from benchmarks.cik import gnomon_forecaster
    from benchmarks.cik.run_cik import build_method, build_parser

    monkeypatch.setattr(gnomon_forecaster, "OpenRouterClient",
                        lambda *args, **kwargs: object())
    args = build_parser().parse_args([
        "--method", "gnomon-conditional", "--model", "x/y",
        "--output-dir", str(tmp_path)])
    method = build_method(args)

    assert method.mode == "agent"
    assert method.future_context is True
    assert method.structural_context is False
    assert "future=on" in method.cache_name


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


# -- superseded tool results ------------------------------------------------

def _forecast_payload(channels, path="/artifacts/f1", support="supported",
                      warning="short history"):
    return {
        "status": "complete", "artifact_path": path,
        "headline": f"{len(channels)} channel(s), weakest tier {support}.",
        "results": [
            {"series": name, "support": support,
             "support_assessment": {"reasons": [{"code": support}],
                                    "assumptions": []},
             "warnings": [f"{name}: {warning}"],
             "notes": [], "selected_model": "seasonal_naive",
             "forecast": [{"timestamp": f"t{i}", "q10": 1.0, "q50": 2.0,
                           "q90": 3.0} for i in range(29)]}
            for name in channels
        ],
    }


def _tool_message(payload):
    return {"role": "tool", "tool_call_id": "c", "content": json.dumps(payload)}


def test_superseding_forecast_compacts_the_older_result_to_disclosures():
    from benchmarks.cik.mcp_agent import ToolMessageLog

    log = ToolMessageLog()
    old = _tool_message(_forecast_payload(["hr"], path="/artifacts/f1",
                                          support="best_effort",
                                          warning="only 12 observations"))
    assert log.record("gnomon_forecast", {"target_column": "hr"}, old) == 0
    new = _tool_message(_forecast_payload(["hr"], path="/artifacts/f2",
                                          warning="short history"))
    assert log.record("gnomon_forecast", {"target_column": "hr"}, new) == 1

    stub = json.loads(old["content"])
    assert stub["harness_superseded"] is True
    # Disclosures the later result does NOT carry stay verbatim, with
    # the path to the full numbers.
    assert stub["artifact_path"] == "/artifacts/f1"
    result, = stub["results"]
    assert result["support"] == "best_effort"
    assert result["warnings"] == ["hr: only 12 observations"]
    assert result["support_assessment"]["reasons"] == [
        {"code": "best_effort"}]
    assert "gnomon_get_artifact" in stub["harness_note"]
    # What it drops is the bulk, and only the bulk.
    assert "forecast" not in result
    assert len(old["content"]) < len(new["content"])
    # The live result is untouched.
    assert "harness_superseded" not in json.loads(new["content"])


def test_disclosures_identical_in_the_live_result_become_a_marker():
    from benchmarks.cik.mcp_agent import ToolMessageLog

    # The measured pathology: five consecutive calls over the same
    # channels, near-identical results. The words are one message down,
    # attached to the live numbers — the stub says so instead of
    # repeating six channels of identical degradation warnings.
    log = ToolMessageLog()
    old = _tool_message(_forecast_payload(["hr", "spo2"], path="/a/f1"))
    log.record("gnomon_forecast", {"target_column": "hr,spo2"}, old)
    new = _tool_message(_forecast_payload(["hr", "spo2"], path="/a/f1"))
    assert log.record("gnomon_forecast",
                      {"target_column": "hr,spo2"}, new) == 1

    stub = json.loads(old["content"])
    assert stub["harness_superseded"] is True
    assert stub["artifact_path"] == "/a/f1"
    assert "identical" in stub["unchanged"]
    assert len(old["content"]) < 700
    # A partially identical result keeps the differing disclosure
    # verbatim and markers only what the live result already carries.
    log2 = ToolMessageLog()
    payload = _forecast_payload(["hr", "spo2"], path="/a/f3")
    payload["results"][0]["warnings"] = ["hr: sensor swapped"]
    changed = _tool_message(payload)
    log2.record("gnomon_forecast", {"target_column": "hr,spo2"}, changed)
    live = _tool_message(_forecast_payload(["hr", "spo2"], path="/a/f4"))
    assert log2.record("gnomon_forecast",
                       {"target_column": "hr,spo2"}, live) == 1
    stub2 = json.loads(changed["content"])
    hr = next(r for r in stub2["results"] if r["series"] == "hr")
    assert hr["warnings"] == ["hr: sensor swapped"]  # differs: verbatim
    assert "support_assessment" in hr["unchanged"]
    spo2 = next(r for r in stub2["results"] if r["series"] == "spo2")
    assert "warnings" not in spo2
    assert "warnings" in spo2["unchanged"]


def test_batched_forecast_supersedes_the_per_channel_calls_it_covers():
    from benchmarks.cik.mcp_agent import ToolMessageLog

    log = ToolMessageLog()
    hr = _tool_message(_forecast_payload(["hr"]))
    spo2 = _tool_message(_forecast_payload(["spo2"]))
    resp = _tool_message(_forecast_payload(["resp"]))
    log.record("gnomon_forecast", {"target_column": "hr"}, hr)
    log.record("gnomon_forecast", {"target_column": "spo2"}, spo2)
    log.record("gnomon_forecast", {"target_column": "resp"}, resp)
    batch = _tool_message(_forecast_payload(["hr", "spo2"], path="/a/batch"))
    compacted = log.record("gnomon_forecast",
                           {"target_column": "hr,spo2"}, batch)
    assert compacted == 2
    assert json.loads(hr["content"])["harness_superseded"] is True
    assert json.loads(spo2["content"])["harness_superseded"] is True
    # A channel the batch did not cover keeps its full result.
    assert "harness_superseded" not in json.loads(resp["content"])


@pytest.mark.parametrize("changed", [
    {"input": "other.csv"},
    {"horizon": 14},
    {"as_of": "2026-01-01T00:00:00Z"},
    {"threshold": 90.0},
    {"candidates": ["theta"]},
    {"repair": "aggressive"},
    {"context_events_file": "events.json"},
])
def test_forecast_semantic_changes_do_not_supersede(changed):
    from benchmarks.cik.mcp_agent import ToolMessageLog

    log = ToolMessageLog()
    base_args = {"input": "series.csv", "target_column": "hr", "horizon": 7}
    old = _tool_message(_forecast_payload(["hr"]))
    log.record("gnomon_forecast", base_args, old)

    new_args = {**base_args, **changed}
    new = _tool_message(_forecast_payload(["hr"], path="/a/changed"))
    assert log.record("gnomon_forecast", new_args, new) == 0
    assert "harness_superseded" not in json.loads(old["content"])


def test_forecast_format_only_change_can_supersede():
    from benchmarks.cik.mcp_agent import ToolMessageLog

    log = ToolMessageLog()
    old = _tool_message(_forecast_payload(["hr"]))
    log.record("gnomon_forecast",
               {"target_column": "hr", "horizon": 7, "format": "full"}, old)
    new = _tool_message(_forecast_payload(["hr"], path="/a/brief"))
    assert log.record(
        "gnomon_forecast",
        {"target_column": "hr", "horizon": 7, "format": "brief"}, new,
    ) == 1


def test_single_target_results_key_on_the_argument_not_the_placeholder():
    from benchmarks.cik.mcp_agent import ToolMessageLog

    # A single-target run names its one result "__default__"; keying on
    # that placeholder would collide forecasts of DIFFERENT columns.
    log = ToolMessageLog()
    hr = _tool_message(_forecast_payload(["__default__"]))
    log.record("gnomon_forecast", {"target_column": "hr"}, hr)
    spo2 = _tool_message(_forecast_payload(["__default__"]))
    assert log.record("gnomon_forecast", {"target_column": "spo2"},
                      spo2) == 0
    assert "harness_superseded" not in json.loads(hr["content"])
    hr_again = _tool_message(_forecast_payload(["__default__"]))
    assert log.record("gnomon_forecast", {"target_column": "hr"},
                      hr_again) == 1
    assert json.loads(hr["content"])["harness_superseded"] is True


def test_errors_neither_supersede_nor_get_compacted():
    from benchmarks.cik.mcp_agent import ToolMessageLog

    log = ToolMessageLog()
    good = _tool_message(_forecast_payload(["hr"]))
    log.record("gnomon_forecast", {"target_column": "hr"}, good)
    error = _tool_message({"status": "error", "error": {
        "code": "AMBIGUOUS_SCHEMA", "message": "…",
        "repair_options": [{"action": "supply_arguments"}]}})
    assert log.record("gnomon_forecast", {"target_column": "hr"},
                      error) == 0
    assert "harness_superseded" not in json.loads(good["content"])
    # And a later success leaves the (small, repair-carrying) error alone.
    retry = _tool_message(_forecast_payload(["hr"], path="/a/f3"))
    log.record("gnomon_forecast", {"target_column": "hr"}, retry)
    assert "harness_superseded" not in json.loads(error["content"])


def test_other_tools_supersede_only_on_identical_arguments():
    from benchmarks.cik.mcp_agent import ToolMessageLog

    log = ToolMessageLog()
    first = _tool_message({"status": "valid", "series": [{"name": "cpu"}]})
    log.record("gnomon_inspect", {"input": "a.csv"}, first)
    other = _tool_message({"status": "valid", "series": [{"name": "mem"}]})
    # Different arguments are new evidence, not a replacement.
    assert log.record("gnomon_inspect",
                      {"input": "a.csv", "target_column": "mem"},
                      other) == 0
    repeat = _tool_message({"status": "valid", "series": [{"name": "cpu"}]})
    assert log.record("gnomon_inspect", {"input": "a.csv"}, repeat) == 1
    stub = json.loads(first["content"])
    assert stub["harness_superseded"] is True
    assert stub["status"] == "valid"
