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
    _task_companion_evidence,
    _transformation_repair_hints,
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

    def __init__(self, steps, compiler_output=None):
        self.steps = list(steps)
        default = json.dumps({
            "events": [], "claims": [], "forecast_candidate": None})
        self.compiler_outputs = (list(compiler_output)
                                 if isinstance(compiler_output, list)
                                 else [compiler_output or default])
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.completion_temperatures = []
        self.completion_prompts = []

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

    def completions(self, messages, *, n=1, temperature=None):
        self.completion_temperatures.append(temperature)
        self.completion_prompts.append(messages[-1]["content"])
        self.total_prompt_tokens += 100
        self.total_completion_tokens += 25
        value = (self.compiler_outputs.pop(0) if len(self.compiler_outputs) > 1
                 else self.compiler_outputs[0])
        return [value for _ in range(n)]

    @property
    def usage_summary(self):
        return {"model": "scripted", "requests": 0,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens}


def test_companion_evidence_is_bounded_and_pre_cutoff_only():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {"predictor": range(40), "target": range(100, 140)},
        index=pd.date_range("2024-01-01", periods=40, freq="h"),
    )
    evidence = _task_companion_evidence(SimpleNamespace(past_time=frame))
    assert "predictor" in evidence
    assert "target" not in evidence
    observed = [float(row.rsplit(",", 1)[-1])
                for row in evidence.splitlines()[2:]]
    assert observed == list(map(float, range(8, 40)))
    assert len(evidence.splitlines()) == 34


def test_compiler_contract_separates_history_from_future_covariates():
    from benchmarks.cik.mcp_agent import DOSSIER_INSTRUCTIONS

    assert "NEVER copy those historical rows into covariate_tables" in \
        DOSSIER_INSTRUCTIONS
    assert "exact requested forecast timestamp" in DOSSIER_INSTRUCTIONS


def test_transformation_repair_hints_are_verbatim_and_constant_specific():
    failures = [{"violations": [{"message":
        "Transformation constant 37.5 (literal) is absent from every cited source span."}]}]
    context = ("Maximum speed is 3000 rpm and pressure is 37.5 Pa.\n"
               "An unrelated threshold is 20 Pa.")
    assert _transformation_repair_hints(failures, context) == [
        "Maximum speed is 3000 rpm and pressure is 37.5 Pa."]


def _forecaster(steps, tmp_path, sessions=None, profile=None,
                compiler_output=None):
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
    proposal = json.dumps({"events": [{
            "event_type": "constraint:announced_cap",
            "effective_start": task.future_time[0],
            "effective_end": task.future_time[-1],
            "confidence": 1.0,
            "source_span": span,
            "rationale": "The task states a future cap.",
        }], "claims": [], "forecast_candidate": None})
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


def test_evidence_binds_only_cited_host_timestamped_covariates(tmp_path):
    task = _task()
    date = task.future_time[0].split("T", 1)[0]
    span = f"On {date} the published weather input is 2.0."
    task.scenario = span
    compiler_output = json.dumps({
        "events": [], "claims": [], "forecast_candidate": None,
        "covariate_tables": [{
            "name": "weather", "type": "continuous",
            "rows": [{
                "document_index": 0,
                "timestamp": task.future_time[0],
                "source_time_span": date,
                "value": 2.0,
                "evidence_quote": span,
                # Host ownership must override this attempted backdate.
                "known_at": task.past_time[0][0],
            }],
        }],
    })
    sessions = []
    forecaster = _forecaster(
        [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
        tmp_path, sessions=sessions, profile="evidence",
        compiler_output=compiler_output)
    _, extra = forecaster(task, 1)
    arguments = sessions[0].calls[0][1]
    assert arguments["covariate_mapping"] == [{
        "name": "weather", "type": "continuous",
        "availability": "future_known",
    }]
    assert arguments["covariates"] == [{
        "timestamp": task.future_time[0],
        "known_at": task.past_time[-1][0],
        "weather": 2.0,
    }]
    assert extra["context_compilation"]["covariate_tables"] == 1
    receipt = json.loads(Path(
        extra["context_compilation"]["receipt_path"]).read_text())
    row = receipt["covariates"]["tables"][0]["rows"][0]
    assert row["provenance"]["evidence_quote"] == span
    assert row["known_at"] == task.past_time[-1][0]


def test_evidence_retains_a_cited_sealed_llm_candidate(tmp_path):
    task = _task()
    span = "A closure is expected throughout the forecast window."
    task.scenario = span
    rows = [
        {"timestamp": stamp, "q10": 125 + index, "q50": 128 + index,
         "q90": 131 + index}
        for index, stamp in enumerate(task.future_time)
    ]
    compiler_output = json.dumps({
        "events": [],
        "claims": [{
            "source_span": span,
            "relation": "supports_decrease",
            "effective_start": task.future_time[0],
            "effective_end": task.future_time[-1],
            "mechanism": "closure",
            "confidence": 0.8,
        }],
        "forecast_candidate": {"quantiles": rows,
                               "rationale": "lower activity"},
    })
    forecaster = _forecaster(
        [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
        tmp_path, profile="evidence", compiler_output=compiler_output)
    _, extra = forecaster(task, 1)
    summary = extra["context_compilation"]
    assert summary["claim_count"] == 1
    assert summary["candidate_available"] is True
    assert extra["llm_candidate_shadow"]["support"] == "prior_assisted"
    assert extra["llm_candidate_shadow"]["automation_eligible"] is False
    receipt = json.loads(Path(summary["receipt_path"]).read_text())
    dossier = receipt["dossier"]
    assert dossier["candidate_support"] == "prior_assisted"
    assert dossier["automation_eligible"] is False
    assert dossier["primary_forecast_unchanged"] is True
    assert len(dossier["seal_sha256"]) == 64


def test_shadow_role_scores_candidate_without_replacing_canonical(tmp_path):
    task = _task()
    span = "A closure is expected throughout the forecast window."
    task.scenario = span
    rows = [
        {"timestamp": stamp, "q10": 124 + index, "q50": 127 + index,
         "q90": 130 + index}
        for index, stamp in enumerate(task.future_time)
    ]
    compiler_output = json.dumps({
        "events": [],
        "claims": [{
            "source_span": span, "relation": "supports_decrease",
            "effective_start": task.future_time[0],
            "effective_end": task.future_time[-1],
            "mechanism": "closure", "confidence": 0.8,
        }],
        "forecast_candidate": {"quantiles": rows, "rationale": "lower"},
    })
    client = ScriptedClient(
        [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
        compiler_output)
    forecaster = McpAgentForecaster(
        "x/y", client=client,
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence",
        output_role="llm_candidate_shadow")
    samples, extra = forecaster(task, 1)
    assert [row[0] for row in samples[0]] == [127, 128, 129, 130]
    assert extra["route"] == "llm_candidate_shadow"
    assert extra["candidate_support"] == "prior_assisted"
    assert extra["automation_eligible"] is False
    assert extra["primary_forecast_unchanged"] is True


def test_best_effort_role_uses_verified_product_publication(tmp_path):
    task = _task()
    span = "A closure is expected throughout the forecast window."
    task.scenario = span
    rows = [{"timestamp": stamp, "q10": 124 + index,
             "q50": 127 + index, "q90": 130 + index}
            for index, stamp in enumerate(task.future_time)]
    compiler_output = json.dumps({
        "events": [], "claims": [{
            "source_span": span, "relation": "supports_decrease",
            "effective_start": task.future_time[0],
            "effective_end": task.future_time[-1],
            "mechanism": "closure", "confidence": 0.8}],
        "forecast_candidate": {"quantiles": rows, "rationale": "lower"},
    })
    forecaster = McpAgentForecaster(
        "x/y", client=ScriptedClient(
            [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
            compiler_output),
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence",
        output_role="publication_best_effort")
    samples, extra = forecaster(task, 1)
    assert [row[0] for row in samples[0]] == [127, 128, 129, 130]
    assert extra["route"] == "publication_best_effort"
    assert extra["publication"]["recommended_support"] == "prior_assisted"
    assert extra["publication"]["primary_forecast_unchanged"] is True
    assert extra["publication"]["automation"]["eligible"] is False


def test_best_effort_role_exercises_live_safe_transformation_surface(tmp_path):
    task = _task()
    span = "A new policy makes each future value exactly half the usual value."
    task.scenario = span
    compiler_output = json.dumps({
        "events": [],
        "claims": [{
            "source_span": span, "relation": "supports_decrease",
            "effective_start": task.future_time[0],
            "effective_end": task.future_time[-1],
            "mechanism": "stated multiplicative rule", "confidence": .9,
        }],
        "transformations": [{"transformation": {
            "known_at": task.past_time[-1][0], "claim_ids": ["claim-1"],
            "lane": "prior_assisted", "output_unit": "unknown",
            "expression": {"op": "multiply", "args": [
                {"op": "primary", "quantile": "q50"},
                {"op": "literal", "value": .5,
                 "unit": "dimensionless"},
            ]},
        }}],
    })
    sessions = []
    def factory(cwd):
        session = InProcessMcpSession(cwd)
        sessions.append(session)
        return session
    forecaster = McpAgentForecaster(
        "x/y", client=ScriptedClient(
            [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
            compiler_output),
        session_factory=factory, work_dir=str(tmp_path), profile="evidence",
        output_role="publication_best_effort")
    _, extra = forecaster(task, 1)
    publication = extra["publication"]
    assert extra["context_compilation"]["transformations_proposed"] == 1
    assert sessions[0].calls[0][1]["context_submission"]["transformations"]
    assert publication["recommended_scenario_id"] == "transformation-1"
    assert publication["recommended_support"] == "prior_assisted"
    assert publication["automation"]["eligible"] is False
    assert publication["primary_forecast_unchanged"] is True


def test_transformation_gets_one_bounded_provenance_repair(tmp_path):
    task = _task()
    span = "A new policy makes each future value exactly half the usual value."
    task.scenario = span
    claim = {
        "source_span": span, "relation": "supports_decrease",
        "effective_start": task.future_time[0],
        "effective_end": task.future_time[-1],
        "mechanism": "stated multiplicative rule", "confidence": .9,
    }

    def dossier(multiplier):
        return json.dumps({
            "events": [], "claims": [claim],
            "transformations": [{"transformation": {
                "known_at": task.past_time[-1][0],
                "claim_ids": ["claim-1"], "lane": "prior_assisted",
                "output_unit": "unknown",
                "expression": {"op": "multiply", "args": [
                    {"op": "primary", "quantile": "q50"},
                    {"op": "literal", "value": multiplier,
                     "unit": "dimensionless"},
                ]},
            }}],
        })

    client = ScriptedClient(
        [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
        [dossier(.7), dossier(.5)],
    )
    forecaster = McpAgentForecaster(
        "x/y", client=client,
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence",
        output_role="publication_best_effort")
    _, extra = forecaster(task, 1)

    assert extra["publication"]["recommended_scenario_id"] == "transformation-1"
    assert not any("transformation_preflight_rejected" in reason for reason in
                   extra["context_compilation"].get("rejections", []))
    assert client.total_prompt_tokens >= 200
    assert client.completion_temperatures == [0, 0]


def test_sealed_candidate_survives_rejected_relational_transform(tmp_path):
    task = _task()
    span = "A new policy makes each future value exactly half the usual value."
    task.scenario = span
    rows = [{"timestamp": stamp, "q10": 124 + index,
             "q50": 127 + index, "q90": 130 + index}
            for index, stamp in enumerate(task.future_time)]
    compiler_output = json.dumps({
        "events": [],
        "claims": [{
            "source_span": span, "relation": "supports_decrease",
            "effective_start": task.future_time[0],
            "effective_end": task.future_time[-1],
            "mechanism": "stated multiplicative rule", "confidence": .9,
        }],
        "forecast_candidate": {
            "quantiles": rows,
            "rationale": "A sealed probabilistic fallback to the relation.",
        },
        "transformations": [{"transformation": {
            "known_at": task.past_time[-1][0], "claim_ids": ["claim-1"],
            "lane": "prior_assisted", "output_unit": "unknown",
            "expression": {"op": "multiply", "args": [
                {"op": "primary", "quantile": "q50"},
                {"op": "literal", "value": .7,
                 "unit": "dimensionless"},
            ]},
        }}],
    })
    forecaster = McpAgentForecaster(
        "x/y", client=ScriptedClient(
            [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
            compiler_output),
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence",
        output_role="publication_best_effort")
    samples, extra = forecaster(task, 1)

    assert [row[0] for row in samples[0]] == [127, 128, 129, 130]
    assert extra["publication"]["recommended_scenario_id"] == "prior-assisted-1"
    assert extra["publication"]["primary_forecast_unchanged"] is True
    assert extra["publication"]["automation"]["eligible"] is False
    receipt = json.loads(Path(
        extra["context_compilation"]["receipt_path"]).read_text())
    assert any("transformation_preflight_rejected" in reason
               for reason in receipt["rejections"])


def test_single_repair_exposes_effect_and_candidate_failures(tmp_path):
    task = _task()
    span = "Demand doubles during the forecast window."
    task.scenario = span
    claim = {
        "source_span": span, "relation": "supports_increase",
        "effective_start": task.future_time[0],
        "effective_end": task.future_time[-1],
        "mechanism": "stated multiplier", "confidence": .9,
    }
    invalid = json.dumps({
        "events": [], "claims": [claim],
        "forecast_candidate": {
            "quantiles": [{"timestamp": stamp, "q10": 0,
                           "q50": 0, "q90": 0}
                          for stamp in task.future_time],
            "rationale": "Placeholder; Gnomon must apply it.",
        },
        "effect_proposal": {
            "shape": "cross_series_relationship", "unit": "target_units",
            "location": 1, "lower": 1, "upper": 1, "confidence": .8,
            "delay_steps": 0, "duration_steps": 4,
            "scope": {"kind": "single_series", "series": ["*"]},
            "claim_ids": ["claim-1"], "rationale": "doubling",
            "uncertainty_basis": "stated rule",
        },
    })
    repaired = json.dumps({"events": [], "claims": [claim],
                           "forecast_candidate": None,
                           "effect_proposal": None})
    client = ScriptedClient(
        [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
        [invalid, repaired])
    forecaster = McpAgentForecaster(
        "x/y", client=client,
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence")
    forecaster(task, 1)

    repair_prompt = client.completion_prompts[1]
    assert '"effect_proposal"' in repair_prompt
    assert '"forecast_candidate"' in repair_prompt
    assert "declares itself incomplete" in repair_prompt
    assert "CROSS_SERIES_SCOPE_REQUIRED" in repair_prompt
    assert client.completion_temperatures == [0, 0]


def test_shadow_role_requires_evidence_profile():
    with pytest.raises(ValueError, match="requires the evidence profile"):
        McpAgentForecaster("x/y", profile="full",
                           output_role="llm_candidate_shadow")


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


def test_dossier_keeps_historical_context_as_claim_not_executable_event(tmp_path):
    task = _task()
    span = "A closure occurred during the first week of January."
    task.scenario = span
    compiler_output = json.dumps({
        "events": [{
            "event_type": "historical_closure",
            "effective_start": task.past_time[0][0],
            "effective_end": task.past_time[6][0],
            "source_span": span,
            "confidence": 1.0,
        }],
        "claims": [{
            "source_span": span, "relation": "supports_decrease",
            "effective_start": task.past_time[0][0],
            "effective_end": task.past_time[6][0],
            "mechanism": "closure", "confidence": 1.0,
        }],
        "forecast_candidate": None,
    })
    sessions = []
    forecaster = _forecaster(
        [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
        tmp_path, sessions=sessions, profile="evidence",
        compiler_output=compiler_output)
    _, extra = forecaster(task, 1)
    assert sessions[0].calls[0][1]["context_events"] == []
    assert extra["context_compilation"]["event_count"] == 0
    assert extra["context_compilation"]["claim_count"] == 1
    assert extra["context_compilation"]["rejection_count"] == 1


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


def test_cache_name_separates_provider_endpoints():
    engy = McpAgentForecaster("x/y", client=ScriptedClient([]), profile="evidence")
    other_client = ScriptedClient([])
    other_client.base_url = "https://other.invalid/v1"
    other = McpAgentForecaster("x/y", client=other_client, profile="evidence")
    assert engy.cache_name != other.cache_name


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
