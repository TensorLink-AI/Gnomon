"""TemporalBench's gnomon-mcp arm: real MCP server in-process, scripted model.

Follows ``test_cik_mcp_agent.py``: the chat client is a script; the MCP
session is the real server code run in-process (CiK's
``InProcessMcpSession``), so the artifact-route tests exercise an actual
``gnomon_forecast`` through the real tool surface. Under test: the
per-channel exits and route taxonomy, the artifact-channel binding
(an artifact cannot be submitted for a channel it did not forecast, and
a batched multi-target artifact binds each channel to its OWN result),
the engine-abstention story through the tool surface (unsupported
artifact rejected with the best_effort recovery named, and a
model-driven ``best_effort: true`` retry accepted with its label), the
path jail, the token bounds on tool results, the last-call protocol
that keeps a breached cap from voiding completed work, and the T1/T3
MCQ contract.
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

from benchmarks.cik import mcp_agent as cik_mcp_agent
from benchmarks.cik.mcp_agent import InProcessMcpSession
from benchmarks.common.openrouter import OpenRouterError
from benchmarks.temporalbench import mcp_agent
from benchmarks.temporalbench.mcp_agent import (
    MAX_ROUNDS,
    bounded_tool_text,
    compact_context_compilation_for_prompt,
    compile_row_context,
    mcq_row,
    preferred_execution_tool,
    run_row,
)


def test_context_prompt_projection_is_bounded_and_keeps_routing_facts() -> None:
    events = [{
        "event_type": f"type-{index % 12}",
        "known_at": f"2026-01-{(index % 28) + 1:02d}T00:00:00Z",
        "claim": "large repeated claim that belongs only in the receipt " * 8,
    } for index in range(32)]
    compilation = {
        "receipt_id": "receipt-123",
        "events": events,
        "hypotheses": [{"claim": "h"}] * 5,
        "rejected": [
            {"reason_code": "missing_time", "source": "long text" * 50},
            {"reason_code": "missing_time"},
            {"reason_code": "unsupported_transform"},
        ],
    }

    projected = compact_context_compilation_for_prompt(compilation)

    assert projected["receipt_id"] == "receipt-123"
    assert projected["accepted_event_count"] == 32
    assert projected["accepted_hypothesis_count"] == 5
    assert projected["rejected_count"] == 3
    assert len(projected["event_types"]) == 8
    assert projected["event_types_omitted"] == 4
    assert projected["known_at_min"] == "2026-01-01T00:00:00Z"
    assert projected["known_at_max"] == "2026-01-28T00:00:00Z"
    assert projected["rejection_code_counts"] == {
        "missing_time": 2, "unsupported_transform": 1}
    assert projected["execution_binding"] == "host-bound complete receipt"
    assert len(json.dumps(projected)) < 800
    assert "large repeated claim" not in json.dumps(projected)


def test_temporal_question_compiler_uses_text_not_labels_and_reuses_receipt(
        tmp_path) -> None:
    row = _row(sparse_temp=False)
    row["mcq"] = {"volatility_change": {
        "question": "Will the fleet become more volatile?",
        "options": ["increased", "decreased"], "label": "decreased"}}
    client = ScriptedClient([{"tool_calls": [("submit_temporal_intent", {
        "status": "compiled", "questions": [{
            "id": "v", "verb": "predict", "property": "volatility",
            "target": {"kind": "aggregate", "members": ["hr", "spo2"]},
            "horizon": 4}]})]}])
    receipts = tmp_path / "receipts"
    first = mcp_agent.compile_row_temporal_questions(
        row, client, ["hr", "spo2"], str(receipts))
    assert not first["rejected"], first
    assert first["questions"][0]["target"]["aggregation"] == \
        "median_normalized_scale_ratio"
    rendered = next(receipts.iterdir()).read_text()
    assert "decreased" not in rendered
    reused = mcp_agent.compile_row_temporal_questions(
        row, ScriptedClient([]), ["hr", "spo2"], str(receipts))
    assert reused["receipt_reused"] is True
    assert reused["compiler_called"] is False


def test_t3_question_compiler_reads_pack_text_but_seals_oracle_fields(
        tmp_path) -> None:
    row = _t3_row()
    row["pack"][0]["evidence"] = {"answer_key_fact": 999}
    client = ScriptedClient([{"tool_calls": [("submit_temporal_intent", {
        "status": "compiled", "questions": [{
            "id": "q1", "verb": "describe", "property": "level",
            "target": "hr"}]})]}])

    result = mcp_agent.compile_row_temporal_questions(
        row, client, ["hr"], str(tmp_path / "receipts"))

    assert result["questions"][0]["property"] == "level"
    request = client.requests[0]
    rendered = json.dumps(request, sort_keys=True)
    assert "answer_key_fact" not in rendered
    assert '"label"' not in rendered
    assert "Higher" not in rendered


def test_failed_temporal_receipt_is_diagnostic_not_permanent_cache(
        tmp_path) -> None:
    row = _row(sparse_temp=False)
    row["mcq"] = {"volatility_change": {
        # Deliberately lacks an explicit property so deterministic recovery
        # cannot turn the malformed provider payload into a valid receipt.
        "question": "Will hr change?",
        "options": ["increased", "decreased"], "label": "decreased"}}
    receipts = tmp_path / "receipts"
    failed_client = ScriptedClient([{"tool_calls": [
        ("submit_temporal_intent", {
            "status": "compiled", "questions": "malformed"})]}])
    failed = mcp_agent.compile_row_temporal_questions(
        row, failed_client, ["hr", "spo2"], str(receipts))
    assert not failed["questions"]

    recovered_client = ScriptedClient([{"tool_calls": [
        ("submit_temporal_intent", {
            "status": "compiled", "questions": [{
                "id": "v", "verb": "predict", "property": "volatility",
                "target": "hr", "horizon": 4}]})]}])
    recovered = mcp_agent.compile_row_temporal_questions(
        row, recovered_client, ["hr", "spo2"], str(receipts))
    assert recovered["prior_failed_receipt"] is True
    assert recovered["questions"][0]["property"] == "volatility"
    assert (receipts / f"{row['id']}.json").exists()
    assert (receipts / f"{row['id']}.retry.json").exists()


def test_t1_compiler_uses_public_question_names_and_seals_options(
        tmp_path) -> None:
    row = {
        "id": "t1-explicit", "tier": "T1", "meta": {
            "main_key": "heart_rate", "n_horizon": 1},
        "labels": {"trend": "constant", "outliers": "stable"},
        "prompt": (
            "Task: answer about heart_rate.\n"
            "1) Trend: {\"upward\", \"downward\", \"constant\"}\n"
            "2) Outliers: {\"sudden_spike\", \"level_shift\", \"stable\"}\n"
            "Input (JSON): {\"heart_rate\":[1,2,3],\"secret\":999}\n"),
    }
    client = ScriptedClient([{"tool_calls": [("submit_temporal_intent", {
        "status": "compiled", "questions": [
            {"id": "trend", "verb": "describe", "property": "trend",
             "target": "heart_rate"},
            # The proposer makes the dangerous semantic substitution. The
            # explicit router must recover the actual observed request.
            {"id": "outliers", "verb": "describe", "property": "extreme",
             "target": "heart_rate"},
        ]})]}])

    result = mcp_agent.compile_row_temporal_questions(
        row, client, ["heart_rate"], str(tmp_path / "receipts"))

    assert [item["property"] for item in result["questions"]] == [
        "trend", "disturbance"]
    rendered = json.dumps(client.requests[0], sort_keys=True)
    assert "sudden_spike" not in rendered
    assert "level_shift" not in rendered
    assert "secret" not in rendered


def test_t1_binding_excludes_coordinate_arrays_from_series() -> None:
    row = {
        "tier": "T1", "meta": {"main_key": "heart_rate"},
        "prompt": ("Input (JSON):\n{\"heart_rate\":[70,null,72],"
                   "\"time_position_in_day\":[0,30,60]}")}
    run = object.__new__(mcp_agent._McqRun)
    assert run._row_channels(row) == {"heart_rate": [70.0, 72.0]}


def test_every_forecast_profile_has_a_host_compiled_first_tool():
    assert {
        profile: preferred_execution_tool(profile, True, host_compiled=True)
        for profile in ("core", "describe", "evidence", "mega", "full")
    } == {
        "core": "gnomon_forecast",
        "describe": "gnomon_forecast",
        "evidence": "gnomon_forecast",
        "mega": "gnomon_run",
        "full": "gnomon_forecast",
    }


def test_forecast_host_instruction_preserves_typed_abstention() -> None:
    assert "support: abstained" in mcp_agent.SYSTEM
    assert "choosing `Uncertain`" in mcp_agent.SYSTEM
    assert preferred_execution_tool("core", True) is None
    assert preferred_execution_tool("describe", True) is None
    assert preferred_execution_tool("full", True) is None
    assert preferred_execution_tool("full", False) is None


def test_future_covariates_compile_to_each_targets_compressed_axis():
    run = object.__new__(mcp_agent._Run)
    run.channels = {"a": [1.0, 3.0], "b": [4.0, 5.0, 6.0]}
    run.epoch = mcp_agent.EPOCH
    row = {"input": {
        "history": {
            "a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0],
            "time_position_in_day": [10.0, 20.0, 30.0],
        },
        "future_covariates": {"time_position_in_day": [40.0, 50.0]},
    }}
    arguments = run._row_covariates(row)
    rows = arguments["covariates"]
    by_series = {name: [item for item in rows if item["series"] == name]
                 for name in ("a", "b")}
    assert [item["time_position_in_day"] for item in by_series["a"]] == [
        10.0, 30.0, 40.0, 50.0]
    assert [item["time_position_in_day"] for item in by_series["b"]] == [
        10.0, 20.0, 30.0, 40.0, 50.0]
    assert arguments["covariate_series_column"] == "series"
    assert arguments["covariate_mapping"][0]["type"] == "cyclic_1440"


def test_single_target_covariates_use_artifact_default_identity_and_declared_type():
    run = object.__new__(mcp_agent._Run)
    run.channels = {"value": [1.0, 2.0, 3.0]}
    run.epoch = mcp_agent.EPOCH
    arguments = run._row_covariates({"input": {
        "history": {"value": [1.0, 2.0, 3.0], "driver": [0.0, 1.0, 0.0]},
        "future_covariates": {"driver": [1.0, 1.0]},
        "covariate_mapping": [{"name": "driver", "type": "binary",
                               "availability": "future_known"}],
    }})
    assert {row["series"] for row in arguments["covariates"]} == {"__default__"}
    assert arguments["covariate_mapping"] == [{
        "name": "driver", "type": "binary", "availability": "future_known"}]


def test_context_compiler_grounds_quotes_and_rejects_invented_text():
    narrative = ("Event context:\nMedication adjusted at "
                 "2122-04-03T08:00:00.\nInput (JSON): {}")
    raw = {"events": [
        {"document_index": 0, "event_type": "medication_adjustment",
         "entity_scope": ["heart_rate"],
         "effective_start": "2122-04-03T08:00:00+00:00",
         "effective_end": "2122-04-03T09:00:00+00:00",
         "known_at": "2122-04-03T08:00:00+00:00",
         "evidence_quote": "Medication adjusted at 2122-04-03T08:00:00."},
        {"document_index": 0, "event_type": "invented",
         "entity_scope": ["heart_rate"],
         "effective_start": "2122-04-03T08:00:00+00:00",
         "effective_end": "2122-04-03T09:00:00+00:00",
         "known_at": "2122-04-03T08:00:00+00:00",
         "evidence_quote": "A sentence absent from the prompt."},
    ], "hypotheses": []}

    class Client:
        model = "test-model"

        def chat(self, messages, **kwargs):
            call = SimpleNamespace(
                id="context", type="function",
                function=SimpleNamespace(
                    name="submit_context", arguments=json.dumps(raw)))
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[call]))])

    result = compile_row_context({
        "id": "context-row", "tier": "T4", "prompt": narrative,
        "meta": {"target_keys": ["heart_rate"]},
    }, Client())
    assert result["attempted"] is True
    assert len(result["events"]) == 1
    assert len(result["rejected"]) == 1
    assert result["events"][0]["attributes"]["evidence_quote"] in narrative


def test_prevalidated_context_bypasses_provider_compilation():
    expected = {"events": [{"event_id": "event_1"}], "hypotheses": [],
                "rejected": [], "compiler_called": False,
                "receipt_reused": True}
    result = compile_row_context(
        {"_validated_context": expected},
        SimpleNamespace(chat=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("provider must not be called"))))
    assert result == {"attempted": True, **expected}


def test_context_compiler_receipt_is_replayed_without_another_model_call(tmp_path):
    narrative = ("Event context:\nMedication adjusted at "
                 "2122-04-03T08:00:00.\nInput (JSON): {}")
    raw = {"events": [{
        "document_index": 0, "event_type": "medication_adjustment",
        "entity_scope": ["heart_rate"],
        "effective_start": "2122-04-03T08:00:00+00:00",
        "effective_end": "2122-04-03T09:00:00+00:00",
        "known_at": "2122-04-03T08:00:00+00:00",
        "evidence_quote": "Medication adjusted at 2122-04-03T08:00:00.",
        "effect_family": "level_shift", "direction": "unknown",
        "duration": "temporary",
    }], "hypotheses": []}

    class Client:
        model = "test-model"
        calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            call = SimpleNamespace(
                id="context", type="function",
                function=SimpleNamespace(
                    name="submit_context", arguments=json.dumps(raw)))
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[call]))])

    row = {"id": "context-row", "tier": "T4", "prompt": narrative,
           "meta": {"target_keys": ["heart_rate"]}}
    client = Client()
    first = compile_row_context(row, client, str(tmp_path))
    replay = compile_row_context(row, client, str(tmp_path))
    assert client.calls == 1
    assert first["receipt_id"] == replay["receipt_id"]
    assert first["compiler_called"] is True
    assert replay["compiler_called"] is False
    assert replay["receipt_reused"] is True


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
        self.requests = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(self, messages, *, n=1, tools=None, tool_choice=None, **kwargs):
        self.requests.append({"messages": messages, "tools": tools,
                              "tool_choice": tool_choice})
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


def _factory(sessions=None):
    def factory(cwd):
        session = InProcessMcpSession(cwd)
        if sessions is not None:
            sessions.append(session)
        return session

    return factory


def _run(row, steps, tmp_path, sessions=None):
    return run_row(row, ScriptedClient(steps),
                   session_factory=_factory(sessions), work_dir=str(tmp_path))


def _run_mcq(row, steps, tmp_path, sessions=None):
    return mcq_row(row, ScriptedClient(steps),
                   session_factory=_factory(sessions), work_dir=str(tmp_path))


def test_mcq_row_accepts_mcp_timeout_for_t1_t3(tmp_path) -> None:
    """The common runner forwards this option for every tier."""
    outcome = mcq_row(
        _t1_row(), ScriptedClient([{"tool_calls": [("submit_answer", {
            "answers": {"trend": "upward", "volatility": "increased"},
            "reasoning": "bounded"
        })]}]), session_factory=_factory(), work_dir=str(tmp_path),
        mcp_call_timeout=12.0)
    assert "answer" in outcome


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


def test_artifact_binding_preserves_typed_primary_relationship(monkeypatch):
    run = object.__new__(mcp_agent._Run)
    run.artifact_paths = {"/sealed/artifact"}
    run.horizon = 1
    run._available_sensitivity = {}
    run._pending_support = {}
    run.covariate_execution = {}
    run.context_execution = {}
    artifact = {
        "task": {"schema": {"target_column": "value"}},
        "results": [{
            "series": "__default__",
            "forecast": [{"point": 10.0, "q50": 10.0}],
            "support": "supported",
            "context_outcome": {
                "status": "rejected",
                "events": ["structural-1"],
                "canonical_primary_preserved": True,
                "primary_forecast_changed": False,
                "automation_eligible": False,
                "relationship_to_primary": "no_distinct_numeric_path",
                "selected_output_role":
                    "primary_forecast_already_noncontinuing",
            },
        }],
    }
    import gnomon.artifacts
    monkeypatch.setattr(gnomon.artifacts, "read_artifact",
                        lambda _path: artifact)

    assert run._artifact_channel_rows(
        "/sealed/artifact", "value") == [10.0]
    execution = run.context_execution["value"]
    assert execution["relationship_to_primary"] == \
        "no_distinct_numeric_path"
    assert execution["selected_output_role"] == \
        "primary_forecast_already_noncontinuing"

    artifact["results"][0]["context_outcome"] = {
        "status": "partially_represented",
        "events": ["structural-1"],
        "dispositions": [{
            "context_id": "structural-1", "disposition": "scenario"}],
        "canonical_primary_preserved": True,
        "primary_forecast_changed": False,
        "automation_eligible": False,
    }
    run.context_execution = {}
    assert run._artifact_channel_rows(
        "/sealed/artifact", "value") == [10.0]
    assert run.context_execution["value"]["status"] == \
        "partially_represented"
    assert run.context_execution["value"]["scenario_only"] == 1


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


def test_host_compiled_forecast_cannot_be_bypassed_by_direct_submission(
    tmp_path,
):
    """Providers may ignore forced tool_choice; the harness must not.

    A direct fallback is never allowed in a product-contract run; otherwise
    the measured product arm is secretly the baseline arm.
    """
    row = _row(sparse_temp=False)
    row["_host_compiled_forecast"] = True

    def bypass(_messages):
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"values": VALUES},
                         "spo2": {"values": VALUES}},
            "mcq": {},
        })]}

    def recover(messages):
        payload = _last_tool_payload(messages)
        assert payload["accepted"] is False
        assert payload["problems"] == [
            "host_execution_required: submit a Gnomon artifact or abstain; "
            "model-authored forecast values bypass the "
            "product contract for channel(s): hr, spo2"
        ]
        return {"tool_calls": [_forecast_call(messages, "hr,spo2")]}

    def submit(messages):
        payload = _last_tool_payload(messages)
        path = payload["artifact_path"]
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"artifact_path": path},
                         "spo2": {"artifact_path": path}},
            "mcq": {},
        })]}

    outcome = _run(row, [bypass, recover, submit], tmp_path)
    assert outcome["channel_route"] == {"hr": "gnomon", "spo2": "gnomon"}
    assert outcome["mcp"]["calls"] == 1
    assert outcome["mcp"]["tool_sequence"][0]["tool"] == "submit_answer"
    assert outcome["mcp"]["tool_sequence"][0]["submit_rejected"]


def test_host_compiled_execution_tool_has_no_model_authored_arguments(tmp_path):
    row = _row(sparse_temp=False)
    row["_host_compiled_forecast"] = True
    row["_require_gnomon_execution"] = True
    client = ScriptedClient([
        {"tool_calls": [("gnomon_forecast", {
            "input": "/outside/ignored.csv",
            "context_events": [{"invented": "x" * 10_000}],
        })]},
        lambda messages: {"tool_calls": [("submit_answer", {
            "forecast": {key: {"artifact_path": _last_tool_payload(
                messages)["artifact_path"]} for key in ("hr", "spo2")},
        })]},
    ])
    outcome = run_row(
        row, client, session_factory=_factory(), work_dir=str(tmp_path),
        profile="core")
    first_tools = client.requests[0]["tools"]
    forecast = next(spec for spec in first_tools
                    if spec["function"]["name"] == "gnomon_forecast")
    assert forecast["function"]["parameters"] == {
        "type": "object", "additionalProperties": False}
    assert outcome["channel_route"] == {"hr": "gnomon", "spo2": "gnomon"}
    assert outcome["mcp"]["calls"] == 1
    assert len(client.requests) == 1
    assert outcome["mcp"]["tool_sequence"][-1] == {
        "host_submission": "complete_artifact"}


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


# -- caps end the run without discarding what it produced -------------------

def test_rounds_cap_abstains_every_channel(tmp_path):
    steps = [{"content": "Working on it."}, {"content": "Still thinking."},
             {"content": "Still nothing."}]  # the last call, unanswered
    outcome = _run(_row(sparse_temp=False), steps, tmp_path)
    assert outcome["answer"]["forecast"] == {}
    assert set(outcome["channel_route"].values()) == {"abstain"}
    assert all("no submission" in reason for reason in outcome["abstained"])
    # The row is marked as one the harness voided, so the runner keeps it
    # out of the accuracy denominators instead of scoring it zero.
    assert "no submission" in outcome["row_abstained"]


def test_token_cap_keeps_the_answer_the_last_call_produces(tmp_path):
    """The measured failure this exists for: a run that had done the work
    was voided by the cap check that ran before its submission."""
    steps = [
        {"content": "thinking", "bump_tokens": 600_000},
        {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"values": VALUES},
                         "spo2": {"values": VALUES}},
            "mcq": {"q1": "Higher"},
        })]},
    ]
    outcome = _run(_row(sparse_temp=False), steps, tmp_path)
    assert outcome["answer"]["forecast"]["hr"] == VALUES
    assert outcome["answer"]["mcq"] == {"q1": "Higher"}
    assert "cap:tokens" in outcome["last_call"]
    assert "row_abstained" not in outcome


def test_token_cap_abstains_when_the_last_call_answers_in_prose(tmp_path):
    """Prose twice becomes a typed, scoreable abstention per channel."""
    steps = [{"content": "thinking", "bump_tokens": 600_000},
             {"content": "I will keep thinking."},
             {"content": "Still prose, sorry."}]
    outcome = _run(_row(sparse_temp=False), steps, tmp_path)
    assert set(outcome["channel_route"].values()) == {"abstain"}
    assert "cap:tokens" in outcome["last_call"]
    assert "row_abstained" not in outcome
    assert any(entry.get("submission_fallback") == "typed_abstention"
               for entry in outcome["mcp"]["tool_sequence"])


def test_last_call_provider_failure_remains_retryable_infrastructure(tmp_path):
    def unavailable(_messages):
        raise OpenRouterError("provider timed out")

    with pytest.raises(OpenRouterError, match="provider timed out"):
        _run(_row(sparse_temp=False), [
            {"content": "thinking", "bump_tokens": 600_000}, unavailable,
        ], tmp_path)


# -- the envelope is repaired; the answer never is --------------------------

def _submitted(messages=None):
    return {"tool_calls": [("submit_answer", {
        "forecast": {"hr": {"values": VALUES}, "spo2": {"values": VALUES}},
        "mcq": {"q1": "Higher"},
    })]}


def test_double_encoded_forecast_is_repaired_and_accepted(tmp_path):
    """The measured failure: a complete answer, JSON-encoded into a
    string, thrown away as an abstention the model never made."""
    outcome = _run(_row(sparse_temp=False), [
        {"tool_calls": [("submit_answer", {
            "forecast": json.dumps(
                {"hr": {"values": VALUES}, "spo2": {"values": VALUES}}),
            "mcq": json.dumps({"q1": "Higher"}),
        })]},
    ], tmp_path)
    assert outcome["answer"]["forecast"]["hr"] == VALUES
    assert outcome["answer"]["mcq"] == {"q1": "Higher"}
    assert outcome["channel_route"] == {"hr": "direct", "spo2": "direct"}
    # The repair is disclosed, never silent.
    coerced = [entry["coerced"] for entry in outcome["mcp"]["tool_sequence"]
               if "coerced" in entry]
    assert coerced and set(coerced[0]) == {"forecast", "mcq"}


def test_double_encoded_channel_entry_is_repaired(tmp_path):
    outcome = _run(_row(sparse_temp=False), [
        {"tool_calls": [("submit_answer", {
            "forecast": {"hr": json.dumps({"values": VALUES}),
                         "spo2": {"values": VALUES}},
            "mcq": {},
        })]},
    ], tmp_path)
    assert outcome["answer"]["forecast"]["hr"] == VALUES
    coerced = [entry["coerced"] for entry in outcome["mcp"]["tool_sequence"]
               if "coerced" in entry]
    assert coerced and "forecast.hr" in coerced[0]


def test_a_string_that_is_not_json_is_left_for_the_validator(tmp_path):
    """Only the envelope may be repaired. Nonsense stays nonsense —
    the harness never invents the answer it wishes it had received."""
    arguments, repaired = mcp_agent.coerce_json_containers(
        {"forecast": "the heart rate goes up", "mcq": "42"})
    assert repaired == []
    assert arguments["forecast"] == "the heart rate goes up"
    # A JSON scalar is not a container either.
    assert arguments["mcq"] == "42"


def test_last_call_rejection_gets_one_repair_round(tmp_path):
    """A rejection at the last call used to die with the correct repair
    message beside it and no round left to use it."""
    steps = [
        {"content": "thinking", "bump_tokens": 600_000},
        {"tool_calls": [("submit_answer", {
            "forecast": {"not_a_channel": {"values": VALUES}}, "mcq": {},
        })]},
        _submitted,
    ]
    outcome = _run(_row(sparse_temp=False), steps, tmp_path)
    assert outcome["answer"]["forecast"]["hr"] == VALUES
    assert "cap:tokens" in outcome["last_call"]
    assert "row_abstained" not in outcome
    assert any("last_call_repair" in entry
               for entry in outcome["mcp"]["tool_sequence"])


def test_last_call_prose_gets_one_repair_round(tmp_path):
    steps = [{"content": "thinking", "bump_tokens": 600_000},
             {"content": "The heart rate will hold near 97."},
             _submitted]
    outcome = _run(_row(sparse_temp=False), steps, tmp_path)
    assert outcome["answer"]["forecast"]["hr"] == VALUES
    assert "row_abstained" not in outcome


def test_the_last_call_repair_is_not_unlimited(tmp_path):
    """Two attempts, not a retry loop: a model that will not produce a
    submission must still end as an honest abstention."""
    bad = {"tool_calls": [("submit_answer", {
        "forecast": {"not_a_channel": {"values": VALUES}}, "mcq": {},
    })]}
    outcome = _run(_row(sparse_temp=False),
                   [{"content": "x", "bump_tokens": 600_000}, bad, bad],
                   tmp_path)
    assert outcome["answer"]["forecast"] == {}
    assert set(outcome["channel_route"].values()) == {"abstain"}
    assert "cap:tokens" in outcome["last_call"]
    assert "row_abstained" not in outcome


def test_last_call_offers_only_the_submit_tool(tmp_path):
    """Nothing more may be computed on a spent budget — but the model
    must still be able to hand over what it already has."""
    offered = []

    def spend(messages):
        return {"content": "thinking", "bump_tokens": 600_000}

    def record(messages):
        return {"content": "no"}

    client = ScriptedClient([spend, record])
    original = client.chat

    def chat(messages, *, n=1, tools=None, tool_choice=None):
        offered.append([tool["function"]["name"] for tool in tools or []])
        return original(messages, n=n, tools=tools, tool_choice=tool_choice)

    client.chat = chat
    run_row(_row(sparse_temp=False), client,
            session_factory=_factory(), work_dir=str(tmp_path))
    assert "gnomon_forecast" in offered[0]
    assert offered[-1] == ["submit_answer"]


def test_spent_tool_budget_does_not_void_the_row(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_agent, "MAX_MCP_CALLS", 1)

    def first_call(messages):
        return {"tool_calls": [_forecast_call(messages, "hr")]}

    def second_call(messages):
        return {"tool_calls": [_forecast_call(messages, "spo2")]}

    def submit(messages):
        payload = _last_tool_payload(messages)
        assert payload["code"] == "TOOL_BUDGET_SPENT"
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"values": VALUES},
                         "spo2": {"values": VALUES}}, "mcq": {},
        })]}

    outcome = _run(_row(sparse_temp=False),
                   [first_call, second_call, submit], tmp_path)
    assert outcome["answer"]["forecast"]["spo2"] == VALUES
    assert outcome["mcp"]["calls"] == 1
    assert outcome["mcp"]["schema_bytes"] > 0
    assert outcome["mcp"]["product_schema_bytes"] > 0
    assert outcome["mcp"]["harness_schema_bytes"] > 0
    assert outcome["mcp"]["schema_bytes"] <= (
        outcome["mcp"]["product_schema_bytes"] +
        outcome["mcp"]["harness_schema_bytes"])


def test_product_contract_submit_schema_cannot_spend_tokens_on_reasoning():
    run = object.__new__(mcp_agent._Run)
    run.row = {"_require_gnomon_execution": True}
    parameters = run._submit_tool()["function"]["parameters"]
    assert "reasoning" not in parameters["properties"]
    assert parameters["required"] == ["forecast"]
    assert parameters["additionalProperties"] is False
    assert "reasoning" in mcp_agent.SUBMIT_TOOL["function"]["parameters"][
        "properties"]


def test_context_contract_requires_bounded_typed_gate_citations():
    run = object.__new__(mcp_agent._Run)
    run.row = {"_require_gnomon_execution": True,
               "_require_context_explanation": True}
    run.temporal_compilation = {}

    parameters = run._submit_tool()["function"]["parameters"]

    assert parameters["required"] == [
        "forecast", "reasoning", "cited_context_gate_codes",
        "context_automation_eligible", "canonical_primary_preserved",
        "cited_scenario_consequences"]
    citations = parameters["properties"]["cited_context_gate_codes"]
    assert citations["maxItems"] == 8
    assert citations["items"] == {"type": "string"}
    assert "disposition is rejected or scenario" in citations["description"]
    assert "Do not cite successful" in citations["description"]
    assert parameters["properties"]["reasoning"]["maxLength"] == 600
    assert parameters["properties"]["context_automation_eligible"] == {
        "type": "boolean",
        "description": (
            "Copy context_outcome.automation_eligible exactly. False means "
            "context evidence alone cannot authorize automation; do not "
            "weaken it to 'not requested'."),
    }
    assert parameters["properties"]["canonical_primary_preserved"][
        "type"] == "boolean"


def test_context_authority_omission_gets_one_artifact_reuse_repair():
    run = object.__new__(mcp_agent._Run)
    run.row = {"_require_gnomon_execution": True,
               "_require_context_explanation": True}
    run.target_keys = ["value"]
    run.horizon = 1
    run.submission = None
    run.mcp_calls = 1
    run.trace = []
    run.artifact_paths = set()
    run.context_execution = {}
    run._project_receipt_choices = lambda: {}

    def artifact_rows(path, channel):
        run._pending_support[channel] = "supported"
        run.context_execution[channel] = {
            "automation_eligible": False,
            "canonical_primary_preserved": True,
            "rejection_codes": [],
        }
        return [10.0]

    run._artifact_channel_rows = artifact_rows
    base = {
        "forecast": {"value": {"artifact_path": "/sealed/artifact.json"}},
        "cited_context_gate_codes": [],
        "context_automation_eligible": False,
        "canonical_primary_preserved": True,
        "cited_scenario_consequences": [],
    }

    rejected = run._handle_submit({
        **base, "reasoning": "The primary forecast is available."})
    assert rejected["accepted"] is False
    assert rejected["ready_to_retry"] == {
        "tool": "submit_answer", "reuse_forecast_artifact": True,
        "rerun_gnomon": False}
    assert any("context_authority_omitted" in problem
               for problem in rejected["problems"])
    assert run.submission is None

    misattributed = run._handle_submit({
        **base,
        "reasoning": ("The canonical primary remains preserved. Automation "
                      "is not eligible because it was not requested."),
    })
    assert misattributed["accepted"] is False
    assert any("context_authority_misattributed" in problem
               for problem in misattributed["problems"])
    assert run.submission is None

    both_true = run._handle_submit({
        **base,
        "reasoning": (
            "Automation was not requested; separately, context evidence "
            "alone cannot authorize automation. The canonical primary "
            "remains preserved."),
    })
    assert both_true["accepted"] is True
    run.submission = None

    accepted = run._handle_submit({
        **base,
        "reasoning": ("The canonical primary remains preserved; context "
                      "evidence alone cannot authorize automation."),
    })
    assert accepted["accepted"] is True


def test_context_scenario_consequence_requires_exact_artifact_reuse_repair():
    run = object.__new__(mcp_agent._Run)
    run.row = {"_require_gnomon_execution": True,
               "_require_context_explanation": True}
    run.target_keys = ["value"]
    run.horizon = 1
    run.submission = None
    run.mcp_calls = 1
    run.trace = []
    run.artifact_paths = set()
    run.context_execution = {}
    run._project_receipt_choices = lambda: {}
    summary = (
        "Conditional scenario q50 is 10 at 2026-01-01 and 12 at "
        "2026-01-02 (delta 2); the canonical primary remains unchanged.")

    def artifact_rows(path, channel):
        run._pending_support[channel] = "supported"
        run.context_execution[channel] = {
            "automation_eligible": False,
            "canonical_primary_preserved": True,
            "rejection_codes": [],
            "scenario_consequence_summaries": [summary],
        }
        return [10.0]

    run._artifact_channel_rows = artifact_rows
    base = {
        "forecast": {"value": {"artifact_path": "/sealed/artifact.json"}},
        "reasoning": ("The canonical primary remains preserved and context "
                      "cannot authorize automation."),
        "cited_context_gate_codes": [],
        "context_automation_eligible": False,
        "canonical_primary_preserved": True,
    }

    rejected = run._handle_submit({
        **base, "cited_scenario_consequences": []})
    assert rejected["accepted"] is False
    assert any("scenario_consequence_omitted" in problem
               for problem in rejected["problems"])
    assert run.submission is None

    accepted = run._handle_submit({
        **base, "cited_scenario_consequences": [summary],
        "reasoning": (base["reasoning"] +
                      " The conditional scenario runs from q50 10 to 12."),
    })
    assert accepted["accepted"] is True
    assert run.submission["context_consequence_projection"]["matched"] == [
        summary]


def test_context_scenario_cannot_be_described_as_admitted():
    run = object.__new__(mcp_agent._Run)
    run.row = {"_require_gnomon_execution": True,
               "_require_context_explanation": True}
    run.target_keys = ["value"]
    run.horizon = 1
    run.submission = None
    run.mcp_calls = 1
    run.trace = []
    run.artifact_paths = set()
    run.context_execution = {}
    run._project_receipt_choices = lambda: {}

    def artifact_rows(path, channel):
        run._pending_support[channel] = "supported"
        run.context_execution[channel] = {
            "admitted": 0,
            "applied": 0,
            "scenario_only": 1,
            "automation_eligible": False,
            "canonical_primary_preserved": True,
            "rejection_codes": [],
            "scenario_consequence_summaries": [],
        }
        return [10.0]

    run._artifact_channel_rows = artifact_rows
    base = {
        "forecast": {"value": {"artifact_path": "/sealed/artifact.json"}},
        "cited_context_gate_codes": [],
        "context_automation_eligible": False,
        "canonical_primary_preserved": True,
        "cited_scenario_consequences": [],
    }

    rejected = run._handle_submit({
        **base,
        "reasoning": ("The context event was admitted as a scenario. The "
                      "canonical primary remains preserved and context "
                      "evidence cannot authorize automation."),
    })
    assert rejected["accepted"] is False
    assert any("scenario_admission_conflated" in problem
               for problem in rejected["problems"])
    assert run.submission is None

    accepted = run._handle_submit({
        **base,
        "reasoning": ("The context event was represented as a scenario, not "
                      "admitted to the numeric forecast. The canonical "
                      "primary remains preserved and context evidence cannot "
                      "authorize automation."),
    })
    assert accepted["accepted"] is True


def test_context_gate_citation_must_be_visible_in_reasoning():
    run = object.__new__(mcp_agent._Run)
    run.row = {"_require_gnomon_execution": True,
               "_require_context_explanation": True}
    run.target_keys = ["value"]
    run.horizon = 1
    run.submission = None
    run.mcp_calls = 1
    run.trace = []
    run.artifact_paths = set()
    run.context_execution = {}
    run._project_receipt_choices = lambda: {}

    def artifact_rows(path, channel):
        run._pending_support[channel] = "supported"
        run.context_execution[channel] = {
            "automation_eligible": False,
            "canonical_primary_preserved": True,
            "rejection_codes": ["separated_model_folds_available"],
            "scenario_consequence_summaries": [],
        }
        return [10.0]

    run._artifact_channel_rows = artifact_rows
    base = {
        "forecast": {"value": {"artifact_path": "/sealed/artifact.json"}},
        "cited_context_gate_codes": ["separated_model_folds_available"],
        "context_automation_eligible": False,
        "canonical_primary_preserved": True,
        "cited_scenario_consequences": [],
    }
    rejected = run._handle_submit({
        **base,
        "reasoning": ("The canonical primary remains preserved and context "
                      "evidence cannot authorize automation."),
    })
    assert rejected["accepted"] is False
    assert any("context_gate_not_human_visible" in problem
               for problem in rejected["problems"])
    assert run.submission is None

    accepted = run._handle_submit({
        **base,
        "reasoning": ("The canonical primary remains preserved and context "
                      "evidence cannot authorize automation. The "
                      "separated_model_folds_available gate failed, so the "
                      "structural effect lacks separated evaluations."),
    })
    assert accepted["accepted"] is True


def test_typed_questions_require_explicit_synthesis_and_basis_maps():
    run = object.__new__(mcp_agent._Run)
    run.row = {"_require_gnomon_execution": True}
    run.temporal_compilation = {"questions": [{"id": "q1"}]}
    parameters = run._submit_tool()["function"]["parameters"]
    assert parameters["required"] == ["forecast", "mcq", "choice_basis"]


def test_natural_routing_still_requires_product_execution(tmp_path):
    run = object.__new__(mcp_agent._Run)
    run.row = {"_require_gnomon_execution": True}
    # A prior inspect/forecast call does not authorize replacing Gnomon's
    # published trajectory with model-authored numbers.
    run.mcp_calls = 1
    run.target_keys = ["hr", "spo2"]
    run.horizon = len(VALUES)
    run.submission = None
    result = run._handle_submit({"forecast": {
        "hr": {"values": VALUES}, "spo2": {"values": VALUES}}})
    assert result["accepted"] is False
    assert "host_execution_required" in result["problems"][0]


def test_evidence_profile_cannot_publish_informed_direct_values(tmp_path):
    """The governed profile enforces immutability without a private flag."""
    client = ScriptedClient([
        {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"values": VALUES},
                         "spo2": {"values": VALUES}},
        })]},
        {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"abstain": True},
                         "spo2": {"abstain": True}},
        })]},
    ])
    outcome = run_row(
        _row(sparse_temp=False), client, session_factory=_factory(),
        work_dir=str(tmp_path), profile="evidence")
    first = outcome["mcp"]["tool_sequence"][0]
    assert first["submit_rejected"]
    assert "host_execution_required" in first["submit_rejected"][0]
    assert set(outcome["channel_route"].values()) == {"abstain"}


def test_complete_artifact_survives_final_submission_format_failure(tmp_path):
    def forecast(messages):
        return {"tool_calls": [_forecast_call(messages, "hr,spo2")]}

    outcome = _run(_row(sparse_temp=False), [
        forecast,
        {"content": "I would submit the artifact."},
        {"content": "Still prose."},
    ], tmp_path)
    assert outcome["channel_route"] == {"hr": "gnomon", "spo2": "gnomon"}
    assert "row_abstained" not in outcome
    assert {entry.get("submission_fallback") for entry in
            outcome["mcp"]["tool_sequence"]} == {None, "complete_artifact"}


def test_single_target_default_artifact_closes_engine_browsing(tmp_path):
    row = _row(sparse_temp=False)
    row["meta"]["target_keys"] = ["hr"]
    row["input"]["history"] = {"hr": row["input"]["history"]["hr"]}
    row["ground_truth"] = {"hr": row["ground_truth"]["hr"]}

    def forecast(messages):
        return {"tool_calls": [_forecast_call(messages, "hr")]}

    outcome = _run(row, [forecast, {"content": "prose"},
                         {"content": "still prose"}], tmp_path)
    assert outcome["mcp"]["calls"] == 1
    assert any(entry.get("last_call") ==
               "forecast artifact ready; engine browsing closed"
               for entry in outcome["mcp"]["tool_sequence"])
    assert outcome["channel_route"]["hr"] == "gnomon"


# -- one batched call for every channel -------------------------------------

def test_one_multi_target_artifact_serves_each_of_its_channels(tmp_path):
    """`target_column: "hr,spo2"` is one run, one artifact, one result
    per channel — and each channel must bind to its OWN result."""
    seen = {}

    def batched(messages):
        return {"tool_calls": [_forecast_call(messages, "hr,spo2")]}

    def submit(messages):
        payload = _last_tool_payload(messages)
        seen["path"] = path = payload["artifact_path"]
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"artifact_path": path},
                         "spo2": {"artifact_path": path}},
            "mcq": {"q1": "Higher"},
        })]}

    outcome = _run(_row(sparse_temp=False), [batched, submit], tmp_path)
    assert outcome["channel_route"] == {"hr": "gnomon", "spo2": "gnomon"}
    assert outcome["mcp"]["calls"] == 1  # one call, not one per channel

    from gnomon.artifacts import read_artifact

    artifact = read_artifact(seen["path"])
    by_series = {item["series"]: [float(row["q50"])
                                  for row in item["forecast"]]
                 for item in artifact["results"]}
    assert outcome["answer"]["forecast"]["hr"] == by_series["hr"]
    assert outcome["answer"]["forecast"]["spo2"] == by_series["spo2"]
    assert by_series["hr"] != by_series["spo2"]  # not results[0] twice


def test_string_mcq_is_rejected_for_repair_instead_of_crashing(tmp_path):
    seen = {}

    def forecast(messages):
        return {"tool_calls": [_forecast_call(messages, "hr,spo2")]}

    def malformed_submit(messages):
        payload = _last_tool_payload(messages)
        seen["path"] = payload["artifact_path"]
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"artifact_path": seen["path"]},
                         "spo2": {"artifact_path": seen["path"]}},
            "mcq": "Higher",
        })]}

    def repaired_submit(messages):
        payload = _last_tool_payload(messages)
        assert payload["accepted"] is False
        assert payload["problems"] == [
            "mcq must be an object mapping question ids to answers"]
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"artifact_path": seen["path"]},
                         "spo2": {"artifact_path": seen["path"]}},
            "mcq": {"q1": "Higher"},
        })]}

    outcome = _run(_row(sparse_temp=False),
                   [forecast, malformed_submit, repaired_submit], tmp_path)
    assert outcome["channel_route"] == {"hr": "gnomon", "spo2": "gnomon"}


def test_multi_target_artifact_rejected_for_a_channel_it_skipped(tmp_path):
    def batched(messages):
        return {"tool_calls": [_forecast_call(messages, "hr,spo2")]}

    def mislabel(messages):
        payload = _last_tool_payload(messages)
        return {"tool_calls": [("submit_answer", {
            "forecast": {"temperature_c":
                         {"artifact_path": payload["artifact_path"]}},
        })]}

    def recover(messages):
        payload = _last_tool_payload(messages)
        assert payload["accepted"] is False
        assert any("'hr', 'spo2'" in problem
                   for problem in payload["problems"])
        return {"tool_calls": [("submit_answer", {
            "forecast": {"temperature_c": {"abstain": True}}, "mcq": {},
        })]}

    outcome = _run(_row(), [batched, mislabel, recover], tmp_path)
    assert outcome["channel_route"]["temperature_c"] == "abstain"


def test_system_prompt_names_the_jail_and_the_batched_call(tmp_path):
    """Two round-trips the first measured sweep spent on discovery: the
    run directory (learned by rejection) and the comma list (six calls
    where one would do)."""
    seen = {}

    def capture(messages):
        seen["system"] = messages[0]["content"]
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"values": VALUES},
                         "spo2": {"values": VALUES}}, "mcq": {},
        })]}

    _run(_row(sparse_temp=False), [capture], tmp_path)
    jail = str(Path(_csv_path([{"content": seen["system"]}])).parent)
    assert f"only read and write inside {jail}" in seen["system"]
    assert f"output_dir={jail}/gnomon-output" in seen["system"]
    assert '"hr,spo2"' in seen["system"]


def test_evidence_profile_uses_lossless_long_panel_for_sparse_channels(tmp_path):
    seen = {}

    def forecast(messages):
        seen["system"] = messages[0]["content"]
        csv_path = Path(_csv_path(messages))
        seen["header"] = csv_path.read_text().splitlines()[0]
        return {"tool_calls": [_forecast_call(messages, "hr,spo2")]}

    def submit(messages):
        payload = _last_tool_payload(messages)
        path = payload["artifact_path"]
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"artifact_path": path},
                         "spo2": {"artifact_path": path},
                         "temperature_c": {"artifact_path": path}},
            "mcq": {"q1": "Higher"},
        })]}

    outcome = run_row(
        _row(sparse_temp=True), ScriptedClient([forecast, submit]),
        session_factory=_factory(), work_dir=str(tmp_path), profile="evidence")

    assert seen["header"] == "timestamp,series,value"
    assert "series_column is `series`" in seen["system"]
    assert outcome["channel_route"] == {
        "hr": "gnomon", "spo2": "gnomon", "temperature_c": "gnomon"}
    assert outcome["mcp"]["calls"] == 1


# -- tool results are bounded without losing their disclosures --------------

def test_short_tool_results_pass_through_verbatim():
    text = json.dumps({"status": "ok", "results": [{"forecast": [1, 2, 3]}]})
    assert bounded_tool_text(text, limit=10_000) == (text, False)


def test_long_tool_results_shrink_bulk_and_keep_every_disclosure():
    payload = {
        "status": "complete",
        "artifact_path": "/run/out",
        "results": [{
            "series": "hr", "support": "best_effort",
            "selected_model": "seasonal_naive",
            "warnings": ["NO RELIABLE FORECAST: " + "w" * 200] * 6,
            "support_assessment": {"status": "unsupported",
                                   "recovery": ["extend the history"]},
            "forecast": [{"timestamp": f"t{i}", "q50": i, "q10": i,
                          "q90": i, "point": i} for i in range(400)],
        }],
    }
    text = json.dumps(payload)
    bounded, truncated = bounded_tool_text(text, limit=4_000)
    assert truncated and len(bounded) <= 4_000
    result = json.loads(bounded)["results"][0]
    # Bulk shrunk, and visibly so — never a short list posing as complete.
    assert result["forecast"]["harness_truncated"] is True
    assert result["forecast"]["items_total"] == 400
    assert result["forecast"]["head"][0]["q50"] == 0
    assert result["forecast"]["tail"][-1]["q50"] == 399
    # The shortening is the harness's doing, so its consequence is the
    # harness's to disclose: the measured failure was a model that read
    # "truncated", assumed the numbers were lost, and burned its
    # remaining rounds re-inspecting the file to rebuild them.
    remedy = result["forecast"]["remedy"]
    assert "artifact_path" in remedy and "do NOT need" in remedy
    # Epistemics verbatim: all six warnings, the assessment, the support
    # label, and the path to the complete numbers.
    assert result["warnings"] == payload["results"][0]["warnings"]
    assert result["support_assessment"] == \
        payload["results"][0]["support_assessment"]
    assert result["support"] == "best_effort"
    assert json.loads(bounded)["artifact_path"] == "/run/out"


def test_a_hard_squeeze_stays_parseable_and_keeps_the_disclosures():
    """Cutting the serialized text would be shorter code and a worse
    answer: unparseable JSON, with the artifact_path the model needs to
    submit possibly inside the part that was cut."""
    payload = {
        "status": "complete",
        "artifact_path": "/run/out",
        "results": [{
            "series": name, "support": "supported",
            "warnings": [], "support_assessment": {"status": "ok"},
            "forecast": [{"timestamp": f"t{i}", "q50": i} for i in range(60)],
            "evidence": {"folds": [{"model": "x", "score": i}
                                   for i in range(40)]},
        } for name in ("hr", "spo2", "resp")],
    }
    bounded, truncated = bounded_tool_text(json.dumps(payload), limit=1_500)
    assert truncated
    parsed = json.loads(bounded)  # the point of the test
    assert parsed["artifact_path"] == "/run/out"
    # Every channel still there with its support label; only bulk left.
    assert [r["series"] for r in parsed["results"]] == ["hr", "spo2", "resp"]
    assert all(r["support"] == "supported" for r in parsed["results"])
    assert parsed["harness_dropped"]  # and it names what went
    assert all(path.startswith("results[") for path in parsed["harness_dropped"])


def test_non_json_tool_output_is_cut_with_the_cut_named():
    bounded, truncated = bounded_tool_text("x" * 500, limit=100)
    assert truncated and "cut at 100 characters of 500" in bounded


def test_a_truncated_result_is_recorded_in_the_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_agent, "MAX_TOOL_RESULT_CHARS", 200)

    def call_forecast(messages):
        return {"tool_calls": [_forecast_call(messages, "hr")]}

    def submit(messages):
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"values": VALUES},
                         "spo2": {"values": VALUES}}, "mcq": {},
        })]}

    outcome = _run(_row(sparse_temp=False), [call_forecast, submit], tmp_path)
    forecast_step, = [step for step in outcome["mcp"]["tool_sequence"]
                      if step["tool"] == "gnomon_forecast"]
    assert forecast_step["truncated"] is True


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


# -- T1/T3: the tier's own answer shape ------------------------------------

def _t1_row() -> dict:
    return {
        "id": "tb-mcp-t1", "tier": "T1", "source_dataset": "MIMIC",
        "prompt": 'Describe the history. Input (JSON): {"hr": [70, 71, 72, '
                  '73, 74, 75]}\nAnswer trend and volatility.',
        "labels": {"trend": "upward", "volatility": "increased"},
    }


def _t3_row() -> dict:
    return {
        "id": "tb-mcp-t3", "tier": "T3", "source_dataset": "MIMIC",
        "prompt": 'Answer the pack. Input (JSON): {"hr": [70, 71, 72, 73]}',
        "pack": [{"question": "Is hr higher at the end?",
                  "options": ["Higher", "Lower"], "label": "Higher"},
                 {"question": "Is hr level stable?",
                  "options": ["Yes", "No"], "label": "No"}],
    }


def test_t1_answers_are_the_fields_the_scorer_reads(tmp_path):
    from benchmarks.temporalbench.scoring import score_t1

    outcome = _run_mcq(_t1_row(), [
        {"tool_calls": [("submit_answer", {
            "answers": {"trend": "Upward", "volatility": "increased"},
        })]},
    ], tmp_path)
    assert outcome["answer"] == {"trend": "Upward", "volatility": "increased"}
    assert outcome["abstained"] == []
    assert score_t1(_t1_row(), outcome["answer"])["correct"] == 2


def test_t3_answers_are_the_pack_list_in_order(tmp_path):
    from benchmarks.temporalbench.scoring import score_t3

    outcome = _run_mcq(_t3_row(), [
        {"tool_calls": [("submit_answer", {"answers": ["Higher", "No"]})]},
    ], tmp_path)
    assert outcome["answer"] == {"answers": ["Higher", "No"]}
    assert score_t3(_t3_row(), outcome["answer"]["answers"])["correct"] == 2


def test_evidence_t3_describe_uses_host_resolved_panel_binding(tmp_path):
    """The agent chooses the verb/questions, never the data schema."""
    row = _t3_row()

    def submit_after_describe(messages):
        payload = _last_tool_payload(messages)
        assert payload.get("status") != "error", payload
        return {"tool_calls": [("submit_answer", {
            "answers": ["Higher", "No"],
        })]}

    client = ScriptedClient([
        {"tool_calls": [("gnomon_describe", {
            "input": "/outside/invented.csv",
            "target_column": "invented",
            "frequency": "10min",
        })]},
        submit_after_describe,
    ])
    outcome = mcq_row(
        row, client, session_factory=_factory(), work_dir=str(tmp_path),
        profile="evidence")

    assert outcome["answer"] == {"answers": ["Higher", "No"]}
    assert outcome["mcp"]["tool_sequence"][0]["host_data_binding"] == \
        "long_panel"
    assert outcome["mcp"]["tool_sequence"][0]["is_error"] is False


def test_evidence_t3_attaches_compiled_pack_questions_to_describe(tmp_path):
    client = ScriptedClient([
        {"tool_calls": [("submit_temporal_intent", {
            "status": "compiled", "questions": [{
                "id": "q1", "verb": "describe", "property": "level",
                "target": "hr"}]})]},
        {"tool_calls": [("gnomon_describe", {})]},
        {"tool_calls": [("submit_answer", {
            "answers": ["Higher", "No"],
        })]},
    ])

    outcome = mcq_row(
        _t3_row(), client, session_factory=_factory(),
        work_dir=str(tmp_path), profile="evidence",
        compile_questions=True,
        question_receipts_dir=str(tmp_path / "question-receipts"))

    first = outcome["mcp"]["tool_sequence"][0]
    assert first["host_data_binding"] == "long_panel"
    # The deterministic property router restores the second explicit level
    # question omitted by the mocked semantic proposal.
    assert first["compiled_questions"] == 2
    assert first["is_error"] is False
    assert outcome["mcp"]["calls"] == 1
    receipts = outcome["mcp"]["temporal_answer_receipts"]
    assert len(receipts) == 1
    assert receipts[0]["source"] == "inline_describe"
    assert receipts[0]["primary_forecast_unchanged"] is None
    assert len(receipts[0]["answers"]) == 2
    assert len(client.requests) == 3  # compiler, describe, forced submission


def test_mcq_submit_schema_is_the_row_s_own_shape():
    from benchmarks.temporalbench.mcp_agent import mcq_submit_tool

    t1_tool, _ = mcq_submit_tool(_t1_row())
    answers = t1_tool["function"]["parameters"]["properties"]["answers"]
    assert sorted(answers["properties"]) == ["trend", "volatility"]
    assert sorted(answers["required"]) == ["trend", "volatility"]

    t3_tool, rule = mcq_submit_tool(_t3_row())
    answers = t3_tool["function"]["parameters"]["properties"]["answers"]
    assert answers["minItems"] == answers["maxItems"] == 2
    assert "2 questions" in rule


def test_mcq_tiers_hold_the_same_unpruned_tool_surface(tmp_path, monkeypatch):
    """The arm's question is whether tool access helps a question tier;
    withdrawing the tools would answer it by construction."""
    monkeypatch.setenv("GNOMON_MCP_PROFILE", "full")
    seen = {}

    def capture(messages):
        return {"tool_calls": [("submit_answer", {"answers": {
            "trend": "upward", "volatility": "increased"}})]}

    client = ScriptedClient([capture])
    original = client.chat

    def chat(messages, *, n=1, tools=None, tool_choice=None):
        seen["tools"] = [tool["function"]["name"] for tool in tools]
        seen["system"] = messages[0]["content"]
        return original(messages, n=n, tools=tools, tool_choice=tool_choice)

    client.chat = chat
    mcq_row(_t1_row(), client, session_factory=_factory(),
            work_dir=str(tmp_path))
    assert {"gnomon_forecast", "gnomon_detect_anomalies", "gnomon_inspect",
            "submit_answer"} <= set(seen["tools"])
    # ...but the forecast contract is absent rather than restated: this
    # tier has no horizon, no channels, and no artifact exits.
    assert "horizon" not in seen["system"]
    assert "artifact_path" not in seen["system"]
    assert "only read and write inside" in seen["system"]


def test_missing_mcq_fields_are_named_once_then_the_answer_stands(tmp_path):
    def partial(messages):
        return {"tool_calls": [("submit_answer",
                                {"answers": {"trend": "upward"}})]}

    def retry(messages):
        payload = _last_tool_payload(messages)
        assert payload["accepted"] is False
        assert "volatility" in payload["problems"][0]
        return {"tool_calls": [("submit_answer", {"answers": {
            "trend": "upward", "volatility": "decreased"}})]}

    outcome = _run_mcq(_t1_row(), [partial, retry], tmp_path)
    assert outcome["answer"]["volatility"] == "decreased"


def test_a_second_incomplete_mcq_submission_is_kept_not_voided(tmp_path):
    """A validator that keeps rejecting turns an answer the model did
    produce into an abstention — a worse report than a partial answer."""
    steps = [
        {"tool_calls": [("submit_answer", {"answers": {"trend": "upward"}})]},
        {"tool_calls": [("submit_answer", {"answers": {"trend": "upward"}})]},
    ]
    outcome = _run_mcq(_t1_row(), steps, tmp_path)
    assert outcome["answer"] == {"trend": "upward"}
    assert "volatility" in outcome["submit_problems"][0]
    assert "row_abstained" not in outcome


def test_mcq_row_that_never_submits_is_voided_not_scored_zero(tmp_path):
    steps = [{"content": "thinking"}, {"content": "still thinking"},
             {"content": "no answer"}]
    outcome = _run_mcq(_t1_row(), steps, tmp_path)
    assert outcome["answer"] == {}
    assert "no submission" in outcome["row_abstained"]


# -- run_temporalbench wiring ----------------------------------------------

def test_answer_row_dispatches_every_tier_to_its_mcp_path(monkeypatch):
    from benchmarks.temporalbench import run_temporalbench
    from benchmarks.temporalbench.mcp_agent import mcq_row as real_mcq

    calls = []
    monkeypatch.setattr(mcp_agent, "run_row",
                        lambda row, client: calls.append(("forecast", row)))
    monkeypatch.setattr(mcp_agent, "mcq_row",
                        lambda row, client: calls.append(("mcq", row)))
    for tier in ("T1", "T2", "T3", "T4"):
        run_temporalbench.answer_row({"tier": tier, "prompt": "x"},
                                     "gnomon-mcp", None)
    assert [kind for kind, _ in calls] == ["mcq", "forecast", "mcq",
                                           "forecast"]
    assert real_mcq is not None  # the tier restriction is gone, not moved


def test_answer_row_passes_the_experiment_profile(monkeypatch):
    from benchmarks.temporalbench import run_temporalbench

    seen = []
    monkeypatch.setattr(
        mcp_agent, "run_row",
        lambda row, client, **kwargs: seen.append(kwargs) or {},
    )
    run_temporalbench.answer_row(
        {"tier": "T2", "prompt": "x"}, "gnomon-mcp", None,
        mcp_profile="core", mcp_call_timeout=17,
    )
    assert seen == [{"profile": "core", "mcp_call_timeout": 17}]


def test_answer_row_passes_question_compiler_to_mcq_path(monkeypatch):
    """T3 question packs use the same host compiler as forecast MCQs."""
    from benchmarks.temporalbench import run_temporalbench

    seen = []
    monkeypatch.setattr(
        mcp_agent, "mcq_row",
        lambda row, client, **kwargs: seen.append(kwargs) or {},
    )
    run_temporalbench.answer_row(
        {"tier": "T1", "prompt": "x"}, "gnomon-mcp", None,
        mcp_profile="evidence", compile_context=True,
        context_receipts_dir="context", compile_questions=True,
        question_receipts_dir="questions", mcp_call_timeout=17,
        model_evidence_registry="registry.json",
    )
    assert seen == [{
        "profile": "evidence",
        "compile_context": True,
        "context_receipts_dir": "context",
        "mcp_call_timeout": 17,
        "compile_questions": True,
        "question_receipts_dir": "questions",
    }]


def test_mcp_condition_keeps_the_requested_tiers(tmp_path, monkeypatch):
    """`--condition gnomon-mcp --tiers T1,T2,T3,T4` must run all four:
    the arm is no longer T2/T4 by construction."""
    import benchmarks.temporalbench.run_temporalbench as runner

    seen = {}
    monkeypatch.setattr(runner, "load_official_metrics", lambda _dir: None)
    monkeypatch.setattr(runner, "OpenRouterClient",
                        lambda *a, **k: SimpleNamespace(
                            base_url="http://x", usage_summary={}))

    def iter_rows(_dir, *, tiers, datasets, limit):
        seen["tiers"] = tiers
        return iter(())

    monkeypatch.setattr(runner, "iter_rows", iter_rows)
    monkeypatch.setattr(sys, "argv", [
        "run_temporalbench", "--data-dir", str(tmp_path),
        "--condition", "gnomon-mcp", "--model", "x/y",
        "--tiers", "T1,T2,T3,T4", "--output-dir", str(tmp_path / "out"),
    ])
    assert runner.main() == 0
    assert seen["tiers"] == ("T1", "T2", "T3", "T4")


def test_fully_errored_run_is_diagnostic_not_success(tmp_path, monkeypatch):
    import benchmarks.temporalbench.run_temporalbench as runner

    monkeypatch.setattr(runner, "load_official_metrics", lambda _dir: None)
    monkeypatch.setattr(runner, "OpenRouterClient",
                        lambda *a, **k: SimpleNamespace(
                            base_url="http://x", usage_summary={}))
    monkeypatch.setattr(runner, "iter_rows", lambda _dir, **kwargs: iter([
        {"id": "bad", "tier": "T1", "prompt": "x", "labels": {}}
    ]))
    monkeypatch.setattr(
        runner, "answer_row",
        lambda *a, **k: (_ for _ in ()).throw(
            ValueError("cached temporal-intent receipt does not match input")),
    )
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "run_temporalbench", "--data-dir", str(tmp_path),
        "--condition", "gnomon-mcp", "--model", "x/y",
        "--tiers", "T1", "--output-dir", str(output),
    ])

    assert runner.main() == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["run_status"] == "failed"
    assert summary["terminal_error_breakdown"] == {
        "ValueError: cached temporal-intent receipt does not match input": 1}


def test_voided_rows_stay_out_of_the_accuracy_denominators(tmp_path,
                                                           monkeypatch):
    """A row the harness ended without an answer is not a wrong answer.
    Counting it as one reported a token cap as a 0% tier score."""
    import benchmarks.temporalbench.run_temporalbench as runner

    rows = [{"id": "answered", "tier": "T1", "prompt": "x",
             "labels": {"trend": "upward"}},
            {"id": "voided", "tier": "T1", "prompt": "x",
             "labels": {"trend": "upward"}}]
    outcomes = {
        "answered": {"answer": {"trend": "upward"}, "abstained": [],
                     "mcp": {"calls": 2, "run_tokens": 120,
                             "schema_bytes": 2000}},
        "voided": {"answer": {}, "abstained": ["cap:tokens exceeded"],
                   "row_abstained": "cap:tokens exceeded",
                   "mcp": {"calls": 4, "run_tokens": 280,
                           "schema_bytes": 2000}},
    }
    monkeypatch.setattr(runner, "load_official_metrics", lambda _dir: None)
    monkeypatch.setattr(runner, "OpenRouterClient",
                        lambda *a, **k: SimpleNamespace(
                            base_url="http://x", usage_summary={}))
    monkeypatch.setattr(runner, "iter_rows",
                        lambda _dir, **kwargs: iter(rows))
    monkeypatch.setattr(runner, "answer_row",
                        lambda row, *a, **k: outcomes[row["id"]])
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "run_temporalbench", "--data-dir", str(tmp_path),
        "--condition", "gnomon-mcp", "--model", "x/y",
        "--tiers", "T1", "--output-dir", str(output),
    ])
    assert runner.main() == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["rows"] == 2
    assert summary["rows_voided_by_harness"] == 1
    # One scored row, answered correctly — not 50% over a denominator
    # that includes a row nobody answered.
    assert summary["choice_rows_scored_by_tier"] == {"T1": 1}
    # ...and a question tier's voided row is not a forecast channel the
    # engine declined: that counter covers T2/T4 channels only.
    assert summary["forecast_channels_abstained"] == 0
    assert summary["choice_accuracy_by_tier_scored_only"] == {"T1": 1.0}
    assert summary["mcp_economics"] == {
        "cumulative_tokens": 400,
        "mean_tokens_per_attempted_row": 200.0,
            "calls_median": 4,
            "calls_p95": 4,
            "surface_required_calls_mean": 3.0,
            "redundant_calls_total": 0,
            "redundant_calls_mean": 0.0,
        "schema_bytes": [2000],
        "rows_answered": 1,
        "rows_attempted": 2,
        "answer_yield": 0.5,
    }
    records = [json.loads(line) for line in
               (output / "gnomonbench.jsonl").read_text().splitlines()]
    voided, = [r for r in records if r["task_id"] == "voided"]
    assert voided["choice_correct"] is None
    assert voided["row_abstained"] == "cap:tokens exceeded"
    assert voided["appropriate_abstention"] is True
    # Provenance: the endpoint that served the model, in the manifest.
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["base_url"] == "http://x"
    assert manifest["mcp_profile"] == "evidence"


def test_row_offset_makes_long_sweeps_shardable(tmp_path, monkeypatch):
    import benchmarks.temporalbench.run_temporalbench as runner

    rows = [{"id": f"row{i}", "tier": "T1", "prompt": "x",
             "labels": {"trend": "upward"}} for i in range(4)]
    seen = []
    requested = {}
    monkeypatch.setattr(runner, "load_official_metrics", lambda _dir: None)
    monkeypatch.setattr(runner, "OpenRouterClient",
                        lambda *a, **k: SimpleNamespace(
                            base_url="http://x", usage_summary={}))
    def rows_for_run(_dir, **kwargs):
        requested["limit"] = kwargs["limit"]
        return iter(rows[:kwargs["limit"]])
    monkeypatch.setattr(runner, "iter_rows", rows_for_run)
    monkeypatch.setattr(
        runner, "answer_row",
        lambda row, *a, **k: seen.append(row["id"]) or {
            "answer": {"trend": "upward"}, "abstained": []},
    )
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "run_temporalbench", "--data-dir", str(tmp_path),
        "--condition", "gnomon-mcp", "--model", "x/y", "--tiers", "T1",
        "--offset", "2", "--limit", "1", "--output-dir", str(output),
    ])
    assert runner.main() == 0
    assert requested["limit"] == 3
    assert seen == ["row2"]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["row_offset"] == 2
    assert summary["rows"] == 1


def test_surface_experiment_enforces_the_precommitted_call_ceiling():
    """Completed work gets a final submission round after four calls."""
    assert MAX_ROUNDS == 10
    assert mcp_agent.MAX_MCP_CALLS == 4
    assert cik_mcp_agent.MAX_ROUNDS == 10, (
        "CiK's cap moved; revisit the matched harness posture"
    )


def test_wall_clock_cap_ends_the_run_as_a_named_abstention(
    tmp_path, monkeypatch,
):
    """The arm had rounds/calls/token caps but no wall cap: a slow
    provider could park a row for hours without breaching anything."""
    monkeypatch.setattr(mcp_agent, "MAX_WALL_SECONDS", -1.0)
    # The breach fires before the first round; the last-call message is
    # answered in prose, so the row abstains with the cap named.
    outcome = _run(_row(), [{"content": "out of time"}], tmp_path)
    assert "cap:wall_clock" in outcome["row_abstained"]


# -- superseded results are compacted out of the running history ------------

def test_superseded_forecast_is_compacted_in_the_message_history(tmp_path):
    """A batched call retires the single-channel call it covers: the
    older tool message is replaced in place by its disclosures plus the
    artifact_path — the bulk stops being re-sent every remaining round,
    and what Gnomon said about its numbers does not."""
    seen = {}

    def forecast_hr(messages):
        return {"tool_calls": [_forecast_call(messages, "hr")]}

    def forecast_batch(messages):
        return {"tool_calls": [_forecast_call(messages, "hr,spo2")]}

    def submit(messages):
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_messages) == 2
        stub = json.loads(tool_messages[0]["content"])
        assert stub["harness_superseded"] is True
        assert stub["artifact_path"], "the path to the full numbers stays"
        result, = stub["results"]
        assert result["support"], "the support label stays"
        # The assessment is verbatim, or marked as riding character-
        # identical on the live batched result one message down.
        assert result.get("support_assessment") \
            or "support_assessment" in result.get("unchanged", "")
        assert "forecast" not in result, "the bulk goes"
        live = json.loads(tool_messages[1]["content"])
        assert "harness_superseded" not in live
        seen["artifact_path"] = live["artifact_path"]
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"artifact_path": live["artifact_path"]},
                         "spo2": {"artifact_path": live["artifact_path"]}},
            "mcq": {"q1": "Higher"},
        })]}

    outcome = _run(_row(sparse_temp=False),
                   [forecast_hr, forecast_batch, submit], tmp_path)
    assert outcome["channel_route"] == {"hr": "gnomon", "spo2": "gnomon"}
    # The adapter decision is disclosed in the trace.
    assert {"superseded": 1} in outcome["mcp"]["tool_sequence"]


def test_forecasts_of_different_channels_are_both_kept(tmp_path):
    def forecast_hr(messages):
        return {"tool_calls": [_forecast_call(messages, "hr")]}

    def forecast_spo2(messages):
        return {"tool_calls": [_forecast_call(messages, "spo2")]}

    def submit(messages):
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        payloads = [json.loads(m["content"]) for m in tool_messages]
        assert not any(p.get("harness_superseded") for p in payloads), \
            "different channels are parallel evidence, not a supersession"
        return {"tool_calls": [("submit_answer", {
            "forecast": {"hr": {"artifact_path": payloads[0]["artifact_path"]},
                         "spo2": {"artifact_path": payloads[1]["artifact_path"]}},
            "mcq": {"q1": "Higher"},
        })]}

    outcome = _run(_row(sparse_temp=False),
                   [forecast_hr, forecast_spo2, submit], tmp_path)
    assert outcome["channel_route"] == {"hr": "gnomon", "spo2": "gnomon"}


def test_host_projects_all_unambiguous_canonical_receipt_answers(tmp_path):
    from benchmarks.temporalbench.mcp_agent import _Run

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "temporal_answers.json").write_text(json.dumps({
        "primary_forecast_unchanged": True,
        "answers": [
            {"question": {"id": "q1"},
             "best_estimate": {"value": "higher",
                               "support": "supported",
                               "automation_eligible": True}},
            {"question": {"id": "q2"},
             "best_estimate": {"value": "stable",
                               "support": "supported",
                               "automation_eligible": True}},
            {"question": {"id": "q3"},
             "best_estimate": {"value": "weaker",
                               "support": "weak",
                               "automation_eligible": False}},
            {"question": {"id": "q4"},
             "best_estimate": {"value": "increased",
                               "support": "weak",
                               "automation_eligible": False},
             "answer": {"reasoning": {"adjudication": {
                 "relationship": "alternative_preferred",
                 "alternative": {"value": "decreased"},
                 "synthesis_eligibility": {"eligible": True}}}}},
        ],
    }), encoding="utf-8")
    run = _Run.__new__(_Run)
    run.artifact_paths = {str(artifact)}
    run.row = {"mcq": {
        "level": {"options": ["Higher", "Lower", "Similar", "Uncertain"]},
        "volatility": {"options": [
            "increased", "decreased", "constant", "Uncertain"]},
        "seasonality": {"options": ["fixed", "shifting", "no", "Uncertain"]},
        "weak_volatility": {"options": [
            "increased", "decreased", "constant", "Uncertain"]},
    }}

    projected = run._project_receipt_choices()

    assert projected["level"]["display_value"] == "Higher"
    assert projected["volatility"]["display_value"] == "constant"
    assert "seasonality" not in projected  # ``weaker`` is not ``no``.
    assert projected["weak_volatility"]["display_value"] == "increased"
    assert projected["weak_volatility"]["automation_eligible"] is False
    assert projected["level"]["authority"] == "binding"
    assert projected["weak_volatility"]["authority"] == "advisory"
    assert projected["weak_volatility"]["has_computed_opposition"] is True
    assert projected["weak_volatility"]["computed_alternative"] == "decreased"


def test_submission_binds_supported_but_preserves_weak_synthesis() -> None:
    from benchmarks.temporalbench.mcp_agent import _Run

    run = _Run.__new__(_Run)
    run.row = {"mcq": {"strong": {}, "weak": {}}}
    run.target_keys = ["x"]
    run.horizon = 2
    run.submission = None
    run.mcp_calls = 0
    run.trace = []
    run.artifact_paths = set()
    run._project_receipt_choices = lambda: {
        "strong": {"display_value": "Higher", "authority": "binding"},
        "weak": {"display_value": "increased", "authority": "advisory",
                 "has_computed_opposition": True,
                 "computed_alternative": "decreased"},
    }
    accepted = run._handle_submit({
        "forecast": {"x": {"values": [1.0, 2.0]}},
        "mcq": {"strong": "Lower", "weak": "decreased"},
        "choice_basis": {"weak": {
            "kind": "computed_opposition", "evidence": "observed disagrees"}},
    })
    assert accepted["accepted"] is True
    assert run.submission["mcq"] == {
        "strong": "Higher", "weak": "decreased"}
    assert run.submission["canonical_mcq"] == {
        "strong": "Higher", "weak": "increased"}
    assert run.submission["synthesized_mcq"] == {
        "strong": "Lower", "weak": "decreased"}
    assert run.submission["choice_authority"]["weak"] == "advisory_override"


def test_unsubstantiated_weak_override_keeps_canonical_default() -> None:
    from benchmarks.temporalbench.mcp_agent import _Run
    run = _Run.__new__(_Run)
    run.row = {"mcq": {"weak": {}}, "prompt": "No opposing statement."}
    run.target_keys = ["x"]
    run.horizon = 1
    run.submission = None
    run.mcp_calls = 0
    run.trace = []
    run.artifact_paths = set()
    run._project_receipt_choices = lambda: {
        "weak": {"display_value": "increased", "authority": "advisory",
                 "has_computed_opposition": False}}
    accepted = run._handle_submit({
        "forecast": {"x": {"values": [1.0]}},
        "mcq": {"weak": "decreased"},
    })
    assert accepted["accepted"] is True
    assert run.submission["mcq"]["weak"] == "increased"
    assert run.submission["synthesized_mcq"]["weak"] == "decreased"
    assert run.submission["choice_authority"]["weak"] == \
        "advisory_canonical_default"


def test_computed_opposition_only_authorizes_its_projected_alternative() -> None:
    from benchmarks.temporalbench.mcp_agent import _Run
    run = _Run.__new__(_Run)
    run.row = {"mcq": {"weak": {}}, "prompt": ""}
    run.target_keys = ["x"]
    run.horizon = 1
    run.submission = None
    run.mcp_calls = 0
    run.trace = []
    run.artifact_paths = set()
    run._project_receipt_choices = lambda: {
        "weak": {"display_value": "increased", "authority": "advisory",
                 "has_computed_opposition": True,
                 "computed_alternative": "decreased"}}
    accepted = run._handle_submit({
        "forecast": {"x": {"values": [1.0]}},
        "mcq": {"weak": "constant"},
        "choice_basis": {"weak": {
            "kind": "computed_opposition", "evidence": "two receipts"}},
    })
    assert accepted["accepted"] is True
    assert run.submission["mcq"]["weak"] == "increased"
    assert run.submission["choice_basis"] == {}


def test_exact_context_quote_needs_outcome_backed_adjudication() -> None:
    from benchmarks.temporalbench.mcp_agent import _Run
    run = _Run.__new__(_Run)
    quote = "A deploy may increase latency."
    run.row = {"mcq": {"weak": {}}, "prompt": quote}
    run.target_keys = ["x"]
    run.horizon = 1
    run.submission = None
    run.mcp_calls = 0
    run.trace = []
    run.artifact_paths = set()
    run._project_receipt_choices = lambda: {
        "weak": {"display_value": "stable", "authority": "advisory",
                 "has_computed_opposition": False,
                 "computed_alternative": None}}
    accepted = run._handle_submit({
        "forecast": {"x": {"values": [1.0]}},
        "mcq": {"weak": "increased"},
        "choice_basis": {"weak": {"kind": "task_context",
                                    "evidence": quote}},
    })
    assert accepted["accepted"] is True
    assert run.submission["mcq"]["weak"] == "stable"
    assert run.submission["synthesized_mcq"]["weak"] == "increased"


def test_validated_context_can_publish_exact_adjudicated_alternative() -> None:
    from benchmarks.temporalbench.mcp_agent import _Run
    run = _Run.__new__(_Run)
    quote = "Repeated scored deploys increased latency."
    run.row = {"mcq": {"weak": {}}, "prompt": quote}
    run.target_keys = ["x"]
    run.horizon = 1
    run.submission = None
    run.mcp_calls = 0
    run.trace = []
    run.artifact_paths = set()
    run._project_receipt_choices = lambda: {
        "weak": {"display_value": "stable", "authority": "advisory",
                 "has_computed_opposition": True,
                 "computed_alternative": "increased"}}
    accepted = run._handle_submit({
        "forecast": {"x": {"values": [1.0]}},
        "mcq": {"weak": "increased"},
        "choice_basis": {"weak": {"kind": "task_context",
                                    "evidence": quote}},
    })
    assert accepted["accepted"] is True
    assert run.submission["mcq"]["weak"] == "increased"
    assert run.submission["choice_authority"]["weak"] == "advisory_override"
