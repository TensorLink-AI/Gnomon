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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik import mcp_agent as cik_mcp_agent
from benchmarks.cik.mcp_agent import InProcessMcpSession
from benchmarks.temporalbench import mcp_agent
from benchmarks.temporalbench.mcp_agent import (
    MAX_ROUNDS,
    bounded_tool_text,
    mcq_row,
    run_row,
)


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
    """Prose twice — the repair round offered and refused — abstains."""
    steps = [{"content": "thinking", "bump_tokens": 600_000},
             {"content": "I will keep thinking."},
             {"content": "Still prose, sorry."}]
    outcome = _run(_row(sparse_temp=False), steps, tmp_path)
    assert any("cap:tokens" in reason for reason in outcome["abstained"])
    assert "cap:tokens" in outcome["row_abstained"]


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
    assert "cap:tokens" in outcome["row_abstained"]


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
        "pack": [{"options": ["Higher", "Lower"], "label": "Higher"},
                 {"options": ["Yes", "No"], "label": "No"}],
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


def test_mcq_tiers_hold_the_same_unpruned_tool_surface(tmp_path):
    """The arm's question is whether tool access helps a question tier;
    withdrawing the tools would answer it by construction."""
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
        mcp_profile="core",
    )
    assert seen == [{"profile": "core"}]


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
    assert manifest["mcp_profile"] == "full"


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
