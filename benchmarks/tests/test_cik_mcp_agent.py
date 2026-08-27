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
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import benchmarks.cik.mcp_agent as mcp_agent_module
from benchmarks.cik.mcp_agent import (
    _forecast_grid_prompt, _transformation_literal_values,
    _verbatim_constant_lines, _verbatim_literal_claim_ids,
    _verbatim_semantic_constant_lines,
    _bind_verbatim_literal_units, _canonicalize_timestamped_series_values,
    _bind_missing_transformation_claim_windows,
    _future_series_values, _verbatim_series_lines,
    _verbatim_series_claim_ids,
    _expand_change_point_series_values,
    _simplify_identity_literals,
    _merge_transformation_repair,
    _restore_cited_power_literals,
    _bind_transformation_provenance,
    _select_publication_fail_closed,
    _has_material_numeric_context,
    _validated_item_count,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_material_numeric_context_ignores_calendar_not_business_quantities():
    assert not _has_material_numeric_context(
        "Maintenance begins 2026-01-03T14:30:00+00:00 and ends at 16:00.")
    assert _has_material_numeric_context(
        "Maintenance begins 2026-01-03 and lasts for 6 days.")
    assert _has_material_numeric_context(
        "The comparable site's maximum was 25.83 at 21:10:00.")
    assert _has_material_numeric_context("Demand is bounded below by -2.5.")


def test_validator_diagnostic_counts_accept_count_or_collection_shapes():
    assert _validated_item_count(2) == 2
    assert _validated_item_count([{"id": 1}]) == 1
    assert _validated_item_count(None) == 0
    assert _validated_item_count(-1) == 0


def test_regular_long_forecast_grid_is_compact_but_exact():
    stamps = [f"2026-01-01T00:{minute:02d}:00+00:00" for minute in range(40)]
    rendered = json.loads(_forecast_grid_prompt(stamps))
    assert rendered == {
        "kind": "regular_host_grid",
        "first": stamps[0], "last": stamps[-1], "steps": 40,
        "step_seconds": 60.0,
        "anchor_rule": (
            "quantile anchors must use first, last, or another timestamp "
            "obtained by adding an integer number of step_seconds to first"),
    }
    assert len(_forecast_grid_prompt(stamps)) < len(json.dumps(stamps)) / 3


def test_irregular_forecast_grid_is_never_misdescribed_as_regular():
    stamps = ["2026-01-01T00:00:00+00:00",
              "2026-01-01T00:01:00+00:00",
              "2026-01-01T00:03:00+00:00"] * 12
    assert json.loads(_forecast_grid_prompt(stamps)) == stamps


def test_transformation_constant_citations_are_completed_verbatim_only():
    wrapper = {"transformation": {"expression": {
        "op": "divide", "args": [
            {"op": "power", "args": [
                {"op": "series", "name": "speed"},
                {"op": "literal", "value": 2}]},
            {"op": "literal", "value": 3000},
            {"op": "literal", "value": 37.5},
        ]}}}
    assert _transformation_literal_values(wrapper) == [2.0, 3000.0, 37.5]
    context = ("Pressure follows the square of speed.\n"
               "The maximal fan speed is 3000 rpm and pressure is 37.5 Pa.")
    assert _verbatim_constant_lines(
        wrapper, context, ["Pressure follows the square of speed."]) == [
            "The maximal fan speed is 3000 rpm and pressure is 37.5 Pa."]
    assert _verbatim_constant_lines(
        wrapper, context,
        ["Pressure follows the square of speed and exponent 2.",
         "The maximal fan speed is 3000 rpm and pressure is 37.5 Pa."]) == []
    assert _verbatim_literal_claim_ids(wrapper, [
        {"claim_id": "claim-1", "source_span": "Pressure follows speed."},
        {"claim_id": "claim-2", "source_span": context.splitlines()[1]},
    ]) == ["claim-2"]
    assert _verbatim_semantic_constant_lines(
        wrapper, "Pressure is proportional to the square of speed.", []) == [
            "Pressure is proportional to the square of speed."]

    wrapper["units"] = {"primary": "Pa", "speed": "rpm"}
    wrapper["transformation"]["output_unit"] = "Pa"
    bound, changes = _bind_verbatim_literal_units(wrapper, [
        {"source_span": context.splitlines()[1]}])
    args = bound["transformation"]["expression"]["args"]
    assert changes == 2
    assert args[1]["unit"] == "rpm"
    assert args[2]["unit"] == "Pa"
    assert args[0]["args"][1].get("unit") is None
    assert bound["transformation"]["literal_unit_binding"] \
        == "verbatim_source_adjacency"


def test_exact_cited_power_is_restored_without_general_algebra_inference():
    wrapper = {"transformation": {
        "claim_ids": ["claim-1"],
        "expression": {"op": "divide", "args": [
            {"op": "power", "args": [
                {"op": "series", "name": "speed"},
                {"op": "literal", "value": 2}]},
            {"op": "literal", "value": 9_000_000},
        ]},
    }}
    claims = [{"claim_id": "claim-1", "source_span": (
        "Pressure follows the square of speed over maximal speed; "
        "maximal speed is 3000 rpm.")}]
    restored, changes = _restore_cited_power_literals(wrapper, claims)
    denominator = restored["transformation"]["expression"]["args"][1]
    assert changes == 1
    assert denominator == {"op": "power", "args": [
        {"op": "literal", "value": 3000.0},
        {"op": "literal", "value": 2},
    ]}
    assert wrapper["transformation"]["expression"]["args"][1]["value"] \
        == 9_000_000

    # An exact arithmetic coincidence is insufficient without cited power
    # semantics, and ambiguous bases are never guessed between.
    unchanged, count = _restore_cited_power_literals(wrapper, [{
        "claim_id": "claim-1", "source_span": "maximal speed is 3000 rpm"}])
    assert count == 0
    assert unchanged == wrapper


def test_repaired_transformation_rebinds_host_claim_ids_transactionally():
    raw = {"transformations": [{
        "transformation": {
            "claim_ids": ["compiler-claim"],
            "expression": {"op": "multiply", "args": [
                {"op": "literal", "value": 37.5},
                {"op": "series", "name": "rpm_in"},
            ]},
        },
        "series_values": {"rpm_in": {
            "values": [298.0, 298.0],
            "source_claim_ids": ["compiler-claim"],
        }},
    }]}
    claims = [
        {"claim_id": "claim-1", "source_span": "max pressure is 37.5 Pa"},
        {"claim_id": "claim-2", "source_span": "speed changes to 298.0"},
    ]
    bound = _bind_transformation_provenance(raw, claims)
    wrapper = bound["transformations"][0]
    assert wrapper["transformation"]["claim_ids"] == [
        "compiler-claim", "claim-1", "claim-2"]
    assert wrapper["series_values"]["rpm_in"]["source_claim_ids"] == [
        "compiler-claim", "claim-2"]
    assert raw["transformations"][0]["series_values"]["rpm_in"][
        "source_claim_ids"] == ["compiler-claim"]

    context = ("Pressure follows the square of speed.\n"
               "The maximal fan speed is 3000 rpm and pressure is 37.5 Pa.")
    ratio_wrapper = {"transformation": {"output_unit": "Pa", "expression": {
        "op": "multiply", "args": [
            {"op": "literal", "value": 37.5},
            {"op": "power", "args": [
                {"op": "divide", "args": [
                    {"op": "series", "name": "rpm_in"},
                    {"op": "literal", "value": 3000}]},
                {"op": "literal", "value": 2}]}]}},
        "units": {"primary": "Pa", "series_name": "Pa"}}
    ratio_bound, changes = _bind_verbatim_literal_units(ratio_wrapper, [
        {"source_span": context.splitlines()[1]}])
    ratio = ratio_bound["transformation"]["expression"]["args"][1]["args"][0]
    assert changes == 2
    assert ratio["args"][1]["unit"] == "rpm"
    assert ratio_bound["units"]["rpm_in"] == "rpm"


def test_incompatible_scenario_ranking_retains_live_publication(monkeypatch):
    publication = {
        "publication_seal_sha256": "already-verified",
        "candidate_portfolio": [{"scenario_id": "candidate"}],
    }

    def reject(*_args, **_kwargs):
        raise ValueError("ranking omitted one live scenario")

    monkeypatch.setattr("gnomon.publication.select_publication", reject)
    retained, error = _select_publication_fail_closed(
        publication, {"selected_scenario_id": "candidate"})
    assert retained is publication
    assert error == (
        "selector incompatible with live portfolio: "
        "ranking omitted one live scenario")


def test_single_live_scenario_needs_no_selector_and_reports_no_error():
    publication = {
        "recommended_scenario_id": "primary",
        "candidate_portfolio": [{"scenario_id": "primary"}],
    }

    retained, error = _select_publication_fail_closed(publication, None)

    assert retained is publication
    assert error is None


def test_pre_call_ranking_is_completed_for_extra_live_scenarios(monkeypatch):
    publication = {"candidate_portfolio": [
        {"scenario_id": "primary"}, {"scenario_id": "candidate"},
        {"scenario_id": "product-sensitivity"}]}
    observed = {}

    def accept(payload, selection):
        observed.update(selection)
        return {**payload, "selected": selection["selected_scenario_id"]}

    monkeypatch.setattr("gnomon.publication.select_publication", accept)
    selected, error = _select_publication_fail_closed(publication, {
        "selected_scenario_id": "candidate",
        "ranking": ["candidate", "primary"],
    })
    assert error is None
    assert selected["selected"] == "candidate"
    assert observed["ranking"] == [
        "candidate", "primary", "product-sensitivity"]
    assert observed["host_completed_live_portfolio"] is True


def test_timestamped_future_series_normalizes_only_on_exact_host_grid():
    stamps = ["2026-01-01T00:00:00+00:00",
              "2026-01-01T01:00:00+00:00"]
    wrapper = {"series_values": {"price": {"values": [
        {"timestamp": stamps[1], "value": 4},
        {"timestamp": stamps[0], "value": 3}]}}}
    canonical, changes = _canonicalize_timestamped_series_values(
        wrapper, stamps)
    assert changes == 1
    assert canonical["series_values"]["price"]["values"] == [3, 4]
    assert canonical["series_values"]["price"]["syntax_canonicalization"] \
        == "timestamped_rows_exact_host_grid"
    off_grid = {"series_values": {"price": {"values": [
        {"timestamp": stamps[0], "value": 3}]}}}
    unchanged, changes = _canonicalize_timestamped_series_values(
        off_grid, stamps)
    assert changes == 0
    assert unchanged == off_grid


def test_only_missing_transformation_claim_windows_bind_to_cutoff():
    cutoff = "2026-01-01T00:00:00+00:00"
    raw = {"transformations": [{}], "claims": [
        {"source_span": "static formula", "effective_start": None,
         "effective_end": None},
        {"source_span": "dated schedule",
         "effective_start": "2026-01-02T00:00:00+00:00",
         "effective_end": "2026-01-03T00:00:00+00:00"},
    ]}
    bound, changes = _bind_missing_transformation_claim_windows(raw, cutoff)
    assert changes == 1
    assert bound["claims"][0]["effective_start"] == cutoff
    assert bound["claims"][0]["effective_end"] == cutoff
    assert bound["claims"][0]["effective_window_binding"] \
        == "undated_transformation_specification_at_cutoff"
    assert bound["claims"][1] == raw["claims"][1]
    untouched, changes = _bind_missing_transformation_claim_windows(
        {"claims": raw["claims"]}, cutoff)
    assert changes == 0
    assert untouched == {"claims": raw["claims"]}


def test_future_series_values_bind_only_to_verbatim_source_lines():
    wrapper = {"series_values": {"rpm_in": {"values": [
        {"timestamp": "t1", "value": 1591.7},
        {"timestamp": "t2", "value": 1591.7},
        {"timestamp": "t3", "value": 298.0}]}}}
    context = ("Speed starts at 1591.7 rpm.\n"
               "At 05:32:42 it changes to 298.0 rpm.")
    assert _future_series_values(wrapper) == {"rpm_in": [1591.7, 298.0]}
    assert _verbatim_series_lines(wrapper, context, []) == context.splitlines()
    claims = [
        {"claim_id": "claim-1", "source_span": context.splitlines()[0]},
        {"claim_id": "claim-2", "source_span": context.splitlines()[1]},
    ]
    assert _verbatim_series_claim_ids(wrapper, claims) == {
        "rpm_in": ["claim-1", "claim-2"]}
    assert _verbatim_series_lines(
        wrapper, context, [context.splitlines()[0]]) == [context.splitlines()[1]]
    punctuated = [{"claim_id": "claim-3",
                   "source_span": "It changes to 1591.7."}]
    assert _verbatim_series_claim_ids(wrapper, punctuated) == {
        "rpm_in": ["claim-3"]}


def test_compact_change_point_schedule_resolves_unique_host_clock():
    stamps = [f"1970-01-01T05:32:{second:02d}+00:00"
              for second in range(8, 14)]
    wrapper = {"series_values": {"rpm_in": {
        "initial_value": 1591.7,
        "change_points": [{"timestamp": "05:32:11", "value": 298.0}],
    }}}
    expanded, changes = _expand_change_point_series_values(wrapper, stamps)
    payload = expanded["series_values"]["rpm_in"]
    assert changes == 1
    assert payload["values"] == [1591.7, 1591.7, 1591.7, 298.0, 298.0, 298.0]
    assert payload["resolved_change_points"] == [
        {"timestamp": stamps[3], "value": 298.0}]

    ambiguous = stamps + ["1970-01-02T05:32:11+00:00"]
    unchanged, changes = _expand_change_point_series_values(wrapper, ambiguous)
    assert changes == 0
    assert "values" not in unchanged["series_values"]["rpm_in"]

    with_history = {"series_values": {"rpm_in": {
        "initial_value": 285.5,
        "change_points": [
            {"timestamp": "1970-01-01T05:31:26+00:00", "value": 285.9},
            {"timestamp": "1970-01-01T05:32:07+00:00", "value": 1591.7},
            {"timestamp": "05:32:11", "value": 298.0}],
    }}}
    expanded, changes = _expand_change_point_series_values(with_history, stamps)
    assert changes == 1
    assert expanded["series_values"]["rpm_in"]["values"] == [
        1591.7, 1591.7, 1591.7, 298.0, 298.0, 298.0]


def test_only_exact_algebraic_identities_are_simplified():
    wrapper = {"transformation": {"expression": {
        "op": "divide", "args": [
            {"op": "add", "args": [
                {"op": "series", "name": "x"},
                {"op": "literal", "value": 0}]},
            {"op": "literal", "value": 1}]}}}
    simplified, changes = _simplify_identity_literals(wrapper)
    assert changes == 2
    assert simplified["transformation"]["expression"] == {
        "op": "series", "name": "x"}
    assert simplified["transformation"]["identity_simplification"] \
        == "exact_algebraic_identities_removed"
    material = {"transformation": {"expression": {
        "op": "multiply", "args": [
            {"op": "series", "name": "x"},
            {"op": "literal", "value": 1.01}]}}}
    unchanged, changes = _simplify_identity_literals(material)
    assert changes == 0
    assert unchanged == material


def test_transformation_repair_cannot_delete_or_rewrite_prior_claims():
    prior = {"claims": [
        {"source_span": "Rule A", "relation": "unknown"}],
        "transformations": [{"old": True}], "events": [{"kept": True}]}
    repaired = {"claims": [
        {"source_span": "Rule A", "relation": "supports_increase"},
        {"source_span": "Parameter B", "relation": "unknown"}],
        "transformations": [{"new": True}], "events": []}
    merged = _merge_transformation_repair(prior, repaired)
    assert merged["claims"] == [
        {"source_span": "Rule A", "relation": "unknown"},
        {"source_span": "Parameter B", "relation": "unknown"}]
    assert merged["transformations"] == [{"new": True}]
    assert merged["events"] == [{"kept": True}]

from benchmarks.cik.gnomon_forecaster import GnomonAbstained
from benchmarks.cik.mcp_agent import (
    MAX_MCP_CALLS,
    MAX_ROUNDS,
    InProcessMcpSession,
    McpAgentForecaster,
    _has_explicit_lag_relationship,
    _extract_explicit_driver_schedule,
    _compiler_target_evidence,
    _task_companion_evidence,
    _task_companion_histories,
    _task_target_name,
    _transformation_repair_hints,
    jail_violations,
    openai_tool_specs,
)


@pytest.mark.parametrize("text", [
    "sales[t] = 1 + 0.5 sales[t-1] + 2 campaign[t-2]",
    "The coefficient at lag 1 affects demand as 0.7 * demand.",
    r"X_1^{t} = -0.567 * X_0^{t-1} + 0.505 * X_1^{t-2}",
])
def test_explicit_lag_relationship_router_is_syntax_based(text):
    assert _has_explicit_lag_relationship(text) is True
    assert _has_explicit_lag_relationship(
        "A campaign may improve sales next quarter.") is False


def test_explicit_driver_schedule_requires_named_complete_ranges():
    text = ("X_0 takes a value of 0.2 from 2026-01-01 to 2026-01-03, "
            "0.4 from 2026-01-04 to 2026-01-05.\n"
            "X_1 is 99 from 2026-01-01 to 2026-01-05.")
    result = _extract_explicit_driver_schedule(
        text, series="X_0", cutoff="2026-01-03T00:00:00+00:00",
        future_timestamps=["2026-01-04T00:00:00+00:00",
                           "2026-01-05T00:00:00+00:00"])
    assert result is not None
    historical, future = result
    assert historical == [{"start": "2026-01-01", "end": "2026-01-03",
                           "value": .2, "source_claim_ids": ["claim-1"]}]
    assert future == [.4, .4]
    assert _extract_explicit_driver_schedule(
        text, series="X_0", cutoff="2026-01-03T00:00:00+00:00",
        future_timestamps=["2026-01-06T00:00:00+00:00"]) is None


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


def _irregular_hourly_task():
    from datetime import datetime, timedelta, timezone

    epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
    observed_hours = [index for index in range(73) if index != 20]
    return SimpleNamespace(
        past_time=[((epoch + timedelta(hours=index)).isoformat(),
                    50.0 + index % 24) for index in observed_hours],
        future_time=[(epoch + timedelta(hours=73 + index)).isoformat()
                     for index in range(4)],
        background="Hourly operations telemetry.", constraints=None,
        scenario="Values will stay in their usual range.",
        name="IrregularHourlyTask", seed=1,
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
        self.completion_reasoning_efforts = []
        self.completion_request_timeouts = []
        self.completion_transport_retries = []
        self.completion_prompts = []

    def chat(self, messages, *, n=1, tools=None, tool_choice=None,
             request_timeout=None, transport_retries=None):
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

    def completions(self, messages, *, n=1, temperature=None,
                    reasoning_effort=None, request_timeout=None,
                    transport_retries=None):
        self.completion_temperatures.append(temperature)
        self.completion_reasoning_efforts.append(reasoning_effort)
        self.completion_request_timeouts.append(request_timeout)
        self.completion_transport_retries.append(transport_retries)
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


class DelayedCompilerClient(ScriptedClient):
    def completions(self, *args, **kwargs):
        time.sleep(.02)
        return super().completions(*args, **kwargs)


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


def test_numeric_cik_columns_use_context_semantic_aliases():
    pd = pytest.importorskip("pandas")
    task = SimpleNamespace(past_time=pd.DataFrame(
        {0: [1.0, 2.0], 1: [3.0, 4.0]},
        index=pd.date_range("2026-01-01", periods=2, freq="D")))
    assert _task_target_name(task) == "X_1"
    assert set(_task_companion_histories(task)) == {"X_0"}
    assert "timestamp,X_0" in _task_companion_evidence(task)


def test_compiler_target_evidence_summarizes_all_and_bounds_raw_tail():
    timestamps = [f"2024-01-{index + 1:02d}T00:00:00+00:00"
                  for index in range(20)]
    values = list(map(float, range(20)))
    evidence = _compiler_target_evidence(timestamps, values, limit=4)
    assert '"observations": 20' in evidence
    assert '"minimum": 0.0' in evidence and '"maximum": 19.0' in evidence
    rows = evidence.split("timestamp,value\n", 1)[1].splitlines()
    assert [float(row.rsplit(",", 1)[-1]) for row in rows] == [16, 17, 18, 19]


def test_compiler_contract_separates_history_from_future_covariates():
    from benchmarks.cik.mcp_agent import DOSSIER_INSTRUCTIONS

    assert "NEVER copy those historical rows into covariate_tables" in \
        DOSSIER_INSTRUCTIONS
    assert "exact requested forecast timestamp" in DOSSIER_INSTRUCTIONS


def test_compiler_contract_preserves_historical_observation_semantics():
    from benchmarks.cik.mcp_agent import (
        DOSSIER_INSTRUCTIONS,
        OBSERVATION_INSTRUCTIONS,
        _expects_historical_zero_interpretation,
    )

    assert "readings were corrupted" in DOSSIER_INSTRUCTIONS
    assert "Do not rewrite history" in DOSSIER_INSTRUCTIONS
    assert "sealed forecast_candidate estimated from unaffected history" in \
        DOSSIER_INSTRUCTIONS
    assert _expects_historical_zero_interpretation(
        "Historical maintenance caused no withdrawals recorded. There is no "
        "future maintenance.")
    assert _expects_historical_zero_interpretation(
        "Maintenance resulted in no withdrawals recorded. The ATM will not "
        "be in maintenance in the future.")
    assert not _expects_historical_zero_interpretation(
        "The site was closed for maintenance and may close again.")
    assert "do not guess a mask" in OBSERVATION_INSTRUCTIONS
    assert "sealed forecast_candidate" in OBSERVATION_INSTRUCTIONS
    assert "cannot edit the immutable primary" in OBSERVATION_INSTRUCTIONS
    assert "recurring_clock_window" in OBSERVATION_INSTRUCTIONS


def test_transformation_repair_hints_are_verbatim_and_constant_specific():
    failures = [{"violations": [{"message":
        "Transformation constant 37.5 (literal) is absent from every cited source span."}]}]
    context = ("Maximum speed is 3000 rpm and pressure is 37.5 Pa.\n"
               "An unrelated threshold is 20 Pa.")
    assert _transformation_repair_hints(failures, context) == [
        "Maximum speed is 3000 rpm and pressure is 37.5 Pa."]


def _forecaster(steps, tmp_path, sessions=None, profile=None,
                compiler_output=None, output_role="canonical"):
    def factory(cwd):
        session = InProcessMcpSession(cwd)
        if sessions is not None:
            sessions.append(session)
        return session

    return McpAgentForecaster(
        "x/y", client=ScriptedClient(steps, compiler_output), session_factory=factory,
        work_dir=str(tmp_path), trace_dir=tmp_path / "traces", profile=profile,
        output_role=output_role,
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
    """Known forecast intent does not pay a redundant tool-choice turn."""
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
    assert len(client.steps) == 1  # conversational tool-choice script unused
    assert len(samples[0]) == 4
    trace = json.loads(next((tmp_path / "traces").glob("*.json")).read_text())
    assert trace["trace"][0]["host_bound_submission"] == {
        "accepted": True, "route": "gnomon"}


def test_evidence_executes_one_server_authored_grid_repair(tmp_path):
    client = ScriptedClient([])
    forecaster = McpAgentForecaster(
        "x/y", client=client,
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), trace_dir=tmp_path / "traces",
        profile="evidence",
    )

    samples, extra = forecaster(_irregular_hourly_task(), 1)

    assert extra["route"] == "gnomon"
    assert extra["mcp_calls"] == 2
    assert len(samples[0]) == 4
    trace = json.loads(next((tmp_path / "traces").glob("*.json")).read_text())
    assert trace["trace"][0]["code"] == "IRREGULAR_TIME_GRID"
    assert trace["trace"][1] == {
        "governed_recovery": "server_authored_aggressive_repair",
        "source_error_code": "IRREGULAR_TIME_GRID",
        "model_turn_required": False,
    }
    assert trace["trace"][2]["host_bound_submission"] == {
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


def test_numeric_context_gets_one_bounded_sufficiency_repair(tmp_path):
    task = _task()
    span = "A comparable site reached 120 on 2023-06-20."
    task.scenario = span
    empty = json.dumps({
        "events": [], "claims": [], "hypotheses": [],
        "covariate_tables": [], "transformations": [],
        "observation_interpretations": [], "effect_proposal": None,
        "forecast_candidate": None,
    })
    repaired = json.dumps({
        "events": [],
        "claims": [{
            "source_span": span, "relation": "unknown",
            "effective_start": "2023-06-20T00:00:00+00:00",
            "effective_end": "2023-06-20T23:59:59+00:00",
            "mechanism": "A comparable historical episode.",
            "confidence": 0.5,
        }],
        "hypotheses": [{
            "kind": "historical_analogue", "claim_ids": ["claim-1"],
            "target_series": ["*"], "predictor_series": None,
            "known_at": task.past_time[-1][0], "lag_steps": 0,
            "direction": "unknown",
            "rationale": "Keep the reference visible without treating one episode as validation.",
        }],
        "covariate_tables": [], "transformations": [],
        "observation_interpretations": [], "effect_proposal": None,
        "forecast_candidate": None,
    })
    forecaster = _forecaster(
        [], tmp_path, profile="evidence", compiler_output=[empty, repaired])

    _, extra = forecaster(task, 1)

    compilation = extra["context_compilation"]
    assert compilation["claim_count"] == 1
    assert compilation["hypothesis_count"] == 1
    receipt = json.loads(Path(compilation["receipt_path"]).read_text())
    assert [call["stage"] for call in receipt["compiler"]["calls"]] == [
        "initial_compile", "dossier_repair"]
    assert receipt["dossier"]["hypotheses"][0]["kind"] \
        == "historical_analogue"


def test_hypothesis_knowledge_time_is_bound_by_host_without_llm_repair(
        tmp_path):
    task = _task()
    span = "A comparable site reached 120 on 2023-06-20."
    task.scenario = span
    compiler_output = json.dumps({
        "events": [],
        "claims": [{
            "source_span": span, "relation": "unknown",
            "effective_start": "2023-06-20T00:00:00+00:00",
            "effective_end": "2023-06-20T23:59:59+00:00",
            "mechanism": "A comparable historical episode.",
            "confidence": 0.5,
        }],
        "hypotheses": [{
            "kind": "historical_analogue", "claim_ids": ["claim-1"],
            "target_series": ["*"], "predictor_series": None,
            # Model-authored future metadata must not trigger an LLM repair.
            "known_at": "2099-01-01T00:00:00+00:00", "lag_steps": 0,
            "direction": "unknown", "rationale": "Weak external analogue.",
        }],
        "covariate_tables": [], "transformations": [],
        "observation_interpretations": [], "effect_proposal": None,
        "forecast_candidate": None,
    })
    forecaster = _forecaster(
        [], tmp_path, profile="evidence", compiler_output=compiler_output)

    _, extra = forecaster(task, 1)

    receipt = json.loads(Path(extra["context_compilation"][
        "receipt_path"]).read_text())
    assert len(receipt["compiler"]["calls"]) == 1
    assert receipt["dossier"]["hypotheses"][0]["known_at"] == \
        task.past_time[-1][0]
    assert receipt["compiler"]["repair_decisions"][0][
        "hypothesis_violation_codes"] == []


def test_accepted_historical_observation_count_does_not_crash_receipt(
        tmp_path):
    task = _task()
    task.past_time[9] = (task.past_time[9][0], 0.0)
    task.past_time[10] = (task.past_time[10][0], 0.0)
    start, end = task.past_time[9][0], task.past_time[10][0]
    span = (f"Maintenance from {start} to {end} resulted in no requests "
            "recorded. There is no future maintenance.")
    task.scenario = span
    compiler_output = json.dumps({
        "events": [],
        "claims": [{
            "source_span": span, "relation": "unknown",
            "effective_start": start, "effective_end": end,
            "mechanism": "Historical readings were maintenance artifacts.",
            "confidence": 1.0,
        }],
        "observation_interpretations": [{
            "kind": "historical_contamination", "claim_ids": ["claim-1"],
            "predicate": {"op": "equals", "value": 0.0},
            "window": "cited_window",
            "rationale": "Exclude only the cited maintenance zeros.",
        }],
        "hypotheses": [], "covariate_tables": [], "transformations": [],
        "effect_proposal": None, "forecast_candidate": None,
    })
    forecaster = _forecaster(
        [], tmp_path, profile="evidence", compiler_output=compiler_output)

    _, extra = forecaster(task, 1)

    receipt = json.loads(Path(extra["context_compilation"][
        "receipt_path"]).read_text())
    decision = receipt["compiler"]["repair_decisions"][0]
    assert decision["accepted_observation_interpretations"] == 1
    assert decision["triggered"] is False


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
    selection_output = json.dumps({
        "selected_scenario_id": "prior-assisted-1",
        "ranking": ["prior-assisted-1", "primary"],
        "cited_claim_ids": ["claim-1"], "counterevidence_claim_ids": [],
        "confidence": .6, "rationale": "The cited closure supports the path.",
        "what_would_change_selection": "Observed activity during the closure.",
    })
    forecaster = McpAgentForecaster(
        "x/y", client=ScriptedClient(
            [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
            [compiler_output, selection_output]),
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), trace_dir=tmp_path / "traces", profile="evidence",
        output_role="publication_best_effort")
    forecaster.benchmark_seed = 41
    samples, extra = forecaster(task, 1)
    assert [row[0] for row in samples[0]] == [127, 128, 129, 130]
    assert extra["route"] == "publication_best_effort"
    assert extra["publication"]["recommended_support"] == "prior_assisted"
    assert extra["publication"]["primary_forecast_unchanged"] is True
    assert extra["publication"]["automation"]["eligible"] is False
    assert extra["publication"]["context_summary"]["status"] == "used"
    trace = json.loads(next((tmp_path / "traces").glob("*.json")).read_text())
    assert trace["seed"] == 41
    assert trace["final_submission"]["recommended_scenario_id"] \
        == "prior-assisted-1"
    assert trace["final_submission"]["primary_forecast_unchanged"] is True
    assert trace["final_submission"]["automation_eligible"] is False
    assert trace["final_submission"]["context_summary"]["status"] == "used"


def test_live_evidence_dominance_is_not_reported_as_selector_failure():
    from benchmarks.cik.mcp_agent import _select_publication_fail_closed
    from gnomon.publication import publish_result

    result = {
        "support": "supported",
        "forecast": [{"timestamp": "2026-01-02T00:00:00+00:00",
                      "point": 10, "q10": 9, "q50": 10, "q90": 11}],
        "primary_forecast": [{"timestamp": "2026-01-02T00:00:00+00:00",
                              "point": 9, "q10": 8, "q50": 9, "q90": 10}],
        "context_outcome": {
            "status": "applied", "admission_basis": "historical_fold_ablation",
            "events": ["event-1"],
        },
    }
    publication = publish_result(result, mode="best_effort")
    retained, reason = _select_publication_fail_closed(publication, None)
    assert retained is publication
    assert reason == "selector skipped: governed evidence dominance"


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
    assert client.completion_reasoning_efforts == ["none", "none"]
    assert client.completion_request_timeouts == [60, 60]
    assert client.completion_transport_retries == [0, 0]


def test_accepted_effect_does_not_recompile_for_malformed_optional_lane(
        tmp_path):
    task = _task()
    span = "Demand will be 2 times the usual level tomorrow."
    task.scenario = span
    dossier = json.dumps({
        "claims": [{
            "source_span": span, "relation": "supports_increase",
            "effective_start": task.future_time[0],
            "effective_end": task.future_time[-1], "confidence": .9,
        }],
        "effect_proposal": {
            "shape": "temporary_pulse", "unit": "fraction_of_level",
            "location": 1, "lower": 1, "upper": 1, "confidence": .9,
            "delay_steps": 0, "duration_steps": 1,
            "scope": {"kind": "single_series", "series": ["*"]},
            "claim_ids": ["claim-1"], "composition": "scenario_only",
        },
        # This side lane is deliberately malformed. Its critique remains in
        # the receipt, but it must not replace the valid executable effect.
        "forecast_candidate": {"quantiles": "not-an-array"},
    })
    client = ScriptedClient(
        [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
        dossier)
    forecaster = McpAgentForecaster(
        "x/y", client=client,
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence",
        output_role="publication_best_effort")
    _, extra = forecaster(task, 1)

    assert len(client.completion_prompts) == 1
    receipt = json.loads(Path(extra["context_compilation"][
        "receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["dossier"]["effect_proposal_critique"]["status"] \
        == "accepted"
    assert receipt["dossier"]["candidate_critique"]["status"] == "rejected"
    assert receipt["compiler"]["repair_decisions"] == [{
        "stage": "dossier_probe", "triggered": False,
        "accepted_executable": True, "effect_status": "accepted",
        "effect_violation_codes": [], "rejected_effect_fields": {},
        "candidate_status": "rejected",
        "candidate_reasons": [
            "forecast_candidate quantiles must match the requested horizon"],
        "accepted_observation_interpretations": 0,
        "rejected_hypotheses": 0,
        "hypothesis_violation_codes": [],
        "rejected_observation_interpretations": 0,
        "required_observation_lane_missing": False,
        "numeric_context_unresolved": False,
        "top_level_rejections": 1,
    }]
    assert client.completion_reasoning_efforts == ["none"]
    assert client.completion_transport_retries == [0]
    assert extra["scenario_selector"] == {
        "attempted": False, "accepted": False,
        "disposition": "skipped_evidence_dominance",
        "error": "selector skipped: governed evidence dominance",
    }


def test_valid_effect_skips_repair_of_malformed_optional_transformation(
        tmp_path):
    task = _task()
    span = "Demand will be 2 times the usual level tomorrow."
    task.scenario = span
    dossier = json.dumps({
        "claims": [{
            "source_span": span, "relation": "supports_increase",
            "effective_start": task.future_time[0],
            "effective_end": task.future_time[-1], "confidence": .9,
        }],
        "effect_proposal": {
            "shape": "temporary_pulse", "unit": "fraction_of_level",
            "location": 1, "lower": 1, "upper": 1, "confidence": .9,
            "delay_steps": 0, "duration_steps": 1,
            "scope": {"kind": "single_series", "series": ["*"]},
            "claim_ids": ["claim-1"], "composition": "scenario_only",
        },
        "transformations": [{
            "transformation": {
                "known_at": task.past_time[-1][0],
                "claim_ids": ["claim-1"], "lane": "scenario_only",
                "output_unit": "target_units",
                "expression": {"op": "series", "name": "future_input"},
            },
            "units": {"primary": "target_units",
                      "future_input": "target_units"},
            # Deliberately one row for a four-step horizon.
            "series_values": {"future_input": {
                "values": [2.0], "known_at": task.past_time[-1][0],
                "source_claim_ids": ["claim-1"],
            }},
        }],
        "events": [], "hypotheses": [], "covariate_tables": [],
        "observation_interpretations": [], "forecast_candidate": None,
    })
    forecaster = _forecaster(
        [], tmp_path, profile="evidence", compiler_output=dossier,
        output_role="publication_best_effort")

    _, extra = forecaster(task, 1)

    receipt = json.loads(Path(extra["context_compilation"][
        "receipt_path"]).read_text())
    assert len(receipt["compiler"]["calls"]) == 1
    decision = next(item for item in receipt["compiler"]["repair_decisions"]
                    if item["stage"] == "transformation_preflight")
    assert decision == {
        "stage": "transformation_preflight", "triggered": False,
        "failure_count": 1, "repair_already_used": False,
        "violation_codes": ["HORIZON_MISMATCH"],
        "alternative_executable_available": True,
        "skip_reason": "valid_non_transform_executable",
    }
    assert extra["publication"]["recommended_scenario_id"] == \
        "effect-composed-1"
    dispositions = extra["publication"]["context_dispositions"]
    assert any(item.get("reason_code") == "HORIZON_MISMATCH"
               and item.get("disposition") == "rejected"
               for item in dispositions)
    assert any(item.get("reason_code") ==
               "duplicate_transformation_preflight_summary"
               and item.get("disposition") == "superseded"
               for item in dispositions)


def test_exact_lag_claims_get_one_focused_sufficiency_repair(tmp_path):
    task = _task()
    equation = "Parents for sales at lag 1 affect it as 0.5 * sales."
    task.scenario = equation
    claim = {
        "source_span": equation, "relation": "supports_increase",
        "effective_start": task.future_time[0],
        "effective_end": task.future_time[-1], "confidence": .9,
    }
    first = json.dumps({"events": [], "claims": [claim],
                        "covariate_tables": [], "transformations": []})
    repaired = json.dumps({
        "events": [], "claims": [claim], "covariate_tables": [],
        "transformations": [{"transformation": {
            "known_at": task.past_time[-1][0], "claim_ids": ["claim-1"],
            "lane": "prior_assisted", "output_unit": "value",
            "expression": {
                "op": "recursive_linear", "output_unit": "value",
                "intercept": 0,
                "autoregressive_terms": [{"lag": 1, "coefficient": .5}],
                "driver_terms": [],
            }}}],
    })
    client = ScriptedClient(
        [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
        [first, repaired])
    forecaster = McpAgentForecaster(
        "x/y", client=client,
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence",
        output_role="publication_best_effort")
    _, extra = forecaster(task, 1)
    # Repair recovers an executable scenario, but a cited recurrence is not
    # recommendation authority until it beats last-value in governed replay.
    assert extra["publication"]["recommended_scenario_id"] == "primary"
    scenario = next(item for item in extra["publication"]["candidate_portfolio"]
                    if item["scenario_id"] == "transformation-1")
    assert scenario["selection_eligible"] is False
    assert client.completion_reasoning_efforts == ["none", "none"]
    assert "exact cited lag equations" in client.completion_prompts[1]


def test_explicit_equation_contract_host_grounds_document_without_copy_repair(
        tmp_path):
    task = _task()
    equation = "sales[t] = 0.5 * sales[t-1]"
    task.scenario = equation
    compiler_output = json.dumps({
        "claims": [],
        "transformations": [{
            # Compiler attempts a future knowledge time; the host owns this.
            "known_at": task.future_time[-1], "claim_id": "missing",
            "historically_testable": "sales",
            "recursive_linear": {"intercept": 0,
                                 "autoregressive_terms": [
                                     {"lag": 1, "coefficient": .5}],
                                 "driver_terms": [],
                                 "series_values": {}}}],
    })
    client = ScriptedClient(
        [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
        compiler_output)
    forecaster = McpAgentForecaster(
        "x/y", client=client,
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence",
        output_role="publication_best_effort")
    _, extra = forecaster(task, 1)
    assert len(client.completion_prompts) == 1
    assert extra["context_compilation"]["claim_count"] == 1
    assert not any("UNVERIFIED_CLAIMS" in item for item in
                   extra["context_compilation"].get("rejections", []))
    assert extra["publication"]["primary_forecast_unchanged"] is True
    assert any(item["scenario_id"] == "transformation-1" for item in
               extra["publication"]["candidate_portfolio"])


def test_single_verified_claim_rebinds_stale_transformation_id(tmp_path):
    task = _task()
    span = "A new policy makes each future value exactly half the usual value."
    task.scenario = span
    compiler_output = json.dumps({
        "events": [],
        "claims": [{
            "source_span": span, "relation": "supports_decrease",
            "effective_start": task.future_time[0],
            "effective_end": task.future_time[-1],
            "mechanism": "stated multiplier", "confidence": .9,
        }],
        "transformations": [{"transformation": {
            "known_at": task.past_time[-1][0],
            "claim_ids": ["claim-99"], "lane": "prior_assisted",
            "output_unit": "unknown",
            "expression": {"op": "multiply", "args": [
                {"op": "primary", "quantile": "q50"},
                {"op": "literal", "value": .5,
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
    _, extra = forecaster(task, 1)

    assert extra["publication"]["recommended_scenario_id"] == "transformation-1"
    receipt = json.loads(Path(
        extra["context_compilation"]["receipt_path"]).read_text())
    transformation = receipt["transformations"][0]["transformation"]
    assert transformation["claim_ids"] == ["claim-1"]
    assert transformation["citation_binding"] == "single_verified_claim"


def test_single_verified_claim_rebinds_effect_and_hypothesis_ids(tmp_path):
    task = _task()
    span = "Demand increases by 10 units throughout the forecast window."
    task.scenario = span
    compiler_output = json.dumps({
        "events": [],
        "claims": [{
            "source_span": span, "relation": "supports_increase",
            "effective_start": task.future_time[0],
            "effective_end": task.future_time[-1],
            "mechanism": "stated increment", "confidence": .9,
        }],
        "effect_proposal": {
            "shape": "level_shift", "unit": "target_units",
            "location": 10, "lower": 8, "upper": 12, "confidence": .8,
            "delay_steps": 0, "duration_steps": None,
            "scope": {"kind": "single_series", "series": ["value"]},
            "claim_ids": ["claim-99"], "rationale": "stated increment",
            "uncertainty_basis": "bounded around stated value",
        },
        "hypotheses": [{
            "kind": "additive_change", "claim_ids": ["claim-99"],
            "target_series": ["value"], "predictor_series": None,
            "known_at": task.past_time[-1][0], "lag_steps": 0,
            "direction": "increase", "rationale": "stated increment",
        }],
        "forecast_candidate": None, "covariate_tables": [],
        "transformations": [],
    })
    forecaster = McpAgentForecaster(
        "x/y", client=ScriptedClient(
            [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
            compiler_output),
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence",
        output_role="publication_best_effort")
    _, extra = forecaster(task, 1)
    receipt = json.loads(Path(
        extra["context_compilation"]["receipt_path"]).read_text())
    assert receipt["dossier"]["effect_proposal"]["claim_ids"] == ["claim-1"]
    assert receipt["dossier"]["hypotheses"][0]["claim_ids"] == ["claim-1"]
    assert extra["publication"]["recommended_scenario_id"] == "effect-composed-1"


def test_invalid_claim_receives_the_one_bounded_dossier_repair(tmp_path):
    task = _task()
    span = "A historical outage corrupted readings, but it has ended."
    task.scenario = span
    bad = json.dumps({
        "events": [], "claims": [{
            "source_span": span, "relation": "unknown",
            "effective_start": "not-a-date", "effective_end": "not-a-date",
            "confidence": "high",
        }], "hypotheses": [], "effect_proposal": None,
        "forecast_candidate": None, "covariate_tables": [],
        "transformations": [],
    })
    good = json.dumps({
        "events": [], "claims": [{
            "source_span": span, "relation": "unknown",
            "effective_start": task.past_time[0][0],
            "effective_end": task.past_time[-1][0], "confidence": .8,
        }], "hypotheses": [{
            "kind": "unsupported", "claim_ids": ["claim-1"],
            "target_series": ["value"], "predictor_series": None,
            "known_at": task.past_time[-1][0], "lag_steps": 0,
            "direction": "unknown", "rationale": "measurement artifact",
        }], "effect_proposal": None, "forecast_candidate": None,
        "covariate_tables": [], "transformations": [],
    })
    client = ScriptedClient(
        [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
        [bad, good])
    forecaster = McpAgentForecaster(
        "x/y", client=client,
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence",
        output_role="publication_best_effort")
    _, extra = forecaster(task, 1)
    assert len(client.completion_prompts) == 2
    assert "one complete corrected dossier" in client.completion_prompts[1]
    assert extra["context_compilation"]["claim_count"] == 1
    assert extra["context_compilation"]["hypothesis_count"] == 1


def test_literal_zero_claim_uses_deterministic_override_lane(tmp_path):
    task = _task()
    start, end = task.future_time[1], task.future_time[2]
    span = f"Readings are zero from {start} to {end}."
    task.scenario = span
    compiler_output = json.dumps({
        "events": [],
        "claims": [{
            "source_span": span, "relation": "supports_decrease",
            "effective_start": start, "effective_end": end,
            "mechanism": "stated outage", "confidence": 1,
        }],
        "hypotheses": [], "effect_proposal": None,
        "forecast_candidate": None, "covariate_tables": [],
        "transformations": [],
    })
    forecaster = McpAgentForecaster(
        "x/y", client=ScriptedClient(
            [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
            compiler_output),
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence",
        output_role="publication_best_effort")
    _, extra = forecaster(task, 1)
    receipt = json.loads(Path(
        extra["context_compilation"]["receipt_path"]).read_text())
    assert receipt["events"][0]["event_type"] == \
        "override:stated_absolute_value"
    publication = extra["publication"]
    assert publication["recommended_scenario_id"] == "context_conditioned"
    rows = publication["recommended_forecast"]
    assert [rows[index]["q50"] for index in (1, 2)] == [0.0, 0.0]
    assert publication["primary_forecast_unchanged"] is True
    assert publication["automation"]["eligible"] is False


def test_immutable_primary_role_scores_preserved_path_not_context_projection(
        tmp_path):
    task = _task()
    start, end = task.future_time[1], task.future_time[2]
    span = f"Readings are zero from {start} to {end}."
    task.scenario = span
    compiler_output = json.dumps({
        "events": [],
        "claims": [{
            "source_span": span, "relation": "supports_decrease",
            "effective_start": start, "effective_end": end,
            "mechanism": "stated outage", "confidence": 1,
        }],
        "hypotheses": [], "effect_proposal": None,
        "forecast_candidate": None, "covariate_tables": [],
        "transformations": [],
    })
    forecaster = McpAgentForecaster(
        "x/y", client=ScriptedClient(
            [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
            compiler_output),
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence",
        output_role="immutable_primary")

    samples, extra = forecaster(task, 1)

    assert extra["route"] == "immutable_primary_diagnostic"
    assert extra["diagnostic_only"] is True
    assert extra["context_recommendation_ignored"] is True
    assert samples[0][1][0] != 0.0
    assert samples[0][2][0] != 0.0


def test_immutable_primary_role_uses_public_path_when_context_did_not_apply(
        tmp_path):
    forecaster = _forecaster(
        [], tmp_path, profile="evidence", compiler_output=json.dumps({
            "events": [], "claims": [], "hypotheses": [],
            "effect_proposal": None, "forecast_candidate": None,
            "covariate_tables": [], "transformations": [],
        }), output_role="immutable_primary")

    samples, extra = forecaster(_task(), 1)

    assert len(samples[0]) == len(_task().future_time)
    assert extra["route"] == "immutable_primary_diagnostic"
    assert extra["context_recommendation_ignored"] is True


def test_undated_general_rule_is_retained_with_actionable_recovery(tmp_path):
    task = _task()
    span = "Demand typically falls during public holidays."
    task.background = span
    clean_dossier = {
        "events": [],
        "claims": [{
            "source_span": span, "relation": "supports_decrease",
            "effective_start": None, "effective_end": None,
            "timing_status": "unresolved_trigger",
            "mechanism": "Holiday demand effect", "confidence": .7,
        }],
        "hypotheses": [{
            "kind": "unsupported", "claim_ids": ["claim-1"],
            "target_series": ["*"], "predictor_series": None,
            "known_at": task.past_time[-1][0], "lag_steps": 0,
            "direction": "decrease", "rationale": "Trigger date missing.",
        }],
        "effect_proposal": None, "forecast_candidate": None,
        "covariate_tables": [], "transformations": [],
    }
    invalid_dossier = {**clean_dossier, "transformations": [{
        "transformation": {
            "known_at": task.past_time[-1][0],
            "claim_ids": ["claim-1"], "lane": "scenario_only",
            "output_unit": "value", "expression": {
                "op": "add", "args": [
                    {"op": "primary", "quantile": "q50"},
                    {"op": "literal", "value": 1, "unit": "value"},
                ],
            },
        },
        "units": {"primary": "value"}, "series_values": {},
    }]}
    forecaster = McpAgentForecaster(
        "x/y", client=ScriptedClient(
            [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
            [json.dumps(invalid_dossier), json.dumps(clean_dossier)]),
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence",
        output_role="publication_best_effort")

    _, extra = forecaster(task, 1)

    assert extra["context_compilation"]["claim_count"] == 1
    assert extra["context_compilation"]["hypothesis_count"] == 1
    receipt = json.loads(Path(extra["context_compilation"][
        "receipt_path"]).read_text())
    decision = next(item for item in receipt["compiler"]["repair_decisions"]
                    if item["stage"] == "transformation_preflight")
    assert decision["triggered"] is True
    assert decision["violation_codes"] == ["UNRESOLVED_TRIGGER_TIMING"]
    assert receipt["rejections"] == []
    disposition = next(item for item in extra["publication"][
        "context_dispositions"] if item.get("claim_id") == "claim-1")
    assert disposition["reason_code"] == "trigger_timing_unresolved"
    assert disposition["recovery_action"]["code"] == "provide_dated_trigger"
    assert extra["publication"]["recommended_scenario_id"] == "primary"
    assert extra["publication"]["automation"]["eligible"] is False
    assert extra["scenario_selector"] == {
        "attempted": False, "accepted": False,
        "disposition": "not_required", "error": None,
    }


def test_transformation_preflight_repairs_malformed_future_series(tmp_path):
    task = _task()
    span = "The future input is 2.0 throughout the forecast window."
    task.scenario = span
    claim = {
        "source_span": span, "relation": "unknown",
        "effective_start": task.future_time[0],
        "effective_end": task.future_time[-1],
        "mechanism": "stated future input", "confidence": 1,
    }

    def dossier(values):
        return json.dumps({
            "events": [], "claims": [claim],
            "transformations": [{
                "transformation": {
                    "known_at": task.past_time[-1][0],
                    "claim_ids": ["claim-1"], "lane": "prior_assisted",
                    "output_unit": "unknown",
                    "expression": {"op": "series", "name": "future_input"},
                },
                "units": {"future_input": "unknown"},
                "series_values": {"future_input": {
                    "values": values, "known_at": task.past_time[-1][0],
                    "source_claim_ids": ["claim-1"],
                }},
            }],
        })

    client = ScriptedClient(
        [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
        [dossier(["not-numeric"] * 4), dossier([2.0] * 4)])
    forecaster = McpAgentForecaster(
        "x/y", client=client,
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence",
        output_role="publication_best_effort")
    _, extra = forecaster(task, 1)

    assert extra["publication"]["recommended_scenario_id"] == "transformation-1"
    assert client.completion_reasoning_efforts == ["none", "none"]
    assert "NON_NUMERIC_VALUE" in client.completion_prompts[1]


def test_source_schedule_normalizes_malformed_range_rows_without_repair(tmp_path):
    pd = pytest.importorskip("pandas")
    task = _task()
    timestamps = [timestamp for timestamp, _ in task.past_time]
    target = [value for _, value in task.past_time]
    cutoff = timestamps[-1]
    task.past_time = pd.DataFrame(
        {0: [1.0] * len(target), 1: target},
        index=pd.to_datetime(timestamps),
    )
    span = (
        "X_1[t] = 0.5 X_1[t-1] + 2 X_0[t-1]. "
        "X_0 is 1 from 2024-01-01 to 2024-03-12, "
        "2 from 2024-03-13 to 2024-03-16."
    )
    task.scenario = span
    compiler_output = json.dumps({
        "events": [],
        "claims": [{
            "source_span": span, "relation": "supports_increase",
            "effective_start": task.future_time[0],
            "effective_end": task.future_time[-1],
            "mechanism": "stated recurrence and driver schedule",
            "confidence": 1,
        }],
        "transformations": [{
            "transformation": {
                "known_at": cutoff,
                "claim_ids": ["claim-1"],
                "lane": "historically_testable",
                "output_unit": "target_units",
                "expression": {
                    "op": "recursive_linear", "output_unit": "target_units",
                    "intercept": 0,
                    "autoregressive_terms": [{"lag": 1, "coefficient": .5}],
                    "driver_terms": [{
                        "series": "X_0", "lag": 1, "coefficient": 2}],
                },
            },
            "units": {"primary": "target_units", "X_0": "target_units"},
            # This is a common LLM representation: semantically exact but not
            # executable by the numeric AST until the cited ranges are
            # expanded over the host-owned forecast grid.
            "series_values": {"X_0": {
                "values": [{
                    "start": "2024-03-13T00:00:00+00:00",
                    "end": "2024-03-16T00:00:00+00:00", "value": 2,
                }],
                "known_at": cutoff,
                "source_claim_ids": ["claim-1"],
            }},
            "historical_series_segments": {"X_0": [{
                "start": "2024-01-01", "end": "2024-03-12", "value": 1,
                "source_claim_ids": ["claim-1"],
            }]},
        }],
    })
    client = ScriptedClient(
        [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
        compiler_output)
    forecaster = McpAgentForecaster(
        "x/y", client=client,
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), profile="evidence",
        output_role="publication_best_effort")

    _, extra = forecaster(task, 1)

    receipt = json.loads(Path(
        extra["context_compilation"]["receipt_path"]).read_text())
    supplied = receipt["transformations"][0]["series_values"]["X_0"]
    assert supplied["values"] == [2.0, 2.0, 2.0, 2.0]
    assert supplied["syntax_canonicalization"] == "cited_range_schedule"
    assert len(client.completion_prompts) == 1
    assert receipt["rejections"] == []


def test_candidate_survives_but_cannot_replace_rejected_transform(tmp_path):
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

    assert len(samples[0]) == 4
    publication = extra["publication"]
    assert publication["recommended_scenario_id"] == "primary"
    assert publication["primary_forecast_unchanged"] is True
    assert publication["automation"]["eligible"] is False
    candidate = next(item for item in publication["candidate_portfolio"]
                     if item["role"] == "model_authored")
    assert candidate["selection_eligible"] is False
    assert candidate["forecast"][0]["q50"] == 127
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
    assert client.completion_reasoning_efforts == ["none", "none"]


def test_context_workflow_deadline_prevents_stacked_repair_timeouts(
        tmp_path, monkeypatch):
    task = _task()
    span = "Demand doubles during the forecast window."
    task.scenario = span
    invalid = json.dumps({
        "events": [], "claims": [{
            "source_span": span, "relation": "supports_increase",
            "effective_start": task.future_time[0],
            "effective_end": task.future_time[-1], "confidence": .9}],
        "forecast_candidate": {"quantiles": [
            {"timestamp": stamp, "q10": 0, "q50": 0, "q90": 0}
            for stamp in task.future_time], "rationale": "placeholder"},
    })
    client = DelayedCompilerClient(
        [{"tool_calls": [("gnomon_forecast", {"frequency": "D"})]}],
        [invalid, invalid])
    monkeypatch.setattr(mcp_agent_module,
                        "MAX_CONTEXT_COMPILATION_SECONDS", .01)
    forecaster = McpAgentForecaster(
        "x/y", client=client,
        session_factory=lambda cwd: InProcessMcpSession(cwd),
        work_dir=str(tmp_path), trace_dir=tmp_path / "traces",
        profile="evidence")
    forecaster(task, 1)
    assert len(client.completion_prompts) == 1
    receipt = json.loads(next(
        (tmp_path / "traces" / "context-receipts").glob("*.json")
    ).read_text())
    assert receipt["compiler"]["workflow_budget_seconds"] == .01
    assert receipt["compiler"]["calls"][0]["stage"] == "initial_compile"
    assert any("deadline exhausted" in item
               for item in receipt["rejections"])


def test_shadow_role_requires_evidence_profile():
    with pytest.raises(ValueError, match="requires the evidence profile"):
        McpAgentForecaster("x/y", profile="full",
                           output_role="llm_candidate_shadow")


def test_evidence_exposes_only_the_task_required_forecast_tool(tmp_path):
    class InspectingClient(ScriptedClient):
        def chat(self, messages, *, n=1, tools=None, tool_choice=None,
                 request_timeout=None, transport_retries=None):
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
