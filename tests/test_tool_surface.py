"""Tool-surface gating and token discipline.

The MCP surface competes for model attention and context tokens. Retired
compatibility tools stay absent, profiles remain exact, and no default
response may exceed its size budget.
"""

from __future__ import annotations

import pytest


def _names(monkeypatch) -> list[str]:
    from gnomon.toolspec import visible_tools

    monkeypatch.delenv("GNOMON_MCP_PROFILE", raising=False)
    return [tool["name"] for tool in visible_tools()]


def test_retired_tools_are_not_defined_or_flag_restorable(monkeypatch) -> None:
    from gnomon.mcp_server import _handle
    from gnomon.toolspec import runner_for

    retired = {
        "gnomon_record_decision", "gnomon_resolve_decision",
        "gnomon_list_open_forecasts", "gnomon_model_performance",
        "gnomon_covariate_guide", "gnomon_propose_covariates",
        "gnomon_proposer_skill", "gnomon_compile_task",
        "gnomon_validate_plan", "gnomon_execute_plan", "gnomon_get_run",
    }
    assert retired.isdisjoint(_names(monkeypatch))
    monkeypatch.setenv("GNOMON_V02_COMPAT", "1")
    monkeypatch.setenv("GNOMON_EXPERIMENTAL_PLANNER", "1")
    assert retired.isdisjoint(_names(monkeypatch))
    assert all(runner_for(name) is None for name in retired)


def test_status_sections_preserve_current_tracking_reads(monkeypatch, tmp_path) -> None:
    from gnomon.toolspec import _run_model_performance, _run_open_forecasts, _run_status
    from gnomon.tracking import TrackingStore

    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "registry.db"))
    TrackingStore().register(
        forecast_id="fc_status", project="status", selected_model="drift",
        support="supported", artifact_path=str(tmp_path), horizon=3,
    )
    args = {"project": "status"}
    assert _run_status({**args, "section": "open_forecasts"}) == _run_open_forecasts(args)
    assert _run_status({**args, "section": "performance"}) == _run_model_performance(args)

    full = _run_status(args)
    decisions = _run_status({**args, "section": "decisions"})
    assert decisions["unresolved_decisions"] == full["unresolved_decisions"]
    assert decisions["decision_summary"] == full["decision_summary"]
    assert "open_forecasts" not in decisions

    import pytest
    from gnomon.contracts import GnomonError
    with pytest.raises(GnomonError):
        _run_status({"section": "performance"})
    with pytest.raises(GnomonError):
        _run_status({"section": "effects"})
    with pytest.raises(GnomonError):
        _run_status({"project": "status", "section": "effect_prior"})
    effects = _run_status({"project": "status", "section": "effects"})
    assert effects == {"schema_version": "0.1", "project": "status",
                       "effects": []}


def test_effect_prior_and_robust_decision_are_agent_callable(monkeypatch, tmp_path) -> None:
    from gnomon.toolspec import _run_status, _run_unified
    from gnomon.tracking import TrackingStore

    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "registry.db"))
    prior = {
        "prior_id": "vendor-1", "event_type": "promotion", "target": "sales",
        "domain": "retail", "population": "*", "unit": "units",
        "effect_family": "additive", "mean": 4.0, "standard_error": 1.0,
        "delay_steps": [0, 1], "duration_steps": [2, 4], "sample_size": 20,
        "source": "study", "version": "2026-01", "known_at": "2026-01-01T00:00:00+00:00",
    }
    resolved = _run_status({
        "section": "effect_prior", "project": "shop", "event_type": "promotion",
        "series": "sales", "as_of": "2026-02-01T00:00:00+00:00",
        "target": "sales", "domain": "retail", "unit": "units",
        "external_priors": [prior],
    })
    assert resolved["resolution"]["selected_lane"] == "external_prior"
    assert resolved["resolution"]["may_affect_primary_forecast"] is False

    result = _run_unified({
        "question": {"kind": "robust_decision"},
        "decision_id": "decision-1", "project": "shop", "forecast_id": "forecast-1",
        "scenario_ids": ["promotion"],
        "actions": [{"name": "wait"}, {"name": "stock"}],
        "utilities": {
            "wait": {"primary": 2.0, "promotion": 1.0},
            "stock": {"primary": 0.0, "promotion": 5.0},
        },
    })
    assert result["decision"]["selected_action"] == "wait"
    assert result["decision"]["scenario_probabilities"] is None
    assert TrackingStore().get_decision_artifact("decision-1") is not None


def test_surviving_covariate_contract_is_complete(monkeypatch) -> None:
    from gnomon.toolspec import TOOLS

    by_name = {tool["name"]: tool for tool in TOOLS}
    validate = by_name["gnomon_validate_covariates"]["description"]
    assert "known_at" in validate
    assert "name:type:future_known" in validate
    forecast_props = by_name["gnomon_forecast"]["inputSchema"]["properties"]
    assert "covariates_file" in forecast_props
    assert "covariate_mapping" in forecast_props
# --- Fix 7: task profiles narrow the surface -------------------------------

def test_profiles_select_documented_subsets(monkeypatch) -> None:
    from gnomon.toolspec import PROFILES, visible_tools

    monkeypatch.delenv("GNOMON_MCP_PROFILE", raising=False)
    assert {tool["name"] for tool in visible_tools()} == PROFILES["evidence"]
    monkeypatch.setenv("GNOMON_MCP_PROFILE", "full")
    full = {tool["name"] for tool in visible_tools()}

    for profile, expected in PROFILES.items():
        monkeypatch.setenv("GNOMON_MCP_PROFILE", profile)
        names = {tool["name"] for tool in visible_tools()}
        assert names == expected, profile
        if profile not in {"describe", "evidence", "mega"}:
            assert names <= full

    # A narrowed profile exposes exactly its declared tools.
    monkeypatch.setenv("GNOMON_MCP_PROFILE", "core")
    names = {tool["name"] for tool in visible_tools()}
    assert "gnomon_get_artifact" not in names
    assert len(names) == 6

    monkeypatch.setenv("GNOMON_MCP_PROFILE", "full")
    assert {tool["name"] for tool in visible_tools()} == full


def test_unknown_profile_fails_loudly(monkeypatch) -> None:
    import pytest

    from gnomon.toolspec import visible_tools

    monkeypatch.setenv("GNOMON_MCP_PROFILE", "everything")
    with pytest.raises(ValueError):
        visible_tools()


def test_capabilities_report_the_active_profile(monkeypatch) -> None:
    from gnomon.runtime import capabilities

    monkeypatch.delenv("GNOMON_MCP_PROFILE", raising=False)
    payload = capabilities()["mcp_profile"]
    assert payload["active"] == "evidence"
    assert payload["visible_tools"] == [
        "gnomon_describe", "gnomon_forecast", "gnomon_select_scenario"]
    assert payload["available"] == [
        "core", "data", "decision", "describe", "evidence", "mega", "full"]

    monkeypatch.setenv("GNOMON_MCP_PROFILE", "core")
    payload = capabilities()["mcp_profile"]
    assert payload["active"] == "core"
    assert "gnomon_decide" not in payload["visible_tools"]
    assert "gnomon_forecast" in payload["visible_tools"]


# --- Fix 6: the writable workspace is disclosed ----------------------------

def test_capabilities_disclose_the_workspace(monkeypatch, tmp_path) -> None:
    from gnomon.runtime import capabilities
    from gnomon.toolspec import TOOLS, runner_for

    monkeypatch.chdir(tmp_path)
    workspace = capabilities()["workspace"]
    assert workspace["cwd"] == str(tmp_path)
    assert workspace["default_output_dir"] == str(tmp_path / "gnomon-output")

    # A forecast with output_dir omitted writes inside the disclosed dir.
    data = tmp_path / "series.csv"
    from datetime import date, timedelta
    start = date(2026, 1, 1)
    data.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=day)},{100 + day}" for day in range(40)
    ) + "\n")
    payload = runner_for("gnomon_forecast")({
        "input": str(data), "horizon": 3,
    })
    assert payload["artifact_path"].startswith(
        workspace["default_output_dir"])

    # And the schema tells the agent to read it from capabilities rather
    # than guess.
    tool = next(t for t in TOOLS if t["name"] == "gnomon_forecast")
    description = tool["inputSchema"]["properties"]["output_dir"]["description"]
    assert "gnomon_capabilities" in description


# --- Fix 5: the batch form leads the forecast description ------------------

def test_batch_syntax_leads_the_forecast_description(tmp_path) -> None:
    from gnomon.toolspec import TOOLS, runner_for

    tool = next(t for t in TOOLS if t["name"] == "gnomon_forecast")
    head = ". ".join(tool["description"].split(". ")[:2])
    assert '"cpu,mem,requests"' in head
    assert '"auto"' in head
    target = tool["inputSchema"]["properties"]["target_column"]["description"]
    assert '"cpu,mem,requests"' in target

    # And the syntax it advertises is the one the runner accepts.
    wide = tmp_path / "wide.csv"
    from datetime import date, timedelta
    start = date(2026, 1, 1)
    rows = ["timestamp,cpu,mem"]
    rows += [f"{start + timedelta(days=d)},{50 + d},{70 + d}"
             for d in range(40)]
    wide.write_text("\n".join(rows) + "\n")
    payload = runner_for("gnomon_forecast")({
        "input": str(wide), "target_column": "cpu,mem", "horizon": 5,
        "output_dir": str(tmp_path / "out"),
    })
    series = {result["series"] for result in payload["results"]}
    assert series == {"cpu", "mem"}
    import json
    from gnomon.toolspec import RESPONSE_BUDGET_BYTES
    assert len(json.dumps(payload)) < RESPONSE_BUDGET_BYTES * 2


def test_batched_forecast_accepts_scoped_validated_context(tmp_path) -> None:
    """Host-compiled context must not force wide data back to N calls."""
    from datetime import date, timedelta
    from gnomon.toolspec import runner_for

    wide = tmp_path / "wide-context.csv"
    start = date(2026, 1, 1)
    rows = ["timestamp,cpu,mem"] + [
        f"{start + timedelta(days=d)},{50 + d},{70 + d}"
        for d in range(40)
    ]
    wide.write_text("\n".join(rows) + "\n")
    event = {
        "event_id": "maintenance-1", "event_type": "maintenance",
        "entity_scope": ["cpu"],
        "effective_start": "2026-01-10T00:00:00+00:00",
        "effective_end": "2026-01-11T00:00:00+00:00",
        "known_at": "2026-01-09T00:00:00+00:00",
        "attributes": {"evidence_quote": "Maintenance on January 10."},
        "source": {"type": "test", "reference": "fixture"},
        "created_by": "llm",
    }
    payload = runner_for("gnomon_forecast")({
        "input": str(wide), "time_column": "timestamp",
        "target_column": "cpu,mem", "frequency": "D", "horizon": 2,
        "context_events": [event], "output_dir": str(tmp_path / "out-context"),
    })
    assert {row["series"] for row in payload["results"]} == {"cpu", "mem"}
    by_series = {row["series"]: row for row in payload["results"]}
    assert by_series["cpu"]["context_outcome"]["status"] == "scenario_only"
    assert by_series["cpu"]["context_outcome"][
        "selected_projection_differs_from_primary"] is False
    assert by_series["cpu"]["context_outcome"][
        "canonical_primary_preserved"] is True
    assert by_series["cpu"]["context_outcome"][
        "automation_eligible"] is False
    assert by_series["cpu"]["context_outcome"]["authority_summary"] == (
        "Context did not change the canonical primary forecast. Context "
        "evidence alone cannot authorize automation.")
    assert by_series["mem"]["context_outcome"]["status"] == "not_considered"
    assert by_series["mem"]["context_outcome"]["authority_summary"] == (
        "Context did not change the canonical primary forecast. Context "
        "evidence alone cannot authorize automation.")


def test_batched_forecast_emits_scoped_sealed_publications(tmp_path) -> None:
    from datetime import date, timedelta
    from gnomon.toolspec import runner_for

    wide = tmp_path / "wide-publication.csv"
    start = date(2026, 1, 1)
    wide.write_text("\n".join(["timestamp,cpu,mem"] + [
        f"{start + timedelta(days=day)},{70 + day % 4},{40 + day % 3}"
        for day in range(40)
    ]) + "\n")
    payload = runner_for("gnomon_forecast")({
        "input": str(wide), "time_column": "timestamp",
        "target_column": "cpu,mem", "frequency": "D", "horizon": 1,
        "context_events": [{
            "event_id": "cpu-cap", "claim_kind": "max",
            "entity_scope": ["cpu"],
            "effective_start": "2026-02-10",
            "effective_end": "2026-02-10",
            "known_at": "2026-02-09",
            "source_span": "CPU is capped at 60 on 2026-02-10.",
        }],
        "publication_mode": "best_effort",
        "automation_policy": {"allow": False},
        "output_dir": str(tmp_path / "out-wide-publication"),
    })

    publications = {item["series"]: item for item in payload["publications"]}
    results = {item["series"]: item for item in payload["results"]}
    assert set(publications) == {"cpu", "mem"}
    assert publications["cpu"]["recommended_scenario_id"] == \
        "context_conditioned"
    assert publications["mem"]["recommended_scenario_id"] == "primary"
    assert publications["mem"]["context_dispositions"] == []
    assert results["cpu"]["context_outcome"][
        "selected_projection_differs_from_primary"] is True
    assert results["cpu"]["context_outcome"][
        "canonical_primary_preserved"] is True
    assert payload["publication_summary"] == {
        "mode": "best_effort", "series_count": 2,
        "primary_forecast_unchanged": True,
        "automation_eligible": False,
        "scenario_count": 3,
    }
    assert all(item["publication_path"] for item in publications.values())


def test_qualitative_context_lane_produces_non_automatable_sensitivity(
        tmp_path) -> None:
    import json
    from datetime import date, timedelta
    from gnomon.toolspec import runner_for

    path = tmp_path / "qualitative.csv"
    start = date(2026, 1, 1)
    path.write_text("\n".join(["timestamp,demand"] + [
        f"{start + timedelta(days=day)},{100 + day % 4}"
        for day in range(40)
    ]) + "\n")
    source = (
        "A confirmed campaign begins on 2026-02-10 and is expected to "
        "increase demand, but its magnitude is unknown.")

    payload = runner_for("gnomon_forecast")({
        "input": str(path), "time_column": "timestamp",
        "target_column": "demand", "frequency": "D", "horizon": 2,
        "qualitative_context_events": [{
            "event_id": "campaign-1",
            "entity_scope": ["demand"],
            "effective_start": "2026-02-10T00:00:00+00:00",
            "effective_end": "2026-02-11T00:00:00+00:00",
            "known_at": "2026-02-09T00:00:00+00:00",
            "direction": "increase", "effect_family": "temporary_pulse",
            "duration": "temporary", "source_span": source,
            "source_reference": "campaign-plan",
        }],
        "publication_mode": "scenario",
        "output_dir": str(tmp_path / "out-qualitative"),
    })

    result = payload["results"][0]
    assert result["context_outcome"]["status"] == "scenario_only"
    assert result["context_outcome"][
        "selected_projection_differs_from_primary"] is False
    assert result["context_outcome"]["canonical_primary_preserved"] is True
    assert "primary_forecast_changed" not in result["context_outcome"]
    assert result["context_outcome"]["sensitivity_scenarios_produced"] == 1
    assert result["context_outcome"]["hypotheses"][0]["direction"] == "increase"
    publication = payload["publication"]
    assert publication["primary_forecast_unchanged"] is True
    assert publication["automation"]["eligible"] is False
    assert publication["scenario_count"] == 2
    assert any(item["role"] == "conditional_sensitivity"
               for item in publication["selection_contract"]["scenarios"])
    assert "response_schema" not in publication["selection_contract"]
    assert "temporal_state" not in publication["selection_contract"]
    assert len(json.dumps(publication["selection_contract"])) < 3_500


def test_validated_structural_context_reaches_scenario_lane_automatically(
        tmp_path) -> None:
    import json
    from datetime import date, timedelta
    from gnomon.toolspec import runner_for

    path = tmp_path / "structural-context.csv"
    start = date(2026, 1, 1)
    path.write_text("\n".join(["timestamp,value"] + [
        f"{start + timedelta(days=day)},{100 + day * 0.5}"
        for day in range(60)
    ]) + "\n")
    source = (
        "From 2026-03-02T00:00:00+00:00, the prior trend will cease "
        "and the series will continue without that drift.")

    payload = runner_for("gnomon_forecast")({
        "input": str(path), "time_column": "timestamp",
        "target_column": "value", "frequency": "D", "horizon": 7,
        "context_events": [{
            "event_id": "repair-1", "event_type": "structural:repair",
            "entity_scope": ["value"],
            "effective_start": "2026-03-02T00:00:00+00:00",
            "effective_end": "2026-03-08T00:00:00+00:00",
            "known_at": "2026-03-01T00:00:00+00:00",
            "attributes": {"source_span": source,
                           "effect": "trend_ceases"},
            "source": {"type": "ticket", "reference": "OPS-42"},
            "created_by": "llm",
        }],
        # No structural_events switch: a validated typed event must not be
        # silently discarded by the ordinary agent path.
        "publication_mode": "scenario",
        "output_dir": str(tmp_path / "out-structural-context"),
    })

    result = payload["results"][0]
    assert result["context_outcome"]["status"] in {
        "scenario_only", "partially_represented",
    }
    assert result["context_outcome"]["canonical_primary_preserved"] is True
    scenario = next(item for item in result["sensitivity_scenarios"]
                    if item["support"] == "prior_assisted_structural")
    assert scenario["automation_eligible"] is False
    assert scenario["primary_forecast_changed"] is False
    assert scenario["effect"]["provenance"]["provenance_class"] == \
        "human_assumption"
    assert scenario["effect"]["shape"] == "trend_change"
    assert scenario["consequence"]["status"] == "numeric_difference"
    assert "forecast_preview" not in scenario
    assert scenario["consequence"]["first_affected_timestamp"] == \
        "2026-03-03T00:00:00"
    assert "canonical primary remains unchanged" in \
        scenario["consequence_summary"]
    # Brief MCP responses omit the duplicate scenario forecast preview while
    # keeping the established publication contract and exact consequence.
    assert payload["publication"]["projection"] == "compact"
    assert len(json.dumps(payload, default=str)) < 16_500


def test_brief_compacts_large_repeated_context_without_losing_counts(
        tmp_path) -> None:
    import json
    from datetime import date, timedelta

    from gnomon.toolspec import runner_for

    path = tmp_path / "repeated-context.csv"
    start = date(2026, 1, 1)
    path.write_text("\n".join(["timestamp,demand"] + [
        f"{start + timedelta(days=day)},{100 + day % 4}"
        for day in range(40)
    ]) + "\n")
    events = [{
        "event_id": f"campaign-{index}",
        "entity_scope": ["demand"],
        "effective_start": "2026-02-10T00:00:00+00:00",
        "effective_end": "2026-02-11T00:00:00+00:00",
        "known_at": "2026-02-09T00:00:00+00:00",
        "direction": "increase", "effect_family": "temporary_pulse",
        "duration": "temporary", "source_span": (
            f"Campaign pulse {index} starts on 2026-02-10 and is expected "
            "to increase demand; "
            "its magnitude is unknown."),
        "source_reference": "campaign-plan",
    } for index in range(10)]

    payload = runner_for("gnomon_forecast")({
        "input": str(path), "time_column": "timestamp",
        "target_column": "demand", "frequency": "D", "horizon": 2,
        "qualitative_context_events": events,
        "publication_mode": "scenario",
        "output_dir": str(tmp_path / "out-repeated-context"),
    })

    outcome = payload["results"][0]["context_outcome"]
    assert outcome["event_count"] == 10
    assert len(outcome["events"]) == 4
    assert outcome["events_omitted"] == 6
    assert outcome["disposition_counts"] == {"scenario": 10}
    assert outcome["hypothesis_count"] == 10
    assert len(outcome["hypotheses"]) == 1
    hypothesis = outcome["hypotheses"][0]
    assert hypothesis["count"] == 10
    assert hypothesis["representative_event_id"] == "campaign-0"
    assert hypothesis["direction"] == "increase"
    assert hypothesis["numeric_status"] == (
        "standardized_sensitivity_not_event_effect")
    assert hypothesis["may_affect_primary_forecast"] is False
    # The protected decision/support contract may exceed the ordinary 9 KiB
    # bulk budget, but repeated events must not grow the answer linearly.
    assert len(json.dumps(payload)) <= 16_000

    source = (
        "A confirmed campaign begins on 2026-02-10 and is expected to "
        "increase demand, but its magnitude is unknown.")
    strict = runner_for("gnomon_forecast")({
            "input": str(path), "time_column": "timestamp",
            "target_column": "demand", "frequency": "D", "horizon": 2,
            "qualitative_context_events": [{
                "event_id": "campaign-1", "entity_scope": ["*"],
                "effective_start": "2026-02-10T00:00:00+00:00",
                "effective_end": "2026-02-11T00:00:00+00:00",
                "known_at": "2026-02-09T00:00:00+00:00",
                "direction": "increase", "effect_family": "temporary_pulse",
                "duration": "temporary", "source_span": source,
            }], "publication_mode": "strict",
            "output_dir": str(tmp_path / "out-qualitative-strict"),
    })
    assert strict["publication"]["mode"] == "strict"
    assert strict["publication"]["context_summary"]["status"] == "scenario_only"
    assert strict["publication"]["recommended_scenario_id"] == "primary"

    advisory = runner_for("gnomon_forecast")({
        "input": str(path), "time_column": "timestamp",
        "target_column": "demand", "frequency": "D", "horizon": 1,
        "publication_mode": "strict", "_mcp_agent_boundary": True,
        "automation_policy": {
            "authorize": True, "policy_id": "model-invented",
            "minimum_support": "supported"},
        "output_dir": str(tmp_path / "out-advisory-automation"),
    })
    assert advisory["publication"]["automation"]["eligible"] is False
    assert advisory["publication"]["automation"]["reason_code"] == \
        "untrusted_authorization_channel"


def test_compact_literal_context_is_compiled_and_source_bound() -> None:
    from gnomon.toolspec import _context_events_from

    events = _context_events_from({"context_events": [{
        "event_id": "closure-1", "claim_kind": "exact",
        "entity_scope": ["*"],
        "effective_start": "2026-02-10T00:00:00+00:00",
        "effective_end": "2026-02-10T00:00:00+00:00",
        "known_at": "2026-02-09T00:00:00+00:00",
        "source_span": "The warehouse is closed on 2026-02-10: demand is exactly 0.",
        "source_reference": "signed-notice",
    }]})

    assert len(events) == 1
    assert events[0].event_type == "override:literal_exact"
    assert events[0].attributes["source_span"].endswith("exactly 0.")
    assert events[0].source.reference == "signed-notice"


def test_external_prediction_cannot_masquerade_as_literal_constraint() -> None:
    from gnomon.toolspec import _context_events_from

    arguments = {"target_column": "throughput", "context_events": [{
        "event_id": "vendor-forecast", "claim_kind": "min",
        "effective_start": "2026-02-10T00:00:00+00:00",
        "effective_end": "2026-02-10T23:59:59+00:00",
        "known_at": "2026-02-09T00:00:00+00:00",
        "source_span": (
            "The vendor forecasts throughput will be at least 500 units on "
            "2026-02-10."),
    }]}
    assert _context_events_from(arguments) == []
    assert arguments["context_rejections"] == [{
        "context_id": "vendor-forecast",
        "reason_code": "external_prediction_not_constraint",
        "reason": ("The quoted source predicts a value; it does not state a "
                   "binding constraint or observed outcome."),
        "source_span": (
            "The vendor forecasts throughput will be at least 500 units on "
            "2026-02-10."),
    }]

    shortened = {
        "target_column": "throughput",
        "_trusted_context_source_text": (
            "A vendor forecast predicts throughput will be at least 500 "
            "units on 2026-02-10."),
        "context_events": [{
            "event_id": "shortened", "claim_kind": "min",
            "effective_start": "2026-02-10T00:00:00+00:00",
            "effective_end": "2026-02-10T23:59:59+00:00",
            "known_at": "2026-02-09T00:00:00+00:00",
            "source_span": (
                "throughput will be at least 500 units on 2026-02-10"),
        }]}
    assert _context_events_from(shortened) == []
    assert shortened["context_rejections"][0]["reason_code"] == \
        "external_prediction_not_constraint"

    # Future tense is itself a prediction.  A numeric grammar must not turn
    # it into authority merely because the words also parse as a bound.
    bare = {"target_column": "throughput", "context_events": [{
        "event_id": "bare-cap-forecast", "claim_kind": "max",
        "effective_start": "2026-02-10T00:00:00+00:00",
        "effective_end": "2026-02-10T23:59:59+00:00",
        "known_at": "2026-02-09T00:00:00+00:00",
        "source_span": "throughput will not exceed 80 on 2026-02-10",
    }]}
    assert _context_events_from(bare) == []
    assert bare["context_rejections"][0]["reason_code"] == \
        "external_prediction_not_constraint"


def test_context_span_must_exist_in_host_bound_document() -> None:
    from gnomon.toolspec import _context_events_from

    arguments = {
        "target_column": "demand",
        "_trusted_context_source_text": "The warehouse remains open.",
        "context_events": [{
            "event_id": "invented", "claim_kind": "exact",
            "effective_start": "2026-02-10T00:00:00+00:00",
            "effective_end": "2026-02-10T23:59:59+00:00",
            "known_at": "2026-02-09T00:00:00+00:00",
            "source_span": "The warehouse is closed; demand is exactly 0.",
        }]}
    assert _context_events_from(arguments) == []
    assert arguments["context_rejections"][0]["reason_code"] == \
        "source_span_not_in_context_document"


def test_binding_rule_with_predictive_language_remains_literal_context() -> None:
    from gnomon.toolspec import _context_events_from

    events = _context_events_from({"target_column": "throughput",
        "context_events": [{
            "event_id": "binding-cap", "claim_kind": "max",
            "effective_start": "2026-02-10T00:00:00+00:00",
            "effective_end": "2026-02-10T23:59:59+00:00",
            "known_at": "2026-02-09T00:00:00+00:00",
            "source_span": (
                "The forecast policy requires throughput not exceed 80 on "
                "2026-02-10."),
        }]})
    assert len(events) == 1
    assert events[0].event_type == "constraint:literal_max"


def test_task_forecast_instruction_does_not_poison_later_binding_context() -> None:
    from gnomon.toolspec import _context_events_from

    source = (
        "Forecast staffing for next week. A contingency plan requires at "
        "least 18 staff on 2026-02-10."
    )
    events = _context_events_from({
        "target_column": "staffing",
        "_trusted_context_source_text": source,
        "context_events": [{
            "event_id": "contingency-floor",
            "claim_kind": "min",
            "effective_start": "2026-02-10T00:00:00+00:00",
            "effective_end": "2026-02-10T23:59:59+00:00",
            "known_at": "2026-02-09T00:00:00+00:00",
            "source_span": (
                "A contingency plan requires at least 18 staff on "
                "2026-02-10"
            ),
        }],
    })

    assert len(events) == 1
    assert events[0].event_type == "constraint:literal_min"
    assert events[0].entity_scope == ("staffing",)


def test_compact_literal_context_rejects_ambiguous_or_unsourced_forms() -> None:
    import pytest
    from gnomon.contracts import GnomonError
    from gnomon.toolspec import _context_events_from

    base = {
        "event_id": "cap-1", "claim_kind": "max", "entity_scope": ["*"],
        "effective_start": "2026-02-10T00:00:00+00:00",
        "effective_end": "2026-02-10T01:00:00+00:00",
        "known_at": "2026-02-09T00:00:00+00:00",
        "source_span": "Capacity is capped at exactly 40 units.",
    }
    compatible = _context_events_from({"context_events": [{
        **base, "event_type": "constraint:legacy",
        "attributes": {"source_span": "stale duplicate"},
    }]})
    assert compatible[0].event_type == "constraint:literal_max"
    assert compatible[0].attributes["source_span"] == base["source_span"]
    with pytest.raises(GnomonError, match="conflicts with legacy"):
        _context_events_from({"context_events": [{
            **base, "event_type": "override:cap",
        }]})
    with pytest.raises(GnomonError, match="requires verbatim source_span"):
        _context_events_from({"context_events": [{**base, "source_span": ""}]})
    with pytest.raises(GnomonError, match="unknown context claim_kind"):
        _context_events_from({"context_events": [{**base, "claim_kind": "around"}]})

    repaired = _context_events_from({"target_column": "value", "context_events": [{
        **base, "claim_kind": "exact",
        "source_span": "A signed notice caps capacity at exactly 40 units.",
    }]})
    assert repaired[0].event_type == "constraint:literal_exact"
    assert repaired[0].attributes["compiler_normalizations"][0] == {
        "code": "literal_claim_reclassified_from_source",
        "from": "override", "to": "constraint",
        "reason": "the quoted text, not the model label, determines semantics",
    }

    scoped = _context_events_from({
        "target_column": "cpu,memory", "context_events": [{
            **base, "entity_scope": [],
            "source_span": "A signed rule caps CPU at exactly 40 units.",
        }],
    })
    assert scoped[0].entity_scope == ("cpu",)
    with pytest.raises(GnomonError, match="names exactly one"):
        _context_events_from({
            "target_column": "cpu,memory", "context_events": [{
                **base, "entity_scope": [],
                "source_span": "A signed rule caps both systems at 40 units.",
            }],
        })


def test_compact_literal_context_auto_enables_governed_future_lane(tmp_path) -> None:
    from datetime import date, timedelta
    from gnomon.toolspec import runner_for

    path = tmp_path / "literal.csv"
    start = date(2026, 1, 1)
    path.write_text("\n".join(["timestamp,demand"] + [
        f"{start + timedelta(days=day)},{100 + day % 4}" for day in range(40)
    ]) + "\n")
    payload = runner_for("gnomon_forecast")({
        "input": str(path), "time_column": "timestamp",
        "target_column": "demand", "frequency": "D", "horizon": 2,
        "context_events": [{
            "event_id": "closure-1", "claim_kind": "exact",
            "entity_scope": ["demand"],
            "effective_start": "2026-02-10T00:00:00+00:00",
            "effective_end": "2026-02-10T00:00:00+00:00",
            "known_at": "2026-02-09T00:00:00+00:00",
            "source_span": (
                "The warehouse is closed on 2026-02-10, so demand is exactly 0."
            ),
        }],
        "publication_mode": "best_effort",
        "output_dir": str(tmp_path / "out-literal"),
    })

    result = payload["results"][0]
    assert result["context_outcome"]["status"] == "applied"
    assert result["context_outcome"]["canonical_primary_preserved"] is True
    assert payload["publication"]["recommended_scenario_id"] == \
        "context_conditioned"
    assert payload["publication"]["primary_forecast_unchanged"] is True
    assert payload["publication"]["automation"]["eligible"] is False


def test_qualitative_context_lane_rejects_numeric_claims_and_missing_sources():
    import pytest
    from gnomon.contracts import GnomonError
    from gnomon.toolspec import _context_events_from

    base = {
        "event_id": "campaign-1",
        "effective_start": "2026-02-10T00:00:00+00:00",
        "effective_end": "2026-02-11T00:00:00+00:00",
        "known_at": "2026-02-09T00:00:00+00:00",
        "direction": "increase", "effect_family": "temporary_pulse",
        "duration": "temporary", "source_span": "Demand should increase.",
    }
    with pytest.raises(GnomonError) as numeric:
        _context_events_from({
            "qualitative_context_events": [{**base, "magnitude": 25}],
        })
    assert numeric.value.details["unknown_fields"] == ["magnitude"]

    with pytest.raises(GnomonError) as unsourced:
        _context_events_from({
            "qualitative_context_events": [{**base, "source_span": ""}],
        })
    assert unsourced.value.details["missing_fields"] == ["source_span"]

    ambiguous = {"qualitative_context_events": [{
        **base, "source_span": "The campaign starts sometime next month."
    }]}
    assert _context_events_from(ambiguous) == []
    assert ambiguous["context_rejections"][0]["reason_code"] == \
        "ambiguous_timing"

    daily = _context_events_from({
        "frequency": "D", "qualitative_context_events": [{
            **base,
            "effective_start": "2026-02-10T00:00:00",
            "effective_end": "2026-02-10",
            "known_at": "2026-02-09",
            "source_span": "The campaign starts on 2026-02-10.",
        }],
    })
    assert daily[0].effective_start == "2026-02-10T00:00:00+00:00"
    assert daily[0].effective_end == "2026-02-10T23:59:59+00:00"
    assert len(daily[0].attributes["compiler_normalizations"]) == 3


def test_context_ref_conflicts_with_qualitative_inline_context():
    import pytest
    from gnomon.contracts import GnomonError
    from gnomon.toolspec import _materialise_context

    with pytest.raises(GnomonError) as raised:
        _materialise_context({
            "context_ref": "ctx_existing",
            "qualitative_context_events": [{"event_id": "campaign-1"}],
        })
    assert set(raised.value.details["conflicts"]) == {
        "context_ref", "qualitative_context_events",
    }


def test_direct_context_rejection_is_receipted_without_changing_numbers(
        tmp_path) -> None:
    import pytest
    from datetime import date, timedelta
    from gnomon.contracts import GnomonError
    from gnomon.toolspec import runner_for

    path = tmp_path / "rejected-context.csv"
    start = date(2026, 1, 1)
    path.write_text("\n".join(["timestamp,demand"] + [
        f"{start + timedelta(days=day)},{100 + day % 4}"
        for day in range(40)
    ]) + "\n")
    arguments = {
        "input": str(path), "time_column": "timestamp",
        "target_column": "demand", "frequency": "D", "horizon": 1,
        "context_rejections": [{
            "context_id": "late-notice", "reason_code": "known_after_cutoff",
            "reason": "The notice was first knowable after the forecast cutoff.",
            "source_span": "The notice was published on 2026-02-11.",
        }],
        "publication_mode": "best_effort",
        "output_dir": str(tmp_path / "out-rejected-context"),
    }
    payload = runner_for("gnomon_forecast")(arguments)
    publication = payload["publication"]
    assert publication["context_summary"]["status"] == "rejected"
    assert publication["recommended_scenario_id"] == "primary"
    assert publication["primary_forecast_unchanged"] is True
    assert publication["automation"]["eligible"] is False
    assert publication["context_summary"][
        "context_evidence_automation_eligible"] is False
    assert publication["context_summary"][
        "canonical_primary_preserved"] is True
    rejection = publication["context_dispositions"][0]
    assert rejection["reason_code"] == "known_after_cutoff"
    assert rejection["source_span"] == \
        "The notice was published on 2026-02-11."

    alias_arguments = {**arguments, "context_rejections": [{
        "event_id": "late-notice-alias",
        "reason_code": "known_after_cutoff",
        "reason": "The notice was first knowable after the forecast cutoff.",
        "source_span": "The notice was published on 2026-02-11.",
    }], "output_dir": str(tmp_path / "out-rejected-context-alias")}
    alias_payload = runner_for("gnomon_forecast")(alias_arguments)
    assert alias_payload["publication"]["context_dispositions"][0][
        "context_id"] == "late-notice-alias"

    conflicting = {**arguments, "context_events": [{
        "event_id": "late-notice", "claim_kind": "max",
        "effective_start": "2026-02-10", "effective_end": "2026-02-10",
        "known_at": "2026-02-09",
        "source_span": "Demand is capped at 120 on 2026-02-10.",
    }], "output_dir": str(tmp_path / "out-conflicting-disposition")}
    with pytest.raises(GnomonError) as conflict:
        runner_for("gnomon_forecast")(conflicting)
    assert conflict.value.code == "CONTEXT_DISPOSITION_CONFLICT"
    assert conflict.value.details["context_ids"] == ["late-notice"]

    invalid = {**arguments, "context_rejections": [{
        "context_id": "late-notice", "reason_code": "known_after_cutoff",
        "reason": "Known later.",
    }]}
    with pytest.raises(GnomonError) as raised:
        runner_for("gnomon_forecast")(invalid)
    assert raised.value.details["missing_fields"] == ["source_span"]


def test_context_ref_replays_validated_context_without_resending(
        tmp_path, monkeypatch) -> None:
    from datetime import date, timedelta
    from gnomon.toolspec import runner_for

    monkeypatch.setenv("GNOMON_CONTEXT_STORE", str(tmp_path / "context-store"))
    monkeypatch.setenv("GNOMON_CONTEXT_NAMESPACE", "test-project")
    wide = tmp_path / "wide-context-ref.csv"
    start = date(2026, 1, 1)
    wide.write_text("\n".join(["timestamp,cpu"] + [
        f"{start + timedelta(days=day)},{50 + day}" for day in range(40)
    ]) + "\n")
    event = {
        "event_id": "maintenance-1", "event_type": "maintenance",
        "entity_scope": ["cpu"],
        "effective_start": "2026-01-10T00:00:00+00:00",
        "effective_end": "2026-01-11T00:00:00+00:00",
        "known_at": "2026-01-09T00:00:00+00:00",
        "source": {"type": "test", "reference": "fixture"},
    }
    first = runner_for("gnomon_forecast")({
        "input": str(wide), "time_column": "timestamp",
        "target_column": "cpu", "frequency": "D", "horizon": 2,
        "context_events": [event], "output_dir": str(tmp_path / "first"),
    })
    assert first["context_cache"]["status"] == "stored"
    replay = runner_for("gnomon_forecast")({
        "input": str(wide), "time_column": "timestamp",
        "target_column": "cpu", "frequency": "D", "horizon": 2,
        "context_ref": first["context_ref"],
        "output_dir": str(tmp_path / "replay"),
    })
    assert replay["context_cache"]["status"] == "hit"
    assert replay["context_cache"]["compiler_reused"] is True
    assert replay["results"] == first["results"]


def test_context_ref_conflict_and_namespace_miss_fail_closed(
        tmp_path, monkeypatch) -> None:
    from gnomon.contracts import GnomonError
    from gnomon.toolspec import _materialise_context

    monkeypatch.setenv("GNOMON_CONTEXT_STORE", str(tmp_path / "store"))
    with pytest.raises(GnomonError, match="replaces"):
        _materialise_context({"context_ref": "context_x", "context_events": []})
    with pytest.raises(GnomonError, match="unknown"):
        _materialise_context({"context_ref": "context_missing"})


def test_context_ref_is_refiltered_for_historical_as_of(
        tmp_path, monkeypatch) -> None:
    from datetime import date, timedelta
    from gnomon.toolspec import runner_for

    monkeypatch.setenv("GNOMON_CONTEXT_STORE", str(tmp_path / "store"))
    path = tmp_path / "as-of-context.csv"
    start = date(2026, 1, 1)
    path.write_text("\n".join(["timestamp,cpu"] + [
        f"{start + timedelta(days=day)},{50 + day}" for day in range(60)
    ]) + "\n")
    event = {
        "event_id": "future-announcement", "event_type": "maintenance",
        "entity_scope": ["cpu"],
        "effective_start": "2026-02-05T00:00:00+00:00",
        "effective_end": "2026-02-06T00:00:00+00:00",
        "known_at": "2026-02-01T00:00:00+00:00",
        "source": {"type": "test", "reference": "later-notice"},
    }
    current = runner_for("gnomon_forecast")({
        "input": str(path), "time_column": "timestamp",
        "target_column": "cpu", "frequency": "D", "horizon": 2,
        "context_events": [event], "output_dir": str(tmp_path / "current"),
    })
    replay = runner_for("gnomon_forecast")({
        "input": str(path), "time_column": "timestamp",
        "target_column": "cpu", "frequency": "D", "horizon": 2,
        "as_of": "2026-01-30T00:00:00",
        "context_ref": current["context_ref"],
        "output_dir": str(tmp_path / "historical"),
    })
    assert replay["results"][0].get("context_outcome") in (
        None, {"status": "not_considered", "primary_forecast_changed": False,
               "events": []})


# --- Fix 4: brief by default, hard response budget -------------------------

def _panel_csv(tmp_path, rows_per_series: int = 60):
    # Regular daily grid per series, distinct values.
    lines = ["timestamp,series,value"]
    from datetime import date, timedelta
    start = date(2026, 1, 1)
    for series in ("api", "batch"):
        base = 100 if series == "api" else 500
        for day in range(rows_per_series):
            lines.append(f"{start + timedelta(days=day)},{series},{base + day}")
    path = tmp_path / "panel.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_forecast_defaults_to_brief(monkeypatch, tmp_path) -> None:
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": "examples/daily_requests.csv", "horizon": 3,
        "output_dir": str(tmp_path),
    })
    assert payload["format"] == "brief"
    # Brief still carries the whole epistemic contract.
    result = payload["results"][0]
    assert "support_assessment" in result
    assert "warnings" in result
    assert payload["series_end"] == "2026-02-04T00:00:00"
    assert payload["frequency"] == "D"
    assert payload["verb"] == "forecast"
    assert payload["reasoning"]["sufficiency"] == {
        "sufficient_for": ["forecast:canonical_answer", "explain_support"],
        "further_calls_add_nothing_for": [
            "forecast:canonical_answer", "explain_support"],
        "requires_follow_up": False,
    }
    assert payload["reasoning"]["resolution"]["kind"] == "complete"
    from datetime import datetime
    assert datetime.fromisoformat(payload["wall_clock_now"]).tzinfo is not None
    assert "latest observation" in payload["staleness"]


def test_brief_forecast_previews_both_ends_of_long_horizon(tmp_path) -> None:
    from gnomon.toolspec import FORECAST_PREVIEW_ROWS, runner_for

    payload = runner_for("gnomon_forecast")({
        "input": "examples/daily_requests.csv", "horizon": 24,
        "output_dir": str(tmp_path),
    })
    result = payload["results"][0]
    assert result["forecast_rows"] == 24
    assert len(result["forecast"]) == FORECAST_PREVIEW_ROWS
    assert result["forecast_preview"] == {
        "strategy": "first_and_last",
        "returned_rows": FORECAST_PREVIEW_ROWS,
        "omitted_middle_rows": 12,
        "full_path": "artifact.results[].forecast",
    }
    assert result["forecast"][0]["timestamp"] < result["forecast"][-1]["timestamp"]


def test_multi_series_forecast_respects_the_budget(tmp_path) -> None:
    import csv
    import json

    from gnomon.toolspec import RESPONSE_BUDGET_BYTES, runner_for

    data = _panel_csv(tmp_path)
    payload = runner_for("gnomon_forecast")({
        "input": str(data), "series_column": "series", "horizon": 30,
        "output_dir": str(tmp_path / "out"),
    })
    raw = json.dumps(payload)
    assert payload.get("truncated") is True
    assert "artifact" in payload["truncation"]["note"]
    # The trim leaves headroom-sized responses: within budget plus the
    # protected epistemics and the truncation metadata itself.
    assert len(raw) < RESPONSE_BUDGET_BYTES * 2
    for result in payload["results"]:
        assert result["support_assessment"], "epistemics must survive the trim"
        assert result["forecast_rows"] == 30
        assert len(result["forecast"]) < 30

    # The artifact on disk still carries every row.
    with open(payload["artifact_path"] + "/forecast.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 60


def test_budget_never_trims_the_contract() -> None:
    from gnomon.toolspec import enforce_response_budget

    payload = {
        "status": "complete",
        "artifact_path": "/tmp/a",
        "warnings": [f"warning {index}" for index in range(400)],
        "support_assessment": {
            "reasons": [{"code": "x", "message": "m" * 40}] * 50,
        },
        "bulk": list(range(5000)),
    }
    trimmed = enforce_response_budget(payload)
    assert trimmed["truncated"] is True
    assert trimmed["warnings"] == payload["warnings"]
    assert trimmed["support_assessment"] == payload["support_assessment"]
    assert len(trimmed["bulk"]) == 5
    numeric = next(item for item in trimmed["truncation"]["trimmed"]
                   if item["path"] == "bulk")
    assert numeric["summary"]["min"] == 0
    assert numeric["summary"]["max"] == 4999

    error = {"status": "error", "error": {"code": "X", "message": "y" * 20000}}
    assert enforce_response_budget(error) is error


# --- capabilities within the budget, without hiding anything ---------------

def test_capabilities_brief_fits_the_budget_and_hides_nothing() -> None:
    import json

    from gnomon.runtime import capabilities
    from gnomon.toolspec import CAPABILITIES_RESPONSE_BUDGET_BYTES, runner_for

    full = capabilities()
    brief = runner_for("gnomon_capabilities")({})
    assert len(json.dumps(brief, default=str)) <= CAPABILITIES_RESPONSE_BUDGET_BYTES
    # Every section survives, and every capability NAME survives: a brief
    # view that hid one would make the command lie about the build.
    assert set(full) <= set(brief)
    assert brief["operators"]["available"] == sorted(full["operators"])
    assert brief["macros"]["available"] == sorted(full["macros"])
    assert brief["models"]["baselines"] == full["models"]["baselines"]
    assert brief["models"]["statistical"] == full["models"]["statistical"]
    assert brief["models"]["tsfm_capabilities"]["models"] \
        == sorted(full["models"]["tsfm_capabilities"])
    assert brief["features"] == full["features"]
    assert brief["frequencies"] == full["frequencies"]
    assert brief["mcp_profile"] == full["mcp_profile"]
    # And the view names what was elided and how to get it back.
    view = brief["view"]
    assert view["format"] == "brief"
    assert view["sections_available"] == sorted(full)
    assert view["elided"]
    assert "full" in view["note"] and "sections" in view["note"]


def test_capabilities_full_and_sections_are_verbatim() -> None:
    from gnomon.runtime import capabilities
    from gnomon.toolspec import runner_for

    full = runner_for("gnomon_capabilities")({"format": "full"})
    # The explicit ask gets everything: no budget trim, no truncated flag
    # — cutting a capability list would misreport the build.
    assert full == capabilities()
    assert "truncated" not in full

    section = runner_for("gnomon_capabilities")(
        {"sections": ["models", "workspace"]})
    assert section["models"] == full["models"]
    assert section["workspace"] == full["workspace"]
    assert "operators" not in section
    assert section["view"]["sections_available"] == sorted(full)


def test_unknown_capabilities_section_fails_loudly_with_the_names() -> None:
    import pytest

    from gnomon.contracts import GnomonError
    from gnomon.toolspec import runner_for

    with pytest.raises(GnomonError) as caught:
        runner_for("gnomon_capabilities")({"sections": ["nope"]})
    assert caught.value.code == "INVALID_ARGUMENTS"
    assert "nope" in caught.value.message
    assert "models" in caught.value.details["sections_available"]


def test_budget_never_drops_a_channel_from_a_results_list() -> None:
    import json

    from gnomon.toolspec import RESPONSE_BUDGET_BYTES, enforce_response_budget

    # Six channels — past the head/tail window — each carrying its
    # support state. The trim must descend into the entries, never cut
    # the list: five results out of six is a silently vanished channel.
    payload = {
        "status": "complete", "artifact_path": "/tmp/a",
        "results": [
            {
                "series": f"ch{index}", "support": "supported",
                "support_assessment": {"reasons": []}, "warnings": [],
                "forecast": [
                    {"timestamp": f"2026-01-01T{hour:02d}:00:00+00:00",
                     "q10": 1.0, "q50": 2.0, "q90": 3.0}
                    for hour in range(24)
                ],
            }
            for index in range(6)
        ],
    }
    assert len(json.dumps(payload)) > RESPONSE_BUDGET_BYTES
    trimmed = enforce_response_budget(payload)
    assert trimmed["truncated"] is True
    assert len(trimmed["results"]) == 6
    for result in trimmed["results"]:
        assert result["support"] == "supported"
        assert result["support_assessment"] == {"reasons": []}
        assert len(result["forecast"]) == 5  # the bulk inside still trims


# --- the forecast schema stays terse ---------------------------------------

def test_forecast_schema_is_a_description_diet_not_a_capability_cut() -> None:
    import json

    from gnomon.toolspec import TOOLS

    tool = next(t for t in TOOLS if t["name"] == "gnomon_forecast")
    spec = {"name": tool["name"], "description": tool["description"],
            "parameters": tool["inputSchema"]}
    # The measured pre-diet size was 10,635 characters (~26% of the
    # whole per-round schema payload). Hold the line below 9,000.
    assert len(json.dumps(spec)) < 9_000


def test_forecast_schema_makes_automation_contract_explicit() -> None:
    from gnomon.toolspec import TOOLS

    tool = next(t for t in TOOLS if t["name"] == "gnomon_forecast")
    policy = tool["inputSchema"]["properties"]["automation_policy"]
    assert policy["additionalProperties"] is False
    assert policy["required"] == ["authorize", "policy_id", "minimum_support"]
    assert policy["properties"]["minimum_support"]["enum"] == [
        "supported", "context_trusted"]
    # Every parameter and every enum survives: batching, format,
    # minimum_support, and best_effort stay discoverable from the tool
    # surface alone.
    props = tool["inputSchema"]["properties"]
    assert {"target_column", "horizon", "format", "candidates",
            "minimum_support", "best_effort", "future_events",
            "structural_events", "covariates", "covariates_file",
            "covariate_mapping", "context_events", "context_events_file",
            "repair", "threshold", "project", "as_of",
            "observations"} <= set(props)
    assert props["format"]["enum"] == ["full", "brief"]
    assert props["minimum_support"]["enum"] == [
        "supported", "conditionally_supported", "best_effort"]
    assert props["repair"]["enum"] == ["off", "safe", "aggressive"]
    assert '"auto"' in props["target_column"]["description"]


def test_evidence_pack_schema_meets_the_12kb_experiment_budget(monkeypatch) -> None:
    import json

    from gnomon.toolspec import visible_tools

    monkeypatch.setenv("GNOMON_MCP_PROFILE", "evidence")
    specs = [{"type": "function", "function": {
        "name": tool["name"], "description": tool["description"],
        "parameters": tool["inputSchema"],
    }} for tool in visible_tools()]
    assert len(json.dumps(specs, separators=(",", ":"))) <= 12 * 1024


# --- gnomon_inspect batches a wide file ------------------------------------

def _wide_csv(tmp_path, columns=("cpu", "mem", "requests"), days: int = 30):
    from datetime import date, timedelta

    lines = ["timestamp," + ",".join(columns)]
    start = date(2026, 1, 1)
    for day in range(days):
        lines.append(f"{start + timedelta(days=day)},"
                     + ",".join(str(50 + day + 10 * i)
                                for i in range(len(columns))))
    path = tmp_path / "wide.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_inspect_batches_channels_like_forecast(tmp_path) -> None:
    import json

    from gnomon.toolspec import RESPONSE_BUDGET_BYTES, runner_for

    path = _wide_csv(tmp_path)
    payload = runner_for("gnomon_inspect")(
        {"input": str(path), "target_column": "auto"})
    assert payload["status"] == "valid"
    assert sorted(payload["targets"]) == ["cpu", "mem", "requests"]
    for name, report in payload["targets"].items():
        assert report["status"] == "valid"
        # The combined view labels each report by its column, not the
        # loader's "__default__" placeholder.
        assert report["series"][0]["name"] == name
        assert report["data_quality"]["status"] == "clean"
    assert "gnomon_forecast" in payload["suggested_next"]
    assert "cpu,mem,requests" in payload["suggested_next"]
    assert len(json.dumps(payload, default=str)) < RESPONSE_BUDGET_BYTES

    named = runner_for("gnomon_inspect")(
        {"input": str(path), "target_column": "cpu,requests"})
    assert sorted(named["targets"]) == ["cpu", "requests"]

    single = runner_for("gnomon_inspect")(
        {"input": str(path), "target_column": "mem"})
    assert "targets" not in single  # the single-target shape is unchanged
    assert single["schema"]["target_column"] == "mem"


def test_short_history_threshold_brief_deduplicates_without_losing_evidence(
        tmp_path) -> None:
    import json

    from gnomon.artifacts import read_artifact
    from gnomon.toolspec import RESPONSE_BUDGET_BYTES, runner_for

    source = tmp_path / "short-threshold.csv"
    source.write_text(
        "timestamp,value\n" + "\n".join(
            f"2026-06-{day:02d},{value}" for day, value in enumerate(
                [101, 104, 109, 111, 114, 118, 121, 123], 23)) + "\n",
        encoding="utf-8")
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "time_column": "timestamp",
        "target_column": "value", "horizon": 7, "threshold": 125,
        "format": "brief", "output_dir": str(tmp_path / "out"),
    })
    result = payload["results"][0]
    artifact_result = read_artifact(payload["artifact_path"])["results"][0]
    assert len(json.dumps(payload).encode()) \
        <= RESPONSE_BUDGET_BYTES
    assert "warnings" not in result
    assert {group["message"] for group in payload["limitation_groups"]} == \
        set(artifact_result["warnings"])
    artifact_reason_codes = {
        item["code"] for item in artifact_result["support_assessment"]["reasons"]}
    wire_reason_codes = {
        item["code"] for item in result["support_assessment"]["reasons"]} | \
        set(result["support_assessment"]["grouped_reason_codes"])
    assert wire_reason_codes == artifact_reason_codes
    assert result["threshold"]["bounded_assessment"][
        "automation_eligible"] is False
    assert result["model_assisted"]["primary_forecast_unchanged"] is True
    artifact_disclosures = {
        item["code"] for item in
        artifact_result["support_assessment"]["disclosures"]}
    wire_disclosures = {
        item["code"] for item in
        result["support_assessment"]["disclosures"]}
    assert artifact_disclosures - wire_disclosures == {"model_assisted_lane"}
    assert result["model_assisted"]["location"] == \
        "artifact.results[].model_assisted"


def test_inspect_defaults_to_every_candidate_instead_of_refusing(
        tmp_path) -> None:
    from gnomon.toolspec import runner_for

    # The benchmark's exact failing call: a wide CSV, no target_column.
    # 27 AMBIGUOUS_SCHEMA rounds per sweep said the refusal was the
    # defect — inspection is read-only, so inspect everything and say so.
    path = _wide_csv(tmp_path)
    payload = runner_for("gnomon_inspect")({"input": str(path)})
    assert payload["status"] == "valid"
    assert sorted(payload["targets"]) == ["cpu", "mem", "requests"]
    assert any("all of them were inspected" in item
               for item in payload["assumptions"])


def test_describe_answers_wide_data_without_a_backtest(tmp_path, monkeypatch) -> None:
    from gnomon.toolspec import runner_for

    monkeypatch.setenv("GNOMON_MCP_PROFILE", "describe")
    path = _wide_csv(tmp_path, columns=("cpu", "mem"), days=40)
    payload = runner_for("gnomon_describe")({"input": str(path)})
    assert payload["status"] == "valid"
    assert sorted(payload["reports"]) == ["cpu", "mem"]
    assert payload["reports"]["cpu"]["level"]["latest"] == 89
    assert payload["reports"]["cpu"]["trend"]["direction"] == "up"
    assert "predictive_direction" in payload["reports"]["cpu"][
        "temporal_profile"]["volatility"]
    assert "marginal_variability" in payload["reports"]["cpu"]["temporal_profile"]
    assert payload["reports"]["cpu"]["series_end"] == payload["series_end"]
    assert payload["triage"]["ranking_rule"] == "largest absolute final-step change"
    assert payload["triage"]["remainder_preserved"] is True
    assert payload["suggested_next"] == []


def test_forecast_triage_contract_preserves_remainder():
    from gnomon.toolspec import _attach_multiseries_triage

    payload = _attach_multiseries_triage({
        "artifact_path": "/tmp/artifact",
        "results": [{"series": name, "notability": score, "support": "degraded"}
                    for name, score in (("a", 1), ("b", 4), ("c", 2), ("d", 3))],
    })
    assert payload["triage"]["ranking_rule"]
    assert payload["triage"]["remainder_count"] == 1
    assert payload["triage"]["remainder_preserved"] is True
    # Canonical copyable fields: the winner and the artifact identity are
    # deterministic response fields, not facts an agent must restate.
    assert payload["triage"]["most_notable"] == "b"
    assert payload["triage"]["artifact"]["artifact_path"] == "/tmp/artifact"


def test_wide_response_bounding_refreshes_the_one_canonical_triage_block():
    from gnomon.toolspec import _attach_multiseries_triage, triage_wide_response

    attached = _attach_multiseries_triage({
        "forecast_id": "fc-1", "artifact_path": "/tmp/artifact",
        "results": [{"series": name, "notability": score, "support": "degraded",
                     "support_assessment": {"status": "degraded"}}
                    for name, score in (("a", 1), ("b", 4), ("c", 2),
                                        ("d", 3), ("e", 5))],
    })
    bounded = triage_wide_response(attached)
    assert "series_triage" not in bounded
    triage = bounded["triage"]
    # The attach-time ranking rule survives the bounding pass verbatim.
    assert triage["ranking_rule"] == \
        "threshold crossing, then relative path movement"
    assert triage["most_notable"] == "e"
    assert triage["series_count"] == 5
    assert triage["returned"] == 3
    assert triage["remainder_count"] == 2
    assert triage["remainder_tiers"] == {"degraded": 2}
    assert triage["remainder_preserved"] is True
    assert triage["artifact"] == {"forecast_id": "fc-1",
                                  "artifact_path": "/tmp/artifact"}
    assert [row["series"] for row in bounded["results"]] == ["e", "b", "d"]


def test_budget_trim_never_drops_the_triage_disclosures():
    from gnomon.toolspec import enforce_response_budget, triage_wide_response

    payload = triage_wide_response({
        "artifact_path": "/tmp/artifact",
        "results": [{"series": f"s{index}", "notability": index,
                     "support": "degraded",
                     "bulk": list(range(400))}
                    for index in range(8)],
    })
    trimmed = enforce_response_budget(payload, budget_bytes=1024)
    assert trimmed["truncated"] is True
    assert trimmed["triage"] == payload["triage"]


def test_describe_spike_response_is_json_serializable(tmp_path, monkeypatch):
    import json
    from gnomon.toolspec import runner_for

    monkeypatch.setenv("GNOMON_MCP_PROFILE", "describe")
    path = _wide_csv(tmp_path, columns=("api", "worker", "db"), days=12)
    payload = runner_for("gnomon_describe")({"input": str(path)})
    assert json.loads(json.dumps(payload))["triage"]["remainder_preserved"] is True


def test_forecast_returns_canonical_temporal_facts_without_artifact_read(
        tmp_path) -> None:
    import json
    from pathlib import Path
    from gnomon.toolspec import runner_for

    path = _wide_csv(tmp_path, columns=("cpu",), days=40)
    payload = runner_for("gnomon_forecast")({
        "input": str(path), "horizon": 2,
        "output_dir": str(tmp_path / "canonical-facts"),
    })
    facts = payload["results"][0]["temporal_facts"]
    assert facts["seasonal_period_steps"] == 7
    assert facts["seasonal_period_label"] == "weekly"
    assert "predictive_direction" in facts["temporal_profile"]["volatility"]
    assert "marginal_variability" in facts["temporal_profile"]
    assert facts["source"] == "computed_from_observations"
    identity = payload["results"][0]["execution_identity"]
    assert identity["publish_matches_evaluated"] is True
    assert identity["evaluated"]["name"] == identity["published"]["name"]
    artifact = json.loads(
        (Path(payload["artifact_path"]) / "artifact.json").read_text())
    # Merely requesting a forecast must not mutate the frozen artifact
    # contract. Rich reasoning evidence is opt-in through typed questions.
    assert not [item for item in artifact["evidence"]
                if item["kind"] == "temporal_profile"]


def test_typed_question_returns_compact_answer_without_changing_primary(
        tmp_path) -> None:
    import json
    from pathlib import Path

    from gnomon.toolspec import runner_for

    path = _wide_csv(tmp_path, columns=("cpu",), days=80)
    plain = runner_for("gnomon_forecast")({
        "input": str(path), "horizon": 2,
        "output_dir": str(tmp_path / "plain"),
    })
    asked = runner_for("gnomon_forecast")({
        "input": str(path), "horizon": 2,
        "output_dir": str(tmp_path / "asked"),
        "questions": [{"id": "v1", "verb": "predict", "target": "cpu",
                           "property": "volatility", "horizon": 7}],
    })
    replay = runner_for("gnomon_forecast")({
        "input": str(path), "horizon": 2,
        "output_dir": str(tmp_path / "replay"),
        "questions": [{"id": "v1", "verb": "predict", "target": "cpu",
                       "property": "volatility", "horizon": 7}],
    })
    assert asked["results"] == plain["results"]
    assert asked["answers"][0]["question"]["target"] == "cpu"
    assert asked["answers"][0]["artifact_id"] == asked["artifact_id"]
    assert set(asked["answers"][0]) == {
        "question", "best_estimate", "synthesis_policy", "decision_rule",
        "answer", "headline", "limitations", "artifact_id", "support"}
    policy = asked["answers"][0]["synthesis_policy"]
    assert policy["canonical_immutable"] is True
    assert policy["synthesis_must_be_separately_labelled"] is True
    assert policy["publication_default"] in {"canonical", "labelled_synthesis"}
    assert asked["answers"][0]["best_estimate"]["value"] == \
        asked["answers"][0]["answer"]["direction"]
    # The full evidence remains in temporal_answers.json; the turn-by-turn
    # answer must stay small enough to quote without repaying receipt bulk.
    assert len(json.dumps(asked["answers"][0], sort_keys=True)) < 6000
    assert "calibration" not in asked["answers"][0]
    assert asked["answers"][0]["answer"]["property_distribution"]["quantity"] == \
        "future_to_reference_residual_scale_ratio"
    assert asked["answer_cache"]["status"] == "stored"
    assert replay["answer_cache"]["status"] == "hit"
    assert replay["answers"] == asked["answers"]
    receipt = json.loads(Path(asked["answer_receipt"]).read_text())
    assert receipt["answers"][0]["calibration"]["direction_candidate"]
    assert receipt["primary_forecast_unchanged"] is True


def test_typed_aggregate_binds_panel_series_not_value_column(tmp_path) -> None:
    import json
    from datetime import date, timedelta
    from pathlib import Path
    from gnomon.toolspec import runner_for

    source = tmp_path / "panel.csv"
    start = date(2026, 1, 1)
    rows = ["timestamp,series,value"]
    for index in range(80):
        stamp = start + timedelta(days=index)
        rows.append(f"{stamp},cpu,{10 + index % 7}")
        rows.append(f"{stamp},mem,{20 + (index % 5) * 2}")
    source.write_text("\n".join(rows) + "\n")

    result = runner_for("gnomon_forecast")({
        "input": str(source), "time_column": "timestamp",
        "target_column": "value", "series_column": "series",
        "frequency": "D", "horizon": 3,
        "output_dir": str(tmp_path / "out"),
        "questions": [{
            "id": "fleet", "verb": "predict", "property": "volatility",
            "target": {"kind": "aggregate", "members": ["cpu", "mem"]},
            "horizon": 3,
        }],
    })

    assert result["answers"][0]["question"]["target"]["members"] == [
        "cpu", "mem"]
    assert result["answers"][0]["decision_rule"]["aggregation"] == \
        "median_normalized_scale_ratio"
    receipt = json.loads(Path(result["answer_receipt"]).read_text())
    assert receipt["primary_forecast_unchanged"] is True
    assert receipt["artifact_id"] == result["artifact_id"]


def test_daily_requests_predictive_volatility_is_strict_json(tmp_path) -> None:
    """Regression: missing Brier diagnostics once leaked IEEE infinity.

    MCP's strict encoder then crashed and mislabeled the valid forecast as a
    tracking error.  The real example fixture pins the complete answer path.
    """
    import json
    from pathlib import Path
    from gnomon.mcp_server import _handle

    source = Path(__file__).resolve().parents[1] / "examples" / "daily_requests.csv"
    result = _handle({"method": "tools/call", "params": {
        "name": "gnomon_forecast", "arguments": {
            "input": str(source), "time_column": "timestamp",
            "target_column": "requests", "frequency": "D", "horizon": 14,
            "output_dir": str(tmp_path / "daily-volatility"),
            "questions": [{"id": "v", "verb": "predict",
                           "target": "requests", "property": "volatility",
                           "horizon": 14}],
        }}})
    encoded = json.dumps(result, allow_nan=False)
    assert result["isError"] is False
    answer = result["structuredContent"]["answers"][0]["answer"]
    assert answer["direction"] in {
        "increased", "decreased", "stable", "uncertain"}
    assert "Infinity" not in encoded and "NaN" not in encoded


def test_irregular_grid_error_carries_literal_aggressive_repair_call(tmp_path) -> None:
    from gnomon.contracts import GnomonError
    from gnomon.toolspec import runner_for

    source = tmp_path / "irregular.csv"
    source.write_text(
        "timestamp,value\n"
        "2026-01-01T00:00:00+00:00,1\n"
        "2026-01-01T01:00:00+00:00,2\n"
        "2026-01-01T02:30:00+00:00,3\n",
        encoding="utf-8")
    with pytest.raises(GnomonError) as caught:
        runner_for("gnomon_forecast")({
            "input": str(source), "time_column": "timestamp",
            "target_column": "value", "frequency": "h", "horizon": 1,
        })
    repair = caught.value.to_dict()["error"]["repair_options"][0]
    assert repair["tool_call"]["name"] == "gnomon_forecast"
    assert repair["tool_call"]["arguments"]["repair"] == "aggressive"


def test_multi_target_forecast_preserves_typed_questions(tmp_path) -> None:
    from gnomon.toolspec import runner_for

    path = _wide_csv(tmp_path, columns=("cpu", "mem"), days=80)
    payload = runner_for("gnomon_forecast")({
        "input": str(path), "target_column": "cpu,mem", "horizon": 2,
        "output_dir": str(tmp_path / "multi-asked"),
        "questions": [
            {"id": "cpu-v", "verb": "predict", "target": "cpu",
             "property": "volatility", "horizon": 7},
            {"id": "mem-v", "verb": "predict", "target": "mem",
             "property": "volatility", "horizon": 7},
        ],
    })
    assert [item["question"]["id"] for item in payload["answers"]] == [
        "cpu-v", "mem-v"]
    assert all(item["artifact_id"] == payload["artifact_id"]
               for item in payload["answers"])


def test_cross_unit_aggregate_preserves_per_series_evidence(
        tmp_path, monkeypatch) -> None:
    from gnomon.toolspec import runner_for

    monkeypatch.setenv("GNOMON_MCP_PROFILE", "evidence")
    path = _wide_csv(tmp_path, columns=("cpu", "mem"), days=100)
    payload = runner_for("gnomon_describe")({
        "input": str(path), "target_column": "cpu,mem",
        "questions": [{
            "id": "fleet-v", "verb": "predict", "property": "volatility",
            "horizon": 7, "target": {
                "kind": "aggregate", "members": ["cpu", "mem"],
                "aggregation": "median_normalized_scale_ratio"},
        }],
    })
    answer = payload["answers"][0]
    assert answer["question"]["target"]["aggregation"] == \
        "median_normalized_scale_ratio"
    assert [item["question"]["target"] for item in answer["per_series"]] == [
        "cpu", "mem"]


def test_describe_brief_projects_typed_answers_within_agent_budget(
        tmp_path, monkeypatch) -> None:
    import json
    from gnomon.toolspec import runner_for

    monkeypatch.setenv("GNOMON_MCP_PROFILE", "evidence")
    path = _wide_csv(tmp_path, columns=("cpu", "mem"), days=100)
    payload = runner_for("gnomon_describe")({
        "input": str(path), "target_column": "cpu,mem", "format": "brief",
        "questions": [{
            "id": "fleet-v", "verb": "predict", "property": "volatility",
            "horizon": 7, "target": {
                "kind": "aggregate", "members": ["cpu", "mem"],
                "aggregation": "median_normalized_scale_ratio"},
        }],
    })

    assert payload["view"]["format"] == "brief"
    assert payload["answers"][0]["headline"]
    assert "per_series" not in payload["answers"][0]
    assert payload["answers"][0]["reasoning"][
        "primary_forecast_unchanged"] is True
    assert len(json.dumps(payload, sort_keys=True)) < 12_000


def test_mega_profile_is_three_tools_and_runs_a_descriptive_question(
        tmp_path, monkeypatch) -> None:
    from gnomon.toolspec import runner_for, visible_tools

    monkeypatch.setenv("GNOMON_MCP_PROFILE", "mega")
    assert {tool["name"] for tool in visible_tools()} == {
        "gnomon_inspect", "gnomon_run", "gnomon_track"}
    path = _wide_csv(tmp_path, columns=("cpu",), days=40)
    inspected = runner_for("gnomon_inspect")({"input": str(path)})
    described = runner_for("gnomon_run")({
        "data_ref": inspected["data_ref"],
        "question": {"kind": "describe"},
    })
    assert described["reports"]["cpu"]["trend"]["direction"] == "up"
    # Lenient discriminant form seen in real agent transcripts.
    described_short = runner_for("gnomon_run")({
        "data_ref": inspected["data_ref"], "question": "describe",
    })
    assert described_short["reports"]["cpu"]["trend"]["direction"] == "up"
    forecast = runner_for("gnomon_run")({
        "data_ref": inspected["data_ref"],
        "question": {"kind": "forecast"},
        "horizon": 3,
        "output_dir": str(tmp_path / "mega-output"),
    })
    assert forecast["status"] == "complete"
    assert forecast["data_ref"] == inspected["data_ref"]
    assert forecast["results"][0]["forecast_rows"] == 3


def test_data_ref_reuses_resolved_data_and_schema_across_verbs(tmp_path) -> None:
    from gnomon.toolspec import runner_for

    path = _wide_csv(tmp_path, columns=("cpu",), days=40)
    inspected = runner_for("gnomon_inspect")({
        "input": str(path), "target_column": "cpu",
    })
    data_ref = inspected["data_ref"]
    assert data_ref.startswith("data_")

    forecast = runner_for("gnomon_forecast")({
        "data_ref": data_ref,
        "horizon": 3,
        "output_dir": str(tmp_path / "out"),
    })
    assert forecast["data_ref"] == data_ref
    assert forecast["status"] in {"complete", "abstained"}


def test_unknown_data_ref_returns_a_recoverable_error() -> None:
    import pytest

    from gnomon.contracts import GnomonError
    from gnomon.toolspec import runner_for

    with pytest.raises(GnomonError) as caught:
        runner_for("gnomon_inspect")({"data_ref": "data_expired"})
    assert caught.value.code == "INVALID_ARGUMENTS"
    assert caught.value.repair_options[0]["action"] == "resupply_data"


def test_data_ref_rejects_a_conflicting_schema_override(tmp_path) -> None:
    import pytest

    from gnomon.contracts import GnomonError
    from gnomon.toolspec import runner_for

    path = _wide_csv(tmp_path, columns=("cpu", "mem"))
    inspected = runner_for("gnomon_inspect")({
        "input": str(path), "target_column": "cpu",
    })
    with pytest.raises(GnomonError) as caught:
        runner_for("gnomon_forecast")({
            "data_ref": inspected["data_ref"],
            "target_column": "mem",
            "horizon": 3,
        })
    assert caught.value.code == "INVALID_ARGUMENTS"
    assert caught.value.details["conflicts"] == ["target_column"]


def test_inline_observations_have_an_explicit_wire_budget() -> None:
    import pytest

    from gnomon.contracts import GnomonError
    from gnomon.toolspec import runner_for

    rows = [{"timestamp": f"2026-01-{(i % 28) + 1:02d}", "value": i}
            for i in range(501)]
    with pytest.raises(GnomonError) as caught:
        runner_for("gnomon_inspect")({"observations": rows})
    assert caught.value.details == {"observations": 501, "maximum": 500}


def test_session_data_refs_are_bounded_and_oldest_expires(monkeypatch) -> None:
    import pytest

    from gnomon.contracts import GnomonError
    from gnomon import toolspec

    monkeypatch.setattr(toolspec, "_MAX_SESSION_DATA_REFS", 2)
    toolspec._SESSION_DATA_REFS.clear()
    _, first = toolspec._register_data_ref({"input": "first.csv"})
    toolspec._register_data_ref({"input": "second.csv"})
    toolspec._register_data_ref({"input": "third.csv"})
    assert len(toolspec._SESSION_DATA_REFS) == 2
    with pytest.raises(GnomonError) as caught:
        toolspec._resolve_data_ref({"data_ref": first})
    assert caught.value.repair_options[0]["action"] == "resupply_data"


def test_inspect_multi_reports_a_failing_channel_in_place(tmp_path) -> None:
    from datetime import date, timedelta

    from gnomon.toolspec import runner_for

    lines = ["timestamp,cpu,note"]
    start = date(2026, 1, 1)
    for day in range(30):
        lines.append(f"{start + timedelta(days=day)},{50 + day},flaky")
    path = tmp_path / "mixed.csv"
    path.write_text("\n".join(lines) + "\n")

    payload = runner_for("gnomon_inspect")(
        {"input": str(path), "target_column": "cpu,note"})
    assert payload["status"] == "partial"
    assert payload["targets"]["cpu"]["status"] == "valid"
    failing = payload["targets"]["note"]
    assert failing["status"] == "error"
    assert failing["error"]["code"]
    assert "cpu" in payload["suggested_next"]
    assert "note" not in payload["suggested_next"].split('"')[1]


def test_agent_response_contract_groups_limitations_and_names_tier_floor():
    from gnomon.toolspec import apply_response_contract

    payload = apply_response_contract({
        "status": "complete", "forecast_id": "fc_1",
        "results": [
            {"series": "api", "support": "supported", "warnings": [],
             "support_assessment": {"status": "supported",
                                    "recovery_actions": []}},
            {"series": "worker", "support": "weakly_supported",
             "warnings": ["pooled residuals"],
             "support_assessment": {
                 "status": "conditionally_supported",
                 "recovery_actions": [{"code": "provide_more_history",
                                       "message": "Add observations."}]}},
            {"series": "billing", "support": "weakly_supported",
             "warnings": ["pooled residuals"],
             "support_assessment": {"status": "conditionally_supported",
                                    "recovery_actions": []}},
        ],
    })

    assert payload["artifact_id"] == "fc_1"
    assert payload["tier_floor"] == "conditionally_supported"
    [group] = payload["limitation_groups"]
    assert group["affected_series_count"] == 2
    assert group["examples"] == ["worker", "billing"]
    assert payload["recovery_actions"][0]["code"] == "provide_more_history"


def test_response_contract_fields_cannot_be_trimmed():
    from gnomon.toolspec import enforce_response_budget

    payload = {
        "headline": "Quote this.",
        "tier_floor": "conditionally_supported",
        "limitation_groups": [{"code": "X", "message": "Keep this."}],
        "recovery_actions": [{"code": "retry", "message": "Try again."}],
        "artifact_id": "fc_1", "artifact_path": "/tmp/fc_1",
        "bulk": list(range(10_000)),
    }
    trimmed = enforce_response_budget(payload)
    for key in ("headline", "tier_floor", "limitation_groups",
                "recovery_actions", "artifact_id", "artifact_path"):
        assert trimmed[key] == payload[key]


def test_ambiguous_forecast_repairs_are_ready_to_issue(tmp_path):
    import pytest

    from gnomon.contracts import GnomonError
    from gnomon.toolspec import runner_for

    path = _wide_csv(tmp_path, columns=("cpu", "mem"))
    with pytest.raises(GnomonError) as caught:
        runner_for("gnomon_forecast")({"input": str(path), "horizon": 3})
    repairs = caught.value.repair_options
    assert repairs
    for repair in repairs:
        call = repair["tool_call"]
        assert call["name"] == "gnomon_forecast"
        assert call["arguments"]["input"] == str(path)
        assert call["arguments"]["horizon"] == 3
        assert call["arguments"]["target_column"] in {"cpu", "mem", "auto"}
