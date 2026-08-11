"""Tool-surface gating and token discipline.

The MCP surface competes for model attention and context tokens; these
tests pin the subtraction: deprecated and near-duplicate tools stay off
the default surface (returning only under GNOMON_V02_COMPAT=1), and no
default response may balloon past the size budget.
"""

from __future__ import annotations


def _names(monkeypatch, compat: bool = False) -> list[str]:
    from gnomon.toolspec import visible_tools

    if compat:
        monkeypatch.setenv("GNOMON_V02_COMPAT", "1")
    else:
        monkeypatch.delenv("GNOMON_V02_COMPAT", raising=False)
    return [tool["name"] for tool in visible_tools()]


# --- Fix 1: the deprecated decision pair is opt-in -------------------------

def test_deprecated_decision_pair_hidden_by_default(monkeypatch) -> None:
    from gnomon.toolspec import runner_for

    names = _names(monkeypatch)
    assert "gnomon_record_decision" not in names
    assert "gnomon_resolve_decision" not in names
    assert runner_for("gnomon_record_decision") is None
    assert runner_for("gnomon_resolve_decision") is None


def test_v02_compat_restores_decision_pair(monkeypatch, tmp_path) -> None:
    from gnomon.toolspec import runner_for
    from gnomon.tracking import TrackingStore

    names = _names(monkeypatch, compat=True)
    assert "gnomon_record_decision" in names
    assert "gnomon_resolve_decision" in names

    # They do not just appear — they still function end to end.
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "registry.db"))
    TrackingStore().register(
        forecast_id="fc_compat", project="compat", selected_model="drift",
        support="supported", artifact_path=str(tmp_path),
    )
    record = runner_for("gnomon_record_decision")({
        "decision_id": "d1", "project": "compat", "forecast_id": "fc_compat",
        "action": "scale_up", "expected_outcome": "load absorbed",
    })
    assert record["status"] == "ok"
    resolved = runner_for("gnomon_resolve_decision")({
        "decision_id": "d1", "actual_outcome": "load absorbed",
        "correct": True,
    })
    assert resolved["status"] == "ok"
    assert resolved["decision"]["correct"] is True


# --- Fix 2: the tracking reads consolidate into gnomon_status --------------

def test_tracking_read_tools_hidden_by_default(monkeypatch) -> None:
    names = _names(monkeypatch)
    assert "gnomon_list_open_forecasts" not in names
    assert "gnomon_model_performance" not in names
    assert "gnomon_status" in names
    compat_names = _names(monkeypatch, compat=True)
    assert "gnomon_list_open_forecasts" in compat_names
    assert "gnomon_model_performance" in compat_names


def test_status_sections_match_the_retired_tools(monkeypatch, tmp_path) -> None:
    from gnomon.toolspec import (
        _run_model_performance,
        _run_open_forecasts,
        _run_status,
    )
    from gnomon.tracking import TrackingStore

    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "registry.db"))
    TrackingStore().register(
        forecast_id="fc_status", project="status", selected_model="drift",
        support="supported", artifact_path=str(tmp_path), horizon=3,
    )
    args = {"project": "status"}
    assert _run_status({**args, "section": "open_forecasts"}) \
        == _run_open_forecasts(args)
    assert _run_status({**args, "section": "performance"}) \
        == _run_model_performance(args)

    # The decisions slice is exactly the decision fields of the full view.
    full = _run_status(args)
    decisions = _run_status({**args, "section": "decisions"})
    assert decisions["unresolved_decisions"] == full["unresolved_decisions"]
    assert decisions["decision_summary"] == full["decision_summary"]
    assert "open_forecasts" not in decisions

    # performance still requires a project, as the standalone tool did.
    import pytest

    from gnomon.contracts import GnomonError
    with pytest.raises(GnomonError):
        _run_status({"section": "performance"})


# --- Fix 3: guide/proposer/duplicate tools leave the default surface -------

def test_covariate_guide_and_proposer_tools_gated(monkeypatch) -> None:
    names = _names(monkeypatch)
    for gated in ("gnomon_covariate_guide", "gnomon_propose_covariates",
                  "gnomon_proposer_skill"):
        assert gated not in names, gated
    # The survivors still cover the workflow: validation carries the
    # format contract, and the forecast takes every covariate argument.
    from gnomon.toolspec import TOOLS
    by_name = {tool["name"]: tool for tool in TOOLS}
    validate = by_name["gnomon_validate_covariates"]["description"]
    assert "known_at" in validate
    assert "name:type:future_known" in validate
    forecast_props = by_name["gnomon_forecast"]["inputSchema"]["properties"]
    assert "covariates_file" in forecast_props
    assert "covariate_mapping" in forecast_props

    compat = set(_names(monkeypatch, compat=True))
    for gated in ("gnomon_covariate_guide", "gnomon_propose_covariates",
                  "gnomon_proposer_skill"):
        assert gated in compat, gated


# --- Fix 7: task profiles narrow the surface -------------------------------

def test_profiles_select_documented_subsets(monkeypatch) -> None:
    from gnomon.toolspec import PROFILES, visible_tools

    monkeypatch.delenv("GNOMON_V02_COMPAT", raising=False)
    monkeypatch.delenv("GNOMON_MCP_PROFILE", raising=False)
    full = {tool["name"] for tool in visible_tools()}

    for profile, expected in PROFILES.items():
        monkeypatch.setenv("GNOMON_MCP_PROFILE", profile)
        names = {tool["name"] for tool in visible_tools()}
        assert names == expected, profile
        assert names <= full

    # A narrowed profile narrows everything: compat tools appear only
    # under full.
    monkeypatch.setenv("GNOMON_MCP_PROFILE", "core")
    monkeypatch.setenv("GNOMON_V02_COMPAT", "1")
    names = {tool["name"] for tool in visible_tools()}
    assert "gnomon_record_decision" not in names

    monkeypatch.setenv("GNOMON_MCP_PROFILE", "full")
    monkeypatch.delenv("GNOMON_V02_COMPAT", raising=False)
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
    assert payload["active"] == "full"
    assert payload["available"] == ["core", "data", "decision", "full"]

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


def test_capabilities_reports_compat_state(monkeypatch) -> None:
    from gnomon.runtime import capabilities

    monkeypatch.delenv("GNOMON_V02_COMPAT", raising=False)
    assert capabilities()["compat"]["v02_tools"] is False
    monkeypatch.setenv("GNOMON_V02_COMPAT", "1")
    assert capabilities()["compat"]["v02_tools"] is True


# --- capabilities within the budget, without hiding anything ---------------

def test_capabilities_brief_fits_the_budget_and_hides_nothing() -> None:
    import json

    from gnomon.runtime import capabilities
    from gnomon.toolspec import RESPONSE_BUDGET_BYTES, runner_for

    full = capabilities()
    brief = runner_for("gnomon_capabilities")({})
    assert len(json.dumps(brief, default=str)) <= RESPONSE_BUDGET_BYTES
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
