"""All three front doors reach the same product.

The CLI had five verbs, the bitemporal store, and `--as-of`. MCP had four
verbs and no store at all. The Python API had one verb. A front door that
reaches less than the product is a front door onto a different product.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import json

import pytest

from gnomon.ids import FixedClock

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))
REPO = Path(__file__).resolve().parent.parent


def _csv(tmp_path: Path, periods: int = 90) -> Path:
    noise = [0.5, -0.3, 0.2, -0.4, 0.1, 0.3, -0.2, -0.1, 0.4, -0.5]
    start = date(2026, 1, 1)
    rows = [
        f"{(start + timedelta(days=i)).isoformat()},{100 + i * 0.4 + noise[i % 10] * 6:.4f}"
        for i in range(periods)
    ]
    path = tmp_path / "series.csv"
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _call(name: str, arguments: dict):
    from gnomon import mcp_server

    result = mcp_server._handle({
        "method": "tools/call", "params": {"name": name, "arguments": arguments},
    })
    return json.loads(result["content"][0]["text"]), result["isError"]


# -- H10: the Python API is the product, not a corner of it --------------

def test_python_api_exports_all_five_verbs():
    import gnomon

    for verb in ("forecast", "investigate_change", "decide", "monitor",
                 "detect_anomalies"):
        assert verb in gnomon.__all__, f"{verb} is not a Python API export"
        assert callable(getattr(gnomon, verb))


def test_python_api_exposes_the_stores():
    import gnomon

    assert "TemporalStore" in gnomon.__all__
    assert "TrackingStore" in gnomon.__all__


def test_python_api_verbs_actually_run(tmp_path):
    import gnomon

    source = str(_csv(tmp_path))
    common = dict(
        time_column="timestamp", target_column="value",
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    payload, path = gnomon.investigate_change(source, **common)
    assert (path / "artifact.json").is_file()
    assert payload["investigation_id"]

    payload, _ = gnomon.decide(
        source, horizon=5, threshold=120.0,
        actions=[{"name": "scale_up"}, {"name": "do_nothing"}], **common,
    )
    assert payload["decision_id"]

    payload, _ = gnomon.monitor(
        source, horizon=5, threshold=120.0, alert_cost=1.0, miss_cost=20.0, **common,
    )
    assert payload["monitor_id"]

    payload, _ = gnomon.detect_anomalies(source, include_tsfm=False, **common)
    assert payload["anomaly_id"]


# -- H9: MCP reaches the bitemporal store --------------------------------

def test_mcp_exposes_ingest_and_list_datasets():
    from gnomon.toolspec import TOOLS

    names = {tool["name"] for tool in TOOLS}
    assert {"gnomon_ingest", "gnomon_list_datasets"} <= names


def test_mcp_forecast_and_inspect_accept_as_of():
    from gnomon.toolspec import TOOLS

    for name in ("gnomon_forecast", "gnomon_inspect"):
        tool = next(item for item in TOOLS if item["name"] == name)
        properties = tool["inputSchema"]["properties"]
        assert "as_of" in properties, f"{name} cannot replay"
        assert "store_path" in properties, f"{name} cannot reach a store"
    forecast = next(item for item in TOOLS if item["name"] == "gnomon_forecast")
    assert "store:" in forecast["inputSchema"]["properties"]["input"]["description"]


def test_the_whole_vintage_workflow_runs_over_mcp(tmp_path):
    """ingest → list → forecast at an as_of → inspect, agent-side only."""
    store_path = str(tmp_path / "store.db")
    source = str(REPO / "examples" / "messy_requests_revisions.csv")

    report, error = _call("gnomon_ingest", {
        "input": source, "dataset": "requests", "time_column": "timestamp",
        "target_column": "requests", "known_at_column": "published",
        "store_path": store_path,
    })
    assert not error, report
    assert report["rows_added"] == report["rows_seen"]
    assert report["revisions_created"] > 0

    listing, error = _call("gnomon_list_datasets", {"store_path": store_path})
    assert not error
    dataset = listing["datasets"][0]
    assert dataset["input_ref"] == "store:requests"
    assert dataset["known_time_provenance"] == "recorded"

    payload, error = _call("gnomon_forecast", {
        "input": "store:requests", "time_column": "timestamp",
        "target_column": "requests", "horizon": 7, "as_of": "2026-06-03",
        "store_path": store_path, "output_dir": str(tmp_path / "out"),
    })
    assert not error, payload
    result = payload["results"][0]
    assert result["support"] == "supported"
    assert result["forecast_rows"] == 7

    inspection, error = _call("gnomon_inspect", {
        "input": "store:requests", "time_column": "timestamp",
        "target_column": "requests", "as_of": "2026-06-03",
        "store_path": store_path,
    })
    assert not error
    assert inspection["status"] == "valid"


def test_as_of_over_mcp_actually_restricts_what_is_seen(tmp_path):
    store_path = str(tmp_path / "store.db")
    source = str(REPO / "examples" / "messy_requests_revisions.csv")
    _call("gnomon_ingest", {
        "input": source, "dataset": "requests", "time_column": "timestamp",
        "target_column": "requests", "known_at_column": "published",
        "store_path": store_path,
    })
    early, _ = _call("gnomon_inspect", {
        "input": "store:requests", "time_column": "timestamp",
        "target_column": "requests", "as_of": "2026-05-01",
        "store_path": store_path,
    })
    late, _ = _call("gnomon_inspect", {
        "input": "store:requests", "time_column": "timestamp",
        "target_column": "requests", "store_path": store_path,
    })
    assert early["series"][0]["observations"] < late["series"][0]["observations"]


# -- H17: the router's answer has a consumer -----------------------------

def test_forecast_accepts_the_routers_candidates(tmp_path):
    import gnomon.runtime as runtime_module

    seen: dict = {}
    original = runtime_module.evaluate_stage

    def spy(state, **kwargs):
        seen["config"] = kwargs.get("config")
        return original(state, **kwargs)

    runtime_module.evaluate_stage = spy
    try:
        artifact, _ = runtime_module.forecast(
            str(_csv(tmp_path)), time_column="timestamp", target_column="value",
            horizon=7, output=str(tmp_path / "out"), clock=CLOCK,
            candidates=["drift", "theta"],
        )
    finally:
        runtime_module.evaluate_stage = original

    assert seen["config"].models.statistical_candidates == ["drift", "theta"]
    scored = {
        name for name, value in artifact.results[0].selection_scores.items()
        if value is not None
    }
    assert scored <= {"drift", "theta", "last_value", "seasonal_naive"}
    assert "ets" not in scored, "a model outside the pool competed"


def test_baselines_survive_a_candidate_restriction(tmp_path):
    from gnomon.runtime import forecast

    artifact, _ = forecast(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=7, output=str(tmp_path / "out"), clock=CLOCK,
        candidates=["drift"],
    )
    assert artifact.results[0].strongest_baseline in {"last_value", "seasonal_naive"}


def test_unknown_candidate_is_refused(tmp_path):
    from gnomon.contracts import GnomonError
    from gnomon.runtime import forecast

    with pytest.raises(GnomonError) as raised:
        forecast(
            str(_csv(tmp_path)), time_column="timestamp", target_column="value",
            horizon=7, output=str(tmp_path / "out"), clock=CLOCK,
            candidates=["prophet"],
        )
    assert raised.value.code == "UNKNOWN_MODEL"


def test_route_names_its_consumer():
    from gnomon.toolspec import TOOLS

    route = next(item for item in TOOLS if item["name"] == "gnomon_route")
    assert "gnomon_forecast" in route["description"]
    forecast = next(item for item in TOOLS if item["name"] == "gnomon_forecast")
    assert "candidates" in forecast["inputSchema"]["properties"]


# -- H23: investigate answers in the reader's terms ----------------------

def test_investigate_ranks_changes_and_marks_the_dominant_one(tmp_path):
    from gnomon.macros import investigate_change

    payload, path = investigate_change(
        str(REPO / "examples" / "messy_requests.csv"),
        time_column="timestamp", target_column="requests",
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    changes = payload["results"][0]["ranked_changes"]
    assert changes, "no changes were reported"
    shares = [item["share_of_variation_explained"] for item in changes]
    assert shares == sorted(shares, reverse=True), "changes are not ranked"
    assert changes[0]["dominant"] is True
    assert not changes[-1]["dominant"], "a noise-level change was called dominant"
    # Timestamps, not model-internal indices.
    assert "index" not in changes[0]
    datetime.fromisoformat(changes[0]["timestamp"])


def test_every_macro_writes_a_summary(tmp_path):
    from gnomon.macros import decide, detect_anomalies, investigate_change, monitor

    source = str(_csv(tmp_path))
    common = dict(
        time_column="timestamp", target_column="value",
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    paths = [
        investigate_change(source, **common)[1],
        detect_anomalies(source, include_tsfm=False, **common)[1],
        decide(source, horizon=5, threshold=120.0,
               actions=[{"name": "scale_up"}, {"name": "do_nothing"}], **common)[1],
        monitor(source, horizon=5, threshold=120.0,
                alert_cost=1.0, miss_cost=20.0, **common)[1],
    ]
    for path in paths:
        summary = path / "summary.md"
        assert summary.is_file(), f"{path.name} has no summary.md"
        text = summary.read_text(encoding="utf-8")
        assert text.startswith("# "), text[:80]
        assert len(text.splitlines()) > 2


# -- M3: adapted_alpha is reachable --------------------------------------

def test_coverage_adaptation_is_reachable_and_says_it_is_not_applied(tmp_path):
    """A complete, tested capability with zero call sites outside its tests."""
    from gnomon.tracking import TrackingStore

    store = TrackingStore(tmp_path / "registry.db")
    store.record_coverage_outcome(
        "sre-demo", "__default__", "fc_1", "2026-01-05T00:00:00", covered=10, points=10,
    )
    store.record_coverage_outcome(
        "sre-demo", "__default__", "fc_2", "2026-01-12T00:00:00", covered=9, points=10,
    )
    scopes = store.coverage_adaptation("sre-demo")
    assert [item["scope"] for item in scopes] == ["__default__"]
    assert scopes[0]["observations"] == 2
    assert 0 < scopes[0]["alpha"] < 1
    assert scopes[0]["applied_to_published_intervals"] is False
    assert store.status("sre-demo")["coverage_adaptation"] == scopes


def test_track_coverage_is_a_cli_command(tmp_path, capsys, monkeypatch):
    import gnomon.tracking as tracking_module
    from gnomon.cli import main
    from gnomon.tracking import TrackingStore

    registry = tmp_path / "registry.db"
    # DEFAULT_REGISTRY_PATH is resolved at import, so the env var is read
    # too early to redirect a store the CLI constructs with no argument.
    monkeypatch.setattr(tracking_module, "DEFAULT_REGISTRY_PATH", registry)
    store = TrackingStore(registry)
    store.record_coverage_outcome(
        "sre-demo", "__default__", "fc_1", "2026-01-05T00:00:00", covered=8, points=10,
    )
    assert main(["track", "coverage", "--project", "sre-demo"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scopes"][0]["scope"] == "__default__"
    assert "not applied" in payload["note"]


# -- H22: install.sh can install the checkout ----------------------------

def test_install_script_has_a_local_mode():
    script = (REPO / "install.sh").read_text(encoding="utf-8")
    assert "--local" in script
    assert "SCRIPT_DIR" in script


def test_getting_started_uses_the_local_mode():
    doc = (REPO / "docs" / "getting-started.md").read_text(encoding="utf-8")
    assert "bash install.sh --local" in doc
    assert "/root/Gnomon" not in doc, "a reviewer's local path is in the published docs"


def test_no_published_doc_ships_a_local_path():
    for doc in (REPO / "docs").glob("*.md"):
        if doc.name.startswith("codebase-review"):
            continue  # the review quotes the paths it is reporting
        assert "/root/Gnomon" not in doc.read_text(encoding="utf-8"), doc.name
