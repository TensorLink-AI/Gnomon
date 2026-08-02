"""Correct behaviour, presented so it cannot be misread.

Each of these was honest output that a reader would take the wrong way:
a probability whose event went unnamed, a unit that differed between two
adjacent fields, a silent mutation, a capability that reads as a promise.
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


# -- M1: the exceedance event is named --------------------------------------

def test_decide_defines_the_exceedance_event(tmp_path):
    """`exceed` was the per-step maximum, labelled plainly, with no
    definition — and it is lower than the any-step probability a
    horizon-level decision usually turns on."""
    from gnomon.macros import decide

    payload, _ = decide(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=7, threshold=115.0,
        actions=[{"name": "scale_up"}, {"name": "do_nothing"}],
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    exceedance = payload["exceedance"]
    assert exceedance is not None
    assert exceedance["peak_step_exceedance"] == payload["scenario_probabilities"]["exceed"]
    assert exceedance["any_step_exceedance_if_independent"] >= exceedance["peak_step_exceedance"]
    assert "largest single-step" in exceedance["event_definition"]
    assert len(exceedance["per_step"]) == 7
    # The expected-utility mass stays exactly two scenarios.
    assert set(payload["scenario_probabilities"]) == {"exceed", "no_exceed"}


# -- M6: data in the future is remarked on ----------------------------------

def test_inspect_remarks_on_entirely_future_data(tmp_path):
    from gnomon.runtime import inspect_dataset

    start = date(2027, 1, 1)
    rows = [f"{(start + timedelta(days=i)).isoformat()},{100 + i}" for i in range(40)]
    path = tmp_path / "future.csv"
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n", encoding="utf-8")

    report = inspect_dataset(
        str(path), time_column="timestamp", target_column="value", clock=CLOCK,
    )
    assert report["status"] == "valid", "future data is unusual, not invalid"
    assert report["data_quality"]["temporal_position"] == "entirely_in_the_future"
    assert "after the current instant" in report["data_quality"]["note"]


def test_inspect_says_nothing_about_ordinary_past_data(tmp_path):
    from gnomon.runtime import inspect_dataset

    report = inspect_dataset(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        clock=CLOCK,
    )
    assert "temporal_position" not in report["data_quality"]


# -- M8: monitor and decide say what they do --------------------------------

def test_monitor_and_decide_lead_with_what_they_do_not_do():
    from gnomon.toolspec import TOOLS

    by_name = {tool["name"]: tool["description"] for tool in TOOLS}
    assert by_name["gnomon_monitor"].startswith("Defines an alert policy; does not run one.")
    assert by_name["gnomon_decide"].startswith("Compares actions; does not take one.")


# -- M9: units are labelled -------------------------------------------------

def test_mape_documents_that_it_is_a_percentage():
    from gnomon.tracking import mape_score

    assert "percent" in (mape_score.__doc__ or "")
    # 100 vs 110 is 10% error, reported as 10.0 not 0.1.
    assert mape_score([100.0], [110.0]) == pytest.approx(10.0)


def test_wape_is_a_fraction():
    from gnomon.evaluation import error_score

    assert error_score([100.0], [110.0]) == pytest.approx(0.1)


# -- M15: the actuals response wears the standard envelope ------------------

def test_actuals_response_carries_schema_version_and_support(tmp_path, capsys, monkeypatch):
    import gnomon.tracking as tracking_module
    from gnomon.cli import main
    from gnomon.runtime import forecast
    from gnomon.tracking import TrackingStore, register_artifact

    registry = tmp_path / "registry.db"
    monkeypatch.setattr(tracking_module, "DEFAULT_REGISTRY_PATH", registry)
    TrackingStore(registry)

    source = _csv(tmp_path, periods=60)
    artifact, path = forecast(
        str(source), time_column="timestamp", target_column="value", horizon=3,
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    register_artifact(artifact, "demo", str(path))

    rows = [
        f"{row['timestamp'][:10]},{row['point']:.4f}"
        for row in artifact.results[0].forecast
    ]
    actuals = tmp_path / "actuals.csv"
    actuals.write_text("timestamp,value\n" + "\n".join(rows) + "\n", encoding="utf-8")

    assert main([
        "track", "actuals", "--project", "demo", "--file", str(actuals),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "0.1"
    assert payload["status"] == "ok"
    if payload.get("scored"):
        assert payload["support_assessment"]["status"] == "supported"
        codes = {
            item["code"] for item in payload["support_assessment"]["disclosures"]
        }
        assert "hindsight_is_observational" in codes


# -- M16: sorting is disclosed ----------------------------------------------

def test_out_of_order_rows_are_recorded_as_a_repair(tmp_path):
    """Correct, and previously invisible — which is why a genuinely
    unsorted file never raised a data-quality signal."""
    from gnomon.runtime import inspect_dataset

    start = date(2026, 1, 1)
    dates = [start + timedelta(days=i) for i in range(40)]
    dates[5], dates[6] = dates[6], dates[5]
    rows = [f"{day.isoformat()},{100 + index}" for index, day in enumerate(dates)]
    path = tmp_path / "unsorted.csv"
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n", encoding="utf-8")

    report = inspect_dataset(
        str(path), time_column="timestamp", target_column="value", clock=CLOCK,
    )
    codes = {item["code"] for item in report["data_quality"]["repairs"]}
    assert "timestamps_reordered" in codes


def test_sorted_input_records_nothing(tmp_path):
    from gnomon.runtime import inspect_dataset

    report = inspect_dataset(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        clock=CLOCK,
    )
    codes = {item["code"] for item in report["data_quality"]["repairs"]}
    assert "timestamps_reordered" not in codes


# -- M19: compare is reachable from a project -------------------------------

def test_compare_finds_its_own_ids(tmp_path, capsys, monkeypatch):
    import gnomon.tracking as tracking_module
    from gnomon.cli import main
    from gnomon.tracking import TrackingStore

    registry = tmp_path / "registry.db"
    monkeypatch.setattr(tracking_module, "DEFAULT_REGISTRY_PATH", registry)
    store = TrackingStore(registry)
    for index in (1, 2):
        store.register(
            f"fc_{index}", "demo", "__default__", horizon=2, frequency="D",
            selected_model="ets", naive_error=10.0,
            artifact_path=str(tmp_path / "artifact"),
            created_at=f"2026-0{index}-01T00:00:00+00:00",
        )
        store.score_forecast(
            f"fc_{index}", [100.0, 110.0], [95.0, 105.0], project="demo",
        )

    assert main(["track", "compare", "--project", "demo", "--latest", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["compared"] == 1
    assert {payload["comparisons"][0]["forecast_a"]["id"],
            payload["comparisons"][0]["forecast_b"]["id"]} == {"fc_1", "fc_2"}


def test_compare_without_ids_or_project_says_what_to_pass(tmp_path, capsys, monkeypatch):
    import gnomon.tracking as tracking_module
    from gnomon.cli import main
    from gnomon.tracking import TrackingStore

    registry = tmp_path / "registry.db"
    monkeypatch.setattr(tracking_module, "DEFAULT_REGISTRY_PATH", registry)
    TrackingStore(registry)
    assert main(["track", "compare"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "INVALID_ARGUMENTS"
    assert error["repair_options"]


# -- L2 / L5: help text and units -------------------------------------------

def test_every_forecast_flag_has_help():
    from gnomon.cli import build_parser

    parser = build_parser()
    forecast = next(
        action for action in parser._subparsers._actions
        if hasattr(action, "choices") and action.choices
    ).choices["forecast"]
    missing = [
        option.option_strings[0] for option in forecast._actions
        if option.option_strings and not option.help
        and option.help is not None or (option.option_strings and not option.help)
    ]
    assert not missing, f"flags without help: {missing}"


def test_season_period_carries_its_unit():
    from gnomon.fingerprint import series_fingerprint

    values = [100 + (index % 7) * 5 for index in range(60)]
    fingerprint = series_fingerprint(values, "W")
    assert fingerprint["season_period_unit"] == "W"
    assert fingerprint["season_period_label"].endswith("x W")


# -- M18: build outputs are not in git --------------------------------------

def test_binary_caches_are_not_tracked():
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "metric_scaling_cache/"],
        cwd=REPO, capture_output=True, text=True, check=False,
    ).stdout.strip()
    assert not tracked, f"binary cache files are tracked: {tracked.splitlines()[:3]}"


def test_caches_are_ignored():
    ignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    for entry in ("inference_cache/", "metric_scaling_cache/", "results/"):
        assert entry in ignore, f"{entry} is not ignored"


# -- M14: the context feature is documented ---------------------------------

def test_context_events_are_documented_with_the_shipped_example():
    concepts = (REPO / "docs" / "concepts.md").read_text(encoding="utf-8")
    assert "examples/context_events.json" in concepts
    assert "known_at" in concepts
    assert (REPO / "examples" / "context_events.json").is_file()


# -- M11 / M12: the docs match the product ----------------------------------

def test_python_api_does_not_miscount_the_artifact_files():
    doc = (REPO / "docs" / "python-api.md").read_text(encoding="utf-8")
    assert "four\nstandard artifact files" not in doc
    assert "results-and-artifacts.md" in doc


def test_the_example_config_does_not_call_implemented_models_planned():
    example = (REPO / "gnomon.yaml.example").read_text(encoding="utf-8")
    for line in example.splitlines():
        if "(planned)" in line and "ets" in line:
            pytest.fail(f"ets is implemented but listed as planned: {line.strip()}")
