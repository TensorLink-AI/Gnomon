"""Multi-target batching: one invocation, several columns of a wide file.

The contract under test is surface-and-concurrency only:

- parity: every batched channel carries exactly the numbers a sequential
  single-target run of that column produces;
- isolation: a channel that abstains (or whose column cannot even be
  loaded) is disclosed in its own result and never blocks the others;
- determinism: the combined artifact is a pure function of the task —
  identical at any worker count;
- identity: multi-target IDs are new; single-target invocations are
  untouched (the golden tests pin those byte-for-byte).
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gnomon.contracts import GnomonError
from gnomon.data import resolve_target_spec
from gnomon.ids import FixedClock
from gnomon.runtime import forecast, forecast_multi

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))
START = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _write_wide(path: Path, rows: int = 120, channels: int = 3,
                junk_channel: bool = False) -> list[str]:
    names = [f"ch{i}" for i in range(channels)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp"] + names + (["junk"] if junk_channel else []))
        for k in range(rows):
            stamp = (START + timedelta(hours=k)).isoformat()
            values = [
                repr(round(10 * (i + 1) + math.sin(2 * math.pi * (k + i) / 24), 6))
                for i in range(channels)
            ]
            writer.writerow([stamp] + values + (["not-a-number"] if junk_channel else []))
    return names


def test_batched_channels_match_sequential_single_target_runs(tmp_path):
    path = tmp_path / "wide.csv"
    names = _write_wide(path)
    artifact, _ = forecast_multi(
        str(path), time_column="timestamp", target_columns=names,
        horizon=12, output=str(tmp_path / "multi"), clock=CLOCK,
    )
    assert [result.series for result in artifact.results] == names
    for result in artifact.results:
        single, _ = forecast(
            str(path), time_column="timestamp", target_column=result.series,
            horizon=12, output=str(tmp_path / "single"), clock=CLOCK,
        )
        expected = single.results[0]
        assert result.support == expected.support
        assert result.selected_model == expected.selected_model
        assert result.selection_scores == expected.selection_scores
        assert result.test_scores == expected.test_scores
        assert result.interval_coverage == expected.interval_coverage
        assert result.warnings == expected.warnings
        assert result.forecast == expected.forecast


def test_channel_abstention_is_isolated_and_disclosed(tmp_path):
    path = tmp_path / "wide.csv"
    names = _write_wide(path, junk_channel=True)
    artifact, _ = forecast_multi(
        str(path), time_column="timestamp", target_columns=names + ["junk"],
        horizon=12, output=str(tmp_path / "multi"), clock=CLOCK,
    )
    by_series = {result.series: result for result in artifact.results}
    failed = by_series["junk"]
    assert failed.support == "unsupported"
    assert failed.forecast == []
    assert failed.support_assessment["status"] == "unsupported"
    # The reason is carried verbatim: code and message, never summarised.
    assert any("INVALID_TARGET" in warning for warning in failed.warnings)
    assert failed.support_assessment["reasons"][0]["code"] == "INVALID_TARGET"
    assert failed.support_assessment["recovery_actions"]
    for name in names:
        assert by_series[name].support != "unsupported"
        assert by_series[name].forecast


def test_missing_column_is_isolated_too(tmp_path):
    path = tmp_path / "wide.csv"
    names = _write_wide(path)
    artifact, _ = forecast_multi(
        str(path), time_column="timestamp",
        target_columns=[names[0], "no_such_column"],
        horizon=6, output=str(tmp_path / "multi"), clock=CLOCK,
    )
    by_series = {result.series: result for result in artifact.results}
    assert by_series["no_such_column"].support == "unsupported"
    assert any("MISSING_COLUMNS" in warning
               for warning in by_series["no_such_column"].warnings)
    assert by_series[names[0]].forecast


def test_artifact_is_identical_at_any_worker_count(tmp_path):
    path = tmp_path / "wide.csv"
    names = _write_wide(path, channels=4)
    texts = []
    for workers in (1, 4):
        _, directory = forecast_multi(
            str(path), time_column="timestamp", target_columns=names,
            horizon=8, output=str(tmp_path / f"w{workers}"), clock=CLOCK,
            max_workers=workers,
        )
        texts.append((directory / "artifact.json").read_text(encoding="utf-8"))
    assert texts[0] == texts[1]


def test_multi_target_id_is_deterministic_and_distinct(tmp_path):
    path = tmp_path / "wide.csv"
    names = _write_wide(path)
    first, _ = forecast_multi(
        str(path), time_column="timestamp", target_columns=names,
        horizon=12, output=str(tmp_path / "a"), clock=CLOCK,
    )
    second, _ = forecast_multi(
        str(path), time_column="timestamp", target_columns=names,
        horizon=12, output=str(tmp_path / "b"), clock=CLOCK,
    )
    assert first.forecast_id == second.forecast_id
    single, _ = forecast(
        str(path), time_column="timestamp", target_column=names[0],
        horizon=12, output=str(tmp_path / "c"), clock=CLOCK,
    )
    assert single.forecast_id != first.forecast_id
    reordered, _ = forecast_multi(
        str(path), time_column="timestamp",
        target_columns=list(reversed(names)),
        horizon=12, output=str(tmp_path / "d"), clock=CLOCK,
    )
    assert reordered.forecast_id != first.forecast_id


def test_all_channels_failing_raises_the_diagnosis(tmp_path):
    path = tmp_path / "wide.csv"
    _write_wide(path)
    with pytest.raises(GnomonError) as excinfo:
        forecast_multi(
            str(path), time_column="timestamp",
            target_columns=["ghost_a", "ghost_b"],
            horizon=6, output=str(tmp_path / "multi"), clock=CLOCK,
        )
    assert excinfo.value.code == "MISSING_COLUMNS"


def test_store_inputs_are_refused(tmp_path):
    with pytest.raises(GnomonError) as excinfo:
        forecast_multi(
            "store:whatever", time_column="t", target_columns=["a", "b"],
            horizon=3, output=str(tmp_path), clock=CLOCK,
        )
    assert excinfo.value.code == "INVALID_ARGUMENTS"


def test_resolve_target_spec_expands_auto_and_lists(tmp_path):
    path = tmp_path / "wide.csv"
    names = _write_wide(path)
    assert resolve_target_spec(str(path), "auto") == names
    assert resolve_target_spec(str(path), "ch1, ch0") == ["ch1", "ch0"]
    assert resolve_target_spec(str(path), "ch2") == ["ch2"]
    with pytest.raises(GnomonError) as excinfo:
        resolve_target_spec(str(path), "ch0,ch0")
    assert excinfo.value.code == "INVALID_ARGUMENTS"


def test_duplicate_targets_are_refused_at_the_runtime_level(tmp_path):
    # Result series and evidence records are keyed by target name; a
    # repeated name would collide identifiers inside one artifact, so the
    # runtime refuses even though the CLI/MCP spec parser already dedupes.
    path = tmp_path / "wide.csv"
    names = _write_wide(path)
    with pytest.raises(GnomonError) as excinfo:
        forecast_multi(
            str(path), time_column="timestamp",
            target_columns=[names[0], names[0]],
            horizon=6, output=str(tmp_path / "multi"), clock=CLOCK,
        )
    assert excinfo.value.code == "INVALID_ARGUMENTS"
    assert excinfo.value.details["duplicates"] == [names[0]]


def test_auto_includes_channels_with_missing_cells(tmp_path):
    """A channel with a gap is still a channel. `auto` excluding it
    silently would hide an abstention the run owes the caller; instead
    the channel is included and abstains loudly on its own, without
    blocking the clean ones."""
    path = tmp_path / "gappy.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "good", "gappy", "label"])
        for k in range(60):
            writer.writerow([
                (START + timedelta(hours=k)).isoformat(),
                repr(10.0 + k % 5),
                "" if k == 30 else repr(20.0 + k % 3),
                "text",  # never a candidate
            ])
    assert resolve_target_spec(str(path), "auto") == ["good", "gappy"]
    artifact, _ = forecast_multi(
        str(path), time_column="timestamp", target_columns=["good", "gappy"],
        horizon=5, output=str(tmp_path / "multi"), clock=CLOCK,
    )
    by_series = {result.series: result for result in artifact.results}
    assert by_series["good"].forecast
    assert by_series["gappy"].support == "unsupported"
    assert any("IRREGULAR_TIME_GRID" in warning
               for warning in by_series["gappy"].warnings)


def test_auto_does_not_read_zero_as_missing(tmp_path):
    path = tmp_path / "zeros.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "zeroes", "vals"])
        for day in range(1, 29):
            writer.writerow([f"2025-01-{day:02d}", 0, 5 + day % 3])
    assert resolve_target_spec(str(path), "auto") == ["zeroes", "vals"]


def test_auto_against_a_store_input_is_a_clear_refusal():
    with pytest.raises(GnomonError) as excinfo:
        resolve_target_spec("store:vitals", "auto")
    assert excinfo.value.code == "INVALID_ARGUMENTS"
    assert "store" in excinfo.value.message


def test_cli_multi_target_and_auto(tmp_path, capsys, monkeypatch):
    from gnomon.cli import main

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "wide.csv"
    names = _write_wide(path)
    assert main([
        "forecast", str(path), "--target", ",".join(names[:2]),
        "--horizon", "6", "--output", str(tmp_path / "out"),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [result["series"] for result in payload["results"]] == names[:2]

    assert main([
        "forecast", str(path), "--target", "auto",
        "--horizon", "6", "--output", str(tmp_path / "out"),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [result["series"] for result in payload["results"]] == names
    assumptions = payload["results"][0]["support_assessment"]["assumptions"]
    assert any("--target auto expanded" in item for item in assumptions)


def test_cli_rejects_flags_multi_target_cannot_honour(tmp_path, capsys):
    from gnomon.cli import main

    path = tmp_path / "wide.csv"
    _write_wide(path)
    code = main([
        "forecast", str(path), "--target", "ch0,ch1", "--horizon", "6",
        "--project", "demo", "--output", str(tmp_path / "out"),
    ])
    assert code == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "INVALID_ARGUMENTS"
    assert "--project" in error["message"]


def test_mcp_forecast_tool_batches_comma_targets(tmp_path):
    from gnomon.toolspec import runner_for

    path = tmp_path / "wide.csv"
    names = _write_wide(path)
    payload = runner_for("gnomon_forecast")({
        "input": str(path), "time_column": "timestamp",
        "target_column": ",".join(names), "horizon": 6,
        "output_dir": str(tmp_path / "out"),
    })
    assert [result["series"] for result in payload["results"]] == names
    for result in payload["results"]:
        assert result["support_assessment"] is not None
