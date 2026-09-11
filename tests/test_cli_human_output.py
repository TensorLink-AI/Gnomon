"""forecast/evaluate print a compact table on a terminal and the JSON envelope elsewhere."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from gnomon import cli
from gnomon.cli import _forecast_text, main

EXAMPLE = str(Path(__file__).resolve().parents[1] / "examples" / "messy_requests.csv")
FORECAST = ["forecast", EXAMPLE, "--time", "timestamp", "--target", "requests", "--horizon", "7",
            "--provider", "seasonal_naive", "--season", "7"]
EVALUATE = ["evaluate", "--input", EXAMPLE, "--time", "timestamp", "--target", "requests", "--candidates",
            "seasonal_naive", "historical_mean", "--baseline", "last_value", "--season", "7", "--horizon", "7", "--folds", "3"]


@pytest.fixture
def tty(monkeypatch):
    monkeypatch.setattr(cli, "_stdout_is_tty", lambda: True)


def test_forecast_on_a_terminal_is_a_short_table(tty, capsys):
    assert main(FORECAST) == 0
    lines = capsys.readouterr().out.rstrip("\n").split("\n")
    assert len(lines) <= 12 and len(lines) == 7 + 3
    assert all(line.startswith("2026-06-") and len(line.split()) == 2 for line in lines[:7])
    assert lines[7].startswith("provider: seasonal_naive (gnomon/")
    assert lines[8].startswith("execution: ")
    assert lines[9].startswith("snapshot: snapshot_") and lines[9].endswith(" as_of=latest")


def test_request_form_discloses_no_snapshot(tty, capsys):
    assert main(["infer", "--provider", "last_value", "--request", '{"history":[10,12,11],"horizon":2}']) == 0
    lines = capsys.readouterr().out.rstrip("\n").split("\n")
    assert lines[:2] == ["+1  11", "+2  11"]
    assert lines[-1] == "snapshot: request supplied directly"


def test_json_flag_and_non_tty_keep_the_envelope(tty, capsys):
    assert main([*FORECAST, "--json"]) == 0
    forced = json.loads(capsys.readouterr().out)
    assert forced["status"] == "ok" and forced["provider"] == "seasonal_naive"
    piped = subprocess.run([sys.executable, "-m", "gnomon", *FORECAST], capture_output=True, text=True)
    assert piped.returncode == 0
    envelope = json.loads(piped.stdout)
    assert envelope["result"]["point"] == forced["result"]["point"]
    assert set(envelope) == set(forced)


def test_errors_stay_json_on_a_terminal(tty, capsys):
    assert main(["forecast", "missing.csv", "--horizon", "2"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "INPUT_NOT_FOUND"


def test_evaluate_on_a_terminal_is_a_fold_table_with_a_ranking_line(tty, capsys):
    assert main(EVALUATE) == 0
    lines = capsys.readouterr().out.rstrip("\n").split("\n")
    assert lines[0].split() == ["origin", "last_value", "seasonal_naive", "historical_mean"]
    assert len(lines) == 1 + 3 + 2
    assert all(len(row.split()) == 4 and row.startswith("2026-05-") for row in lines[1:4])
    assert lines[4].startswith("ranking: last_value (mae ") and lines[4].endswith("[complete]")
    assert lines[5].startswith("study: ")
    assert main([*EVALUATE, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_quantile_bounds_are_printed_when_present():
    text = _forecast_text({"status": "ok", "provider": "p", "revision": None, "execution_id": "x", "snapshot": None,
                           "result": {"point": [1.5], "timestamps": ["2026-01-02"],
                                      "quantiles": [{"0.1": 1.0, "0.5": 1.5, "0.9": 2.25}]}})
    assert text.split("\n") == ["2026-01-02  1.5  [1 2.25]", "provider: p (unversioned)", "execution: x",
                                "snapshot: request supplied directly"]


def test_terminal_keeps_repair_assumption_and_unit_disclosures(tty, tmp_path, capsys):
    path = tmp_path / "dollars.csv"
    rows = ["2026-01-%02d,$%d" % (day, 100 + day) for day in range(1, 15)] + ["2026-01-14,$114"]
    path.write_text("timestamp,requests\n" + "\n".join(rows) + "\n")
    assert main(["forecast", str(path), "--horizon", "2", "--provider", "last_value", "--target", "requests",
                 "--unit", "widgets"]) == 0
    lines = capsys.readouterr().out.rstrip("\n").split("\n")
    assert [line.split(":")[0] for line in lines[2:]] == ["provider", "execution", "snapshot", "repairs", "unit"]
    assert lines[5].startswith("repairs: 2 (") and "numeric_format_normalisedx15" in lines[5] and "duplicate_row_collapsedx1" in lines[5]
    assert "assumptive" not in lines[5], "safe repair invents nothing, and says so by omission"
    assert lines[6] == "unit: widgets"
    # The bare form's inferred choices stay visible too.
    assert main(["forecast", str(EXAMPLE), "--horizon", "2"]) == 0
    last = capsys.readouterr().out.rstrip("\n").split("\n")[-1]
    assert last.startswith("assumed: ") and "provider=last_value" in last and "target_column=requests" in last
    assert main([*EVALUATE, "--unit", "widgets"]) == 0
    assert capsys.readouterr().out.rstrip("\n").split("\n")[-1] == "unit: widgets"
