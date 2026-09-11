"""--time and --target are aliases for --time-column and --target-column."""

import json
from pathlib import Path

import pytest

from gnomon.cli import build_parser, main

EXAMPLE = str(Path(__file__).resolve().parents[1] / "examples" / "messy_requests.csv")


def _run(capsys, argv):
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_inspect_accepts_short_flags_without_ambiguity(capsys):
    # Before the alias, --time was an ambiguous prefix of --time-column and --timezone.
    code, result = _run(capsys, ["inspect", EXAMPLE, "--time", "timestamp", "--target", "requests"])
    assert code == 0 and result["status"] == "ok"


def test_short_and_long_flags_are_the_same_dest_on_every_data_command(capsys):
    long_form = _run(capsys, ["describe", EXAMPLE, "--statistic", "latest", "--time-column", "timestamp",
                              "--target-column", "requests", "--brief"])[1]
    short_form = _run(capsys, ["describe", EXAMPLE, "--statistic", "latest", "--time", "timestamp",
                               "--target", "requests", "--brief"])[1]
    assert long_form["value"] == short_form["value"]
    code, forecast = _run(capsys, ["forecast", "--input", EXAMPLE, "--horizon", "2", "--provider", "last_value",
                                   "--time", "timestamp", "--target", "requests"])
    assert code == 0 and forecast["result"]["point"] == [long_form["value"]] * 2
    code, study = _run(capsys, ["evaluate", "--input", EXAMPLE, "--candidates", "historical_mean",
                                "--baseline", "last_value", "--horizon", "2", "--folds", "2",
                                "--time", "timestamp", "--target", "requests"])
    assert code == 0 and study["status"] == "complete"
    parsed = build_parser().parse_args(["route", "--input", EXAMPLE, "--study", "STUDY", "--source-as-of", "2026-06-01T00:00:00Z",
                                        "--recorded-as-of", "2099-01-01T00:00:00Z", "--ledger-path", "x.db",
                                        "--time", "timestamp", "--target", "requests"])
    assert parsed.time_column == "timestamp" and parsed.target_column == "requests"


def test_short_flags_count_as_explicit_arguments_in_errors(capsys):
    code, payload = _run(capsys, ["inspect", EXAMPLE, "--time", "ts", "--target", "requests"])
    assert code == 2 and payload["error"]["code"] == "MISSING_COLUMNS"
    explicit = payload["error"]["details"]["explicit_input_options"]
    assert explicit["time_column"] == "ts" and explicit["target_column"] == "requests"


@pytest.mark.parametrize("command", ["inspect", "describe", "infer", "evaluate", "route"])
def test_help_keeps_the_long_form_canonical(command, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args([command, "--help"])
    text = capsys.readouterr().out
    assert "--time-column NAME, --time NAME" in text
    assert "--target-column NAME, --target NAME" in text
