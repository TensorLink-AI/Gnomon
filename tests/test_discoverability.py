"""Discoverability: errors teach the short path, docs lead with it.

The loudest agent complaint was flag verbosity — mostly a teaching
problem. These tests pin the fixes: ambiguity errors show a minimal
working invocation built from the actual file, unknown flags get a
nearest-valid suggestion, capabilities reports the new surface, and the
docs lead the forecast examples with the minimal form.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from gnomon.cli import main

REPO = Path(__file__).resolve().parent.parent


def _ambiguous_file(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "hr", "spo2"])
        for day in range(1, 10):
            writer.writerow([f"2026-01-{day:02d}", 70 + day, 97.0])


def test_ambiguous_target_error_shows_a_minimal_invocation(tmp_path, capsys):
    path = tmp_path / "two_numeric.csv"
    _ambiguous_file(path)
    assert main(["forecast", str(path), "--horizon", "3"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "AMBIGUOUS_SCHEMA"
    assert error["details"]["candidates"] == ["hr", "spo2"]
    suggested = error["details"]["suggested_invocation"]
    assert suggested.startswith("gnomon forecast ")
    assert "--target <hr|spo2>" in suggested
    assert suggested in error["message"]
    # The batching escape hatch is offered right where the ambiguity is.
    assert any(option["action"] == "forecast_all_candidates"
               for option in error["repair_options"])


def test_unknown_flag_suggests_the_valid_one(tmp_path, capsys):
    path = tmp_path / "data.csv"
    _ambiguous_file(path)
    assert main(["forecast", str(path), "--column", "hr"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "INVALID_ARGUMENTS"
    assert "Did you mean --target instead of --column?" in error["message"]
    assert {"unknown": "--column", "suggestion": "--target"} in (
        error["details"]["flag_suggestions"]
    )
    assert any(option["action"] == "rename_flag"
               for option in error["repair_options"])


def test_misspelled_flag_suggests_the_nearest_one(tmp_path, capsys):
    path = tmp_path / "data.csv"
    _ambiguous_file(path)
    assert main(["forecast", str(path), "--tagret", "hr"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert "Did you mean --target instead of --tagret?" in error["message"]


def test_flag_valid_only_on_another_command_is_not_suggested_as_itself(capsys):
    assert main(["capabilities", "--brief"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert "Did you mean --brief instead of --brief?" not in error["message"]
    assert not error["details"].get("flag_suggestions")
    assert error["repair_options"] == [{
        "action": "show_usage",
        "description": "Run `gnomon --help` for the full argument list.",
    }]


def test_capabilities_report_the_new_surface():
    from gnomon.runtime import capabilities

    payload = capabilities()
    assert payload["features"]["multi_target_batching"] is True
    assert payload["features"]["brief_output"] is True
    surface = payload["forecast_surface"]
    assert "--target" in surface["multi_target_batching"]["cli"]
    assert surface["brief_output"]["cli"] == "--brief"


def test_readme_leads_the_forecast_example_with_the_minimal_form():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    minimal = readme.index("gnomon forecast examples/daily_requests.csv --horizon 3")
    explicit = readme.index("--time timestamp --target requests \\")
    assert minimal < explicit, "the README teaches the long path first"
    assert "--target auto" in readme
    assert "--brief" in readme


def test_cli_reference_states_what_is_inferred_and_when_inference_refuses():
    reference = (REPO / "docs" / "cli-reference.md").read_text(encoding="utf-8")
    assert "gnomon forecast data.csv" in reference
    for phrase in ("`--time` is inferred", "`--target` is inferred",
                   "Inference refuses", "--target auto", "`--brief`"):
        assert phrase in reference, phrase
