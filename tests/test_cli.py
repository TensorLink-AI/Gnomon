from __future__ import annotations

import json
from pathlib import Path

from gnomon.cli import main


def test_capabilities_are_honest(capsys) -> None:
    assert main(["capabilities"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["interfaces"]["cli"] is True
    assert result["interfaces"]["mcp"] is True
    assert result["features"]["context_events"] is True
    assert result["features"]["actual_scoring"] is True
    assert result["features"]["decision_outcomes"] is True
    assert isinstance(result["models"]["tsfm"], list)
    assert isinstance(result["models"].get("tsfm_available"), list)


def test_missing_input_returns_structured_error(capsys) -> None:
    result = main([
        "inspect", "/does/not/exist.csv", "--time", "timestamp", "--target", "value"
    ])
    assert result == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "error"
    assert error["error"]["code"] == "INPUT_NOT_FOUND"


def test_describe_cli_executes_typed_statistical_question(
        tmp_path: Path, capsys) -> None:
    path = tmp_path / "series.csv"
    path.write_text(
        "timestamp,value\n" + "".join(
            f"2024-01-{index + 1:02d},{(-1) ** index}\n"
            for index in range(20)),
        encoding="utf-8",
    )
    questions = json.dumps([{
        "id": "adf", "verb": "test", "property": "stationarity",
        "target": "value", "method": "adf",
    }])
    assert main(["describe", str(path), "--questions", questions,
                 "--format", "full"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["answers"][0]["answer"]["executable"]["kind"] == \
        "fitted_stationarity_test"
    assert payload["dataset_contract"]["shape"] == "univariate"


def test_forecast_cli_exposes_the_weakest_authority_floor(
        tmp_path: Path, capsys) -> None:
    source = Path(__file__).resolve().parent.parent / "examples" / "daily_requests.csv"
    assert main([
        "forecast", str(source), "--time", "timestamp", "--target", "requests",
        "--frequency", "D", "--horizon", "7", "--output", str(tmp_path),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    row_tiers = {
        row["tier"] for result in payload["results"]
        for row in result["forecast_preview"]
    }
    assert payload["tier_floor"] in row_tiers
