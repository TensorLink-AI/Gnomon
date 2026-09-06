import json
import pytest

from gnomon.cli import main


def test_observed_mean_and_forecast_use_same_input(tmp_path, capsys):
    source = tmp_path / "data.csv"
    source.write_text("timestamp,value\n2025-01-01,1\n2025-01-02,9\n2025-01-03,2\n")
    assert main(["describe", str(source), "--statistic", "mean"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["value"] == 4 and result["count"] == 3
    assert result["action_authorized"] is False
    assert main(["infer", "--input", str(source), "--horizon", "2", "--provider", "last_value"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["result"]["point"] == [2, 2]
    assert result["input"]["snapshot"]["known_time_assumed"] is True


@pytest.mark.parametrize("command", ["forecast", "investigate", "detect", "decide", "monitor",
                                    "track", "tsfm", "context", "covariates", "report", "eval"])
def test_retired_commands_fail_without_silent_aliases(command, capsys):
    assert main([command]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "INVALID_ARGUMENTS"


@pytest.mark.parametrize("profile", ["core", "data", "decision", "evidence", "full", "mega"])
def test_retired_profiles_do_not_restore_legacy_runtime(profile, capsys):
    assert main(["mcp", "serve", "--profile", profile]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "INVALID_ARGUMENTS" and "Retired" in error["message"]


def test_missing_input_is_structured(tmp_path, capsys):
    assert main(["inspect", str(tmp_path / "missing.csv")]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "INPUT_NOT_FOUND"


def test_temporal_calendar_shift(tmp_path, capsys):
    assert main(["temporal", "--arguments", json.dumps({
        "operation": "shift", "value": "2024-01-31", "amount": 1, "unit": "months",
        "mode": "calendar", "invalid_date": "clamp",
    })]) == 0
    assert "2024-02-29" in capsys.readouterr().out


def test_invalid_self_check_does_not_claim_success(capsys):
    assert main(["self-check", "leakage", "--cases", "0"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "INVALID_ARGUMENTS"
