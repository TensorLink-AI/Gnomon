import json
import subprocess
import sys
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


def test_unknown_command_is_a_structured_error(capsys):
    assert main(["not-a-command"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "INVALID_ARGUMENTS"


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
@pytest.mark.parametrize("arguments", [
    ["inspect", "-"], ["describe", "-", "--statistic", "mean"],
    ["infer", "--input", "-", "--provider", "last_value", "--horizon", "2"],
])
def test_piped_csv_uses_the_current_snapshot_contract(arguments):
    result = subprocess.run([sys.executable, "-m", "gnomon", *arguments, "--frequency", "D"],
        input="timestamp,value\n2026-01-01,1\n2026-01-02,2\n", text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    if arguments[0] == "describe":
        assert payload["value"] == 1.5
    elif arguments[0] == "infer":
        assert payload["result"]["point"] == [2, 2]
    else:
        assert payload["status"] == "ok"


def test_stdin_limit_is_enforced_before_materializing_a_snapshot(monkeypatch, capsys):
    import io
    from types import SimpleNamespace
    from gnomon import cli
    monkeypatch.setattr(cli, "MAX_STDIN_BYTES", 16)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"x" * 17)))
    assert cli.main(["inspect", "-"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "INPUT_TOO_LARGE"
