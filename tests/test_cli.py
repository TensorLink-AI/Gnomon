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
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "INVALID_ARGUMENTS"
    assert captured.err == ""


def test_subprocess_errors_are_parseable_from_the_same_channel_as_successes():
    result = subprocess.run([sys.executable, "-m", "gnomon", "not-a-command"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(result.stdout)["error"]["code"] == "INVALID_ARGUMENTS"
    assert result.stderr == ""


def test_missing_input_is_structured(tmp_path, capsys):
    assert main(["inspect", str(tmp_path / "missing.csv")]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "INPUT_NOT_FOUND"
    assert captured.err == ""


def test_temporal_calendar_shift(tmp_path, capsys):
    assert main(["temporal", "--arguments", json.dumps({
        "operation": "shift", "value": "2024-01-31", "amount": 1, "unit": "months",
        "mode": "calendar", "invalid_date": "clamp",
    })]) == 0
    assert "2024-02-29" in capsys.readouterr().out


def test_invalid_self_check_does_not_claim_success(capsys):
    assert main(["self-check", "leakage", "--cases", "0"]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "INVALID_ARGUMENTS"
    assert captured.err == ""
@pytest.mark.parametrize("arguments", [
    ["inspect", "-"], ["inspect", "--input", "-"],
    ["describe", "-", "--statistic", "mean"], ["describe", "--input", "-", "--statistic", "mean"],
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
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "INPUT_TOO_LARGE"
    assert captured.err == ""


@pytest.mark.parametrize("provider", ["seasonal_naive", "historical_mean"])
def test_column_defaults_and_explicit_mapping_are_provider_independent(provider, tmp_path, capsys):
    source = tmp_path / "data.csv"
    source.write_text("ts,value\n2026-01-01,1\n2026-01-02,3\n2026-01-03,5\n")
    args = ["infer", "--provider", provider, "--input", str(source), "--horizon", "2"]
    assert main(args) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "MISSING_COLUMNS"
    assert error["details"]["missing_columns"] == ["timestamp"]
    assert "--time-column ts" in error["repair_options"][0]["description"]
    assert main([*args, "--time-column", "ts"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["result"]["point"] == ([5, 5] if provider == "seasonal_naive" else [3, 3])
    source.write_text(source.read_text().replace("ts,value", "timestamp,value"))
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["result"] == result["result"]


@pytest.mark.parametrize("command,extra", [("inspect", []), ("describe", ["--statistic", "mean"])])
def test_input_alias_preserves_positional_behavior(command, extra, tmp_path, capsys):
    source = tmp_path / "data.csv"
    source.write_text("timestamp,value\n2026-01-01,1\n2026-01-02,3\n2026-01-03,5\n")
    assert main([command, str(source), *extra]) == 0
    positional = json.loads(capsys.readouterr().out)
    assert main([command, "--input", str(source), *extra]) == 0
    assert json.loads(capsys.readouterr().out) == positional
    assert main([command, str(source), "--input", str(source), *extra]) == 2
    assert "not both" in json.loads(capsys.readouterr().out)["error"]["message"]
    assert main([command, *extra]) == 2
    assert "--input" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_bare_self_check_has_usage_guidance_without_evidence_rejection(capsys):
    assert main(["self-check"]) == 2
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["error"]["code"] == "INVALID_ARGUMENTS"
    assert "gnomon self-check --help" in result["error"]["repair_options"][0]["description"]
    assert "rejection" not in result
    assert captured.err == ""


@pytest.mark.parametrize("command", ["evaluate", "route"])
def test_schema_is_available_without_config_and_describes_inline_data(command, capsys):
    assert main([command, "--schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    inline = schema["oneOf"][0]
    assert "data" in inline["required"] and "data_ref" not in inline["properties"]
    assert inline["properties"]["data"]["required"] == ["input"]
    assert "time_column" in inline["properties"]["data"]["properties"]
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert '"data":{"input":"data.csv"}' in help_text
    assert "TOML" in help_text and "ledger_path" in help_text and "--schema" in help_text


def test_help_examples_evaluate_then_route_across_sessions(tmp_path, monkeypatch, capsys):
    from gnomon.cli import _EXAMPLES
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.csv").write_text("timestamp,value\n" + "".join(
        f"2026-01-{day:02d}T00:00:00Z,{day}\n" for day in range(1, 32)))
    config = tmp_path / "providers.toml"
    config.write_text('schema_version = 1\nledger_path = "ledger.db"\n')
    assert main(["evaluate", "--providers-config", str(config), "--arguments", _EXAMPLES["evaluate"]]) == 0
    study = json.loads(capsys.readouterr().out)
    assert study["usage"]["provider_calls"] == 8 and study["recorded"] is True
    route = json.loads(_EXAMPLES["route"])
    route["study_id"] = study["study_id"]
    argument_file = tmp_path / "route.json"
    argument_file.write_text(json.dumps(route))
    assert main(["route", "--providers-config", str(config), "--arguments", "@" + str(argument_file)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["matched_folds"] == 4 and result["provider_calls"] == 0


@pytest.mark.parametrize("arguments,message", [
    ({"data": "data.csv"}, "data must be an inspection object"),
    ({"arguments": {}}, "unknown arguments"),
    ([], "JSON argument must be an object"),
    ({}, "missing arguments"),
])
def test_evaluate_invalid_shapes_explain_the_accepted_object(arguments, message, capsys):
    assert main(["evaluate", "--arguments", json.dumps(arguments)]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert message in error["message"]
    assert error["details"]["example_arguments"]["data"] == {"input": "data.csv"}
    assert error["details"]["schema_command"] == "gnomon evaluate --schema"


def test_json_provider_config_explains_toml_without_echoing_contents(tmp_path, capsys):
    path = tmp_path / "providers.json"
    path.write_text('{"token":"private-test-token"}')
    assert main(["route", "--providers-config", str(path), "--arguments", "{}"]) == 2
    captured = capsys.readouterr()
    error = json.loads(captured.out)["error"]
    assert "TOML, not JSON" in error["message"] and "ledger_path" in error["message"]
    assert "private-test-token" not in captured.out


def test_evaluate_identifies_invalid_nested_budget(tmp_path, capsys):
    from gnomon.cli import _EXAMPLES
    source = tmp_path / "data.csv"
    source.write_text("timestamp,value\n2026-01-01,1\n2026-01-02,3\n2026-01-03,5\n")
    args = {**json.loads(_EXAMPLES["evaluate"]), "data": {"input": str(source)}, "budget": 8}
    assert main(["evaluate", "--arguments", json.dumps(args)]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["message"] == "budget must be an object"
    assert error["details"]["example_arguments"]["budget"] == {"max_calls": 8}


def test_cli_cache_discloses_disabled_bypassed_and_fresh_session_misses(tmp_path, capsys):
    args = ["infer", "--provider", "last_value", "--request", '{"history":[1,2],"horizon":1}']
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["cache"]["status"] == "disabled"
    config = tmp_path / "providers.toml"
    config.write_text("cache_size = 8\n")
    for _ in range(2):
        assert main([*args, "--providers-config", str(config)]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["cache_hit"] is False
        assert result["cache"]["status"] == "miss"
        assert result["cache"]["scope"] == "session" and result["cache"]["persistent"] is False
    assert main([*args, "--no-cache", "--providers-config", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["cache"]["status"] == "bypassed"
