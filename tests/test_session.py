from io import StringIO
import json
import subprocess
import sys

import pytest

from gnomon import ForecastRequest, ForecastResult, GnomonSession, InferenceEngine, TemporalLedger
from gnomon.cli import main
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.mcp_server import _handle, serve
from test_ephemeris import service  # noqa: F401


def preferred(request):
    return ForecastResult((request.history[-1] + 1,) * request.horizon,
                          timestamps=request.future_timestamps, series_id=request.series_id, unit=request.unit)


def model_factory():
    return preferred


def secret_failure(request):
    raise RuntimeError("token=supersecret")


def config(tmp_path, text=""):
    path = tmp_path / "providers.toml"
    path.write_text('schema_version=1\nledger_path="ledger.db"\n' + text)
    return path


def request():
    return {"history": [1, 2], "horizon": 1, "series_id": "shop",
            "future_timestamps": ["2025-01-03T00:00:00Z"]}


def test_cache_policy_and_outcomes_match_reuse_in_a_long_lived_session(tmp_path):
    path = config(tmp_path, "cache_size = 8\n")
    with GnomonSession.from_config(path) as session:
        policy = session.capabilities()["cache"]
        assert {key: policy[key] for key in ("enabled", "max_entries", "scope", "persistent")} == {
            "enabled": True, "max_entries": 8, "scope": "session", "persistent": False}
        args = {"provider": "last_value", "request": request()}
        first = session.call("gnomon_forecast", args, compact=False)
        second = session.call("gnomon_forecast", args, compact=False)
        assert first["cache"]["status"] == "miss" and first["cache_hit"] is False
        assert second["cache"]["status"] == "hit" and second["cache_hit"] is True
        bypass = session.call("gnomon_forecast", {**args, "use_cache": False}, compact=False)
        assert bypass["cache"]["status"] == "bypassed" and bypass["cache_hit"] is False
        session.engine.register("unversioned", preferred, deterministic=True)
        ineligible = session.call("gnomon_forecast", {**args, "provider": "unversioned"}, compact=False)
        assert ineligible["cache"]["status"] == "ineligible"
        assert ineligible["cache"]["provider_eligible"] is False and ineligible["cache_hit"] is False


def call_message(name, args):
    return {"method": "tools/call", "params": {"name": name, "arguments": args}}


def test_python_cli_and_mcp_use_identical_request_provider_and_ledger_contracts(tmp_path, capsys):
    path = config(tmp_path, f'[providers.custom]\nkind="callable"\nentrypoint="{__name__}:preferred"\nrevision="v1"\n')
    args = {"provider": "custom", "request": request()}
    with GnomonSession.from_config(path) as session:
        python = session.call("gnomon_forecast", args)
        mcp = _handle(call_message("gnomon_forecast", args), session=session)
        assert mcp["isError"] is False
        mcp = mcp["structuredContent"]
    assert main(["infer", "--provider", "custom", "--request", json.dumps(request()), "--providers-config", str(path)]) == 0
    cli = json.loads(capsys.readouterr().out)
    assert len({python["execution_id"], cli["execution_id"], mcp["execution_id"]}) == 3
    assert len({python["fingerprint"], cli["fingerprint"], mcp["fingerprint"]}) == 1
    for response in (python, cli, mcp):
        assert response["recorded"] is True
        assert response["result"]["point"] in ([3], (3,))
        assert response["evidence"] == "inference_only" and not response["action_authorized"]
        assert TemporalLedger(tmp_path / "ledger.db").execution(response["execution_id"])["provider"] == "custom"


def test_factory_plugin_uses_per_request_lifecycle(tmp_path):
    path = config(tmp_path, f'[providers.custom]\nkind="factory"\nentrypoint="{__name__}:model_factory"\n')
    with GnomonSession.from_config(path) as session:
        assert session.engine.capabilities()["custom"]["lifecycle"] == "fresh_per_request"
        assert session.call("gnomon_forecast", {"provider": "custom", "request": request()})["result"]["point"] == (3,)


@pytest.mark.parametrize("with_cutoffs", [False, True])
def test_direct_request_provenance_matches_recorded_execution_without_claiming_snapshot_verification(tmp_path, with_cutoffs):
    payload = request()
    if with_cutoffs:
        payload.update(timestamps=["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"],
                       cutoff="2025-01-02T00:00:00Z", known_time_cutoff="2025-01-02T00:00:00Z",
                       recorded_time_cutoff="2025-01-03T00:00:00Z", snapshot_id="caller-declared-id")
    with GnomonSession.from_config(config(tmp_path)) as session:
        forecast = session.forecast("last_value", ForecastRequest.from_dict(payload))
        assert forecast["recorded"] is True
        assert "snapshot" not in forecast
        provenance = forecast["request_provenance"]
        assert provenance["source"] == "caller_supplied_request"
        stored = session.ledger.execution(forecast["execution_id"])["request"]
        from gnomon.inference import _freeze_request
        from dataclasses import asdict
        canonical = asdict(_freeze_request(payload))
        for key in ("cutoff", "known_time_cutoff", "recorded_time_cutoff", "snapshot_id", "series_id"):
            assert provenance[key] == stored[key] == canonical.get(key)
        assert provenance["history_end"] == (canonical["timestamps"][-1] if with_cutoffs else None)


def test_mcp_session_never_accepts_agent_controlled_urls_imports_or_ledger_paths(tmp_path):
    path = config(tmp_path)
    with GnomonSession.from_config(path) as session:
        for key in ("base_url", "entrypoint", "token_env", "ledger_path", "providers_config"):
            result = _handle(call_message("gnomon_forecast", {
                "provider": "last_value", "request": request(), key: "not-authorized",
            }), session=session)
            assert result["isError"] and result["structuredContent"]["error"]["code"] == "INVALID_ARGUMENTS"
        invalid = _handle(call_message("gnomon_forecast", {"provider": "last_value", "request": {
            **request(), "training_data_url": "https://example.invalid"}}), session=session)
        assert invalid["isError"]


def test_outcome_writes_are_a_startup_permission_not_a_tool_argument(tmp_path):
    path = config(tmp_path)
    with GnomonSession.from_config(path) as session:
        args = {"operation": "append_actual", "series_id": "shop", "valid_time": "2025-01-03T00:00:00Z",
                "value": 3, "source_available_at": "2025-01-04T00:00:00Z"}
        result = _handle(call_message("gnomon_ledger", args), session=session)
        assert result["isError"] and result["structuredContent"]["error"]["code"] == "OUTCOME_WRITES_DISABLED"
        schemas = json.dumps(session.tools())
        assert "append_actual" not in schemas
    path = config(tmp_path, 'allow_outcome_writes=true\n')
    with GnomonSession.from_config(path) as session:
        run = session.call("gnomon_forecast", {"provider": "last_value", "request": request()})
        assert session.call("gnomon_ledger", args)["status"] == "ok"
        scored = session.call("gnomon_ledger", {"operation": "evaluate", "execution_id": run["execution_id"]})
        assert scored["result"]["mae"] == 1


def test_subprocess_cli_and_stdio_mcp_reach_configured_ephemeris(tmp_path, service):
    url, state = service
    path = config(tmp_path, f'[providers.remote]\nkind="ephemeris"\nbase_url="{url}"\ndiscover=true\n')
    cli = subprocess.run([sys.executable, "-m", "gnomon.cli", "infer", "--providers-config", str(path),
                          "--provider", "remote/not-in-gnomon", "--request", json.dumps(request())],
                         text=True, capture_output=True)
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["result"]["point"] == [2.5]
    messages = [
        {"id": 1, "method": "tools/list"},
        {"id": 2, **call_message("gnomon_forecast", {"provider": "remote", "request": request()})},
    ]
    mcp = subprocess.run([sys.executable, "-m", "gnomon.cli", "mcp", "serve", "--providers-config", str(path)],
                         input="".join(json.dumps(m) + "\n" for m in messages), capture_output=True, text=True)
    assert mcp.returncode == 0, mcp.stderr
    listed, forecast = [json.loads(line)["result"] for line in mcp.stdout.splitlines()]
    assert {t["name"] for t in listed["tools"]} == {
        "gnomon_capabilities", "gnomon_forecast", "gnomon_ledger", "gnomon_inspect", "gnomon_describe", "gnomon_evaluate", "gnomon_route", "gnomon_read"}
    assert forecast["structuredContent"]["result"]["point"] == [2.5]
    assert [r[0] for r in state["requests"]] == ["GET", "POST", "GET", "POST"]


def test_session_can_be_injected_without_any_configuration_file(tmp_path):
    ledger = TemporalLedger(tmp_path / "ledger.db")
    engine = InferenceEngine(ledger=ledger)
    engine.register("custom", preferred)
    session = GnomonSession(engine)
    stdin, stdout = StringIO(json.dumps({"id": 1, **call_message("gnomon_forecast", {
        "provider": "custom", "request": request()})}) + "\n"), StringIO()
    assert serve(stdin, stdout, session=session) == 0
    run = json.loads(stdout.getvalue())["result"]["structuredContent"]
    assert run["recorded"] is True and ledger.execution(run["execution_id"])


def test_inference_cli_does_not_import_context_stack_or_read_ambient_config(tmp_path):
    # No implicit cwd search for a file capable of importing arbitrary code.
    (tmp_path / "providers.toml").write_text('[providers.bad]\nkind="callable"\nentrypoint="nonexistent:bad"')
    source = """
import sys
from gnomon.cli import main
assert main(['infer', '--provider', 'last_value', '--request', '{"history":[1,2],"horizon":1}']) == 0
from gnomon import GnomonSession
from gnomon.mcp_server import _handle
session = GnomonSession.from_config()
assert _handle({'method':'tools/list'}, session=session)['tools']
reply = _handle({'method':'tools/call', 'params': {'name':'gnomon_forecast',
                'arguments': {'provider':'last_value', 'request': {'history':[1,2], 'horizon':1}}}}, session=session)
assert reply['isError'] is False
for name in ('toolspec', 'runtime', 'evaluation', 'publication', 'llm_dossier', 'context_intelligence'):
    assert 'gnomon.' + name not in sys.modules, name
"""
    result = subprocess.run([sys.executable, "-c", source], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_provider_load_failure_names_the_safe_root_cause(tmp_path, capsys):
    path = config(tmp_path, '[providers.custom]\nkind="callable"\nentrypoint="missing_forecast_package:model"\n')
    assert main(["capabilities", "--providers-config", str(path)]) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["error"]["code"] == "PROVIDER_LOAD_FAILED"
    assert payload["error"]["details"] == {
        "provider": "custom", "entrypoint": "missing_forecast_package:model",
        "stage": "import_module", "missing_module": "missing_forecast_package",
    }
    assert payload["error"]["repair_options"][0]["action"] == "install_provider_dependency"


def test_cli_provider_attribute_failure_is_actionable(tmp_path, capsys):
    path = config(tmp_path, '[providers.custom]\nkind="callable"\nentrypoint="json:not_a_real_provider"\n')
    assert main(["capabilities", "--providers-config", str(path)]) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["error"]["code"] == "PROVIDER_LOAD_FAILED"
    assert payload["error"]["details"]["stage"] == "resolve_attribute"


def test_cli_unexpected_provider_failure_keeps_secrets_redacted(tmp_path, capsys):
    path = config(tmp_path, f'[providers.bad]\nkind="callable"\nentrypoint="{__name__}:secret_failure"\n')
    assert main(["infer", "--providers-config", str(path), "--provider", "bad",
                 "--request", json.dumps(request())]) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == "" and "supersecret" not in captured.out
    assert payload["error"]["code"] == "EXECUTION_FAILED"
    detail = payload['error']['details']
    assert detail['stage'] == 'provider_execution' and detail['provider'] == 'bad'
    assert detail['error_category'] == 'execution' and detail['diagnostic_ref'].startswith('failure_')
    assert payload['execution_diagnostics']['provider_calls'] == 1


def test_unexpected_provider_exceptions_are_secret_safe_at_mcp_boundary():
    engine = InferenceEngine()
    def failure(r):
        raise RuntimeError("token=supersecret")
    engine.register("bad", failure)
    result = _handle(call_message("gnomon_forecast", {"provider": "bad", "request": request()}), session=GnomonSession(engine))
    assert result["isError"] and "supersecret" not in json.dumps(result)


@pytest.mark.parametrize("changes", [{"history": [True, 2]}, {"history": ["1", 2]}, {"series_id": {}}, {"unit": 1}])
def test_json_contract_rejects_values_that_violate_its_schema(changes):
    with pytest.raises(ForecastAdapterError):
        ForecastRequest.from_dict({**request(), **changes})


def test_config_and_session_reject_truthy_permission_strings(tmp_path):
    with pytest.raises(ForecastAdapterError):
        GnomonSession.from_config(config(tmp_path, 'allow_outcome_writes="false"\n'))


def test_cli_ledger_uses_the_same_typed_dispatch(tmp_path, capsys):
    path = config(tmp_path)
    with GnomonSession.from_config(path):
        pass  # Reads require a ledger previously initialized for recording.
    assert main(["ledger", "--providers-config", str(path), "--arguments", '{"operation":"pending"}']) == 0
    assert json.loads(capsys.readouterr().out)["result"] == []
