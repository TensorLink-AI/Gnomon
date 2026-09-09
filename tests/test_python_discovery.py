"""First-use Python requests and discoverable CLI/MCP contracts."""

import json

import pytest

from gnomon import AdapterCapabilities, ForecastRequest, ForecastResult, GnomonSession, InferenceEngine
from gnomon.cli import main
from gnomon.contracts import GnomonError
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.session import ledger_schema


def test_dict_forecasts_match_typed_requests_cli_and_ledger(tmp_path, capsys):
    config = tmp_path / "providers.toml"
    config.write_text('ledger_path="ledger.db"\ncache_size=4\n')
    request = {"history": [1, 2, 3, 4], "horizon": 3, "season": 2}
    with GnomonSession.from_config(config) as session:
        direct = session.forecast("seasonal_naive", request)
        typed = session.forecast("seasonal_naive", ForecastRequest.from_dict(request))
        assert typed["cache_hit"] is True
        assert direct["fingerprint"] == typed["fingerprint"]
        assert direct["result"]["point"] == (3, 4, 3)
        request["history"][-1] = 100
        stored = session.ledger.execution(direct["execution_id"])["request"]
        assert stored["history"] == [1, 2, 3, 4]
    assert main(["infer", "--provider", "seasonal_naive", "--request", json.dumps(stored),
                 "--providers-config", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["fingerprint"] == direct["fingerprint"]


@pytest.mark.parametrize("payload,message", [
    (None, "ForecastRequest or dict"), ([], "ForecastRequest or dict"),
    ('{"history":[1],"horizon":1}', "ForecastRequest or dict"),
    ({}, "missing forecast request fields: history, horizon"),
    ({"history": [1], "horizon": 1, "season_length": 7}, "unknown forecast request fields: season_length"),
])
def test_invalid_python_requests_fail_before_dispatch_with_actionable_errors(payload, message):
    calls = []
    with InferenceEngine() as engine:
        engine.register("test", lambda r: calls.append(r))
        with GnomonSession(engine) as session:
            with pytest.raises(ForecastAdapterError, match=message):
                session.forecast("test", payload)
            assert not calls


def test_empty_engine_explains_registration_and_built_in_entry_point():
    with InferenceEngine() as engine:
        with pytest.raises(ForecastAdapterError) as exc:
            engine.forecast("last_value", {"history": [1], "horizon": 1})
        assert "engine.register" in str(exc.value)
        assert "GnomonSession.from_config()" in str(exc.value)
    with GnomonSession.from_config() as session:
        assert session.forecast("last_value", {"history": [1], "horizon": 1})["result"]["point"] == (1,)


def test_engine_batches_accept_mixed_requests_and_validate_before_dispatch():
    calls = []
    def forecast(request):
        calls.append(request)
        return ForecastResult((request.history[-1],) * request.horizon)
    with InferenceEngine() as engine:
        engine.register("custom", forecast)
        batch = [{"history": [1, 2], "horizon": 1}, ForecastRequest((3, 4), 2)]
        assert [run.result.point for run in engine.forecast_batch("custom", batch)] == [(2,), (4, 4)]
        assert all(isinstance(request, ForecastRequest) for request in calls)
        calls.clear()
        with pytest.raises(ForecastAdapterError, match="ForecastRequest or dict"):
            engine.forecast_batch("custom", [batch[0], None])
        assert not calls


def test_provider_schemas_disclose_season_and_declared_limits_without_mutating_shared_schema():
    with GnomonSession.from_config() as session:
        caps = session.call("gnomon_capabilities", {})
        # The built-in discovery response fits the ordinary MCP response budget.
        assert not caps.get("partial")
        seasonal = caps['request_schemas'][caps['providers']['seasonal_naive']['request_schema_ref'].rsplit('/', 1)[1]]
        assert seasonal["properties"]["season"]["default"] == 1
        assert "--season" in seasonal["properties"]["season"]["description"]
        assert "season_length" not in seasonal["properties"]
        assert seasonal["additionalProperties"] is False
        assert seasonal["properties"]["quantiles"]["maxItems"] == 0
        session.engine.register("limited", lambda request: ForecastResult((1,) * request.horizon),
                                capabilities=AdapterCapabilities(min_history=4, max_horizon=2, frequencies=("D",), future_covariates=True))
        limited = session.capabilities(brief=False)["providers"]["limited"]["request_schema"]["properties"]
        assert limited["history"]["minItems"] == 4
        assert limited["horizon"]["maximum"] == 2
        assert limited["frequency"]["enum"] == [None, "D"]
        assert "maxItems" not in limited["future_covariates"]
        assert "maxItems" not in limited["future_covariates"]["items"]
        with pytest.raises(ForecastAdapterError, match="history"):
            session.forecast("limited", {"history": [1], "horizon": 1})
        limited["history"]["minItems"] = 900
        assert session.capabilities(brief=False)["providers"]["last_value"]["request_schema"]["properties"]["history"]["minItems"] == 1


def test_top_level_help_shows_nested_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "self-check leakage" in help_text
    assert "mcp serve" in help_text


def test_ledger_discovery_explains_permissions_units_and_cutoffs(tmp_path):
    with GnomonSession.from_config(ledger_path=tmp_path / "ledger.db") as session:
        with pytest.raises(GnomonError) as exc:
            session.call("gnomon_ledger", {"operation": "append_actual"})
        assert exc.value.code == "OUTCOME_WRITES_DISABLED"
        assert exc.value.details["config_setting"] == "allow_outcome_writes = true"
        assert exc.value.details["cli_option"] == "--providers-config"
    variants = {s["properties"]["operation"]["const"]: s for s in ledger_schema()["oneOf"]}
    assert "all units" in variants["search"]["properties"]["unit"]["description"]
    actuals = variants["actuals_as_of"]["properties"]
    assert "only unitless" in actuals["unit"]["description"]
    assert "unbounded" in actuals["source_as_of"]["description"]
