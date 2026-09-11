"""Adapter kinds: dependency-free wiring tests, plus per-package smoke tests that skip when the package is absent."""

import json
import math

import pytest

from gnomon import GnomonSession
from gnomon.adapters import ADAPTERS, _base, adapter_kinds, install_command
from gnomon.cli import main
from gnomon.forecast_adapter import ForecastAdapterError, ForecastRequest
from gnomon.session import configuration_schema, resolved_configuration

HISTORY = [round(10 + 3 * math.sin(i / 2) + 0.1 * i, 6) for i in range(40)]
STAMPS = [f"2026-01-{d:02d}T00:00:00+00:00" for d in range(1, 32)] + [f"2026-02-{d:02d}T00:00:00+00:00" for d in range(1, 10)]
FUTURE = [f"2026-02-{d:02d}T00:00:00+00:00" for d in range(10, 13)]


def config(tmp_path, body):
    path = tmp_path / "providers.toml"
    path.write_text("schema_version = 1\n[providers.p]\n" + body)
    return path


# --- wiring, no third-party package needed ------------------------------------

def test_registry_and_configuration_schema_expose_every_kind():
    kinds = adapter_kinds()
    assert set(kinds) == set(ADAPTERS)
    for kind, entry in kinds.items():
        assert entry["install"] == install_command(ADAPTERS[kind]) == f"python -m pip install 'gnomon-forecast[{kind}]'"
        assert type(entry["installed"]) is bool and entry["default_model"] and entry["summary"]
    providers = configuration_schema()["properties"]["providers"]["additionalProperties"]
    assert providers["properties"]["kind"]["enum"] == ["ephemeris", "callable", "factory", *ADAPTERS]
    assert providers["adapter_kinds"] == kinds


def test_config_schema_cli_lists_adapter_kinds_without_loading_them(capsys):
    assert main(["capabilities", "--config-schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert set(schema["properties"]["providers"]["additionalProperties"]["adapter_kinds"]) == set(ADAPTERS)


def test_missing_adapter_package_names_the_install_extra(tmp_path, monkeypatch, capsys):
    import gnomon.session as session_module

    def absent(kind, name, spec):
        raise ModuleNotFoundError("No module named 'statsforecast'", name="statsforecast")
    monkeypatch.setattr(session_module, "build_provider", absent)
    path = config(tmp_path, 'kind = "statsforecast"\n')
    assert main(["capabilities", "--providers-config", str(path)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "PROVIDER_LOAD_FAILED"
    assert payload["error"]["details"] == {
        "provider": "p", "kind": "statsforecast", "stage": "import_adapter_dependency",
        "missing_module": "statsforecast", "install": install_command(ADAPTERS["statsforecast"])}
    assert payload["error"]["repair_options"][0]["action"] == "install_provider_dependency"


def test_adapter_spec_rejects_unknown_and_invalid_fields(tmp_path, capsys):
    path = config(tmp_path, 'kind = "statsforecast"\nentrypoint = "x:y"\n')
    assert main(["capabilities", "--providers-config", str(path)]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["details"]["rejected_fields"] == ["entrypoint"]
    path = config(tmp_path, 'kind = "nope"\n')
    assert main(["capabilities", "--providers-config", str(path)]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["details"]["rejected_fields"] == ["providers.p.kind"]


def test_resolved_configuration_reports_adapter_model_without_importing(tmp_path):
    path = config(tmp_path, 'kind = "prophet"\nmodel = "Prophet"\nseed = 4\n[providers.p.options]\nuncertainty_samples = 10\n')
    resolved = resolved_configuration(path)
    assert resolved["providers"]["p"] == {"kind": "prophet", "model": "Prophet", "seed": 4,
                                          "entrypoint_imported": False, "remote_endpoint_configured": False}


def test_unknown_provider_that_matches_an_adapter_kind_gets_an_operator_hint():
    with GnomonSession.from_config() as session:
        with pytest.raises(ForecastAdapterError, match=r"adapter kind, not a registered provider name") as info:
            session.forecast("prophet", {"history": [1, 2, 3], "horizon": 1})
        assert "gnomon-forecast[prophet]" in str(info.value)
        with pytest.raises(ForecastAdapterError) as info:
            session.forecast("nothing", {"history": [1, 2, 3], "horizon": 1})
        assert "adapter kind" not in str(info.value)


# --- shared helpers -------------------------------------------------------------

def test_levels_and_quantile_row_repair():
    assert _base.levels([0.1, 0.5, 0.9]) == [80]
    assert _base.levels([0.025, 0.975, 0.3]) == [40, 95]
    assert _base.level_for(0.025) == 95 and _base.level_for(0.1) == 80
    request = ForecastRequest((1.0, 2.0, 3.0), 2, quantiles=(0.1, 0.5, 0.9))
    rows, fixed = _base.make_rows(request, lambda q: {0.1: [1.0, 5.0], 0.5: [2.0, 4.0], 0.9: [3.0, 6.0]}[q])
    assert fixed == 1 and rows[1] == {0.1: 5.0, 0.5: 5.0, 0.9: 6.0} and rows[0] == {0.1: 1.0, 0.5: 2.0, 0.9: 3.0}
    result = _base.finish(request, [1.5, 2.5], kind="test", quantiles=rows, fixed=fixed)
    assert result.metadata["quantile_crossings_repaired"] == 1 and result.metadata["adapter_kind"] == "test"
    assert _base.make_rows(ForecastRequest((1.0, 2.0), 1), lambda q: [0.0]) == (None, 0)


def test_paired_covariates_and_date_requirements():
    # Dependency-free paths first: no numpy or pandas is imported for undated, covariate-free requests.
    assert _base.paired_covariates(ForecastRequest((1.0, 2.0), 1)) == (None, None, ())
    with pytest.raises(ForecastAdapterError, match="both past_covariates and future_covariates"):
        _base.paired_covariates(ForecastRequest((1.0, 2.0, 3.0), 2, past_covariates=((1.0,), (2.0,), (3.0,))))
    with pytest.raises(ForecastAdapterError, match="needs request timestamps"):
        _base.require_dates("prophet", ForecastRequest((1.0, 2.0), 1))
    undated = ForecastRequest((1.0, 2.0, 3.0), 2)
    index, freq, note = _base.history_index(undated)
    assert list(index) == [0, 1, 2] and freq is None and note == "integer_index_no_timestamps_supplied"
    assert list(_base.future_index(undated, index, freq)) == [3, 4]
    pytest.importorskip("numpy")
    both = ForecastRequest((1.0, 2.0, 3.0), 2, past_covariates=((1.0,), (2.0,), (3.0,)),
                           future_covariates=((4.0,), (5.0,)), future_covariate_names=("x",))
    hist, fut, names = _base.paired_covariates(both)
    assert hist.shape == (3, 1) and fut.shape == (2, 1) and names == ("x",)
    pytest.importorskip("pandas")
    dated = ForecastRequest(tuple(HISTORY), 3, timestamps=tuple(STAMPS), frequency="D")
    index, freq = _base.require_dates("prophet", dated)
    assert freq == "D" and [str(t.date()) for t in _base.future_index(dated, index, freq)] == ["2026-02-10", "2026-02-11", "2026-02-12"]
    inferred = ForecastRequest(tuple(HISTORY), 3, timestamps=tuple(STAMPS))
    assert _base.history_index(inferred)[1] == "D"


# --- per-package smoke tests, skipped when the package is not installed ------

CASES = {
    "statsforecast": ('kind = "statsforecast"\nmodel = "AutoETS"\n', ("statsforecast",), {}),
    "statsmodels": ('kind = "statsmodels"\nmodel = "ETS"\nseed = 1\n', ("statsmodels",), {}),
    "prophet": ('kind = "prophet"\nseed = 1\n[providers.p.options]\nuncertainty_samples = 100\n', ("prophet",), {"dated": True, "samples": True}),
    "mlforecast": ('kind = "mlforecast"\nseed = 1\n[providers.p.options]\nlags = [1, 2, 3]\n'
                   '[providers.p.options.regressor]\nn_estimators = 10\nmin_child_samples = 2\n', ("mlforecast", "lightgbm"), {}),
    "skforecast": ('kind = "skforecast"\nseed = 1\n[providers.p.options]\nlags = 3\nn_boot = 20\n', ("skforecast",), {}),
    "sktime": ('kind = "sktime"\nmodel = "sktime.forecasting.theta:ThetaForecaster"\n', ("sktime",), {}),
    "darts": ('kind = "darts"\nmodel = "ExponentialSmoothing"\nseed = 1\n[providers.p.options]\nnum_samples = 30\n', ("darts",), {"samples": True}),
    "gluonts": ('kind = "gluonts"\nmodel = "gluonts.model.npts:NPTSPredictor"\nseed = 1\n[providers.p.options]\nnum_samples = 30\n', ("gluonts",), {"samples": True}),
    "neuralforecast": ('kind = "neuralforecast"\nmodel = "NHITS"\nseed = 1\n[providers.p.options]\nmax_steps = 3\n', ("neuralforecast", "torch"), {}),
}


@pytest.mark.parametrize("kind", sorted(CASES))
def test_adapter_forecasts_through_a_configured_session(tmp_path, kind):
    body, modules, flags = CASES[kind]
    for module in modules:
        pytest.importorskip(module)
    request = {"history": HISTORY, "horizon": 3, "season": 7, "quantiles": [0.1, 0.5, 0.9], "series_id": "s", "unit": "u"}
    if flags.get("dated"):
        request.update(timestamps=STAMPS, future_timestamps=FUTURE, frequency="D")
    with GnomonSession.from_config(config(tmp_path, body)) as session:
        caps = session.capabilities(brief=False)["providers"]["p"]
        assert caps["revision"].startswith(kind + "/") and caps["lifecycle"] == "fresh_per_request"
        assert caps["capabilities"]["quantiles"] is True
        for _ in range(2):  # a fresh fitting object per call
            result = session.forecast("p", request)["result"]
            assert len(result["point"]) == 3 and all(math.isfinite(v) for v in result["point"])
            assert len(result["quantiles"]) == 3
            for row in result["quantiles"]:
                assert list(row) == ["0.1", "0.5", "0.9"] or list(row) == [0.1, 0.5, 0.9]
                values = list(row.values())
                assert values == sorted(values)
            assert result["metadata"]["adapter_kind"] == kind and result["metadata"]["protocol_version"]
            assert result["series_id"] == "s" and result["unit"] == "u"
        if flags.get("samples"):
            sampled = session.forecast("p", {**request, "samples": 4})["result"]
            assert len(sampled["sample_paths"]) == 4 and all(len(path) == 3 for path in sampled["sample_paths"])


def test_prophet_refuses_to_invent_dates(tmp_path):
    pytest.importorskip("prophet")
    with GnomonSession.from_config(config(tmp_path, CASES["prophet"][0])) as session:
        with pytest.raises(Exception) as info:
            session.forecast("p", {"history": HISTORY, "horizon": 2})
        assert "timestamps" in str(info.value) or getattr(info.value, "details", {}).get("missing_fields") == ["timestamps"]


def test_adapters_accept_paired_covariates_and_reject_one_sided(tmp_path):
    pytest.importorskip("statsforecast")
    body = 'kind = "statsforecast"\nmodel = "AutoARIMA"\n'
    with GnomonSession.from_config(config(tmp_path, body)) as session:
        request = {"history": HISTORY, "horizon": 3, "season": 1, "timestamps": STAMPS, "future_timestamps": FUTURE,
                   "frequency": "D", "past_covariates": [[float(i % 7)] for i in range(40)],
                   "future_covariates": [[3.0], [4.0], [5.0]], "future_covariate_names": ["dow"]}
        assert len(session.forecast("p", request)["result"]["point"]) == 3
        one_sided = {k: v for k, v in request.items() if k not in ("future_covariates", "future_covariate_names")}
        with pytest.raises(Exception, match="past_covariates and future_covariates"):
            session.forecast("p", one_sided)
