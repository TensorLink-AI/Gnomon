from dataclasses import replace
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from gnomon import ForecastRequest, InferenceEngine, EphemerisProvider
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.http_transport import InferenceHTTPError, JSONTransport


@pytest.fixture
def service():
    state = {"requests": [], "status": 200, "raw": None, "redirect": None}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            state["requests"].append(("GET", self.path, dict(self.headers), None))
            models = [
                {"name": "not-in-gnomon", "enabled": True, "healthy": True, "covariates": True},
                {"name": "offline", "enabled": True, "healthy": False},
                {"name": "disabled", "enabled": False, "healthy": True},
            ]
            self.respond(models if state.get("gateway") else {"models": models})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(("POST", self.path, dict(self.headers), body))
            series = [row["values"] for row in body["series"]] if state.get("gateway") else body["series"]
            response = {"forecasts": [
                {"quantiles": {str(q): [row[-1] + q] * body["horizon"] for q in body["quantiles"]}}
                for row in series
            ], "meta": {"mode": body["mode"], "models_used": [body.get("model", "remote-chosen")],
                        "request_id": "remote-request", "notes": []}}
            if state.get("transform"):
                response = state["transform"](response)
            self.respond(response)
        def respond(self, value):
            self.send_response(state["status"])
            if state["redirect"]:
                self.send_header("Location", state["redirect"])
            raw = state["raw"] if state["raw"] is not None else json.dumps(value).encode()
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}/custom/base", state
    server.shutdown()
    server.server_close()
    worker.join()


def test_discovery_and_short_history_inference_over_real_http(service, monkeypatch):
    url, state = service
    monkeypatch.setenv("TEST_EPHEMERIS_KEY", "supersecret")
    provider = EphemerisProvider(url, token_env="TEST_EPHEMERIS_KEY")
    engine = InferenceEngine()
    assert provider.register_models(engine) == ["ephemeris/not-in-gnomon"]
    run = engine.forecast("ephemeris/not-in-gnomon", ForecastRequest((1, 2), 2, quantiles=(.1, .9)))
    assert run.result.point == (2.5, 2.5)
    assert run.result.metadata["point_definition"] == "median"
    assert provider.name == "ephemeris/route"
    assert run.result.metadata["provider"] == "ephemeris"
    assert run.provider == "ephemeris/not-in-gnomon"
    assert run.result.metadata["revision_attested"] is False
    assert run.revision is None and run.evidence == "inference_only"
    assert state["requests"][-1][1] == "/custom/base/forecast"
    assert state["requests"][-1][2]["Authorization"] == "Bearer supersecret"
    assert state["requests"][-1][3]["quantiles"] == [.1, .5, .9]
    assert "supersecret" not in json.dumps(run.to_dict())


def test_native_batch_named_covariates_and_series_alignment(service):
    url, state = service
    engine = InferenceEngine()
    engine.register("remote", EphemerisProvider(url))
    req = ForecastRequest((1, 2), 1, frequency="D", series_id="one", unit="USD",
                          future_timestamps=("2025-01-03",),
                          past_covariates=((10, 20), (11, 21)), past_covariate_names=("price", "holiday"),
                          future_covariates=((22,),), future_covariate_names=("holiday",))
    result = engine.forecast_batch("remote", [req, replace(req, history=(3, 4), series_id="two")])
    assert [r.result.point for r in result] == [(2.5,), (4.5,)]
    assert [r.result.series_id for r in result] == ["one", "two"]
    assert len(state["requests"]) == 1
    assert state["requests"][0][3]["covariates"][0] == {
        "past": {"price": [10, 11], "holiday": [20, 21]}, "future": {"holiday": [22]}}


@pytest.mark.parametrize("automatic", [True, False])
def test_gateway_discovery_and_batch_contract(service, automatic):
    url, state = service
    state["gateway"] = True
    provider = EphemerisProvider(url + "/api/v1/" if automatic else url,
                                 api_format="auto" if automatic else "gateway")
    engine = InferenceEngine()
    assert provider.register_models(engine) == ["ephemeris/not-in-gnomon"]
    req = ForecastRequest((1, 2), 1, frequency="D", series_id="one", unit="widgets",
                          future_timestamps=("2025-01-03",), quantiles=(.1, .9),
                          past_covariates=((10,), (11,)), past_covariate_names=("price",),
                          future_covariates=((12,),), future_covariate_names=("price",))
    results = engine.forecast_batch("ephemeris/not-in-gnomon",
                                   [req, replace(req, history=(3, 4), series_id="two")])
    assert [r.result.point for r in results] == [(2.5,), (4.5,)]
    assert [r.result.series_id for r in results] == ["one", "two"]
    assert all(r.result.unit == "widgets" and r.result.timestamps == r.request.future_timestamps for r in results)
    body = state["requests"][-1][3]
    assert "freq" not in body and "covariates" not in body
    assert body["model"] == "not-in-gnomon"
    assert body["series"][0] == {"values": [1, 2], "freq": "D", "covariates": {
        "past": {"price": [10, 11]}, "future": {"price": [12]}}}
    assert body["quantiles"] == [.1, .5, .9]


def test_gateway_configuration_and_optional_frequency(service, tmp_path):
    from gnomon import GnomonSession
    url, state = service
    state["gateway"] = True
    config = tmp_path / "providers.toml"
    config.write_text(f'[providers.remote]\nkind="ephemeris"\nbase_url="{url}"\n'
                      'api_format="gateway"\ndiscover=true\n')
    with GnomonSession.from_config(config) as session:
        result = session.forecast("remote", {"history": [1, 2], "horizon": 1})
    assert tuple(result["result"]["point"]) == (2.5,)
    assert state["requests"][-1][3]["series"] == [{"values": [1.0, 2.0]}]


def test_format_override_and_invalid_discovery(service):
    url, state = service
    provider = EphemerisProvider(url + "/api/v1", api_format="direct")
    provider.forecast(ForecastRequest((1, 2), 1))
    assert state["requests"][-1][3]["series"] == [[1, 2]]
    for raw in (b'[1]', b'[{"name": 1}]', b'{"models": null}'):
        state["raw"] = raw
        with pytest.raises(ForecastAdapterError, match="models response"):
            provider.models()
    with pytest.raises(ForecastAdapterError, match="api_format"):
        EphemerisProvider(url, api_format="invalid")


@pytest.mark.parametrize("kwargs", [
    {"season": 7}, {"related_series": ((3, 4),)}, {"samples": 3},
    {"quantiles": (.1234567,)}, {"future_covariates": ((5,),)},
])
def test_unsupported_semantics_do_not_reach_service(service, kwargs):
    url, state = service
    with pytest.raises(ForecastAdapterError):
        EphemerisProvider(url).forecast(ForecastRequest((1, 2), 1, **kwargs))
    assert not state["requests"]


@pytest.mark.parametrize("transform", [
    lambda r: {**r, "forecasts": []},
    lambda r: {**r, "meta": {**r["meta"], "mode": "wrong"}},
    lambda r: {**r, "meta": {**r["meta"], "models_used": []}},
    lambda r: {**r, "forecasts": [{"quantiles": {"0.5": [float("nan")]}}]},
    lambda r: {**r, "forecasts": [{"quantiles": {"0.5": [3, 4]}}]},
    lambda r: {**r, "forecasts": [{"quantiles": {"0.1": [4], "0.5": [3]}}]},
])
def test_malformed_remote_results_fail_closed(service, transform):
    url, state = service
    state["transform"] = transform
    with pytest.raises(ForecastAdapterError):
        EphemerisProvider(url).forecast(ForecastRequest((1, 2), 1))


def test_no_implicit_post_retry_or_redirect_and_no_secret_in_error(service, monkeypatch):
    url, state = service
    monkeypatch.setenv("TEST_EPHEMERIS_KEY", "supersecret")
    provider = EphemerisProvider(url, token_env="TEST_EPHEMERIS_KEY")
    state["status"] = 503
    state["raw"] = b"upstream-error: supersecret"
    with pytest.raises(InferenceHTTPError) as error:
        provider.forecast(ForecastRequest((1, 2), 1))
    assert error.value.status == 503
    assert "supersecret" not in str(error.value)
    assert len(state["requests"]) == 1
    state["status"] = 302
    state["redirect"] = url + "/stolen"
    with pytest.raises(InferenceHTTPError) as error:
        provider.models()
    assert error.value.status == 302
    assert len(state["requests"]) == 2


def test_get_retries_are_bounded(service):
    url, state = service
    state["status"] = 503
    with pytest.raises(InferenceHTTPError):
        JSONTransport(url, get_retries=1).call("/models")
    assert len(state["requests"]) == 2


def test_response_size_and_json_shape_are_bounded(service):
    url, state = service
    transport = JSONTransport(url, max_bytes=100)
    state["raw"] = b"x" * 101
    with pytest.raises(InferenceHTTPError, match="byte limit"):
        transport.call("/models")
    for raw in (b"{invalid", b"[]"):
        state["raw"] = raw
        with pytest.raises(InferenceHTTPError, match="JSON"):
            transport.call("/models")
    state["requests"].clear()
    with pytest.raises(InferenceHTTPError, match="byte limit"):
        transport.call("/forecast", {"huge": "x" * 100})
    assert state["requests"] == []


@pytest.mark.parametrize("url, kwargs", [
    ("https://token@example.com", {}), ("https://example.com?key=secret", {}),
    ("file:///etc/passwd", {}), ("http://example.com", {}),
    ("http://example.com", {"allow_http": True, "token_env": "SECRET"}),
    ("https://example.com", {"timeout": 0}),
    ("https://example.com", {"get_retries": 4}),
    ("https://example.com", {"auth_header": "X-Token\r\nHost"}),
])
def test_transport_configuration_is_operator_owned_and_validated(url, kwargs):
    with pytest.raises(ForecastAdapterError):
        JSONTransport(url, **kwargs)
