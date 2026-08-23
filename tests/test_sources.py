import io
import json

import pytest

from gnomon.contracts import GnomonError
from gnomon.sources import (
    materialize_agent_source, materialize_prometheus, materialize_stdin,
)


def test_stdin_is_bounded_and_materialized(tmp_path):
    path = materialize_stdin(io.StringIO("timestamp,value\n2026-01-01,1\n"))
    assert path.read_text(encoding="utf-8").startswith("timestamp,value")


def test_prometheus_range_response_becomes_panel_csv(monkeypatch):
    payload = json.dumps({"status": "success", "data": {"result": [{
        "metric": {"job": "api"}, "values": [[1, "2.5"], [2, "3.5"]],
    }]}}).encode()

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self, _): return payload

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    path = materialize_prometheus(
        "prom+https://metrics.example/api/v1/query_range?query=up&start=1&end=2&step=1"
    )
    text = path.read_text(encoding="utf-8")
    assert "timestamp,series,value" in text
    assert "job=api" in text


@pytest.mark.parametrize("reference", [
    "prom+https://user:secret@metrics.example/api/v1/query_range?query=up&start=1&end=2&step=1",
    "prom+https://metrics.example/api/v1/query?query=up&start=1&end=2&step=1",
    "prom+https://metrics.example/api/v1/query_range?query=up",
])
def test_prometheus_rejects_unsafe_or_incomplete_references(reference):
    with pytest.raises(GnomonError):
        materialize_prometheus(reference)


def test_agent_prometheus_source_requires_allowlisted_host(monkeypatch):
    monkeypatch.delenv("GNOMON_PROMETHEUS_ALLOWED_HOSTS", raising=False)
    with pytest.raises(GnomonError) as caught:
        materialize_agent_source(
            "prom+https://metrics.example/api/v1/query_range?"
            "query=up&start=1&end=2&step=1")
    assert caught.value.code == "PROMETHEUS_HOST_NOT_ALLOWED"


def test_mcp_inspect_uses_governed_prometheus_connector(monkeypatch):
    payload = json.dumps({"status": "success", "data": {"result": [{
        "metric": {"job": "api"},
        "values": [[index * 86400, str(index)] for index in range(1, 9)],
    }]}}).encode()

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self, _): return payload

    monkeypatch.setenv("GNOMON_MCP_PROFILE", "core")
    monkeypatch.setenv("GNOMON_PROMETHEUS_ALLOWED_HOSTS", "metrics.example")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    from gnomon.toolspec import runner_for
    result = runner_for("gnomon_inspect")({
        "input": (
            "prom+https://metrics.example/api/v1/query_range?"
            "query=up&start=1&end=9&step=86400"),
    })
    assert result["status"] == "valid"
    assert result["schema"]["series_column"] == "series"
    assert any("governed read-only Prometheus" in item
               for item in result["assumptions"])
