import io
import json

import pytest

from gnomon.contracts import GnomonError
from gnomon.sources import materialize_prometheus, materialize_stdin


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
