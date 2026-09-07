"""Optional real MCP container combinations, not a paid agent comparison."""

import os
import json
from datetime import datetime, timedelta, timezone

import pytest

from benchmarks.workflow import service_backend
from benchmarks.workflow.service_backend import CombinedBackend, McpBackend, source_fingerprint

CASE = {"id": "service", "question": "Forecast the next value.", "available_at_cutoff": {"series": [3, 7],
        "files": {"series.csv": "timestamp,value\n2025-01-01T00:00:00+00:00,3\n2025-01-02T00:00:00+00:00,7\n"}}}


@pytest.fixture
def options():
    software = os.environ.get("GNOMON_TEST_SOFTWARE_IMAGE")
    service = os.environ.get("GNOMON_TEST_SERVICE_IMAGE")
    if not software or not service:
        pytest.skip("explicit software and service image IDs required; no automatic build/pull")
    return {"software_image": software, "service_image": service, "docker_host": "unix:///var/run/docker.sock"}


@pytest.fixture(params=["lean", "full"])
def backend(request, tmp_path, options):
    value = CombinedBackend(arm=request.param, case=CASE, options=options, workspace=tmp_path, timeout=45)
    try:
        yield value
    finally:
        value.close()


def test_actual_tools_keep_original_schema_and_identical_ordinary_access(backend):
    tools = backend.tools()
    assert tools[0]["name"] == "python"
    assert len({tool["name"] for tool in tools}) == len(tools)
    forecast = next(tool for tool in tools if tool["name"] == "gnomon_forecast")
    full = backend.service.provenance["feature_arm"] == "full"
    variants = forecast["inputSchema"]["oneOf"]
    assert all("provider" in variant["required"] for variant in variants)
    assert all("input" not in variant["properties"] for variant in variants)
    assert backend.service.provenance["feature_arm"] in {"lean", "full"}
    assert len(tools) == (10 if full else 7)
    reply = backend.call("python", {"code": "import json; from pathlib import Path; print(json.loads(Path('/tmp/case.json').read_text())['available_at_cutoff']['series'])"}, timeout=5)
    assert reply.value["stdout"].strip() == "[3, 7]"
    capabilities = backend.call("gnomon_capabilities", {}, timeout=5).value
    assert not capabilities["isError"]
    assert backend.provenance["service"]["software"]["gnomon_source_sha256"] == source_fingerprint()


def test_mcp_errors_repair_without_closing_or_changing_arguments(backend):
    failed = backend.call("gnomon_forecast", {"made_up": True}, timeout=5).value
    assert failed["isError"]
    assert not backend.service.closed
    valid = backend.call("gnomon_capabilities", {}, timeout=5).value
    assert not valid["isError"]


def test_current_lean_session_references_survive_across_calls(tmp_path, options):
    backend = service_backend.lean(case=CASE, options=options, workspace=tmp_path, timeout=30)
    try:
        result = backend.call("gnomon_forecast", {"provider": "last_value", "request": {"history": [3, 7], "horizon": 1}}, timeout=5).value
        assert not result["isError"]
        value = result["structuredContent"]
        assert value["result"]["point"] == [7]
        inspected = backend.call("gnomon_inspect", {"input": "/tmp/data/series.csv", "frequency": "D"}, timeout=5).value
        assert not inspected["isError"], inspected
        reference = inspected["structuredContent"]["data_ref"]
        described = backend.call("gnomon_describe", {"data_ref": reference, "statistic": "mean"}, timeout=5).value
        assert not described["isError"] and described["structuredContent"]["value"] == 5
        # Public case state and the installed library live in different containers;
        # Python cannot import/call Gnomon or modify the service's filesystem.
        result = backend.call("python", {"code": "import importlib.util; print(importlib.util.find_spec('gnomon'))"}, timeout=5).value
        assert result["stdout"].strip() == "None"
        assert backend.software.container != backend.service.container
    finally:
        backend.close()


def test_service_deadline_removes_both_containers_on_close(tmp_path, options):
    backend = service_backend.lean(case=CASE, options=options, workspace=tmp_path, timeout=30)
    try:
        with pytest.raises(TimeoutError):
            backend.service._rpc("ping", {}, timeout=0)
        assert backend.service.closed and backend.service.process is None
    finally:
        backend.close()
    assert backend.software.closed


def test_full_forecast_uses_same_contract_and_records_evidence(tmp_path, options):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    csv = "timestamp,value\n" + "".join(f"{(start + timedelta(days=i)).isoformat()},{10+i%7}\n" for i in range(28))
    case = {**CASE, "available_at_cutoff": {"files": {"history.csv": csv}}}
    backend = service_backend.full(case=case, options=options, workspace=tmp_path, timeout=45)
    try:
        inspected = backend.call("gnomon_inspect", {"input": "/tmp/data/history.csv",
                                                   "frequency": "D"}, timeout=5).value
        assert not inspected["isError"], inspected
        result = backend.call("gnomon_forecast", {"provider": "last_value",
            "data_ref": inspected["structuredContent"]["data_ref"], "horizon": 2}, timeout=30).value
        assert not result["isError"], result
        assert result["structuredContent"]
        identifier = result["structuredContent"]["execution_id"]
        saved = backend.call("gnomon_ledger", {"operation": "execution", "execution_id": identifier}, timeout=5).value
        assert not saved["isError"]
        assert saved["structuredContent"]["result"]["result"]["point"] == [16, 16]
        assert backend.service.provenance["execution_options"] == {"ledger": True, "temporal": True}
    finally:
        backend.close()


def test_wrong_installed_source_refused_before_serving(tmp_path, options, monkeypatch):
    monkeypatch.setattr(service_backend, "source_fingerprint", lambda: "wrong-checkout")
    with pytest.raises(ValueError, match="contents differ"):
        McpBackend(arm="lean", case=CASE,
                   options={"image": options["service_image"], "docker_host": options["docker_host"]},
                   workspace=tmp_path, timeout=30)


def test_cleanup_attempts_every_component_even_if_one_fails():
    calls = []
    class Backend:
        def __init__(self, fail):
            self.fail = fail
        def close(self):
            calls.append(self.fail)
            if self.fail:
                raise RuntimeError("unavailable daemon")
    backend = CombinedBackend.__new__(CombinedBackend)
    backend.service, backend.software = Backend(True), Backend(False)
    with pytest.raises(RuntimeError, match="could not confirm cleanup"):
        backend.close()
    assert calls == [True, False]


def test_package_identity_covers_nonpython_resources(tmp_path):
    (tmp_path / "module.py").write_text("value = 1\n")
    resource = tmp_path / "catalog.json"
    resource.write_text('{"version":1}')
    first = source_fingerprint(tmp_path)
    resource.write_text('{"version":2}')
    second = source_fingerprint(tmp_path)
    assert first != second
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/cache.pyc").write_bytes(b"unrelated-bytecode")
    assert source_fingerprint(tmp_path) == second


@pytest.mark.parametrize("arm,value", [("unknown", {"ledger": True}), ("lean", False),
    ("lean", {"ledger": "true"}), ("lean", {"allow_outcome_writes": True})])
def test_unsupported_startup_options_fail_before_container_work(arm, value):
    with pytest.raises(ValueError):
        service_backend._execution_options(arm, value)


def test_lean_ledger_and_temporal_options_are_startup_only(tmp_path, options):
    backend = service_backend.lean(case=CASE, options={**options, "execution_options": {"ledger": True, "temporal": True}},
                                   workspace=tmp_path, timeout=30)
    try:
        names = {tool["name"] for tool in backend.tools()}
        assert {"gnomon_ledger", "gnomon_route", "gnomon_temporal"} <= names
        result = backend.call("gnomon_forecast", {"provider": "last_value", "request": {"history": [3, 7],
            "horizon": 1, "series_id": "shop", "future_timestamps": ["2025-01-03T00:00:00Z"]}}, timeout=5).value
        assert not result["isError"]
        identifier = result["structuredContent"]["execution_id"]
        saved = backend.call("gnomon_ledger", {"operation": "execution", "execution_id": identifier}, timeout=5).value
        assert not saved["isError"]
        assert saved["structuredContent"]["result"]["result"]["point"] == [7]
        denied = backend.call("gnomon_ledger", {"operation": "append_actual", "series_id": "shop",
            "valid_time": "2025-01-03T00:00:00Z", "value": 8, "source_available_at": "2025-01-04T00:00:00Z"}, timeout=5).value
        assert denied["isError"] and denied["structuredContent"]["error"]["code"] == "OUTCOME_WRITES_DISABLED"
        # The test operator, not an exposed agent tool, reveals a synthetic actual.
        # This exercises the service/ledger path; it does not implement the staged
        # agent protocol or manufacture an answer on the agent's behalf.
        update = {"series_id": "shop", "valid_time": "2025-01-03T00:00:00Z", "value": 8,
                  "source_available_at": "2025-01-04T00:00:00Z"}
        script = "import json,sys; from gnomon import TemporalLedger; TemporalLedger('/tmp/ledger.sqlite3').append_actual(**json.load(sys.stdin))"
        backend.service._command(["exec", "--user=65534:65534", "-i", backend.service.container,
                                  "python", "-I", "-c", script], input=json.dumps(update).encode())
        scored = backend.call("gnomon_ledger", {"operation": "evaluate", "execution_id": identifier}, timeout=5).value
        assert scored["structuredContent"]["result"]["mae"] == 1
        original = backend.call("gnomon_ledger", {"operation": "execution", "execution_id": identifier}, timeout=5).value
        assert original["structuredContent"]["result"]["result"]["point"] == [7]
        assert backend.service.provenance["execution_options"] == {"ledger": True, "temporal": True}
    finally:
        backend.close()
