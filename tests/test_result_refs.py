import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gnomon.contracts import GnomonError
from gnomon.result_refs import ResultLimits, ResultReferences, encode
from gnomon import GnomonSession, ForecastResult, InferenceEngine
from gnomon.mcp_server import serve, _handle, PROTOCOL_VERSION


def collect(store, ref, pointer=""):
    pages, offset = [], 0
    while True:
        page = store.read(ref, pointer=pointer, offset=offset, max_chars=73)
        assert len(encode(page).encode("utf-8")) <= store.limits.max_response_bytes
        pages.append(page["text"])
        if page["next_offset"] is None:
            return "".join(pages), page["root_sha256"]
        assert page["next_offset"] > offset
        offset = page["next_offset"]


def test_unicode_json_pages_round_trip_exactly_and_pointer_is_not_a_path():
    store = ResultReferences(ResultLimits(max_response_bytes=2048))
    value = {"result": {"point": list(range(1200)), "a/~": "🌡\"\\\n" * 1000}, "unit": "°C"}
    try:
        receipt = store.project(value)
        assert receipt["partial"] and "result" not in receipt
        ref = receipt["result_ref"]
        text, digest = collect(store, ref)
        assert json.loads(text) == value
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == digest
        selected, _ = collect(store, ref, "/result/a~1~0")
        assert json.loads(selected) == value["result"]["a/~"]
        assert json.loads(collect(store, ref, "/result/point/12")[0]) == 12
        with pytest.raises(GnomonError, match="unknown"):
            store.read("/etc/passwd")
        for pointer in ("/result/point/01", "/result/~x", "/result/~~0", "/absent"):
            with pytest.raises(GnomonError):
                store.read(ref, pointer=pointer)
    finally:
        store.close()


def test_retention_eviction_integrity_and_cleanup(tmp_path):
    store = ResultReferences(ResultLimits(max_response_bytes=2048, max_result_bytes=8192,
                                        max_retained_bytes=8192, max_results=2))
    one = store.put({"a": "x" * 3000})
    two = store.put({"a": "y" * 3000})
    store.value(one)  # refresh the retained receipt, not the underlying model
    three = store.put({"a": "z" * 3000})
    assert store.value(one)["a"].startswith("x")
    with pytest.raises(GnomonError, match="expired"):
        store.value(two)
    path = store._entries[three][0]
    assert path.stat().st_mode & 0o777 == 0o600
    path.write_text("{}")
    with pytest.raises(GnomonError, match="integrity"):
        store.value(three)
    directory = Path(store._directory.name)
    store.close()
    assert not directory.exists()


def test_oversized_result_does_not_evict_prior_receipts_and_discloses_completed_work():
    store = ResultReferences(ResultLimits(max_response_bytes=2048, max_result_bytes=3000, max_retained_bytes=6000))
    try:
        old = store.put({"x": 2})
        with pytest.raises(GnomonError) as caught:
            store.project({"execution_id": "run-1", "recorded": True, "result": "x" * 3100})
        error = caught.value.to_dict()["error"]
        assert error["code"] == "RESULT_RETENTION_LIMIT"
        assert error["details"]["execution_id"] == "run-1"
        assert error["details"]["operation_completed"] is True
        assert store.value(old) == {"x": 2}
        assert len(list(Path(store._directory.name).iterdir())) == 1
    finally:
        store.close()


def test_forecast_projection_retrieves_without_rerunning_and_full_mode_is_explicit():
    calls = []
    def forecast(request):
        calls.append(request)
        return ForecastResult(tuple(range(request.horizon)))
    engine = InferenceEngine()
    engine.register("user", forecast)
    with GnomonSession(engine) as session:
        args = {"provider": "user", "request": {"history": [1, 2], "horizon": 3000}}
        result = session.call("gnomon_forecast", args)
        assert result["status"] == "result_available" and result["action_authorized"] is False
        ref = result["result_ref"]
        page = session.call("gnomon_read", {"result_ref": ref, "pointer": "/result/point/2999"})
        assert json.loads(page["text"]) == 2999 and len(calls) == 1
        full = session.call("gnomon_forecast", args, compact=False)
        assert len(full["result"]["point"]) == 3000 and "result_ref" not in full
    with pytest.raises(GnomonError, match="unknown"):
        session.results.read(ref)


def test_malformed_rpc_lines_do_not_kill_or_execute_the_server(monkeypatch):
    import gnomon.mcp_server as server
    monkeypatch.setattr(server, "MAX_REQUEST_BYTES", 512)
    output = StringIO()
    malformed = ["[]", "null", "false", "{", '{"method":"ping","id":NaN}',
                 '{"method":"ping","id":1e999}', '{"method":"ping","params":[]}',
                 '{"method":"\\ud800","id":1}', '{"method":"ping","id":true}',
                 '{"method":"ping","id":"' + "x" * 260 + '"}',
                 '"' + "🌡" * 800 + '"']
    source = StringIO("\n".join([*malformed, '{"id":99,"method":"ping"}']) + "\n")
    with GnomonSession.from_config() as session:
        assert serve(source, output, session=session) == 0
    responses = list(map(json.loads, output.getvalue().splitlines()))
    assert len(responses) == len(malformed) + 1
    assert all("error" in r for r in responses[:-1])
    assert responses[-1] == {"jsonrpc": "2.0", "id": 99, "result": {}}


def test_bad_tool_arguments_are_not_coerced_to_empty_and_version_is_supported():
    with GnomonSession.from_config() as session:
        for arguments in ([], None, 0, ""):
            result = _handle({"method": "tools/call", "params": {
                "name": "gnomon_capabilities", "arguments": arguments}}, session=session)
            assert result["isError"]
    assert _handle({"method": "initialize", "params": {"protocolVersion": "invented"}})["protocolVersion"] == PROTOCOL_VERSION


def test_real_stdio_retrieves_large_results_and_survives_invalid_utf8(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    process = subprocess.Popen([sys.executable, "-m", "gnomon", "mcp", "serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmp_path, env=env)
    def exchange(message):
        process.stdin.write(json.dumps(message).encode() + b"\n")
        process.stdin.flush()
        line = process.stdout.readline()
        assert len(line) < 4 * 8192 + 1024
        return json.loads(line)
    try:
        process.stdin.write(b"\xff\n")
        process.stdin.flush()
        assert "error" in json.loads(process.stdout.readline())
        result = exchange({"id": 1, "method": "tools/call", "params": {
            "name": "gnomon_forecast", "arguments": {"provider": "last_value",
                "request": {"history": [1, 7], "horizon": 4000}}}})["result"]["structuredContent"]
        assert result["partial"] and result["summary"]["evidence"] == "inference_only"
        read = exchange({"id": 2, "method": "tools/call", "params": {
            "name": "gnomon_read", "arguments": {"result_ref": result["result_ref"], "pointer": "/result/point/3999"}}})
        assert json.loads(read["result"]["structuredContent"]["text"]) == 7
        process.stdin.close()
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_large_cli_output_is_full_and_never_an_expired_reference(tmp_path):
    result = subprocess.run([sys.executable, "-m", "gnomon", "infer", "--provider", "last_value",
        "--request", json.dumps({"history": [1, 7], "horizon": 4000})],
        text=True, capture_output=True, timeout=30, cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["result"]["point"] == [7] * 4000 and "result_ref" not in payload


@pytest.mark.parametrize("options", [{"max_results": 0}, {"max_response_bytes": True},
    {"max_response_bytes": 100}, {"max_result_bytes": 8000}, {"max_retained_bytes": 10000}])
def test_invalid_operator_capacities_fail(options):
    with pytest.raises(ValueError):
        ResultLimits(**options)


def test_large_tool_errors_share_the_bounded_projection_and_never_suggest_retry():
    for maximum in (3000, 30000):
        with GnomonSession.from_config() as session:
            session.results.close()
            session.results = ResultReferences(ResultLimits(max_response_bytes=2048,
                max_result_bytes=maximum, max_retained_bytes=maximum))
            result = _handle({"method": "tools/call", "params": {
                "name": "gnomon_capabilities", "arguments": {"x" * 10000: True}}}, session=session)
            assert result["isError"]
            error = result["structuredContent"]
            assert len(encode(error).encode("utf-8")) <= 2048
            assert error["error"]["repair_options"]
            if maximum == 3000:
                assert error["error"]["code"] == "ERROR_DETAIL_LIMIT"
            else:
                assert error["partial"] and error["error"]["retryable"] is False
                original = session.results.value(error["result_ref"])
                assert original["error"]["code"] == "INVALID_ARGUMENTS"
