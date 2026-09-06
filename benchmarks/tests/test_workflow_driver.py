"""Scripted local HTTP exercises the real driver; no model-quality evidence."""

from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import sys
import threading

import pytest

from benchmarks.workflow.driver import endpoint, run
from benchmarks.workflow.matched import ARMS, prepare, public_context, normalized_rows
from benchmarks.workflow.run_workflow import case_payload, run_command
from benchmarks.workflow.schema import Case
from benchmarks.workflow.scoring import score_run

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "benchmarks/workflow/driver.py"


@pytest.fixture
def model_server():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, dict(self.headers), body))
            if self.path.startswith("/redirect"):
                self.send_response(307)
                self.send_header("Location", "/stolen/chat/completions")
                self.end_headers()
                return
            if len(body["messages"]) == 2:
                name = body["tools"][0]["function"]["name"]
                arguments = ({"provider": "last_value", "request": {"history": [3, 7], "horizon": 1}}
                             if name == "gnomon_forecast" else {})
            else:
                name, arguments = "submit_answer", {"status": "answered", "support": "supported", "numbers": {"next": 7}}
            value = {"choices": [{"message": {"role": "assistant", "content": None,
                     "tool_calls": [{"id": "call", "type": "function", "function": {
                         "name": name, "arguments": json.dumps(arguments)}}]}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01}}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def experiment(tmp_path, monkeypatch, model_server):
    url, requests = model_server
    module = tmp_path / "fixture_backend.py"
    module.write_text('''from gnomon import GnomonSession
from benchmarks.workflow.bounded_agent import ToolReply
class Backend:
    startup_cost_usd = 0
    def __init__(self, *, case, options, workspace, timeout):
        assert 'experiment' not in case and 'oracle' not in case
        assert workspace.is_dir() and timeout > 0
        self.session = GnomonSession.from_config()
        self.marker = options['marker']
    def tools(self):
        forecast = next(t for t in self.session.tools() if t['name'] == 'gnomon_forecast')
        return [forecast, {'name':self.marker, 'description':'Scripted binding marker, not an ordinary software baseline', 'inputSchema':{'type':'object'}}]
    def call(self, name, arguments, *, timeout):
        return ToolReply(self.session.call(name, arguments), 0)
    def close(self):
        self.session.close()
make = Backend
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("TEST_DRIVER_TOKEN", "fixture-only-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-be-used")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Answer the supplied task with available tools.")
    provider = tmp_path / "driver.json"
    config = {"schema_version": 1, "llm": {"base_url": url + "/prefix", "token_env": "TEST_DRIVER_TOKEN"},
              "backends": {arm: {"factory": "fixture_backend:make", "options": {"marker": arm}} for arm in ARMS}}
    provider.write_text(json.dumps(config))
    spec = {"schema_version": 1, "evidence_kind": "scripted", "command": [sys.executable, str(DRIVER)],
            "driver_files": [str(DRIVER), str(module)],
            "common": {"model": {"id": "scripted/not-a-real-model", "revision": None},
                       "generation": {"temperature": 0, "max_output_tokens": 100},
                       "prompt_file": str(prompt), "provider_config_file": str(provider),
                       "budget": {"timeout_seconds": 5, "jobs": 1, "infrastructure_retries": 0,
                                  "max_tool_calls": 2, "max_rounds": 3, "max_tokens": 1000}},
            "arms": {arm: {"description": "Fixture: all use same real Gnomon backend, NOT a surface ablation",
                           "tool_contract": "current forecast plus binding marker", "guidance": f"Marker is {arm}."} for arm in ARMS}}
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(spec))
    case = Case.from_dict({"id": "next", "kind": "synthetic", "domain": "forecasting",
                          "question": "Use last value to forecast the next observation.",
                          "available_at_cutoff": {"series": [3, 7]}, "answer_schema": {"numbers": ["next"]},
                          "oracle": {"numbers": {"next": 7}}})
    def contract(arm="lean"):
        return prepare(path, [case], command=shlex.join(spec["command"]), arm=arm, timeout=5, jobs=1, retries=0)
    return case, contract, config, provider, requests


@pytest.mark.parametrize("arm", ARMS)
def test_real_shared_command_binds_all_controls_and_current_forecast(experiment, tmp_path, arm):
    case, contract, _, _, requests = experiment
    identity = contract(arm)
    observation = run_command([case], shlex.join(identity["pinned"]["command"]), 5,
                              experiment=identity, checkpoint_path=tmp_path / "observations.jsonl")[0]
    assert observation.status == "answered" and observation.numbers["next"] == 7
    assert observation.metadata["experiment_id"] == identity["experiment_id"]
    assert observation.metadata["arm"] == arm
    assert observation.metadata["model"]["revision"] is None
    assert [t["name"] for t in observation.metadata["tool_inventory"]] == ["gnomon_forecast", arm]
    assert observation.cumulative_tokens == 30 and observation.cost_usd == 0.02
    assert len(requests) == 2
    for path, headers, body in requests:
        assert path == "/prefix/chat/completions" and headers["Authorization"] == "Bearer fixture-only-token"
        assert body["model"] == "scripted/not-a-real-model" and body["temperature"] == 0 and body["max_tokens"] == 100
        assert body["tool_choice"] == "auto"
        content = json.dumps(body["messages"])
        assert "oracle" not in content and "fixture-only-token" not in content and "provider_config_file" not in content
    rows = normalized_rows(score_run([case], [observation]))
    assert rows[0]["completed"] and rows[0]["budget_exceeded"] is False
    assert rows[0]["temporal_leakage"] is None


def payload(experiment):
    case, contract, *_ = experiment
    return {**case_payload(case), "experiment": public_context(contract())}


def test_changed_pinned_config_refused_before_import_or_network(experiment):
    case = payload(experiment)
    experiment[3].write_text("{}")
    with pytest.raises(ValueError, match="pinned content"):
        run(case)
    assert experiment[4] == []


def test_missing_named_token_never_falls_back_to_ambient_key(experiment, monkeypatch):
    monkeypatch.delenv("TEST_DRIVER_TOKEN")
    with pytest.raises(ValueError, match="credential is absent"):
        run(payload(experiment))
    assert experiment[4] == []


def test_redirect_does_not_forward_credential_and_preserves_unknown_usage(experiment):
    case = payload(experiment)
    config = deepcopy(experiment[2])
    config["llm"]["base_url"] = config["llm"]["base_url"].replace("/prefix", "/redirect")
    data = json.dumps(config).encode()
    experiment[3].write_bytes(data)
    case["experiment"]["common"]["provider_config_sha256"] = hashlib.sha256(data).hexdigest()
    result = run(case)
    assert result.status == "error" and result.cost_usd is None
    assert "cumulative_tokens" not in result.metadata["resource_fields"]
    assert len(experiment[4]) == 1 and experiment[4][0][0].startswith("/redirect/")


@pytest.mark.parametrize("url", ["https://example.invalid", "http://example.invalid", "http://localhost:1234",
    "https://user:pass@example.invalid", "https://example.invalid?key=secret", "https://example.invalid/#fragment",
    "https://example.invalid\n"])
def test_unsafe_or_unapproved_endpoint_refused_without_network(url):
    with pytest.raises(ValueError):
        endpoint(url, allow_model_requests=False)


def test_remote_opt_in_still_requires_tls():
    assert endpoint("https://example.invalid/prefix/", allow_model_requests=True) == "https://example.invalid/prefix"
    with pytest.raises(ValueError):
        endpoint("http://example.invalid", allow_model_requests=True)
