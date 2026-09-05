"""Real optional container checks. No remote model call or agent-quality claim."""

import json
import io
import os
import subprocess
import time

import pytest

from benchmarks.workflow.process import ProcessLimit
from benchmarks.workflow.software_backend import SoftwareBackend

CASE = {"id": "software", "question": "Forecast from public history.", "available_at_cutoff": {"series": [3, 7]}}


@pytest.mark.parametrize("options", [{}, {"image": "latest", "docker_host": "unix:///var/run/docker.sock"},
    {"image": "sha256:" + "a"*64, "docker_host": "tcp://remote:2375"}])
def test_invalid_or_mutable_runtime_configuration_precedes_execution(tmp_path, options):
    with pytest.raises(ValueError):
        SoftwareBackend(case=CASE, options=options, workspace=tmp_path, timeout=10)


@pytest.mark.parametrize("files", [{"../outside": "x"}, {"/root/secret": "x"}, {"file.csv": 1},
                                 {str(i): "x" for i in range(33)}])
def test_public_file_validation_precedes_daemon_access(tmp_path, files):
    with pytest.raises(ValueError, match="public files"):
        SoftwareBackend(case={**CASE, "available_at_cutoff": {"files": files}},
                        options={"image": "sha256:" + "a"*64, "docker_host": "unix:///nonexistent-docker.sock"},
                        workspace=tmp_path, timeout=10)


@pytest.mark.parametrize("failure", ["different_owner", "unreachable_daemon"])
def test_cleanup_never_deletes_an_unverified_target(failure):
    backend = SoftwareBackend.__new__(SoftwareBackend)
    backend.closed, backend.container, backend.name, backend.owner = False, "a"*64, "unused", "ours"
    calls = []
    def command(arguments, **kwargs):
        calls.append(arguments)
        if failure == "different_owner":
            return subprocess.CompletedProcess([], 0, json.dumps({"Id": "a"*64, "Config": {"Labels": {"gnomon.agent-owner": "theirs"}}}).encode(), b"")
        return subprocess.CompletedProcess([], 1, b"", b"dial unix socket: no such file or directory")
    backend._command = command
    with pytest.raises(RuntimeError):
        backend.close()
    assert not backend.closed and len(calls) == 1 and "rm" not in calls[0]


@pytest.fixture
def backend(tmp_path, monkeypatch):
    image = os.environ.get("GNOMON_TEST_SOFTWARE_IMAGE")
    if not image:
        pytest.skip("explicit local GNOMON_TEST_SOFTWARE_IMAGE required; no automatic build/pull")
    monkeypatch.setenv("MODEL_SECRET_MUST_NOT_CROSS", "private-fixture-token")
    instance = SoftwareBackend(case=CASE, options={"image": image, "docker_host": "unix:///var/run/docker.sock"},
                               workspace=tmp_path, timeout=45)
    try:
        yield instance
    finally:
        instance.close()


def python(backend, code, timeout=20):
    return backend.call("python", {"code": code}, timeout=timeout).value


def test_real_software_forecasts_without_any_gnomon_adapter(backend):
    value = python(backend, '''import json,numpy as np,pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive,AutoARIMA,AutoETS
df=pd.DataFrame({'unique_id':['series']*40,'ds':pd.date_range('2025-01-01',periods=40),'y':[10.,20.,30.,40.]*10})
model=StatsForecast(models=[SeasonalNaive(season_length=4),AutoARIMA(season_length=4),AutoETS(season_length=4)],freq='D',n_jobs=1)
result=model.forecast(df=df,h=2,level=[80])
print(json.dumps({'columns':list(result.columns),'naive':result['SeasonalNaive'].tolist(),'finite':bool(np.isfinite(result.select_dtypes('number')).all().all())}))
''', timeout=30)
    assert value["returncode"] == 0, value
    result = json.loads(value["stdout"])
    assert result["naive"] == [10, 20] and result["finite"]
    assert {"AutoARIMA", "AutoETS"} <= set(result["columns"])
    assert backend.provenance["software"]["distributions"]["statsforecast"]
    assert "gnomon-forecast" not in backend.provenance["software"]["distributions"]


def test_public_case_files_persist_and_python_error_can_be_repaired(backend):
    first = python(backend, "import json; from pathlib import Path; p=json.loads(Path('/tmp/case.json').read_text()); Path('/tmp/saved.json').write_text(json.dumps(p)); print(p['available_at_cutoff']['series'])")
    assert first["stdout"].strip() == "[3, 7]"
    bad = python(backend, "raise ValueError('agent can repair this')")
    assert bad["returncode"] != 0 and "ValueError" in bad["stderr"] and not backend.closed
    repaired = python(backend, "import json; from pathlib import Path; print(json.loads(Path('/tmp/saved.json').read_text())['id'])")
    assert repaired["stdout"].strip() == "software"


def test_network_host_files_credentials_and_timer_permissions(backend):
    value = python(backend, '''import json,os,socket,signal
from pathlib import Path
denied=[]
def visible(path):
    try: return Path(path).exists()
    except OSError: return False
for name,operation in [('network',lambda:socket.create_connection(('1.1.1.1',443),timeout=0.2)),('root_write',lambda:Path('/outside').write_text('x')),('timer_signal',lambda:os.kill(1,signal.SIGSTOP))]:
    try: operation()
    except OSError: denied.append(name)
print(json.dumps({'uid':os.getuid(),'secret':os.environ.get('MODEL_SECRET_MUST_NOT_CROSS'),'checkout':visible('/root/Gnomon'),'docker_socket':visible('/var/run/docker.sock'),'denied':denied}))
''')
    assert value["returncode"] == 0, value
    assert json.loads(value["stdout"]) == {"uid": 65534, "secret": None, "checkout": False,
                                          "docker_socket": False, "denied": ["network", "root_write", "timer_signal"]}
    info = json.loads(backend._command(["inspect", backend.container, "--format", "{{json .HostConfig}}"]).stdout)
    assert info["NetworkMode"] == "none" and info["ReadonlyRootfs"] and not info["Binds"]
    assert info["Memory"] == 1_073_741_824 and info["PidsLimit"] == 128


@pytest.mark.parametrize("code", ["import time; time.sleep(30)", "import os;\nwhile True: os.write(1,b'x'*8192)"])
def test_timeout_or_output_overflow_removes_own_container(backend, code):
    with pytest.raises(ProcessLimit):
        python(backend, code, timeout=0.2)
    assert backend.closed
    result = backend._command(["inspect", backend.container], cleanup=True)
    assert result.returncode != 0 and b"no such" in result.stderr.lower()


def test_pid_one_lifetime_ends_without_host_cleanup(tmp_path):
    image = os.environ.get("GNOMON_TEST_SOFTWARE_IMAGE")
    if not image:
        pytest.skip("explicit local image required")
    instance = SoftwareBackend(case=CASE, options={"image": image, "docker_host": "unix:///var/run/docker.sock"},
                               workspace=tmp_path, timeout=3)
    try:
        deadline = time.monotonic() + 7
        while time.monotonic() < deadline:
            result = instance._command(["inspect", instance.container], cleanup=True)
            if result.returncode != 0:
                assert b"no such" in result.stderr.lower()
                break
            time.sleep(0.2)
        else:
            pytest.fail("container outlived its independent lifetime limit")
    finally:
        instance.close()


def test_shared_agent_loop_executes_real_ordinary_software_and_records_identity(tmp_path):
    from benchmarks.common.openrouter import OpenRouterClient
    from benchmarks.workflow.bounded_agent import run_agent
    image = os.environ.get("GNOMON_TEST_SOFTWARE_IMAGE")
    if not image:
        pytest.skip("explicit local image required")
    class ScriptedModel:
        def open(self, request, **kwargs):
            body = json.loads(request.data)
            if len(body["messages"]) == 2:
                name = "python"
                arguments = {"code": "import json,statistics; from pathlib import Path; print(statistics.mean(json.loads(Path('/tmp/case.json').read_text())['available_at_cutoff']['series']))"}
            else:
                tool_value = json.loads(body["messages"][-1]["content"])
                name = "submit_answer"
                arguments = {"status": "answered", "support": "supported", "numbers": {"mean": float(tool_value["stdout"])}}
            response = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01}}
            return io.BytesIO(json.dumps(response).encode())
    result = run_agent(CASE, prompt="Calculate the mean using available software.",
                       budget={"max_rounds": 3, "max_tool_calls": 2, "max_tokens": 1000, "timeout_seconds": 30},
                       client_factory=lambda: OpenRouterClient("scripted/not-real", api_key="unused", request_opener=ScriptedModel()),
                       backend_factory=lambda: SoftwareBackend(case=CASE, options={"image": image, "docker_host": "unix:///var/run/docker.sock"}, workspace=tmp_path, timeout=30))
    assert result.status == "answered" and result.numbers == {"mean": 5}
    assert result.metadata["backend_provenance"]["image"] == image
    assert result.metadata["backend_provenance"]["software"]["distributions"]["statsforecast"]
    assert result.metadata["cleanup_errors"] == []
    assert result.tool_calls == 1 and result.cumulative_tokens == 30 and result.cost_usd == 0.02
