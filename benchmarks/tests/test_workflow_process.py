"""Real process boundaries, not mocked timeout return values."""

import os
from pathlib import Path
import sys
import time

import pytest

from benchmarks.workflow.process import ProcessLimit, run_process


def invoke(code, **kwargs):
    return run_process([sys.executable, "-c", code], input=b"", timeout=2, **kwargs)


def test_reads_both_pipes_and_delivers_input():
    result = run_process([sys.executable, "-c", "import sys; print(sys.stdin.read()); print('err',file=sys.stderr)"],
                         input=b"payload", timeout=2)
    assert result.returncode == 0 and result.stdout == b"payload\n" and result.stderr == b"err\n"


@pytest.mark.parametrize("fd", [1, 2])
def test_output_is_bounded_while_reading(fd):
    with pytest.raises(ProcessLimit, match="output_limit"):
        invoke(f"import os;\nwhile True: os.write({fd},b'x'*8192)", stdout_limit=1024, stderr_limit=1024)


def test_input_limit_precedes_process_creation():
    with pytest.raises(ProcessLimit, match="input_limit"):
        run_process(["does-not-exist"], input=b"x"*1_048_577, timeout=1)


def _running(pid):
    # Linux may retain a killed, unreaped orphan zombie; it cannot do more work.
    status = Path(f"/proc/{pid}/stat")
    try:
        return status.read_text().split(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        # The process can be reaped while /proc is being read. Check its PID
        # directly too, preserving the fallback on platforms without /proc.
        pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.parametrize("alive", [False, True])
@pytest.mark.parametrize("error", [FileNotFoundError, ProcessLookupError])
def test_process_probe_handles_stat_disappearing(monkeypatch, alive, error):
    def missing_stat(_path):
        raise error

    def probe(pid, signal):
        assert (pid, signal) == (12345, 0)
        if not alive:
            raise ProcessLookupError

    monkeypatch.setattr(Path, "read_text", missing_stat)
    monkeypatch.setattr(os, "kill", probe)
    assert _running(12345) is alive


@pytest.mark.parametrize("parent_exit", [False, True])
def test_owned_descendants_are_killed_even_when_parent_exits(tmp_path, parent_exit):
    path = tmp_path / "pid"
    code = ("import subprocess,sys,time; from pathlib import Path; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],"
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,stdin=subprocess.DEVNULL); "
            f"Path({str(path)!r}).write_text(str(p.pid)); " + ("pass" if parent_exit else "time.sleep(30)"))
    if parent_exit:
        assert invoke(code).returncode == 0
    else:
        with pytest.raises(ProcessLimit, match="timeout"):
            run_process([sys.executable, "-c", code], input=b"", timeout=0.5)
    pid = int(path.read_text())
    deadline = time.monotonic() + 2
    while _running(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _running(pid)


def test_closed_pipes_do_not_evade_timeout():
    with pytest.raises(ProcessLimit, match="timeout"):
        run_process([sys.executable, "-c", "import os,time; os.close(1); os.close(2); time.sleep(30)"],
                    input=b"", timeout=0.1)
