"""Subprocess execution of model-written Python.

TSAIA's protocol is that the model answers by writing code against
pre-loaded variables. Running it is therefore part of the benchmark, not an
optional extra — but it is still arbitrary code from an untrusted generator,
so:

- execution is **off by default** and requires an explicit opt-in;
- it happens in a fresh subprocess in a throwaway directory, never in the
  harness process;
- wall-clock, address-space, and output size are capped;
- the network proxy variables are stripped from the child environment.

This is containment against a careless model, not against a hostile one. For
adversarial inputs, run the whole harness inside a container — the README
says so, and so does the CLI when you enable execution.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MEMORY_MB = 2048
MAX_OUTPUT_CHARS = 20000

RESULT_VARIABLE = "predictions"


@dataclass
class ExecutionResult:
    ok: bool
    value: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    timed_out: bool = False
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stdout": self.stdout[:2000],
            "stderr": self.stderr[:2000],
            "error": self.error,
            "timed_out": self.timed_out,
            "duration_seconds": round(self.duration_seconds, 4),
        }


@dataclass
class SandboxConfig:
    enabled: bool = False
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    memory_mb: int = DEFAULT_MEMORY_MB
    python: str = field(default_factory=lambda: sys.executable)
    extra_syspath: tuple[str, ...] = ()
    result_variable: str = RESULT_VARIABLE


_PREAMBLE = textwrap.dedent(
    '''
    import json, os, pickle, sys, traceback

    _bundle = json.loads(os.environ["AIONBENCH_SANDBOX"])
    for _entry in _bundle["syspath"]:
        if _entry not in sys.path:
            sys.path.insert(0, _entry)

    if _bundle["memory_mb"]:
        try:
            import resource
            _limit = _bundle["memory_mb"] * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (_limit, _limit))
        except Exception:
            pass

    _globals = {"__name__": "__aionbench_answer__"}
    if _bundle["variables_path"]:
        with open(_bundle["variables_path"], "rb") as _handle:
            _globals.update(pickle.load(_handle))

    with open(_bundle["code_path"], "r", encoding="utf-8") as _handle:
        _code = _handle.read()

    _outcome = {"ok": False, "error": None}
    try:
        exec(compile(_code, "<model_answer>", "exec"), _globals)
        _outcome["ok"] = True
    except BaseException:
        _outcome["error"] = traceback.format_exc(limit=12)

    _name = _bundle["result_variable"]
    if _outcome["ok"] and _name not in _globals:
        _outcome["ok"] = False
        _outcome["error"] = (
            "The code ran but never assigned `%s`." % _name
        )

    if _outcome["ok"]:
        try:
            with open(_bundle["result_path"], "wb") as _handle:
                pickle.dump(_globals[_name], _handle)
        except BaseException:
            _outcome["ok"] = False
            _outcome["error"] = "`%s` could not be serialised: %s" % (
                _name, traceback.format_exc(limit=4),
            )

    sys.stderr.write("\\n__AIONBENCH_OUTCOME__" + json.dumps(_outcome))
    '''
).strip()


class SandboxDisabled(RuntimeError):
    """Raised when a code condition runs without execution being enabled."""


def run_code(
    code: str,
    *,
    variables: Mapping[str, Any] | None = None,
    variables_path: str | os.PathLike[str] | None = None,
    config: SandboxConfig | None = None,
) -> ExecutionResult:
    """Execute ``code`` and return whatever it assigned to the result variable.

    ``variables`` are injected into the code's globals. ``variables_path``
    points at an already-pickled mapping (how TSAIA ships its executor
    variables) and is used verbatim to avoid a needless re-pickle.
    """
    config = config or SandboxConfig()
    if not config.enabled:
        raise SandboxDisabled(
            "Code execution is disabled. Pass --allow-code-execution to run "
            "model-written Python, ideally inside a container."
        )
    if not code or not code.strip():
        return ExecutionResult(ok=False, error="No code to execute.")

    with tempfile.TemporaryDirectory(prefix="aionbench-exec-") as raw_dir:
        directory = Path(raw_dir)
        code_path = directory / "answer.py"
        code_path.write_text(code, encoding="utf-8")

        vars_path: str | None = None
        if variables_path is not None:
            vars_path = str(Path(variables_path))
        elif variables:
            vars_path = str(directory / "variables.pkl")
            with open(vars_path, "wb") as handle:
                pickle.dump(dict(variables), handle)

        result_path = directory / "result.pkl"
        bundle = {
            "code_path": str(code_path),
            "variables_path": vars_path,
            "result_path": str(result_path),
            "result_variable": config.result_variable,
            "memory_mb": config.memory_mb,
            "syspath": list(config.extra_syspath),
        }

        runner_path = directory / "_runner.py"
        runner_path.write_text(_PREAMBLE, encoding="utf-8")

        completed, timed_out, duration = _spawn(config, runner_path, bundle, directory)

        stdout = (completed.stdout if completed else "")[:MAX_OUTPUT_CHARS]
        stderr_raw = (completed.stderr if completed else "")
        stderr, outcome = _split_outcome(stderr_raw)

        if timed_out:
            return ExecutionResult(
                ok=False, stdout=stdout, stderr=stderr[:MAX_OUTPUT_CHARS],
                error=f"Execution exceeded {config.timeout_seconds}s.",
                timed_out=True, duration_seconds=duration,
            )
        if outcome is None:
            return ExecutionResult(
                ok=False, stdout=stdout, stderr=stderr[:MAX_OUTPUT_CHARS],
                error="The sandbox process died before reporting an outcome "
                      "(most often the memory cap or a hard crash).",
                duration_seconds=duration,
            )
        if not outcome.get("ok"):
            return ExecutionResult(
                ok=False, stdout=stdout, stderr=stderr[:MAX_OUTPUT_CHARS],
                error=str(outcome.get("error") or "Unknown execution failure."),
                duration_seconds=duration,
            )

        try:
            with open(result_path, "rb") as handle:
                value = pickle.load(handle)
        except (OSError, pickle.UnpicklingError) as exc:
            return ExecutionResult(
                ok=False, stdout=stdout, stderr=stderr[:MAX_OUTPUT_CHARS],
                error=f"Result could not be read back: {exc}",
                duration_seconds=duration,
            )

        return ExecutionResult(
            ok=True, value=value, stdout=stdout,
            stderr=stderr[:MAX_OUTPUT_CHARS], duration_seconds=duration,
        )


def _spawn(
    config: SandboxConfig,
    runner_path: Path,
    bundle: Mapping[str, Any],
    cwd: Path,
) -> tuple[subprocess.CompletedProcess[str] | None, bool, float]:
    import time

    environment = {
        key: value for key, value in os.environ.items()
        # Keep the child off the network by default and away from anything
        # that would let generated code phone home with the benchmark data.
        if key.upper() not in {
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
            "OPENROUTER_API_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN",
        }
    }
    environment["AIONBENCH_SANDBOX"] = json.dumps(dict(bundle))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    started = time.monotonic()
    try:
        completed = subprocess.run(
            [config.python, str(runner_path)],
            capture_output=True, text=True, cwd=str(cwd),
            env=environment, timeout=config.timeout_seconds, check=False,
        )
        return completed, False, time.monotonic() - started
    except subprocess.TimeoutExpired:
        return None, True, time.monotonic() - started


def _split_outcome(stderr: str) -> tuple[str, dict[str, Any] | None]:
    marker = "\n__AIONBENCH_OUTCOME__"
    index = stderr.rfind(marker)
    if index < 0:
        return stderr, None
    try:
        return stderr[:index], json.loads(stderr[index + len(marker):])
    except json.JSONDecodeError:
        return stderr, None
