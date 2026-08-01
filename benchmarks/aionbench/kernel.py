"""A persistent Python kernel, matching TSAIA's execution semantics.

TSAIA's CodeAct agent runs against a live Jupyter notebook: variables are
injected once into ``globals()``, the model gets up to five execution turns,
and **state persists between them** — a model can define a helper in turn one
and call it in turn four. The answer is never parsed from prose; it is read
out of the kernel afterwards as ``globals()["predictions"]``.

A one-shot ``exec`` cannot reproduce that, so this is a real long-lived
subprocess speaking a length-prefixed JSON protocol over its stdio. It is not
Jupyter — no ZMQ, no kernel spec, no dependency — but it matches the three
things that affect a score: persistence across turns, injection into globals,
and pickled retrieval of a named variable.

Reference: ``tools/codeact_agent.py`` in USC-Melady/TSAIA
(``MyNotebookExecutor.inject_available_variables`` / ``access_variable``).

Containment is the same as the one-shot sandbox: wall-clock and memory caps,
provider and proxy credentials stripped, a throwaway working directory. That
is protection against a careless model, not a hostile one.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .contracts import BenchError

# Mirrors TSAIA's own cap; the agent loop enforces the turn count, this is a
# backstop so a wedged kernel cannot outlive the cell that owns it.
DEFAULT_TURN_TIMEOUT = 120
DEFAULT_MEMORY_MB = 4096
STARTUP_TIMEOUT = 60

_STRIPPED_ENV_PREFIXES = ("OPENROUTER", "OPENAI", "ANTHROPIC", "HF_", "HUGGINGFACE", "AWS_", "GOOGLE")
_STRIPPED_ENV_EXACT = ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "GEMINI_API_KEY")


class KernelDisabled(BenchError):
    """Raised when code execution was not explicitly enabled for the run."""


@dataclass
class KernelConfig:
    enabled: bool = False
    turn_timeout_seconds: int = DEFAULT_TURN_TIMEOUT
    memory_mb: int = DEFAULT_MEMORY_MB
    extra_syspath: tuple[str, ...] = ()


@dataclass
class ExecResult:
    """One execution turn's outcome, shaped like TSAIA's ServiceResponse."""

    ok: bool
    output: str = ""
    error: str = ""
    timed_out: bool = False

    @property
    def content(self) -> str:
        """What TSAIA feeds back to the model as the execution output."""
        return self.output if self.ok else (self.output + self.error).strip()


# The child program. Reads {"code": ...} or {"fetch": name} frames, replies
# with one frame each. Kept as a string so the kernel has no import-time
# dependency on anything in this package.
_CHILD = r'''
import base64, io, json, os, pickle, resource, struct, sys, traceback

def _read():
    header = sys.__stdin__.buffer.read(4)
    if len(header) < 4:
        return None
    (size,) = struct.unpack("!I", header)
    return json.loads(sys.__stdin__.buffer.read(size).decode("utf-8"))

def _write(payload):
    blob = json.dumps(payload).encode("utf-8")
    sys.__stdout__.buffer.write(struct.pack("!I", len(blob)) + blob)
    sys.__stdout__.buffer.flush()

limit_mb = int(os.environ.get("AIONBENCH_KERNEL_MEM_MB", "0"))
if limit_mb:
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_mb * 1024 * 1024,) * 2)
    except (ValueError, OSError):
        pass

NAMESPACE = {"__name__": "__main__", "__builtins__": __builtins__}
_write({"ready": True})

while True:
    message = _read()
    if message is None or message.get("shutdown"):
        break

    if "fetch" in message:
        # Mirrors MyNotebookExecutor.access_variable: pickle the name out of
        # globals, or report that it was never bound.
        name = message["fetch"]
        if name not in NAMESPACE:
            _write({"found": False})
            continue
        try:
            _write({"found": True,
                    "value": base64.b64encode(pickle.dumps(NAMESPACE[name])).decode("ascii")})
        except Exception as exc:
            _write({"found": True, "error": "%s: %s" % (type(exc).__name__, exc)})
        continue

    if "inject" in message:
        try:
            NAMESPACE.update(pickle.loads(base64.b64decode(message["inject"])))
            _write({"ok": True})
        except Exception as exc:
            _write({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
        continue

    stdout, stderr = io.StringIO(), io.StringIO()
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout, stderr
    try:
        exec(compile(message["code"], "<execute>", "exec"), NAMESPACE, NAMESPACE)
        ok, error = True, ""
    except BaseException:
        ok = False
        error = traceback.format_exc()
    finally:
        sys.stdout, sys.stderr = real_out, real_err
    _write({"ok": ok, "output": stdout.getvalue() + stderr.getvalue(), "error": error})
'''


class PersistentKernel:
    """A subprocess whose globals survive between ``run`` calls."""

    def __init__(self, config: KernelConfig | None = None, workdir: str | Path | None = None):
        self.config = config or KernelConfig()
        if not self.config.enabled:
            raise KernelDisabled(
                "CODE_EXECUTION_DISABLED",
                "This suite runs model-written Python. Pass --allow-code-execution "
                "to enable it, ideally inside a container.",
            )
        self._workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="aionbench-kernel-"))
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._process = self._spawn()
        self.turns = 0

    # -- lifecycle ---------------------------------------------------------

    def _spawn(self) -> subprocess.Popen:
        environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith(_STRIPPED_ENV_PREFIXES) and key not in _STRIPPED_ENV_EXACT
        }
        environment["AIONBENCH_KERNEL_MEM_MB"] = str(self.config.memory_mb)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if self.config.extra_syspath:
            existing = environment.get("PYTHONPATH", "")
            joined = os.pathsep.join([*self.config.extra_syspath, existing]) if existing \
                else os.pathsep.join(self.config.extra_syspath)
            environment["PYTHONPATH"] = joined

        process = subprocess.Popen(
            [sys.executable, "-u", "-c", _CHILD],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=str(self._workdir), env=environment,
        )
        handshake = self._receive(process, STARTUP_TIMEOUT)
        if not handshake or not handshake.get("ready"):
            process.kill()
            raise BenchError("KERNEL_START_FAILED", "The execution kernel did not start.")
        return process

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self._send({"shutdown": True})
                self._process.wait(timeout=5)
            except Exception:  # noqa: BLE001 - shutdown is best-effort
                self._process.kill()
                self._process.wait(timeout=5)

    def __enter__(self) -> PersistentKernel:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def alive(self) -> bool:
        return self._process.poll() is None

    # -- protocol ----------------------------------------------------------

    def _send(self, payload: Mapping[str, Any]) -> None:
        blob = json.dumps(payload).encode("utf-8")
        assert self._process.stdin is not None
        self._process.stdin.write(struct.pack("!I", len(blob)) + blob)
        self._process.stdin.flush()

    @staticmethod
    def _receive(process: subprocess.Popen, timeout: int) -> dict[str, Any] | None:
        """Read one length-prefixed frame, killing the child if it stalls."""
        import threading

        result: list[dict[str, Any] | None] = [None]

        def reader() -> None:
            stream = process.stdout
            if stream is None:
                return
            header = stream.read(4)
            if len(header) < 4:
                return
            (size,) = struct.unpack("!I", header)
            body = stream.read(size)
            if body is not None and len(body) == size:
                result[0] = json.loads(body.decode("utf-8"))

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            process.kill()
            # Reap it, or `alive` keeps reporting True and the caller loops
            # feeding turns to a corpse.
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - kill is reliable
                pass
            return None
        return result[0]

    # -- operations --------------------------------------------------------

    def inject(self, variables: Mapping[str, Any]) -> None:
        """Put ``variables`` into the kernel's globals.

        TSAIA does this by base64-pickling the dict and calling
        ``globals().update(...)`` inside the notebook. Same wire format here,
        so a variable that survives their injection survives ours.
        """
        try:
            blob = base64.b64encode(pickle.dumps(variables)).decode("ascii")
        except Exception as exc:  # noqa: BLE001 - dataset assets vary
            raise BenchError(
                "EXECUTOR_VARIABLES_UNPICKLABLE",
                f"Could not serialise executor variables: {type(exc).__name__}: {exc}",
            ) from exc

        self._send({"inject": blob})
        reply = self._receive(self._process, self.config.turn_timeout_seconds)
        if not reply or not reply.get("ok"):
            detail = (reply or {}).get("error", "kernel died during injection")
            raise BenchError("EXECUTOR_VARIABLES_INJECTION_FAILED", str(detail))

    def run(self, code: str) -> ExecResult:
        """Execute one turn. Failures are returned, never raised."""
        if not code.strip():
            return ExecResult(ok=False, error="Empty code block.")
        if not self.alive:
            return ExecResult(ok=False, error="The execution kernel is no longer running.")

        self.turns += 1
        try:
            self._send({"code": code})
        except (BrokenPipeError, OSError) as exc:
            return ExecResult(ok=False, error=f"Kernel unreachable: {exc}")

        reply = self._receive(self._process, self.config.turn_timeout_seconds)
        if reply is None:
            # Killed on timeout. The kernel is gone and its state with it —
            # the same thing that happens to a hung Jupyter kernel.
            return ExecResult(
                ok=False, timed_out=True,
                error=f"Execution exceeded {self.config.turn_timeout_seconds}s and was terminated.",
            )
        return ExecResult(
            ok=bool(reply.get("ok")),
            output=str(reply.get("output") or ""),
            error=str(reply.get("error") or ""),
        )

    def fetch(self, name: str = "predictions") -> tuple[bool, Any]:
        """Read a variable out of the kernel. Returns ``(found, value)``.

        ``found`` is False when the name was never bound — which is how a
        model that talked about its answer without assigning it gets scored,
        rather than being credited for prose.
        """
        if not self.alive:
            return False, None
        try:
            self._send({"fetch": name})
        except (BrokenPipeError, OSError):
            return False, None

        reply = self._receive(self._process, self.config.turn_timeout_seconds)
        if not reply or not reply.get("found"):
            return False, None
        if "error" in reply:
            return True, None       # bound, but not transportable
        try:
            return True, pickle.loads(base64.b64decode(reply["value"]))
        except Exception:  # noqa: BLE001 - unpickling a model-built object
            return True, None
