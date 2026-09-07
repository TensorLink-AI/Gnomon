"""Local stdio MCP regression client; not a generated-code security sandbox."""

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


class StdioMcpSession:
    """Newline-delimited JSON-RPC client over a `gnomon mcp serve` child.

    The subprocess runs with the jail as its working directory, so any
    relative path that slips through argument screening still lands
    inside the jail.

    Every call carries a timeout: ``readline`` on a wedged server blocks
    forever, and the runners' own caps (rounds, calls, tokens, wall
    clock) are all checked *between* reads, so a single hung call used to
    stall an entire sequential run with no summary ever written. On
    timeout the child is killed and the call raises; the runners already
    treat transport death as a disclosed harness failure.
    """

    #: Regression tests override this ceiling with their own deadline.
    DEFAULT_CALL_TIMEOUT_SECONDS = 600.0

    def __init__(self, cwd: str | Path, command: list[str] | None = None,
                 call_timeout: float | None = None):
        child_env = dict(os.environ)
        # The child deliberately runs from a disposable jail, so its normal
        # import path would resolve the environment's last installed wheel.
        # Benchmarks must exercise the same working-tree revision as the host
        # adapter. Pin that source explicitly and retain any caller path after
        # it; the manifest/cache contract then describes one coherent build.
        repository_root = Path(__file__).resolve().parents[2]
        source_paths = [str(repository_root / "src"), str(repository_root)]
        inherited_path = child_env.get("PYTHONPATH")
        child_env["PYTHONPATH"] = os.pathsep.join(
            [*source_paths, *([inherited_path] if inherited_path else [])])
        self._proc = subprocess.Popen(
            command or [sys.executable, "-m", "gnomon", "mcp", "serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=str(cwd), text=True,
            env=child_env,
        )
        self._next_id = 0
        self.call_timeout = (self.DEFAULT_CALL_TIMEOUT_SECONDS
                             if call_timeout is None else float(call_timeout))

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        import threading

        self._next_id += 1
        request = {"jsonrpc": "2.0", "id": self._next_id,
                   "method": method, "params": params}
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        timed_out: list[bool] = []

        def _kill() -> None:
            timed_out.append(True)
            self._proc.kill()

        timer = threading.Timer(self.call_timeout, _kill)
        timer.start()
        try:
            line = self._proc.stdout.readline()
        finally:
            timer.cancel()
        if not line:
            if timed_out:
                raise RuntimeError(
                    f"gnomon mcp server did not answer {method} within "
                    f"{self.call_timeout:.0f}s and was killed"
                )
            raise RuntimeError("gnomon mcp server closed its stdout")
        message = json.loads(line)
        if "error" in message:
            raise RuntimeError(f"MCP {method} failed: {message['error']}")
        return message["result"]

    def initialize(self) -> dict[str, Any]:
        return self._rpc("initialize", {"protocolVersion": "2025-06-18"})

    def list_tools(self) -> list[dict[str, Any]]:
        return self._rpc("tools/list", {})["tools"]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> None:
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
