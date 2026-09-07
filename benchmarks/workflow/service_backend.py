"""Current execution services alongside identical ordinary software.

The full arm enables the optional ledger and temporal tools. Each agent filesystem
is confined to explicitly owned no-network containers.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

from benchmarks.workflow.agent_metrics import _decode_record
from .bounded_agent import ToolReply
from .matched import fingerprint
from .software_backend import SoftwareBackend, REQUIRED


def source_fingerprint(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2] / "src/gnomon"
    return fingerprint({str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(root.rglob("*")) if path.is_file()
                        and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}})


def _execution_options(arm, value):
    if arm not in {"lean", "full"}:
        raise ValueError("service backend requires the lean or full feature arm")
    if value is not None and not isinstance(value, dict):
        raise ValueError("execution_options must be an object")
    value = {**({"ledger": True, "temporal": True} if arm == "full" else {}), **(value or {})}
    if set(value) - {"ledger", "temporal"} or any(type(item) is not bool for item in value.values()):
        raise ValueError("ledger/temporal startup options must be boolean")
    return value


class McpBackend(SoftwareBackend):
    def __init__(self, *, arm, case, options, workspace, timeout, execution_options=None):
        execution_options = _execution_options(arm, execution_options)
        self.process = None
        self.request_id = 0
        super().__init__(case=case, options=options, workspace=workspace, timeout=timeout)
        try:
            extra = []
            if execution_options:
                config = "schema_version=1\nallow_outcome_writes=false\n"
                if execution_options.get("ledger"):
                    config += 'ledger_path="/tmp/ledger.sqlite3"\n'
                if execution_options.get("temporal"):
                    config += 'enable_temporal=true\n'
                script = ("from pathlib import Path; import sys; p=Path('/tmp/gnomon-operator.toml'); "
                          "p.write_bytes(sys.stdin.buffer.read()); p.chmod(0o444)")
                self._command(["exec", "--user=0:0", "-i", self.container, "python", "-I", "-c", script], input=config.encode())
                extra = ["--providers-config", "/tmp/gnomon-operator.toml"]
            self.process = subprocess.Popen([
                *self.argv, "exec", "--user=65534:65534", "-i", self.container,
                "python", "-I", "-m", "gnomon", "mcp", "serve", *extra],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                start_new_session=True, env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"})
            for stream in (self.process.stdin, self.process.stdout):
                os.set_blocking(stream.fileno(), False)
            initialized = self._rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                                   "clientInfo": {"name": "gnomon-matched", "version": "1"}}, timeout=timeout)
            if initialized.get("protocolVersion") != "2025-06-18":
                raise ValueError("service protocol differs from requested contract")
            self.notify_initialized = True
            self.inventory = self._rpc("tools/list", {}, timeout=timeout)["tools"]
            if not isinstance(self.inventory, list):
                raise ValueError("invalid service tool inventory")
            self.provenance.update(feature_arm=arm, execution_options=execution_options,
                                   server_info=initialized.get("serverInfo"),
                                   tool_inventory_sha256=fingerprint(self.inventory))
        except BaseException:
            self.close()
            raise

    def _validate_inventory(self, inventory):
        if not (REQUIRED | {"gnomon-forecast"}) <= inventory["distributions"].keys():
            raise ValueError("service image lacks required installed software")
        if inventory["gnomon_source_sha256"] != source_fingerprint():
            raise ValueError("installed service package contents differ from this checkout")

    def _rpc(self, method, params, *, timeout):
        if self.closed:
            raise RuntimeError("service backend is closed")
        self.request_id += 1
        wire = json.dumps({"jsonrpc": "2.0", "id": self.request_id, "method": method,
                           "params": params}, allow_nan=False).encode() + b"\n"
        if getattr(self, "notify_initialized", False):
            wire = b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n' + wire
            self.notify_initialized = False
        if len(wire) > 1_048_576:
            raise ValueError("service request exceeds byte limit")
        deadline = min(self.deadline, time.monotonic() + timeout)
        output, sent = bytearray(), 0
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE, "write")
                selector.register(self.process.stdout, selectors.EVENT_READ, "read")
                while b"\n" not in output:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("service response exceeded deadline")
                    for key, _ in selector.select(min(remaining, 0.1)):
                        if key.data == "write":
                            sent += os.write(key.fd, wire[sent:sent + 8192])
                            if sent == len(wire):
                                selector.unregister(self.process.stdin)
                        else:
                            chunk = os.read(key.fd, min(8192, 1_048_577 - len(output)))
                            if not chunk:
                                raise RuntimeError("service closed its response stream")
                            output.extend(chunk)
                            if len(output) > 1_048_576:
                                raise ValueError("service response exceeds byte limit")
            result = _decode_record(output.decode())
            if (result.get("jsonrpc") != "2.0" or type(result.get("id")) is not int
                    or result["id"] != self.request_id or "error" in result or "result" not in result):
                raise ValueError("service response does not match the request")
            return result["result"]
        except BaseException:
            self.close()
            raise

    def tools(self):
        return self.inventory

    def call(self, name, arguments, *, timeout):
        if name not in {tool["name"] for tool in self.inventory}:
            raise ValueError("tool is not in the discovered service inventory")
        return ToolReply(self._rpc("tools/call", {"name": name, "arguments": arguments}, timeout=timeout), 0)

    def close(self):
        try:
            super().close()
        finally:
            process = self.process
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                for stream in (process.stdin, process.stdout):
                    stream.close()
                process.wait(timeout=5)
                self.process = None


class CombinedBackend:
    startup_cost_usd = 0

    def __init__(self, *, arm, case, options, workspace, timeout):
        if (not {"software_image", "service_image", "docker_host"} <= options.keys()
                or set(options) - {"software_image", "service_image", "docker_host", "execution_options"}):
            raise ValueError("combined backend requires both pinned images and local docker_host")
        execution_options = _execution_options(arm, options.get("execution_options"))
        self.software = self.service = None
        started = time.monotonic()
        try:
            for directory in ("software", "service"):
                (workspace / directory).mkdir(mode=0o700)
            self.software = SoftwareBackend(case=case, options={"image": options["software_image"], "docker_host": options["docker_host"]},
                                            workspace=workspace / "software", timeout=timeout)
            self.service = McpBackend(arm=arm, case=case, options={"image": options["service_image"], "docker_host": options["docker_host"]},
                                      workspace=workspace / "service", timeout=timeout - (time.monotonic() - started),
                                      execution_options=execution_options)
            ordinary = self.software.provenance["software"]
            service = self.service.provenance["software"]
            if (ordinary["python"] != service["python"] or ordinary["distributions"] != {
                    key: value for key, value in service["distributions"].items() if key != "gnomon-forecast"}):
                raise ValueError("service and ordinary software dependencies differ")
            self.provenance = {"ordinary": self.software.provenance, "service": self.service.provenance,
                               "resources": "two_independent_container_budgets_not_equal_total_compute_to_ordinary"}
        except BaseException:
            self.close()
            raise

    def tools(self):
        return [*self.software.tools(), *self.service.tools()]

    def call(self, name, arguments, *, timeout):
        return (self.software if name == "python" else self.service).call(name, arguments, timeout=timeout)

    def reveal(self, revealed, *, timeout):
        """Deliver common operator data after commitment; never calculate answers."""
        started = time.monotonic()
        first = self.software.reveal(revealed, timeout=timeout)
        second = self.service.reveal(revealed, timeout=timeout - (time.monotonic() - started))
        actuals = revealed.get("actuals", [])
        if not isinstance(actuals, list) or len(actuals) > 1000:
            raise ValueError("revealed actuals must be a bounded row list")
        ingested = None
        if actuals and self.service.provenance["execution_options"].get("ledger"):
            script = ("import json,sys; from gnomon import TemporalLedger\n"
                      "ledger=TemporalLedger('/tmp/ledger.sqlite3')\n"
                      "for row in json.load(sys.stdin): ledger.append_actual(**row)\n")
            self.service._command(["exec", "--user=65534:65534", "-i", self.service.container,
                                   "python", "-I", "-c", script], input=json.dumps(actuals, allow_nan=False).encode(),
                                  timeout=timeout - (time.monotonic() - started))
            ingested = len(actuals)
        return ToolReply({"software": first.value, "service": second.value, "ledger_actuals_ingested": ingested}, 0)

    def close(self):
        failed = False
        for backend in (self.service, self.software):
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    failed = True
        if failed:
            raise RuntimeError("one or more combined backends could not confirm cleanup")


def lean(**kwargs):
    return CombinedBackend(arm="lean", **kwargs)


def full(**kwargs):
    return CombinedBackend(arm="full", **kwargs)
