"""Ordinary Python/software tools in one ephemeral, resource-limited container.

No Gnomon/model adapters, generated host execution, host mounts, environment
credentials or network. Docker and the pinned image are trusted operator inputs;
containers share the host kernel and are not an adversarial security certification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import time
import uuid

from gnomon.agent_eval import _decode_record
from .bounded_agent import ToolReply
from .process import run_process

REQUIRED = {"numpy", "pandas", "scipy", "statsforecast"}


def public_files(value):
    if (not isinstance(value, dict) or len(value) > 32 or any(
            not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)
            or not isinstance(text, str) for name, text in value.items())):
        raise ValueError("public files require at most32 simple filenames and UTF-8 text contents")
    return value


def _absent(result, target):
    message = result.stderr.lower()
    return result.returncode != 0 and any(
        (prefix + target).encode() in message for prefix in ("no such container: ", "no such object: "))


BOOTSTRAP = """import json,sys,importlib.metadata,importlib.util,platform,hashlib
from pathlib import Path
data=sys.stdin.buffer.read()
Path('/tmp/case.json').write_bytes(data)
files=json.loads(data).get('available_at_cutoff',{}).get('files',{})
Path('/tmp/data').mkdir()
for name,text in files.items():
    Path('/tmp/data',name).write_text(text,encoding='utf-8')
spec=importlib.util.find_spec('gnomon')
source=None
if spec is not None:
    root=Path(spec.origin).parent
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.pyo'}}
    source=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
print(json.dumps({'python':platform.python_version(), 'distributions':dict(sorted(
    (d.metadata['Name'].lower(),d.version) for d in importlib.metadata.distributions())), 'gnomon_source_sha256':source}))
"""


class SoftwareBackend:
    startup_cost_usd = 0  # Local infrastructure is excluded, not described as free.

    def __init__(self, *, case, options, workspace, timeout):
        if not isinstance(options, dict) or set(options) != {"image", "docker_host"}:
            raise ValueError("software backend requires pinned image and local docker_host")
        image, host = options["image"], options["docker_host"]
        if not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
            raise ValueError("software image must be an already installed immutable image ID")
        if not isinstance(host, str) or not host.startswith("unix:///") or any(c in host for c in "\r\n\x00"):
            raise ValueError("software backend requires an explicit local Unix Docker socket")
        if type(timeout) not in (int, float) or not 0 < timeout <= 3600:
            raise ValueError("invalid software backend lifetime")
        if "oracle" in case or "experiment" in case or "stages" in case:
            raise ValueError("software backend accepts only a public single-stage case")
        public_files(case.get("available_at_cutoff", {}).get("files", {}))
        public = json.dumps(case, allow_nan=False, sort_keys=True).encode()
        if len(public) > 1_048_576:
            raise ValueError("public case exceeds byte limit")
        socket = Path(host[7:])
        if not socket.is_absolute() or not stat.S_ISSOCK(socket.stat().st_mode):
            raise ValueError("docker_host must identify a local socket")
        docker = shutil.which("docker")
        if docker is None:
            raise ValueError("Docker is required for the ordinary software backend")
        config = Path(workspace) / "docker-client"
        config.mkdir(mode=0o700)  # Never consume a user's client auth/configuration.
        self.argv = [docker, "--config", str(config), "--host", host]
        self.name = "gnomon-agent-" + uuid.uuid4().hex
        self.owner = uuid.uuid4().hex
        self.container = None
        self.deadline = time.monotonic() + timeout
        self.closed = False
        self.provenance = {}
        try:
            info = _decode_record(self._command(["image", "inspect", image, "--format", "{{json .}}"]).stdout.decode())
            if info["Id"] != image or info["Config"].get("Volumes"):
                raise ValueError("software image identity changed or declares automatic volumes")
            # PID1 has a finite lifetime independent of the driver. Killing the
            # host process therefore does not leave ordinary work running forever.
            lifetime = max(1, int(self.deadline - time.monotonic()) + 1)
            command = ["create", "--pull=never", "--rm", "--name", self.name,
                       "--label", "gnomon.agent-owner=" + self.owner,
                       "--network=none", "--read-only", "--cap-drop=ALL",
                       "--security-opt=no-new-privileges", "--pids-limit=128",
                       "--memory=1g", "--memory-swap=1g", "--cpus=2", "--no-healthcheck",
                       "--user=0:0", "--workdir=/tmp",
                       "--tmpfs=/tmp:rw,nosuid,nodev,size=128m,mode=1777",
                       "--env=OPENBLAS_NUM_THREADS=1", "--env=OMP_NUM_THREADS=1",
                       "--env=NUMBA_NUM_THREADS=1", "--env=NUMBA_CACHE_DIR=/tmp/numba",
                       "--env=PYTHONDONTWRITEBYTECODE=1", "--entrypoint=python", image,
                       "-I", "-S", "-c", f"import time; time.sleep({lifetime})"]
            identifier = self._command(command).stdout.decode().strip()
            if not re.fullmatch(r"[0-9a-f]{64}", identifier):
                raise ValueError("invalid created container identity")
            self.container = identifier
            self._command(["start", identifier])
            bootstrap = self._command(["exec", "--user=65534:65534", "-i", identifier, "python", "-I", "-c", BOOTSTRAP], input=public)
            inventory = _decode_record(bootstrap.stdout.decode())
            self._validate_inventory(inventory)
            self.provenance = {"image": image, "software": inventory,
                               "case_sha256": hashlib.sha256(public).hexdigest(),
                               "isolation": "local_docker_no_network_no_host_mounts_readonly_root_nonroot_agent_root_timer",
                               "limits": {"memory_bytes": 1_073_741_824, "tmp_bytes": 134_217_728,
                                          "pids": 128, "cpus": 2, "lifetime_seconds": lifetime},
                               "state": "files_in_tmp_persist_between_calls_python_globals_do_not"}
        except BaseException:
            self.close()
            raise

    def _validate_inventory(self, inventory):
        if (not REQUIRED <= inventory["distributions"].keys() or inventory["gnomon_source_sha256"] is not None
                or any(key.startswith("gnomon") for key in inventory["distributions"])):
            raise ValueError("ordinary image requires the declared software and no installed Gnomon")

    def _command(self, arguments, *, input=b"", timeout=None, cleanup=False, check=True):
        remaining = 5.0 if cleanup else self.deadline - time.monotonic()
        if timeout is not None:
            remaining = min(remaining, timeout)
        if remaining <= 0:
            raise TimeoutError("software backend lifetime exhausted")
        result = run_process([*self.argv, *arguments], input=input, timeout=remaining,
                             stdout_limit=524_288, stderr_limit=65_536,
                             env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"})
        if result.returncode and check and not cleanup:
            raise RuntimeError("software container command failed")
        return result

    def tools(self):
        libraries = ", ".join(f"{key} {self.provenance['software']['distributions'][key]}" for key in sorted(REQUIRED))
        return [{"name": "python", "description":
                 f"Execute ordinary Python with {libraries} and the standard library. "
                 "Public task data is /tmp/case.json. Print results to stdout. Files under /tmp persist "
                 "and supplied public files are under /tmp/data. "
                 "between calls; each call has fresh Python globals. Network and host files are unavailable. "
                 "Use installed library APIs directly; no Gnomon package is installed.",
                 "inputSchema": {"type": "object", "additionalProperties": False,
                                 "required": ["code"], "properties": {"code": {"type": "string", "maxLength": 65536}}}}]

    def reveal(self, revealed, *, timeout):
        """Operator event: add newly revealed files, never overwrite agent state."""
        files = public_files(revealed.get("files", {}))
        if files:
            script = ("import json,sys; from pathlib import Path\n"
                      "for name,text in json.load(sys.stdin).items():\n"
                      "    with Path('/tmp/data',name).open('x',encoding='utf-8') as stream: stream.write(text)\n")
            self._command(["exec", "--user=65534:65534", "-i", self.container, "python", "-I", "-c", script],
                          input=json.dumps(files, allow_nan=False).encode(), timeout=timeout)
        return ToolReply({"created_files": sorted(files)}, 0)

    def call(self, name, arguments, *, timeout):
        if self.closed:
            raise RuntimeError("software backend is closed")
        if (name != "python" or not isinstance(arguments, dict) or set(arguments) != {"code"}
                or not isinstance(arguments["code"], str) or len(arguments["code"].encode()) > 65_536):
            raise ValueError("python requires bounded code text")
        try:
            result = self._command(["exec", "--user=65534:65534", "-i", self.container, "python", "-I", "-"],
                                   input=arguments["code"].encode(), timeout=timeout, check=False)
        except BaseException:
            # Killing docker exec's host CLI alone would leave the code running.
            self.close()
            raise
        return ToolReply({"stdout": result.stdout.decode("utf-8", errors="replace"),
                          "stderr": result.stderr.decode("utf-8", errors="replace"),
                          "returncode": result.returncode}, 0)

    def close(self):
        if self.closed:
            return
        # Resolve the exact created ID/name and verify our unguessable ownership
        # label. Never prune, enumerate, or remove another container by pattern.
        target = self.container or self.name
        result = self._command(["container", "inspect", target,
                                "--format", "{{json .}}"], cleanup=True)
        if result.returncode == 0:
            info = _decode_record(result.stdout.decode())
            if info.get("Config", {}).get("Labels", {}).get("gnomon.agent-owner") != self.owner:
                raise RuntimeError("container ownership differs; refusing cleanup")
            identifier = info["Id"]
            if not re.fullmatch(r"[0-9a-f]{64}", identifier):
                raise RuntimeError("invalid container cleanup identity")
            removed = self._command(["rm", "--force", identifier], cleanup=True)
            if removed.returncode and not _absent(removed, identifier):
                raise RuntimeError("software container cleanup failed")
        elif not _absent(result, target):
            raise RuntimeError("software container cleanup could not verify state")
        self.closed = True


make = SoftwareBackend
