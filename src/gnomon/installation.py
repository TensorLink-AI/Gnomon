"""Explicit management of environments created by install.sh."""

import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import urllib.parse
import urllib.request
from uuid import uuid4

from .build_info import build_info, source_fingerprint
from .contracts import GnomonError

RECEIPT = "gnomon-install.json"


def environment_info():
    package = Path(__file__).resolve().parent
    return {"schema_version": "1", "status": "ok", "distribution": "gnomon-forecast", "import_name": "gnomon",
            "python": sys.executable, "prefix": sys.prefix, "package_path": str(package),
            "site_packages": str(package.parent), "build": build_info(),
            "managed_install": (Path(sys.prefix) / RECEIPT).is_file(),
            "python_command": "gnomon python", "install_into_your_python": "python -m pip install gnomon-forecast"}


def _context():
    prefix = Path(sys.prefix).resolve()
    path = prefix / RECEIPT
    if not path.is_file():
        raise GnomonError("NOT_MANAGED_INSTALL", "Release management requires an install.sh environment. "
                          "For a pip installation, use this Python's pip to update; run gnomon environment for its path.")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        root = Path(receipt["install_root"])
        command = Path(receipt["command"])
        if not root.is_absolute():
            raise ValueError("relative install root")
        root = root.resolve()
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise GnomonError("INVALID_INSTALL_RECEIPT", "Install receipt is unreadable or malformed; rerun install.sh.") from exc
    if receipt.get("schema_version") != 1 or prefix.parent != root / "releases":
        raise GnomonError("INVALID_INSTALL_RECEIPT", "Install receipt does not match this environment.")
    if not command.is_absolute() or command.name != "gnomon" or not command.is_symlink():
        raise GnomonError("INVALID_INSTALL_RECEIPT", "The managed gnomon command must be an absolute symlink.")
    target = command.resolve()
    active = target.parent.parent
    if target.name != "gnomon" or target.parent.name != "bin" or active.parent != root / "releases":
        raise GnomonError("INVALID_INSTALL_RECEIPT", "The command points outside the managed releases directory.")
    return prefix, root, command, active, receipt


@contextmanager
def _locked():
    import fcntl
    _, root, _, _, _ = _context()
    with (root / ".install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _release_info(path, active, running):
    receipt = {}
    if (path / RECEIPT).is_file():
        try:
            receipt = json.loads((path / RECEIPT).read_text(encoding="utf-8"))
            if not isinstance(receipt, dict):
                receipt = {}
        except (ValueError, OSError):
            pass
    packages = sorted(path.glob("lib/python*/site-packages/gnomon"))
    info = receipt.get("build")
    if info is None and packages:
        package = packages[0]
        try:
            if (package / "_build_info.json").is_file():
                info = json.loads((package / "_build_info.json").read_text(encoding="utf-8"))
            else:
                tree = ast.parse((package / "product_contract.py").read_text(encoding="utf-8"))
                version = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets))
                info = {"package_version": version, "commit": None, "source_sha256": source_fingerprint(package),
                        "provenance": "legacy_install_commit_unknown"}
        except (OSError, ValueError, StopIteration, SyntaxError):
            info = None
    installing = (path / ".installing").exists()
    usable = os.access(path / "bin" / "gnomon", os.X_OK) and bool(packages) and not installing
    return {"release_id": path.name, "path": str(path), "active": path == active, "running": path == running,
            "usable": usable, "installing": installing, "installed_at": receipt.get("installed_at"), "build": info,
            "requested_ref": receipt.get("requested_ref"), "source": receipt.get("source", "legacy_install")}


def _live_releases(root):
    """Find legacy Python/CLI processes which predate startup leases.

    Linux exposes argv, cwd and activated virtualenv paths. On platforms where
    that inspection is unavailable, legacy environments stay protected.
    """
    proc = Path("/proc")
    if not proc.is_dir():
        return set(), False
    live, complete = set(), True
    base = str(root / "releases") + os.sep
    try:
        processes = list(proc.iterdir())
    except OSError:
        return set(), False
    for process in processes:
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            values = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
            values += [item.split("=", 1)[1] for item in
                       (process / "environ").read_bytes().decode(errors="replace").split("\0")
                       if item.startswith("VIRTUAL_ENV=")]
            values.append(str((process / "cwd").readlink()))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            complete = False
            continue
        for value in values:
            if value.startswith(base):
                live.add(root / "releases" / value[len(base):].split(os.sep, 1)[0])
    return live, complete


@contextmanager
def _unused_release(root, path):
    """Hold an exclusive lease while checking or removing an environment."""
    import fcntl
    locks = root / ".release-locks"
    locks.mkdir(exist_ok=True)
    with (locks / (path.name + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            try:
                yield True
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _usage(path, root, live, complete):
    if path in live:
        return "in_use"
    with _unused_release(root, path) as unused:
        if not unused:
            return "in_use"
    # Old installations cannot reliably be inspected on every OS. Missing
    # process visibility is a reason to preserve a release, never to delete it.
    leased = any(path.glob("lib/python*/site-packages/gnomon_release_lease.pth"))
    return "idle" if complete or leased else "unknown"


def releases(*, prune=False, apply=False, keep=1):
    with _locked():
        return _releases(prune=prune, apply=apply, keep=keep)


def _releases(*, prune=False, apply=False, keep=1):
    if type(keep) is not int or keep < 0 or (apply and not prune):
        raise GnomonError("INVALID_ARGUMENTS", "Use releases --prune [--keep N] to preview, and add --apply to remove old installs.")
    running, root, command, active, _ = _context()
    paths = sorted((p for p in (root / "releases").iterdir() if p.is_dir() and not p.is_symlink()),
                   key=lambda p: p.name, reverse=True)
    entries = [_release_info(p, active, running) for p in paths]
    live, complete = _live_releases(root)
    for entry, path in zip(entries, paths):
        entry["usage"] = "in_use" if entry["running"] else _usage(path, root, live, complete)
        entry["running"] = entry["usage"] == "in_use"
    inactive = [entry for entry in entries if not entry["active"] and entry["usage"] == "idle" and not entry["installing"]]
    candidates = inactive[keep:] if prune else []
    removed, skipped = [], []
    if apply:
        for entry in candidates:
            # Recheck the active link before every deletion, including when a
            # caller rolled back while inspecting the preview.
            target = Path(entry["path"])
            if target == command.resolve().parent.parent or target == running or target.is_symlink():
                continue
            live, complete = _live_releases(root)
            if _usage(target, root, live, complete) != "idle":
                skipped.append(entry["release_id"])
                continue
            with _unused_release(root, target) as unused:
                if unused:
                    shutil.rmtree(target)
                    removed.append(entry["release_id"])
                else:
                    skipped.append(entry["release_id"])
    return {"schema_version": "1", "status": "ok", "install_root": str(root), "releases": entries,
            "prune_candidates": [entry["release_id"] for entry in candidates], "removed": removed,
            "apply": apply, "keep_inactive": keep, "skipped_in_use": skipped,
            "protected": "active_running_installing_and_unknown_usage_environments"}


def rollback(release_id):
    with _locked():
        return _rollback(release_id)


def _rollback(release_id):
    running, root, command, active, _ = _context()
    if not isinstance(release_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", release_id) or release_id in {".", ".."}:
        raise GnomonError("INVALID_ARGUMENTS", "Select an exact release_id from gnomon releases.")
    target = root / "releases" / release_id
    if target.is_symlink() or not target.is_dir() or not _release_info(target, active, running)["usable"]:
        raise GnomonError("RELEASE_NOT_FOUND", "That release is unavailable or incomplete; run gnomon releases.")
    temporary = command.parent / (".gnomon-switch-" + uuid4().hex)
    try:
        temporary.symlink_to(target / "bin" / "gnomon")
        os.replace(temporary, command)
    finally:
        temporary.unlink(missing_ok=True)
    return {"schema_version": "1", "status": "ok", "previous_release": active.name, "active_release": release_id,
            "command": str(command), "management_command": str(running / "bin" / "gnomon"),
            "next_step": "Run gnomon --version to verify the selected build. Older builds may need management_command for release management."}


def update(ref):
    prefix, root, command, active, receipt = _context()
    repository = receipt.get("repository")
    if not repository or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise GnomonError("INVALID_INSTALL_RECEIPT", "No valid update repository is recorded; rerun install.sh with --repository.")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", ref) or ".." in ref:
        raise GnomonError("INVALID_ARGUMENTS", "Version must be a Git tag, branch or commit.")
    commit = _resolve_ref(repository, ref)
    installed = _release_info(active, active, prefix)
    build = installed.get("build") or {}
    packages = list(active.glob("lib/python*/site-packages/gnomon"))
    if (build.get("commit") == commit and build.get("dirty") is not True and installed["usable"]
            and (installed["source"] == "github" or build.get("dirty") is False)
            and len(packages) == 1 and build.get("source_sha256") == source_fingerprint(packages[0])):
        return {"schema_version": "1", "status": "ok", "changed": False, "reason": "already_up_to_date",
                "active_release": active.name, "release": installed}
    installer = prefix / "install.sh"
    if not installer.is_file():
        raise GnomonError("INSTALLER_NOT_FOUND", "Rerun install.sh to upgrade this legacy installation.")
    environment = dict(os.environ)
    environment.pop("GNOMON_LOCAL", None)
    environment["GNOMON_UPDATE_REF"] = ref
    with TemporaryDirectory(prefix="gnomon-update-") as directory:
        requirements = Path(directory) / "requirements.txt"
        frozen = subprocess.run([str(active / "bin/python"), "-m", "pip", "freeze", "--all",
                                 "--exclude", "gnomon-forecast", "--exclude", "pip"],
                                capture_output=True, text=True, env=environment)
        if frozen.returncode:
            raise GnomonError("DEPENDENCY_EXPORT_FAILED", "Could not export the active environment's dependencies. "
                              "Repair its pip installation before updating; the active command is unchanged.")
        requirements.write_text(frozen.stdout, encoding="utf-8")
        requirements.chmod(0o600)
        result = subprocess.run(["bash", str(installer), "--repository", repository, "--version", commit,
                                 "--requirements", str(requirements),
                                 "--install-root", str(root), "--bin-dir", str(command.parent)],
                                stdout=sys.stderr, stderr=sys.stderr, env=environment)
    if result.returncode:
        raise GnomonError("UPDATE_FAILED", "Installation failed; the previous command remains active. See stderr for installer diagnostics.")
    active = command.resolve().parent.parent
    return {"schema_version": "1", "status": "ok", "changed": True, "active_release": active.name,
            "release": _release_info(active, active, prefix),
            "management_command": str(prefix / "bin" / "gnomon"),
            "next_step": "Run gnomon --version to verify the selected build. Older builds may need management_command for release management."}


def _resolve_ref(repository, ref):
    """Resolve once before creating an environment; the installer receives a SHA."""
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    endpoint = f"repos/{repository}/commits/{urllib.parse.quote(ref, safe='')}"
    try:
        if shutil.which("gh") and subprocess.run(["gh", "auth", "status"], capture_output=True).returncode == 0:
            result = subprocess.run(["gh", "api", endpoint, "--jq", ".sha"], check=True, capture_output=True, text=True)
            commit = result.stdout.strip()
        else:
            request = urllib.request.Request("https://api.github.com/" + endpoint, headers={"User-Agent": "gnomon-updater"})
            with urllib.request.urlopen(request, timeout=30) as response:
                commit = json.load(response)["sha"]
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("invalid commit")
        return commit
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        raise GnomonError("UPDATE_REF_NOT_FOUND", "Could not resolve the requested Git ref. Check the repository, "
                          "version and GitHub access; no installation was changed.", {"repository": repository, "ref": ref}) from exc
