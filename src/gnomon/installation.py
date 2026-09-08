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
    inactive = [entry for entry in entries if not entry["active"] and not entry["running"] and not entry["installing"]]
    candidates = inactive[keep:] if prune else []
    removed = []
    if apply:
        for entry in candidates:
            # Recheck the active link before every deletion, including when a
            # caller rolled back while inspecting the preview.
            target = Path(entry["path"])
            if target == command.resolve().parent.parent or target == running or target.is_symlink():
                continue
            shutil.rmtree(target)
            removed.append(entry["release_id"])
    return {"schema_version": "1", "status": "ok", "install_root": str(root), "releases": entries,
            "prune_candidates": [entry["release_id"] for entry in candidates], "removed": removed,
            "apply": apply, "keep_inactive": keep, "protected": "active_running_and_installing_environments"}


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
    prefix, root, command, _, receipt = _context()
    repository = receipt.get("repository")
    if not repository or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise GnomonError("INVALID_INSTALL_RECEIPT", "No valid update repository is recorded; rerun install.sh with --repository.")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", ref) or ".." in ref:
        raise GnomonError("INVALID_ARGUMENTS", "Version must be a Git tag, branch or commit.")
    installer = prefix / "install.sh"
    if not installer.is_file():
        raise GnomonError("INSTALLER_NOT_FOUND", "Rerun install.sh to upgrade this legacy installation.")
    environment = dict(os.environ)
    environment.pop("GNOMON_LOCAL", None)
    result = subprocess.run(["bash", str(installer), "--repository", repository, "--version", ref,
                             "--install-root", str(root), "--bin-dir", str(command.parent)],
                            stdout=sys.stderr, stderr=sys.stderr, env=environment)
    if result.returncode:
        raise GnomonError("UPDATE_FAILED", "Installation failed; the previous command remains active. See stderr for installer diagnostics.")
    active = command.resolve().parent.parent
    return {"schema_version": "1", "status": "ok", "active_release": active.name,
            "release": _release_info(active, active, prefix),
            "management_command": str(prefix / "bin" / "gnomon"),
            "next_step": "Run gnomon --version to verify the selected build. Older builds may need management_command for release management."}
