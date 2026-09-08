"""Build identity that survives wheels, source archives, and isolated installs."""

import ast
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def source_fingerprint(package):
    digest = hashlib.sha256()
    for path in sorted(Path(package).rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git(root, *arguments):
    try:
        return subprocess.run(["git", "-C", str(root), *arguments], check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=3).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_build_info(package):
    package = Path(package).resolve()
    tree = ast.parse((package / "product_contract.py").read_text(encoding="utf-8"))
    version = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__version__"
                                                          for target in node.targets))
    fingerprint = source_fingerprint(package)
    saved = package / "_build_info.json"
    previous = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else {}
    root = package.parent.parent
    checkout = (root / ".git").exists() and (root / "pyproject.toml").is_file()
    if not checkout and previous.get("source_sha256") == fingerprint and previous.get("package_version") == version:
        return previous
    commit, dirty = None, None
    if checkout:
        commit = _git(root, "rev-parse", "HEAD")
        status = _git(root, "status", "--porcelain", "--untracked-files=normal", "--",
                      "src/gnomon", "pyproject.toml", "build_hook.py", "install.sh")
        dirty = bool(status) if status is not None else None
    else:
        commit = os.environ.get("GNOMON_BUILD_COMMIT") or previous.get("commit")
        dirty = True if previous else None
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        commit = None
    local = (["g" + commit[:12]] if commit else []) + ["s" + fingerprint[:12]]
    if dirty:
        local.append("dirty")
    return {"package_version": version, "build_id": version + "+" + ".".join(local),
            "commit": commit, "dirty": dirty, "source_sha256": fingerprint,
            "fingerprint_scope": "gnomon_python_sources", "provenance": "git" if checkout else "archive" if commit else "source_only"}


@lru_cache(maxsize=1)
def _build_info():
    package = Path(__file__).resolve().parent
    return source_build_info(package)


def build_info():
    return dict(_build_info())
