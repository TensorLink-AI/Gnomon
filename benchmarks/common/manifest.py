"""Git and run provenance for the current matched evaluation harness."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"

def code_revision() -> str | None:
    """The commit and tracked-tree state that produced a benchmark run."""
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(root),
        )
        tracked = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=5, cwd=str(root),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    if not revision or result.returncode != 0 or tracked.returncode != 0:
        return None
    return revision + ("+dirty" if tracked.stdout.strip() else "")


def write_manifest(output_dir: Path, **fields: Any) -> Path:
    """Record one run's provenance beside its results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in fields.items() if value is not None}
    payload.setdefault("code_revision", code_revision())
    path = output_dir / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path
