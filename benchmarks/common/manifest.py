"""Provenance for one benchmark run.

A comparison is only meaningful when both arms answered the same
questions about the same target. Three times in one session the suite
produced confidently-formatted numbers for arms that had not: a control
scored on MACD values against a treatment forecasting prices, arms
reading different sample sets, and two different F1 conventions. Nothing
in the output said so — it had to be caught by reading.

So every run records what it was, and `benchmarks/report.py` refuses to
compare runs whose manifests disagree on the things that must match.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"

#: Fields that must be identical for two runs to be comparable at all.
#: `condition` deliberately is not one of them — differing conditions is
#: the entire point of a comparison.
COMPARABLE_FIELDS = ("benchmark", "target")


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


def read_manifest(run_dir: Path) -> dict[str, Any]:
    """The run's manifest, or an empty dict for runs made before manifests
    existed — absence is reported by the caller, never guessed around."""
    path = Path(run_dir) / MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def incompatibilities(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """Why these two runs cannot be compared; empty when they can."""
    problems = []
    for field in COMPARABLE_FIELDS:
        left_value, right_value = left.get(field), right.get(field)
        if left_value is None or right_value is None:
            continue  # unknown is reported separately, not treated as equal
        if left_value != right_value:
            problems.append(
                f"{field} differs: {left_value!r} vs {right_value!r}"
            )
    return problems
