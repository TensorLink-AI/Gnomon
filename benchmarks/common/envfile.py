"""Minimal ``.env`` support, standard library only.

Lets ``OPENROUTER_API_KEY`` (or any variable) live in an untracked
``.env`` file instead of the shell profile. Real environment variables
always win — the file only fills in what is not already set, so
per-invocation overrides keep working. Lines are ``KEY=VALUE`` with
optional ``export`` prefix, ``#`` comments, and single or double quotes
around the value. No interpolation, no multiline values — anything
needing those belongs in a real environment manager.
"""

from __future__ import annotations

import os
from pathlib import Path


def parse_env_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load_env_file(*directories: Path) -> dict[str, str]:
    """Apply the first ``.env`` found in ``directories`` (default: the
    working directory, then the repository root). Returns the variables
    actually set; existing environment always wins."""
    if not directories:
        directories = (Path.cwd(), Path(__file__).resolve().parents[2])
    applied: dict[str, str] = {}
    for directory in directories:
        candidate = Path(directory) / ".env"
        if not candidate.is_file():
            continue
        for key, value in parse_env_text(
            candidate.read_text(encoding="utf-8")
        ).items():
            if key not in os.environ:
                os.environ[key] = value
                applied[key] = value
        break
    return applied
