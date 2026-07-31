"""Leakage lint: operators and pipeline stages must not read files directly.

All data access in the execution path goes through a Snapshot. The only
place allowed to touch raw inputs is the load stage (and the store's own
ingestion). This test makes a direct read a CI violation rather than a
code-review hope.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "aion"

# Modules whose non-load functions must be snapshot-only. Extend as the
# operator substrate grows.
GUARDED_MODULES = ["pipeline.py", "operators.py", "macros.py", "adjudication.py"]

# Functions allowed to perform raw input reads.
ALLOWED_FUNCTIONS = {"load_stage"}

FORBIDDEN_CALLS = {
    "open", "load_observations", "read_text", "read_bytes",
    "DictReader", "read_table", "read_csv",
}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def test_no_direct_reads_outside_load_stage():
    violations: list[str] = []
    for module_name in GUARDED_MODULES:
        module_path = SRC / module_name
        if not module_path.is_file():
            continue
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for function in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            if function.name in ALLOWED_FUNCTIONS:
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.Call):
                    name = _call_name(node)
                    if name in FORBIDDEN_CALLS:
                        violations.append(
                            f"{module_name}:{node.lineno} {function.name}() calls {name}()"
                        )
    assert not violations, (
        "Direct data reads inside operators/stages violate the snapshot "
        "policy:\n" + "\n".join(violations)
    )
