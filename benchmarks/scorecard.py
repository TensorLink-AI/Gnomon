"""Validate the three-layer product scorecard used by the v0.6 loop.

The scorecard keeps forecast/output quality, reasoning preservation, and
topology/operational behavior separate.  It is deliberately a validator, not
an aggregator: an experiment must preregister the meaning and gate of every
metric rather than letting this module choose a favorable statistic after the
run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.1"
LAYER_NAMES = ("output", "reasoning", "topology")
DECISIONS = {"continue", "promote", "revise", "reject"}
OPERATORS = {"lte", "gte", "eq", "between"}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value == "unknown":
        raise ValueError(f"{field} must be known non-empty text")
    return value


def _number(value: Any, *, field: str) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _count(value: Any, *, field: str, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be {qualifier}")
    return value


def _gate_result(metric: dict[str, Any], *, field: str) -> bool:
    value = _number(metric.get("value"), field=f"{field}.value")
    gate = _mapping(metric.get("gate"), field=f"{field}.gate")
    operator = gate.get("operator")
    if operator not in OPERATORS:
        raise ValueError(f"{field}.gate.operator must be one of {sorted(OPERATORS)}")
    if operator == "between":
        bounds = gate.get("threshold")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"{field}.gate.threshold must be [lower, upper]")
        lower = _number(bounds[0], field=f"{field}.gate.threshold[0]")
        upper = _number(bounds[1], field=f"{field}.gate.threshold[1]")
        if lower > upper:
            raise ValueError(f"{field}.gate.threshold bounds are reversed")
        actual = lower <= value <= upper
    else:
        threshold = _number(
            gate.get("threshold"), field=f"{field}.gate.threshold")
        if operator == "lte":
            actual = value <= threshold
        elif operator == "gte":
            actual = value >= threshold
        else:
            actual = math.isclose(value, threshold, rel_tol=1e-12, abs_tol=1e-12)
    if not isinstance(gate.get("passed"), bool):
        raise ValueError(f"{field}.gate.passed must be boolean")
    if gate["passed"] is not actual:
        raise ValueError(f"{field}.gate.passed contradicts value and threshold")
    return actual


def _validate_accounting(value: Any, *, field: str) -> tuple[bool, int]:
    accounting = _mapping(value, field=field)
    expected = _count(accounting.get("expected"), field=f"{field}.expected",
                      positive=True)
    completed = _count(accounting.get("completed"), field=f"{field}.completed")
    answered = _count(accounting.get("answered"), field=f"{field}.answered")
    abstained = _count(accounting.get("abstained"), field=f"{field}.abstained")
    failed = _count(accounting.get("failed"), field=f"{field}.failed")
    if completed != answered + abstained + failed:
        raise ValueError(
            f"{field}.completed must equal answered + abstained + failed")
    if completed > expected:
        raise ValueError(f"{field}.completed cannot exceed expected")
    return completed == expected, expected


def _validate_evidence(value: Any, *, field: str, root: Path) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must contain retained evidence")
    for index, item in enumerate(value):
        entry = _mapping(item, field=f"{field}[{index}]")
        relative = _text(entry.get("path"), field=f"{field}[{index}].path")
        expected_digest = _text(
            entry.get("sha256"), field=f"{field}[{index}].sha256")
        if len(expected_digest) != 64 or any(
                character not in "0123456789abcdef" for character in expected_digest):
            raise ValueError(f"{field}[{index}].sha256 must be lowercase SHA-256")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(
                f"{field}[{index}].path must stay within the evidence root")
        path = (root / candidate).resolve()
        if not path.is_relative_to(root):
            raise ValueError(
                f"{field}[{index}].path must stay within the evidence root")
        if not path.is_file():
            raise ValueError(f"missing scorecard evidence: {relative}")
        if _digest(path) != expected_digest:
            raise ValueError(f"scorecard evidence digest mismatch: {relative}")


def _validate_layer(name: str, value: Any, *, root: Path) -> tuple[bool, bool]:
    layer = _mapping(value, field=f"layers.{name}")
    _text(layer.get("evaluated_commit"),
          field=f"layers.{name}.evaluated_commit")
    _text(layer.get("dataset_identity"),
          field=f"layers.{name}.dataset_identity")
    _text(layer.get("configuration_identity"),
          field=f"layers.{name}.configuration_identity")
    complete, expected = _validate_accounting(
        layer.get("accounting"), field=f"layers.{name}.accounting")
    metrics = layer.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"layers.{name}.metrics must not be empty")
    names: set[str] = set()
    gate_results: list[bool] = []
    for index, item in enumerate(metrics):
        metric = _mapping(item, field=f"layers.{name}.metrics[{index}]")
        metric_name = _text(
            metric.get("name"), field=f"layers.{name}.metrics[{index}].name")
        if metric_name in names:
            raise ValueError(f"duplicate metric in {name}: {metric_name}")
        names.add(metric_name)
        _text(metric.get("unit"), field=f"layers.{name}.metrics[{index}].unit")
        denominator = _count(
            metric.get("denominator"),
            field=f"layers.{name}.metrics[{index}].denominator",
            positive=True)
        if denominator > expected:
            raise ValueError(
                f"layers.{name}.metrics[{index}].denominator cannot exceed "
                "the layer's expected population")
        gate_results.append(_gate_result(
            metric, field=f"layers.{name}.metrics[{index}]"))
    status = layer.get("status")
    if status not in {"pass", "fail"}:
        raise ValueError(f"layers.{name}.status must be pass or fail")
    passed = complete and all(gate_results)
    if (status == "pass") is not passed:
        raise ValueError(
            f"layers.{name}.status contradicts completeness or metric gates")
    _validate_evidence(
        layer.get("evidence"), field=f"layers.{name}.evidence", root=root)
    return complete, passed


def validate_payload(payload: Any, *, root: Path) -> dict[str, Any]:
    """Validate and return a compact decision summary.

    ``root`` is the repository or artifact root against which evidence paths
    and digests are resolved.
    """
    scorecard = _mapping(payload, field="scorecard")
    if scorecard.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported product scorecard schema")
    _text(scorecard.get("scorecard_id"), field="scorecard_id")
    _text(scorecard.get("scorecard_commit"), field="scorecard_commit")
    if not isinstance(scorecard.get("dirty_tree"), bool):
        raise ValueError("dirty_tree must be boolean")
    scope = scorecard.get("scope")
    if scope not in {"smoke", "subset", "full"}:
        raise ValueError("scope must be smoke, subset, or full")

    layers = _mapping(scorecard.get("layers"), field="layers")
    if set(layers) != set(LAYER_NAMES):
        raise ValueError(
            f"layers must contain exactly {', '.join(LAYER_NAMES)}")
    complete: dict[str, bool] = {}
    passed: dict[str, bool] = {}
    for name in LAYER_NAMES:
        complete[name], passed[name] = _validate_layer(
            name, layers[name], root=root)

    invariants = _mapping(scorecard.get("invariants"), field="invariants")
    if not invariants:
        raise ValueError("invariants must not be empty")
    invariant_passed: dict[str, bool] = {}
    for name, raw in invariants.items():
        invariant = _mapping(raw, field=f"invariants.{name}")
        if invariant.get("status") not in {"pass", "fail"}:
            raise ValueError(f"invariants.{name}.status must be pass or fail")
        invariant_passed[name] = invariant["status"] == "pass"
        _validate_evidence(
            invariant.get("evidence"), field=f"invariants.{name}.evidence",
            root=root)

    decision = _mapping(scorecard.get("decision"), field="decision")
    decision_status = decision.get("status")
    if decision_status not in DECISIONS:
        raise ValueError(f"decision.status must be one of {sorted(DECISIONS)}")
    _text(decision.get("reason"), field="decision.reason")
    if decision_status == "promote":
        if scope != "full":
            raise ValueError("only a full scorecard may promote a change")
        if scorecard["dirty_tree"]:
            raise ValueError("a dirty-tree scorecard cannot promote a change")
        if not all(complete.values()):
            raise ValueError("a scorecard with incomplete rows cannot promote")
        if not all(passed.values()):
            raise ValueError("a scorecard with a failing layer cannot promote")
        if not all(invariant_passed.values()):
            raise ValueError("a scorecard with a failing invariant cannot promote")

    return {
        "scorecard_id": scorecard["scorecard_id"],
        "scope": scope,
        "layers": {name: {
            "complete": complete[name], "passed": passed[name]}
            for name in LAYER_NAMES},
        "invariants_passed": all(invariant_passed.values()),
        "decision": decision_status,
    }


def validate(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_payload(payload, root=(root or Path.cwd()).resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a three-layer Gnomon product scorecard.")
    parser.add_argument("scorecard", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(validate(args.scorecard, root=args.root),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
