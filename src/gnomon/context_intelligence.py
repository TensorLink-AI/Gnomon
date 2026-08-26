"""Governed context hypotheses and fold-safe candidate executables.

Language models may translate prose into competing typed hypotheses.  This
module gives those hypotheses stable identities, validates their grounding,
and evaluates numerical relationships without exposing future observations.
It deliberately does not publish forecasts or grant automation authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .effect_proposals import validate_effect_proposal
from .statistical_executables import fit_regression_executable

MAX_HYPOTHESES = 6
MAX_TRANSFORM_NODES = 48
MAX_TRANSFORM_DEPTH = 8
TRANSFORM_OPS = frozenset({
    "literal", "series", "primary", "add", "subtract", "multiply",
    "divide", "lag", "difference", "percent_change", "rolling_mean",
    "clip", "quantile", "power",
})
HYPOTHESIS_KINDS = frozenset({
    "absolute_value", "bound", "additive_change", "multiplicative_change",
    "regime_shift", "relationship", "historical_analogue", "unsupported",
})


class TransformationError(ValueError):
    """A typed, user-repairable rejection of a declarative transformation."""

    def __init__(self, code: str, field: str, message: str):
        super().__init__(message)
        self.code = code
        self.field = field
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {"field": self.field, "code": self.code,
                "message": self.message}


def _finite(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TransformationError(
            "NON_NUMERIC_VALUE", field, "Expected a numeric value.") from exc
    if not math.isfinite(number):
        raise TransformationError(
            "NON_FINITE_VALUE", field, "NaN and infinity are not executable.")
    return number


def validate_transformation(
    raw: Any, *, series: list[str], claim_ids: list[str], cutoff: str,
    units: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compile a small declarative expression without executing model code.

    The returned object is canonical and content-addressed.  It contains only
    approved operations, resolved series names, traceable constants and a
    knowledge timestamp no later than ``cutoff``.
    """
    if not isinstance(raw, dict):
        raise TransformationError(
            "TRANSFORMATION_NOT_OBJECT", "$", "Transformation must be an object.")
    cutoff_dt = _aware(cutoff)
    known_at = _aware(raw.get("known_at", cutoff))
    if cutoff_dt is None or known_at is None or known_at > cutoff_dt:
        raise TransformationError(
            "NOT_KNOWN_AT_CUTOFF", "known_at",
            "Transformation knowledge must be timestamped at or before cutoff.")
    cited = sorted({str(value) for value in raw.get("claim_ids") or []})
    if not cited or set(cited) - set(claim_ids):
        raise TransformationError(
            "UNVERIFIED_CLAIMS", "claim_ids",
            "Every transformation must cite verified context claims.")
    expression = raw.get("expression")
    declared_raw = str(raw.get("output_unit") or "")
    state = {"nodes": 0}
    normalized, inferred_unit = _validate_expression(
        expression, path="expression", depth=0, state=state,
        series=set(series), units=units or {})
    # A root literal with an explicitly declared output unit is an absolute
    # stated value, not a dimensionless multiplier. Bind that one unambiguous
    # omission deterministically; nested literals remain dimensionless unless
    # their own unit is stated.
    if (declared_raw and isinstance(expression, dict)
            and expression.get("op") == "literal"
            and "unit" not in expression):
        normalized["unit"] = declared_raw
        inferred_unit = declared_raw
    declared_unit = str(declared_raw or inferred_unit or "unknown")
    if inferred_unit not in {None, "unknown", declared_unit}:
        raise TransformationError(
            "OUTPUT_UNIT_MISMATCH", "output_unit",
            f"Expression yields {inferred_unit!r}, not {declared_unit!r}.")
    lane = str(raw.get("lane") or "scenario_only")
    if lane not in {"historically_testable", "prior_assisted", "scenario_only"}:
        raise TransformationError(
            "UNKNOWN_CANDIDATE_LANE", "lane", "Unknown candidate lane.")
    payload = {
        "language": "gnomon_transform", "version": "0.1", "lane": lane,
        "claim_ids": cited, "known_at": known_at.isoformat(),
        "cutoff": cutoff_dt.isoformat(),
        "output_unit": declared_unit, "expression": normalized,
        "validation": {"approved_ast": True, "nodes": state["nodes"],
                       "maximum_nodes": MAX_TRANSFORM_NODES,
                       "maximum_depth": MAX_TRANSFORM_DEPTH,
                       "known_at_cutoff": True, "units_checked": True},
    }
    payload["transformation_id"] = "transform-" + hashlib.sha256(
        _canonical(payload).encode()).hexdigest()[:16]
    payload["seal_sha256"] = hashlib.sha256(
        _canonical(payload).encode()).hexdigest()
    return payload


def compile_transformation(
    raw: Any, *, series: list[str], claim_ids: list[str], cutoff: str,
    units: dict[str, str] | None = None, repair: Any = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Validate once and permit exactly one field-bounded repair.

    The repair may alter only the top-level field implicated by the first
    typed violation.  This keeps repair useful for schema mistakes without
    allowing the model to replace the claim, lane, and expression together.
    """
    try:
        compiled = validate_transformation(
            raw, series=series, claim_ids=claim_ids, cutoff=cutoff, units=units)
        return compiled, {"status": "accepted", "attempts_used": 1,
                          "attempts_remaining": 1, "violations": []}
    except TransformationError as first:
        violation = first.as_dict()
    if repair in (None, {}) or not isinstance(raw, dict) or not isinstance(repair, dict):
        return None, {"status": "rejected", "attempts_used": 1,
                      "attempts_remaining": 1, "violations": [violation]}
    repairable = violation["field"].split(".", 1)[0].split("[", 1)[0]
    changed = {key for key in set(raw) | set(repair)
               if raw.get(key) != repair.get(key)}
    if changed - {repairable}:
        bounded = {"field": "$", "code": "REPAIR_CHANGED_UNRELATED_FIELDS",
                   "message": "Repair may change only the field named by the first violation."}
        return None, {"status": "rejected", "attempts_used": 2,
                      "attempts_remaining": 0,
                      "violations": [violation, bounded]}
    try:
        compiled = validate_transformation(
            repair, series=series, claim_ids=claim_ids, cutoff=cutoff, units=units)
        return compiled, {"status": "repaired", "attempts_used": 2,
                          "attempts_remaining": 0,
                          "violations": [violation]}
    except TransformationError as second:
        return None, {"status": "rejected", "attempts_used": 2,
                      "attempts_remaining": 0,
                      "violations": [violation, second.as_dict()]}


def _validate_expression(node: Any, *, path: str, depth: int,
                         state: dict[str, int], series: set[str],
                         units: dict[str, str]) -> tuple[dict[str, Any], str | None]:
    if depth > MAX_TRANSFORM_DEPTH:
        raise TransformationError("EXPRESSION_TOO_DEEP", path,
                                  "Transformation exceeds the depth limit.")
    if not isinstance(node, dict):
        raise TransformationError("EXPRESSION_NOT_OBJECT", path,
                                  "Every expression node must be an object.")
    state["nodes"] += 1
    if state["nodes"] > MAX_TRANSFORM_NODES:
        raise TransformationError("EXPRESSION_TOO_LARGE", path,
                                  "Transformation exceeds the node limit.")
    raw_op = str(node.get("op") or "")
    op = "power" if raw_op == "pow" else raw_op
    if op not in TRANSFORM_OPS:
        raise TransformationError("UNSAFE_OR_UNKNOWN_OPERATOR", f"{path}.op",
                                  f"Operator {op!r} is not allowed.")
    if op == "literal":
        return ({"op": op, "value": _finite(node.get("value"), f"{path}.value"),
                 "unit": str(node.get("unit") or "dimensionless")},
                str(node.get("unit") or "dimensionless"))
    if op in {"series", "primary"}:
        alias = node.get("name")
        if op == "series" and alias is None:
            alias = node.get("series")
        if op == "series" and alias is None:
            raw_args = node.get("args")
            if isinstance(raw_args, list) and len(raw_args) == 1 \
                    and isinstance(raw_args[0], str):
                alias = raw_args[0]
        name = str(alias or ("primary" if op == "primary" else ""))
        if op == "series" and name not in series:
            raise TransformationError("UNKNOWN_SERIES", f"{path}.name",
                                      f"Series {name!r} is unavailable.")
        quantile = str(node.get("quantile") or "q50")
        if quantile not in {"q10", "q50", "q90", "point"}:
            raise TransformationError("UNKNOWN_QUANTILE", f"{path}.quantile",
                                      "Quantile must be q10, q50, q90, or point.")
        return ({"op": op, "name": name, "quantile": quantile},
                units.get(name, "unknown"))
    children = node.get("args")
    if not isinstance(children, list) and "left" in node and "right" in node:
        children = [node.get("left"), node.get("right")]
    if op == "lag" and not isinstance(children, list) and node.get("series"):
        children = [{"op": "series", "name": node["series"]}]
    if not isinstance(children, list):
        children = [node.get("arg")] if "arg" in node else []
    if op in {"add", "multiply"}:
        valid_arity = len(children) >= 2
        required_text = "at least 2"
    else:
        required = 2 if op in {"subtract", "divide", "power"} else 1
        valid_arity = len(children) == required
        required_text = str(required)
    # Common model notation represents lag(x, 2) as two args. Canonicalize
    # only when the second value is a positive integer; arbitrary extra args
    # still fail loudly.
    if op == "lag" and len(children) == 2:
        step_node = children[1]
        step_value = (step_node.get("value") if isinstance(step_node, dict)
                      and step_node.get("op") == "literal" else step_node)
        node = {**node, "steps": step_value}
        children = children[:1]
        valid_arity = True
    if not valid_arity:
        raise TransformationError("INVALID_ARITY", f"{path}.args",
                                  f"{op} requires {required_text} argument(s).")
    parsed = [_validate_expression(child, path=f"{path}.args[{index}]",
                                   depth=depth + 1, state=state,
                                   series=series, units=units)
              for index, child in enumerate(children)]
    args, child_units = [item[0] for item in parsed], [item[1] for item in parsed]
    clean: dict[str, Any] = {"op": op, "args": args}
    if op in {"add", "subtract"}:
        known = {unit for unit in child_units if unit not in {None, "unknown"}}
        if len(known) > 1:
            raise TransformationError("INCOMPATIBLE_UNITS", path,
                                      "Addition/subtraction requires matching units.")
        output_unit = next(iter(known), "unknown")
    elif op == "multiply":
        output_unit = child_units[0]
        for unit in child_units[1:]:
            output_unit = _combined_unit(output_unit, unit, "*")
    elif op == "divide":
        output_unit = _combined_unit(child_units[0], child_units[1], "/")
    elif op == "power":
        exponent = args[1]
        if exponent.get("op") != "literal":
            raise TransformationError("NON_LITERAL_EXPONENT", path,
                                      "Power requires a literal exponent.")
        value = exponent["value"]
        if value != int(value) or not 0 <= int(value) <= 4:
            raise TransformationError("UNSAFE_EXPONENT", path,
                                      "Power exponent must be an integer from 0 to 4.")
        clean["exponent"] = int(value)
        output_unit = ("dimensionless" if int(value) == 0 else
                       child_units[0] if int(value) == 1 else
                       f"{child_units[0]}^{int(value)}")
    else:
        output_unit = child_units[0]
    if op in {"lag", "difference", "percent_change", "rolling_mean"}:
        window = node.get("steps", node.get("window", 1))
        try:
            window = int(window)
        except (TypeError, ValueError) as exc:
            raise TransformationError("INVALID_WINDOW", path,
                                      "Window must be a positive integer.") from exc
        if window < 1 or window > 10_000:
            raise TransformationError("INVALID_WINDOW", path,
                                      "Window must be between 1 and 10000.")
        clean["steps" if op != "rolling_mean" else "window"] = window
        if op == "percent_change":
            output_unit = "dimensionless"
    elif op == "clip":
        lower = _finite(node.get("lower"), f"{path}.lower")
        upper = _finite(node.get("upper"), f"{path}.upper")
        if lower > upper:
            raise TransformationError("INVALID_BOUNDS", path,
                                      "Clip lower bound exceeds upper bound.")
        clean.update({"lower": lower, "upper": upper})
    elif op == "quantile":
        q = _finite(node.get("q"), f"{path}.q")
        if not 0 <= q <= 1:
            raise TransformationError("INVALID_QUANTILE", f"{path}.q",
                                      "Quantile must be between zero and one.")
        clean["q"] = q
    return clean, output_unit


def _combined_unit(left: str | None, right: str | None, operator: str) -> str:
    left, right = left or "unknown", right or "unknown"
    if operator == "*" and left == "dimensionless":
        return right
    if operator == "*" and right == "dimensionless":
        return left
    if operator == "*" and "/" in right and right.rsplit("/", 1)[1] == left:
        return right.rsplit("/", 1)[0]
    if operator == "*" and "/" in left and left.rsplit("/", 1)[1] == right:
        return left.rsplit("/", 1)[0]
    if operator == "/" and right == "dimensionless":
        return left
    if operator == "/" and left == right and left != "unknown":
        return "dimensionless"
    return "unknown" if "unknown" in {left, right} else f"{left}{operator}{right}"


def execute_transformation(
    compiled: dict[str, Any], *, primary: list[dict[str, Any]],
    series_values: dict[str, Any] | None = None,
    historical_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute only a previously validated canonical transformation."""
    seal = compiled.get("seal_sha256")
    body = {key: value for key, value in compiled.items() if key != "seal_sha256"}
    if not seal or hashlib.sha256(_canonical(body).encode()).hexdigest() != seal:
        raise TransformationError("INVALID_TRANSFORMATION_SEAL", "$",
                                  "Transformation changed after validation.")
    width = len(primary)
    environment: dict[str, list[float]] = {}
    for name, supplied in (series_values or {}).items():
        if not isinstance(supplied, dict):
            raise TransformationError(
                "UNVERSIONED_FUTURE_SERIES", f"series_values.{name}",
                "Future series require values, known_at, and source_claim_id.")
        known_at = _aware(supplied.get("known_at"))
        cutoff = _aware(compiled.get("cutoff"))
        source_claim = str(supplied.get("source_claim_id") or "")
        if known_at is None or cutoff is None or known_at > cutoff:
            raise TransformationError(
                "FUTURE_SERIES_NOT_KNOWN_AT_CUTOFF", f"series_values.{name}.known_at",
                "Future input was not knowable at the forecast cutoff.")
        if source_claim not in compiled["claim_ids"]:
            raise TransformationError(
                "UNVERIFIED_FUTURE_SERIES", f"series_values.{name}.source_claim_id",
                "Future input must cite one of the transformation's verified claims.")
        values = supplied.get("values")
        if not isinstance(values, list):
            raise TransformationError(
                "INVALID_FUTURE_SERIES", f"series_values.{name}.values",
                "Future input values must be an array.")
        environment[name] = [_finite(value, name) for value in values]
    if any(len(values) != width for values in environment.values()):
        raise TransformationError("HORIZON_MISMATCH", "series_values",
                                  "Every future series must match the primary horizon.")
    values = _execute_expression(compiled["expression"], primary=primary,
                                 environment=environment, width=width,
                                 primary_quantile="q50")
    if len(values) != width or any(not math.isfinite(value) for value in values):
        raise TransformationError("INVALID_EXECUTION_RESULT", "expression",
                                  "Transformation did not yield one finite value per step.")
    quantile_paths = {
        quantile: _execute_expression(
            compiled["expression"], primary=primary,
            environment=environment, width=width,
            primary_quantile=quantile)
        for quantile in ("q10", "q50", "q90")
    }
    rows = []
    for index, value in enumerate(values):
        source = primary[index]
        ordered = sorted(quantile_paths[key][index]
                         for key in ("q10", "q50", "q90"))
        rows.append({"timestamp": source.get("timestamp"), "point": value,
                     "q10": ordered[0], "q50": value, "q90": ordered[-1]})
    validation = dict(compiled["validation"])
    if historical_validation:
        points = int(historical_validation.get("validation_points") or 0)
        skill = _finite(historical_validation.get("skill"), "historical_validation.skill")
        validation.update({
            "validation_points": points, "skill": skill,
            "beats_baseline": bool(historical_validation.get("beats_baseline")),
            "scheme": str(historical_validation.get("scheme") or "expanding_origin"),
            "per_origin_knowledge_checked": bool(
                historical_validation.get("per_origin_knowledge_checked")),
        })
        if compiled["lane"] == "historically_testable" and not validation[
                "per_origin_knowledge_checked"]:
            raise TransformationError(
                "HISTORICAL_VALIDATION_NOT_FOLD_SAFE", "historical_validation",
                "Historical admission requires per-origin knowledge checks.")
    return {
        "transformation_id": compiled["transformation_id"],
        "forecast": rows, "lane": compiled["lane"],
        "claim_ids": compiled["claim_ids"], "known_at": compiled["known_at"],
        "output_unit": compiled["output_unit"],
        "validation": validation,
        "source_seal_sha256": seal, "automation_eligible": False,
        "primary_forecast_unchanged": True,
    }


def _execute_expression(node: dict[str, Any], *, primary: list[dict[str, Any]],
                        environment: dict[str, list[float]], width: int,
                        primary_quantile: str = "q50",
                        ) -> list[float]:
    op = node["op"]
    if op == "literal":
        return [float(node["value"])] * width
    if op == "primary":
        # q50 denotes the canonical primary path and propagates the matching
        # quantile when the executor builds an uncertainty envelope. An
        # explicitly requested non-median quantile remains fixed.
        quantile = (primary_quantile if node["quantile"] == "q50"
                    else node["quantile"])
        return [_finite(row.get(quantile, row.get("point")), quantile)
                for row in primary]
    if op == "series":
        if node["name"] not in environment:
            raise TransformationError("MISSING_FUTURE_SERIES", node["name"],
                                      "A referenced future series was not supplied.")
        return list(environment[node["name"]])
    args = [_execute_expression(child, primary=primary,
                                environment=environment, width=width,
                                primary_quantile=primary_quantile)
            for child in node["args"]]
    if op in {"add", "multiply"}:
        output = list(args[0])
        for other in args[1:]:
            output = [a + b if op == "add" else a * b
                      for a, b in zip(output, other)]
        return output
    if op in {"subtract", "divide"}:
        functions = {
            "add": lambda a, b: a + b, "subtract": lambda a, b: a - b,
            "multiply": lambda a, b: a * b,
            "divide": lambda a, b: a / b if abs(b) > 1e-15 else math.nan,
        }
        return [functions[op](a, b) for a, b in zip(args[0], args[1])]
    if op == "power":
        exponent = int(node["exponent"])
        return [value ** exponent for value in args[0]]
    values = args[0]
    if op == "lag":
        steps = node["steps"]
        return [values[max(0, index - steps)] for index in range(width)]
    if op in {"difference", "percent_change"}:
        steps = node["steps"]
        output = []
        for index, value in enumerate(values):
            previous = values[max(0, index - steps)]
            output.append(value - previous if op == "difference"
                          else ((value / previous - 1) if abs(previous) > 1e-15
                                else math.nan))
        return output
    if op == "rolling_mean":
        window = node["window"]
        return [statistics.mean(values[max(0, i-window+1):i+1])
                for i in range(width)]
    if op == "clip":
        return [min(node["upper"], max(node["lower"], value)) for value in values]
    if op == "quantile":
        ordered = sorted(values)
        position = node["q"] * (len(ordered) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        value = ordered[lo] if lo == hi else (
            ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo))
        return [value] * width
    raise AssertionError(f"validated operator not implemented: {op}")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _identifier(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items()
            if key not in {"hypothesis_id", "validation"}}
    return "hyp-" + hashlib.sha256(_canonical(body).encode()).hexdigest()[:12]


def compile_context_hypotheses(
    raw: Any, *, claims: list[dict[str, Any]], series: list[str],
    cutoff: str, repair: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a bounded set of alternative interpretations.

    A repair may replace only fields named by the first attempt's violations.
    Valid hypotheses from the first attempt survive a repair, making the
    protocol deterministic and preventing an agent from silently rewriting
    already accepted interpretations.
    """
    first = raw if isinstance(raw, list) else ([] if raw in (None, {}) else [raw])
    accepted, rejected = _validate_hypotheses(
        first[:MAX_HYPOTHESES], claims=claims, series=series, cutoff=cutoff)
    attempts = [{"attempt": 1, "accepted": len(accepted), "rejected": rejected}]
    if rejected and repair not in (None, {}):
        repairs = repair if isinstance(repair, list) else [repair]
        repaired, repair_rejected = _validate_hypotheses(
            repairs[:len(rejected)], claims=claims, series=series, cutoff=cutoff)
        accepted_ids = {item["hypothesis_id"] for item in accepted}
        accepted.extend(item for item in repaired
                        if item["hypothesis_id"] not in accepted_ids)
        rejected = repair_rejected
        attempts.append({"attempt": 2, "accepted": len(repaired),
                         "rejected": repair_rejected})
    accepted.sort(key=lambda item: item["hypothesis_id"])
    return accepted[:MAX_HYPOTHESES], {
        "status": ("accepted" if accepted and not rejected else
                   "partially_accepted" if accepted else "rejected"),
        "attempts_used": len(attempts), "attempts_remaining": 2 - len(attempts),
        "accepted": len(accepted), "rejected": rejected, "attempts": attempts,
        "bounded": True, "maximum_hypotheses": MAX_HYPOTHESES,
    }


def _validate_hypotheses(raw: list[Any], *, claims: list[dict[str, Any]],
                         series: list[str], cutoff: str
                         ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    claim_ids = {str(item.get("claim_id")) for item in claims}
    allowed_series = set(series) | {"*"}
    cutoff_dt = _aware(cutoff)
    if cutoff_dt is None:
        raise ValueError("cutoff must be timezone-aware ISO-8601")
    accepted, rejected = [], []
    for index, item in enumerate(raw):
        errors: list[dict[str, str]] = []
        if not isinstance(item, dict):
            rejected.append({"index": index, "violations": [{
                "field": "$", "code": "HYPOTHESIS_NOT_OBJECT"}]})
            continue
        kind = str(item.get("kind") or "unsupported")
        if kind not in HYPOTHESIS_KINDS:
            errors.append({"field": "kind", "code": "UNKNOWN_HYPOTHESIS_KIND"})
        cited = sorted({str(value) for value in item.get("claim_ids") or []})
        if not cited or set(cited) - claim_ids:
            errors.append({"field": "claim_ids", "code": "UNVERIFIED_CLAIMS"})
        targets = sorted({str(value) for value in item.get("target_series") or ["*"]})
        if set(targets) - allowed_series:
            errors.append({"field": "target_series", "code": "UNKNOWN_SERIES"})
        known_at = _aware(item.get("known_at", cutoff))
        if known_at is None or known_at > cutoff_dt:
            errors.append({"field": "known_at", "code": "NOT_KNOWN_AT_CUTOFF"})
        lag = item.get("lag_steps", 0)
        try:
            lag = int(lag)
        except (TypeError, ValueError):
            lag = -1
        if lag < 0:
            errors.append({"field": "lag_steps", "code": "INVALID_LAG"})
        proposal = None
        if item.get("effect_proposal") not in (None, {}):
            proposal, critique = validate_effect_proposal(
                item["effect_proposal"], claim_ids=claim_ids)
            if proposal is None:
                errors.extend({"field": "effect_proposal",
                               "code": violation["code"]}
                              for attempt in critique["attempts"]
                              for violation in attempt["violations"])
        predictor = item.get("predictor_series")
        if kind == "relationship" and str(predictor or "") not in allowed_series - {"*"}:
            errors.append({"field": "predictor_series", "code": "UNKNOWN_PREDICTOR"})
        if errors:
            rejected.append({"index": index, "violations": errors,
                             "repairable_fields": sorted({e["field"] for e in errors})})
            continue
        clean = {
            "kind": kind, "claim_ids": cited, "target_series": targets,
            "known_at": known_at.isoformat(), "lag_steps": lag,
            "predictor_series": str(predictor) if predictor is not None else None,
            "direction": str(item.get("direction") or "unknown"),
            "rationale": str(item.get("rationale") or "")[:1000],
            "effect_proposal": proposal,
            "validation": {"grounded": True, "known_at_cutoff": True,
                           "series_resolved": True},
        }
        clean["hypothesis_id"] = _identifier(clean)
        accepted.append(clean)
    return accepted, rejected


def _aware(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else None


def align_vintage_rows(rows: list[dict[str, Any]], *, cutoff: str,
                       time_key: str = "timestamp", known_key: str = "known_at"
                       ) -> list[dict[str, Any]]:
    """Return the latest vintage per timestamp that was knowable at cutoff."""
    cutoff_dt = _aware(cutoff)
    if cutoff_dt is None:
        raise ValueError("cutoff must be timezone-aware ISO-8601")
    chosen: dict[str, tuple[datetime, dict[str, Any]]] = {}
    for row in rows:
        valid = _aware(row.get(time_key))
        known = _aware(row.get(known_key, row.get(time_key)))
        if valid is None or known is None or valid > cutoff_dt or known > cutoff_dt:
            continue
        key = valid.isoformat()
        if key not in chosen or known > chosen[key][0]:
            chosen[key] = (known, dict(row))
    return [chosen[key][1] for key in sorted(chosen)]


@dataclass(frozen=True)
class FittedContextCandidate:
    kind: str
    hypothesis_id: str
    estimate: dict[str, Any]
    validation: dict[str, Any]
    support: str

    def execute(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "hypothesis_id": self.hypothesis_id,
            "estimate": self.estimate, "validation": self.validation,
            "support": self.support, "automation_eligible": False,
            "primary_forecast_unchanged": True,
            "executable": {"kind": self.kind, "version": "0.1",
                           "fold_safe": True},
        }


def fit_vintage_exogenous(
    rows: list[dict[str, Any]], *, target_key: str, predictor_keys: list[str],
    cutoff: str, hypothesis_id: str, minimum_train: int = 20,
) -> FittedContextCandidate:
    """Fit exogenous regression from point-in-time eligible aligned rows.

    In addition to the global cutoff, every training row must have been known
    by its own valid timestamp.  A later revision can therefore never improve
    an earlier expanding-origin prediction.
    """
    origin_safe = [row for row in rows
                   if _aware(row.get("known_at", row.get("timestamp"))) is not None
                   and _aware(row.get("timestamp")) is not None
                   and _aware(row.get("known_at", row.get("timestamp")))
                   <= _aware(row.get("timestamp"))]
    eligible = align_vintage_rows(origin_safe, cutoff=cutoff)
    eligible = [row for row in eligible
                if target_key in row
                and all(name in row for name in predictor_keys)]
    fitted = fit_regression_executable(
        [float(row[target_key]) for row in eligible],
        {name: [float(row[name]) for row in eligible] for name in predictor_keys},
        target=target_key, minimum_train=minimum_train)
    result = fitted.execute()
    validation = dict(result["estimate"]["validation"])
    skill = float(validation["skill_vs_mean_baseline"])
    validation.update({
        "skill": skill,
        "beats_baseline": result["direction"] == "predictive_contribution",
        "vintage_cutoff": cutoff, "per_origin_knowledge_checked": True,
    })
    return FittedContextCandidate(
        "fitted_vintage_exogenous", hypothesis_id,
        {"coefficients": result["estimate"]["coefficients"],
         "coefficient_intervals_95": result["estimate"]["coefficient_intervals_95"]},
        validation, result["support"])


def fit_lagged_relationship(
    target_rows: list[dict[str, Any]], predictor_rows: list[dict[str, Any]], *,
    target_key: str, predictor_key: str, cutoff: str, hypothesis_id: str,
    lags: list[int] | None = None, minimum_train: int = 20,
) -> FittedContextCandidate:
    """Choose a lag using expanding-origin predictions, never full-data fit."""
    target = align_vintage_rows(target_rows, cutoff=cutoff)
    predictor = align_vintage_rows(predictor_rows, cutoff=cutoff)
    x_by_time = {str(row["timestamp"]): float(row[predictor_key]) for row in predictor}
    y = [(str(row["timestamp"]), float(row[target_key])) for row in target
         if str(row["timestamp"]) in x_by_time]
    candidates = sorted(set(lags or [0, 1, 2, 3, 6, 12]))
    scores: list[dict[str, Any]] = []
    for lag in candidates:
        pairs = [(x_by_time[y[i-lag][0]], y[i][1]) for i in range(lag, len(y))]
        predictions, actuals, baselines = [], [], []
        for origin in range(max(minimum_train, 3), len(pairs)):
            train = pairs[:origin]
            xbar = statistics.mean(item[0] for item in train)
            ybar = statistics.mean(item[1] for item in train)
            denom = sum((item[0] - xbar) ** 2 for item in train)
            slope = (sum((a-xbar)*(b-ybar) for a, b in train) / denom
                     if denom > 1e-12 else 0.0)
            predictions.append(ybar + slope * (pairs[origin][0] - xbar))
            actuals.append(pairs[origin][1]); baselines.append(ybar)
        if not actuals:
            continue
        mse = statistics.mean((a-b)**2 for a, b in zip(actuals, predictions))
        base = statistics.mean((a-b)**2 for a, b in zip(actuals, baselines))
        scores.append({"lag_steps": lag, "validation_points": len(actuals),
                       "mse": mse, "baseline_mse": base,
                       "skill": 1 - mse / max(base, 1e-12)})
    if not scores:
        raise ValueError("insufficient vintage-aligned history for lag validation")
    best = max(scores, key=lambda item: (item["skill"], -item["lag_steps"]))
    # Multiplicity-aware admission: require more evidence as more lags compete.
    threshold = min(.25, .02 + .01 * math.log2(max(1, len(scores))))
    supported = best["skill"] >= threshold and best["validation_points"] >= 8
    return FittedContextCandidate(
        "fitted_lagged_relationship", hypothesis_id,
        {"selected_lag_steps": best["lag_steps"], "skill": best["skill"]},
        {"scheme": "expanding_origin", "candidates": scores,
         "admission_threshold": threshold, "beats_baseline": supported,
         "vintage_cutoff": cutoff},
        "supported" if supported else "weak")


def fit_historical_analogue(
    episodes: list[dict[str, Any]], *, query_features: dict[str, float],
    cutoff: str, hypothesis_id: str, k: int = 5,
) -> FittedContextCandidate:
    """Evaluate nearest historical episodes with leave-one-episode-out skill."""
    cutoff_dt = _aware(cutoff)
    if cutoff_dt is None:
        raise ValueError("cutoff must be timezone-aware ISO-8601")
    names = tuple(sorted(query_features))
    eligible = [item for item in episodes
                if _aware(item.get("outcome_known_at")) is not None
                and _aware(item["outcome_known_at"]) <= cutoff_dt
                and all(name in (item.get("features") or {}) for name in names)
                and math.isfinite(float(item.get("outcome")))]
    if len(eligible) < 5:
        raise ValueError("historical analogues require five resolved episodes")
    scales = {name: max(statistics.pstdev(
        [float(item["features"][name]) for item in eligible]), 1e-12)
        for name in names}
    def distance(features: dict[str, float], item: dict[str, Any]) -> float:
        return math.sqrt(sum(((float(features[name]) -
                              float(item["features"][name])) / scales[name]) ** 2
                             for name in names))
    errors, baseline_errors = [], []
    for held in eligible:
        train = [item for item in eligible if item is not held]
        nearest = sorted(train, key=lambda item: distance(held["features"], item))[:k]
        prediction = statistics.mean(float(item["outcome"]) for item in nearest)
        truth = float(held["outcome"])
        errors.append(abs(truth - prediction))
        baseline_errors.append(abs(truth - statistics.mean(
            float(item["outcome"]) for item in train)))
    nearest = sorted(eligible, key=lambda item: distance(query_features, item))[:k]
    outcomes = [float(item["outcome"]) for item in nearest]
    skill = 1 - statistics.mean(errors) / max(statistics.mean(baseline_errors), 1e-12)
    supported = skill >= .02 and len(errors) >= 8
    return FittedContextCandidate(
        "fitted_historical_analogue", hypothesis_id,
        {"location": statistics.mean(outcomes), "lower": min(outcomes),
         "upper": max(outcomes), "matched_episode_ids": [
             str(item.get("episode_id")) for item in nearest]},
        {"scheme": "leave_one_episode_out", "episodes": len(errors),
         "mae": statistics.mean(errors),
         "global_mean_mae": statistics.mean(baseline_errors),
         "skill": skill, "beats_baseline": supported,
         "outcomes_known_by": cutoff},
        "supported" if supported else "weak")


def candidate_evidence_score(candidate: dict[str, Any]) -> dict[str, Any]:
    """Rank evidence, not provenance prestige or an LLM confidence claim."""
    validation = candidate.get("validation") or {}
    points = int(validation.get("validation_points") or
                 validation.get("episodes") or 0)
    skill = float(validation.get("skill") or 0.0)
    supported = bool(validation.get("beats_baseline"))
    score = max(-1.0, min(1.0, skill)) * min(1.0, points / 20.0)
    return {"score": score, "validation_points": points,
            "beats_baseline": supported,
            "decisive": supported and points >= 8 and score > 0,
            "automation_eligible": False,
            "basis": "out_of_sample_skill_times_evidence_fraction"}
