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
import re
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


def canonicalize_recursive_wrapper(
    wrapper: Any, *, target_name: str, driver_names: list[str],
) -> tuple[Any, dict[str, Any]]:
    """Recognize a verbose linear lag equation and bind it to safe recursion.

    This is syntax normalization, not equation inference: every term,
    coefficient and lag must already be explicit. Target-lag arrays are
    discarded rather than trusted; duplicate driver schedules must agree.
    """
    if not isinstance(wrapper, dict):
        return wrapper, {"status": "not_applicable"}
    transformation = wrapper.get("transformation", wrapper)
    if not isinstance(transformation, dict):
        return wrapper, {"status": "not_applicable"}
    expression = transformation.get("expression")
    if not isinstance(expression, dict):
        return wrapper, {"status": "not_applicable"}

    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.casefold())

    if expression.get("op") == "recursive_linear":
        # Some compilers encode the lag twice (series="X_0_lag2", lag=2).
        # Treat that as syntax only when the suffix and typed lag agree; the
        # future schedules must still be identical below before they collapse
        # onto the governed base-series history.
        lag_aliases: dict[str, str] = {}
        for term in expression.get("driver_terms") or []:
            if not isinstance(term, dict):
                continue
            source = str(term.get("series") or "")
            match = re.fullmatch(r"(.+?)[_-]?lag[_-]?(\d+)", source, re.I)
            if match and int(match.group(2)) == int(term.get("lag") or 0):
                lag_aliases[source] = match.group(1).rstrip("_-")
        actual_by_alias: dict[str, str] = {}
        for actual in driver_names:
            canonical_actual = lag_aliases.get(actual, actual)
            base = normalize(canonical_actual)
            source = normalize(actual)
            for alias in (source, base, "future" + base, base + "future",
                          "schedule" + base, base + "schedule",
                          "forecast" + base, base + "forecast"):
                if (alias in actual_by_alias
                        and actual_by_alias[alias] != canonical_actual):
                    return wrapper, {"status": "rejected",
                                     "reason": f"ambiguous driver alias {alias!r}"}
                actual_by_alias[alias] = canonical_actual
        rebound_terms = []
        aliases: dict[str, list[str]] = {}
        changed = False
        for term in expression.get("driver_terms") or []:
            if not isinstance(term, dict):
                return wrapper, {"status": "rejected",
                                 "reason": "recursive driver term is not an object"}
            source = str(term.get("series") or "")
            actual = actual_by_alias.get(normalize(source))
            if actual is None:
                return wrapper, {"status": "already_canonical"}
            rebound_terms.append({**term, "series": actual})
            aliases.setdefault(actual, []).append(source)
            changed = changed or actual != source
        if not changed:
            return wrapper, {"status": "already_canonical"}
        supplied = wrapper.get("series_values") or {}
        canonical_values: dict[str, Any] = {}
        for actual, names in aliases.items():
            payloads = [supplied.get(name) for name in names]
            if any(not isinstance(item, dict) for item in payloads):
                return wrapper, {"status": "rejected",
                                 "reason": f"missing future schedule for {actual!r}"}
            values = [item.get("values") for item in payloads]
            if any(value != values[0] for value in values[1:]):
                return wrapper, {"status": "rejected",
                                 "reason": f"conflicting schedules for {actual!r}"}
            claim_ids = sorted({str(claim_id) for item in payloads
                                for claim_id in item.get("source_claim_ids") or []})
            canonical_values[actual] = {
                **payloads[0], "source_claim_ids": claim_ids,
                "canonicalized_from": sorted(names)}
        units = dict(wrapper.get("units") or {})
        canonical_units = {"primary": units.get(
            "primary", transformation.get("output_unit", "unknown"))}
        for actual, names in aliases.items():
            canonical_units[actual] = next(
                (units[name] for name in names if name in units), "unknown")
        return ({**wrapper,
                 "transformation": {**transformation, "expression": {
                     **expression, "driver_terms": rebound_terms},
                     "syntax_canonicalization": "recursive_driver_alias"},
                 "series_values": canonical_values, "units": canonical_units},
                {"status": "canonicalized", "target": target_name,
                 "drivers": sorted(canonical_values)})

    def flatten(node: Any) -> list[Any] | None:
        if not isinstance(node, dict):
            return None
        if node.get("op") == "add":
            output = []
            for child in node.get("args") or []:
                terms = flatten(child)
                if terms is None:
                    return None
                output.extend(terms)
            return output
        return [node]

    driver_by_normal: dict[str, str] = {}
    for name in driver_names:
        base = normalize(name)
        for alias in (base, "future" + base, base + "future",
                      "schedule" + base, base + "schedule"):
            driver_by_normal[alias] = name
    target_key = normalize(target_name)
    target_aliases = {target_key, "future" + target_key, target_key + "future"}
    ar_terms, driver_terms = [], []
    aliases: dict[str, list[str]] = {}
    for term in flatten(expression) or []:
        if not isinstance(term, dict) or term.get("op") != "multiply":
            return wrapper, {"status": "not_applicable"}
        args = term.get("args") or []
        if len(args) != 2:
            return wrapper, {"status": "not_applicable"}
        literal = next((item for item in args if isinstance(item, dict)
                        and item.get("op") == "literal"), None)
        value_node = next((item for item in args if isinstance(item, dict)
                           and item is not literal), None)
        if literal is None or value_node is None:
            return wrapper, {"status": "not_applicable"}
        if value_node.get("op") == "series":
            name = str(value_node.get("name") or "")
            match = re.fullmatch(r"(.+?)[_-]?lag[_-]?(\d+)", name, re.I)
            if not match:
                return wrapper, {"status": "not_applicable"}
            base, lag = normalize(match.group(1)), int(match.group(2))
        elif value_node.get("op") == "lag":
            lag_args = value_node.get("args") or []
            source = lag_args[0] if len(lag_args) == 1 else None
            if not isinstance(source, dict) or source.get("op") != "series":
                return wrapper, {"status": "not_applicable"}
            name = str(source.get("name") or "")
            base = normalize(name)
            try:
                lag = int(value_node.get("steps"))
            except (TypeError, ValueError):
                return wrapper, {"status": "not_applicable"}
        else:
            return wrapper, {"status": "not_applicable"}
        coefficient = _finite(literal.get("value"), "expression.coefficient")
        if base in target_aliases:
            ar_terms.append({"lag": lag, "coefficient": coefficient})
        elif base in driver_by_normal:
            actual = driver_by_normal[base]
            driver_terms.append({"series": actual, "lag": lag,
                                 "coefficient": coefficient})
            aliases.setdefault(actual, []).append(name)
        else:
            return wrapper, {"status": "not_applicable",
                             "reason": f"unresolved lagged series {name!r}"}
    if not ar_terms:
        return wrapper, {"status": "not_applicable"}

    supplied = wrapper.get("series_values") or {}
    canonical_values: dict[str, Any] = {}
    for actual, names in aliases.items():
        payloads = [supplied.get(name) for name in names]
        if not payloads or any(not isinstance(item, dict) for item in payloads):
            return wrapper, {"status": "rejected",
                             "reason": f"missing future schedule for {actual!r}"}
        values = [item.get("values") for item in payloads]
        if any(value != values[0] for value in values[1:]):
            return wrapper, {"status": "rejected",
                             "reason": f"conflicting lag schedules for {actual!r}"}
        claim_ids = sorted({str(claim_id) for item in payloads
                            for claim_id in item.get("source_claim_ids") or []})
        canonical_values[actual] = {
            **payloads[0], "source_claim_ids": claim_ids,
            "canonicalized_from": sorted(names),
        }
    output_unit = str(transformation.get("output_unit") or "unknown")
    canonical = {
        **transformation,
        "expression": {
            "op": "recursive_linear", "output_unit": output_unit,
            "intercept": 0.0,
            "autoregressive_terms": sorted(ar_terms, key=lambda item: item["lag"]),
            "driver_terms": sorted(driver_terms,
                                   key=lambda item: (item["series"], item["lag"])),
        },
        "syntax_canonicalization": "verbose_lag_arrays_to_recursive_linear",
    }
    units = dict(wrapper.get("units") or {})
    canonical_units = {"primary": units.get("primary", output_unit)}
    for actual, names in aliases.items():
        canonical_units[actual] = next(
            (units[name] for name in names if name in units), "unknown")
    return ({**wrapper, "transformation": canonical,
             "series_values": canonical_values, "units": canonical_units},
            {"status": "canonicalized", "target": target_name,
             "drivers": sorted(canonical_values)})


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


def expand_cited_history_segments(
    segments_by_series: Any, *, timestamps: list[datetime], cutoff: datetime,
    claim_spans: dict[str, str], allowed_claim_ids: list[str],
) -> dict[str, list[float]]:
    """Expand source-entailed historical ranges onto a governed grid.

    This is an explicit representation bridge, never an inferred scale or
    repair of the structured data. Every range endpoint and value must occur
    in a verified claim, ranges may not overlap, and the complete grid must be
    covered before the document-supplied series can be used for replay.
    """
    if not segments_by_series:
        return {}
    if not isinstance(segments_by_series, dict):
        raise TransformationError(
            "INVALID_HISTORY_SEGMENTS", "historical_series_segments",
            "Historical series segments must be an object keyed by series.")
    allowed = set(allowed_claim_ids)
    output: dict[str, list[float]] = {}
    for name, raw_segments in segments_by_series.items():
        if not isinstance(raw_segments, list) or not raw_segments:
            raise TransformationError(
                "INVALID_HISTORY_SEGMENTS", f"historical_series_segments.{name}",
                "Each historical driver requires one or more cited ranges.")
        expanded: list[float | None] = [None] * len(timestamps)
        for index, segment in enumerate(raw_segments):
            field = f"historical_series_segments.{name}[{index}]"
            if not isinstance(segment, dict):
                raise TransformationError("INVALID_HISTORY_SEGMENT", field,
                                          "A history range must be an object.")
            claim_ids = {str(value) for value in
                         segment.get("source_claim_ids") or []}
            if not claim_ids or not claim_ids.issubset(allowed):
                raise TransformationError("UNVERIFIED_CLAIMS", field,
                                          "A history range must cite verified claims.")
            start_raw, end_raw = str(segment.get("start") or ""), str(
                segment.get("end") or "")
            try:
                start, end = (datetime.fromisoformat(start_raw),
                              datetime.fromisoformat(end_raw))
                if start.tzinfo is None:
                    start = start.replace(tzinfo=cutoff.tzinfo)
                if end.tzinfo is None:
                    end = end.replace(tzinfo=cutoff.tzinfo)
            except (TypeError, ValueError):
                start = end = None
            if start is None or end is None or start > end or end > cutoff:
                raise TransformationError("INVALID_HISTORY_RANGE", field,
                                          "History ranges must end by the cutoff.")
            value = _finite(segment.get("value"), field + ".value")
            source = " ".join(claim_spans.get(claim_id, "")
                              for claim_id in claim_ids)
            value_tokens = {match.group(0) for match in re.finditer(
                r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
                source.replace(",", ""))}
            def endpoint_entailed(raw: str, parsed: datetime) -> bool:
                if raw in source:
                    return True
                return (len(raw) > 10 and raw[:10] in source
                        and parsed.hour == parsed.minute == parsed.second
                        == parsed.microsecond == 0)

            if (not endpoint_entailed(start_raw, start)
                    or not endpoint_entailed(end_raw, end)
                    or not any(math.isclose(value, float(token), rel_tol=1e-12,
                                            abs_tol=1e-12)
                               for token in value_tokens)):
                raise TransformationError(
                    "UNENTAILED_HISTORY_RANGE", field,
                    "Range endpoints and value must occur in the cited source.")
            for position, timestamp in enumerate(timestamps):
                if start <= timestamp <= end:
                    if expanded[position] is not None:
                        raise TransformationError(
                            "OVERLAPPING_HISTORY_RANGES", field,
                            "Historical ranges may not overlap on the governed grid.")
                    expanded[position] = value
        if any(value is None for value in expanded):
            raise TransformationError(
                "INCOMPLETE_HISTORY_COVERAGE",
                f"historical_series_segments.{name}",
                "Cited ranges must cover every governed pre-cutoff timestamp.")
        output[str(name)] = [float(value) for value in expanded]
    return output


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


def _model_authored_constants(node: Any, *, role: str = "literal") \
        -> list[tuple[float, str]]:
    """Return every numeric choice embedded in an untrusted AST."""
    if not isinstance(node, dict):
        return []
    op = str(node.get("op") or "")
    found: list[tuple[float, str]] = []
    if op == "literal":
        try:
            found.append((float(node.get("value")), role))
        except (TypeError, ValueError):
            pass
    if op == "reference_power":
        for key in ("input_reference", "input_ref", "output_reference", "output_ref"):
            if key not in node:
                continue
            raw = node[key]
            raw = raw.get("value") if isinstance(raw, dict) else raw
            try:
                found.append((float(raw), key))
            except (TypeError, ValueError):
                pass
        try:
            found.append((float(node.get("exponent", 1)), "exponent"))
        except (TypeError, ValueError):
            pass
    if op == "linear_combination":
        for term in node.get("terms") or []:
            if not isinstance(term, dict):
                continue
            try:
                found.append((float(term.get("coefficient")), "coefficient"))
            except (TypeError, ValueError):
                pass
    if op == "recursive_linear":
        for term in [*(node.get("autoregressive_terms") or []),
                     *(node.get("driver_terms") or [])]:
            if not isinstance(term, dict):
                continue
            for key in ("coefficient", "lag"):
                try:
                    found.append((float(term.get(key)), key))
                except (TypeError, ValueError):
                    pass
        if "intercept" in node:
            try:
                found.append((float(node.get("intercept")), "intercept"))
            except (TypeError, ValueError):
                pass
        if "intercept" in node:
            try:
                found.append((float(node.get("intercept")), "intercept"))
            except (TypeError, ValueError):
                pass
    for key in ("steps", "window", "lower", "upper", "quantile"):
        value = node.get(key)
        if value is None:
            continue
        try:
            found.append((float(value), key))
        except (TypeError, ValueError):
            pass
    for index, child in enumerate(node.get("args") or []):
        child_role = "exponent" if op in {"power", "pow"} and index == 1 else "literal"
        found.extend(_model_authored_constants(child, role=child_role))
    for key in ("arg", "left", "right", "series"):
        if isinstance(node.get(key), dict):
            found.extend(_model_authored_constants(node[key]))
    return found


def _constant_is_entailed(value: float, *, role: str, text: str) -> bool:
    # An omitted intercept is canonically represented as additive identity.
    # It adds no information and cannot move the path; requiring prose to say
    # “plus zero” would reject ordinary equations for formatting alone.
    if role == "intercept" and value == 0:
        return True
    # Dates and clock times are provenance, not formula constants.
    clean = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b",
                   " ", text)
    clean = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", clean)
    # Equations commonly typeset a signed coefficient as ``- 2``. Preserve
    # that sign when extracting entailed constants instead of treating it as
    # an unrelated positive two.
    clean = re.sub(r"([+-])\s+(?=\d)", r"\1", clean)
    numbers = []
    for token in re.findall(
            r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
            clean.replace(",", "")):
        try:
            numbers.append(float(token))
        except ValueError:
            pass
    if any(math.isclose(value, item, rel_tol=1e-12, abs_tol=1e-12)
           for item in numbers):
        return True
    words = clean.casefold()
    word_values = {
        0.25: r"\b(?:a\s+quarter|one\s+quarter|quarter)\b",
        0.5: r"\b(?:a\s+half|one\s+half|half)\b",
        2.0: r"\b(?:twice|double)\b",
        3.0: r"\btriple\b",
    }
    if any(math.isclose(value, expected) and re.search(pattern, words)
           for expected, pattern in word_values.items()):
        return True
    return (role == "exponent"
            and ((value == 2 and re.search(r"\b(?:square|squared|quadratic)\b", words))
                 or (value == 3 and re.search(r"\b(?:cube|cubed|cubic)\b", words))))


def validate_transformation(
    raw: Any, *, series: list[str], claim_ids: list[str], cutoff: str,
    units: dict[str, str] | None = None,
    claim_spans: dict[str, str] | None = None,
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
    expression, derived_coefficient_units = _bind_linear_units(
        expression, output_unit=declared_raw, units=units or {})
    if claim_spans is not None:
        cited_text = " ".join(str(claim_spans.get(item) or "") for item in cited)
        for value, role in _model_authored_constants(expression):
            if not _constant_is_entailed(value, role=role, text=cited_text):
                raise TransformationError(
                    "UNENTAILED_TRANSFORMATION_CONSTANT", "expression",
                    f"Transformation constant {value:g} ({role}) is absent "
                    "from every cited source span.")
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
                       "known_at_cutoff": True, "units_checked": True,
                       "coefficient_units_derived": derived_coefficient_units,
                       "constants_entailed": claim_spans is not None},
    }
    payload["transformation_id"] = "transform-" + hashlib.sha256(
        _canonical(payload).encode()).hexdigest()[:16]
    payload["seal_sha256"] = hashlib.sha256(
        _canonical(payload).encode()).hexdigest()
    return payload


def _bind_linear_units(node: Any, *, output_unit: str,
                       units: dict[str, str]) -> tuple[Any, bool]:
    """Canonicalize omitted coefficient units in an additive equation.

    LLMs normally write ``a*x + b*y`` without spelling ``a`` as
    ``target_unit/x_unit``.  When the transformation declares one output unit,
    that conversion is mathematically forced.  We derive only that forced
    metadata; values, operators, series and signs remain untouched.
    """
    if not output_unit or not isinstance(node, dict):
        return node, False

    def static_unit(value: Any) -> str:
        if not isinstance(value, dict):
            return "unknown"
        op = str(value.get("op") or "")
        if op == "literal":
            return str(value.get("unit") or "dimensionless")
        if op in {"series", "primary"}:
            name = str(value.get("name") or value.get("series") or
                       ("primary" if op == "primary" else ""))
            return str(units.get(name) or "unknown")
        args = value.get("args") or ([value.get("arg")]
                                     if "arg" in value else [])
        if op in {"lag", "difference", "rolling_mean", "quantile", "clip"} \
                and args:
            return static_unit(args[0])
        if op == "percent_change":
            return "dimensionless"
        return "unknown"

    def bind(value: Any, expected: str | None = None) -> tuple[Any, bool]:
        if not isinstance(value, dict):
            return value, False
        clean = dict(value)
        op = str(clean.get("op") or "")
        changed = False
        if op in {"add", "subtract"}:
            children = clean.get("args")
            if not isinstance(children, list) and "left" in clean and "right" in clean:
                children = [clean.get("left"), clean.get("right")]
                clean.pop("left", None)
                clean.pop("right", None)
            if isinstance(children, list):
                rebound = [bind(child, expected or output_unit)
                           for child in children]
                clean["args"] = [item[0] for item in rebound]
                changed = any(item[1] for item in rebound)
            return clean, changed
        if op == "literal" and expected and "unit" not in clean:
            clean["unit"] = expected
            return clean, True
        if op == "multiply":
            children = list(clean.get("args") or [])
            if len(children) == 2 and expected:
                for literal_index, other_index in ((0, 1), (1, 0)):
                    literal = children[literal_index]
                    if not isinstance(literal, dict) or literal.get("op") != "literal" \
                            or "unit" in literal:
                        continue
                    input_unit = static_unit(children[other_index])
                    if input_unit != "unknown":
                        literal = dict(literal)
                        literal["unit"] = ("dimensionless"
                                           if input_unit == expected else
                                           f"{expected}/{input_unit}")
                        children[literal_index] = literal
                        changed = True
                        break
            rebound = [bind(child) for child in children]
            clean["args"] = [item[0] for item in rebound]
            return clean, changed or any(item[1] for item in rebound)
        children = clean.get("args")
        if isinstance(children, list):
            rebound = [bind(child) for child in children]
            clean["args"] = [item[0] for item in rebound]
            changed = any(item[1] for item in rebound)
        return clean, changed

    # Only additive roots define a forced common target unit. A standalone
    # multiplication may intentionally produce a compound unit and is left
    # untouched unless the explicit linear_combination macro is used.
    if str(node.get("op") or "") not in {"add", "subtract"}:
        return node, False
    return bind(node, output_unit)


def compile_transformation(
    raw: Any, *, series: list[str], claim_ids: list[str], cutoff: str,
    units: dict[str, str] | None = None, repair: Any = None,
    claim_spans: dict[str, str] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Validate once and permit exactly one field-bounded repair.

    The repair may alter only the top-level field implicated by the first
    typed violation.  This keeps repair useful for schema mistakes without
    allowing the model to replace the claim, lane, and expression together.
    """
    try:
        compiled = validate_transformation(
            raw, series=series, claim_ids=claim_ids, cutoff=cutoff, units=units,
            claim_spans=claim_spans)
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
            repair, series=series, claim_ids=claim_ids, cutoff=cutoff, units=units,
            claim_spans=claim_spans)
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
    if raw_op == "recursive_linear":
        output_unit = str(node.get("output_unit") or "")
        if not output_unit:
            raise TransformationError(
                "MISSING_OUTPUT_UNIT", f"{path}.output_unit",
                "A recursive linear equation requires its target unit.")
        autoregressive = node.get("autoregressive_terms") or []
        drivers = node.get("driver_terms") or []
        if not isinstance(autoregressive, list) or len(autoregressive) > 16 \
                or not isinstance(drivers, list) or len(drivers) > 32 \
                or not autoregressive and not drivers:
            raise TransformationError(
                "INVALID_RECURSIVE_TERMS", path,
                "A recursive equation requires 1..48 bounded AR/driver terms.")

        def term(raw: Any, *, driver: bool, index: int) -> dict[str, Any]:
            field = "driver_terms" if driver else "autoregressive_terms"
            if not isinstance(raw, dict):
                raise TransformationError(
                    "INVALID_RECURSIVE_TERM", f"{path}.{field}[{index}]",
                    "Each recursive term must be an object.")
            coefficient = _finite(
                raw.get("coefficient"), f"{path}.{field}[{index}].coefficient")
            try:
                lag = int(raw.get("lag"))
            except (TypeError, ValueError) as exc:
                raise TransformationError(
                    "INVALID_WINDOW", f"{path}.{field}[{index}].lag",
                    "Recursive lags must be positive integers.") from exc
            if lag < 1 or lag > 10_000:
                raise TransformationError(
                    "INVALID_WINDOW", f"{path}.{field}[{index}].lag",
                    "Recursive lags must be between 1 and 10000.")
            clean = {"coefficient": coefficient, "lag": lag}
            if driver:
                name = str(raw.get("series") or "")
                if name not in series:
                    raise TransformationError(
                        "UNKNOWN_SERIES", f"{path}.{field}[{index}].series",
                        f"Series {name!r} is unavailable.")
                clean["series"] = name
                clean["coefficient_unit"] = (
                    "dimensionless" if units.get(name) == output_unit else
                    f"{output_unit}/{units.get(name, 'unknown')}")
            else:
                clean["coefficient_unit"] = "dimensionless"
            return clean

        clean = {
            "op": "recursive_linear", "output_unit": output_unit,
            "intercept": _finite(node.get("intercept", 0.0),
                                 f"{path}.intercept"),
            "autoregressive_terms": [term(value, driver=False, index=index)
                                     for index, value in enumerate(autoregressive)],
            "driver_terms": [term(value, driver=True, index=index)
                             for index, value in enumerate(drivers)],
        }
        state["nodes"] += len(autoregressive) + len(drivers)
        if state["nodes"] > MAX_TRANSFORM_NODES:
            raise TransformationError("EXPRESSION_TOO_LARGE", path,
                                      "Transformation exceeds the node limit.")
        return clean, output_unit
    if raw_op == "linear_combination":
        # Compact safe macro for equations whose coefficients convert several
        # input units into one target unit.  The coefficient units are derived
        # by the engine (target/input), never trusted from model text, and the
        # macro expands to the ordinary sealed arithmetic AST.
        output_unit = str(node.get("output_unit") or "")
        terms = node.get("terms")
        if not output_unit:
            raise TransformationError(
                "MISSING_OUTPUT_UNIT", f"{path}.output_unit",
                "A linear combination requires its target unit.")
        if not isinstance(terms, list) or not 1 <= len(terms) <= 32:
            raise TransformationError(
                "INVALID_LINEAR_TERMS", f"{path}.terms",
                "A linear combination requires between 1 and 32 terms.")
        expanded_terms: list[dict[str, Any]] = []
        for index, term in enumerate(terms):
            if not isinstance(term, dict):
                raise TransformationError(
                    "INVALID_LINEAR_TERM", f"{path}.terms[{index}]",
                    "Each linear term must be an object.")
            name = str(term.get("series") or "")
            if name not in series:
                raise TransformationError(
                    "UNKNOWN_SERIES", f"{path}.terms[{index}].series",
                    f"Series {name!r} is unavailable.")
            input_unit = str(units.get(name) or "unknown")
            coefficient_unit = ("dimensionless" if input_unit == output_unit
                                else f"{output_unit}/{input_unit}")
            expanded_terms.append({"op": "multiply", "args": [
                {"op": "literal", "value": term.get("coefficient"),
                 "unit": coefficient_unit},
                {"op": "series", "name": name,
                 "quantile": term.get("quantile", "q50")},
            ]})
        if "intercept" in node:
            expanded_terms.append({
                "op": "literal", "value": node.get("intercept"),
                "unit": output_unit})
        expanded = (expanded_terms[0] if len(expanded_terms) == 1 else
                    {"op": "add", "args": expanded_terms})
        return _validate_expression(
            expanded, path=path, depth=depth + 1, state=state,
            series=series, units=units)
    if raw_op == "reference_power":
        # Compact safe macro for common reference laws:
        # output_ref * (series / input_ref) ** exponent. It expands into the
        # ordinary canonical AST before sealing; execution learns no new
        # operator and never evaluates model-authored code.
        series_node = node.get("series")
        if isinstance(series_node, str):
            series_node = {"op": "series", "name": series_node}

        def reference_literal(value: Any, unit_key: str) -> dict[str, Any]:
            if isinstance(value, dict):
                return {"op": "literal", **value}
            return {"op": "literal", "value": value,
                    "unit": node.get(unit_key)}

        expanded = {
            "op": "multiply", "args": [
                reference_literal(
                    node.get("output_reference", node.get("output_ref")),
                    "output_unit"),
                {"op": "power", "args": [
                    {"op": "divide", "args": [
                        series_node,
                        reference_literal(
                            node.get("input_reference", node.get("input_ref")),
                            "input_unit"),
                    ]},
                    {"op": "literal", "value": node.get("exponent", 1)},
                ]},
            ],
        }
        return _validate_expression(
            expanded, path=path, depth=depth + 1, state=state,
            series=series, units=units)
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
        output_unit = ("dimensionless" if child_units[0] == "dimensionless"
                       or int(value) == 0 else
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
    claim_spans: dict[str, str] | None = None,
    history_values: list[float] | None = None,
    history_series: dict[str, list[float]] | None = None,
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
                "Future series require values, known_at, and source_claim_id(s).")
        known_at = _aware(supplied.get("known_at"))
        cutoff = _aware(compiled.get("cutoff"))
        source_claims = [str(value) for value in
                         supplied.get("source_claim_ids") or []]
        singular_claim = str(supplied.get("source_claim_id") or "")
        if singular_claim:
            source_claims.append(singular_claim)
        source_claims = sorted(set(source_claims))
        if known_at is None or cutoff is None or known_at > cutoff:
            raise TransformationError(
                "FUTURE_SERIES_NOT_KNOWN_AT_CUTOFF", f"series_values.{name}.known_at",
                "Future input was not knowable at the forecast cutoff.")
        if not source_claims or set(source_claims) - set(compiled["claim_ids"]):
            raise TransformationError(
                "UNVERIFIED_FUTURE_SERIES", f"series_values.{name}.source_claim_ids",
                "Future input must cite one or more of the transformation's verified claims.")
        values = supplied.get("values")
        if not isinstance(values, list):
            raise TransformationError(
                "INVALID_FUTURE_SERIES", f"series_values.{name}.values",
                "Future input values must be an array.")
        finite_values = [_finite(value, name) for value in values]
        span = " ".join(str((claim_spans or {}).get(claim_id) or "")
                        for claim_id in source_claims)
        cited_numbers = []
        for token in re.findall(
                r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
                span.replace(",", "")):
            try:
                cited_numbers.append(float(token))
            except ValueError:
                pass
        missing = [value for value in set(finite_values)
                   if not any(math.isclose(value, cited, rel_tol=1e-12,
                                           abs_tol=1e-12)
                              for cited in cited_numbers)]
        if missing:
            raise TransformationError(
                "UNENTAILED_FUTURE_SERIES_VALUES",
                f"series_values.{name}.values",
                "Every supplied future value must occur in its cited source span.")
        environment[name] = finite_values
    if any(len(values) != width for values in environment.values()):
        raise TransformationError("HORIZON_MISMATCH", "series_values",
                                  "Every future series must match the primary horizon.")
    if compiled["expression"].get("op") == "recursive_linear":
        values, lower_widths, upper_widths = _execute_recursive_linear(
            compiled["expression"], primary=primary, environment=environment,
            history_values=history_values or [],
            history_series=history_series or {})
        quantile_paths = {
            "q10": [value - width for value, width in zip(values, lower_widths)],
            "q50": values,
            "q90": [value + width for value, width in zip(values, upper_widths)],
        }
    else:
        values = _execute_expression(compiled["expression"], primary=primary,
                                     environment=environment, width=width,
                                     primary_quantile="q50")
        quantile_paths = {
            quantile: _execute_expression(
                compiled["expression"], primary=primary,
                environment=environment, width=width,
                primary_quantile=quantile)
            for quantile in ("q10", "q50", "q90")
        }
    if len(values) != width or any(not math.isfinite(value) for value in values):
        raise TransformationError("INVALID_EXECUTION_RESULT", "expression",
                                  "Transformation did not yield one finite value per step.")
    rows = []
    for index, value in enumerate(values):
        source = primary[index]
        ordered = sorted(quantile_paths[key][index]
                         for key in ("q10", "q50", "q90"))
        rows.append({"timestamp": source.get("timestamp"), "point": value,
                     "q10": ordered[0], "q50": value, "q90": ordered[-1]})
    validation = dict(compiled["validation"])
    if compiled["expression"].get("op") == "recursive_linear":
        expression = compiled["expression"]
        order = max((term["lag"] for term in
                     expression["autoregressive_terms"]), default=0)
        coefficients = [0.0] * order
        for term in expression["autoregressive_terms"]:
            coefficients[term["lag"] - 1] += float(term["coefficient"])
        impulse_history = [0.0] * max(0, order - 1) + ([1.0] if order else [])
        impulse = []
        for _ in range(max(256, width * 4)):
            value = sum(coefficient * impulse_history[-lag]
                        for lag, coefficient in enumerate(coefficients, 1))
            impulse_history.append(value)
            impulse.append(value)
        impulse_peak = max((abs(value) for value in impulse), default=0.0)
        impulse_tail = max((abs(value) for value in impulse[-32:]), default=0.0)
        primary_width = max((
            _finite(row.get("q90", row.get("q50", row.get("point"))), "primary.q90")
            - _finite(row.get("q10", row.get("q50", row.get("point"))), "primary.q10")
            for row in primary), default=0.0)
        candidate_width = max((row["q90"] - row["q10"] for row in rows),
                              default=0.0)
        interval_growth = (candidate_width / primary_width
                           if primary_width > 1e-15 else
                           1.0 if candidate_width <= 1e-15 else math.inf)
        validation.update({
            "recurrence_uncertainty": "linear_state_covariance",
            "recurrence_impulse_peak": impulse_peak,
            "recurrence_impulse_tail": impulse_tail,
            "recurrence_stable": impulse_peak <= 100 and impulse_tail <= 10,
            "interval_growth_ratio": interval_growth,
            "recurrence_plausibility_passed": (
                impulse_peak <= 100 and impulse_tail <= 10
                and interval_growth <= 20),
        })
        validation.update(_replay_recursive_linear(
            expression, history_values=history_values or [],
            history_series=history_series or {}))
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


def _replay_recursive_linear(
    node: dict[str, Any], *, history_values: list[float],
    history_series: dict[str, list[float]], minimum_points: int = 8,
) -> dict[str, Any]:
    """Test fixed recurrence claims on pre-cutoff observations.

    Each origin uses observed lags only.  The recurrence is not refit, and the
    comparison is the robust last-value forecast on the identical origins.
    This is deliberately a small admission test, not a claim of causality.
    """
    ar_terms = node.get("autoregressive_terms") or []
    driver_terms = node.get("driver_terms") or []
    maximum_lag = max(
        [int(term["lag"]) for term in [*ar_terms, *driver_terms]],
        default=0)
    target = [float(value) for value in history_values]
    required_drivers = {str(term["series"]) for term in driver_terms}
    drivers = {name: [float(value) for value in history_series.get(name) or []]
               for name in required_drivers}
    aligned = bool(target) and all(len(values) == len(target)
                                   for values in drivers.values())
    if not aligned or len(target) <= maximum_lag:
        return {
            "recurrence_replay_scheme": "fixed_equation_expanding_origin",
            "recurrence_replay_points": 0,
            "recurrence_replay_beats_baseline": False,
            "recurrence_replay_admitted": False,
            "recurrence_replay_reason": "insufficient_aligned_history",
            "per_origin_knowledge_checked": True,
        }
    candidate_errors: list[float] = []
    baseline_errors: list[float] = []
    intercept = float(node.get("intercept") or 0.0)
    for origin in range(max(1, maximum_lag), len(target)):
        prediction = intercept
        prediction += sum(float(term["coefficient"])
                          * target[origin - int(term["lag"])]
                          for term in ar_terms)
        prediction += sum(float(term["coefficient"])
                          * drivers[str(term["series"])][
                              origin - int(term["lag"])]
                          for term in driver_terms)
        candidate_errors.append(abs(target[origin] - prediction))
        baseline_errors.append(abs(target[origin] - target[origin - 1]))
    candidate_mae = sum(candidate_errors) / len(candidate_errors)
    baseline_mae = sum(baseline_errors) / len(baseline_errors)
    skill = (1.0 - candidate_mae / baseline_mae
             if baseline_mae > 1e-15 else
             1.0 if candidate_mae <= 1e-15 else -math.inf)
    enough = len(candidate_errors) >= minimum_points
    beats = candidate_mae < baseline_mae
    return {
        "recurrence_replay_scheme": "fixed_equation_expanding_origin",
        "recurrence_replay_points": len(candidate_errors),
        "recurrence_replay_candidate_mae": candidate_mae,
        "recurrence_replay_baseline_mae": baseline_mae,
        "recurrence_replay_skill": skill,
        "recurrence_replay_beats_baseline": beats,
        "recurrence_replay_admitted": enough and beats,
        "recurrence_replay_reason": (
            "admitted" if enough and beats else
            "insufficient_validation_points" if not enough else
            "did_not_beat_last_value"),
        "per_origin_knowledge_checked": True,
    }


def _execute_recursive_linear(
    node: dict[str, Any], *, primary: list[dict[str, Any]],
    environment: dict[str, list[float]], history_values: list[float],
    history_series: dict[str, list[float]],
) -> tuple[list[float], list[float], list[float]]:
    """Execute a sealed ARX recurrence with conservative width propagation."""
    ar_terms = node["autoregressive_terms"]
    driver_terms = node["driver_terms"]
    max_target_lag = max((term["lag"] for term in ar_terms), default=0)
    if len(history_values) < max_target_lag:
        raise TransformationError(
            "MISSING_RECURSIVE_HISTORY", "history_values",
            f"Recursive target requires {max_target_lag} trusted historical values.")
    for term in driver_terms:
        name, lag = term["series"], term["lag"]
        if name not in environment:
            raise TransformationError(
                "MISSING_FUTURE_SERIES", name,
                "A recursive driver requires its cited future schedule.")
        if len(history_series.get(name) or []) < lag:
            raise TransformationError(
                "MISSING_RECURSIVE_HISTORY", f"history_series.{name}",
                f"Recursive driver {name!r} requires {lag} trusted historical values.")
    output: list[float] = []
    lower_widths: list[float] = []
    upper_widths: list[float] = []
    order = max_target_lag
    coefficients = [0.0] * order
    for term in ar_terms:
        coefficients[term["lag"] - 1] += float(term["coefficient"])
    lower_cov = [[0.0] * order for _ in range(order)]
    upper_cov = [[0.0] * order for _ in range(order)]

    def advance_covariance(covariance: list[list[float]],
                           innovation_width: float
                           ) -> tuple[list[list[float]], float]:
        if order == 0:
            return covariance, innovation_width
        transition = [coefficients] + [
            [1.0 if column == row - 1 else 0.0 for column in range(order)]
            for row in range(1, order)]
        projected = [[sum(
            transition[i][left] * covariance[left][right]
            * transition[j][right]
            for left in range(order) for right in range(order))
            for j in range(order)] for i in range(order)]
        projected[0][0] += innovation_width ** 2
        return projected, math.sqrt(max(0.0, projected[0][0]))

    for index, row in enumerate(primary):
        value = float(node["intercept"])
        for term in ar_terms:
            position = index - term["lag"]
            prior = (output[position] if position >= 0
                     else float(history_values[position]))
            value += term["coefficient"] * prior
        for term in driver_terms:
            position = index - term["lag"]
            source = (environment[term["series"]][position] if position >= 0
                      else history_series[term["series"]][position])
            value += term["coefficient"] * float(source)
        q50 = _finite(row.get("q50", row.get("point")), "primary.q50")
        innovation_lower = max(0.0, q50 - _finite(row.get("q10", q50), "primary.q10"))
        innovation_upper = max(0.0, _finite(row.get("q90", q50), "primary.q90") - q50)
        lower_cov, lower_width = advance_covariance(
            lower_cov, innovation_lower)
        upper_cov, upper_width = advance_covariance(
            upper_cov, innovation_upper)
        output.append(value)
        lower_widths.append(lower_width)
        upper_widths.append(upper_width)
    return output, lower_widths, upper_widths


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
