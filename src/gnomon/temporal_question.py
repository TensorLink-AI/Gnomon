"""Typed, validated temporal intent shared by every public surface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Iterable

from .contracts import GnomonError


VERBS = frozenset({"describe", "predict", "compare", "detect", "decide"})
PROPERTIES = frozenset({
    "level", "trend", "seasonality", "volatility", "regime", "extreme",
    "dependence",
})
MEASURES = frozenset({
    "point", "slope", "period", "residual_scale", "marginal_variability",
    "change", "maximum", "minimum", "correlation",
})
CONTEXT_POLICIES = frozenset({"ignore", "measure", "scenario"})
AGGREGATIONS = {
    "volatility": "median_normalized_scale_ratio",
    "seasonality": "median_alignment",
}

_PROPERTY_ALIASES = {
    "average": "level", "mean": "level", "value": "level",
    "direction": "trend", "growth": "trend",
    "seasonal": "seasonality", "cycle": "seasonality",
    "variance": "volatility", "variability": "volatility",
    "noise": "volatility", "dispersion": "volatility",
    "changepoint": "regime", "shift": "regime",
    "outlier": "extreme", "peak": "extreme",
    "relationship": "dependence", "correlation": "dependence",
}


@dataclass(frozen=True)
class TemporalQuestion:
    id: str
    verb: str
    target: str
    property: str
    horizon: int | None = None
    comparison: dict[str, Any] | None = None
    measure: str | None = None
    context_policy: str = "ignore"
    scope: str = "series"
    members: tuple[str, ...] = ()
    aggregation: str | None = None
    answer_vocabulary: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {key: value for key, value in asdict(self).items()
                   if value is not None and key not in {"scope", "members",
                                                        "aggregation"}}
        if self.scope != "series":
            payload["target"] = {
                "kind": self.scope, "members": list(self.members),
                **({"aggregation": self.aggregation}
                   if self.aggregation else {}),
            }
        return payload


def _repair(raw: Any, targets: list[str]) -> dict[str, Any]:
    seed = raw if isinstance(raw, dict) else {}
    return {"tool": "gnomon_describe", "arguments": {
        "target_column": targets[0] if len(targets) == 1 else "<target>",
        "questions": [{
            "id": str(seed.get("id", "q1")),
            "verb": str(seed.get("verb", "describe")),
            "target": targets[0] if len(targets) == 1 else "<target>",
            "property": str(seed.get("property", "level")),
        }],
    }}


def compile_temporal_question(
    raw: dict[str, Any], *, available_targets: Iterable[str],
    default_verb: str = "describe", default_horizon: int | None = None,
) -> TemporalQuestion:
    """Validate a structured question and bind it to one exact target.

    Natural-language interpretation belongs to the host model.  This boundary
    accepts only a small alias set whose meanings are invariant; it never
    guesses a target or silently changes the requested quantity.
    """
    targets = sorted({str(item) for item in available_targets})
    if not isinstance(raw, dict):
        raise GnomonError("INVALID_TEMPORAL_QUESTION",
                          "Each temporal question must be an object.",
                          {"repair_call": _repair(raw, targets)})
    verb = str(raw.get("verb", default_verb)).strip().lower()
    prop_raw = str(raw.get("property", "")).strip().lower()
    prop = _PROPERTY_ALIASES.get(prop_raw, prop_raw)
    target_raw = raw.get("target", "")
    scope, members, aggregation = "series", (), None
    if isinstance(target_raw, dict):
        scope = str(target_raw.get("kind", "")).strip().lower()
        members = tuple(str(item) for item in target_raw.get("members", []))
        aggregation = target_raw.get("aggregation")
        # Each aggregatable property has exactly one public cross-unit
        # meaning. Canonicalising an omitted name is semantic normalization,
        # not numerical inference: aggregate scope has already been selected.
        # Canonicalising the omitted name is not numerical inference; the
        # scope already selected that contract, and no competing aggregation
        # exists to choose between.
        if scope == "aggregate" and aggregation is None:
            aggregation = AGGREGATIONS.get(prop)
        target = members[0] if scope == "series" and len(members) == 1 else "*"
        if scope not in {"series", "pair", "each", "aggregate"}:
            failures = {"target.kind": {
                "received": scope, "allowed": ["series", "pair", "each", "aggregate"]}}
        else:
            failures = {}
        invalid_members = [item for item in members if item not in targets]
        if not members or invalid_members:
            failures["target.members"] = {"invalid": invalid_members,
                                           "allowed": targets}
        allowed_aggregation = AGGREGATIONS.get(prop)
        if scope == "aggregate" and aggregation != allowed_aggregation:
            failures["target.aggregation"] = {
                "allowed": ([allowed_aggregation] if allowed_aggregation else [])}
        if scope == "aggregate" and prop not in AGGREGATIONS:
            failures["target.kind"] = (
                "aggregate scope is defined only for properties with an "
                "auditable cross-series decision rule")
        if scope in {"aggregate", "pair"} and len(members) < 2:
            failures["target.members"] = f"{scope} scope requires at least two members"
        if scope == "pair" and len(members) != 2:
            failures["target.members"] = "pair scope requires exactly two members"
        if scope == "pair" and len(set(members)) != len(members):
            failures["target.members"] = "pair scope requires two distinct members"
        if scope == "pair" and prop != "dependence":
            failures["target.kind"] = (
                "pair scope is defined only for cross-series dependence")
        if prop == "dependence" and scope != "pair":
            failures["target.kind"] = (
                "predictive dependence requires pair scope and two members")
    else:
        target = str(target_raw).strip()
        failures: dict[str, Any] = {}
    if not target and len(targets) == 1:
        target = targets[0]
    if verb not in VERBS:
        failures["verb"] = {"received": verb, "allowed": sorted(VERBS)}
    if prop not in PROPERTIES:
        failures["property"] = {"received": prop_raw,
                                "allowed": sorted(PROPERTIES)}
    if scope == "series" and target not in targets:
        failures["target"] = {"received": target, "allowed": targets}
    measure = raw.get("measure")
    if measure is not None and str(measure) not in MEASURES:
        failures["measure"] = {"received": measure,
                               "allowed": sorted(MEASURES)}
    policy = str(raw.get("context_policy", "ignore"))
    if policy not in CONTEXT_POLICIES:
        failures["context_policy"] = {
            "received": policy, "allowed": sorted(CONTEXT_POLICIES)}
    horizon_raw = raw.get("horizon", default_horizon)
    horizon: int | None = None
    if horizon_raw is not None:
        if isinstance(horizon_raw, bool) or not isinstance(horizon_raw, int) \
                or horizon_raw < 1:
            failures["horizon"] = "must be a positive integer number of periods"
        else:
            horizon = horizon_raw
    comparison = raw.get("comparison")
    if comparison is not None and not isinstance(comparison, dict):
        failures["comparison"] = "must be an object"
    vocabulary = raw.get("answer_vocabulary")
    if vocabulary is not None and (not isinstance(vocabulary, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            or not key.strip() or not value.strip()
            for key, value in vocabulary.items())):
        failures["answer_vocabulary"] = (
            "must map non-empty canonical strings to non-empty display strings")
    if failures:
        raise GnomonError(
            "INVALID_TEMPORAL_QUESTION",
            "The temporal question is materially ambiguous or unsupported.",
            {"fields": failures, "repair_call": _repair(raw, targets)},
        )
    return TemporalQuestion(
        id=str(raw.get("id") or "q1"), verb=verb, target=target,
        property=prop, horizon=horizon, comparison=comparison,
        measure=str(measure) if measure is not None else None,
        context_policy=policy,
        scope=scope, members=members, aggregation=(
            str(aggregation) if aggregation is not None else None),
        answer_vocabulary=(dict(vocabulary)
                           if isinstance(vocabulary, dict) else None),
    )


def compile_temporal_questions(
    raw: Any, *, available_targets: Iterable[str], default_verb: str,
    default_horizon: int | None = None,
) -> list[TemporalQuestion]:
    if raw is None:
        return []
    # Some structured-output providers return a JSON-encoded array inside the
    # schema field. Accept that transport representation exactly once, then
    # apply the same deterministic validator. Never iterate over characters.
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise GnomonError(
                "INVALID_TEMPORAL_QUESTION",
                "questions was a string but not a JSON-encoded array.",
            ) from error
        raw = decoded
    if not isinstance(raw, list) or not raw:
        raise GnomonError("INVALID_TEMPORAL_QUESTION",
                          "questions must be a non-empty array.")
    compiled = [compile_temporal_question(
        item, available_targets=available_targets, default_verb=default_verb,
        default_horizon=default_horizon) for item in raw]
    ids = [item.id for item in compiled]
    if len(ids) != len(set(ids)):
        raise GnomonError("INVALID_TEMPORAL_QUESTION",
                          "Every temporal question id must be unique.")
    return compiled
