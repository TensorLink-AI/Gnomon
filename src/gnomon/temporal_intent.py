"""LLM-assisted temporal intent proposal with deterministic acceptance."""

from __future__ import annotations

import copy
import re
from typing import Any

from .llm import LLMAdapter
from .temporal_question import (
    AGGREGATIONS, TemporalQuestion, compile_temporal_question,
    compile_temporal_questions,
)

INTENT_COMPILER_VERSION = "0.6"
# A structured intent is tiny, but reasoning providers may spend substantially
# more tokens deciding it before emitting the tool call. Measured 700-token
# caps produced syntactically valid `compiled` envelopes with no questions.
INTENT_COMPILER_MAX_TOKENS = 10000


INTENT_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["compiled", "refused"]},
        "refusal_reason": {"type": "string"},
        "questions": {"type": "array", "maxItems": 8, "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "id": {"type": "string"},
                "verb": {"type": "string", "enum": [
                    "describe", "predict", "compare", "detect", "decide",
                    "test", "decompose", "regress"]},
                "property": {"type": "string", "enum": [
                    "level", "trend", "seasonality", "volatility", "regime",
                    "extreme", "disturbance", "dependence", "stationarity", "decomposition",
                    "regression"]},
                "target": {"oneOf": [
                    {"type": "string"},
                    {"type": "object", "additionalProperties": False,
                     "properties": {
                         "kind": {"type": "string", "enum": [
                             "pair", "each", "aggregate"]},
                         "members": {"type": "array", "items": {"type": "string"}},
                         "aggregation": {"type": "string", "enum": [
                             "median_normalized_scale_ratio",
                             "median_alignment"]}},
                     "required": ["kind", "members"]}]},
                "horizon": {"type": "integer", "minimum": 1},
                "measure": {"type": "string", "enum": [
                    "point", "slope", "period", "residual_scale",
                    "marginal_variability", "change", "maximum", "minimum",
                    "correlation"]},
                "context_policy": {"type": "string", "enum": [
                    "ignore", "measure", "scenario"]},
                "method": {"type": "string"},
                "period": {"type": "integer", "minimum": 2},
                "seasonal_period": {"type": "integer", "minimum": 2},
                "differencing": {"type": "integer", "minimum": 0,
                                  "maximum": 2},
                "explanatory_variables": {"type": "array",
                    "items": {"type": "string"}},
                "validation": {"type": "object"},
                "answer_vocabulary": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}},
            }, "required": ["id", "verb", "property", "target"]}},
    }, "required": ["status", "questions"],
}


def _question_segments(text: str) -> list[str]:
    """Return ordered question clauses without interpreting their answer."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    questions = [line for line in lines if "?" in line]
    return questions or [item.strip() + "?" for item in text.split("?")
                         if item.strip()]


_PROPERTY_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("stationarity", re.compile(r"\b(stationar|unit root|adf|kpss)\w*\b", re.I)),
    ("decomposition", re.compile(r"\b(decompos|stl)\w*\b", re.I)),
    ("regression", re.compile(r"\b(regress|exogenous|predictor|coefficient)\w*\b", re.I)),
    ("volatility", re.compile(
        r"\b(volatil\w*|variance|variab\w*|dispersion|nois\w*)\b", re.I)),
    ("seasonality", re.compile(r"\b(season|periodic|cycle|phase alignment)\w*\b", re.I)),
    ("trend", re.compile(
        r"\b(trend|slope|growth rate|decline rate|moving up|moving down)\w*\b",
        re.I)),
    ("regime", re.compile(r"\b(regime|structural break|change point|changepoint)\w*\b", re.I)),
    ("disturbance", re.compile(r"\b(outlier|anomal|spike|disturbance)\w*\b", re.I)),
    ("extreme", re.compile(r"\b(extreme|maximum|minimum|peak|tail risk)\w*\b", re.I)),
    ("dependence", re.compile(
        r"\b(correlat\w*|depend\w*|related|relationship between)\b", re.I)),
    ("level", re.compile(r"\b(median|mean|average|level|higher|lower)\w*\b", re.I)),
)


def _explicit_property(segment: str) -> str | None:
    """Return a property only when lexical evidence is unique.

    This is a semantic parser for standard statistical terms, not a fuzzy
    classifier. Ambiguous clauses remain under LLM interpretation.
    """
    matches = [prop for prop, pattern in _PROPERTY_CUES if pattern.search(segment)]
    # A seasonality question often contains the word correlation to describe
    # its alignment measure. The requested property remains seasonality.
    if "seasonality" in matches and "dependence" in matches:
        matches.remove("dependence")
    return matches[0] if len(set(matches)) == 1 else None


def _named_targets(segment: str, available_targets: list[str]) -> list[str]:
    lowered = segment.lower()
    positioned = []
    for target in available_targets:
        starts = [match.start() for alias in {
            target.lower(), target.lower().replace("_", " ")}
            for match in [re.search(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered)]
            if match]
        if starts:
            positioned.append((min(starts), target))
    return [target for _, target in sorted(positioned)]


def _canonical_measure(prop: str, segment: str) -> str | None:
    lowered = segment.lower()
    if prop == "seasonality" and "alignment" in lowered:
        return "change"
    if any(word in lowered for word in ("change", "higher", "lower", "future")):
        return "change"
    if prop == "level" and any(word in lowered for word in ("median", "mean", "average")):
        return "change"
    return None


def _explicit_horizon(segment: str) -> int | None:
    match = re.search(
        r"\b(?:next|for)\s+(\d+)\s+(?:periods?|steps?|points?)\b",
        segment, re.I)
    return int(match.group(1)) if match else None


def _proposal_has_unknown_target(question: dict[str, Any],
                                 available_targets: list[str]) -> bool:
    target = question.get("target")
    if isinstance(target, str):
        return target not in available_targets
    if isinstance(target, dict):
        members = target.get("members")
        return (not isinstance(members, list)
                or any(member not in available_targets for member in members))
    return False


def _recover_explicit_questions(
    text: str, available_targets: list[str], default_verb: str,
    default_horizon: int | None,
) -> list[dict[str, Any]]:
    """Recover only fully explicit intents from malformed model structure.

    This is deliberately narrower than the LLM compiler: every clause must
    contain a unique statistical property and an unambiguous target binding.
    It cannot choose among unnamed series or infer a semantic substitution.
    """
    segments = _question_segments(text)
    recovered: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        prop = _explicit_property(segment)
        named = _named_targets(segment, available_targets)
        if prop is None:
            return []
        if len(named) == 1:
            target: Any = named[0]
        elif len(named) == 2 and prop == "dependence":
            target = {"kind": "pair", "members": named}
        elif len(available_targets) == 1:
            target = available_targets[0]
        else:
            return []
        question: dict[str, Any] = {
            "id": f"q{index + 1}", "verb": default_verb,
            "property": prop, "target": target,
        }
        horizon = _explicit_horizon(segment)
        if horizon is not None:
            question["horizon"] = horizon
        elif default_horizon is not None:
            question["horizon"] = default_horizon
        measure = _canonical_measure(prop, segment)
        if measure:
            question["measure"] = measure
        recovered.append(question)
    return recovered


def _route_explicit_questions(
    text: str, questions: Any, available_targets: list[str],
    default_verb: str, default_horizon: int | None,
) -> Any:
    """Bind explicit property slots, then let arguments remain validated.

    Numbered/line-separated questions form an ordered contract. For clauses
    with an unambiguous statistical term, the model cannot change the
    property, omit the slot, or inherit a singular target into a clearly
    fleet-level volatility/seasonality question.
    """
    if not isinstance(questions, list):
        return questions
    segments = _question_segments(text)
    routed: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        prop = _explicit_property(segment)
        proposed = copy.deepcopy(questions[index]) \
            if index < len(questions) and isinstance(questions[index], dict) else {}
        if prop is None:
            if proposed:
                routed.append(proposed)
            continue
        # Do not launder an invented model target into a valid one. The normal
        # validator must retain and reject that safety failure.
        if _proposal_has_unknown_target(proposed, available_targets):
            routed.append(proposed)
            continue
        prior_property = proposed.get("property")
        proposed["id"] = str(proposed.get("id") or f"q{index + 1}")
        proposed["property"] = prop
        if prior_property and prior_property != prop:
            for key in ("measure", "method", "period", "seasonal_period", "differencing",
                        "explanatory_variables", "validation"):
                proposed.pop(key, None)
            proposed["verb"] = {
                "stationarity": "test", "decomposition": "decompose",
                "regression": "regress", "dependence": "compare",
            }.get(prop, default_verb)
        else:
            proposed["verb"] = str(proposed.get("verb") or default_verb)
        horizon = _explicit_horizon(segment)
        if horizon is not None:
            proposed["horizon"] = horizon
        elif default_horizon is not None and proposed.get("horizon") is None:
            proposed["horizon"] = default_horizon
        measure = _canonical_measure(prop, segment)
        if measure:
            proposed["measure"] = measure
        named = _named_targets(segment, available_targets)
        each_scope = bool(re.search(r"\b(each|every|separately)\b", segment, re.I))
        if each_scope:
            proposed["target"] = {"kind": "each", "members": list(
                named or available_targets)}
        elif len(named) == 1:
            proposed["target"] = named[0]
        elif len(named) >= 2 and prop == "dependence":
            proposed["target"] = {"kind": "pair", "members": named}
        elif len(named) >= 2 and prop in {"volatility", "seasonality"}:
            proposed["target"] = {
                "kind": "aggregate", "members": named,
                "aggregation": AGGREGATIONS[prop],
            }
        elif len(available_targets) == 1:
            proposed["target"] = available_targets[0]
        elif not named and prop in {"volatility", "seasonality"}:
            proposed["target"] = {
                "kind": "aggregate", "members": list(available_targets),
                "aggregation": AGGREGATIONS[prop],
            }
        routed.append(proposed)
    # If no clause was explicit, retain the model proposal unchanged.
    return routed if any(_explicit_property(item) for item in segments) else questions


def _resolve_discourse_focus(
    text: str, questions: Any, available_targets: list[str],
) -> Any:
    """Bind omitted targets to the nearest explicit series deterministically.

    The LLM proposes semantic structure, but it cannot silently broaden a
    singular discourse focus into a cross-series aggregate. Explicit
    collective language remains authoritative.
    """
    if not isinstance(questions, list):
        return questions
    segments = _question_segments(text)
    if len(segments) != len(questions):
        return questions
    aliases = {
        target: {target.lower(), target.lower().replace("_", " ")}
        for target in available_targets
    }
    collective = re.compile(
        r"\b(all|across|each|every|fleet|group|metrics|series|channels)\b",
        re.IGNORECASE)
    focus: str | None = None
    resolved = copy.deepcopy(questions)
    for segment, raw in zip(segments, resolved):
        lowered = segment.lower()
        named = [target for target, names in aliases.items()
                 if any(re.search(rf"\b{re.escape(name)}\b", lowered)
                        for name in names)]
        if len(named) == 1:
            focus = named[0]
            raw["target"] = focus
        elif not named and focus is not None and not collective.search(segment):
            raw["target"] = focus
    return resolved


def _normalize_nonsemantic_optionals(questions: Any) -> Any:
    """Drop optional fields that encode absence rather than meaning.

    Some structured-output models emit ``horizon: 0`` for a descriptive
    question about the present.  Zero is not a forecast horizon and the
    field is optional, so retaining it turns a correct intent into a
    validation failure.  Canonicalize that representation only for
    descriptive questions; predictive requests with invalid horizons must
    still fail rather than silently changing the requested window.
    """
    if not isinstance(questions, list):
        return questions
    normalized = copy.deepcopy(questions)
    for raw in normalized:
        if not isinstance(raw, dict) or raw.get("verb") != "describe":
            continue
        horizon = raw.get("horizon")
        if isinstance(horizon, int) and not isinstance(horizon, bool) and horizon <= 0:
            raw.pop("horizon", None)
    return normalized


def compile_temporal_text(
    text: str, *, available_targets: list[str], adapter: LLMAdapter,
    default_verb: str = "describe", default_horizon: int | None = None,
) -> list[TemporalQuestion]:
    """Propose intent from text, then pass it through the normal compiler.

    The model never sees data, forecast values, benchmark labels, or answer
    options. Its output has no numerical authority: the exact same validator
    used for explicit questions accepts or refuses it.
    """
    prompt = (
        "Compile the user's temporal request into typed questions. Do not "
        "answer it. Available targets: " + repr(available_targets) + ". "
        + (f"The host-resolved forecast horizon is {default_horizon} periods; "
           "when the request says 'forecast horizon' without another number, "
           "use that value. " if default_horizon is not None else "") +
        "Allowed properties: level, trend, seasonality, volatility, regime, "
        "extreme, dependence, stationarity, decomposition, regression. "
        "Use test/stationarity for ADF or KPSS, decompose/decomposition for "
        "a requested fixed-period decomposition, and regress/regression for "
        "a target with explicit explanatory_variables. Preserve an explicitly "
        "requested method and period exactly; never translate ADF into anomaly "
        "detection, STL into generic season discovery, or regression into a "
        "forecast. Target is an exact series name, or an object "
        "with kind pair/each/aggregate and explicit members. Dependence or "
        "correlation between two named series uses pair. The only cross-unit "
        "volatility aggregate is median_normalized_scale_ratio; a seasonality "
        "aggregate is median_alignment. A request about one "
        "named series always uses that series-name string as target. Aggregate "
        "means combining two or more different series; a median or comparison "
        "over time for one series is not aggregate. Use status=refused and "
        "an empty questions array when the property or target is materially "
        "ambiguous; never choose a target merely because it is available. "
        "History, future, and forecast horizon name time windows, never target "
        "groups. A singular collective request about a fleet, group, or all "
        "metrics means aggregate over all available targets unless a subset "
        "is named; "
        "words like each, every, or separately mean each. Return exactly one "
        "A question asking for one volatility-change answer across several "
        "series is aggregate with median_normalized_scale_ratio; use each "
        "only when the user asks for a separate answer per series. "
        "History, future, and forecast are comparison windows, never members "
        "of pair scope. "
        "Never invent answer_vocabulary; preserve it only when the user's "
        "request explicitly supplies a canonical-to-display mapping. Return "
        "exactly one question unless the user explicitly asks several "
        "distinct questions. In a numbered question list, a question that "
        "omits the target inherits the nearest explicitly named target only "
        "when that reference is unambiguous. Do not duplicate one request as both each and "
        "aggregate. Do not emit extra measures. User request:\n" + text
    )
    proposed = adapter.complete(prompt, INTENT_SCHEMA)
    if proposed.get("status") == "refused":
        from .contracts import GnomonError
        raise GnomonError(
            "INVALID_TEMPORAL_QUESTION",
            str(proposed.get("refusal_reason") or
                "The temporal request is materially ambiguous."),
            {"compiler_status": "refused"},
        )
    raw_questions = proposed.get("questions") or []
    if isinstance(raw_questions, str):
        import json
        try:
            decoded = json.loads(raw_questions)
            if isinstance(decoded, list):
                raw_questions = decoded
        except (TypeError, ValueError):
            pass
    routed_questions = _route_explicit_questions(
        text, raw_questions, available_targets,
        default_verb, default_horizon)
    if not isinstance(raw_questions, list):
        routed_questions = _recover_explicit_questions(
            text, available_targets, default_verb, default_horizon)
    if not routed_questions:
        from .contracts import GnomonError
        raise GnomonError(
            "INVALID_TEMPORAL_QUESTION",
            "The intent compiler returned compiled status without a question.",
            {"compiler_status": "malformed", "compiler_proposal": proposed},
        )
    try:
        proposed_questions = routed_questions
        proposed_questions = _resolve_discourse_focus(
            text, proposed_questions, available_targets)
        proposed_questions = _normalize_nonsemantic_optionals(
            proposed_questions)
        return compile_temporal_questions(
            proposed_questions, available_targets=available_targets,
            default_verb=default_verb, default_horizon=default_horizon)
    except Exception as error:
        # Preserve the proposal in the typed error receipt. It remains
        # powerless, but operators can distinguish model interpretation from
        # deterministic validation failure.
        if hasattr(error, "details"):
            error.details["compiler_proposal"] = proposed
        raise


def compile_temporal_text_receipt(
    text: str, *, available_targets: list[str], adapter: LLMAdapter,
    default_verb: str = "describe", default_horizon: int | None = None,
) -> dict[str, Any]:
    """Compile independently: one bad proposal cannot erase valid siblings."""
    # Reuse the canonical prompt/schema by capturing the proposal once.
    class Capture:
        def __init__(self):
            self.proposal: dict[str, Any] | None = None

        def complete(self, prompt: str, response_schema: dict[str, Any]):
            response = adapter.complete(prompt, response_schema)
            self.proposal = copy.deepcopy(response)
            return response

    capture = Capture()
    try:
        compiled = compile_temporal_text(
            text, available_targets=available_targets, adapter=capture,
            default_verb=default_verb, default_horizon=default_horizon)
        return {"proposed": capture.proposal, "accepted": compiled,
                "rejected": []}
    except Exception as whole_error:
        proposal = capture.proposal or {}
        accepted, rejected = [], []
        proposed_questions = proposal.get("questions") or []
        proposal_was_list = isinstance(proposed_questions, list)
        if isinstance(proposed_questions, str):
            import json
            try:
                proposed_questions = json.loads(proposed_questions)
            except (TypeError, ValueError):
                proposed_questions = []
        if proposal_was_list:
            proposed_questions = _route_explicit_questions(
                text, proposed_questions, available_targets,
                default_verb, default_horizon)
        proposed_questions = _resolve_discourse_focus(
            text, proposed_questions, available_targets)
        proposed_questions = _normalize_nonsemantic_optionals(
            proposed_questions)
        for index, raw in enumerate(proposed_questions):
            try:
                accepted.append(compile_temporal_question(
                    raw, available_targets=available_targets,
                    default_verb=default_verb,
                    default_horizon=default_horizon))
            except Exception as error:
                rejected.append({"index": index, "proposal": raw,
                                 "type": type(error).__name__,
                                 "message": str(error),
                                 "details": getattr(error, "details", {})})
        if not accepted and not rejected:
            rejected.append({"type": type(whole_error).__name__,
                             "message": str(whole_error),
                             "details": getattr(whole_error, "details", {})})
        return {"proposed": proposal, "accepted": accepted,
                "rejected": rejected}
