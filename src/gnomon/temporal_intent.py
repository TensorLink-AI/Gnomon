"""LLM-assisted temporal intent proposal with deterministic acceptance."""

from __future__ import annotations

import copy
import re
from typing import Any

from .llm import LLMAdapter
from .temporal_question import (
    TemporalQuestion, compile_temporal_question, compile_temporal_questions,
)

INTENT_COMPILER_VERSION = "0.5"


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
                    "describe", "predict", "compare", "detect", "decide"]},
                "property": {"type": "string", "enum": [
                    "level", "trend", "seasonality", "volatility", "regime",
                    "extreme", "dependence"]},
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
        "extreme, dependence. Target is an exact series name, or an object "
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
    try:
        proposed_questions = _resolve_discourse_focus(
            text, proposed.get("questions"), available_targets)
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
        if isinstance(proposed_questions, str):
            import json
            try:
                proposed_questions = json.loads(proposed_questions)
            except (TypeError, ValueError):
                proposed_questions = []
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
