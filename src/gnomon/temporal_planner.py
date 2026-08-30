"""Deterministic evidence planning for typed temporal questions.

The planner decides what kind of inference a question requests and whether the
execution result contains the evidence needed for that inference.  It does not
choose a numerical answer: the fitted executable remains the sole authority.
"""

from __future__ import annotations

from typing import Any

from .temporal_question import TemporalQuestion
from .temporal_adjudication import adjudicate_temporal_evidence


PLANNER_VERSION = "0.4"

_RECOVERY = {
    "rolling_origin_scale_fit": "collect more history or request a shorter horizon",
    "rolling_origin_property_fit": "collect more history or request a shorter horizon",
    "future_phase_evidence": "use a phase-aware predictive model or provide a timed causal hypothesis",
    "positive_transition_evidence": "provide a leading signal or a repeated historical transition",
    "tail_calibration": "collect more tail events or shorten the evaluation horizon",
    "paired_rolling_origin_fit": "collect longer aligned histories for both series",
    "conditional_effect_receipt": "provide prior occurrences with known-at times and outcomes",
    "observed_cycles": "collect at least two complete observed cycles",
}

_PREDICTIVE_REQUIREMENTS = {
    "level": ("immutable_forecast_projection",),
    "trend": ("rolling_origin_property_fit",),
    "seasonality": ("future_phase_evidence",),
    "volatility": ("rolling_origin_scale_fit",),
    "regime": ("positive_transition_evidence",),
    "extreme": ("tail_calibration",),
    "disturbance": ("observed_disturbance_cannot_be_predicted",),
    "dependence": ("paired_rolling_origin_fit",),
}

_OBSERVED_REQUIREMENTS = {
    "level": ("two_observed_windows",),
    "trend": ("two_observed_windows",),
    "seasonality": ("observed_cycles",),
    "volatility": ("detrended_observed_windows",),
    "regime": ("observed_changepoint_evidence",),
    "extreme": ("observed_tail_evidence",),
    "disturbance": ("observed_anomaly_and_regime_evidence",),
    "dependence": ("paired_observations",),
}


def inference_mode(question: TemporalQuestion) -> str:
    """Classify requested authority independently of answer contents."""
    if question.context_policy == "scenario":
        return "conditional"
    if question.verb in {"predict", "decide"} or question.horizon is not None:
        return "predictive"
    if question.verb in {"describe", "detect"}:
        return "observed"
    comparison = question.comparison or {}
    names = " ".join(str(value).lower() for value in comparison.values())
    return "predictive" if any(word in names for word in
                               ("future", "forecast", "horizon")) else "observed"


def _evidence_rows(result: dict[str, Any],
                   observed_evidence: dict[str, Any] | None = None,
                   ) -> list[dict[str, Any]]:
    answer = result.get("answer") or {}
    executable = answer.get("executable") or {}
    kind = str(executable.get("kind") or (
        "fitted_volatility" if answer.get("property_distribution") else
        "fitted_executable"))
    rows: list[dict[str, Any]] = []
    if executable:
        rows.append({
            "kind": kind,
            "direction": answer.get("direction"),
            "support": answer.get("support"),
            "folds": ((answer.get("property_distribution") or {}).get("folds")
                      or (result.get("calibration") or {}).get("folds")),
        })
    if observed_evidence:
        estimate = observed_evidence.get("estimate")
        if isinstance(estimate, dict):
            estimate = {key: estimate[key] for key in (
                "phase_shift_steps", "strength_ratio", "template_correlation")
                        if key in estimate}
        rows.append({
            "kind": "observed_transition",
            "direction": observed_evidence.get("direction"),
            "support": observed_evidence.get("support"),
            "identifiable": observed_evidence.get("identifiable"),
            "estimate": estimate,
            "window_steps": (observed_evidence.get("diagnostics") or {}).get(
                "window_steps"),
        })
    panel = result.get("observed_panel_evidence") or {}
    if panel:
        rows.append({
            "kind": "cross_series_observed_evidence",
            "direction": panel.get("direction"),
            "support": panel.get("support"),
            "effective_series": panel.get("effective_series"),
            "agreement": panel.get("agreement"),
            "predictive_assumption": "recent_regime_persists",
        })
    return rows[:3]


def _available(requirement: str, result: dict[str, Any],
               evidence: list[dict[str, Any]]) -> bool:
    answer = result.get("answer") or {}
    executable = answer.get("executable") or {}
    kind = str(executable.get("kind", ""))
    folds = ((answer.get("property_distribution") or {}).get("folds")
             or (result.get("calibration") or {}).get("folds") or 0)
    if requirement == "immutable_forecast_projection":
        return kind == "published_forecast_projection"
    if requirement in {"rolling_origin_property_fit", "tail_calibration",
                       "positive_transition_evidence"}:
        return kind == "fitted_temporal_property" and int(folds) > 0
    if requirement == "rolling_origin_scale_fit":
        return bool(answer.get("property_distribution")) and int(folds) > 0
    if requirement == "paired_rolling_origin_fit":
        return kind == "fitted_dependence" and int(folds) > 0
    if requirement == "future_phase_evidence":
        return (kind == "published_forecast_projection"
                and answer.get("direction") not in {None, "uncertain"})
    if requirement == "conditional_effect_receipt":
        return bool(result.get("conditional_effect"))
    # Observed descriptive executables expose a computed estimate. Aggregate
    # panel evidence is an explicit observed receipt.
    return answer.get("estimate") is not None or any(
        row["kind"] == "cross_series_observed_evidence" for row in evidence)


def build_evidence_plan(question: TemporalQuestion,
                        result: dict[str, Any], *,
                        observed_evidence: dict[str, Any] | None = None,
                        analogues: dict[str, Any] | None = None,
                        discrimination: dict[str, Any] | None = None,
                        ) -> dict[str, Any]:
    """Return a bounded reasoning pack that cannot override the answer."""
    mode = inference_mode(question)
    requirements = list(
        _OBSERVED_REQUIREMENTS[question.property] if mode == "observed"
        else _PREDICTIVE_REQUIREMENTS[question.property])
    if mode == "conditional":
        requirements.append("conditional_effect_receipt")
    evidence = _evidence_rows(result, observed_evidence)
    missing = [item for item in requirements
               if not _available(item, result, evidence)]
    canonical = (result.get("best_estimate") or {}).get("value")
    contradictions: list[dict[str, Any]] = []
    equivalents = {
        "seasonality": {"stable": "continued", "phase_shifted": "shifting"},
    }
    for row in evidence:
        direction = row.get("direction")
        comparable = equivalents.get(question.property, {}).get(
            str(direction), direction)
        if (row["kind"] in {"cross_series_observed_evidence",
                            "observed_transition"}
                and mode in {"predictive", "conditional"}
                and canonical not in {None, "uncertain"}
                and comparable not in {None, "uncertain", canonical}):
            contradictions.append({
                "between": ["canonical_predictive_answer", row["kind"]],
                "canonical": canonical, "observed": direction,
                "resolution": "retain canonical answer; disclose observed disagreement",
            })
    identifiable = not missing and canonical not in {None, "uncertain"}
    plan = {
        "version": PLANNER_VERSION,
        "mode": mode,
        "identifiable": identifiable,
        # The canonical value and support already sit one level above this
        # pack. Repeating them here costs every subsequent agent turn and
        # creates a second field that could drift.
        "authority": "fitted_executable",
        "llm_role": "explain_and_qualify_only",
        "primary_forecast_unchanged": True,
    }
    if evidence:
        plan["evidence"] = evidence
    if missing:
        plan["missing_evidence"] = missing[:3]
        plan["recovery"] = list(dict.fromkeys(
            _RECOVERY[item] for item in missing if item in _RECOVERY))[:2]
    else:
        plan["basis"] = requirements[:3]
    if contradictions:
        plan["contradictions"] = contradictions[:2]
    # A compact contrastive projection gives the LLM the structure humans use
    # to reason: what supports the answer, what pushes against it, and what is
    # still unknown.  These are references to computed receipts, never new
    # numerical claims authored by the model.
    because = [{"evidence": row["kind"],
                "direction": row.get("direction"),
                "support": row.get("support")}
               for row in evidence
               if row.get("direction") in {canonical, None}
               and row.get("support") != "abstained"]
    against = [{"evidence": item["between"][1],
                "direction": item["observed"]}
               for item in contradictions]
    unknown = list(missing)
    if analogues and analogues.get("available"):
        analogue_direction = analogues.get("consensus_direction")
        analogue_row = {
            "evidence": "historical_analogues",
            "direction": analogue_direction,
            "agreement": analogues.get("agreement"),
        }
        if analogue_direction == canonical:
            because.append(analogue_row)
        elif analogue_direction not in {None, "uncertain"}:
            against.append(analogue_row)
        else:
            unknown.append("historical_analogue_outcomes_disagree")
        plan["historical_analogues"] = {
            key: analogues[key] for key in (
                "version", "window_steps", "consensus_direction",
                "agreement", "matches") if key in analogues
        }
    elif analogues and analogues.get("reason"):
        unknown.append(str(analogues["reason"]))
    plan["contrast"] = {
        "because": because[:3], "against": against[:2],
        "unknown": list(dict.fromkeys(unknown))[:3],
    }
    suggestions = list(plan.get("recovery") or [])
    if contradictions:
        suggestions.append(
            "wait for the next window and re-evaluate the observed/predictive disagreement")
    if analogues and not analogues.get("available"):
        suggestions.append("collect enough history for non-overlapping analogue outcomes")
    plan["suggested_next"] = list(dict.fromkeys(suggestions))[:2]
    plan["adjudication"] = adjudicate_temporal_evidence(
        result, evidence=evidence, analogues=analogues,
        conditional_effect=result.get("conditional_effect"), missing=missing)
    # The evidence dossier: the same receipts reorganised so the model can
    # reason toward a conclusion instead of paraphrasing one — observations,
    # temporal properties, evidence for and against every interpretation
    # still compatible with the data, sufficiency, and what would
    # distinguish the alternatives. A supported canonical stays binding;
    # anywhere weaker the model selects and verify_packet_selection is the
    # gate its conclusion must pass.
    if discrimination is not None:
        # The measured held-out discrimination between competing surrogates
        # (gnomon.discrimination) — evidence the model reasons from, kept in
        # the receipt in full; the packet carries the merged weights.
        plan["discrimination"] = discrimination
    from .reasoning_packet import build_reasoning_packet
    plan["packet"] = build_reasoning_packet(
        result, mode=mode, property=question.property, missing=missing,
        adjudication=plan["adjudication"],
        vocabulary=question.answer_vocabulary,
        discrimination=discrimination,
    )
    return plan


def compact_evidence_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Project a full reasoning receipt into the bounded agent contract.

    The receipt retains diagnostics, executable requirements, analogue
    matches and contradiction provenance.  The inline projection only tells
    an agent what supports the answer, what pushes against it, what remains
    unknown, and the single best next step.  It deliberately contains no
    second copy of the canonical answer and therefore cannot rewrite it.
    """
    contrast = plan.get("contrast") or {}
    projected = {
        "version": plan.get("version", PLANNER_VERSION),
        "authority": "fitted_executable",
        "primary_forecast_unchanged": True,
        "because": list(contrast.get("because") or [])[:1],
        "against": list(contrast.get("against") or [])[:1],
        "unknown": list(contrast.get("unknown") or [])[:2],
        "next": list(plan.get("suggested_next") or [])[:1],
        "details_in_answer_receipt": True,
    }
    packet = plan.get("packet") or {}
    if packet:
        # The dossier's bounded wire form: which interpretations remain
        # live, how far the evidence goes, and the single best
        # discriminator. Values only — the full packet stays in the receipt.
        projected["packet"] = {
            "interpretations": [
                {"value": row.get("value"), "support": row.get("support"),
                 "compatible": bool(row.get("compatible")),
                 "decision_eligible": bool(row.get("decision_eligible")),
                 "supporting": list(row.get("supporting") or [])[:4],
                 "conflicting": list(row.get("conflicting") or [])[:4],
                 **({"conditional_only": True}
                    if row.get("conditional_only") else {})}
                for row in (packet.get("interpretations") or [])[:4]
            ],
            "sufficiency": (packet.get("evidence_sufficiency") or {}).get(
                "level"),
            **({"separation": (packet.get("evidence_sufficiency") or {}).get(
                "separation")}
               if "separation" in (packet.get("evidence_sufficiency") or {})
               else {}),
            "discriminator": next(
                iter(packet.get("discriminators") or []), None),
            "selector": (packet.get("selection_contract") or {}).get(
                "selector"),
            "selection_must_cite_evidence": (
                packet.get("selection_contract") or {}).get(
                    "selection_must_cite_evidence"),
        }
    adjudication = plan.get("adjudication") or {}
    eligibility = adjudication.get("synthesis_eligibility") or {}
    alternative = adjudication.get("alternative") or {}
    ranked = list(adjudication.get("candidates") or [])[:3]
    projected["adjudication"] = {
        "relationship": adjudication.get("relationship", "unresolved"),
        "alternative": ({
            "value": alternative.get("value"),
            "support": alternative.get("support"),
        } if alternative else None),
        "synthesis_eligible": bool(eligibility.get("eligible")),
        "what_would_flip": list(adjudication.get("what_would_flip") or [])[:1],
        "ranked_hypotheses": [{
            "value": row.get("value"),
            "support": row.get("support"),
            "evidence_weight": row.get("evidence_weight"),
            "conditional_only": bool(row.get("conditional_only")),
        } for row in ranked],
        "weight_meaning": "receipt evidence weight, not probability",
    }
    return projected
