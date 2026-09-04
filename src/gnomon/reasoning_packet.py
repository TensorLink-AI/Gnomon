"""The evidence dossier: a reasoning packet the model concludes from.

The cross-model evaluation (docs/cross-model-evaluation-2026-08.md) found
that the compiled evidence packet improved behaviour but not discrimination:
it handed the model a governed conclusion rather than the evidence needed to
reason toward one.  This module builds the restructured packet that record
adopted — observations, relevant temporal properties, supporting and
conflicting evidence, the interpretations still compatible with the data,
evidence sufficiency, and what would distinguish the alternatives — and the
deterministic check Gnomon runs over the model's selection afterwards.

The packet organises receipts that already exist; it computes no new numbers
and cannot rewrite the canonical answer.  A ``supported`` canonical stays
binding only when the evidence requirements for the requested inference are
met; a supported observed measurement cannot become a binding forecast.
Otherwise the packet names the model as selector and
:func:`verify_packet_selection` is the gate its conclusion must pass: the
selected interpretation must be one the packet admits as compatible, and
every cited evidence source must exist and support it.
"""

from __future__ import annotations

from typing import Any

PACKET_VERSION = "0.2"

#: Directions that are support states or absences, not selectable answers.
_NON_ANSWERS = {None, "", "uncertain", "unknown", "abstained", "weak"}


def _interpretations(
    canonical: str | None, canonical_support: str,
    adjudication: dict[str, Any], vocabulary: dict[str, Any] | None,
    *, evidence_authoritative: bool = True,
) -> list[dict[str, Any]]:
    """Every interpretation the receipts mention, plus the caller's own
    vocabulary, each with its evidence for and against."""
    candidates = {
        str(row.get("value")): row
        for row in (adjudication.get("candidates") or [])
        if isinstance(row, dict) and row.get("value") not in _NON_ANSWERS
    }
    values = list(candidates)
    for label in (vocabulary or {}):
        if label not in candidates and label not in _NON_ANSWERS:
            values.append(str(label))
    if canonical not in _NON_ANSWERS and str(canonical) not in values:
        values.insert(0, str(canonical))

    supported_values = ({
        value for value, row in candidates.items()
        if str(row.get("support")) == "supported"
    } if evidence_authoritative else set())
    if (evidence_authoritative and canonical_support == "supported"
            and canonical not in _NON_ANSWERS):
        supported_values.add(str(canonical))
    rows: list[dict[str, Any]] = []
    for value in values:
        row = candidates.get(value, {})
        sources = [
            str(item.get("kind"))
            for item in (row.get("sources") or [])
            if isinstance(item, dict) and item.get("kind")
        ]
        against = sorted({
            kind
            for other, other_row in candidates.items()
            if other != value
            for kind in (
                str(item.get("kind"))
                for item in (other_row.get("sources") or [])
                if isinstance(item, dict) and item.get("kind")
                and str(item.get("support")) != "abstained"
            )
        })
        has_evidence = bool(sources) and str(
            row.get("support", "abstained")) != "abstained"
        # An interpretation is excluded only when supported evidence names a
        # different answer and nothing non-abstained names this one.  Absent
        # any supported evidence, every mentioned interpretation remains
        # live — that is what "underpowered" means.
        compatible = has_evidence or value == canonical or not supported_values
        rows.append({
            "value": value,
            "support": str(row.get("support", "abstained")),
            "evidence_weight": row.get("evidence_weight", 0.0),
            "supporting": sources,
            "conflicting": against,
            "compatible": bool(compatible),
            "conditional_only": bool(row.get("conditional_only")),
            "decision_eligible": bool(
                compatible and not row.get("conditional_only")),
        })
    return rows


def _sufficiency(canonical_support: str, interpretations: list[dict[str, Any]],
                 missing: list[str]) -> dict[str, Any]:
    """The admission vocabulary — sufficient / mixed / insufficient."""
    non_abstained = [row for row in interpretations
                     if row["support"] != "abstained"]
    if canonical_support == "supported" and not missing:
        level = "sufficient"
    elif non_abstained:
        level = "mixed"
    else:
        level = "insufficient"
    return {
        "level": level,
        "missing_evidence": list(missing)[:3],
        "compatible_interpretations": sum(
            row["compatible"] for row in interpretations),
        "meaning": (
            "the evidence distinguishes one conclusion" if level == "sufficient"
            else "the evidence narrows but does not settle the conclusion"
            if level == "mixed"
            else "the evidence does not distinguish any conclusion"),
    }


def build_reasoning_packet(
    result: dict[str, Any], *,
    mode: str,
    property: str,
    missing: list[str],
    adjudication: dict[str, Any],
    vocabulary: dict[str, Any] | None = None,
    discrimination: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the dossier from receipts the planner already computed."""
    best = result.get("best_estimate") or {}
    canonical = best.get("value")
    canonical_support = str(best.get("support") or "abstained")
    answer = result.get("answer") or {}
    # A supported measurement binds only the inference it actually earned.
    # For example, a supported observed trend is not a supported forecast of
    # the next window when rolling-origin predictive evidence is missing.
    # Keep the measurement and its support visible, but do not let that label
    # silently acquire authority across the observation/forecast boundary.
    binding = canonical_support == "supported" and not missing
    interpretations = _interpretations(
        canonical if canonical is None else str(canonical),
        canonical_support, adjudication, vocabulary,
        evidence_authoritative=not missing)
    observations = {
        "direction": answer.get("direction"),
        "estimate": answer.get("estimate"),
        "interval": answer.get("interval"),
    }
    temporal_properties: dict[str, Any] = {
        "property": property,
        "inference_mode": mode,
    }
    discriminators = list(dict.fromkeys(
        list(adjudication.get("what_would_flip") or []) + list(missing)))[:3]
    discriminating: dict[str, Any] | None = None
    if discrimination and discrimination.get("identifiable"):
        # The distinguishing computation actually ran: merge the measured
        # held-out fit weights onto the interpretations, and surface any
        # interpretation only the surrogates mention. Fit weights are
        # evidence over the surrogate set, never probabilities, and never
        # move the canonical answer.
        by_value = {str(row.get("value")): row
                    for row in (discrimination.get("hypotheses") or [])}
        best_fit = discrimination.get("best")
        separation = str(discrimination.get("separation") or "none")
        known = {row["value"] for row in interpretations}
        for value, hypothesis in by_value.items():
            weight = float(hypothesis.get("relative_weight") or 0.0)
            if value in known or weight <= 0.0 or value in _NON_ANSWERS:
                continue
            interpretations.append({
                "value": value,
                "support": "weak" if weight >= .6 else "abstained",
                "evidence_weight": 0.0,
                "supporting": ["held_out_hypothesis_fit"],
                "conflicting": [],
                "compatible": True,
                "conditional_only": False,
                # This is a selectable interpretation in the model lane,
                # not an authority upgrade. A binding canonical is still
                # enforced independently by verify_packet_selection.
                "decision_eligible": True,
            })
        for row in interpretations:
            hypothesis = by_value.get(row["value"])
            if hypothesis is not None:
                row["held_out_fit"] = hypothesis.get("relative_weight")
                if (row["value"] == best_fit and separation != "none"
                        and "held_out_hypothesis_fit" not in row["supporting"]):
                    row["supporting"] = [*row["supporting"],
                                         "held_out_hypothesis_fit"]
        discriminating = {
            "kind": "held_out_hypothesis_fit",
            "best": best_fit,
            "separation": separation,
            "holdout_steps": discrimination.get("holdout_steps"),
            "weights_are_fit_evidence_not_probabilities": True,
        }
    elif discrimination is not None:
        discriminating = {
            "kind": "held_out_hypothesis_fit",
            "ran": False,
            "reason": discrimination.get("reason"),
        }
    return {
        "version": PACKET_VERSION,
        "observations": observations,
        "temporal_properties": temporal_properties,
        # The raw evidence rows live once, in the plan beside this packet;
        # each interpretation's supporting/conflicting kinds index them.
        "interpretations": interpretations,
        "evidence_sufficiency": {
            **_sufficiency(canonical_support, interpretations, missing),
            **({"separation": discriminating["separation"]}
               if discriminating and "separation" in discriminating else {}),
        },
        **({"discriminating_evidence": discriminating}
           if discriminating else {}),
        "discriminators": discriminators,
        "selection_contract": {
            "selector": "gnomon_canonical" if binding else "model",
            "canonical": {"value": canonical, "support": canonical_support,
                          "role": "binding" if binding else
                          "default_not_command"},
            "inference_authority": {
                "mode": mode,
                "requirements_satisfied": not missing,
                "missing_evidence": list(missing)[:3],
            },
            "verification": "verify_packet_selection",
            "selection_must_cite_evidence": not binding,
            "primary_forecast_unchanged": True,
        },
    }


def verify_packet_selection(
    packet: dict[str, Any], selection: dict[str, Any],
) -> list[dict[str, Any]]:
    """Gnomon's check that a model conclusion follows from the packet.

    ``selection`` is ``{"value": ..., "cited_evidence": [kinds...]}``.  The
    verdict is a list of typed violations, empty when the selection stands.
    Deterministic and judgement-free, in the style of :mod:`gnomon.verifier`:
    it rejects selections the supplied evidence cannot carry, never
    selections it merely disagrees with.
    """
    violations: list[dict[str, Any]] = []
    value = str(selection.get("value") or "").strip()
    contract = packet.get("selection_contract") or {}
    canonical = contract.get("canonical") or {}
    interpretations = {
        str(row.get("value")).strip().lower(): row
        for row in (packet.get("interpretations") or [])
        if isinstance(row, dict)
    }
    if not value:
        return [{"code": "SELECTION_MISSING",
                 "message": "The selection names no interpretation."}]
    row = interpretations.get(value.lower())
    canonical_value = ("" if canonical.get("value") is None
                       else str(canonical.get("value")).strip().lower())
    if (canonical.get("role") == "binding"
            and value.lower() != canonical_value):
        violations.append({
            "code": "SELECTION_OVERRIDES_BINDING",
            "message": (
                f"The canonical answer {canonical.get('value')!r} is "
                f"supported and binding; the packet does not authorise "
                f"selecting {value!r} instead."),
        })
    if row is None:
        violations.append({
            "code": "SELECTION_NOT_IN_PACKET",
            "message": (
                f"{value!r} is not an interpretation this packet presents; "
                f"the packet's interpretations are "
                f"{sorted(interpretations)}."),
        })
        return violations
    if not row.get("compatible"):
        violations.append({
            "code": "SELECTION_INCOMPATIBLE",
            "message": (
                f"The packet's evidence excludes {value!r}: supported "
                f"evidence names a different interpretation and nothing "
                f"non-abstained names this one."),
        })
    if value.lower() != canonical_value and not row.get("decision_eligible"):
        violations.append({
            "code": "SELECTION_NOT_DECISION_ELIGIBLE",
            "message": (
                f"The evidence packet presents {value!r} for comparison but "
                "does not authorize it as a decision; conditional or "
                "under-supported alternatives remain advisory evidence."),
        })
    if contract.get("selection_must_cite_evidence"):
        cited = [str(item) for item in
                 (selection.get("cited_evidence") or [])]
        if not cited:
            violations.append({
                "code": "SELECTION_UNCITED",
                "message": (
                    "A non-binding selection must cite the packet evidence "
                    "it rests on."),
            })
        known = {str(item.get("kind")) for item in
                 (packet.get("evidence") or []) if isinstance(item, dict)}
        known.update(kind for entry in interpretations.values()
                     for kind in (entry.get("supporting") or []))
        supporting = set(row.get("supporting") or [])
        for kind in cited:
            if kind not in known:
                violations.append({
                    "code": "SELECTION_EVIDENCE_MISSING",
                    "message": (
                        f"The selection cites {kind!r}, which this packet "
                        f"does not contain."),
                })
            elif kind not in supporting:
                violations.append({
                    "code": "SELECTION_EVIDENCE_CONTRARY",
                    "message": (
                        f"The cited evidence {kind!r} does not support "
                        f"{value!r} in this packet."),
                })
    return violations


#: A rejected selection gets exactly one repair round. The second failure
#: is terminal: the host publishes the canonical default, labelled — a
#: model that cannot ground its conclusion twice does not get a third
#: attempt at wearing down the verifier.
MAX_REPAIR_ROUNDS = 1


def repair_selection(packet: dict[str, Any],
                     selection: dict[str, Any]) -> dict[str, Any]:
    """Verify a selection and, when it fails, build the one repair turn.

    Returns ``{"accepted": True, "violations": []}`` for a grounded
    selection.  Otherwise the ``repair`` block is a complete, deterministic
    instruction a host can hand back to the model verbatim: the allowed
    interpretations, the evidence kinds each may cite, and the canonical
    default to fall back to.  Rejection converts into accuracy only when
    the model learns *why*; a bare retry re-samples the same mistake.
    """
    violations = verify_packet_selection(packet, selection)
    if not violations:
        return {"accepted": True, "violations": []}
    contract = packet.get("selection_contract") or {}
    canonical = contract.get("canonical") or {}
    compatible = [row for row in (packet.get("interpretations") or [])
                  if isinstance(row, dict) and row.get("compatible")]
    codes = {str(item.get("code")) for item in violations}
    if "SELECTION_OVERRIDES_BINDING" in codes:
        instruction = (
            f"The canonical answer {canonical.get('value')!r} is supported "
            f"and binding; return it unchanged."
        )
    elif contract.get("selection_must_cite_evidence"):
        instruction = (
            "Select one interpretation from allowed_values and cite only "
            "evidence kinds listed for it in citable_evidence. If none "
            "matches your reasoning, return the canonical default."
        )
    else:
        instruction = (
            "Select one interpretation from allowed_values, or return the "
            "canonical default."
        )
    return {
        "accepted": False,
        "violations": violations,
        "repair": {
            "rounds": MAX_REPAIR_ROUNDS,
            "instruction": instruction,
            "allowed_values": [row["value"] for row in compatible],
            "citable_evidence": {
                row["value"]: list(row.get("supporting") or [])
                for row in compatible
            },
            "canonical_default": {"value": canonical.get("value"),
                                  "support": canonical.get("support")},
            "after_failed_repair": "publish_canonical_default_labelled",
        },
    }
