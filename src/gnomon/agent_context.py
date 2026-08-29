"""Provider-neutral helpers for governed model-authored context priors.

Gnomon does not call an LLM. Host integrations may use these helpers to ask
their own model for independent point paths, then deterministically validate,
aggregate, and seal those paths through :mod:`gnomon.llm_dossier`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from datetime import datetime
from typing import Any


SAMPLED_PRIOR_MIN_PATHS = 3
SAMPLED_PRIOR_MIN_VALID_FRACTION = 0.75
SAMPLED_PRIOR_MIN_DIRECTION_AGREEMENT = 0.60
SAMPLED_PRIOR_MAX_PAIRWISE_TO_POINTWISE_RATIO = 2.0
SAMPLED_PRIOR_MAX_ANCHORS = 32


def _seal(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def seal_temporal_decision_prior(
    answer: dict[str, Any], *, question_sha256: str,
    proposer_id: str, model: str,
) -> dict[str, Any]:
    """Validate and host-seal a decision captured before Gnomon evidence.

    This receipt preserves an LLM's independent temporal prior so a later
    evidence call can reconcile rather than anchor on the first authoritative-
    looking number it sees. It is never historical evidence, never modifies
    the primary forecast, and never grants automation authority. The host owns
    the capture-order assertion; a model cannot self-assert it inside
    ``answer``.
    """
    if (not isinstance(question_sha256, str) or len(question_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in question_sha256)):
        raise ValueError("question_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(proposer_id, str) or not proposer_id.strip():
        raise ValueError("proposer_id is required")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model is required")
    prediction = answer.get("breach_expected")
    action = str(answer.get("action") or "").strip().lower()
    if not isinstance(prediction, bool):
        raise ValueError("breach_expected must be boolean")
    if action not in {"act", "monitor"}:
        raise ValueError("action must be act or monitor")
    probability = answer.get("breach_probability")
    if probability is not None:
        if (isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(float(probability))
                or not 0 <= float(probability) <= 1):
            raise ValueError("breach_probability must be in [0, 1] or null")
        probability = float(probability)
    step = answer.get("first_breach_step")
    if step is not None and (isinstance(step, bool)
                             or not isinstance(step, int) or step < 1):
        raise ValueError("first_breach_step must be a positive integer or null")
    body = {
        "schema_version": "0.1",
        "kind": "temporal_decision_prior",
        "question_sha256": question_sha256,
        "proposer": {"proposer_id": proposer_id, "model": model},
        "prediction": "breach" if prediction else "no_breach",
        "breach_probability": probability,
        "first_breach_step": step,
        "action": action,
        "capture": {
            "phase": "before_gnomon_evidence",
            "host_attested": True,
        },
        "support": "prior_assisted",
        "automation_eligible": False,
    }
    return {**body, "seal_sha256": _seal(body)}


def verify_temporal_decision_prior(receipt: dict[str, Any]) -> bool:
    if not isinstance(receipt, dict) or not receipt.get("seal_sha256"):
        return False
    body = {key: value for key, value in receipt.items()
            if key != "seal_sha256"}
    return (
        receipt.get("kind") == "temporal_decision_prior"
        and receipt.get("support") == "prior_assisted"
        and receipt.get("automation_eligible") is False
        and (receipt.get("capture") or {}).get("phase") ==
        "before_gnomon_evidence"
        and (receipt.get("capture") or {}).get("host_attested") is True
        and receipt["seal_sha256"] == _seal(body)
    )


def build_temporal_decision_reconciliation(
    primary_packet: dict[str, Any], prior_receipt: dict[str, Any],
    *, question_sha256: str,
    proposer_skill: dict[str, Any] | None = None,
    decision_cutoff: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic contrast between immutable evidence and prior."""
    if not verify_temporal_decision_prior(prior_receipt):
        raise ValueError("temporal decision prior seal is invalid")
    if prior_receipt.get("question_sha256") != question_sha256:
        raise ValueError("temporal decision prior belongs to another question")
    skill = None
    if proposer_skill is not None:
        proposer = prior_receipt.get("proposer") or {}
        if proposer_skill.get("proposer_id") != proposer.get("proposer_id"):
            raise ValueError("proposer skill belongs to another proposer")
        if (proposer_skill.get("rule")
                != "paired_categorical_sign_test_and_shrunk_net_v1"
                or not isinstance(proposer_skill.get("resolved"), int)
                or proposer_skill.get("resolved", 0) < 1
                or proposer_skill.get("support_upgrade_allowed") is not False
                or proposer_skill.get("automation_upgrade_allowed") is not False):
            raise ValueError("proposer skill schema is invalid")
        known_at = proposer_skill.get("known_at")
        if not isinstance(known_at, str) or not decision_cutoff:
            raise ValueError(
                "proposer skill requires known_at and decision_cutoff")
        try:
            known = datetime.fromisoformat(known_at.replace("Z", "+00:00"))
            cutoff = datetime.fromisoformat(
                decision_cutoff.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("proposer skill timing is invalid") from error
        if known.tzinfo is None or cutoff.tzinfo is None or known > cutoff:
            raise ValueError("proposer skill was not known by the decision cutoff")
        skill = dict(proposer_skill)
    analysis = primary_packet.get("threshold_analysis") or {}
    event = analysis.get("horizon_event") or {}
    decision = primary_packet.get("governed_decision") or {}
    probability = event.get("probability_any_breach")
    engine_prediction = (
        "breach" if probability is not None and float(probability) >= .5
        else "no_breach" if probability is not None else "indeterminate")
    engine_action = (decision.get("recommended_action")
                     or decision.get("advisory_action"))
    body = {
        "schema_version": "0.1",
        "kind": "temporal_decision_reconciliation",
        "question_sha256": question_sha256,
        "primary_packet_sha256": _seal(primary_packet),
        "immutable_primary": {
            "support": primary_packet.get("support"),
            "event_probability": probability,
            "prediction": engine_prediction,
            "action_reference": engine_action,
            "human_action_authority": decision.get(
                "human_action_authority", "unavailable"),
            "automation_eligible": decision.get(
                "automation_eligible") is True,
        },
        "independent_prior": prior_receipt,
        "proposer_skill": skill,
        "conflict": {
            "prediction": prior_receipt.get("prediction") != engine_prediction,
            "action": (engine_action is not None
                       and prior_receipt.get("action") != engine_action),
            "probability_delta": (
                round(float(prior_receipt["breach_probability"])
                      - float(probability), 6)
                if prior_receipt.get("breach_probability") is not None
                and probability is not None else None),
        },
        "selection_policy": {
            "human_may_select": True,
            "must_cite_selected_source": True,
            "must_state_counterevidence": True,
            "must_state_what_would_change": True,
            "may_edit_numeric_inputs": False,
            "may_upgrade_support": False,
            "automation_eligible": False,
            "prior_has_outcome_skill": bool(
                skill and skill.get("graduated_for_human_prior") is True),
        },
        "primary_forecast_unchanged": True,
    }
    return {**body, "seal_sha256": _seal(body)}


def verify_temporal_decision_reconciliation(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or not payload.get("seal_sha256"):
        return False
    body = {key: value for key, value in payload.items()
            if key != "seal_sha256"}
    return (
        payload.get("kind") == "temporal_decision_reconciliation"
        and payload.get("primary_forecast_unchanged") is True
        and (payload.get("selection_policy") or {}).get(
            "automation_eligible") is False
        and payload["seal_sha256"] == _seal(body)
    )


def seal_temporal_decision_selection(
    reconciliation: dict[str, Any], selection: dict[str, Any],
) -> dict[str, Any]:
    """Validate and seal the human-facing choice over two visible sources."""
    if not verify_temporal_decision_reconciliation(reconciliation):
        raise ValueError("temporal decision reconciliation seal is invalid")
    source = str(selection.get("selected_source") or "").strip().lower()
    counter = str(selection.get("counterevidence_source") or "").strip().lower()
    allowed = {"independent_prior", "immutable_primary", "synthesis"}
    if source not in allowed:
        raise ValueError("selected_source is invalid")
    conflict = reconciliation.get("conflict") or {}
    has_conflict = bool(conflict.get("prediction") or conflict.get("action"))
    if has_conflict and (counter not in allowed or counter == source):
        raise ValueError(
            "a conflicting selection requires a distinct counterevidence_source")
    action = str(selection.get("action") or "").strip().lower()
    if action not in {"act", "monitor"}:
        raise ValueError("selection action must be act or monitor")
    confidence = str(selection.get("confidence") or "").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        raise ValueError("selection confidence must be low, medium, or high")
    what_changes = " ".join(str(
        selection.get("what_would_change") or "").split())
    if not what_changes or len(what_changes) > 300:
        raise ValueError("what_would_change must contain 1-300 characters")
    automation = str(selection.get("automation_action") or "").strip().lower()
    if automation != "withhold":
        raise ValueError("reconciled decisions cannot authorize automation")
    body = {
        "schema_version": "0.1",
        "kind": "temporal_decision_selection",
        "reconciliation_seal_sha256": reconciliation["seal_sha256"],
        "selected_source": source,
        "counterevidence_source": counter if has_conflict else counter or None,
        "action": action,
        "confidence": confidence,
        "what_would_change": what_changes,
        "automation_action": "withhold",
        "support": "prior_assisted",
        "automation_eligible": False,
        "primary_forecast_unchanged": True,
    }
    return {**body, "seal_sha256": _seal(body)}


def verify_temporal_decision_selection(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or not payload.get("seal_sha256"):
        return False
    body = {key: value for key, value in payload.items()
            if key != "seal_sha256"}
    return (
        payload.get("kind") == "temporal_decision_selection"
        and payload.get("primary_forecast_unchanged") is True
        and payload.get("automation_eligible") is False
        and payload["seal_sha256"] == _seal(body)
    )


def decision_selection_synthesis_payload(
    reconciliation: dict[str, Any], selection: dict[str, Any],
) -> dict[str, Any]:
    """Project a sealed selection into the existing outcome ledger schema."""
    if not verify_temporal_decision_reconciliation(reconciliation):
        raise ValueError("temporal decision reconciliation seal is invalid")
    if not verify_temporal_decision_selection(selection):
        raise ValueError("temporal decision selection seal is invalid")
    if (selection.get("reconciliation_seal_sha256")
            != reconciliation.get("seal_sha256")):
        raise ValueError("selection belongs to another reconciliation")
    prior = reconciliation.get("independent_prior") or {}
    proposer = prior.get("proposer") or {}
    return {
        "label": "hypothesis_ranking",
        "value": selection["action"],
        "primary_forecast_unchanged": True,
        "scenario_role": "temporal_decision_selection",
        "candidate_origin": "model_authored_decision_prior",
        "proposer_id": proposer.get("proposer_id"),
        "model": proposer.get("model"),
        "support": "prior_assisted",
        "automation_eligible": False,
        "selection_seal_sha256": selection["seal_sha256"],
    }


def _json_objects(text: str) -> list[dict[str, Any]]:
    """Extract JSON objects without trusting markdown or surrounding prose."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    offset = 0
    while offset < len(text):
        start = text.find("{", offset)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        offset = start + consumed
    return objects


def empirical_quantile(values: list[float], probability: float) -> float:
    """Dependency-free linear empirical quantile for bounded path draws."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a quantile from no values")
    position = min(1.0, max(0.0, probability)) * (len(ordered) - 1)
    left = math.floor(position)
    right = math.ceil(position)
    weight = position - left
    return ordered[left] + (ordered[right] - ordered[left]) * weight


def sample_path_stability(
    paths: list[list[float]], history_values: list[float] | None,
) -> dict[str, Any]:
    """Return scale-free draw dispersion, explicitly not forecast skill."""
    if not paths or not paths[0]:
        raise ValueError("at least one non-empty sampled path is required")
    history = [float(value) for value in history_values or []
               if math.isfinite(float(value))]
    increments = [abs(right - left) for left, right in zip(history, history[1:])
                  if abs(right - left) > 0]
    if increments:
        scale = statistics.median(increments)
        scale_basis = "median_nonzero_history_increment"
    elif len(history) >= 2 and max(history) > min(history):
        scale = max(history) - min(history)
        scale_basis = "history_range"
    else:
        scale = max(1.0, abs(statistics.median(history)) * .01) if history else 1.0
        scale_basis = "level_floor"
    scale = max(float(scale), 1e-12)

    widths = []
    for index in range(len(paths[0])):
        point_values = [path[index] for path in paths]
        widths.append((empirical_quantile(point_values, .9) -
                       empirical_quantile(point_values, .1)) / scale)
    pairwise = [
        statistics.mean(abs(a - b) for a, b in zip(left, right)) / scale
        for left_index, left in enumerate(paths)
        for right in paths[left_index + 1:]
    ]
    tolerance = scale * 1e-6
    direction_agreement = []
    for index in range(1, len(paths[0])):
        signs = []
        for path in paths:
            change = path[index] - path[index - 1]
            signs.append(1 if change > tolerance else
                         -1 if change < -tolerance else 0)
        direction_agreement.append(max(signs.count(-1), signs.count(0),
                                       signs.count(1)) / len(signs))
    return {
        "version": "0.2",
        "interpretation": "stability_not_historical_skill",
        "scale_basis": scale_basis,
        "path_count": len(paths),
        "horizon": len(paths[0]),
        "median_pointwise_q80_width_scaled": statistics.median(widths),
        "mean_pointwise_q80_width_scaled": statistics.mean(widths),
        "p90_pointwise_q80_width_scaled": empirical_quantile(widths, .9),
        "median_pairwise_mae_scaled": (
            statistics.median(pairwise) if pairwise else 0.0),
        "max_pairwise_mae_scaled": max(pairwise) if pairwise else 0.0,
        "mean_direction_agreement": (
            statistics.mean(direction_agreement)
            if direction_agreement else 1.0),
        "unanimous_direction_fraction": (
            sum(value == 1.0 for value in direction_agreement) /
            len(direction_agreement) if direction_agreement else 1.0),
    }


def sampled_prior_sufficiency(
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Assess whether sampled paths are coherent enough to headline.

    This is deliberately an operational publication gate, not historical
    skill evidence.  It asks only whether the host received enough valid,
    mutually coherent paths to summarize the model's prior without presenting
    transport failures or a visibly multimodal elicitation as one answer.
    Candidates that fail remain useful labelled scenarios and retain their
    immutable receipts; they simply cannot displace the primary forecast.
    """
    requested = diagnostics.get("requested")
    accepted = diagnostics.get("accepted")
    reasons: list[dict[str, Any]] = []
    if (isinstance(requested, bool) or not isinstance(requested, int)
            or isinstance(accepted, bool) or not isinstance(accepted, int)
            or requested < 1 or accepted < 0 or accepted > requested):
        return {
            "version": "0.1",
            "eligible_for_human_recommendation": False,
            "reason_codes": ["invalid_path_accounting"],
            "reasons": [{
                "code": "invalid_path_accounting",
                "message": "Sample request accounting is invalid.",
            }],
            "historical_skill_evidence": False,
            "automation_eligible": False,
        }

    valid_fraction = accepted / requested
    if accepted < SAMPLED_PRIOR_MIN_PATHS:
        reasons.append({
            "code": "too_few_valid_paths",
            "message": (
                f"Only {accepted} valid paths survived; at least "
                f"{SAMPLED_PRIOR_MIN_PATHS} are required."),
        })
    if valid_fraction < SAMPLED_PRIOR_MIN_VALID_FRACTION:
        reasons.append({
            "code": "low_valid_path_fraction",
            "message": (
                f"Only {valid_fraction:.0%} of requested paths were valid; "
                f"at least {SAMPLED_PRIOR_MIN_VALID_FRACTION:.0%} are "
                "required for a headline recommendation."),
        })

    stability = diagnostics.get("stability")
    if not isinstance(stability, dict):
        reasons.append({
            "code": "stability_not_measured",
            "message": "Host-observed path stability was not measured.",
        })
        direction_agreement = None
        pairwise_ratio = None
    else:
        try:
            direction_agreement = float(stability["mean_direction_agreement"])
            median_pairwise = float(stability["median_pairwise_mae_scaled"])
            mean_width = float(
                stability["mean_pointwise_q80_width_scaled"])
            # A zero median marginal width is not permission for materially
            # different paths to headline as one distribution. That pattern
            # occurs when draws agree at many timestamps but disagree across
            # whole trajectories; the old absolute floor silently waived the
            # relative-coherence gate. Keep only a numerical epsilon in the
            # denominator so identical paths pass and contradictory paths are
            # visibly demoted, independent of units.
            pairwise_limit = (
                SAMPLED_PRIOR_MAX_PAIRWISE_TO_POINTWISE_RATIO
                * max(mean_width, 1e-6))
            pairwise_ratio = median_pairwise / max(mean_width, 1e-12)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            reasons.append({
                "code": "invalid_stability_diagnostics",
                "message": "Host-observed stability diagnostics are invalid.",
            })
            direction_agreement = None
            pairwise_ratio = None
        else:
            valid_stability = bool(
                stability.get("interpretation") ==
                "stability_not_historical_skill"
                and stability.get("path_count") == accepted
                and all(math.isfinite(value) and value >= 0 for value in (
                    direction_agreement, median_pairwise, mean_width))
                and direction_agreement <= 1)
            if not valid_stability:
                reasons.append({
                    "code": "invalid_stability_diagnostics",
                    "message": "Host-observed stability diagnostics are invalid.",
                })
            elif direction_agreement < SAMPLED_PRIOR_MIN_DIRECTION_AGREEMENT:
                reasons.append({
                    "code": "directionally_unstable_paths",
                    "message": (
                        "Sampled paths do not agree sufficiently on temporal "
                        "direction to support one headline path."),
                })
            if valid_stability and median_pairwise > pairwise_limit:
                reasons.append({
                    "code": "dispersed_sampled_paths",
                    "message": (
                        "Typical path-to-path disagreement is too large "
                        "relative to the sampled marginal spread."),
                })

    return {
        "version": "0.1",
        "eligible_for_human_recommendation": not reasons,
        "reason_codes": [str(item["code"]) for item in reasons],
        "reasons": reasons,
        "requested_paths": requested,
        "accepted_paths": accepted,
        "valid_path_fraction": round(valid_fraction, 6),
        "minimum_paths": SAMPLED_PRIOR_MIN_PATHS,
        "minimum_valid_fraction": SAMPLED_PRIOR_MIN_VALID_FRACTION,
        "minimum_direction_agreement":
            SAMPLED_PRIOR_MIN_DIRECTION_AGREEMENT,
        "maximum_pairwise_to_pointwise_ratio":
            SAMPLED_PRIOR_MAX_PAIRWISE_TO_POINTWISE_RATIO,
        "observed_direction_agreement": direction_agreement,
        "observed_pairwise_to_pointwise_ratio": pairwise_ratio,
        "interpretation": "elicitation_sufficiency_not_historical_skill",
        "historical_skill_evidence": False,
        "automation_eligible": False,
    }


def candidate_from_sampled_paths(
    outputs: list[str], future_timestamps: list[str],
    *, history_values: list[float] | None = None,
    path_transform: Any = None,
    allowed_claim_ids: set[str] | None = None,
    required_claim_groups: list[set[str]] | None = None,
    single_choice_claim_ids: set[str] | None = None,
    require_rationale: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Validate independent model paths on a host-owned grid and aggregate."""
    accepted: list[list[float]] = []
    rejection_reasons: list[str] = []
    response_shapes: list[dict[str, Any]] = []
    rationales: list[str] = []
    accepted_claim_ids: list[list[str]] = []
    accepted_single_choices: list[str | None] = []
    expected = len(future_timestamps)
    anchor_indices = sampled_path_anchor_indices(expected)
    display_timestamps = []
    for timestamp in future_timestamps:
        try:
            display_timestamps.append(datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")).strftime(
                    "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            display_timestamps.append(timestamp)
    for output in outputs:
        objects = _json_objects(output)
        first = objects[0] if objects else {}
        raw = first.get("forecast_path") if isinstance(first, dict) else None
        values = raw.get("values") if isinstance(raw, dict) else None
        supplied_claim_ids = (raw.get("claim_ids")
                              if isinstance(raw, dict) else None)
        shape = {
            "sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "characters": len(output),
            "json_objects": len(objects),
            "forecast_path_present": isinstance(raw, dict),
            "observed_values": len(values) if isinstance(values, list) else None,
        }
        if not isinstance(values, list):
            match = re.search(
                r"<forecast>\s*(.*?)\s*</forecast>", output,
                flags=re.IGNORECASE | re.DOTALL)
            parsed_values: list[float] = []
            parsed_timestamps: list[str] = []
            if match:
                for line in match.group(1).splitlines():
                    line = line.strip().strip("(), ")
                    if not line:
                        continue
                    parts = line.rsplit(",", 1)
                    if len(parts) != 2:
                        parsed_values = []
                        break
                    parsed_timestamps.append(parts[0].strip(" '\""))
                    try:
                        parsed_values.append(float(parts[1].strip()))
                    except ValueError:
                        parsed_values = []
                        break
            if parsed_values and parsed_timestamps == display_timestamps:
                values = parsed_values
                shape["fallback_format"] = "timestamp_value_rows"
                shape["observed_values"] = len(values)
        if not isinstance(values, list) or len(values) not in {
                expected, len(anchor_indices)}:
            shape["status"] = "rejected_wrong_shape"
            response_shapes.append(shape)
            rejection_reasons.append(
                f"forecast response requires {expected} full-grid values or "
                f"{len(anchor_indices)} host-selected anchor values")
            continue
        claim_ids: list[str] = []
        selected_choice: str | None = None
        if allowed_claim_ids is not None:
            if not isinstance(supplied_claim_ids, list):
                shape["status"] = "rejected_missing_claim_ids"
                response_shapes.append(shape)
                rejection_reasons.append(
                    "forecast_path requires grounded claim_ids")
                continue
            claim_ids = list(dict.fromkeys(
                str(item) for item in supplied_claim_ids))
            if (not claim_ids
                    or any(item not in allowed_claim_ids for item in claim_ids)):
                shape["status"] = "rejected_unknown_claim_ids"
                response_shapes.append(shape)
                rejection_reasons.append(
                    "forecast_path cites an unknown claim_id")
                continue
            missing_group = next((group for group in required_claim_groups or []
                                  if not group.intersection(claim_ids)), None)
            if missing_group is not None:
                shape["status"] = "rejected_missing_claim_group"
                response_shapes.append(shape)
                rejection_reasons.append(
                    "forecast_path omits a required evidence class")
                continue
            choices = ((single_choice_claim_ids or set()).intersection(
                claim_ids))
            if single_choice_claim_ids is not None and len(choices) != 1:
                shape["status"] = "rejected_ambiguous_reference_choice"
                response_shapes.append(shape)
                rejection_reasons.append(
                    "forecast_path must cite exactly one comparable range")
                continue
            selected_choice = next(iter(choices), None)
        try:
            path = [float(value) for value in values]
        except (TypeError, ValueError):
            shape["status"] = "rejected_non_numeric"
            response_shapes.append(shape)
            rejection_reasons.append("forecast_path contains a non-number")
            continue
        if not all(math.isfinite(value) for value in path):
            shape["status"] = "rejected_non_finite"
            response_shapes.append(shape)
            rejection_reasons.append("forecast_path contains a non-finite value")
            continue
        if len(path) == len(anchor_indices) and len(anchor_indices) < expected:
            path = _interpolate_sampled_anchors(
                path, anchor_indices, future_timestamps)
            shape["path_representation"] = "host_anchor_linear_interpolation"
            shape["anchor_count"] = len(anchor_indices)
        else:
            shape["path_representation"] = "full_host_grid"
        if path_transform is not None:
            try:
                path = [float(value) for value in path_transform(path)]
            except (TypeError, ValueError, OverflowError):
                shape["status"] = "rejected_transformation"
                response_shapes.append(shape)
                rejection_reasons.append(
                    "forecast_path failed the governed transformation")
                continue
            if len(path) != expected or not all(
                    math.isfinite(value) for value in path):
                shape["status"] = "rejected_transformation_shape"
                response_shapes.append(shape)
                rejection_reasons.append(
                    "governed transformation returned an invalid path")
                continue
        shape["status"] = "accepted"
        if claim_ids:
            shape["claim_ids"] = claim_ids
        response_shapes.append(shape)
        accepted.append(path)
        accepted_claim_ids.append(claim_ids)
        accepted_single_choices.append(selected_choice)
        rationale = " ".join(str(
            raw.get("rationale") or "" if isinstance(raw, dict) else ""
        ).split())
        if require_rationale and not rationale:
            accepted.pop()
            accepted_claim_ids.pop()
            accepted_single_choices.pop()
            shape["status"] = "rejected_missing_external_assumption"
            rejection_reasons.append(
                "forecast_path must state its external matching assumption")
            continue
        rationales.append(rationale[:300])
    diagnostics = {
        "requested": len(outputs), "accepted": len(accepted),
        "rejected": len(outputs) - len(accepted),
        "rejection_reasons": rejection_reasons[:8],
        # Structural diagnostics are sufficient to debug provider formatting
        # without persisting raw model text or user context in the receipt.
        "response_shapes": response_shapes,
        "aggregation": "linear_empirical_marginal_q10_q50_q90",
        "timestamp_binding": "host_grid_order",
        "request_mode": "concurrent_single_sample_requests",
    }
    if single_choice_claim_ids is not None and accepted:
        counts = {choice: accepted_single_choices.count(choice)
                  for choice in set(accepted_single_choices) if choice}
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        majority_choice = ordered[0][0] if ordered else None
        majority_count = ordered[0][1] if ordered else 0
        required_majority = len(accepted) // 2 + 1
        diagnostics["reference_selection"] = {
            "counts": counts,
            "required_majority": required_majority,
            "selected_claim_id": (
                majority_choice if majority_count >= required_majority else None),
            "interpretation": "majority_consistent_comparable_not_skill",
        }
        if majority_count < required_majority:
            diagnostics["rejection_reasons"].append(
                "sampled paths did not agree on one comparable")
            diagnostics["accepted"] = 0
            diagnostics["rejected"] = len(outputs)
            diagnostics["accepted_after_reference_consensus"] = 0
            return None, diagnostics
        accepted_before_consensus = len(accepted)
        retained = [index for index, choice in enumerate(
            accepted_single_choices) if choice == majority_choice]
        accepted = [accepted[index] for index in retained]
        accepted_claim_ids = [accepted_claim_ids[index] for index in retained]
        rationales = [rationales[index] for index in retained]
        diagnostics["accepted"] = len(accepted)
        diagnostics["rejected"] = len(outputs) - len(accepted)
        diagnostics["accepted_after_reference_consensus"] = len(accepted)
        if len(accepted) < accepted_before_consensus:
            diagnostics["rejection_reasons"].append(
                "paths citing minority comparables were excluded")
    if not accepted:
        return None, diagnostics
    diagnostics["stability"] = sample_path_stability(accepted, history_values)
    rows = []
    for index, timestamp in enumerate(future_timestamps):
        point_values = [path[index] for path in accepted]
        rows.append({
            "timestamp": timestamp,
            "q10": empirical_quantile(point_values, .1),
            "q50": empirical_quantile(point_values, .5),
            "q90": empirical_quantile(point_values, .9),
        })
    candidate = {
        "quantiles": rows,
        "_validated_sample_paths": accepted,
        "rationale": (
            f"Host-aggregated {len(accepted)} sampled model-authored point "
            "paths into empirical marginal quantiles. "
            + ("Sample rationales: " + " | ".join(
                item for item in rationales if item)[:600]
               if any(rationales) else "")),
    }
    if allowed_claim_ids is not None:
        candidate["_selected_claim_ids"] = sorted(set().union(
            *(set(items) for items in accepted_claim_ids)))
    return candidate, diagnostics


def sampled_path_anchor_indices(
        horizon: int, *, maximum: int = SAMPLED_PRIOR_MAX_ANCHORS,
) -> list[int]:
    """Return deterministic near-uniform anchors including both boundaries."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 2:
        raise ValueError("maximum must be an integer of at least two")
    if horizon <= maximum:
        return list(range(horizon))
    indices = [round(index * (horizon - 1) / (maximum - 1))
               for index in range(maximum)]
    # Rounding a monotone grid cannot duplicate at horizon > maximum, but keep
    # the invariant explicit so a future selection rule cannot silently alter
    # the wire contract.
    if len(set(indices)) != maximum or indices[0] != 0 \
            or indices[-1] != horizon - 1:
        raise AssertionError("invalid sampled-path anchor grid")
    return indices


def _interpolate_sampled_anchors(
    values: list[float], indices: list[int], timestamps: list[str],
) -> list[float]:
    """Linearly interpolate validated anchors on the host-owned time grid."""
    if len(values) != len(indices) or not indices:
        raise ValueError("anchor values and indices must align")
    parsed = [datetime.fromisoformat(value.replace("Z", "+00:00"))
              for value in timestamps]
    if any(right <= left for left, right in zip(parsed, parsed[1:])):
        raise ValueError("future timestamps must be strictly increasing")
    result = [0.0] * len(timestamps)
    for segment, (left_index, right_index) in enumerate(
            zip(indices, indices[1:])):
        left_value, right_value = values[segment], values[segment + 1]
        left_time, right_time = parsed[left_index], parsed[right_index]
        span = (right_time - left_time).total_seconds()
        if span <= 0:
            raise ValueError("anchor timestamps must be strictly increasing")
        for index in range(left_index, right_index + 1):
            fraction = (parsed[index] - left_time).total_seconds() / span
            result[index] = left_value + fraction * (right_value - left_value)
    return result


def build_sampled_context_prior_prompt(
    *, timestamps: list[str], values: list[float],
    future_timestamps: list[str], context: str,
    temporal_facts: dict[str, Any] | None = None,
    claim_catalog: dict[str, str] | None = None,
    single_choice_claim_ids: set[str] | None = None,
    external_matching_assumption_required: bool = False,
) -> str:
    """Build a compact numeric prompt for a host's own model provider."""
    history_values = ",".join(f"{float(value):.12g}" for value in values)

    def grid_summary(grid: list[str]) -> str:
        parsed = [datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")) for timestamp in grid]
        if len(parsed) < 2:
            return f"timestamps={grid!r}"
        steps = [(right - left).total_seconds()
                 for left, right in zip(parsed, parsed[1:])]
        if steps and max(steps) == min(steps):
            return (f"start={grid[0]}, end={grid[-1]}, "
                    f"step_seconds={steps[0]:.12g}, count={len(grid)}")
        return "timestamps=" + json.dumps(grid, separators=(",", ":"))

    anchor_indices = sampled_path_anchor_indices(len(future_timestamps))
    sparse = len(anchor_indices) < len(future_timestamps)
    output_instruction = (
        f"Return exactly {len(anchor_indices)} finite values at these zero-based "
        f"future-grid indices, in order: {anchor_indices}. The host will "
        "linearly interpolate between these fixed time anchors."
        if sparse else
        "Return exactly one finite value per future grid point, in order.")
    representation = ("host-selected sparse anchors" if sparse
                      else "the complete host grid")
    citation_instruction = ""
    json_example = '{"forecast_path":{"values":[1.0,2.0],"rationale":"brief basis"}}'
    if claim_catalog is not None:
        allowed = sorted(claim_catalog)
        choices = sorted(single_choice_claim_ids or set())
        citation_instruction = (
            "Every path must cite the source facts it uses. Include relevant "
            "target-descriptor claim IDs and exactly one comparable-range "
            f"claim ID from {choices}. Allowed claim IDs: {allowed}. "
            "Do not cite an alternative comparable as support.\n")
        if external_matching_assumption_required:
            citation_instruction += (
                "The source does not state matching attributes for the "
                "comparables. The rationale must name the external matching "
                "assumption used to choose one. Treat it as prior knowledge, "
                "not source-grounded or automation-safe evidence.\n")
        json_example = (
            '{"forecast_path":{"values":[1.0,2.0],'
            '"claim_ids":["claim-1","claim-3"],'
            '"rationale":"brief analogue basis"}}')
    return f"""\
I have a time series forecasting task for you.

Here is context known at the forecast cutoff. Factor in relevant background
knowledge, satisfy any stated constraints, and respect any stated scenarios.
<context>
{context}
</context>

The host owns the historical grid ({grid_summary(timestamps)}). Values below
are in exact grid order:
<history>
[{history_values}]
</history>

Deterministic past-only temporal reference (descriptive evidence, not proof of
future skill):
<temporal_facts>
{json.dumps(temporal_facts or {}, sort_keys=True, separators=(",", ":"))}
</temporal_facts>

Predict the future grid ({grid_summary(future_timestamps)}).

{output_instruction}
{citation_instruction}
Return only compact JSON for {representation}. Do not echo timestamps:

{json_example}

Use no observations after the cutoff.
"""


def candidate_temporal_facts(
    timestamps: list[str], values: list[float], *, horizon: int,
) -> dict[str, Any]:
    """Pre-shape full past history into a compact forecast argument.

    Raw prompts stay bounded, but the candidate still receives deterministic
    level/trend/season evidence computed over every pre-cutoff observation.
    The seasonal reference is a robust same-phase median, aligned to the
    requested future steps; it is a baseline to reason from, not an admitted
    model or an automation signal.
    """
    if horizon < 1 or not values or len(timestamps) != len(values):
        raise ValueError("temporal facts require aligned non-empty history")
    from .temporal import default_season, detect_season, infer_frequency

    parsed = [datetime.fromisoformat(value.replace("Z", "+00:00"))
              for value in timestamps]
    frequency = infer_frequency(parsed)
    detected, strength, source = detect_season(values, frequency)
    calendar = default_season(frequency)
    period = calendar if calendar >= 2 and len(values) >= 2 * calendar else detected
    if period < 2 or len(values) < 2 * period:
        period = 1
    differences = [right - left for left, right in zip(values, values[1:])]
    reference: list[float] = []
    if period > 1:
        first = max(0, len(values) - 4 * period)
        for step in range(min(horizon, 64)):
            phase = (len(values) + step) % period
            phase_values = [float(values[index]) for index in
                            range(first, len(values)) if index % period == phase]
            reference.append(round(statistics.median(phase_values), 12))
    return {
        "observations": len(values),
        "frequency": frequency,
        "last_value": round(float(values[-1]), 12),
        "median_first_difference": round(
            statistics.median(differences), 12) if differences else 0.0,
        "seasonal_period": period,
        "seasonal_strength": round(float(strength), 6),
        "seasonal_basis": source,
        "seasonal_reference_next": reference,
        "seasonal_reference_interpretation": (
            "past-only same-phase median baseline; descriptive, not "
            "historical skill evidence"),
    }


def build_relationship_prior_prompt(
    *, context: str, target_name: str, driver_name: str,
) -> str:
    """Ask a host model for a tiny declarative relationship prior.

    This is intentionally not a forecast request. The model may contribute
    domain knowledge about functional form, while the host retains all numeric
    execution, scaling, uncertainty, path binding, and publication authority.
    """
    return f"""\
Extract the named relationship between target {target_name!r} and driver
{driver_name!r} from the supplied context using your general domain knowledge.
Return ONLY compact JSON in this exact shape:
{{"relationship_prior":{{"family":"linear|power","exponent":1.0,
"rationale":"brief named-law basis"}}}}

Use family linear only for an affine/linear relationship and exponent 1.
Use family power only for a power law and give its dimensionless exponent.
Do not output coefficients, intercepts, forecasts, observations, code, support,
or automation advice. If the named law does not determine either family,
return {{"relationship_prior":null}}.

<context>
{context}
</context>
"""


def candidate_from_relationship_prior_specs(
    outputs: list[str], *, target_history: list[float],
    driver_history: list[float], future_driver: list[float],
    future_timestamps: list[str], claim_ids: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Execute stable model-authored relationship forms as a sealed prior.

    Repeated model outputs contribute only an allowed family and exponent.
    Gnomon fits the scale from observed history and deterministically constructs
    the path and uncertainty. Agreement is elicitation evidence, never
    historical skill, and the resulting candidate cannot authorize automation.
    """
    if len(target_history) != len(driver_history) or len(target_history) < 4:
        raise ValueError("relationship prior requires aligned target/driver history")
    if len(future_driver) != len(future_timestamps) or not future_timestamps:
        raise ValueError("future driver must match the host-owned grid")
    target = [float(value) for value in target_history]
    driver = [float(value) for value in driver_history]
    future = [float(value) for value in future_driver]
    if not all(math.isfinite(value) for value in [*target, *driver, *future]):
        raise ValueError("relationship prior inputs must be finite")

    accepted: list[tuple[str, float]] = []
    shapes: list[dict[str, Any]] = []
    reasons: list[str] = []
    for output in outputs:
        objects = _json_objects(output)
        first = objects[0] if objects else {}
        raw = first.get("relationship_prior") if isinstance(first, dict) else None
        shape = {
            "sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "characters": len(output), "json_objects": len(objects),
        }
        if raw is None:
            shape["status"] = "withheld"
            shapes.append(shape); reasons.append("relationship prior withheld")
            continue
        if not isinstance(raw, dict):
            shape["status"] = "rejected_shape"
            shapes.append(shape); reasons.append("relationship prior must be an object")
            continue
        family = str(raw.get("family") or "").strip().casefold()
        try:
            exponent = float(raw.get("exponent"))
        except (TypeError, ValueError):
            exponent = math.nan
        valid = (family in {"linear", "power"}
                 and math.isfinite(exponent) and -4 <= exponent <= 4
                 and (family != "linear" or math.isclose(exponent, 1.0)))
        if not valid:
            shape["status"] = "rejected_semantics"
            shapes.append(shape); reasons.append("unsupported family or exponent")
            continue
        if family == "power" and min([*driver, *future]) <= 0:
            shape["status"] = "rejected_domain"
            shapes.append(shape); reasons.append(
                "power prior requires positive driver values")
            continue
        accepted.append((family, exponent))
        shape.update({"status": "accepted", "family": family,
                      "exponent": exponent})
        shapes.append(shape)
    requested = len(outputs)
    family_counts = {family: sum(item[0] == family for item in accepted)
                     for family in {item[0] for item in accepted}}
    winning_family = (max(family_counts, key=lambda key: (
        family_counts[key], key)) if family_counts else None)
    winning = [item[1] for item in accepted if item[0] == winning_family]
    family_agreement = (len(winning) / len(accepted) if accepted else 0.0)
    valid_fraction = len(accepted) / requested if requested else 0.0
    exponent_width = (max(winning) - min(winning) if winning else math.inf)
    eligible = bool(len(accepted) >= 3 and valid_fraction >= .75
                    and family_agreement >= .8 and exponent_width <= .5)
    diagnostics = {
        "version": "0.1", "requested": requested,
        "accepted": len(accepted), "rejected": requested - len(accepted),
        "response_shapes": shapes, "rejection_reasons": reasons[:8],
        "family_counts": family_counts, "winning_family": winning_family,
        "family_agreement": family_agreement,
        "valid_fraction": valid_fraction, "exponent_width": exponent_width,
        "eligible_for_human_recommendation": eligible,
        "interpretation": "domain_prior_stability_not_historical_skill",
        "historical_skill_evidence": False, "automation_eligible": False,
    }
    if not eligible or winning_family is None:
        return None, diagnostics
    exponent = statistics.median(winning)

    def basis(value: float, family: str, power: float) -> float:
        return value if family == "linear" else value ** power

    bases = [basis(value, winning_family, exponent) for value in driver]
    if winning_family == "linear":
        xbar, ybar = statistics.mean(bases), statistics.mean(target)
        denom = sum((value - xbar) ** 2 for value in bases)
        slope = (sum((x - xbar) * (y - ybar)
                     for x, y in zip(bases, target)) / denom
                 if denom > 1e-12 else 0.0)
        intercept = ybar - slope * xbar
        predict = lambda value, exp=exponent: intercept + slope * basis(
            value, winning_family, exp)
        fitted = [predict(value) for value in driver]
    else:
        scales = [y / basis(x, winning_family, exponent)
                  for x, y in zip(driver, target)]
        scale = statistics.median(scales)
        predict = lambda value, exp=exponent: scale * basis(
            value, winning_family, exp)
        fitted = [predict(value) for value in driver]
    residuals = [actual - estimate for actual, estimate in zip(target, fitted)]
    center = statistics.median(residuals)
    mad = statistics.median(abs(value - center) for value in residuals)
    base_width = max(1.2815515655446004 * 1.4826 * mad,
                     max(abs(value) for value in residuals),
                     max(statistics.median(abs(value) for value in target), 1.0)
                     * 1e-6)
    rows = []
    for timestamp, value in zip(future_timestamps, future):
        alternatives = [predict(value, exp) for exp in winning]
        point = predict(value)
        width = max(base_width, max(abs(item - point) for item in alternatives))
        rows.append({"timestamp": timestamp, "point": point,
                     "q10": point - width, "q50": point,
                     "q90": point + width})
    candidate = {
        "kind": "model_authored_relationship_prior",
        "forecast": rows,
        "quantiles": [{key: row[key] for key in
                       ("timestamp", "q10", "q50", "q90")} for row in rows],
        "claim_ids": list(dict.fromkeys(str(item) for item in claim_ids)),
        "rationale": (
            "Host-executed relationship prior: the model supplied only a "
            "bounded functional family; Gnomon fitted scale and uncertainty."),
        "provenance_class": "model_authored_relationship_prior",
        "validation": {
            "scheme": "repeated_domain_prior_elicitation",
            "family": winning_family, "exponent": exponent,
            "elicitation": diagnostics, "historical_skill_evidence": False,
            "beats_baseline": False,
        },
        "support": "prior_assisted", "selection_eligible": True,
        "automation_eligible": False, "primary_forecast_unchanged": True,
        "executable": {"kind": "sealed_relationship_prior", "version": "0.1",
                       "family": winning_family, "exponent": exponent},
    }
    return candidate, diagnostics


def recommended_sample_count(horizon: int) -> int:
    """Bound host inference while tolerating one rejected long path."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return 4 if horizon >= 96 else 5


def recommended_initial_sample_count(horizon: int) -> int:
    """Use the bounded cap so the published median is not a three-draw fluke.

    Path agreement is only a coherence diagnostic and raw spread never becomes
    calibrated uncertainty. Even so, the human-facing centre is an empirical
    median: starting with only three stochastic draws made that centre needlessly
    brittle while saving latency only on concurrent requests. The existing
    horizon-aware cap (four or five) remains the hard cost boundary.
    """
    return recommended_sample_count(horizon)
