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
) -> dict[str, Any]:
    """Build a deterministic contrast between immutable evidence and prior."""
    if not verify_temporal_decision_prior(prior_receipt):
        raise ValueError("temporal decision prior seal is invalid")
    if prior_receipt.get("question_sha256") != question_sha256:
        raise ValueError("temporal decision prior belongs to another question")
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
        "version": "0.1",
        "interpretation": "stability_not_historical_skill",
        "scale_basis": scale_basis,
        "path_count": len(paths),
        "horizon": len(paths[0]),
        "median_pointwise_q80_width_scaled": statistics.median(widths),
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


def candidate_from_sampled_paths(
    outputs: list[str], future_timestamps: list[str],
    *, history_values: list[float] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Validate independent model paths on a host-owned grid and aggregate."""
    accepted: list[list[float]] = []
    rejection_reasons: list[str] = []
    rationales: list[str] = []
    expected = len(future_timestamps)
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
        if not isinstance(values, list) or len(values) != expected:
            rejection_reasons.append(
                f"forecast response requires {expected} host-grid-bound values")
            continue
        try:
            path = [float(value) for value in values]
        except (TypeError, ValueError):
            rejection_reasons.append("forecast_path contains a non-number")
            continue
        if not all(math.isfinite(value) for value in path):
            rejection_reasons.append("forecast_path contains a non-finite value")
            continue
        accepted.append(path)
        rationale = " ".join(str(
            raw.get("rationale") or "" if isinstance(raw, dict) else ""
        ).split())
        if rationale:
            rationales.append(rationale[:300])
    diagnostics = {
        "requested": len(outputs), "accepted": len(accepted),
        "rejected": len(outputs) - len(accepted),
        "rejection_reasons": rejection_reasons[:8],
        "aggregation": "linear_empirical_marginal_q10_q50_q90",
        "timestamp_binding": "host_grid_order",
        "request_mode": "concurrent_single_sample_requests",
    }
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
            + ("Sample rationales: " + " | ".join(rationales[:2])
               if rationales else "")),
    }
    return candidate, diagnostics


def build_sampled_context_prior_prompt(
    *, timestamps: list[str], values: list[float],
    future_timestamps: list[str], context: str,
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

Predict the future grid ({grid_summary(future_timestamps)}).

Return only compact JSON with exactly one finite value per future grid point,
in order. Do not echo timestamps:

{{"forecast_path":{{"values":[1.0,2.0],"rationale":"brief basis"}}}}

Use no observations after the cutoff.
"""


def recommended_sample_count(horizon: int) -> int:
    """Bound host inference while tolerating one rejected long path."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return 4 if horizon >= 96 else 5
