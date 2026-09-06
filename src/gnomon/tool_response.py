"""Pure legacy response projections shared by CLI and MCP; no registry or execution imports."""

from __future__ import annotations

import json

from typing import Any

from .contracts import ForecastArtifact


FORECAST_PREVIEW_ROWS = 12


FORECAST_PREVIEW_SMALL_HORIZON = 16


def _bounded_forecast_preview(rows: list[Any]) -> tuple[list[Any], int]:
    """Keep both decision-near and horizon-end rows in a brief response."""
    # Avoid hiding a support-tier transition to save only a handful of rows;
    # short split-horizon answers are more useful intact.
    if len(rows) <= FORECAST_PREVIEW_SMALL_HORIZON:
        return list(rows), 0
    head = FORECAST_PREVIEW_ROWS // 2
    tail = FORECAST_PREVIEW_ROWS - head
    return [*rows[:head], *rows[-tail:]], len(rows) - FORECAST_PREVIEW_ROWS


def apply_response_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Add the compact agent-facing envelope without rewriting artifacts.

    Existing verb payloads remain authoritative and byte-compatible on disk;
    this projection adds stable routing fields to MCP responses. Repeated
    warning text is grouped with a count instead of asking an agent to infer
    prevalence from prose. The complete per-series warnings remain in the
    artifact and in each result.
    """
    if payload.get("status") == "error" or "error" in payload:
        from .reasoning_boundary import apply_reasoning_boundary
        return apply_reasoning_boundary(payload)
    result = dict(payload)

    if "artifact_id" not in result:
        for key in ("forecast_id", "investigation_id", "anomaly_id",
                    "decision_id", "monitor_id", "route_id"):
            if result.get(key):
                result["artifact_id"] = result[key]
                break

    entries = [item for item in result.get("results", [])
               if isinstance(item, dict)]
    triggers = [item for item in result.get("triggers", [])
                if isinstance(item, dict)]
    warning_series: dict[str, set[str]] = {}
    warning_examples: dict[str, list[str]] = {}
    recoveries: list[dict[str, Any]] = []

    def add_recoveries(assessment: object) -> None:
        if not isinstance(assessment, dict):
            return
        actions = assessment.get("recovery_actions") or []
        if not isinstance(actions, list):
            return
        for action in actions:
            if isinstance(action, dict) and action not in recoveries:
                recoveries.append(action)

    root_recoveries = result.get("recovery_actions") or []
    if isinstance(root_recoveries, list):
        for action in root_recoveries:
            if isinstance(action, dict) and action not in recoveries:
                recoveries.append(action)
    add_recoveries(result.get("support_assessment"))
    from .support import payload_support_tier
    for entry in entries:
        assessment = entry.get("support_assessment") or {}
        series = str(entry.get("series") or "__default__")
        for warning in entry.get("warnings") or []:
            text = str(warning)
            warning_series.setdefault(text, set()).add(series)
            examples = warning_examples.setdefault(text, [])
            if series not in examples and len(examples) < 3:
                examples.append(series)
        add_recoveries(assessment)
    for trigger in triggers:
        add_recoveries(trigger.get("support_assessment"))

    tier_floor = payload_support_tier(result)
    if tier_floor is not None:
        # Recompute rather than trusting a pre-existing summary. A bounded
        # response may carry a floor calculated before its weakest series was
        # moved into triage.remainder_tiers.
        result["tier_floor"] = tier_floor
    if warning_series and "limitation_groups" not in result:
        result["limitation_groups"] = [
            {
                "code": "RUNTIME_WARNING",
                "message": message,
                "affected_series_count": len(warning_series[message]),
                "examples": warning_examples[message],
            }
            for message in sorted(warning_series)
        ]
    bounded_threshold_brief = any(
        isinstance(entry.get("threshold"), dict)
        and bool(entry["threshold"].get("bounded_assessment"))
        for entry in entries)
    if (result.get("format") == "brief" and result.get("limitation_groups")
            and bounded_threshold_brief):
        # The grouped form above retains every distinct warning verbatim plus
        # affected-series counts/examples. Repeating those same strings under
        # every result makes wide and fold-starved threshold answers expensive
        # without adding evidence. The sealed artifact remains the per-series
        # source.
        for entry in entries:
            warnings = {str(value) for value in entry.get("warnings") or []}
            assessment = entry.get("support_assessment") or {}
            if isinstance(assessment, dict):
                compact = dict(assessment)
                reasons = [dict(value) for value in
                           compact.get("reasons") or []]
                grouped_reason_codes = {
                    "warning", "degraded_evaluation",
                    "selection_underpowered",
                } if warnings else set()
                filtered_reasons = [
                    value for value in reasons
                    if value.get("code") not in grouped_reason_codes
                ]
                if len(filtered_reasons) != len(reasons):
                    compact["reasons"] = filtered_reasons
                    compact["grouped_reason_codes"] = sorted({
                        str(value.get("code")) for value in reasons
                        if value.get("code") in grouped_reason_codes})
                disclosures = [dict(value) for value in
                               compact.get("disclosures") or []]
                if entry.get("model_assisted"):
                    filtered_disclosures = [
                        value for value in disclosures
                        if value.get("code") != "model_assisted_lane"]
                    if len(filtered_disclosures) != len(disclosures):
                        compact["disclosures"] = filtered_disclosures
                entry["support_assessment"] = compact
            threshold = entry.get("threshold") or {}
            if isinstance(threshold, dict) and threshold.get(
                    "bounded_assessment"):
                # The structured threshold block carries probability status,
                # range relation, conflict, and automation eligibility. Drop
                # only the note that restates those same fields in prose.
                notes = [
                    note for note in entry.get("notes") or []
                    if not (str(note).startswith("threshold ")
                            and "no crossing probability" in str(note))
                ]
                if notes:
                    entry["notes"] = notes
                else:
                    entry.pop("notes", None)
            if entry.get("warnings"):
                entry.pop("warnings", None)
        # artifact_path already names the complete result; this prose merely
        # restated the same pointer and consumed every subsequent agent turn.
        result.pop("note", None)
    if recoveries:
        result["recovery_actions"] = recoveries
    from .reasoning_boundary import apply_reasoning_boundary
    bounded = apply_reasoning_boundary(result)
    if (bounded.get("format") == "brief"
            and bounded.get("agent_response_contract")
            and isinstance(bounded.get("reasoning"), dict)):
        # The sealed response contract supersedes the generic reasoning
        # frame's duplicate pointers. Keep the terminal/sufficiency fields
        # hosts already consume, while full responses retain the complete
        # reasoning receipt.
        reasoning = bounded["reasoning"]
        compact_reasoning: dict[str, Any] = {}
        resolution = reasoning.get("resolution")
        if isinstance(resolution, dict) and resolution.get("kind"):
            compact_reasoning["resolution"] = {
                "kind": resolution["kind"]}
        bounded["reasoning"] = compact_reasoning
    return bounded


def compact_publication_for_wire(payload: dict[str, Any]) -> dict[str, Any]:
    """Project a signed publication without repeating its forecast arrays.

    The complete, verifiable receipt remains at ``publication_path``.  Agents
    get the decision contract and authority fields needed to reason or invoke
    ``gnomon_select_scenario``; requesting ``format=full`` bypasses this
    projection at the runner boundary.
    """
    publication = payload.get("publication")
    path = payload.get("publication_path")
    if not isinstance(publication, dict) or not path:
        return payload
    keys = (
        "schema_version", "artifact_id", "mode", "recommended_scenario_id",
        "recommended_support", "primary_scenario_id",
        "primary_forecast_unchanged", "scenario_count",
        "context_dispositions", "context_summary", "temporal_state",
        "scenario_selection",
        "recommendation_authority", "automation", "calibration_lineage",
        "selection_contract",
        "candidate_admission", "publication_seal_sha256",
    )
    projection = {key: publication[key] for key in keys if key in publication}
    dispositions = list(projection.get("context_dispositions") or [])
    if dispositions and isinstance(projection.get("context_summary"), dict):
        # Global automation may truthfully say "not requested", but that is
        # not the context-authority answer an agent needs.  Put the two
        # context-specific facts beside the disposition so the model never
        # has to infer them from unrelated publication policy fields.
        projection["context_summary"] = {
            **projection["context_summary"],
            "canonical_primary_preserved": bool(
                publication.get("primary_forecast_unchanged", True)),
            "context_evidence_automation_eligible": False,
        }
    if len(dispositions) > 4:
        counts: dict[str, int] = {}
        for disposition in dispositions:
            label = str(disposition.get("disposition") or "unknown")
            counts[label] = counts.get(label, 0) + 1
        grouped: dict[str, dict[str, Any]] = {}
        for disposition in dispositions:
            evidence = disposition.get("source_evidence") or {}
            source = evidence.get("source") or {}
            signature = json.dumps({
                "disposition": disposition.get("disposition"),
                "reason_code": disposition.get("reason_code"),
                "reason": disposition.get("reason"),
                "source_type": source.get("type"),
                "source_reference": source.get("reference"),
                "known_at": evidence.get("known_at"),
                "receipt_id": evidence.get("receipt_id"),
            }, sort_keys=True, separators=(",", ":"))
            if signature not in grouped:
                grouped[signature] = {
                    **disposition,
                    "representative_context_id": disposition.get(
                        "context_id"),
                    "count": 0,
                }
            grouped[signature]["count"] += 1
        visible_dispositions = list(grouped.values())[:4]
        projection["context_dispositions"] = visible_dispositions
        projection["context_disposition_counts"] = counts
        projection["context_dispositions_omitted"] = (
            len(dispositions) - len(visible_dispositions))
        projection["context_dispositions_location"] = (
            "receipt.context_dispositions")
    contract = projection.get("selection_contract")
    if isinstance(contract, dict):
        claims = [item for item in contract.get("claims") or []
                  if isinstance(item, dict)]
        claim_groups: dict[str, dict[str, Any]] = {}
        for claim in claims:
            shared = {key: value for key, value in claim.items()
                      if key != "claim_id"}
            signature = json.dumps(
                shared, sort_keys=True, separators=(",", ":"))
            if signature not in claim_groups:
                claim_groups[signature] = {
                    **claim,
                    "representative_claim_id": claim.get("claim_id"),
                    "count": 0,
                }
            claim_groups[signature]["count"] += 1
        compact_contract = {key: contract.get(key) for key in (
            "selection_required", "deterministic_scenario_id",
            "selection_basis")}
        if contract.get("selection_required") is True:
            compact_contract.update({
                "instruction": (
                    "Rank eligible scenario_ids using cited claims and "
                    "counterevidence; give confidence, rationale, and what "
                    "would change the selection. Do not alter numbers, "
                    "support, or automation."),
                "scenarios": [{
                    key: ((list(scenario.get(key) or [])[:4])
                          if key == "claim_ids" else scenario.get(key))
                    for key in (
                        "scenario_id", "role", "support", "claim_ids",
                        "human_selection_eligible", "forecast_seal", "summary")
                    if scenario.get(key) is not None
                } | ({
                    "claim_count": len(scenario.get("claim_ids") or []),
                    "claim_ids_omitted": max(
                        0, len(scenario.get("claim_ids") or []) - 4),
                    "claim_ids_location": "receipt.selection_contract.scenarios",
                } if len(scenario.get("claim_ids") or []) > 4 else {}) | ({"evidence": {
                    key: value for key, value in
                    (scenario.get("derivation") or {}).items()
                    if value not in (None, False, [], {}, "not_applicable")
                }} if any(value not in (None, False, [], {}, "not_applicable")
                          for value in (scenario.get("derivation") or {}).values())
                     else {})
                    for scenario in contract.get("scenarios") or []
                    if isinstance(scenario, dict)],
                "claims": list(claim_groups.values())[:4],
                **({
                    "claim_count": len(claims),
                    "claims_omitted": len(claims) - min(
                        4, len(claim_groups)),
                    "claims_location": "receipt.selection_contract.claims",
                } if len(claims) > min(4, len(claim_groups)) else {}),
                "observation_evidence": contract.get(
                    "observation_evidence") or [],
            })
        projection["selection_contract"] = compact_contract
    projection.update({
        "projection": "compact",
        "receipt_path": str(path),
        "receipt_is_complete_and_sealed": True,
    })
    return {**payload, "publication": projection}


def apply_temporal_grounding(payload: dict[str, Any]) -> dict[str, Any]:
    """Echo the data boundary and wall clock on every data-bearing response."""
    from datetime import datetime, timezone

    result = dict(payload)
    now = datetime.now(timezone.utc)
    result["wall_clock_now"] = now.isoformat()
    series_end = result.get("series_end")
    frequency = result.get("frequency")
    if series_end is None:
        series = result.get("series") or []
        ends = [row.get("end") for row in series if isinstance(row, dict) and row.get("end")]
        if ends:
            series_end = max(ends)
        frequency = (result.get("schema") or {}).get("frequency")
    if series_end is None:
        rows = result.get("results") or []
        first = next((row.get("forecast", [])[0].get("timestamp")
                      for row in rows if isinstance(row, dict) and row.get("forecast")), None)
        frequency = frequency or ((result.get("task") or {}).get("schema") or {}).get("frequency")
        if first and frequency:
            from .temporal import frequency_step
            step = frequency_step(str(frequency))
            if step is not None:
                parsed = datetime.fromisoformat(str(first).replace("Z", "+00:00"))
                series_end = (parsed - step).isoformat()
    if series_end is not None:
        result["series_end"] = str(series_end)
        try:
            parsed_end = datetime.fromisoformat(str(series_end).replace("Z", "+00:00"))
            if parsed_end.tzinfo is None:
                parsed_end = parsed_end.replace(tzinfo=timezone.utc)
            gap = now - parsed_end.astimezone(timezone.utc)
            from .temporal import frequency_step
            step = frequency_step(str(frequency)) if frequency else None
            if step is not None and gap > step:
                result["staleness"] = (
                    f"The latest observation is {gap.days} days behind the wall clock "
                    f"({series_end} versus {result['wall_clock_now']})."
                )
        except (TypeError, ValueError):
            pass
    return result


def _triage_artifact_identity(payload: dict[str, Any]) -> dict[str, Any]:
    """Artifact identity fields the triage block carries verbatim.

    Ranking rules, preserved remainders, and artifact identity are canonical
    response fields copied deterministically — never facts an LLM must
    remember to restate from elsewhere in the payload.
    """
    identity = {key: payload.get(key) for key in ("forecast_id", "artifact_path")
                if payload.get(key) is not None}
    return {"artifact": identity} if identity else {}


def triage_wide_response(payload: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    """Bound a wide response while leaving the immutable artifact complete.

    The bounded view refreshes the one canonical ``triage`` block instead of
    attaching a second, differently-shaped summary: the ranking rule, the
    most notable series, the preserved remainder, and the artifact identity
    stay deterministic fields with one name each, whichever surface bounded
    the response.
    """
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) <= top_k:
        return payload
    ranked = sorted(rows, key=lambda row: (
        -float(row.get("notability", 0.0)), str(row.get("series", ""))))
    remainder = ranked[top_k:]
    tier_counts: dict[str, int] = {}
    from .support import result_support_tier
    for row in remainder:
        tier = str(result_support_tier(row) or "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    existing = (payload.get("triage")
                if isinstance(payload.get("triage"), dict) else {})
    return {
        **payload,
        "results": ranked[:top_k],
        "triage": {
            **existing,
            "series_count": len(rows), "returned": top_k,
            "ranking_rule": existing.get(
                "ranking_rule",
                "threshold crossing, then relative forecast movement"),
            "most_notable": (str(ranked[0].get("series"))
                             if isinstance(ranked[0], dict)
                             and ranked[0].get("series") is not None else
                             existing.get("most_notable")),
            "remainder_count": len(remainder),
            "remainder_tiers": dict(sorted(tier_counts.items())),
            "remainder_preserved": True,
            **_triage_artifact_identity(payload),
            "full_results": existing.get("full_results") or (
                "Use gnomon_get_artifact with series/fields/where/"
                "order_by/limit selectors on artifact_path."),
        },
    }


def compact_support_details(
    payload: dict[str, Any], *, force: bool = False,
) -> dict[str, Any]:
    """Keep claim-bearing support fields inline; sensitivity lives in artifact."""
    rows = payload.get("results")
    if not isinstance(rows, list) or (len(rows) <= 1 and not force):
        return payload
    compacted = []
    # The per-result support contract remains frozen for existing consumers.
    # Sensitivity is diagnostic bulk; the complete block remains in artifact.
    keep = {"status", "reasons", "grouped_reason_codes",
            "recovery_actions", "assumptions", "disclosures",
            "legacy_support", "measured_coverage"}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("support_assessment"), dict):
            compacted.append(row)
            continue
        assessment = row["support_assessment"]
        projected = {key: value for key, value in assessment.items()
                     if key in keep and value not in (None, [], {})}
        compacted.append({**row, "support_assessment": projected})
    return {**payload, "results": compacted}


def disclose_assumptions(payload: Any, assumptions: list[str]) -> Any:
    """Attach caller-level inferences to every result's support assessment.

    An inference the caller is not told about is a guess. These ride in the
    same `assumptions` list as `known_time_assumed`, so an agent reading the
    envelope finds them where it already looks. Payloads without a results
    list carry them at the top level instead.
    """
    if not assumptions or not isinstance(payload, dict):
        return payload
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return {**payload, "assumptions": assumptions}
    decorated = []
    for result in results:
        if not isinstance(result, dict):
            decorated.append(result)
            continue
        assessment = dict(result.get("support_assessment") or {})
        assessment["assumptions"] = list(assessment.get("assumptions", [])) + assumptions
        decorated.append({**result, "support_assessment": assessment})
    return {**payload, "results": decorated}


def _result_authority_projection(item: Any) -> dict[str, Any]:
    """Disambiguate path support only when another claim is weaker."""
    from .support import forecast_path_support_tier, result_support_tier

    path_tier = forecast_path_support_tier(item)
    result_tier = result_support_tier(item)
    if result_tier is None or result_tier == path_tier:
        return {}
    return {"support_scope": "forecast_path", "tier_floor": result_tier}


def forecast_summary(artifact: ForecastArtifact, path: Any) -> dict[str, Any]:
    """The compact forecast payload shared by the CLI and every adapter.

    The first forecast rows are inlined so an agent can quote numbers without
    a second read; the full series always lives in forecast.csv."""
    from .support import (
        artifact_headline, forecast_notability, payload_support_tier,
    )
    from .temporal_profile import compact_temporal_profile

    def response_facts(item: Any) -> dict[str, Any] | None:
        if not item.temporal_facts:
            return None
        facts = dict(item.temporal_facts)
        if facts.get("temporal_profile") and item.support not in {
                "best_effort", "unsupported", "invalid", "inconclusive"}:
            facts["temporal_profile"] = compact_temporal_profile(
                facts["temporal_profile"])
        else:
            facts.pop("temporal_profile", None)
        return facts

    payload = {
        "schema_version": "0.1",
        "status": "complete",
        "forecast_id": artifact.forecast_id,
        "artifact_path": str(path),
        # One deterministic sentence, template-generated from the
        # assessment, naming the weakest tier present — the sentence an
        # agent may relay verbatim. Required in every format.
        "headline": artifact_headline(artifact.results),
        **_forecast_temporal_boundary(artifact),
        "results": [
            {
                "series": item.series, "support": item.support,
                **_result_authority_projection(item),
                "support_assessment": item.support_assessment,
                "selected_model": item.selected_model,
                "interval_coverage": item.interval_coverage,
                "warnings": item.warnings,
                # Epistemic disclosures ride in every format: brief already
                # carried notes; full dropping them meant the verbose mode
                # disclosed less than the compact one.
                "notes": item.notes,
                "forecast_preview": item.forecast[:FORECAST_PREVIEW_ROWS],
                "forecast_rows": len(item.forecast),
                **({
                    "primary_forecast_preview":
                        item.primary_forecast[:FORECAST_PREVIEW_ROWS],
                    "primary_forecast_rows": len(item.primary_forecast),
                    "forecast_role": "context_conditioned_projection",
                    "primary_forecast_location":
                        "artifact.results[].primary_forecast",
                } if item.primary_forecast else {
                    "forecast_role": "primary_forecast",
                }),
                "threshold": item.threshold,
                "context": item.context,
                "context_outcome": item.context_outcome,
                **({"sensitivity_scenarios": [{
                    "events": scenario.get("events", []),
                    "support": scenario.get("support"),
                    "primary_forecast_changed": False,
                    "assumed_effect": scenario.get("assumed_effect"),
                    "assumed_effect_unit": scenario.get("assumed_effect_unit"),
                    "assumptions": scenario.get("assumptions", []),
                    "forecast_rows": len(scenario.get("forecast", [])),
                    "location": "artifact.results[].sensitivity_scenarios",
                } for scenario in item.sensitivity_scenarios]}
                   if item.sensitivity_scenarios else {}),
                "covariates": item.covariates,
                "notability": forecast_notability(item),
                "execution_identity": _execution_identity(artifact, item),
                **({"temporal_facts": response_facts(item)}
                   if item.temporal_facts else {}),
                **_model_assisted_summary(item),
            }
            for item in artifact.results
        ],
    }
    _attach_tsfm_on_ramp(payload, artifact)
    payload = _attach_multiseries_triage(payload)
    payload["tier_floor"] = payload_support_tier(payload)
    return payload


def _model_assisted_summary(item: Any) -> dict[str, Any]:
    """The lane's bounded response form: label, model, validation, and a
    points preview — the full points array stays in the artifact. Absent
    entirely when the series earned no lane, so untouched responses stay
    byte-identical."""
    lane = getattr(item, "model_assisted", None)
    if not lane:
        return {}
    points = list(lane.get("points") or [])
    return {"model_assisted": {
        "support": lane.get("support"),
        "selected_model": lane.get("selected_model"),
        "points_preview": points[:FORECAST_PREVIEW_ROWS],
        "points_total": len(points),
        "timestamps_match_primary_forecast": True,
        "validation": lane.get("validation"),
        "automation_eligible": False,
        "primary_forecast_unchanged": True,
        "location": "artifact.results[].model_assisted",
    }}


def _attach_tsfm_on_ramp(payload: dict[str, Any],
                         artifact: ForecastArtifact) -> None:
    """Turn the eligible-but-absent disclosure into an exact one-command path."""
    absent = [note for result in artifact.results for note in result.notes
              if note.startswith("No foundation-model candidate competed:")]
    if not absent:
        return
    preferred = "toto2_4m" if any("toto2_4m" in note for note in absent) \
        else None
    if preferred is None:
        return
    payload["tsfm_on_ramp"] = {
        "candidate": preferred,
        "reason": "eligible for at least one series but not installed",
        "command": f"gnomon tsfm install {preferred}",
        "mcp_tool_call": {
            "name": "gnomon_install_tsfm",
            "arguments": {"name": preferred},
        },
        "mcp_profile_required": "full",
        "admission": (
            "After installation the candidate enters the same out-of-sample "
            "contest; installation does not guarantee publication."
        ),
    }


def _compact_sensitivity_projection(items: list[dict[str, Any]]) \
        -> list[dict[str, Any]]:
    """Group numerically identical scenarios for a bounded wire response."""
    def compact_effect(effect: Any) -> dict[str, Any] | None:
        if not isinstance(effect, dict):
            return None
        distribution = effect.get("distribution") or {}
        provenance = effect.get("provenance") or {}
        return {
            "shape": effect.get("shape"),
            "distribution": {
                key: distribution.get(key) for key in (
                    "distribution", "location", "lower", "upper", "unit")
                if distribution.get(key) is not None
            },
            "provenance": {
                key: provenance.get(key) for key in (
                    "provenance_class", "observed", "known_at")
                if provenance.get(key) is not None
            },
        }

    grouped: dict[str, dict[str, Any]] = {}
    for scenario in items:
        assumptions = [str(value) for value in scenario.get("assumptions", [])
                       if not str(value).startswith("assumes ")]
        key = json.dumps({
            "support": scenario.get("support"),
            "assumed_effect": scenario.get("assumed_effect"),
            "assumed_effect_unit": scenario.get("assumed_effect_unit"),
            "assumptions": assumptions,
            "effect": scenario.get("effect"),
            "forecast": scenario.get("forecast") or [],
        }, sort_keys=True, default=str, separators=(",", ":"))
        events = [str(value) for value in scenario.get("events", [])]
        if key not in grouped:
            grouped[key] = {
                "events": [], "support": scenario.get("support"),
                "primary_forecast_changed": False,
                **({"assumed_effect": scenario.get("assumed_effect")}
                   if scenario.get("assumed_effect") is not None else {}),
                **({"assumed_effect_unit": scenario.get(
                    "assumed_effect_unit")}
                   if scenario.get("assumed_effect_unit") is not None else {}),
                "assumptions": assumptions,
                **({"effect": compact_effect(scenario.get("effect")),
                    "consequence": scenario.get("consequence"),
                    "consequence_summary": scenario.get(
                        "consequence_summary")}
                   if (scenario.get("effect") and scenario.get("support") ==
                       "prior_assisted_structural") else {}),
                "automation_eligible": bool(
                    scenario.get("automation_eligible", False)),
                "selection_eligible": bool(
                    scenario.get("selection_eligible", True)),
                **({"intervals_available": False}
                   if scenario.get("intervals_available") is False else {}),
                "forecast_rows": len(scenario.get("forecast", [])),
                "location": "artifact.results[].sensitivity_scenarios",
            }
        grouped[key]["events"].extend(events)
    projected = []
    for scenario in grouped.values():
        events = sorted(set(scenario["events"]))
        scenario["event_count"] = len(events)
        scenario["events"] = events[:4]
        if len(events) > 4:
            scenario["events_omitted"] = len(events) - 4
        projected.append(scenario)
    return projected


def brief_summary(artifact: ForecastArtifact, path: Any) -> dict[str, Any]:
    """The compact forecast payload: q50 path, one q10–q90 interval, the
    selection, and every disclosure — roughly summary.md as JSON.

    What it drops is bulk only: the extra quantile levels, the raw
    `point` path beside its bias correction, and the context/covariate
    gate detail (all still in the artifact on disk, which is written
    unchanged). What it may never drop is epistemics: the support state,
    every warning, every abstention reason, every recovery action, and
    every disclosure ride along verbatim — an abstention serialises the
    same structured support assessment full mode carries. Hiding
    disclosures is the one thing this codebase exists to not do.
    """
    from .support import forecast_notability, payload_support_tier
    from .temporal_profile import compact_temporal_profile

    def context_outcome_projection(item: Any) -> dict[str, Any] | None:
        if not item.context_outcome:
            return None
        projected = dict(item.context_outcome)
        changed = projected.pop("primary_forecast_changed", None)
        if changed is not None:
            # The persisted v0.2 field means that the selected conditional
            # projection differs from the history-only primary. Its old name
            # is easily misread as mutation, so the public brief names the
            # distinction while the frozen artifact remains byte-compatible.
            projected["selected_projection_differs_from_primary"] = bool(changed)
            projected["canonical_primary_preserved"] = bool(
                projected.get("canonical_primary_preserved", True))
        canonical_preserved = bool(
            projected.get("canonical_primary_preserved", True))
        projection_differs = bool(
            projected.get("selected_projection_differs_from_primary", False))
        automation_eligible = projected.get("automation_eligible") is True
        projected["authority_summary"] = (
            ("Context changed the selected conditional projection; "
             "the canonical primary forecast remains preserved. "
             if projection_differs else
             "Context did not change the canonical primary forecast. ") +
            ("An explicit automation policy is still required."
             if automation_eligible else
             "Context evidence alone cannot authorize automation.")
        )
        projected["canonical_primary_preserved"] = canonical_preserved
        # Repeated operational events can number in the hundreds.  Their
        # individual receipts remain in the immutable artifact; the agent
        # response carries the decision-relevant aggregate and a bounded
        # preview so context does not crowd the forecast out of its budget.
        events = list(projected.get("events") or [])
        if len(events) > 4:
            projected["event_count"] = len(events)
            projected["events"] = events[:4]
            projected["events_omitted"] = len(events) - 4
            projected["events_location"] = (
                "artifact.results[].context_outcome.events")
        context_evidence = list(projected.get("context_evidence") or [])
        if len(context_evidence) > 4:
            evidence_groups: dict[str, dict[str, Any]] = {}
            for evidence in context_evidence:
                source = evidence.get("source") or {}
                # Missing provenance remains distinct; only exact validated
                # source identities are safe to collapse for the wire.
                signature_values = {
                    "source_type": source.get("type"),
                    "source_reference": source.get("reference"),
                    "known_at": evidence.get("known_at"),
                    "receipt_id": evidence.get("receipt_id"),
                }
                if not source.get("reference"):
                    signature_values["context_id"] = evidence.get(
                        "context_id")
                signature = json.dumps(
                    signature_values, sort_keys=True, separators=(",", ":"))
                if signature not in evidence_groups:
                    evidence_groups[signature] = {
                        **evidence,
                        "representative_context_id": evidence.get(
                            "context_id"),
                        "count": 0,
                    }
                evidence_groups[signature]["count"] += 1
            evidence_preview = list(evidence_groups.values())[:4]
            projected["context_evidence_count"] = len(context_evidence)
            projected["context_evidence"] = evidence_preview
            projected["context_evidence_omitted"] = \
                len(context_evidence) - len(evidence_preview)
            projected["context_evidence_location"] = (
                "artifact.results[].context_outcome.context_evidence")
        dispositions = list(projected.get("dispositions") or [])
        if len(dispositions) > 4:
            counts: dict[str, int] = {}
            for disposition in dispositions:
                label = str(disposition.get("disposition") or "unknown")
                counts[label] = counts.get(label, 0) + 1
            projected["disposition_counts"] = counts
            projected["dispositions"] = dispositions[:4]
            projected["dispositions_omitted"] = len(dispositions) - 4
            projected["dispositions_location"] = (
                "artifact.results[].context_outcome.dispositions")
        hypotheses = list(projected.get("hypotheses") or [])
        if hypotheses:
            signatures: dict[str, dict[str, Any]] = {}
            for hypothesis in hypotheses:
                signature_fields = {
                    key: hypothesis.get(key) for key in (
                        "direction", "duration", "duration_steps",
                        "effect_family", "entity_kind", "entity_scope",
                        "grounding_status", "numeric_status",
                        "may_affect_numbers", "may_affect_primary_forecast",
                    ) if (key in hypothesis and hypothesis.get(key) is not None
                          and hypothesis.get(key) != "unknown")
                }
                signature = json.dumps(
                    signature_fields, sort_keys=True, separators=(",", ":"))
                if signature not in signatures:
                    signatures[signature] = {
                        **signature_fields, "count": 0,
                        "representative_event_id": hypothesis.get("event_id"),
                    }
                signatures[signature]["count"] += 1
            projected["hypothesis_count"] = len(hypotheses)
            projected["hypotheses"] = list(signatures.values())
            projected["hypotheses_location"] = (
                "artifact.results[].context_outcome.hypotheses")
        if projected.get("conditional_forecasts_produced") == 0:
            projected.pop("conditional_forecasts_produced", None)
        excluded = list(projected.get("excluded") or [])
        if len(excluded) > 4:
            reason_counts: dict[str, int] = {}
            reason_examples: dict[str, dict[str, Any]] = {}
            for exclusion in excluded:
                reason = str(exclusion.get("reason") or "unspecified")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                reason_examples.setdefault(reason, exclusion)
            projected["excluded"] = list(reason_examples.values())[:4]
            projected["excluded_count"] = len(excluded)
            projected["excluded_reason_counts"] = reason_counts
            projected["excluded_omitted"] = len(excluded) - 4
            projected["excluded_location"] = (
                "artifact.results[].context_outcome.excluded")
        return projected

    def response_facts(item: Any) -> dict[str, Any] | None:
        if not item.temporal_facts:
            return None
        facts = dict(item.temporal_facts)
        if facts.get("temporal_profile") and item.support not in {
                "best_effort", "unsupported", "invalid", "inconclusive"}:
            facts["temporal_profile"] = compact_temporal_profile(
                facts["temporal_profile"])
        else:
            facts.pop("temporal_profile", None)
        return facts
    results = []
    for item in artifact.results:
        preview, omitted_middle = _bounded_forecast_preview(item.forecast)
        results.append({
            "series": item.series,
            "support": item.support,
            **_result_authority_projection(item),
            "selected_model": item.selected_model,
            "interval_coverage": item.interval_coverage,
            # Verbatim, never summarised: the same objects full mode carries.
            "warnings": item.warnings,
            "support_assessment": item.support_assessment,
            "notes": item.notes,
            "forecast": [
                {"timestamp": row["timestamp"], "q50": row["q50"],
                 "q10": row["q10"], "q90": row["q90"],
                 # The unstrippable label rides on every row in every
                 # format; brief may drop quantile levels, never the tier.
                 **({"tier": row["tier"]} if "tier" in row else {})}
                for row in preview
            ],
            # The row count survives even when the budget trims the rows.
            "forecast_rows": len(item.forecast),
            **({"forecast_preview": {
                "strategy": "first_and_last",
                "returned_rows": len(preview),
                "omitted_middle_rows": omitted_middle,
                "full_path": "artifact.results[].forecast",
            }} if omitted_middle else {}),
            **({
                "primary_forecast_preview": [
                    {"timestamp": row["timestamp"], "q50": row["q50"],
                     "q10": row["q10"], "q90": row["q90"],
                     **({"tier": row["tier"]} if "tier" in row else {})}
                    for row in item.primary_forecast[:FORECAST_PREVIEW_ROWS]
                ],
                "primary_forecast_rows": len(item.primary_forecast),
                "forecast_role": "context_conditioned_projection",
                "primary_forecast_location":
                    "artifact.results[].primary_forecast",
            } if item.primary_forecast else {
                "forecast_role": "primary_forecast",
            }),
            "notability": forecast_notability(item),
            "execution_identity": _execution_identity(artifact, item),
            **({"temporal_facts": response_facts(item)}
               if item.temporal_facts else {}),
            **({"threshold": item.threshold} if item.threshold else {}),
            **({"context_outcome": context_outcome_projection(item)}
               if item.context_outcome else {}),
            **({"sensitivity_scenarios": _compact_sensitivity_projection(
                item.sensitivity_scenarios)}
               if item.sensitivity_scenarios else {}),
            **_model_assisted_summary(item),
        })
    from .support import artifact_headline
    payload = {
        "schema_version": "0.1",
        "status": "complete",
        "format": "brief",
        "forecast_id": artifact.forecast_id,
        "artifact_path": str(path),
        "headline": artifact_headline(artifact.results),
        **_forecast_temporal_boundary(artifact),
        "note": (
            "Brief output: q50 with the q10-q90 interval per step. The full "
            "artifact (all quantile levels, evidence, lineage) is on disk at "
            "artifact_path, unchanged."
        ),
        "results": results,
    }
    _attach_tsfm_on_ramp(payload, artifact)
    payload = _attach_multiseries_triage(payload)
    payload["tier_floor"] = payload_support_tier(payload)
    return payload


def _execution_identity(artifact: ForecastArtifact, item: Any) -> dict[str, Any]:
    """Expose the persisted evaluation/publication seam in the first response."""
    candidate = next((evidence.payload for evidence in artifact.evidence
                      if evidence.kind == "final_candidate"
                      and evidence.series == item.series), None)
    evaluated = dict(candidate) if candidate else {
        "kind": "builtin", "name": item.selected_model,
        "data_fingerprint": artifact.source_fingerprint,
    }
    published = {"name": item.selected_model,
                 "data_fingerprint": artifact.source_fingerprint}
    return {
        "evaluated": evaluated,
        "published": published,
        "publish_matches_evaluated": (
            evaluated.get("name") == published["name"]),
        "runtime_version": artifact.runtime_version or None,
    }


def _attach_multiseries_triage(payload: dict[str, Any]) -> dict[str, Any]:
    """Bounded, typed summary for wide forecasts; the artifact keeps all rows."""
    results = [item for item in payload.get("results", [])
               if isinstance(item, dict)]
    if len(results) <= 3:
        return payload
    ranked = sorted(results, key=lambda item: (
        -float(item.get("notability") or 0.0), str(item.get("series") or "")))
    notable = [{"series": item.get("series"),
                "notability": item.get("notability"),
                "support": item.get("support")}
               for item in ranked[:3]]
    return {**payload, "triage": {
        "series_count": len(results),
        "ranking_rule": "threshold crossing, then relative path movement",
        "most_notable": notable[0]["series"] if notable else None,
        "notable": notable,
        "remainder_count": len(results) - len(notable),
        # A bounded response is not data loss: every omitted series remains
        # addressable in the immutable artifact named below.
        "remainder_preserved": True,
        **_triage_artifact_identity(payload),
        "full_results": {
            "tool_call": {"name": "gnomon_get_artifact", "arguments": {
                "artifact_path": payload.get("artifact_path"),
                "order_by": "notability", "limit": len(results),
            }},
        },
    }}


def _forecast_temporal_boundary(artifact: ForecastArtifact) -> dict[str, Any]:
    """Derive the last observed instant from the first forecast grid point."""
    from datetime import datetime
    from .temporal import frequency_step

    first = next((item.forecast[0].get("timestamp") for item in artifact.results
                  if item.forecast), None)
    step = frequency_step(artifact.task.schema.frequency)
    if first is None or step is None:
        return {"frequency": artifact.task.schema.frequency}
    parsed = datetime.fromisoformat(str(first).replace("Z", "+00:00"))
    return {"series_end": (parsed - step).isoformat(),
            "frequency": artifact.task.schema.frequency}


_CAPABILITIES_PROSE_LIMIT = 100


def _brief_capabilities(full: dict[str, Any]) -> dict[str, Any]:
    """The default capabilities view, sized to the response budget.

    Every top-level section of the full payload is present and every
    capability *name* (tools, models, operators, macros, features,
    frequencies, flags) survives verbatim — a brief view that hid a
    capability would make the command lie about the build. What is
    elided, and said to be elided, is prose: long explanatory strings
    and the per-operator / per-TSFM metadata blocks, all of which the
    caller gets verbatim with ``format: "full"`` or ``sections``.
    """
    elided: list[str] = []

    def compact(node: Any, path: str) -> Any:
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                where = f"{path}.{key}" if path else str(key)
                if isinstance(value, str) and len(value) > _CAPABILITIES_PROSE_LIMIT:
                    elided.append(where)
                    continue
                out[key] = compact(value, where)
            return out
        if isinstance(node, list):
            return [compact(item, path) for item in node]
        return node

    brief: dict[str, Any] = {}
    for key, value in full.items():
        if key in ("operators", "macros") and isinstance(value, dict):
            # The names are the capability; the per-entry contracts
            # (summary, minimum data, abstention policy) are detail.
            brief[key] = {"available": sorted(value)}
            elided.append(f"{key}.<details>")
        elif key == "models" and isinstance(value, dict):
            models = dict(value)
            matrix = models.get("tsfm_capabilities")
            if isinstance(matrix, dict):
                models["tsfm_capabilities"] = {"models": sorted(matrix)}
                elided.append("models.tsfm_capabilities.<details>")
            statsforecast = models.get("statsforecast")
            if isinstance(statsforecast, dict):
                # Retain every machine-actionable availability and model
                # name; move explanatory install/admission prose to the full
                # or named-section view so one optional plugin cannot break
                # the bounded default response.
                # In brief form the model names are the capability. Install
                # state, version range, activation, and admission policy stay
                # verbatim in the full/named models view.
                prefix = "statsforecast_"
                names = sorted(statsforecast.get("models") or [])
                models["statsforecast"] = {
                    "prefix": prefix,
                    "models": [name.removeprefix(prefix) for name in names],
                }
                elided.append("models.statsforecast.<details>")
            brief[key] = compact(models, "models")
        elif key == "workspace" and isinstance(value, dict):
            # Absolute checkout paths are environment detail, not a
            # capability. Repeating cwd inside default_output_dir made the
            # supposedly bounded brief depend on the host path length (and
            # overflow in CI). The relative default is directly executable;
            # exact absolute paths remain available from the full or named
            # workspace section.
            brief[key] = {"default_output_dir": "./gnomon-output"}
            elided.append("workspace.<absolute_paths>")
        elif key == "general_frequencies" and isinstance(value, dict):
            # The regex is the executable capability; its prose restatement
            # is detail available in the named/full view.
            brief[key] = {"pattern": value.get("pattern")}
            elided.append("general_frequencies.<details>")
        elif key == "product_contract" and isinstance(value, dict):
            # Keep the deployment identity and every withheld public claim in
            # the ambient view; positioning prose stays in the explicit
            # section. This remains actionable while fitting even the broad
            # full-profile tool list inside the fixed response budget.
            brief[key] = {
                "default_mcp_profile": value.get("default_mcp_profile"),
                "offline_builtin_runtime": value.get("offline_builtin_runtime"),
                "current_evidence_release": value.get("current_evidence_release"),
                "withheld_claims": sorted(
                    name for name in (
                        "forecast_superiority", "agent_choice_lift",
                        "regulatory_certification",
                    ) if value.get(name) in {"not_established", "not_claimed"}
                ),
            }
            elided.append("product_contract.<positioning_details>")
        elif key == "features" and isinstance(value, dict):
            # A boolean map repeats JSON punctuation and ``true`` for every
            # feature. Preserve every name and state in two lists; callers
            # requesting the section or full view still receive the exact
            # map. This recovered enough budget for the product claim
            # contract without hiding capabilities.
            brief[key] = {
                "enabled": sorted(name for name, enabled in value.items()
                                  if enabled is True),
                "disabled": sorted(name for name, enabled in value.items()
                                   if enabled is False),
            }
            elided.append("features.<boolean_map>")
        else:
            brief[key] = compact(value, key)
    brief["view"] = {
        "format": "brief",
        "sections_available": sorted(full),
        "elided": sorted({path.split(".", 1)[0] for path in elided}),
        "note": "full or sections",
    }
    return brief
