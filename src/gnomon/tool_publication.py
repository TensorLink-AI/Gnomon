"""Legacy publication projection over immutable forecast artifacts."""

from __future__ import annotations

import json

from typing import Any

from .contracts import ForecastArtifact, GnomonError
from .tool_input import (
    _parse_as_of,
)


def _attach_publication(payload: dict[str, Any], artifact: ForecastArtifact,
                        path: Any, arguments: dict[str, Any], *,
                        result_index: int = 0) -> None:
    """MCP projection boundary; forecast artifacts remain byte-immutable."""
    mode = str(arguments.get("publication_mode") or "strict")
    dossiers = arguments.get("temporal_dossiers") or []
    submission = arguments.get("context_submission") or {}
    if not isinstance(submission, dict):
        raise GnomonError("INVALID_ARGUMENTS", "context_submission must be an object")
    submission = dict(submission)
    direct_rejections = arguments.get("context_rejections")
    if direct_rejections is not None:
        if not isinstance(direct_rejections, list):
            raise GnomonError(
                "INVALID_ARGUMENTS", "context_rejections must be an array")
        allowed_rejection_fields = {
            "context_id", "event_id", "reason_code", "reason", "source_span"}
        required_rejection_fields = {
            "context_id", "reason_code", "reason", "source_span"}
        normalized_rejections = []
        trusted_source = str(arguments.get(
            "_trusted_context_source_text") or "").strip()
        for index, item in enumerate(direct_rejections, 1):
            if not isinstance(item, dict):
                raise GnomonError(
                    "INVALID_ARGUMENTS",
                    "context_rejections items must be objects",
                    {"item_index": index})
            unknown = sorted(set(item) - allowed_rejection_fields)
            normalized = dict(item)
            event_alias = str(normalized.pop("event_id", "") or "").strip()
            context_id = str(normalized.get("context_id") or "").strip()
            if event_alias and context_id and event_alias != context_id:
                raise GnomonError(
                    "INVALID_ARGUMENTS",
                    "context_rejections event_id alias conflicts with context_id",
                    {"item_index": index})
            if event_alias:
                normalized["context_id"] = event_alias
            if trusted_source:
                normalized.setdefault("context_id", f"context-rejection-{index}")
                normalized.setdefault("reason_code", "context_unresolved")
                normalized.setdefault(
                    "reason", "Context was rejected by the agent compiler; "
                    "see the typed reason code and host-bound source.")
                normalized.setdefault("source_span", trusted_source)
            missing = sorted(key for key in required_rejection_fields
                             if not str(normalized.get(key) or "").strip())
            if unknown or missing:
                raise GnomonError(
                    "INVALID_ARGUMENTS",
                    "context_rejections require context_id, reason_code, "
                    "reason, and verbatim source_span",
                    {"item_index": index, "unknown_fields": unknown,
                     "missing_fields": missing})
            normalized_rejections.append(normalized)
        submission["rejections"] = [
            *(submission.get("rejections") or []), *normalized_rejections]
    raw_proposal = submission.get("proposal")
    raw_model_candidate = submission.get("model_candidate")
    deterministic_compile = submission.get("compile")
    if deterministic_compile not in (None, "deterministic_linear"):
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "context_submission.compile supports only deterministic_linear")
    if raw_model_candidate is not None and raw_proposal is not None:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "context_submission accepts proposal or model_candidate, not both")
    if deterministic_compile and (raw_proposal is not None
                                  or raw_model_candidate is not None
                                  or submission.get("transformations")):
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "deterministic context compilation cannot be combined with an "
            "agent proposal or caller-authored transformations")
    selection = arguments.get("scenario_selection")
    policy = arguments.get("automation_policy")
    if mode == "strict" and not (dossiers or raw_proposal
                                  or deterministic_compile
                                  or submission.get("transformations")
                                  or submission.get("rejections")
                                  or arguments.get("_context_was_supplied")
                                  or selection or policy):
        return
    if result_index < 0 or result_index >= len(artifact.results):
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "Publication result index is outside the artifact.")
    from .publication import (compile_dossier_for_result, publish_result,
                              write_publication)
    artifact_payload = artifact.to_dict()
    result = artifact_payload["results"][result_index]
    # Ungrouped artifacts use ``__default__`` as their storage series key.
    # Preserve the semantic target supplied by the caller for claim-ownership
    # checks at the publication boundary; this metadata never changes points.
    result["target_identity"] = str(arguments.get("target_column") or "")
    # Candidate plausibility is defined at the last *observed* boundary.  Load
    # that boundary through the exact same repair/as-of/store seam as the
    # immutable forecast.  Using the primary's future q50 path here makes a
    # candidate's validity depend on a competing forecast and can reject a
    # perfectly plausible conditional path.
    from .pipeline import load_stage
    publication_input = load_stage(
        arguments["input"], time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
        store_path=arguments.get("store_path"),
        repair=arguments.get("repair", "safe"),
        regrid=arguments.get("regrid"))
    result_series = artifact.results[result_index].series
    observations = publication_input.groups.get(result_series)
    if observations is None and len(publication_input.groups) == 1:
        observations = next(iter(publication_input.groups.values()))
    if not observations:
        raise GnomonError(
            "EMPTY_SNAPSHOT",
            "Publication validation could not resolve the forecast's observed history.")
    governed_history = [float(item.value) for item in observations]
    model_candidate_paths: list[list[float]] | None = None
    model_candidate_diagnostics: dict[str, Any] | None = None
    if raw_model_candidate is not None:
        if not isinstance(raw_model_candidate, dict):
            raise GnomonError(
                "INVALID_ARGUMENTS", "model_candidate must be an object")
        allowed = {
            "source_spans", "quantiles", "sample_paths", "rationale",
            "temperature", "governed_fallback",
        }
        unknown = sorted(set(raw_model_candidate) - allowed)
        if unknown:
            raise GnomonError(
                "INVALID_ARGUMENTS", "model_candidate has unknown fields",
                {"unknown_fields": unknown})
        governed_fallback = raw_model_candidate.get("governed_fallback")
        if governed_fallback not in {
                None, "categorical_state_mapping_not_admitted"}:
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "model_candidate.governed_fallback is unsupported")
        if governed_fallback is not None:
            from .llm_dossier import verify_temporal_dossier_seal
            fallback_proven = any(
                verify_temporal_dossier_seal(item)
                and (item.get("candidate_critique") or {}).get(
                    "candidate_origin") ==
                    "governed_categorical_state_mapping"
                and (item.get("candidate_critique") or {}).get(
                    "selection_eligible") is False
                for item in dossiers if isinstance(item, dict))
            if not fallback_proven:
                raise GnomonError(
                    "INVALID_ARGUMENTS",
                    "model_candidate.governed_fallback requires a sealed "
                    "failed categorical mapping dossier")
        context_text = str(submission.get("text") or "")
        known_at = str(submission.get("known_at") or "")
        spans = raw_model_candidate.get("source_spans")
        if (not context_text or not known_at or not isinstance(spans, list)
                or not 1 <= len(spans) <= 8
                or any(not isinstance(span, str) or not span.strip()
                       or span not in context_text for span in spans)):
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "model_candidate requires text, known_at, and 1-8 exact "
                "source_spans copied from text")
        quantiles = raw_model_candidate.get("quantiles")
        sample_paths = raw_model_candidate.get("sample_paths")
        if (quantiles is None) == (sample_paths is None):
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "model_candidate requires exactly one of quantiles or sample_paths")
        forecast_rows = result.get("primary_forecast") or result.get("forecast") or []
        future_timestamps = [str(row.get("timestamp")) for row in forecast_rows]
        claim_ids = [f"claim-{index}" for index in range(1, len(spans) + 1)]
        if sample_paths is not None:
            if (not isinstance(sample_paths, list)
                    or not 3 <= len(sample_paths) <= 16):
                raise GnomonError(
                    "INVALID_ARGUMENTS",
                    "model_candidate.sample_paths requires 3-16 paths")
            from .agent_context import candidate_from_sampled_paths
            serialized = [json.dumps({"forecast_path": {
                "values": path,
                "rationale": str(raw_model_candidate.get("rationale") or ""),
            }}) for path in sample_paths]
            candidate, model_candidate_diagnostics = (
                candidate_from_sampled_paths(
                    serialized, future_timestamps,
                    history_values=governed_history))
            if candidate is None:
                raise GnomonError(
                    "INVALID_ARGUMENTS",
                    "model_candidate.sample_paths did not contain a valid "
                    "host-grid-bound path",
                    {"diagnostics": model_candidate_diagnostics})
            model_candidate_paths = candidate.pop("_validated_sample_paths")
            candidate.pop("_selected_claim_ids", None)
        else:
            candidate = {
                "quantiles": quantiles,
                "rationale": str(raw_model_candidate.get("rationale") or
                                 "Caller-supplied model forecast candidate."),
            }
        candidate["claim_ids"] = claim_ids
        raw_proposal = {
            "events": [],
            "claims": [{
                "source_span": span,
                "relation": "unknown",
                "effective_start": None,
                "effective_end": None,
                "timing_status": "atemporal_context",
                "mechanism": "model-authored forecast prior",
                "confidence": 0.5,
            } for span in spans],
            "hypotheses": [],
            "effect_proposal": None,
            "forecast_candidate": candidate,
            "covariate_tables": [],
            "transformations": [],
            "observation_interpretations": [],
        }
    raw_context_rejections = submission.get("rejections") or []
    if not isinstance(raw_context_rejections, list):
        raise GnomonError(
            "INVALID_ARGUMENTS", "context_submission.rejections must be a list")
    if len(raw_context_rejections) > 16:
        raise GnomonError(
            "INVALID_ARGUMENTS", "context_submission.rejections is limited to 16 items")
    result["context_rejections"] = []
    for index, item in enumerate(raw_context_rejections, 1):
        if isinstance(item, str):
            code, separator, reason = item.partition(":")
            result["context_rejections"].append({
                "context_id": f"context-submission-{index}",
                "reason_code": code.strip() if separator else "context_unresolved",
                "reason": reason.strip() if separator else item.strip(),
            })
        elif isinstance(item, dict):
            result["context_rejections"].append({
                "context_id": str(item.get("context_id") or
                                  f"context-submission-{index}"),
                "reason_code": str(item.get("reason_code") or
                                   "context_unresolved"),
                "reason": str(item.get("reason") or
                              "Supplied context could not be grounded or executed."),
                **({"source_span": str(item["source_span"])}
                   if item.get("source_span") else {}),
            })
        else:
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "context_submission.rejections items must be strings or objects")

    def recurrence_observations_for(target: str) -> list[Any]:
        from .pipeline import load_stage
        loaded = load_stage(
            arguments["input"], time_column=arguments["time_column"],
            target_column=target, series_column=arguments.get("series_column"),
            frequency=arguments.get("frequency"),
            as_of=_parse_as_of(arguments.get("as_of")),
            store_path=arguments.get("store_path"),
            regrid=arguments.get("regrid"),
            repair=arguments.get("repair", "safe"))
        if len(loaded.groups) != 1:
            raise GnomonError(
                "AMBIGUOUS_RECURSIVE_HISTORY",
                "Recursive context execution requires exactly one series per target.")
        return list(next(iter(loaded.groups.values())))
    if deterministic_compile:
        context_text = str(submission.get("text") or "")
        known_at = str(submission.get("known_at") or "")
        if not context_text or not known_at:
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "deterministic context compilation requires text and known_at")
        forecast_timestamps = [str(row["timestamp"])
                               for row in result.get("forecast") or []
                               if isinstance(row, dict) and row.get("timestamp")]
        from .relationship_text import compile_linear_relationship_text
        compiled_relationship = compile_linear_relationship_text(
            context_text,
            target_name=str(arguments.get("target_column") or ""),
            cutoff=known_at, future_timestamps=forecast_timestamps)
        if compiled_relationship is None:
            result["context_rejections"].append({
                "context_id": "deterministic-linear-compiler",
                "reason_code": "DETERMINISTIC_RELATIONSHIP_UNRESOLVED",
                "reason": (
                    "The cited text was not a complete, mechanically "
                    "checkable linear lag specification with a full future "
                    "driver schedule. No partial arithmetic was executed."),
                "source_span": context_text,
            })
        else:
            raw_proposal, compilation_kind = compiled_relationship
            submission["transformations"] = list(
                raw_proposal.get("transformations") or [])
            submission["compiler"] = (
                "gnomon:deterministic_linear:" + compilation_kind)
    if raw_proposal is not None:
        context_text = str(submission.get("text") or "")
        known_at = str(submission.get("known_at") or "")
        if not context_text or not known_at:
            raise GnomonError("INVALID_ARGUMENTS",
                              "context_proposal requires context_text and context_known_at")
        dossier, dossier_rejections = compile_dossier_for_result(
            raw_proposal, context_text=context_text, known_at=known_at,
            result=result,
            compiler_model=str(submission.get("compiler") or "agent"),
            history=governed_history,
            prefer_explicit_forecast_candidate=(
                raw_model_candidate is not None))
        if model_candidate_paths is not None:
            if not isinstance(dossier.get("forecast_candidate"), dict):
                # An optional interpretation candidate must not erase a
                # successfully computed immutable primary. Preserve the
                # validator's reasons as a typed disposition so an agent can
                # repair the candidate or simply use the primary.
                result["context_rejections"].append({
                    "context_id": "model-authored-forecast-candidate",
                    "reason_code": "model_candidate_validation_failed",
                    "reason": "The supplied model-authored candidate did not "
                              "pass the governed forecast-candidate contract.",
                    "violations": dossier_rejections[:8],
                    "sampling_diagnostics": model_candidate_diagnostics,
                })
            else:
                from .agent_context import sample_path_stability
                from .llm_dossier import attach_host_candidate_elicitation
                temperature = raw_model_candidate.get("temperature", 1.0)
                try:
                    temperature = float(temperature)
                    stability = sample_path_stability(
                        model_candidate_paths,
                        governed_history)
                    dossier = attach_host_candidate_elicitation(
                        dossier, requested_paths=len(sample_paths),
                        accepted_paths=len(model_candidate_paths),
                        aggregation="linear_empirical_marginal_q10_q50_q90",
                        temperature=temperature, stability=stability,
                        request_mode="batch_request",
                        sample_paths=model_candidate_paths,
                        governed_fallback=governed_fallback)
                except (TypeError, ValueError) as exc:
                    raise GnomonError("INVALID_ARGUMENTS", str(exc)) from exc
        dossiers = [*dossiers, dossier]
    transformations = submission.get("transformations") or []
    if transformations:
        from .context_intelligence import (
            compile_transformation,
            execute_transformation,
            load_recurrence_history,
        )
        claims = [claim for dossier in dossiers
                  for claim in dossier.get("claims") or []]
        claim_ids = [str(claim.get("claim_id")) for claim in claims
                     if claim.get("claim_id")]
        claim_spans = {str(claim.get("claim_id")): str(
            claim.get("source_span") or "") for claim in claims}
        cutoff = str(submission.get("known_at") or "")
        if not cutoff:
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "context transformations require context_submission.known_at")
        result["transformation_candidates"] = []
        result["transformation_rejections"] = []
        if len(transformations) > 6:
            result["transformation_rejections"].append({
                "transformation_id": "transformation-overflow",
                "reason_code": "bounded_transformation_overflow",
                "reason": "Only the first six transformations were evaluated.",
            })
        for index, item in enumerate(transformations[:6], 1):
            wrapper = item if isinstance(item, dict) else {}
            compiled, critique = compile_transformation(
                wrapper.get("transformation", wrapper),
                series=list((wrapper.get("series_values") or {}).keys()),
                claim_ids=claim_ids, cutoff=cutoff,
                units=wrapper.get("units"), repair=wrapper.get("repair"),
                claim_spans=claim_spans)
            if compiled is None:
                result["transformation_rejections"].append({
                    "transformation_id": f"transformation-{index}",
                    "reason_code": "transformation_validation_failed",
                    "reason": "Declarative transformation was rejected.",
                    "violations": critique["violations"],
                })
                continue
            try:
                target_history, driver_history = load_recurrence_history(
                    compiled.get("expression"),
                    wrapper.get("historical_series_segments"),
                    target_name=str(arguments["target_column"]),
                    observations_for=recurrence_observations_for,
                    verified_claim_ids=claim_ids,
                    verified_claim_spans=claim_spans)
                candidate = execute_transformation(
                    compiled,
                    primary=(result.get("primary_forecast") or
                             result.get("forecast") or []),
                    series_values=wrapper.get("series_values"),
                    historical_validation=wrapper.get("historical_validation"),
                    claim_spans=claim_spans,
                    history_values=target_history,
                    history_series=driver_history)
                if wrapper.get("historical_series_segments"):
                    candidate["validation"]["recurrence_history_source"] = (
                        "document_cited_segments")
                    candidate["validation"]["document_history_series"] = sorted(
                        wrapper["historical_series_segments"])
            except (ValueError, GnomonError) as exc:
                result["transformation_rejections"].append({
                    "transformation_id": compiled["transformation_id"],
                    "reason_code": getattr(exc, "code", "transformation_execution_failed"),
                    "reason": str(exc),
                    "violations": [getattr(exc, "as_dict", lambda: {})()],
                })
                continue
            result["transformation_candidates"].append(candidate)
    candidate_outcome_evidence = None
    if mode == "best_effort" and arguments.get("project"):
        from .tracking import TrackingStore
        cutoff = (arguments.get("as_of") or artifact.task.as_of
                  or artifact.created_at)
        if cutoff is not None:
            series_name = artifact.results[result_index].series
            candidate_outcome_evidence = TrackingStore().candidate_outcome_summary(
                str(arguments["project"]), series=str(series_name),
                resolved_before=str(cutoff))
    calibration_evidence = None
    if isinstance(policy, dict) and policy.get("action_tier") is not None:
        series_name = artifact.results[result_index].series
        rolling = next((
            item for item in artifact_payload.get("evidence") or []
            if item.get("kind") == "rolling_evaluation"
            and item.get("series") == series_name
            and not str(item.get("evidence_id") or "").endswith(":prefix")
        ), None)
        if rolling is not None:
            evidence_payload = rolling.get("payload") or {}
            quantiles = tuple(artifact.task.quantiles)
            calibration_evidence = {
                "artifact_id": artifact.forecast_id,
                "series": series_name,
                "selected_model": result.get("selected_model"),
                "horizon": artifact.task.horizon,
                "nominal_coverage": (
                    float(max(quantiles) - min(quantiles))
                    if quantiles else None),
                "measured_interval_coverage": evidence_payload.get(
                    "measured_interval_coverage"),
                "coverage_points": artifact.task.horizon,
                "residual_fold_count": evidence_payload.get(
                    "residual_fold_count"),
                "residuals_pooled_across_selection": evidence_payload.get(
                    "residuals_pooled_across_selection"),
                "cutoff_status": (
                    "explicit_as_of" if artifact.task.as_of
                    else "artifact_snapshot"),
                "prospective_validation_status": evidence_payload.get(
                    "prospective_validation_status"),
            }
    try:
            publication = publish_result(
                result, mode=mode,
                dossiers=list(dossiers), scenario_selection=selection,
                automation_policy=policy,
                automation_authority=not bool(arguments.get(
                    "_mcp_agent_boundary")),
                calibration_evidence=calibration_evidence,
                candidate_outcome_evidence=candidate_outcome_evidence,
                prior_compromise_history=(
                    governed_history if mode == "best_effort" else None),
                allow_uncertainty_limited_prior=bool(
                    submission.get("allow_prior_compromise", False)),
                artifact_id=artifact.forecast_id)
    except ValueError as exc:
        raise GnomonError("INVALID_ARGUMENTS", str(exc)) from exc
    payload["publication"] = publication
    payload["publication_path"] = str(write_publication(path, publication))
