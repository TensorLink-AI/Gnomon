"""Shared semantic completion, recovery and independently checkable evidence."""
import hashlib
import json
import math

CUTOFF_SEMANTICS = {
    'forecast_origin': 'Last history instant for one prediction task (request.cutoff / fold.origin). Later rescoring never moves it.',
    'observation_valid_time': 'When an observation applies, distinct from when it became available.',
    'source_availability': 'known_time in TemporalStore; source_available_at for ledger actuals. Inclusive visibility boundary.',
    'local_recording': 'recorded_at, assigned when the observation or evidence was ingested locally. Inclusive visibility boundary.',
    'snapshot_as_of': 'Frozen source-availability boundary. A narrower snapshot cannot be widened by a later query.',
    'study_evidence_recorded_as_of': 'Selects studies/executions recorded by this boundary, independently of observation valid times.',
    'routing_source_as_of': 'Input snapshot source boundary and prospective forecast boundary: new forecast timestamps must follow it.',
    'rescore_source_as_of': 'Actual source-availability boundary only. Predictions, valid times and original forecast origins are reused unchanged.',
}


def completion(payload):
    """Technical operation completeness; never assert the business intent was right."""
    status = payload.get('status')
    evidence = None
    task = status not in ('error', 'unscored', 'partial', 'result_available')
    if status is None:
        task = False
    if 'scoring_status' in payload:
        task = evidence = payload.get('complete', False)
    elif 'folds' in payload or 'fallback_used' in payload:
        task = evidence = (not payload.get('fallback_used', False) and status == 'ok') if 'fallback_used' in payload else status == 'complete'
    return {**payload, 'operation_succeeded': status != 'error', 'task_completed': task,
            'evidence_complete': evidence, 'completion_scope': 'requested_technical_operation_not_business_intent'}


def recovery_metadata(details):
    from .recovery import example_changes
    supplied = details.get('supplied_arguments', details.get('input_options', {}))
    example = details.get('example_arguments')
    changed = details.get('changed_fields', example_changes(supplied, example) if isinstance(example, dict) else [])
    return {'supplied_arguments': supplied, 'preserved_fields': details.get('preserved_fields', [
        k for k in supplied if isinstance(example, dict) and k in example and not example_changes({k: supplied[k]}, {k: example[k]})]),
        'changed_fields': changed, 'choices_required': details.get('choices_required', {}),
        'rejected_fields': details.get('rejected_fields', []),
        'example_kind': details.get('example_kind', 'no_example'),
        'example_runnable': details.get('example_runnable', False),
        'admissible': details.get('admissible'), 'example_arguments_ref': '/error/details/example_arguments' if example is not None else None}


def repair_budgets(details):
    """Expose independent budget scopes; null means this diagnostic did not measure it."""
    drop, gap, counts = details.get('drop_budget', {}), details.get('repair_budget', {}), details.get('repair_counts', {})
    total = details.get('total_observations', details.get('original_observations', gap.get('denominator')))
    conflicts = details.get('conflicting_rows', counts.get('conflicts'))
    fills = gap.get('filled', counts.get('filled'))
    def budget(scope, proposed, denominator, maximum, admissible=None, **extra):
        return {'scope': scope, 'proposed_count': proposed, 'denominator': denominator,
                'max_fraction': maximum, 'admissible': admissible, **extra}
    return {'drop_budget': budget('all_input_rows', drop.get('dropped_rows'), drop.get('total_rows'), .05,
                drop.get('within_budget'), scan_complete=drop.get('scan_complete'),
                predicted_post_repair_count=drop.get('predicted_rows_after_drops')),
        'gap_fill_budget': budget('shared_fill_conflict_fraction_of_original_observations', fills, total, .30,
                False if counts.get('gap_run_at_least', 0) > details.get('max_gap_run', float('inf')) else gap.get('within_budget'), max_gap_run=details.get('max_gap_run', gap.get('max_gap_run')),
                gap_run_at_least=counts.get('gap_run_at_least'),
                predicted_post_repair_count=total + fills if total is not None and fills is not None and conflicts == 0 else None),
        'conflict_resolution_budget': budget('shared_fill_conflict_fraction_of_original_observations', conflicts, total, .30,
                conflicts / max(1, total) <= .30 if conflicts is not None and total is not None else None),
        'timestamp_alignment_budget': {'scope': 'bounded_timestamp_jitter_not_charged_to_fill_conflict_budget',
                'proposed_count': counts.get('aligned_timestamps_not_charged'), 'admissible': None},
        'guidance': 'Null counts/admissibility were not measured. Passing one budget does not establish that the whole repair is admissible. Fills and conflicts share a combined 30% cap.'}


def scored_pairs(pairs):
    pairs = [(float(p), float(a)) for p, a in pairs]
    errors = [p - a for p, a in pairs]
    encoded = json.dumps(pairs, separators=(',', ':'), allow_nan=False)
    return {'n': len(pairs), 'errors': errors, 'absolute_error_sum': math.fsum(abs(e) for e in errors),
            'squared_error_sum': math.fsum(e * e for e in errors), 'error_sum': math.fsum(errors),
            'denominator': len(pairs), 'pairs_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
            'hash_encoding': 'UTF-8 compact JSON array of [float_prediction,float_actual] pairs in fold/time order'}


def verify_builtin(provider, request, result):
    from .forecast_adapter import ForecastAdapterError
    history, horizon, season = request.history, request.horizon, request.season
    if provider == 'last_value':
        expected, calculation = [history[-1]] * horizon, 'repeat final observed value'
    elif provider == 'historical_mean':
        expected, calculation = [math.fsum(history) / len(history)] * horizon, 'sum(history) / len(history)'
    elif provider == 'seasonal_naive':
        expected, calculation = [history[-season + i % season] for i in range(horizon)], 'repeat the last season observations'
    else:
        raise ForecastAdapterError('Independent built-in verification supports last_value, historical_mean and seasonal_naive.')
    return {'scope': 'deterministic_builtin_arithmetic', 'calculation': calculation, 'expected': expected,
            'verified': len(expected) == len(result.point) and all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(expected, result.point)),
            'effective_season': season, 'history_count': len(history), 'horizon': horizon}
