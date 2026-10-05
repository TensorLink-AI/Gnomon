"""Deterministic presentation of existing results. No inference, reads or writes."""
from copy import deepcopy


def _pick(value, keys):
    return {key: deepcopy(value[key]) for key in keys if key in value}


def _followup(purpose, tool, arguments, *, requires=(), effect='read'):
    return {'purpose': purpose, 'tool': tool, 'arguments': arguments,
            'requires': list(requires), 'effect': effect}


def attach_summary(payload, operation, *, ledger=False, outcome_writes=False):
    """Add an operation-specific overview without changing original result fields.

    JSON pointers refer to this full response. Follow-ups are suggestions only;
    `requires` lists missing inputs and effects distinguish reads from writes.
    """
    if operation in {'read', 'capabilities', 'temporal'}:
        return payload
    operation = payload.get('operation', operation)
    nested = payload.get('result')
    native = nested if isinstance(nested, dict) else {}
    summary = {'schema_version': '1', 'operation': operation,
               'status': {'operation': payload.get('status', 'unknown')},
               'scope': _pick(payload, ('series_id', 'unit', 'variable', 'frequency', 'horizon',
                                       'source_as_of', 'recorded_as_of', 'window')),
               'result': {'details_pointer': '/result' if nested is not None else ''},
               'basis': {}, 'limitations': [],
               'references': _pick(payload, ('execution_id', 'study_id', 'data_ref', 'snapshot_id',
                                             'request_fingerprint', 'recorded', 'reference_scope')),
               'followups': []}
    scope, result, basis = summary['scope'], summary['result'], summary['basis']
    limits, followups = summary['limitations'], summary['followups']
    if 'scoring_status' in payload:
        summary['status']['scoring'] = payload['scoring_status']
    if 'evidence_complete' in payload:
        summary['status']['evidence_complete'] = payload['evidence_complete']

    if 'completion' in payload and 'execution_id' in payload:
        summary['operation'] = 'forecast'
        receipt = payload['completion']
        scope.update(_pick(receipt, ('series_id', 'unit', 'horizon')))
        scope.update(_pick(payload.get('request_provenance', {}), (
            'history_start', 'history_end', 'cutoff', 'known_time_cutoff', 'recorded_time_cutoff')))
        times = receipt.get('future_timestamps', [])
        if times:
            scope['target_window'] = {'start': times[0], 'end': times[-1], 'count': len(times)}
        result.update(_pick(payload, ('provider', 'revision', 'cache_hit')))
        points = native.get('point', [])
        result.update(point_count=len(points), point_pointer='/result/point')
        if len(points) <= 8:
            result['point'] = deepcopy(points)
        result['timestamps_pointer'] = '/completion/future_timestamps'
        if native.get('quantiles'):
            result['quantiles_pointer'] = '/result/quantiles'
        basis['kind'] = 'caller_selected_provider'
        limits.append('Forecasting alone does not establish accuracy or calibrated uncertainty.')
        routing = payload.get('routing')
        if routing:
            basis.update(kind='configured_router', **_pick(routing, (
                'router', 'served_provider', 'reason', 'evidence_based', 'evidence_level',
                'matched_origins', 'effective_n', 'evidence_as_of')))
            basis['details_pointer'] = '/routing'
            summary['references'].update(_pick(routing, ('routing_decision_id',)))
            if not routing.get('evidence_based'):
                limits.append('The router did not select this forecast on sufficient comparative evidence.')
        if 'memory' in payload:
            basis['memory_pointer'] = '/memory'
        if ledger and payload.get('recorded'):
            execution = payload['execution_id']
            followups.append(_followup('inspect_recorded_forecast', 'gnomon_ledger',
                {'operation': 'execution', 'execution_id': execution}))
            if scope.get('series_id') and times:
                followups.append(_followup('score_available_outcomes', 'gnomon_ledger',
                    {'operation': 'evaluate', 'execution_id': execution, 'allow_partial': True},
                    requires=('source_as_of', 'recorded_as_of'), effect='ledger_write'))
                if outcome_writes:
                    followups.append(_followup('submit_observed_actual', 'gnomon_ledger',
                        {'operation': 'append_actual', 'series_id': scope['series_id'], 'unit': scope.get('unit')},
                        requires=('valid_time', 'value', 'source_available_at'), effect='ledger_write'))
                else:
                    limits.append('Actual submission is disabled in this session; the host must ingest outcomes.')
            else:
                limits.append('Outcome scoring requires a recorded series_id and explicit future timestamps.')
        else:
            limits.append('This forecast is not retained in a ledger for later outcome review.')

    elif operation == 'inspect' and 'series' in payload:
        series = payload['series']
        result.update(series_count=len(series), series=deepcopy(series[:4]), series_pointer='/series')
        if len(series) > 4:
            result['omitted_series'] = len(series) - 4
        basis.update(kind='frozen_snapshot', snapshot_pointer='/snapshot', readiness_pointer='/readiness')
        scope.update(_pick(payload, ('timezone', 'unit_basis')))
        repairs = payload.get('repairs', [])
        if repairs:
            limits.append(f'{len(repairs)} data repairs recorded; inspect /repairs before historical evaluation.')
        args = {'data_ref': payload['data_ref']}
        requires = ['statistic']
        if len(series) == 1:
            args['series_id'] = series[0]['series_id']
        else:
            requires.append('series_id')
        followups.append(_followup('calculate_observed_statistic', 'gnomon_describe', args, requires=requires))

    elif operation == 'describe':
        result.update(_pick(payload, ('statistic', 'value', 'count')))
        basis['kind'] = 'observed_statistic'

    elif operation == 'route':
        result.update(_pick(payload, ('recommendation', 'recommendation_role')))
        basis.update(_pick(payload, ('basis', 'reason', 'fallback_used', 'matched_folds',
                                    'min_folds', 'min_improvement')))
        basis['scores_pointer'] = '/scores'
        if payload.get('fallback_used'):
            limits.append('This is a baseline fallback, not a demonstrated comparative win.')
        limits.append('This recommendation has not executed a new forecast.')

    elif 'study_id' in payload and 'scores' in payload:
        summary['operation'] = 'model_comparison'
        result.update(scores_pointer='/scores', ranking_pointer='/ranking')
        basis.update(_pick(payload, ('baseline', 'ranking_policy', 'replay', 'known_time_assumed',
                                     'metric_version', 'aggregation')))
        basis['kind'] = payload.get('evidence', 'saved_model_comparison')
        basis['coverage'] = _pick(payload.get('usage', {}), ('requested_folds', 'planned_folds', 'matched_folds', 'stop_reason'))
        ranking = payload.get('ranking', [])
        result['ranked_models'] = len(ranking)
        result['scores'] = {p: deepcopy(payload['scores'][p]) for p in ranking[:4]}
        result['ranking'] = ranking[:4]
        if len(ranking) > 4:
            result['omitted_models'] = len(ranking) - 4
        limits.append('Ranking applies only to the tested models and matched historical outcomes; future superiority is not established.')
        if payload.get('status') != 'complete':
            limits.append('The requested comparison is not fully scored; inspect /usage and /issues.')
        followups.append(_followup('retrieve_saved_comparison', 'gnomon_evaluate', {'study_id': payload['study_id']}))

    elif operation in {'evaluate', 'review_decision'} and nested is not None:
        summary['operation'] = 'outcome_review'
        scope.update(_pick(payload.get('query', {}), ('execution_id', 'execution_ids', 'decision_id',
            'series_id', 'horizon', 'unit', 'source_as_of', 'recorded_as_of')))
        basis['kind'] = 'recorded_outcomes'
        records = nested if isinstance(nested, list) else [nested]
        result['record_count'] = len(records)
        result['records'] = [_pick(r, ('execution_id', 'decision_id', 'n', 'complete', 'metrics',
            'mae', 'rmse', 'rmsle', 'bias', 'scoring_status', 'review_ready',
            'source_as_of', 'recorded_as_of',
            'coverage_basis', 'current_coverage_basis')) for r in records[:4]]
        for source, target in zip(records, result['records']):
            for field in ('coverage', 'current_coverage'):
                if isinstance(source.get(field), dict):
                    target[field] = _pick(source[field], ('required_steps', 'matched_steps', 'fraction', 'unit'))
        if len(records) > 4:
            result['omitted_records'] = len(records) - 4
        if payload.get('scoring_status') != 'complete':
            limits.append('Outcomes are not fully scored; current metrics cover only the disclosed matched observations.')

    else:
        result.update(_pick(native or payload, ('value', 'statistic', 'status', 'scoring_status', 'reason', 'count', 'n')))
        basis.update(_pick(payload, ('evidence', 'selection_reason', 'evidence_based', 'fallback_used')))
        if payload.get('fallback_used'):
            limits.append('A fallback is not evidence that the selected model outperforms alternatives.')

    # All fields are copied from execution facts. No model-generated narrative is treated as evidence.
    return {**payload, 'agent_summary': summary}
