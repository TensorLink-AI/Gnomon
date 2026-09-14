"""Source fragment for an undeployed common-lab capsule.

The capsule builder replaces only frozen lab.backtest with this function. Its
globals are the existing lab helpers, with all executions still through core.
Do not import this fragment as a standalone execution API.
"""


def backtest(config, *, initial_baseline=False):
    """Retain three CV results and one unselected production forecast per config."""
    config = configuration(config)
    cid = config_id(config)
    task = core.read('task.json')
    expected = {end: core.request_at(end)[0] for end in (*FOLDS, 730)}

    def validate(row, end):
        ex = row.get('execution', {})
        point = row.get('point')
        if (row.get('request') != expected[end] or row.get('config') != config
                or row.get('config_id') != cid or row.get('task_origin') != task['origin']
                or row.get('kind') != ('forecast' if end == 730 else 'backtest')
                or ex.get('provider') != config['model'] + '_' + cid
                or ex.get('revision') != 'ml-lab-v1:' + cid
                or not ex.get('execution_id') or ex.get('result', {}).get('point') != point
                or not isinstance(point, (list, tuple)) or len(point) != expected[end]['horizon']
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in point)):
            raise Rejected('COLLECTION_IDENTITY_MISMATCH', 'Stored execution does not match the current task and configuration.')
        if end == 730 and (row.get('actual') is not None or row.get('metrics') is not None):
            raise Rejected('COLLECTION_FUTURE_ACTUALS', 'Production collection cannot contain current-task actuals or scores.')
        if end != 730:
            score = (row.get('metrics') or {}).get('rmsle')
            if type(score) not in (int, float) or not math.isfinite(score) or score < 0:
                raise Rejected('COLLECTION_INVALID_CV_SCORE', 'Completed CV requires a finite nonnegative RMSLE.')
        return row

    existing = {}
    for row in current_records():
        if row['event'] != 'result' or row.get('config_id') != cid:
            continue
        kind = row.get('kind')
        if kind not in ('backtest', 'forecast'):
            continue
        ends = [end for end, request in expected.items()
                if row.get('request') == request and (kind == 'forecast') == (end == 730)]
        if len(ends) != 1:
            raise Rejected('COLLECTION_IDENTITY_MISMATCH', 'Stored execution does not match a current fold or forecast origin.')
        end = ends[0]
        if end in existing:
            raise Rejected('AMBIGUOUS_COLLECTION_EXECUTIONS', 'Multiple stored results match the same configuration and origin.')
        existing[end] = validate(row, end)

    missing = [end for end in (*FOLDS, 730) if end not in existing]
    if missing:
        if budget()['phase'] == 'selection' and not initial_baseline:
            raise Rejected('SELECTION_PHASE_RESERVED',
                           'Collection is closed. Explicitly commit an already backtested configuration.',
                           numerical_calls_started=0)
        reserve = 0 if initial_baseline else 1
        admit(len(missing), reserve=reserve)
        core.append('experiments.jsonl', {
            'event': 'collection_batch_admitted', 'task_origin': task['origin'],
            'config_id': cid, 'config': config, 'fits': len(missing), 'missing_ends': missing,
            'remaining_before': budget()['numerical_remaining'], 'reserve': reserve,
            'phase': budget()['phase'], 'initial_baseline': initial_baseline, 'selection_changed': False})
        for end in missing:
            seconds = budget()['seconds_remaining_approx']
            if seconds is None or seconds <= 0:
                raise Rejected('COLLECTION_DEADLINE',
                               'Collection stopped before the next fit; completed evidence and checkpoint remain intact.',
                               completed_ends=sorted(existing), missing_ends=[e for e in missing if e not in existing])
            try:
                existing[end] = validate(core.execute(config, end, 'forecast' if end == 730 else 'backtest'), end)
            except Exception as exc:
                raise Rejected('COLLECTION_FIT_FAILED',
                               'A metered collection fit failed. Completed evidence and checkpoint remain intact.',
                               failed_end=end, completed_ends=sorted(existing),
                               cause_type=type(exc).__name__, cause_code=getattr(exc, 'code', None),
                               next_call={'operation': 'backtest', 'arguments': {'config': config},
                                          'condition': 'Retry only while exploration and the existing budget permit.'}) from exc
    # Both new results and exact retries are validated; no checkpoint publication.
    rows = [existing[end] for end in FOLDS]
    forecast = existing[730]
    return {'config': config, 'config_id': cid,
            'folds': [{'origin': r['request']['cutoff'], 'metrics': r['metrics'],
                       'execution_id': r['execution']['execution_id']} for r in rows],
            'mean_rmsle': sum(r['metrics']['rmsle'] for r in rows) / len(FOLDS),
            'reused': not missing,
            'collection': {'status': 'complete', 'execution_id': forecast['execution']['execution_id'],
                           'new_fits': len(missing), 'reused_fits': 4 - len(missing),
                           'production_actuals_known': False, 'selection_changed': False,
                           'task_origin': task['origin'],
                           'next_call': {'operation': 'commit', 'arguments': {'execution_id': forecast['execution']['execution_id']}}}}
