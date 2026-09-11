"""Bounded decision summaries and lessons backed by immutable ledger evidence.

Context is caller-declared evidence, never inferred from demand. No agent/model
is invoked; lesson text remains a hypothesis even when a score is complete.
The existing decisions/outcomes tables retain backwards compatibility.
"""
from copy import deepcopy
import hashlib
import json
from statistics import mean
from uuid import uuid4

from .forecast_adapter import ForecastAdapterError, point_error_metrics
from .ledger import _json, _time, _METRIC_VERSION

SUMMARY_KIND = 'forecast_decision_summary/1'
LESSON_KIND = 'forecast_lesson/1'


def _fail(field, message):
    raise ForecastAdapterError(message, details={'rejected_fields': [field], 'field': field})


def _text(value, field, maximum=1000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        _fail(field, f'{field} must be a nonempty string of at most {maximum} characters')
    return value


def _texts(value, field):
    if not isinstance(value, list) or len(value) > 8:
        _fail(field, f'{field} must be an array of at most 8 concise statements')
    return [_text(v, field, 500) for v in value]


def validate_context(context):
    if not isinstance(context, list) or len(context) > 16:
        _fail('context', 'context must be an array of at most 16 explicit labels')
    result, keys = [], set()
    required = {'key', 'value', 'valid_from', 'valid_to', 'source_available_at', 'source_ref'}
    for item in context:
        if not isinstance(item, dict) or set(item) != required:
            _fail('context', 'Each context label requires exactly key, value, valid_from, valid_to, source_available_at and source_ref')
        label = {k: _text(item[k], 'context.' + k, 500 if k == 'source_ref' else 128)
                 for k in ('key', 'value', 'source_ref')}
        label.update({k: _time(item[k], 'context.' + k) for k in ('valid_from', 'valid_to', 'source_available_at')})
        if label['valid_from'] >= label['valid_to']:
            _fail('context.valid_to', 'Context valid intervals must be nonempty half-open [valid_from, valid_to)')
        if label['key'] in keys:
            _fail('context.key', 'Context keys must be distinct within one decision')
        keys.add(label['key'])
        result.append(label)
    return result


def record_decision_summary(ledger, *, execution_id, rationale, assumptions,
                            invalidation_conditions, context, evidence_refs=None):
    """Record a concise hypothesis tied to one executed forecast; no action executed."""
    execution_id = _text(execution_id, 'execution_id', 128)
    rationale = _text(rationale, 'rationale')
    assumptions = _texts(assumptions, 'assumptions')
    invalidation = _texts(invalidation_conditions, 'invalidation_conditions')
    context = validate_context(context)
    refs = [] if evidence_refs is None else evidence_refs
    if not isinstance(refs, list) or len(refs) > 16:
        _fail('evidence_refs', 'evidence_refs must contain at most 16 execution/study references')
    now = ledger._now()
    for label in context:
        label['recorded_at'] = now
    with ledger._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        execution = ledger._execution(conn, execution_id)
        if execution['recorded_at'] > now:
            _fail('execution_id', 'Execution was not recorded by the decision recording time')
        req = execution['request']
        if not req.get('timestamps') or not req.get('future_timestamps') or req.get('series_id') in (None, '__default__'):
            _fail('execution_id', 'Decision summaries require a named series, history timestamps and future timestamps')
        origin = _time(req.get('cutoff') or req['timestamps'][-1])
        for label in context:
            if label['source_available_at'] > now:
                _fail('context.source_available_at', 'A recorded context assertion cannot claim future source availability; append it after it becomes available')
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {'kind', 'id'} or ref['kind'] not in ('execution', 'study'):
                _fail('evidence_refs', 'Each evidence reference requires kind execution/study and id')
            _text(ref['id'], 'evidence_refs.id', 128)
            table, key = ('executions', 'execution_id') if ref['kind'] == 'execution' else ('studies', 'study_id')
            row = conn.execute(f'SELECT recorded_at FROM {table} WHERE {key}=?', (ref['id'],)).fetchone()
            if row is None or row[0] > now:
                _fail('evidence_refs', 'Referenced evidence must exist at the decision recording time')
        decision_id = str(uuid4())
        from .final_selection import forecast_request_fingerprint
        summary = {'kind': SUMMARY_KIND, 'execution_id': execution_id,
                   'provider': execution['provider'], 'revision': execution['revision'],
                   'request_fingerprint': forecast_request_fingerprint(req), 'series_id': req['series_id'],
                   'unit': req['unit'], 'horizon': req['horizon'], 'origin': origin,
                   'rationale': rationale, 'assumptions': assumptions,
                   'invalidation_conditions': invalidation, 'context': context,
                   'evidence_refs': refs, 'assertion_basis': 'caller_declared',
                   'business_explanation_validated': False}
        payload = {'decision_id': decision_id, 'execution_ids': [execution_id],
                   'policy': {'kind': SUMMARY_KIND}, 'inputs': summary,
                   'action': {'provider': execution['provider'], 'execution_id': execution_id},
                   'authorization_ref': None, 'meaning': 'Recorded forecast selection summary; no business action executed'}
        conn.execute('INSERT INTO decisions VALUES (?,?,?)', (decision_id, now, _json(payload)))
    return {'decision_id': decision_id, 'recorded_at': now, 'summary': summary}


def _summary(conn, decision_id, recorded):
    _text(decision_id, 'decision_id', 128)
    row = conn.execute('SELECT payload_json, recorded_at FROM decisions WHERE decision_id=?', (decision_id,)).fetchone()
    if row is None or row[1] > recorded:
        _fail('decision_id', 'Decision does not exist at recorded_as_of')
    payload = json.loads(row[0])
    summary = payload.get('inputs', {})
    if summary.get('kind') != SUMMARY_KIND:
        _fail('decision_id', 'This operation requires a structured forecast decision summary')
    return payload, row[1], hashlib.sha256(row[0].encode()).hexdigest()


def _review(ledger, conn, decision_id, source, recorded):
    decision, decision_recorded, digest = _summary(conn, decision_id, recorded)
    summary = decision['inputs']
    execution = ledger._execution(conn, summary['execution_id'])
    if execution['recorded_at'] > recorded:
        _fail('recorded_as_of', 'Execution was not recorded by this cutoff')
    req = execution['request']
    times, pairs = ledger._pairs(conn, req, source, recorded)
    metrics = point_error_metrics([(execution['result']['point'][i], row['value']) for i, row in pairs])
    complete = len(pairs) == len(times)
    score_status = 'complete' if complete else 'partial' if pairs else 'pending'
    return {'kind': 'decision_review/1', 'decision_id': decision_id,
            'decision_recorded_at': decision_recorded, 'original_decision_sha256': digest,
            'execution_id': summary['execution_id'], 'provider': execution['provider'],
            'revision': execution['revision'], 'summary': summary,
            'source_as_of': source, 'recorded_as_of': recorded,
            'status': 'ready_for_review' if complete else 'waiting_for_actuals',
            'review_ready': complete, 'scoring_status': score_status,
            'coverage': ledger._coverage(conn, req, times, pairs, source, recorded),
            'metrics': metrics, 'metric_version': _METRIC_VERSION, 'actual_ids': [a['actual_id'] for _, a in pairs],
            'scored_pairs': [{'step': i, 'timestamp': times[i], 'prediction': execution['result']['point'][i],
                             'actual': a['value'], 'actual_id': a['actual_id'],
                             'source_available_at': a['source_available_at'], 'recorded_at': a['recorded_at']}
                            for i, a in pairs],
            'business_explanation_validated': False,
            'next_step': 'review_assumptions_and_record_a_lesson' if complete else 'wait_for_matching_actuals',
            'provider_calls': 0, 'ledger_writes': 0}


def review_decision(ledger, *, decision_id, source_as_of, recorded_as_of):
    """Read a cutoff-bound review packet. Complete coverage invites review, not belief."""
    source, recorded = _time(source_as_of), _time(recorded_as_of)
    with ledger._connect() as conn:
        conn.execute('BEGIN')
        return _review(ledger, conn, decision_id, source, recorded)


def record_lesson(ledger, *, decision_id, lesson, source_as_of, recorded_as_of, previous_lesson_id=None):
    """Append a hypothesis and immutable complete review. Versions form one checked chain."""
    lesson = _text(lesson, 'lesson', 2000)
    if previous_lesson_id is not None:
        _text(previous_lesson_id, 'previous_lesson_id', 128)
    source, recorded, now = _time(source_as_of), _time(recorded_as_of), ledger._now()
    if max(source, recorded) > now:
        _fail('source_as_of/recorded_as_of', 'Saved lessons require evidence cutoffs no later than the recording clock')
    with ledger._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        review = _review(ledger, conn, decision_id, source, recorded)
        if not review['review_ready']:
            raise ForecastAdapterError('Lesson requires complete matching-unit actuals; use review_decision to inspect missing evidence',
                                       details={'coverage': review['coverage']})
        rows = conn.execute("SELECT outcome_id, payload_json FROM decision_outcomes WHERE decision_id=? "
                            "AND json_extract(payload_json, '$.outcome.kind')=? ORDER BY rowid DESC LIMIT 1",
                            (decision_id, LESSON_KIND)).fetchall()
        prior = json.loads(rows[0][1])['outcome'] if rows else None
        if prior and prior['lesson'] == lesson and prior['review'] == review and prior['previous_lesson_id'] == previous_lesson_id:
            return {'lesson_id': rows[0][0], 'version': prior['version'], 'reused': True}
        expected = rows[0][0] if rows else None
        if previous_lesson_id != expected:
            raise ForecastAdapterError('Supply the latest lesson ID to append a new version; previous versions are immutable',
                                       details={'expected_previous_lesson_id': expected, 'supplied_previous_lesson_id': previous_lesson_id})
        lesson_id = str(uuid4())
        outcome = {'kind': LESSON_KIND, 'lesson': lesson, 'version': prior['version'] + 1 if prior else 1,
                   'previous_lesson_id': previous_lesson_id, 'review': review,
                   'assertion_basis': 'agent_or_operator_hypothesis', 'business_explanation_validated': False}
        payload = {'outcome_id': lesson_id, 'decision_id': decision_id, 'outcome': outcome, 'source_available_at': source}
        conn.execute('INSERT INTO decision_outcomes VALUES (?,?,?,?)', (lesson_id, decision_id, now, _json(payload)))
    return {'lesson_id': lesson_id, 'version': outcome['version'], 'reused': False}


def export_lesson(ledger, *, lesson_id, recorded_as_of):
    """Portable compact memory with exact numerical verification and immutable IDs."""
    _text(lesson_id, 'lesson_id', 128)
    recorded = _time(recorded_as_of)
    with ledger._connect() as conn:
        conn.execute('BEGIN')
        row = conn.execute('SELECT payload_json, recorded_at FROM decision_outcomes WHERE outcome_id=?', (lesson_id,)).fetchone()
        if row is None or row[1] > recorded:
            _fail('lesson_id', 'Lesson does not exist at recorded_as_of')
        payload = json.loads(row[0])
        outcome = payload['outcome']
        if outcome.get('kind') != LESSON_KIND:
            _fail('lesson_id', 'Outcome is not a structured lesson')
        review = outcome['review']
        _, _, original_hash = _summary(conn, payload['decision_id'], recorded)
        if original_hash != review['original_decision_sha256']:
            _fail('decision_id', 'Original decision integrity check failed')
    return {'schema_version': '1', 'kind': LESSON_KIND, 'lesson_id': lesson_id,
            'decision_id': payload['decision_id'], 'version': outcome['version'],
            'previous_lesson_id': outcome['previous_lesson_id'], 'recorded_at': row[1],
            'lesson': outcome['lesson'], 'assertion_basis': outcome['assertion_basis'],
            'business_explanation_validated': False, 'provider': review['provider'],
            'execution_id': review['execution_id'], 'revision': review['revision'],
            'series_id': review['summary']['series_id'], 'unit': review['summary']['unit'],
            'horizon': review['summary']['horizon'], 'request_fingerprint': review['summary']['request_fingerprint'],
            'metrics': review['metrics'], 'metric_version': review['metric_version'],
            'actual_ids': review['actual_ids'],
            'scored_pairs_sha256': hashlib.sha256(_json(review['scored_pairs']).encode()).hexdigest(),
            'context': review['summary']['context'], 'original_decision_sha256': original_hash,
            'lesson_sha256': hashlib.sha256(row[0].encode()).hexdigest(),
            'verification_call': {'operation': 'review_decision', 'decision_id': payload['decision_id'],
                                  'source_as_of': review['source_as_of'], 'recorded_as_of': review['recorded_as_of']},
            'immutable_evidence_call': {'operation': 'decision', 'decision_id': payload['decision_id'],
                                        'recorded_as_of': row[1]},
            'memory_key': lesson_id}


def put_lesson(store, namespace, exported_lesson):
    """Explicitly write an exported lesson to a caller-owned LangGraph-compatible store.

    No dependency or store is created. Pass a persistent store for durability.
    The caller authorizes this external write; IDs allow idempotent put retries.
    """
    if not isinstance(namespace, tuple) or not namespace or any(not isinstance(v, str) or not v for v in namespace):
        _fail('namespace', 'namespace must be a nonempty tuple of nonempty strings')
    if not isinstance(exported_lesson, dict) or exported_lesson.get('kind') != LESSON_KIND:
        _fail('exported_lesson', 'Use TemporalLedger.export_lesson to obtain a portable lesson')
    key = _text(exported_lesson.get('lesson_id'), 'lesson_id', 128)
    store.put(namespace, key, deepcopy(exported_lesson))
    return {'memory_key': key, 'namespace': list(namespace)}


def validate_filters(filters):
    if not isinstance(filters, dict) or not 1 <= len(filters) <= 16:
        _fail('context_filters', 'Supply 1 to 16 exact context key/value filters')
    return {_text(k, 'context_filters.key', 128): _text(v, 'context_filters.value', 128) for k, v in filters.items()}


def context_records(conn, series_id, recorded, start, end):
    rows = conn.execute("SELECT payload_json, recorded_at FROM decisions WHERE recorded_at<=? "
                        "AND json_extract(payload_json, '$.inputs.kind')=? "
                        "AND json_extract(payload_json, '$.inputs.series_id')=? "
                        "AND json_extract(payload_json, '$.inputs.origin') BETWEEN ? AND ? ORDER BY rowid LIMIT 1001",
                        (recorded, SUMMARY_KIND, series_id, start, end)).fetchall()
    if len(rows) > 1000:
        _fail('context_filters', 'Context comparison exceeds 1000 decision summaries; narrow the origin window. No partial ranking returned')
    return [(json.loads(r[0]), r[1]) for r in rows]


def match_context(records, selected, origin, source, filters):
    ids = {r['execution_id'] for r in selected}
    first_target = min(_time(t) for r in selected for t in r['request']['future_timestamps'])
    eligible, excluded = [], []
    for decision, recorded in records:
        summary = decision['inputs']
        if summary['execution_id'] not in ids:
            continue
        labels = {v['key']: v for v in summary['context']}
        reason = None
        if not all(k in labels for k in filters):
            reason = 'missing_context_labels'
        elif recorded >= first_target:
            reason = 'context_recorded_after_target'
        elif any(labels[k]['source_available_at'] > min(source, origin) for k in filters):
            reason = 'context_source_unavailable_at_origin'
        elif any(not labels[k]['valid_from'] <= origin < labels[k]['valid_to'] for k in filters):
            reason = 'context_not_valid_at_origin'
        if reason:
            excluded.append({'decision_id': decision['decision_id'], 'reason': reason})
        else:
            eligible.append((decision['decision_id'], tuple(labels[k]['value'] for k in filters)))
    values = {v for _, v in eligible}
    if len(values) > 1:
        return False, {'reason': 'conflicting_context_labels', 'context_exclusions': excluded,
                       'decision_ids': [i for i, _ in eligible]}
    matches = bool(values) and next(iter(values)) == tuple(filters.values())
    return matches, {'reason': None if matches else 'context_filter_mismatch' if values else 'no_visible_context',
                     'decision_ids': [i for i, _ in eligible], 'context_exclusions': excluded}


def compare_context(ledger, *, context_filters, **kwargs):
    """Exact ex-ante labels filter the existing matched production comparison."""
    from .ledger_history import compare_history
    filters = validate_filters(context_filters)
    answer = compare_history(ledger, **kwargs, context_filters=filters)
    answer['context_filters'] = filters
    answer['context_basis'] = 'caller_labels_source_available_by_origin_and_recorded_before_first_target'
    answer['uncertainty'] = {'method': 'descriptive_origin_ranges', 'confidence_interval': None,
        'limitation': 'Origins may overlap and share actuals; no independence, causal explanation or superiority claim.',
        'models': [{'provider': m['provider'], 'origin_count': answer['matched_origins'],
                    'min_origin_mae': min(v), 'max_origin_mae': max(v), 'mean_origin_mae': mean(v)}
                   for m in answer['models']
                   for v in [[next(x['mae'] for x in o['models'] if x['provider'] == m['provider']) for o in answer['origins']]]]}
    if answer.get('metric') == 'rmsle':
        answer['uncertainty']['selected_metric'] = 'rmsle'
        answer['uncertainty']['selected_metric_ranges'] = [
            {'provider': m['provider'], 'min': min(values), 'max': max(values), 'mean': mean(values)}
            for m in answer['models']
            for values in [[next(x['rmsle'] for x in o['models'] if x['provider'] == m['provider'])
                            for o in answer['origins']]]]
    return answer


MEMORY_PARAMETERS = {
    'record_decision_summary': ('execution_id', 'rationale', 'assumptions', 'invalidation_conditions', 'context', 'evidence_refs'),
    'compare_context': ('series_id', 'horizon', 'providers', 'unit', 'start', 'end', 'source_as_of', 'recorded_as_of', 'context_filters',
                        'metric', 'recent_origins', 'negative_predictions'),
    'retrieve_context': ('series_id', 'horizon', 'providers', 'unit', 'start', 'end', 'source_as_of', 'recorded_as_of',
                         'context_candidates', 'min_origins', 'metric', 'recent_origins', 'negative_predictions'),
    'review_decision': ('decision_id', 'source_as_of', 'recorded_as_of'),
    'record_lesson': ('decision_id', 'lesson', 'source_as_of', 'recorded_as_of', 'previous_lesson_id'),
    'export_lesson': ('lesson_id', 'recorded_as_of'),
}
MEMORY_REQUIRED = {k: tuple(p for p in v if p not in {'unit', 'evidence_refs', 'previous_lesson_id', 'metric', 'recent_origins', 'negative_predictions', 'min_origins'})
                   for k, v in MEMORY_PARAMETERS.items()}
MEMORY_WRITES = {'record_decision_summary', 'record_lesson'}
MEMORY_DESCRIPTIONS = {
    'retrieve_context': 'Read caller-ordered progressively broader context cohorts in one database snapshot. Select the first meeting min_origins, never the best observed score. Explicit {} opts into unfiltered evidence. Count eligibility is not confidence; no provider selection, calls or writes.',
    'record_decision_summary': 'Record a concise caller hypothesis tied to an existing execution. Context source availability is explicit; recording time is assigned by the ledger. No business action is executed.',
    'compare_context': 'Read complete matched production origins with exact context filters. Labels must be source-available by origin, locally recorded before the first target and visible at recorded_as_of. Conflicts exclude an origin. Descriptive ranges are not confidence intervals or causal evidence.',
    'review_decision': 'Read-only outcome-triggered review packet. review_ready requires complete matching-unit actuals. Poll with explicit advancing cutoffs; no background agent is run and no business explanation is validated.',
    'record_lesson': 'Append a bounded hypothesis with a complete immutable review packet. Require the latest previous_lesson_id for a new version. Exact latest retries reuse the lesson. Original decisions and prior lessons remain unchanged.',
    'export_lesson': 'Read portable JSON with immutable IDs, hashes and exact numerical verification calls. Does not contact any memory service. Hypotheses remain unverified.',
}
_text_schema = {'type': 'string', 'minLength': 1, 'maxLength': 128}
_context_fields = {k: {**_text_schema} for k in ('key', 'value', 'valid_from', 'valid_to', 'source_available_at', 'source_ref')}
_context_fields['source_ref']['maxLength'] = 500
for _field in ('valid_from', 'valid_to', 'source_available_at'):
    _context_fields[_field]['description'] = 'Explicit timezone timestamp; valid interval is half-open. Source availability is independent of the valid interval.'
MEMORY_PROPERTIES = {
    'context_candidates': {'type': 'array', 'minItems': 1, 'maxItems': 8,
                           'description': 'Most-specific-first exact filters. Each later entry must strictly remove filters without changing values. Empty {} explicitly permits unfiltered fallback.',
                           'items': {'type': 'object', 'maxProperties': 16, 'propertyNames': _text_schema,
                                     'additionalProperties': _text_schema}},
    'min_origins': {'type': 'integer', 'minimum': 1, 'maximum': 1000, 'default': 4,
                    'description': 'Required complete matched origins for retrieval eligibility, not statistical significance.'},
    'rationale': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
    'lesson': {'type': 'string', 'minLength': 1, 'maxLength': 2000},
    'assumptions': {'type': 'array', 'maxItems': 8, 'items': {'type': 'string', 'minLength': 1, 'maxLength': 500}},
    'invalidation_conditions': {'type': 'array', 'maxItems': 8, 'items': {'type': 'string', 'minLength': 1, 'maxLength': 500}},
    'context': {'type': 'array', 'maxItems': 16, 'items': {'type': 'object', 'additionalProperties': False,
                'required': list(_context_fields), 'properties': _context_fields}},
    'context_filters': {'type': 'object', 'minProperties': 1, 'maxProperties': 16,
                        'propertyNames': _text_schema, 'additionalProperties': _text_schema},
    'evidence_refs': {'type': 'array', 'maxItems': 16, 'items': {'type': 'object', 'additionalProperties': False,
                      'required': ['kind', 'id'], 'properties': {'kind': {'enum': ['execution', 'study']}, 'id': _text_schema}}},
    'previous_lesson_id': {'type': ['string', 'null'], 'minLength': 1},
}
MEMORY_EXAMPLES = {
    'retrieve_context': {'series_id': 'SERIES_ID', 'horizon': 2,
                         'providers': {'last_value': 'REVISION', 'historical_mean': 'REVISION'},
                         'start': 'ORIGIN_START', 'end': 'ORIGIN_END', 'source_as_of': 'SOURCE_CUTOFF',
                         'recorded_as_of': 'RECORDING_CUTOFF',
                         'context_candidates': [{'promotion': 'planned', 'demand': 'sparse'}, {'demand': 'sparse'}, {}]},
    'record_decision_summary': {'execution_id': 'EXECUTION_ID', 'rationale': 'RATIONALE',
                              'assumptions': [], 'invalidation_conditions': [], 'context': []},
    'compare_context': {'series_id': 'SERIES_ID', 'horizon': 2,
                        'providers': {'last_value': 'REVISION', 'historical_mean': 'REVISION'},
                        'start': 'ORIGIN_START', 'end': 'ORIGIN_END', 'source_as_of': 'SOURCE_CUTOFF',
                        'recorded_as_of': 'RECORDING_CUTOFF', 'context_filters': {'promotion': 'planned'}},
    'review_decision': {'decision_id': 'DECISION_ID', 'source_as_of': 'SOURCE_CUTOFF', 'recorded_as_of': 'RECORDING_CUTOFF'},
    'record_lesson': {'decision_id': 'DECISION_ID', 'lesson': 'LESSON_HYPOTHESIS',
                      'source_as_of': 'SOURCE_CUTOFF', 'recorded_as_of': 'RECORDING_CUTOFF'},
    'export_lesson': {'lesson_id': 'LESSON_ID', 'recorded_as_of': 'RECORDING_CUTOFF'},
}
