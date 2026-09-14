"""Recording-visible development reviews; original087 remains reproducible."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

from .agent_review import brief, compact_bytes
from .ml_ledger_cards import development_cards, instant


def same_value(supplied, recorded):
    """JSON-equivalent sequences/numbers, without equating booleans and numbers."""
    if isinstance(supplied, (tuple, list)) and isinstance(recorded, (tuple, list)):
        return len(supplied) == len(recorded) and all(same_value(a, b) for a, b in zip(supplied, recorded))
    if type(supplied) is dict and type(recorded) is dict:
        return supplied.keys() == recorded.keys() and all(same_value(supplied[k], recorded[k]) for k in supplied)
    if type(supplied) in (int, float) and type(recorded) in (int, float):
        return math.isfinite(supplied) and math.isfinite(recorded) and supplied == recorded
    return type(supplied) is type(recorded) and supplied == recorded


def validate_request(supplied, recorded, *, complete=False):
    if type(supplied) is not dict or (complete and supplied.keys() != recorded.keys()):
        raise ValueError('Event envelope disagrees with ledger execution: request fields')
    required = ('series_id', 'unit', 'horizon', 'history', 'cutoff', 'timestamps', 'future_timestamps')
    for key in required:
        if key not in supplied:
            raise ValueError(f'Event request missing required identity: {key}')
    for key, value in supplied.items():
        if key not in recorded:
            raise ValueError(f'Event request has an unknown field: {key}')
        expected = recorded[key]
        if key in ('cutoff', 'known_time_cutoff', 'recorded_time_cutoff') and value is not None and expected is not None:
            equal = instant(value) == instant(expected)
        elif key in ('timestamps', 'future_timestamps'):
            equal = [instant(t) for t in value] == [instant(t) for t in expected]
        else:
            equal = same_value(value, expected)
        if not equal:
            raise ValueError(f'Event request disagrees with ledger execution: {key}')


def visible_records(db, records, task):
    """Validate public execution references before discovery or pair ordering.

    The supplied event log is an index, not authority for recording visibility.
    Every forecast reference is checked against ledger.execution. Future-recorded
    identities are withheld from the usable catalogue. Their exclusions belong
    to operator evidence, not to the candidate list used by the agent.
    """
    cutoff = instant(task['origin'])
    executions = {}
    visible = []
    excluded = []
    for row in records:
        if row.get('event') != 'result' or row.get('kind') != 'forecast':
            continue
        envelope = row['execution']
        eid = envelope['execution_id']
        if eid not in executions:
            try:
                executions[eid] = db.execution(eid)
            except (ValueError, KeyError) as exc:
                raise ValueError('Review references an unverifiable ledger execution') from exc
        stored = executions[eid]
        if stored.get('execution_id') != eid:
            raise ValueError('Ledger returned a different execution identity')
        # Never use recording time copied into the event envelope as authority.
        recorded = instant(stored['recorded_at'])
        if recorded > cutoff:
            excluded.append({'execution_id': eid, 'reason': 'execution_not_recorded_by_query',
                             'recorded_at': stored['recorded_at'], 'recorded_as_of': task['origin']})
            continue
        for key in ('provider', 'revision', 'fingerprint', 'result'):
            if key not in envelope or not same_value(envelope[key], stored[key]):
                raise ValueError(f'Event envelope disagrees with ledger execution: {key}')
        request = stored['request']
        if 'request' in envelope:
            validate_request(envelope['request'], request, complete=True)
        validate_request(row['request'], request)
        metadata = stored['result'].get('metadata') or {}
        if 'config' in metadata and not same_value(row['config'], metadata['config']):
            raise ValueError('Event configuration disagrees with ledger result metadata: config')
        item = deepcopy(row)
        item['request'] = deepcopy(request)
        item['execution'] = deepcopy(stored)
        visible.append(item)
    return visible, excluded, len(executions)


def review(db, records, task, evidence_math, evidence_path, **options):
    """Query public1.2.0 ledger evidence without executing forecasts or writes."""
    visible, excluded, reads = visible_records(db, records, task)
    full = development_cards(db, visible, task, evidence_math, **options)
    full['catalog_excluded'].extend(excluded)
    full['ledger_queries'] += reads
    full['catalog_visibility'] = {
        'authority': 'ledger.execution.recorded_at', 'recorded_as_of': task['origin'],
        'cutoff_inclusive': True, 'execution_reads': reads,
        'future_recorded_exclusions': len(excluded),
        'scoring_eligibility': 'Still requires ex-ante recording and matched actuals in compare_history.',
    }
    data = compact_bytes(full)
    result = brief(full, task, evidence_path, hashlib.sha256(data).hexdigest())
    result['schema_version'] = 'agent-review-088'
    result['catalog_visibility'] = deepcopy(full['catalog_visibility'])
    with Path(evidence_path).open('xb') as stream:
        stream.write(data)
    return result
