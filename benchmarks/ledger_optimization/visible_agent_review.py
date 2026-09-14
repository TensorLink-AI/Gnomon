"""Recording-visible development reviews; original087 remains reproducible."""
from copy import deepcopy
import hashlib
from pathlib import Path

from .agent_review import brief, compact_bytes
from .ml_ledger_cards import development_cards, instant


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
        for key in ('provider', 'revision', 'fingerprint', 'request', 'result'):
            if key not in envelope or envelope[key] != stored[key]:
                raise ValueError(f'Event envelope disagrees with ledger execution: {key}')
        request = stored['request']
        supplied = row['request']
        for key in ('series_id', 'unit', 'horizon', 'history'):
            if supplied.get(key) != request.get(key):
                raise ValueError(f'Event request disagrees with ledger execution: {key}')
        for key in ('cutoff',):
            if instant(supplied[key]) != instant(request[key]):
                raise ValueError(f'Event request disagrees with ledger execution: {key}')
        for key in ('timestamps', 'future_timestamps'):
            if [instant(t) for t in supplied[key]] != [instant(t) for t in request[key]]:
                raise ValueError(f'Event request disagrees with ledger execution: {key}')
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
