"""Plan outcome visibility for every already executed production forecast.

No predictions, scoring, ledger mutations, or agent selection occur here.
The caller supplies only outcomes whose synthetic recording time has arrived.
"""
from datetime import datetime
import math


def instant(value):
    value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        raise ValueError('Outcome maturation requires timezone-aware timestamps')
    return value


def plan(events, prior, task):
    now = instant(task['origin'])
    outcomes = {}
    for outcome in prior:
        origin = instant(outcome['origin'])
        if origin in outcomes:
            raise ValueError('Ambiguous outcome origin')
        outcomes[origin] = outcome
    done = {e['execution_id'] for e in events if e['event'] == 'matured'}
    groups = {}
    excluded = []
    seen = set()
    for result in events:
        if result['event'] != 'result' or result['kind'] != 'forecast':
            continue
        eid = result['execution']['execution_id']
        if eid in seen:
            raise ValueError('Duplicate production execution identity')
        seen.add(eid)
        if eid in done:
            continue
        origin = instant(result['task_origin'])
        outcome = outcomes.get(origin)
        request = result['request']
        reason = None
        if outcome is None:
            reason = 'outcome_not_supplied'
        elif instant(outcome['outcome_recorded_at']) > now:
            reason = 'outcome_not_recorded_by_query'
        elif (request['series_id'] != task['series_id'] or request['unit'] != task['unit']
              or outcome.get('series_id', task['series_id']) != task['series_id']
              or outcome.get('unit', task['unit']) != task['unit']):
            reason = 'series_or_unit_mismatch'
        elif instant(request['cutoff']) != origin:
            reason = 'forecast_origin_mismatch'
        else:
            times = [instant(t) for t in request['future_timestamps']]
            outcome_times = [instant(t) for t in outcome['future_timestamps']]
            if (request['horizon'] != task['horizon'] or
                    len(times) != request['horizon'] or
                    len(times) != len(result['point']) or
                    len(times) != len(outcome['actual']) or
                    times != outcome_times or times != sorted(set(times))):
                reason = 'target_identity_mismatch'
            elif not times or min(times) <= origin or max(times) > now:
                reason = 'target_not_mature'
            elif instant(outcome['outcome_recorded_at']) < max(times):
                reason = 'invalid_outcome_recording_time'
            elif any(not math.isfinite(v) or v < 0 for v in outcome['actual']):
                reason = 'invalid_outcome_values'
        if reason:
            excluded.append({'execution_id': eid, 'reason': reason})
            continue
        group = groups.setdefault(origin, {'outcome': outcome, 'results': []})
        group['results'].append(result)
    return list(groups.values()), excluded
