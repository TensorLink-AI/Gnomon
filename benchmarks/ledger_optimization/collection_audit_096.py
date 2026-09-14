"""Independent sequential admission audit for the undeployed collection lab."""
import hashlib
import json


def audit_collection_events(events, job):
    """Check actual attempts against admitted missing work, including failures.

    Retrying a failed fit costs another attempt; successful partial results must
    be reused. Direct production fits remain valid for explicit commit recovery
    only after all three CV origins have successful results. The parent audit
    separately reconciles guarded calls, selections, metrics and visibility.
    """
    ends = (688, 702, 716, 730)
    cutoffs = {end: job['request']['timestamps'][end-1] for end in ends}
    attempts, successes, known, batches = {}, set(), {}, 0
    active = None

    def require(ok, message):
        if not ok:
            raise ValueError('Collection audit: ' + message)

    def cid_for(config):
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]

    for event in events:
        if event.get('task_origin') != job['origin']:
            continue
        kind = event['event']
        if kind == 'batch_admitted':
            raise ValueError('Collection audit: legacy CV-only admission in collection session')
        if kind == 'collection_batch_admitted':
            cid = event['config_id']
            require(cid == cid_for(event['config']), 'admission config identity')
            missing = [end for end in ends if (cid, end) not in known]
            require(type(event['fits']) is int and event['fits'] == len(missing) > 0, 'exact missing fit count')
            require(event['missing_ends'] == missing, 'exact missing origins')
            require(type(event['initial_baseline']) is bool, 'explicit baseline marker')
            if event['initial_baseline']:
                require(event['config'] == {'model': 'seasonal', 'season': 7}, 'only fixed baseline may waive reserve')
            reserve = 0 if event['initial_baseline'] else 1
            require(event['reserve'] == reserve, 'final-fit reserve')
            require(event['remaining_before'] == 60-len(attempts), 'remaining budget reconciles')
            require(event['remaining_before'] >= len(missing)+reserve, 'batch fits in budget')
            require(event['phase'] == 'exploration' or (event['phase'] == 'selection' and event['initial_baseline']), 'admission phase')
            require(event['selection_changed'] is False, 'collection cannot select')
            active = {'cid': cid, 'ends': missing.copy()}
            batches += 1
        elif kind == 'attempt':
            aid, cid, request = event['attempt_id'], event['config_id'], event['request']
            require(aid not in attempts, 'unique attempt ID')
            require(cid == cid_for(event['config']), 'attempt configuration identity')
            matched = [end for end in ends if request['cutoff'] == cutoffs[end]]
            require(len(matched) == 1, 'attempt belongs to a declared origin')
            end = matched[0]
            require(event['kind'] == ('forecast' if end == 730 else 'backtest'), 'attempt kind/origin')
            require(request['series_id'] == job['series_id'] and request['unit'] == job['request']['unit'], 'attempt task identity')
            if active and active['cid'] == cid and active['ends'] and active['ends'][0] == end:
                active['ends'].pop(0)
            else:
                require(end == 730 and all((cid, e) in known for e in ends[:-1]), 'unadmitted fit')
                active = None
            require((cid, end) not in known, 'successful execution was refitted')
            attempts[aid] = event
            require(len(attempts) <= 60, 'fit cap exceeded')
        elif kind == 'result':
            aid = event['attempt_id']
            require(aid in attempts and aid not in successes, 'result has one preceding attempt')
            attempt = attempts[aid]
            require(all(event[k] == attempt[k] for k in ('request', 'config', 'config_id', 'kind')), 'result matches attempt')
            end = next(e for e in ends if cutoffs[e] == event['request']['cutoff'])
            known[event['config_id'], end] = event
            successes.add(aid)
    return {'admitted_batches': batches, 'numerical_attempts': len(attempts),
            'successful_results': len(successes), 'failed_or_unfinished_attempts': len(attempts)-len(successes),
            'production_results': sum(end == 730 for cid, end in known)}
