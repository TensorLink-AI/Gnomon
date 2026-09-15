"""Audit annotation facts and timing from visible CSVs and actual tool replies.

Does not call the annotation or trusted-record reader. The established compact
renderer is reused only after independently checking its input facts/references.
"""
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path

from .contrast_view_100 import contrast_view


def audit_annotations(folder):
    folder = Path(folder); project = folder/'project'; checks = 0
    def check(label, value):
        nonlocal checks
        checks += 1
        if not value: raise ValueError('Annotation audit: '+label)
    def equal(a, b):
        if type(a) is dict and type(b) is dict:
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if type(a) is list and type(b) is list:
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        if type(a) in (int, float) and type(b) in (int, float):
            return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-12
        return type(a) is type(b) and a == b
    def read(name):
        p = Path(name)
        check('relative evidence path', not p.is_absolute() and '..' not in p.parts)
        for parent in [p, *p.parents]: check('no evidence symlink', not (project/parent).is_symlink())
        return (project/p).read_bytes()
    task = json.loads(read('task.json')); arm = json.loads(read('backend.json'))['arm']
    query = {k: task[k] for k in ('series_id', 'unit', 'horizon', 'origin')}
    check('frozen lab horizon', query['horizon'] == 14)
    history = list(csv.DictReader(io.StringIO(read('history.csv').decode())))
    check('frozen history length', len(history) == 730)
    names = [k for k in history[0] if k not in ('timestamp', 'value')]
    expected, actuals = {}, {}
    for end in (688, 702, 716):
        training, targets = history[:end], history[end:end+14]
        at = training[-1]['timestamp']
        expected[at] = {'history': [float(r['value']) for r in training],
                        'timestamps': [r['timestamp'] for r in training],
                        'future_timestamps': [r['timestamp'] for r in targets],
                        'past_covariates': [[float(r[k]) for k in names] for r in training],
                        'future_covariates': [[float(r[k]) for k in names] for r in targets],
                        'past_covariate_names': names, 'future_covariate_names': names,
                        'horizon': 14, 'series_id': query['series_id'], 'unit': query['unit'],
                        'cutoff': at, 'frequency': 'D', 'season': 7, 'known_time_cutoff': at}
        actuals[at] = [float(r['value']) for r in targets]
        check('CV targets already observed', datetime.fromisoformat(targets[-1]['timestamp']) <= datetime.fromisoformat(query['origin']))
    raw_log = read('experiments.jsonl')
    annotations = [json.loads(line) for line in read('comparison-annotations.jsonl').splitlines()]
    annotations = [r for r in annotations if r['annotation']['query'] == query]
    events = [json.loads(line) for line in (folder/'boundary-events.jsonl').read_bytes().splitlines()]
    requests, replies = {}, []
    for e in events:
        if e.get('stage') == 'requested' and e.get('tool') == 'lab':
            requests[e['tool_call_id']] = json.loads(e['raw_arguments'])
        if e.get('tool') != 'lab' and e.get('stage') != 'host_completion_check': continue
        if 'result' not in e or e['result'].get('status') != 'ok': continue
        envelope = e['result']['result']
        if type(envelope) is not dict or 'stdout' not in envelope: continue
        reply = json.loads(envelope['stdout'])
        args = requests[e['tool_call_id']] if e.get('tool') == 'lab' else {'operation': 'status'}
        replies.append((args, reply))
    check('every successful annotation retained', len(annotations) == sum('evidence_summary' in r for _, r in replies))
    all_rows = [json.loads(line) for line in raw_log.splitlines()]
    attempts = [r for r in all_rows if r.get('task_origin') == query['origin'] and r['event'] == 'attempt']
    previous_attempts, annotation_index, previous_prefix = 0, 0, 0
    latest_review, reviewed = None, False
    for args, reply in replies:
        n = reply['budget']['numerical_attempts']
        check('attempt count stays within budget', type(n) is int and previous_attempts <= n <= 60)
        check('returned new-fit count', n-previous_attempts == reply['numerical_calls_started'])
        check('no fits from evidence operations', args['operation'] in ('start', 'backtest', 'commit') or n == previous_attempts)
        for attempt in attempts[previous_attempts:n]:
            requested = ({'model': 'seasonal', 'season': 7} if args['operation'] == 'start' else args.get('config', {}))
            check('new fits preserve requested settings', all(attempt['config'].get(k) == v for k, v in requested.items()))
        previous_attempts = n
        if reply.get('status') != 'ok':
            check('rejection does not advertise a completed annotation', 'evidence_summary' not in reply)
            continue
        check('successful lab reply has annotation', 'evidence_summary' in reply)
        receipt = annotations[annotation_index]; annotation_index += 1
        annotation = reply['evidence_summary']
        check('annotation matches retained response', receipt['annotation'] == annotation)
        check('operation and pair preserved', receipt['operation'] == args['operation'] and receipt['requested_pair'] == args.get('pair'))
        check('query identity', annotation['query'] == query)
        for name, digest in receipt['input_sha256'].items(): check('input digest', hashlib.sha256(read(name)).hexdigest() == digest)
        check('all input digests supplied', set(receipt['input_sha256']) == {'task.json', 'history.csv', 'future.csv', 'backend.json'})
        ref = receipt['execution_log_prefix']; size = ref['bytes']
        check('monotonic available log prefix', type(size) is int and previous_prefix <= size <= len(raw_log))
        previous_prefix = size; prefix = raw_log[:size]
        check('execution prefix digest', hashlib.sha256(prefix).hexdigest() == ref['sha256'])
        check('prefix ends at record boundary', not prefix or prefix.endswith(b'\n'))
        rows = [json.loads(line) for line in prefix.splitlines()]
        rows = [r for r in rows if r.get('task_origin') == query['origin']]
        prefix_attempts = [r for r in rows if r['event'] == 'attempt']
        check('no later fit attempts included', len(prefix_attempts) == n and prefix_attempts == attempts[:n])
        allowed = {r['attempt_id'] for r in prefix_attempts}
        grouped, configs, executions = {}, {}, set()
        for row in rows:
            if row['event'] != 'result': continue
            check('result belongs to an admitted attempt', row['attempt_id'] in allowed)
            if row['kind'] != 'backtest': continue
            request = row['request']; at = request['cutoff']; ex = row['execution']; cid = row['config_id']
            check('exact visible CV request', request == expected.get(at))
            check('actuals equal visible targets', row['actual'] == actuals[at])
            check('configuration fingerprint', hashlib.sha256(json.dumps(row['config'], sort_keys=True).encode()).hexdigest()[:16] == cid)
            check('versioned provider', ex['provider'] == row['config']['model']+'_'+cid and ex['revision'] == 'ml-lab-v1:'+cid)
            check('execution point correspondence', ex['result']['point'] == row['point'] and len(row['point']) == 14)
            check('valid scored values', all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in row['point']))
            eid = ex['execution_id']; check('unique CV execution', eid not in executions); executions.add(eid)
            score = math.sqrt(sum((math.log1p(p)-math.log1p(a))**2 for p, a in zip(row['point'], actuals[at], strict=True))/14)
            check('reported RMSLE', equal(score, row['metrics']['rmsle']) and row['metrics']['n'] == 14)
            folds = grouped.setdefault(cid, {}); check('unique CV origin', at not in folds)
            folds[at] = {'origin': at, 'target_end': request['future_timestamps'][-1], 'n': 14,
                         'rmsle': score, 'execution_id': eid}
            configs[cid] = {'config_id': cid, 'config': row['config'], 'provider': ex['provider'], 'revision': ex['revision']}
        complete = {c: {**configs[c], 'folds': [f[t] for t in sorted(f)]} for c, f in grouped.items() if set(f) == set(expected)}
        excluded = [{'config_id': c, 'reason': 'incomplete_current_cv', 'completed_folds': len(f), 'required_folds': 3,
                     'missing_origins': sorted(set(expected)-set(f))} for c, f in sorted(grouped.items()) if set(f) != set(expected)]
        scores = {c: sum(f['rmsle'] for f in r['folds'])/3 for c, r in complete.items()}
        ordered = sorted(scores, key=lambda c: (scores[c], c))
        table = {'configurations': [{'config_id': c, 'config': configs[c]['config'], 'mean_rmsle': scores[c],
                                    'rank': 1+sum(v < scores[c] for v in scores.values()),
                                    'fold_rmsle': [f['rmsle'] for f in complete[c]['folds']]} for c in ordered],
                 'fold_origins': sorted(expected) if complete else [], 'excluded': excluded}
        check('independent current table and ranks', equal(annotation['current_cv'], table))
        if arm == 'ledger' and args['operation'] == 'review':
            reviewed = True
            latest_review = reply['result'] if reply['result'].get('schema_version') == 'agent-review-088' else None
            check('review is current or explicitly empty', latest_review is not None or reply['result'].get('status') == 'insufficient_evidence')
        pair, focus = None, None
        if args.get('pair') is not None:
            focus = 'explicit_review_pair'
            if all(c in complete for c in args['pair']): pair = args['pair']
            else: check('missing pair does not become a different task', annotation['comparison_status'] == 'requested_pair_lacks_complete_current_cv')
        elif len(ordered) >= 2:
            new = reply['result'].get('config_id') if args['operation'] == 'backtest' else None
            pair = [new, next(c for c in ordered if c != new)] if new in complete else ordered[:2]
            focus = 'new_backtest_vs_lowest_current_cv_alternative' if new in complete else 'two_lowest_current_cv_means'
        else: check('insufficient current pair identified', annotation['comparison_status'] == 'fewer_than_two_complete_current_configurations')
        check('focus rule', annotation.get('focus_rule') == focus)
        if pair is not None:
            view = annotation['comparison']; ref = view['evidence']; raw = read(ref['path'])
            check('stored artifact integrity', len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'])
            artifact = json.loads(raw)
            check('artifact contains only available current folds', equal(artifact['current'], {'query': query, 'runs': [complete[c] for c in pair]}))
            kwargs = {'review': latest_review, 'full_evidence_bytes': read(latest_review['full_evidence']['path'])} if latest_review is not None else {}
            expected_view, expected_raw = contrast_view(artifact['current'], **kwargs)
            if reviewed and latest_review is None: expected_view['history_status'] = 'last_review_had_no_historical_catalog'
            check('presentation matches verified inputs and latest requested history', view == expected_view and raw == expected_raw)
        else: check('no unrequested pair artifact', 'comparison' not in annotation)
        check('annotation declares no additional execution', annotation['annotation_diagnostics'] == {'provider_calls': 0, 'additional_ledger_queries': 0, 'forecast_selection_made': False})
    return {'checks': checks, 'annotations': annotation_index, 'passed': True,
            'scope': 'CV facts independently reconstructed from visible CSVs; prefix attempt counts matched to actual lab responses. Established renderer verifies presentation only after input checks. No fits or queries.'}
