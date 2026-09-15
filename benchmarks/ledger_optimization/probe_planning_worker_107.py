"""Actual Hermes boundary, five synthetic origins and three arms; no Engy calls."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
from unittest.mock import patch


def synthetic_jobs():
    times = [(datetime(2020, 1, 1, tzinfo=timezone.utc)+timedelta(days=i)).isoformat() for i in range(800)]
    values = [float(10+i%7+3*(i%11 == 0)) for i in range(800)]
    jobs = []
    for n in range(5):
        begin, end = n*14, n*14+730; future = times[end:end+14]
        jobs.append({'series_id': 'synthetic-planning-worker-107', 'round': n, 'origin': times[end-1],
                     'future_timestamps': future, 'outcome_recorded_at': future[-1], 'actual': values[end:end+14],
                     'request': {'history': values[begin:end], 'timestamps': times[begin:end],
                                 'future_timestamps': future, 'unit': 'widgets',
                                 'past_covariate_names': ['promo'], 'future_covariate_names': ['promo'],
                                 'past_covariates': [[float(i%11 == 0)] for i in range(begin, end)],
                                 'future_covariates': [[float(i%11 == 0)] for i in range(end, end+14)]}})
    return jobs


def probe(capsule, runtime, output):
    capsule, runtime, output = map(lambda p: Path(p).resolve(), (capsule, runtime, output))
    if any(k.startswith('benchmarks.hermes_ml_checkpoint_v6') for k in sys.modules):
        raise ValueError('Fresh process required')
    manifest = json.loads((capsule/'capsule.json').read_text())
    if manifest['status'] != 'offline_recipe_plan_worker_not_dispatch_ready':
        raise ValueError('Exact isolated recipe-plan prototype required')
    package = capsule/'benchmarks/hermes_ml_checkpoint_v6'
    for name, sha in manifest['sources'].items():
        assert hashlib.sha256((package/name).read_bytes()).hexdigest() == sha
    os.environ['LEDGER_ML_RUNTIME_ROOT'] = str(runtime)
    sys.path.insert(0, str(capsule))
    sys.modules['benchmarks'].__path__ = [str(capsule/'benchmarks')]
    from benchmarks.hermes_ml_checkpoint_v6 import run, service_admission
    from benchmarks.hermes_ml_checkpoint_v6.analyze import analyze
    assert run.HERE == package and run.OTHER == runtime
    output.mkdir(parents=True, exist_ok=False)
    runtimes = {a: runtime/('plain-venv' if a == 'plain' else 'gnomon-venv')/'bin/python' for a in run.ARMS}
    inventory = run.runtime_inventory()
    info = json.loads(subprocess.check_output([str(runtimes['gnomon']), '-I', '-c',
        'import json;from gnomon.build_info import build_info;print(json.dumps(build_info()))'], text=True))
    assert info['package_version'] == '1.2.0' and info['source_sha256'] == run.BUILD_SHA
    jobs = synthetic_jobs(); series = jobs[0]['series_id']; seed = manifest.get('requested_seed', 7)
    run.dump(output/'manifest.json', {'planned': 15, 'requested_seed': seed, 'synthetic': True,
                                     'sources': manifest['sources'], 'inventory': inventory, 'build': info})
    run.dump(output/'host-jobs.json', {series: jobs})
    current, wire, checks, tables = {}, [], [], {}
    config = {'model': 'ridge', 'window': 365, 'lags': 14, 'alpha': 10.0}

    def check(label, condition):
        checks.append({'assertion': label, 'passed': bool(condition)})
        if not condition: raise AssertionError(label)

    class Response:
        status = 200
        def __init__(self, body): self.body = body
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(self.body).encode()

    def upstream(request, **kwargs):
        if request.full_url != 'https://api.engy.ai/v1/chat/completions':
            raise AssertionError('Unexpected outbound URL')
        payload = json.loads(request.data); current['n'] += 1; n = current['n']
        check('fixed model and request settings', payload['model'] == 'deepseek-v4.1-flash'
              and payload['temperature'] == .2 and payload['max_tokens'] == 3072 and payload['seed'] == seed)
        check('bounded scripted conversation', n <= 6)
        wire.append({'arm': current['arm'], 'round': current['round'], 'request': payload, 'synthetic': True})
        replies, pages = [], []
        for message in payload['messages']:
            if message.get('role') != 'tool': continue
            envelope = json.loads(message['content']); result = envelope.get('result')
            if type(result) is not dict: continue
            if 'stdout' in result:
                answer = json.loads(result['stdout'])
                if answer.get('status') == 'ok': replies.append(answer)
            elif 'text' in result and 'path' in result: pages.append(result)
        warm = current['arm'] == 'ledger' and current['round'] == 4
        if current['arm'] != 'ledger':
            check('control receives no historical recipe plan', all('recipe_plan' not in r for r in replies))
        if n == 2:
            review = next(r for r in reversed(replies) if r['operation'] == 'review')
            if warm:
                plan = review['recipe_plan']
                check('warm agent sees explicit start prerequisite', plan['next_call']['arguments'] == {'operation': 'start'})
                check('conditional recipe not presented as currently admissible', len(plan['recipes']) == 1
                      and plan['recipes'][0]['arguments']['config'] == config
                      and not plan['recipes'][0]['admissible_now']
                      and plan['recipes'][0]['conditional_budget_feasible'])
                check('four matured origins support the recipe', plan['recipes'][0]['historical_origins_with_paired_evidence'] == 4)
                current['reference'] = plan['evidence']
        if n == 3 and warm:
            started = next(r for r in reversed(replies) if r['operation'] == 'start')
            check('actual checkpoint changes next action to backtest', started['recipe_plan']['next_call']['arguments'] ==
                  {'operation': 'backtest', 'config': config} and started['budget']['numerical_attempts'] == 4)
        if n == 4:
            tested = next(r for r in reversed(replies) if r['operation'] == 'backtest')
            check('real common backtest costs eight total attempts', tested['budget']['numerical_attempts'] == 8)
            tables[current['arm'], current['round']] = tested['evidence_summary']['current_cv']
            if not warm: current['reference'] = tested['evidence_summary']['comparison']['evidence']
        if n == 5:
            ref = current['reference']; page = next(p for p in pages if p['path'] == ref['path'])
            raw = page['text'].encode()
            check('actual Hermes evidence read returns full bytes and hash', page['next_offset'] is None
                  and len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == page['sha256'] == ref['sha256'])
            if warm:
                artifact = json.loads(raw)
                check('full original plan preserves missing-checkpoint state', artifact['plan']['observed_state']['checkpoint_available'] is False)
                check('full original plan is for this task', artifact['query']['origin'] == jobs[4]['origin']
                      and artifact['query']['series_id'] == series)
        if n == 6:
            committed = next(r for r in reversed(replies) if r['operation'] == 'commit')
            check('commit selects an executed forecast after comparison', committed['result']['selection_after_comparison'])
            check('commit does not suggest more exploration', 'recipe_plan' not in committed)
        marker = f'RECIPE_SEED_{seed}_ARM_{current["arm"]}_ORIGIN_{current["round"]}'
        actions = {1: [('lab', {'operation': 'review'})],
                   2: [('lab', {'operation': 'start'}), ('memory', {'target': 'memory', 'action': 'add', 'content': marker})],
                   3: [('lab', {'operation': 'backtest', 'config': config})],
                   4: [('evidence_read', {'path': current.get('reference', {}).get('path', 'unused'), 'max_chars': 16384})],
                   5: [('lab', {'operation': 'commit', 'config': config})], 6: []}[n]
        calls = [{'id': f'synthetic-{len(wire)}-{i}', 'type': 'function',
                  'function': {'name': name, 'arguments': json.dumps(args)}} for i, (name, args) in enumerate(actions)]
        message = {'role': 'assistant', 'content': None if calls else 'Synthetic sequence complete.'}
        if calls: message['tool_calls'] = calls
        return Response({'id': f'synthetic-{len(wire)}', 'object': 'chat.completion', 'created': 0,
                         'model': 'deepseek-v4.1-flash', 'choices': [{'index': 0, 'message': message,
                          'finish_reason': 'tool_calls' if calls else 'stop'}],
                         'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}})

    def ready(_):
        return 200, b'{"choices":[{"message":{"content":"READY"}}],"usage":{"total_tokens":0},"synthetic":true}', 0.

    try:
        with patch.object(urllib.request, 'urlopen', upstream), patch.object(service_admission, 'probe', ready), \
             patch.object(run, 'key', side_effect=AssertionError('No credentials in synthetic preflight')):
            prior = {a: [] for a in run.ARMS}
            for job in jobs:
                offset = job['round'] % 3
                for arm in run.ARMS[offset:]+run.ARMS[:offset]:
                    current.clear(); current.update(arm=arm, round=job['round'], n=0)
                    row = run.execute(output, arm, series, job, prior[arm], runtimes[arm], 'synthetic-only')
                    check(f'{arm} origin {job["round"]}: completed', row['valid'] and row['workflow_complete'])
                    check(f'{arm} origin {job["round"]}: budget', row['api_calls'] == 6 and row['numerical_attempts'] == 8)
                    project = output/arm/series/f'round-{job["round"]}'/'project'
                    selected = json.loads((project/'checkpoint.json').read_text())
                    check('explicit Ridge selection, two current configurations', selected['config'] == config
                          and selected['selection_after_comparison'] and len(selected['compared_configurations']) == 2)
            report = analyze(output)
            check('complete original independent workflow and comparison audit', report['complete']
                  and not report['audit_failures'] and not report['shutdown_record_gaps'])
            for arm in run.ARMS:
                for n in range(1, 5):
                    first = next(w['request']['messages'] for w in wire if w['arm'] == arm and w['round'] == n)
                    check('own native memory persists', f'RECIPE_SEED_{seed}_ARM_{arm}_ORIGIN_0' in json.dumps(first))
                    check('other arms native memory absent', all(f'RECIPE_SEED_{seed}_ARM_{other}_' not in json.dumps(first)
                                                               for other in run.ARMS if other != arm))
            for n in range(5):
                check('identical numerical CV table across three arms', tables['plain', n] == tables['gnomon', n] == tables['ledger', n])
            check('exactly 90 scripted responses', len(wire) == 90)
            check('runtime unchanged', run.runtime_inventory() == inventory)
            check('frozen capsule sources unchanged', all(hashlib.sha256((package/n).read_bytes()).hexdigest() == s for n, s in manifest['sources'].items()))
            result = {'passed': True, 'requested_seed': seed, 'checks': checks, 'sessions': 15,
                      'scripted_model_responses': len(wire), 'numerical_attempts': 120,
                      'independent_existing_audit_checks': report['audit_checks'],
                      'independent_recipe_annotation_audit_completed': False,
                      'engy_calls': 0, 'final_gate_opened': False, 'sources': manifest['sources'],
                      'scope': 'Actual Hermes/tool transport with scripted responses; original workflow audit and recipe handoff assertions. Separate full recipe audit remains required. No efficacy result.'}
            run.dump(output/'passed.json', result)
            return result
    finally:
        run.dump(output/'probe-evidence.json', {'checks': checks, 'wire': wire, 'engy_calls': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capsule', 'runtime', 'output'): parser.add_argument('--'+name, required=True, type=Path)
    a = parser.parse_args(); result = probe(a.capsule, a.runtime, a.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'sources')}, indent=2))
