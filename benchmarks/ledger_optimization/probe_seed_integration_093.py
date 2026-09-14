"""Full synthetic two-origin integration for one isolated seed capsule.

Run in a fresh process on the pinned Hermes/Gnomon runtimes after paid work is
idle. Real worker, models, ledger and native memory; scripted upstream responses.
No credentials or real dataset are read, and no Engy request is sent.
"""
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
    times = [(datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)).isoformat() for i in range(800)]
    values = [float(10 + i % 7 + 3 * (i % 11 == 0)) for i in range(800)]
    jobs = []
    for number in range(2):
        begin, end = number*14, number*14+730
        future = times[end:end+14]
        jobs.append({'series_id': 'synthetic-seed-integration', 'round': number, 'origin': times[end-1],
                     'future_timestamps': future, 'outcome_recorded_at': future[-1], 'actual': values[end:end+14],
                     'request': {'history': values[begin:end], 'timestamps': times[begin:end],
                                 'future_timestamps': future, 'unit': 'widgets',
                                 'past_covariate_names': ['promo'], 'future_covariate_names': ['promo'],
                                 'past_covariates': [[float(i % 11 == 0)] for i in range(begin, end)],
                                 'future_covariates': [[float(i % 11 == 0)] for i in range(end, end+14)]}})
    return jobs


def probe(capsule, runtime, output):
    capsule, runtime, output = (Path(p).resolve() for p in (capsule, runtime, output))
    if any(k.startswith('benchmarks.hermes_ml_checkpoint_v6') for k in sys.modules):
        raise ValueError('Fresh process required for each isolated seed capsule')
    manifest = json.loads((capsule/'capsule.json').read_text())
    package = capsule/'benchmarks/hermes_ml_checkpoint_v6'
    for name, digest in manifest['sources'].items():
        assert hashlib.sha256((package/name).read_bytes()).hexdigest() == digest
    os.environ['LEDGER_ML_RUNTIME_ROOT'] = str(runtime)
    sys.path.insert(0, str(capsule))
    from benchmarks.hermes_ml_checkpoint_v6 import run, service_admission
    from benchmarks.hermes_ml_checkpoint_v6.analyze import analyze
    assert run.HERE == package and run.REPO == capsule and run.OTHER == runtime
    output.mkdir(parents=True, exist_ok=False)
    runtimes = {a: runtime/('plain-venv' if a == 'plain' else 'gnomon-venv')/'bin/python' for a in run.ARMS}
    inventory = run.runtime_inventory()
    info = json.loads(subprocess.check_output([str(runtimes['gnomon']), '-I', '-c',
        'import json;from gnomon.build_info import build_info;print(json.dumps(build_info()))'], text=True))
    assert info['package_version'] == '1.2.0' and info['source_sha256'] == run.BUILD_SHA
    jobs = synthetic_jobs(); series = jobs[0]['series_id']; seed = manifest['requested_seed']
    run.dump(output/'manifest.json', {'planned': 6, 'requested_seed': seed, 'synthetic': True,
                                     'sources': manifest['sources'], 'inventory': inventory, 'build': info})
    run.dump(output/'host-jobs.json', {series: jobs})
    current = {}; wire = []; checks = []
    config = {'model': 'ridge', 'window': 90, 'lags': 7, 'alpha': 10}
    def check(label, value):
        checks.append({'assertion': label, 'passed': bool(value)})
        if not value: raise AssertionError(label)
    class Response:
        status = 200
        def __init__(self, body): self.body = body
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(self.body).encode()
    def upstream(request, **kwargs):
        if request.full_url != 'https://api.engy.ai/v1/chat/completions':
            raise AssertionError('Unexpected outbound URL; no real network allowed')
        payload = json.loads(request.data); current['n'] += 1; n = current['n']
        check('forwarded requested seed', payload['seed'] == seed)
        check('unchanged model parameters', payload['model'] == 'deepseek-v4.1-flash'
              and payload['temperature'] == .2 and payload['max_tokens'] == 3072)
        wire.append({'arm': current['arm'], 'round': current['round'], 'request': payload, 'synthetic': True})
        if n > 4: raise AssertionError('Unexpected extra model request')
        marker = f'CAPSULE_SEED_{seed}_ARM_{current["arm"]}_ORIGIN_{current["round"]}'
        operations = {1: [('lab', {'operation': 'review'}), ('lab', {'operation': 'start'})],
                      2: [('lab', {'operation': 'backtest', 'config': config}),
                          ('memory', {'target': 'memory', 'action': 'add', 'content': marker})],
                      3: [('lab', {'operation': 'commit', 'config': config}),
                          ('notes_write', {'path': 'decision.json', 'text': '{"rationale":"synthetic seed integration"}'})],
                      4: []}[n]
        calls = [{'id': f'synthetic-{len(wire)}-{i}', 'type': 'function',
                  'function': {'name': name, 'arguments': json.dumps(arguments)}}
                 for i, (name, arguments) in enumerate(operations)]
        message = {'role': 'assistant', 'content': None if calls else 'Synthetic workflow complete.'}
        if calls: message['tool_calls'] = calls
        return Response({'id': f'synthetic-{len(wire)}', 'object': 'chat.completion', 'created': 0,
                         'model': 'deepseek-v4.1-flash', 'choices': [{'index': 0, 'message': message,
                          'finish_reason': 'tool_calls' if calls else 'stop'}],
                         'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}})
    def ready(_):
        return 200, b'{"choices":[{"message":{"content":"READY"}}],"usage":{"total_tokens":0},"synthetic":true}', 0.
    try:
        with patch.object(urllib.request, 'urlopen', upstream), patch.object(service_admission, 'probe', ready), \
             patch.object(run, 'key', side_effect=AssertionError('No credentials in integration probe')):
            prior = {a: [] for a in run.ARMS}
            for job in jobs:
                offset = job['round'] % 3
                for arm in run.ARMS[offset:] + run.ARMS[:offset]:
                    current.update(arm=arm, round=job['round'], n=0)
                    row = run.execute(output, arm, series, job, prior[arm], runtimes[arm], 'synthetic-only')
                    check(f'{arm} round {job["round"]}: full workflow', row['valid'] and row['workflow_complete'])
                    check(f'{arm} round {job["round"]}: exact budgets', row['api_calls'] == 4 and row['numerical_attempts'] == 8)
            report = analyze(output)
            check('independent complete audit', report['complete'] and not report['audit_failures'] and not report['shutdown_record_gaps'])
            for arm in run.ARMS:
                own = f'CAPSULE_SEED_{seed}_ARM_{arm}_ORIGIN_0'
                second = [v['request']['messages'] for v in wire if v['arm'] == arm and v['round'] == 1]
                check(arm + ': own prior native memory loaded', own in json.dumps(second[0]))
                for other in run.ARMS:
                    if other != arm:
                        check(arm + ': other-arm memory absent', f'CAPSULE_SEED_{seed}_ARM_{other}_' not in json.dumps(second))
                other_seed = 19 if seed == 7 else 7
                check(arm + ': other-seed memory absent', f'CAPSULE_SEED_{other_seed}_' not in json.dumps(second))
                check(arm + ': native memory saved', own in (output/arm/series/'round-1/memory.json').read_text())
            check('24 scripted model responses', len(wire) == 24)
            check('runtime inventory unchanged', run.runtime_inventory() == inventory)
            check('capsule sources unchanged', all(hashlib.sha256((package/name).read_bytes()).hexdigest() == digest
                                                   for name, digest in manifest['sources'].items()))
            result = {'passed': True, 'requested_seed': seed, 'checks': checks,
                      'scripted_model_responses': len(wire), 'numerical_attempts': 48,
                      'independent_audit_checks': report['audit_checks'], 'engy_calls': 0,
                      'final_gate_opened': False, 'sources': manifest['sources'],
                      'scope': 'Real worker/models/native memory and ledger with scripted replies. No model efficacy or provider seed-support claim.'}
            run.dump(output/'passed.json', result)
            return result
    finally:
        run.dump(output/'probe-evidence.json', {'checks': checks, 'wire': wire, 'engy_calls': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.capsule, args.runtime, args.output)
    print(json.dumps({k: result[k] for k in ('passed', 'requested_seed', 'numerical_attempts', 'independent_audit_checks', 'engy_calls')}))
