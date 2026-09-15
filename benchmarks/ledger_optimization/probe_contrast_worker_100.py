"""Synthetic Hermes contrast integration, including reading the full evidence artifact.

Run in a fresh process on the pinned Hermes/Gnomon runtimes on the local isolated runtime, separate from paid pod work. Real worker, models, ledger and native memory; scripted upstream responses.
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


from .probe_workflow_worker_097 import synthetic_jobs, verify_collection_project


def probe(capsule, runtime, output):
    capsule, runtime, output = (Path(p).resolve() for p in (capsule, runtime, output))
    if any(k.startswith('benchmarks.hermes_ml_checkpoint_v6') for k in sys.modules):
        raise ValueError('Fresh process required for each isolated collection capsule')
    manifest = json.loads((capsule/'capsule.json').read_text())
    package = capsule/'benchmarks/hermes_ml_checkpoint_v6'
    parent = sys.modules.get('benchmarks')
    if parent is not None:
        parent.__path__ = [str(capsule/'benchmarks')]
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
    jobs = synthetic_jobs(); series = jobs[0]['series_id']; seed = 7
    assert manifest['common_to_all_arms'] is True and 'workflow_audit_sha256' in manifest
    run.dump(output/'manifest.json', {'planned': 6, 'requested_seed': seed, 'synthetic': True,
                                     'sources': manifest['sources'], 'inventory': inventory, 'build': info})
    run.dump(output/'host-jobs.json', {series: jobs})
    current = {}; wire = []; checks = []; common_tables = {}
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
        if n > 5: raise AssertionError('Unexpected extra model request')
        if n == 1:
            check('agent sees four-fit collection budget', 'fresh backtest/collection batch costs four fits' in json.dumps(payload['messages']))
            check('tool schema describes unselected collection', 'collects an unselected forecast' in json.dumps(payload['tools']))
        lab_replies, read_replies = [], []
        for message in payload['messages']:
            if message.get('role') != 'tool': continue
            envelope = json.loads(message['content'])
            result = envelope.get('result')
            if type(result) is not dict: continue
            if 'stdout' in result:
                reply = json.loads(result['stdout'])
                if reply.get('status') == 'ok': lab_replies.append(reply)
            elif 'text' in result and 'path' in result: read_replies.append(result)
        if n == 2:
            start = next(r for r in lab_replies if r['operation'] == 'start')
            table = start['evidence_summary']['current_cv']['configurations']
            check('baseline scores are returned before exploration', len(table) == 1 and len(table[0]['fold_rmsle']) == 3)
        if n == 3:
            summary = next(r for r in reversed(lab_replies) if r['operation'] == 'backtest')['evidence_summary']
            check('comparison is available after ML backtest', summary['comparison_status'] == 'available')
            current['reference'] = summary['comparison']['evidence']
            common_tables[(current['arm'], current['round'])] = summary['current_cv']
            state = summary['comparison']['history_status']
            expected_state = ('last_review_had_no_historical_catalog' if current['round'] == 0 else 'pair_on_supplied_page') if current['arm'] == 'ledger' else 'not_requested'
            check('history state describes the arm and current origin', state == expected_state)
        if n == 4:
            ref = current['reference']
            page = next(r for r in read_replies if r['path'] == ref['path'])
            raw = page['text'].encode()
            check('Hermes received complete artifact bytes', page['next_offset'] is None and len(raw) == ref['bytes'])
            check('Hermes artifact digest matches returned reference', page['sha256'] == hashlib.sha256(raw).hexdigest() == ref['sha256'])
            full = json.loads(raw)
            check('artifact is bound to this exact synthetic task', full['current']['query']['series_id'] == series and full['current']['query']['origin'] == jobs[current['round']]['origin'])
            check('artifact preserves six current execution references', sum(len(r['folds']) for r in full['current']['runs']) == 6)
        marker = f'COLLECTION_SEED_{seed}_ARM_{current["arm"]}_ORIGIN_{current["round"]}'
        operations = {1: [('lab', {'operation': 'review'}), ('lab', {'operation': 'start'})],
                      2: [('lab', {'operation': 'backtest', 'config': config}),
                          ('memory', {'target': 'memory', 'action': 'add', 'content': marker})],
                      3: [('lab', {'operation': 'backtest', 'config': config}),
                          ('evidence_read', {'path': current.get('reference', {}).get('path', 'unused'), 'max_chars': 16384})],
                      4: [('lab', {'operation': 'commit', 'config': {'model': 'seasonal', 'season': 7}}),
                          ('notes_write', {'path': 'decision.json', 'text': '{"rationale":"synthetic collection integration"}'})],
                      5: []}[n]
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
                    check(f'{arm} round {job["round"]}: exact budgets', row['api_calls'] == 5 and row['numerical_attempts'] == 8)
                    verify_collection_project(output/arm/series/f'round-{job["round"]}'/'project', job, check)

            report = analyze(output)
            check('independent complete audit', report['complete'] and not report['audit_failures'] and not report['shutdown_record_gaps'])
            for arm in run.ARMS:
                own = f'COLLECTION_SEED_{seed}_ARM_{arm}_ORIGIN_0'
                second = [v['request']['messages'] for v in wire if v['arm'] == arm and v['round'] == 1]
                check(arm + ': own prior native memory loaded', own in json.dumps(second[0]))
                for other in run.ARMS:
                    if other != arm:
                        check(arm + ': other-arm memory absent', f'COLLECTION_SEED_{seed}_ARM_{other}_' not in json.dumps(second))
                other_seed = 19 if seed == 7 else 7
                check(arm + ': other-seed memory absent', f'COLLECTION_SEED_{other_seed}_' not in json.dumps(second))
                check(arm + ': native memory saved', own in (output/arm/series/'round-1/memory.json').read_text())
            for round_number in (0, 1):
                check('identical common CV table in all arms', common_tables[('plain', round_number)] == common_tables[('gnomon', round_number)] == common_tables[('ledger', round_number)])
            check('30 scripted model responses', len(wire) == 30)
            check('all six reminders independently audited', all(row['workflow_progress_audit']['reminder_requests'] == [4] for row in report['rows']))
            check('runtime inventory unchanged', run.runtime_inventory() == inventory)
            check('capsule sources unchanged', all(hashlib.sha256((package/name).read_bytes()).hexdigest() == digest
                                                   for name, digest in manifest['sources'].items()))
            result = {'passed': True, 'requested_seed': seed, 'checks': checks,
                      'scripted_model_responses': len(wire), 'numerical_attempts': 48,
                      'independent_audit_checks': report['audit_checks'], 'engy_calls': 0,
                      'final_gate_opened': False, 'sources': manifest['sources'],
                      'scope': 'Real Hermes worker/transport/model integration with scripted replies. All arms receive identical current CV tables, baseline scores are exposed, artifact references are read through actual Hermes tool dispatch and hashes verified, and historical contrast appears only for the ledger arm after maturation. Original fit/request budgets retained. Not a paid or efficacy test.'}
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
