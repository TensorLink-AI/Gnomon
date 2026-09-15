"""Full synthetic 72+240-session host/worker integration, with no Engy access."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import urllib.request
from unittest.mock import patch

from . import planning_host_107 as host
from . import control_collection_096 as controller
from . import continue_collection_096 as continuation


def fixture():
    times = [(datetime(2020, 1, 1, tzinfo=timezone.utc)+timedelta(days=i)).isoformat() for i in range(1200)]
    jobs = {}
    for s in range(4):
        series = f'synthetic-recipe-host-{s}'
        values = [float(10+s+i%7+3*(i%(11+s) == 0)) for i in range(1200)]; rows = []
        for n in range(26):
            begin, end = n*14, n*14+730; future = times[end:end+14]
            rows.append({'series_id': series, 'round': n, 'origin': times[end-1],
                'future_timestamps': future, 'outcome_recorded_at': future[-1], 'actual': values[end:end+14],
                'request': {'history': values[begin:end], 'timestamps': times[begin:end],
                    'future_timestamps': future, 'unit': 'widgets',
                    'past_covariate_names': ['promo'], 'future_covariate_names': ['promo'],
                    'past_covariates': [[float(i%(11+s) == 0)] for i in range(begin, end)],
                    'future_covariates': [[float(i%(11+s) == 0)] for i in range(end, end+14)]}})
        jobs[series] = rows
    return jobs


def worker(root, stage):
    settings = host.read(root/'settings.json'); plan = host.read(root/'plan.json')
    jobs = host.read(root/'jobs.json'); output = root/stage
    helper = continuation.configure(settings['capsule'], settings['runtime'], root/'jobs.json')
    run = helper.run
    from benchmarks.hermes_ml_checkpoint_v6 import service_admission
    contexts = {}; lock = threading.Lock(); checks = []; active = 0; maximum = 0
    config = {'model': 'ridge', 'window': 365, 'lags': 14, 'alpha': 10.0}

    def check(label, value):
        with lock: checks.append({'assertion': label, 'passed': bool(value)})
        if not value: raise AssertionError(label)

    class Response:
        status = 200
        def __init__(self, body): self.body = body
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(self.body).encode()

    def upstream(request, **kwargs):
        if request.full_url != 'https://api.engy.ai/v1/chat/completions': raise AssertionError('No external request permitted')
        token = request.get_header('Authorization').removeprefix('Bearer ')
        with lock:
            current = contexts[token]; current['requests'] += 1; n = current['requests']; meta = dict(current)
        payload = json.loads(request.data)
        check('fixed request settings', payload['seed'] == 7 and payload['model'] == 'deepseek-v4.1-flash'
              and payload['temperature'] == .2 and payload['max_tokens'] == 3072 and n <= 4)
        if stage == 'complete' and n == 1 and meta['round'] == 6:
            text = json.dumps(payload['messages'])
            check('own pilot memory carried into continuation', f"RECIPE107/{meta['series']}/{meta['arm']}/r5" in text)
            for other_s in jobs:
                for other_a in host.ARMS:
                    if (other_s, other_a) != (meta['series'], meta['arm']):
                        check('no cross-arm or cross-series memory', f'RECIPE107/{other_s}/{other_a}/r' not in text)
        if n == 2:
            replies = []
            for message in payload['messages']:
                if message.get('role') != 'tool': continue
                envelope = json.loads(message['content']).get('result', {})
                if 'stdout' in envelope: replies.append(json.loads(envelope['stdout']))
            if meta['arm'] != 'ledger': check('controls receive no historical plan', all('recipe_plan' not in r for r in replies))
            elif meta['round'] >= 4:
                review = next(r for r in replies if r['operation'] == 'review')
                start = next(r for r in replies if r['operation'] == 'start')
                check('mature review proposes explicit prerequisite', review['recipe_plan']['next_call']['arguments'] == {'operation': 'start'})
                check('real checkpoint enables historical backtest', start['recipe_plan']['next_call']['arguments'] == {'operation': 'backtest', 'config': config})
        marker = f"RECIPE107/{meta['series']}/{meta['arm']}/r{meta['round']}"
        actions = {1: [('lab', {'operation': 'review'}), ('lab', {'operation': 'start'})],
                   2: [('lab', {'operation': 'backtest', 'config': config}),
                       ('memory', {'target': 'memory', 'action': 'add', 'content': marker})],
                   3: [('lab', {'operation': 'commit', 'config': config})], 4: []}[n]
        calls = [{'id': f'{token}-{n}-{i}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}
                 for i, (name, args) in enumerate(actions)]
        message = {'role': 'assistant', 'content': None if calls else 'Synthetic host workflow complete.'}
        if calls: message['tool_calls'] = calls
        return Response({'id': f'{token}-{n}', 'object': 'chat.completion', 'created': 0,
                         'model': 'deepseek-v4.1-flash', 'choices': [{'index': 0, 'message': message,
                             'finish_reason': 'tool_calls' if calls else 'stop'}],
                         'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}})

    original = run.execute
    def execute(*args):
        nonlocal active, maximum
        _, arm, series, job = args[:4]; token = f'synthetic-{series}-{arm}-{job["round"]}'
        with lock:
            if token in contexts: raise AssertionError('Session executed twice')
            contexts[token] = {'arm': arm, 'series': series, 'round': job['round'], 'requests': 0}
            active += 1; maximum = max(maximum, active)
        try:
            result = original(*args[:-1], token)
            check('completed full workflow', result['valid'] and result['workflow_complete'])
            check('bounded calls and fits', result['api_calls'] == 4 and result['numerical_attempts'] == 8)
            return result
        finally:
            with lock: active -= 1

    def ready(_):
        return 200, b'{"choices":[{"message":{"content":"READY"}}],"usage":{"total_tokens":0},"synthetic":true}', 0.0

    with ExitStack() as stack:
        # Only the synthetic source hash and upstream responses are substituted.
        stack.enter_context(patch.object(run, 'SOURCE_SHA', host.sha(root/'jobs.json')))
        stack.enter_context(patch.object(run, 'execute', execute))
        stack.enter_context(patch.object(run, 'key', side_effect=AssertionError('No credential file access')))
        stack.enter_context(patch.object(service_admission, 'probe', ready))
        stack.enter_context(patch.object(urllib.request, 'urlopen', upstream))
        try:
            result = host.execute_stage(helper, root/'plan.json', jobs, output, stage=stage,
                credential=lambda: 'synthetic-unused-host-key', pilot=root/'pilot', pilot_launch=root/'pilot-launch')
            expected = 72 if stage == 'pilot' else 240
            check('each new session executed once', len(contexts) == expected)
            check('exact scripted request count', sum(c['requests'] for c in contexts.values()) == expected*4)
            check('two series executed concurrently', maximum == 2)
            host.dump(output/'host-probe.json', {'passed': True, 'checks': checks, 'new_sessions': expected,
                 'numerical_attempts': expected*8, 'scripted_responses': expected*4, 'engy_calls': 0,
                 'host_sources': host.source_identity(), 'stage_check': result, 'maximum_parallel_sessions': maximum})
        finally:
            if output.exists(): host.dump(output/'probe-progress.json', {'checks': checks, 'contexts': contexts})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--runtime', type=Path); parser.add_argument('--capsule', type=Path)
    parser.add_argument('--role', choices=('orchestrate', 'control', 'worker'), default='orchestrate')
    parser.add_argument('--stage', choices=('pilot', 'complete')); args = parser.parse_args(); root = args.root.absolute()
    if args.role == 'worker': return worker(root, args.stage)
    if args.role == 'control':
        command = [sys.executable, '-m', 'benchmarks.ledger_optimization.probe_planning_host_107',
                   '--root', str(root), '--role', 'worker', '--stage', args.stage]
        result = controller.supervise(command, root/(args.stage+'-launch'), root/args.stage)
        print(json.dumps(result), flush=True)
        if not result['complete'] or args.stage == 'pilot' and not result['continuation_gate_passed']: raise SystemExit(1)
        return
    if args.runtime is None or args.capsule is None: raise ValueError('Explicit runtime and capsule required')
    root.mkdir(parents=True, exist_ok=False)
    host.dump(root/'launch.json', {**controller.identity(__import__('os').getpid()), 'automatic_retry': False})
    jobs = fixture(); host.validate_jobs(jobs); host.dump(root/'jobs.json', jobs)
    manifest = host.read(Path('results/planning-worker-107-synthetic-001/manifest.json'))
    capsule = host.read(args.capsule/'capsule.json')
    if manifest['sources'] != capsule['sources']: raise ValueError('Require the already verified recipe worker')
    sources = host.source_identity()
    plan = {'synthetic': True, 'requested_seed': 7, 'capsule': capsule, 'host_sources': sources,
            'runtime_inventory': manifest['inventory'], 'build': manifest['build'],
            'arms': list(host.ARMS), 'planned': {'pilot_sessions': 72, 'continuation_sessions': 240, 'total_sessions': 312},
            'task_source_sha256': host.sha(root/'jobs.json'),
            'jobs_sha256': hashlib.sha256(json.dumps(jobs, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'final_gate_opened': False}
    host.dump(root/'plan.json', plan)
    host.dump(root/'settings.json', {'capsule': str(args.capsule.resolve()), 'runtime': str(args.runtime.resolve())})
    completed = []
    try:
        for stage in ('pilot', 'complete'):
            command = [sys.executable, '-m', 'benchmarks.ledger_optimization.probe_planning_host_107',
                       '--root', str(root), '--role', 'control', '--stage', stage]
            with (root/(stage+'-controller.stdout')).open('x') as out, (root/(stage+'-controller.stderr')).open('x') as err:
                child = subprocess.Popen(command, stdout=out, stderr=err)
                host.dump(root/(stage+'-controller-process.json'), controller.identity(child.pid)); status = child.wait()
            host.dump(root/(stage+'-controller-exit.json'), {'argv': command, 'exit_status': status})
            if status: raise RuntimeError(f'{stage} failed; no automatic retry')
            completed.append(stage); host.dump(root/(stage+'-finished.json'), {'complete': True})
        if sources != host.source_identity(): raise ValueError('Host sources changed during preflight')
        host.dump(root/'passed.json', {'passed': True, 'host_sources': sources, 'engy_calls': 0,
            'synthetic_sessions': 312, 'retained_pilot_sessions': 72, 'new_continuation_sessions': 240,
            'numerical_attempts': 2496, 'scripted_responses': 1248, 'final_gate_opened': False,
            'pilot_terminal_sha256': host.sha(root/'pilot-launch/FINISHED.json'),
            'complete_terminal_sha256': host.sha(root/'complete-launch/FINISHED.json'),
            'scope': 'Full synthetic host execution, actual Hermes and numerical tools; no paid efficacy claim.'})
    except BaseException as exc:
        host.dump(root/'INCOMPLETE.json', {'type': type(exc).__name__, 'completed': completed, 'automatic_retry': False}); raise


if __name__ == '__main__': main()
