"""Offline real-Hermes memory/skill persistence across two forecast origins."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

from benchmarks.hermes_ml_checkpoint_v4 import run as common
from benchmarks.hermes_ml_checkpoint_v4.transport import dump, sha, proxy, MODEL
from .run import HERE, prepare, native_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline-preflight', type=Path, required=True)
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    sources = {p.name: sha(p) for p in HERE.iterdir() if p.is_file()}
    shared = {p.name: sha(p) for p in common.HERE.iterdir() if p.is_file()}
    baseline = json.loads(args.baseline_preflight.read_text())
    assert baseline['passed'] and baseline['tested_sources'] == shared
    inventory = common.runtime_inventory()
    assert not inventory['plain']['gnomon']
    jobs = next(iter(json.loads(common.TASK_SOURCE.read_text()).values()))[:2]
    work, home = root / 'work', root / 'home'
    work.mkdir(); home.mkdir()
    python = common.OTHER / 'plain-venv/bin/python'
    env = common.environment(home, work, python)
    dump(home / 'config.yaml', {'terminal': {'backend': 'local', 'cwd': str(work)},
         'memory': {'memory_enabled': True, 'user_profile_enabled': True},
         'compression': {'enabled': False}, 'mcp_servers': {}})
    config = json.dumps({'model': 'ridge', 'window': 180, 'lags': 14, 'alpha': 10})
    skill = ('---\nname: retail-series-review\ndescription: Use when reviewing a retail series. Check evidence units.\n---\n'
             'Check units before comparing observations. SYNTHETIC_LESSON_ONE.\n')
    def tool(name, arguments, index):
        return {'id': f'call_{index}_{name}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments)}}
    class Reply:
        status = 200
        def __init__(self, value): self.value = value
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(self.value).encode()
    prior = []
    results = []
    for index, job in enumerate(jobs):
        prepare(work, job, prior, 'plain')
        sync = subprocess.run([str(python), 'lab.py', 'sync'], cwd=work, env=env, capture_output=True, text=True)
        assert sync.returncode == 0, sync.stderr
        out = root / f'origin-{index}'; out.mkdir()
        calls = []
        def fake(request, timeout):
            assert request.full_url == 'https://api.engy.ai/v1/chat/completions'
            payload = json.loads(request.data); calls.append(payload); n = len(calls)
            assert payload['model'] == MODEL == 'deepseek-v4.1-flash'
            if index == 1 and n == 1:
                assert 'SYNTHETIC_MEMORY_POINTER' in json.dumps(payload['messages'])
            commands = ['python lab.py start', "python lab.py backtest --config '" + config + "'",
                        "python lab.py commit --config '" + config + "'"]
            shift = 1 if index else 0
            if index == 1 and n == 1:
                batch = [tool('skill_view', {'name': 'retail-series-review'}, n)]
            elif 1 <= n - shift <= 3:
                batch = [tool('terminal', {'command': commands[n-shift-1], 'timeout': 60}, n)]
            elif n - shift == 4:
                if index == 0:
                    batch = [tool('memory', {'target': 'memory', 'action': 'add', 'content':
                             'Environment convention: SYNTHETIC_MEMORY_POINTER identifies this preflight; consult retail-series-review.'}, n),
                             tool('skill_manage', {'operations': [{'name': 'retail-series-review', 'action': 'create', 'content': skill}]}, n)]
                else:
                    batch = [tool('skill_manage', {'operations': [{'name': 'retail-series-review', 'action': 'patch',
                             'old_string': 'SYNTHETIC_LESSON_ONE', 'new_string': 'SYNTHETIC_LESSON_TWO'}]}, n)]
            else:
                assert n - shift == 5
                batch = []
            message = {'role': 'assistant', 'content': None, 'tool_calls': batch} if batch else {'role': 'assistant', 'content': 'Complete.'}
            return Reply({'id': f'mock-{index}-{n}', 'object': 'chat.completion', 'created': int(time.time()), 'model': MODEL,
                 'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if batch else 'stop'}],
                 'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20}})
        with patch('urllib.request.urlopen', side_effect=fake):
            with proxy(out, 'synthetic-test-key', work=work, deadline=time.time()+480) as url:
                result = subprocess.run([str(python), str(common.HERE/'worker.py'), str(work), str(out), url, 'alone'],
                     cwd=work, env=env, capture_output=True, text=True, timeout=180)
        (out/'stdout.txt').write_text(result.stdout); (out/'stderr.txt').write_text(result.stderr)
        assert result.returncode == 0, result.stderr
        grade = common.assess(work, job, {}, b'')
        assert grade['workflow_complete'] and grade['valid'] and grade['numerical_attempts'] == 8
        assert len(calls) == (5 if index == 0 else 6)
        messages = json.loads((out/'hermes-result.json').read_text())['messages']
        if index:
            assert any(m.get('role') == 'tool' and 'SYNTHETIC_LESSON_ONE' in str(m.get('content')) for m in messages)
        state = native_state(home); dump(out/'native-state.json', state)
        decoded = {name: base64.b64decode(v['base64']).decode() for name, v in state.items()}
        assert any('SYNTHETIC_MEMORY_POINTER' in text for name, text in decoded.items() if name.startswith('memories/'))
        marker = 'SYNTHETIC_LESSON_TWO' if index else 'SYNTHETIC_LESSON_ONE'
        assert any(marker in text for name, text in decoded.items() if name.endswith('SKILL.md'))
        prior.append({'series_id': job['series_id'], 'unit': job['request']['unit'], 'origin': job['origin'],
             'outcome_recorded_at': job['outcome_recorded_at'], 'actual': job['actual'], 'future_timestamps': job['future_timestamps'],
             'execution_id': grade['execution_id'], 'point': grade['point'], 'config': grade['config'],
             'fallback_used': grade['fallback_used'], 'rmsle': grade['rmsle']})
        results.append({'origin': job['origin'], 'requests': len(calls), 'grade': grade})
    assert inventory == common.runtime_inventory()
    assert sources == {p.name: sha(p) for p in HERE.iterdir() if p.is_file()}
    assert shared == {p.name: sha(p) for p in common.HERE.iterdir() if p.is_file()}
    dump(root/'passed.json', {'passed': True, 'tested_sources': sources, 'common_sources': shared,
         'paid_api_calls': 0, 'inherited_preflight_sha256': sha(args.baseline_preflight),
         'checks': ['Native memory write survives into next-origin model prompt',
                    'Native skill create, next-origin read and update persist',
                    'Both origins complete the unchanged backtest/commit workflow',
                    'Same numerical code, budgets and isolated no-Gnomon runtime'], 'origins': results})
    print(json.dumps({'passed': True, 'paid_api_calls': 0, 'native_origins_tested': 2}))


if __name__ == '__main__':
    main()
