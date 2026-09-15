"""Execute one proposed sequence after four synthetic matured ledger origins.

Uses frozen lab files and published 1.2.0, with scripted CLI actions, no agent,
API credentials, Engy calls, real development values or held-out values.
"""
import argparse
import ast
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from .planning_sequence_107 import planning_sequence


def probe(root, runtime):
    root, runtime = Path(root).resolve(), Path(runtime).resolve()
    root.mkdir(parents=True, exist_ok=False)
    work = root/'work'; work.mkdir()
    package = Path('results/contrast-capsule-100-offline-003/capsule/benchmarks/hermes_ml_checkpoint_v6').resolve()
    capsule = json.loads((package.parent.parent/'capsule.json').read_text())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for name, expected in capsule['sources'].items():
        assert sha(package/name) == expected
    run_ast = ast.parse((package/'run.py').read_text())
    files = next(ast.literal_eval(n.value) for n in run_ast.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'PROJECT_FILES' for t in n.targets))
    prepare_node = next(n for n in run_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    def dump(path, value):
        Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    ns = {'datetime': datetime, 'csv': csv, 'dump': dump, 'PROJECT_FILES': files,
          'shutil': shutil, 'HERE': package}
    exec(compile(ast.Module(body=[prepare_node], type_ignores=[]), str(package/'run.py'), 'exec'), ns)
    numerical_ast = ast.parse((package/'numerical.py').read_text())
    config_node = next(n for n in numerical_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'configuration')
    cfg_ns = {'math': math}
    exec(compile(ast.Module(body=[config_node], type_ignores=[]), str(package/'numerical.py'), 'exec'), cfg_ns)
    canonicalize = cfg_ns['configuration']
    python = runtime/'gnomon-venv/bin/python'
    env = {k: v for k, v in os.environ.items() if k in ('PATH', 'LANG', 'LC_ALL', 'TZ')}
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    inventory_code = 'import json,importlib.metadata; print(json.dumps({d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}))'
    inventory = subprocess.run([str(python), '-I', '-c', inventory_code], capture_output=True, text=True, env=env, check=True)
    expected = json.loads(Path('results/m5-ml-capsule-offline-001/synthetic-worker-002/manifest.json').read_text())['inventory']['gnomon']['packages']
    assert json.loads(inventory.stdout) == expected
    dump(root/'runtime.json', {'packages': json.loads(inventory.stdout), 'python': str(python)})
    times = [(datetime(2020, 1, 1, tzinfo=timezone.utc)+timedelta(days=i)).isoformat() for i in range(800)]
    values = [float(10+i%7+3*(i%11 == 0)) for i in range(800)]
    jobs = []
    for n in range(5):
        begin = n*14; end = begin+730; future = times[end:end+14]
        jobs.append({'series_id': 'synthetic-planning-sequence-107', 'round': n, 'origin': times[end-1],
                     'future_timestamps': future, 'outcome_recorded_at': future[-1], 'actual': values[end:end+14],
                     'request': {'history': values[begin:end], 'timestamps': times[begin:end],
                                 'future_timestamps': future, 'past_covariate_names': ['promo'], 'future_covariate_names': ['promo'],
                                 'past_covariates': [[float(i%11 == 0)] for i in range(begin, end)],
                                 'future_covariates': [[float(i%11 == 0)] for i in range(end, end+14)], 'unit': 'widgets'}})
    dump(root/'synthetic-jobs.json', jobs)
    dump(root/'inputs.json', {'capsule': str(package.parent.parent), 'capsule_sha256': sha(package.parent.parent/'capsule.json'),
                             'sources': capsule['sources'], 'probe_sha256': sha(__file__),
                             'sequence_sha256': sha(Path(__file__).with_name('planning_sequence_107.py')),
                             'engy_calls': 0, 'data_scope': 'synthetic_only'})
    config = canonicalize({'model': 'ridge', 'window': 365, 'lags': 14, 'alpha': 10})
    prior, total_commands, checks, source_hashes = [], 0, 0, {}
    for job in jobs:
        n = job['round']; out = root/f'round-{n}'; out.mkdir()
        previous = (work/'experiments.jsonl').read_bytes() if (work/'experiments.jsonl').exists() else b''
        ns['prepare'](work, job, prior, 'ledger')
        source_hashes = {name: sha(work/name) for name in files}
        assert source_hashes == {name: capsule['sources'][name] for name in files}
        count = 0; deadline = time.time()+480

        def call(operation, config=None):
            nonlocal count, total_commands
            count += 1; total_commands += 1
            # Scripted tool-step accounting; these are not API requests.
            dump(work/'agent-budget.json', {'forwarded_requests': count, 'remaining_requests': 16-count, 'deadline_epoch': deadline})
            command = [str(python), 'lab.py', operation]
            if config is not None: command += ['--config', json.dumps(config)]
            result = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, timeout=90)
            stem = out/f'{count:02d}-{operation}'
            stem.with_suffix('.stdout').write_text(result.stdout); stem.with_suffix('.stderr').write_text(result.stderr)
            dump(stem.with_suffix('.json'), {'argv': command, 'cwd': str(work), 'exit_status': result.returncode})
            assert result.returncode == 0, result.stderr or result.stdout
            payload = json.loads(result.stdout); assert payload['status'] == 'ok'
            return payload

        call('sync')
        if n < 4:
            call('start'); call('backtest', config); call('commit', config)
        else:
            review_call = call('review'); review = review_call['result']
            assert not (work/'checkpoint.json').exists()
            current = {'query': review['query'], 'executed_configurations': [], 'checkpoint_available': False,
                       'budget': review_call['budget'], 'initial_checkpoint': {
                           'config': canonicalize({'model': 'seasonal', 'season': 7}), 'max_numerical_attempts': 4}}
            full_path = work/review['full_evidence']['path']
            full_bytes = full_path.read_bytes()
            result = planning_sequence(review, full_bytes, current, canonicalize)
            dump(out/'planning-sequence.json', result)
            assert result['next_call']['arguments'] == {'operation': 'start'}
            assert len(result['recipes']) == 1 and result['recipes'][0]['historical_origins_with_paired_evidence'] == 4
            conditional = result['recipes'][0]['next_call']
            assert not conditional['admissible_now'] and conditional['conditional_budget_feasible']
            assert conditional['arguments']['config'] == config
            started = call('start')
            assert started['budget']['numerical_attempts'] == 4
            assert (work/'checkpoint.json').is_file()
            # Re-render with actual established state and fresh returned budget.
            baseline = json.loads((work/'checkpoint.json').read_text())
            current.update(checkpoint_available=True, budget=started['budget'], executed_configurations=[{
                k: baseline[k] for k in ('config_id', 'config', 'provider', 'revision')}])
            refreshed = planning_sequence(review, full_bytes, current, canonicalize)
            dump(out/'planning-after-checkpoint.json', refreshed)
            assert refreshed['next_call']['arguments'] == conditional['arguments']
            assert refreshed['next_call']['admissible_now']
            tested = call('backtest', refreshed['next_call']['arguments']['config'])
            assert tested['budget']['numerical_attempts'] == 8
            committed = call('commit', config)
            assert committed['budget']['numerical_attempts'] == 8
            assert full_path.read_bytes() == full_bytes
            checks += 15
        checkpoint = json.loads((work/'checkpoint.json').read_text())
        assert checkpoint['selection_after_comparison'] and not checkpoint['baseline_only']
        assert checkpoint['future_timestamps'] == job['future_timestamps']
        assert checkpoint['series_id'] == job['series_id'] and checkpoint['unit'] == 'widgets'
        assert len(checkpoint['compared_configurations']) == 2 and checkpoint['config'] == config
        raw = (work/'experiments.jsonl').read_bytes(); assert raw.startswith(previous)
        records = [json.loads(line) for line in raw.splitlines()]
        current_records = [r for r in records if r.get('task_origin') == job['origin']]
        assert sum(r['event'] == 'attempt' for r in current_records) == 8
        assert sum(r['event'] == 'result' for r in current_records) == 8
        for record in current_records:
            if record['event'] != 'result': continue
            assert record['request']['series_id'] == job['series_id'] and record['request']['unit'] == 'widgets'
            if record['kind'] == 'forecast':
                assert record['actual'] is None and record['metrics'] is None
            else:
                independently = math.sqrt(math.fsum((math.log1p(p)-math.log1p(a))**2
                                          for p, a in zip(record['point'], record['actual'], strict=True))/14)
                assert abs(independently-record['metrics']['rmsle']) < 1e-12
            checks += 3
        assert {name: sha(work/name) for name in files} == source_hashes
        shutil.copytree(work, out/'project')
        prior.append({k: job[k] for k in ('series_id', 'origin', 'actual', 'future_timestamps', 'outcome_recorded_at')})
        checks += 8
    for name, expected in capsule['sources'].items(): assert sha(package/name) == expected
    result = {'status': 'synthetic_checkpoint_recipe_sequence_passed', 'checks': checks,
              'synthetic_origins': 5, 'matured_origins_before_proposed_sequence': 4,
              'numerical_attempts': 40, 'estimator_fits': 20, 'scripted_tool_commands': total_commands,
              'proposed_sequence_numerical_attempts': 8, 'engy_calls': 0,
              'original_frozen_sources_unchanged': True, 'prior_event_prefixes_preserved': True,
              'final_gate_opened': False, 'objective_established': False,
              'scope': 'One scripted recipe plan through the frozen common lab and real published-1.2.0 ledger. Not an agent or accuracy comparison.'}
    dump(root/'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(probe(a.output, a.runtime), indent=2))
