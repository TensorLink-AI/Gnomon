"""Local synthetic integration: real models, guarded lab, published 1.2.0 storage.

No Hermes agent, Engy call, production dataset or pod runtime is involved.
Run with the isolated Gnomon interpreter and a separate matching plain venv.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def probe(capsule, plain_python, output):
    capsule, output = Path(capsule).resolve(), Path(output).resolve()
    plain_python = Path(plain_python).absolute()  # Preserve venv symlink identity.
    if any(k.startswith('benchmarks.hermes_ml_checkpoint_v6') for k in sys.modules):
        raise ValueError('Fresh process required')
    manifest = json.loads((capsule / 'capsule.json').read_text())
    package = capsule / 'benchmarks/hermes_ml_checkpoint_v6'
    for name, digest in manifest['sources'].items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == digest
    sys.path.insert(0, str(capsule))
    from benchmarks.hermes_ml_checkpoint_v6 import run
    from benchmarks.hermes_ml_checkpoint_v6.execution_boundary_093 import LabBoundary
    assert run.HERE == package
    from gnomon.build_info import build_info
    info = build_info()
    assert info['package_version'] == '1.2.0' and info['source_sha256'] == run.BUILD_SHA
    output.mkdir(parents=True, exist_ok=False)
    checks, calls, results = [], [], {}

    def check(name, passed):
        checks.append({'assertion': name, 'passed': bool(passed)})
        if not passed:
            raise AssertionError(name)

    metadata_code = ('import json,sys,importlib.metadata as m,importlib.util as u;'
                     'print(json.dumps({"python":sys.version,"gnomon":u.find_spec("gnomon") is not None,'
                     '"packages":{d.metadata["Name"].lower():d.version for d in m.distributions()}}))')
    inventory = {a: json.loads(subprocess.check_output([str(py), '-I', '-c', metadata_code], text=True))
                 for a, py in [('plain', plain_python), ('gnomon', Path(sys.executable))]}
    check('Plain runtime has no Gnomon', not inventory['plain']['gnomon'])
    check('Gnomon runtime is pinned 1.2.0', inventory['gnomon']['packages']['gnomon-forecast'] == '1.2.0')
    stripped = {a: {k: v for k, v in d['packages'].items() if k != 'gnomon-forecast'} for a, d in inventory.items()}
    check('All other installed package versions match', stripped['plain'] == stripped['gnomon'])
    times = [(datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)).isoformat() for i in range(760)]
    values = [float(10 + i % 7 + 3 * (i % 11 == 0)) for i in range(760)]
    jobs = []
    for n in range(2):
        begin, end = n * 14, n * 14 + 730
        future = times[end:end + 14]
        jobs.append({'series_id': 'synthetic-collection', 'round': n, 'origin': times[end-1],
                     'future_timestamps': future, 'outcome_recorded_at': future[-1], 'actual': values[end:end+14],
                     'request': {'history': values[begin:end], 'timestamps': times[begin:end],
                                 'future_timestamps': future, 'unit': 'widgets',
                                 'past_covariate_names': ['promo'], 'future_covariate_names': ['promo'],
                                 'past_covariates': [[float(i % 11 == 0)] for i in range(begin, end)],
                                 'future_covariates': [[float(i % 11 == 0)] for i in range(end, end+14)]}})
    (output / 'host-jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    config = {'model': 'ridge', 'window': 90, 'lags': 7, 'alpha': 10}
    try:
        for arm in ('plain', 'gnomon', 'ledger'):
            work = output / arm / 'project'
            work.mkdir(parents=True)
            python = plain_python if arm == 'plain' else Path(sys.executable)
            prior, original_forecasts = [], {}
            for job in jobs:
                n = job['round']
                destination = output / arm / f'round-{n}'
                destination.mkdir()
                prefix = (work / 'experiments.jsonl').read_bytes() if (work / 'experiments.jsonl').exists() else b''
                run.prepare(work, job, prior, arm)
                (work / 'agent-budget.json').write_text(json.dumps({'deadline_epoch': time.time()+480,
                                                                  'remaining_requests': 16, 'forwarded_requests': 0}))
                protected = {p.name: run.sha(p) for p in work.iterdir()
                             if p.is_file() and (p.name in run.PROJECT_FILES or p.name in
                                                ('task.json', 'history.csv', 'future.csv', 'backend.json', 'previous_runs.json'))}
                boundary = LabBoundary(work, python, protected)

                def call(operation, **arguments):
                    args = {'operation': operation, **arguments}
                    envelope = boundary.dispatch('lab', args)
                    calls.append({'arm': arm, 'round': n, 'arguments': args, 'response': envelope})
                    check(f'{arm}/{n}/{operation}: guarded call succeeded', envelope['status'] == 'ok')
                    payload = json.loads(envelope['result']['stdout'])
                    check(f'{arm}/{n}/{operation}: lab succeeded', envelope['result']['exit_code'] == 0 and payload['status'] == 'ok')
                    return payload

                sync = call('sync')
                check(f'{arm}/{n}: exact mature execution count', len(sync['result']['matured_execution_ids']) == 2 * n)
                call('start')
                before = (work / 'checkpoint.json').read_bytes()
                collected = call('backtest', config=config)
                check(f'{arm}/{n}: four collection fits', collected['numerical_calls_started'] == 4)
                check(f'{arm}/{n}: collection did not change selection', (work / 'checkpoint.json').read_bytes() == before)
                reused = call('backtest', config=config)
                check(f'{arm}/{n}: exact repeat fits nothing', reused['numerical_calls_started'] == 0 and reused['result']['reused'])
                check(f'{arm}/{n}: exact production identity reused', reused['result']['collection']['execution_id'] == collected['result']['collection']['execution_id'])
                call('commit', config={'model': 'seasonal', 'season': 7})
                selected = json.loads((work / 'checkpoint.json').read_text())
                check(f'{arm}/{n}: explicit baseline selection completed comparison', selected['selection_after_comparison'] and selected['config']['model'] == 'seasonal')
                status = call('status')
                check(f'{arm}/{n}: total eight actual attempts', status['budget']['numerical_attempts'] == 8)
                review = call('review')
                events = run.records(work)
                forecasts = [r for r in events if r['event'] == 'result' and r['kind'] == 'forecast' and r['task_origin'] == job['origin']]
                check(f'{arm}/{n}: two unscored current forecasts', len(forecasts) == 2 and all(r['actual'] is None and r['metrics'] is None for r in forecasts))
                check(f'{arm}/{n}: old event prefix unchanged', (work / 'experiments.jsonl').read_bytes().startswith(prefix))
                if n:
                    matured = [r for r in events if r['event'] == 'matured']
                    check(f'{arm}: selected and unselected forecasts mature', len(matured) == 2 and sum(r['selected_for_submission'] for r in matured) == 1)
                    check(f'{arm}: matured predictions unchanged', all(r['point'] == original_forecasts[r['execution_id']] for r in matured))
                    if arm == 'ledger':
                        check('Ledger reports paired historical evidence', any(w.get('matched_origins') == 1 for c in review['result']['cards'] for w in c['windows'].values()))
                        check('Both original executions have complete ledger scores', all(r['ledger_score']['complete'] for r in matured))
                else:
                    original_forecasts = {r['execution']['execution_id']: r['point'] for r in forecasts}
                results[arm, n] = {r['config_id']: r['point'] for r in forecasts}
                prior.append({k: job[k] for k in ('series_id', 'origin', 'future_timestamps', 'outcome_recorded_at', 'actual')}
                             | {'unit': 'widgets', 'execution_id': selected['execution_id']})
                shutil.copytree(work, destination / 'project')
                (destination / 'protected.json').write_text(json.dumps(protected, indent=2) + '\n')
        for n in range(2):
            check(f'Origin {n}: actual predictions identical across all arms', results['plain', n] == results['gnomon', n] == results['ledger', n])
        for name, digest in manifest['sources'].items():
            check(f'Source unchanged: {name}', run.sha(package / name) == digest)
        report = {'status': 'local_guarded_collection_integration_passed', 'checks': checks,
                  'runtime': inventory, 'gnomon_build': info, 'actual_numerical_attempts': 48,
                  'engy_calls': 0, 'agent_sessions': 0, 'synthetic_origins': 2,
                  'scope': 'Real models, guarded lab subprocesses, checkpoint files and published 1.2.0 storage. Synthetic inputs and scripted tool calls, no Hermes agent/API transport or efficacy comparison. Pod worker integration and final dispatch gates remain outstanding.',
                  'target_met': False, 'final_holdout_accessed': False}
        (output / 'passed.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ('checks', 'runtime')}, indent=2))
    finally:
        (output / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
        (output / 'calls.json').write_text(json.dumps(calls, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule', required=True, type=Path)
    parser.add_argument('--plain-python', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    probe(args.capsule, args.plain_python, args.output)
