"""Opt-in live Ditto handoff. Creates synthetic evidence in a dedicated workspace.

Run from a source checkout with the hosted package installed. No credentials
are printed or saved in the report. Existing evidence is resumed on rerun.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import subprocess
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from hosted_probe_support import serving, call
from gnomon_hosted.storage import Store, now


def checked(result):
    if result.get('status') != 'ok':
        raise RuntimeError(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--graph', required=True)
    parser.add_argument('--hermes-root', type=Path)
    parser.add_argument('--hermes-python', default=sys.executable)
    args = parser.parse_args()
    os.umask(0o077)
    # Deliberately parse one literal variable; never source or execute .env text.
    key = None
    for line in args.env_file.read_text().splitlines():
        line = line.strip().removeprefix('export ')
        if line.startswith('DITTO_API_KEY='):
            key = line.split('=', 1)[1].strip().strip('\"\'')
    if not key:
        raise RuntimeError('DITTO_API_KEY is missing')
    env = {'DITTO_API_KEY': key}
    path = args.root / 'smoke-state.json'
    if path.exists():
        state = json.loads(path.read_text())
        store = Store(args.root)
    else:
        store = Store.initialize(args.root)
        project = store.create_project('live-synthetic-handoff')['project_id']
        state = {'project': project}
        scopes = {
            'hermes-a': ['forecast.create', 'decision.create', 'evidence.read'],
            'outcomes': ['actual.create'],
            'reviewer': ['decision.create', 'memory.export', 'evidence.read'],
            'hermes-b': ['evidence.read'],
        }
        state['tokens'] = {name: store.issue_token(project, name, permissions)['token'] for name, permissions in scopes.items()}
        path.write_text(json.dumps(state))
    store.configure_ditto(state['project'], 'https://api.heyditto.ai/mcp', 'DITTO_API_KEY', args.graph)
    def save():
        path.write_text(json.dumps(state))
    def invoke(url, principal, tool, arguments):
        if args.hermes_root and principal in ('hermes-a', 'hermes-b'):
            process = subprocess.run([args.hermes_python, str(Path(__file__).with_name('hermes_client.py')),
                '--hermes-root', str(args.hermes_root), '--url', url + '/mcp'],
                input=json.dumps({'tool': tool, 'arguments': arguments}), text=True, capture_output=True, timeout=90,
                env={**os.environ, 'GNOMON_SERVICE_TOKEN': state['tokens'][principal]})
            if process.returncode:
                raise RuntimeError('Hermes probe failed: ' + process.stderr[-3000:])
            return checked(json.loads(process.stdout))
        return checked(call(url, state['tokens'][principal], tool, arguments))
    with serving(args.root, env=env) as url:
        if 'request' not in state:
            at = datetime.now(timezone.utc)
            state['request'] = {'history': [10, 11, 12], 'horizon': 1, 'series_id': 'gnomon-live-smoke-widgets', 'unit': 'widgets',
                'timestamps': [(at - timedelta(days=i)).isoformat() for i in (3, 2, 1)],
                'future_timestamps': [(at + timedelta(seconds=5)).isoformat()]}
            save()
        run = invoke(url, 'hermes-a', 'gnomon_forecast', {'provider': 'last_value', 'request': state['request'], 'idempotency_key': 'forecast'})
        decision = invoke(url, 'hermes-a', 'gnomon_ledger', {'operation': 'record_decision_summary',
            'execution_id': run['result']['execution_id'], 'rationale': 'Synthetic acceptance fixture: last value persists.',
            'assumptions': ['Synthetic data only'], 'invalidation_conditions': ['Observed value differs'], 'context': [],
            'idempotency_key': 'decision'})
        state['decision'] = decision['result']['result']['decision_id']
        state['execution'] = run['result']['execution_id']
        save()
    # All client sessions and the server stop. An independent outcome principal
    # later appends the observation, after its real valid time.
    target = datetime.fromisoformat(state['request']['future_timestamps'][0])
    while datetime.now(timezone.utc) <= target:
        time.sleep(.05)
    with serving(args.root, env=env) as url:
        if 'actual' not in state:
            state['actual'] = {'operation': 'append_actual', 'series_id': state['request']['series_id'], 'unit': 'widgets',
                'valid_time': target.isoformat(), 'value': 13, 'source_available_at': now(), 'source_ref': 'synthetic-live-smoke', 'idempotency_key': 'actual'}
            save()
        invoke(url, 'outcomes', 'gnomon_ledger', state['actual'])
        if 'cutoffs' not in state:
            state['cutoffs'] = {'source_as_of': now(), 'recorded_as_of': now()}
            save()
        review = invoke(url, 'reviewer', 'gnomon_hosted', {'action': 'review.save', 'decision_id': state['decision'], **state['cutoffs'], 'idempotency_key': 'review'})
        assert review['result']['review']['metrics']['mae'] == 1
        lesson = invoke(url, 'reviewer', 'gnomon_ledger', {'operation': 'record_lesson', 'decision_id': state['decision'], **state['cutoffs'],
            'lesson': 'Synthetic Gnomon acceptance test: forecast 12 widgets, actual 13 widgets, MAE 1. Cause is unverified.', 'idempotency_key': 'lesson'})
        state['lesson'] = lesson['result']['result']['lesson_id']
        if 'export_cutoff' not in state:
            state['export_cutoff'] = now()
            save()
        export = invoke(url, 'reviewer', 'gnomon_hosted', {'action': 'export.enqueue', 'lesson_id': state['lesson'],
            'recorded_as_of': state['export_cutoff'], 'idempotency_key': 'export'})
        state['export'] = export['result']['export_id']
        save()
        delivered = invoke(url, 'reviewer', 'gnomon_hosted', {'action': 'export.deliver', 'export_id': state['export']})
        print(json.dumps({'delivery': delivered}), flush=True)
        if delivered['state'] != 'acknowledged':
            raise RuntimeError('Delivery not acknowledged; inspect export before retrying.')
        state['memory'] = delivered['memory_id']
        save()
    # A fresh server and fresh Ditto ClientSession fetch full text and verify
    # its digest, reference and score, under the second agent's read-only key.
    with serving(args.root, env=env) as url:
        for attempt in range(6):
            recalled = invoke(url, 'hermes-b', 'gnomon_hosted', {'action': 'memory.recall',
                'query': 'Synthetic Gnomon acceptance test widgets forecast 12 actual 13', 'source_as_of': now(), 'recorded_as_of': now()})
            if recalled['returned']:
                break
            time.sleep(2)  # bounded read-only retry for asynchronous Ditto indexing
        assert recalled['returned'] == 1, recalled
        assert recalled['lessons'][0]['current_review']['metrics']['mae'] == 1
        report = {'status': 'ok', 'graph': args.graph, 'service_id': store.service_id, 'project_id': state['project'],
            'execution_id': state['execution'], 'lesson_id': state['lesson'], 'export_id': state['export'], 'memory_id': state['memory'],
            'verified_mae': 1, 'server_restarts': 2, 'narrative_verified': False,
            'client': 'Hermes MCP discovery/registry/transport; no LLM calls' if args.hermes_root else 'MCP Python SDK'}
        (args.root / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
