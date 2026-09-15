"""Offline positive and tampering checks against retained synthetic Hermes calls.

Run with SOURCE_WORKER_ROOT NEW_OUTPUT_ROOT. Copies evidence; never changes source
records, invokes a model, or contacts an API. This is an integration probe, not a
claim of forecasting efficacy or a replacement for the underlying run audit.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from .contrast_audit_100 import audit_annotations


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def save(path, rows):
    path.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows))


def change_annotations(folder, change):
    """Change both copies so rejection cannot rely on a redundant-copy mismatch."""
    p = folder/'project/comparison-annotations.jsonl'; records = lines(p)
    for r in records: change(r['annotation'])
    save(p, records)
    p = folder/'boundary-events.jsonl'; events = lines(p)
    for e in events:
        result = e.get('result', {}).get('result', {})
        if type(result) is not dict or 'stdout' not in result: continue
        reply = json.loads(result['stdout'])
        if 'evidence_summary' in reply:
            change(reply['evidence_summary']); result['stdout'] = json.dumps(reply)+'\n'
    save(p, events)


def probe(source, output):
    source = Path(source).resolve(); output = Path(output).resolve()
    if output == source or source in output.parents: raise ValueError('Output must be outside source')
    output.mkdir(parents=True, exist_ok=False)
    folders = sorted(source.glob('*/synthetic-collection-worker/round-*'))
    if len(folders) != 6: raise ValueError('Expected all six synthetic sessions')
    before = {str(p.relative_to(source)): digest(p) for f in folders for p in f.rglob('*') if p.is_file()}
    positive = {str(f.relative_to(source)): audit_annotations(f) for f in folders}
    seed = source/'ledger/synthetic-collection-worker/round-1'
    cases = []

    def trial(name, mutate, expected):
        target = output/name; target.mkdir()
        shutil.copytree(seed/'project', target/'project')
        shutil.copy2(seed/'boundary-events.jsonl', target/'boundary-events.jsonl')
        mutate(target)
        try: audit_annotations(target)
        except ValueError as exc:
            reason = str(exc)
            if expected not in reason: raise AssertionError((name, expected, reason)) from exc
        else: raise AssertionError(name+' was accepted')
        cases.append({'case': name, 'rejected': True, 'reason': reason})

    def score(a):
        if a['current_cv']['configurations']: a['current_cv']['configurations'][0]['mean_rmsle'] += .25
    trial('fabricated-mean', lambda f: change_annotations(f, score), 'independent current table')

    def rank(a):
        if a['current_cv']['configurations']: a['current_cv']['configurations'][0]['rank'] += 1
    trial('fabricated-rank', lambda f: change_annotations(f, rank), 'independent current table')

    def later_prefix(f):
        p = f/'project/comparison-annotations.jsonl'; records = lines(p)
        raw = (f/'project/experiments.jsonl').read_bytes()
        task = json.loads((f/'project/task.json').read_bytes())
        current = next(r for r in records if r['annotation']['query']['origin'] == task['origin'])
        current['execution_log_prefix'] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        save(p, records)
    trial('future-log-prefix', later_prefix, 'no later fit attempts included')

    def missing(f):
        p = f/'project/comparison-annotations.jsonl'; records = lines(p)
        task = json.loads((f/'project/task.json').read_bytes())
        idx = next(i for i, r in enumerate(records) if r['annotation']['query']['origin'] == task['origin'])
        save(p, records[:idx]+records[idx+1:])
    trial('missing-annotation', missing, 'every successful annotation retained')

    def corrupt(f):
        for p in (f/'project/comparison-evidence').glob('*.json'): p.write_bytes(p.read_bytes()+b' ')
    trial('corrupt-artifact', corrupt, 'stored artifact integrity')

    def artifact_facts(f):
        refs = {}
        for p in (f/'project/comparison-evidence').glob('*.json'):
            obj = json.loads(p.read_bytes()); obj['current']['runs'][0]['folds'][0]['rmsle'] += .25
            raw = json.dumps(obj, sort_keys=True).encode(); sha = hashlib.sha256(raw).hexdigest()
            new = p.with_name(sha+'.json'); new.write_bytes(raw)
            refs[str(p.relative_to(f/'project'))] = {'path': str(new.relative_to(f/'project')), 'sha256': sha, 'bytes': len(raw)}
        def replace(a):
            if 'comparison' in a: a['comparison']['evidence'] = deepcopy(refs[a['comparison']['evidence']['path']])
        change_annotations(f, replace)
    trial('self-consistent-false-artifact', artifact_facts, 'artifact contains only available current folds')

    def actuals(f):
        p = f/'project/experiments.jsonl'; old = p.read_bytes(); rows = lines(p)
        for r in rows:
            if r['event'] == 'result' and r['kind'] == 'backtest': r['actual'][0] += 1
        save(p, rows); new_lines = p.read_bytes().splitlines(keepends=True)
        q = f/'project/comparison-annotations.jsonl'; records = lines(q)
        for r in records:
            n = len(old[:r['execution_log_prefix']['bytes']].splitlines())
            raw = b''.join(new_lines[:n])
            r['execution_log_prefix'] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        save(q, records)
    trial('wrong-actuals-rehashed-prefix', actuals, 'actuals equal visible targets')

    trial('false-extra-fit-accounting', lambda f: change_annotations(f, lambda a: a['annotation_diagnostics'].update(provider_calls=1)), 'annotation declares no additional execution')
    after = {str(p.relative_to(source)): digest(p) for f in folders for p in f.rglob('*') if p.is_file()}
    if before != after: raise AssertionError('Original source evidence changed')
    report = {'passed': True, 'positive_sessions': positive, 'negative_cases': cases,
              'source_sha256': before, 'source_unchanged': True, 'provider_calls': 0, 'engy_calls': 0,
              'scope': 'Six real synthetic Hermes session artifacts; deliberate mutations on separate copies. No forecasting efficacy claim.'}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': True, 'sessions': len(positive), 'annotations': sum(r['annotations'] for r in positive.values()),
                      'checks': sum(r['checks'] for r in positive.values()), 'tampering_rejected': len(cases)}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('source'); parser.add_argument('output')
    args = parser.parse_args(); probe(args.source, args.output)
