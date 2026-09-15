"""Independently recalculate every candidate-107 score without executing models.

This verifies arithmetic and the full planned denominator. It does not replace
the host, recipe, temporal, archive or cost audits or admit a continuation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from .analyze_guarded_093 import paired_series_contrast

ARMS = ('plain', 'gnomon', 'ledger')
FIELDS = ('series_id', 'round', 'origin', 'future_timestamps',
          'outcome_recorded_at', 'actual', 'request')


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def calculate(jobs, grades):
    expected = {}
    for series, tasks in jobs.items():
        for task in tasks:
            key = (series, task['round'])
            if task['series_id'] != series or key in expected:
                raise ValueError('Duplicate or inconsistent task identity')
            if len(task['actual']) != 14 or not all(map(number, task['actual'])):
                raise ValueError('Expected fourteen finite nonnegative actuals')
            expected[key] = task
    if not expected:
        raise ValueError('No planned tasks')
    rows, seen, delta = [], set(), 0.0
    for grade in grades:
        key = (grade['arm'], grade['series_id'], grade['round'])
        if key in seen or key[0] not in ARMS or key[1:] not in expected:
            raise ValueError('Duplicate, extra or unrelated grade')
        seen.add(key)
        task = expected[key[1:]]
        point = grade['point']
        if grade['origin'] != task['origin'] or len(point) != 14 or not all(map(number, point)):
            raise ValueError('Forecast origin, horizon or point contract mismatch')
        score = math.sqrt(sum((math.log1p(p)-math.log1p(a))**2
                             for p, a in zip(point, task['actual'], strict=True))/14)
        if not number(grade['rmsle']) or abs(score-grade['rmsle']) > 1e-12:
            raise ValueError('Stored score differs from independent arithmetic')
        for field in ('valid', 'workflow_complete', 'fallback_used'):
            if type(grade[field]) is not bool:
                raise ValueError('Completion flags must be explicit booleans')
        delta = max(delta, abs(score-grade['rmsle']))
        rows.append({**{k: grade[k] for k in ('arm', 'series_id', 'round', 'origin',
                       'valid', 'workflow_complete', 'fallback_used')}, 'rmsle': score})
    if seen != {(arm, *key) for arm in ARMS for key in expected}:
        raise ValueError('Incomplete planned denominator; do not drop missing sessions')
    keyed = {(r['arm'], r['series_id'], r['round']): r for r in rows}
    summary = {}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        summary[arm] = {'sessions': len(selected), 'mean_rmsle': mean(r['rmsle'] for r in selected),
                        'valid': sum(r['valid'] for r in selected),
                        'workflow_complete': sum(r['workflow_complete'] for r in selected),
                        'fallbacks': sum(r['fallback_used'] for r in selected)}
        for name, low, high in [('cold', 0, 3), ('mature', 10, 25), ('late', 22, 25)]:
            window = [r for r in selected if low <= r['round'] <= high]
            summary[arm][name] = {'cases': len(window),
                'mean_rmsle': mean(r['rmsle'] for r in window) if window else None}
    return {'sessions': len(rows), 'matched_cases': len(expected), 'arms': summary,
            'max_score_recalculation_delta': delta,
            'contrasts': {'ledger_vs_'+arm: paired_series_contrast([
                (keyed['ledger', *key], keyed[arm, *key]) for key in sorted(expected)])
                for arm in ('gnomon', 'plain')}, 'rows': rows}


def analyze(root, task_source, output):
    root, task_source, output = map(lambda p: Path(p).absolute(), (root, task_source, output))
    if output.exists() or output.is_symlink() or root == output or root in output.parents or output in root.parents:
        raise ValueError('Fresh analysis outside original evidence required')
    inputs = {}
    def read(path):
        raw = path.read_bytes(); inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    plan = read(root/'prospective-plan.json'); manifest = read(root/'manifest.json')
    if plan.get('final_gate_opened') is not False or type(plan.get('synthetic')) is not bool:
        raise ValueError('Only explicitly labeled development evidence is supported')
    source = read(task_source)
    if inputs[str(task_source)] != plan['task_source_sha256']:
        raise ValueError('Task source differs from prospective plan')
    pilot = manifest['pilot']
    if type(pilot) is not bool:
        raise ValueError('Explicit stage required')
    total, new, retained = (72, 72, 0) if pilot else (312, 240, 72)
    if (manifest['planned'] != total or manifest['requested_seed'] != 7
            or manifest['model'] != 'deepseek-v4.1-flash'
            or manifest['build'] != plan['build'] or manifest['build']['package_version'] != '1.2.0'):
        raise ValueError('Stage, runtime or agent identity mismatch')
    if read(root/'complete.json') != {'completed': total, 'planned': total, 'new_sessions': new,
                                     'retained_pilot_sessions': retained, 'source_unchanged': True}:
        raise ValueError('Complete stage accounting required')
    if read(root/'runner-exit.json') != {'exit_status': 0, 'continuation_launched': False}:
        raise ValueError('Stage did not finish successfully; retain its rejection evidence separately')
    if len(source) != 4 or any(len(tasks) != 26 for tasks in source.values()):
        raise ValueError('Expected the frozen four-series development cohort')
    expected = {s: [{k: j[k] for k in FIELDS} for j in (tasks[:6] if pilot else tasks)]
                for s, tasks in source.items()}
    jobs = read(root/'host-jobs.json')
    if jobs != expected:
        raise ValueError('Saved task values differ from authenticated original jobs')
    paths = sorted(root.glob('*/*/round-*/grade.json'))
    grades = []
    for path in paths:
        g = read(path)
        if path.relative_to(root).parts[:3] != (g['arm'], g['series_id'], 'round-'+str(g['round'])):
            raise ValueError('Grade location and identity disagree')
        grades.append(g)
    result = calculate(jobs, grades)
    for path, digest in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Input changed during arithmetic verification')
    if paths != sorted(root.glob('*/*/round-*/grade.json')):
        raise ValueError('Grade set changed during verification')
    result.update({'status': 'arithmetic_verified', 'synthetic': plan['synthetic'],
        'stage': 'pilot' if pilot else 'complete', 'inputs': inputs,
        'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'bootstrap_source_sha256': hashlib.sha256(Path(__file__).with_name('analyze_guarded_093.py').read_bytes()).hexdigest(),
        'provider_calls': 0, 'engy_calls': 0, 'original_inputs_unchanged': True,
        'full_integrity_audit_claimed': False, 'cost_audit_claimed': False,
        'objective_established': False, 'final_gate_opened': False,
        'limitations': ['All planned cases including fallbacks remain in the primary mean.',
                       'Four reused development series and one requested seed; exploratory uncertainty only.',
                       'Independent host, archive, recipe and cost audits remain required.']})
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'task-source', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    result = analyze(**vars(parser.parse_args()))
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'inputs')}, indent=2))
