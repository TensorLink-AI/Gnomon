"""Score a complete planned agent submission set; missing cases keep fallbacks."""
import json
from datetime import date, timedelta
from pathlib import Path

from .agent import load_case, resolve
from .data import dump, load_panel, origin, sha
from .models import forecast, metrics
from .run import fingerprint, summarize


def score_submissions(panel, baseline_run, submissions, output, arm):
    if arm not in ('hermes', 'gnomon', 'ledger'):
        raise ValueError('Unknown comparison arm')
    baseline_run, output = Path(baseline_run), Path(output)
    if output.exists():
        raise ValueError('Use a fresh scoring destination')
    _, frame = load_panel(panel)
    plan = json.loads((baseline_run/'plan.json').read_text())
    if plan['source_manifest_sha256'] != sha(Path(panel)/'manifest.json'):
        raise ValueError('Agent cases and scoring panel differ')
    supplied = json.loads(Path(submissions).read_text())
    if not isinstance(supplied, list) or len({r['case_id'] for r in supplied}) != len(supplied):
        raise ValueError('Provide one submission record per case, without duplicates')
    indexed = {r['case_id']: r for r in supplied}
    cases = sorted((baseline_run/'agent-cases').iterdir())
    expected_cases = {s+'-'+origin(i).isoformat() for s in plan['series'] for i in plan['origins']}
    if (len(cases) != plan['planned_cases'] or {p.name for p in cases} != expected_cases
            or set(indexed)-expected_cases):
        raise ValueError('Case set missing or submissions include unrelated cases')
    rows = []
    for case in cases:
        task, history = load_case(case); day = date.fromisoformat(task['origin'])
        original_history = frame[(frame.series_id == task['series_id']) & (frame.date.dt.date <= day)].sort_values('date')
        expected_id = fingerprint(task['series_id'], original_history.value,
            original_history.date.dt.strftime('%Y-%m-%d').tolist(), task['future_timestamps'])
        if task['case_id'] != case.name or task['request_fingerprint'] != expected_id:
            raise ValueError('Agent task or history differs from the original host-owned case')
        actual = frame[(frame.series_id == task['series_id']) & (frame.date.dt.date > day)
                       & (frame.date.dt.date <= day+timedelta(days=14))].sort_values('date').value.tolist()
        if len(actual) != task['horizon']:
            raise ValueError('Missing scorer-owned actuals')
        submitted = indexed.get(task['case_id'])
        resolution = resolve(case, submitted['executions'], submitted.get('execution_id')) if submitted else {'resolved':False,'status':'missing_submission'}
        expected_backend = 'direct' if arm == 'hermes' else 'gnomon'
        if resolution['resolved'] and resolution['execution']['backend'] != expected_backend:
            resolution = {'resolved': False, 'status': 'wrong_execution_backend'}
        pred = resolution.get('execution') if resolution['resolved'] else forecast('seasonal_naive_7', history.value, 14, day)
        rows.append({'case_id': task['case_id'], 'method': arm, 'resolved': resolution['resolved'],
            'resolution_status': resolution['status'], 'fallback_used': not resolution['resolved'] or pred['fallback_used'],
            'clipped_points': pred['clipped_points'], 'metrics': metrics(pred['point'], actual, history.value),
            'execution_id': pred.get('execution_id'), 'point': pred['point']})
    output.mkdir(parents=True)
    result = {'scope': 'scoring_imported_typed_executions_not_an_agent_runner', 'cases': len(rows),
        'resolved': sum(r['resolved'] for r in rows), 'summary': summarize(rows), 'rows': rows,
        'ledger_use_verified': False, 'agent_budget_compliance_verified': False,
        'objective_established': False, 'submissions_sha256': sha(submissions)}
    dump(output/'report.json', result)
    return result
