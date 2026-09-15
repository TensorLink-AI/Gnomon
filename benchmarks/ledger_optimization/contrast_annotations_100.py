"""Host-owned annotations for a future capsule; caller holds the lab lock."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import tempfile

from .contrast_records_100 import current_runs
from .contrast_view_100 import contrast_view
from .paired_consistency_098 import paired_consistency


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _project_path(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Evidence requires a relative project path')
    cursor = Path('.')
    for part in path.parts:
        cursor = cursor/part
        if cursor.is_symlink(): raise ValueError('Symlink evidence paths are not supported')
    return path


def _atomic(path, raw, *, immutable=False):
    path = _project_path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != raw: raise ValueError('Existing content-addressed evidence changed')
        return
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def annotate(core, operation, result, *, requested_pair=None):
    """Annotate trusted completed lab state, without another provider/query call."""
    if operation not in ('start', 'backtest', 'review', 'status', 'commit'):
        return None
    task = core.read('task.json')
    arm = core.read('backend.json')['arm']
    query = {k: task[k] for k in ('series_id', 'unit', 'horizon', 'origin')}
    prefix = Path('experiments.jsonl').read_bytes() if Path('experiments.jsonl').exists() else b''
    records = [json.loads(line) for line in prefix.splitlines()]
    expected = [core.request_at(end) for end in (688, 702, 716)]
    current = current_runs(task, records, [r for r, _ in expected], core.configuration,
                           arm=arm, expected_actuals={r['cutoff']: a for r, a in expected})
    runs = {r['config_id']: r for r in current['runs']}
    scores = {cid: mean(f['rmsle'] for f in r['folds']) for cid, r in runs.items()}
    order = sorted(runs, key=lambda cid: (scores[cid], cid))
    annotation = {'schema_version': 'lab-contrast-100', 'query': query, 'metric': 'rmsle',
                  'current_cv': {'configurations': [
                      {'config_id': cid, 'config': runs[cid]['config'], 'mean_rmsle': scores[cid],
                       'rank': 1+sum(s < scores[cid] for s in scores.values()),
                       'fold_rmsle': [f['rmsle'] for f in runs[cid]['folds']]}
                      for cid in order],
                      'fold_origins': [f['origin'] for f in runs[order[0]]['folds']] if order else [],
                      'excluded': current['excluded']},
                  'annotation_diagnostics': {'provider_calls': 0, 'additional_ledger_queries': 0,
                                             'forecast_selection_made': False}}
    state_path = Path('comparison-state.json')
    cached = None
    if arm == 'ledger':
        if operation == 'review':
            if result.get('schema_version') == 'agent-review-088':
                if result['query'] != query: raise ValueError('Review belongs to another task')
                raw = _project_path(result['full_evidence']['path']).read_bytes()
                paired_consistency(result, raw)
                cached = {'query': query, 'review': deepcopy(result)}
            else:
                if result.get('status') != 'insufficient_evidence':
                    raise ValueError('Unexpected historical review schema')
                cached = {'query': query, 'review': None}
            _atomic(state_path, _json(cached))
        elif state_path.exists():
            state = json.loads(_project_path(str(state_path)).read_bytes())
            if state['query'] == query: cached = state
    pair = None
    if requested_pair is not None:
        if len(requested_pair) != 2 or len(set(requested_pair)) != 2:
            raise ValueError('Comparison requires two distinct configurations')
        annotation['focus_rule'] = 'explicit_review_pair'
        if all(cid in runs for cid in requested_pair): pair = list(requested_pair)
        else: annotation['comparison_status'] = 'requested_pair_lacks_complete_current_cv'
    elif len(order) >= 2:
        newest = result.get('config_id') if operation == 'backtest' else None
        if newest in runs:
            pair = [newest, next(cid for cid in order if cid != newest)]
            annotation['focus_rule'] = 'new_backtest_vs_lowest_current_cv_alternative'
        else:
            pair = order[:2]
            annotation['focus_rule'] = 'two_lowest_current_cv_means'
    else:
        annotation['comparison_status'] = 'fewer_than_two_complete_current_configurations'
    if pair is not None:
        inputs = {'query': query, 'runs': [runs[cid] for cid in pair]}
        historical = cached['review'] if cached else None
        if historical is not None:
            view, raw = contrast_view(inputs, review=historical,
                                      full_evidence_bytes=_project_path(historical['full_evidence']['path']).read_bytes())
        else:
            view, raw = contrast_view(inputs)
            if cached is not None:
                view['history_status'] = 'last_review_had_no_historical_catalog'
        _atomic(view['evidence']['path'], raw, immutable=True)
        if hashlib.sha256(_project_path(view['evidence']['path']).read_bytes()).hexdigest() != view['evidence']['sha256']:
            raise ValueError('Evidence persistence verification failed')
        annotation.update(comparison_status='available', comparison=view)
    evidence = {'operation': operation, 'requested_pair': requested_pair,
                'execution_log_prefix': {'bytes': len(prefix), 'sha256': hashlib.sha256(prefix).hexdigest()},
                'input_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                 for name in ('task.json', 'history.csv', 'future.csv', 'backend.json')},
                'annotation': annotation}
    with Path('comparison-annotations.jsonl').open('ab') as handle:
        handle.write(_json(evidence)+b'\n')
    if Path('experiments.jsonl').exists() and Path('experiments.jsonl').read_bytes() != prefix:
        raise ValueError('Execution log changed during annotation')
    return annotation
