"""Read-only audit of matured production evidence in the completed ML v1 trial.

Counts visibility/coverage, not forecasting performance or a counterfactual win.
Never executes models, updates a ledger, or accepts confirmation data.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / 'results/hermes-ml-checkpoint-120-v1/evaluation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instant(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError('Explicit timezone required')
    return result


def coverage(events, prior, task):
    """Find already executed forecasts with a complete matching matured outcome."""
    outcomes = {p['origin']:p for p in prior}
    assert len(outcomes) == len(prior), 'Ambiguous outcome origin'
    scored = {e['execution_id'] for e in events if e['event']=='matured'}
    now = instant(task['origin'])
    visible = defaultdict(list)
    for event in events:
        if event['event']!='result' or event['kind']!='forecast':
            continue
        outcome = outcomes.get(event['task_origin'])
        if outcome is None or instant(outcome['outcome_recorded_at']) > now:
            continue
        request = event['request']
        if (request['series_id'] != task['series_id'] or request['unit'] != task['unit']
                or request['future_timestamps'] != outcome['future_timestamps']
                or instant(request['cutoff']) != instant(outcome['origin'])):
            continue
        times = [instant(t) for t in request['future_timestamps']]
        if not times or max(times)>now or min(times)<=instant(request['cutoff']):
            continue
        assert len(times)==len(event['point'])==len(outcome['actual'])==request['horizon']
        visible[event['task_origin']].append({
            'execution_id':event['execution']['execution_id'],
            'config_id':event['config_id'],
            'matured_event_present':event['execution']['execution_id'] in scored})
    return [{'origin':origin,'executions':rows,
             'distinct_configs':len({r['config_id'] for r in rows}),
             'unscored_executions':sum(not r['matured_event_present'] for r in rows)}
            for origin,rows in sorted(visible.items())]


def audit():
    assert (SOURCE.parent/'FINISHED.json').exists()
    before={}; series=[]; reviews=[]
    for work in sorted((SOURCE/'ledger').glob('*/work')):
        paths={n:work/n for n in ['experiments.jsonl','review-log.jsonl','previous_runs.json','task.json']}
        for path in paths.values():before[str(path.relative_to(REPO))]=sha(path)
        events=[json.loads(s) for s in paths['experiments.jsonl'].read_text().splitlines()]
        prior=json.loads(paths['previous_runs.json'].read_text())
        task=json.loads(paths['task.json'].read_text())
        origins=coverage(events,prior,task)
        series.append({'series_id':task['series_id'],'as_of':task['origin'],'origins':origins})
        for s in paths['review-log.jsonl'].read_text().splitlines():
            r=json.loads(s)
            reviews.append({'series_id':task['series_id'],'origin':r['task_origin'],
                'record_count':r['record_count'],'shown_records':len(r['records']),
                'comparison_count':r['comparison_count'],
                'shown_comparisons':len(r['matched_comparisons']),
                'shown_production_comparisons':sum(p['kind']=='forecast' for p in r['matched_comparisons']),
                'ledger_queries':r['ledger_queries'],'payload_bytes':len(s.encode())})
    origins=[o for s in series for o in s['origins']]
    assert all(sha(REPO/n)==h for n,h in before.items())
    return {'scope':'Completed checkpoint-v1 development only; read-only coverage audit',
        'source_hashes':before,'source_unchanged':True,'series':series,'reviews':reviews,
        'summary':{'review_calls':len(reviews),
            'reviews_with_truncated_comparisons':sum(r['comparison_count']>r['shown_comparisons'] for r in reviews),
            'median_review_bytes':median(r['payload_bytes'] for r in reviews),
            'maximum_available_comparisons':max(r['comparison_count'] for r in reviews),
            'shown_production_comparisons':sum(r['shown_production_comparisons'] for r in reviews),
            'matured_origins':len(origins),
            'origins_with_multiple_executed_configurations':sum(o['distinct_configs']>1 for o in origins),
            'matching_executed_forecasts':sum(len(o['executions']) for o in origins),
            'matching_forecasts_without_matured_event':sum(o['unscored_executions'] for o in origins)},
        'limits':['No numerical rescoring, provider call, ledger write or original grade change.',
                  'Source/recording visibility follows the original synthetic replay assumptions.',
                  'Missing matched evidence does not establish that exposing it would improve selections.',
                  'This is a benchmark-integration omission, not a failure of TemporalLedger.evaluate.',
                  'The frozen checkpoint-v2 pilot retains v1 maturation to isolate orchestration changes.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    path=parser.parse_args().output
    value=audit()
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value['summary'],indent=2))
