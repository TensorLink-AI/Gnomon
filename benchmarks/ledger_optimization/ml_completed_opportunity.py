"""Read-only opportunity audit on completed v4 development executions.

Hindsight scores diagnose portfolio headroom, not an attainable policy. Historical
coverage uses only outcomes already visible at each origin. No held-out data or
provider is accessed, and no recorded selection is replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from benchmarks.hermes_ml_checkpoint_v4.analyze import rmsle
from benchmarks.hermes_ml_checkpoint_v4.maturation import instant


def visible_history(events, now, configurations):
    rows = {}
    for event in events:
        if event['event'] != 'matured' or event['config_id'] not in configurations:
            continue
        if instant(event['forecast_origin']) >= instant(now):
            continue
        if instant(event['outcome_recorded_at']) > instant(now):
            continue
        score = rmsle(event['point'], event['actual'])
        assert abs(score-event['metrics']['rmsle']) < 1e-12
        key = event['config_id'], event['forecast_origin']
        if key in rows:
            assert abs(rows[key]-score) < 1e-12
        rows[key] = score
    return rows


def audit(root, output):
    root, output = Path(root), Path(output)
    source = root/'report.json'
    report = json.loads(source.read_text())
    assert report['complete'] and not report['audit_failures']
    jobs = json.loads((root/'host-jobs.json').read_text())
    hashes = {}
    def read(path):
        data = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        return data
    read(source); read(root/'host-jobs.json')
    rows = []
    for series, tasks in jobs.items():
        last = max(task['round'] for task in tasks)
        path = root/'ledger'/series/f'round-{last:02d}'/'project/experiments.jsonl'
        if not path.exists():
            path = root/'ledger'/series/f'round-{last}'/'project/experiments.jsonl'
        events = [json.loads(line) for line in read(path).splitlines()]
        for task in tasks:
            grade = next(r for r in report['rows'] if r['arm']=='ledger'
                         and r['series_id']==series and r['round']==task['round'])
            options = {}
            for event in events:
                if event['event']!='result' or event['kind']!='forecast' or event['task_origin']!=task['origin']:
                    continue
                request = event['request']
                assert request['series_id']==series and request['unit']==task['request']['unit']
                assert request['cutoff']==task['origin'] and request['future_timestamps']==task['future_timestamps']
                score = rmsle(event['point'], task['actual'])
                cid = event['config_id']
                if cid in options:
                    assert abs(options[cid]-score)<1e-12
                options[cid] = score
            historical = visible_history(events, task['origin'], options)
            origins = {cid:{origin for config,origin in historical if config==cid} for cid in options}
            shared = set.intersection(*origins.values()) if len(origins)>=2 else set()
            best = min([grade['rmsle'], *options.values()])
            rows.append({'series_id':series,'round':task['round'],
                         'submitted_rmsle':grade['rmsle'],'hindsight_best_executed_rmsle':best,
                         'executed_configurations':len(options),
                         'configurations_with_visible_history':sum(bool(v) for v in origins.values()),
                         'all_options_matched_past_origins':len(shared)})
    assert len(rows)==104
    actual=mean(r['submitted_rmsle'] for r in rows)
    oracle=mean(r['hindsight_best_executed_rmsle'] for r in rows)
    summary={'tasks':len(rows),'submitted_rmsle':actual,'hindsight_best_executed_rmsle':oracle,
             'restricted_hindsight_reduction':1-oracle/actual,
             'tasks_with_better_executed_forecast':sum(r['hindsight_best_executed_rmsle']<r['submitted_rmsle']-1e-12 for r in rows),
             'tasks_with_at_least_two_executed_configurations':sum(r['executed_configurations']>=2 for r in rows),
             'tasks_with_at_least_three_all_options_matched_past_origins':sum(r['all_options_matched_past_origins']>=3 for r in rows)}
    for name,digest in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    result={'summary':summary,'rows':rows,'source_hashes':hashes,'provider_calls':0,
            'limitations':['Reused development data; hindsight is not a deployable selection rule.',
                          'Coverage is restricted to already executed configurations; no claim about unexecuted alternatives.',
                          'No original forecast, selection, grade, outcome or ledger is modified.']}
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root');parser.add_argument('output')
    args=parser.parse_args()
    print(json.dumps(audit(args.root,args.output)['summary'],indent=2))
