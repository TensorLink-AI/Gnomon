"""Read-only, hindsight-only headroom in the completed v1 development portfolios.

This is not a policy, treatment estimate, or upper bound over unexecuted models.
It never reads reserved outcomes or runs another numerical/provider call.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

REPO=Path(__file__).resolve().parents[2]
SOURCE=REPO/'results/hermes-ml-checkpoint-120-v1/evaluation'


def metric(point,actual):
    assert len(point)==len(actual)==14
    return math.sqrt(sum((math.log1p(p)-math.log1p(a))**2 for p,a in zip(point,actual,strict=True))/14)


def audit():
    assert (SOURCE.parent/'FINISHED.json').exists()
    jobs=json.loads((SOURCE/'host-jobs.json').read_text())
    index={(s,j['round']):j for s,rows in jobs.items() for j in rows}
    hashes={str((SOURCE/'host-jobs.json').relative_to(REPO)):
            hashlib.sha256((SOURCE/'host-jobs.json').read_bytes()).hexdigest()};rows=[]
    for grade_path in sorted(SOURCE.glob('*/*/round-*/grade.json')):
        grade=json.loads(grade_path.read_text());job=index[grade['series_id'],grade['round']]
        path=grade_path.parent/'project/experiments.jsonl'
        events=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        for p in [grade_path,*([path] if path.exists() else [])]:
            hashes[str(p.relative_to(REPO))]=hashlib.sha256(p.read_bytes()).hexdigest()
        options=[]
        for e in events:
            if e['event']!='result' or e['kind']!='forecast' or e['task_origin']!=job['origin']:continue
            req=e['request']
            assert req['series_id']==job['series_id'] and req['unit']==job['request']['unit']
            assert req['cutoff']==job['origin'] and req['future_timestamps']==job['future_timestamps']
            options.append({'execution_id':e['execution']['execution_id'],'config_id':e['config_id'],
                            'rmsle':metric(e['point'],job['actual'])})
        score=metric(grade['point'],job['actual'])
        assert abs(score-grade['rmsle'])<1e-12
        # Keep the predeclared fallback when no eligible execution exists; never drop a case.
        best=min([score,*[o['rmsle'] for o in options]])
        rows.append({'arm':grade['arm'],'series_id':grade['series_id'],'round':grade['round'],
                     'actual_submitted_rmsle':score,'restricted_hindsight_rmsle':best,
                     'executed_options':options,'fallback_used':grade['fallback_used']})
    summary={}
    for arm in ('plain','gnomon','ledger'):
        subset=[r for r in rows if r['arm']==arm];assert len(subset)==32
        actual=mean(r['actual_submitted_rmsle'] for r in subset)
        best=mean(r['restricted_hindsight_rmsle'] for r in subset)
        summary[arm]={'cases':32,'actual_mean_rmsle':actual,'restricted_hindsight_mean_rmsle':best,
                      'hindsight_reduction_fraction':1-best/actual,
                      'cases_with_better_executed_option':sum(r['restricted_hindsight_rmsle']<r['actual_submitted_rmsle']-1e-12 for r in subset)}
    assert all(hashlib.sha256((REPO/n).read_bytes()).hexdigest()==v for n,v in hashes.items())
    return {'scope':'Completed v1 development only, all 96 cases retained','summary':summary,
            'source_hashes':hashes,'rows':rows,'provider_calls':0,'ledger_writes':0,
            'limits':['Uses future targets in hindsight; never used as agent input or scored as a causal policy.',
                      'Restricted to already-executed production forecasts plus the originally graded forecast/fallback.',
                      'Does not bound additional model/configuration exploration or establish an attainable 20% gain.',
                      'No original score, execution, ledger or outcome was modified.']}


if __name__=='__main__':
    result=audit()
    out=REPO/'benchmarks/ledger_optimization/evidence/ml-portfolio-headroom-018.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))
