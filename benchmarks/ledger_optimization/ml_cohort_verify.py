"""Independent saved-pair/visibility checks for the fixed development cohort."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def verify(root, source):
    root,source=Path(root),Path(source)
    manifest=json.loads((root/'manifest.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_sha256']
    jobs=json.loads(source.read_text());report=json.loads((root/'report.json').read_text())
    checks=0;all_scores=[];hashes={};models=['seasonal','ridge','random_forest']
    def metric(point,actual):
        nonlocal checks
        assert len(point)==len(actual)==14
        assert all(math.isfinite(v) and v>=0 for v in point+actual)
        checks+=2
        return math.sqrt(mean((math.log1p(p)-math.log1p(a))**2 for p,a in zip(point,actual)))
    for series,tasks in jobs.items():
        previous=[]
        for task in sorted(tasks,key=lambda t:t['round']):
            path=root/f'{series}-{task["round"]:02d}.json';data=path.read_bytes()
            hashes[path.name]=hashlib.sha256(data).hexdigest();row=json.loads(data)
            assert row['request']==task['request'] and row['actual']==task['actual'];checks+=2
            cv={};scores={}
            for model in models:
                scores[model]=metric(row['point'][model],task['actual'])
                assert abs(scores[model]-row['scores'][model])<1e-12;checks+=1
                folds=row['folds'][model];assert [f['end'] for f in folds]==[688,702,716]
                values=[]
                for f in folds:
                    end=f['end'];r=f['request'];original=task['request']
                    assert r['history']==original['history'][:end]
                    assert r['timestamps']==original['timestamps'][:end]
                    assert r['future_timestamps']==original['timestamps'][end:end+14]
                    assert r['past_covariates']==original['past_covariates'][:end]
                    assert r['future_covariates']==original['past_covariates'][end:end+14]
                    assert f['actual']==original['history'][end:end+14]
                    assert r['cutoff']==r['known_time_cutoff']==original['timestamps'][end-1]
                    value=metric(f['point'],f['actual'])
                    assert abs(value-f['metrics']['rmsle'])<1e-12
                    values.append(value);checks+=8
                cv[model]=mean(values)
                assert abs(cv[model]-row['cv'][model])<1e-12;checks+=1
            naive=[task['request']['history'][-7+i%7] for i in range(14)]
            assert row['point']['seasonal']==naive;checks+=1
            now=datetime.fromisoformat(task['origin'])
            visible=[p for p in previous if datetime.fromisoformat(p['origin'])<now
                     and datetime.fromisoformat(p['outcome_recorded_at'])<=now
                     and datetime.fromisoformat(p['future_timestamps'][-1])<=now][-4:]
            cv_model=min(models,key=lambda m:(cv[m],models.index(m)))
            past_model=cv_model if len(visible)<3 else min(models,key=lambda m:
                (mean(p['scores'][m] for p in visible),cv[m],models.index(m)))
            assert row['selection']['cv']==cv_model and row['selection']['past']==past_model
            assert row['selection']['history_origins']==[p['origin'] for p in visible]
            checks+=3
            all_scores.append({'cv_rmsle':scores[cv_model],'past_rmsle':scores[past_model],
                               'hindsight_rmsle':min(scores.values())})
            previous.append({**task,'scores':scores})
    assert len(all_scores)==104 and report['numerical_fits']==1248 and report['api_calls']==0
    for key,value in report['means'].items():
        assert abs(mean(r[key] for r in all_scores)-value)<1e-12;checks+=1
    for name,digest in hashes.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    result={'checks':checks,'failures':[],'saved_pair_hashes':hashes,'provider_calls':0,
            'report_sha256':hashlib.sha256((root/'report.json').read_bytes()).hexdigest()}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root');parser.add_argument('source');args=parser.parse_args()
    result=verify(args.root,args.source)
    print(json.dumps({k:v for k,v in result.items() if k!='saved_pair_hashes'},indent=2))
