"""Frozen default-cohort development screen; no API calls or held-out access."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean
import time
import sys

from benchmarks.hermes_ml_checkpoint_v4 import numerical

JOBS_SHA = '380261912295bbe21f3c998a8c6f7048cde1fb0869d37bb4db44ad3a72b28f25'
NUMERICAL_SHA = 'a71b75f06ad69fcca06275046c5e74565b15d7d6dd2538df0fe827c448fc0413'
MODELS = ('seasonal', 'ridge', 'random_forest')


def select(current_cv, history, origin):
    """Accept only complete commonly matched, closed and recorded horizons."""
    now = datetime.fromisoformat(origin)
    if now.tzinfo is None:
        raise ValueError('Explicit timezone required')
    eligible = []
    seen = set()
    for row in history:
        at, recorded, closed = (datetime.fromisoformat(row[k]) for k in
                                ('origin', 'outcome_recorded_at', 'last_target'))
        if any(t.tzinfo is None for t in (at, recorded, closed)):
            raise ValueError('Explicit evidence timezone required')
        if at >= now or recorded > now or closed > now:
            continue
        if recorded < closed or closed <= at:
            raise ValueError('Invalid observation availability')
        if at in seen:
            raise ValueError('Duplicate historical origin')
        seen.add(at)
        if set(row['scores']) != set(MODELS):
            raise ValueError('Incomplete matched cohort')
        eligible.append(row)
    eligible.sort(key=lambda row: datetime.fromisoformat(row['origin']))
    cv = min(MODELS, key=lambda model: (current_cv[model], MODELS.index(model)))
    recent = eligible[-4:]
    if len(recent)<3:
        return {'cv':cv, 'past':cv, 'matched_origins':len(recent), 'history_origins':[r['origin'] for r in recent],
                'basis':'insufficient_history_use_current_cv'}
    past = min(MODELS, key=lambda model: (mean(r['scores'][model] for r in recent),
                                         current_cv[model], MODELS.index(model)))
    return {'cv':cv, 'past':past, 'matched_origins':len(recent), 'history_origins':[r['origin'] for r in recent],
            'basis':'recent_common_past_origins'}


def fold_request(request, end):
    r = deepcopy(request)
    r['history'] = request['history'][:end]
    r['timestamps'] = request['timestamps'][:end]
    r['past_covariates'] = request['past_covariates'][:end]
    r['future_timestamps'] = request['timestamps'][end:end+14]
    r['future_covariates'] = request['past_covariates'][end:end+14]
    r['cutoff'] = r['known_time_cutoff'] = r['timestamps'][-1]
    return r, request['history'][end:end+14]


def run(source, output):
    source, output = Path(source), Path(output)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==JOBS_SHA, 'Only frozen development jobs accepted'
    assert hashlib.sha256(Path(numerical.__file__).read_bytes()).hexdigest()==NUMERICAL_SHA
    output.mkdir(parents=True, exist_ok=False)
    configs = {model:numerical.configuration({'model':model}) for model in MODELS}
    def save(name,value):
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save('manifest.json',{'source_sha256':JOBS_SHA,'numerical_sha256':NUMERICAL_SHA,
         'screen_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'protocol_sha256':hashlib.sha256(Path(__file__).with_name('ML_COHORT_032.md').read_bytes()).hexdigest(),
         'python':sys.version,'executable':sys.executable,
         'packages':{name:importlib.metadata.version(name) for name in ('numpy','scikit-learn')},
         'configs':configs,'planned_fits':1248,'planned_cases':104,'api_calls':0})
    jobs=json.loads(source.read_text());all_rows=[];fits=0;started=time.monotonic()
    try:
        for series,tasks in sorted(jobs.items()):
            history=[]
            for task in sorted(tasks,key=lambda task:task['round']):
                assert task['series_id']==series and task['request']['series_id']==series
                assert len(task['request']['history'])==730 and task['request']['horizon']==14
                cv={};predictions={};folds={}
                for model,config in configs.items():
                    folds[model]=[]
                    for end in (688,702,716):
                        request,actual=fold_request(task['request'],end)
                        fits+=1
                        point=numerical.predict(request,config)
                        fold={'end':end,'request':request,'point':point,'actual':actual,
                              'metrics':numerical.metrics(point,actual)}
                        folds[model].append(fold)
                    cv[model]=mean(f['metrics']['rmsle'] for f in folds[model])
                    fits+=1
                    predictions[model]=numerical.predict(task['request'],config)
                chosen=select(cv,history,task['origin'])
                # Current production targets are scored only after both choices.
                scores={model:numerical.metrics(point,task['actual'])['rmsle']
                        for model,point in predictions.items()}
                row={'series_id':series,'round':task['round'],'origin':task['origin'],
                     'request':task['request'],'actual':task['actual'],'cv':cv,'selection':chosen,
                     'point':predictions,'scores':scores,'folds':folds}
                save(f'{series}-{task["round"]:02d}.json',row)
                all_rows.append({'series_id':series,'round':task['round'],'selection':chosen,
                     'cv_rmsle':scores[chosen['cv']], 'past_rmsle':scores[chosen['past']],
                     'hindsight_rmsle':min(scores.values())})
                history.append({'origin':task['origin'],'last_target':task['future_timestamps'][-1],
                                'outcome_recorded_at':task['outcome_recorded_at'],'scores':scores})
                save('status.json',{'completed_cases':len(all_rows),'numerical_fits':fits,
                                   'seconds':time.monotonic()-started})
        assert len(all_rows)==104 and fits==1248
        means={key:mean(r[key] for r in all_rows) for key in ('cv_rmsle','past_rmsle','hindsight_rmsle')}
        result={'means':means,'past_relative_reduction':1-means['past_rmsle']/means['cv_rmsle'],
                'hindsight_relative_reduction':1-means['hindsight_rmsle']/means['cv_rmsle'],
                'numerical_fits':fits,'api_calls':0,'seconds':time.monotonic()-started,'rows':all_rows,
                'limitation':'Fixed development cohort mechanism screen, not an agent treatment effect or held-out result.'}
        save('report.json',result)
        return result
    except BaseException as error:
        save('FAILED.json',{'error':type(error).__name__,'message':str(error),'numerical_fits':fits,
                           'completed_cases':len(all_rows),'seconds':time.monotonic()-started})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source');parser.add_argument('output')
    args=parser.parse_args()
    result=run(args.source,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
