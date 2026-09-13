"""Matched numerical lab. Evidence logging is common; only ledger is treatment."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from numerical import configuration, config_id, predict, metrics
from maturation import plan


def read(name):
    return json.loads(Path(name).read_text())


def dump(name,value):
    Path(name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def append(name,value):
    with Path(name).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')


def logs():
    return [json.loads(line) for line in Path('experiments.jsonl').read_text().splitlines()] if Path('experiments.jsonl').exists() else []


def request_at(end):
    task=read('task.json')
    with open('history.csv') as f: rows=list(csv.DictReader(f))
    with open('future.csv') as f: future=list(csv.DictReader(f))
    training=rows[:end];targets=(rows+future)[end:end+14]
    names=[k for k in rows[0] if k not in ('timestamp','value')]
    request={'history':[float(r['value']) for r in training],
        'timestamps':[r['timestamp'] for r in training],
        'future_timestamps':[r['timestamp'] for r in targets],
        'past_covariates':[[float(r[k]) for k in names] for r in training],
        'future_covariates':[[float(r[k]) for k in names] for r in targets],
        'past_covariate_names':names,'future_covariate_names':names,
        'horizon':14,'series_id':task['series_id'],'unit':task['unit'],
        'cutoff':training[-1]['timestamp'],'frequency':'D','season':7,
        'known_time_cutoff':training[-1]['timestamp']}
    return request,([float(r['value']) for r in targets] if end<len(rows) else None)


def ledger(at, create=True):
    from gnomon import TemporalLedger
    from gnomon.ids import FixedClock
    return TemporalLedger('ledger.db',clock=FixedClock(datetime.fromisoformat(at)),create=create)


def score_in_ledger(db, execution_id, request, actual, now):
    db.append_actual(actuals=[{'series_id':request['series_id'],'unit':request['unit'],
        'valid_time':t,'source_available_at':t,'value':v,
        'source_ref':'visible-replay-observation'}
        for t,v in zip(request['future_timestamps'],actual,strict=True)])
    return db.evaluate(execution_id,source_as_of=now,recorded_as_of=now,allow_partial=False)


def execute(config,end,kind):
    task=read('task.json');backend=read('backend.json')['arm']
    attempts=[r for r in logs() if r.get('event')=='attempt' and r['task_origin']==task['origin']]
    if len(attempts)>=60:raise ValueError('60 numerical attempt budget exhausted')
    request,actual=request_at(end);cid=config_id(config)
    attempt={'event':'attempt','attempt_id':str(uuid4()),'task_origin':task['origin'],
             'config':config,'config_id':cid,'kind':kind,'request':request}
    append('experiments.jsonl',attempt)
    db=None
    if backend=='plain':
        point=predict(request,config)
        execution={'execution_id':str(uuid4()),'provider':config['model']+'_'+cid,
                   'revision':'ml-lab-v1:'+cid,'result':{'point':point}}
    else:
        from gnomon import ForecastResult, InferenceEngine, GnomonSession
        from gnomon.forecast_adapter import AdapterCapabilities
        if backend=='ledger':db=ledger(task['origin'])
        engine=InferenceEngine(ledger=db)
        provider=config['model']+'_'+cid
        def call(r):
            return ForecastResult(point=tuple(predict(request,config)),series_id=r.series_id,
                                  unit=r.unit,timestamps=r.future_timestamps,
                                  metadata={'config':config,'retrospective':kind=='backtest'})
        engine.register(provider,call,revision='ml-lab-v1:'+cid,lifecycle='stateless',
                        deterministic=True,capabilities=AdapterCapabilities(past_covariates=True,future_covariates=True))
        execution=GnomonSession(engine=engine).forecast(provider,request)
        if execution['status']!='ok':raise ValueError(execution)
        point=execution['result']['point']
    result={**attempt,'event':'result','execution':execution,'point':point,
            'actual':actual,'metrics':metrics(point,actual) if actual is not None else None}
    if db is not None and actual is not None:
        result['ledger_score']=score_in_ledger(db,execution['execution_id'],request,actual,task['origin'])
    append('experiments.jsonl',result)
    return result


def sync():
    """Score all eligible prior executions, without changing submitted forecasts."""
    task=read('task.json');arm=read('backend.json')['arm']
    groups,excluded=plan(logs(),read('previous_runs.json'),task)
    matured=[]
    for group in groups:
        p=group['outcome'];originals=group['results'];scores={}
        if arm=='ledger':
            db=ledger(task['origin'])
            # One observation batch per origin, not one revision per forecast.
            # Reuse already visible values when recovering an interrupted sync.
            visible={datetime.fromisoformat(a['valid_time']):a for a in db.actuals_as_of(
                task['series_id'],unit=task['unit'],source_as_of=task['origin'],
                recorded_as_of=task['origin'])}
            actuals=[]
            for t,v in zip(p['future_timestamps'],p['actual'],strict=True):
                previous=visible.get(datetime.fromisoformat(t))
                if previous is not None:
                    if previous['value'] != v:
                        raise ValueError('Visible actual conflicts with frozen replay outcome')
                    continue
                actuals.append({'series_id':task['series_id'],'unit':task['unit'],
                    'valid_time':t,'source_available_at':t,'value':v,
                    'source_ref':'visible-replay-observation'})
            if actuals:db.append_actual(actuals=actuals)
            ids=[r['execution']['execution_id'] for r in originals]
            values=db.evaluate(execution_ids=ids,source_as_of=task['origin'],
                               recorded_as_of=task['origin'],allow_partial=False)
            scores=dict(zip(ids,values,strict=True))
        for original in originals:
            eid=original['execution']['execution_id']
            value={'event':'matured','execution_id':eid,'task_origin':task['origin'],
                'forecast_origin':original['task_origin'],'config':original['config'],
                'config_id':original['config_id'],'actual':p['actual'],'point':original['point'],
                'selected_for_submission':eid==p.get('execution_id'),
                'outcome_recorded_at':p['outcome_recorded_at'],
                'metrics':metrics(original['point'],p['actual'])}
            if eid in scores:value['ledger_score']=scores[eid]
            append('experiments.jsonl',value);matured.append(eid)
    return {'matured_execution_ids':matured,'excluded':excluded,'provider_calls':0,
            'selection_changed':False}


def review(offset=0,limit=12,pair=None):
    from ledger_cards import development_cards,compact_cards
    import dev_evidence_summary
    task=read('task.json');rows=logs()
    if read('backend.json')['arm']!='ledger':raise ValueError('Ledger review requires ledger arm')
    if not Path('ledger.db').exists():
        return {'status':'insufficient_evidence','cards':[],'record_count':0,'ledger_queries':0,'provider_calls':0}
    full=development_cards(ledger(task['origin'],False),rows,task,dev_evidence_summary,
                           offset=offset,limit=limit,pair=pair)
    directory=Path('review-evidence');directory.mkdir(exist_ok=True)
    index=len(list(directory.glob('*.json')))
    path=directory/f'{index:06}.json'
    with path.open('x') as f:f.write(json.dumps(full,indent=2,allow_nan=False)+'\n')
    value=compact_cards(full,str(path))
    value['record_count']=sum(r['event']=='matured' for r in rows)
    append('review-log.jsonl',{'task_origin':task['origin'],**value})
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__,epilog='Config: ridge(window90..730,lags7..56,alpha.01..10000); random_forest(window90..730,lags7..56,depth2..16); seasonal(season1..28). Three fixed rolling folds. Use review for persistent evidence.')
    parser.add_argument('operation',choices=['backtest','forecast','review','sync'])
    parser.add_argument('--config',type=json.loads)
    args=parser.parse_args()
    if args.operation=='sync':sync();return
    if args.operation=='review':print(json.dumps(review()));return
    config=configuration(args.config or {})
    if args.operation=='backtest':
        rows=[execute(config,end,'backtest') for end in (688,702,716)]
        print(json.dumps({'config':config,'config_id':config_id(config),
            'folds':[{'origin':r['request']['cutoff'],'metrics':r['metrics'],'execution_id':r['execution']['execution_id']} for r in rows],
            'mean_rmsle':sum(r['metrics']['rmsle'] for r in rows)/3}))
    else:
        r=execute(config,730,'forecast')
        dump('execution.json',r['execution'])
        dump('forecast.json',{'point':r['point'],'method':config,'config_id':r['config_id'],
                             'execution_id':r['execution']['execution_id']})
        print(json.dumps({'saved':'forecast.json','execution_id':r['execution']['execution_id'],'config':config}))


if __name__=='__main__':main()
