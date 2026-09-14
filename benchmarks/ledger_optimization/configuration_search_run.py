"""Frozen066 chronological real-development search with common logical budgets."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
from pathlib import Path
from statistics import mean
import time
from . import configuration_search as search
from .configured_hourly import catalog,config_id,predict,revision
from .broad_screen import rmsle


def encoded(value):return json.dumps(value,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(encoded(value)).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


class Execution:
    def __init__(self,root,task,starters,arm,predictor=predict):
        self.root=Path(root);self.task=task;self.starters=starters;self.arm=arm;self.predictor=predictor;self.events=[]
        self.home=self.root/'cases'/task['task_id']/arm;self.home.mkdir(parents=True,exist_ok=False)
        self.cache=self.root/'forecasts'/task['task_id'];self.cache.mkdir(parents=True,exist_ok=True)
    def call(self,config,end):
        if len(self.events)>=60:raise ValueError('Numerical attempt budget exhausted')
        if (self.root/'STOP').exists():raise RuntimeError('Peer failure requested stop')
        cid=config_id(config);request={'task_id':self.task['task_id'],'input_sha256':digest(self.task),'config_id':cid,'revision':revision(),'history_end':end}
        rid=digest(request);path=self.cache/(rid+'.json');index=len(self.events)+1
        event={'attempt':index,'arm':self.arm,'execution_id':digest({'request':rid,'arm':self.arm,'attempt':index}),
            'config_id':cid,'config':config,'history_end':end,'request_id':rid,'cache_ref':str(path.relative_to(self.root)),
            'completed':False,'physical_started':False,'estimator_fit':config['kind'] in ('ridge','forest')}
        self.events.append(event);log=self.home/f'attempt-{index:02d}.json';write(log,event);start=time.monotonic()
        try:
            if path.exists():
                cached=json.loads(path.read_text());event['resolution']='local_cache'
                if cached['request']!=request:raise ValueError('Cache request identity mismatch')
                point=cached['point']
            elif cid in self.starters:
                original=self.starters[cid];point=original['production'] if end==730 else original['folds'][str(end)]
                event['resolution']='inherited_source'
                cached={'request':request,'point':point,'origin':'original038_043','source_ref':original['source_ref']}
            else:
                event.update(resolution='new_computation',physical_started=True);write(log,event)
                low=datetime.fromisoformat(self.task['origin'])-timedelta(hours=730)
                labels=[low+timedelta(hours=i) for i in range(end+24)]
                point=self.predictor(self.task['history'][:end],labels[:end],labels[end:],config)
                cached={'request':request,'point':point,'origin':'configured064','physical_seconds':time.monotonic()-start}
            if end not in (658,682,706,730) or len(point)!=24 or any(not math.isfinite(v) or v<0 for v in point):raise ValueError('Invalid complete forecast contract')
            if not path.exists():write(path,cached)
            event['completed']=True;return point,event.copy()
        except BaseException as error:
            event.update(error=type(error).__name__,message=str(error));raise
        finally:event['seconds']=time.monotonic()-start;write(log,event)
    def backtest(self,config):
        search.admit_backtest(len(self.events));folds=[]
        for end in (658,682,706):
            point,event=self.call(config,end);actual=self.task['history'][end:end+24]
            folds.append({'end':end,'point':point,'actual':actual,'rmsle':rmsle(point,actual),'execution_id':event['execution_id'],'cache_ref':event['cache_ref']})
        return {'config_id':config_id(config),'config':config,'cv_rmsle':mean(f['rmsle'] for f in folds),'folds':folds}


def run_case(root,task,starters,arm,history,predictor=predict):
    executor=Execution(root,task,starters,arm,predictor);start=time.monotonic();cpu=time.process_time();backtests=[];proposals=[]
    current={k:task[k] for k in ('series_id','domain','origin','features')};current['arm']=arm
    try:
        for row in catalog()[:6]:
            backtests.append(executor.backtest(row['config']));executor.call(row['config'],730)
        if len(executor.events)!=24:raise AssertionError('Common starter charge differs')
        for step in range(11):
            search.admit_backtest(len(executor.events))
            proposal=search.suggest(current,backtests,history if arm=='ledger' else [])
            write(executor.home/f'proposal-{step:02d}.json',proposal);proposals.append(str((executor.home/f'proposal-{step:02d}.json').relative_to(Path(root))))
            backtests.append(executor.backtest(proposal['next_config']))
        selected=search.select(backtests);point,execution=executor.call(selected['config'],730)
        if len(executor.events)!=58 or len(backtests)!=17:raise AssertionError('Unequal fixed search budget')
        study={**current,'source_available_at':task['origin'],'recorded_at':task['origin'],'evidence_kind':'executed_backtests','revision':revision(),
            'backtests':[{k:r[k] for k in ('config','config_id','cv_rmsle')} for r in backtests],
            'study_ref':str((executor.home/'decision.json').relative_to(Path(root)))}
        result={'task_id':task['task_id'],'arm':arm,'series_id':task['series_id'],'round':task['round'],'origin':task['origin'],
            'backtests':backtests,'selected':selected,'point':point,'final_execution_id':execution['execution_id'],'final_cache_ref':execution['cache_ref'],
            'study':study,'proposal_refs':proposals,'logical_attempts':len(executor.events),'physical_computations':sum(e['physical_started'] for e in executor.events),
            'physical_estimator_fits':sum(e['physical_started'] and e['estimator_fit'] for e in executor.events),
            'reused_computations':sum(not e['physical_started'] for e in executor.events),'surrogate_solves':len(proposals),
            'seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu}
        write(executor.home/'decision.json',result);return result
    except BaseException as error:
        write(executor.home/'FAILED.json',{'error':type(error).__name__,'message':str(error),'logical_attempts_started':len(executor.events),
            'completed_attempts':sum(e['completed'] for e in executor.events),'physical_started':sum(e['physical_started'] for e in executor.events),'seconds':time.monotonic()-start})
        raise


def domain_run(domain,items,root):
    root=Path(root);histories={'control':[],'ledger':[]};rows=[];costs={a:{'logical_attempts':0,'physical_computations':0,'physical_estimator_fits':0,'reused_computations':0,'surrogate_solves':0,'seconds':0.,'cpu_seconds':0.,'cases':0} for a in histories}
    start=time.monotonic();pending=None
    try:
        origins=sorted({item['input']['origin'] for item in items})
        for origin in origins:
            batch=[]
            for item in sorted((r for r in items if r['input']['origin']==origin),key=lambda r:r['input']['series_id']):
                task=item['input'];decisions={}
                # Alternate physical cache ownership, not information or policy.
                order=('control','ledger') if (task['round']+8)%2==0 else ('ledger','control')
                for arm in order:
                    pending={'task_id':task['task_id'],'series_id':task['series_id'],'round':task['round'],'arm':arm}
                    decisions[arm]=run_case(root,task,item['starters'],arm,histories[arm])
                    for key in costs[arm]:
                        if key=='cases':costs[arm][key]+=1
                        else:costs[arm][key]+=decisions[arm][key]
                    write(root/f'status-{domain}.json',{'pending':pending,'completed_arm_cases':sum(c['cases'] for c in costs.values()),'costs':costs,'seconds':time.monotonic()-start})
                batch.append((item,decisions))
            # Only after every same-origin decision: score/mature records for later queries.
            for item,decisions in batch:
                task=item['input'];actual=item['target'];scores={a:rmsle(d['point'],actual) for a,d in decisions.items()}
                if task['round']>=0:scores.update({a:rmsle(p,actual) for a,p in item['guards'].items()})
                result={'task_id':task['task_id'],'series_id':task['series_id'],'round':task['round'],'origin':origin,'last_target':item['last_target'],
                    'actual':actual,'scores':scores,'point':{**{a:d['point'] for a,d in decisions.items()},**item['guards']},
                    'selected':{a:d['selected']['config_id'] for a,d in decisions.items()},'production_source_available_at':item['last_target'],'production_recorded_at':item['last_target'],
                    'production_outcomes_used_by_search':False,'warmup':task['round']<0}
                write(root/'outcomes'/(task['task_id']+'.json'),result);rows.append(result)
                for arm in histories:histories[arm].append(decisions[arm]['study'])
        report={'domain':domain,'cases':len(rows),'rows':rows,'costs':costs,'seconds':time.monotonic()-start};write(root/f'completed-{domain}.json',report);return report
    except BaseException as error:
        (root/'STOP').touch();write(root/f'FAILED-{domain}.json',{'pending':pending,'error':type(error).__name__,'message':str(error),'costs':costs,'completed_outcomes':len(rows),'seconds':time.monotonic()-start});raise


def prepare_items(source,warm_source,original,warm,guard,output):
    here=Path(__file__).parent;receipts={}
    def read_hashed(path,expected):
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Changed source '+str(path))
        return json.loads(raw)
    spans=read_hashed(source,'60c8db1de333290d7cc04ca3a52dc4c4d46d8b93867105c5cb9be03adb8de15f')
    earlier=read_hashed(warm_source,'32a6c000a32021656431af81ab39f4106c1b9ca63649da19278b605aae589df4')
    def load(root,name,prefix):
        path=here/'evidence'/name;receipts[name]=hashlib.sha256(path.read_bytes()).hexdigest();receipt=json.loads(path.read_text());rows=[]
        for n,h in receipt['files'].items():
            if n.startswith(prefix):rows.append((read_hashed(Path(root)/n,h),{'root':str(root),'file':n,'sha256':h}))
        return rows
    scored=load(original,'broad-screen-038.json',('electricity:','pedestrian:'));early=load(warm,'broad-warm-screen-043.json',('warmup-electricity:','warmup-pedestrian:'))
    contexts=load(warm,'broad-warm-screen-043.json',('episodes.json',))[0][0];guards=load(guard,'broad-lifetime-memory-061.json',('electricity:','pedestrian:'))
    if tuple(map(len,(scored,early,contexts,guards)))!=(416,125,541,416):raise ValueError('Full frozen cohort required')
    index=lambda values:{(r['series_id'],r['origin']):r for r in values}
    meta=index(contexts);guard_index=index([r for r,_ in guards]);items=[]
    names=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')
    if len(meta)!=541 or len(guard_index)!=416:raise ValueError('Duplicate metadata identity')
    for raw,ref in scored+early:
        key=raw['series_id'],raw['origin'];span=(earlier if raw['round']<0 else spans)[key[0]];low,stop=raw['history_indices'];values=span['values'][low:stop];target=span['values'][stop:stop+24]
        if len(values)!=730 or len(target)!=24 or any(v is None or not math.isfinite(v) or v<0 for v in values+target):raise ValueError('Missing or invalid prepared task')
        if digest(values)!=raw['history_sha256'] or target!=raw['actual'] or raw['target_indices']!=[stop,stop+24]:raise ValueError('Task observations changed')
        at=datetime.fromisoformat(span['start_label']).replace(tzinfo=timezone.utc)+timedelta(hours=stop)
        if at.isoformat()!=raw['origin'] or (at+timedelta(hours=24)).isoformat()!=raw['last_target']:raise ValueError('Task clock changed')
        task_id=digest({'series_id':key[0],'origin':key[1]});task={'task_id':task_id,'series_id':key[0],'domain':meta[key]['domain'],'round':raw['round'],'origin':raw['origin'],
            'history':values,'features':meta[key]['features'],'unit':span['unit'],'source_sha256':span['source_sha256']}
        starter={}
        for name,row in zip(names,catalog()[:6],strict=True):
            folds={str(end):raw['folds'][name][i]['point'] for i,end in enumerate((658,682,706))}
            for i,end in enumerate((658,682,706)):
                if raw['folds'][name][i]['actual']!=values[end:end+24]:raise ValueError('Original CV actuals changed')
            starter[row['config_id']]={'production':raw['point'][name],'folds':folds,'source_ref':ref}
        guard_points={}
        if raw['round']>=0:
            g=guard_index[key]
            if g['actual']!=target:raise ValueError('Guard actuals changed')
            guard_points={'strong_block_cv':g['point']['block_cv'],'lifetime_ledger':g['point']['lifetime_ledger']}
        write(Path(output)/'inputs'/(task_id+'.json'),task)
        items.append({'input':task,'starters':starter,'target':target,'last_target':raw['last_target'],'guards':guard_points})
    if len({r['input']['task_id'] for r in items})!=541:raise ValueError('Duplicate task')
    return items,receipts


def summary(rows):
    arms=('control','ledger','strong_block_cv','lifetime_ledger');means={a:mean(r['scores'][a] for r in rows) for a in arms}
    return {'cases':len(rows),'mean_rmsle':means,'ledger_reduction':{a:1-means['ledger']/means[a] for a in arms if a!='ledger'}}


def run(source,warm_source,original,warm,guard,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'inputs').mkdir();(output/'outcomes').mkdir();here=Path(__file__).parent;start=time.monotonic()
    try:
        items,receipts=prepare_items(source,warm_source,original,warm,guard,output)
        write(output/'manifest.json',{'protocol':'CONFIGURATION_SEARCH_RUN_066.md','code_sha256':{n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in
            ('CONFIGURATION_SEARCH_RUN_066.md','configuration_search_run.py','configuration_search.py','configured_hourly.py','CONFIGURATION_SEARCH_065.md')},
            'source_receipts_sha256':receipts,'source_span_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'warm_span_sha256':hashlib.sha256(Path(warm_source).read_bytes()).hexdigest(),
            'revision':revision(),'packages':{n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn')},'workers':2,'scored_tasks':416,'warmup_tasks':125,
            'unavailable_warmups':3,'nominal_utc_and_period_end_assumptions':True,'validation_or_final_access':False,'api_calls':0,'runtime_kind':'standalone_numerical_proxy_not_Gnomon_or_Hermes_execution'})
        results={};failures={}
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs={pool.submit(domain_run,d,[r for r in items if r['input']['domain']==d],str(output)):d for d in ('electricity','pedestrian')}
            for future in as_completed(jobs):
                domain=jobs[future]
                try:results[domain]=future.result()
                except BaseException as error:failures[domain]={'error':type(error).__name__,'message':str(error)};(output/'STOP').touch()
        if failures:raise RuntimeError('Domain failures: '+json.dumps(failures))
        rows=[r for d in results.values() for r in d['rows'] if not r['warmup']]
        if len(rows)!=416:raise ValueError('Incomplete scored denominator')
        overall=summary(rows);domains={d:summary([r for r in rows if r['series_id'].startswith(d+':')]) for d in ('electricity','pedestrian')}
        costs={arm:{k:sum(d['costs'][arm][k] for d in results.values()) for k in results['electricity']['costs'][arm]} for arm in ('control','ledger')}
        reduction=overall['ledger_reduction'];passed=reduction['control']>=.2 and reduction['strong_block_cv']>=.2 and reduction['lifetime_ledger']>0 and all(v>0 for d in domains.values() for v in d['ledger_reduction'].values())
        report={'overall':overall,'domains':domains,'phases':{'early_0_7':summary([r for r in rows if r['round']<8]),'later_8_25':summary([r for r in rows if r['round']>=8])},
            'costs':costs,'seconds':time.monotonic()-start,'completed_arm_cases':sum(c['cases'] for c in costs.values()),'scored_tasks':416,'warmup_tasks':125,
            'inherited_original_forecast_computations':12984,'guard_extra_historical_forecast_computations':12600,'api_calls':0,'development_gate_passed':passed,
            'limitation':'Reused development numerical tuning-reuse proxy. No agent treatment estimate or confirmatory interval.'}
        write(output/'report.json',report);return report
    except BaseException as error:
        write(output/'FAILED.json',{'error':type(error).__name__,'message':str(error),'seconds':time.monotonic()-start});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','warm_source','original','warm','guard','output'):parser.add_argument(name)
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))
