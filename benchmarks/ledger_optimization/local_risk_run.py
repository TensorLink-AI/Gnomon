"""Frozen081per-step conditional model-risk development comparison."""
import argparse,hashlib,importlib.metadata,json,time
from datetime import timedelta
from pathlib import Path
from statistics import mean
import numpy as np
from .search_ensemble_run import Source
from .guarded_correction import MODELS,instant,visible,risk
from .local_risk import fit,predict_matrices
from .conditional_risk import fit_weights


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def summarize(rows):
    names=('control','ledger','uncorrected045','strong050','lifetime061','incumbent068');means={a:mean(r['scores'][a] for r in rows) for a in names}
    return {'cases':len(rows),'mean_rmsle':means,'ledger_reduction':{a:1-means['ledger']/means[a] for a in names if a!='ledger'}}


def run(source,anchors,incumbent,output):
    src=Source(source);out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir();here=Path(__file__).parent;start=time.monotonic();cpu=time.process_time();rows=[];pending=None;projection={};comparison_access={};counts={'forests_started':0,'forests_completed':0,'trees':0,'weight_fits_started':0,'weight_fits_completed':0,'weight_iterations':0}
    ar_path=here/'evidence/broad-ensemble-045.json';ir_path=here/'evidence/search-ensemble-068.json';ar=json.loads(ar_path.read_text());ir=json.loads(ir_path.read_text())
    def bound(folder,name,receipt,label):
        p=Path(folder)/name
        if sha(p)!=receipt['files'][name]:raise ValueError('Changed frozen comparison '+label+'/'+name)
        comparison_access[label+'/'+name]=receipt['files'][name];return json.loads(p.read_text())
    def fixed(tid):
        if tid in projection:return projection[tid]
        d=src.decision(tid,'control');groups=d['backtests'][:6];ids={m:g['config_id'] for m,g in zip(MODELS,groups,strict=True)};point={}
        for i,m in enumerate(MODELS):
            e=src.read(f'cases/{tid}/control/attempt-{4*i+4:02d}.json');f=src.read(e['cache_ref'])
            if e['config_id']!=ids[m] or f['request']['config_id']!=ids[m] or f['request']['history_end']!=730:raise ValueError('Fixed production identity')
            point[m]=f['point']
        cv=[]
        for i,end in enumerate((658,682,706)):
            if any(g['folds'][i]['end']!=end or g['folds'][i]['actual']!=groups[0]['folds'][i]['actual'] for g in groups):raise ValueError('Unmatched fixed CV')
            cv.append({'point':{m:g['folds'][i]['point'] for m,g in zip(MODELS,groups,strict=True)},'actual':groups[0]['folds'][i]['actual']})
        projection[tid]=(point,cv,ids);return point,cv,ids
    try:
        tasks=[src.read(n) for n in sorted(src.inventory) if n.startswith('inputs/')];metadata=[]
        if len(tasks)!=541 or len({t['task_id'] for t in tasks})!=541:raise ValueError('Complete541tasks required')
        for t in tasks:
            end=(instant(t['origin'])+timedelta(hours=24)).isoformat();metadata.append({**{k:t[k] for k in ('task_id','series_id','domain','origin')},'last_target':end,'source_available_at':end,'recorded_at':end})
        anchor_map={}
        for n in ar['files']:
            if n.startswith(('electricity:','pedestrian:')):
                r=bound(anchors,n,ar,'045');anchor_map[r['series_id'],r['origin']]={'file':n,'weights':r['control_fit']['weights'],'point':r['point']['cv_ensemble']}
        if len(anchor_map)!=416:raise ValueError('Complete416anchors required')
        write(out/'manifest.json',{'protocol':'LOCAL_RISK_081.md','code_sha256':{n:sha(here/n) for n in ('LOCAL_RISK_080.md','LOCAL_RISK_081.md','local_risk.py','local_risk_run.py','guarded_correction.py','conditional_risk.py','search_ensemble_run.py')},'packages':{n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn')},'source_receipt_sha256':src.receipt_sha256,'anchor_receipt_sha256':sha(ar_path),'incumbent_receipt_sha256':sha(ir_path),'runtime_kind':'standalone_numerical_proxy','validation_or_final_access':False,'api_calls':0,'new_base_forecasts':0,'inherited_forecast_computations':49616,'inherited045anchor_fits':416,'inherited068blend_fits':832,'inherited_search_surrogate_solves':11902,'inherited_search_logical_attempts_per_arm':31378,'timing':'Nominal period-end recording/source availability.'});write(out/'metadata.json',metadata)
        for task in sorted((t for t in tasks if t['round']>=0),key=lambda t:(instant(t['origin']),t['series_id'])):
            tid=task['task_id'];origin=task['origin'];pending={'task_id':tid,'series_id':task['series_id'],'round':task['round']};point,cv,ids=fixed(tid);anchor=anchor_map[task['series_id'],origin];history=[];refs=[]
            for m in visible(metadata,origin,task['domain']):
                op,_,oi=fixed(m['task_id']);o=src.read('outcomes/'+m['task_id']+'.json')
                if oi!=ids or instant(m['origin']).timetz()!=instant(origin).timetz():raise ValueError('Fixed model/phase mismatch')
                if any(instant(o[rk])!=instant(m[mk]) for rk,mk in (('last_target','last_target'),('production_source_available_at','source_available_at'),('production_recorded_at','recorded_at'))):raise ValueError('Incorrect historical availability metadata')
                history.append({'point':op,'actual':o['actual']});refs.append({**m,'config_ids':oi})
            records={};points={}
            for arm in ('control','ledger'):
                pairs=cv+history if arm=='ledger' else cv;masses=[1/6]*3+[.5/len(history)]*len(history) if arm=='ledger' and history else [1/3]*3;counts['forests_started']+=1;pending.update(arm=arm,stage='forest');write(out/'attempt.json',{'pending':pending,**counts,'completed_cases':len(rows)})
                model,info=fit(pairs,masses,anchor['weights']);counts['forests_completed']+=1;counts['trees']+=info['tree_count'];name=f'models/{tid}-{arm}.json';write(out/name,{'task_id':tid,'arm':arm,'origin':origin,'historical_references':refs if arm=='ledger' else [],'masses':masses,'fit':info});matrices=predict_matrices(model,point,anchor['weights']);certificates=[]
                try:
                    for h,g in enumerate(matrices):
                        counts['weight_fits_started']+=1;pending.update(stage='weight',lead=h);f=fit_weights(g,anchor['weights']);certificates.append(f);counts['weight_fits_completed']+=1;counts['weight_iterations']+=f['iterations']
                except BaseException:
                    write(out/'partial-weight-fits.json',{'pending':pending,'matrices':matrices.tolist(),'completed_certificates':certificates});raise
                logs=np.log1p(np.asarray([point[m] for m in MODELS],dtype=float)).T;weights=np.array([f['weights'] for f in certificates]);forecast=np.expm1(np.sum(logs*weights,axis=1))
                if not np.isfinite(forecast).all():raise ValueError('Nonfinite forecast')
                points[arm]=forecast.tolist();records[arm]={'model_file':name,'model_sha256':sha(out/name),'matrices':matrices.tolist(),'weight_fits':certificates,'point':forecast.tolist()}
            # Current production outcomes are scored only after both arm outputs.
            outcome=src.read('outcomes/'+tid+'.json');prior=bound(incumbent,tid+'.json',ir,'068')
            if outcome['actual']!=prior['actual']:raise ValueError('Mismatched current outcomes')
            points.update(uncorrected045=anchor['point'],strong050=outcome['point']['strong_block_cv'],lifetime061=outcome['point']['lifetime_ledger'],incumbent068=prior['point']['ledger'])
            row={**{k:task[k] for k in ('task_id','series_id','domain','origin','round')},'config_ids':ids,'anchor':anchor,'records':records,'point':points,'actual':outcome['actual'],'scores':{k:risk(p,outcome['actual']) for k,p in points.items()}};write(out/(tid+'.json'),row);rows.append(row);write(out/'status.json',{'completed_cases':len(rows),**counts,'seconds':time.monotonic()-start})
            if len(rows)%16==0:print(json.dumps({'completed_cases':len(rows),'seconds':time.monotonic()-start}),flush=True)
        if len(rows)!=416:raise ValueError('Incomplete scored cohort')
        overall=summarize(rows);domains={d:summarize([r for r in rows if r['domain']==d]) for d in ('electricity','pedestrian')};g=overall['ledger_reduction'];guards=('control','strong050','lifetime061','incumbent068');passed=g['control']>=.2 and g['strong050']>=.2 and all(g[a]>0 for a in guards) and all(d['ledger_reduction'][a]>0 for d in domains.values() for a in guards)
        report={'overall':overall,'domains':domains,'phases':{'early_0_7':summarize([r for r in rows if r['round']<8]),'later_8_25':summarize([r for r in rows if r['round']>=8])},'completed_cases':len(rows),**counts,'seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,'new_base_forecasts':0,'api_calls':0,'development_gate_passed':passed,'limitation':'Repeated-development numerical lead-conditioned risk experiment; squared-log training surrogate; no held-out/agent claim.'};write(out/'report.json',report);return report
    except BaseException as e:
        write(out/'FAILED.json',{'pending':pending,'error':type(e).__name__,'message':str(e),'certificate':getattr(e,'certificate',None),'completed_cases':len(rows),**counts,'seconds':time.monotonic()-start});raise
    finally:write(out/'source_access.json',src.accessed);write(out/'comparison_access.json',comparison_access)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','anchors','incumbent','output'):p.add_argument(n)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
