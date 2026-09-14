"""Frozen079daily error-transition correction over audited fixed forecasts."""
import argparse,hashlib,json,time
from datetime import timedelta
from pathlib import Path
from statistics import mean
from .search_ensemble_run import Source
from .guarded_correction import instant,visible,risk
from .error_transition import MODELS,errors,fit,apply


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def summarize(rows):
    names=('control','ledger','uncorrected045','strong050','lifetime061','incumbent068');means={a:mean(r['scores'][a] for r in rows) for a in names}
    return {'cases':len(rows),'mean_rmsle':means,'ledger_reduction':{a:1-means['ledger']/means[a] for a in names if a!='ledger'}}


def run(source,anchors,incumbent,output):
    src=Source(source);out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'transitions').mkdir();here=Path(__file__).parent;start=time.monotonic();cpu=time.process_time();rows=[];fits=0;pending=None;projection={};transition_cache={};comparisons={}
    ar_path=here/'evidence/broad-ensemble-045.json';ir_path=here/'evidence/search-ensemble-068.json';ar=json.loads(ar_path.read_text());ir=json.loads(ir_path.read_text())
    def bound(folder,name,receipt,label):
        p=Path(folder)/name
        if sha(p)!=receipt['files'][name]:raise ValueError('Changed frozen comparison '+label+'/'+name)
        comparisons[label+'/'+name]=receipt['files'][name];return json.loads(p.read_text())
    def fixed(tid):
        if tid in projection:return projection[tid]
        d=src.decision(tid,'control');groups=d['backtests'][:6];ids={m:g['config_id'] for m,g in zip(MODELS,groups,strict=True)};point={}
        for i,m in enumerate(MODELS):
            e=src.read(f'cases/{tid}/control/attempt-{4*i+4:02d}.json');f=src.read(e['cache_ref'])
            if e['config_id']!=ids[m] or f['request']['config_id']!=ids[m] or f['request']['history_end']!=730:raise ValueError('Changed fixed production config')
            point[m]=f['point']
        cv=[]
        for i,end in enumerate((658,682,706)):
            if any(g['folds'][i]['end']!=end or g['folds'][i]['actual']!=groups[0]['folds'][i]['actual'] for g in groups):raise ValueError('Unmatched fixed CV evidence')
            cv.append({'point':{m:g['folds'][i]['point'] for m,g in zip(MODELS,groups,strict=True)},'actual':groups[0]['folds'][i]['actual']})
        projection[tid]=(point,cv,ids);return point,cv,ids
    try:
        inputs=[src.read(n) for n in sorted(src.inventory) if n.startswith('inputs/')];metadata=[]
        if len(inputs)!=541 or len({t['task_id'] for t in inputs})!=541:raise ValueError('Complete original cohort required')
        for t in inputs:
            end=(instant(t['origin'])+timedelta(hours=24)).isoformat();metadata.append({**{k:t[k] for k in ('task_id','series_id','domain','origin')},'last_target':end,'source_available_at':end,'recorded_at':end})
        anchors_bykey={}
        for n in ar['files']:
            if n.startswith(('electricity:','pedestrian:')):
                r=bound(anchors,n,ar,'045');anchors_bykey[r['series_id'],r['origin']]={'file':n,'weights':r['control_fit']['weights'],'point':r['point']['cv_ensemble']}
        if len(anchors_bykey)!=416:raise ValueError('Missing frozen045anchors')
        write(out/'manifest.json',{'protocol':'ERROR_TRANSITION_079.md','code_sha256':{n:sha(here/n) for n in ('ERROR_TRANSITION_079.md','error_transition.py','error_transition_run.py','guarded_correction.py','search_ensemble_run.py')},'source_receipt_sha256':src.receipt_sha256,'anchor_receipt_sha256':sha(ar_path),'incumbent_receipt_sha256':sha(ir_path),'runtime_kind':'standalone_numerical_proxy','validation_or_final_access':False,'api_calls':0,'new_base_forecasts':0,'inherited_forecast_computations':49616,'inherited045anchor_fits':416,'inherited068blend_fits':832,'inherited_search_surrogate_solves':11902,'inherited_search_logical_attempts_per_arm':31378,'timing':'Nominal period-end availability,24-hour transition origins.'});write(out/'metadata.json',metadata)
        for task in sorted((r for r in inputs if r['round']>=0),key=lambda r:(instant(r['origin']),r['series_id'])):
            tid=task['task_id'];pending={'task_id':tid,'series_id':task['series_id'],'round':task['round']};origin=task['origin'];point,cv,ids=fixed(tid);a=anchors_bykey[task['series_id'],origin];cv_errors=[errors(p['point'],p['actual']).tolist() for p in cv];current=[{'previous_error':cv_errors[i],'next_error':cv_errors[i+1]} for i in range(2)];history=[];refs=[]
            for m in visible(metadata,origin,task['domain']):
                if instant(m['origin']).timetz()!=instant(origin).timetz():raise ValueError('Unmatched relative lead phase')
                oid=m['task_id']
                if oid not in transition_cache:
                    op,ocv,oi=fixed(oid);actual=src.read('outcomes/'+oid+'.json')
                    if any(instant(actual[rk])!=instant(m[mk]) for rk,mk in (('last_target','last_target'),('production_source_available_at','source_available_at'),('production_recorded_at','recorded_at'))):raise ValueError('Source timing differs from metadata')
                    transition={'previous_error':errors(ocv[2]['point'],ocv[2]['actual']).tolist(),'next_error':errors(op,actual['actual']).tolist()};transition_cache[oid]={'metadata':m,'config_ids':oi,'previous_origin':(instant(m['origin'])-timedelta(hours=24)).isoformat(),'transition':transition};write(out/'transitions'/(oid+'.json'),transition_cache[oid])
                h=transition_cache[oid]
                if h['config_ids']!=ids:raise ValueError('Model revision/configuration identity differs across transitions')
                history.append(h['transition']);refs.append({'file':'transitions/'+oid+'.json','sha256':sha(out/'transitions'/(oid+'.json')),'metadata':m})
            records={};points={}
            for arm in ('control','ledger'):
                transitions=current+history if arm=='ledger' else current;masses=[.25,.25]+[.5/len(history)]*len(history) if arm=='ledger' and history else [.5,.5];fits+=1;write(out/'attempt.json',{'pending':pending,'arm':arm,'fits_started':fits,'completed_cases':len(rows)})
                f=fit(transitions,masses);application=apply(point,cv_errors[2],a['weights'],f);records[arm]={'fit':f,'masses':masses,'historical_references':refs if arm=='ledger' else [],'application':application};points[arm]=application['point']
            # Both fitted forecasts exist before current production actual scoring.
            outcome=src.read('outcomes/'+tid+'.json');prior=bound(incumbent,tid+'.json',ir,'068')
            if outcome['actual']!=prior['actual']:raise ValueError('Unmatched comparison outcome')
            points.update(uncorrected045=a['point'],strong050=outcome['point']['strong_block_cv'],lifetime061=outcome['point']['lifetime_ledger'],incumbent068=prior['point']['ledger'])
            row={**pending,'origin':origin,'domain':task['domain'],'config_ids':ids,'anchor':a,'current_transition_origins':[(instant(origin)-timedelta(hours=h)).isoformat() for h in (72,48,24)],'current_transitions':current,'last_cv_error':cv_errors[2],'records':records,'point':points,'actual':outcome['actual'],'scores':{k:risk(p,outcome['actual']) for k,p in points.items()}};write(out/(tid+'.json'),row);rows.append(row);write(out/'status.json',{'completed_cases':len(rows),'transition_fits':fits,'seconds':time.monotonic()-start})
        if len(rows)!=416:raise ValueError('Incomplete experiment')
        overall=summarize(rows);domains={d:summarize([r for r in rows if r['domain']==d]) for d in ('electricity','pedestrian')};g=overall['ledger_reduction'];guards=('control','strong050','lifetime061','incumbent068');passed=g['control']>=.2 and g['strong050']>=.2 and all(g[k]>0 for k in guards) and all(d['ledger_reduction'][k]>0 for d in domains.values() for k in guards)
        report={'overall':overall,'domains':domains,'phases':{'early_0_7':summarize([r for r in rows if r['round']<8]),'later_8_25':summarize([r for r in rows if r['round']>=8])},'completed_cases':len(rows),'transition_fits':fits,'normal_equation_systems':6*fits,'stored_historical_transitions':len(transition_cache),'clipped_model_leads':{a:sum(len(v) for r in rows for v in r['records'][a]['application']['clipped_leads'].values()) for a in ('control','ledger')},'seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,'new_base_forecasts':0,'api_calls':0,'development_gate_passed':passed,'limitation':'Reused development numerical error-transition experiment, not held-out/agent evidence. Squared training surrogate versus mean-case RMSLE evaluation.'};write(out/'report.json',report);return report
    except BaseException as e:
        write(out/'FAILED.json',{'pending':pending,'error':type(e).__name__,'message':str(e),'completed_cases':len(rows),'fits_started':fits,'seconds':time.monotonic()-start});raise
    finally:write(out/'source_access.json',src.accessed);write(out/'comparison_access.json',comparisons)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','anchors','incumbent','output'):p.add_argument(n)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
