"""Frozen078 guarded forest correction with two explicit evidence cutoffs."""
import argparse,hashlib,importlib.metadata,json,time
from datetime import timedelta
from pathlib import Path
from statistics import mean
from scipy.optimize import minimize
from .ensemble_refinement import isolated
from .search_ensemble_run import Source
from .guarded_correction import MODELS,instant,visible,training,fit,apply,decide,risk,features


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def summarize(rows):
    names=('control','ledger','uncorrected045','strong050','lifetime061','incumbent068');means={a:mean(r['scores'][a] for r in rows) for a in names}
    return {'cases':len(rows),'mean_rmsle':means,'ledger_reduction':{a:1-means['ledger']/means[a] for a in names if a!='ledger'}}


def run(source,anchors,incumbent,output):
    src=Source(source);out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir();here=Path(__file__).parent;started=time.monotonic();cpu=time.process_time();rows=[];pending=None
    counts={'guard_baseline_fits':0,'guard_baseline_iterations':0,'guard_forests':0,'production_forests':0,'trees':0};accessed={};projection_cache={}
    baseline=isolated('benchmarks.ledger_optimization._guard_baseline078',here/'evidence_ensemble.py')
    def refined(*args,**kwargs):
        kwargs['options']={**kwargs['options'],'ftol':1e-12};return minimize(*args,**kwargs)
    baseline.minimize=refined
    ar=here/'evidence/broad-ensemble-045.json';ir=here/'evidence/search-ensemble-068.json';anchor_receipt=json.loads(ar.read_text());incumbent_receipt=json.loads(ir.read_text())
    def bound(root,name,receipt,label):
        p=Path(root)/name
        if sha(p)!=receipt['files'][name]:raise ValueError('Changed frozen '+label+' '+name)
        accessed[label+'/'+name]=receipt['files'][name];return json.loads(p.read_text())
    def fixed(tid):
        if tid in projection_cache:return projection_cache[tid]
        d=src.decision(tid,'control');groups=d['backtests'][:6];ids={m:g['config_id'] for m,g in zip(MODELS,groups,strict=True)};points={}
        for i,m in enumerate(MODELS):
            e=src.read(f'cases/{tid}/control/attempt-{4*i+4:02d}.json');f=src.read(e['cache_ref'])
            if e['config_id']!=ids[m] or f['request']['config_id']!=ids[m] or f['request']['history_end']!=730:raise ValueError('Fixed production configuration mismatch')
            points[m]=f['point']
        pairs=[]
        for i,end in enumerate((658,682,706)):
            if any(g['folds'][i]['end']!=end or g['folds'][i]['actual']!=groups[0]['folds'][i]['actual'] for g in groups):raise ValueError('Unmatched fixed CV folds')
            pairs.append({'point':{m:g['folds'][i]['point'] for m,g in zip(MODELS,groups,strict=True)},'actual':groups[0]['folds'][i]['actual']})
        projection_cache[tid]=(points,pairs,ids);return points,pairs,ids
    try:
        inputs=[src.read(n) for n in sorted(src.inventory) if n.startswith('inputs/')]
        if len(inputs)!=541 or len({r['task_id'] for r in inputs})!=541:raise ValueError('Complete541task population required')
        metadata=[]
        for task in inputs:
            end=(instant(task['origin'])+timedelta(hours=24)).isoformat();metadata.append({**{k:task[k] for k in ('task_id','series_id','domain','origin')},'last_target':end,'source_available_at':end,'recorded_at':end})
        anchor_index={}
        for n in anchor_receipt['files']:
            if n.startswith(('electricity:','pedestrian:')):
                a=bound(anchors,n,anchor_receipt,'045');anchor_index[a['series_id'],a['origin']]={'file':n,'weights':a['control_fit']['weights'],'point':a['point']['cv_ensemble']}
        if len(anchor_index)!=416:raise ValueError('Complete045baseline required')
        manifest={'protocol':'GUARDED_CORRECTION_078.md','code_sha256':{n:sha(here/n) for n in ('GUARDED_CORRECTION_077.md','GUARDED_CORRECTION_078.md','guarded_correction.py','guarded_correction_run.py','evidence_ensemble.py','ensemble_refinement.py','search_ensemble_run.py')},'packages':{n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn')},'source_receipt_sha256':src.receipt_sha256,'anchor_receipt_sha256':sha(ar),'incumbent_receipt_sha256':sha(ir),'runtime_kind':'standalone_numerical_proxy','api_calls':0,'new_base_forecasts':0,'validation_or_final_access':False,'inherited_forecast_computations':49616,'inherited045anchor_fits':416,'inherited068blend_fits':832,'inherited_search_surrogate_solves':11902,'inherited_search_logical_attempts_per_arm':31378,'timing':'Nominal period-end visibility; historical stage origins simulated.'}
        write(out/'manifest.json',manifest);write(out/'metadata.json',metadata)
        def past(stage_origin,domain):
            refs=visible(metadata,stage_origin,domain);pairs=[];saved=[]
            for ref in refs:
                r=src.read('outcomes/'+ref['task_id']+'.json');point,_,ids=fixed(ref['task_id'])
                if any(instant(r[rk])!=instant(ref[mk]) for rk,mk in (('last_target','last_target'),('production_source_available_at','source_available_at'),('production_recorded_at','recorded_at'))):raise ValueError('Source visibility differs from nominal metadata')
                if instant(ref['origin']).timetz()!=instant(stage_origin).timetz():raise ValueError('Different historical lead phase')
                pairs.append({'point':point,'actual':r['actual']});saved.append({**ref,'config_ids':ids})
            return pairs,saved
        def forest(tid,arm,stage,cutoff,current,history,refs,weights):
            counts[stage+'_forests']+=1;write(out/'attempt.json',{'pending':pending,'stage':stage,'arm':arm,**counts,'completed_cases':len(rows)})
            pairs,masses=training(current,history);model,info=fit(pairs,masses,weights);counts['trees']+=info['tree_count'];name=f'{tid}-{arm}-{stage}.json'
            write(out/'models'/name,{'task_id':tid,'arm':arm,'stage':stage,'cutoff':cutoff,'current_cv_ends':[658,682] if stage=='guard' else [658,682,706],'historical_references':refs,'masses':masses,'fit':info})
            return model,{'file':'models/'+name,'sha256':sha(out/'models'/name),'training_cases':len(pairs),'historical_cases':len(history)}
        for task in sorted((r for r in inputs if r['round']>=0),key=lambda r:(instant(r['origin']),r['series_id'])):
            tid=task['task_id'];pending={'task_id':tid,'series_id':task['series_id'],'round':task['round']};now=task['origin'];cutoff=(instant(now)-timedelta(hours=24)).isoformat();point,cv,ids=fixed(tid);a=anchor_index[task['series_id'],now]
            counts['guard_baseline_fits']+=1;gb=baseline.fit(cv[:2],[.5,.5]);counts['guard_baseline_iterations']+=gb['iterations'];history,refs=past(cutoff,task['domain']);records={}
            # Both held-back predictions exist before held-back actual scoring.
            for arm in ('control','ledger'):
                hp,hr=(history,refs) if arm=='ledger' else ([],[]);model,ref=forest(tid,arm,'guard',cutoff,cv[:2],hp,hr,gb['weights']);prediction=apply(cv[2]['point'],gb['weights'],model);records[arm]={'guard_model':ref,'guard_prediction':prediction}
            for arm in ('control','ledger'):
                v=records[arm]['guard_prediction'];records[arm]['decision']=decide(v['base_point'],v['point'],cv[2]['actual'])
            production_history,production_refs=(past(now,task['domain']) if records['ledger']['decision']['correction_enabled'] else ([],[]));points={}
            for arm in ('control','ledger'):
                if records[arm]['decision']['correction_enabled']:
                    hp,hr=(production_history,production_refs) if arm=='ledger' else ([],[]);model,ref=forest(tid,arm,'production',now,cv,hp,hr,a['weights']);prediction=apply(point,a['weights'],model);records[arm].update(production_model=ref,production_prediction=prediction);points[arm]=prediction['point']
                    if max(abs(x-y) for x,y in zip(prediction['base_point'],a['point'],strict=True))>1e-12:raise ValueError('045production baseline changed')
                else:records[arm].update(production_model=None,production_prediction=None);points[arm]=a['point']
            # Current outcome is used only after all arm decisions and predictions.
            outcome=src.read('outcomes/'+tid+'.json');prior=bound(incumbent,tid+'.json',incumbent_receipt,'068')
            if outcome['actual']!=prior['actual']:raise ValueError('Unmatched comparison actuals')
            points.update(uncorrected045=a['point'],strong050=outcome['point']['strong_block_cv'],lifetime061=outcome['point']['lifetime_ledger'],incumbent068=prior['point']['ledger'])
            row={**pending,'origin':now,'domain':task['domain'],'guard_origin':cutoff,'current_config_ids':ids,'anchor':a,'guard_baseline':gb,'records':records,'point':points,'actual':outcome['actual'],'scores':{k:risk(v,outcome['actual']) for k,v in points.items()}}
            write(out/(tid+'.json'),row);rows.append(row);write(out/'status.json',{'completed_cases':len(rows),**counts,'seconds':time.monotonic()-started})
            if len(rows)%32==0:print(json.dumps({'completed_cases':len(rows),'seconds':time.monotonic()-started}),flush=True)
        if len(rows)!=416:raise ValueError('Incomplete scored cohort')
        overall=summarize(rows);domains={d:summarize([r for r in rows if r['domain']==d]) for d in ('electricity','pedestrian')};g=overall['ledger_reduction'];guards=('control','strong050','lifetime061','incumbent068');passed=g['control']>=.2 and g['strong050']>=.2 and all(g[a]>0 for a in guards) and all(d['ledger_reduction'][a]>0 for d in domains.values() for a in guards)
        report={'overall':overall,'domains':domains,'phases':{'early_0_7':summarize([r for r in rows if r['round']<8]),'later_8_25':summarize([r for r in rows if r['round']>=8])},'completed_cases':len(rows),**counts,'correction_enabled':{a:sum(r['records'][a]['decision']['correction_enabled'] for r in rows) for a in ('control','ledger')},'seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,'new_base_forecasts':0,'api_calls':0,'development_gate_passed':passed,'limitation':'Repeated-development numerical guarded-correction experiment, not an actual-agent or held-out result.'}
        write(out/'report.json',report);return report
    except BaseException as e:
        write(out/'FAILED.json',{'pending':pending,'error':type(e).__name__,'message':str(e),'certificate':getattr(e,'certificate',None),'completed_cases':len(rows),**counts,'seconds':time.monotonic()-started});raise
    finally:write(out/'source_access.json',src.accessed);write(out/'comparison_access.json',accessed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','anchors','incumbent','output'):p.add_argument(n)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
