"""Frozen074 chronological, executed-candidate memory strength experiment."""
import argparse
from collections import Counter
from datetime import datetime,timedelta
import hashlib
from itertools import groupby
import json
from pathlib import Path
from statistics import mean
import time
from scipy.optimize import minimize
from .ensemble_refinement import isolated
from .search_ensemble_run import Source,sha,write
from .search_ensemble import fit,combine
from .memory_strength import STRENGTHS,KEYS,blend_inputs,choose,risk,instant
from .lifetime_context import retrieve


def digest(value):return hashlib.sha256(json.dumps(value,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verified_rows(root,receipt,predicate):
    rows=[]
    for name,h in json.loads(Path(receipt).read_text())['files'].items():
        if predicate(name):
            p=Path(root)/name
            if sha(p)!=h:raise ValueError('Changed frozen evidence '+str(p))
            rows.append((json.loads(p.read_text()),str(p),h))
    return rows


def summary(rows):
    names=('control','ledger','fixed_half','strong_block_cv','lifetime_ledger','incumbent068')
    means={a:mean(r['scores'][a] for r in rows) for a in names}
    return {'cases':len(rows),'mean_rmsle':means,'ledger_reduction':{a:1-means['ledger']/means[a] for a in names if a!='ledger'}}


def run(source,anchors,incumbent,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'fits').mkdir();(out/'cases').mkdir();(out/'decisions').mkdir()
    here=Path(__file__).parent;src=Source(source);start=time.monotonic();cpu=time.process_time()
    counts={'logical_blend_requests':0,'physical_blend_fits':0,'cache_hits':0,'warm_anchor_fits':0,'blend_iterations':0,'anchor_iterations':0}
    fitcache={};trials=[];scored=[];completed=0;pending=None
    warm=isolated('benchmarks.ledger_optimization._warm_anchor074',here/'evidence_ensemble.py')
    def refined(*args,**kwargs):
        kwargs['options']={**kwargs['options'],'ftol':1e-12};return minimize(*args,**kwargs)
    warm.minimize=refined
    try:
        inputs=[src.read(n) for n in sorted(src.inventory) if n.startswith('inputs/')];bykey={(r['series_id'],r['origin']):r for r in inputs}
        if len(inputs)!=541 or len(bykey)!=541:raise ValueError('Expected541distinct inputs')
        contexts=[{**{k:r[k] for k in ('series_id','origin','features','domain')},'last_target':(instant(r['origin'])+timedelta(hours=24)).isoformat(),'outcome_recorded_at':(instant(r['origin'])+timedelta(hours=24)).isoformat()} for r in inputs]
        ai={(r['series_id'],r['origin']):{'weights':r['control_fit']['weights'],'file':p,'sha256':h} for r,p,h in verified_rows(anchors,here/'evidence/broad-ensemble-045.json',lambda n:n.startswith(('electricity:','pedestrian:')))}
        old={r['task_id']:r for r,_,_ in verified_rows(incumbent,here/'evidence/search-ensemble-068.json',lambda n:n.endswith('.json') and len(n)==69)}
        if len(ai)!=416 or len(old)!=416:raise ValueError('Incomplete frozen comparisons')
        write(out/'manifest.json',{'protocol':'MEMORY_STRENGTH_074.md','code_sha256':{n:sha(here/n) for n in ('MEMORY_STRENGTH_073.md','MEMORY_STRENGTH_074.md','memory_strength_run.py','memory_strength.py','search_ensemble.py','search_ensemble_run.py','lifetime_context.py','evidence_ensemble.py','ensemble_refinement.py')},'source_receipt_sha256':src.receipt_sha256,'anchor_receipt_sha256':sha(here/'evidence/broad-ensemble-045.json'),'incumbent_receipt_sha256':sha(here/'evidence/search-ensemble-068.json'),'runtime_kind':'standalone_numerical_proxy','api_calls':0,'new_provider_calls':0,'validation_or_final_access':False,'inherited_forecast_computations':49616,'inherited_anchor_fits':416,'inherited_search_surrogate_solves':11902,'inherited_search_logical_attempts_per_arm':31378,'nominal_period_end_availability':True,'candidate_recording_semantics':'Simulated decision origin; actual wall execution is this development run.'})
        write(out/'contexts.json',contexts)
        # Finish every decision in an equal-instant batch before exposing outcomes.
        ordered=sorted(inputs,key=lambda r:(instant(r['origin']),r['series_id']))
        for origin,batch in groupby(ordered,key=lambda r:instant(r['origin'])):
            decisions=[]
            for task in batch:
                tid=task['task_id'];pending={'task_id':tid,'stage':'anchor'};key=task['series_id'],task['origin']
                if key in ai:anchor_source=ai[key]
                else:
                    counts['warm_anchor_fits']+=1
                    pairs=src.current_pairs(tid,'control');pairs=[{'point':{m:v for m,v in p['point'].items() if m!='search_selected'},'actual':p['actual']} for p in pairs]
                    fitted=warm.fit(pairs,[1/3]*3);counts['anchor_iterations']+=fitted['iterations'];anchor_source={'weights':fitted['weights'],'fit':fitted,'pairs':pairs,'masses':[1/3]*3,'warmup':True}
                anchor=anchor_source['weights']+[0.];retrieval=retrieve(task,contexts);records={}
                for arm in ('control','ledger'):
                    pending={'task_id':tid,'arm':arm,'stage':'candidates'};current=src.current_pairs(tid,arm);points,ids=src.points(tid,arm);past=[];references=[]
                    neighbors=retrieval['selected'] if arm=='ledger' and retrieval['ready'] else []
                    for ref in neighbors:
                        previous=bykey[ref['series_id'],ref['origin']];history=src.read(f'outcomes/{previous["task_id"]}.json')
                        if not (instant(previous['origin'])<origin and instant(history['production_source_available_at'])<=origin and instant(history['production_recorded_at'])<=origin):raise ValueError('Unavailable historical evidence')
                        pp,pi=src.points(previous['task_id'],arm);past.append({'point':pp,'actual':history['actual']})
                        references.append({'task_id':previous['task_id'],'series_id':previous['series_id'],'origin':previous['origin'],'config_ids':pi,'source_available_at':history['production_source_available_at'],'recorded_at':history['production_recorded_at']})
                    selection=choose({**task,'arm':arm},trials if arm=='ledger' else [],neighbors)
                    candidates={}
                    for strength,label in zip(STRENGTHS,KEYS,strict=True):
                        training=blend_inputs(current,past,strength);binding={k:training[k] for k in ('pairs','masses')};binding['anchor']=anchor;fid=digest(binding)
                        counts['logical_blend_requests']+=1;hit=fid in fitcache
                        if hit:counts['cache_hits']+=1
                        else:
                            counts['physical_blend_fits']+=1;write(out/'attempt.json',{'pending':pending,'strength':strength,**counts,'completed_tasks':completed})
                            fitted=fit(training['pairs'],training['masses'],anchor);counts['blend_iterations']+=fitted['iterations'];fitcache[fid]=fitted
                            if fitted['input_sha256']!=fid:raise ValueError('Canonical training cache binding')
                            write(out/'fits'/(fid+'.json'),{'training':binding,'fit':fitted})
                        candidates[label]={'execution_id':digest({'task_id':tid,'arm':arm,'strength':strength}),'task_id':tid,'arm':arm,'requested_strength':strength,'effective_strength':training['effective_strength'],'forecast_recorded_at':task['origin'],'fit_ref':fid,'cache_hit':hit,'point':combine(points,fitcache[fid]['weights'])}
                    selected=next(k for k in KEYS if float(k)==selection['requested_strength'])
                    records[arm]={'selection':selection,'candidates':candidates,'selected_key':selected,'current_points':points,'current_config_ids':ids,'historical_references':references}
                decision={k:task[k] for k in ('task_id','series_id','origin','round','domain')};decision.update(records=records,retrieval=retrieval,anchor_source=anchor_source)
                write(out/'decisions'/(tid+'.json'),decision);decisions.append(decision)
            for decision in decisions:
                tid=decision['task_id'];outcome=src.read(f'outcomes/{tid}.json');actual=outcome['actual'];points={}
                for arm,record in decision['records'].items():
                    points[arm]=record['candidates'][record['selected_key']]['point']
                    trials.append({**{k:decision[k] for k in ('task_id','series_id','origin','domain')},'arm':arm,'last_target':outcome['last_target'],'source_available_at':outcome['production_source_available_at'],'recorded_at':outcome['production_recorded_at'],'actual':actual,'candidates':record['candidates']})
                points['fixed_half']=decision['records']['ledger']['candidates']['0.5']['point']
                if decision['round']>=0:
                    prior=old[tid]
                    if prior['actual']!=actual or any(max(abs(a-b) for a,b in zip(points[new],prior['point'][previous],strict=True))>1e-12 for new,previous in (('control','control'),('fixed_half','ledger'))):raise ValueError('Failed068forecast parity')
                    points.update({a:outcome['point'][a] for a in ('strong_block_cv','lifetime_ledger')});points['incumbent068']=prior['point']['ledger']
                row={**{k:decision[k] for k in ('task_id','series_id','origin','round','domain')},'actual':actual,'point':points,'scores':{a:risk(p,actual) for a,p in points.items()},'ledger_selected_strength':decision['records']['ledger']['selection']['requested_strength'],'trial_history_ready':decision['records']['ledger']['selection']['ready']}
                write(out/'cases'/(tid+'.json'),row);completed+=1
                if row['round']>=0:scored.append(row)
            write(out/'status.json',{'completed_tasks':completed,'completed_scored':len(scored),**counts,'seconds':time.monotonic()-start})
            print(json.dumps({'completed_tasks':completed,'completed_scored':len(scored),'seconds':time.monotonic()-start}),flush=True)
        overall=summary(scored);domains={d:summary([r for r in scored if r['domain']==d]) for d in ('electricity','pedestrian')};g=overall['ledger_reduction'];guards=('control','strong_block_cv','lifetime_ledger','incumbent068')
        report={'overall':overall,'domains':domains,'phases':{'early_0_7':summary([r for r in scored if r['round']<8]),'later_8_25':summary([r for r in scored if r['round']>=8])},'scored_strength_counts':dict(Counter(str(r['ledger_selected_strength']) for r in scored)),'mature_scored_cases':sum(r['trial_history_ready'] for r in scored),'completed_tasks':completed,'completed_scored':len(scored),**counts,'seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,'new_provider_calls':0,'api_calls':0,'development_gate_passed':g['control']>=.2 and g['strong_block_cv']>=.2 and all(g[a]>0 for a in guards) and all(d['ledger_reduction'][a]>0 for d in domains.values() for a in guards),'limitation':'Repeated-development numerical proxy; not a held-out or agent result.'}
        if completed!=541 or len(scored)!=416 or counts['logical_blend_requests']!=5410:raise ValueError('Incomplete experiment')
        write(out/'report.json',report);return report
    except BaseException as e:
        write(out/'FAILED.json',{'pending':pending,'error':type(e).__name__,'message':str(e),'certificate':getattr(e,'certificate',None),'completed_tasks':completed,**counts,'seconds':time.monotonic()-start});raise
    finally:write(out/'source_access.json',src.accessed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','anchors','incumbent','output'):p.add_argument(n)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
