"""Hindsight-only diagnostic of available production forecasts from frozen066."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time


def score(p,y):
    if len(p)!=len(y) or not y or any(not math.isfinite(v) or v<0 for v in p+y):raise ValueError('Complete finite nonnegative pairs required')
    return math.sqrt(math.fsum((math.log1p(a)-math.log1p(b))**2 for a,b in zip(p,y,strict=True))/len(y))
def best(scores):
    if not scores or any(not math.isfinite(v) or v<0 for v in scores.values()):raise ValueError('Valid candidate scores required')
    v=min(scores.values());ties=sorted(k for k,s in scores.items() if s==v)
    return {'score':v,'selected':ties[0],'ties':ties}
def mean(values):return math.fsum(values)/len(values)


def case(task,decisions,produced):
    actual=task['actual'];original={r['config_id'] for r in decisions['control']['backtests'][:6]}
    if original!={r['config_id'] for r in decisions['ledger']['backtests'][:6]}:raise ValueError('Different starter configurations')
    scores={cid:score(point,actual) for cid,point in produced.items()};common={cid:scores[cid] for cid in original};arms={};choices={'original_six':best(common)}
    for arm,d in decisions.items():
        selected=d['selected']['config_id'];six=min(d['backtests'][:6],key=lambda b:b['cv_rmsle']);cid=six['config_id'];full=min(d['backtests'],key=lambda b:b['cv_rmsle'])
        if full['config_id']!=selected or d['point']!=produced[selected]:raise ValueError('Selection/production identity mismatch')
        if full['cv_rmsle']>six['cv_rmsle']:raise ValueError('Expanded CV minimum worsened')
        available={**common,selected:scores[selected]};choices[arm]=best(available)
        delta=scores[cid]-scores[selected]
        arms[arm]={'selected_config_id':selected,'selected_extra':selected not in original,'original_cv_selected':cid,
            'original_cv_rmsle':six['cv_rmsle'],'expanded_cv_rmsle':full['cv_rmsle'],'cv_improvement':six['cv_rmsle']-full['cv_rmsle'],
            'production_change_vs_original_cv':delta,'production_comparison':'improved' if delta>1e-12 else 'worsened' if delta < -1e-12 else 'tied',
            'produced_configurations':len(available),'tested_configurations':len(d['backtests']),'unobserved_production_configurations':len(d['backtests'])-len(available),
            'selected_score':scores[selected],'original_cv_selected_score':scores[cid],'selection_regret':scores[selected]-choices[arm]['score']}
    if arms['control']['original_cv_selected']!=arms['ledger']['original_cv_selected']:raise ValueError('Different original CV decisions')
    union=original|{a['selected_config_id'] for a in arms.values()}
    if set(produced)!=union:raise ValueError('Unselected or unavailable production forecast included')
    choices['union']=best(scores)
    final={'original_cv':arms['control']['original_cv_selected_score'],**{a:r['selected_score'] for a,r in arms.items()},
        **{a:score(task['point'][a],actual) for a in ('strong_block_cv','lifetime_ledger')},**{'oracle_'+a:r['score'] for a,r in choices.items()}}
    return {**{k:task[k] for k in ('task_id','series_id','round','origin')},'scores':final,'arms':arms,'hindsight_choices':choices,
        'production_scores':scores,'same_selection':arms['control']['selected_config_id']==arms['ledger']['selected_config_id'],
        'diagnostic_only':True,'uses_current_future_actuals':True}


def summary(rows):
    means={k:mean([r['scores'][k] for r in rows]) for k in rows[0]['scores']};guard=means['strong_block_cv']
    return {'cases':len(rows),'mean_rmsle':means,'same_selection_cases':sum(r['same_selection'] for r in rows),
        'ceiling_vs_strong_block_cv':{a:{'maximum_reduction':1-means[a]/guard,'twenty_percent_status':'impossible_for_these_saved_choices' if means[a]>.8*guard else 'not_ruled_out_not_established'} for a in means if a.startswith('oracle_')},
        'arms':{a:{'extra_selections':sum(r['arms'][a]['selected_extra'] for r in rows),'production_comparisons':dict(Counter(r['arms'][a]['production_comparison'] for r in rows)),
            'mean_cv_improvement':mean([r['arms'][a]['cv_improvement'] for r in rows]),'mean_selection_regret':mean([r['arms'][a]['selection_regret'] for r in rows]),
            'configuration_counts':dict(sorted(Counter(r['arms'][a]['selected_config_id'] for r in rows).items()))} for a in ('control','ledger')}}


def run(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=False);here=Path(__file__).parent;start=time.monotonic()
    receipt_path=here/'evidence/configuration-search-066.json';receipt=json.loads(receipt_path.read_text())
    index_path=source/'SHA256SUMS.json'
    if hashlib.sha256(index_path.read_bytes()).hexdigest()!=receipt['files']['SHA256SUMS.json']:raise ValueError('Changed source inventory')
    inventory=json.loads(index_path.read_text());accessed={}
    def read(name):
        raw=(source/name).read_bytes();h=hashlib.sha256(raw).hexdigest()
        if h!=inventory[name]:raise ValueError('Changed source '+name)
        accessed[name]=h;return json.loads(raw)
    rows=[]
    for name in sorted(inventory):
        if not name.startswith('outcomes/'):continue
        task=read(name)
        if task['warmup']:continue
        decisions={a:read(f'cases/{task["task_id"]}/{a}/decision.json') for a in ('control','ledger')};produced={}
        for arm,d in decisions.items():
            for i,b in enumerate(d['backtests'][:6]):
                event=read(f'cases/{task["task_id"]}/{arm}/attempt-{4*i+4:02d}.json');forecast=read(event['cache_ref'])
                if event['config_id']!=b['config_id'] or event['history_end']!=730:raise ValueError('Starter production binding')
                if b['config_id'] in produced and produced[b['config_id']]!=forecast['point']:raise ValueError('Shared forecasts differ')
                produced[b['config_id']]=forecast['point']
            final=read(d['final_cache_ref'])
            if final['request']['config_id']!=d['selected']['config_id']:raise ValueError('Selected config identity')
            produced[d['selected']['config_id']]=final['point']
        row=case(task,decisions,produced);rows.append(row);(output/(task['task_id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    if len(rows)!=416 or len({r['task_id'] for r in rows})!=416:raise ValueError('Full scored cohort required')
    report={'overall':summary(rows),'domains':{d:summary([r for r in rows if r['series_id'].startswith(d+':')]) for d in ('electricity','pedestrian')},
        'phases':{'early_0_7':summary([r for r in rows if r['round']<8]),'later_8_25':summary([r for r in rows if r['round']>=8])},
        'seconds':time.monotonic()-start,'provider_calls':0,'api_calls':0,'new_weight_fits':0,'diagnostic_only':True,'uses_current_future_actuals':True,
        'inherited_forecast_computations':49616,'limitation':'Saved-produced-forecast hindsight only. Not a bound on unexecuted configurations, new combinations, prospective policy skill or agent performance.'}
    manifest={'protocol':'CONFIGURATION_DIAGNOSTIC_067.md','code_sha256':{n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ('CONFIGURATION_DIAGNOSTIC_067.md','configuration_diagnostic.py')},
        'source_receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),'source_inventory_sha256':receipt['files']['SHA256SUMS.json'],'accessed_source_files':accessed,
        'validation_or_final_access':False,'policy_changed':False}
    for name,r in [('report.json',report),('manifest.json',manifest)]: (output/name).write_text(json.dumps(r,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('output');print(json.dumps(run(**vars(p.parse_args())),indent=2))
