"""Frozen062 outcome-supervised retrieval with the unchanged block RMSLE objective."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean
import time
from .learned_context import retrieve,train_at
from .intraday_ensemble import fit,combine
from .validation_run import cv_pairs
from .broad_screen import rmsle

ARMS=('global_cv','block_cv','legacy_ledger','recent_ledger','lifetime_ledger','learned_ledger')


def summarize(rows):
    means={a:mean(r['scores'][a] for r in rows) for a in ARMS}
    return {'cases':len(rows),'mean_rmsle':means,'reduction':{a:1-means['learned_ledger']/means[a] for a in ARMS[:-1]}}


def merge_contexts(original,added,raw_keys,allowed_ids):
    fields=('series_id','origin','last_target','outcome_recorded_at','domain','features')
    project=lambda rows:[{k:r[k] for k in fields} for r in rows]
    original,added=project(original),project(added)
    keys=lambda rows:[(r['series_id'],r['origin']) for r in rows]
    old_keys,new_keys=keys(original),keys(added)
    if len(set(old_keys))!=len(old_keys) or len(set(new_keys))!=len(new_keys) or set(old_keys)&set(new_keys):raise ValueError('Overlapping or duplicated memory episode')
    if {r['series_id'] for r in added}!=set(allowed_ids) or set(allowed_ids)&{r['series_id'] for r in original}:raise ValueError('Unapproved memory series')
    if set(old_keys+new_keys)!=set(raw_keys):raise ValueError('Context/execution identity mismatch')
    if any(r['domain']!=r['series_id'].split(':')[0] for r in original+added):raise ValueError('Domain identity mismatch')
    return original+added


def run(original,warm,control,incumbent,memory,recent,lifetime,output):
    original,warm,control,incumbent,memory,recent,lifetime,output=map(Path,(original,warm,control,incumbent,memory,recent,lifetime,output));here=Path(__file__).parent
    receipts={}
    def load(root,name,prefixes):
        path=here/'evidence'/name;receipts[name]=hashlib.sha256(path.read_bytes()).hexdigest();receipt=json.loads(path.read_text());rows=[]
        for filename,h in receipt['files'].items():
            if not filename.startswith(prefixes):continue
            raw=(root/filename).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Changed source: '+filename)
            rows.append(json.loads(raw))
        return rows
    scored=load(original,'broad-screen-038.json',('electricity:','pedestrian:'))
    early=load(warm,'broad-warm-screen-043.json',('warmup-electricity:','warmup-pedestrian:'))
    episodes=load(warm,'broad-warm-screen-043.json',('episodes.json',))[0]
    anchors=load(control,'broad-ensemble-045.json',('electricity:','pedestrian:'))
    old=load(incumbent,'broad-intraday-050.json',('electricity:','pedestrian:'))
    recent_rows=load(recent,'broad-memory-breadth-060.json',('electricity:','pedestrian:'))
    lifetime_rows=load(lifetime,'broad-lifetime-memory-061.json',('electricity:','pedestrian:'))
    if len(lifetime_rows)!=416:raise ValueError('Full lifetime comparator required')
    if len(recent_rows)!=416:raise ValueError('Full recent-memory comparator required')
    if tuple(map(len,(scored,early,episodes,anchors,old)))!=(416,125,541,416,416):raise ValueError('Original development cohort required')
    def index(rows):
        values={(r['series_id'],r['origin']):r for r in rows}
        if len(values)!=len(rows):raise ValueError('Duplicate task identity')
        return values
    added=load(memory,'broad-memory-forecasts-059.json',('raw/',))
    added_context=load(memory,'broad-memory-forecasts-059.json',('contexts.json',))[0]
    if len(added)!=525 or len(added_context)!=525:raise ValueError('All525 added episodes required')
    if any(r.get('role')!='additional_memory_training' or r.get('included_in_scored_denominator') is not False for r in added):raise ValueError('Added records must be training only')
    identity_path=here/'evidence/broad-memory-identity-057.json'
    receipts[identity_path.name]=hashlib.sha256(identity_path.read_bytes()).hexdigest()
    allowed=json.loads(identity_path.read_text())['selection']
    added_ids={d+':'+r['series_name'] for d,p in allowed.items() for r in p['memory_training']}
    if {r['series_id'] for r in added}!=added_ids or added_ids & {r['series_id'] for r in scored}:raise ValueError('Training/scored identity mismatch')
    raw=index(scored+early+added);anchors=index(anchors);old=index(old);recent=index(recent_rows);lifetime=index(lifetime_rows)
    contexts=merge_contexts(episodes,added_context,set(raw),added_ids)
    if len(raw)!=1066 or len(contexts)!=1066:raise ValueError('Full expanded history required')
    meta=index(contexts)
    output.mkdir(parents=True,exist_ok=False)
    def save(n,v):(output/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    save('manifest.json',{'protocol':'LEARNED_RETRIEVAL_062.md','code_sha256':{n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in
        ('LEARNED_RETRIEVAL_062.md','learned_screen.py','learned_context.py','conditional_risk.py','intraday_ensemble.py','validation_run.py','broad_screen.py')},
        'source_receipts_sha256':receipts,'packages':{n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn')},
        'original_development_only':True,'validation_or_final_access':False,'api_calls':0,'inherited_forecast_computations':25584,
        'original_evidence_forecast_computations':12984,'added_evidence_forecast_computations':12600,
        'additional_forecast_computations':0,'added_memory_series':sorted(added_ids),'contexts':1066})
    save('contexts.json',contexts);start=time.monotonic();cpu=time.process_time();rows=[];fits=iterations=0;pending=None;forests={};forest_fits=0
    try:
        for source in sorted(scored,key=lambda r:(r['origin'],r['series_id'])):
            key=source['series_id'],source['origin'];pending={'series_id':key[0],'round':source['round']}
            domain=meta[key]['domain'];forest_key=(domain,key[1]);forest_ref=f'forest-{domain}-{source["round"]:02d}.json'
            if forest_key not in forests:
                model,training=train_at(contexts,raw,domain,key[1]);forests[forest_key]=(model,training)
                if model is not None:forest_fits+=1
                save(forest_ref,training)
            model,training=forests[forest_key]
            retrieval=retrieve(model,training,meta[key]['features'])
            selected=[(r,w) for r,w in zip(training['eligible'],retrieval['record_weights']) if w>0]
            past=[raw[r['series_id'],r['origin']] for r,w in selected]
            pairs=cv_pairs(source);anchor=anchors[key]['control_fit']['weights']
            if retrieval['ready']:
                fits+=1;fitted=fit(pairs+[{'point':r['point'],'actual':r['actual']} for r in past],[1/6]*3+[.5*w for r,w in selected],anchor);iterations+=fitted['iterations']
            else:fitted=old[key]['fits']['block_cv']
            points={a:old[key]['point'][{'legacy_ledger':'block_ledger'}.get(a,a)] for a in ('global_cv','block_cv','legacy_ledger')}
            points['recent_ledger']=recent[key]['point']['expanded_ledger']
            points['lifetime_ledger']=lifetime[key]['point']['lifetime_ledger']
            points['learned_ledger']=combine(source['point'],fitted['weights'])
            result={**{k:source[k] for k in ('series_id','round','origin','last_target')},'retrieval':retrieval,'forest_ref':forest_ref,'fit':fitted,
                'point':points,'actual':source['actual'],'scores':{a:rmsle(p,source['actual']) for a,p in points.items()},
                'fallback':None if retrieval['ready'] else 'insufficient_visible_context',
                'added_memory_mass':sum(w for r,w in selected if r['series_id'] in added_ids),
                'selected_records':len(selected),'effective_records':retrieval.get('effective_records',0.)}
            save(f'{key[0]}-{source["round"]:02d}.json',result)
            rows.append({k:result[k] for k in ('series_id','round','scores','fallback','added_memory_mass','selected_records','effective_records')})
            save('status.json',{'completed_cases':len(rows),'weight_fits':fits,'iterations':iterations,'seconds':time.monotonic()-start})
        overall=summarize(rows);domains={d:summarize([r for r in rows if r['series_id'].startswith(d+':')]) for d in ('electricity','pedestrian')}
        passed=overall['reduction']['block_cv']>=.2 and overall['reduction']['global_cv']>=.2 and overall['reduction']['legacy_ledger']>0 and overall['reduction']['recent_ledger']>0 and overall['reduction']['lifetime_ledger']>0 and all(v>0 for d in domains.values() for v in d['reduction'].values())
        report={'overall':overall,'domains':domains,'rows':rows,'weight_fits':fits,'iterations':iterations,'seconds':time.monotonic()-start,
            'cpu_seconds':time.process_time()-cpu,'api_calls':0,'additional_forecast_computations':0,'inherited_forecast_computations':25584,
            'original_evidence_forecast_computations':12984,'added_evidence_forecast_computations':12600,
            'added_evidence_estimator_fits':6300,'evidence_forest_fits':forest_fits,'evidence_trees':forest_fits*64,
            'cases_using_added_memory':sum(r['added_memory_mass']>0 for r in rows),
            'mean_added_memory_mass':mean(r['added_memory_mass'] for r in rows),
            'mean_selected_records':mean(r['selected_records'] for r in rows),
            'mean_effective_records':mean(r['effective_records'] for r in rows),
            'development_gate_passed':passed,'limitation':'Original reused development cases only; no confirmatory interval or agent result.'}
        save('report.json',report);return report
    except BaseException as e:
        save('FAILED.json',{'task':pending,'error':type(e).__name__,'message':str(e),'certificate':getattr(e,'certificate',None),
            'completed_cases':len(rows),'evidence_forest_fits':forest_fits,'weight_fits_started':fits,'completed_iterations':iterations,'seconds':time.monotonic()-start})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('original','warm','control','incumbent','memory','recent','lifetime','output'):p.add_argument(n)
    r=run(**vars(p.parse_args()));print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
