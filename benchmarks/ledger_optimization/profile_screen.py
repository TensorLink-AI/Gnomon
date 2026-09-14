"""Original-development056 screen of observed error-profile evidence retrieval."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean
import time
from .error_profile import profile,retrieve_profile
from .intraday_ensemble import fit,combine
from .validation_run import cv_pairs
from .broad_screen import rmsle

ARMS=('global_cv','block_cv','legacy_ledger','profile_ledger')


def summarize(rows):
    means={a:mean(r['scores'][a] for r in rows) for a in ARMS}
    return {'cases':len(rows),'mean_rmsle':means,'reduction':{a:1-means['profile_ledger']/means[a] for a in ARMS[:-1]}}


def run(original,warm,control,incumbent,output):
    original,warm,control,incumbent,output=map(Path,(original,warm,control,incumbent,output));here=Path(__file__).parent
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
    if tuple(map(len,(scored,early,episodes,anchors,old)))!=(416,125,541,416,416):raise ValueError('Original development cohort required')
    def index(rows):
        values={(r['series_id'],r['origin']):r for r in rows}
        if len(values)!=len(rows):raise ValueError('Duplicate task identity')
        return values
    raw=index(scored+early);anchors=index(anchors);old=index(old);metadata=index(episodes)
    contexts=[{**{k:r[k] for k in ('series_id','origin','last_target','outcome_recorded_at','domain','features')},
        'error_profile':profile(raw[r['series_id'],r['origin']]['folds'])} for r in episodes]
    meta=index(contexts)
    output.mkdir(parents=True,exist_ok=False)
    def save(n,v):(output/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    save('manifest.json',{'protocol':'ERROR_PROFILE_056.md','code_sha256':{n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in
        ('ERROR_PROFILE_056.md','profile_screen.py','error_profile.py','context_ensemble.py','intraday_ensemble.py','validation_run.py','broad_screen.py')},
        'source_receipts_sha256':receipts,'packages':{n:importlib.metadata.version(n) for n in ('numpy','scipy')},
        'original_development_only':True,'validation_or_final_access':False,'api_calls':0,'inherited_forecast_computations':12984,
        'additional_forecast_computations':0,'feature_groups':[12,48],'distance_weights':[1,1]})
    save('contexts.json',contexts);start=time.monotonic();cpu=time.process_time();rows=[];fits=iterations=0;pending=None
    try:
        for source in sorted(scored,key=lambda r:(r['origin'],r['series_id'])):
            key=source['series_id'],source['origin'];pending={'series_id':key[0],'round':source['round']}
            retrieval=retrieve_profile(meta[key],contexts);past=[raw[r['series_id'],r['origin']] for r in retrieval['selected']]
            pairs=cv_pairs(source);anchor=anchors[key]['control_fit']['weights']
            if retrieval['ready']:
                if len(past)!=16:raise ValueError('Sixteen contexts required')
                fits+=1;fitted=fit(pairs+[{'point':r['point'],'actual':r['actual']} for r in past],[1/6]*3+[.5/16]*16,anchor);iterations+=fitted['iterations']
            else:fitted=old[key]['fits']['block_cv']
            points={a:old[key]['point'][{'legacy_ledger':'block_ledger'}.get(a,a)] for a in ARMS[:-1]}
            points['profile_ledger']=combine(source['point'],fitted['weights'])
            result={**{k:source[k] for k in ('series_id','round','origin','last_target')},'retrieval':retrieval,'fit':fitted,
                'point':points,'actual':source['actual'],'scores':{a:rmsle(p,source['actual']) for a,p in points.items()},
                'fallback':None if retrieval['ready'] else 'insufficient_visible_context'}
            save(f'{key[0]}-{source["round"]:02d}.json',result)
            rows.append({k:result[k] for k in ('series_id','round','scores','fallback')})
            save('status.json',{'completed_cases':len(rows),'weight_fits':fits,'iterations':iterations,'seconds':time.monotonic()-start})
        overall=summarize(rows);domains={d:summarize([r for r in rows if r['series_id'].startswith(d+':')]) for d in ('electricity','pedestrian')}
        passed=overall['reduction']['block_cv']>=.2 and overall['reduction']['global_cv']>=.2 and overall['reduction']['legacy_ledger']>0 and all(v>0 for d in domains.values() for v in d['reduction'].values())
        report={'overall':overall,'domains':domains,'rows':rows,'weight_fits':fits,'iterations':iterations,'seconds':time.monotonic()-start,
            'cpu_seconds':time.process_time()-cpu,'api_calls':0,'additional_forecast_computations':0,'inherited_forecast_computations':12984,
            'development_gate_passed':passed,'limitation':'Original reused development cases only; no confirmatory interval or agent result.'}
        save('report.json',report);return report
    except BaseException as e:
        save('FAILED.json',{'task':pending,'error':type(e).__name__,'message':str(e),'certificate':getattr(e,'certificate',None),
            'completed_cases':len(rows),'weight_fits_started':fits,'completed_iterations':iterations,'seconds':time.monotonic()-start})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('original','warm','control','incumbent','output'):p.add_argument(n)
    r=run(**vars(p.parse_args()));print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
