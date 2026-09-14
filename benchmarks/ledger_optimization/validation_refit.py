"""Uniform055 numerical refit of all533 frozen054 raw cases; no new forecasts."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
import numpy as np

from .validation_run import digest,fit_case,uncertainty,ARMS,DOMAINS
from .broad_intraday import summarize
from .broad_screen import rmsle
from .certified_global import fit


def run(original,output):
    original,output=map(Path,(original,output));here=Path(__file__).parent
    read=lambda p:json.loads(p.read_text());receipt=read(here/'evidence/broad-validation-054-failed.json')
    for name,h in receipt['files'].items():
        if digest(original/name)!=h:raise ValueError('Frozen054 evidence changed: '+name)
    inherited=read(original/'manifest.json')
    for name,h in inherited['code_sha256'].items():
        if digest(here/name)!=h:raise ValueError('Locked implementation changed: '+name)
    output.mkdir(parents=True,exist_ok=False);(output/'cases').mkdir()
    shutil.copytree(original/'raw',output/'raw')
    for n in ('contexts.json','excluded-warmup.json'):shutil.copyfile(original/n,output/n)
    def save(name,value):(output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save('manifest.json',{**inherited,'numerical_amendment':'VALIDATION_REFIT_055.md','original_failed_run':str(original),
        'original_receipt_sha256':digest(here/'evidence/broad-validation-054-failed.json'),
        'code_sha256':{**inherited['code_sha256'],**{n:digest(here/n) for n in ('validation_refit.py','certified_global.py','VALIDATION_REFIT_055.md')}},
        'global_ftol':1e-14,'global_search_epsilon':1e-6,'global_certificate':'exact_objective_norm_dual_support',
        'additional_forecast_computations':0,'additional_estimator_fits':0,'inherited_forecast_computations':12792})
    rawrows=[read(p) for p in sorted((output/'raw').glob('*.json'))];raw={(r['series_id'],r['origin']):r for r in rawrows}
    contexts=read(output/'contexts.json');meta={(r['series_id'],r['origin']):r for r in contexts}
    if len(raw)!=len(rawrows) or len(raw)!=533 or set(meta)!=set(raw):raise ValueError('Full matched cohort required')
    rows=[];cost={'weight_fits':0,'weight_fits_started':0,'completed_weight_iterations':0};pending=None;start=time.monotonic();cpu=time.process_time()
    def charge(arm):cost['weight_fits_started']+=1;pending['arm']=arm
    def status(phase):save('status.json',{'phase':phase,'completed_raw_cases':533,'completed_scored_cases':len(rows),**cost,'seconds':time.monotonic()-start,'pending':pending})
    try:
        for row in sorted((r for r in rawrows if r['round']>=0),key=lambda r:(r['origin'],r['series_id'])):
            key=row['series_id'],row['origin'];now=datetime.fromisoformat(key[1]);pending={'series_id':key[0],'round':row['round']};status('refitting')
            proposal=fit_case({k:v for k,v in row.items() if k not in ('actual','scores')},meta[key],contexts,
                {k:v for k,v in raw.items() if datetime.fromisoformat(v['origin'])<now and datetime.fromisoformat(v['last_target'])<=now},fit,charge)
            cost['weight_fits']+=proposal['weight_fits']
            cost['completed_weight_iterations']+=sum(proposal['fits'][a]['iterations'] for a in (ARMS if proposal['fallback'] is None else ARMS[:2]))
            result={**{k:row[k] for k in ('series_id','round','origin','last_target')},**proposal,'actual':row['actual'],
                'scores':{a:rmsle(p,row['actual']) for a,p in proposal['point'].items()}}
            save(f'cases/{key[0]}-{row["round"]:02d}.json',result)
            rows.append({k:result[k] for k in ('series_id','round','scores','fallback')});status('refitting')
        if len(rows)!=416:raise ValueError('Full416-case denominator required')
        interval,si,oi,reductions=uncertainty(rows);np.savez_compressed(output/'bootstrap.npz',series_indices=si,origin_indices=oi,reductions=reductions)
        overall=summarize(rows);domains={d:summarize([r for r in rows if r['series_id'].startswith(d+':')]) for d in DOMAINS}
        passed=(overall['primary_reduction']>=.2 and overall['reduction_vs_global_guard']>=.2
            and all(x[0]>0 for x in interval['interval_95'].values())
            and all(v['primary_reduction']>0 and v['reduction_vs_global_guard']>0 for v in domains.values()))
        report={'overall':overall,'domains':domains,'uncertainty':interval,'validation_gate_passed':passed,'rows':rows,**cost,
            'cold_start_fallbacks':sum(r['fallback'] is not None for r in rows),
            'forecast_computations':12792,'estimator_fits':6396,'additional_forecast_computations':0,'additional_estimator_fits':0,
            'usable_warmup_cases':117,'unavailable_warmup_cases':11,'inherited_failed_run':receipt['failure'],
            'api_calls':0,'seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,'goal_achieved':False,
            'limitation':'Disjoint-series numerical validation after disclosed uniform numerical amendment; no agent/final evaluation.'}
        save('report.json',report);status('complete');return report
    except BaseException as e:
        save('FAILED.json',{'task':pending,'error':type(e).__name__,'message':str(e),'certificate':getattr(e,'certificate',None),**cost,
            'completed_scored_cases':len(rows),'additional_forecast_computations':0,'seconds':time.monotonic()-start})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('original');p.add_argument('output')
    r=run(**vars(p.parse_args()));print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
