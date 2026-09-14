"""Independently audit new historical cases before expanded-ledger comparison."""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import json
import math
from pathlib import Path

MODELS=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')


def audit(source,directory):
    source,directory=map(Path,(source,directory));here=Path(__file__).parent;read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=0
    def check(ok,message):
        nonlocal checks
        checks+=1
        if not ok:raise AssertionError(message)
    def avg(v):return math.fsum(v)/len(v)
    def near(a,b,tol=1e-9):check(math.isclose(a,b,rel_tol=tol,abs_tol=tol),f'{a} != {b}')
    def metric(p,y):return math.sqrt(avg([(math.log1p(a)-math.log1p(b))**2 for a,b in zip(p,y,strict=True)]))
    manifest=read(directory/'manifest.json');receipt=read(here/'evidence/broad-memory-source-058.json')
    check(sha(here/'evidence/broad-memory-source-058.json')==manifest['source_receipt_sha256'],'Source receipt hash')
    for name,h in receipt['files'].items():check(sha(source/name)==h,'Prepared source unchanged')
    for name,h in manifest['code_sha256'].items():check(sha(here/name)==h,'Frozen generator/code')
    spans={'main':read(source/'memory-spans.json'),'warm':read(source/'warmup-spans.json')};tasks=read(source/'warmup-boundaries.json')
    expected={(s,r) for s in spans['main'] for r in range(25)}|{(t['series_id'],t['round']) for t in tasks if t['ready']}
    rows=[read(p) for p in sorted((directory/'raw').glob('*.json'))];contexts=read(directory/'contexts.json');context={(r['series_id'],r['origin']):r for r in contexts}
    check(len(rows)==len(expected)==len(context)==len(contexts)==525,'Full unique cohort')
    check({(r['series_id'],r['round']) for r in rows}==expected,'Training tasks only, none removed')
    check(read(directory/'excluded-warmup.json')==[t for t in tasks if not t['ready']],'Exact warm-up exclusions')
    jobs=[read(p) for p in sorted((directory/'jobs').glob('*.json'))];jobmap={(r['series_id'],r['round']):r for r in jobs}
    check(len(jobs)==len(jobmap)==525 and set(jobmap)==expected,'Every task has cost record')
    for row in rows:
        series,round_number=row['series_id'],row['round'];warm=round_number<0;index=round_number+8 if warm else round_number
        span=spans['warm' if warm else 'main'][series];stop=730+168*index;history=span['values'][stop-730:stop];target=span['values'][stop:stop+24]
        start=datetime.fromisoformat(span['start_label']).replace(tzinfo=timezone.utc)
        check(row['history_indices']==[stop-730,stop] and row['target_indices']==[stop,stop+24],'Exact slices')
        check(len(history)==730 and len(target)==24 and all(v is not None for v in history+target),'Observed whole task')
        check(row['origin']==(start+timedelta(hours=stop)).isoformat() and row['last_target']==(start+timedelta(hours=stop+24)).isoformat(),'Dates and one-second phase')
        check(row['history_sha256']==hashlib.sha256(json.dumps(history,separators=(',',':')).encode()).hexdigest() and row['actual']==target,'Raw-source history/target identity')
        check(row['role']=='additional_memory_training' and row['included_in_scored_denominator'] is False,'Non-evaluation role')
        for m in MODELS:
            check(len(row['point'][m])==24 and all(math.isfinite(v) and v>=0 for v in row['point'][m]),'Forecast contract')
            near(row['scores'][m],metric(row['point'][m],target))
            for f,end in zip(row['folds'][m],(658,682,706),strict=True):
                check(f['end']==end and f['actual']==history[end:end+24],'CV lies inside observed history')
                near(f['rmsle'],metric(f['point'],f['actual']))
            near(row['cv'][m],avg([f['rmsle'] for f in row['folds'][m]]))
            if m in ('daily','weekly','weekly_mean'):
                for end,point in [(730,row['point'][m])]+[(f['end'],f['point']) for f in row['folds'][m]]:
                    h=history[:end]
                    expected_point=[h[-24+i] for i in range(24)] if m=='daily' else [h[-168+i] for i in range(24)] if m=='weekly' else [avg([h[-168*k+i] for k in (1,2,3)]) for i in range(24)]
                    for a,b in zip(point,expected_point,strict=True):near(a,b)
        c=context[series,row['origin']];z=[math.log1p(v) for v in history];mean=avg(z)
        x=[math.log1p(row['cv'][m]) for m in MODELS]+[mean,math.sqrt(avg([(v-mean)**2 for v in z])),sum(v==0 for v in history)/730,
            avg(z[-168:])-avg(z[-336:-168]),avg([abs(z[t]-z[t-24]) for t in range(24,730)]),avg([abs(z[t]-z[t-168]) for t in range(168,730)])]
        for a,b in zip(c['features'],x,strict=True):near(a,b)
        check(c['last_target']==c['outcome_recorded_at']==row['last_target'] and c['domain']==series.split(':')[0],'Explicit period-end availability')
        check('actual' not in c and 'scores' not in c,'No production outcomes in query features')
        cost=jobmap[series,round_number];check(cost['status']=='complete' and cost['forecast_computations_started']==24 and cost['estimator_fits_started']==12,'Per-case cost')
    done=read(directory/'COMPLETED.json')
    check(done['cases_completed']==done['cases_started']==525 and done['cases_failed']==done['unfinished_cases']==0,'Completion accounting')
    check(done['forecast_computations_started']==sum(j['forecast_computations_started'] for j in jobs)==12600,'All forecast calls')
    check(done['estimator_fits_started']==sum(j['estimator_fits_started'] for j in jobs)==6300,'All estimator fits')
    check(done['worker_cpu_accounting_complete'] is True and all('cpu_seconds' in j for j in jobs),'CPU accounting present')
    near(done['worker_cpu_seconds'],math.fsum(j['cpu_seconds'] for j in jobs))
    check(done['api_calls']==0 and manifest['validation_or_final_access'] is False,'No paid or protected data work')
    result={'checks':checks,'failures':0,'historical_cases':525,'forecast_computations':12600,'estimator_fits':6300,
        'verifier_sha256':sha(Path(__file__)),'completed_sha256':sha(directory/'COMPLETED.json'),
        'scope':'Independent simple forecasts, all CV/production scores, history/target hashes, dates, predecision features, role and cost completeness. ML estimators not refitted; frozen causal code verified.'}
    out=directory/'verification.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('directory');print(json.dumps(audit(**vars(p.parse_args())),indent=2))
