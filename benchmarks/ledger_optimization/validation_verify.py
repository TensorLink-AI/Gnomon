"""Independent arithmetic, causal evidence and paired-resampling audit for054."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

MODELS = ('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')
ARMS = ('global_cv','block_cv','block_ledger')
DOMAINS = ('electricity','pedestrian')


def avg(x):return math.fsum(x)/len(x)
def metric(p,y):return math.sqrt(avg([(math.log1p(a)-math.log1p(b))**2 for a,b in zip(p,y,strict=True)]))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def encoded(x):return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()


def block_objective(w,pairs,masses,anchor):
    loss = .01/4*math.fsum((w[b][j]-anchor[j])**2 for b in range(4) for j in range(6))
    gradient = [[.02/4*(w[b][j]-anchor[j]) for j in range(6)] for b in range(4)]
    for pair,mass in zip(pairs,masses,strict=True):
        logs = [[math.log1p(pair['point'][m][h]) for m in MODELS] for h in range(24)]
        error = [math.fsum(w[h//6][j]*logs[h][j] for j in range(6))-math.log1p(pair['actual'][h]) for h in range(24)]
        norm = math.sqrt(avg([e*e for e in error])+1e-12); loss += mass*norm
        for b in range(4):
            for j in range(6):gradient[b][j] += mass*math.fsum(logs[h][j]*error[h] for h in range(b*6,(b+1)*6))/(24*norm)
    gap = math.fsum(math.fsum(w[b][j]*gradient[b][j] for j in range(6))-min(gradient[b]) for b in range(4))
    return loss,gap


def global_objective(w,pairs):
    value = 1e-6*math.fsum((v-1/6)**2 for v in w); gradient = [2e-6*(v-1/6) for v in w]; omitted = 0.
    for pair in pairs:
        x = [[math.log1p(pair['point'][m][h]) for m in MODELS] for h in range(24)]
        error = [math.fsum(w[j]*x[h][j] for j in range(6))-math.log1p(pair['actual'][h]) for h in range(24)]
        norm = math.sqrt(avg([e*e for e in error]));value += norm/3
        if norm <= 1e-8:omitted += norm/3
        else:
            for j in range(6):gradient[j] += math.fsum(x[h][j]*error[h] for h in range(24))/(72*norm)
    return value,math.fsum(w[j]*gradient[j] for j in range(6))-min(gradient)+omitted


def audit(source,directory):
    source,directory = map(Path,(source,directory));here = Path(__file__).parent
    if (directory/'verification.json').exists():raise FileExistsError('Retain existing audit')
    read = lambda p:json.loads(p.read_text());checks=0
    def check(ok,message):
        nonlocal checks
        checks += 1
        if not ok:raise AssertionError(message)
    def near(a,b,tol=1e-9):check(math.isclose(a,b,rel_tol=tol,abs_tol=tol),f'{a} != {b}')
    manifest = read(directory/'manifest.json');receipt = read(here/'evidence/broad-validation-source-053.json')
    check(sha(here/'evidence/broad-validation-source-053.json') == manifest['source_receipt_sha256'],'Source receipt')
    for name,h in receipt['files'].items():check(sha(source/name) == h,'Frozen source file')
    for name,h in manifest['code_sha256'].items():check(sha(here/name) == h,'Frozen implementation')
    spans=read(source/'scored-spans.json');warm=read(source/'warmup-spans.json');tasks=read(source/'warmup-boundaries.json')
    rawrows=[read(p) for p in sorted((directory/'raw').glob('*.json'))]
    raw={(r['series_id'],r['origin']):r for r in rawrows}
    check(len(rawrows) == len(raw) == 533,'Complete unique raw cohort')
    expected={(s,i) for s in spans for i in range(26)}|{(t['series_id'],t['round']) for t in tasks if t['ready']}
    check({(r['series_id'],r['round']) for r in rawrows} == expected,'No removed or extra cases')
    check(read(directory/'excluded-warmup.json') == [t for t in tasks if not t['ready']],'Exact unavailable warmup cohorts')
    contexts=read(directory/'contexts.json');meta={(r['series_id'],r['origin']):r for r in contexts}
    check(len(contexts) == len(meta) == 533 and set(meta) == set(raw),'Context identity')
    for row in rawrows:
        key=row['series_id'],row['origin'];r=row['round'];span=(warm if r<0 else spans)[key[0]];i=r+8 if r<0 else r
        stop=730+168*i;observed=span['values'][stop-730:stop];actual=span['values'][stop:stop+24]
        start=datetime.fromisoformat(span['start_label']).replace(tzinfo=timezone.utc)
        check(row['history_indices'] == [stop-730,stop] and row['target_indices'] == [stop,stop+24],'Raw slices')
        check(row['origin'] == (start+timedelta(hours=stop)).isoformat() and row['last_target'] == (start+timedelta(hours=stop+24)).isoformat(),'Raw dates')
        check(row['history_sha256'] == encoded(observed) and row['actual'] == actual,'Exact history and targets')
        for m in MODELS:
            point=row['point'][m]
            check(len(point)==24 and all(math.isfinite(v) and v>=0 for v in point),'Valid point contract')
            near(row['scores'][m],metric(point,actual))
            for f,end in zip(row['folds'][m],(658,682,706),strict=True):
                check(f['end']==end and f['actual']==observed[end:end+24],'CV actuals')
                near(f['rmsle'],metric(f['point'],f['actual']))
            near(row['cv'][m],avg([f['rmsle'] for f in row['folds'][m]]))
            if m in ('daily','weekly','weekly_mean'):
                for end,pred in [(730,point)]+[(f['end'],f['point']) for f in row['folds'][m]]:
                    history=observed[:end]
                    want=[history[-24+h] for h in range(24)] if m=='daily' else [history[-168+h] for h in range(24)] if m=='weekly' else [avg([history[-168*k+h] for k in (1,2,3)]) for h in range(24)]
                    for a,b in zip(pred,want,strict=True):near(a,b)
        z=[math.log1p(v) for v in observed];mu=avg(z)
        feats=[math.log1p(row['cv'][m]) for m in MODELS]+[mu,math.sqrt(avg([(v-mu)**2 for v in z])),sum(v==0 for v in observed)/730,
            avg(z[-168:])-avg(z[-336:-168]),avg([abs(z[h]-z[h-24]) for h in range(24,730)]),avg([abs(z[h]-z[h-168]) for h in range(168,730)])]
        for a,b in zip(meta[key]['features'],feats,strict=True):near(a,b)
        check(meta[key]['outcome_recorded_at']==row['last_target'] and meta[key]['domain']==key[0].split(':')[0],'Availability assumption')
        check('scores' not in meta[key] and 'actual' not in meta[key],'No outcome in context')
    cases=[read(p) for p in sorted((directory/'cases').glob('*.json'))];check(len(cases)==416,'Full scored cohort')
    checked=[];fits=iterations=refinements=0
    for case in sorted(cases,key=lambda r:(r['origin'],r['series_id'])):
        key=case['series_id'],case['origin'];row=raw[key];now=datetime.fromisoformat(key[1]);current=meta[key]
        check(case['round']==row['round'] and case['actual']==row['actual'],'Mixture task identity')
        pool=[e for e in contexts if e['domain']==current['domain'] and datetime.fromisoformat(e['origin'])<now
            and datetime.fromisoformat(e['last_target'])<=now and datetime.fromisoformat(e['outcome_recorded_at'])<=now]
        latest=sorted({e['origin'] for e in pool})[-8:];pool=sorted([e for e in pool if e['origin'] in latest],key=lambda e:(e['origin'],e['series_id']))
        retrieval=case['retrieval'];ready=len(pool)>=16 and len(latest)>=3
        check(retrieval['ready']==ready and retrieval['distinct_origins']==len(latest),'Cold start criterion')
        if pool:
            locations=[avg([e['features'][j] for e in pool]) for j in range(12)]
            scales=[max(.1,math.sqrt(avg([(e['features'][j]-locations[j])**2 for e in pool]))) for j in range(12)]
            for a,b in zip(retrieval['location'],locations,strict=True):near(a,b)
            for a,b in zip(retrieval['scale'],scales,strict=True):near(a,b)
            distances=[math.fsum(((e['features'][j]-current['features'][j])/scales[j])**2 for j in range(12)) for e in pool]
            check([(e['series_id'],e['origin']) for e in retrieval['candidates']]==[(e['series_id'],e['origin']) for e in pool],'All eligible candidates')
            for a,b in zip(retrieval['candidates'],distances,strict=True):near(a['distance'],b)
            wanted=sorted(zip(pool,distances),key=lambda pair:(pair[1],pair[0]['origin'],pair[0]['series_id']))[:16] if ready else []
        else:wanted=[]
        refs=[(e['series_id'],e['origin']) for e,_ in wanted]
        check([(e['series_id'],e['origin']) for e in retrieval['selected']]==refs,'Exact nearest contexts')
        check(case['current_features']==current['features'],'Current features')
        pairs=[{'point':{m:row['folds'][m][i]['point'] for m in MODELS},'actual':row['folds'][MODELS[0]][i]['actual']} for i in range(3)]
        anchor=case['fits']['global_cv'];w=anchor['weights'];near(math.fsum(w),1.,1e-10);check(min(w)>=0,'Simplex anchor')
        check(anchor['input_sha256']==encoded({'pairs':pairs,'masses':[1/3]*3}),'Anchor training provenance')
        value,gap=global_objective(w,pairs);near(value,anchor['objective']);near(gap,anchor['convex_gap_bound'],1e-8)
        check(anchor['success'] and gap<=1e-5+1e-10,'Global certificate');fits+=1;iterations+=anchor['iterations']
        for arm in ('block_cv','block_ledger'):
            fitted=case['fits'][arm];weights=fitted['weights']
            for block in weights:near(math.fsum(block),1.,1e-10);check(min(block)>=0,'Block simplex')
            use_history=arm=='block_ledger' and ready
            training=pairs+[{'point':raw[k]['point'],'actual':raw[k]['actual']} for k in refs] if use_history else pairs
            masses=[1/6]*3+[.5/16]*16 if use_history else [1/3]*3
            check(fitted['input_sha256']==encoded({'pairs':training,'masses':masses,'anchor':w}),'Exact causal fit inputs')
            check(fitted['anchor']==w,'Same current CV anchor')
            value,gap=block_objective(weights,training,masses,w);near(value,fitted['objective']);near(gap,fitted['convex_gap_bound'],1e-8)
            check(fitted['success'] and gap<=1e-5+1e-10,'Block certificate')
            initial,_=block_objective([w]*4,training,masses,w);near(initial,fitted['initial_objective']);check(value<=initial+1e-8,'Objective improvement')
            if arm=='block_cv' or ready:
                fits+=1;iterations+=fitted['iterations'];refinements+=len(fitted['certificate_refinements'])
        check((case['fallback'] is None)==ready,'Fallback disclosure')
        if not ready:check(case['fits']['block_ledger']==case['fits']['block_cv'] and case['point']['block_ledger']==case['point']['block_cv'],'Exact control fallback')
        for arm in ARMS:
            weights=case['fits'][arm]['weights'];expected_point=[]
            for h in range(24):
                v=weights if arm=='global_cv' else weights[h//6]
                expected_point.append(math.expm1(math.fsum(v[j]*math.log1p(row['point'][m][h]) for j,m in enumerate(MODELS))))
            for a,b in zip(case['point'][arm],expected_point,strict=True):near(a,b)
            near(case['scores'][arm],metric(expected_point,row['actual']))
        checked.append({k:case[k] for k in ('series_id','round','scores','fallback')})
    report=read(directory/'report.json');check(report['rows']==checked,'Report row completeness')
    def aggregate(observed,rows):
        check(observed['cases']==len(rows),'Aggregation count')
        means={a:avg([r['scores'][a] for r in rows]) for a in ARMS}
        for a in ARMS:near(observed['mean_rmsle'][a],means[a])
        near(observed['primary_reduction'],1-means['block_ledger']/means['block_cv'])
        near(observed['reduction_vs_global_guard'],1-means['block_ledger']/means['global_cv'])
    aggregate(report['overall'],checked)
    for d in DOMAINS:aggregate(report['domains'][d],[r for r in checked if r['series_id'].startswith(d+':')])
    with np.load(directory/'bootstrap.npz') as bundle:
        si,oi,red=(bundle[n] for n in ('series_indices','origin_indices','reductions'))
    rng=np.random.default_rng(20260914);expected_si=rng.integers(0,8,size=(10000,2,8));starts=rng.integers(0,26,size=(10000,2,7))
    expected_oi=np.asarray([[[((int(v)+j)%26) for v in starts[r,d] for j in range(4)][:26] for d in range(2)] for r in range(10000)])
    check(np.array_equal(si,expected_si) and np.array_equal(oi,expected_oi),'Frozen random draws and block geometry')
    matrix=np.empty((2,8,26,3));index={(r['series_id'],r['round']):r for r in checked}
    for d,domain in enumerate(DOMAINS):
        names=sorted({r['series_id'] for r in checked if r['series_id'].startswith(domain+':')})
        check(report['uncertainty']['series_order'][domain]==names,'Bootstrap identity order')
        for s,name in enumerate(names):
            for t in range(26):matrix[d,s,t]=[index[name,t]['scores'][a] for a in ARMS]
    audited=[]
    for r in range(10000):
        total=np.zeros(3)
        for d in range(2):
            weights=np.outer(np.bincount(si[r,d],minlength=8),np.bincount(oi[r,d],minlength=26))/(2*8*26)
            total += (matrix[d]*weights[...,None]).sum(axis=(0,1))
        rr=1-total[2]/total[:2];audited.append(rr)
        for a,b in zip(red[r],rr,strict=True):near(float(a),float(b),1e-11)
    intervals=np.quantile(audited,[.025,.975],axis=0)
    for i,arm in enumerate(ARMS[:2]):
        for a,b in zip(report['uncertainty']['interval_95'][arm],intervals[:,i],strict=True):near(a,float(b))
    gate=(report['overall']['primary_reduction']>=.2 and report['overall']['reduction_vs_global_guard']>=.2 and bool((intervals[0]>0).all())
        and all(v['primary_reduction']>0 and v['reduction_vs_global_guard']>0 for v in report['domains'].values()))
    check(report['validation_gate_passed']==gate and report['goal_achieved'] is False,'Validation not goal completion')
    check(report['weight_fits']==report['weight_fits_started']==fits and report['completed_weight_iterations']==iterations,'Weight costs')
    check(report['forecast_computations']==12792 and report['estimator_fits']==6396 and report['api_calls']==0,'Shared forecast costs')
    result={'checks':checks,'failures':0,'raw_cases':533,'scored_cases':416,'weight_fits':fits,'certificate_refinements':refinements,
        'report_sha256':sha(directory/'report.json'),'verifier_sha256':sha(__file__),
        'scope':'Independent scalar scores/features/simple baselines, causal retrieval, training hashes, convex fit certificates, every derived point, costs, and all paired bootstrap replicates. ML estimators not refitted; frozen causal code and exact history hashes audited.'}
    (directory/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('directory')
    print(json.dumps(audit(**vars(p.parse_args())),indent=2))
