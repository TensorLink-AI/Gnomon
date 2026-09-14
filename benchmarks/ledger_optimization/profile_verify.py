"""Independent056 profile arithmetic, temporal retrieval and fitted-evidence audit."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from .validation_verify import block_objective,metric,MODELS,avg,encoded

ARMS=('global_cv','block_cv','legacy_ledger','profile_ledger')


def audit(original,warm,control,incumbent,directory):
    original,warm,control,incumbent,directory=map(Path,(original,warm,control,incumbent,directory));here=Path(__file__).parent
    checks=0
    def check(ok,message):
        nonlocal checks
        checks+=1
        if not ok:raise AssertionError(message)
    def near(a,b,tol=1e-9):check(math.isclose(a,b,abs_tol=tol,rel_tol=tol),f'{a} != {b}')
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=read(directory/'manifest.json')
    for name,h in manifest['code_sha256'].items():check(sha(here/name)==h,'Code frozen')
    for name,h in manifest['source_receipts_sha256'].items():check(sha(here/'evidence'/name)==h,'Receipt frozen')
    def load(root,receipt,prefix):
        rows=[]
        for n,h in read(here/'evidence'/receipt)['files'].items():
            if n.startswith(prefix):check(sha(root/n)==h,'Source frozen');rows.append(read(root/n))
        return rows
    scored=load(original,'broad-screen-038.json',('electricity:','pedestrian:'))
    early=load(warm,'broad-warm-screen-043.json',('warmup-electricity:','warmup-pedestrian:'))
    oldmeta=load(warm,'broad-warm-screen-043.json',('episodes.json',))[0]
    anchors=load(control,'broad-ensemble-045.json',('electricity:','pedestrian:'))
    old=load(incumbent,'broad-intraday-050.json',('electricity:','pedestrian:'))
    def index(rows):
        out={(r['series_id'],r['origin']):r for r in rows};check(len(out)==len(rows),'Unique identities');return out
    raw=index(scored+early);oldmeta=index(oldmeta);anchors=index(anchors);old=index(old)
    contexts=read(directory/'contexts.json');meta=index(contexts);check(len(raw)==len(meta)==541 and set(meta)==set(raw),'Full original cohort')
    for key,row in raw.items():
        context=meta[key];source=oldmeta[key]
        for field in ('series_id','origin','last_target','outcome_recorded_at','domain','features'):check(context[field]==source[field],'Original predecision context')
        check('actual' not in context and 'scores' not in context,'No production outcome in profile')
        expected=[]
        for model in MODELS:
            for block in range(4):
                errors=[math.log1p(f['point'][h])-math.log1p(f['actual'][h]) for f in row['folds'][model] for h in range(block*6,(block+1)*6)]
                expected.extend((avg(errors),math.sqrt(avg([e*e for e in errors]))))
        check(len(context['error_profile'])==48,'Profile shape')
        for a,b in zip(context['error_profile'],expected,strict=True):near(a,b)
    results=[];fits=iterations=0
    for source in sorted(scored,key=lambda r:(r['origin'],r['series_id'])):
        key=source['series_id'],source['origin'];now=datetime.fromisoformat(key[1]);current=meta[key]
        row=read(directory/f'{key[0]}-{source["round"]:02d}.json');ret=row['retrieval']
        pool=[r for r in contexts if r['domain']==current['domain'] and datetime.fromisoformat(r['origin'])<now and datetime.fromisoformat(r['last_target'])<=now and datetime.fromisoformat(r['outcome_recorded_at'])<=now]
        origins=sorted({r['origin'] for r in pool})[-8:];pool=sorted([r for r in pool if r['origin'] in origins],key=lambda r:(r['origin'],r['series_id']))
        check(ret['ready']==(len(pool)>=16 and len(origins)>=3),'Ready rule')
        check([(r['series_id'],r['origin']) for r in ret['candidates']]==[(r['series_id'],r['origin']) for r in pool],'All visible candidates')
        components=[]
        for field,n,location,scale in [('features',12,'location','scale'),('error_profile',48,'error_location','error_scale')]:
            center=[avg([r[field][j] for r in pool]) for j in range(n)]
            spread=[max(.1,math.sqrt(avg([(r[field][j]-center[j])**2 for r in pool]))) for j in range(n)]
            for a,b in zip(ret[location],center,strict=True):near(a,b)
            for a,b in zip(ret[scale],spread,strict=True):near(a,b)
            components.append([avg([((r[field][j]-current[field][j])/spread[j])**2 for j in range(n)]) for r in pool])
        distances=[]
        for c,a,b in zip(ret['candidates'],*components,strict=True):
            near(c['legacy_component'],a);near(c['legacy_distance'],12*a);near(c['profile_component'],b);near(c['distance'],a+b);distances.append(a+b)
        wanted=sorted(zip(pool,distances),key=lambda p:(p[1],p[0]['origin'],p[0]['series_id']))[:16] if ret['ready'] else []
        refs=[(r['series_id'],r['origin']) for r,_ in wanted]
        check([(r['series_id'],r['origin']) for r in ret['selected']]==refs,'Exact chosen neighbors')
        pairs=[{'point':{m:source['folds'][m][i]['point'] for m in MODELS},'actual':source['folds'][MODELS[0]][i]['actual']} for i in range(3)]
        training=pairs+[{'point':raw[k]['point'],'actual':raw[k]['actual']} for k in refs] if ret['ready'] else pairs
        masses=[1/6]*3+[.5/16]*16 if ret['ready'] else [1/3]*3;anchor=anchors[key]['control_fit']['weights'];fitted=row['fit'];weights=fitted['weights']
        check(fitted['input_sha256']==encoded({'pairs':training,'masses':masses,'anchor':anchor}),'Causal training provenance')
        check(fitted['anchor']==anchor and fitted['success'],'Fixed anchor and solver acceptance')
        for block in weights:near(math.fsum(block),1.,1e-10);check(min(block)>=0,'Simplex')
        value,gap=block_objective(weights,training,masses,anchor);near(value,fitted['objective']);near(gap,fitted['convex_gap_bound'],1e-8);check(gap<=1e-5+1e-10,'Certificate')
        check(row['actual']==source['actual'] and row['round']==source['round'],'Task unchanged')
        for arm in ARMS[:-1]:check(row['point'][arm]==old[key]['point'][{'legacy_ledger':'block_ledger'}.get(arm,arm)],'Comparator byte-identical')
        for h,p in enumerate(row['point']['profile_ledger']):near(p,math.expm1(math.fsum(weights[h//6][j]*math.log1p(source['point'][m][h]) for j,m in enumerate(MODELS))))
        for arm in ARMS:near(row['scores'][arm],metric(row['point'][arm],source['actual']))
        if ret['ready']:fits+=1;iterations+=fitted['iterations']
        else:check(row['point']['profile_ledger']==row['point']['block_cv'],'Cold start fallback')
        results.append({k:row[k] for k in ('series_id','round','scores','fallback')})
    report=read(directory/'report.json');check(report['rows']==results and len(results)==416,'Full result denominator')
    def summary(observed,rows):
        means={a:avg([r['scores'][a] for r in rows]) for a in ARMS}
        check(observed['cases']==len(rows),'Aggregate count')
        for a,v in means.items():near(observed['mean_rmsle'][a],v)
        for a in ARMS[:-1]:near(observed['reduction'][a],1-means['profile_ledger']/means[a])
    summary(report['overall'],results)
    for d in ('electricity','pedestrian'):summary(report['domains'][d],[r for r in results if r['series_id'].startswith(d+':')])
    reductions=report['overall']['reduction'];gate=reductions['block_cv']>=.2 and reductions['global_cv']>=.2 and reductions['legacy_ledger']>0 and all(v>0 for d in report['domains'].values() for v in d['reduction'].values())
    check(report['development_gate_passed']==gate,'All comparator gates')
    check(report['weight_fits']==fits and report['iterations']==iterations,'Fit costs')
    check(report['api_calls']==report['additional_forecast_computations']==0 and report['inherited_forecast_computations']==12984,'Forecast costs')
    output={'checks':checks,'failures':0,'cases':416,'weight_fits':fits,'report_sha256':sha(directory/'report.json'),'verifier_sha256':sha(Path(__file__)),
        'scope':'Independent scalar CV profiles, distance families/scales, all visible/selected records, training hashes, certificates, forecasts, scores, fixed comparators and costs. Reused development data, not statistical confirmation.'}
    target=directory/'verification.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(output,indent=2)+'\n');return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('original','warm','control','incumbent','directory'):p.add_argument(n)
    print(json.dumps(audit(**vars(p.parse_args())),indent=2))
