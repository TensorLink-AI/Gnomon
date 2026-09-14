"""Independent061 audit of mature lifetime retrieval and all comparator scores."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from .validation_verify import block_objective,metric,MODELS,avg,encoded

ARMS=('global_cv','block_cv','legacy_ledger','recent_ledger','lifetime_ledger')


def audit(original,warm,control,incumbent,memory,recent,directory):
    original,warm,control,incumbent,memory,recent,directory=map(Path,(original,warm,control,incumbent,memory,recent,directory));here=Path(__file__).parent
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
    recent_rows=load(recent,'broad-memory-breadth-060.json',('electricity:','pedestrian:'))
    def index(rows):
        out={(r['series_id'],r['origin']):r for r in rows};check(len(out)==len(rows),'Unique identities');return out
    added=load(memory,'broad-memory-forecasts-059.json',('raw/',))
    added_meta=load(memory,'broad-memory-forecasts-059.json',('contexts.json',))[0]
    identities=read(here/'evidence/broad-memory-identity-057.json')['selection']
    added_ids={d+':'+r['series_name'] for d,part in identities.items() for r in part['memory_training']}
    check(len(added)==len(added_meta)==525 and {r['series_id'] for r in added}==added_ids,'All named memory cases')
    check(not(added_ids & {r['series_id'] for r in scored}),'No extra scored identity')
    check(all(r['role']=='additional_memory_training' and r['included_in_scored_denominator'] is False for r in added),'Training-only source roles')
    raw=index(scored+early+added);oldmeta=index(oldmeta+added_meta);anchors=index(anchors);old=index(old);recent=index(recent_rows)
    contexts=read(directory/'contexts.json');meta=index(contexts);check(len(raw)==len(meta)==1066 and set(meta)==set(raw),'Full expanded cohort')
    for key,row in raw.items():
        context=meta[key];source=oldmeta[key]
        for field in ('series_id','origin','last_target','outcome_recorded_at','domain','features'):check(context[field]==source[field],'Original predecision context')
        check('actual' not in context and 'scores' not in context,'No production outcome in profile')
    results=[];fits=iterations=0
    for source in sorted(scored,key=lambda r:(r['origin'],r['series_id'])):
        key=source['series_id'],source['origin'];now=datetime.fromisoformat(key[1]);current=meta[key]
        row=read(directory/f'{key[0]}-{source["round"]:02d}.json');ret=row['retrieval']
        pool=[r for r in contexts if r['domain']==current['domain'] and datetime.fromisoformat(r['origin'])<now and datetime.fromisoformat(r['last_target'])<=now and datetime.fromisoformat(r['outcome_recorded_at'])<=now]
        origins=sorted({r['origin'] for r in pool});pool=sorted([r for r in pool if r['origin'] in origins],key=lambda r:(r['origin'],r['series_id']))
        check(ret['ready']==(len(pool)>=16 and len(origins)>=3),'Ready rule')
        check([(r['series_id'],r['origin']) for r in ret['candidates']]==[(r['series_id'],r['origin']) for r in pool],'All visible candidates')
        components=[]
        for field,n,location,scale in [('features',12,'location','scale')]:
            center=[avg([r[field][j] for r in pool]) for j in range(n)]
            spread=[max(.1,math.sqrt(avg([(r[field][j]-center[j])**2 for r in pool]))) for j in range(n)]
            for a,b in zip(ret[location],center,strict=True):near(a,b)
            for a,b in zip(ret[scale],spread,strict=True):near(a,b)
            components.append([avg([((r[field][j]-current[field][j])/spread[j])**2 for j in range(n)]) for r in pool])
        distances=[]
        for c,a in zip(ret['candidates'],components[0],strict=True):
            near(c['distance'],12*a);distances.append(12*a)
        wanted=sorted(zip(pool,distances),key=lambda p:(p[1],p[0]['origin'],p[0]['series_id']))[:16] if ret['ready'] else []
        refs=[(r['series_id'],r['origin']) for r,_ in wanted]
        check([(r['series_id'],r['origin']) for r in ret['selected']]==refs,'Exact chosen neighbors')
        check(row['added_memory_neighbors']==sum(k[0] in added_ids for k in refs),'Added-memory use count')
        old_window=set(origins[-8:])
        check(ret['candidates_outside_recent_eight']==sum(r['origin'] not in old_window for r in pool),'Old eligible records')
        check(ret['selected_outside_recent_eight']==row['older_memory_neighbors']==sum(k[1] not in old_window for k in refs),'Old selected records')
        check(ret['origin_window']=='all_visible','Lifetime-window disclosure')
        pairs=[{'point':{m:source['folds'][m][i]['point'] for m in MODELS},'actual':source['folds'][MODELS[0]][i]['actual']} for i in range(3)]
        training=pairs+[{'point':raw[k]['point'],'actual':raw[k]['actual']} for k in refs] if ret['ready'] else pairs
        masses=[1/6]*3+[.5/16]*16 if ret['ready'] else [1/3]*3;anchor=anchors[key]['control_fit']['weights'];fitted=row['fit'];weights=fitted['weights']
        check(fitted['input_sha256']==encoded({'pairs':training,'masses':masses,'anchor':anchor}),'Causal training provenance')
        check(fitted['anchor']==anchor and fitted['success'],'Fixed anchor and solver acceptance')
        for block in weights:near(math.fsum(block),1.,1e-10);check(min(block)>=0,'Simplex')
        value,gap=block_objective(weights,training,masses,anchor);near(value,fitted['objective']);near(gap,fitted['convex_gap_bound'],1e-8);check(gap<=1e-5+1e-10,'Certificate')
        check(row['actual']==source['actual'] and row['round']==source['round'],'Task unchanged')
        for arm in ('global_cv','block_cv','legacy_ledger'):check(row['point'][arm]==old[key]['point'][{'legacy_ledger':'block_ledger'}.get(arm,arm)],'Comparator byte-identical')
        check(row['point']['recent_ledger']==recent[key]['point']['expanded_ledger'],'Recent-memory comparator unchanged')
        for h,p in enumerate(row['point']['lifetime_ledger']):near(p,math.expm1(math.fsum(weights[h//6][j]*math.log1p(source['point'][m][h]) for j,m in enumerate(MODELS))))
        for arm in ARMS:near(row['scores'][arm],metric(row['point'][arm],source['actual']))
        if ret['ready']:fits+=1;iterations+=fitted['iterations']
        else:check(row['point']['lifetime_ledger']==row['point']['block_cv'],'Cold start fallback')
        results.append({k:row[k] for k in ('series_id','round','scores','fallback','added_memory_neighbors','older_memory_neighbors')})
    report=read(directory/'report.json');check(report['rows']==results and len(results)==416,'Full result denominator')
    def summary(observed,rows):
        means={a:avg([r['scores'][a] for r in rows]) for a in ARMS}
        check(observed['cases']==len(rows),'Aggregate count')
        for a,v in means.items():near(observed['mean_rmsle'][a],v)
        for a in ARMS[:-1]:near(observed['reduction'][a],1-means['lifetime_ledger']/means[a])
    summary(report['overall'],results)
    for d in ('electricity','pedestrian'):summary(report['domains'][d],[r for r in results if r['series_id'].startswith(d+':')])
    reductions=report['overall']['reduction'];gate=reductions['block_cv']>=.2 and reductions['global_cv']>=.2 and reductions['legacy_ledger']>0 and reductions['recent_ledger']>0 and all(v>0 for d in report['domains'].values() for v in d['reduction'].values())
    check(report['development_gate_passed']==gate,'All comparator gates')
    check(report['weight_fits']==fits and report['iterations']==iterations,'Fit costs')
    check(report['api_calls']==report['additional_forecast_computations']==0 and report['inherited_forecast_computations']==25584,'Forecast costs')
    check(report['added_evidence_forecast_computations']==12600 and report['original_evidence_forecast_computations']==12984 and report['added_evidence_estimator_fits']==6300,'Separated historical cost')
    check(report['cases_using_added_memory']==sum(r['added_memory_neighbors']>0 for r in results),'Added-memory case count')
    near(report['mean_added_neighbors'],avg([r['added_memory_neighbors'] for r in results]))
    check(report['cases_using_older_memory']==sum(r['older_memory_neighbors']>0 for r in results),'Older-memory case count')
    near(report['mean_older_neighbors'],avg([r['older_memory_neighbors'] for r in results]))
    output={'checks':checks,'failures':0,'cases':416,'weight_fits':fits,'report_sha256':sha(directory/'report.json'),'verifier_sha256':sha(Path(__file__)),
        'scope':'Independent expanded-cohort identities, scalar distance/scales, all visible/selected records, training hashes, certificates, forecasts, scores, fixed comparators and costs. Reused development data, not statistical confirmation.'}
    target=directory/'verification.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(output,indent=2)+'\n');return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('original','warm','control','incumbent','memory','recent','directory'):p.add_argument(n)
    print(json.dumps(audit(**vars(p.parse_args())),indent=2))
