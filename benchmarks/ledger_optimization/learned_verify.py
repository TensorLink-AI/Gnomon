"""Independent062 audit of temporal training labels, tree weights and block objectives."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import numpy as np
from pathlib import Path
from .validation_verify import block_objective,metric,MODELS,avg,encoded

ARMS=('global_cv','block_cv','legacy_ledger','recent_ledger','lifetime_ledger','learned_ledger')


def audit(original,warm,control,incumbent,memory,recent,lifetime,directory):
    original,warm,control,incumbent,memory,recent,lifetime,directory=map(Path,(original,warm,control,incumbent,memory,recent,lifetime,directory));here=Path(__file__).parent
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
    lifetime_rows=load(lifetime,'broad-lifetime-memory-061.json',('electricity:','pedestrian:'))
    def index(rows):
        out={(r['series_id'],r['origin']):r for r in rows};check(len(out)==len(rows),'Unique identities');return out
    added=load(memory,'broad-memory-forecasts-059.json',('raw/',))
    added_meta=load(memory,'broad-memory-forecasts-059.json',('contexts.json',))[0]
    identities=read(here/'evidence/broad-memory-identity-057.json')['selection']
    added_ids={d+':'+r['series_name'] for d,part in identities.items() for r in part['memory_training']}
    check(len(added)==len(added_meta)==525 and {r['series_id'] for r in added}==added_ids,'All named memory cases')
    check(not(added_ids & {r['series_id'] for r in scored}),'No extra scored identity')
    check(all(r['role']=='additional_memory_training' and r['included_in_scored_denominator'] is False for r in added),'Training-only source roles')
    raw=index(scored+early+added);oldmeta=index(oldmeta+added_meta);anchors=index(anchors);old=index(old);recent=index(recent_rows);lifetime=index(lifetime_rows)
    contexts=read(directory/'contexts.json');meta=index(contexts);check(len(raw)==len(meta)==1066 and set(meta)==set(raw),'Full expanded cohort')
    for key,row in raw.items():
        context=meta[key];source=oldmeta[key]
        for field in ('series_id','origin','last_target','outcome_recorded_at','domain','features'):check(context[field]==source[field],'Original predecision context')
        check('actual' not in context and 'scores' not in context,'No production outcome in profile')
    def leaf(tree, features):
        node = 0; depth = 0
        while tree['children_left'][node] != -1:
            check(depth < 4, 'tree depth and acyclic traversal')
            j = tree['feature'][node]
            check(0 <= j < 12 and math.isfinite(tree['threshold'][node]), 'valid split')
            node = tree['children_left'][node] if features[j] <= tree['threshold'][node] else tree['children_right'][node]
            check(0 <= node < len(tree['feature']), 'tree child index'); depth += 1
        check(tree['children_right'][node] == -1, 'leaf structure')
        return node
    forests={}
    def forest(name,domain,origin):
        if name in forests:return forests[name]
        record=read(directory/name);now=datetime.fromisoformat(origin)
        pool=sorted([r for r in contexts if r['domain']==domain and datetime.fromisoformat(r['origin'])<now and datetime.fromisoformat(r['last_target'])<=now and datetime.fromisoformat(r['outcome_recorded_at'])<=now],key=lambda r:(r['origin'],r['series_id']))
        refs=[{'series_id':r['series_id'],'origin':r['origin']} for r in pool]
        check(record['eligible']==refs,'All and only visible historical labels')
        check(record['domain']==domain and record['origin']==record['effective_source_as_of']==record['effective_recorded_as_of']==origin,'Forest effective cutoffs')
        ready=len(pool)>=32 and len({r['origin'] for r in pool})>=3
        check(record['ready']==ready,'Minimum training support')
        if not ready:forests[name]=(record,[],[]);return forests[name]
        check(record['features']==[r['features'] for r in pool],'Predecision training features')
        settings=dict(n_estimators=64,max_depth=4,min_samples_leaf=8,max_features=1.,bootstrap=False,random_state=17,n_jobs=1,criterion='squared_error')
        check(record['settings']==settings and len(record['trees'])==64,'Frozen estimator settings')
        for r,observed in zip(pool,record['targets'],strict=True):
            past=raw[r['series_id'],r['origin']];residual=[]
            for m in MODELS:
                cv=avg([metric(f['point'],f['actual']) for f in past['folds'][m]])
                near(past['cv'][m],cv)
                residual.append(metric(past['point'][m],past['actual'])-cv)
            center=avg(residual)
            for value,delta in zip(observed,residual,strict=True):near(value,delta-center)
            near(math.fsum(observed),0.)
        check(record['input_sha256']==encoded({'features':record['features'],'targets':record['targets']}),'Forest input hash')
        groups=[];features=np.asarray(record['features'],dtype=np.float32).tolist()
        for tree in record['trees']:
            sizes=[len(tree[k]) for k in ('children_left','children_right','feature','threshold')]
            check(len(set(sizes))==1 and sizes[0]>0,'Tree array structure')
            membership={}
            for i,x in enumerate(features):membership.setdefault(leaf(tree,x),[]).append(i)
            check(all(len(g)>=8 for g in membership.values()),'Minimum leaf support')
            groups.append(membership)
        forests[name]=(record,groups,pool);return forests[name]
    results=[];fits=iterations=0
    for source in sorted(scored,key=lambda r:(r['origin'],r['series_id'])):
        key=source['series_id'],source['origin'];now=datetime.fromisoformat(key[1]);current=meta[key]
        row=read(directory/f'{key[0]}-{source["round"]:02d}.json');ret=row['retrieval']
        expected_ref=f'forest-{current["domain"]}-{source["round"]:02d}.json'
        check(row['forest_ref']==expected_ref,'One forest per domain/origin')
        record,groups,pool=forest(expected_ref,current['domain'],key[1]);ready=record['ready']
        check(ret['ready']==ready and ret['current_features']==current['features'],'Query contains only current predecision features')
        relevance=[0.]*len(pool);leaves=[];x=np.asarray(current['features'],dtype=np.float32).tolist()
        if ready:
            for tree,group in zip(record['trees'],groups,strict=True):
                node=leaf(tree,x);leaves.append(node);support=group[node]
                for i in support:relevance[i]+=1/(64*len(support))
            check(leaves==ret['leaves'],'Independent query tree traversal')
            check(len(ret['record_weights'])==len(relevance),'All historical weights retained')
            for a,b in zip(ret['record_weights'],relevance,strict=True):near(a,b,1e-12)
            near(math.fsum(relevance),1.,1e-10);check(min(relevance)>=0,'Nonnegative evidence weights')
            for j,v in enumerate(ret['predicted_contrast']):near(v,math.fsum(w*t[j] for w,t in zip(relevance,record['targets'],strict=True)))
            near(ret['effective_records'],1/math.fsum(w*w for w in relevance))
        else:check(ret['record_weights']==[] and ret['leaves']==[],'Unsupported cold start has no inferred evidence')
        refs=[(r['series_id'],r['origin']) for r,w in zip(pool,relevance,strict=True) if w>0]
        positive=[w for w in relevance if w>0]
        near(row['added_memory_mass'],math.fsum(w for r,w in zip(pool,relevance,strict=True) if r['series_id'] in added_ids))
        check(row['selected_records']==len(refs),'Every positive record retained')
        near(row['effective_records'],ret.get('effective_records',0.))
        pairs=[{'point':{m:source['folds'][m][i]['point'] for m in MODELS},'actual':source['folds'][MODELS[0]][i]['actual']} for i in range(3)]
        training=pairs+[{'point':raw[k]['point'],'actual':raw[k]['actual']} for k in refs] if ret['ready'] else pairs
        masses=[1/6]*3+[.5*w for w in positive] if ret['ready'] else [1/3]*3;anchor=anchors[key]['control_fit']['weights'];fitted=row['fit'];weights=fitted['weights']
        check(fitted['input_sha256']==encoded({'pairs':training,'masses':masses,'anchor':anchor}),'Causal training provenance')
        check(fitted['anchor']==anchor and fitted['success'],'Fixed anchor and solver acceptance')
        for block in weights:near(math.fsum(block),1.,1e-10);check(min(block)>=0,'Simplex')
        value,gap=block_objective(weights,training,masses,anchor);near(value,fitted['objective']);near(gap,fitted['convex_gap_bound'],1e-8);check(gap<=1e-5+1e-10,'Certificate')
        check(row['actual']==source['actual'] and row['round']==source['round'],'Task unchanged')
        for arm in ('global_cv','block_cv','legacy_ledger'):check(row['point'][arm]==old[key]['point'][{'legacy_ledger':'block_ledger'}.get(arm,arm)],'Comparator byte-identical')
        check(row['point']['recent_ledger']==recent[key]['point']['expanded_ledger'],'Recent-memory comparator unchanged')
        check(row['point']['lifetime_ledger']==lifetime[key]['point']['lifetime_ledger'],'Lifetime comparator unchanged')
        for h,p in enumerate(row['point']['learned_ledger']):near(p,math.expm1(math.fsum(weights[h//6][j]*math.log1p(source['point'][m][h]) for j,m in enumerate(MODELS))))
        for arm in ARMS:near(row['scores'][arm],metric(row['point'][arm],source['actual']))
        if ret['ready']:fits+=1;iterations+=fitted['iterations']
        else:check(row['point']['learned_ledger']==row['point']['block_cv'],'Cold start fallback')
        results.append({k:row[k] for k in ('series_id','round','scores','fallback','added_memory_mass','selected_records','effective_records')})
    report=read(directory/'report.json');check(report['rows']==results and len(results)==416,'Full result denominator')
    def summary(observed,rows):
        means={a:avg([r['scores'][a] for r in rows]) for a in ARMS}
        check(observed['cases']==len(rows),'Aggregate count')
        for a,v in means.items():near(observed['mean_rmsle'][a],v)
        for a in ARMS[:-1]:near(observed['reduction'][a],1-means['learned_ledger']/means[a])
    summary(report['overall'],results)
    for d in ('electricity','pedestrian'):summary(report['domains'][d],[r for r in results if r['series_id'].startswith(d+':')])
    reductions=report['overall']['reduction'];gate=reductions['block_cv']>=.2 and reductions['global_cv']>=.2 and reductions['legacy_ledger']>0 and reductions['recent_ledger']>0 and reductions['lifetime_ledger']>0 and all(v>0 for d in report['domains'].values() for v in d['reduction'].values())
    check(report['development_gate_passed']==gate,'All comparator gates')
    check(report['weight_fits']==fits and report['iterations']==iterations,'Fit costs')
    check(report['api_calls']==report['additional_forecast_computations']==0 and report['inherited_forecast_computations']==25584,'Forecast costs')
    check(report['added_evidence_forecast_computations']==12600 and report['original_evidence_forecast_computations']==12984 and report['added_evidence_estimator_fits']==6300,'Separated historical cost')
    check(report['cases_using_added_memory']==sum(r['added_memory_mass']>0 for r in results),'Added-memory usage')
    for output_name,row_name in [('mean_added_memory_mass','added_memory_mass'),('mean_selected_records','selected_records'),('mean_effective_records','effective_records')]:near(report[output_name],avg([r[row_name] for r in results]))
    trained=sum(record['ready'] for record,_,_ in forests.values())
    check(report['evidence_forest_fits']==trained and report['evidence_trees']==trained*64,'Forest training costs')
    output={'checks':checks,'failures':0,'cases':416,'weight_fits':fits,'report_sha256':sha(directory/'report.json'),'verifier_sha256':sha(Path(__file__)),
        'evidence_forest_fits':trained,'scope':'Independent maturity filters, scalar centered RMSLE labels, saved-tree traversal, historical/query leaf weights, training hashes, objective certificates, forecasts, scores, fixed comparators and costs. Does not independently implement optimal tree splitting. Reused development data, not statistical confirmation.'}
    target=directory/'verification.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(output,indent=2)+'\n');return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('original','warm','control','incumbent','memory','recent','lifetime','directory'):p.add_argument(n)
    print(json.dumps(audit(**vars(p.parse_args())),indent=2))
