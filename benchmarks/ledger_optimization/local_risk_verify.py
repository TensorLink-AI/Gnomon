"""Independent081tree evidence, per-lead convex weights and temporal audit."""
from datetime import datetime,timedelta
import hashlib,json,math,time
from pathlib import Path
import sys
import numpy as np

NAMES=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')
RECIPE={'n_estimators':32,'max_depth':4,'min_samples_leaf':24,'max_features':1.,'bootstrap':False,'random_state':17,'n_jobs':1,'criterion':'squared_error'}


def verify(root,source,anchors,incumbent):
    root,source,anchors,incumbent=map(Path,(root,source,anchors,incumbent));here=Path(__file__).parent;start=time.monotonic();checks=0
    def check(v,m):
        nonlocal checks
        checks+=1
        if not v:raise AssertionError(m)
    def near(a,b):check(np.allclose(a,b,atol=1e-9,rtol=1e-8),'Independent numerical disagreement')
    def load(p):return json.loads(Path(p).read_text())
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def dt(s):
        t=datetime.fromisoformat(s)
        if t.tzinfo is None:raise ValueError('Explicit time required')
        return t
    def risk(p,y):return math.sqrt(math.fsum((math.log1p(a)-math.log1p(b))**2 for a,b in zip(p,y,strict=True))/24)
    manifest=load(root/'manifest.json');sr=load(here/'evidence/configuration-search-066.json');ar=load(here/'evidence/broad-ensemble-045.json');ir=load(here/'evidence/search-ensemble-068.json');inventory=load(source/'SHA256SUMS.json');cache={}
    check(sha(source/'SHA256SUMS.json')==sr['files']['SHA256SUMS.json'],'066inventory identity')
    for n,k in (('configuration-search-066.json','source_receipt_sha256'),('broad-ensemble-045.json','anchor_receipt_sha256'),('search-ensemble-068.json','incumbent_receipt_sha256')):check(sha(here/'evidence'/n)==manifest[k],'Frozen receipt')
    for n,h in manifest['code_sha256'].items():check(sha(here/n)==h,'Frozen source code')
    def read(n):
        if n not in cache:check(sha(source/n)==inventory[n],'Source file hash');cache[n]=load(source/n)
        return cache[n]
    tasks=[read(n) for n in inventory if n.startswith('inputs/')];byid={t['task_id']:t for t in tasks};metadata=load(root/'metadata.json');check(len(tasks)==len(byid)==len(metadata)==541,'Complete541context population')
    for m in metadata:
        t=byid[m['task_id']];end=(dt(t['origin'])+timedelta(hours=24)).isoformat();check(m=={**{k:t[k] for k in ('task_id','series_id','domain','origin')},'last_target':end,'source_available_at':end,'recorded_at':end},'Exact nominal metadata')
    projection={}
    def fixed(tid):
        if tid in projection:return projection[tid]
        d=read(f'cases/{tid}/control/decision.json');groups=d['backtests'][:6];ids={n:g['config_id'] for n,g in zip(NAMES,groups,strict=True)};point={}
        for i,n in enumerate(NAMES):
            e=read(f'cases/{tid}/control/attempt-{4*i+4:02d}.json');f=read(e['cache_ref']);check(e['config_id']==f['request']['config_id']==ids[n] and f['request']['history_end']==730,'Executed fixed model configuration');point[n]=f['point']
        cv=[]
        for i,end in enumerate((658,682,706)):
            check(all(g['folds'][i]['end']==end and g['folds'][i]['actual']==groups[0]['folds'][i]['actual'] for g in groups),'Exact matched CV folds');cv.append({'point':{n:g['folds'][i]['point'] for n,g in zip(NAMES,groups,strict=True)},'actual':groups[0]['folds'][i]['actual']})
        projection[tid]=(point,cv,ids);return point,cv,ids
    def features(point,anchor):
        logs=np.array([[math.log1p(point[n][h]) for n in NAMES] for h in range(24)]);base=np.array([math.fsum(anchor[j]*logs[h,j] for j in range(6)) for h in range(24)]);mu=math.fsum(base)/24;scale=max(math.sqrt(math.fsum((b-mu)**2 for b in base)/24),.1)
        return np.array([[(v-mu)/scale for v in logs[h]]+[base[h],scale,math.sin(2*math.pi*h/24),math.cos(2*math.pi*h/24)] for h in range(24)],dtype=np.float32)
    def grams(point,actual):
        e=np.array([[math.log1p(point[n][h])-math.log1p(actual[h]) for n in NAMES] for h in range(24)]);return np.array([np.outer(row,row).ravel() for row in e])
    rows=[];forest_count=tree_count=weight_count=iterations=0;model_files=set()
    for task in sorted((t for t in tasks if t['round']>=0),key=lambda t:(dt(t['origin']),t['series_id'])):
        tid=task['task_id'];r=load(root/(tid+'.json'));point,cv,ids=fixed(tid);now=dt(task['origin']);a=r['anchor'];anchor=a['weights'];qx=features(point,anchor)
        for k in ('task_id','series_id','domain','origin','round'):check(r[k]==task[k],'Task identity')
        check(r['config_ids']==ids,'Current fixed configuration IDs');check(sha(anchors/a['file'])==ar['files'][a['file']],'045anchor identity');saved=load(anchors/a['file']);check(anchor==saved['control_fit']['weights'] and a['point']==saved['point']['cv_ensemble'],'Unchanged045baseline')
        eligible=sorted([m for m in metadata if m['domain']==task['domain'] and dt(m['origin'])<now and all(dt(m[k])<=now for k in ('last_target','source_available_at','recorded_at'))],key=lambda m:(dt(m['origin']),m['series_id']))
        for arm in ('control','ledger'):
            rec=r['records'][arm];check(sha(root/rec['model_file'])==rec['model_sha256'],'Saved forest identity');mf=load(root/rec['model_file']);model_files.add(rec['model_file']);f=mf['fit'];forest_count+=1;check(mf['task_id']==tid and mf['arm']==arm and mf['origin']==task['origin'] and f['anchor']==anchor,'Forest task/arm/anchor identity')
            pairs=list(cv);refs=[]
            for m in eligible if arm=='ledger' else []:
                pp,_,pi=fixed(m['task_id']);o=read('outcomes/'+m['task_id']+'.json');check(pi==ids and dt(m['origin']).timetz()==now.timetz(),'Same fixed configs and lead phase');check(all(dt(o[rk])==dt(m[mk]) for rk,mk in (('last_target','last_target'),('production_source_available_at','source_available_at'),('production_recorded_at','recorded_at'))),'Actual visibility matches eligible metadata');pairs.append({'point':pp,'actual':o['actual']});refs.append({**m,'config_ids':pi})
            q=[1/6]*3+[.5/len(eligible)]*len(eligible) if arm=='ledger' and eligible else [1/3]*3;check(mf['masses']==q and mf['historical_references']==refs,'All and only mature historical training');check(f['case_count']==len(pairs) and f['row_count']==24*len(pairs),'Training rows');check(f['input_sha256']==hashlib.sha256(json.dumps({'pairs':pairs,'masses':q,'anchor':anchor},separators=(',',':')).encode()).hexdigest(),'Full training hash')
            for k,v in RECIPE.items():check(f['recipe'][k]==v,'Frozen shared recipe')
            x=np.concatenate([features(p['point'],anchor) for p in pairs]);y=np.concatenate([grams(p['point'],p['actual']) for p in pairs]);w=np.repeat(np.array(q)/24,24);predictions=[];check(len(f['trees'])==f['tree_count']==32,'Full forest')
            for t in f['trees']:
                left=t['children_left'];right=t['children_right'];feature=t['feature'];threshold=t['threshold'];values=np.array(t['value'])[:,:,0];stack=[(0,np.arange(len(x)),0)];seen=set()
                while stack:
                    node,index,depth=stack.pop();seen.add(node);check(depth<=4 and len(index)==t['n_node_samples'][node],'Tree depth and sample count');near(w[index].sum(),t['weighted_n_node_samples'][node]);near(np.sum(w[index,None]*y[index],axis=0)/w[index].sum(),values[node])
                    if left[node]==-1:check(right[node]==-1 and len(index)>=24,'Minimum leaf support')
                    else:
                        check(0<=feature[node]<10 and 0<=left[node]<len(left) and 0<=right[node]<len(left),'Valid tree split');mask=x[index,feature[node]].astype(np.float64)<=threshold[node];stack.extend(((left[node],index[mask],depth+1),(right[node],index[~mask],depth+1)))
                check(seen==set(range(len(left))),'Every node supported');outputs=[]
                for row in qx:
                    node=0
                    while left[node]!=-1:node=left[node] if float(row[feature[node]])<=threshold[node] else right[node]
                    outputs.append(values[node])
                predictions.append(outputs);tree_count+=1
            matrices=np.mean(predictions,axis=0).reshape(24,6,6);near(rec['matrices'],matrices);check(np.linalg.eigvalsh(matrices).min()>=-1e-9,'Conditional PSD error matrices');forecast=[];check(len(rec['weight_fits'])==24,'All24weight certificates')
            for h,(g,certificate) in enumerate(zip(matrices,rec['weight_fits'],strict=True)):
                weights=np.array(certificate['weights']);aa=np.array(anchor);check(weights.shape==(6,) and np.isfinite(weights).all() and weights.min()>=0,'Simplex feasibility');near(weights.sum(),1.);near(certificate['anchor'],anchor);near(certificate['matrix'],g);value=float(weights@g@weights+.001*np.square(weights-aa).sum());initial=float(aa@g@aa);gradient=2*g@weights+.002*(weights-aa);gap=float(weights@gradient-gradient.min());near(certificate['objective'],value);near(certificate['initial_objective'],initial);near(certificate['convex_gap_bound'],gap);near(certificate['minimum_eigenvalue'],np.linalg.eigvalsh(g).min());check(certificate['success'] and gap<=1e-5+1e-9 and value<=initial+1e-8,'Independent convex weight certificate');iterations+=certificate['iterations'];weight_count+=1
                p=math.expm1(math.fsum(weights[j]*math.log1p(point[n][h]) for j,n in enumerate(NAMES)));forecast.append(p);check(min(point[n][h] for n in NAMES)-1e-9<=p<=max(point[n][h] for n in NAMES)+1e-9,'In-range combined forecast')
            near(rec['point'],forecast);check(r['point'][arm]==rec['point'],'Selected computed forecast')
        outcome=read('outcomes/'+tid+'.json');check(sha(incumbent/(tid+'.json'))==ir['files'][tid+'.json'],'068guard identity');prior=load(incumbent/(tid+'.json'));check(r['actual']==outcome['actual']==prior['actual'],'Unchanged scored actuals');check(r['point']['uncorrected045']==a['point'] and r['point']['strong050']==outcome['point']['strong_block_cv'] and r['point']['lifetime061']==outcome['point']['lifetime_ledger'] and r['point']['incumbent068']==prior['point']['ledger'],'All stronger comparisons unchanged')
        for name,p in r['point'].items():near(r['scores'][name],risk(p,r['actual']))
        rows.append(r)
        if len(rows)%16==0:print(json.dumps({'audited_cases':len(rows),'trees':tree_count,'seconds':time.monotonic()-start}),flush=True)
    report=load(root/'report.json');check(len(rows)==report['completed_cases']==416 and report['forests_started']==report['forests_completed']==forest_count==832,'Complete cases/forests');check(report['trees']==tree_count==26624 and report['weight_fits_started']==report['weight_fits_completed']==weight_count==19968 and report['weight_iterations']==iterations,'Exact tree/solve costs');check({str(p.relative_to(root)) for p in (root/'models').glob('*.json')}==model_files,'All model artifacts accounted')
    def summarize(group,saved):
        check(saved['cases']==len(group),'Summary denominator');means={a:math.fsum(risk(r['point'][a],r['actual']) for r in group)/len(group) for a in group[0]['point']}
        for a,v in means.items():near(saved['mean_rmsle'][a],v)
        for a,v in means.items():
            if a!='ledger':near(saved['ledger_reduction'][a],1-means['ledger']/v)
    summarize(rows,report['overall'])
    for d in ('electricity','pedestrian'):summarize([r for r in rows if r['domain']==d],report['domains'][d])
    summarize([r for r in rows if r['round']<8],report['phases']['early_0_7']);summarize([r for r in rows if r['round']>=8],report['phases']['later_8_25'])
    g=report['overall']['ledger_reduction'];guards=('control','strong050','lifetime061','incumbent068');passed=g['control']>=.2 and g['strong050']>=.2 and all(g[a]>0 for a in guards) and all(d['ledger_reduction'][a]>0 for d in report['domains'].values() for a in guards);check(report['development_gate_passed']==passed,'Frozen gate')
    for n,h in load(root/'source_access.json').items():check(inventory[n]==h and sha(source/n)==h,'All source accesses')
    for n,h in load(root/'comparison_access.json').items():
        label,name=n.split('/',1);folder,receipt=(anchors,ar) if label=='045' else (incumbent,ir);check(receipt['files'][name]==h and sha(folder/name)==h,'All comparison accesses')
    for k,v in {'inherited_forecast_computations':49616,'inherited045anchor_fits':416,'inherited068blend_fits':832,'inherited_search_surrogate_solves':11902,'inherited_search_logical_attempts_per_arm':31378}.items():check(manifest[k]==v,'Inherited costs')
    check(report['new_base_forecasts']==report['api_calls']==0 and manifest['validation_or_final_access'] is False,'No base/API/protected access')
    result={'checks':checks,'failures':0,'cases':416,'forests':forest_count,'trees':tree_count,'certified_weight_fits':weight_count,'seconds':time.monotonic()-start,'new_model_or_weight_fits':0,'api_calls':0,'verifier_sha256':sha(__file__),'scope':'All source/config/temporal cohorts, every tree node weighted36-output Gram mean and sample support, all query matrices/PSD checks,19,968independent simplex certificates, predictions/scores and costs.'};(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(verify(*sys.argv[1:]),indent=2))
