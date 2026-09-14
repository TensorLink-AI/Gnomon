"""Independent078 temporal training, tree support, guard and scoring audit."""
from datetime import datetime,timedelta
import hashlib,json,math,time
from pathlib import Path
import sys
import numpy as np

NAMES=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')
RECIPE={'n_estimators':32,'max_depth':4,'min_samples_leaf':24,'max_features':1.,'bootstrap':False,'random_state':17,'n_jobs':1,'criterion':'squared_error'}


def split_mask(values,threshold):
    """Preserve the tree's double threshold when comparing float32 features."""
    return np.asarray(values,dtype=np.float64)<=float(threshold)


def verify(root,source,anchors,incumbent):
    root,source,anchors,incumbent=map(Path,(root,source,anchors,incumbent));here=Path(__file__).parent;started=time.monotonic();checks=0
    def load(p):return json.loads(Path(p).read_text())
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def check(v,m):
        nonlocal checks
        checks+=1
        if not v:raise AssertionError(m)
    def near(a,b):check(np.allclose(a,b,atol=1e-9,rtol=1e-8),'Independent numerical disagreement')
    def dt(s):
        t=datetime.fromisoformat(s)
        if t.tzinfo is None:raise ValueError('Explicit time required')
        return t
    def risk(p,y):return math.sqrt(math.fsum((math.log1p(a)-math.log1p(b))**2 for a,b in zip(p,y,strict=True))/24)
    manifest=load(root/'manifest.json');sr=load(here/'evidence/configuration-search-066.json');ar=load(here/'evidence/broad-ensemble-045.json');ir=load(here/'evidence/search-ensemble-068.json');inventory=load(source/'SHA256SUMS.json');cache={}
    check(sha(source/'SHA256SUMS.json')==sr['files']['SHA256SUMS.json'],'066inventory identity')
    for receipt_name,key in (('configuration-search-066.json','source_receipt_sha256'),('broad-ensemble-045.json','anchor_receipt_sha256'),('search-ensemble-068.json','incumbent_receipt_sha256')):check(sha(here/'evidence'/receipt_name)==manifest[key],'Frozen receipt identity')
    for n,h in manifest['code_sha256'].items():check(sha(here/n)==h,'Frozen experiment code')
    def read(n):
        if n not in cache:check(sha(source/n)==inventory[n],'Source file identity');cache[n]=load(source/n)
        return cache[n]
    tasks=[read(n) for n in inventory if n.startswith('inputs/')];taskmap={r['task_id']:r for r in tasks};metadata=load(root/'metadata.json');check(len(tasks)==len(taskmap)==len(metadata)==541,'Complete original population')
    for r in metadata:
        task=taskmap[r['task_id']];end=(dt(task['origin'])+timedelta(hours=24)).isoformat();check(r=={**{k:task[k] for k in ('task_id','series_id','domain','origin')},'last_target':end,'source_available_at':end,'recorded_at':end},'Exact nominal maturity metadata')
    projected={}
    def fixed(tid):
        if tid in projected:return projected[tid]
        decision=read(f'cases/{tid}/control/decision.json');groups=decision['backtests'][:6];ids={n:g['config_id'] for n,g in zip(NAMES,groups,strict=True)};points={}
        for i,n in enumerate(NAMES):
            e=read(f'cases/{tid}/control/attempt-{4*i+4:02d}.json');f=read(e['cache_ref']);check(e['config_id']==f['request']['config_id']==ids[n] and f['request']['history_end']==730,'Fixed configuration production binding');points[n]=f['point']
        pairs=[]
        for i,end in enumerate((658,682,706)):
            check(all(g['folds'][i]['end']==end and g['folds'][i]['actual']==groups[0]['folds'][i]['actual'] for g in groups),'Matched fixed CV folds')
            pairs.append({'point':{n:g['folds'][i]['point'] for n,g in zip(NAMES,groups,strict=True)},'actual':groups[0]['folds'][i]['actual']})
        projected[tid]=(points,pairs,ids);return points,pairs,ids
    def design(point,weights):
        logs=np.asarray([[math.log1p(point[n][h]) for n in NAMES] for h in range(24)]);base=np.array([math.fsum(weights[j]*logs[h,j] for j in range(6)) for h in range(24)]);location=math.fsum(base)/24;scale=max(math.sqrt(math.fsum((b-location)**2 for b in base)/24),.1)
        x=np.array([[(v-location)/scale for v in logs[h]]+[base[h],scale,math.sin(2*math.pi*h/24),math.cos(2*math.pi*h/24)] for h in range(24)],dtype=np.float32)
        return x,base
    model_files=set();forest_count=tree_count=guard_count=production_count=baseline_iterations=0
    def audit_forest(ref,tid,arm,stage,cutoff,current,weights,query):
        nonlocal forest_count,tree_count,guard_count,production_count
        p=root/ref['file'];check(sha(p)==ref['sha256'],'Forest artifact identity');r=load(p);f=r['fit'];model_files.add(ref['file']);forest_count+=1;guard_count+=stage=='guard';production_count+=stage=='production'
        check(r['task_id']==tid and r['arm']==arm and r['stage']==stage and dt(r['cutoff'])==dt(cutoff),'Stage/task binding');check(r['current_cv_ends']==([658,682] if stage=='guard' else [658,682,706]),'No held-back label in guard training')
        now=dt(cutoff);domain=taskmap[tid]['domain'];eligible=sorted([m for m in metadata if m['domain']==domain and dt(m['origin'])<now and all(dt(m[k])<=now for k in ('last_target','source_available_at','recorded_at'))],key=lambda m:(dt(m['origin']),m['series_id'])) if arm=='ledger' else []
        pairs=list(current);refs=[]
        for m in eligible:
            point,_,ids=fixed(m['task_id']);out=read('outcomes/'+m['task_id']+'.json');check(dt(m['origin']).timetz()==now.timetz(),'Lead-phase consistency');check(all(dt(out[rk])==dt(m[mk]) for rk,mk in (('last_target','last_target'),('production_source_available_at','source_available_at'),('production_recorded_at','recorded_at'))),'Actual source timing matches stage eligibility');pairs.append({'point':point,'actual':out['actual']});refs.append({**m,'config_ids':ids})
        q=([.5/len(current)]*len(current)+[.5/len(eligible)]*len(eligible)) if eligible else [1/len(current)]*len(current)
        check(r['historical_references']==refs and r['masses']==q,'All and only stage-visible outcomes with exact masses');check(ref['training_cases']==f['case_count']==len(pairs) and ref['historical_cases']==len(eligible) and f['row_count']==24*len(pairs),'Training row counts')
        check(f['weights']==weights,'Stage baseline identity');fingerprint=hashlib.sha256(json.dumps({'pairs':pairs,'masses':q,'weights':weights},separators=(',',':')).encode()).hexdigest();check(f['training_sha256']==fingerprint,'Full training input hash')
        for k,v in RECIPE.items():check(f['recipe'][k]==v,'Fixed shared forest recipe')
        xs=[];ys=[]
        for pair in pairs:
            x,b=design(pair['point'],weights);xs.append(x);ys.append(np.log1p(pair['actual'])-b)
        x=np.concatenate(xs);y=np.concatenate(ys);w=np.repeat(np.array(q)/24,24);qx,base=design(query,weights);predictions=[];check(len(f['trees'])==f['tree_count']==32,'All32trees retained')
        for t in f['trees']:
            left=t['children_left'];right=t['children_right'];feature=t['feature'];threshold=t['threshold'];value=np.array(t['value'])[:,0,0];visited=set();stack=[(0,np.arange(len(x)),0)]
            while stack:
                node,indices,depth=stack.pop();visited.add(node);check(depth<=4 and len(indices)==t['n_node_samples'][node],f'Tree depth and sample support: task={tid} arm={arm} stage={stage} tree={tree_count} node={node} depth={depth} reconstructed={len(indices)} saved={t["n_node_samples"][node]}');near(w[indices].sum(),t['weighted_n_node_samples'][node]);near(float(w[indices]@y[indices]/w[indices].sum()),value[node])
                if left[node]==-1:check(right[node]==-1 and len(indices)>=24,'Minimum leaf support')
                else:
                    check(0<=feature[node]<10 and 0<=left[node]<len(left) and 0<=right[node]<len(left),'Valid split indices');mask=split_mask(x[indices,feature[node]],threshold[node]);stack.extend(((left[node],indices[mask],depth+1),(right[node],indices[~mask],depth+1)))
            check(visited==set(range(len(left))),'Every saved node supported');node=np.zeros(24,dtype=int)
            for depth in range(5):
                active=np.array([left[n]!=-1 for n in node])
                if not active.any():break
                check(depth<4,'Prediction depth bound')
                for i in np.where(active)[0]:n=node[i];node[i]=left[n] if split_mask(qx[i,feature[n]],threshold[n]) else right[n]
            predictions.append(value[node]);tree_count+=1
        correction=np.mean(predictions,axis=0);raw=base+correction;point=np.expm1(np.maximum(raw,0.));return {'point':point,'base_point':np.expm1(base),'correction_log':correction,'raw_predicted_log':raw,'clipped_leads':np.where(raw<0)[0].tolist()}
    def check_application(saved,expected):
        for k,v in expected.items():
            if k=='clipped_leads':check(saved[k]==v,'Clipping disclosure')
            else:near(saved[k],v)
    rows=[];enabled={'control':0,'ledger':0}
    for task in sorted((r for r in tasks if r['round']>=0),key=lambda r:(dt(r['origin']),r['series_id'])):
        tid=task['task_id'];r=load(root/(tid+'.json'));point,cv,ids=fixed(tid);origin=task['origin'];cutoff=(dt(origin)-timedelta(hours=24)).isoformat();gb=r['guard_baseline'];weights=gb['weights'];baseline_iterations+=gb['iterations']
        for k in ('task_id','series_id','domain','origin','round'):check(r[k]==task[k],'Current task identity')
        check(r['guard_origin']==cutoff and r['current_config_ids']==ids,'Held-back origin and fixed configs')
        check(gb['input_sha256']==hashlib.sha256(json.dumps({'pairs':cv[:2],'masses':[.5,.5]},separators=(',',':')).encode()).hexdigest(),'Guard baseline excludes third-fold actuals');w=np.array(weights);check(w.shape==(6,) and np.isfinite(w).all() and w.min()>=0,'Baseline simplex');near(w.sum(),1.)
        value=1e-6*np.square(w-1/6).sum();initial=0.;gradient=2e-6*(w-1/6);omitted=0.
        for pair in cv[:2]:
            x=np.log1p(np.array([pair['point'][n] for n in NAMES]).T);y=np.log1p(pair['actual']);error=x@w-y;norm=math.sqrt(float(error@error)/24);value+=.5*norm;initial+=.5*math.sqrt(float(np.mean((x@np.full(6,1/6)-y)**2)))
            if norm<=1e-8:omitted+=.5*norm
            else:gradient+=x.T@error/(48*norm)
        gap=float(w@gradient-gradient.min()+omitted);near(gb['objective'],value);near(gb['initial_objective'],initial);near(gb['convex_gap_bound'],gap);check(gb['success'] and gap<=1e-5+1e-9 and value<=initial+1e-8,'Independent guard baseline certificate')
        a=r['anchor'];check(sha(anchors/a['file'])==ar['files'][a['file']],'Production045anchor identity');saved_a=load(anchors/a['file']);check(a['weights']==saved_a['control_fit']['weights'] and a['point']==saved_a['point']['cv_ensemble'],'Unchanged three-fold production baseline')
        for arm in ('control','ledger'):
            record=r['records'][arm];v=audit_forest(record['guard_model'],tid,arm,'guard',cutoff,cv[:2],weights,cv[2]['point']);check_application(record['guard_prediction'],v);decision=record['decision'];before=risk(record['guard_prediction']['base_point'],cv[2]['actual']);after=risk(record['guard_prediction']['point'],cv[2]['actual']);chosen=before-after>1e-12
            near(decision['baseline_rmsle'],before);near(decision['corrected_rmsle'],after);near(decision['improvement'],before-after);check(decision['correction_enabled']==chosen and decision['threshold']==1e-12 and decision['cause']==('lower_held_back_fold_rmsle' if chosen else 'no_validated_improvement'),'Held-back decision uses only its scores');enabled[arm]+=chosen
            if chosen:
                p=audit_forest(record['production_model'],tid,arm,'production',origin,cv,a['weights'],point);check_application(record['production_prediction'],p);check(r['point'][arm]==record['production_prediction']['point'],'Selected corrected forecast');near(record['production_prediction']['base_point'],a['point'])
            else:check(record['production_model'] is None and record['production_prediction'] is None and r['point'][arm]==a['point'],'Rejected correction preserves exact baseline without fitting')
        outcome=read('outcomes/'+tid+'.json');check(sha(incumbent/(tid+'.json'))==ir['files'][tid+'.json'],'068comparison identity');prior=load(incumbent/(tid+'.json'));check(r['actual']==outcome['actual']==prior['actual'],'Identical scored actuals');check(r['point']['uncorrected045']==a['point'] and r['point']['strong050']==outcome['point']['strong_block_cv'] and r['point']['lifetime061']==outcome['point']['lifetime_ledger'] and r['point']['incumbent068']==prior['point']['ledger'],'All stronger guards unchanged')
        for name,p in r['point'].items():near(r['scores'][name],risk(p,r['actual']))
        rows.append(r)
        if len(rows)%32==0:print(json.dumps({'audited_cases':len(rows),'trees':tree_count,'seconds':time.monotonic()-started}),flush=True)
    report=load(root/'report.json');check(len(rows)==report['completed_cases']==report['guard_baseline_fits']==416,'Complete scored cohort');check(report['guard_baseline_iterations']==baseline_iterations,'Baseline iterations');check(report['guard_forests']==guard_count==832 and report['production_forests']==production_count==sum(enabled.values()) and report['trees']==tree_count==32*forest_count,'Exact conditional forest costs');check(report['correction_enabled']==enabled,'Reported branch decisions');check({str(p.relative_to(root)) for p in (root/'models').glob('*.json')}==model_files,'No unreported model files')
    def summarize(group,saved):
        check(saved['cases']==len(group),'Summary denominator');means={a:math.fsum(risk(r['point'][a],r['actual']) for r in group)/len(group) for a in group[0]['point']}
        for a,v in means.items():near(saved['mean_rmsle'][a],v)
        for a,v in means.items():
            if a!='ledger':near(saved['ledger_reduction'][a],1-means['ledger']/v)
    summarize(rows,report['overall'])
    for d in ('electricity','pedestrian'):summarize([r for r in rows if r['domain']==d],report['domains'][d])
    summarize([r for r in rows if r['round']<8],report['phases']['early_0_7']);summarize([r for r in rows if r['round']>=8],report['phases']['later_8_25'])
    g=report['overall']['ledger_reduction'];guards=('control','strong050','lifetime061','incumbent068');passed=g['control']>=.2 and g['strong050']>=.2 and all(g[a]>0 for a in guards) and all(d['ledger_reduction'][a]>0 for d in report['domains'].values() for a in guards);check(report['development_gate_passed']==passed,'Frozen promotion gate')
    for n,h in load(root/'source_access.json').items():check(inventory[n]==h and sha(source/n)==h,'All accessed066source identity')
    for n,h in load(root/'comparison_access.json').items():
        label,name=n.split('/',1);folder,receipt=(anchors,ar) if label=='045' else (incumbent,ir);check(receipt['files'][name]==h and sha(folder/name)==h,'All accessed comparison identity')
    check(manifest['validation_or_final_access'] is False and report['new_base_forecasts']==report['api_calls']==0,'No protected evaluation or base/API calls')
    for k,v in {'inherited_forecast_computations':49616,'inherited045anchor_fits':416,'inherited068blend_fits':832,'inherited_search_surrogate_solves':11902,'inherited_search_logical_attempts_per_arm':31378}.items():check(manifest[k]==v,'Inherited costs preserved')
    result={'checks':checks,'failures':0,'cases':416,'guard_baseline_certificates':416,'forests':forest_count,'trees':tree_count,'seconds':time.monotonic()-started,'new_forest_fits':0,'new_base_forecasts':0,'api_calls':0,'verifier_sha256':sha(__file__),'scope':'All temporal stage cohorts/config bindings, independent guard baseline certificates, every saved tree node sample support and weighted mean, forest predictions, held-back branch decisions, current forecasts, scores and costs. Source forecasts reused from audited066.'}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(verify(*sys.argv[1:]),indent=2))
