"""Independent079raw errors, normal equations, chronology and output audit."""
from datetime import datetime,timedelta
import hashlib,json,math,time
from pathlib import Path
import sys
import numpy as np

NAMES=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')


def verify(root,source,anchors,incumbent):
    root,source,anchors,incumbent=map(Path,(root,source,anchors,incumbent));here=Path(__file__).parent;start=time.monotonic();checks=0
    def check(v,m):
        nonlocal checks
        checks+=1
        if not v:raise AssertionError(m)
    def near(a,b):check(np.allclose(a,b,atol=1e-9,rtol=1e-8),'Independent numeric disagreement')
    def load(p):return json.loads(Path(p).read_text())
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def dt(s):
        t=datetime.fromisoformat(s);check(t.tzinfo is not None,'Explicit timezone');return t
    def risk(p,y):return math.sqrt(math.fsum((math.log1p(a)-math.log1p(b))**2 for a,b in zip(p,y,strict=True))/24)
    def errors(p,y):return [[math.log1p(y[h])-math.log1p(p[n][h]) for h in range(24)] for n in NAMES]
    manifest=load(root/'manifest.json');receipts={k:load(here/'evidence'/n) for k,n in (('066','configuration-search-066.json'),('045','broad-ensemble-045.json'),('068','search-ensemble-068.json'))};inventory=load(source/'SHA256SUMS.json');cache={}
    check(sha(source/'SHA256SUMS.json')==receipts['066']['files']['SHA256SUMS.json'],'066inventory identity')
    for k,n in (('source_receipt_sha256','configuration-search-066.json'),('anchor_receipt_sha256','broad-ensemble-045.json'),('incumbent_receipt_sha256','search-ensemble-068.json')):check(sha(here/'evidence'/n)==manifest[k],'Frozen receipt')
    for n,h in manifest['code_sha256'].items():check(sha(here/n)==h,'Frozen candidate code')
    def read(n):
        if n not in cache:check(sha(source/n)==inventory[n],'Source file identity');cache[n]=load(source/n)
        return cache[n]
    tasks=[read(n) for n in inventory if n.startswith('inputs/')];byid={t['task_id']:t for t in tasks};meta=load(root/'metadata.json');check(len(tasks)==len(byid)==len(meta)==541,'Full original cohort')
    for m in meta:
        t=byid[m['task_id']];end=(dt(t['origin'])+timedelta(hours=24)).isoformat();check(m=={**{k:t[k] for k in ('task_id','series_id','domain','origin')},'last_target':end,'source_available_at':end,'recorded_at':end},'Nominal metadata')
    projection={}
    def fixed(tid):
        if tid in projection:return projection[tid]
        d=read(f'cases/{tid}/control/decision.json');groups=d['backtests'][:6];ids={n:g['config_id'] for n,g in zip(NAMES,groups,strict=True)};point={}
        for i,n in enumerate(NAMES):
            e=read(f'cases/{tid}/control/attempt-{4*i+4:02d}.json');f=read(e['cache_ref']);check(e['config_id']==f['request']['config_id']==ids[n] and f['request']['history_end']==730,'Fixed production identities');point[n]=f['point']
        cv=[]
        for i,end in enumerate((658,682,706)):
            check(all(g['folds'][i]['end']==end and g['folds'][i]['actual']==groups[0]['folds'][i]['actual'] for g in groups),'Matched CV ends/actuals')
            cv.append({'point':{n:g['folds'][i]['point'] for n,g in zip(NAMES,groups,strict=True)},'actual':groups[0]['folds'][i]['actual']})
        projection[tid]=(point,cv,ids);return point,cv,ids
    saved_history={}
    for p in (root/'transitions').glob('*.json'):
        r=load(p);m=r['metadata'];point,cv,ids=fixed(m['task_id']);outcome=read('outcomes/'+m['task_id']+'.json');check(p.stem==m['task_id'] and r['config_ids']==ids,'Historical fixed-config identity');check(r['previous_origin']==(dt(m['origin'])-timedelta(hours=24)).isoformat(),'Exact24-hour historical transition')
        for rk,mk in (('last_target','last_target'),('production_source_available_at','source_available_at'),('production_recorded_at','recorded_at')):check(dt(outcome[rk])==dt(m[mk]),'Historical source and recording availability')
        near(r['transition']['previous_error'],errors(cv[2]['point'],cv[2]['actual']));near(r['transition']['next_error'],errors(point,outcome['actual']));saved_history[p.stem]=r
    rows=[];clipped=dict.fromkeys(('control','ledger'),0);used=set();systems=0
    for task in sorted((t for t in tasks if t['round']>=0),key=lambda t:(dt(t['origin']),t['series_id'])):
        tid=task['task_id'];r=load(root/(tid+'.json'));point,cv,ids=fixed(tid);now=dt(task['origin']);last=errors(cv[2]['point'],cv[2]['actual']);curr=[{'previous_error':errors(cv[i]['point'],cv[i]['actual']),'next_error':errors(cv[i+1]['point'],cv[i+1]['actual'])} for i in range(2)]
        for k in ('task_id','series_id','domain','origin','round'):check(r[k]==task[k],'Scored identity')
        check(r['config_ids']==ids and r['current_transition_origins']==[(now-timedelta(hours=h)).isoformat() for h in (72,48,24)],'Current24-hour transition origins')
        for i in range(2):
            for k in ('previous_error','next_error'):near(r['current_transitions'][i][k],curr[i][k])
        near(r['last_cv_error'],last);a=r['anchor'];check(sha(anchors/a['file'])==receipts['045']['files'][a['file']],'045anchor identity');aa=load(anchors/a['file']);check(a['weights']==aa['control_fit']['weights'] and a['point']==aa['point']['cv_ensemble'],'Unchanged baseline')
        eligible=sorted([m for m in meta if m['domain']==task['domain'] and dt(m['origin'])<now and all(dt(m[k])<=now for k in ('last_target','source_available_at','recorded_at'))],key=lambda m:(dt(m['origin']),m['series_id']))
        for arm in ('control','ledger'):
            record=r['records'][arm];f=record['fit'];transitions=list(r['current_transitions']);expected=[]
            for m in eligible if arm=='ledger' else []:
                h=saved_history[m['task_id']];check(h['metadata']==m and h['config_ids']==ids and dt(m['origin']).timetz()==now.timetz(),'Matched historical domain/config/lead phase');name='transitions/'+m['task_id']+'.json';expected.append({'file':name,'sha256':sha(root/name),'metadata':m});transitions.append(h['transition']);used.add(m['task_id'])
            q=[.25,.25]+[.5/len(eligible)]*len(eligible) if arm=='ledger' and eligible else [.5,.5];check(record['historical_references']==expected and record['masses']==q,'All and only eligible transitions');check(f['case_count']==len(transitions) and f['rows_per_model']==24*len(transitions) and f['penalty']==.1,'Regression dimensions/penalty');check(f['input_sha256']==hashlib.sha256(json.dumps({'transitions':transitions,'masses':q},separators=(',',':')).encode()).hexdigest(),'Exact training hash');corrected=[]
            for j,n in enumerate(NAMES):
                s=f['systems'][n];previous=np.array([t['previous_error'][j] for t in transitions]);following=np.array([t['next_error'][j] for t in transitions]);mass=np.array(q)[:,None]/24
                sx=float(np.sum(mass*previous));sxx=float(np.sum(mass*previous*previous));sy=float(np.sum(mass*following));sxy=float(np.sum(mass*previous*following));gram=np.array([[1.,sx],[sx,sxx]]);matrix=gram+.1*np.eye(2);rhs=np.array([sy,sxy]);det=matrix[0,0]*matrix[1,1]-matrix[0,1]**2;theta=np.array([(matrix[1,1]*sy-matrix[0,1]*sxy)/det,(matrix[0,0]*sxy-matrix[0,1]*sy)/det]);residual=theta[0]+theta[1]*previous-following;objective=float(np.sum(mass*residual**2)+.1*(theta@theta));initial=float(np.sum(mass*following**2));gradient=2*(matrix@theta-rhs)
                near(s['gram'],gram);near(s['rhs'],rhs);near(s['regularized_matrix'],matrix);near(s['coefficients'],theta);near(s['objective'],objective);near(s['zero_objective'],initial);near(s['gradient'],gradient);near(s['eigenvalues'],np.linalg.eigvalsh(matrix));check(det>0 and np.linalg.eigvalsh(matrix).min()>=.1-1e-10 and np.max(np.abs(gradient))<=1e-9 and objective<=initial+1e-9,'Independent unique ridge solution');systems+=1
                correction=theta[0]+theta[1]*np.array(last[j]);raw=np.log1p(point[n])+correction;prediction=np.expm1(np.maximum(raw,0.));app=record['application'];near(app['predicted_error'][n],correction);near(app['corrected_model_points'][n],prediction);check(app['clipped_leads'][n]==np.where(raw<0)[0].tolist(),'Model clipping');clipped[arm]+=len(app['clipped_leads'][n]);corrected.append(np.log1p(prediction))
            result=np.expm1(np.array(a['weights'])@np.array(corrected));near(record['application']['point'],result);check(r['point'][arm]==record['application']['point'],'Executed corrected blend')
        outcome=read('outcomes/'+tid+'.json');check(sha(incumbent/(tid+'.json'))==receipts['068']['files'][tid+'.json'],'068guard identity');prior=load(incumbent/(tid+'.json'));check(r['actual']==outcome['actual']==prior['actual'],'Unchanged actuals');check(r['point']['uncorrected045']==a['point'] and r['point']['strong050']==outcome['point']['strong_block_cv'] and r['point']['lifetime061']==outcome['point']['lifetime_ledger'] and r['point']['incumbent068']==prior['point']['ledger'],'All strong guards unchanged')
        for n,p in r['point'].items():near(r['scores'][n],risk(p,r['actual']))
        rows.append(r)
    report=load(root/'report.json');check(len(rows)==report['completed_cases']==416 and report['transition_fits']==832 and report['normal_equation_systems']==systems==4992,'All case/fitting costs');check(set(saved_history)==used and report['stored_historical_transitions']==len(used),'All used historical artifacts');check(report['clipped_model_leads']==clipped,'Clipped leads count')
    def summarize(group,saved):
        check(saved['cases']==len(group),'Summary denominator');means={a:math.fsum(risk(r['point'][a],r['actual']) for r in group)/len(group) for a in group[0]['point']}
        for a,v in means.items():near(saved['mean_rmsle'][a],v)
        for a,v in means.items():
            if a!='ledger':near(saved['ledger_reduction'][a],1-means['ledger']/v)
    summarize(rows,report['overall'])
    for d in ('electricity','pedestrian'):summarize([r for r in rows if r['domain']==d],report['domains'][d])
    summarize([r for r in rows if r['round']<8],report['phases']['early_0_7']);summarize([r for r in rows if r['round']>=8],report['phases']['later_8_25'])
    g=report['overall']['ledger_reduction'];guards=('control','strong050','lifetime061','incumbent068');passed=g['control']>=.2 and g['strong050']>=.2 and all(g[a]>0 for a in guards) and all(d['ledger_reduction'][a]>0 for d in report['domains'].values() for a in guards);check(report['development_gate_passed']==passed,'Frozen gate');check(report['new_base_forecasts']==report['api_calls']==0 and manifest['validation_or_final_access'] is False,'No base/API/protected access')
    for n,h in load(root/'source_access.json').items():check(inventory[n]==h and sha(source/n)==h,'Accessed source hashes')
    for n,h in load(root/'comparison_access.json').items():
        label,name=n.split('/',1);folder=anchors if label=='045' else incumbent;check(receipts[label]['files'][name]==h and sha(folder/name)==h,'Comparison access hashes')
    result={'checks':checks,'failures':0,'cases':416,'certified_systems':systems,'seconds':time.monotonic()-start,'new_base_forecasts':0,'api_calls':0,'verifier_sha256':sha(__file__),'scope':'All raw signed errors, temporal transition/config alignment, exact weighted moments and analytical2x2solutions, corrections/clips/blends, score aggregates and source/cost identities.'}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(verify(*sys.argv[1:]),indent=2))
