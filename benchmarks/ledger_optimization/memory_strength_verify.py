"""Independent074 source, candidate, chronology and convex-certificate audit."""
from collections import Counter
from datetime import datetime,timedelta
import hashlib,json,math
from pathlib import Path
import sys,time
import numpy as np

NAMES=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest','search_selected')
KEYS=('0','0.25','0.5','0.75','1');TIES=('0.5','0.25','0.75','0','1')


def verify(root,source):
    root,source=Path(root),Path(source);here=Path(__file__).parent;start=time.monotonic();checks=0
    def check(v,m):
        nonlocal checks
        checks+=1
        if not v:raise AssertionError(m)
    def near(a,b):check(np.allclose(a,b,atol=1e-9,rtol=1e-8),'Independent numeric disagreement')
    def load(p):return json.loads(Path(p).read_text())
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def digest(v):return hashlib.sha256(json.dumps(v,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    def dt(s):
        t=datetime.fromisoformat(s);check(t.tzinfo is not None,'Explicit timezone');return t
    def score(p,y):return math.sqrt(math.fsum((math.log1p(a)-math.log1p(b))**2 for a,b in zip(p,y,strict=True))/24)
    manifest=load(root/'manifest.json');receipt=load(here/'evidence/configuration-search-066.json')
    check(sha(here/'evidence/configuration-search-066.json')==manifest['source_receipt_sha256'],'Source receipt identity');check(sha(source/'SHA256SUMS.json')==receipt['files']['SHA256SUMS.json'],'066inventory identity')
    inventory=load(source/'SHA256SUMS.json');cache={}
    def read(n):
        if n not in cache:
            check(sha(source/n)==inventory[n],'Frozen source identity');cache[n]=load(source/n)
        return cache[n]
    for n,h in manifest['code_sha256'].items():check(sha(here/n)==h,'Frozen candidate code')
    ar=load(here/'evidence/broad-ensemble-045.json');ir=load(here/'evidence/search-ensemble-068.json')
    check(sha(here/'evidence/broad-ensemble-045.json')==manifest['anchor_receipt_sha256'],'Anchor receipt')
    check(sha(here/'evidence/search-ensemble-068.json')==manifest['incumbent_receipt_sha256'],'Incumbent receipt')
    tasks=[read(n) for n in inventory if n.startswith('inputs/')];bykey={(r['series_id'],r['origin']):r for r in tasks};contexts=load(root/'contexts.json')
    check(len(tasks)==len(bykey)==len(contexts)==541,'Full source population')
    for c in contexts:
        task=bykey[c['series_id'],c['origin']];end=(dt(c['origin'])+timedelta(hours=24)).isoformat()
        check(c=={**{k:task[k] for k in ('series_id','origin','features','domain')},'last_target':end,'outcome_recorded_at':end},'Nominal context metadata')
    projected_cache={}
    def projected(tid,arm):
        if (tid,arm) in projected_cache:return projected_cache[tid,arm]
        d=read(f'cases/{tid}/{arm}/decision.json');selected=next(b for b in d['backtests'] if b['config_id']==d['selected']['config_id']);groups=d['backtests'][:6]+[selected];points={}
        for i,name in enumerate(NAMES[:6]):
            e=read(f'cases/{tid}/{arm}/attempt-{4*i+4:02d}.json');f=read(e['cache_ref']);check(f['request']['history_end']==730 and f['request']['config_id']==groups[i]['config_id']==e['config_id'],'Executed production identity');points[name]=f['point']
        f=read(d['final_cache_ref']);check(f['point']==d['point'] and f['request']['config_id']==selected['config_id'],'Selected production identity');points['search_selected']=f['point'];pairs=[]
        for i,end in enumerate((658,682,706)):
            check(all(g['folds'][i]['end']==end and g['folds'][i]['actual']==groups[0]['folds'][i]['actual'] for g in groups),'Matched CV evidence')
            pairs.append({'point':{m:g['folds'][i]['point'] for m,g in zip(NAMES,groups,strict=True)},'actual':groups[0]['folds'][i]['actual']})
        result=({m:g['config_id'] for m,g in zip(NAMES,groups,strict=True)},points,pairs);projected_cache[tid,arm]=result;return result
    certified={};iterations=0
    def certify(fid):
        nonlocal iterations
        if fid in certified:return certified[fid]
        entry=load(root/'fits'/(fid+'.json'));training=entry['training'];f=entry['fit'];check(digest(training)==fid==f['input_sha256'],'Exact fit-input binding')
        a=np.array(training['anchor']);w=np.array(f['weights']);check(w.shape==(4,7) and np.isfinite(w).all() and w.min()>=0,'Simplex feasibility');near(w.sum(axis=1),[1]*4);near(f['anchor'],a)
        def objective(weights):
            value=.01/4*np.square(weights-a).sum();gradient=.02/4*(weights-a)
            for pair,mass in zip(training['pairs'],training['masses'],strict=True):
                design=np.array([[math.log1p(pair['point'][n][h]) for n in NAMES] for h in range(24)]);error=np.array([design[h]@weights[h//6] for h in range(24)])-np.log1p(pair['actual']);norm=math.sqrt(float(np.mean(error**2))+1e-12);value+=mass*norm
                for h in range(24):gradient[h//6]+=mass*error[h]*design[h]/(24*norm)
            return float(value),gradient
        value,gradient=objective(w);initial,_=objective(np.tile(a,(4,1)));gap=float(np.sum(w*gradient)-gradient.min(axis=1).sum());near(f['objective'],value);near(f['initial_objective'],initial);near(f['convex_gap_bound'],gap)
        check(f['success'] and gap<=1e-5+1e-9 and value<=initial+1e-8,'Independent convergence certificate');iterations+=f['iterations'];certified[fid]=entry;return entry
    decisions={};rows=[];seenfits=set();logical=hits=warmcount=warmiters=readycount=0;strengths=Counter()
    for task in sorted(tasks,key=lambda r:(dt(r['origin']),r['series_id'])):
        tid=task['task_id'];r=load(root/'decisions'/(tid+'.json'));row=load(root/'cases'/(tid+'.json'));now=dt(task['origin']);actual=read(f'outcomes/{tid}.json')
        for k in ('task_id','series_id','origin','round','domain'):check(r[k]==row[k]==task[k],'Decision identity')
        check(row['actual']==actual['actual'],'Original outcomes unchanged')
        eligible=sorted([c for c in contexts if c['domain']==task['domain'] and datetime.fromisoformat(c['origin'])<now and datetime.fromisoformat(c['last_target'])<=now and datetime.fromisoformat(c['outcome_recorded_at'])<=now],key=lambda c:(c['origin'],c['series_id']))
        enough=len(eligible)>=16 and len({c['origin'] for c in eligible})>=3;ret=r['retrieval'];check(ret['ready']==enough,'Raw-evidence readiness')
        if eligible:
            x=np.asarray([c['features'] for c in eligible]);scale=np.maximum(x.std(axis=0),.1);distance=np.square((x-task['features'])/scale).sum(axis=1);near(ret['scale'],scale);near(ret['location'],x.mean(axis=0));neighbors=[{'series_id':c['series_id'],'origin':c['origin'],'distance':float(d)} for c,d in zip(eligible,distance,strict=True)]
        else:neighbors=[]
        chosen=sorted(neighbors,key=lambda c:(c['distance'],c['origin'],c['series_id']))[:16] if enough else []
        check(ret['candidates']==neighbors and ret['selected']==chosen,'Independent causal retrieval')
        anchor_ref=r['anchor_source'];anchor=anchor_ref['weights']+[0.]
        if task['round']>=0:
            path=Path(anchor_ref['file']);check(sha(path)==anchor_ref['sha256']==ar['files'][path.name],'Frozen045anchor');check(load(path)['control_fit']['weights']==anchor_ref['weights'],'Anchor weights unchanged')
        else:
            warmcount+=1;_,_,cp=projected(tid,'control');pairs=[{'point':{n:p['point'][n] for n in NAMES[:6]},'actual':p['actual']} for p in cp];check(anchor_ref['pairs']==pairs and anchor_ref['masses']==[1/3]*3,'Warm045training inputs');f=anchor_ref['fit'];w=np.array(f['weights']);check(w.min()>=0 and w.shape==(6,),'Warm simplex');near(w.sum(),1);check(f['input_sha256']==digest({'pairs':pairs,'masses':[1/3]*3}),'Warm input hash')
            value=1e-6*np.square(w-1/6).sum();initial=0.;gradient=2e-6*(w-1/6);omitted=0.
            for p in pairs:
                design=np.log1p(np.array([p['point'][n] for n in NAMES[:6]]).T);y=np.log1p(p['actual']);error=design@w-y;norm=math.sqrt(float(np.mean(error**2)));value+=norm/3;initial+=math.sqrt(float(np.mean((design@np.full(6,1/6)-y)**2)))/3
                if norm<=1e-8:omitted+=norm/3
                else:gradient+=design.T@error/(72*norm)
            gap=float(w@gradient-gradient.min()+omitted);near(f['objective'],value);near(f['initial_objective'],initial);near(f['convex_gap_bound'],gap);check(f['success'] and gap<=1e-5+1e-9,'Warm convex certificate');check(f['weights']==anchor_ref['weights'],'Warm anchor identity');warmiters+=f['iterations']
        for arm in ('control','ledger'):
            record=r['records'][arm];ids,points,cp=projected(tid,arm);check(record['current_points']==points and record['current_config_ids']==ids,'Bound current configuration slots');past=[];refs=[];cohort=[]
            for n in chosen if arm=='ledger' else []:
                oldtask=bykey[n['series_id'],n['origin']];oid=oldtask['task_id'];out=read(f'outcomes/{oid}.json');oi,op,_=projected(oid,arm);check(dt(out['production_source_available_at'])<=now and dt(out['production_recorded_at'])<=now and dt(oldtask['origin'])<now,'Historical availability')
                check(oid in decisions,'Candidate decision already executed');prior=decisions[oid];cohort.append((oldtask,out,prior['records'][arm]['candidates']));past.append({'point':op,'actual':out['actual']});refs.append({'task_id':oid,'series_id':oldtask['series_id'],'origin':oldtask['origin'],'config_ids':oi,'source_available_at':out['production_source_available_at'],'recorded_at':out['production_recorded_at']})
            check(record['historical_references']==refs,'Exact causal raw evidence')
            selection=record['selection'];trialready=len(cohort)==16 and len({o['origin'] for o,_,_ in cohort})>=3
            check(selection['ready']==trialready and selection['missing_trial_records']==[],'Complete candidate cohort')
            check(selection['eligible_records']==(len(eligible) if arm=='ledger' else 0),'Eligible trial population')
            expected=.5
            if trialready:
                means={k:math.fsum(score(c[k]['point'],out['actual']) for _,out,c in cohort)/16 for k in KEYS};minimum=min(means.values());ties=[k for k in TIES if means[k]==minimum];expected=float(ties[0])
                for k,v in means.items():near(selection['candidate_score_means'][k],v)
                check(selection['ties']==ties and selection['selection_basis']=='lowest_matched_historical_candidate_rmsle','Historically selected strength')
                expected_cohort=[{**{k:o[k] for k in ('task_id','series_id','domain','origin')},'arm':arm,'last_target':out['last_target'],'source_available_at':out['production_source_available_at'],'recorded_at':out['production_recorded_at']} for o,out,_ in cohort];check(selection['cohort']==expected_cohort,'Selection evidence identity')
            else:check(selection['candidate_score_means'] is None and selection['cohort']==[] and selection['selection_basis']=='insufficient_matched_history','Explicit default')
            check(selection['requested_strength']==expected and float(record['selected_key'])==expected,'Selection matches available risks');check(selection['provider_calls']==selection['new_weight_fits']==0,'Read-only choice')
            check(set(record['candidates'])==set(KEYS),'All candidates executed')
            for k in KEYS:
                c=record['candidates'][k];g=float(k) if past else 0.;pairs=(cp if g<1 else [])+(past if g>0 else []);masses=([(1-g)/3]*3 if g<1 else [])+([g/16]*16 if g>0 else []);binding={'pairs':pairs,'masses':masses,'anchor':anchor};fid=digest(binding);logical+=1;cached=fid in seenfits;hits+=cached;seenfits.add(fid)
                check(c['cache_hit']==cached and c['fit_ref']==fid,'Physical versus logical accounting');check(c['execution_id']==digest({'task_id':tid,'arm':arm,'strength':float(k)}) and c['task_id']==tid and c['arm']==arm and c['requested_strength']==float(k) and c['effective_strength']==g,'Logical candidate identity');check(c['forecast_recorded_at']==task['origin'],'Simulated decision recording')
                entry=certify(fid);check(entry['training']==binding,'Exact candidate training');w=entry['fit']['weights'];point=[math.expm1(math.fsum(w[h//6][j]*math.log1p(points[n][h]) for j,n in enumerate(NAMES))) for h in range(24)];near(c['point'],point)
            check(row['point'][arm]==record['candidates'][record['selected_key']]['point'],'Selected executed forecast')
        check(row['point']['fixed_half']==r['records']['ledger']['candidates']['0.5']['point'],'Fixed-half diagnostic')
        for name,point in row['point'].items():near(row['scores'][name],score(point,row['actual']))
        if task['round']>=0:
            p=Path('results/search-ensemble-068-001')/(tid+'.json');check(sha(p)==ir['files'][p.name],'068frozen comparison');prior=load(p);check(row['actual']==prior['actual'],'Matched comparison outcome')
            for a,b in (('control','control'),('fixed_half','ledger'),('incumbent068','ledger')):check(max(abs(x-y) for x,y in zip(row['point'][a],prior['point'][b],strict=True))<=1e-12,'068parity')
            for a in ('strong_block_cv','lifetime_ledger'):check(row['point'][a]==actual['point'][a],'Existing guards unchanged')
            rows.append(row);strengths[str(r['records']['ledger']['selection']['requested_strength'])]+=1;readycount+=r['records']['ledger']['selection']['ready']
        check(row['ledger_selected_strength']==r['records']['ledger']['selection']['requested_strength'] and row['trial_history_ready']==r['records']['ledger']['selection']['ready'],'Selection report identity');decisions[tid]=r
    report=load(root/'report.json');check(len(decisions)==report['completed_tasks']==541 and len(rows)==report['completed_scored']==416,'Complete chronology');check(logical==report['logical_blend_requests']==5410 and hits==report['cache_hits'] and len(certified)==report['physical_blend_fits'],'All fit costs');check(warmcount==report['warm_anchor_fits']==125 and warmiters==report['anchor_iterations'] and iterations==report['blend_iterations'],'All anchor and iteration costs');check(report['scored_strength_counts']==dict(strengths) and report['mature_scored_cases']==readycount,'Readiness and choices')
    check({p.stem for p in (root/'fits').glob('*.json')}==set(certified),'No unreported fit artifacts')
    def summarize(group,saved):
        check(saved['cases']==len(group),'Summary denominator');means={a:math.fsum(score(r['point'][a],r['actual']) for r in group)/len(group) for a in group[0]['point']}
        for a,v in means.items():near(saved['mean_rmsle'][a],v)
        for a,v in means.items():
            if a!='ledger':near(saved['ledger_reduction'][a],1-means['ledger']/v)
    summarize(rows,report['overall'])
    for d in ('electricity','pedestrian'):summarize([r for r in rows if r['domain']==d],report['domains'][d])
    summarize([r for r in rows if r['round']<8],report['phases']['early_0_7']);summarize([r for r in rows if r['round']>=8],report['phases']['later_8_25'])
    g=report['overall']['ledger_reduction'];guards=('control','strong_block_cv','lifetime_ledger','incumbent068');passed=g['control']>=.2 and g['strong_block_cv']>=.2 and all(g[a]>0 for a in guards) and all(d['ledger_reduction'][a]>0 for d in report['domains'].values() for a in guards);check(passed==report['development_gate_passed'],'Frozen gate');check(report['api_calls']==report['new_provider_calls']==0,'No source/API calls')
    for n,h in load(root/'source_access.json').items():check(inventory[n]==h and sha(source/n)==h,'Full accessed-source integrity')
    result={'checks':checks,'failures':0,'tasks':541,'scored_cases':416,'logical_candidates':logical,'certified_blend_fits':len(certified),'certified_warm_anchors':125,'seconds':time.monotonic()-start,'api_calls':0,'provider_calls':0,'verifier_sha256':sha(__file__),'scope':'All source/configuration bindings, chronological candidate availability and selection, masses, cache accounting, independent convex certificates, forecast parity and score aggregates. Simulated historical timing; numerical development only.'}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(verify(*sys.argv[1:]),indent=2))
