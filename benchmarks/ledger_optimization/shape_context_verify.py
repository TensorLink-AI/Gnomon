"""Independent072 observed-shape retrieval and blend, convex certificate, temporal retrieval and source audit."""
from datetime import datetime,timedelta
import hashlib,json,math
from pathlib import Path
import sys,time
import numpy as np

NAMES=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest','search_selected')

def verify(root,source,incumbent):
    root,source,incumbent=Path(root),Path(source),Path(incumbent);here=Path(__file__).parent;start=time.monotonic();checks=0
    def check(v,m):
        nonlocal checks
        checks+=1
        if not v:raise AssertionError(m)
    def near(a,b):check(np.allclose(a,b,atol=1e-9,rtol=1e-8),'Numerical disagreement')
    def load(p):return json.loads(Path(p).read_text())
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def score(p,y):return math.sqrt(math.fsum((math.log1p(a)-math.log1p(b))**2 for a,b in zip(p,y,strict=True))/len(y))
    manifest=load(root/'manifest.json');source_receipt=here/'evidence/configuration-search-066.json';receipt=load(source_receipt)
    check(sha(source_receipt)==manifest['source_receipt_sha256'],'066receipt identity');check(sha(source/'SHA256SUMS.json')==receipt['files']['SHA256SUMS.json'],'066inventory identity')
    inventory=load(source/'SHA256SUMS.json');cache={}
    def read(name):
        if name not in cache:
            p=source/name;check(sha(p)==inventory[name],'Source binding');cache[name]=load(p)
        return cache[name]
    for n,h in manifest['code_sha256'].items():check(sha(here/n)==h,'Frozen068code')
    tasks=[read(n) for n in inventory if n.startswith('inputs/')];check(len(tasks)==541,'Full context population');bykey={(r['series_id'],r['origin']):r for r in tasks}
    contexts=load(root/'contexts.json');check(len(contexts)==541,'Full saved metadata')
    for c in contexts:
        task=bykey[c['series_id'],c['origin']];end=(datetime.fromisoformat(c['origin'])+timedelta(hours=24)).isoformat()
        check(c=={**{k:task[k] for k in ('series_id','origin','features','domain')},'last_target':end,'outcome_recorded_at':end},'Exact nominal context metadata')
    def projected(tid,arm):
        d=read(f'cases/{tid}/{arm}/decision.json');selected=next(r for r in d['backtests'] if r['config_id']==d['selected']['config_id']);groups=d['backtests'][:6]+[selected]
        ids={name:g['config_id'] for name,g in zip(NAMES,groups,strict=True)};points={}
        for i,name in enumerate(NAMES[:6]):
            e=read(f'cases/{tid}/{arm}/attempt-{4*i+4:02d}.json');points[name]=read(e['cache_ref'])['point']
        points['search_selected']=d['point'];pairs=[]
        for i in range(3):pairs.append({'point':{name:g['folds'][i]['point'] for name,g in zip(NAMES,groups,strict=True)},'actual':groups[0]['folds'][i]['actual']})
        return ids,points,pairs
    profiles=load(root/'profiles.json');check(len(profiles)==541,'Full observed-profile population')
    for tid,p in profiles.items():
        t=read('inputs/'+tid+'.json');values=[math.log1p(v) for v in t['history'][-672:]];center=math.fsum(values)/672;spread=max(.1,math.sqrt(math.fsum((v-center)**2 for v in values)/672));origin=datetime.fromisoformat(t['origin'])
        labels=[origin-timedelta(hours=672-i) for i in range(672)];hour_groups=[[v for v,at in zip(values,labels,strict=True) if at.hour==h] for h in range(24)];week_groups=[[v for v,at in zip(values,labels,strict=True) if at.weekday()==d] for d in range(7)]
        vector=[(math.fsum(g)/len(g)-center)/spread for g in hour_groups+week_groups];near(p['vector'],vector);near(p['center'],center);near(p['scale'],spread)
        check(p['hour_counts']==[len(g) for g in hour_groups] and p['weekday_counts']==[len(g) for g in week_groups],'Exact calendar group coverage')
        check(p['history_start']==labels[0].isoformat() and p['history_end']==labels[-1].isoformat() and p['origin']==t['origin'] and p['observations']==672,'Observed profile interval')
        encoded=json.dumps({'history':t['history'],'origin':t['origin']},separators=(',',':')).encode();check(p['input_sha256']==hashlib.sha256(encoded).hexdigest(),'Profile input binding')
    rows=[];iterations=0;ready=0
    for task in sorted((r for r in tasks if r['round']>=0),key=lambda r:(r['origin'],r['series_id'])):
        tid=task['task_id'];r=load(root/(tid+'.json'));now=datetime.fromisoformat(task['origin']);current=read(f'outcomes/{tid}.json')
        check(r['series_id']==task['series_id'] and r['origin']==task['origin'] and r['round']==task['round'],'Current task identity')
        eligible=sorted([c for c in contexts if c['domain']==task['domain'] and datetime.fromisoformat(c['origin'])<now and datetime.fromisoformat(c['last_target'])<=now and datetime.fromisoformat(c['outcome_recorded_at'])<=now],key=lambda c:(c['origin'],c['series_id']))
        enough=len(eligible)>=16 and len({c['origin'] for c in eligible})>=3;ret=r['retrieval'];check(ret['ready']==enough,'Historical readiness')
        if eligible:
            x=np.array([c['features'] for c in eligible]);scale=np.maximum(x.std(axis=0),.1);distance=np.sum(((x-task['features'])/scale)**2,axis=1);near(ret['scale'],scale);near(ret['location'],x.mean(axis=0))
            candidates=[{'series_id':c['series_id'],'origin':c['origin'],'distance':float(d)} for c,d in zip(eligible,distance,strict=True)]
        else:candidates=[]
        check(ret['query_shape']==profiles[tid],'Query profile binding')
        if candidates:
            past_profiles=[profiles[bykey[c['series_id'],c['origin']]['task_id']] for c in candidates];x=np.array([p['vector'] for p in past_profiles]);scale=np.maximum(.1,x.std(axis=0));distances=np.sum(((x-profiles[tid]['vector'])/scale)**2,axis=1)
            near(ret['shape_scale'],scale);near(ret['shape_location'],x.mean(axis=0))
            candidates=[{**c,'coarse_distance':c['distance'],'shape_distance':float(d),'distance':c['distance']/12+float(d)/31,'shape_input_sha256':p['input_sha256']} for c,d,p in zip(candidates,distances,past_profiles,strict=True)]
        selected=sorted(candidates,key=lambda c:(c['distance'],c['origin'],c['series_id']))[:16] if enough else []
        check(ret['candidates']==candidates and ret['selected']==selected,'All and only causal shape-matched neighbors')
        recent=set(sorted({c['origin'] for c in candidates})[-8:]);check(ret['selected_outside_recent_eight']==sum(c['origin'] not in recent for c in selected),'Reordered older-origin count')
        check(ret['retrieval_policy']=='coarse_plus_observed_calendar_shape' and ret['shape_dimensions']==31 and ret['coarse_dimensions']==12,'Declared distance contract')
        anchor_ref=r['anchor_source'];check(sha(anchor_ref['file'])==anchor_ref['sha256'],'045anchor source');anchor=load(anchor_ref['file'])['control_fit']['weights']+[0.]
        for arm in ('control','ledger'):
            record=r['records'][arm];f=record['fit'];ids,points,pairs=projected(tid,arm);expected_refs=[]
            check(record['current_points']==points and record['current_config_ids']==ids,'Current seven slots bound to actual066 executions')
            if arm=='ledger' and enough:
                ready+=1
                for ref in selected:
                    old=bykey[ref['series_id'],ref['origin']];out=read(f'outcomes/{old["task_id"]}.json');oldids,oldpoints,_=projected(old['task_id'],arm)
                    check(datetime.fromisoformat(out['production_source_available_at'])<=now and datetime.fromisoformat(out['production_recorded_at'])<=now,'Actual production visibility')
                    pairs.append({'point':oldpoints,'actual':out['actual']});expected_refs.append({'task_id':old['task_id'],'series_id':old['series_id'],'origin':old['origin'],'arm':arm,'config_ids':oldids,
                        'source_available_at':out['production_source_available_at'],'recorded_at':out['production_recorded_at']})
                masses=[1/6]*3+[.5/16]*16
            else:masses=[1/3]*3
            check(record['historical_references']==expected_refs and record['pairs']==pairs and record['masses']==masses,'Exact causal training inputs and masses')
            check(record['evidence_ready']==bool(expected_refs),'Arm-specific memory readiness')
            check(record['selected_slot_duplicates_original']==(ids['search_selected'] in [ids[m] for m in NAMES[:6]]),'Duplicate-slot disclosure')
            encoded=json.dumps({'pairs':pairs,'masses':masses,'anchor':anchor},separators=(',',':')).encode();check(hashlib.sha256(encoded).hexdigest()==f['input_sha256'],'Full training input hash');near(f['anchor'],anchor)
            w=np.asarray(f['weights']);check(w.shape==(4,7) and np.isfinite(w).all() and w.min()>=0,'Weight feasibility');near(w.sum(axis=1),np.ones(4));iterations+=f['iterations']
            def objective(weights):
                loss=.01/4*np.square(weights-anchor).sum();gradient=.02/4*(weights-anchor)
                for p,mass in zip(pairs,masses,strict=True):
                    design=np.asarray([[math.log1p(p['point'][m][h]) for m in NAMES] for h in range(24)])
                    predicted=np.array([np.dot(design[h],weights[h//6]) for h in range(24)]);error=predicted-np.log1p(p['actual']);norm=math.sqrt(float(np.mean(error**2))+1e-12);loss+=mass*norm
                    for h in range(24):gradient[h//6]+=mass*error[h]*design[h]/(24*norm)
                return float(loss),gradient
            value,gradient=objective(w);initial,_=objective(np.tile(anchor,(4,1)));gap=float(np.sum(w*gradient)-np.min(gradient,axis=1).sum())
            near(f['objective'],value);near(f['initial_objective'],initial);near(f['convex_gap_bound'],gap)
            check(f['success'] and gap<=1e-5+1e-9 and value<=initial+1e-8,'Independent frozen convergence certificate')
            output=[math.expm1(math.fsum(w[h//6,j]*math.log1p(points[m][h]) for j,m in enumerate(NAMES))) for h in range(24)];near(r['point'][arm],output)
        check(r['actual']==current['actual'],'Unchanged scored actuals')
        for a in ('strong_block_cv','lifetime_ledger'):check(r['point'][a]==current['point'][a],'Strong guard unchanged')
        for a in ('control','ledger'):check(r['point']['search_'+a]==current['point'][a],'Previous search unchanged')
        for a,p in r['point'].items():near(r['scores'][a],score(p,r['actual']))
        rows.append(r)
    report=load(root/'report.json');check(len(rows)==416 and report['completed_cases']==416 and report['weight_fits']==832,'Complete cases and fits');check(report['iterations']==iterations and report['ledger_ready_cases']==ready,'Iteration/readiness totals')
    def summary(group,saved):
        check(saved['cases']==len(group),'Summary denominator');means={a:math.fsum(score(r['point'][a],r['actual']) for r in group)/len(group) for a in group[0]['point']}
        for a,v in means.items():near(saved['mean_rmsle'][a],v)
        for a,v in means.items():
            if a!='ledger':near(saved['ledger_reduction'][a],1-means['ledger']/v)
    summary(rows,report['overall'])
    for d in ('electricity','pedestrian'):summary([r for r in rows if r['series_id'].startswith(d+':')],report['domains'][d])
    summary([r for r in rows if r['round']<8],report['phases']['early_0_7']);summary([r for r in rows if r['round']>=8],report['phases']['later_8_25'])
    g=report['overall']['ledger_reduction'];passed=g['control']>=.2 and g['strong_block_cv']>=.2 and g['lifetime_ledger']>0 and g['incumbent_search_ledger']>0 and all(d['ledger_reduction'][a]>0 for d in report['domains'].values() for a in ('control','strong_block_cv','lifetime_ledger','incumbent_search_ledger'))
    check(report['development_gate_passed']==passed,'Frozen gate');check(report['new_provider_calls']==report['api_calls']==0,'No new forecasts/API calls')
    for k,v in {'inherited_forecast_computations':49616,'inherited_anchor_fits':416,'inherited_search_surrogate_solves':11902,'inherited_search_logical_attempts_per_arm':31378}.items():check(report[k]==v,'Inherited costs')
    for n,h in load(root/'source_access.json').items():check(inventory[n]==h and sha(source/n)==h,'Complete access receipt')
    amendment=load(root/'amendment.json')
    for n,h in amendment['code_sha256'].items():check(sha(here/n)==h,'Frozen070amendment')
    oldreceipt=here/'evidence/search-ensemble-068.json';check(sha(oldreceipt)==amendment['incumbent_receipt_sha256'],'Incumbent receipt')
    oldhashes=load(oldreceipt)['files'];comparison=load(root/'comparison.json');check(len(comparison)==416,'Full incumbent comparison')
    bytid={r['task_id']:r for r in rows};check({r['task_id'] for r in comparison}==set(bytid),'Exact comparison cohort')
    for c in comparison:
        n=c['task_id']+'.json';check(sha(incumbent/n)==oldhashes[n],'Incumbent source identity');old=load(incumbent/n);current=bytid[c['task_id']]
        check(all(current[k]==old[k] for k in ('series_id','origin','round','actual')),'Matched incumbent task/actuals')
        check(current['point']['control']==old['point']['control'] and current['records']['control']==old['records']['control'],'Matched control reproduces068exactly')
        for a in current['scores']:near(c['scores'][a],current['scores'][a])
        for a in ('control','ledger'):near(c['scores']['incumbent_search_'+a],score(old['point'][a],old['actual']))
    def comparison_summary(group,saved):
        for a in ('incumbent_search_control','incumbent_search_ledger'):
            value=math.fsum(r['scores'][a] for r in group)/len(group);near(saved['mean_rmsle'][a],value);near(saved['ledger_reduction'][a],1-saved['mean_rmsle']['ledger']/value)
    comparison_summary(comparison,report['overall'])
    for d in ('electricity','pedestrian'):comparison_summary([r for r in comparison if r['series_id'].startswith(d+':')],report['domains'][d])
    comparison_summary([r for r in comparison if r['round']<8],report['phases']['early_0_7']);comparison_summary([r for r in comparison if r['round']>=8],report['phases']['later_8_25'])
    check(report['additional_inherited_068comparison_weight_fits']==832,'Additional incumbent fitting costs')
    check(amendment['block_simplexes']==4 and amendment['retrieval_policy']=='coarse_plus_observed_calendar_shape' and amendment['weights_per_simplex']==7 and amendment['base068modified'] is False,'Actual amendment scope')
    for n,h in load(root/'profile_source_access.json').items():check(n.startswith('inputs/') and inventory[n]==h and sha(source/n)==h,'Observed-only profile source accesses')
    result={'checks':checks,'failures':0,'cases':len(rows),'certified_weight_fits':832,'profiles_validated':len(profiles),'seconds':time.monotonic()-start,'provider_calls':0,'api_calls':0,'verifier_sha256':sha(__file__),
        'scope':'All causal neighbors, configuration-bound training pairs, simplex feasibility, independent convex objective/gradient certificates, produced blends and score aggregates. Reuses previously audited066 raw forecasts.'}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(verify(*sys.argv[1:]),indent=2))
