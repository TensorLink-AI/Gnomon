"""Audit frozen066 inputs, execution budgets, causal proposals and all scores.

Search reconstruction uses independent block Cholesky algebra. Prediction replay
covers one deterministic request per used configuration, not every estimator fit.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from .search_schur_audit import AcquisitionAudit


def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def score(p,y): return math.sqrt(math.fsum((math.log1p(a)-math.log1p(b))**2 for a,b in zip(p,y,strict=True))/len(y))


def verify(root):
    root=Path(root);here=Path(__file__).parent;start=time.monotonic();checks=0;proposals=0
    def check(ok,message):
        nonlocal checks
        checks+=1
        if not ok: raise AssertionError(message)
    def near(a,b):check(np.allclose(a,b,atol=1e-9,rtol=1e-8),'Numerical disagreement')
    manifest=load(root/'manifest.json');report=load(root/'report.json')
    for name,h in manifest['code_sha256'].items():check(sha(here/name)==h,'Frozen code '+name)
    for name,h in manifest['source_receipts_sha256'].items():check(sha(here/'evidence'/name)==h,'Frozen receipt '+name)
    spans=[]
    for path,key in [('results/broad-panel-037-001/development-spans.json','source_span_sha256'),('results/broad-warmup-042-001/warmup-spans.json','warm_span_sha256')]:
        check(sha(path)==manifest[key],'Source span identity');spans.append(load(path))
    def historical(directory,receipt,prefix):
        rows={}
        for name,h in load(here/'evidence'/receipt)['files'].items():
            if name.startswith(prefix):
                path=Path(directory)/name;check(sha(path)==h,'Source record identity');r=load(path);rows[r['series_id'],r['origin']]=r
        return rows
    originals=historical('results/broad-screen-038-001','broad-screen-038.json',('electricity:','pedestrian:'))
    originals.update(historical('results/broad-warm-screen-043-001','broad-warm-screen-043.json',('warmup-electricity:','warmup-pedestrian:')))
    guards=historical('results/broad-lifetime-memory-061-001','broad-lifetime-memory-061.json',('electricity:','pedestrian:'))
    ep=Path('results/broad-warm-screen-043-001/episodes.json');check(sha(ep)==load(here/'evidence/broad-warm-screen-043.json')['files']['episodes.json'],'Context source identity')
    contexts={(r['series_id'],r['origin']):r for r in load(ep)}
    catalogue=load('results/common-configuration-064-001/catalogue.json');order={r['config_id']:i for i,r in enumerate(catalogue)}
    configs={r['config_id']:r['config'] for r in catalogue};names=dict(zip(order,('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')))
    inputs=[load(p) for p in sorted((root/'inputs').glob('*.json'))];check(len(inputs)==541,'Complete input count')
    check({(r['series_id'],r['origin']) for r in inputs}==set(originals),'Exactly frozen cohort')
    decisions={};histories={'control':[],'ledger':[]};costs={a:Counter() for a in histories};cache={};owners=Counter();replay={};outcomes=[]
    for task in inputs:
        for arm in histories:
            p=root/'cases'/task['task_id']/arm/'decision.json';r=load(p);decisions[task['task_id'],arm]=r;histories[arm].append(r['study'])
    for ti,task in enumerate(inputs):
        tid=task['task_id'];key=task['series_id'],task['origin'];raw=originals[key];span=spans[task['round']<0][key[0]];low,end=raw['history_indices'];y=span['values'][end:end+24]
        check(task['history']==span['values'][low:end] and len(task['history'])==730,'Exact observed history')
        check(digest(task['history'])==raw['history_sha256'] and y==raw['actual'],'Historical source values')
        at=datetime.fromisoformat(span['start_label']).replace(tzinfo=timezone.utc)+timedelta(hours=end)
        check(at.isoformat()==task['origin'] and raw['last_target']==(at+timedelta(hours=24)).isoformat(),'Temporal labels')
        check(tid==digest({'series_id':key[0],'origin':key[1]}) and task['features']==contexts[key]['features'],'Input identity/features')
        check(task['source_sha256']==span['source_sha256'] and task['unit']==span['unit'],'Source/unit identity')
        check(set(task)=={'task_id','series_id','domain','round','origin','history','features','unit','source_sha256'},'Observed-only task input')
        for arm in histories:
            d=decisions[tid,arm];home=root/'cases'/tid/arm;backtests=d['backtests'];study=d['study']
            check(d['task_id']==tid and d['arm']==arm and d['origin']==task['origin'] and d['round']==task['round'],'Decision identity')
            check(len(backtests)==17 and len({r['config_id'] for r in backtests})==17,'Seventeen unique configurations')
            check(study=={**{k:task[k] for k in ('series_id','domain','origin','features')},'arm':arm,'source_available_at':task['origin'],'recorded_at':task['origin'],
                'evidence_kind':'executed_backtests','revision':manifest['revision'],'backtests':[{k:r[k] for k in ('config','config_id','cv_rmsle')} for r in backtests],
                'study_ref':str((home/'decision.json').relative_to(root))},'Exact immutable backtest evidence')
            audit=AcquisitionAudit(study,histories[arm] if arm=='ledger' else [],catalogue,check,near)
            for step in range(11):
                p=home/f'proposal-{step:02d}.json';proposal=load(p);audit.verify(proposal,backtests[:6+step]);proposals+=1
                check(proposal['next_config_id']==backtests[6+step]['config_id'],'Proposed configuration actually tested')
                check(d['proposal_refs'][step]==str(p.relative_to(root)),'Proposal reference')
            expected=[]
            for i,b in enumerate(backtests):
                check(b['config']==configs[b['config_id']],'Configuration catalogue membership')
                if i<6:check(b['config_id']==catalogue[i]['config_id'],'Common starter order')
                check(len(b['folds'])==3,'Three observed CV folds')
                for endpoint,f in zip((658,682,706),b['folds'],strict=True):
                    check(f['end']==endpoint and f['actual']==task['history'][endpoint:endpoint+24],'Observed CV target interval')
                    near(f['rmsle'],score(f['point'],f['actual']));expected.append((b['config_id'],endpoint,f))
                near(b['cv_rmsle'],math.fsum(f['rmsle'] for f in b['folds'])/3)
                if i<6:expected.append((b['config_id'],730,None))
            selected=min(backtests,key=lambda b:(b['cv_rmsle'],order[b['config_id']]))
            check(d['selected']['config_id']==selected['config_id'] and d['selected']['config']==selected['config'],'Selection by current observed CV')
            expected.append((selected['config_id'],730,None));events=[load(p) for p in sorted(home.glob('attempt-*.json'))]
            check(len(events)==len(expected)==58 and d['logical_attempts']==58,'Equal numerical budget')
            for i,(event,(cid,endpoint,fold)) in enumerate(zip(events,expected,strict=True),1):
                req={'task_id':tid,'input_sha256':digest(task),'config_id':cid,'revision':manifest['revision'],'history_end':endpoint};rid=digest(req)
                ref=f'forecasts/{tid}/{rid}.json';check(event['attempt']==i and event['arm']==arm and event['config_id']==cid and event['config']==configs[cid] and event['history_end']==endpoint,'Attempt sequence')
                check(event['completed'] and event['request_id']==rid and event['cache_ref']==ref and event['execution_id']==digest({'request':rid,'arm':arm,'attempt':i}),'Execution identity/success')
                if ref not in cache:
                    cached=load(root/ref);cache[ref]=cached;check(cached['request']==req,'Cache request binding')
                    check(len(cached['point'])==24 and all(math.isfinite(v) and v>=0 for v in cached['point']),'Forecast contract')
                    if cid in names:
                        p=raw['point'][names[cid]] if endpoint==730 else raw['folds'][names[cid]][(endpoint-658)//24]['point']
                        check(cached['origin']=='original038_043' and cached['point']==p,'Exact inherited forecast')
                        sr=cached['source_ref'];check(sha(Path(sr['root'])/sr['file'])==sr['sha256'],'Inherited source link')
                    else:check(cached['origin']=='configured064','New configuration origin')
                    if cid not in replay or ref<replay[cid][0]:replay[cid]=(ref,task,endpoint)
                cached=cache[ref]
                check(event['physical_started']==(event['resolution']=='new_computation'),'Physical accounting')
                check(event['estimator_fit']==(configs[cid]['kind'] in ('ridge','forest')),'Estimator accounting')
                if event['resolution']!='local_cache':owners[ref]+=1
                if fold:check(fold['execution_id']==event['execution_id'] and fold['cache_ref']==ref and fold['point']==cached['point'],'Fold bound to execution')
                if i==58:check(d['final_execution_id']==event['execution_id'] and d['final_cache_ref']==ref and d['point']==cached['point'],'Final forecast bound to selection')
            physical=sum(e['physical_started'] for e in events);fits=sum(e['physical_started'] and e['estimator_fit'] for e in events)
            for k,v in {'logical_attempts':58,'physical_computations':physical,'physical_estimator_fits':fits,'reused_computations':58-physical,'surrogate_solves':11}.items():check(d[k]==v,'Decision cost '+k);costs[arm][k]+=v
            costs[arm]['cases']+=1;costs[arm]['seconds']+=d['seconds'];costs[arm]['cpu_seconds']+=d['cpu_seconds']
        out=load(root/'outcomes'/(tid+'.json'));check(out['actual']==y and out['origin']==task['origin'] and out['round']==task['round'],'Outcome identity')
        check(out['warmup']==(task['round']<0) and not out['production_outcomes_used_by_search'],'Warmup and unused production label contract')
        check(out['production_source_available_at']==out['production_recorded_at']==out['last_target']==raw['last_target'],'Production availability boundary')
        for arm in histories:check(out['point'][arm]==decisions[tid,arm]['point'] and out['selected'][arm]==decisions[tid,arm]['selected']['config_id'],'Outcome execution')
        if not out['warmup']:
            check(out['point']['strong_block_cv']==guards[key]['point']['block_cv'] and out['point']['lifetime_ledger']==guards[key]['point']['lifetime_ledger'],'Existing strong guards unchanged')
        for arm,p in out['point'].items():near(out['scores'][arm],score(p,y))
        outcomes.append(out)
        if ti%20==0:print(json.dumps({'audited_tasks':ti+1,'proposals':proposals,'checks':checks}),flush=True)
    check(all(v==1 for v in owners.values()) and set(owners)==set(cache),'Exactly one physical/inherited cache owner')
    check(len(list((root/'forecasts').glob('*/*.json')))==len(cache),'No orphan cached executions')
    for arm in histories:
        for k,v in costs[arm].items():near(report['costs'][arm][k],v)
    scored=[r for r in outcomes if not r['warmup']];check(len(scored)==416 and len(outcomes)-len(scored)==125,'Complete scored and warmup denominators')
    def summary(rows,observed):
        check(observed['cases']==len(rows),'Summary denominator')
        means={a:math.fsum(score(r['point'][a],r['actual']) for r in rows)/len(rows) for a in ('control','ledger','strong_block_cv','lifetime_ledger')}
        for a,v in means.items():near(observed['mean_rmsle'][a],v)
        for a in ('control','strong_block_cv','lifetime_ledger'):near(observed['ledger_reduction'][a],1-means['ledger']/means[a])
    summary(scored,report['overall'])
    for domain in ('electricity','pedestrian'):summary([r for r in scored if r['series_id'].startswith(domain+':')],report['domains'][domain])
    summary([r for r in scored if r['round']<8],report['phases']['early_0_7']);summary([r for r in scored if r['round']>=8],report['phases']['later_8_25'])
    reduction=report['overall']['ledger_reduction'];passed=reduction['control']>=.2 and reduction['strong_block_cv']>=.2 and reduction['lifetime_ledger']>0 and all(v>0 for d in report['domains'].values() for v in d['ledger_reduction'].values())
    check(report['development_gate_passed']==passed,'Frozen gate');check(proposals==11902,'All acquisition proposals audited')
    from .configured_hourly import predict
    replayed=[];replay_start=time.monotonic()
    for cid,(ref,task,end) in sorted(replay.items()):
        low=datetime.fromisoformat(task['origin'])-timedelta(hours=730);labels=[low+timedelta(hours=i) for i in range(end+24)]
        point=predict(task['history'][:end],labels[:end],labels[end:],configs[cid]);near(point,cache[ref]['point']);replayed.append({'config_id':cid,'cache_ref':ref})
    result={'checks':checks,'failures':0,'tasks':541,'arm_cases':1082,'proposals_reconstructed':proposals,'cached_requests_validated':len(cache),
        'audit_prediction_replays':replayed,'audit_forecast_computations':len(replayed),'audit_estimator_fits':sum(configs[r['config_id']]['kind'] in ('ridge','forest') for r in replayed),
        'replay_seconds':time.monotonic()-replay_start,'seconds':time.monotonic()-start,'api_calls':0,
        'scope':'Every input, CV score, target score, proposal and logical attempt; one deterministic saved request per used configuration replayed with frozen adapter. No independent reimplementation of sklearn estimators. Numerical proxy only.',
        'verifier_sha256':{n:sha(here/n) for n in ('configuration_search_run_verify.py','search_schur_audit.py')}}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    root=Path(sys.argv[1])
    try: print(json.dumps(verify(root),indent=2))
    except BaseException as e:
        path=root/'verification-failure.json';path.write_text(json.dumps({'error':type(e).__name__,'message':str(e)},indent=2)+'\n');raise
