"""Independent saved-pair audit and honest three-arm outcome report."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from collections import Counter
from statistics import mean
from datetime import datetime

from .transport import dump,sha


def rmsle(point,actual):
    assert len(point)==len(actual)==14
    return math.sqrt(sum((math.log1p(p)-math.log1p(a))**2 for p,a in zip(point,actual,strict=True))/14)


def analyze(root):
    manifest=json.loads((root/'manifest.json').read_text())
    jobs=json.loads((root/'host-jobs.json').read_text());index={(s,j['round']):j for s,js in jobs.items() for j in js}
    rows=[];checks=0;failures=[]
    for p in sorted(root.glob('*/*/round-*/grade.json')):
        r=json.loads(p.read_text());job=index[r['series_id'],r['round']];project=p.parent/'project'
        assert abs(rmsle(r['point'],job['actual'])-r['rmsle'])<1e-12;checks+=1
        for name,digest in json.loads((p.parent/'input-hashes.json').read_text()).items():
            if sha(project/name)!=digest:failures.append(str(p)+': changed '+name)
            checks+=1
        log=project/'experiments.jsonl'
        events=[json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        old=json.loads((p.parent/'prior-evidence.json').read_text())
        assert hashlib.sha256((log.read_bytes() if log.exists() else b'')[:old['bytes']]).hexdigest()==old['sha256']
        attempts=sum(e['event']=='attempt' and e['task_origin']==job['origin'] for e in events)
        assert attempts<=60 and attempts==r['numerical_attempts']
        for admission in events:
            if admission['event']=='batch_admitted':
                assert admission['fits']==3 and admission['reserve']==1 and admission['remaining_before']>=4
                assert admission['phase']!='selection' or admission['initial_baseline']
        checks+=3
        if r['valid']:
            checkpoint=json.loads((project/'checkpoint.json').read_text())
            fingerprint=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            assert json.loads((project/'checkpoints'/(checkpoint['checkpoint_id']+'.json')).read_text())==checkpoint
            selected=[e for e in events if e['event']=='result' and e['kind']=='forecast'
                      and e['task_origin']==job['origin'] and e['execution']['execution_id']==checkpoint['execution_id']]
            assert len(selected)==1 and selected[0]['point']==r['point']==checkpoint['point']
            assert checkpoint['request_fingerprint']==fingerprint(selected[0]['request'])
            eligible={};selected_complete=None
            for e in events:
                if e.get('task_origin')!=job['origin']:continue
                if e['event']=='result' and e['kind']=='backtest':
                    eligible.setdefault(e['config_id'],{})[e['request']['cutoff']]=e['config']
                elif e['event']=='selection' and e['checkpoint_id']==checkpoint['checkpoint_id']:
                    complete={cid:next(iter(v.values())) for cid,v in eligible.items()
                              if set(v)=={job['request']['timestamps'][i] for i in (687,701,715)}}
                    selected_complete=(len(complete)>=2 and any(c['model']!='seasonal' for c in complete.values())
                                       and checkpoint['config_id'] in complete)
                    assert e['checkpoint_sha256']==fingerprint(checkpoint)
                    assert sorted(complete)==e['compared_configurations']
                    assert e['selection_after_comparison']==selected_complete
            assert selected_complete is not None and r['workflow_complete']==selected_complete
            assert r['baseline_checkpoint_only']==(not selected_complete)
            checks+=7
        successful=sum(e['event']=='result' and e['task_origin']==job['origin'] for e in events)
        if r['numerical_successes']!=successful:
            r['raw_grade_numerical_successes']=r['numerical_successes']
            r['numerical_successes']=successful
            r['diagnostic_correction']='Count successful numerical executions independently of missing final submission.'
        for event in events:
            if event['event']!='result' or event['task_origin']!=job['origin']:continue
            req=event['request'];n=len(req['history'])
            assert req['history']==job['request']['history'][:n]
            assert req['timestamps']==job['request']['timestamps'][:n]
            assert max(req['timestamps'])<=req['cutoff']<min(req['future_timestamps'])
            assert req['past_covariates']==job['request']['past_covariates'][:n]
            if event['kind']=='backtest':
                assert event['actual']==job['request']['history'][n:n+14]
                assert abs(rmsle(event['point'],event['actual'])-event['metrics']['rmsle'])<1e-12
                assert req['future_covariates']==job['request']['past_covariates'][n:n+14]
                if event.get('ledger_score'):
                    assert event['ledger_score']['n']==14
                    assert abs(event['ledger_score']['mae']-event['metrics']['mae'])<1e-10
            else:
                assert event['actual'] is None and req['future_timestamps']==job['future_timestamps']
            checks+=1
        # Independent closure check: every past production result with a visible
        # host outcome must have exactly one matching maturation, selected or not.
        host_by_origin={j['origin']:j for j in jobs[r['series_id']]}
        matured={}
        for event in events:
            if event['event']=='matured':
                assert event['execution_id'] not in matured
                matured[event['execution_id']]=event
        expected_ids=set()
        for event in events:
            if event['event']!='result' or event['kind']!='forecast':continue
            past=host_by_origin[event['task_origin']]
            if datetime.fromisoformat(past['outcome_recorded_at'])>datetime.fromisoformat(job['origin']):continue
            eid=event['execution']['execution_id'];expected_ids.add(eid)
            value=matured[eid];req=event['request']
            assert req['series_id']==r['series_id'] and req['unit']==past['request']['unit']
            assert req['horizon']==14 and req['future_timestamps']==past['future_timestamps']
            assert req['cutoff']==past['origin']
            assert value['actual']==past['actual'] and value['point']==event['point']
            assert abs(value['metrics']['rmsle']-rmsle(event['point'],past['actual']))<1e-12
            assert datetime.fromisoformat(past['outcome_recorded_at'])<=datetime.fromisoformat(value['task_origin'])<=datetime.fromisoformat(job['origin'])
            if r['arm']=='ledger':
                assert value['ledger_score']['n']==14
                assert abs(value['ledger_score']['mae']-sum(abs(p-a) for p,a in zip(event['point'],past['actual'],strict=True))/14)<1e-10
            checks+=5
        assert set(matured)==expected_ids
        sync=json.loads((p.parent/'host-sync.json').read_text())
        assert sync['exit_code']==0
        sync_result=json.loads(sync['stdout'])
        assert sync_result['numerical_calls_started']==0 and sync_result['result']['provider_calls']==0
        r['matured_production_executions']=len(matured)
        r['matured_unselected_executions']=sum(not e['selected_for_submission'] for e in matured.values())
        checks+=2
        # Include every API request/error, not merely successful final conversations.
        receipt=[json.loads(f.read_text()) for f in p.parent.glob('api-*-receipt.json')]
        assert len(receipt)<=16
        orchestration=json.loads((p.parent/'orchestration.json').read_text())
        assert orchestration['corrections']<=2
        assert len(list(p.parent.glob('hermes-attempt-*.json')))==orchestration['attempts']
        for request in p.parent.glob('api-*-forwarded.json'):
            number=int(request.name.split('-')[1])
            if number>=13:
                intervention=json.loads(request.with_name(request.name.replace('-forwarded','-intervention')).read_text())
                assert intervention['phase']=='selection'
                assert json.loads(request.read_text())['messages'][-1]['content']==intervention['notice']
        checks+=3
        r['http_errors']=sum(v['status']!=200 for v in receipt)
        rows.append(r)
    arms={}
    for arm in ('plain','gnomon','ledger'):
        subset=[r for r in rows if r['arm']==arm]
        if not subset:continue
        arms[arm]={'tasks':len(subset),'valid':sum(r['valid'] for r in subset),
            'workflow_complete':sum(r['workflow_complete'] for r in subset),
            'corrections':sum(r['corrections'] for r in subset),
            'blocked_api_requests':sum(r['blocked_api_requests'] for r in subset),
            'orchestration_stops':dict(Counter(r['orchestration_stop'] for r in subset)),
            'baseline_checkpoint_only':sum(r['baseline_checkpoint_only'] for r in subset),
            'checkpoint_publications':sum(r['checkpoint_publications'] for r in subset),
            'review_calls':sum(r['review_calls'] for r in subset),
            'mean_rmsle':mean(r['rmsle'] for r in subset),
            'valid_only_rmsle':mean(r['rmsle'] for r in subset if r['valid']) if any(r['valid'] for r in subset) else None,
            'tokens':sum(r['tokens'] for r in subset),'api_calls':sum(r['api_calls'] for r in subset),
            'usage_complete':all(r['usage_complete'] and r['responses']==r['api_calls'] for r in subset),
            'seconds':sum(r['seconds'] for r in subset),'http_errors':sum(r['http_errors'] for r in subset),
            'ledger_reviews':sum(r['ledger_reviews'] for r in subset),
            'numerical_attempts':sum(r['numerical_attempts'] for r in subset),
            'numerical_successes':sum(r['numerical_successes'] for r in subset),
            'distinct_configs_per_task':mean(r['backtested_configurations'] for r in subset),
            'per_series':{s:mean(r['rmsle'] for r in subset if r['series_id']==s) for s in jobs if any(r['series_id']==s for r in subset)},
            'dollars':None}
        for name,test in [('cold',lambda r:r['round']<4),('later',lambda r:r['round']>=22)]:
            values=[r['rmsle'] for r in subset if test(r)]
            arms[arm][name+'_rmsle']=mean(values) if values else None
    keyed={(r['arm'],r['series_id'],r['round']):r for r in rows};contrasts={}
    for treatment,control in [('ledger','gnomon'),('ledger','plain'),('gnomon','plain')]:
        pairs=[(r,keyed[control,r['series_id'],r['round']]) for r in rows if r['arm']==treatment and (control,r['series_id'],r['round']) in keyed]
        if not pairs:continue
        base=mean(b['rmsle'] for a,b in pairs);new=mean(a['rmsle'] for a,b in pairs)
        series=sorted({a['series_id'] for a,b in pairs});rng=random.Random(142)
        boots=[]
        for _ in range(2000):
            chosen=[rng.choice(series) for _ in series]
            sample=[(a,b) for s in chosen for a,b in pairs if a['series_id']==s]
            boots.append(1-mean(a['rmsle'] for a,b in sample)/mean(b['rmsle'] for a,b in sample))
        boots.sort()
        contrasts[treatment+'_vs_'+control]={'pairs':len(pairs),'relative_rmsle_reduction':1-new/base,
            'wins':sum(a['rmsle']<b['rmsle']-1e-12 for a,b in pairs),
            'losses':sum(a['rmsle']>b['rmsle']+1e-12 for a,b in pairs),
            'exploratory_series_bootstrap_95_interval':[boots[49],boots[1949]],
            'warning':'Only four reused development series, one requested seed: not confirmatory.'}
    full_keys=[(s,n) for s,n in index if all(
        (a,s,n) in keyed and keyed[a,s,n]['valid'] and keyed[a,s,n]['workflow_complete']
        for a in ('plain','gnomon','ledger'))]
    successful_matched={'tasks_per_arm':len(full_keys),
        'tasks':[{'series_id':s,'round':n} for s,n in full_keys],
        'mean_rmsle':{a:mean(keyed[a,s,n]['rmsle'] for s,n in full_keys) if full_keys else None
                      for a in ('plain','gnomon','ledger')},
        'warning':'Selected successful subset; diagnostic only, not an unbiased overall treatment effect.'}
    value={'complete':len(rows)==manifest['planned'],'arms':arms,'contrasts':contrasts,
        'all_three_full_workflows':successful_matched,
        'audit_checks':checks,'audit_failures':failures,'rows':rows,
        'interpretation':'A supported ML iteration workflow comparison; no causal/business or general ledger superiority claim.'}
    dump(root/'report.json',value)
    lines=['# Checkpoint-first Hermes ML iteration — Gnomon 1.2.0','',
        'Matched development experiment. Full completion requires two distinct three-fold backtests, at least one ML model, and explicit selection of a backtested execution AFTER comparison. An initial baseline alone is not full completion.','',
        '| Arm | Forecasts valid | Workflows complete | Mean RMSLE | Tokens | Ledger reviews |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,a in arms.items():lines.append(f'| {arm} | {a["valid"]}/{a["tasks"]} | {a["workflow_complete"]}/{a["tasks"]} | {a["mean_rmsle"]:.6f} | {a["tokens"]:,} | {a["ledger_reviews"]} |')
    lines+=['','Invalid submissions include a predeclared last-value fallback. Valid-only metrics and every failed task remain in report.json. Sessions were not rerun; bounded within-session corrections are counted separately.','',
            f'All-three-completed subset: {len(full_keys)} matched tasks per arm; means {successful_matched["mean_rmsle"]}. This selected subset is diagnostic, not an unbiased overall effect.','']
    for name,c in contrasts.items():lines.append(f'- {name}: {c["relative_rmsle_reduction"]:.2%} relative RMSLE reduction across {c["pairs"]} pairs.')
    lines+=['',f'Independent checks: {checks}; integrity failures: {len(failures)}.',
        'Four reused development series and one requested seed cannot establish broad superiority. API dollar cost was not supplied. The ledger treatment includes a public-API review helper; it does not isolate the database from its usability support.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    r=analyze(args.root);print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
