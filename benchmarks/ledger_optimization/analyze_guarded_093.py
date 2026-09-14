"""Local post-run audit with explicit undefined zero-control contrasts.

Derived from the frozen guarded-v6 auditor. Session checks, scores and bootstrap
sampling are unchanged; zero denominators no longer prevent preserving an audit.
Do not deploy over the live frozen runner.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from collections import Counter
from statistics import mean
from datetime import datetime

from benchmarks.hermes_ml_checkpoint_v6.transport import dump,sha


def paired_series_contrast(pairs):
    """Keep every draw; undefined relative draws make the relative interval null."""
    if not pairs:
        raise ValueError('At least one matched pair is required')
    for a,b in pairs:
        if a['series_id'] != b['series_id']:
            raise ValueError('Pair series identity must match')
        for row in (a,b):
            score=row['rmsle']
            if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or score<0:
                raise ValueError('RMSLE must be finite and nonnegative')
    base=mean(b['rmsle'] for a,b in pairs);new=mean(a['rmsle'] for a,b in pairs)
    series=sorted({a['series_id'] for a,b in pairs});rng=random.Random(142)
    boots=[];absolute=[];undefined=0
    for _ in range(2000):
        chosen=[rng.choice(series) for _ in series]
        sample=[(a,b) for s in chosen for a,b in pairs if a['series_id']==s]
        control=mean(b['rmsle'] for a,b in sample)
        treatment=mean(a['rmsle'] for a,b in sample)
        absolute.append(control-treatment)
        if control==0:
            undefined+=1
        else:
            boots.append(1-treatment/control)
    boots.sort();absolute.sort()
    return {'pairs':len(pairs),'relative_rmsle_reduction':1-new/base if base else None,
        'relative_rmsle_status':'defined' if base else 'undefined_zero_control_mean',
        'wins':sum(a['rmsle']<b['rmsle']-1e-12 for a,b in pairs),
        'losses':sum(a['rmsle']>b['rmsle']+1e-12 for a,b in pairs),
        'exploratory_series_bootstrap_95_interval':[boots[49],boots[1949]] if undefined==0 else None,
        'relative_interval_status':'defined' if undefined==0 else 'undefined_zero_control_resample',
        'bootstrap_draws':2000,'undefined_relative_draws':undefined,
        'absolute_rmsle_reduction':base-new,
        'exploratory_absolute_series_bootstrap_95_interval':[absolute[49],absolute[1949]],
        'warning':'Only four reused development series, one requested seed: not confirmatory. No zero-control draw was discarded or assigned a relative improvement. Absolute differences are diagnostic, not a substitute target.'}


def rmsle(point,actual):
    assert len(point)==len(actual)==14
    return math.sqrt(sum((math.log1p(p)-math.log1p(a))**2 for p,a in zip(point,actual,strict=True))/14)


def audit_orchestration(folder, row):
    """Keep process termination distinct from independently validated checkpoints."""
    path = folder / 'orchestration.json'
    if path.exists():
        value = json.loads(path.read_text())
        assert value['corrections'] <= 2
        assert len(list(folder.glob('hermes-attempt-*.json'))) == value['attempts']
        return {'status': 'shutdown_record_present'}
    process = json.loads((folder / 'process.json').read_text())
    command = json.loads((folder / 'command.json').read_text())
    assert process['exit_code'] == row['exit_code'] and row['exit_code'] in (-15, -9)
    assert row['seconds'] >= command['timeout'] == 520
    assert row['orchestration_stop'] == 'worker_failed' and row['corrections'] == 0
    assert not list(folder.glob('hermes-attempt-*.json'))
    assert not (folder / 'corrections.jsonl').exists()
    requests = list(folder.glob('api-*-request.json'))
    assert 1 <= len(requests) <= 16
    # In this supported recovery there was exactly one user turn, not hidden
    # corrective attempts. Do not invent a successful orchestration transcript.
    for request in requests:
        messages = json.loads(request.read_text())['messages']
        assert sum(m.get('role') == 'user' for m in messages) == 1
    return {'status': 'missing_shutdown_record_after_worker_termination',
            'exit_code': row['exit_code'], 'observed_requests': len(requests),
            'single_attempt_verified_from_wire': True,
            'checkpoint_scoring_changed': False}


def analyze(root, output=None):
    root = Path(root)
    output = Path(output) if output is not None else root
    output.mkdir(parents=True, exist_ok=True)
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
        assert sync['status']=='ok' and sync['result']['exit_code']==0
        sync_result=json.loads(sync['result']['stdout'])
        assert sync_result['numerical_calls_started']==0 and sync_result['result']['provider_calls']==0
        r['matured_production_executions']=len(matured)
        r['matured_unselected_executions']=sum(not e['selected_for_submission'] for e in matured.values())
        checks+=2
        # The original counter-only audit missed arbitrary terminal modelling.
        # Require the guarded surface and reconcile every dispatched lab result.
        tools=json.loads((p.parent/'tools.json').read_text())
        assert {t['function']['name'] for t in tools}=={
            'lab','evidence_read','project_list','notes_write','data_summary',
            'memory','skills_list','skill_view','skill_manage'}
        boundary=[json.loads(line) for line in (p.parent/'boundary-events.jsonl').read_text().splitlines()]
        requested=[e for e in boundary if e['stage']=='requested']
        returned=[e for e in boundary if e['stage']=='returned']
        assert [(e['tool_call_id'],e['tool']) for e in requested]==[(e['tool_call_id'],e['tool']) for e in returned]
        guarded_fits=0
        for event in returned:
            result=event['result']
            if event['tool']=='lab' and result['status']=='ok':
                assert result['result']['execution_scope']=='verified_metered_lab'
                guarded_fits+=json.loads(result['result']['stdout'])['numerical_calls_started']
        assert guarded_fits==attempts
        checks+=4
        # Audit the complete evidence referenced by each current-origin ledger review.
        reviews=project/'review-log.jsonl'
        if reviews.exists():
            originals={e['execution']['execution_id']:e for e in events if e['event']=='result'}
            for line in reviews.read_text().splitlines():
                review=json.loads(line)
                if review['task_origin']!=job['origin']:continue
                evidence=project/review['full_evidence']['path']
                assert sha(evidence)==review['full_evidence']['sha256']
                full=json.loads(evidence.read_text())
                assert full['metric']=='rmsle' and full['as_of']==job['origin'] and full['provider_calls']==0
                for card in full['cards']:
                    for window in card['windows'].values():
                        recomputed={provider:[] for provider in card['providers']}
                        for origin in window['origins']:
                            for model in origin['models']:
                                event=originals[model['execution_id']]
                                assert event['kind']=='forecast'
                                old_job=host_by_origin[event['task_origin']]
                                assert datetime.fromisoformat(old_job['outcome_recorded_at'])<=datetime.fromisoformat(job['origin'])
                                assert datetime.fromisoformat(origin['origin'])==datetime.fromisoformat(old_job['origin'])
                                assert event['execution']['provider']==model['provider'] and event['execution']['revision']==model['revision']
                                score=rmsle(event['point'],old_job['actual'])
                                assert abs(score-model['rmsle'])<1e-12
                                recomputed[model['provider']].append(score);checks+=5
                        assert all(len(v)==window['matched_origins'] for v in recomputed.values())
                        for rank in window.get('ranking',[]):
                            score=mean(recomputed[rank['provider']]);assert abs(score-rank['score'])<1e-12
                            assert rank['rank']==1+sum(v['score']<rank['score'] for v in window['ranking']);checks+=2
                checks+=1
        # Readiness is task-free, before agent execution, separately metered.
        admission=p.parent/'service-admission'
        admitted=json.loads((admission/'status.json').read_text())
        assert admitted['ready'] and not admitted['agent_started'] and admitted['agent_requests']==0
        assert 1<=admitted['probe_count']<=10
        canary={'model':'deepseek-v4.1-flash','messages':[{'role':'user','content':'Reply with the word READY.'}],
                'temperature':0,'seed':7,'max_tokens':16,'stream':False}
        probe_receipts=[]
        for request in sorted(admission.glob('probe-*-request.json')):
            assert json.loads(request.read_text())==canary
            probe_receipts.append(json.loads(request.with_name(request.name.replace('-request','-receipt')).read_text()))
            checks+=1
        assert len(probe_receipts)==admitted['probe_count']
        r['admission_api_calls']=len(probe_receipts)
        r['admission_reported_tokens']=sum((x['usage'] or {}).get('total_tokens',0) for x in probe_receipts)
        r['admission_wait_seconds']=admitted['known_wait_seconds']+sum(x['seconds'] for x in probe_receipts)
        checks+=3
        # Include every API request/error, not merely successful final conversations.
        receipt=[json.loads(f.read_text()) for f in p.parent.glob('api-*-receipt.json')]
        assert len(receipt)<=16
        r['orchestration_audit']=audit_orchestration(p.parent,r)
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
            'admission_api_calls':sum(r['admission_api_calls'] for r in subset),
            'admission_reported_tokens':sum(r['admission_reported_tokens'] for r in subset),
            'admission_wait_seconds':sum(r['admission_wait_seconds'] for r in subset),
            'tokens':sum(r['tokens'] for r in subset),'api_calls':sum(r['api_calls'] for r in subset),
            'usage_complete':all(r['usage_complete'] and r['responses']==r['api_calls'] for r in subset),
            'seconds':sum(r['seconds'] for r in subset),'http_errors':sum(r['http_errors'] for r in subset),
            'ledger_reviews':sum(r['ledger_reviews'] for r in subset),
            'numerical_attempts':sum(r['numerical_attempts'] for r in subset),
            'numerical_successes':sum(r['numerical_successes'] for r in subset),
            'distinct_configs_per_task':mean(r['backtested_configurations'] for r in subset),
            'per_series':{s:mean(r['rmsle'] for r in subset if r['series_id']==s) for s in jobs if any(r['series_id']==s for r in subset)},
            'dollars':None}
        for name,test in [('cold',lambda r:r['round']<4),('mature',lambda r:r['round']>=10),('later',lambda r:r['round']>=22)]:
            values=[r['rmsle'] for r in subset if test(r)]
            arms[arm][name+'_rmsle']=mean(values) if values else None
    keyed={(r['arm'],r['series_id'],r['round']):r for r in rows};contrasts={}
    for treatment,control in [('ledger','gnomon'),('ledger','plain'),('gnomon','plain')]:
        pairs=[(r,keyed[control,r['series_id'],r['round']]) for r in rows if r['arm']==treatment and (control,r['series_id'],r['round']) in keyed]
        if not pairs:continue
        contrasts[treatment+'_vs_'+control]=paired_series_contrast(pairs)
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
    value['shutdown_record_gaps']=[{'arm':r['arm'],'series_id':r['series_id'],'round':r['round'],
        **r['orchestration_audit']} for r in rows if r['orchestration_audit']['status']!='shutdown_record_present']
    dump(output/'report.json',value)
    lines=['# Checkpoint-first Hermes ML iteration — Gnomon 1.2.0','',
        'Matched development experiment. Full completion requires two distinct three-fold backtests, at least one ML model, and explicit selection of a backtested execution AFTER comparison. An initial baseline alone is not full completion.','',
        '| Arm | Forecasts valid | Workflows complete | Mean RMSLE | Tokens | Ledger reviews |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,a in arms.items():lines.append(f'| {arm} | {a["valid"]}/{a["tasks"]} | {a["workflow_complete"]}/{a["tasks"]} | {a["mean_rmsle"]:.6f} | {a["tokens"]:,} | {a["ledger_reviews"]} |')
    lines+=['','Invalid submissions include a predeclared last-value fallback. Valid-only metrics and every failed task remain in report.json. Sessions were not rerun; bounded within-session corrections are counted separately.','',
            f'All-three-completed subset: {len(full_keys)} matched tasks per arm; means {successful_matched["mean_rmsle"]}. This selected subset is diagnostic, not an unbiased overall effect.','']
    for name,c in contrasts.items():
        relative=c['relative_rmsle_reduction']
        label=f'{relative:.2%}' if relative is not None else 'undefined (zero control mean)'
        lines.append(f'- {name}: {label} relative RMSLE reduction across {c["pairs"]} pairs.')
    lines+=['',f'Independent checks: {checks}; integrity failures: {len(failures)}.',
        'Four reused development series and one requested seed cannot establish broad superiority. API dollar cost was not supplied. The ledger treatment includes the separately frozen corrected comparison and brief review helper; it does not isolate the database from its usability support.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    r=analyze(args.root);print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
