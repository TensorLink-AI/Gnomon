"""One-shot full-span exploratory replay; includes the previously reserved period."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from benchmarks.online_retail_ii.tsfm.agent import load_case, resolve
from benchmarks.online_retail_ii.full_span.data import dump, sha
from benchmarks.online_retail_ii.tsfm.ledger import build_history
from benchmarks.online_retail_ii.full_span.models import forecast, metrics
from benchmarks.online_retail_ii.full_span.baselines import fingerprint, summarize
from benchmarks.online_retail_ii.tsfm.transport import proxy, SECONDS, REQUEST_LIMIT, MAX_TOKENS, MODEL

ARMS=('ledger_tsfm',)
SEEDS=(7,19)


def source_files(repo):
    own=list((repo/'benchmarks/online_retail_ii').rglob('*.py'))
    shared=[repo/'benchmarks/ledger_optimization'/n for n in ('hermes_boundary_093.py','execution_boundary_093.py')]
    shared.append(repo/'benchmarks/hermes_ml_checkpoint_v6/boundary_schemas_093.py')
    shared.append(repo/'benchmarks/hermes_ml_checkpoint_v6/hermes_boundary_093.py')
    shared.append(repo/'benchmarks/hermes_ml_checkpoint_v6/execution_boundary_093.py')
    return {str(p.relative_to(repo)):sha(p) for p in [*own,*shared]}


def credential(path):
    for line in Path(path).read_text().splitlines():
        if line.startswith('ENGY_API_KEY='):
            value=line.split('=',1)[1].strip().strip('"').strip("'")
            if value:return value
    raise ValueError('Engy credential unavailable')


def environment(home,repo,worker_python):
    allowed=('PATH','LANG','LC_ALL','TZ','HOME','USER','SHELL','SSL_CERT_FILE',
             'SSL_CERT_DIR','REQUESTS_CA_BUNDLE','CURL_CA_BUNDLE','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY')
    env={k:v for k,v in os.environ.items() if k in allowed}
    env.update(HERMES_HOME=str(home),PYTHONPATH=str(repo),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        HERMES_DISABLE_TELEMETRY='1',HERMES_DISABLE_LAZY_INSTALLS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',PATH=str(Path(worker_python).parent)+':'+env.get('PATH','/usr/bin:/bin'))
    return env


def execute_one(args,case,arm,seed,key,frozen,stub=False):
    repo=Path(args.repo); root=Path(args.output)
    task,history=load_case(case)
    base=root/f'seed-{seed}'/arm/task['series_id'];home=base/'home';out=base/task['origin']
    out.mkdir(parents=True);home.mkdir(exist_ok=True)
    if source_files(repo)!=frozen:raise ValueError('Frozen worker or numerical source changed')
    dump(home/'config.yaml',{'compression':{'enabled':False},'skills':{'external_dirs':[]},
        'memory':{'memory_enabled':True,'user_profile_enabled':True},'mcp_servers':{}})
    started=time.monotonic()
    if arm=='ledger_tsfm':build_history(case,out/'ledger')
    evidence_seconds=time.monotonic()-started
    cv=json.loads((case/'current-cv.json').read_text())
    matured=json.loads((case/'matured-outcomes.json').read_text())
    interface={'hermes':'Forecasts use the shared numerical functions directly. No Gnomon ledger interface.',
               'gnomon':'Forecasts use the same numerical functions through Gnomon 1.2.0. Ledger disabled.',
               'ledger_tsfm':'Forecasts use the same numerical functions through Gnomon 1.2.0. retail ledger returns verified lifetime/recent RMSLE and sample counts from the same matured outcomes.'}[arm]
    prompt=('Forecast recorded UK product sales, not latent demand. Select a model for the next 14 days. '
        'Twelve candidates are registered; only those marked available may execute now. Optimize RMSLE; smaller is better. Current CV uses up to three completed earlier 14-day folds when history permits. No CV exists at the first origin. Inspect model_availability: do not request a model before its minimum observed history. Scores null mean unavailable, not zero error. '
        'You may inspect history and matured raw outcomes with retail. Never infer stockouts, promotions or causal patterns from zeros alone. '
        'Use retail forecast with provider, then retail select with its execution_id. Four numerical attempts, '
        '12 model requests, 3072 output tokens per request, 480 seconds, at most two correction turns within the SAME budget. '
        'A single matching execution is recoverable even if the final response is prose; multiple executions require selection. '
        'When selected, finish concisely; optionally return {"execution_id":"..."}. '
        'Native text memory/skills persist for this product, arm and seed only; use them if useful and distinguish assumptions from facts. '
        +interface+' The two TSFM options are paracast_route and paracast_ensemble2 (top_k=2). You choose among these and the ten existing models. API medians are used as point forecasts. They are retained once per task and replayed through Gnomon within the same numerical selection budget. Training cutoffs are unknown; do not claim historical deployability. '+ '\n'+json.dumps({'task':task,'last28_sales':history.value.tail(28).tolist(),
          'matured_origins':len(matured['records']),'current_cv':{p:{'mean_rmsle':v['mean_rmsle'],'eligible':v['eligible']}
          for p,v in cv['candidates'].items()}},allow_nan=False))
    (out/'prompt.txt').write_text(prompt)
    dump(out/'inputs.json',{'case':str(case),'hashes':{p.name:sha(p) for p in case.iterdir() if p.is_file()},
                         'arm':arm,'seed':seed,'home':str(home)})
    deadline=time.time()+SECONDS
    worker_python=args.plain_python if arm=='hermes' else args.gnomon_python
    with proxy(out,key,seed,deadline,stub) as url:
        argv=[worker_python,'-m','benchmarks.online_retail_ii.tsfm.worker',
            '--case',str(case),'--output',str(out),'--numerical-python',sys.executable,'--repo',str(repo),
            '--arm',arm,'--base-url',url,'--prompt',str(out/'prompt.txt'),'--seed',str(seed),'--deadline',str(deadline)]
        dump(out/'command.json',{'argv':argv,'cwd':str(home),'timeout':SECONDS+45})
        with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
            process=subprocess.Popen(argv,cwd=home,env=environment(home,repo,worker_python),
                stdout=stdout,stderr=stderr,start_new_session=True)
            try:code=process.wait(timeout=SECONDS+45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:code=process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
        dump(out/'process.json',{'exit_code':code})
    selection=json.loads((out/'selection.json').read_text())['execution_id'] if (out/'selection.json').exists() else None
    resolution=resolve(case,out/'executions',selection)
    result=resolution.get('execution') if resolution['resolved'] else forecast('seasonal_naive_7',history.value,14,
        __import__('datetime').date.fromisoformat(task['origin']))
    events=[json.loads(line) for line in (out/'boundary-events.jsonl').read_text().splitlines()] if (out/'boundary-events.jsonl').exists() else []
    receipts=[json.loads(p.read_text()) for p in out.glob('api-*-receipt.json')]
    responses=[]
    for p in out.glob('api-*-response.json'):
        try:response=json.loads(p.read_text())
        except (ValueError,UnicodeError):response={}
        responses.append(response if isinstance(response,dict) else {})
    usage=[r.get('usage') for r in responses]
    ledger_calls=sum(e.get('stage')=='requested' and e.get('tool')=='retail'
        and json.loads(e['raw_arguments']).get('operation')=='ledger' for e in events
        if e.get('stage')=='requested' and e.get('tool')=='retail' and is_json_object(e.get('raw_arguments')))
    attempts=sorted(out.glob('hermes-attempt-*.json'))
    final=json.loads(attempts[-1].read_text()).get('final_response') if attempts else None
    try:
        parsed=json.loads(final) if isinstance(final,str) else None
        final_conformant=(isinstance(parsed,dict) and set(parsed)=={'execution_id'} and
                          resolution['resolved'] and parsed['execution_id']==result['execution_id'])
    except (ValueError,TypeError):final_conformant=False
    row={'case_id':task['case_id'],'origin':task['origin'],'series_id':task['series_id'],
        'arm':arm,'method':arm,'seed':seed,'resolved':resolution['resolved'],'resolution_status':resolution['status'],
        'execution_id':result.get('execution_id'),'selected_explicitly':selection is not None,
        'strict_final_conformant':bool(final_conformant),
        'fallback_used':not resolution['resolved'] or result['fallback_used'],'provider':result.get('provider'),
        'point':result['point'],'clipped_points':result['clipped_points'],'exit_code':code,
        'api_calls':len(receipts),'api_errors':sum(r['status']!=200 for r in receipts),
        'reported_tokens':sum(u.get('total_tokens',0) for u in usage if isinstance(u,dict)),
        'unknown_usage_calls':sum(not isinstance(u,dict) for u in usage),'ledger_queries':ledger_calls,
        'native_memory_calls':sum(e.get('stage')=='requested' and e.get('tool') in ('memory','skills_list','skill_view','skill_manage') for e in events),
        'seconds':time.monotonic()-started,'evidence_preparation_seconds':evidence_seconds,
        'stub':stub,'cost_dollars':None}
    dump(out/'result.json',row)
    if source_files(repo)!=frozen:raise ValueError('Source drift during execution')
    return row


def is_json_object(raw):
    try:return isinstance(json.loads(raw),dict)
    except (ValueError,TypeError):return False


def runtime_inventory(python):
    code='''import importlib.metadata as m,importlib.util as u,json,hashlib,pathlib
spec=u.find_spec("run_agent")
print(json.dumps({"packages":{d.metadata["Name"]:d.version for d in m.distributions()},
"hermes_source":spec.origin,"hermes_sha256":hashlib.sha256(pathlib.Path(spec.origin).read_bytes()).hexdigest(),
"gnomon_available":u.find_spec("gnomon") is not None}))'''
    return json.loads(subprocess.check_output([python,'-I','-c',code],text=True))


def run(args):
    from copy import copy
    from .client import Client,PROVIDERS,POLICIES,digest
    from .prepare import prepare_case
    repo=Path(args.repo).resolve();root=Path(args.output).resolve();baseline=Path(args.baseline).resolve()
    args.repo=str(repo);args.output=str(root)
    if root.exists():raise ValueError('Fresh output required')
    baseplan=json.loads((baseline/'plan.json').read_text())
    if baseplan['phase']!='full_span' or baseplan['planned_cases']!=2448 or baseplan['smoke']:raise ValueError('Wrong task set')
    for name,version in baseplan['runtime'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Numerical runtime drift: '+name)
    for name,d in baseplan['code_sha256'].items():
        if sha(repo/'benchmarks/online_retail_ii/full_span'/name)!=d:raise ValueError('Original baseline source drift: '+name)
    if sha(args.manifest)!=baseplan['source_manifest_sha256']:raise ValueError('Different cohort')
    frozen=source_files(repo);runtime=runtime_inventory(args.gnomon_python)
    reference_plan=json.loads(Path(args.reference_plan).read_text())
    if runtime!=reference_plan['runtime_inventory']['ledger']:raise ValueError('Agent runtime differs from existing ledger arm')
    from benchmarks.online_retail_ii.full_span.data import load_panel
    manifest,panel=load_panel(args.panel)
    cases=sorted((baseline/'agent-cases').iterdir());tasks={c:load_case(c)[0] for c in cases}
    if len(cases)!=2448 or manifest!=json.loads(Path(args.manifest).read_text()):raise ValueError('Incomplete panel')
    origins=sorted({t['origin'] for t in tasks.values()})
    actuals={};panels={s:f.sort_values('date').assign(day=lambda f:f.date.dt.strftime('%Y-%m-%d')) for s,f in panel.groupby('series_id')}
    for c,t in tasks.items():
        past=panels[t['series_id']];visible=past[past.day<=t['origin']]
        if fingerprint(t['series_id'],visible.value,visible.day,t['future_timestamps'])!=t['request_fingerprint']:raise ValueError('Task history mismatch')
        actuals[t['case_id']]=past[past.day.isin(t['future_timestamps'])].value.tolist()
    root.mkdir(parents=True);client=Client(args.chutes_key,root/'host-api')
    health=client.call('health');dump(root/'catalog-status.json',client.catalog())
    if health.get('status')!='ok':raise ValueError('TSFM service unhealthy')
    image=health.get('image');revision={p:'paracast/'+digest({'policy':POLICIES[p],'image':image,'endpoint':'chutes-direct','training_cutoff':'unknown'}) for p in PROVIDERS}
    plan={'model':MODEL,'gnomon':'1.2.0','arm':'ledger_tsfm','seeds':SEEDS,'planned_sessions':4896,
        'cases':2448,'periods':51,'source_hashes':frozen,'baseline_plan_sha256':sha(baseline/'plan.json'),
        'reference_plan_sha256':sha(args.reference_plan),'runtime_inventory':runtime,'policies':POLICIES,'revisions':revision,
        'initial_image':image,'feedback_enabled':False,'metric':'mean_per_case_rmsle','fresh_native_memory':True,
        'tsfm_point':'API 0.5 quantile clipped at zero','one_remote_call_per_case_per_policy':True,
        'historical_cv_reuses_origin_frozen_predictions':True,'same_remote_forecast_for_both_seeds':True,
        'limits':{'api_requests':REQUEST_LIMIT,'output_tokens_per_request':MAX_TOKENS,'seconds':SECONDS,'numerical_attempts':4},
        'interpretation':'additional-model-access arm, not an isolated ledger benefit test',
        'scope':'retrospective full-span exploratory; external pretraining overlap unknown'}
    dump(root/'plan.json',plan)
    pilot_series=[sorted(s for s,v in manifest['selected_products'].items() if v['stratum']==kind)[0] for kind in ('sparse','intermittent','frequent')]
    store={};prepared={};rows=[];done=set();api_scores=[]
    if args.reuse_preparation:
        previous=Path(args.reuse_preparation)
        if (previous/'scores.jsonl').exists() or list(previous.glob('seed-*')):raise ValueError('Preparation reuse requires zero prior agent attempts')
        previous_plan=json.loads((previous/'plan.json').read_text())
        if previous_plan['policies']!=POLICIES or previous_plan['revisions']!=revision or previous_plan['baseline_plan_sha256']!=sha(baseline/'plan.json'):
            raise ValueError('Preparation belongs to a different forecast protocol')
        for source in cases:
            old=previous/'cases'/source.name
            if not all((old/(p+'-forecast.json')).exists() for p in PROVIDERS):continue
            old_task,_=load_case(old)
            if old_task['request_fingerprint']!=tasks[source]['request_fingerprint']:raise ValueError('Preparation task identity changed')
            destination=root/'cases'/source.name;destination.parent.mkdir(parents=True,exist_ok=True)
            __import__('shutil').copytree(old,destination);prepared[source]=destination
            _,history=load_case(source)
            for provider in PROVIDERS:
                value=json.loads((old/(provider+'-forecast.json')).read_text())
                if value['point_sha256']!=digest(value['point']):raise ValueError('Preparation point changed')
                __import__('shutil').copytree(previous/'host-api'/value['api_receipt_label'],root/'host-api'/value['api_receipt_label'])
                store[old_task['series_id'],old_task['origin'],provider]=value
                score={'case_id':old_task['case_id'],'origin':old_task['origin'],'series_id':old_task['series_id'],'method':provider,**value,
                    'metrics':metrics(value['point'],actuals[old_task['case_id']],history.value)}
                api_scores.append(score)
                with (root/'tsfm-baseline-scores.jsonl').open('a') as f:f.write(json.dumps(score,allow_nan=False)+'\n')
        dump(root/'reused-preparation.json',{'source':str(previous),'cases':len(prepared),'forecast_calls_reused':len(api_scores),'agent_attempts_reused':0})
    def prepare(selected):
        missing=[c for c in selected if c not in prepared]
        if not missing:return
        if source_files(repo)!=frozen:raise ValueError('Source drift')
        snapshot_label='snapshot-'+str(len(prepared))
        current_health=client.call('health',label=snapshot_label+'-health')
        dump(root/(snapshot_label+'-catalog-status.json'),client.catalog(snapshot_label+'-models'))
        if current_health.get('status')!='ok' or current_health.get('image')!=image:raise ValueError('TSFM health/image changed mid-run')
        # At most two concurrent remote forecasts; one series per request so the
        # router cannot choose a model using another product's batch context.
        def one(c):
            destination=root/'cases'/c.name
            return c,prepare_case(c,destination,client,revision,store),destination
        with ThreadPoolExecutor(max_workers=2) as pool:
            for c,t,destination in pool.map(one,missing):
                prepared[c]=destination
                _,history=load_case(c)
                for provider in PROVIDERS:
                    value=store[t['series_id'],t['origin'],provider]
                    score={'case_id':t['case_id'],'origin':t['origin'],'series_id':t['series_id'],'method':provider,
                        **value,'metrics':metrics(value['point'],actuals[t['case_id']],history.value)}
                    api_scores.append(score)
                    with (root/'tsfm-baseline-scores.jsonl').open('a') as f:f.write(json.dumps(score,allow_nan=False)+'\n')
        failures=sum(not store[tasks[c]['series_id'],tasks[c]['origin'],p]['available'] for c in missing for p in PROVIDERS)
        dump(root/'preparation.json',{'prepared_cases':len(prepared),'planned_cases':2448,'last_batch_cases':len(missing),
            'last_batch_failed_remote_forecasts':failures,'remote_forecasts':len(api_scores),'tsfm_baselines':summarize(api_scores)})
        if failures>.1*len(missing)*len(PROVIDERS):raise ValueError('More than 10 percent TSFM failures; paid dispatch paused')
    def stage(selected):
        for day in origins:
            batch=[(c,seed) for c in selected if tasks[c]['origin']==day for seed in SEEDS if (c,seed) not in done]
            if not batch:continue
            prepare(sorted({c for c,s in batch}))
            # All dispatched results are retained even if one worker fails.
            errors=[]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures={pool.submit(execute_one,args,prepared[c],'ledger_tsfm',seed,engy,frozen,False):(c,seed) for c,seed in batch}
                for future in as_completed(futures):
                    c,seed=futures[future]
                    try:row=future.result()
                    except Exception as exc:errors.append({'case':c.name,'seed':seed,'cause':type(exc).__name__});continue
                    _,history=load_case(c);row['metrics']=metrics(row['point'],actuals[row['case_id']],history.value)
                    rows.append(row);done.add((c,seed))
                    with (root/'scores.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                    usage={p:sum(r['provider']==p for r in rows) for p in PROVIDERS}
                    dump(root/'progress.json',{'completed_sessions':len(rows),'planned_sessions':4896,'resolved':sum(r['resolved'] for r in rows),
                        'summary':summarize(rows),'selected_tsfm_providers':usage,'remote_forecasts':len(api_scores),
                        'credits_charged':sum(r.get('api_meta',{}).get('credits_charged',0) for r in api_scores),
                        'scope':plan['scope'],'stub':False})
                    print(json.dumps({'completed':len(rows),'case':row['case_id'],'seed':seed,'provider':row['provider'],'resolved':row['resolved']}),flush=True)
            if errors:
                dump(root/'dispatch-errors.json',errors);raise ValueError('Agent stage failed; completed futures retained')
    # Before paid agent calls, exercise both remote policies using real Hermes
    # and synthetic Engy replies, including a matured origin and native memory.
    pilot=[c for c in cases if tasks[c]['series_id'] in pilot_series and tasks[c]['origin'] in origins[:2]]
    for day in origins[:2]:prepare([c for c in pilot if tasks[c]['origin']==day])
    pre=copy(args);pre.output=str(root/'preflight');pre_rows=[]
    for day in origins[:2]:
        c=next(c for c in pilot if tasks[c]['series_id']==pilot_series[0] and tasks[c]['origin']==day)
        for seed in SEEDS:
            row=execute_one(pre,prepared[c],'ledger_tsfm',seed,'',frozen,True);pre_rows.append(row)
            expected=PROVIDERS[0] if seed==7 else PROVIDERS[1]
            assert row['resolved'] and row['exit_code']==0 and row['api_errors']==0 and row['native_memory_calls']>=1
            assert row['provider']==expected and row['point']==store[tasks[c]['series_id'],day,expected]['point']
    dump(root/'preflight/FINISHED.json',{'stub':True,'sessions':4,'resolved':4,'paid_engy_calls':0,'both_tsfm_policies_verified':True})
    engy=credential(args.credentials)
    stage(pilot)
    passed=len(rows)==12 and all(r['exit_code']==0 for r in rows) and sum(r['resolved'] for r in rows)>=.9*len(rows)
    dump(root/'pilot-gate.json',{'passed':passed,'sessions':len(rows),'accuracy_gate':False})
    if not passed:raise ValueError('New-arm pilot completion gate failed')
    stage(cases)
    dump(root/'FINISHED.json',{'sessions':len(rows),'resolved':sum(r['resolved'] for r in rows),'summary':summarize(rows),
        'tsfm_baselines':summarize(api_scores),'remote_forecasts':len(api_scores),'engy_calls':sum(r['api_calls'] for r in rows),
        'credits_charged':sum(r.get('api_meta',{}).get('credits_charged',0) for r in api_scores),'scope':plan['scope'],
        'objective_established':False})


def main():
    p=argparse.ArgumentParser()
    for name in ('repo','baseline','manifest','panel','output','gnomon-python','credentials','chutes-key','reference-plan'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--reuse-preparation')
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        if Path(args.output).exists():dump(Path(args.output)/'INCOMPLETE.json',{'cause':type(exc).__name__,'message':str(exc)})
        raise
if __name__=='__main__':main()
