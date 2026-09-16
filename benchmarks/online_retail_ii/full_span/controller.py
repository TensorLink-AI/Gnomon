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

from benchmarks.online_retail_ii.full_span.agent import load_case, resolve
from benchmarks.online_retail_ii.full_span.data import dump, sha
from benchmarks.online_retail_ii.full_span.ledger import build_history
from benchmarks.online_retail_ii.full_span.models import forecast, metrics
from benchmarks.online_retail_ii.full_span.baselines import fingerprint, summarize
from benchmarks.online_retail_ii.full_span.transport import proxy, SECONDS, REQUEST_LIMIT, MAX_TOKENS, MODEL

ARMS=('hermes','gnomon','ledger')
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
    if arm=='ledger':build_history(case,out/'ledger')
    evidence_seconds=time.monotonic()-started
    cv=json.loads((case/'current-cv.json').read_text())
    matured=json.loads((case/'matured-outcomes.json').read_text())
    interface={'hermes':'Forecasts use the shared numerical functions directly. No Gnomon ledger interface.',
               'gnomon':'Forecasts use the same numerical functions through Gnomon 1.2.0. Ledger disabled.',
               'ledger':'Forecasts use the same numerical functions through Gnomon 1.2.0. retail ledger returns verified lifetime/recent RMSLE and sample counts from the same matured outcomes.'}[arm]
    prompt=('Forecast recorded UK product sales, not latent demand. Select a model for the next 14 days. '
        'Ten candidates are registered; only those marked available may execute now. Optimize RMSLE; smaller is better. Current CV uses up to three completed earlier 14-day folds when history permits. No CV exists at the first origin. Inspect model_availability: do not request a model before its minimum observed history. Scores null mean unavailable, not zero error. '
        'You may inspect history and matured raw outcomes with retail. Never infer stockouts, promotions or causal patterns from zeros alone. '
        'Use retail forecast with provider, then retail select with its execution_id. Four numerical attempts, '
        '12 model requests, 3072 output tokens per request, 480 seconds, at most two correction turns within the SAME budget. '
        'A single matching execution is recoverable even if the final response is prose; multiple executions require selection. '
        'When selected, finish concisely; optionally return {"execution_id":"..."}. '
        'Native text memory/skills persist for this product, arm and seed only; use them if useful and distinguish assumptions from facts. '
        +interface+'\n'+json.dumps({'task':task,'last28_sales':history.value.tail(28).tolist(),
          'matured_origins':len(matured['records']),'current_cv':{p:{'mean_rmsle':v['mean_rmsle'],'eligible':v['eligible']}
          for p,v in cv['candidates'].items()}},allow_nan=False))
    (out/'prompt.txt').write_text(prompt)
    dump(out/'inputs.json',{'case':str(case),'hashes':{p.name:sha(p) for p in case.iterdir() if p.is_file()},
                         'arm':arm,'seed':seed,'home':str(home)})
    deadline=time.time()+SECONDS
    worker_python=args.plain_python if arm=='hermes' else args.gnomon_python
    with proxy(out,key,seed,deadline,stub) as url:
        argv=[worker_python,'-m','benchmarks.online_retail_ii.full_span.worker',
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
    repo=Path(args.repo).resolve(); args.repo=str(repo)
    root=Path(args.output).resolve();args.output=str(root)
    if root.exists():raise ValueError('Fresh one-shot output required; never overwrite/retry prior sessions')
    baseline=Path(args.baseline).resolve();plan=json.loads((baseline/'plan.json').read_text())
    if plan['phase']!='full_span' or plan['planned_cases']!=2448 or plan['smoke']:
        raise ValueError('Only the frozen 2448-case full-span exploratory replay is admitted')
    if json.loads((baseline/'report.json').read_text())['status']!='complete':raise ValueError('Baseline incomplete')
    for package,version in plan['runtime'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Numerical runtime mismatch: '+package)
    for name,digest in plan['code_sha256'].items():
        if sha(repo/'benchmarks/online_retail_ii/full_span'/name)!=digest:raise ValueError('Baseline adapter drift: '+name)
    cases=sorted((baseline/'agent-cases').iterdir())
    if len(cases)!=2448:raise ValueError('Incomplete case set')
    frozen=source_files(repo)
    manifest=json.loads(Path(args.manifest).read_text())
    if sha(args.manifest)!=plan['source_manifest_sha256']:
        raise ValueError('Selection manifest differs from the baseline plan')
    runtimes={a:runtime_inventory(args.plain_python if a=='hermes' else args.gnomon_python) for a in ARMS}
    if runtimes['hermes']['gnomon_available'] or not all(runtimes[a]['gnomon_available'] for a in ('gnomon','ledger')):
        raise ValueError('Hermes runtime registry separation failed')
    stripped=lambda r:{k.lower():v for k,v in r['packages'].items() if k.lower()!='gnomon-forecast'}
    if stripped(runtimes['hermes'])!=stripped(runtimes['gnomon']) or runtimes['hermes']['hermes_sha256']!=runtimes['gnomon']['hermes_sha256']:
        raise ValueError('Hermes runtime parity failed')
    products=manifest['selected_products']
    pilot_series=[sorted(s for s in products if products[s]['stratum']==kind)[0]
                  for kind in ('sparse','intermittent','frequent')]
    case_tasks={c:load_case(c)[0] for c in cases}
    origins=sorted({t['origin'] for t in case_tasks.values()})
    jobs=[(c,arm,seed) for day in origins for c in cases if case_tasks[c]['origin']==day for seed in SEEDS for arm in ARMS]
    pilot=[j for j in jobs if case_tasks[j[0]]['series_id'] in pilot_series and case_tasks[j[0]]['origin'] in origins[:2]]
    root.mkdir(parents=True)
    frozen_plan={'model':MODEL,'gnomon':'1.2.0','arms':ARMS,'seeds':SEEDS,'cases_per_arm_seed':2448,
        'planned_sessions':14688,'pilot_sessions':len(pilot),'pilot_series':pilot_series,'limits':{
        'api_requests':REQUEST_LIMIT,'output_tokens_per_request':MAX_TOKENS,'seconds':SECONDS,'numerical_attempts':4},
        'native_memory':'isolated_per_product_arm_seed_persistent_across_origins',
        'source_hashes':frozen,'runtime_inventory':runtimes,'baseline_plan_sha256':sha(baseline/'plan.json'),'stub':args.stub,
        'final_evidence_opened':True,'scope':'full_span_exploratory_not_untouched_final','gate':'all pilot workers exit 0 and at least 90 percent resolved; no accuracy gate'}
    dump(root/'plan.json',frozen_plan)
    rows=[];done=set()
    actuals={}
    # Host scores include predictions but not actual vectors; panel remains inaccessible to tools.
    from benchmarks.online_retail_ii.full_span.data import load_panel
    panel_manifest,panel=load_panel(args.panel)
    if panel_manifest!=manifest:raise ValueError('Host panel manifest mismatch')
    series_panels={s:g.sort_values('date').assign(day=g.sort_values('date').date.dt.strftime('%Y-%m-%d')) for s,g in panel.groupby('series_id')}
    for c in cases:
        task=case_tasks[c]
        series=series_panels[task['series_id']]
        past=series[series.day<=task['origin']]
        if task['request_fingerprint']!=fingerprint(task['series_id'],past.value,past.date.dt.strftime('%Y-%m-%d'),task['future_timestamps']):
            raise ValueError('Case differs from host source history')
        actuals[task['case_id']]=series[series.day.isin(task['future_timestamps'])].value.tolist()
    key='' if args.stub else credential(args.credentials)
    def stage(selected):
        for day in origins:
            batch=[j for j in selected if case_tasks[j[0]]['origin']==day and tuple(map(str,j)) not in done]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures={pool.submit(execute_one,args,*j,key,frozen,args.stub):j for j in batch}
                for future in as_completed(futures):
                    row=future.result();j=futures[future];_,history=load_case(j[0])
                    row['metrics']=metrics(row['point'],actuals[row['case_id']],history.value)
                    rows.append(row);done.add(tuple(map(str,j)))
                    with (root/'scores.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                    dump(root/'progress.json',{'completed_sessions':len(rows),'planned_sessions':14688,
                        'resolved':sum(r['resolved'] for r in rows),'summary':summarize(rows),
                        'stub':args.stub,'final_evidence_opened':True,'scope':'full_span_exploratory_not_untouched_final'})
                    print(json.dumps({'completed':len(rows),'case':row['case_id'],'arm':row['arm'],
                                      'seed':row['seed'],'resolved':row['resolved']}),flush=True)
    if args.stub:
        stage([j for j in pilot if j[2]==7 and case_tasks[j[0]]['series_id']==pilot_series[0]])
        if len(rows)!=6 or not all(r['resolved'] and r['exit_code']==0 for r in rows):
            raise ValueError('Real Hermes synthetic transport preflight failed')
        dump(root/'FINISHED.json',{'stub':True,'sessions':6,'resolved':6,'paid_calls':0})
        return
    stage(pilot)
    gate=all(r['exit_code']==0 for r in rows) and sum(r['resolved'] for r in rows)>=.9*len(rows)
    dump(root/'pilot-gate.json',{'passed':gate,'sessions':len(rows),'resolved':sum(r['resolved'] for r in rows)})
    if not gate:raise ValueError('Pilot interface quality gate failed; paid continuation stopped; retain every attempt')
    stage(jobs)
    dump(root/'FINISHED.json',{'sessions':len(rows),'resolved':sum(r['resolved'] for r in rows),
        'summary':summarize(rows),'stub':False,'api_calls':sum(r['api_calls'] for r in rows),
        'reported_tokens':sum(r['reported_tokens'] for r in rows),'unknown_usage_calls':sum(r['unknown_usage_calls'] for r in rows),
        'final_evidence_opened':True,'scope':'full_span_exploratory_not_untouched_final','objective_established':False})


def main():
    p=argparse.ArgumentParser()
    for name in ('repo','baseline','manifest','panel','output','plain-python','gnomon-python'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--credentials');p.add_argument('--stub',action='store_true');p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        if Path(args.output).exists():dump(Path(args.output)/'INCOMPLETE.json',{'cause':type(exc).__name__,'message':str(exc)})
        raise


if __name__=='__main__':main()
