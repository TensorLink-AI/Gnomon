"""Audited continuation of a stopped retail run; never rerun a completed session."""
import shutil
from .run import *
from . import run as original_controller
from benchmarks.online_retail_ii.resume_ledger import build_history
original_controller.build_history=build_history


def row_key(row):return (row['case_id'],row['arm'],row['seed'])


def recover_rows(root,cases,actuals):
    """Score every retained result, including futures completed after controller failure."""
    root=Path(root);indexed={load_case(c)[0]['case_id']:c for c in cases}
    prior=[json.loads(s) for s in (root/'scores.jsonl').read_text().splitlines()]
    existing={row_key(r):r for r in prior}
    if len(existing)!=len(prior):raise ValueError('Duplicate saved score')
    recovered=[];seen=set()
    for path in sorted(root.glob('seed-*/*/*/*/result.json')):
        result=json.loads(path.read_text());key=row_key(result)
        if key in seen:raise ValueError('Duplicate retained result')
        seen.add(key)
        if result['arm'] not in ARMS or result['seed'] not in SEEDS or result['case_id'] not in indexed:
            raise ValueError('Result outside original task set')
        case=indexed[result['case_id']];task,history=load_case(case)
        expected=root/f"seed-{result['seed']}"/result['arm']/task['series_id']/task['origin']
        if path.parent!=expected:raise ValueError('Result stored under wrong task')
        inputs=json.loads((path.parent/'inputs.json').read_text())
        if inputs['hashes']!={p.name:sha(p) for p in case.iterdir() if p.is_file()}:
            raise ValueError('Retained result input drift')
        if result['series_id']!=task['series_id'] or result['origin']!=task['origin'] or result['stub']:
            raise ValueError('Retained result identity mismatch')
        scored={**result,'metrics':metrics(result['point'],actuals[result['case_id']],history.value)}
        if key in existing:
            if scored!=existing[key]:raise ValueError('Saved score differs from retained result')
        else:recovered.append(scored)
    if not set(existing)<=seen:raise ValueError('Saved score lacks retained result')
    return prior,recovered


def run(args):
    if args.stub:raise ValueError('Use the dedicated zero-cost preflight; continuation is not a stub run')
    repo=Path(args.repo).resolve(); args.repo=str(repo)
    root=Path(args.output).resolve();args.output=str(root)
    if root.exists():raise ValueError('Fresh continuation root required')
    previous=Path(args.resume_from).resolve()
    if (previous/'FINISHED.json').exists():raise ValueError('Previous run already finished')
    previous_plan=json.loads((previous/'plan.json').read_text())
    if previous_plan['stub'] or previous_plan['planned_sessions']!=3744:raise ValueError('Wrong predecessor')
    original_gate=json.loads((previous/'pilot-gate.json').read_text())
    if not original_gate['passed'] or original_gate['sessions']!=36:raise ValueError('Original pilot gate did not pass')
    for name,digest in previous_plan['source_hashes'].items():
        if sha(repo/name)!=digest:raise ValueError('Original executed source changed: '+name)
    baseline=Path(args.baseline).resolve();plan=json.loads((baseline/'plan.json').read_text())
    if plan['phase']!='development' or plan['planned_cases']!=624 or plan['smoke']:
        raise ValueError('Only the frozen 624-case development run is admitted')
    if json.loads((baseline/'report.json').read_text())['status']!='complete':raise ValueError('Baseline incomplete')
    for package,version in plan['runtime'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Numerical runtime mismatch: '+package)
    for name,digest in plan['code_sha256'].items():
        if sha(repo/'benchmarks/online_retail_ii'/name)!=digest:raise ValueError('Baseline adapter drift: '+name)
    cases=sorted((baseline/'agent-cases').iterdir())
    if len(cases)!=624:raise ValueError('Incomplete case set')
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
    if runtimes!=previous_plan['runtime_inventory']:raise ValueError('Runtime changed since original execution')
    products=manifest['selected_products']
    pilot_series=[sorted(s for s in products if products[s]['stratum']==kind)[0]
                  for kind in ('sparse','intermittent','frequent')]
    case_tasks={c:load_case(c)[0] for c in cases}
    origins=sorted({t['origin'] for t in case_tasks.values()})
    jobs=[(c,arm,seed) for day in origins for c in cases if case_tasks[c]['origin']==day for seed in SEEDS for arm in ARMS]
    pilot=[j for j in jobs if case_tasks[j[0]]['series_id'] in pilot_series and case_tasks[j[0]]['origin'] in origins[:2]]
    shutil.copytree(previous,root)
    preserved=root/'pre-resume';preserved.mkdir()
    for name in ('plan.json','progress.json','pilot-gate.json','INCOMPLETE.json'):
        if (root/name).exists():shutil.move(root/name,preserved/name)
    frozen_plan={'model':MODEL,'gnomon':'1.2.0','arms':ARMS,'seeds':SEEDS,'cases_per_arm_seed':624,
        'planned_sessions':3744,'pilot_sessions':len(pilot),'pilot_series':pilot_series,'limits':{
        'api_requests':REQUEST_LIMIT,'output_tokens_per_request':MAX_TOKENS,'seconds':SECONDS,'numerical_attempts':4},
        'native_memory':'isolated_per_product_arm_seed_persistent_across_origins',
        'resumed_from':str(previous),'original_plan_sha256':sha(previous/'plan.json'),
        'source_hashes':frozen,'runtime_inventory':runtimes,'baseline_plan_sha256':sha(baseline/'plan.json'),'stub':args.stub,
        'final_evidence_opened':False,'gate':'all pilot workers exit 0 and at least 90 percent resolved; no accuracy gate'}
    dump(root/'plan.json',frozen_plan)
    rows=[];done=set()
    actuals={}
    # Host scores include predictions but not actual vectors; panel remains inaccessible to tools.
    from benchmarks.online_retail_ii.data import load_panel
    panel_manifest,panel=load_panel(args.panel)
    if panel_manifest!=manifest:raise ValueError('Host panel manifest mismatch')
    panels={s:f.sort_values('date').assign(day=lambda f:f.date.dt.strftime('%Y-%m-%d')) for s,f in panel.groupby('series_id')}
    for c in cases:
        task=case_tasks[c];series=panels[task['series_id']]
        past=series[series.day<=task['origin']]
        if task['request_fingerprint']!=fingerprint(task['series_id'],past.value,past.date.dt.strftime('%Y-%m-%d'),task['future_timestamps']):
            raise ValueError('Case differs from host source history')
        actuals[task['case_id']]=series[series.day.isin(task['future_timestamps'])].value.tolist()
    rows,recovered=recover_rows(root,cases,actuals)
    with (root/'scores.jsonl').open('a') as stream:
        for row in recovered:stream.write(json.dumps(row,allow_nan=False)+'\n')
    rows.extend(recovered)
    completed={row_key(r) for r in rows}
    done={tuple(map(str,j)) for j in jobs if (case_tasks[j[0]]['case_id'],j[1],j[2]) in completed}
    abandoned=[]
    for c,arm,seed in jobs:
        task=case_tasks[c];out=root/f'seed-{seed}'/arm/task['series_id']/task['origin']
        if (task['case_id'],arm,seed) in completed:continue
        if out.exists():
            # Only numerical-evidence preparation failures can be retried here.
            if any(out.glob('api-*')) or (out/'command.json').exists() or (out/'boundary-events.jsonl').exists():
                raise ValueError('Unresolved paid attempt requires explicit accounting; refusing rerun')
            dest=preserved/'aborted-preparation'/out.relative_to(root)
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(out,dest);abandoned.append(str(dest.relative_to(root)))
    dump(root/'resume-receipt.json',{'previous':str(previous),'previous_scores_sha256':sha(previous/'scores.jsonl'),
        'retained_sessions':len(rows),'recovered_unaggregated_sessions':len(recovered),'pending_sessions':3744-len(rows),
        'abandoned_preparation':abandoned,'no_paid_attempt_retried':True,'completed_scores_reverified':True})
    dump(root/'progress.json',{'completed_sessions':len(rows),'planned_sessions':3744,'resolved':sum(r['resolved'] for r in rows),
        'summary':summarize(rows),'stub':False,'final_evidence_opened':False,'resumed':True})
    key=credential(args.credentials)
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
                    dump(root/'progress.json',{'completed_sessions':len(rows),'planned_sessions':3744,
                        'resolved':sum(r['resolved'] for r in rows),'summary':summarize(rows),
                        'stub':args.stub,'final_evidence_opened':False})
                    print(json.dumps({'completed':len(rows),'case':row['case_id'],'arm':row['arm'],
                                      'seed':row['seed'],'resolved':row['resolved']}),flush=True)
    if args.stub:
        stage([j for j in pilot if j[2]==7 and case_tasks[j[0]]['series_id']==pilot_series[0]])
        if len(rows)!=6 or not all(r['resolved'] and r['exit_code']==0 for r in rows):
            raise ValueError('Real Hermes synthetic transport preflight failed')
        dump(root/'FINISHED.json',{'stub':True,'sessions':6,'resolved':6,'paid_calls':0})
        return
    stage(pilot)
    pilot_keys={(case_tasks[c]['case_id'],a,s) for c,a,s in pilot}
    pilot_rows=[r for r in rows if row_key(r) in pilot_keys]
    gate=len(pilot_rows)==36 and all(r['exit_code']==0 for r in pilot_rows) and sum(r['resolved'] for r in pilot_rows)>=.9*len(pilot_rows)
    dump(root/'pilot-gate.json',{'passed':gate,'sessions':len(pilot_rows),'resolved':sum(r['resolved'] for r in pilot_rows),'retained_original_sessions':True})
    if not gate:raise ValueError('Pilot interface quality gate failed; paid continuation stopped; retain every attempt')
    stage(jobs)
    dump(root/'FINISHED.json',{'sessions':len(rows),'resolved':sum(r['resolved'] for r in rows),
        'summary':summarize(rows),'stub':False,'api_calls':sum(r['api_calls'] for r in rows),
        'reported_tokens':sum(r['reported_tokens'] for r in rows),'unknown_usage_calls':sum(r['unknown_usage_calls'] for r in rows),
        'final_evidence_opened':False,'objective_established':False})


def main():
    p=argparse.ArgumentParser()
    for name in ('repo','baseline','manifest','panel','output','plain-python','gnomon-python'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--resume-from',required=True)
    p.add_argument('--credentials');p.add_argument('--stub',action='store_true');p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        if Path(args.output).exists():dump(Path(args.output)/'INCOMPLETE.json',{'cause':type(exc).__name__,'message':str(exc)})
        raise


if __name__=='__main__':main()
