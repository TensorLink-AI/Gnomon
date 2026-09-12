"""Fresh interleaved Hermes ML iteration trial; target values remain host-only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
from statistics import mean
import subprocess
import threading
import time

from .transport import proxy,dump,sha,MODEL

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OTHER=Path(os.environ.get('LEDGER_ML_RUNTIME_ROOT', '/root/Gnomon/results/hermes-ml-runtime-v3')).resolve()
ARMS=('plain','gnomon','ledger')
SOURCE_SHA='836e05229535f144acab46e94d48fb0fbddf2acc9b9bf4696369f5149efb9ec9'
BUILD_SHA='9723394ccb6d9e11991b312e01bac47c767c69407b6b33d36971cb6e48b6a22e'
LOCK=threading.Lock()


def key():
    for line in (REPO/'.env').read_text().splitlines():
        if line.startswith('ENGY_API_KEY='):return line.split('=',1)[1].strip().strip('"').strip("'")
    raise RuntimeError('Engy credential unavailable')


def environment(home,work,python):
    names=('PATH','LANG','LC_ALL','TZ','HOME','USER','SHELL','SSL_CERT_FILE',
           'SSL_CERT_DIR','REQUESTS_CA_BUNDLE','CURL_CA_BUNDLE','HTTP_PROXY',
           'HTTPS_PROXY','ALL_PROXY','NO_PROXY')
    env={k:os.environ[k] for k in names if k in os.environ}
    env.update(HERMES_HOME=str(home),TERMINAL_CWD=str(work),
        PATH=str(python.parent)+':'+env.get('PATH','/usr/bin:/bin'),
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HERMES_DISABLE_TELEMETRY='1',
        HERMES_DISABLE_LAZY_INSTALLS='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    return env


def runtime_inventory():
    runtimes={a:OTHER/('plain-venv' if a=='plain' else 'gnomon-venv')/'bin/python' for a in ARMS}
    code='import json,importlib.metadata,importlib.util;print(json.dumps({"gnomon":importlib.util.find_spec("gnomon") is not None,"packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}}))'
    inventory={a:json.loads(subprocess.check_output([str(py),'-I','-c',code],text=True)) for a,py in runtimes.items()}
    assert inventory['plain']['gnomon'] is False and inventory['gnomon']['gnomon'] is True
    strip=lambda d:{k:v for k,v in d['packages'].items() if k.lower()!='gnomon-forecast'}
    assert strip(inventory['plain'])==strip(inventory['gnomon'])==strip(inventory['ledger']), 'Runtime package parity failed'
    return inventory


def prepare(work,job,prior,arm):
    req=job['request'];now=datetime.fromisoformat(job['origin'])
    assert len(req['history'])==len(req['timestamps'])==730
    assert max(map(datetime.fromisoformat,req['timestamps']))<=now
    assert min(map(datetime.fromisoformat,req['future_timestamps']))>now
    assert all(datetime.fromisoformat(p['outcome_recorded_at'])<=now for p in prior)
    with (work/'history.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['timestamp','value']+req['past_covariate_names'])
        w.writerows([t,y,*c] for t,y,c in zip(req['timestamps'],req['history'],req['past_covariates'],strict=True))
    with (work/'future.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['timestamp']+req['future_covariate_names'])
        w.writerows([t,*c] for t,c in zip(req['future_timestamps'],req['future_covariates'],strict=True))
    dump(work/'task.json',{'series_id':job['series_id'],'origin':job['origin'],'round':job['round'],
        'horizon':14,'unit':req['unit'],'future_timestamps':req['future_timestamps'],
        'availability':'Synthetic replay: source at valid time, recording at horizon close, not measured vintages.'})
    dump(work/'previous_runs.json',prior);dump(work/'backend.json',{'arm':arm})
    for n in ('lab.py','core.py','numerical.py','check_forecast.py','policy.py','maturation.py'):shutil.copyfile(HERE/n,work/n)
    extra={'plain':'Gnomon is absent. Your executions and metrics are persisted in experiments.jsonl; lab.py review summarizes those same raw numerical facts.',
        'gnomon':'Gnomon 1.2.0 executes the shared models through its typed public API. Ledger is disabled. Your executions and metrics persist in experiments.jsonl; lab.py review summarizes those raw facts.',
        'ledger':'Gnomon 1.2.0 executes the same models and persists them and visible actuals in TemporalLedger. Run `python lab.py review` to retrieve saved metrics and matched-origin comparisons through public ledger APIs. Review your accumulated evidence before iterating. Raw experiments.jsonl and previous_runs.json contain the same numerical facts. Ledger access uses your normal time/call budget. Do not modify ledger.db directly.'}[arm]
    (work/'TASK.md').write_text((HERE/'TASK.md').read_text()+'\n'+extra+'\n')
    for name in ('forecast.json','execution.json','decision.json','checkpoint.json'):(work/name).unlink(missing_ok=True)


def command(args,cwd,env,out,timeout=520):
    dump(out/'command.json',{'argv':args,'cwd':str(cwd),'timeout':timeout})
    with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
        p=subprocess.Popen(args,cwd=cwd,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
    dump(out/'process.json',{'exit_code':code})
    return code


def records(work):
    p=work/'experiments.jsonl'
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def assess(work,job,protected,prefix):
    from .check_forecast import validate
    forecast=None;cause=None;binding=None;selection=None;comparison_at_selection=False
    runs=[r for r in records(work) if r['event']=='result' and r['task_origin']==job['origin']]
    try:
        forecast=json.loads((work/'checkpoint.json').read_text())
        cause=validate(forecast,14)
        if not (work/'experiments.jsonl').read_bytes().startswith(prefix):cause='Previous evidence modified'
        if any(not (work/n).exists() or sha(work/n)!=h for n,h in protected.items()):cause='Protected input modified'
        matches=[r for r in runs if r['kind']=='forecast' and r['execution']['execution_id']==forecast.get('execution_id')]
        if len(matches)!=1:cause='Final forecast not bound to one executed configuration'
        else:
            binding=matches[0]
            if (binding['point']!=forecast['point'] or binding['request']['future_timestamps']!=job['future_timestamps']
                    or forecast['future_timestamps']!=job['future_timestamps']
                    or forecast['series_id']!=job['series_id'] or forecast['task_origin']!=job['origin']
                    or forecast['unit']!=job['request']['unit'] or forecast['config_id']!=binding['config_id']):
                cause='Submitted points or task identity differ from execution'
        from uuid import UUID
        UUID(forecast['checkpoint_id'])
        immutable=json.loads((work/'checkpoints'/(forecast['checkpoint_id']+'.json')).read_text())
        if immutable!=forecast:cause='Checkpoint differs from immutable saved record'
        fingerprint=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if binding and forecast['request_fingerprint']!=fingerprint(binding['request']):cause='Request fingerprint mismatch'
        folds={};configs={}
        expected={job['request']['timestamps'][i-1] for i in (688,702,716)}
        for event in records(work):
            if event.get('task_origin')!=job['origin']:continue
            if event['event']=='result' and event['kind']=='backtest':
                folds.setdefault(event['config_id'],set()).add(event['request']['cutoff'])
                configs[event['config_id']]=event['config']
            elif event['event']=='selection' and event['checkpoint_id']==forecast['checkpoint_id']:
                selection=event
                eligible={cid for cid,times in folds.items() if times==expected}
                comparison_at_selection=(len(eligible)>=2 and any(configs[c]['model']!='seasonal' for c in eligible)
                                         and forecast['config_id'] in eligible)
                if (event['checkpoint_sha256']!=fingerprint(forecast)
                        or sorted(eligible)!=event['compared_configurations']
                        or event['execution_id']!=forecast['execution_id']
                        or event['selection_after_comparison']!=comparison_at_selection):
                    cause='Selection record does not match the evidence available at selection'
        if selection is None:cause='No explicit selection record for checkpoint'
        if forecast['selection_after_comparison']!=comparison_at_selection:cause='Checkpoint completion claim inconsistent with selection evidence'
    except (OSError,ValueError,TypeError,KeyError) as exc:cause=type(exc).__name__+': '+str(exc)
    valid=cause is None
    point=forecast['point'] if valid else [job['request']['history'][-1]]*14
    backtests={}
    for r in runs:
        if r['kind']=='backtest':backtests.setdefault(r['config_id'],set()).add(r['request']['cutoff'])
    complete_ids={c for c,v in backtests.items() if len(v)==3}
    ml=any(r['config_id'] in complete_ids and r['config']['model']!='seasonal' for r in runs)
    workflow=bool(valid and len(complete_ids)>=2 and ml and binding['config_id'] in complete_ids and comparison_at_selection)
    score=math.sqrt(mean((math.log1p(p)-math.log1p(a))**2 for p,a in zip(point,job['actual'],strict=True)))
    review_path=work/'review_calls.jsonl'
    reviews=[json.loads(line) for line in review_path.read_text().splitlines()] if review_path.exists() else []
    return {'valid':valid,'workflow_complete':workflow,'baseline_checkpoint_only':valid and not workflow,
        'selection_after_comparison':comparison_at_selection,'fallback_used':not valid,'cause':cause,
        'point':point,'rmsle':score,'backtested_configurations':len(complete_ids),
        'execution_id':binding['execution']['execution_id'] if valid else None,
        'config':binding['config'] if valid else None,
        'numerical_successes':len(runs),'numerical_attempts':sum(r['event']=='attempt' and r['task_origin']==job['origin'] for r in records(work)),
        'checkpoint_publications':sum(r['event']=='selection' and r['task_origin']==job['origin'] for r in records(work)),
        'review_calls':sum(r['task_origin']==job['origin'] for r in reviews),
        'ledger_reviews':sum(r['task_origin']==job['origin'] and r['ledger_queries']>0 for r in reviews)}


def execute(root,arm,series,job,prior,python,api_key):
    base=root/arm/series;work=base/'work';home=base/'home';out=base/f'round-{job["round"]}'
    out.mkdir(parents=True);work.mkdir(exist_ok=True);home.mkdir(exist_ok=True)
    dump(home/'config.yaml',{'terminal':{'backend':'local','cwd':str(work)},'compression':{'enabled':False},
        'memory':{'memory_enabled':True,'user_profile_enabled':True},'agent':{'max_turns':16},'mcp_servers':{}})
    prepare(work,job,prior,arm)
    env=environment(home,work,python)
    sync=subprocess.run([str(python),'lab.py','sync'],cwd=work,env=env,capture_output=True,text=True)
    dump(out/'host-sync.json',{'exit_code':sync.returncode,'stdout':sync.stdout,'stderr':sync.stderr})
    if sync.returncode:raise RuntimeError('Host visibility synchronization failed; see '+str(out))
    protected={n:sha(work/n) for n in ('task.json','history.csv','future.csv','previous_runs.json','lab.py','core.py','numerical.py','backend.json','check_forecast.py','policy.py','maturation.py')}
    prefix=(work/'experiments.jsonl').read_bytes() if (work/'experiments.jsonl').exists() else b''
    dump(out/'input-hashes.json',protected)
    dump(out/'prior-evidence.json',{'bytes':len(prefix),'sha256':hashlib.sha256(prefix).hexdigest()})
    started=time.monotonic()
    with proxy(out,api_key,work=work,deadline=time.time()+480) as url:
        code=command([str(python),str(HERE/'worker.py'),str(work),str(out),url,'alone'],work,env,out)
    row={'arm':arm,'series_id':series,'round':job['round'],'origin':job['origin'],
         **assess(work,job,protected,prefix),'seconds':time.monotonic()-started,'exit_code':code}
    responses=[]
    for p in out.glob('api-*-response.json'):
        try:responses.append(json.loads(p.read_text()))
        except ValueError:responses.append({'error':'invalid_response'})
    usages=[r.get('usage') or {} for r in responses]
    row.update(api_calls=len(list(out.glob('api-*-forwarded.json'))),responses=len(responses),
        tokens=sum(u.get('total_tokens',0) for u in usages),usage_complete=bool(usages) and all('total_tokens' in u for u in usages),
        api_errors=sum(bool(r.get('error')) for r in responses),
        prior_outcomes=len(prior),decision_summary_present=(work/'decision.json').exists())
    orchestration=json.loads((out/'orchestration.json').read_text()) if (out/'orchestration.json').exists() else {}
    row.update(corrections=orchestration.get('corrections',0),
               orchestration_stop=orchestration.get('stop_reason','worker_failed'),
               blocked_api_requests=len(list(out.glob('blocked-request-*.json'))),
               selection_interventions=len(list(out.glob('api-*-intervention.json'))))
    dump(out/'grade.json',row)
    for p in work.rglob('*'):
        if p.is_file() and not p.is_symlink() and not any(x.startswith('.') or x=='__pycache__' for x in p.relative_to(work).parts):
            target=out/'project'/p.relative_to(work);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    dump(out/'memory.json',{p.name:p.read_text() for p in (home/'memories').glob('*.md')})
    prior.append({'series_id':job['series_id'],'unit':job['request']['unit'],'origin':job['origin'],'future_timestamps':job['future_timestamps'],
        'outcome_recorded_at':job['outcome_recorded_at'],'point':row['point'],'actual':job['actual'],
        'execution_id':row['execution_id'],'config':row['config'],'fallback_used':row['fallback_used'],'rmsle':row['rmsle']})
    with LOCK:
        completed=len(list(root.glob('*/*/round-*/grade.json')))
        dump(root/'status.json',{'completed':completed,'planned':json.loads((root/'manifest.json').read_text())['planned'],
             'at':datetime.now(timezone.utc).isoformat(),'last':{k:row[k] for k in ('arm','series_id','round','valid','workflow_complete')}})
        print(json.dumps({'completed':completed,**{k:row[k] for k in ('arm','series_id','round','valid','workflow_complete')}}),flush=True)
    return row


def chain(index,series,jobs,root,runtimes,api_key):
    prior={a:[] for a in ARMS};rows=[]
    for i,job in enumerate(jobs):
        offset=(index+i)%3
        for arm in ARMS[offset:]+ARMS[:offset]:
            rows.append(execute(root,arm,series,job,prior[arm],runtimes[arm],api_key))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',action='store_true')
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    source=OTHER/'setup/recovered-task-source/host-jobs.json';assert sha(source)==SOURCE_SHA
    raw=json.loads(source.read_text());keys=sorted(raw)
    jobs={s:[{k:j[k] for k in ('request','actual','origin','outcome_recorded_at','future_timestamps','series_id','round')} for j in (raw[s][:3] if args.pilot else raw[s])] for s in keys}
    runtimes={a:OTHER/('plain-venv' if a=='plain' else 'gnomon-venv')/'bin/python' for a in ARMS}
    inventory=runtime_inventory()
    build=json.loads(subprocess.check_output([str(runtimes['gnomon']),'-I','-c','import json;from gnomon.build_info import build_info;print(json.dumps(build_info()))'],text=True))
    assert build['source_sha256']==BUILD_SHA and build['package_version']=='1.2.0'
    frozen={p.name:sha(p) for p in HERE.iterdir() if p.is_file()}
    dump(root/'manifest.json',{'planned':sum(map(len,jobs.values()))*3,'pilot':args.pilot,'build':build,'runtimes':{a:str(p) for a,p in runtimes.items()},'inventory':inventory,'sources':frozen,'source_jobs_sha256':SOURCE_SHA,'model':MODEL})
    dump(root/'host-jobs.json',jobs);shutil.copytree(HERE,root/'frozen-source',ignore=shutil.ignore_patterns('__pycache__'))
    rows=[];api_key=key()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(chain,i,s,jobs[s],root,runtimes,api_key) for i,s in enumerate(keys)]
        for f in as_completed(futures):rows+=f.result()
    assert all(sha(HERE/n)==h for n,h in frozen.items())
    final_inventory=runtime_inventory()
    dump(root/'final-runtime-inventory.json',final_inventory)
    assert inventory==final_inventory, 'Runtime package inventory changed during the experiment'
    dump(root/'complete.json',{'completed':len(rows),'planned':sum(map(len,jobs.values()))*3,'source_unchanged':True})


if __name__=='__main__':main()
