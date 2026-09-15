"""Full synthetic M5 host integration, with real workers and no paid network.

Uses explicit synthetic input-hash substitutions, not real development/final
values. The production CLI's plan/predecessor admission remains tested separately.
Each controller and worker is a separate actual subprocess; stage counts, fits,
copied audits and archive checks run through production host functions.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
import urllib.request
from unittest.mock import patch

from . import launch_m5_ml as launch
from . import m5_ml_controller as controller
from . import m5_ml_stage_checks as stages
from . import m5_ml_terminal_prefix as terminal
from . import m5_ml_launch_inputs as inputs
from . import continue_collection_096 as continuation
from .m5_ml_development_contract import describe_cohort
from .control_collection_096 import identity
from benchmarks.tests.test_m5_ml_development_contract import fixture


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def setup(root, stack):
    settings=launch.read(root/'settings.json');jobs_bytes=(root/'jobs.json').read_bytes();manifest_bytes=(root/'panel.json').read_bytes()
    cohort=describe_cohort(json.loads(manifest_bytes),json.loads(jobs_bytes));digest=hashlib.sha256(jobs_bytes).hexdigest()
    def authenticate(m,j):
        if m!=manifest_bytes or j!=jobs_bytes:raise ValueError('Synthetic fixture bytes changed')
        return deepcopy(cohort)
    stack.enter_context(patch.object(stages,'authenticated_contract',authenticate))
    for module in (stages,terminal,controller):stack.enter_context(patch.object(module,'DEVELOPMENT_JOBS_SHA',digest))
    return settings,manifest_bytes,jobs_bytes


def arguments(root,stage,settings):
    return SimpleNamespace(output=root/('pilot' if stage=='pilot' else 'complete'),
        launch=root/('pilot-launch' if stage=='pilot' else 'complete-launch'),stage=stage,
        runtime=Path(settings['runtime']),capsule=Path(settings['capsule']),
        pilot_root=root/'pilot',pilot_launch=root/'pilot-launch',plan=root/'plan.json',
        host_preflight=root/'synthetic-bootstrap.json',credentials_file=root/'synthetic-credential.txt',
        reservation=Path(launch.read(root/(stage+'-reservation.json'))['path']) if (root/(stage+'-reservation.json')).exists() else None)


def worker(root,stage):
    with ExitStack() as stack:
        settings,manifest_bytes,jobs_bytes=setup(root,stack);args=arguments(root,stage,settings);plan=launch.read(args.plan)
        helper=continuation.configure(args.capsule,args.runtime,root/'jobs.json');run=helper.run
        from benchmarks.hermes_ml_checkpoint_v6 import service_admission
        stack.enter_context(patch.object(run,'SOURCE_SHA',hashlib.sha256(jobs_bytes).hexdigest()))
        if run.runtime_inventory()!=plan['runtime_inventory']:raise ValueError('Actual runtime differs from frozen seed proof')
        prefix=None
        if stage=='complete':
            prefix=terminal.verify_terminal_prefix(args.pilot_root,args.pilot_launch,args.plan,args.capsule,
                plan['runtime_inventory'],manifest_bytes,jobs_bytes)
        launch.consume_reservation(args,plan)
        try:launch.consume_reservation(args,plan)
        except FileExistsError:reentry=True
        else:raise AssertionError('Worker reservation reused')
        lock=threading.Lock();contexts={};wire=[];checks=[];active=0;maximum=0
        config={'model':'ridge','window':90,'lags':7,'alpha':10}
        def check(name,ok):
            with lock:checks.append({'assertion':name,'passed':bool(ok)})
            if not ok:raise AssertionError(name)
        class Response:
            status=200
            def __init__(self,body):self.body=body
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return json.dumps(self.body).encode()
        def upstream(request,**kwargs):
            if request.full_url!='https://api.engy.ai/v1/chat/completions':raise AssertionError('No real network permitted')
            token=request.get_header('Authorization').removeprefix('Bearer ')
            with lock:
                current=contexts[token];current['n']+=1;n=current['n'];meta=dict(current)
            payload=json.loads(request.data);check('forwarded requested seed',payload['seed']==settings['seed'])
            if n>4:raise AssertionError('Unexpected extra scripted request')
            with lock:wire.append({'arm':meta['arm'],'series_id':meta['series'],'round':meta['round'],'request':payload})
            marker=f"S{settings['seed']}/{meta['series']}/{meta['arm']}/r{meta['round']}"
            operations={1:[('lab',{'operation':'review'}),('lab',{'operation':'start'})],
                2:[('lab',{'operation':'backtest','config':config}),('memory',{'target':'memory','action':'add','content':marker})],
                3:[('lab',{'operation':'commit','config':config}),('notes_write',{'path':'decision.json','text':'{"rationale":"synthetic host integration"}'})],4:[]}[n]
            calls=[{'id':f"{meta['series']}-{meta['arm']}-{meta['round']}-{n}-{i}",'type':'function',
                    'function':{'name':name,'arguments':json.dumps(arguments)}} for i,(name,arguments) in enumerate(operations)]
            message={'role':'assistant','content':None if calls else 'Synthetic workflow complete.'}
            if calls:message['tool_calls']=calls
            return Response({'id':f'synthetic-{token}-{n}','object':'chat.completion','created':0,
                'model':'deepseek-v4.1-flash','choices':[{'index':0,'message':message,'finish_reason':'tool_calls' if calls else 'stop'}],
                'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}})
        original=run.execute
        def execute(*values):
            nonlocal active,maximum
            _,arm,series,job=values[:4];token=f'synthetic-{settings["seed"]}-{series}-{arm}-{job["round"]}'
            with lock:
                if token in contexts:raise AssertionError('Duplicate session execution')
                contexts[token]={'arm':arm,'series':series,'round':job['round'],'n':0}
                active+=1;maximum=max(maximum,active)
            try:
                result=original(*values[:-1],token)
                check(token+' complete',result['valid'] and result['workflow_complete'])
                check(token+' bounded calls/fits',result['api_calls']==4 and result['numerical_attempts']==8)
                return result
            finally:
                with lock:active-=1
        def ready(_):return 200,b'{"choices":[{"message":{"content":"READY"}}],"usage":{"total_tokens":0},"synthetic":true}',0.0
        stack.enter_context(patch.object(urllib.request,'urlopen',upstream))
        stack.enter_context(patch.object(service_admission,'probe',ready))
        stack.enter_context(patch.object(run,'execute',execute))
        try:
            launch.execute_stage(args,helper,plan,manifest_bytes,jobs_bytes,prefix)
            expected=72 if stage=='pilot' else 552
            check('full stage executed once',len(contexts)==expected and len(wire)==expected*4)
            check('two series overlapped',maximum==2)
            if stage=='complete':
                check('original prefix unchanged',terminal.inventory(args.pilot_root)==prefix['pilot_files'])
                for series in json.loads(jobs_bytes):
                    for arm in run.ARMS:
                        messages=[json.dumps(w['request']['messages']) for w in wire if w['series_id']==series and w['arm']==arm and w['round']==3]
                        own=f'S{settings["seed"]}/{series}/{arm}/r2'
                        check(own+' memory restored',any(own in message for message in messages))
                        for other_series in json.loads(jobs_bytes):
                            for other_arm in run.ARMS:
                                if (other_series,other_arm)!=(series,arm):
                                    forbidden=f'S{settings["seed"]}/{other_series}/{other_arm}/r'
                                    check('other memory excluded',all(forbidden not in m for m in messages))
                        check('other seed excluded',all(f'S{19 if settings["seed"]==7 else 7}/' not in m for m in messages))
            dump(args.output/'integrated-probe.json',{'passed':True,'checks':checks,'sessions':expected,
                'requested_seed':settings['seed'],'maximum_parallel_sessions':maximum,'reservation_reentry_rejected':reentry,
                'provider_fits':expected*8,'scripted_responses':len(wire),'engy_calls':0,'host_sources':launch.source_identity()})
        finally:
            # Wire already retained per session; this index carries metadata and
            # checks without creating a second full transcript copy.
            if args.output.exists():dump(args.output/'integration-checks.json',{'checks':checks,'requests':len(wire),'host_sources':launch.source_identity()})


def control(root,stage):
    with ExitStack() as stack:
        settings,manifest_bytes,jobs_bytes=setup(root,stack);args=arguments(root,stage,settings);plan=launch.read(args.plan)
        reservation=inputs.reserve_stage(args.plan,plan,stage=stage,output=args.output,controller=args.launch)
        dump(root/(stage+'-reservation.json'),{'path':str(reservation)})
        command=[sys.executable,'-m','benchmarks.ledger_optimization.probe_m5_integrated_host',
                 '--root',str(root),'--role','worker','--stage',stage]
        result=controller.supervise(command,args.launch,args.output,manifest_bytes,jobs_bytes,
                                    stage=stage,credentials_file=args.credentials_file)
        print(json.dumps(result),flush=True)
        if not result['complete'] or stage=='pilot' and not result['continuation_gate_passed']:raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--role',choices=('orchestrate','control','worker'),default='orchestrate')
    parser.add_argument('--stage',choices=('pilot','complete'));parser.add_argument('--runtime',type=Path)
    args=parser.parse_args();root=args.root.absolute()
    if args.role=='worker':return worker(root,args.stage)
    if args.role=='control':return control(root,args.stage)
    if args.runtime is None:raise ValueError('Explicit existing runtime required for orchestration')
    root.mkdir(parents=True,exist_ok=False)
    dump(root/'launch.json',{**identity(__import__('os').getpid()),'at':datetime.now(timezone.utc).isoformat(),'automatic_retry':False})
    manifest,jobs=fixture();initial_sources=launch.source_identity();completed=[]
    options={7:('results/m5-ml-capsule-offline-001/capsule','results/m5-ml-capsule-offline-001/synthetic-worker-002'),
             19:('results/m5-ml-seed-19-offline-001/capsule','results/m5-ml-seed-19-offline-001/probe')}
    try:
        for seed,(capsule,proof) in options.items():
            home=root/f'seed-{seed}';home.mkdir();capsule=Path(capsule).resolve();proof=Path(proof)
            dump(home/'panel.json',manifest);dump(home/'jobs.json',jobs)
            runtime=launch.read(proof/'manifest.json');cap=launch.read(capsule/'capsule.json')
            dump(home/'settings.json',{'seed':seed,'capsule':str(capsule),'runtime':str(args.runtime.resolve())})
            plan={'requested_seed':seed,'capsule':cap,'capsule_sha256':inputs.sha(capsule/'capsule.json'),
                'task_source_sha256':inputs.sha(home/'jobs.json'),'runtime_inventory':runtime['inventory'],'build':runtime['build'],
                'arms':['plain','gnomon','ledger'],'planned':{'pilot_sessions':72,'continuation_sessions':552,'total_sessions':624},
                'dispatch_registry':str(root/'synthetic-reservations'),'final_gate_opened':False,'synthetic':True}
            dump(home/'plan.json',plan);dump(home/'synthetic-bootstrap.json',{'synthetic':True,'engy_calls':0})
            (home/'synthetic-credential.txt').write_text('ENGY_API_KEY=synthetic-host-probe-credential\n')
            for stage in ('pilot','complete'):
                command=[sys.executable,'-m','benchmarks.ledger_optimization.probe_m5_integrated_host',
                         '--root',str(home),'--role','control','--stage',stage]
                with (home/(stage+'-controller.stdout')).open('x') as out,(home/(stage+'-controller.stderr')).open('x') as err:
                    child=subprocess.Popen(command,stdout=out,stderr=err)
                    dump(home/(stage+'-controller-process.json'),identity(child.pid));status=child.wait()
                dump(home/(stage+'-controller-exit.json'),{'exit_status':status,'argv':command})
                if status:raise RuntimeError(f'Seed {seed} {stage} failed; no retry')
                completed.append({'seed':seed,'stage':stage});dump(root/'progress.json',{'completed':completed})
        if initial_sources!=launch.source_identity():raise ValueError('Host sources changed during integration')
        dump(root/'passed.json',{'status':'m5_integrated_host_preflight_passed','passed':True,'engy_calls':0,
            'host_sources':initial_sources,'plan_hashes':{str(k):v for k,v in launch.PLAN_HASHES.items()},
            'runtime_inventory':runtime['inventory'],'build':runtime['build'],'requested_seeds':[7,19],
            'series_per_seed':8,'pilot_sessions_per_seed':72,'resumed_sessions_per_seed':552,
            'full_workflows_per_seed':624,'parallel_series':2,'original_prefix_unchanged':True,
            'copied_prefix_audit_passed':True,'cross_seed_state_isolated':True,'reservation_reentry_rejected':True,
            'checks':[{'assertion':f'seed {v["seed"]} {v["stage"]} complete','passed':True} for v in completed],
            'scope':'Full synthetic host/worker integration; fixed data-hash authentication substituted only for synthetic fixture bytes. No real Engy request or final-target access.'})
    except BaseException as exc:
        dump(root/'INCOMPLETE.json',{'type':type(exc).__name__,'completed':completed,'automatic_retry':False});raise


if __name__=='__main__':main()
