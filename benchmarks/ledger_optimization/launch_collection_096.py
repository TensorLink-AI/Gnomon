"""Gated 36-session collection pilot. Never starts its continuation or final test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_terminal(identity, proc=Path('/proc')):
    path=proc/str(identity['pid'])/'stat'
    if not path.exists():
        return
    fields=path.read_text().split(') ',1)[1].split()
    boot=(proc/'sys/kernel/random/boot_id').read_text().strip()
    if boot==identity['boot_id'] and fields[19]==str(identity['start_ticks']) and fields[0]!='Z':
        raise ValueError('Predecessor process is still live')


def verify_predecessors(paid, seed, proc=Path('/proc')):
    paid,seed=Path(paid),Path(seed)
    for root in (paid,seed):
        identity=read(root/'launch.json')
        require_terminal(identity,proc)
        if (root/'INCOMPLETE.json').exists() or not (root/'FINISHED.json').exists():
            raise ValueError('Predecessor has not completed cleanly')
    identity=read(paid/'launch.json')
    require_terminal({**read(paid/'development-process.json'),'boot_id':identity['boot_id']},proc)
    if read(paid/'development-exit.json')['exit_status']!=0 or read(paid/'AUDITED.json').get('complete') is not True:
        raise ValueError('Paid predecessor exit/audit failed')
    result=read(seed/'passed.json')
    if (result.get('passed') is not True or result['engy_calls']!=0
            or {r['seed'] for r in result['results']}!={7,19}):
        raise ValueError('Queued seed integration incomplete')
    seed_identity=read(seed/'launch.json')
    for number in (7,19):
        require_terminal({**read(seed/f'process-{number}.json'),'boot_id':seed_identity['boot_id']},proc)
        if read(seed/f'exit-{number}.json')['exit_code']!=0:
            raise ValueError('Queued seed process failed')
        if sha(seed/f'integration-{number}/passed.json')!=next(r['passed_sha256'] for r in result['results'] if r['seed']==number):
            raise ValueError('Queued seed evidence changed')
    if sha(seed/'passed.json')!=read(seed/'FINISHED.json')['passed_sha256']:
        raise ValueError('Queued seed completion receipt changed')
    return {'paid_finished_sha256':sha(paid/'FINISHED.json'),
            'paid_audit_sha256':sha(paid/'AUDITED.json'),
            'seed_finished_sha256':sha(seed/'FINISHED.json'),
            'seed_passed_sha256':sha(seed/'passed.json')}


def verify_inputs(capsule, plan, preflight, task_source):
    capsule=Path(capsule);plan=read(plan);proof=read(preflight)
    manifest=read(capsule/'capsule.json');package=capsule/'benchmarks/hermes_ml_checkpoint_v6'
    sources={p.name:sha(p) for p in package.iterdir() if p.is_file()}
    if (sha(capsule/'capsule.json')!=plan['capsule_sha256'] or manifest!=plan['capsule']
            or sources!=manifest['sources'] or sources!=proof['tested_sources']):
        raise ValueError('Plan, preflight and current source identities differ')
    if not proof.get('passed') or proof['engy_calls']!=0 or not proof['checks'] or not all(c['passed'] is True for c in proof['checks']):
        raise ValueError('Passing worker preflight required')
    if (sha(task_source)!=plan['task_source_sha256'] or plan['arms']!=['plain','gnomon','ledger']
            or plan['planned']!={'pilot_sessions':36,'continuation_sessions':276,'total_sessions':312}
            or plan['final_gate_opened'] is not False):
        raise ValueError('Frozen development plan mismatch')
    return plan,proof,sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('capsule','plan','preflight','task-source','runtime','paid-predecessor','seed-predecessor','output'):
        parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--credentials-file',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if any(k.startswith('benchmarks.hermes_ml_checkpoint_v6') for k in sys.modules):
        raise ValueError('Fresh launch process required')
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('Require a fresh pilot output; never resume by rerunning this launcher')
    predecessor=verify_predecessors(args.paid_predecessor,args.seed_predecessor)
    plan,proof,sources=verify_inputs(args.capsule,args.plan,args.preflight,args.task_source)
    os.environ['LEDGER_ML_RUNTIME_ROOT']=str(args.runtime.resolve())
    os.environ['LEDGER_ML_TASK_SOURCE']=str(args.task_source.resolve())
    sys.path.insert(0,str(args.capsule.resolve()))
    # -m loads the regular repository parent package before this function.
    # Point subsequent child imports exclusively at the verified capsule.
    parent=sys.modules.get('benchmarks')
    if parent is not None:
        parent.__path__=[str(args.capsule.resolve()/'benchmarks')]
    from benchmarks.hermes_ml_checkpoint_v6 import run
    from benchmarks.hermes_ml_checkpoint_v6.analyze import analyze
    if run.HERE!=args.capsule.resolve()/'benchmarks/hermes_ml_checkpoint_v6':
        raise ValueError('Unexpected worker module location')
    if run.runtime_inventory()!=proof['tested_inventory']:
        raise ValueError('Runtime package versions differ from tested worker')
    if args.check_only:
        print(json.dumps({'checks_passed':True,'paid_calls':0,'output_created':False,'scope':'Read-only launch prerequisites; no dispatch.'}))
        return
    if args.credentials_file is None:
        raise ValueError('Explicit credential file is required after checks pass')
    def key():
        # run.main has rechecked package/build identity before reaching here.
        run.dump(args.output/'accepted-launch.json',
                 {'plan_sha256':sha(args.plan),'preflight_sha256':sha(args.preflight),
                  'launcher_sha256':sha(Path(__file__)),'predecessors':predecessor,
                  'fresh_arm_state':True,'continuation_authorized':False})
        (args.output/'pilot-plan.json').write_bytes(args.plan.read_bytes())
        (args.output/'launcher-source.py').write_bytes(Path(__file__).read_bytes())
        for line in args.credentials_file.read_text().splitlines():
            if line.startswith('ENGY_API_KEY='):
                value=line.split('=',1)[1].strip().strip('"').strip("'")
                if value:return value
        raise ValueError('Engy credential unavailable')
    run.key=key
    sys.argv=['collection-pilot','--output',str(args.output),'--pilot','--preflight',str(args.preflight)]
    try:
        run.main()
        report=analyze(args.output)
        gate={'passed':report['complete'] and not report['audit_failures'] and not report['shutdown_record_gaps'],
              'accuracy_used_for_gate':False,'predecessors':predecessor}
        for arm in run.ARMS:
            rows=[r for r in report['rows'] if r['arm']==arm]
            gate['passed'] &= len(rows)==12 and all(r['valid'] for r in rows) and sum(r['workflow_complete'] for r in rows)>=11
        run.dump(args.output/'GATE.json',gate)
        run.dump(args.output/'runner-exit.json',{'exit_status':0,'continuation_launched':False})
        print(json.dumps({'pilot_complete':True,'continuation_gate_passed':gate['passed'],'continuation_launched':False}))
    except BaseException as exc:
        if args.output.exists():
            run.dump(args.output/'runner-exit.json',{'exit_status':1,'type':type(exc).__name__,'continuation_launched':False})
        raise


if __name__=='__main__':
    main()
