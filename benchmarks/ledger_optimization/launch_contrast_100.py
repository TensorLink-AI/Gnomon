"""Gated candidate-100 pilot. No automatic continuation or final evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .contrast_plan_100 import read, sha, verify
from .contrast_readiness_100 import verify_predecessor

PLAN_SHA = '0d6e759270e23db8d54f5cc7a5a0009a8af366e8d9ace23cf66beb648b76bc27'


def normalized_preflight(plan, worker):
    proof = read(Path(worker)/'passed.json')
    return {**proof, 'tested_sources': proof['sources'],
            'tested_inventory': plan['preflight']['runtime_inventory'],
            'original_worker_proof_sha256': sha(Path(worker)/'passed.json'),
            'normalization': 'Field-name adapter only; same retained real synthetic worker evidence.'}


def verify_inputs(parent_plan, capsule, worker, task_source, plan_path, preflight):
    if sha(plan_path) != PLAN_SHA: raise ValueError('Exact frozen candidate-100 plan required')
    plan=read(plan_path)
    parent, manifest, runtime=verify(parent_plan,capsule,worker,task_source)
    if (plan['capsule'] != manifest or plan['preflight']['runtime_inventory'] != runtime['inventory']
            or read(preflight) != normalized_preflight(plan,worker)):
        raise ValueError('Plan or normalized worker proof mismatch')
    return plan


def pilot_gate(report, plan):
    expected={(a,c['series_id'],c['round'],c['origin']) for a in plan['arms'] for c in plan['cases'] if c['stage']=='pilot'}
    rows=report['rows'];actual={(r['arm'],r['series_id'],r['round'],r['origin']) for r in rows}
    if len(rows)!=36 or actual!=expected: raise ValueError('Pilot cohort mismatch')
    clean=report['complete'] is True and not report['audit_failures'] and not report['shutdown_record_gaps']
    counts={a:{'valid':sum(r['valid'] is True for r in rows if r['arm']==a),
               'full':sum(r['workflow_complete'] is True for r in rows if r['arm']==a)} for a in plan['arms']}
    passed=clean and all(v['valid']==12 and v['full']>=11 for v in counts.values())
    return {'passed':passed,'counts':counts,'accuracy_used_for_gate':False}


def run_pilot(run, analyze, args, plan, predecessor):
    """All source/runtime/predecessor checks precede this credential boundary."""
    def key():
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
    sys.argv=['contrast-pilot','--output',str(args.output),'--pilot','--preflight',str(args.preflight)]
    try:
        run.main()
        report=analyze(args.output)
        gate={**pilot_gate(report,plan),'predecessors':predecessor}
        run.dump(args.output/'GATE.json',gate)
        run.dump(args.output/'runner-exit.json',{'exit_status':0,'continuation_launched':False})
        print(json.dumps({'pilot_complete':True,'continuation_gate_passed':gate['passed'],'continuation_launched':False}))
    except BaseException as exc:
        if args.output.exists():
            run.dump(args.output/'runner-exit.json',{'exit_status':1,'type':type(exc).__name__,'continuation_launched':False})
        raise


def fresh_outputs(args):
    if args.output.is_symlink() or args.predecessor_audit.is_symlink():
        raise ValueError('Fresh separate output directories required')
    outputs=[args.output.resolve(),args.predecessor_audit.resolve()]
    protected=[p.resolve() for p in (args.previous_root,args.previous_launch,args.previous_capsule,args.capsule,args.worker_proof_root,args.runtime)]
    for p in outputs:
        if p.exists() or p.is_symlink():raise ValueError('Fresh separate output directories required')
        if any(p==q or p in q.parents or q in p.parents for q in protected):
            raise ValueError('Outputs must not overlap retained evidence or source')
    if outputs[0]==outputs[1] or outputs[0] in outputs[1].parents or outputs[1] in outputs[0].parents:
        raise ValueError('Pilot and predecessor audit must be separate')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('parent-plan','capsule','worker-proof-root','plan','preflight','task-source','runtime',
                  'previous-root','previous-launch','previous-capsule','predecessor-audit','output'):
        parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--credentials-file',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if any(k.startswith('benchmarks.hermes_ml_checkpoint_v6') for k in sys.modules):
        raise ValueError('Fresh launch process required')
    fresh_outputs(args)
    predecessor=verify_predecessor(args.previous_root,args.previous_launch,args.parent_plan)
    plan=verify_inputs(args.parent_plan,args.capsule,args.worker_proof_root,args.task_source,args.plan,args.preflight)
    # Recheck the old experiment in a fresh process, using its own exact analyzer.
    args.predecessor_audit.mkdir(parents=True)
    command=[sys.executable,'-m','benchmarks.ledger_optimization.recheck_predecessor_100',
             '--capsule',str(args.previous_capsule),'--parent-plan',str(args.parent_plan),
             '--runtime',str(args.runtime),'--root',str(args.previous_root),'--output',str(args.predecessor_audit/'audit')]
    (args.predecessor_audit/'command.json').write_text(json.dumps(command)+'\n')
    with (args.predecessor_audit/'stdout').open('x') as out, (args.predecessor_audit/'stderr').open('x') as err:
        result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
    (args.predecessor_audit/'exit.json').write_text(json.dumps({'exit_code':result.returncode})+'\n')
    if result.returncode: raise ValueError('Independent predecessor re-audit failed; evidence retained')
    # Recheck inventory after analysis to establish original evidence was not changed.
    if verify_predecessor(args.previous_root,args.previous_launch,args.parent_plan)!=predecessor:
        raise ValueError('Predecessor changed during independent re-audit')
    predecessor={**predecessor,'recheck_sha256':sha(args.predecessor_audit/'audit/passed.json')}
    os.environ['LEDGER_ML_RUNTIME_ROOT']=str(args.runtime.resolve())
    os.environ['LEDGER_ML_TASK_SOURCE']=str(args.task_source.resolve())
    sys.path.insert(0,str(args.capsule.resolve()))
    if 'benchmarks' in sys.modules:sys.modules['benchmarks'].__path__=[str(args.capsule.resolve()/'benchmarks')]
    from benchmarks.hermes_ml_checkpoint_v6 import run
    from benchmarks.hermes_ml_checkpoint_v6.analyze import analyze
    if (run.HERE!=args.capsule.resolve()/'benchmarks/hermes_ml_checkpoint_v6'
            or run.runtime_inventory()!=plan['preflight']['runtime_inventory']):
        raise ValueError('Wrong runtime or worker module')
    if args.check_only:
        print(json.dumps({'checks_passed':True,'engy_calls':0,'pilot_created':False,'predecessor_audit':str(args.predecessor_audit)}))
        return
    if args.credentials_file is None:raise ValueError('Credential file required after checks pass')
    run_pilot(run,analyze,args,plan,predecessor)


if __name__=='__main__':main()
