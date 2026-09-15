"""Gated 36-session common workflow-progress pilot. Never starts its continuation or final test."""
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


def verify_predecessor(pilot, launch, proc=Path('/proc')):
    """A failed workflow gate can finish cleanly; do not select on its accuracy."""
    pilot, launch = Path(pilot), Path(launch)
    for name in ('launch.json', 'pilot-process.json'):
        require_terminal(read(launch/name), proc)
    if (launch/'INCOMPLETE.json').exists():
        raise ValueError('Previous pilot has incomplete evidence')
    finished, audited = read(launch/'FINISHED.json'), read(launch/'AUDITED.json')
    report, gate = read(pilot/'report.json'), read(pilot/'GATE.json')
    runner = read(pilot/'runner-exit.json')
    if (finished.get('complete') is not True or audited.get('complete') is not True
            or read(launch/'pilot-exit.json')['exit_status'] != 0
            or runner != {'exit_status':0, 'continuation_launched':False}
            or report.get('complete') is not True or report['audit_failures'] or report['shutdown_record_gaps']
            or gate['accuracy_used_for_gate'] is not False or type(gate['passed']) is not bool):
        raise ValueError('Previous pilot has not completed cleanly')
    if (sha(launch/'evidence.tar.gz') != finished['archive_sha256']
            or sha(launch/'SHA256SUMS.json') != finished['inventory_sha256']):
        raise ValueError('Previous evidence archive or inventory changed')
    inventory = read(launch/'SHA256SUMS.json')
    for label, root in (('pilot',pilot),('launch',launch)):
        for name in ('report.json','GATE.json','runner-exit.json','manifest.json','pilot-plan.json') if label=='pilot' else ('AUDITED.json','pilot-exit.json'):
            if sha(root/name) != inventory[label+'/'+name]:
                raise ValueError('Previous completion evidence changed')
    from .workflow_plan_097 import PARENT_PLAN_SHA
    if sha(pilot/'pilot-plan.json') != PARENT_PLAN_SHA:
        raise ValueError('Expected the exact 096 predecessor')
    parent_plan = read(pilot/'pilot-plan.json')
    rows = report['rows']
    expected = {(a,c['series_id'],c['round']) for c in parent_plan['cases'] if c['stage']=='pilot'
                for a in ('plain','gnomon','ledger')}
    counts = {a:{'rows':sum(r['arm']==a for r in rows),
        'valid':sum(r['arm']==a and r['valid'] for r in rows),
        'full':sum(r['arm']==a and r['workflow_complete'] for r in rows)} for a in ('plain','gnomon','ledger')}
    if (len(rows)!=36 or any(v['rows']!=12 for v in counts.values())
            or {(r['arm'],r['series_id'],r['round']) for r in rows} != expected):
        raise ValueError('Previous pilot cohort incomplete')
    recomputed = all(v['valid']==12 and v['full']>=11 for v in counts.values())
    if gate['passed'] != recomputed or finished['continuation_gate_passed'] != recomputed:
        raise ValueError('Previous gate disagrees with retained counts')
    return {'previous_pilot_finished_sha256':sha(launch/'FINISHED.json'),
            'previous_report_sha256':sha(pilot/'report.json'),
            'previous_gate_passed':gate['passed'],'previous_counts':counts,
            'accuracy_used_for_admission':False}


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
    for field in ('capsule','plan','preflight','task-source','runtime','previous-pilot','previous-launch','output'):
        parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--credentials-file',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if any(k.startswith('benchmarks.hermes_ml_checkpoint_v6') for k in sys.modules):
        raise ValueError('Fresh launch process required')
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('Require a fresh pilot output; never resume by rerunning this launcher')
    predecessor=verify_predecessor(args.previous_pilot,args.previous_launch)
    plan,proof,sources=verify_inputs(args.capsule,args.plan,args.preflight,args.task_source)
    parent_plan=read(args.previous_pilot/'pilot-plan.json')
    unchanged=('arms','cases','seed','agent','gnomon','planned','budgets','pilot_gate','comparison',
               'final_target','final_gate_opened','task_source_sha256')
    if (plan['candidate']!='097_common_workflow_progress'
            or plan['parent_plan_sha256']!=sha(args.previous_pilot/'pilot-plan.json')
            or any(plan[field]!=parent_plan[field] for field in unchanged)):
        raise ValueError('The common progress amendment cannot change frozen comparison rules')
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
    sys.argv=['workflow-pilot','--output',str(args.output),'--pilot','--preflight',str(args.preflight)]
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
