"""Admit one fixed M5 pilot or continuation through the archival controller.

CLI dispatch remains closed until an exact-source integrated host preflight
exists. No final-data path, automatic retry or accuracy-dependent continuation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from . import m5_ml_launch_inputs as inputs
from . import m5_ml_stage_checks as stages
from . import m5_ml_controller as controller
from . import m5_ml_terminal_prefix as terminal
from . import continue_collection_096 as continuation
from .control_continuation_097 import completion as predecessor_completion

PLAN_HASHES = {7: '2fbfc7716c3aaa027d9265a28e2d801c1c17423cbdd06644b8682aab3569e76a',
               19: '643ba1061dffcd2076cd409a311da3441edb52dd57444d1a85c7dc4d10686325'}
HOST_FILES = ('launch_m5_ml.py', 'm5_ml_launch_inputs.py', 'm5_ml_controller.py',
              'm5_ml_terminal_prefix.py', 'm5_ml_stage_checks.py', 'm5_ml_prefix_identity.py',
              'm5_ml_development_contract.py', 'm5_ml_adapter.py', 'm5_ml_panel.py',
              'continue_collection_096.py', 'continue_guarded_093.py',
              'launch_collection_096.py', 'control_collection_096.py',
              'control_continuation_097.py', 'costs_guarded_093.py',
              'probe_m5_integrated_host.py', '../tests/test_m5_ml_development_contract.py')


def source_identity():
    return {name: inputs.sha(Path(__file__).parent/name) for name in HOST_FILES}


def read(path):
    return json.loads(Path(path).read_text())


def validate_paths(args):
    """Outputs must not nest within retained evidence, source or runtime roots."""
    output, launch = args.output.absolute(), args.launch.absolute()
    protected = [getattr(args, n, None) for n in ('capsule', 'worker_proof', 'runtime',
                 'pilot_root', 'pilot_launch', 'predecessor_root', 'predecessor_launch')]
    if output == launch or output in launch.parents or launch in output.parents:
        raise ValueError('Separate output and controller roots required')
    for root in (Path(p).absolute() for p in protected if p is not None):
        if any(root == p or root in p.parents or p in root.parents for p in (output, launch)):
            raise ValueError('Output/controller must be outside all retained and source roots')


def verify_host_preflight(path, plan):
    proof = read(path)
    if (proof.get('status') != 'm5_integrated_host_preflight_passed'
            or proof.get('passed') is not True or proof.get('engy_calls') != 0
            or proof.get('host_sources') != source_identity()
            or proof.get('plan_hashes') != {str(k): v for k, v in PLAN_HASHES.items()}
            or proof.get('runtime_inventory') != plan['runtime_inventory']
            or proof.get('build') != plan['build']
            or proof.get('requested_seeds') != [7, 19]
            or proof.get('series_per_seed') != 8
            or proof.get('pilot_sessions_per_seed') != 72
            or proof.get('resumed_sessions_per_seed') != 552
            or proof.get('full_workflows_per_seed') != 624
            or proof.get('parallel_series') != 2
            or proof.get('original_prefix_unchanged') is not True
            or proof.get('copied_prefix_audit_passed') is not True
            or proof.get('cross_seed_state_isolated') is not True
            or proof.get('reservation_reentry_rejected') is not True
            or not proof.get('checks') or not all(c.get('passed') is True for c in proof['checks'])):
        raise ValueError('Exact-source eight-series, two-seed host integration preflight required')
    return proof


def verify_predecessor(output, launch, parent_plan, *, proc=Path('/proc')):
    """Require terminal candidate-100 evidence; never erase an old audit failure."""
    output, launch = Path(output), Path(launch)
    if inputs.sha(parent_plan) != inputs.PARENT_PLAN_SHA:
        raise ValueError('Fixed candidate-100 predecessor plan required')
    for name in ('launch.json', 'development-process.json'):
        terminal.terminal(read(launch/name), proc)
    if (launch/'INCOMPLETE.json').exists():
        raise ValueError('Predecessor requires explicit retained audit reconciliation; no automatic override')
    status = read(launch/'development-exit.json')['exit_status']
    recomputed = predecessor_completion(output, status)
    if (not recomputed['complete'] or read(launch/'AUDITED.json') != recomputed
            or read(launch/'FINISHED.json').get('complete') is not True):
        raise ValueError('Predecessor completion evidence does not pass')
    terminal.archived_prefix(output, launch)
    parent, manifest, report = read(parent_plan), read(output/'manifest.json'), read(output/'report.json')
    if manifest['sources'] != parent['capsule']['sources'] or manifest['source_jobs_sha256'] != parent['task_source_sha256']:
        raise ValueError('Wrong predecessor source')
    expected = {(a, c['series_id'], c['round'], c['origin']) for a in parent['arms'] for c in parent['cases']}
    if {(r['arm'], r['series_id'], r['round'], r['origin']) for r in report['rows']} != expected or len(report['rows']) != len(expected):
        raise ValueError('Wrong predecessor cohort')
    costs = read(launch/'costs.json')
    if (costs['deduplicated_total']['sessions'] != 312
            or costs['stages']['retained_pilot']['sessions'] != 36
            or costs['stages']['continuation']['sessions'] != 276):
        raise ValueError('Predecessor cost coverage incomplete')
    return {'finished_sha256': inputs.sha(launch/'FINISHED.json'),
            'report_sha256': inputs.sha(output/'report.json'),
            'accuracy_used_for_admission': False}


def prepare_copy(helper, pilot, output, files, manifest, jobs, manifest_bytes, jobs_bytes):
    """Copy immutable state, independently audit it, then extend host metadata."""
    metadata = helper.copy_prefix(pilot, output, files)
    for name in ('manifest.json', 'host-jobs.json'):
        shutil.copyfile(metadata/name, output/name)
    report = helper.analyze(output)
    grades = [read(p) for p in output.glob('*/*/round-*/grade.json')]
    stages.check_development_stage(manifest_bytes, jobs_bytes, grades, report, stage='pilot')
    for name in ('report.json', 'RESULTS.md'):
        (output/name).rename(metadata/('rechecked-'+name))
    helper.run.dump(output/'manifest.json', {**manifest, 'pilot': False, 'planned': 624,
        'pilot_sessions_retained': 72, 'new_sessions_planned': 552,
        'continuation_sources': source_identity(), 'original_pilot': str(pilot),
        'accuracy_used_for_promotion': False})
    helper.run.dump(output/'host-jobs.json', jobs)
    return metadata


def consume_reservation(args, plan):
    if args.reservation is None:
        raise ValueError('Prior one-shot reservation required')
    reservation = read(args.reservation)
    if (reservation.get('plan_sha256') != inputs.sha(args.plan) or reservation.get('stage') != args.stage
            or reservation.get('output') != str(args.output.absolute())
            or reservation.get('controller') != str(args.launch.absolute())
            or args.reservation.parent != Path(plan['dispatch_registry'])):
        raise ValueError('Reservation does not bind this worker command')
    with args.reservation.with_suffix('.consumed.json').open('x') as stream:
        json.dump({'at': datetime.now(timezone.utc).isoformat(), 'automatic_retry': False}, stream)


def execute_stage(args, helper, plan, manifest_bytes, jobs_bytes, prefix):
    """Already admitted worker path. Credential access follows copy/audit."""
    output = args.output.resolve(); tested = source_identity()
    run = helper.run
    runtimes = {a: args.runtime.resolve()/('plain-venv' if a == 'plain' else 'gnomon-venv')/'bin/python' for a in run.ARMS}
    jobs = json.loads(jobs_bytes)
    old_dump = run.dump

    def stamped_dump(path, value):
        if Path(path) == output/'manifest.json':
            value = {**value, 'requested_seed': plan['requested_seed'], 'host_sources': tested}
        old_dump(path, value)

    def key():
        if source_identity() != tested:
            raise ValueError('Host source changed before credentials')
        run.dump(output/'accepted-launch.json', {'plan_sha256': inputs.sha(args.plan),
            'host_preflight_sha256': inputs.sha(args.host_preflight),
            'reservation_sha256': inputs.sha(args.reservation), 'host_sources': tested,
            'stage': args.stage, 'final_gate_opened': False})
        if args.stage == 'pilot':
            (output/'pilot-plan.json').write_bytes(args.plan.read_bytes())
        for line in args.credentials_file.read_text().splitlines():
            if line.startswith('ENGY_API_KEY='):
                value = line.split('=', 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
        raise ValueError('Engy credential unavailable')

    run.dump = stamped_dump
    run.key = key
    try:
        if args.stage == 'pilot':
            proof_path = args.launch/'worker-preflight.json'
            old_dump(proof_path, {'passed': True, 'tested_sources': plan['capsule']['sources'],
                'tested_inventory': plan['runtime_inventory'], 'source': 'Authenticated worker and integrated host proofs',
                'host_preflight_sha256': inputs.sha(args.host_preflight)})
            sys.argv = ['m5-pilot', '--output', str(output), '--pilot', '--preflight', str(proof_path)]
            run.main()
            report = helper.analyze(output)
            checked = stages.check_development_stage(manifest_bytes, jobs_bytes,
                [read(p) for p in output.glob('*/*/round-*/grade.json')], report,
                stage='pilot', require_pilot_quality=False)
            run.dump(output/'GATE.json', {'passed': checked['pilot_quality_passed'], 'accuracy_used_for_gate': False})
            exit_receipt = {'exit_status': 0, 'continuation_launched': False}
        else:
            prepare_copy(helper, args.pilot_root, output, prefix['pilot_files'],
                         read(args.pilot_root/'manifest.json'), jobs, manifest_bytes, jobs_bytes)
            api_key = key()
            results = []
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(helper.continue_chain, i, s, jobs[s], output, runtimes, api_key)
                           for i, s in enumerate(sorted(jobs))]
                for future in as_completed(futures):
                    try:
                        results += future.result()
                    except BaseException:
                        run.STOP.set(); raise
            if len(results) != 552 or helper.inventory(args.pilot_root) != prefix['pilot_files']:
                raise ValueError('Continuation count or original pilot changed')
            runtime = run.runtime_inventory()
            run.dump(output/'final-runtime-inventory.json', runtime)
            if runtime != plan['runtime_inventory']:
                raise ValueError('Runtime changed during continuation')
            run.dump(output/'complete.json', {'completed': 624, 'planned': 624, 'new_sessions': 552,
                'retained_pilot_sessions': 72, 'source_unchanged': True})
            report = helper.analyze(output)
            stages.check_development_stage(manifest_bytes, jobs_bytes,
                [read(p) for p in output.glob('*/*/round-*/grade.json')], report, stage='complete')
            exit_receipt = {'exit_status': 0, 'final_gate_opened': False}
        if source_identity() != tested or {p.name: inputs.sha(p) for p in run.HERE.iterdir() if p.is_file()} != plan['capsule']['sources']:
            raise ValueError('Frozen source changed during stage')
        run.dump(output/'runner-exit.json', exit_receipt)
    except BaseException as exc:
        run.STOP.set()
        if output.exists():
            run.dump(output/'INCOMPLETE.json', {'type': type(exc).__name__, 'automatic_retry': False})
            run.dump(output/'runner-exit.json', {'exit_status': 1, 'final_gate_opened': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('plan', 'parent-plan', 'capsule', 'worker-proof', 'panel-manifest', 'task-source',
                  'runtime', 'host-preflight', 'output', 'launch', 'predecessor-root', 'predecessor-launch'):
        parser.add_argument('--'+field, type=Path, required=True)
    for field in ('pilot-root', 'pilot-launch', 'credentials-file', 'reservation'):
        parser.add_argument('--'+field, type=Path)
    parser.add_argument('--seed', type=int, choices=(7,19), required=True)
    parser.add_argument('--stage', choices=('pilot','complete'), required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.absolute())
    validate_paths(args)
    if inputs.sha(args.plan) != PLAN_HASHES[args.seed]:
        raise ValueError('Exact frozen M5 seed plan required')
    manifest_bytes, jobs_bytes = args.panel_manifest.read_bytes(), args.task_source.read_bytes()
    plan = inputs.verify_plan(args.plan, args.parent_plan, args.capsule, args.worker_proof,
        manifest_bytes, jobs_bytes, seed=args.seed, registry=Path(read(args.plan)['dispatch_registry']))
    verify_host_preflight(args.host_preflight, plan)
    verify_predecessor(args.predecessor_root, args.predecessor_launch, args.parent_plan)
    helper = continuation.configure(args.capsule, args.runtime, args.task_source)
    runtime = helper.run.runtime_inventory()
    build = json.loads(subprocess.check_output([str(args.runtime.resolve()/'gnomon-venv/bin/python'), '-I', '-c',
        'import json;from gnomon.build_info import build_info;print(json.dumps(build_info()))'], text=True))
    if runtime != plan['runtime_inventory'] or build != plan['build']:
        raise ValueError('Actual host runtime/build differs from tested plan')
    prefix = None
    if args.stage == 'complete':
        if args.pilot_root is None or args.pilot_launch is None:
            raise ValueError('Explicit terminal pilot roots required')
        prefix = terminal.verify_terminal_prefix(args.pilot_root,args.pilot_launch,args.plan,
            args.capsule,runtime,manifest_bytes,jobs_bytes)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('Fresh output required; never restart the same run')
    if args.check_only:
        print(json.dumps({'checks_passed':True,'execution_authorized':False,'engy_calls':0}));return
    if args.credentials_file is None:
        raise ValueError('Explicit credential path required after all admission checks')
    if args.worker:
        consume_reservation(args, plan)
        execute_stage(args,helper,plan,manifest_bytes,jobs_bytes,prefix)
    else:
        reservation=inputs.reserve_stage(args.plan,plan,stage=args.stage,output=args.output,controller=args.launch)
        command=[sys.executable,'-m','benchmarks.ledger_optimization.launch_m5_ml',*sys.argv[1:],
                 '--worker','--reservation',str(reservation)]
        result=controller.supervise(command,args.launch,args.output,manifest_bytes,jobs_bytes,
                                    stage=args.stage,credentials_file=args.credentials_file)
        print(json.dumps(result))
        if not result['complete']:raise SystemExit(1)


if __name__=='__main__':main()
