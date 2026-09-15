"""Continue a terminal, accepted 096 pilot once, using its exact worker capsule.

All gates and a copied-prefix audit precede credential access. No final targets
are admitted. This host wrapper reuses the tested 093 state-transfer primitives;
their worker imports are bound to the verified 096 capsule in a fresh process.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .launch_collection_096 import read, sha, require_terminal, verify_inputs


def source_identity():
    root = Path(__file__).parent
    names = ('continue_collection_096.py', 'continue_guarded_093.py',
             'launch_collection_096.py')
    return {name: sha(root/name) for name in names}


def configure(capsule, runtime, task_source=None):
    if any(k.startswith('benchmarks.hermes_ml_checkpoint_v6') for k in sys.modules):
        raise ValueError('Fresh process required for the continuation capsule')
    capsule, runtime = Path(capsule).resolve(), Path(runtime).resolve()
    manifest = read(capsule/'capsule.json')
    package = capsule/'benchmarks/hermes_ml_checkpoint_v6'
    if {p.name: sha(p) for p in package.iterdir() if p.is_file()} != manifest['sources']:
        raise ValueError('Capsule source mismatch')
    os.environ['LEDGER_ML_RUNTIME_ROOT'] = str(runtime)
    if task_source is not None:
        os.environ['LEDGER_ML_TASK_SOURCE'] = str(Path(task_source).resolve())
    sys.path.insert(0, str(capsule))
    parent = sys.modules.get('benchmarks')
    if parent is not None:
        parent.__path__ = [str(capsule/'benchmarks')]
    from . import continue_guarded_093 as helper
    if helper.run.HERE != package or helper.run.OTHER != runtime:
        raise ValueError('Continuation imported an unexpected worker')
    return helper


def verify_gate(helper, pilot, launch, jobs, sources, plan_path, proc=Path('/proc')):
    """Recheck outcomes and immutable receipts, rather than trusting passed=true."""
    if (launch/'INCOMPLETE.json').exists():
        raise ValueError('Pilot controller recorded incomplete evidence')
    for identity_path in ('launch.json', 'pilot-process.json'):
        require_terminal(read(launch/identity_path), proc)
    finished, audited = read(launch/'FINISHED.json'), read(launch/'AUDITED.json')
    gate, runner = read(pilot/'GATE.json'), read(pilot/'runner-exit.json')
    if (finished.get('complete') is not True or audited.get('complete') is not True
            or finished.get('continuation_gate_passed') is not True
            or read(launch/'pilot-exit.json').get('exit_status') != 0
            or runner.get('exit_status') != 0 or runner.get('continuation_launched') is not False
            or gate.get('passed') is not True or gate.get('accuracy_used_for_gate') is not False):
        raise ValueError('Pilot completion gate did not pass')
    manifest, report = read(pilot/'manifest.json'), read(pilot/'report.json')
    if (manifest.get('planned') != 36 or manifest.get('sources') != sources
            or manifest.get('source_jobs_sha256') != helper.run.SOURCE_SHA
            or report.get('complete') is not True or report.get('audit_failures') != []
            or report.get('shutdown_record_gaps') != []):
        raise ValueError('Pilot source or complete independent audit mismatch')
    if (read(pilot/'accepted-launch.json')['plan_sha256'] != sha(plan_path)
            or sha(pilot/'pilot-plan.json') != sha(plan_path)):
        raise ValueError('Pilot did not execute this frozen plan')
    if read(pilot/'host-jobs.json') != {s: values[:3] for s, values in jobs.items()}:
        raise ValueError('Pilot does not match the exact development prefix')
    grades = [read(p) for p in pilot.glob('*/*/round-*/grade.json')]
    expected = {(a, s, n) for a in helper.run.ARMS for s in jobs for n in range(3)}
    if len(grades) != 36 or {(g['arm'], g['series_id'], g['round']) for g in grades} != expected:
        raise ValueError('Missing, extra or duplicate pilot sessions')
    for arm in helper.run.ARMS:
        rows = [g for g in grades if g['arm'] == arm]
        full = sum(g['workflow_complete'] is True for g in rows)
        if len(rows) != 12 or not all(g['valid'] is True for g in rows) or full < 11:
            raise ValueError('Pilot completion threshold failed: ' + arm)
        if report['arms'][arm]['workflow_complete'] != full:
            raise ValueError('Pilot report disagrees with retained grades')
    if (sha(launch/'evidence.tar.gz') != finished['archive_sha256']
            or sha(launch/'SHA256SUMS.json') != finished['inventory_sha256']):
        raise ValueError('Pilot terminal archive or inventory hash mismatch')
    inventory = read(launch/'SHA256SUMS.json')
    files = helper.inventory(pilot)
    if files != {name[6:]: value for name, value in inventory.items() if name.startswith('pilot/')}:
        raise ValueError('Pilot state differs from terminal inventory')
    for name, value in inventory.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in ('launch', 'pilot'):
            raise ValueError('Unsafe terminal inventory path')
        if relative.parts[0] == 'launch' and sha(launch.joinpath(*relative.parts[1:])) != value:
            raise ValueError('Pilot launch evidence changed')
    return files, manifest


def verify_preflight(path, sources, runtime):
    proof = read(path)
    if (proof.get('passed') is not True or proof.get('engy_calls') != 0
            or proof.get('continuation_sources') != source_identity()
            or proof.get('frozen_sources') != sources or proof.get('tested_inventory') != runtime
            or set(proof.get('resumed_arms', [])) != {'plain', 'gnomon', 'ledger'}
            or proof.get('numerical_attempts') != 96
            or not proof.get('checks') or not all(c.get('passed') is True for c in proof['checks'])
            or proof.get('independent_audit_checks', 0) <= 0):
        raise ValueError('Exact-source synthetic continuation preflight required')
    return proof


def prepare_copy(helper, pilot, output, files, manifest, jobs):
    metadata = helper.copy_prefix(pilot, output, files)
    for name in ('manifest.json', 'host-jobs.json'):
        shutil.copyfile(metadata/name, output/name)
    prefix = helper.analyze(output)
    if not prefix['complete'] or prefix['audit_failures'] or prefix['shutdown_record_gaps']:
        raise ValueError('Independent copied-prefix audit failed')
    for name in ('report.json', 'RESULTS.md'):
        (output/name).rename(metadata/('rechecked-' + name))
    helper.run.dump(output/'manifest.json', {**manifest, 'pilot': False, 'planned': 312,
        'pilot_sessions_retained': 36, 'new_sessions_planned': 276,
        'continuation_sources': source_identity(), 'original_pilot': str(pilot),
        'accuracy_used_for_promotion': False})
    helper.run.dump(output/'host-jobs.json', jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('capsule', 'plan', 'preflight', 'task-source', 'runtime', 'pilot-root',
                  'pilot-launch', 'continuation-preflight', 'output'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--credentials-file', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    pilot, launch, output = (p.resolve() for p in (args.pilot_root, args.pilot_launch, args.output))
    if (args.output.exists() or args.output.is_symlink() or pilot in output.parents
            or launch in output.parents or output in pilot.parents or output in launch.parents):
        raise ValueError('Require a fresh continuation root outside the pilot and launch')
    _, proof, sources = verify_inputs(args.capsule, args.plan, args.preflight, args.task_source)
    helper = configure(args.capsule, args.runtime, args.task_source)
    jobs = helper.jobs_from_source(args.task_source)
    files, manifest = verify_gate(helper, pilot, launch, jobs, sources, args.plan)
    runtime = helper.run.runtime_inventory()
    if (runtime != proof['tested_inventory'] or runtime != manifest['inventory']
            or runtime != read(pilot/'final-runtime-inventory.json')):
        raise ValueError('Runtime differs from the tested and completed pilot')
    runtimes = {a: args.runtime.resolve()/('plain-venv' if a == 'plain' else 'gnomon-venv')/'bin/python'
                for a in helper.run.ARMS}
    if {a: str(p) for a, p in runtimes.items()} != manifest['runtimes']:
        raise ValueError('Runtime locations differ from the original pilot')
    build = json.loads(subprocess.check_output([str(runtimes['gnomon']), '-I', '-c',
        'import json;from gnomon.build_info import build_info;print(json.dumps(build_info()))'], text=True))
    if (build != manifest['build'] or build['package_version'] != '1.2.0'
            or build['source_sha256'] != helper.run.BUILD_SHA):
        raise ValueError('Installed build differs from the original pilot')
    verify_preflight(args.continuation_preflight, sources, runtime)
    if args.check_only:
        print(json.dumps({'checks_passed': True, 'paid_calls': 0, 'output_created': False}))
        return
    if args.credentials_file is None:
        raise ValueError('Explicit credential file required after verification')
    tested_sources = source_identity()
    try:
        prepare_copy(helper, pilot, output, files, manifest, jobs)
        helper.run.dump(output/'accepted-launch.json', {'continuation_sources': tested_sources,
            'continuation_preflight_sha256': sha(args.continuation_preflight),
            'pilot_finished_sha256': sha(launch/'FINISHED.json'), 'plan_sha256': sha(args.plan),
            'retained_sessions': 36, 'new_sessions': 276, 'final_gate_opened': False})
        api_key = None
        for line in args.credentials_file.read_text().splitlines():
            if line.startswith('ENGY_API_KEY='):
                api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                break
        if not api_key:
            raise ValueError('Engy credential unavailable')
        results = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(helper.continue_chain, i, s, jobs[s], output, runtimes, api_key)
                       for i, s in enumerate(sorted(jobs))]
            for future in as_completed(futures):
                try:
                    results += future.result()
                except BaseException:
                    helper.run.STOP.set()
                    raise
        if len(results) != 276:
            raise ValueError('Continuation session count mismatch')
        final_runtime = helper.run.runtime_inventory()
        helper.run.dump(output/'final-runtime-inventory.json', final_runtime)
        if (runtime != final_runtime or helper.inventory(pilot) != files
                or source_identity() != tested_sources
                or {p.name: sha(p) for p in helper.run.HERE.iterdir() if p.is_file()} != sources):
            raise ValueError('Pilot, runtime or frozen source changed during continuation')
        helper.run.dump(output/'complete.json', {'completed': 312, 'planned': 312,
            'new_sessions': 276, 'retained_pilot_sessions': 36, 'source_unchanged': True})
        report = helper.analyze(output)
        if not report['complete'] or report['audit_failures'] or report['shutdown_record_gaps']:
            raise ValueError('Independent continuation audit failed')
        helper.run.dump(output/'runner-exit.json', {'exit_status': 0, 'final_gate_opened': False})
    except BaseException as exc:
        helper.run.STOP.set()
        if output.exists():
            helper.run.dump(output/'CONTINUATION_INCOMPLETE.json', {
                'at': datetime.now(timezone.utc).isoformat(), 'type': type(exc).__name__,
                'automatic_retry': False, 'scope': 'Retain all attempts; no selective rerun.'})
            helper.run.dump(output/'runner-exit.json', {'exit_status': 1, 'final_gate_opened': False})
        raise


if __name__ == '__main__':
    main()
