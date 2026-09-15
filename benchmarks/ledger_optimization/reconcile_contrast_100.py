"""Retain and reconcile the known candidate-100 audit failure, never rerun agents.

This produces evidence only. It cannot reserve or dispatch an M5 run, remove an
INCOMPLETE marker, rewrite an original report, or authorize the final gate.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path

from . import continue_collection_096 as continuation
from . import contrast_audit_100 as corrected_audit
from . import m5_ml_terminal_prefix as terminal
from .costs_guarded_093 import summarize

PLAN_SHA = '0d6e759270e23db8d54f5cc7a5a0009a8af366e8d9ace23cf66beb648b76bc27'
KNOWN_CAUSE = 'Annotation audit: presentation matches verified inputs and latest requested history'


def read(path):
    return json.loads(Path(path).read_text())


def cost_facts(value):
    """Only observation time and local evidence location may differ on replay."""
    return {k: v for k, v in value.items() if k not in ('at', 'root')}


def validate_reports(plan, manifest, done, original_exit, original_failure,
                     corrected, costs, final_runtime):
    """Full-cohort consistency only; process/archive/source checks are separate."""
    if (original_exit != {'exit_status': 1, 'final_gate_opened': False}
            or original_failure != KNOWN_CAUSE):
        raise ValueError('Only the reproduced known audit failure is eligible')
    expected = {(a, c['series_id'], c['round'], c['origin'])
                for a in plan['arms'] for c in plan['cases']}
    rows = corrected['rows']
    if (len(expected) != 312 or len(rows) != 312
            or {(r['arm'], r['series_id'], r['round'], r['origin']) for r in rows} != expected
            or sum(r['round'] < 3 for r in rows) != 36
            or corrected['complete'] is not True or corrected['audit_failures']
            or corrected['shutdown_record_gaps'] or corrected['audit_checks'] <= 0):
        raise ValueError('Complete independently audited fixed cohort required')
    if (done != {'completed': 312, 'planned': 312, 'new_sessions': 276,
                 'retained_pilot_sessions': 36, 'source_unchanged': True}
            or manifest['planned'] != 312 or manifest['pilot_sessions_retained'] != 36
            or manifest['new_sessions_planned'] != 276
            or manifest['accuracy_used_for_promotion'] is not False
            or manifest['sources'] != plan['capsule']['sources']
            or manifest['source_jobs_sha256'] != plan['task_source_sha256']
            or manifest['model'] != plan['agent']['model']
            or manifest['build']['package_version'] != plan['gnomon']['version']
            or manifest['build']['source_sha256'] != plan['gnomon']['build_source_sha256']
            or final_runtime != manifest['inventory']):
        raise ValueError('Original execution completeness or identity mismatch')
    if (costs['deduplicated_total']['sessions'] != 312
            or costs['stages']['retained_pilot']['sessions'] != 36
            or costs['stages']['continuation']['sessions'] != 276):
        raise ValueError('Every retained and new session must remain in costs')
    # Do not require positive accuracy, successful agents, or known billing.
    return {'sessions': 312, 'retained_sessions': 36, 'new_sessions': 276,
            'audit_checks': corrected['audit_checks'], 'accuracy_used_for_admission': False}


def replay_audits(helper, output, destination):
    """Reproduce the old error, then change only the annotation audit callable."""
    frozen_analyzer = helper.analyze
    with (destination/'original-audit.stdout').open('x') as stdout, \
            (destination/'original-audit.stderr').open('x') as stderr, \
            redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            frozen_analyzer(output, destination/'original-audit')
        except ValueError as exc:
            cause = str(exc)
            (destination/'original-audit-failure.json').write_text(
                json.dumps({'type': type(exc).__name__, 'cause': cause}, indent=2)+'\n')
            if cause != KNOWN_CAUSE:
                raise ValueError('Original auditor failed for a different reason') from exc
        else:
            raise ValueError('Known frozen audit failure did not reproduce')
    namespace = frozen_analyzer.__globals__
    old = namespace['audit_annotations']
    try:
        namespace['audit_annotations'] = corrected_audit.audit_annotations
        report = frozen_analyzer(output, destination/'corrected-audit')
    finally:
        namespace['audit_annotations'] = old
    return cause, report


def reconcile(output, launch, capsule, runtime, plan_path, destination, *, proc=Path('/proc')):
    output, launch, capsule, runtime, plan_path, destination = map(
        lambda p: Path(p).absolute(), (output, launch, capsule, runtime, plan_path, destination))
    if destination.exists() or destination.is_symlink():
        raise ValueError('Fresh reconciliation destination required')
    for protected in (output, launch, capsule, runtime, plan_path):
        if (destination == protected or destination in protected.parents
                or protected in destination.parents):
            raise ValueError('Reconciliation must be outside retained evidence and runtime')
    if terminal.digest(plan_path) != PLAN_SHA:
        raise ValueError('Exact prospectively frozen candidate-100 plan required')
    for name in ('launch.json', 'development-process.json'):
        identity = read(launch/name)
        # A copied receipt checked against another machine's /proc cannot prove
        # the original process has stopped. This narrow path requires the same
        # boot; a reboot needs separately authenticated host evidence.
        if identity.get('boot_id') != (proc/'sys/kernel/random/boot_id').read_text().strip():
            raise ValueError('Reconcile on the original host and boot; copied process receipts are insufficient')
        terminal.terminal(identity, proc)
    if (not (launch/'INCOMPLETE.json').is_file()
            or read(launch/'development-exit.json') != {'exit_status': 1}
            or read(launch/'FINISHED.json').get('complete') is not False):
        raise ValueError('Original terminal failure and completed archive required')
    original_files = terminal.archived_prefix(output, launch)
    original_launch = terminal.inventory(launch)
    plan = read(plan_path)
    if (terminal.digest(capsule/'capsule.json') != plan['capsule_sha256']
            or read(capsule/'capsule.json') != plan['capsule']):
        raise ValueError('Original capsule identity required')
    source_before = {str(p): terminal.digest(p) for p in
                     (plan_path, Path(__file__), Path(corrected_audit.__file__))}
    destination.mkdir(parents=True)
    dump = lambda name, value: (destination/name).write_text(json.dumps(value, indent=2)+'\n')
    dump('inputs.json', {'output': str(output), 'launch': str(launch), 'capsule': str(capsule),
                        'source_sha256': source_before, 'plan_sha256': PLAN_SHA,
                        'original_archive_sha256': terminal.digest(launch/'evidence.tar.gz')})
    try:
        # configure authenticates all frozen worker sources. analyze reads saved
        # executions; neither run/session/forecast nor key() is called here.
        helper = continuation.configure(capsule, runtime)
        cause, report = replay_audits(helper, output, destination)
        costs = summarize(output)
        checked = validate_reports(plan, read(output/'manifest.json'), read(output/'complete.json'),
                                   read(output/'runner-exit.json'), cause, report, costs,
                                   read(output/'final-runtime-inventory.json'))
        if cost_facts(costs) != cost_facts(read(launch/'costs.json')):
            raise ValueError('Recomputed costs differ from retained controller costs')
        dump('costs.json', costs)
        if (terminal.inventory(output) != original_files or terminal.inventory(launch) != original_launch
                or {p: terminal.digest(p) for p in source_before} != source_before):
            raise ValueError('Original evidence or reconciliation sources changed')
        proof = {**checked, 'status': 'known_audit_failure_reconciled',
                 'original_failure_preserved': True, 'original_exit_status': 1,
                 'reproduced_cause': cause, 'originals_unchanged': True,
                 'corrected_report_sha256': terminal.digest(destination/'corrected-audit/report.json'),
                 'original_finished_sha256': terminal.digest(launch/'FINISHED.json'),
                 'corrected_auditor_sha256': terminal.digest(corrected_audit.__file__),
                 'engy_calls': 0, 'refits': 0, 'execution_authorized': False,
                 'final_gate_opened': False, 'objective_established': False}
        dump('receipt.json', proof)
        return proof
    except BaseException as exc:
        dump('FAILED.json', {'type': type(exc).__name__, 'message': str(exc),
                             'automatic_retry': False, 'execution_authorized': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'launch', 'capsule', 'runtime', 'plan', 'destination'):
        parser.add_argument('--'+name, required=True, type=Path)
    a = parser.parse_args()
    print(json.dumps(reconcile(a.output, a.launch, a.capsule, a.runtime, a.plan, a.destination)))


if __name__ == '__main__':
    main()
