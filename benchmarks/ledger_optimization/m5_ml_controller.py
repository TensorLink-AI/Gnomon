"""Archive one M5 development stage, never retrying or promoting on accuracy.

The caller supplies a separately admitted launch command. This controller
authenticates the development cohort, supervises the process and preserves
complete/failed evidence and costs; it does not replace launch admission.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from . import control_collection_096 as archive
from . import costs_guarded_093 as costs
from . import m5_ml_stage_checks as checks
from .m5_ml_development_contract import DEVELOPMENT_JOBS_SHA


def read(path):
    return json.loads(Path(path).read_text())


def completion(output, status, manifest_bytes, jobs_bytes, *, stage):
    """Evidence completion and pilot quality are deliberately different states."""
    if stage not in ('pilot', 'complete'):
        raise ValueError('Only fixed M5 development stages are supported')
    base = {'stage': stage, 'exit_status': status, 'final_gate_opened': False,
            'accuracy_used_for_completion': False, 'continuation_launched': False}
    if status != 0:
        return {**base, 'complete': False, 'cause': 'stage_process_failed',
                'continuation_gate_passed': False}
    try:
        output = Path(output)
        cohort = checks.authenticated_contract(manifest_bytes, jobs_bytes)
        jobs = json.loads(jobs_bytes)
        expected_jobs = {s: rows[:3] if stage == 'pilot' else rows for s, rows in jobs.items()}
        if read(output/'host-jobs.json') != expected_jobs:
            raise ValueError('Retained job source differs from fixed stage')
        manifest, runner, done = (read(output/n) for n in ('manifest.json', 'runner-exit.json', 'complete.json'))
        count = cohort['decisions_per_seed']['pilot' if stage == 'pilot' else 'total']
        if (type(manifest.get('planned')) is not int or manifest['planned'] != count
                or manifest.get('source_jobs_sha256') != DEVELOPMENT_JOBS_SHA
                or type(manifest.get('requested_seed')) is not int
                or manifest['requested_seed'] not in (7, 19)
                or done.get('completed') != count or done.get('planned') != count
                or done.get('source_unchanged') is not True):
            raise ValueError('Stage count/source/seed completion mismatch')
        expected_runner = ({'exit_status': 0, 'continuation_launched': False} if stage == 'pilot'
                           else {'exit_status': 0, 'final_gate_opened': False})
        if runner != expected_runner:
            raise ValueError('Stage runner exit receipt mismatch')
        if stage == 'complete' and (
                manifest.get('pilot_sessions_retained') != 72 or manifest.get('new_sessions_planned') != 552
                or manifest.get('accuracy_used_for_promotion') is not False
                or done.get('retained_pilot_sessions') != 72 or done.get('new_sessions') != 552):
            raise ValueError('Retained/new session counts mismatch')
        grades = []
        for path in output.rglob('grade.json'):
            row = read(path)
            if path.relative_to(output).parts != (row['arm'], row['series_id'], f"round-{row['round']}", 'grade.json'):
                raise ValueError('Unexpected grade location')
            grades.append(row)
        stage_check = checks.check_development_stage(manifest_bytes, jobs_bytes, grades,
                        read(output/'report.json'), stage=stage, require_pilot_quality=False)
        gate_passed = stage == 'pilot' and stage_check['pilot_quality_passed']
        if stage == 'pilot':
            gate = read(output/'GATE.json')
            if gate.get('accuracy_used_for_gate') is not False or type(gate.get('passed')) is not bool or gate['passed'] != gate_passed:
                raise ValueError('Pilot gate does not match recomputed workflow completion')
        return {**base, 'complete': True, 'cause': None, 'sessions': count,
                'retained_pilot_sessions': 72, 'new_sessions': 0 if stage == 'pilot' else 552,
                'continuation_gate_passed': gate_passed, 'stage_checks': stage_check}
    except (OSError, ValueError, KeyError, TypeError):
        return {**base, 'complete': False, 'cause': 'stage_evidence_missing_or_inconsistent',
                'continuation_gate_passed': False}


def supervise(command, launch, output, manifest_bytes, jobs_bytes, *, stage,
              credentials_file=None):
    """Run one admitted command; retain failure, unknown usage and original bytes."""
    if stage not in ('pilot', 'complete'):
        raise ValueError('Only fixed M5 development stages are supported')
    checks.authenticated_contract(manifest_bytes, jobs_bytes)
    launch, output = Path(launch).absolute(), Path(output).absolute()
    if (launch == output or launch in output.parents or output in launch.parents
            or launch.exists() or launch.is_symlink() or output.exists() or output.is_symlink()):
        raise ValueError('Separate fresh controller and stage output directories required')
    if type(command) is not list or not command or any(not isinstance(a, str) for a in command):
        raise ValueError('Explicit subprocess argv required')
    launch.mkdir(parents=True)
    label = 'pilot' if stage == 'pilot' else 'development'
    archive.dump(launch/'launch.json', {**archive.identity(os.getpid()),
        'at': datetime.now(timezone.utc).isoformat(), 'stage': stage,
        'controller_sha256': archive.sha(__file__),
        'automatic_retry': False, 'final_gate_opened': False})
    archive.dump(launch/'command.json', {'argv': command, 'cwd': str(Path.cwd()), 'output': str(output)})
    for name, module in [('controller-source.py', __file__),
                         ('archive-helper-source.py', archive.__file__),
                         ('cost-accounting-source.py', costs.__file__),
                         ('stage-check-source.py', checks.__file__)]:
        (launch/name).write_bytes(Path(module).read_bytes())
    with (launch/(label+'.stdout')).open('xb') as stdout, (launch/(label+'.stderr')).open('xb') as stderr:
        try:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        except OSError as exc:
            archive.dump(launch/'spawn-error.json', {'type': type(exc).__name__}); status = 127
        else:
            # A short-lived child remains waitable until reaped, so identity is
            # retained even for immediate failure. No retry branch exists.
            archive.dump(launch/(label+'-process.json'), archive.identity(child.pid))
            status = child.wait()
    archive.dump(launch/(label+'-exit.json'), {'exit_status': status})
    result = completion(output, status, manifest_bytes, jobs_bytes, stage=stage)
    if output.exists():
        try:
            accounting = costs.summarize(output)
            if result['complete'] and (
                    accounting['deduplicated_total']['sessions'] != result['sessions']
                    or accounting['stages']['retained_pilot']['sessions'] != 72
                    or accounting['stages']['continuation']['sessions'] != result['new_sessions']):
                result.update(complete=False, cause='cost_session_count_mismatch', continuation_gate_passed=False)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            accounting = {'complete': False, 'cause': 'cost_accounting_failed', 'type': type(exc).__name__}
            result.update(complete=False, cause='cost_accounting_failed', continuation_gate_passed=False)
        archive.dump(launch/'costs.json', accounting)
    archive.dump(launch/'AUDITED.json', result)
    if not result['complete']:
        archive.dump(launch/'INCOMPLETE.json', result)
    forbidden = []
    if credentials_file is not None and (output/'accepted-launch.json').exists():
        for line in Path(credentials_file).read_text().splitlines():
            if line.startswith('ENGY_API_KEY='):
                value = line.split('=', 1)[1].strip().strip('"').strip("'")
                if value:
                    forbidden.append(value.encode())
        if not forbidden:
            raise ValueError('Credential scan unavailable; archive withheld')
    archived = archive.archive_evidence(launch, output, forbidden)
    archive.dump(launch/'FINISHED.json', {**result, **archived,
        'at': datetime.now(timezone.utc).isoformat(), 'further_execution_launched': False})
    return result
