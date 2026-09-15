"""Read-only terminal/archive admission checks for an M5 development pilot.

This is called before copying state or reading credentials. It verifies the
complete fixed pilot and its immutable archive. It does not reserve a dispatch,
copy state, rerun the copied audit, or grant final-data access.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile

from .launch_collection_096 import require_terminal
from . import m5_ml_stage_checks as stage_checks
from .m5_ml_stage_checks import check_continuation_prefix
from .m5_ml_development_contract import DEVELOPMENT_JOBS_SHA


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Real directory required')
    files = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if not stat.S_ISDIR(mode) and not stat.S_ISREG(mode):
                raise ValueError('Links and special files are not accepted evidence')
            if stat.S_ISREG(mode):
                files[path.relative_to(root).as_posix()] = digest(path)
    return files


def terminal(identity, proc):
    if (type(identity.get('pid')) is not int or identity['pid'] <= 0
            or not str(identity.get('start_ticks', '')).isdigit()
            or not isinstance(identity.get('boot_id'), str) or not identity['boot_id']):
        raise ValueError('Complete process identity required')
    require_terminal(identity, proc)


def archived_prefix(pilot, launch):
    """Check live copies and every tar member without extracting the archive."""
    pilot, launch = Path(pilot), Path(launch)
    pilot_files, launch_files = inventory(pilot), inventory(launch)
    finished = read(launch/'FINISHED.json')
    index_path, archive_path = launch/'SHA256SUMS.json', launch/'evidence.tar.gz'
    if (digest(index_path) != finished['inventory_sha256']
            or digest(archive_path) != finished['archive_sha256']):
        raise ValueError('Terminal archive or inventory hash mismatch')
    index = read(index_path)
    if type(index) is not dict or not index:
        raise ValueError('Nonempty archived inventory required')
    for name, value in index.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts or len(path.parts) < 2
                or path.as_posix() != name or path.parts[0] not in ('pilot', 'launch')
                or not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('Unsafe or malformed archive inventory')
    if pilot_files != {k[6:]: v for k, v in index.items() if k.startswith('pilot/')}:
        raise ValueError('Pilot state differs from terminal inventory')
    # These three artifacts are created after collecting the archive file list.
    post_archive = {'FINISHED.json', 'SHA256SUMS.json', 'evidence.tar.gz'}
    if {k: v for k, v in launch_files.items() if k not in post_archive} != {
            k[7:]: v for k, v in index.items() if k.startswith('launch/')}:
        raise ValueError('Controller evidence changed or has unarchived files')
    expected = {**index, 'SHA256SUMS.json': finished['inventory_sha256']}
    seen = set()
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            if member.name not in expected or member.name in seen or not member.isfile():
                raise ValueError('Extra, duplicate or non-regular archive member')
            stream = archive.extractfile(member)
            if stream is None or hashlib.file_digest(stream, 'sha256').hexdigest() != expected[member.name]:
                raise ValueError('Archived bytes differ from terminal evidence')
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Archive omits retained evidence')
    if inventory(pilot) != pilot_files or inventory(launch) != launch_files:
        raise ValueError('Evidence changed during archive validation')
    return pilot_files


def verify_terminal_prefix(pilot, launch, plan_path, capsule_root, runtime_inventory,
                           manifest_bytes, jobs_bytes, *, proc=Path('/proc')):
    """Combine process, source-plan, full pilot and archived-byte checks.

plan_path must be the prospective plan chosen by the launcher, not a plan
substituted from the pilot's own directory. A later dispatch wrapper must bind
its hash to a one-shot reservation and recheck state immediately before use.
"""
    # Authenticate the fixed DEVELOPMENT bytes before parsing any job source.
    # The full prefix check below also binds them to retained grades.
    stage_checks.authenticated_contract(manifest_bytes, jobs_bytes)
    pilot, launch, capsule_root = map(Path, (pilot, launch, capsule_root))
    if pilot.resolve() == launch.resolve() or pilot.resolve() in launch.resolve().parents or launch.resolve() in pilot.resolve().parents:
        raise ValueError('Separate pilot and controller evidence required')
    # Reject links before reading receipt paths; never follow an evidence link.
    before_pilot, before_launch = inventory(pilot), inventory(launch)
    for marker in ('INCOMPLETE.json', 'CONTINUATION_INCOMPLETE.json'):
        if (pilot/marker).exists() or (launch/marker).exists():
            raise ValueError('Incomplete evidence requires separate reconciliation')
    for name in ('launch.json', 'pilot-process.json'):
        terminal(read(launch/name), proc)
    finished, audited = read(launch/'FINISHED.json'), read(launch/'AUDITED.json')
    if (finished.get('complete') is not True or audited.get('complete') is not True
            or finished.get('continuation_gate_passed') is not True
            or audited.get('continuation_gate_passed') is not True
            or finished.get('continuation_launched') is not False
            or read(launch/'pilot-exit.json') != {'exit_status': 0}):
        raise ValueError('Clean terminal pilot controller required')
    runner, gate = read(pilot/'runner-exit.json'), read(pilot/'GATE.json')
    if (runner != {'exit_status': 0, 'continuation_launched': False}
            or gate.get('passed') is not True or gate.get('accuracy_used_for_gate') is not False):
        raise ValueError('Pilot process/completion gate failed')
    capsule_files = inventory(capsule_root)
    capsule = read(capsule_root/'capsule.json')
    package_prefix = 'benchmarks/hermes_ml_checkpoint_v6/'
    # Worker imports may have generated bytecode; the capsule source manifest
    # binds direct source files, exactly as its builder does. All bytes still
    # remain unchanged across this check; runtime reinspection is separate.
    sources = {k[len(package_prefix):]: v for k, v in capsule_files.items()
               if k.startswith(package_prefix) and '/' not in k[len(package_prefix):]}
    if sources != capsule['sources']:
        raise ValueError('Capsule package source identity mismatch')
    plan = read(plan_path)
    if (plan.get('capsule_sha256') != digest(capsule_root/'capsule.json')
            or plan.get('capsule') != capsule
            or plan.get('task_source_sha256') != DEVELOPMENT_JOBS_SHA
            or plan.get('arms') != ['plain', 'gnomon', 'ledger']
            or plan.get('requested_seed') != capsule.get('requested_seed', 7)
            or type(plan.get('requested_seed')) is not int
            or plan.get('runtime_inventory') != runtime_inventory
            or plan.get('planned') != {'pilot_sessions': 72, 'continuation_sessions': 552, 'total_sessions': 624}
            or plan.get('final_gate_opened') is not False):
        raise ValueError('Exact M5 development source/seed/runtime plan required')
    plan_hash = digest(plan_path)
    if (digest(pilot/'pilot-plan.json') != plan_hash
            or read(pilot/'accepted-launch.json').get('plan_sha256') != plan_hash):
        raise ValueError('Pilot did not execute the supplied prospective plan')
    jobs = json.loads(jobs_bytes)
    if read(pilot/'host-jobs.json') != {s: v[:3] for s, v in jobs.items()}:
        raise ValueError('Pilot job prefix differs from fixed source')
    manifest, report = read(pilot/'manifest.json'), read(pilot/'report.json')
    if read(pilot/'final-runtime-inventory.json') != runtime_inventory:
        raise ValueError('Pilot final runtime differs from continuation runtime')
    grade_paths = list(pilot.rglob('grade.json'))
    grades = []
    for path in grade_paths:
        row = read(path)
        if path.relative_to(pilot).parts != (row['arm'], row['series_id'], f"round-{row['round']}", 'grade.json'):
            raise ValueError('Grade file location does not match its task')
        grades.append(row)
    checked = check_continuation_prefix(manifest_bytes, jobs_bytes, grades, report,
                                       pilot_manifest=manifest, capsule=capsule,
                                       runtime_inventory=runtime_inventory)
    files = archived_prefix(pilot, launch)
    if (before_pilot != inventory(pilot) or before_launch != inventory(launch)
            or capsule_files != inventory(capsule_root) or digest(plan_path) != plan_hash):
        raise ValueError('Inputs changed during terminal verification')
    return {'terminal_prefix_checks_passed': True, 'prefix_checks': checked,
            'plan_sha256': plan_hash, 'pilot_files': files,
            'pilot_finished_sha256': digest(launch/'FINISHED.json'),
            'execution_authorized': False, 'final_gate_opened': False,
            'scope': 'Terminal processes, exact prospective plan, full fixed pilot and archived bytes. '
                     'Copied-prefix audit, runtime/build reinspection and one-shot dispatch remain required.'}
