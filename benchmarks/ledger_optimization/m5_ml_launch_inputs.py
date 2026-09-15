"""Authenticate/freeze M5 launch inputs and reserve a stage exactly once.

No worker imports, credentials, API calls or final targets. A frozen plan and
reservation are necessary but insufficient for dispatch: the launcher must
also verify terminal predecessors, actual runtime and copied-state evidence.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from .m5_ml_development_contract import authenticated_contract, DEVELOPMENT_JOBS_SHA

# Retained from the active three-arm prospective plan.
PARENT_PLAN_SHA = '0d6e759270e23db8d54f5cc7a5a0009a8af366e8d9ace23cf66beb648b76bc27'
BUILD_SHA = '9723394ccb6d9e11991b312e01bac47c767c69407b6b33d36971cb6e48b6a22e'
IDENTITIES = {
    7: {'capsule.json': 'e4a51304b46c87c2f83546eae07d983747ef9139db4975064098ebda88fc1f81',
        'passed.json': 'ba697e0de5484af148c6110393e7592433a6b4e4f29f10bdf8cfe134174c93d5',
        'report.json': '59d5a78953b594559d5a34beee7ee22e27116e9eb6728429df403fe56d127ded',
        'manifest.json': '123273702ac98c4eae16a2fde80043e252003ed95ce5b162de1cd44985077ebe'},
    19: {'capsule.json': 'fd5130ceeda34e6524558acc14fc33b9846015b8fd31ceac45eca494a9af2f5a',
         'passed.json': '9096b36c3dd5926ebf9661286c29c1ade29afc57431d89bee2d12a466ba84bd8',
         'report.json': 'fb9ff094f0b2e54c3cafa04bad3513b045910a04d74b146c6e88b42383947097',
         'manifest.json': '094e09cd0c0c7e1fa0591d062f506ef915a78820deaf1f55b9d76e0583e74fd9'},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authenticated_json(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha(path) != expected:
        raise ValueError('Exact frozen input required: '+path.name)
    return json.loads(path.read_text())


def prepare_plan(parent_path, capsule_root, worker_root, manifest_bytes, jobs_bytes,
                 *, seed, registry):
    if type(seed) is not int or seed not in IDENTITIES:
        raise ValueError('Only requested seeds 7 and 19 are admitted')
    cohort = authenticated_contract(manifest_bytes, jobs_bytes)
    parent = authenticated_json(parent_path, PARENT_PLAN_SHA)
    capsule_root, worker_root = Path(capsule_root), Path(worker_root)
    expected = IDENTITIES[seed]
    capsule = authenticated_json(capsule_root/'capsule.json', expected['capsule.json'])
    if capsule.get('requested_seed', 7) != seed:
        raise ValueError('Capsule seed mismatch')
    package = capsule_root/'benchmarks/hermes_ml_checkpoint_v6'
    if package.is_symlink() or any(p.is_symlink() or p.is_dir() and p.name != '__pycache__' for p in package.iterdir()):
        raise ValueError('Unexpected capsule source entries')
    sources = {p.name: sha(p) for p in package.iterdir() if p.is_file()}
    if sources != capsule['sources']:
        raise ValueError('Frozen worker sources changed')
    if sha(capsule_root/'cohort-contract.json') != capsule['cohort_contract_sha256']:
        raise ValueError('Frozen cohort description changed')
    proof, report, runtime = (authenticated_json(worker_root/name, expected[name])
                              for name in ('passed.json', 'report.json', 'manifest.json'))
    task_ids = {(a, 'synthetic-collection-worker', n) for a in ('plain', 'gnomon', 'ledger') for n in (0, 1)}
    if (proof.get('passed') is not True or proof.get('engy_calls') != 0
            or proof.get('requested_seed') != seed or not proof.get('checks')
            or not all(c.get('passed') is True for c in proof['checks'])
            or proof.get('sources') != sources or runtime.get('sources') != sources
            or runtime.get('requested_seed') != seed
            or proof.get('numerical_attempts') != 48 or proof.get('scripted_model_responses') != 30
            or proof.get('independent_audit_checks') != 3859
            or report.get('audit_failures') != [] or report.get('shutdown_record_gaps') != []
            or report.get('audit_checks') != 3859 or len(report.get('rows', [])) != 6
            or {(r['arm'], r['series_id'], r['round']) for r in report['rows']} != task_ids
            or not all(r['valid'] is True and r['workflow_complete'] is True for r in report['rows'])):
        raise ValueError('Complete exact-source worker preflight required')
    if runtime['build'].get('package_version') != '1.2.0' or runtime['build'].get('source_sha256') != BUILD_SHA:
        raise ValueError('Published pinned Gnomon 1.2.0 required')
    registry = Path(registry)
    if not registry.is_absolute() or '..' in registry.parts:
        raise ValueError('Explicit absolute host reservation registry required')
    return {
        'status': 'frozen_development_inputs_not_dispatch_authorization',
        'candidate': 'm5_current_history_contrast', 'requested_seed': seed,
        'parent_plan_sha256': PARENT_PLAN_SHA, 'task_source_sha256': DEVELOPMENT_JOBS_SHA,
        'capsule_sha256': expected['capsule.json'], 'capsule': capsule,
        'arms': ['plain', 'gnomon', 'ledger'], 'cases': cohort['cases'],
        'planned': {'pilot_sessions': 72, 'continuation_sessions': 552, 'total_sessions': 624},
        'agent': {**deepcopy(parent['agent']), 'seed': seed},
        'gnomon': deepcopy(parent['gnomon']), 'budgets': deepcopy(parent['budgets']),
        'runtime_inventory': runtime['inventory'], 'build': runtime['build'],
        'preflight_hashes': {k: v for k, v in expected.items() if k != 'capsule.json'},
        'pilot_gate': {'valid_per_arm': 24, 'full_workflows_at_least_per_arm': 22,
                       'accuracy_threshold': None, 'retain_pilot_once_in_continuation': True},
        'comparison': {**deepcopy(parent['comparison']),
                       'primary': 'ledger versus gnomon within this fresh M5 run',
                       'secondary': 'ledger versus plain within this fresh M5 run'},
        'efficacy_scope': 'Fixed development panel: eight series in two stores, not eight independent stores; no final-evaluation claim.',
        'dispatch_registry': str(registry),
        'final_target': deepcopy(parent['final_target']),
        'execution_authorized': False, 'final_gate_opened': False,
    }


def freeze_plan(output, *args, **kwargs):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('Fresh prospective plan path required')
    plan = prepare_plan(*args, **kwargs)
    with output.open('x') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False); stream.write('\n')
    return plan


def verify_plan(plan_path, *args, **kwargs):
    expected = prepare_plan(*args, **kwargs)
    actual = json.loads(Path(plan_path).read_text())
    if actual != expected:
        raise ValueError('Prospective plan differs from authenticated source, seed or budget')
    return actual


def reserve_stage(plan_path, verified_plan, *, stage, output, controller):
    """Atomic one-shot reservation after launcher admission, before credentials.

The launcher must pass the exact return from verify_plan. Reservation is shared
by source/capsule/seed/stage across output paths, so renaming output is no retry.
No reservation is released automatically after spawn failure or interruption.
"""
    if stage not in ('pilot', 'complete'):
        raise ValueError('Development stage required')
    if json.loads(Path(plan_path).read_text()) != verified_plan:
        raise ValueError('Verified plan changed before reservation')
    output, controller = Path(output).absolute(), Path(controller).absolute()
    registry = Path(verified_plan['dispatch_registry'])
    if (registry.is_symlink() or output.exists() or output.is_symlink()
            or controller.exists() or controller.is_symlink()
            or output == controller or output in controller.parents or controller in output.parents
            or registry == output or output in registry.parents
            or registry == controller or controller in registry.parents):
        raise ValueError('Fresh separate outputs and stable reservation registry required')
    identity = {k: verified_plan[k] for k in ('task_source_sha256', 'capsule_sha256', 'requested_seed')}
    identity['stage'] = stage
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    registry.mkdir(parents=True, exist_ok=True)
    path = registry/(key+'.json')
    value = {**identity, 'at': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha(plan_path),
             'output': str(output), 'controller': str(controller),
             'automatic_retry': False, 'final_gate_opened': False}
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return path
