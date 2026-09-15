"""One-shot paid admission for the frozen candidate-107 development stages."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

from . import planning_host_107 as host
from . import control_collection_096 as control
from . import continue_collection_096 as continuation
from . import reconcile_contrast_100 as reconciliation

PLAN_SHA = '99683f980b5a4b31ff39886d385e57c87b1f0d0b5e714c638323638974c4cb48'
REMOTE_BASE = Path('/root/gnomon-ledger-ml-v3/code/results')
REGISTRY = REMOTE_BASE/'planning-107-dispatch-registry-001'
PATHS = {'pilot': (REMOTE_BASE/'planning-107-pilot-001', REMOTE_BASE/'planning-107-pilot-launch-001'),
         'complete': (REMOTE_BASE/'planning-107-development-001', REMOTE_BASE/'planning-107-continuation-launch-001')}


def dispatch_sources():
    """Bind indirect host imports too; worker code is bound by its capsule."""
    root = Path(__file__).parent
    queue = [Path(__file__).stem, *(Path(n).stem for n in host.source_identity())]; result = {}
    while queue:
        name = queue.pop(); path = root/(name+'.py')
        if path.name in result: continue
        result[path.name] = host.sha(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                names = [node.module.split('.')[0]] if node.module else [n.name.split('.')[0] for n in node.names]
                queue.extend(n for n in names if (root/(n+'.py')).is_file())
    return result


def validate_host_proof(proof, sources):
    required = {'passed': True, 'engy_calls': 0, 'synthetic_sessions': 312,
                'retained_pilot_sessions': 72, 'new_continuation_sessions': 240,
                'numerical_attempts': 2496, 'scripted_responses': 1248, 'final_gate_opened': False}
    if any(type(proof.get(k)) is not type(v) or proof[k] != v for k, v in required.items()):
        raise ValueError('Complete 72+240 synthetic host execution required')
    if proof.get('host_sources') != sources:
        raise ValueError('Host proof does not cover current exact source')
    for key in ('pilot_terminal_sha256', 'complete_terminal_sha256'):
        v = proof.get(key)
        if not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError('Both terminal synthetic stage identities required')


def verify_predecessor(output, launch, original_plan, receipt_root, *, proc=Path('/proc')):
    """Admit only the narrowly reconciled original failure, retaining its status."""
    output, launch, receipt_root = map(Path, (output, launch, receipt_root))
    if host.sha(original_plan) != reconciliation.PLAN_SHA:
        raise ValueError('Original candidate-100 plan identity required')
    for name in ('launch.json', 'development-process.json'):
        identity = host.read(launch/name)
        if identity['boot_id'] != (proc/'sys/kernel/random/boot_id').read_text().strip():
            raise ValueError('Predecessor must be verified on its original host boot')
        host.terminal(identity, proc)
    finished = host.read(launch/'FINISHED.json'); receipt = host.read(receipt_root/'receipt.json')
    if (finished.get('complete') is not False or not (launch/'INCOMPLETE.json').exists()
            or host.read(launch/'development-exit.json') != {'exit_status': 1}
            or receipt.get('status') != 'known_audit_failure_reconciled'
            or receipt.get('original_failure_preserved') is not True
            or receipt.get('original_exit_status') != 1 or receipt.get('originals_unchanged') is not True
            or receipt.get('engy_calls') != 0 or receipt.get('refits') != 0
            or receipt.get('original_finished_sha256') != host.sha(launch/'FINISHED.json')
            or receipt.get('corrected_report_sha256') != host.sha(receipt_root/'corrected-audit/report.json')
            or receipt.get('corrected_auditor_sha256') != host.sha(Path(reconciliation.corrected_audit.__file__))):
        raise ValueError('Exact retained failure and independently corrected report required')
    # Check current live copies and every archived member, without extracting.
    files = host.archived_prefix(output, launch)
    report = host.read(receipt_root/'corrected-audit/report.json')
    costs = host.read(receipt_root/'costs.json')
    checked = reconciliation.validate_reports(host.read(original_plan), host.read(output/'manifest.json'),
        host.read(output/'complete.json'), host.read(output/'runner-exit.json'), receipt['reproduced_cause'],
        report, costs, host.read(output/'final-runtime-inventory.json'))
    if reconciliation.cost_facts(costs) != reconciliation.cost_facts(host.read(launch/'costs.json')):
        raise ValueError('Reconciled and original cost facts differ')
    return {**checked, 'receipt_sha256': host.sha(receipt_root/'receipt.json'),
            'original_finished_sha256': host.sha(launch/'FINISHED.json'),
            'retained_original_files': len(files), 'original_failure_preserved': True}


def verify_inputs(args):
    if host.sha(args.plan) != PLAN_SHA: raise ValueError('Exact prospective candidate-107 plan required')
    plan = host.read(args.plan); capsule = host.read(args.capsule/'capsule.json')
    if (plan['host_sources'] != host.source_identity() or capsule != plan['capsule']
            or host.sha(args.capsule/'capsule.json') != plan['capsule_sha256']
            or plan['synthetic'] is not False or plan['final_gate_opened'] is not False):
        raise ValueError('Prospective source contract changed')
    package = args.capsule/'benchmarks/hermes_ml_checkpoint_v6'
    if {p.name: host.sha(p) for p in package.iterdir() if p.is_file()} != capsule['sources']:
        raise ValueError('Worker source changed')
    if host.sha(args.task_source) != plan['task_source_sha256']: raise ValueError('Development task source changed')
    for name, reference in plan['proofs'].items():
        if host.sha(args.proofs/name) != reference['sha256']:
            raise ValueError('Frozen worker proof changed: '+name)
    proof = host.read(args.host_preflight); validate_host_proof(proof, plan['host_sources'])
    synthetic_plan = host.read(args.proofs/'host-test-plan')
    if (synthetic_plan.get('synthetic') is not True or synthetic_plan.get('final_gate_opened') is not False
            or any(synthetic_plan.get(k) != plan[k] for k in
                   ('capsule', 'host_sources', 'runtime_inventory', 'build', 'requested_seed', 'arms', 'planned'))):
        raise ValueError('Host preflight must use the same worker, runtime and stage contract')
    for stage in ('pilot', 'complete'):
        finished = host.read(args.proofs/(stage+'-host-terminal'))
        if (host.sha(args.proofs/(stage+'-host-terminal')) != proof[stage+'_terminal_sha256']
                or finished.get('complete') is not True or finished.get('exit_status') != 0
                or stage == 'pilot' and finished.get('continuation_gate_passed') is not True):
            raise ValueError('Synthetic terminal archive completion mismatch')
    admission = host.read(args.admission)
    if (admission.get('plan_sha256') != PLAN_SHA
            or admission.get('launcher_sha256') != host.sha(__file__)
            or admission.get('dispatch_sources') != dispatch_sources()
            or admission.get('host_preflight_sha256') != host.sha(args.host_preflight)
            or admission.get('host_test_plan_sha256') != host.sha(args.proofs/'host-test-plan')
            or admission.get('predecessor_receipt_sha256') != host.sha(args.predecessor_receipt/'receipt.json')
            or admission.get('registry') != str(REGISTRY)
            or admission.get('paths') != {k: [str(p) for p in v] for k, v in PATHS.items()}
            or admission.get('final_gate_opened') is not False):
        raise ValueError('Exact source-bound one-shot admission manifest required')
    jobs = {s: [{k: j[k] for k in host.FIELDS} for j in rows] for s, rows in sorted(host.read(args.task_source).items())}
    host.validate_jobs(jobs)
    return plan, jobs


def reserve(stage, binding, *, registry=REGISTRY):
    """Atomic exclusive stage claim; a failed attempt is not silently reusable."""
    if stage not in PATHS: raise ValueError('Unknown stage')
    registry = Path(registry); registry.mkdir(parents=True, exist_ok=True)
    path = registry/stage; path.mkdir(exist_ok=False)
    host.dump(path/'reservation.json', {**binding, 'stage': stage, 'controller': control.identity(os.getpid())})
    return path


def consume(stage, binding, *, registry=REGISTRY):
    path = Path(registry)/stage; receipt = host.read(path/'reservation.json')
    if any(receipt.get(k) != v for k, v in binding.items()) or receipt.get('stage') != stage:
        raise ValueError('Worker reservation differs from the admitted controller')
    identity = receipt['controller']; stat = Path('/proc')/str(identity['pid'])/'stat'
    if not stat.exists(): raise ValueError('Admitting controller is no longer live')
    fields = stat.read_text().split(') ', 1)[1].split()
    if (fields[0] == 'Z' or fields[19] != identity['start_ticks']
            or identity['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        raise ValueError('Admitting controller identity changed')
    host.dump(path/'worker.json', {**control.identity(os.getpid()), **binding})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pilot', 'complete'), required=True)
    parser.add_argument('--role', choices=('controller', 'worker'), default='controller')
    for name in ('plan', 'capsule', 'runtime', 'task-source', 'proofs', 'host-preflight', 'admission',
                 'predecessor-output', 'predecessor-launch', 'predecessor-plan', 'predecessor-receipt'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--credentials-file', type=Path)
    parser.add_argument('--check-only', action='store_true'); args = parser.parse_args()
    output, launch = PATHS[args.stage]
    if output.exists() or output.is_symlink() or args.role == 'controller' and launch.exists():
        raise ValueError('Never restart an existing stage output or controller')
    plan, jobs = verify_inputs(args)
    helper = continuation.configure(args.capsule, args.runtime, args.task_source)
    if helper.run.runtime_inventory() != plan['runtime_inventory']:
        raise ValueError('Actual runtime inventory differs from verified plan')
    python = args.runtime.resolve()/'gnomon-venv/bin/python'
    build = json.loads(subprocess.check_output([str(python), '-I', '-c',
        'import json;from gnomon.build_info import build_info;print(json.dumps(build_info()))'], text=True))
    if build != plan['build']:
        raise ValueError('Actual installed build differs from verified plan')
    predecessor = verify_predecessor(args.predecessor_output, args.predecessor_launch,
                                     args.predecessor_plan, args.predecessor_receipt)
    binding = {'plan_sha256': PLAN_SHA, 'admission_sha256': host.sha(args.admission),
               'launcher_sha256': host.sha(__file__), 'output': str(output), 'launch': str(launch),
               'predecessor_receipt_sha256': predecessor['receipt_sha256']}
    if args.stage == 'complete':
        host.verify_pilot(*PATHS['pilot'], jobs, plan, PLAN_SHA)
    if args.check_only:
        print(json.dumps({'checks_passed': True, 'reserved': False, 'credential_read': False,
                          'provider_calls': 0, 'engy_calls': 0, 'final_gate_opened': False})); return
    if args.credentials_file is None: raise ValueError('Credential file required after read-only checks')
    if args.role == 'controller':
        reserve(args.stage, binding)
        command = [sys.executable, '-m', 'benchmarks.ledger_optimization.launch_planning_107',
                   '--role', 'worker', '--stage', args.stage]
        for name, value in vars(args).items():
            if isinstance(value, Path): command += ['--'+name.replace('_', '-'), str(value.absolute())]
        result = control.supervise(command, launch, output, args.credentials_file)
        print(json.dumps(result))
        if not result['complete']: raise SystemExit(1)
    else:
        consume(args.stage, binding)
        def credential():
            for line in args.credentials_file.read_text().splitlines():
                if line.startswith('ENGY_API_KEY='):
                    value = line.split('=', 1)[1].strip().strip('"').strip("'")
                    if value: return value
            raise ValueError('Engy credential unavailable')
        result = host.execute_stage(helper, args.plan, jobs, output, stage=args.stage, credential=credential,
                                    pilot=PATHS['pilot'][0], pilot_launch=PATHS['pilot'][1])
        print(json.dumps(result))


if __name__ == '__main__': main()
