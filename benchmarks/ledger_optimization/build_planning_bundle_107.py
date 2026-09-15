"""Package verified candidate-107 inputs without reserving or executing a run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

from . import launch_planning_107 as launch

host = launch.host


def verify_archive(path, expected):
    """Verify each regular member, including the inventory, without extraction."""
    seen = set()
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            if member.name in seen or member.name not in expected or not member.isfile():
                raise ValueError('Unexpected, duplicate or non-regular bundle member')
            stream = archive.extractfile(member)
            if stream is None or hashlib.file_digest(stream, 'sha256').hexdigest() != expected[member.name]:
                raise ValueError('Bundle member bytes differ from inventory')
            seen.add(member.name)
    if seen != set(expected): raise ValueError('Bundle omits inventoried files')


def build(*, output, plan, capsule, task_source, host_root, predecessor_receipt):
    paths = {k: Path(v).absolute() for k, v in locals().items()}
    output = paths.pop('output')
    if output.exists() or output.is_symlink() or any(
            p == output or p in output.parents or output in p.parents for p in paths.values()):
        raise ValueError('Fresh bundle directory outside source evidence required')
    plan, capsule, task_source, host_root, predecessor_receipt = (
        paths[k] for k in ('plan', 'capsule', 'task_source', 'host_root', 'predecessor_receipt'))
    if host.sha(plan) != launch.PLAN_SHA: raise ValueError('Exact frozen development plan required')
    frozen = host.read(plan); proof = host.read(host_root/'passed.json')
    launch.validate_host_proof(proof, host.source_identity())
    receipt = host.read(predecessor_receipt/'receipt.json')
    if (receipt.get('status') != 'known_audit_failure_reconciled'
            or receipt.get('original_failure_preserved') is not True
            or receipt.get('corrected_report_sha256') != host.sha(predecessor_receipt/'corrected-audit/report.json')):
        raise ValueError('Completed predecessor reconciliation required before packaging')
    # These checks authenticate the host proof's underlying saved executions;
    # copied terminal JSON alone is not a complete archive verification.
    archive_files = {stage: len(host.archived_prefix(host_root/stage, host_root/(stage+'-launch')))
                     for stage in ('pilot', 'complete')}
    sources = launch.dispatch_sources()
    copies = {plan: 'plan.json', task_source: 'host-jobs.json',
              host_root/'passed.json': 'host-preflight.json', host_root/'plan.json': 'proofs/host-test-plan',
              predecessor_receipt/'receipt.json': 'predecessor-receipt/receipt.json',
              predecessor_receipt/'corrected-audit/report.json': 'predecessor-receipt/corrected-audit/report.json',
              predecessor_receipt/'costs.json': 'predecessor-receipt/costs.json'}
    copies.update({Path(ref['path']): 'proofs/'+name for name, ref in frozen['proofs'].items()})
    copies.update({host_root/(stage+'-launch/FINISHED.json'): 'proofs/'+stage+'-host-terminal'
                   for stage in ('pilot', 'complete')})
    copies.update({Path(__file__).with_name(name): 'code/benchmarks/ledger_optimization/'+name for name in sources})
    copies[Path(__file__)] = 'code/benchmarks/ledger_optimization/'+Path(__file__).name
    for name, digest in host.inventory(capsule).items():
        if '__pycache__' not in Path(name).parts and not name.endswith('.pyc'):
            copies[capsule/name] = 'capsule/'+name
    before = {str(p): host.sha(p) for p in copies}
    output.mkdir(parents=True); payload = output/'payload'; payload.mkdir()
    for source, name in copies.items():
        target = payload/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in ('code/benchmarks/__init__.py', 'code/benchmarks/ledger_optimization/__init__.py'):
        (payload/name).write_text('')
    host.dump(payload/'admission.json', {
        'plan_sha256': launch.PLAN_SHA, 'launcher_sha256': sources['launch_planning_107.py'],
        'dispatch_sources': sources, 'host_preflight_sha256': host.sha(payload/'host-preflight.json'),
        'host_test_plan_sha256': host.sha(payload/'proofs/host-test-plan'),
        'predecessor_receipt_sha256': host.sha(payload/'predecessor-receipt/receipt.json'),
        'registry': str(launch.REGISTRY), 'paths': {k: list(map(str, v)) for k, v in launch.PATHS.items()},
        'final_gate_opened': False})
    args = SimpleNamespace(plan=payload/'plan.json', capsule=payload/'capsule',
        task_source=payload/'host-jobs.json', proofs=payload/'proofs',
        host_preflight=payload/'host-preflight.json', admission=payload/'admission.json',
        predecessor_receipt=payload/'predecessor-receipt')
    launch.verify_inputs(args)
    # Import from the actual isolated package, not the checkout used to build it.
    script = """from pathlib import Path
from types import SimpleNamespace
from benchmarks.ledger_optimization.launch_planning_107 import verify_inputs
p=Path('..').absolute()
verify_inputs(SimpleNamespace(plan=p/'plan.json',capsule=p/'capsule',task_source=p/'host-jobs.json',
proofs=p/'proofs',host_preflight=p/'host-preflight.json',admission=p/'admission.json',
predecessor_receipt=p/'predecessor-receipt'))
print('Isolated copied inputs verified; no reservation, credentials or provider calls.')
"""
    env = dict(os.environ); env.pop('PYTHONPATH', None); env['PYTHONDONTWRITEBYTECODE'] = '1'
    command = [sys.executable, '-c', script]
    result = subprocess.run(command, cwd=payload/'code', env=env, text=True, capture_output=True)
    (output/'copied-inputs.stdout').write_text(result.stdout); (output/'copied-inputs.stderr').write_text(result.stderr)
    host.dump(output/'copied-inputs-command.json', {'argv': command, 'cwd': str(payload/'code'), 'exit_status': result.returncode})
    result.check_returncode()
    if before != {str(p): host.sha(p) for p in copies} or sources != launch.dispatch_sources():
        raise ValueError('Input or launcher sources changed during packaging')
    index = host.inventory(payload); host.dump(payload/'SHA256SUMS.json', index)
    with tarfile.open(output/'dispatch.tar.gz', 'x:gz') as archive:
        for name in sorted([*index, 'SHA256SUMS.json']): archive.add(payload/name, arcname=name, recursive=False)
    verify_archive(output/'dispatch.tar.gz', {**index, 'SHA256SUMS.json': host.sha(payload/'SHA256SUMS.json')})
    result = {'status': 'built_not_deployed_or_launched', 'files': len(index),
        'archive_sha256': host.sha(output/'dispatch.tar.gz'), 'archive_bytes': (output/'dispatch.tar.gz').stat().st_size,
        'plan_sha256': launch.PLAN_SHA, 'admission_sha256': host.sha(payload/'admission.json'),
        'builder_sha256': host.sha(__file__), 'verified_host_archive_files': archive_files,
        'source_inputs': before, 'isolated_copied_inputs_verified': True,
        'engy_calls': 0, 'reserved': False, 'final_gate_opened': False,
        'remaining': 'Original-host predecessor/process/archive and runtime checks before one-shot pilot admission.'}
    host.dump(output/'receipt.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'plan', 'capsule', 'task-source', 'host-root', 'predecessor-receipt'):
        parser.add_argument('--'+name, required=True, type=Path)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))
