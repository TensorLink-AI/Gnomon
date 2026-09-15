"""Wait for the exact 097 processes, then invoke one frozen gated 100 controller.

No credential reads, retries, continuation, final access, or accuracy-based gate.
The existing controller/launcher independently audit all predecessor evidence and
the candidate runtime before any credential read or provider call.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

BUNDLE_SHA = 'a18612bcd7ed3b3dc58b8803cb2471654916cfef7d05f2f7c3b352f29f40a052'
PLAN_SHA = '0d6e759270e23db8d54f5cc7a5a0009a8af366e8d9ace23cf66beb648b76bc27'
FIELDS = ('bundle', 'previous_root', 'previous_launch', 'runtime', 'launch_directory',
          'output', 'credentials_file')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def process_live(record, proc=Path('/proc')):
    """Missing/reused/zombie identities are terminal; unreadable state is an error."""
    try:
        fields = (proc/str(record['pid'])/'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError:
        return False
    boot = (proc/'sys/kernel/random/boot_id').read_text().strip()
    return (boot == record['boot_id'] and fields[19] == str(record['start_ticks'])
            and fields[0] != 'Z')


def validate_bundle(bundle):
    bundle = Path(bundle)
    if (bundle/'RETIRED.json').exists():
        raise ValueError('Bundle retired; no automatic replacement or dispatch')
    if sha(bundle/'dispatch.tar.gz') != BUNDLE_SHA:
        raise ValueError('Exact staged bundle 002 required')
    payload = bundle/'payload'
    if bundle.is_symlink() or payload.is_symlink() or (payload/'SHA256SUMS.json').is_symlink():
        raise ValueError('Bundle symlink rejected')
    with tarfile.open(bundle/'dispatch.tar.gz', 'r:gz') as archive:
        archived_inventory = archive.extractfile('SHA256SUMS.json').read()
    if (payload/'SHA256SUMS.json').read_bytes() != archived_inventory:
        raise ValueError('Inventory differs from the pinned archive')
    inventory = read(payload/'SHA256SUMS.json')
    if len(inventory) != 79 or sha(payload/'plan.json') != PLAN_SHA:
        raise ValueError('Exact corrected plan and bundle inventory required')
    actual = {str(p.relative_to(payload)) for p in payload.rglob('*') if p.is_file()}
    if actual != set(inventory) | {'SHA256SUMS.json'}:
        raise ValueError('Unexpected or missing payload files')
    for name, digest in inventory.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe bundle inventory path')
        if any(p.is_symlink() for p in (payload/relative, *(payload/p for p in relative.parents))):
            raise ValueError('Bundle symlink rejected')
        if sha(payload/relative) != digest:
            raise ValueError('Bundle payload changed')
    return payload


def validate_paths(config, state):
    if set(config) != set(FIELDS) or any(not isinstance(v, str) or not Path(v).is_absolute() for v in config.values()):
        raise ValueError('Exact absolute-path configuration required')
    paths = {k: Path(v) for k, v in config.items()}
    outputs = [Path(state), paths['launch_directory'], paths['output']]
    protected = [paths[k] for k in ('bundle', 'previous_root', 'previous_launch', 'runtime', 'credentials_file')]
    for path in outputs:
        if not path.is_absolute() or path.exists() or path.is_symlink():
            raise ValueError('Separate fresh absolute output paths required')
    for i, path in enumerate(outputs):
        for other in outputs[i+1:] + protected:
            a, b = path.resolve(), other.resolve()
            if a == b or a in b.parents or b in a.parents:
                raise ValueError('Output overlaps source, runtime, credentials or retained evidence')
    return paths


def controller_command(paths, payload):
    fields = {'parent-plan': payload/'parent-plan.json', 'capsule': payload/'capsule',
              'worker-proof-root': payload/'worker-proof', 'plan': payload/'plan.json',
              'preflight': payload/'preflight.json', 'task-source': payload/'host-jobs.json',
              'runtime': paths['runtime'], 'previous-root': paths['previous_root'],
              'previous-launch': paths['previous_launch'], 'previous-capsule': payload/'previous-capsule',
              'predecessor-audit': paths['launch_directory']/'predecessor-audit',
              'output': paths['output'], 'credentials-file': paths['credentials_file']}
    command = [sys.executable, '-u', '-m', 'benchmarks.ledger_optimization.control_contrast_100',
               '--launch-directory', str(paths['launch_directory'])]
    for key, value in fields.items():
        command += ['--'+key, str(value)]
    return command


def run(config, state, *, proc=Path('/proc'), sleep=time.sleep, popen=subprocess.Popen, poll_seconds=30):
    if type(poll_seconds) is not int or not 1 <= poll_seconds <= 60:
        raise ValueError('Poll interval must be 1–60 seconds')
    paths = validate_paths(config, state)
    payload = validate_bundle(paths['bundle'])
    identity_files = [paths['previous_launch']/n for n in ('launch.json', 'development-process.json')]
    identities = [read(p) for p in identity_files]
    identity_hashes = {str(p): sha(p) for p in identity_files}
    state = Path(state); state.mkdir(parents=True)
    dump(state/'config.json', config)
    dump(state/'accepted-wait.json', {'at': now(), 'bundle_sha256': BUNDLE_SHA, 'plan_sha256': PLAN_SHA,
        'waiter_sha256': sha(__file__), 'predecessor_identity_hashes': identity_hashes,
        'automatic_retry': False, 'continuation_authorized': False, 'final_gate_opened': False})
    try:
        while True:
            if any(sha(Path(p)) != h for p, h in identity_hashes.items()):
                raise ValueError('Predecessor process receipt changed')
            live = [process_live(record, proc) for record in identities]
            with (state/'polls.jsonl').open('a') as stream:
                stream.write(json.dumps({'at': now(), 'controller_live': live[0], 'worker_live': live[1]})+'\n')
            if not any(live):
                break
            sleep(poll_seconds)
        # A vanished process alone is not a successful terminal result.
        finished = read(paths['previous_launch']/'FINISHED.json')
        if (paths['previous_launch']/'INCOMPLETE.json').exists() or finished.get('complete') is not True:
            raise ValueError('Predecessor incomplete; candidate remains stopped')
        payload = validate_bundle(paths['bundle'])
        if any(paths[k].exists() or paths[k].is_symlink() for k in ('launch_directory', 'output')):
            raise ValueError('Candidate output already reserved; never retry')
        command = controller_command(paths, payload)
        # The reservation precedes spawning; even spawn failure cannot be retried.
        reservation = paths['bundle']/'dispatch-reservation.json'
        dump(reservation, {'at': now(), 'argv': command, 'wait_state': str(state),
                          'automatic_retry': False, 'predecessor_finished_sha256': sha(paths['previous_launch']/'FINISHED.json')})
        dump(state/'dispatch.json', {'argv': command, 'cwd': str(payload/'code'), 'at': now()})
        env = dict(os.environ)
        for name in ('PYTHONPATH', 'ENGY_API_KEY'):
            env.pop(name, None)
        env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (state/'controller.stdout').open('xb') as out, (state/'controller.stderr').open('xb') as err:
            child = popen(command, cwd=payload/'code', env=env, stdin=subprocess.DEVNULL,
                          stdout=out, stderr=err, start_new_session=True)
            try:
                fields = (proc/str(child.pid)/'stat').read_text().split(') ', 1)[1].split()
                identity = {'start_ticks': fields[19], 'boot_id': (proc/'sys/kernel/random/boot_id').read_text().strip()}
            except FileNotFoundError:
                identity = {'identity_unavailable': 'process already absent; do not infer restart permission'}
            dump(state/'controller-process.json', {'pid': child.pid, 'at': now(), **identity})
            status = child.wait()
        dump(state/'controller-exit.json', {'exit_status': status})
        completion = read(paths['launch_directory']/'FINISHED.json')
        if status != 0 or completion.get('complete') is not True:
            raise ValueError('Candidate controller did not finish cleanly; no retry')
        result = {'status': 'pilot_controller_finished', 'at': now(),
                  'pilot_finished_sha256': sha(paths['launch_directory']/'FINISHED.json'),
                  'continuation_launched': False, 'final_gate_opened': False}
        dump(state/'FINISHED.json', result)
        return result
    except BaseException as exc:
        dump(state/'STOPPED.json', {'at': now(), 'cause': type(exc).__name__, 'message': str(exc),
             'automatic_retry': False, 'continuation_launched': False, 'final_gate_opened': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(read(args.config), args.state)))


if __name__ == '__main__':
    main()
