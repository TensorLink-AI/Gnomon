"""Generate forecasts once on CPU, with durable per-call receipts and bounded workers."""
import os
# Set before scientific libraries load, including in worker processes.
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import fcntl
import json
import math
import multiprocessing as mp
from pathlib import Path
import resource
import time

from .common import digest, identity, read, resources, write
from .data import forward_fill
from .models import forecast


def worker(connection):
    while True:
        try:
            request = connection.recv()
        except EOFError:
            return
        start = time.monotonic()
        before = resource.getrusage(resource.RUSAGE_SELF)
        try:
            point = [float(x) for x in forecast(**request)]
            if len(point) != request['horizon'] or not all(math.isfinite(x) for x in point):
                raise ValueError('invalid forecast shape or nonfinite values')
            result = {'status':'ok', 'point':point}
        except Exception as exc:
            result = {'status':'error', 'error': f'{type(exc).__name__}: {exc}'[:1000]}
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(seconds=time.monotonic()-start,
                      cpu_seconds=usage.ru_utime+usage.ru_stime-before.ru_utime-before.ru_stime,
                      peak_rss_kib=usage.ru_maxrss)
        connection.send(result)


class CPUWorker:
    """One serial process caches library imports, never fitted models or predictions."""
    def __init__(self):
        self.process = None
        self.parent = None

    def close(self):
        if self.parent is not None:
            self.parent.close()
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
        self.process = self.parent = None

    def predict(self, request, timeout):
        started = time.monotonic()
        if self.process is None:
            context = mp.get_context('spawn')
            self.parent, child = context.Pipe()
            self.process = context.Process(target=worker,args=(child,))
            self.process.start()
            child.close()
        try:
            self.parent.send(request)
            if self.parent.poll(timeout):
                result = self.parent.recv()
            else:
                result = {'status':'timeout','error':'per-model deadline','cpu_seconds':None}
                self.close()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            result = {'status':'error','error':'worker exited without a result','cpu_seconds':None}
            self.close()
        result['dispatch_seconds'] = time.monotonic()-started
        return result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def bounded_forecast(request, timeout):
    with CPUWorker() as cpu:
        return cpu.predict(request, timeout)


def validate_manifest(manifest):
    if digest({k:v for k,v in manifest.items() if k != 'sha256'}) != manifest['sha256']:
        raise ValueError('Manifest hash mismatch')
    if manifest['mode'] not in ('pilot','evaluation'):
        raise ValueError('Unknown mode')


def run(manifest, output):
    validate_manifest(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol, spec = manifest['protocol'], manifest['spec']
    pinned = {'manifest_sha256':manifest['sha256'], 'runtime':identity()}
    write(output/'identity.json', pinned)
    if not (output/'resources.json').exists():
        write(output/'resources.json', resources())
    write(output/'manifest.json', manifest)
    cap = protocol['pilot_wall_seconds'] if manifest['mode']=='pilot' else protocol['evaluation_wall_seconds']
    # Durable dispatch records reserve the whole timeout before work. An interrupted
    # call cannot be retried under the same identity, or erase its budget reservation.
    if (output/'COMPLETE.json').exists():
        complete = read(output/'COMPLETE.json')
        receipts = {p.name:digest(read(p)) for p in sorted((output/'calls').glob('*.json'))}
        if complete['identity'] != digest(pinned) or complete['receipts'] != receipts:
            raise ValueError('Completed receipt set changed')
        return
    used = 0.0
    for path in (output/'calls').glob('*.request.json'):
        request_receipt = read(path)
        receipt = path.with_name(path.name.replace('.request.json','.result.json'))
        if not receipt.exists():
            raise ValueError(f'Unfinished attempt: {path}; preserve and inspect before a new run')
        result = read(receipt)
        if result['request_sha256'] != digest(request_receipt):
            raise ValueError('Result/request mismatch')
        used += result['dispatch_seconds']
        if request_receipt['identity'] != digest(pinned):
            raise ValueError('Call identity mismatch')
    with CPUWorker() as cpu:
        for series in manifest['series']:
            for index, phase in zip(series['origins'],series['phases']):
                history = forward_fill(series['values'][:index])[-protocol['max_history']:]
                for name in protocol['models']:
                    key = digest([series['id'],index,name])
                    stem = output/'calls'/key
                    request = {'name':name, 'history':history, 'horizon':spec['horizon'],
                               'season':spec['season'], 'seed':protocol['seed']}
                    plan = {'identity':digest(pinned), 'series':series['id'], 'origin_index':index,
                            'phase':phase, 'model':name, 'request_sha256':digest(request)}
                    result_path = Path(str(stem)+'.result.json')
                    if result_path.exists():
                        if read(str(stem)+'.request.json') != plan:
                            raise ValueError('Saved request differs')
                        continue
                    if used + protocol['model_timeout_seconds'] > cap:
                        raise RuntimeError(f'Wall-budget admission stopped at {used:.2f}s of {cap}s; partial calls retained')
                    if identity() != pinned['runtime']:
                        raise RuntimeError('Sources/dependencies changed during execution')
                    write(str(stem)+'.request.json',plan)
                    result = cpu.predict(request, protocol['model_timeout_seconds'])
                    used += result['dispatch_seconds']
                    if result['status'] != 'ok':
                        result['point'] = [float(x) for x in forecast('seasonal_naive',history,spec['horizon'],spec['season'],protocol['seed'])]
                        result['fallback'] = 'seasonal_naive'
                    result['request_sha256'] = digest(plan)
                    write(result_path, result)
                print(json.dumps({'dataset':spec['id'],'series':series['id'],'origin':index,'phase':phase,
                                  'used_dispatch_seconds':round(used,3)}), flush=True)
    if identity() != pinned['runtime']:
        raise RuntimeError('Sources/dependencies changed during execution')
    receipts = {p.name:digest(read(p)) for p in sorted((output/'calls').glob('*.json'))}
    write(output/'COMPLETE.json', {'identity':digest(pinned), 'receipts':receipts,
                                 'dispatch_seconds':used})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(read(args.manifest),args.output)


if __name__ == '__main__':
    main()
