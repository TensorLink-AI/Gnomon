"""Freeze seed-specific M5 worker copies; never dispatch or open data targets."""
import argparse
import hashlib
import json
from pathlib import Path

from .m5_ml_capsule import replace_once

PARENT_SHA256 = 'e4a51304b46c87c2f83546eae07d983747ef9139db4975064098ebda88fc1f81'
SEEDS = (7, 19)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def build(source, output, seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Only the prospective requested seeds 7 and 19 are supported')
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink() or source in output.resolve().parents:
        raise ValueError('Require a fresh capsule outside the authenticated parent')
    raw = (source/'capsule.json').read_bytes()
    if digest(raw) != PARENT_SHA256:
        raise ValueError('Require the exact verified M5 development capsule')
    parent = json.loads(raw)
    contract = (source/'cohort-contract.json').read_bytes()
    if digest(contract) != parent['cohort_contract_sha256']:
        raise ValueError('Cohort contract changed')
    package = source/'benchmarks/hermes_ml_checkpoint_v6'
    if any(p.is_symlink() or p.is_dir() and p.name != '__pycache__' for p in package.iterdir()):
        raise ValueError('Unexpected parent source entry')
    content = {p.name: p.read_bytes() for p in package.iterdir() if p.is_file()}
    before = {n: digest(v) for n, v in content.items()}
    if before != parent['sources']:
        raise ValueError('Parent worker sources changed')
    replacements = {
        'worker.py': ("request_overrides={'temperature':0.2,'seed':7}",
                      "request_overrides={'temperature':0.2,'seed':"+str(seed)+'}'),
        'transport.py': ('payload.update(model=MODEL, temperature=0.2, seed=7, max_tokens=3072, stream=False)',
                         f'payload.update(model=MODEL, temperature=0.2, seed={seed}, max_tokens=3072, stream=False)'),
        'analyze.py': ("'temperature': .2, 'seed': 7, 'max_tokens': 3072, 'stream': False}",
                       "'temperature': .2, 'seed': "+str(seed)+", 'max_tokens': 3072, 'stream': False}"),
    }
    for name, (old, new) in replacements.items():
        content[name] = replace_once(content[name].decode(), old, new).encode()
    if seed != 7:
        content['PROTOCOL.md'] += (
            '\nRequested agent-seed replication: 19. All three arms use 19 in both '
            'worker and forwarded model requests. The readiness canary remains '
            'seed 7, temperature 0, 16 tokens and is not an agent decision. '
            'Numerical provider random states, data, budgets and memory rules are '
            'unchanged. Use separate processes and fresh seed-specific output roots. '
            'Requested API seeds do not establish provider determinism.\n').encode()
    for name, value in content.items():
        if name.endswith('.py'):
            compile(value, name, 'exec')
    after = {n: digest(v) for n, v in content.items()}
    changed = sorted(n for n in after if before.get(n) != after[n])
    if changed != ([] if seed == 7 else ['PROTOCOL.md', 'analyze.py', 'transport.py', 'worker.py']):
        raise ValueError('Unexpected seed-specific source changes')
    result = {**parent, 'status': 'offline_m5_seed_worker_not_dispatch_ready',
              'parent_capsule_sha256': PARENT_SHA256, 'parent_sources': before,
              'requested_seed': seed, 'sources': after, 'seed_changed_files': changed,
              'builder_sha256': digest(Path(__file__).read_bytes()),
              'parameters': {'model': 'deepseek-v4.1-flash', 'temperature': .2,
                             'seed': seed, 'max_tokens': 3072},
              'readiness_canary_seed': 7, 'numerical_provider_random_states_changed': False,
              'seed_semantics': 'Requested upstream seed; provider determinism or seed honoring not established.',
              'remaining': ['exact-source seed-specific full worker integration',
                            'cohort-aware launch/continuation and cross-seed isolation gates',
                            'prospective runtime, seeds and budget freeze'],
              'execution_authorized': False, 'final_gate_opened': False}
    target = output/'benchmarks/hermes_ml_checkpoint_v6'
    target.mkdir(parents=True, exist_ok=False)
    for name, value in content.items():
        (target/name).write_bytes(value)
    (output/'cohort-contract.json').write_bytes(contract)
    (output/'capsule.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.seed), indent=2))
