"""Build isolated, hash-pinned source copies for prospective seed replication.

This does not dispatch agents, read data/credentials or open the final gate.
Each capsule must run in its own process with its own project/memory directories.
"""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'hermes_ml_checkpoint_v6'
BASE_INVENTORY_SHA256 = '4ead8b05a2b15b2e9cb2a03c33fd7555dc04d1b985d23db2b5993cba1b61b121'
SEEDS = (7, 19)


def inventory(content):
    return {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}


def build(output, seed, source=SOURCE):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Only prospective requested seeds 7 and 19 are supported')
    output, source = Path(output), Path(source)
    if output.exists() or output.is_symlink():
        raise ValueError('Refuse to overwrite a capsule')
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError('Capsule must be outside the frozen source directory')
    content = {}
    for path in source.iterdir():
        if path.is_symlink():
            raise ValueError('Frozen source cannot contain symlinks')
        if path.is_file():
            content[path.name] = path.read_bytes()
        elif path.name != '__pycache__':
            raise ValueError('Unexpected source directory')
    before = inventory(content)
    digest = hashlib.sha256(json.dumps(before, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != BASE_INVENTORY_SHA256:
        raise ValueError('Source differs from the audited 24-file guarded093 base')
    replacements = {
        'worker.py': (b"request_overrides={'temperature':0.2,'seed':7}",
                      ("request_overrides={'temperature':0.2,'seed':" + str(seed) + '}').encode()),
        'transport.py': (b'payload.update(model=MODEL, temperature=0.2, seed=7, max_tokens=3072, stream=False)',
                         f'payload.update(model=MODEL, temperature=0.2, seed={seed}, max_tokens=3072, stream=False)'.encode()),
    }
    for name, (old, new) in replacements.items():
        if content[name].count(old) != 1:
            raise ValueError('Seed replacement must match exactly one declared setting')
        content[name] = content[name].replace(old, new)
    package = output / 'benchmarks/hermes_ml_checkpoint_v6'
    package.mkdir(parents=True, exist_ok=False)
    for name, data in content.items():
        (package / name).write_bytes(data)
    after = inventory(content)
    manifest = {'status': 'source_capsule_prepared_not_dispatched', 'requested_seed': seed,
                'base_inventory_sha256': digest, 'base_sources': before, 'sources': after,
                'changed_files': sorted(n for n in before if before[n] != after[n]),
                'worker_module': 'benchmarks.hermes_ml_checkpoint_v6.worker',
                'isolation': 'Fresh process per capsule; separate memory/projects per seed and arm.',
                'parameters': {'model': 'deepseek-v4.1-flash', 'temperature': .2,
                               'seed': seed, 'max_tokens': 3072},
                'base_protocol_note': 'Copied PROTOCOL.md describes seed 7 base; this manifest declares the requested replication seed.',
                'seed_semantics': 'Requested API seed; provider determinism or honoring seeds is not established.',
                'provider_calls': 0, 'engy_calls': 0, 'final_gate_opened': False}
    (output / 'capsule.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.seed), indent=2))
