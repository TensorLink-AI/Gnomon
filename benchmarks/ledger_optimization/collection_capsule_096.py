"""Build an offline-only common collection capsule; never dispatch or overwrite."""
import ast
import hashlib
import json
from pathlib import Path

from .seed_capsule_093 import SOURCE, BASE_INVENTORY_SHA256, inventory


def build(output, source=SOURCE):
    output, source = Path(output), Path(source)
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError('A fresh capsule outside frozen sources is required')
    content = {}
    for p in source.iterdir():
        if p.is_symlink() or (p.is_dir() and p.name != '__pycache__'):
            raise ValueError('Unexpected source entry')
        if p.is_file():
            content[p.name] = p.read_bytes()
    before = inventory(content)
    digest = hashlib.sha256(json.dumps(before, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != BASE_INVENTORY_SHA256:
        raise ValueError('Frozen 093 inventory mismatch')
    text = content['lab.py'].decode()
    matches = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'backtest']
    if len(matches) != 1:
        raise ValueError('Exactly one original backtest function required')
    fragment = Path(__file__).with_name('collection_backtest_096.py').read_text()
    replacement = next(n for n in ast.parse(fragment).body if isinstance(n, ast.FunctionDef) and n.name == 'backtest')
    lines = text.splitlines(keepends=True)
    node = matches[0]
    content['lab.py'] = (''.join(lines[:node.lineno-1]) + ast.get_source_segment(fragment, replacement)
                         + '\n' + ''.join(lines[node.end_lineno:])).encode()
    compile(content['lab.py'], 'lab.py', 'exec')
    package = output / 'benchmarks/hermes_ml_checkpoint_v6'
    package.mkdir(parents=True)
    for name, data in content.items():
        (package / name).write_bytes(data)
    after = inventory(content)
    manifest = {'status': 'offline_prototype_not_dispatch_ready', 'base_sources': before, 'sources': after,
                'base_inventory_sha256': digest, 'changed_files': [n for n in before if before[n] != after[n]],
                'fragment_sha256': hashlib.sha256(fragment.encode()).hexdigest(),
                'common_to_all_arms': True, 'engy_calls': 0, 'provider_calls': 0,
                'outstanding': ['Actual worker/model and maturation integration',
                                'Agent-visible task/budget descriptions and analyzer admission support',
                                'Frozen prospective protocol and passing dispatch gate'],
                'final_gate_opened': False}
    (output / 'capsule.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
