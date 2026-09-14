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
    def replace_once(name, old, new):
        old, new = old.encode(), new.encode()
        if content[name].count(old) != 1:
            raise ValueError('Expected one declared replacement in ' + name)
        content[name] = content[name].replace(old, new)

    replace_once('lab.py', "'backtest_batches_remaining': max(0, (LIMIT-attempts-1)//3),",
                 "'backtest_batches_remaining': max(0, (LIMIT-attempts-1)//4),\n"
                 "            'fresh_backtest_fits': 4, 'cv_fits': 3, 'production_collection_fits': 1,")
    replace_once('TASK.md', '   compare an ML configuration with the baseline. This is a syntax example, not a',
                 '   compare an ML configuration with the baseline and retain an unselected current-origin\n'
                 '   forecast (three CV fits plus one production fit). Collection does not select it.\n'
                 '   This is a syntax example, not a')
    replace_once('TASK.md', 'three-fold backtest reserves one final fit and is rejected before exceeding the\nbudget.',
                 'fresh backtest/collection batch costs four fits and reserves one final fit. It is\n'
                 'rejected before exceeding the budget. Completed fits are reused on an explicit\n'
                 'retry; failed fits count. The initial baseline uses four fits without another\n'
                 'reserve. A collected production forecast stays unselected until explicit commit.')
    replace_once('boundary_schemas_093.py', 'backtest requires config; ',
                 'backtest requires config and collects an unselected forecast (four fresh fits, with reuse); ')
    content['PROTOCOL.md'] += (
        '\n## Undeployed 096 collection capsule amendment\n\n'
        'Every arm collects a current-origin unselected forecast after each three-fold CV batch.\n'
        'Charge all four fits, retain a final-fit reserve, and reuse completed executions.\n'
        'Outcomes mature under unchanged source/recording rules; collection never selects.\n'
        'Original 093 is frozen separately. No paid dispatch until actual worker integration\n'
        'and a new prospective manifest pass. See capsule.json for outstanding gates.\n').encode()
    audit_path = Path(__file__).with_name('collection_audit_096.py')
    audit_fragment = audit_path.read_text()
    audit_fn = next(n for n in ast.parse(audit_fragment).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'audit_collection_events')
    analyzer_path = Path(__file__).with_name('analyze_guarded_093.py')
    content['analyze.py'] = analyzer_path.read_bytes().replace(
        b'from benchmarks.hermes_ml_checkpoint_v6.transport import dump,sha', b'from .transport import dump,sha')
    replace_once('analyze.py', 'def analyze(root, output=None):',
                 ast.get_source_segment(audit_fragment, audit_fn) + '\n\ndef analyze(root, output=None):')
    replace_once('analyze.py', '        checks+=3\n        if r[\'valid\']:',
                 "        r['collection_audit']=audit_collection_events(events,job)\n"
                 "        assert r['collection_audit']['numerical_attempts']==attempts\n"
                 "        checks+=4\n        if r['valid']:")
    compile(content['lab.py'], 'lab.py', 'exec')
    compile(content['analyze.py'], 'analyze.py', 'exec')
    package = output / 'benchmarks/hermes_ml_checkpoint_v6'
    package.mkdir(parents=True)
    for name, data in content.items():
        (package / name).write_bytes(data)
    after = inventory(content)
    manifest = {'status': 'offline_prototype_not_dispatch_ready', 'base_sources': before, 'sources': after,
                'base_inventory_sha256': digest, 'changed_files': [n for n in before if before[n] != after[n]],
                'fragment_sha256': hashlib.sha256(fragment.encode()).hexdigest(),
                'audit_fragment_sha256': hashlib.sha256(audit_fragment.encode()).hexdigest(),
                'base_local_auditor_sha256': hashlib.sha256(analyzer_path.read_bytes()).hexdigest(),
                'common_to_all_arms': True, 'engy_calls': 0, 'provider_calls': 0,
                'outstanding': ['Hermes worker/API-transport integration with the updated capsule',
                                'Frozen prospective protocol and passing dispatch gate'],
                'final_gate_opened': False}
    (output / 'capsule.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
