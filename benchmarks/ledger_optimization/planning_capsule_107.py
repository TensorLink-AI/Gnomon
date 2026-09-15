"""Build an undispatched recipe-plan worker from the exact candidate-100 source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

PARENT_SHA = 'ea0920fd92a5e888816a93ef3ef95c88dcea8482bd5a541bc987c7f78c082519'
MODULES = ('planning_recipes_106.py', 'planning_sequence_107.py', 'planning_annotations_107.py')


def build(source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink() or source in output.resolve().parents:
        raise ValueError('Fresh capsule outside original source required')
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    if digest((source/'capsule.json').read_bytes()) != PARENT_SHA:
        raise ValueError('Exact prospectively executed candidate-100 capsule required')
    parent = json.loads((source/'capsule.json').read_text())
    package = source/'benchmarks/hermes_ml_checkpoint_v6'
    if any(p.is_symlink() or p.is_dir() and p.name != '__pycache__' for p in package.iterdir()):
        raise ValueError('Unexpected original source entries')
    content = {p.name: p.read_bytes() for p in package.iterdir() if p.is_file()}
    if {name: digest(raw) for name, raw in content.items()} != parent['sources']:
        raise ValueError('Original source changed')
    fragments = {}
    for name in (*MODULES, 'contrast_audit_100.py'):
        raw = Path(__file__).with_name(name).read_bytes(); fragments[name] = digest(raw)
        if name in MODULES and name in content: raise ValueError('New module collision')
        content[name] = re.sub(r'^from \.(\w+) import (.+)$',
                               lambda m: 'if __package__:\n    from .'+m[1]+' import '+m[2]+
                                         '\nelse:\n    from '+m[1]+' import '+m[2],
                               raw.decode(), flags=re.MULTILINE).encode()
    lab = content['lab.py'].decode()
    marker = "            if evidence_summary is not None: answer['evidence_summary'] = evidence_summary\n"
    if lab.count(marker) != 1: raise ValueError('Unexpected response boundary')
    lab = lab.replace('import core\n', 'import core\nfrom planning_annotations_107 import annotate_planning\n', 1)
    lab = lab.replace(marker, marker+
                      "            recipe_plan = annotate_planning(core, args.operation, result, budget=answer['budget'], checkpoint_getter=checkpoint)\n"+
                      "            if recipe_plan is not None: answer['recipe_plan'] = recipe_plan\n")
    content['lab.py'] = lab.encode()
    runner = content['run.py'].decode()
    node = next(n for n in ast.parse(runner).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'PROJECT_FILES' for t in n.targets))
    runner = runner.replace(ast.get_source_segment(runner, node),
                            'PROJECT_FILES='+repr((*ast.literal_eval(node.value), *MODULES)), 1)
    content['run.py'] = runner.encode()
    content['TASK.md'] += (
        '\nIf a lab reply includes recipe_plan, it lists optional historical recipes\n'
        'to test, ordered by evidence support rather than forecast accuracy. Its\n'
        'next_call respects the current checkpoint prerequisite and budget snapshot.\n'
        'A conditional recipe is not admissible until the prerequisite succeeds;\n'
        'recheck time and budget before each step. Do not execute every suggestion.\n'
        'Inspect losses/ties in the referenced evidence and choose explicitly after\n'
        'current backtesting. Full plan artifacts are available through evidence_read.\n').encode()
    content['PROTOCOL.md'] += (
        '\n## Undispatched 107 planning-sequence extension\n\n'
        'Only ledger exposes a compact recipe plan using its latest requested\n'
        'review page. The hook performs no fit, forecast selection or additional\n'
        'historical query. Common numerical, completion and information-access\n'
        'rules are unchanged. Plan artifacts and annotation records are retained.\n'
        'The offline cache-order auditor is corrected; frozen execution behavior\n'
        'is unchanged by that correction. Independent plan audit, actual Hermes\n'
        'preflight and a new prospective paid plan remain required.\n').encode()
    for name, raw in content.items():
        if name.endswith('.py'): compile(raw, name, 'exec')
    hashes = {name: digest(raw) for name, raw in content.items()}
    unchanged = ('numerical.py', 'core.py', 'maturation.py', 'policy.py', 'worker.py',
                 'execution_boundary_093.py', 'transport.py', 'service_admission.py')
    if any(hashes[n] != parent['sources'][n] for n in unchanged):
        raise ValueError('Common execution, model or outcome behavior changed')
    manifest = {**parent, 'candidate': '107_checkpoint_first_historical_recipes',
                'status': 'offline_recipe_plan_worker_not_dispatch_ready',
                'parent_capsule_sha256': PARENT_SHA, 'parent_sources': parent['sources'],
                'sources': hashes, 'recipe_fragment_sha256': fragments,
                'capsule_builder_sha256': digest(Path(__file__).read_bytes()),
                'changed_from_100': sorted(n for n in hashes if hashes[n] != parent['sources'].get(n)),
                'historical_overlay_arm': 'ledger', 'additional_ledger_queries': 0,
                'engy_calls': 0, 'provider_calls': 0, 'final_gate_opened': False,
                'outstanding': ['independent recipe annotation audit', 'actual Hermes boundary preflight',
                                'prospective matched paid plan and dispatch gates']}
    target = output/'benchmarks/hermes_ml_checkpoint_v6'; target.mkdir(parents=True)
    for name, raw in content.items(): (target/name).write_bytes(raw)
    (output/'capsule.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return {k: manifest[k] for k in ('status', 'candidate', 'changed_from_100', 'outstanding')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))
