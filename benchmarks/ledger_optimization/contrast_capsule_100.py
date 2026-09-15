"""Build an isolated, undispatched evidence-contrast capsule from frozen 097."""
import ast
import hashlib
import json
import re
from pathlib import Path

BASE_CAPSULE_SHA = '3b2220767e916c8977317e3916c7bb39b9b68a0717dc43156797e1a982e45dd3'
MODULES = ('paired_consistency_098.py', 'current_cv_pair_100.py',
           'current_history_contrast_100.py', 'contrast_view_100.py',
           'contrast_records_100.py', 'contrast_annotations_100.py')
AUDIT_MODULES = ('contrast_audit_100.py',)


def build(source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink() or source in output.resolve().parents:
        raise ValueError('Require a fresh capsule outside the frozen source')
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    if digest((source/'capsule.json').read_bytes()) != BASE_CAPSULE_SHA:
        raise ValueError('Expected the exact frozen 097 capsule')
    parent = json.loads((source/'capsule.json').read_text())
    package = source/'benchmarks/hermes_ml_checkpoint_v6'
    if any(p.is_symlink() or (p.is_dir() and p.name != '__pycache__') for p in package.iterdir()):
        raise ValueError('Unexpected frozen source entry')
    content = {p.name: p.read_bytes() for p in package.iterdir() if p.is_file() and not p.is_symlink()}
    if {name: digest(raw) for name, raw in content.items()} != parent['sources']:
        raise ValueError('Frozen 097 sources changed')
    fragments = {}
    for name in (*MODULES, *AUDIT_MODULES):
        raw = Path(__file__).with_name(name).read_bytes()
        fragments[name] = digest(raw)
        if name in content: raise ValueError('Unexpected source module collision')
        content[name] = re.sub(r'^from \.(\w+) import (.+)$',
            lambda m: 'if __package__:\n    from .'+m[1]+' import '+m[2]+
                      '\nelse:\n    from '+m[1]+' import '+m[2], raw.decode(), flags=re.MULTILINE).encode()
    lab = content['lab.py'].decode()
    lab = lab.replace('import core\n', 'import core\nfrom contrast_annotations_100 import annotate\n', 1)
    marker = "            answer = {'status': 'ok', 'operation': args.operation, 'result': result, 'budget': budget()}\n"
    if lab.count(marker) != 1: raise ValueError('Unexpected lab response boundary')
    lab = lab.replace(marker,
        "            evidence_summary = annotate(core, args.operation, result, requested_pair=args.pair)\n"+marker+
        "            if evidence_summary is not None: answer['evidence_summary'] = evidence_summary\n")
    content['lab.py'] = lab.encode()
    runner = content['run.py'].decode()
    node = next(n for n in ast.parse(runner).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'PROJECT_FILES' for t in n.targets))
    original = ast.get_source_segment(runner, node)
    files = ast.literal_eval(node.value)
    runner = runner.replace(original, 'PROJECT_FILES='+repr((*files, *MODULES)), 1)
    content['run.py'] = runner.encode()
    analyzer = content['analyze.py'].decode()
    marker = '        rows.append(r)\n'
    if analyzer.count(marker) != 1: raise ValueError('Unexpected analyzer row boundary')
    analyzer = 'from .contrast_audit_100 import audit_annotations\n'+analyzer
    analyzer = analyzer.replace(marker, "        r['contrast_annotation_audit'] = audit_annotations(p.parent)\n"+
        "        checks += r['contrast_annotation_audit']['checks']\n"+marker)
    content['analyze.py'] = analyzer.encode()
    content['TASK.md'] += (
        '\nSuccessful lab replies include evidence_summary with every complete current\n'
        'configuration, calculated CV ranks and fold scores, including the baseline.\n'
        'A focused pair compares the latest trial with the lowest current-CV alternative\n'
        '(or the two lowest current means on other calls). This selects a comparison,\n'
        'not your forecast. Other configurations stay visible in the table. Ledger\n'
        'history uses only the last review you requested for this task; it makes no\n'
        'extra query. Treat absent matches as unknown and one origin as sparse.\n'
        'Full pair evidence is available via evidence_read using the returned path\n'
        'and SHA-256. Retrieval consumes the existing request/time budget.\n').encode()
    content['PROTOCOL.md'] += (
        '\n## Undispatched 100 evidence contrast amendment\n\n'
        'All arms receive the same current-CV table from trusted completed records.\n'
        'Only ledger adds already-retrieved historical comparisons. Full artifacts\n'
        'and execution-prefix hashes are retained. Fits, raw information, budget,\n'
        'model choices and native-memory availability remain as in 097. No new\n'
        'provider or historical query is called by annotation. Source modules are\n'
        'host protected. Requires annotation audit, real Hermes preflight and a\n'
        'new frozen prospective plan after 097 completes. No final-set access.\n').encode()
    for name, raw in content.items():
        if name.endswith('.py'): compile(raw, name, 'exec')
    hashes = {name: digest(raw) for name, raw in content.items()}
    manifest = {**parent, 'candidate': '100_current_history_contrast',
                'status': 'offline_prototype_not_dispatch_ready',
                'parent_capsule_sha256': BASE_CAPSULE_SHA, 'parent_sources': parent['sources'],
                'sources': hashes, 'fragment_sha256': fragments,
                'capsule_builder_sha256': digest(Path(__file__).read_bytes()),
                'changed_from_097': sorted(n for n in content if hashes[n] != parent['sources'].get(n)),
                'common_to_all_arms': True, 'historical_overlay_arm': 'ledger',
                'engy_calls': 0, 'provider_calls': 0, 'final_gate_opened': False,
                'outstanding': ['independent annotation audit', 'real Hermes retrieval preflight',
                                'terminal 097 development audit', 'new prospective plan and pilot']}
    target = output/'benchmarks/hermes_ml_checkpoint_v6'; target.mkdir(parents=True)
    for name, raw in content.items(): (target/name).write_bytes(raw)
    (output/'capsule.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
