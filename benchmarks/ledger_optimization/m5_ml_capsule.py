"""Build an offline M5 development worker from the fixed candidate-100 capsule.

Authenticates existing development artifacts; never opens the source archive,
reserved target values, credentials, or a network connection. The generated
worker still needs separate cohort-aware launch/continuation admission and
exact-source integration tests. Building it does not select a final candidate.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

from .m5_ml_development_contract import authenticated_contract, DEVELOPMENT_JOBS_SHA

PARENT_SHA = 'ea0920fd92a5e888816a93ef3ef95c88dcea8482bd5a541bc987c7f78c082519'
AVAILABILITY = ('M5 development replay: valid timestamps are daily period ends; '
                'source and recording availability are assumed at period end, '
                'not measured vintages. Complete forecast scores mature at horizon close.')
PROMOTION = 'unavailable_assumed_zero; not an observed absence of promotions'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Unexpected frozen source boundary: '+old[:80])
    return text.replace(old, new, 1)


def build(source, manifest_path, jobs_path, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink() or source in output.resolve().parents:
        raise ValueError('Require a fresh capsule outside the frozen parent')
    manifest_bytes = Path(manifest_path).read_bytes()
    jobs_bytes = Path(jobs_path).read_bytes()
    cohort = authenticated_contract(manifest_bytes, jobs_bytes)
    parent_bytes = (source/'capsule.json').read_bytes()
    if digest(parent_bytes) != PARENT_SHA:
        raise ValueError('Require the exact frozen candidate-100 parent')
    parent = json.loads(parent_bytes)
    package = source/'benchmarks/hermes_ml_checkpoint_v6'
    if any(p.is_symlink() or (p.is_dir() and p.name != '__pycache__') for p in package.iterdir()):
        raise ValueError('Unexpected parent source entry')
    content = {p.name: p.read_bytes() for p in package.iterdir() if p.is_file()}
    if {n: digest(raw) for n, raw in content.items()} != parent['sources']:
        raise ValueError('Parent sources changed')
    runner = content['run.py'].decode()
    runner = replace_once(runner,
        "SOURCE_SHA='cf7bdd21e216e84809edb8653710e0c0755402864201400c5749a8dbff00f561'",
        'SOURCE_SHA='+repr(DEVELOPMENT_JOBS_SHA))
    runner = replace_once(runner,
        "TASK_SOURCE=Path(os.environ.get('LEDGER_ML_TASK_SOURCE', str(REPO/'results/ledger-ml-continuous-022/export-002/host-jobs.json'))).resolve()",
        "TASK_SOURCE=Path(os.environ.get('LEDGER_ML_TASK_SOURCE', str(REPO/'development-jobs.json'))).resolve()")
    runner = replace_once(runner,
        "'availability':'Synthetic replay: source at valid time, recording at horizon close, not measured vintages.'",
        "'availability':"+repr(AVAILABILITY)+",'promotion':"+repr(PROMOTION))
    runner = replace_once(runner, "'model':MODEL})",
        "'model':MODEL,'dataset':'m5_development','cohort_contract_sha256':"+
        repr(digest((json.dumps(cohort, sort_keys=True, separators=(',', ':'))+'\n').encode()))+"})")
    content['run.py'] = runner.encode()
    content['TASK.md'] += ('\nM5 data semantics: '+AVAILABILITY+'\n'
        'The onpromotion column is an unavailable-data placeholder set to zero, '
        'not evidence that promotions were absent. Weekday features describe '
        'the sales day preceding its period-end timestamp.\n').encode()
    content['PROTOCOL.md'] += ('\nM5 development integration only: eight fixed items '
        'from two stores, 26 origins, three arms, 72 pilot and 552 continuation '
        'sessions per seed. '+AVAILABILITY+' Promotions: '+PROMOTION+'.\n'
        'No final candidate is selected or reserved data admitted by this build.\n').encode()
    fragments = {}
    for name in ('contrast_audit_100.py', 'm5_ml_development_contrast.py'):
        raw = Path(__file__).with_name(name).read_bytes()
        fragments[name] = digest(raw)
        # Existing capsule convention supports package and subprocess imports.
        content[name] = re.sub(r'^from \.(\w+) import (.+)$',
            lambda m: 'if __package__:\n    from .'+m[1]+' import '+m[2]+
                      '\nelse:\n    from '+m[1]+' import '+m[2], raw.decode(), flags=re.MULTILINE).encode()
    analyzer = content['analyze.py'].decode()
    fn = next(n for n in ast.parse(analyzer).body
              if isinstance(n, ast.FunctionDef) and n.name == 'paired_series_contrast')
    analyzer = replace_once(analyzer, ast.get_source_segment(analyzer, fn),
                           'from .m5_ml_development_contrast import paired_series_contrast')
    analyzer = replace_once(analyzer,
        'Four reused development series and one requested seed cannot establish broad superiority.',
        'Eight fixed development items share two stores. Matched scores are descriptive; '
        'no independent-item uncertainty interval or final superiority claim is supplied.')
    content['analyze.py'] = analyzer.encode()
    for name, raw in content.items():
        if name.endswith('.py'):
            compile(raw, name, 'exec')
    changed = sorted(n for n, raw in content.items() if parent['sources'].get(n) != digest(raw))
    if changed != ['PROTOCOL.md', 'TASK.md', 'analyze.py', 'contrast_audit_100.py',
                   'm5_ml_development_contrast.py', 'run.py']:
        raise ValueError('Unexpected M5 worker source change set')
    cohort_raw = (json.dumps(cohort, sort_keys=True, separators=(',', ':'))+'\n').encode()
    result = {'status': 'offline_m5_development_worker_not_dispatch_ready',
              'parent_capsule_sha256': PARENT_SHA, 'parent_sources': parent['sources'],
              'input_identity': cohort['input_identity'],
              'cohort_contract_sha256': digest(cohort_raw),
              'planned_per_seed': cohort['decisions_per_seed'],
              'changed_modules': changed, 'sources': {n: digest(raw) for n, raw in content.items()},
              'fragment_sources': fragments, 'builder_sha256': digest(Path(__file__).read_bytes()),
              'common_to_all_arms': True,
              'workflow_audit_sha256': parent['workflow_audit_sha256'],
              'arms': cohort['arms'], 'provider_calls': 0, 'engy_calls': 0,
              'selected_final_candidate': False, 'execution_authorized': False,
              'final_gate_opened': False,
              'remaining': ['cohort-aware launch and continuation gates',
                            'exact-source multi-series resumed-state integration',
                            'frozen prospective runtime, seeds and budget plan']}
    target = output/'benchmarks/hermes_ml_checkpoint_v6'
    target.mkdir(parents=True, exist_ok=False)
    for name, raw in content.items():
        (target/name).write_bytes(raw)
    (output/'cohort-contract.json').write_bytes(cohort_raw)
    (output/'capsule.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'manifest', 'jobs', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.manifest, args.jobs, args.output), indent=2))
