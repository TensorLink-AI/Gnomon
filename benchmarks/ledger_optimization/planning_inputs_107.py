"""Freeze candidate-107 development inputs without reserving or spending."""
import argparse
import hashlib
import json
from pathlib import Path

from . import planning_host_107 as host

PARENT_PLAN_SHA = '0d6e759270e23db8d54f5cc7a5a0009a8af366e8d9ace23cf66beb648b76bc27'
TASK_SHA = 'cf7bdd21e216e84809edb8653710e0c0755402864201400c5749a8dbff00f561'


def build(*, parent_plan, task_source, capsule, worker_root, recipe_audit, fault_probe, output):
    paths = {k: Path(v).absolute() for k, v in locals().items()}
    destination = paths.pop('output')
    if destination.exists() or destination.is_symlink() or any(
            p == destination or p in destination.parents or destination in p.parents for p in paths.values()):
        raise ValueError('Fresh plan directory outside source evidence required')
    if host.sha(paths['parent_plan']) != PARENT_PLAN_SHA or host.sha(paths['task_source']) != TASK_SHA:
        raise ValueError('Exact prospectively fixed candidate-100 development jobs required')
    parent = host.read(paths['parent_plan']); cap = host.read(paths['capsule']/'capsule.json')
    if cap['candidate'] != '107_checkpoint_first_historical_recipes' or cap['parent_capsule_sha256'] != parent['capsule_sha256']:
        raise ValueError('Exact candidate-107 lineage required')
    package = paths['capsule']/'benchmarks/hermes_ml_checkpoint_v6'
    if {p.name: host.sha(p) for p in package.iterdir() if p.is_file()} != cap['sources']:
        raise ValueError('Capsule source changed')
    worker = host.read(paths['worker_root']/'passed.json'); manifest = host.read(paths['worker_root']/'manifest.json')
    report = host.read(paths['worker_root']/'report.json'); recipe = host.read(paths['recipe_audit']/'report.json')
    faults = host.read(paths['fault_probe']/'report.json')
    if (worker.get('passed') is not True or worker['sessions'] != 15 or worker['engy_calls'] != 0
            or worker['numerical_attempts'] != 120 or worker['sources'] != cap['sources']
            or not worker['checks'] or not all(c.get('passed') is True for c in worker['checks'])
            or not report['complete'] or report['audit_failures'] or report['shutdown_record_gaps']
            or len(report['rows']) != 15 or not all(r['valid'] and r['workflow_complete'] for r in report['rows'])
            or manifest['sources'] != cap['sources']
            or manifest['inventory'] != parent['preflight']['runtime_inventory']):
        raise ValueError('Complete exact-source actual-worker preflight required')
    if (recipe.get('passed') is not True or recipe['sessions'] != 15 or recipe['annotations'] != 20
            or recipe['engy_calls'] != 0 or recipe['recipe_audit_sha256'] != host.sha(Path(__file__).with_name('audit_planning_107.py'))
            or host.sha(paths['recipe_audit']/'sessions.json') != recipe['sessions_sha256']
            or faults.get('passed') is not True or len(faults['counterexamples']) != 8
            or not all(c['rejected'] and c['audit_read_only'] for c in faults['counterexamples'])
            or faults['auditor_sha256'] != recipe['recipe_audit_sha256']):
        raise ValueError('Independent actual recipe audit and rejection probes required')
    raw = host.read(paths['task_source'])
    jobs = {s: [{k: j[k] for k in host.FIELDS} for j in rows] for s, rows in sorted(raw.items())}
    host.validate_jobs(jobs)
    identities = {(s, j['round'], j['origin']) for s, rows in jobs.items() for j in rows}
    if identities != {(c['series_id'], c['round'], c['origin']) for c in parent['cases']}:
        raise ValueError('Development cohort changed')
    protocol = Path(__file__).with_name('PLANNING_SEQUENCE_107_DEVELOPMENT.md')
    proofs = {name: {'path': str(path), 'sha256': host.sha(path)} for name, path in {
        'worker': paths['worker_root']/'passed.json', 'worker_report': paths['worker_root']/'report.json',
        'worker_manifest': paths['worker_root']/'manifest.json', 'recipe_audit': paths['recipe_audit']/'report.json',
        'fault_probe': paths['fault_probe']/'report.json', 'protocol': protocol}.items()}
    value = {'status': 'frozen_development_plan_requires_host_preflight_and_dispatch_admission',
        'candidate': 107, 'synthetic': False, 'parent_plan_sha256': PARENT_PLAN_SHA,
        'task_source_sha256': TASK_SHA,
        'jobs_sha256': hashlib.sha256(json.dumps(jobs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
        'capsule': cap, 'capsule_sha256': host.sha(paths['capsule']/'capsule.json'),
        'host_sources': host.source_identity(), 'input_builder_sha256': host.sha(__file__),
        'runtime_inventory': manifest['inventory'], 'build': manifest['build'],
        'requested_seed': 7, 'agent': parent['agent'], 'budgets': parent['budgets'], 'arms': list(host.ARMS),
        'planned': {'pilot_sessions': 72, 'continuation_sessions': 240, 'total_sessions': 312},
        'cases': parent['cases'], 'pilot_prefix_origins_per_series': 6,
        'pilot_gate': {'valid_per_arm': 24, 'complete_workflows_per_arm_at_least': 22,
                       'audit_failures': 0, 'accuracy_threshold': None, 'no_selective_reruns': True},
        'proofs': proofs, 'final_target': parent['final_target'], 'final_gate_opened': False,
        'engy_calls': 0, 'dispatch_authorized_by_this_file': False,
        'required_before_paid_dispatch': ['Full exact-source 72+240 synthetic host preflight and terminal archives',
            'Terminal candidate-100 original-host evidence and any explicit audit reconciliation',
            'Separate tested paid admission controller and exclusive one-shot reservation',
            'Fresh output and matched arm state; credentials read only after all checks']}
    destination.mkdir(parents=True); host.dump(destination/'plan.json', value)
    return {'status': value['status'], 'plan_sha256': host.sha(destination/'plan.json'),
            'planned': value['planned'], 'engy_calls': 0, 'final_gate_opened': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('parent-plan', 'task-source', 'capsule', 'worker-root', 'recipe-audit', 'fault-probe', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))
