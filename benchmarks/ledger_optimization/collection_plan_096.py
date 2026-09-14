"""Freeze a development-only collection plan; no execution or credential access."""
import hashlib
import json
from pathlib import Path

from .seed_capsule_093 import BASE_INVENTORY_SHA256

SOURCE_SHA = 'cf7bdd21e216e84809edb8653710e0c0755402864201400c5749a8dbff00f561'
ARMS = ('plain', 'gnomon', 'ledger')
CHANGED = {'lab.py', 'TASK.md', 'boundary_schemas_093.py', 'PROTOCOL.md', 'analyze.py'}


def freeze(capsule, task_source, output):
    capsule, task_source, output = map(Path, (capsule, task_source, output))
    if output.exists() or output.is_symlink():
        raise ValueError('A fresh plan path is required')
    raw = task_source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('Only the frozen reused development cohort is admitted')
    jobs = json.loads(raw)
    if len(jobs) != 4 or any(len(v) != 26 for v in jobs.values()):
        raise ValueError('Expected four development series and 26 origins each')
    manifest = json.loads((capsule/'capsule.json').read_text())
    if set(manifest['changed_files']) != CHANGED or manifest['common_to_all_arms'] is not True:
        raise ValueError('Only the declared common collection amendment is admitted')
    base = manifest['base_sources']
    base_hash = hashlib.sha256(json.dumps(base, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    actual_changes = {n for n in base if base[n] != manifest['sources'].get(n)}
    if (base_hash != BASE_INVENTORY_SHA256 or set(base) != set(manifest['sources'])
            or actual_changes != CHANGED):
        raise ValueError('Unrecognized frozen base or undeclared changes')
    for name, digest in manifest['sources'].items():
        if Path(name).name != name or hashlib.sha256((capsule/'benchmarks/hermes_ml_checkpoint_v6'/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Capsule source mismatch')
    cases = []
    for series, values in sorted(jobs.items()):
        for number, job in enumerate(values):
            if job['series_id'] != series or job['round'] != number:
                raise ValueError('Unexpected series/origin order')
            cases.append({'series_id': series, 'round': number, 'origin': job['origin'],
                          'stage': 'pilot' if number < 3 else 'continuation'})
    plan = {
        'status': 'frozen_development_plan_not_dispatch_authorization',
        'candidate': '096_common_prospective_forecast_collection',
        'task_source_sha256': SOURCE_SHA, 'capsule': manifest,
        'capsule_sha256': hashlib.sha256((capsule/'capsule.json').read_bytes()).hexdigest(),
        'arms': list(ARMS), 'cases': cases, 'seed': 7,
        'agent': {'model': 'deepseek-v4.1-flash', 'temperature': .2, 'max_tokens': 3072},
        'gnomon': {'version': '1.2.0', 'build_source_sha256': '9723394ccb6d9e11991b312e01bac47c767c69407b6b33d36971cb6e48b6a22e'},
        'planned': {'pilot_sessions': 36, 'continuation_sessions': 276, 'total_sessions': 312},
        'budgets': {'numerical_attempts': 60, 'model_requests': 16, 'corrections': 2,
                    'exploration_requests': 12, 'selection_requests': 4, 'selection_seconds': 90,
                    'agent_seconds': 480, 'host_seconds': 520, 'parallel_series': 2,
                    'fresh_collection_fits': 4, 'final_fit_reserve': 1},
        'pilot_gate': {'valid_per_arm': 12, 'full_workflows_at_least_per_arm': 11,
                       'audit_failures': 0, 'accuracy_threshold': None,
                       'retain_pilot_once_in_continuation': True},
        'comparison': {'primary': 'ledger versus gnomon within this new run',
                       'secondary': 'ledger versus plain within this new run',
                       'metric': 'arithmetic mean per-case RMSLE',
                       'include_failures': True, 'old_093_controls_allowed': False,
                       'no_shared_memory_between_arms': True, 'reuse_old_093_state': False,
                       'advance_after_all_arms_at_origin': True},
        'launch_prerequisites': ['093 paid controller terminal and evidence preserved',
                                'queued seed integration complete and independently verified',
                                'exact-capsule Hermes worker/transport integration passes',
                                'runtime and capsule hashes reverified',
                                'fresh independent arm homes and run directory'],
        'efficacy_scope': 'Reused development cohort; not final evidence. No accuracy-dependent continuation or partial-run promotion.',
        'final_target': {'relative_rmsle_reduction': .2, 'paired_95_interval_excludes_zero': True,
                         'untouched_final_required': True},
        'final_gate_opened': False, 'engy_calls': 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        f.write(json.dumps(plan, indent=2)+'\n')
    return plan
