"""Freeze the common progress amendment without changing the comparison or gate."""
import hashlib
import json
from pathlib import Path

PARENT_PLAN_SHA = '7422b334e3fe6b44c3d0a89f86aa7443a58b98ec732a523a575e8472fea31e0c'
PARENT_CAPSULE_SHA = '1935bfc673c4cc71b314d18c86ee732254fb68e07aadb1e91e55b377f39eaaed'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(parent_plan, capsule, task_source, output):
    parent_plan, capsule, task_source, output = map(Path, (parent_plan, capsule, task_source, output))
    if output.exists() or output.is_symlink():
        raise ValueError('Fresh plan required')
    if sha(parent_plan) != PARENT_PLAN_SHA:
        raise ValueError('Exact frozen 096 plan required')
    parent = json.loads(parent_plan.read_text())
    manifest = json.loads((capsule/'capsule.json').read_text())
    if (manifest['parent_capsule_sha256'] != PARENT_CAPSULE_SHA
            or manifest['parent_sources'] != parent['capsule']['sources']
            or manifest['common_to_all_arms'] is not True
            or manifest['final_gate_opened'] is not False
            or sha(task_source) != parent['task_source_sha256']):
        raise ValueError('Candidate or task parent identity mismatch')
    sources = {p.name: sha(p) for p in (capsule/'benchmarks/hermes_ml_checkpoint_v6').iterdir() if p.is_file()}
    changed = sorted(n for n in sources if sources[n] != manifest['parent_sources'].get(n))
    if (sources != manifest['sources'] or set(sources) != set(manifest['parent_sources'])
            or changed != ['PROTOCOL.md', 'TASK.md', 'analyze.py', 'transport.py']
            or manifest['changed_from_096'] != changed):
        raise ValueError('Undeclared worker changes')
    plan = {**parent, 'candidate': '097_common_workflow_progress',
        'capsule': manifest, 'capsule_sha256': sha(capsule/'capsule.json'),
        'parent_plan_sha256': PARENT_PLAN_SHA,
        'amendment': {
            'reason': '096 procedural failures used exploration requests without comparing an ML model.',
            'common_to_all_arms': True, 'reminder_requests': [4, 8, 11],
            'omit_when_seconds_remaining_at_most': 90, 'extra_calls_or_fits': 0,
            'uses_only_agent_visible_state': True, 'model_selection_by_agent': True,
            'current_state_counts_not_accuracy_advice': True,
            '096_pilot_excluded_from_097_scores': True,
            'no_accuracy_dependent_gate': True,
        },
        'launch_prerequisites': ['096 pilot and controller terminal; all evidence and costs preserved',
            'independent reminder audit and exact-capsule Hermes integration pass',
            'current runtime and capsule hashes verified', 'fresh independent arm homes and run directory'],
        'efficacy_scope': 'Reused development cohort with a common workflow amendment. Compare only fresh matched 097 arms. Not final evidence; no outcome-dependent continuation.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(plan, indent=2)+'\n')
    return plan
