"""Read-only predecessor gate for candidate 100; run on the worker host.

No credential access, API calls, process starts or final-data reads. A later
launcher must also verify the candidate plan/runtime and rerun the independent
analysis. This checks terminal identity, immutable receipts and complete cohort.
"""
import json
from pathlib import Path

from .contrast_plan_100 import PARENT_PLAN_SHA, read, sha
from .launch_workflow_097 import require_terminal
from .control_continuation_097 import completion


def verify_predecessor(output, launch, parent_plan, proc=Path('/proc')):
    output, launch, parent_plan = map(Path, (output, launch, parent_plan))
    if sha(parent_plan) != PARENT_PLAN_SHA: raise ValueError('Exact 097 plan required')
    for name in ('launch.json','development-process.json'):
        require_terminal(read(launch/name),proc)
    if (launch/'INCOMPLETE.json').exists(): raise ValueError('Predecessor evidence incomplete')
    finished, audited = read(launch/'FINISHED.json'), read(launch/'AUDITED.json')
    status = read(launch/'development-exit.json')['exit_status']
    recomputed = completion(output,status)
    if not recomputed['complete'] or audited != recomputed or finished.get('complete') is not True:
        raise ValueError('Predecessor terminal audit mismatch')
    if (finished.get('further_execution_launched') is not False
            or finished.get('final_gate_opened') is not False
            or sha(launch/'evidence.tar.gz') != finished['archive_sha256']
            or sha(launch/'SHA256SUMS.json') != finished['inventory_sha256']):
        raise ValueError('Predecessor archive or completion identity changed')
    inventory = read(launch/'SHA256SUMS.json')
    required = {'pilot/'+n for n in ('manifest.json','report.json','complete.json','runner-exit.json','host-jobs.json')}
    required |= {'launch/'+n for n in ('AUDITED.json','costs.json','development-exit.json')}
    if not required <= set(inventory): raise ValueError('Required terminal evidence absent from inventory')
    for label, expected in inventory.items():
        path = Path(label)
        if path.is_absolute() or '..' in path.parts or path.parts[0] not in ('launch','pilot'):
            raise ValueError('Unsafe inventory path')
        root = launch if path.parts[0] == 'launch' else output
        relative = Path(*path.parts[1:])
        if any((root/p).is_symlink() for p in (relative,*relative.parents)):
            raise ValueError('Symlink in retained evidence')
        if sha(root/relative) != expected: raise ValueError('Retained evidence changed: '+label)
    parent = read(parent_plan); manifest = read(output/'manifest.json'); report = read(output/'report.json')
    if manifest['sources'] != parent['capsule']['sources'] or manifest['source_jobs_sha256'] != parent['task_source_sha256']:
        raise ValueError('Wrong predecessor source or task identity')
    expected={(a,c['series_id'],c['round'],c['origin']) for a in parent['arms'] for c in parent['cases']}
    actual={(r['arm'],r['series_id'],r['round'],r['origin']) for r in report['rows']}
    if actual != expected or len(report['rows']) != len(expected): raise ValueError('Wrong predecessor cohort')
    costs=read(launch/'costs.json')
    if (costs['deduplicated_total']['sessions'] != 312 or costs['stages']['retained_pilot']['sessions'] != 36
            or costs['stages']['continuation']['sessions'] != 276):
        raise ValueError('Predecessor cost coverage incomplete')
    return {'predecessor_terminal_verified':True,'files_verified':len(inventory),
            'finished_sha256':sha(launch/'FINISHED.json'),'report_sha256':sha(output/'report.json'),
            'costs_sha256':sha(launch/'costs.json'),'accuracy_used_for_admission':False,
            'engy_calls':0,'final_gate_opened':False,
            'scope':'Terminal process and immutable audit/cohort checks; not candidate dispatch or a fresh numerical re-audit.'}
