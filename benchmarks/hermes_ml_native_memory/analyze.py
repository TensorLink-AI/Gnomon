"""Audit the single follow-up arm and compare only identical original task keys."""
import base64
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from benchmarks.hermes_ml_checkpoint_v4.analyze import analyze as audit
from benchmarks.hermes_ml_checkpoint_v4.transport import dump, sha


def native_calls(folder):
    counts = Counter(); seen = set()
    for path in sorted(folder.glob('hermes-attempt-*.json')):
        for message in json.loads(path.read_text()).get('messages', []):
            for call in message.get('tool_calls') or []:
                function = call.get('function', {})
                identity = json.dumps(call, sort_keys=True)
                if identity in seen:
                    continue
                seen.add(identity)
                counts[function.get('name')] += 1
    return counts


def analyze(root, reference):
    root, reference = Path(root), Path(reference)
    report = audit(root)
    assert set(report['arms']) == {'plain'} and report['arms']['plain']['tasks'] == 104
    memory = []
    for path in root.glob('plain/*/round-*/grade.json'):
        folder = path.parent
        before = json.loads((folder/'native-state-before.json').read_text())
        after = json.loads((folder/'native-state-after.json').read_text())
        for state in (before, after):
            for value in state.values():
                import hashlib
                data = base64.b64decode(value['base64'])
                assert len(data) == value['bytes'] and hashlib.sha256(data).hexdigest() == value['sha256']
        grade = json.loads(path.read_text()); calls = native_calls(folder)
        memory.append({'series_id': grade['series_id'], 'round': grade['round'],
             'memory_calls': calls['memory'], 'skill_view_calls': calls['skill_view'],
             'skill_manage_calls': calls['skill_manage'],
             'prior_memory_present': any(n.startswith('memories/') and v['bytes'] for n,v in before.items()),
             'saved_memory_present': any(n.startswith('memories/') and v['bytes'] for n,v in after.items()),
             'task_skill_present': any('retail-series-review' in n and n.endswith('SKILL.md') for n in after),
             'native_state_changed': before != after})
    index = {(r['series_id'], r['round']): r for r in report['rows']}
    previous = json.loads((reference/'evaluation/report.json').read_text())
    assert previous['complete'] and not previous['audit_failures']
    comparisons = {}
    for arm in ('plain', 'gnomon', 'ledger'):
        old = {(r['series_id'], r['round']): r for r in previous['rows'] if r['arm'] == arm}
        assert set(old) == set(index)
        reference_jobs = json.loads((reference/'evaluation/host-jobs.json').read_text())
        current_jobs = json.loads((root/'host-jobs.json').read_text())
        assert current_jobs == reference_jobs
        new_score = mean(r['rmsle'] for r in index.values())
        old_score = mean(r['rmsle'] for r in old.values())
        comparisons[arm] = {'matched_tasks': len(index), 'native_memory_rmsle': new_score,
             'existing_arm_rmsle': old_score, 'native_memory_relative_error_reduction': 1-new_score/old_score,
             'native_memory_full_workflows': sum(r['workflow_complete'] for r in index.values()),
             'existing_arm_full_workflows': sum(r['workflow_complete'] for r in old.values()),
             'native_memory_reported_tokens': sum(r['tokens'] for r in index.values()),
             'existing_arm_reported_tokens': sum(r['tokens'] for r in old.values()),
             'native_memory_api_calls': sum(r['api_calls'] for r in index.values()),
             'existing_arm_api_calls': sum(r['api_calls'] for r in old.values())}
    summary = {'memory_tool_sessions': sum(r['memory_calls']>0 for r in memory),
         'skill_read_sessions': sum(r['skill_view_calls']>0 for r in memory),
         'skill_update_sessions': sum(r['skill_manage_calls']>0 for r in memory),
         'sessions_with_saved_memory': sum(r['saved_memory_present'] for r in memory),
         'sessions_with_task_skill': sum(r['task_skill_present'] for r in memory)}
    dump(root/'native-memory-report.json', {'summary': summary, 'sessions': memory, 'comparisons': comparisons,
         'reference_report_sha256': sha(reference/'evaluation/report.json'),
         'limitation': 'Separate explicitly prompted development follow-up. Tool calls are invocation counts, not proof of useful lessons. No contemporaneous randomized comparison or held-out superiority claim.'})
    (root/'MEMORY_RESULTS.md').write_text('# Hermes native-memory follow-up\n\nOnly Hermes was rerun; all original evidence remains unchanged.\n\n'
         + json.dumps({'adoption': summary, 'comparisons': comparisons}, indent=2) + '\n')
    return report
