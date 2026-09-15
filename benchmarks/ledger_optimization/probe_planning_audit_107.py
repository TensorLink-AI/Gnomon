"""Tamper copies of retained synthetic replies; never mutate original evidence."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from .audit_planning_107 import audit_session, digest, encoded


def inventory(root):
    return {str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}


def rewrite(folder, operation, mutation):
    """Alter artifact, address, annotation and tool view consistently.

    Rehashing all presentation layers ensures the auditor must check facts,
    rather than merely notice an out-of-date integrity digest.
    """
    project = folder/'project'; path = folder/'boundary-events.jsonl'
    events = [json.loads(line) for line in path.read_text().splitlines()]
    annotations_path = project/'planning-annotations.jsonl'
    annotations = [json.loads(line) for line in annotations_path.read_text().splitlines()]
    for event in events:
        if event.get('tool') != 'lab' or event.get('stage') != 'returned': continue
        payload = event.get('result', {}).get('result', {})
        if 'stdout' not in payload: continue
        reply = json.loads(payload['stdout'])
        if reply.get('operation') != operation or 'recipe_plan' not in reply: continue
        old_view = deepcopy(reply['recipe_plan']); view = reply['recipe_plan']
        old_path = project/view['evidence']['path']; artifact = json.loads(old_path.read_bytes())
        mutation(artifact, view)
        raw = encoded(artifact); sha = digest(raw)
        view['evidence'] = {'path': f'planning-evidence/{sha}.json', 'bytes': len(raw), 'sha256': sha}
        (project/view['evidence']['path']).write_bytes(raw)
        matches = [a for a in annotations if a['view'] == old_view]
        assert len(matches) == 1
        matches[0]['view'] = view
        matches[0]['execution_log_prefix'] = artifact['execution_log_prefix']
        payload['stdout'] = json.dumps(reply)
        path.write_bytes(b'\n'.join(encoded(e) for e in events)+b'\n')
        annotations_path.write_bytes(b'\n'.join(encoded(a) for a in annotations)+b'\n')
        return
    raise ValueError('Required operation has no recipe annotation')


def probe(source, output):
    source = Path(source).resolve(); output = Path(output).resolve()
    if source == output or source in output.parents: raise ValueError('Output must be outside source')
    before = inventory(source); output.mkdir(parents=True, exist_ok=False)
    clean = audit_session(source); results = []

    def support(a, v):
        a['plan']['recipes'][0]['historical_origins_with_paired_evidence'] += 1
        v['recipes'][0]['historical_origins_with_paired_evidence'] += 1

    def budget(a, v):
        a['observed_budget']['numerical_remaining'] += 4
        a['plan']['observed_state']['budget']['numerical_remaining'] += 4
        v['shared_budget']['numerical_remaining'] += 4

    def checkpoint(a, v):
        a['checkpoint'] = None
        a['plan']['observed_state']['checkpoint_available'] = False

    def untried(a, v):
        a['plan']['observed_state']['executed_configurations'] = []

    def premature(a, v):
        a['plan']['recipes'][0]['next_call']['admissible_now'] = True
        v['recipes'][0]['admissible_now'] = True

    def wrong_call(a, v):
        call = {'tool': 'lab', 'arguments': {'operation': 'commit', 'config': {'model': 'seasonal', 'season': 7}}}
        a['plan']['next_call'] = call; v['next_call'] = call

    def review(a, v): a['review_state_sha256'] = '0'*64
    def hidden_query(a, v): a['additional_ledger_queries'] = 1; v['additional_ledger_queries'] = 1

    cases = [
        ('inflated_support', 'review', support, 'recipe fields and union support'),
        ('invented_budget', 'review', budget, 'budget exactly as published'),
        ('lost_checkpoint', 'start', checkpoint, 'latest published checkpoint'),
        ('forgot_executed_config', 'start', untried, 'observed context'),
        ('premature_admission', 'review', premature, 'recipe fields and union support'),
        ('automatic_selection', 'review', wrong_call, 'next action retains prerequisites'),
        ('unrequested_review', 'review', review, 'exact most recently requested review'),
        ('extra_ledger_query', 'review', hidden_query, 'zero extra execution'),
    ]
    for name, operation, mutate, reason in cases:
        copy = output/name; shutil.copytree(source, copy)
        rewrite(copy, operation, mutate)
        after_mutation = inventory(copy)
        try: audit_session(copy)
        except ValueError as exc:
            message = str(exc)
            if reason not in message: raise AssertionError((name, message, reason)) from exc
        else: raise AssertionError('Tampered evidence was accepted: '+name)
        assert inventory(copy) == after_mutation, 'Audit mutated counterexample'
        results.append({'case': name, 'rejected': True, 'cause': message,
                        'all_presentation_hashes_updated': True, 'audit_read_only': True})
        (output/'cases.json').write_text(json.dumps(results, indent=2)+'\n')
    assert inventory(source) == before, 'Original evidence changed'
    report = {'passed': True, 'clean_checks': clean['checks'], 'counterexamples': results,
              'source_files_unchanged': len(before), 'source_inventory_sha256': digest(encoded(before)),
              'numerical_calls': 0, 'engy_calls': 0, 'ledger_queries': 0,
              'auditor_sha256': digest(Path(__file__).with_name('audit_planning_107.py').read_bytes()),
              'probe_sha256': digest(Path(__file__).read_bytes())}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'counterexamples'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    probe(args.session, args.output)
