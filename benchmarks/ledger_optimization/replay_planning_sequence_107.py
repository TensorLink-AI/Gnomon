"""Replay explicit prerequisite plans against the authenticated 106 contexts."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from .planning_sequence_107 import planning_sequence


def replay(destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    hashes, counts = {}, Counter()

    def read(path, expected=None):
        path = Path(path)
        raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
        if expected is not None and sha != expected: raise ValueError('Source hash mismatch: '+str(path))
        hashes[str(path)] = sha
        return raw

    receipt_path = Path('benchmarks/ledger_optimization/evidence/planning-recipes-106-offline-001.json')
    receipt = json.loads(read(receipt_path))
    parent = Path('results/planning-recipes-106-offline-001')
    report = json.loads(read(parent/'report.json', receipt['evidence_sha256'][str(parent/'report.json')]))
    rows = [json.loads(line) for line in read(parent/'reviews.jsonl', receipt['evidence_sha256'][str(parent/'reviews.jsonl')]).splitlines()]
    assert report['counts']['audited_sessions'] == 186 and len(rows) == 63
    package = Path('results/contrast-capsule-100-offline-003/capsule/benchmarks/hermes_ml_checkpoint_v6')
    capsule_path = package.parent.parent/'capsule.json'
    capsule = json.loads(read(capsule_path, report['input_sha256'][str(capsule_path)]))
    numerical = read(package/'numerical.py', capsule['sources']['numerical.py']).decode()
    lab = read(package/'lab.py', capsule['sources']['lab.py']).decode()
    # Authenticate the exact common start implementation used to supply the bound.
    start = next(n for n in ast.parse(lab).body if isinstance(n, ast.FunctionDef) and n.name == 'start')
    start_text = ast.unparse(start)
    assert "configuration({'model': 'seasonal', 'season': 7})" in start_text
    assert 'admit(1 if config_id(config) in tested() else 4)' in start_text
    assert 'backtest(config, initial_baseline=True)' in start_text
    assert "return commit(config, reason='initial_baseline_checkpoint')" in start_text
    config_fn = next(n for n in ast.parse(numerical).body if isinstance(n, ast.FunctionDef) and n.name == 'configuration')
    ns = {'math': math}
    exec(compile(ast.Module(body=[config_fn], type_ignores=[]), str(package/'numerical.py'), 'exec'), ns)
    canonicalize = ns['configuration']; initial_config = canonicalize({'model': 'seasonal', 'season': 7})
    initial_id = hashlib.sha256(json.dumps(initial_config, sort_keys=True).encode()).hexdigest()[:16]
    checks = 0
    for item in rows:
        counts['review_calls'] += 1
        saved = {k: item[k] for k in ('series_id', 'round', 'boundary_line', 'status')}
        if item['status'] == 'no_historical_catalog':
            counts['cold_reviews'] += 1; saved['sequence'] = None
        else:
            current = deepcopy(item['current']); query = current['query']
            suffix = '/ledger/'+item['series_id']+'/round-'+str(item['round'])+'/project/'
            event_suffix = '/ledger/'+item['series_id']+'/round-'+str(item['round'])+'/boundary-events.jsonl'
            event_paths = [p for p in report['input_sha256'] if p.endswith(event_suffix)]
            assert len(event_paths) == 1
            events = [json.loads(x) for x in read(event_paths[0], report['input_sha256'][event_paths[0]]).splitlines()]
            event = events[item['boundary_line']-1]
            assert event['stage'] == 'returned' and event['tool'] == 'lab'
            payload = json.loads(event['result']['result']['stdout'])
            assert payload['operation'] == 'review' and payload['budget'] == current['budget']
            review = payload['result']; assert review['query'] == query
            full_paths = [p for p in report['input_sha256'] if p.endswith(suffix+review['full_evidence']['path'])]
            assert len(full_paths) == 1
            full_bytes = read(full_paths[0], report['input_sha256'][full_paths[0]])
            full = json.loads(full_bytes)
            if not current['checkpoint_available']:
                current['initial_checkpoint'] = {'config': initial_config, 'max_numerical_attempts': 4}
            before = deepcopy((review, current))
            result = planning_sequence(review, full_bytes, current, canonicalize)
            assert before == (review, current) and result['observed_state'] == current
            assert result['numerical_attempts'] == 0 and result['provider_calls'] == 0
            assert not result['state_projected_as_executed'] and not result['forecast_selection_made']
            support = {cid: set() for cid in review['configuration_index']}
            for card in full['cards']:
                origins = {datetime.fromisoformat(o['origin']) for o in card['windows']['lifetime']['origins']}
                assert all(t < datetime.fromisoformat(query['origin']) for t in origins)
                for cid in (card['left_config_id'], card['right_config_id']): support[cid].update(origins)
            excluded = {c['config_id'] for c in current['executed_configurations']}
            if not current['checkpoint_available']: excluded.add(initial_id)
            eligible = sorted((cid for cid in support if cid not in excluded and len(support[cid]) >= 4),
                              key=lambda cid: (-len(support[cid]), cid))
            assert [r['config_id'] for r in result['recipes']] == eligible[:3]
            assert result['eligible_recipes_on_page'] == len(eligible)
            budget = current['budget']
            if current['checkpoint_available']:
                assert result['recipes'] == item['planning']['recipes']
                capacity = item['planning']['sequential_capacity_at_snapshot']
                assert result['prerequisite'] is None
            else:
                capacity = (max(0, min((budget['numerical_remaining']-5)//4,
                                      budget['agent_requests_remaining']-2,
                                      budget['exploration_requests_remaining']-1))
                            if budget['phase'] == 'exploration' else 0)
                assert result['prerequisite']['arguments'] == {'operation': 'start'}
                assert result['prerequisite']['max_numerical_attempts'] == 4
                assert all(not r['next_call']['admissible_now'] for r in result['recipes'])
                assert result['initial_checkpoint_config_id'] == initial_id
                counts['reviews_without_checkpoint'] += 1
            capacity = min(capacity, len(eligible), 3)
            assert result['sequential_capacity_upper_bound'] == capacity
            assert bool(result['next_call']) == (capacity > 0)
            for recipe in result['recipes']:
                cid = recipe['config_id']; call = recipe['next_call']
                assert call['arguments'] == {'operation': 'backtest', 'config': review['configuration_index'][cid]['config']}
                assert call['valid_for_task'] == query
                assert recipe['historical_origins_with_paired_evidence'] == len(support[cid])
                for pointer in recipe['evidence_pointers']:
                    card = full['cards'][int(pointer.split('/')[2])]
                    assert cid in (card['left_config_id'], card['right_config_id'])
                checks += 4
            counts['reviews_with_budget_feasible_first_action'] += capacity > 0
            counts['eligible_recipes'] += len(eligible)
            counts['displayed_recipes'] += len(result['recipes'])
            counts['conditional_sequential_capacity'] += capacity
            saved.update(sequence=result, serialized_bytes=len(json.dumps(result, separators=(',', ':')).encode()))
            checks += 10
        with (destination/'reviews.jsonl').open('a') as stream:
            stream.write(json.dumps(saved, separators=(',', ':'))+'\n')
    for name in ('PLANNING_SEQUENCE_107.md', 'planning_sequence_107.py', 'planning_recipes_106.py',
                 'paired_consistency_098.py', Path(__file__).name): read(Path(__file__).parent/name)
    for path, sha in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
        checks += 1
    result = {'status': 'structural_sequence_replay_passed', 'counts': dict(counts), 'checks': checks,
              'input_sha256': hashes, 'originals_unchanged': True, 'engy_calls': 0, 'refits': 0,
              'synthetic_lab_execution_completed': False, 'agent_efficacy_test_completed': False,
              'live_sources_changed': False, 'final_gate_opened': False, 'objective_established': False,
              'scope': 'Conditional budget feasibility at actual review boundaries, not executed steps or an accuracy result.'}
    (destination/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    return {k: v for k, v in result.items() if k != 'input_sha256'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    print(json.dumps(replay(parser.parse_args().output), indent=2))
