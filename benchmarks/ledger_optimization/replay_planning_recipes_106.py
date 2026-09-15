"""Offline structural replay at every review in the frozen 186-session prefix."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from .planning_recipes_106 import planning_recipes


def replay(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    hashes, counts, results, seen = {}, Counter(), [], set()

    def read(path, expected=None):
        path = Path(path)
        raw = path.read_bytes(); value = hashlib.sha256(raw).hexdigest()
        if expected is not None and value != expected:
            raise ValueError('Retained input hash mismatch: '+str(path))
        hashes[str(path)] = value
        return raw

    receipt_path = Path('benchmarks/ledger_optimization/evidence/contrast-100-development-audit-004.json')
    receipt = json.loads(read(receipt_path))
    assert receipt['combined_verified_sessions'] == 186
    numerical = Path('results/contrast-capsule-100-offline-003/capsule/benchmarks/hermes_ml_checkpoint_v6/numerical.py')
    capsule = json.loads(read(numerical.parent.parent.parent/'capsule.json'))
    source = read(numerical, capsule['sources']['numerical.py']).decode()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'configuration')
    namespace = {'math': math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(numerical), 'exec'), namespace)
    canonicalize = namespace['configuration']
    checks = 0
    for report_name, expected_report in receipt['input_reports'].items():
        report_path = Path(report_name)
        report = json.loads(read(report_path, expected_report))
        assert not report['audit_failures'] and not report['shutdown_record_gaps']
        batch = report_path.parent.parent
        if batch.name == 'contrast-100-final-001':
            base = batch/'original/pilot'
            inventory = json.loads(read(batch/'original/SHA256SUMS.json'))
            inventory = {k[6:]: v for k, v in inventory.items() if k.startswith('pilot/')}
        else:
            base = batch/'snapshot'
            inventory = json.loads(read(batch/'snapshot.json'))['files']

        def retained(path):
            path = Path(path)
            if path.is_symlink() or any(p.is_symlink() for p in path.parents):
                raise ValueError('Evidence links are not allowed')
            relative = path.relative_to(base).as_posix()
            return read(path, inventory[relative])

        for row in report['rows']:
            key = (row['arm'], row['series_id'], row['round'])
            assert key not in seen
            seen.add(key); counts['audited_sessions'] += 1
            if row['arm'] != 'ledger': continue
            counts['ledger_sessions'] += 1
            folder = base/'ledger'/row['series_id']/f"round-{row['round']}"
            project = folder/'project'
            task = json.loads(retained(project/'task.json'))
            events = [json.loads(line) for line in retained(folder/'boundary-events.jsonl').splitlines()]
            log = [json.loads(line) for line in retained(project/'experiments.jsonl').splitlines()]
            executions = {e['execution']['execution_id']: e for e in log
                          if e['event'] == 'result' and e['task_origin'] == task['origin']}
            executed, checkpoint, page_count, backtests = {}, False, 0, 0
            for line, event in enumerate(events, 1):
                if event.get('stage') != 'returned' or event.get('tool') != 'lab': continue
                envelope = event['result']
                if envelope.get('status') != 'ok':
                    counts['rejected_tool_calls'] += 1; continue
                payload = json.loads(envelope['result']['stdout'])
                if payload.get('status') != 'ok':
                    counts['rejected_lab_calls'] += 1; continue
                op, value = payload['operation'], payload['result']
                if op in ('start', 'backtest'):
                    config = value['config']; assert canonicalize(config) == config
                    cid = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
                    if op == 'start':
                        record = executions[value['execution_id']]
                        assert record['kind'] == 'forecast'
                        checkpoint = True
                    else:
                        assert len(value['folds']) == 3
                        for fold in value['folds']:
                            record = executions[fold['execution_id']]
                            assert (record['kind'] == 'backtest' and record['config_id'] == cid
                                    and record['request']['cutoff'] == fold['origin']
                                    and record['metrics'] == fold['metrics'])
                        backtests += 1
                    assert record['config_id'] == cid and record['config'] == config
                    executed[cid] = {'config_id': cid, 'config': config,
                                     **{k: record['execution'][k] for k in ('provider', 'revision')}}
                elif op == 'commit':
                    assert value['execution_id'] in executions
                    checkpoint = True
                elif op == 'review':
                    page_count += 1; counts['review_calls'] += 1
                    counts['reviews_before_current_backtest'] += backtests == 0
                    counts['reviews_without_checkpoint'] += not checkpoint
                    item = {'series_id': row['series_id'], 'round': row['round'], 'boundary_line': line,
                            'current_backtests': backtests, 'checkpoint_available': checkpoint}
                    if 'schema_version' not in value:
                        assert value == {'status': 'insufficient_evidence', 'cards': [],
                                         'record_count': 0, 'ledger_queries': 0, 'provider_calls': 0}
                        counts['cold_reviews_without_catalog'] += 1
                        item.update(status='no_historical_catalog', admissible_recipes=0)
                    else:
                        query = {k: task[k] for k in ('series_id', 'unit', 'horizon', 'origin')}
                        assert value['query'] == query
                        current = {'query': query, 'executed_configurations': deepcopy(list(executed.values())),
                                   'checkpoint_available': checkpoint, 'budget': deepcopy(payload['budget'])}
                        assert current['budget']['fresh_backtest_fits'] == 4
                        assert current['budget']['reserved_final_fits'] == 1
                        ref = project/value['full_evidence']['path']
                        assert ref.resolve().is_relative_to(project.resolve())
                        full_bytes = retained(ref); full = json.loads(full_bytes)
                        original = deepcopy((value, current))
                        rendered = planning_recipes(value, full_bytes, current, canonicalize)
                        assert (value, current) == original
                        # Independent union/count/order calculation from the saved page.
                        support = {}
                        for card in full['cards']:
                            for name in ('left_config_id', 'right_config_id'):
                                support.setdefault(card[name], set()).update(
                                    datetime.fromisoformat(o['origin']) for o in card['windows']['lifetime']['origins'])
                        eligible = sorted((cid for cid, dates in support.items() if len(dates) >= 4 and cid not in executed),
                                          key=lambda cid: (-len(support[cid]), cid))
                        assert rendered['eligible_recipes_on_page'] == len(eligible)
                        assert [r['config_id'] for r in rendered['recipes']] == eligible[:3]
                        budget = current['budget']
                        capacity = (max(0, (budget['numerical_remaining']-1)//4)
                                    if checkpoint and budget['phase'] == 'exploration' else 0)
                        assert rendered['sequential_capacity_at_snapshot'] == min(capacity, len(eligible), 3)
                        for recipe in rendered['recipes']:
                            cid = recipe['config_id']; call = recipe['next_call']
                            assert call['arguments'] == {'operation': 'backtest', 'config': value['configuration_index'][cid]['config']}
                            assert call['valid_for_task'] == query and call['admissible_now'] == (capacity > 0)
                            assert recipe['historical_origins_with_paired_evidence'] == len(support[cid])
                            for pointer in recipe['evidence_pointers']:
                                _, _, index, _, label = pointer.split('/')
                                card = full['cards'][int(index)]
                                assert cid in (card['left_config_id'], card['right_config_id']) and label == 'lifetime'
                            checks += 4
                        counts['reviews_with_eligible_recipes'] += bool(eligible)
                        counts['eligible_recipes'] += len(eligible)
                        counts['displayed_recipes'] += len(rendered['recipes'])
                        counts['reviews_with_admissible_recipe'] += rendered['sequential_capacity_at_snapshot'] > 0
                        counts['admissible_sequential_recipes'] += rendered['sequential_capacity_at_snapshot']
                        counts['partial_pages'] += not value['pagination']['all_pairs_included']
                        item.update(status='rendered', current=current, planning=rendered,
                                    serialized_bytes=len(json.dumps(rendered, separators=(',', ':')).encode()))
                        checks += 5
                    results.append(item)
                    with (destination/'reviews.jsonl').open('a') as stream:
                        stream.write(json.dumps(item, separators=(',', ':'))+'\n')
            counts['ledger_sessions_without_review'] += page_count == 0
    assert len(seen) == 186
    for name in ('PLANNING_RECIPES_106.md', 'planning_recipes_106.py', 'paired_consistency_098.py', Path(__file__).name):
        read(Path(__file__).parent/name)
    for path, expected in list(hashes.items()):
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        checks += 1
    receipt = {'status': 'structural_replay_passed', 'counts': dict(counts), 'checks': checks,
               'input_sha256': hashes, 'originals_unchanged': True,
               'engy_calls': 0, 'refits': 0, 'live_sources_changed': False, 'final_gate_opened': False,
               'objective_established': False, 'forecast_selection_made': False,
               'scope': 'All actual ledger review calls in the fixed 186-session prefix; feasibility only, no accuracy claim.'}
    (destination/'report.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return {k: v for k, v in receipt.items() if k != 'input_sha256'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    print(json.dumps(replay(parser.parse_args().output), indent=2))
