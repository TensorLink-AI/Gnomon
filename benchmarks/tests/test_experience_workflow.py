import copy
import sqlite3

import pytest

from benchmarks.experience_workflow.agent import Boundary, checkpoint
from benchmarks.experience_workflow.audit import run
from benchmarks.experience_workflow.report import summarize
from benchmarks.experience_workflow.scenario import generate, oracle, public_answer, matches, expected_provider
from benchmarks.experience_workflow.storage import Store


def test_reserved_confirmation_worlds_cannot_be_generated():
    with pytest.raises(ValueError, match='development'):
        generate(9000)


def test_independent_oracle_sql_and_public_ledger_across_revisions(tmp_path):
    report = run([100, 101, 102, 103], 16, tmp_path / 'audit')
    assert report['passed'], report['failures'][:3]
    assert report['checks'] == 576
    assert all(report['mutation_failures_detected'].values())
    assert not report['objective_achieved']


def test_generator_reproducible_and_future_truth_not_in_tasks():
    a, b = generate(102), generate(102)
    assert a == b
    for task in a['tasks']:
        assert 'truth' not in task and 'events' not in task
        assert task['request']['timestamps'][-1] == task['now']
        assert min(task['request']['future_timestamps']) > task['now']


def test_sql_cannot_mutate_attach_or_read_host_files(tmp_path):
    store = Store(tmp_path / 'sql', 'sqlite')
    try:
        for sql in ('DELETE FROM actuals', "ATTACH DATABASE '/tmp/private.db' AS private", 'PRAGMA database_list'):
            with pytest.raises(sqlite3.DatabaseError):
                store.sql(sql, {})
        assert store.sql('SELECT 1 AS n', {}) == [{'n': 1}]
    finally:
        store.close()


def test_task_binding_wrong_evidence_and_budget_do_not_pass(tmp_path):
    world = generate(102, 8)
    store = Store(tmp_path / 'sql', 'sqlite')
    try:
        task = world['tasks'][-1]
        store.ingest(world['events'], task['now'])
        b = Boundary(store, world, task)
        expected = {label: public_answer(oracle(world['events'], q)) for label, q in task['queries'].items()}
        bad = b.call('submit_decision', dict(execution_id='another-task', evidence=expected))
        assert 'error' in bad and b.result is None
        b = Boundary(store, world, task)
        completion = b.call('forecast', {'provider': expected_provider(expected['current'])})
        wrong = copy.deepcopy(expected)
        wrong['current']['matched_origins'] += 1
        assert 'error' in b.call('submit_decision', dict(execution_id=completion['execution_id'], evidence=wrong))
        assert b.result is None
        assert b.call('submit_decision', dict(execution_id=completion['execution_id'], evidence=expected))['accepted']
        assert b.result['evidence'] == expected
        assert 'error' in b.call('forecast', {'provider': 'last_value'})
    finally:
        store.close()


def test_bad_sql_is_recoverable_not_harness_crash(tmp_path):
    world = generate(100, 4)
    store = Store(tmp_path / 'sql', 'sqlite')
    try:
        boundary = Boundary(store, world, world['tasks'][0])
        assert 'error' in boundary.call('sql_query', {'sql': 'SELECT no_such_field FROM actuals', 'params': {}})
        assert boundary.call('sql_query', {'sql': 'SELECT 1 AS n', 'params': {}}) == [{'n': 1}]
    finally:
        store.close()


def test_missing_api_usage_is_unknown_and_does_not_pass_cost_gate(tmp_path):
    world = generate(100, 4)
    store = Store(tmp_path / 'sql', 'sqlite')
    try:
        store.ingest(world['events'], world['tasks'][0]['now'])
        reply = {'choices': [{'message': {'role': 'assistant', 'content': 'No decision'}}]}
        row = checkpoint(store, world, world['tasks'][0], 7, 'fake', tmp_path, chat=lambda *a: reply)
        assert row['tokens'] is None and not row['usage_complete'] and row['fallback_used']
        assert row['api_calls'] == 8 and not row['completed']
        assert 'fake' not in (tmp_path / '000-sqlite.jsonl').read_text()
        report = summarize([row], dict(scope='development', expected_decisions=2, rounds=4, agent_seeds=[7]), draws=10)
        assert not report['objective_achieved']
        assert report['gates']['cost_point'] == 'not_met_or_not_established'
    finally:
        store.close()


def test_oracle_requires_full_horizon_not_partial_metrics():
    world = generate(100, 8)
    q = world['tasks'][-1]['queries']['current']
    full = oracle(world['events'], q)
    assert full['matched_origins'] > 0
    remove = set(full['origins'][0]['actual_refs'])
    partial = oracle([e for e in world['events'] if e['event_id'] != next(iter(remove))], q)
    assert partial['matched_origins'] == full['matched_origins'] - 1
    assert not matches(dict(matched_origins=0, scores={}, ranking=[]), full)
