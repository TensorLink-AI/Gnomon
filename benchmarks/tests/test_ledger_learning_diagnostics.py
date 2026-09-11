from datetime import datetime, timedelta, timezone

import pytest

from benchmarks.ledger_optimization.cv_reliability import run
from benchmarks.ledger_optimization.robustness import describe
from benchmarks.ledger_optimization.support_screen import support_packet


def test_cv_reliability_never_learns_from_unavailable_outcomes():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cases = [{'series_id': 's', 'round': i, 'origin': (start+timedelta(days=i)).isoformat(),
              'future_timestamps': [(start+timedelta(days=i+1)).isoformat()],
              'outcome_recorded_at': '2027-01-01T00:00:00Z', 'predictions': {'a': [1], 'b': [2]},
              'scores': {'a': 100, 'b': 0}, 'current_card': {'a': {'cv_rmsle': 1}, 'b': {'cv_rmsle': 2}}}
             for i in range(26)]
    contexts = {('s', i): {'sparsity': 'low', 'trend': 'stable'} for i in range(26)}
    report = run(cases, contexts)
    assert all(row['visible_origins'] == 0 for row in report['cases'])
    assert all(set(row['choices'].values()) == {'a'} for row in report['cases'])
    cases[0]['outcome_recorded_at'] = cases[1]['origin']
    learned = run(cases, contexts)
    assert learned['cases'][1]['choices']['series_all_additive_4'] == 'b'
    assert not learned['changes_forecasts'] and not learned['target_established']


def test_robustness_retains_large_losses_and_requires_the_whole_cohort():
    manifest = {'series': ['s'], 'rounds': [0, 8], 'seeds_requested': [7], 'arms': ['no_ledger', 'ledger']}
    rows = [{'series_id': 's', 'round': r, 'seed': 7, 'arm': arm, 'provider': 'p',
             'rmsle': loss} for r, arm, loss in [(0, 'no_ledger', 1), (0, 'ledger', 1),
                                                (8, 'no_ledger', 2), (8, 'ledger', 4)]]
    report = {'scope': 'synthetic', 'manifest': manifest, 'per_case': rows, 'harness_failures': 0}
    result = describe(report)
    assert result['tail']['ledger']['maximum_rmsle'] == 4
    assert result['paired_differences']['ledger']['mean'] == 1
    assert result['paired_differences']['ledger']['tied_cases'] == 1
    assert result['cold_start']['ledger']['mean_case_rmsle'] == 1
    assert result['mature_history']['ledger']['mean_case_rmsle'] == 4
    report['per_case'].pop()
    with pytest.raises(ValueError, match='every registered'):
        describe(report)


def test_support_packet_uses_paired_history_and_abstains_with_too_little_evidence():
    case = {'origin': '2026-02-01T00:00:00Z', 'current_card': {'a': {'cv_rmsle': 1}, 'b': {'cv_rmsle': 2}}}
    rows = [{'origin': f'2026-01-0{i+1}T00:00:00Z', 'rmsle': [4, 1]} for i in range(4)]
    packet = {'raw_history': {'providers': ['a', 'b'], 'records': rows,
                             'source_as_of': case['origin'], 'recorded_as_of': case['origin']}}
    supported = support_packet(case, packet)
    assert supported['provider'] == 'b' and supported['paired_wins'] == 4
    assert supported['supports_changing_current_cv_choice'] and not supported['forecast_executed']
    assert support_packet({**case, 'actual': [999999]}, packet) == supported
    packet['raw_history']['records'] = rows[:3]
    assert support_packet(case, packet)['provider'] == 'a'
