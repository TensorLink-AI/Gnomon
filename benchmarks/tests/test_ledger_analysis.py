from copy import deepcopy
from itertools import product

import pytest

from benchmarks.ledger_optimization.analysis import analyze, validated_cohort


def fixture_report():
    manifest = {'series': ['s1', 's2'], 'rounds': list(range(8)),
                'seeds_requested': [7, 19], 'arms': ['no_ledger', 'ledger_119', 'treatment'],
                'expected_decisions': 96}
    rows = []
    for s, r, seed, arm in product(manifest['series'], manifest['rounds'], manifest['seeds_requested'], manifest['arms']):
        base = (r+1)*(1 if s == 's1' else 2)*(1 if seed == 7 else 3)
        rows.append({'series_id': s, 'round': r, 'seed': seed, 'arm': arm,
                     'rmsle': base*{'no_ledger': 1, 'ledger_119': .9, 'treatment': .7}[arm],
                     'fallback_used': False, 'resolution_status': 'explicit_selection'})
    return {'scope': 'synthetic test', 'manifest': manifest, 'per_case': rows, 'harness_failures': 0}


def test_paired_bootstrap_preserves_exact_relative_gain_and_is_deterministic():
    report = fixture_report()
    result = analyze(report)
    comparison = result['comparisons']['treatment']
    assert comparison['relative_reduction_vs_no_ledger'] == pytest.approx(.3)
    assert comparison['paired_95_percent_intervals']['relative_vs_no_ledger'] == pytest.approx([.3, .3])
    assert comparison['numerical_criteria_met']
    assert result['target_established'] is False
    assert result['cohort']['matched_case_seed_pairs'] == 32
    assert result['cold_start']['decisions'] == result['mature_history']['decisions'] == 48
    report['per_case'].reverse()
    report['manifest']['series'].reverse()
    assert analyze(report) == result


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'extra', 'negative', 'nan', 'count', 'harness'])
def test_invalid_cohort_cannot_be_silently_averaged(corruption):
    report = fixture_report()
    if corruption == 'missing':
        report['per_case'].pop()
    elif corruption == 'duplicate':
        report['per_case'].append(deepcopy(report['per_case'][0]))
    elif corruption == 'extra':
        report['per_case'].append({**report['per_case'][0], 'series_id': 'unexpected'})
    elif corruption in ('negative', 'nan'):
        report['per_case'][0]['rmsle'] = -1 if corruption == 'negative' else float('nan')
    elif corruption == 'count':
        report['manifest']['expected_decisions'] -= 1
    else:
        report['harness_failures'] = 1
    with pytest.raises(ValueError):
        validated_cohort(report)


def test_sparse_development_origins_cannot_masquerade_as_consecutive_time_blocks():
    report = fixture_report()
    report['manifest']['rounds'] = [0, 8, 17, 25]
    with pytest.raises(ValueError, match='consecutive'):
        validated_cohort(report)


def test_zero_baseline_reports_undefined_relative_gain_without_dropping_draws():
    report = fixture_report()
    for row in report['per_case']:
        row['rmsle'] = 0
    result = analyze(report)
    comparison = result['comparisons']['treatment']
    assert comparison['relative_reduction_vs_no_ledger'] is None
    assert comparison['paired_95_percent_intervals']['relative_vs_no_ledger'] is None
    assert comparison['paired_95_percent_intervals']['absolute_vs_no_ledger'] == [0, 0]
    assert comparison['numerical_criteria_met'] is False


def test_fallbacks_stay_in_the_primary_denominator():
    report = fixture_report()
    treatment = next(row for row in report['per_case'] if row['arm'] == 'treatment')
    treatment.update(fallback_used=True, resolution_status='no_successful_execution', rmsle=1000)
    result = analyze(report)
    assert result['completion']['treatment']['fallbacks'] == 1
    assert result['comparisons']['treatment']['relative_reduction_vs_no_ledger'] < 0
    assert result['cohort']['decisions'] == 96
