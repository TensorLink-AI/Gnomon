from copy import deepcopy

import pytest

from benchmarks.ledger_optimization.opportunity_audit import choices_with_mapping, run, validated
from benchmarks.tests.test_cv_context_screen import case


def test_identity_mapping_reproduces_support_and_corruption_loses_real_signal():
    cases = [case(i) for i in range(26)]
    episodes = validated(cases)
    assert choices_with_mapping(episodes, {'sales': {'a': 'a', 'b': 'b'}}) == [c['support_provider'] for c in episodes]
    assert choices_with_mapping(episodes, {'sales': {'a': 'b', 'b': 'a'}}) == ['a'] * 26
    result = run(cases, trials=16)
    summary = result['summaries']['all']
    assert summary['future_aware_case_oracle_mean_rmsle'] == 0
    assert summary['support_relative_gain_vs_cv'] == pytest.approx(22 / 26)
    assert summary['diagnostic_gates']['true_support_beats_median_corrupted_history']
    assert result['target_established'] is False and result['new_confirmation_authorized'] is False


def test_corrupted_history_keeps_visibility_boundaries_and_original_forecasts():
    episodes = validated([case(i) for i in range(26)])
    before = deepcopy(episodes)
    mapping = {'sales': {'a': 'b', 'b': 'a'}}
    expected = choices_with_mapping(episodes, mapping)
    assert episodes == before
    for c in episodes[8:]:
        c['scores'] = {'a': 0, 'b': 10000}
    assert choices_with_mapping(episodes, mapping)[:9] == expected[:9]
    episodes = deepcopy(before)
    for c in episodes[:4]:
        c['outcome_recorded_at'] = episodes[20]['origin']
    assert choices_with_mapping(episodes, {'sales': {'a': 'a', 'b': 'b'}})[4] == 'a'
    episodes = deepcopy(before)
    for c in episodes[:4]:
        c['future_timestamps'] = [episodes[20]['origin']]
    assert choices_with_mapping(episodes, {'sales': {'a': 'a', 'b': 'b'}})[4] == 'a'


def test_input_permutation_and_repeat_are_deterministic_and_do_not_mutate():
    cases = [case(i) for i in range(26)]
    before = deepcopy(cases)
    result = run(cases, trials=8)
    assert result == run(cases[::-1], trials=8)
    assert cases == before
    assert len(result['corruption_trials']) == 8


def test_zero_reference_is_undefined_and_cannot_pass_target():
    cases = [case(i, cv='b') for i in range(26)]
    summary = run(cases, trials=4)['summaries']['all']
    assert summary['case_oracle_relative_gain_vs_cv'] is None
    assert summary['required_oracle_opportunity_fraction_for_target'] is None
    assert summary['diagnostic_gates']['headroom_reaches_target_against_cv_reference'] is False


@pytest.mark.parametrize('kind', ['nonfinite', 'duplicate_origin', 'missing_round', 'range', 'corrupt_score'])
def test_invalid_evidence_rejected(kind):
    cases = [case(i) for i in range(26)]
    if kind == 'nonfinite':
        cases[0]['current_card']['a']['cv_rmsle'] = float('nan')
    elif kind == 'duplicate_origin':
        cases[1]['origin'] = cases[0]['origin']
    elif kind == 'missing_round':
        cases.pop()
    elif kind == 'range':
        cases[0]['future_timestamps'] = [cases[0]['origin']]
    else:
        cases[0]['scores']['a'] = 123
    with pytest.raises(ValueError):
        run(cases, trials=2)


def test_invalid_mapping_cannot_duplicate_or_drop_provider():
    with pytest.raises(ValueError, match='permutation'):
        choices_with_mapping(validated([case(i) for i in range(26)]), {'sales': {'a': 'b', 'b': 'b'}})
