from copy import deepcopy

import pytest

from benchmarks.ledger_optimization.override_trust_screen import gate, prepare, run
from benchmarks.tests.test_cv_context_screen import case


def test_insufficient_past_decisions_do_not_enable_override():
    episodes = prepare([case(i) for i in range(8)])
    assert episodes[4]['support_provider'] == 'b'
    assert gate(episodes[4], episodes, 2, 0)['provider'] == 'a'
    assert gate(episodes[4], episodes, 2, 0)['reason'] == 'insufficient_past_overrides'
    assert gate(episodes[6], episodes, 2, 0)['provider'] == 'b'
    assert gate(episodes[6], episodes, 2, 0)['matched_origins'] == 2


def test_late_outcomes_and_future_scores_do_not_change_gate():
    episodes = prepare([case(i) for i in range(12)])
    current = episodes[6]
    expected = gate(current, episodes, 2, 0)
    altered = deepcopy(episodes)
    for c in altered[6:]:
        c['scores'] = {'a': 1000000, 'b': 0}
    assert gate(current, altered, 2, 0) == expected
    altered[4]['outcome_recorded_at'] = episodes[10]['origin']
    assert gate(current, altered, 2, 0)['reason'] == 'insufficient_past_overrides'
    altered = deepcopy(episodes)
    altered[4]['future_timestamps'] = [episodes[10]['origin']]
    assert gate(current, altered, 2, 0)['reason'] == 'insufficient_past_overrides'


def test_bad_override_history_is_distinct_from_insufficient_history():
    episodes = prepare([case(i) for i in range(8)])
    for c in episodes[4:6]:
        c['scores'] = {'a': 0, 'b': 2}
    decision = gate(episodes[6], episodes, 2, 0)
    assert decision['provider'] == 'a'
    assert decision['reason'] == 'past_overrides_do_not_support_change'


def test_old_proposal_cannot_use_future_outcomes_and_order_is_irrelevant():
    cases = [case(i) for i in range(26)]
    before = deepcopy(cases)
    episodes = prepare(cases)
    changed = deepcopy(cases)
    for c in changed[8:]:
        c['actual'] = [4]
        c['scores'] = {'a': 0, 'b': episodes[0]['scores']['a']}
    assert [c['support_provider'] for c in prepare(changed)[:8]] == [c['support_provider'] for c in episodes[:8]]
    result = run(cases)
    assert result == run(list(reversed(cases)))
    assert cases == before and result['target_established'] is False
    assert result['selected_validation_relative_gain']['frozen_support'] is None


def test_duplicate_or_corrupt_evidence_rejects():
    cases = [case(i) for i in range(26)]
    with pytest.raises(ValueError, match='Duplicate'):
        prepare(cases + [cases[0]])
    cases[0]['scores']['a'] = 99
    with pytest.raises(ValueError, match='arithmetic'):
        prepare(cases)
