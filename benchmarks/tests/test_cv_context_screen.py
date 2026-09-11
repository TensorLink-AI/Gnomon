from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from benchmarks.ledger_optimization.cv_context_screen import leader, rmsle, run, select, visible_history


def case(round_, *, cv='a'):
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=round_ * 2)
    future = (origin + timedelta(days=1)).isoformat()
    predictions = {'a': [4], 'b': [2]}
    return {'series_id': 'sales', 'round': round_, 'origin': origin.isoformat(),
            'future_timestamps': [future], 'outcome_recorded_at': future,
            'current_card': {p: {'cv_rmsle': 0 if p == cv else 1} for p in predictions},
            'predictions': predictions, 'actual': [2],
            'scores': {p: rmsle(point, [2]) for p, point in predictions.items()}}


def test_condition_uses_original_cv_choice_not_eventual_winner():
    cases = [case(i, cv='a' if i < 4 else 'b') for i in range(6)]
    current = case(7)
    history = visible_history(current, cases, conditioned=True)
    assert len(history) == 4
    assert all(leader(c) == 'a' for c in history)
    assert select(current, history, 4, .5) == 'b'


def test_current_future_other_series_and_late_outcomes_are_excluded():
    current = case(10)
    accepted, late_recording, late_source, other = [case(i) for i in range(4)]
    accepted['outcome_recorded_at'] = current['origin']  # inclusive recording boundary
    accepted['future_timestamps'] = [current['origin']]  # inclusive source boundary
    late_recording['outcome_recorded_at'] = case(11)['origin']
    late_source['future_timestamps'] = [case(11)['origin']]
    other['series_id'] = 'other'
    assert visible_history(current, [accepted, late_recording, late_source, other, current, case(11)],
                           conditioned=True) == [accepted]


def test_future_outcome_changes_do_not_change_current_selection():
    cases = [case(i) for i in range(12)]
    current = cases[5]
    original = select(current, visible_history(current, cases, conditioned=True), 4, .5)
    altered = deepcopy(cases)
    for c in altered[5:]:
        c['actual'] = [1000000]
        c['scores'] = {'a': 0, 'b': 100}
    assert select(current, visible_history(current, altered, conditioned=True), 4, .5) == original


def test_order_invariance_and_original_predictions_are_unchanged():
    cases = [case(i) for i in range(26)]
    before = deepcopy(cases)
    result = run(cases)
    assert run(list(reversed(cases))) == result
    assert cases == before
    assert result['provider_calls'] == 0 and result['target_established'] is False
    assert result['selected_validation_gain_vs_frozen_support'] is None  # zero reference loss
    assert all(r['choices']['leader_context_4_0.5'] == 'a' for r in result['cases'][:4])


def test_duplicate_identity_and_incorrect_cached_scores_reject():
    cases = [case(i) for i in range(26)]
    with pytest.raises(ValueError, match='Duplicate'):
        run(cases + [cases[0]])
    cases[0]['scores']['a'] = 0
    with pytest.raises(ValueError, match='arithmetic'):
        run(cases)
