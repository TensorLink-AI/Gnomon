from copy import deepcopy
import unittest

from benchmarks.ledger_optimization.historical_selection_099 import selections


class HistoricalSelectionTests(unittest.TestCase):
    def test_common_cohort_and_immutability(self):
        cv = {'a': 1., 'b': 2.}
        history = {'2026-01-01T00:00:00Z': {'a': 3., 'b': 1.},
                   '2026-01-02T00:00:00Z': {'a': 4., 'b': 2.},
                   '2026-01-03T00:00:00Z': {'a': 0.}}
        before = deepcopy((cv, history))
        result = selections(cv, history, origin='2026-02-01T00:00:00Z')
        self.assertEqual((cv, history), before)
        self.assertEqual(result['current_cv']['config_id'], 'a')
        self.assertEqual(result['lifetime']['config_id'], 'b')
        self.assertEqual(result['lifetime']['scores'], {'a': 3.5, 'b': 1.5})

    def test_missing_candidate_prevents_unmatched_ranking(self):
        result = selections({'a': 1., 'b': 2., 'new': .5},
                            {'2026-01-01T00:00:00Z': {'a': 0., 'b': 1.},
                             '2026-01-02T00:00:00Z': {'a': 0., 'b': 1.}},
                            origin='2026-02-01T00:00:00Z')
        for name in ('last_4', 'last_12', 'lifetime'):
            self.assertEqual(result[name]['config_id'], 'new')
            self.assertEqual(result[name]['basis'], 'current_cv_fallback')

    def test_recent_window_uses_global_origins_not_last_shared_origins(self):
        history = {f'2026-01-0{i}T00:00:00Z': {'a': 2., 'b': 0.} for i in (1, 2)}
        history.update({f'2026-01-0{i}T00:00:00Z': {'other': 1.} for i in (3, 4, 5, 6)})
        result = selections({'a': 0., 'b': 1.}, history, origin='2026-02-01T00:00:00Z')
        self.assertEqual(result['last_4']['config_id'], 'a')
        self.assertEqual(result['last_4']['matched_origins'], [])
        self.assertEqual(result['lifetime']['config_id'], 'b')

    def test_ties_are_deterministic_and_no_history_is_explicit_fallback(self):
        result = selections({'b': 0., 'a': 0.}, {}, origin='2026-02-01T00:00:00Z')
        self.assertEqual(result['current_cv']['config_id'], 'a')
        self.assertEqual(result['last_4']['basis'], 'current_cv_fallback')
        self.assertEqual(result['last_4']['scores'], {})

    def test_future_duplicate_naive_and_invalid_scores_rejected(self):
        for history in (
            {'2026-02-01T00:00:00Z': {'a': 0.}},
            {'2027-01-01T00:00:00Z': {'a': 0.}},
            {'2026-01-01': {'a': 0.}},
            {'2026-01-01T00:00:00Z': {'a': 0.}, '2026-01-01T01:00:00+01:00': {'a': 1.}},
            {'2026-01-01T00:00:00Z': {'a': float('nan')}},
        ):
            with self.assertRaises(ValueError):
                selections({'a': 0.}, history, origin='2026-02-01T00:00:00Z')
        for value in (float('inf'), -1., True):
            with self.assertRaises(ValueError):
                selections({'a': value}, {}, origin='2026-02-01T00:00:00Z')


if __name__ == '__main__':
    unittest.main()
