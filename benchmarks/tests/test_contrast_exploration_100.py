from copy import deepcopy
import unittest

from benchmarks.ledger_optimization.contrast_exploration_100 import exploration
from benchmarks.tests.test_contrast_history_progress_100 import rows


def fixture():
    data = rows()
    tables = {}
    for row in data:
        row['config'] = {'model': 'ridge', 'window': 365}
        tables[row['arm'], row['series_id'], row['round']] = {'configurations': [
            {'config': {'model': 'ridge', 'window': 365}, 'mean_rmsle': 2.},
            {'config': {'model': 'seasonal', 'season': 7}, 'mean_rmsle': 1.},
        ]}
    return data, tables


class ExplorationTests(unittest.TestCase):
    def test_search_sets_and_failures_are_reported_without_filtering(self):
        data, tables = fixture()
        tables['ledger', 's', 0]['configurations'].append(
            {'config': {'model': 'random_forest', 'depth': 8}, 'mean_rmsle': .5})
        out = exploration(data, tables)
        self.assertEqual(out['primary_all_matched']['matched_cases'], 2)
        self.assertEqual(out['primary_all_matched']['fallbacks']['ledger'], 2)
        self.assertEqual(out['same_complete_cv_set_cases'], 1)
        self.assertEqual(out['same_selected_config_cases'], 2)
        self.assertEqual(out['arms']['ledger']['mean_complete_cv_configs'], 2.5)
        self.assertEqual(out['arms']['ledger']['family_exposure_sessions']['random_forest'], 1)
        self.assertEqual(out['cases'][0]['arms']['ledger']['selected_cv_rank'], 3)
        self.assertEqual(out, exploration(list(reversed(data)), tables))

    def test_ties_empty_tables_and_unmatched_groups_are_explicit(self):
        data, tables = fixture()
        tables['plain', 's', 0]['configurations'][0]['mean_rmsle'] = 1.
        tables['ledger', 's', 0]['configurations'] = []
        out = exploration(data[:-1], tables)
        self.assertEqual(out['arms']['plain']['selected_cv_minimum_sessions'], 1)
        self.assertEqual(out['arms']['ledger']['selected_not_in_final_cv_sessions'], 1)
        self.assertIsNone(out['cases'][0]['arms']['ledger']['selected_cv_rank'])
        self.assertEqual(out['primary_all_matched']['matched_cases'], 1)
        self.assertEqual(out['pending'][0]['missing_arms'], ['ledger'])

    def test_missing_duplicate_or_invalid_scores_reject(self):
        data, tables = fixture()
        with self.assertRaises(ValueError): exploration(data, {})
        for value in (float('nan'), -1, True):
            bad = deepcopy(tables)
            bad['ledger', 's', 0]['configurations'][0]['mean_rmsle'] = value
            with self.subTest(value=value), self.assertRaises(ValueError): exploration(data, bad)
        bad = deepcopy(tables)
        bad['ledger', 's', 0]['configurations'] *= 2
        with self.assertRaises(ValueError): exploration(data, bad)


if __name__ == '__main__':
    unittest.main()
