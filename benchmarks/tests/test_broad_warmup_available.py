import unittest
from benchmarks.ledger_optimization.broad_warmup_available import boundaries


class AvailableWarmupTest(unittest.TestCase):
    def test_missing_rows_exclude_whole_cohort_without_changing_other_origins(self):
        span = {'start_label': '2019-08-12T14:00:00', 'first_scored_origin': '2019-11-07T00:00:00', 'values': [1.]*2074}
        complete = boundaries('s', span)
        span['values'][400] = None
        partial = boundaries('s', span)
        self.assertEqual(len(partial), 8)
        self.assertEqual([r['origin'] for r in partial], [r['origin'] for r in complete])
        self.assertEqual([r['round'] for r in partial if not r['ready']], [-8, -7, -6])
        self.assertTrue(all(r['missing_targets'] == 0 for r in partial))

    def test_zero_is_observed_and_missing_target_excludes(self):
        span = {'start_label': '2019-08-12T14:00:00', 'first_scored_origin': '2019-11-07T00:00:00', 'values': [0.]*2074}
        self.assertTrue(all(r['ready'] for r in boundaries('s', span)))
        span['values'][730] = None
        task = boundaries('s', span)[0]
        self.assertFalse(task['ready']); self.assertEqual(task['missing_targets'], 1)
        self.assertEqual(task['missing_history'], 0)


if __name__ == '__main__':
    unittest.main()
