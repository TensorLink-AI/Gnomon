from copy import deepcopy
import unittest
from benchmarks.ledger_optimization.warm_context import select, features, MODELS, FEATURES


class ContextTest(unittest.TestCase):
    def setUp(self):
        self.cv = {m: (1.2 if m == 'weekly' else 1.) for m in MODELS}
        self.x = features(self.cv, [2.]*730)
        self.rows = [{'series_id': 's'+str(j), 'domain': 'electricity', 'origin': f'2020-01-0{i}T00:00:00Z',
            'last_target': f'2020-01-0{i+1}T00:00:00Z', 'outcome_recorded_at': f'2020-01-0{i+1}T00:00:00Z',
            'features': self.x, 'cv': self.cv, 'scores': {m: (.2 if m == 'weekly' else 1.) for m in MODELS}}
            for i in (1, 2, 3) for j in range(8)]

    def test_constant_context_has_correct_centered_residual_solution(self):
        r = select(self.cv, self.x, 'electricity', '2020-01-05T00:00:00Z', self.rows)
        self.assertEqual(r['provider'], 'weekly')
        self.assertEqual(r['records'], 24); self.assertEqual(r['distinct_origins'], 3)
        self.assertAlmostEqual(r['estimated_rmsle']['weekly'], .7)
        self.assertEqual(len(self.x), len(FEATURES))

    def test_other_domains_same_origin_and_late_outcomes_cannot_change_estimate(self):
        expected = select(self.cv, self.x, 'electricity', '2020-01-05T00:00:00Z', self.rows)
        for fields in ({'domain': 'pedestrian'}, {'outcome_recorded_at': '2021-01-01T00:00:00Z'},
                       {'origin': '2020-01-05T00:00:00Z', 'last_target': '2020-01-06T00:00:00Z'}):
            other = deepcopy(self.rows[0]); other.update(fields); other['scores'] = {m: 9999. for m in MODELS}
            self.assertEqual(select(self.cv, self.x, 'electricity', '2020-01-05T00:00:00Z', self.rows+[other]), expected)

    def test_insufficient_origins_and_duplicate_cohorts(self):
        self.assertEqual(select(self.cv, self.x, 'electricity', '2020-01-05T00:00:00Z', self.rows[:16])['provider'], 'daily')
        with self.assertRaises(ValueError):
            select(self.cv, self.x, 'electricity', '2020-01-05T00:00:00Z', self.rows+self.rows[:1])


if __name__ == '__main__':
    unittest.main()
