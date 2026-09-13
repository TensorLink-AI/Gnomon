from copy import deepcopy
import unittest
from benchmarks.ledger_optimization.broad_calibration import choose, MODELS


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        self.cv = {m: 1. for m in MODELS}
        self.cv['weekly'] = 1.2
        self.history = [{'series_id': 's', 'origin': f'2020-01-0{i}T00:00:00Z',
            'last_target': f'2020-01-0{i+1}T00:00:00Z',
            'outcome_recorded_at': f'2020-01-0{i+1}T00:00:00Z',
            'cv': dict(self.cv), 'scores': {m: (.2 if m == 'weekly' else 1.) for m in MODELS}}
            for i in (1, 2, 3, 4)]

    def test_bias_shrinkage_and_insufficient_evidence(self):
        self.assertEqual(choose(self.cv, self.history[:3], 's', '2020-01-06T00:00:00Z')['provider'], 'daily')
        result = choose(self.cv, self.history, 's', '2020-01-06T00:00:00Z')
        self.assertEqual(result['provider'], 'weekly')
        self.assertEqual(result['bias']['weekly'], -1.)
        self.assertEqual(result['shrinkage_weight'], .5)
        self.assertAlmostEqual(result['estimated_rmsle']['weekly'], .7)

    def test_late_or_future_outcomes_excluded_and_inputs_immutable(self):
        original = deepcopy(self.history)
        expected = choose(self.cv, self.history, 's', '2020-01-06T00:00:00Z')
        row = deepcopy(self.history[-1]); row['outcome_recorded_at'] = '2021-01-01T00:00:00Z'
        row['scores'] = {m: 9999. for m in MODELS}
        self.assertEqual(choose(self.cv, self.history+[row], 's', '2020-01-06T00:00:00Z'), expected)
        row['outcome_recorded_at'] = '2020-01-07T00:00:00Z'; row['last_target'] = row['outcome_recorded_at']
        self.assertEqual(choose(self.cv, self.history+[row], 's', '2020-01-06T00:00:00Z'), expected)
        self.assertEqual(self.history, original)

    def test_duplicate_incomplete_and_foreign_evidence_reject(self):
        variants = [self.history+self.history, deepcopy(self.history), deepcopy(self.history)]
        variants[1][0]['scores'].pop('weekly'); variants[2][0]['series_id'] = 'other'
        for history in variants:
            with self.assertRaises(ValueError):
                choose(self.cv, history, 's', '2020-01-06T00:00:00Z')


if __name__ == '__main__':
    unittest.main()
