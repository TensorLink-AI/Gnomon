"""Independent arithmetic must retain failures and reject damaged evidence."""
import copy
import math
import unittest

from benchmarks.ledger_optimization.verify_planning_scores_107 import calculate


class Scores(unittest.TestCase):
    def setUp(self):
        self.jobs = {'s': [{'series_id': 's', 'round': 0, 'origin': '2026-01-01',
                            'actual': [0]*14}]}
        self.grades = [{'arm': arm, 'series_id': 's', 'round': 0, 'origin': '2026-01-01',
                        'point': [value]*14, 'rmsle': math.log1p(value), 'valid': True,
                        'workflow_complete': True, 'fallback_used': False}
                       for arm, value in [('plain', 3), ('gnomon', 1), ('ledger', 0)]]

    def test_independent_error_and_contrast(self):
        result = calculate(self.jobs, self.grades)
        self.assertEqual(result['matched_cases'], 1)
        self.assertAlmostEqual(result['arms']['gnomon']['mean_rmsle'], math.log(2))
        self.assertEqual(result['contrasts']['ledger_vs_gnomon']['relative_rmsle_reduction'], 1)

    def test_failed_workflow_is_not_dropped(self):
        g = self.grades[-1]
        g.update(valid=False, workflow_complete=False, fallback_used=True, point=[7]*14, rmsle=math.log(8))
        result = calculate(self.jobs, self.grades)
        self.assertEqual(result['sessions'], 3)
        self.assertEqual(result['arms']['ledger']['fallbacks'], 1)
        self.assertAlmostEqual(result['contrasts']['ledger_vs_gnomon']['relative_rmsle_reduction'], -2)

    def test_missing_duplicate_and_unrelated_grades_reject(self):
        for grades in (self.grades[:-1], self.grades+[self.grades[0]],
                       [{**g, 'series_id': 'other'} for g in self.grades]):
            with self.subTest(grades=grades), self.assertRaises(ValueError):
                calculate(self.jobs, grades)

    def test_changed_origin_score_horizon_and_nonfinite_reject(self):
        for update in ({'origin': '2026-01-02'}, {'rmsle': .1}, {'point': [0]},
                       {'point': [float('nan')]*14}, {'point': [True]*14},
                       {'point': [-1]*14}, {'valid': 'true'}):
            grades = copy.deepcopy(self.grades); grades[-1].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                calculate(self.jobs, grades)

    def test_zero_control_relative_contrast_stays_undefined(self):
        self.grades[1].update(point=[0]*14, rmsle=0)
        result = calculate(self.jobs, self.grades)
        contrast = result['contrasts']['ledger_vs_gnomon']
        self.assertIsNone(contrast['relative_rmsle_reduction'])
        self.assertIsNone(contrast['exploratory_series_bootstrap_95_interval'])
        self.assertEqual(contrast['undefined_relative_draws'], 2000)

    def test_per_case_mean_not_pooled_error(self):
        self.jobs['s'].append({**self.jobs['s'][0], 'round': 1})
        extra = [{**g, 'round': 1, 'point': [0]*14, 'rmsle': 0} for g in self.grades]
        result = calculate(self.jobs, self.grades+extra)
        self.assertAlmostEqual(result['arms']['gnomon']['mean_rmsle'], math.log(2)/2)


if __name__ == '__main__':
    unittest.main()
