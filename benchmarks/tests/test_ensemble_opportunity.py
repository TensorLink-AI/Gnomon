import unittest
from benchmarks.ledger_optimization.ensemble_opportunity import interval, summary


class OpportunityIntervalTest(unittest.TestCase):
    def test_regularizer_and_gap_are_removed_conservatively(self):
        lo, hi = interval({'objective': .3, 'convex_gap_bound': 1e-5}, .3)
        self.assertAlmostEqual(lo, .3-1e-5-(5/6)*1e-6)
        self.assertEqual(hi, .3)

    def test_improvement_upper_bound_controls_impossibility_claim(self):
        r = summary([{'control_rmsle': 1., 'lower_bound': .79, 'upper_bound': .81}])
        self.assertFalse(r['twenty_percent_ruled_out'])
        r = summary([{'control_rmsle': 1., 'lower_bound': .81, 'upper_bound': .82}])
        self.assertTrue(r['twenty_percent_ruled_out'])


if __name__ == '__main__':unittest.main()
