import unittest
from benchmarks.ledger_optimization.broad_opportunity import bounds, summarize


class OpportunityTest(unittest.TestCase):
    def test_perfect_filter_cannot_invent_an_unproposed_execution(self):
        scores = {'control': 1., 'proposed_bad': 2., 'unproposed_best': .2}
        r = bounds(scores, 'control', {'past': 'proposed_bad'}, 0)
        self.assertEqual(r['past_perfect_filter'], 1.)
        self.assertEqual(r['proposal_union_oracle'], 1.)
        self.assertEqual(r['all_six_oracle'], .2)
        self.assertEqual(r['all_six_after_4_origins'], 1.)
        self.assertEqual(bounds(scores, 'control', {'past': 'proposed_bad'}, 4)['all_six_after_4_origins'], .2)

    def test_zero_headroom_and_impossible_target_are_explicit(self):
        row = {'bounds': bounds({'a': 1.}, 'a', {'past': 'a'}, 5)}
        r = summarize([row])
        self.assertIsNone(r['oracle_gain_capture_required'])
        self.assertFalse(r['twenty_percent_feasible_within_each_bound']['all_six_oracle'])
        self.assertAlmostEqual(r['remaining_regret_allowed_at_target'], -.2)

    def test_required_capture_uses_total_error_reduction(self):
        row = {'bounds': bounds({'a': 1., 'b': .75}, 'a', {'past': 'b'}, 5)}
        r = summarize([row])
        self.assertAlmostEqual(r['oracle_gain_capture_required'], .8)
        self.assertAlmostEqual(r['remaining_regret_allowed_at_target'], .05)


if __name__ == '__main__':
    unittest.main()
