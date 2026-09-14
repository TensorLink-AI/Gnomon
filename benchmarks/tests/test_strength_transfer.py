import unittest
from benchmarks.ledger_optimization.strength_transfer import KEYS,pair_counts,case_metrics,summarize


class StrengthTransferTests(unittest.TestCase):
    def test_rank_counts_and_ties(self):
        a=dict(zip(KEYS,range(5)));b={k:4-v for k,v in a.items()};same=pair_counts(a,a);reverse=pair_counts(a,b)
        self.assertEqual(same['concordant'],10);self.assertEqual(reverse['discordant'],10)
        tied=dict.fromkeys(KEYS,0.);self.assertEqual(pair_counts(tied,tied)['both_tied'],10)
        self.assertEqual(pair_counts(tied,a)['historical_only_tie'],10);self.assertEqual(pair_counts(a,tied)['realized_only_tie'],10)
    def test_historical_success_can_fail_later(self):
        h={'0':2.,'0.25':1.5,'0.5':1.,'0.75':.75,'1':.5};r={k:2-v for k,v in h.items()};m=case_metrics(h,r,'1')
        self.assertEqual(m['historical_advantage_vs_half'],.5);self.assertEqual(m['realized_advantage_vs_half'],-.5);self.assertEqual(m['realized_comparison'],'worse');self.assertGreaterEqual(m['selected_regret'],m['half_regret'])
    def test_finite_bound_and_aggregate(self):
        h=dict.fromkeys(KEYS,1.);r={k:1.+i/10 for i,k in enumerate(KEYS)};m=case_metrics(h,r,'0.5');row={'realized':r,'selected':1.2,'control':1.1,'strong_block_cv':1.1,'metrics':m};s=summarize([row,row])
        self.assertEqual(s['cases'],2);self.assertAlmostEqual(s['mean_rmsle']['hindsight_minimum'],1.);self.assertFalse(s['finite_family_can_reach_twenty_percent']);self.assertEqual(sum(s['pair_counts'].values()),20);self.assertIsNone(s['untied_pair_concordance'])
    def test_reject_incomplete(self):
        with self.assertRaises(ValueError):pair_counts({},dict.fromkeys(KEYS,1.))
        with self.assertRaises(ValueError):summarize([])

if __name__=='__main__':unittest.main()
