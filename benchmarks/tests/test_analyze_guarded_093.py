import random
import unittest
from statistics import mean
from benchmarks.ledger_optimization.analyze_guarded_093 import paired_series_contrast


def pair(series, treatment, control):
    return ({'series_id':series,'rmsle':treatment},{'series_id':series,'rmsle':control})


class ZeroControlAuditTests(unittest.TestCase):
    def test_positive_data_matches_frozen_bootstrap_exactly(self):
        pairs=[pair('a',1,2),pair('a',3,4),pair('b',5,3),pair('c',2,2)]
        rng=random.Random(142);series=['a','b','c'];draws=[]
        for _ in range(2000):
            chosen=[rng.choice(series) for _ in series]
            sample=[(a,b) for s in chosen for a,b in pairs if a['series_id']==s]
            draws.append(1-mean(a['rmsle'] for a,b in sample)/mean(b['rmsle'] for a,b in sample))
        draws.sort();out=paired_series_contrast(pairs)
        self.assertEqual(out['exploratory_series_bootstrap_95_interval'],[draws[49],draws[1949]])
        self.assertEqual(out['relative_rmsle_reduction'],1-2.75/2.75)
        self.assertEqual(out['undefined_relative_draws'],0)

    def test_zero_control_draws_are_not_discarded(self):
        out=paired_series_contrast([pair('a',0,0),pair('b',1,2)])
        self.assertEqual(out['relative_rmsle_reduction'],.5)
        self.assertGreater(out['undefined_relative_draws'],0)
        self.assertLess(out['undefined_relative_draws'],2000)
        self.assertIsNone(out['exploratory_series_bootstrap_95_interval'])
        self.assertEqual(out['relative_interval_status'],'undefined_zero_control_resample')
        self.assertEqual(out['exploratory_absolute_series_bootstrap_95_interval'],[0,1])

    def test_all_zero_scores_are_not_relative_improvement(self):
        out=paired_series_contrast([pair('a',0,0)])
        self.assertIsNone(out['relative_rmsle_reduction'])
        self.assertEqual(out['undefined_relative_draws'],2000)
        self.assertEqual(out['absolute_rmsle_reduction'],0)
        self.assertEqual(out['wins'],0)
        self.assertEqual(out['losses'],0)

    def test_zero_control_with_positive_treatment_retains_loss(self):
        out=paired_series_contrast([pair('a',1,0)])
        self.assertIsNone(out['relative_rmsle_reduction'])
        self.assertEqual(out['losses'],1)
        self.assertEqual(out['absolute_rmsle_reduction'],-1)
        self.assertEqual(out['exploratory_absolute_series_bootstrap_95_interval'],[-1,-1])

    def test_invalid_pairs_reject(self):
        for pairs in [[],[pair('a',float('nan'),1)],[pair('a',1,-1)],
                      [pair('a',True,1)],[({'series_id':'a','rmsle':1},{'series_id':'b','rmsle':1})]]:
            with self.assertRaises(ValueError):paired_series_contrast(pairs)


if __name__=='__main__':unittest.main()
