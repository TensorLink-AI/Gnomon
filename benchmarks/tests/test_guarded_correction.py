import unittest
import numpy as np
from benchmarks.ledger_optimization.guarded_correction import MODELS,RECIPE,visible,training,features,fit,apply,decide

class GuardedCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.point={m:[float(i+2+h%4) for h in range(24)] for i,m in enumerate(MODELS)};self.weights=[1/6]*6
    def record(self,**changes):
        r={'domain':'d','series_id':'s','origin':'2020-01-01T00:00:00Z','last_target':'2020-01-02T00:00:00Z','source_available_at':'2020-01-02T00:00:00Z','recorded_at':'2020-01-02T00:00:00Z'};r.update(changes);return r
    def test_visibility_inclusive_and_no_future_payload(self):
        r=self.record();later=self.record(series_id='late',recorded_at='2020-01-03T00:00:00Z');future=self.record(series_id='future',origin='2020-01-02T00:00:00Z',last_target='2020-01-03T00:00:00Z');other={'domain':'other'}
        self.assertEqual(visible([later,future,r,other],'2020-01-02T00:00:00Z','d'),[r]);self.assertEqual(visible([r],'2020-01-01T23:59:59Z','d'),[])
    def test_reject_duplicate_and_bad_chronology(self):
        r=self.record()
        with self.assertRaises(ValueError):visible([r,r],'2020-01-03T00:00:00Z','d')
        with self.assertRaises(ValueError):visible([self.record(recorded_at='2020-01-01T12:00:00Z')],'2020-01-03T00:00:00Z','d')
    def test_masses_and_folds(self):
        current=[{'id':1},{'id':2}];past=[{'id':3}]*5;p,q=training(current,past);self.assertEqual(p,current+past);self.assertEqual(q,[.25,.25]+[.1]*5)
        self.assertEqual(training(current,[])[1],[.5,.5])
        with self.assertRaises(ValueError):training([{}],[])
    def test_features_do_not_use_actuals(self):
        x,base=features(self.point,self.weights);self.assertEqual(x.shape,(24,10));self.assertTrue(np.isfinite(x).all());np.testing.assert_allclose(base,np.mean(np.log1p(list(self.point.values())),axis=0))
    def test_zero_residual_and_tree_recipe(self):
        _,base=features(self.point,self.weights);pair={'point':self.point,'actual':np.expm1(base).tolist()};model,info=fit([pair]*2,[.5,.5],self.weights);output=apply(self.point,self.weights,model)
        np.testing.assert_allclose(output['point'],pair['actual'],atol=1e-12);self.assertEqual(info['tree_count'],32)
        for k,v in RECIPE.items():self.assertEqual(info['recipe'][k],v)
    def test_holdback_choice_retains_baseline_on_failure_or_tie(self):
        self.assertFalse(decide([2.]*24,[3.]*24,[2.]*24)['correction_enabled']);self.assertFalse(decide([2.]*24,[2.]*24,[2.]*24)['correction_enabled']);self.assertTrue(decide([1.]*24,[2.]*24,[2.]*24)['correction_enabled'])
    def test_reject_incomplete_and_nonfinite(self):
        with self.assertRaises(ValueError):features({**self.point,'seventh':[1.]*24},self.weights)
        with self.assertRaises(ValueError):fit([{'point':self.point,'actual':[float('nan')]*24}],[1.],self.weights)

if __name__=='__main__':unittest.main()
