import unittest
import numpy as np
from benchmarks.ledger_optimization.conditional_correction import MODELS,fit,apply,raw_features,design,objective

class ConditionalCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.weights=[[1/7]*7 for _ in range(4)]
        self.points={m:[float(2+i+h%3) for h in range(24)] for i,m in enumerate(MODELS)}
    def test_zero_residual(self):
        _,base=raw_features(self.points,self.weights);pairs=[{'point':self.points,'actual':np.expm1(base).tolist()}]*3
        f=fit(pairs,[1/3]*3,self.weights);np.testing.assert_allclose(f['coefficients'],0,atol=1e-10);np.testing.assert_allclose(apply(self.points,f)['point'],pairs[0]['actual'],atol=1e-10)
    def test_constant_bias_shrinks_and_extrapolates(self):
        p={m:[1.]*24 for m in MODELS};pairs=[{'point':p,'actual':[3.]*24}]*3;f=fit(pairs,[1/3]*3,self.weights);a=apply(p,f)
        self.assertLess(f['objective'],f['initial_objective']);self.assertLessEqual(f['suboptimality_bound'],1e-8);self.assertTrue(all(1<v<3.00001 for v in a['point']));self.assertEqual(a['outside_model_range_leads'],list(range(24)))
    def test_analytic_gradient(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(3,24,11));res=rng.normal(size=(3,24));t=rng.normal(size=11);q=np.array([.2,.3,.5]);_,g=objective(t,x,res,q);step=1e-6
        numeric=[(objective(t+np.eye(11)[j]*step,x,res,q)[0]-objective(t-np.eye(11)[j]*step,x,res,q)[0])/(2*step) for j in range(11)];np.testing.assert_allclose(g,numeric,atol=1e-8)
    def test_scaling_bound_to_training_and_current_actual_absent(self):
        f=fit([{'point':self.points,'actual':[4.]*24}]*3,[1/3]*3,self.weights);before=dict(f);first=apply(self.points,f);second=apply(self.points,f);self.assertEqual(first,second);self.assertEqual(before,f);self.assertTrue(all(v>=.1 for v in f['scale']))
    def test_reject_bad_shapes(self):
        with self.assertRaises(ValueError):fit([{'point':self.points,'actual':[1.]}],[1.],self.weights)
        with self.assertRaises(ValueError):fit([{'point':self.points,'actual':[1.]*24}],[.5],self.weights)
        with self.assertRaises(ValueError):raw_features(self.points,[[0.]*7]*4)

if __name__=='__main__':unittest.main()
