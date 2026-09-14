import unittest
import numpy as np
from benchmarks.ledger_optimization.error_transition import MODELS,errors,fit,apply

class ErrorTransitionTests(unittest.TestCase):
    def test_zero_target(self):
        previous=np.ones((6,24)).tolist();f=fit([{'previous_error':previous,'next_error':np.zeros((6,24)).tolist()}],[1.])
        for s in f['systems'].values():self.assertEqual(s['coefficients'],[0.,0.]);self.assertEqual(s['gradient'],[0.,0.])
    def test_known_ridge_solution(self):
        p=np.tile([-1.,1.],12);x=np.tile(p,(6,1));y=2*x;f=fit([{'previous_error':x.tolist(),'next_error':y.tolist()}],[1.])
        for s in f['systems'].values():np.testing.assert_allclose(s['coefficients'],[0.,2/1.1],atol=1e-12);self.assertLess(s['objective'],s['zero_objective'])
    def test_clipping_and_unchanged_zero_correction(self):
        zero=np.zeros((6,24)).tolist();f=fit([{'previous_error':zero,'next_error':zero}],[1.]);point={m:[1.]*24 for m in MODELS};out=apply(point,zero,[1/6]*6,f);np.testing.assert_allclose(out['point'],1.)
        for s in f['systems'].values():s['coefficients']=[-2.,0.]
        out=apply(point,zero,[1/6]*6,f);self.assertEqual(out['point'],[0.]*24);self.assertTrue(all(v==list(range(24)) for v in out['clipped_leads'].values()))
    def test_shapes_and_sign(self):
        point={m:[1.]*24 for m in MODELS};np.testing.assert_allclose(errors(point,[3.]*24),np.log(2.))
        with self.assertRaises(ValueError):errors(point,[1.])
        with self.assertRaises(ValueError):fit([],[])
        with self.assertRaises(ValueError):fit([{'previous_error':[[0.]],'next_error':[[0.]]}],[1.])

if __name__=='__main__':unittest.main()
