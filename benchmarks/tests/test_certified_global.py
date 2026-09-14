import math
import unittest
import numpy as np
from benchmarks.ledger_optimization.certified_global import fit,evaluate
from benchmarks.ledger_optimization.evidence_ensemble import arrays,objective,MODELS

class CertifiedGlobalTests(unittest.TestCase):
    def pair(self,actual):return {'point':{m:[float(i+1)]*24 for i,m in enumerate(MODELS)},'actual':[actual]*24}
    def test_cusp_median_is_certified(self):
        pairs=[self.pair(y) for y in (1.,2.,4.)]
        result=fit(pairs,[1/3]*3)
        predicted=math.expm1(sum(result['weights'][i]*math.log1p(i+1) for i in range(6)))
        self.assertAlmostEqual(predicted,2.,places=4)
        self.assertLessEqual(result['convex_gap_bound'],1e-5)
        self.assertEqual(result['certificate_kind'],'exact_objective_norm_dual_support')
    def test_support_bound_below_random_feasible_values(self):
        pairs=[self.pair(y) for y in (1.,2.,4.)];prepared=list(map(arrays,pairs));q=[1/3]*3;rng=np.random.default_rng(17)
        for w in [np.eye(6)[1],np.full(6,1/6),*rng.dirichlet(np.ones(6),size=8)]:
            _,_,gap,_=evaluate(w,prepared,q);lower=objective(w,prepared,q)[0]-gap
            for v in [np.eye(6)[i] for i in range(6)]+list(rng.dirichlet(np.ones(6),size=40)):
                self.assertLessEqual(lower,objective(v,prepared,q)[0]+1e-12)
    def test_exact_fit_support_zero_is_finite(self):
        p=self.pair(2.);w=np.eye(6)[1]
        value,g,gap,slack=evaluate(w,[arrays(p)],[1.])
        self.assertTrue(np.isfinite(g).all());self.assertEqual(slack,0.)
        self.assertAlmostEqual(value,1e-6+1e-6*5/6)
        self.assertAlmostEqual(gap,2e-6)
    def test_smooth_search_gradient(self):
        p=[arrays(self.pair(8.))];w=np.full(6,1/6);_,g,_,_=evaluate(w,p,[1.])
        for i in range(6):
            d=np.eye(6)[i]*1e-6
            self.assertAlmostEqual(g[i],(evaluate(w+d,p,[1.])[0]-evaluate(w-d,p,[1.])[0])/2e-6,places=7)

if __name__=='__main__':unittest.main()
