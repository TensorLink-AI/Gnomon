import unittest
import numpy as np
from benchmarks.ledger_optimization import norm_risk as nr


class NormRiskTest(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(17)
        self.point={m:self.rng.uniform(0,10,24).tolist() for m in nr.MODELS}
        self.actual=self.rng.uniform(0,10,24).tolist();self.anchor=np.full(6,1/6)

    def errors(self):
        return np.log1p(np.array([self.point[m] for m in nr.MODELS]).T)-np.log1p(self.actual)[:,None]

    def test_upper_bound_and_tangency(self):
        g,info=nr.targets(self.point,self.actual,self.anchor);e=self.errors()
        for w in [np.tile(self.anchor,(24,1)),*self.rng.dirichlet(np.ones(6),size=(20,24))]:
            n=np.sqrt(np.mean(np.sum(e*w,axis=1)**2)+nr.SMOOTHING**2)
            q=np.einsum('hi,hij,hj->h',w,g,w).mean()+info['majorizer_constant']
            self.assertGreaterEqual(q+1e-12,n)
        q=np.einsum('i,hij,j->h',self.anchor,g,self.anchor).mean()+info['majorizer_constant']
        self.assertAlmostEqual(q,info['anchor_smoothed_rmsle'],places=12)
        self.assertGreaterEqual(np.linalg.eigvalsh(g).min(),-1e-9)

    def test_tangent_gradient(self):
        g,info=nr.targets(self.point,self.actual,self.anchor);e=self.errors()
        exact=e*(e@self.anchor)[:,None]/(24*info['anchor_smoothed_rmsle'])
        surrogate=np.einsum('hij,j->hi',g,self.anchor)/12
        np.testing.assert_allclose(surrogate,exact,atol=1e-12)

    def test_exact_fit_scaling_and_solver(self):
        point={m:[float(np.expm1(.1*(i+1)))]*24 for i,m in enumerate(nr.MODELS)}
        actual=[float(np.expm1(.35))]*24
        g,info=nr.targets(point,actual,self.anchor)
        self.assertAlmostEqual(info['anchor_smoothed_rmsle'],1e-6)
        self.assertAlmostEqual(info['gram_scale'],500000.)
        pair={'point':point,'actual':actual}
        model,evidence=nr.fit([pair],[1.],self.anchor.tolist())
        matrices=nr.predict_matrices(model,point,self.anchor)
        result=nr.combine(point,self.anchor,matrices)
        self.assertEqual(result['weight_fit_count'],24)
        np.testing.assert_allclose(result['point'],actual,atol=1e-7)

    def test_rejections(self):
        for a in ([1]*6,[float('nan')]*6,[1/5]*5):
            with self.assertRaises(ValueError):nr.targets(self.point,self.actual,a)
        with self.assertRaises(ValueError):nr.targets(self.point,[-1]*24,self.anchor)
        with self.assertRaises(ValueError):nr.fit([],[],self.anchor)
