import unittest
import numpy as np
from benchmarks.ledger_optimization import lead_ensemble as lead,search_ensemble as block
from benchmarks.tests.test_search_ensemble import pairs


class LeadEnsembleTest(unittest.TestCase):
    def test_repeated_block_weights_have_identical_objective_and_forecast(self):
        p=pairs();a=np.array([1/6]*6+[0.]);w=np.array([[.4,.1,.1,.1,.1,.1,.1],[.1,.4,.1,.1,.1,.1,.1],[.1,.1,.4,.1,.1,.1,.1],[.1,.1,.1,.4,.1,.1,.1]])
        full=np.repeat(w,6,axis=0);old,og=block.objective(w.ravel(),block.prepare(p),np.array([1.]),a);new,ng=lead.objective(full.ravel(),lead.prepare(p),np.array([1.]),a)
        self.assertAlmostEqual(old,new,places=12);self.assertTrue(np.allclose(og.reshape(4,7),ng.reshape(4,6,7).sum(axis=1)))
        self.assertTrue(np.allclose(block.combine(p[0]['point'],w),lead.combine(p[0]['point'],full)))
    def test_gradient_selected_hour_matches_finite_difference(self):
        p=pairs();a=np.array([1/6]*6+[0.]);w=np.full(168,1/7);args=lead.prepare(p),np.array([1.]),a;_,g=lead.objective(w,*args)
        for j in (0,6,7,83,167):
            d=np.zeros(168);d[j]=1e-6;numeric=(lead.objective(w+d,*args)[0]-lead.objective(w-d,*args)[0])/2e-6;self.assertAlmostEqual(g[j],numeric,places=7)
    def test_full_hourly_fit_certificate_and_shapes(self):
        p=pairs();f=lead.fit(p,[1.],[1/6]*6+[0.]);self.assertEqual(np.shape(f['weights']),(24,7));self.assertLessEqual(f['convex_gap_bound'],1e-5)
        point=lead.combine(p[0]['point'],f['weights']);self.assertEqual(len(point),24);self.assertTrue(all(np.isfinite(point)))
    def test_wrong_granularity_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):lead.combine(pairs()[0]['point'],[[1/7]*7]*4)


if __name__=='__main__':unittest.main()
