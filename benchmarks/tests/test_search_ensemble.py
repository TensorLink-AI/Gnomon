import copy
import unittest
import numpy as np
from benchmarks.ledger_optimization import search_ensemble as se
from benchmarks.ledger_optimization.lifetime_context import retrieve


def pairs():
    return [{'point':{m:[float(j+i%4+1) for i in range(24)] for j,m in enumerate(se.MODELS)},'actual':[float(3+i%4) for i in range(24)]}]


class SearchEnsembleTest(unittest.TestCase):
    def test_analytic_gradient_matches_finite_differences(self):
        x=np.full(28,1/7);a=np.full(7,1/7);prepared=se.prepare(pairs());q=np.array([1.]);_,gradient=se.objective(x,prepared,q,a)
        for j in range(28):
            d=np.zeros(28);d[j]=1e-6
            numeric=(se.objective(x+d,prepared,q,a)[0]-se.objective(x-d,prepared,q,a)[0])/2e-6
            self.assertAlmostEqual(gradient[j],numeric,places=7)
    def test_certificate_and_geometric_blend(self):
        p=pairs();anchor=[1/6]*6+[0.];f=se.fit(p,[1.],anchor)
        self.assertLessEqual(f['convex_gap_bound'],1e-5);w=np.array(f['weights']);self.assertTrue(np.allclose(w.sum(axis=1),1.))
        output=se.combine(p[0]['point'],f['weights'])
        for h,v in enumerate(output):self.assertAlmostEqual(v,np.expm1(sum(w[h//6,j]*np.log1p(p[0]['point'][m][h]) for j,m in enumerate(se.MODELS))),places=12)
    def test_selected_slot_may_duplicate_an_original(self):
        p=pairs();p[0]['point']['search_selected']=p[0]['point']['daily'].copy();f=se.fit(p,[1.],[1/6]*6+[0.])
        self.assertLessEqual(f['convex_gap_bound'],1e-5)
    def test_missing_or_unmatched_slots_rejected(self):
        p=pairs();del p[0]['point']['search_selected']
        with self.assertRaises(KeyError):se.fit(p,[1.],[1/6]*6+[0.])
        p=pairs();p[0]['actual']=p[0]['actual'][:-1]
        with self.assertRaises(ValueError):se.fit(p,[1.],[1/6]*6+[0.])
    def test_retrieval_checks_maturity_before_features(self):
        current={'origin':'2020-02-01T00:00:00+00:00','domain':'electricity','features':[0.]*12}
        late={'domain':'electricity','series_id':'electricity:s','origin':'2020-01-01T00:00:00+00:00','last_target':'2020-01-02T00:00:00+00:00','outcome_recorded_at':'2030-01-01T00:00:00+00:00','features':object()}
        self.assertEqual(retrieve(current,[late]),retrieve(current,[]))


if __name__=='__main__':unittest.main()
