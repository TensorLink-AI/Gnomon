import copy
import unittest
import numpy as np
from benchmarks.ledger_optimization import lagged_risk as lr


class LaggedRiskTest(unittest.TestCase):
    def setUp(self):
        self.point={m:[2+i]*24 for i,m in enumerate(lr.MODELS)}
        self.anchor=[1/6]*6
        self.prior={'point':self.point,'actual':[3]*24}

    def test_context_errors_and_missingness(self):
        absent=lr.features(self.point,self.anchor,None)
        present=lr.features(self.point,self.anchor,self.prior)
        self.assertEqual(present.shape,(24,17))
        np.testing.assert_array_equal(present[:,:10],absent[:,:10])
        np.testing.assert_array_equal(absent[:,10:],0)
        np.testing.assert_allclose(present[:,10:16],np.tile(np.log1p(np.arange(2,8))-np.log(4),(24,1)))
        np.testing.assert_array_equal(present[:,-1],1)

    def test_invalid_predecessor_values(self):
        for actual in ([1]*23,[float('nan')]*24,[-1]*24):
            with self.assertRaises(ValueError):lr.features(self.point,self.anchor,{'point':self.point,'actual':actual})

    def test_visibility_and_identity(self):
        ids={m:m for m in lr.MODELS}
        meta={'origin':'2026-01-01T00:00:00Z','last_target':'2026-01-02T00:00:00Z',
              'source_available_at':'2026-01-02T00:00:00Z','recorded_at':'2026-01-02T00:00:00Z','config_ids':ids}
        self.assertTrue(lr.validate_predecessor(meta,'2026-01-02T00:00:00Z',ids))
        self.assertFalse(lr.validate_predecessor(None,'2026-01-02T00:00:00Z',ids))
        with self.assertRaises(ValueError):lr.validate_predecessor(meta,'2026-01-03T00:00:00Z',ids)
        for key,value in [('recorded_at','2026-01-03T00:00:00Z'),('source_available_at','2026-01-03T00:00:00Z'),('origin','2026-01-02T00:00:00Z'),('config_ids',{})]:
            bad={**meta,key:value}
            with self.assertRaises(ValueError):lr.validate_predecessor(bad,'2026-01-02T00:00:00Z',ids)

    def test_psd_prediction_and_simplex(self):
        pair={'point':self.point,'actual':[3]*24,'predecessor':self.prior}
        model,evidence=lr.fit([pair],[1.],self.anchor)
        matrix=lr.predict_matrices(model,self.point,self.anchor,self.prior)
        self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(),-1e-9)
        result=lr.combine(self.point,self.anchor,matrix)
        self.assertEqual(result['weight_fit_count'],24)
        self.assertEqual(evidence['feature_count'],17)
        self.assertTrue(all(2-1e-9<=v<=7+1e-9 for v in result['point']))

    def test_query_cannot_take_current_actual(self):
        # Query context is supplied independently of the target labels used by fit.
        pair={'point':self.point,'actual':[3]*24,'predecessor':copy.deepcopy(self.prior)}
        before=lr.features(pair['point'],self.anchor,pair['predecessor'])
        pair['actual']=[100]*24
        np.testing.assert_array_equal(before,lr.features(pair['point'],self.anchor,pair['predecessor']))
