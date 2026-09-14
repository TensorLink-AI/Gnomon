import unittest
import numpy as np
from benchmarks.ledger_optimization.local_risk import MODELS,gram_targets,fit,predict_matrices,combine

class LocalRiskTests(unittest.TestCase):
    def setUp(self):
        self.anchor=[1/6]*6;self.point={m:[float(2+3*i)]*24 for i,m in enumerate(MODELS)};self.actual=[2.]*12+[17.]*12
    def test_rank_one_error_matrices(self):
        g=gram_targets(self.point,self.actual)
        for h in range(24):
            e=np.log1p([self.point[m][h] for m in MODELS])-np.log1p(self.actual[h]);np.testing.assert_allclose(g[h],np.outer(e,e));self.assertGreaterEqual(np.linalg.eigvalsh(g[h]).min(),-1e-12)
    def test_joint_forest_is_psd_and_lead_conditional(self):
        p={'point':self.point,'actual':self.actual};model,info=fit([p]*6,[1/6]*6,self.anchor);g=predict_matrices(model,self.point,self.anchor)
        self.assertEqual(info['tree_count'],32);self.assertEqual(g.shape,(24,6,6));self.assertGreaterEqual(np.linalg.eigvalsh(g).min(),-1e-9);self.assertFalse(np.allclose(g[3],g[18]))
    def test_zero_risk_preserves_anchor_and_bounds(self):
        result=combine(self.point,self.anchor,np.zeros((24,6,6)))
        for f in result['weight_fits']:np.testing.assert_allclose(f['weights'],self.anchor,atol=1e-10)
        expected=np.expm1(np.mean(np.log1p([2+3*i for i in range(6)])));np.testing.assert_allclose(result['point'],expected)
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):gram_targets(self.point,[1.])
        with self.assertRaises(ValueError):fit([{'point':self.point,'actual':self.actual}],[.2],self.anchor)
        with self.assertRaises(ValueError):combine(self.point,self.anchor,np.zeros((6,6)))

if __name__=='__main__':unittest.main()
