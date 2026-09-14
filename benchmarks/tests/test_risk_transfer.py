import unittest
from benchmarks.ledger_optimization.risk_transfer import quadratic, comparison, loss


class RiskTransferTest(unittest.TestCase):
    def test_quadratic_and_loss(self):
        g=[[float(i==j) for j in range(6)] for i in range(6)]
        self.assertAlmostEqual(quadratic(g,[1/6]*6),1/6)
        self.assertEqual(loss([2,3],[2,3]),0)

    def test_conditional_denominators(self):
        r=comparison([1,2,0,-1],[2,-1,0,-3])
        self.assertEqual(r['predicted_positive'],2)
        self.assertEqual(r['realised_positive_when_predicted_positive'],1)
        self.assertEqual(r['sign_agreement'],3)
        self.assertEqual(r['realised_ties'],1)
        self.assertIsNone(comparison([1,1],[2,3])['pearson'])

    def test_identical_and_opposite_correlations(self):
        self.assertAlmostEqual(comparison([1,2,3],[2,4,6])['pearson'],1)
        self.assertAlmostEqual(comparison([1,2,3],[-2,-4,-6])['pearson'],-1)
        with self.assertRaises(ValueError):comparison([],[])
        with self.assertRaises(ValueError):comparison([1],[1,2])
