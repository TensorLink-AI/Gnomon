import unittest
import numpy as np
from benchmarks.ledger_optimization.intraday_ensemble import MODELS, combine, fit, objective, prepare


class IntradayTest(unittest.TestCase):
    def pair(self):
        return {'point': {m: [float(i+1)]*24 for i, m in enumerate(MODELS)},
                'actual': [2.]*6+[5.]*6+[2.]*6+[5.]*6}

    def test_analytic_gradient(self):
        pair = self.pair(); pair['actual'] = [8.]*24
        inputs = prepare([pair]); q = np.array([1.]); anchor = np.full(6, 1/6); weights = np.full(24, 1/6)
        _, gradient = objective(weights, inputs, q, anchor)
        for j in range(24):
            delta = np.eye(24)[j]*1e-6
            observed = (objective(weights+delta, inputs, q, anchor)[0]-objective(weights-delta, inputs, q, anchor)[0])/2e-6
            self.assertAlmostEqual(observed, gradient[j], places=7)

    def test_block_assignment(self):
        weights = np.eye(6)[[0, 1, 2, 3]]
        actual = combine(self.pair()['point'], weights)
        np.testing.assert_allclose(actual, [1.]*6+[2.]*6+[3.]*6+[4.]*6)

    def test_fit_improves_varying_target_with_certificate(self):
        result = fit([self.pair()], [1.], [1/6]*6)
        self.assertTrue(result['success']); self.assertLessEqual(result['convex_gap_bound'], 1e-5)
        self.assertLess(result['objective'], result['initial_objective'])
        np.testing.assert_allclose(combine(self.pair()['point'], result['weights']), self.pair()['actual'], atol=1e-5)

    def test_invalid_simplex_and_mass_rejected(self):
        with self.assertRaises(ValueError):fit([self.pair()], [2.], [1/6]*6)
        with self.assertRaises(ValueError):combine(self.pair()['point'], [[1.]*6]*4)


if __name__ == '__main__':unittest.main()
