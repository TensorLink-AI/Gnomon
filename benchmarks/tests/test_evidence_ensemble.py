import math
import unittest
import numpy as np
from benchmarks.ledger_optimization.evidence_ensemble import MODELS, fit, combine, arrays, objective


class EnsembleTest(unittest.TestCase):
    def pair(self):
        return {'point': {m: [float(i+1)]*24 for i, m in enumerate(MODELS)}, 'actual': [3.2]*24}

    def test_combination_preserves_identity_forecast_and_log_mean(self):
        point = self.pair()['point']
        self.assertEqual(combine(point, [1., 0., 0., 0., 0., 0.]), [1.]*24)
        expected = math.expm1(sum(math.log1p(i) for i in range(1, 7))/6)
        self.assertAlmostEqual(combine(point, [1/6]*6)[0], expected)

    def test_analytic_gradient_matches_finite_difference(self):
        pair = self.pair(); prepared = [arrays(pair)]; w = np.full(6, 1/6)
        _, grad = objective(w, prepared, [1.])
        for i in range(6):
            delta = np.eye(6)[i]*1e-6
            observed = (objective(w+delta, prepared, [1.])[0]-objective(w-delta, prepared, [1.])[0])/2e-6
            self.assertAlmostEqual(grad[i], observed, places=7)

    def test_solver_has_feasible_certified_solution(self):
        pair = self.pair(); pair['actual'] = [8.]*24
        r = fit([pair], [1.])
        self.assertAlmostEqual(sum(r['weights']), 1.)
        self.assertLessEqual(r['convex_gap_bound'], 1e-5)
        self.assertAlmostEqual(combine(pair['point'], r['weights'])[0], 6., places=5)

    def test_exact_interior_fit_has_valid_near_zero_certificate(self):
        pair = self.pair()
        r = fit([pair], [1.])
        self.assertLessEqual(r['convex_gap_bound'], 1e-5)
        self.assertAlmostEqual(combine(pair['point'], r['weights'])[0], 3.2, places=6)


if __name__ == '__main__':unittest.main()
