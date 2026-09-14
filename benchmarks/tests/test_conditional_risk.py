import unittest
import numpy as np
from benchmarks.ledger_optimization.conditional_risk import MODELS, gram, train, predict, fit_weights, visible


class ConditionalRiskTest(unittest.TestCase):
    def test_gram_diagonals_and_psd(self):
        pair = {'point': {m: [float(i+1)]*24 for i, m in enumerate(MODELS)}, 'actual': [3.]*24}
        g = gram(pair); e = np.log1p(np.arange(1, 7))-np.log1p(3.)
        np.testing.assert_allclose(g, np.outer(e, e), atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(g).min(), -1e-12)

    def test_forest_output_reconstructs_past_psd_targets(self):
        x = [[float(i)]*12 for i in range(40)]; matrices = [(np.eye(6)*(i+1)).tolist() for i in range(40)]
        model, record = train(x, matrices); result = predict(model, record, [10.]*12)
        self.assertAlmostEqual(sum(result['record_weights']), 1.)
        self.assertTrue(all(w >= 0 for w in result['record_weights']))
        np.testing.assert_allclose(result['matrix'], np.einsum('n,nij->ij', result['record_weights'], matrices))

    def test_quadratic_weights_and_non_psd_rejection(self):
        result = fit_weights(np.diag([1., 2., 3., 4., 5., 6.]), [1/6]*6)
        self.assertLessEqual(result['convex_gap_bound'], 1e-5)
        self.assertGreater(result['weights'][0], result['weights'][-1])
        with self.assertRaises(ValueError):fit_weights(-np.eye(6), [1/6]*6)

    def test_temporal_boundaries_and_no_same_origin_outcomes(self):
        row = {'domain': 'electricity', 'series_id': 'electricity:a', 'origin': '2020-01-01T00:00:00+00:00',
               'last_target': '2020-01-02T00:00:00+00:00', 'outcome_recorded_at': '2020-01-03T00:00:00+00:00'}
        self.assertEqual(visible([row], 'electricity', row['origin']), [])
        self.assertEqual(visible([row], 'electricity', row['last_target']), [])
        self.assertEqual(visible([row], 'electricity', row['outcome_recorded_at']), [row])
        self.assertEqual(visible([row], 'pedestrian', row['outcome_recorded_at']), [])
        with self.assertRaises(ValueError):visible([row, row], 'electricity', row['outcome_recorded_at'])


if __name__ == '__main__':unittest.main()
