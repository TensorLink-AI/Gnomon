import math
import unittest
from benchmarks.ledger_optimization.residual_memory import MODELS, fit, apply


class ResidualMemoryTest(unittest.TestCase):
    def pair(self, residual):
        return {'point': {m: [math.expm1(2.)]*24 for m in MODELS},
                'actual': [math.expm1(2.+residual)]*24}

    def test_half_shrink_and_weighted_training(self):
        result = fit([self.pair(1.), self.pair(-1.)], [.75, .25], [1/6]*6)
        for correction in result['correction']:self.assertAlmostEqual(correction, .25)
        self.assertLess(result['max_absolute_gradient'], 1e-12)
        self.assertAlmostEqual(result['objective'], .875)

    def test_stationary_solution_beats_perturbations(self):
        result = fit([self.pair(1.)], [1.], [1/6]*6)
        self.assertAlmostEqual(result['objective'], .5)
        for b in (.4, .6):self.assertGreater((1-b)**2+b*b, result['objective'])

    def test_negative_corrected_log_is_clipped_and_disclosed(self):
        output = apply(self.pair(0.)['point'], [1/6]*6, [-3.]*24)
        self.assertEqual(output['point'], [0.]*24)
        self.assertEqual(output['clipped_leads'], list(range(24)))

    def test_current_forecast_can_be_applied_without_current_actuals(self):
        fitted = fit([self.pair(.5)], [1.], [1/6]*6)
        output = apply(self.pair(0.)['point'], [1/6]*6, fitted['correction'])
        self.assertAlmostEqual(output['point'][0], math.expm1(2.25))
        with self.assertRaises(ValueError):fit([self.pair(1.)], [2.], [1/6]*6)
        invalid = self.pair(0.); invalid['actual'][0] = -1.
        with self.assertRaises(ValueError):fit([invalid], [1.], [1/6]*6)


if __name__ == '__main__':unittest.main()
