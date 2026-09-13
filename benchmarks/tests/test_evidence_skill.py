import math
import unittest
from benchmarks.ledger_optimization.evidence_skill import MODELS, ESTIMATES, assess_risks, ensemble_loss, ordering, summarize


class EvidenceSkillTest(unittest.TestCase):
    def test_common_risk_offset_does_not_change_pairwise_evidence(self):
        actual = {m: float(i) for i, m in enumerate(MODELS)}
        estimates = {method: {m: v+2 for m, v in actual.items()} for method in ESTIMATES}
        scores = assess_risks(estimates, actual)
        for method in ESTIMATES:
            self.assertEqual(scores[method]['contrast_mse'], 0.)
            self.assertEqual(scores[method]['risk_mse'], 4.)
            self.assertEqual(len(scores[method]['contrasts']), 15)

    def test_opposite_ranking_and_tie_counts(self):
        actual = {m: float(i) for i, m in enumerate(MODELS)}
        estimates = {method: {m: 5-v for m, v in actual.items()} for method in ESTIMATES}
        scores = assess_risks(estimates, actual)
        self.assertTrue(all(p['ordering'] == 'opposite' for p in scores['cv']['contrasts']))
        self.assertEqual(ordering(0., 1.), 'predicted_tied')
        self.assertEqual(ordering(1., 0.), 'actual_tied')

    def test_log_ensemble_loss_and_invalid_inputs(self):
        points = {m: [float(i)]*24 for i, m in enumerate(MODELS)}
        expected = abs(sum(math.log1p(i) for i in range(6))/6-math.log1p(2.))
        self.assertAlmostEqual(ensemble_loss(points, [2.]*24, [1/6]*6), expected)
        with self.assertRaises(ValueError):ensemble_loss(points, [2.]*24, [1.]*6)

    def test_zero_denominator_and_gain_accounting(self):
        actual = {m: 1. for m in MODELS}
        assessments = assess_risks({method: actual for method in ESTIMATES}, actual)
        rows = [{'risk_assessment': assessments, 'gain_estimates': dict.fromkeys(ESTIMATES, 0.), 'actual_gain': gain}
                for gain in (1., -2., 0.)]
        r = summarize(rows)
        self.assertIsNone(r['risk_prediction']['historical']['contrast_skill_vs_cv'])
        self.assertEqual(r['actual_gain']['helped'], 1)
        self.assertEqual(r['actual_gain']['hurt'], 1)
        self.assertEqual(r['actual_gain']['sum_gain'], 1.)
        self.assertEqual(r['actual_gain']['sum_loss'], 2.)


if __name__ == '__main__':unittest.main()
