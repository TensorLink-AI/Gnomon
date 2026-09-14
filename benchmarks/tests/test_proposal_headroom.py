import math
import unittest
from benchmarks.ledger_optimization.proposal_headroom import best,envelope,score,summary,BASELINES,ARMS,ORACLES


class ProposalHeadroomTest(unittest.TestCase):
    def test_hindsight_and_ties(self):
        self.assertEqual(best({'a':2.,'b':1.,'c':1.}),{'score':1.,'selected':'b','ties':['b','c']})
        self.assertAlmostEqual(score([0.,3.],[0.,0.]),math.log(4)/math.sqrt(2))

    def test_projection_handles_inside_below_and_above(self):
        result=envelope({'a':[1.,1.,1.],'b':[3.,3.,3.]},[2.,0.,7.])
        for observed,expected in zip(result['log_residual'],[0.,math.log(2),-math.log(2)],strict=True):self.assertAlmostEqual(observed,expected,places=14)
        self.assertAlmostEqual(result['score'],math.log(2)*math.sqrt(2/3))

    def test_envelope_lower_bounds_every_convex_stepwise_combination(self):
        points={'a':[1.,8.,3.],'b':[3.,2.,7.]};actual=[2.,4.,12.];bound=envelope(points,actual)['score']
        for w in (0.,.2,.5,.8,1.):
            mixed=[math.expm1(w*math.log1p(a)+(1-w)*math.log1p(b)) for a,b in zip(points['a'],points['b'])]
            self.assertLessEqual(bound,score(mixed,actual)+1e-12)

    def test_weak_and_strong_control_are_not_conflated(self):
        scores={a:1. for a in (*BASELINES,*ARMS,*ORACLES)};scores.update(selected_cv=2.,block_cv=1.)
        result=summary([{'scores':scores}])['comparisons']
        self.assertEqual(result['selected_cv']['provider_oracle']['maximum_relative_improvement'],.5)
        self.assertEqual(result['block_cv']['provider_oracle']['twenty_percent_status'],'impossible_in_stated_class')

    def test_invalid_inputs(self):
        for p,y in [([],[]),([1.],[1.,2.]),([-1.],[0.]),([float('nan')],[1.])]:
            with self.assertRaises(ValueError):score(p,y)
        with self.assertRaises(ValueError):envelope({},[1.])
        with self.assertRaises(ValueError):best({})


if __name__=='__main__':unittest.main()
