"""Check independent audit arithmetic on analytically known synthetic pairs."""
import math
import unittest
from benchmarks.ledger_optimization.validation_verify import block_objective, global_objective, MODELS

class ValidationVerifierTest(unittest.TestCase):
    def test_equal_models_have_exact_scalar_objective(self):
        pair={'point':{m:[1.]*24 for m in MODELS},'actual':[3.]*24}
        anchor=[1/6]*6
        value,gap=block_objective([anchor]*4,[pair],[1.],anchor)
        self.assertAlmostEqual(value,math.sqrt(math.log(2)**2+1e-12),places=12)
        self.assertAlmostEqual(gap,0.,places=12)
        value,gap=global_objective(anchor,[pair]*3)
        self.assertAlmostEqual(value,math.log(2),places=12)
        self.assertAlmostEqual(gap,0.,places=12)

    def test_anchor_penalty_and_gap(self):
        pair={'point':{m:[1.]*24 for m in MODELS},'actual':[1.]*24}
        anchor=[1/6]*6;w=[[1.,0.,0.,0.,0.,0.]]*4
        value,gap=block_objective(w,[pair],[1.],anchor)
        self.assertAlmostEqual(value,.01*5/6+1e-6,places=12)
        self.assertAlmostEqual(gap,.02,places=12)

if __name__=='__main__':unittest.main()
