import unittest
import numpy as np
from benchmarks.ledger_optimization.guarded_correction_verify import split_mask

class GuardedTreeAuditTests(unittest.TestCase):
    def test_double_threshold_is_not_rounded_to_float32(self):
        values=np.array([1.],dtype=np.float32);threshold=float(np.nextafter(1.,0.))
        self.assertFalse(bool(split_mask(values,threshold)[0]))
        self.assertFalse(bool(split_mask(values[0],threshold)))
    def test_equal_threshold_and_both_sides(self):
        values=np.array([.5,1.,1.5],dtype=np.float32)
        np.testing.assert_array_equal(split_mask(values,1.),[True,True,False])

if __name__=='__main__':unittest.main()
