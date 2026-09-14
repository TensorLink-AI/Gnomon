import unittest
from benchmarks.ledger_optimization.blend_error_geometry import geometry


class GeometryTest(unittest.TestCase):
    def test_inside_floor_zero_but_actual_blend_can_err(self):
        r=geometry([[1.,1.],[3.,3.]],[2.,2.],[1.,1.])
        self.assertEqual(r['range_floor_rmsle'],0.);self.assertEqual(r['position'],['inside']*2)
        self.assertGreater(r['blending_energy'],0.);self.assertEqual(r['cross_energy'],0.)
    def test_outside_decomposition_includes_positive_cross_term(self):
        r=geometry([[2.,2.],[3.,3.]],[1.,4.],[3.,2.])
        self.assertEqual(r['position'],['below','above']);self.assertGreater(r['cross_energy'],0.)
        self.assertAlmostEqual(r['total_energy'],r['range_energy']+r['blending_energy']+r['cross_energy'])
    def test_degenerate_and_boundary_ranges(self):
        r=geometry([[2.,2.]],[2.,3.],[2.,2.])
        self.assertEqual(r['position'],['inside','above']);self.assertEqual(r['blending_energy'],0.)
    def test_invalid_or_outside_blends_rejected(self):
        for args in (([[1.]],[],None),([[1.]],[float('nan')],None),([[1.],[2.]],[1.],[3.])):
            with self.assertRaises(ValueError):geometry(*args)


if __name__=='__main__':unittest.main()
