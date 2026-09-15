"""Historical four-arm experiment tests; not the active three-arm protocol."""
from copy import deepcopy
import unittest

from benchmarks.ledger_optimization import m5_ml_reference_analysis as a
from benchmarks.tests.test_m5_ml_analysis import fixture

IDENTITY={'label':'synthetic-prior-ledger','policy_sha256':'a'*64,'runtime_version':'1.2.0'}


def reference_fixture(value=.9):
    panel,seeds,rows=fixture()
    rows += [{**r,'arm':'ledger_reference','rmsle':value} for r in rows if r['arm']=='ledger']
    return panel,seeds,rows


class ReferenceAnalysisTests(unittest.TestCase):
    def test_full_objective_requires_both_comparisons(self):
        p,s,rows=reference_fixture(.6)
        result=a.analyze(p,s,rows,reference_identity=IDENTITY)
        self.assertTrue(result['numerical_primary_criteria_met'])
        self.assertFalse(result['numerical_all_objective_criteria_met'])
        self.assertFalse(result['contrasts']['ledger_vs_ledger_reference']['point_improvement_met'])
        self.assertFalse(result['target_established'])
        self.assertFalse(result['active_protocol_eligible'])
        self.assertEqual(result['decisions'],4992)
        self.assertEqual(result['matched_cases_per_arm'],1248)

    def test_reference_improvement_and_shared_paired_interval(self):
        p,s,rows=reference_fixture(.9)
        result=a.analyze(p,s,rows,reference_identity=IDENTITY)
        ref=result['contrasts']['ledger_vs_ledger_reference']
        self.assertTrue(result['numerical_all_objective_criteria_met'])
        self.assertAlmostEqual(ref['relative_rmsle_reduction'],1-.7/.9)
        self.assertEqual(ref['paired_95_interval'],[ref['relative_rmsle_reduction']]*2)
        self.assertFalse(result['target_established'])
        self.assertFalse(result['active_protocol_eligible'])
        # Reference identity belongs to the hashed contract, even when scores match.
        identity={**IDENTITY,'policy_sha256':'b'*64}
        other=a.analyze(p,s,rows,reference_identity=identity)
        self.assertNotEqual(result['analysis_input_sha256'],other['analysis_input_sha256'])
        self.assertEqual(result['numerical_input_sha256'],other['numerical_input_sha256'])

    def test_absent_duplicate_or_malformed_reference_cannot_pass(self):
        p,s,rows=reference_fixture()
        for bad in (rows[:-1],rows+[rows[-1]],rows[:3744]):
            with self.assertRaises(ValueError):a.analyze(p,s,bad,reference_identity=IDENTITY)
        bad=deepcopy(rows);bad[-1]['valid']=1
        with self.assertRaises(ValueError):a.analyze(p,s,bad,reference_identity=IDENTITY)

    def test_zero_reference_cannot_establish_improvement(self):
        p,s,rows=reference_fixture(0)
        result=a.analyze(p,s,rows,reference_identity=IDENTITY)
        reference=result['contrasts']['ledger_vs_ledger_reference']
        self.assertIsNone(reference['paired_95_interval'])
        self.assertFalse(reference['point_improvement_met'])
        self.assertFalse(result['numerical_all_objective_criteria_met'])

    def test_identity_must_be_explicit(self):
        p,s,rows=reference_fixture()
        for identity in ({},{**IDENTITY,'runtime_version':'1.1.9'},{**IDENTITY,'policy_sha256':'unknown'}):
            with self.assertRaises(ValueError):a.analyze(p,s,rows,reference_identity=identity)


if __name__=='__main__':unittest.main()
