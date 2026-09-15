from copy import deepcopy
import unittest

from benchmarks.ledger_optimization.m5_ml_prefix_identity import check_prefix_identity


class PrefixIdentityTests(unittest.TestCase):
    def fixture(self, seed=19):
        capsule = {'requested_seed': seed, 'sources': {'worker.py':'worker-digest','numerical.py':'numerical-digest'}}
        runtime = {'gnomon': {'package_version':'1.2.0'}, 'plain': {'gnomon':False}}
        manifest = {'requested_seed': seed, 'sources': deepcopy(capsule['sources']), 'inventory':deepcopy(runtime)}
        return manifest,capsule,runtime

    def test_matching_identity_does_not_grant_dispatch(self):
        for seed in (7,19):
            out=check_prefix_identity(*self.fixture(seed))
            self.assertTrue(out['seed_source_runtime_checks_passed'])
            self.assertFalse(out['operational_gate_passed'])
            self.assertFalse(out['execution_authorized'])
            self.assertFalse(out['final_gate_opened'])

    def test_wrong_seed_rejects_even_with_matching_source_map(self):
        for seed in (7,True,None):
            args=self.fixture();args[0]['requested_seed']=seed
            with self.subTest(seed=seed),self.assertRaisesRegex(ValueError,'seed'):check_prefix_identity(*args)

    def test_wrong_runtime_or_source_rejects(self):
        for field,value in [('sources',{}),('sources',{'worker.py':'changed'}),('inventory',{}),
                            ('inventory',{'gnomon':{'package_version':'1.1.9'}})]:
            args=self.fixture();args[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):check_prefix_identity(*args)


if __name__=='__main__':unittest.main()
