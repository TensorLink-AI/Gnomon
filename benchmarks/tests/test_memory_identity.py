import hashlib
import unittest
from copy import deepcopy
from benchmarks.ledger_optimization.memory_identity import choose

class MemoryIdentityTest(unittest.TestCase):
    def fixture(self):
        rows=[{'series_name':str(i)} for i in range(43)]
        ordered=sorted(rows,key=lambda r:hashlib.sha256(f'20260914:panel035:electricity:{r["series_name"]}'.encode()).hexdigest())
        return rows,{'development':ordered[:8],'reserved':ordered[8:24]},{'validation':ordered[24:32]},ordered
    def test_exact_next_block_disjoint(self):
        rows,old,valid,ordered=self.fixture();out=choose('electricity',rows,old,valid)
        self.assertEqual(out['memory_training'],ordered[32:40]);self.assertEqual(len(out['reserved_final_unchanged']),16)
    def test_protected_cohort_change_rejected(self):
        rows,old,valid,_=self.fixture();valid['validation']=list(reversed(valid['validation']))
        with self.assertRaises(ValueError):choose('electricity',rows,old,valid)
    def test_too_few_or_outcome_fields_rejected(self):
        rows,old,valid,_=self.fixture()
        with self.assertRaises(ValueError):choose('electricity',rows[:39],old,valid)
        rows=deepcopy(rows);rows[0]['actual']=[999]
        with self.assertRaises(ValueError):choose('electricity',rows,old,valid)

if __name__=='__main__':unittest.main()
