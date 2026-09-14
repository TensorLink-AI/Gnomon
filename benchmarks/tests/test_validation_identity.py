import hashlib
import unittest
from benchmarks.ledger_optimization.validation_identity import choose


class ValidationIdentityTest(unittest.TestCase):
    def fixture(self):
        rows = [{'series_name': f's{i}'} for i in range(40)]
        ordered = sorted(rows, key=lambda r: hashlib.sha256(f'20260914:panel035:electricity:{r["series_name"]}'.encode()).hexdigest())
        return rows, {'development': ordered[:8], 'reserved': ordered[8:24]}, ordered

    def test_exact_disjoint_next_eight(self):
        rows, previous, ordered = self.fixture()
        result = choose('electricity', rows, previous)
        self.assertEqual(result['validation'], ordered[24:32])
        self.assertEqual(choose('electricity', rows[::-1], previous), result)

    def test_partition_change_or_duplicate_rejects(self):
        rows, previous, _ = self.fixture()
        with self.assertRaises(ValueError):choose('electricity', rows+[rows[0]], previous)
        previous['reserved'] = previous['reserved'][::-1]
        with self.assertRaises(ValueError):choose('electricity', rows, previous)

    def test_outcome_fields_and_insufficient_pool_reject(self):
        rows, previous, _ = self.fixture()
        with self.assertRaises(ValueError):choose('electricity', rows[:31], previous)
        rows[0]['future_actuals'] = [1, 2, 3]
        with self.assertRaises(ValueError):choose('electricity', rows, previous)


if __name__ == '__main__':unittest.main()
