from datetime import datetime
import hashlib
import json
import unittest
from benchmarks.ledger_optimization.validation_source import split_span


class ValidationSourceTest(unittest.TestCase):
    def fixture(self):
        values = [str(i) for i in range(6298)]
        prefix = hashlib.sha256(json.dumps([float(i) for i in range(1344, 2074)], separators=(',', ':')).encode()).hexdigest()
        return values, prefix

    def test_missing_earlier_history_preserved_with_full_scored_span(self):
        values, prefix = self.fixture(); values[:50] = [None]*50
        scored, warm = split_span('x', values, datetime(2019, 1, 1), 'sha', 'unit', prefix)
        self.assertEqual(len(scored['values']), 4954); self.assertEqual(len(warm['values']), 2074)
        self.assertEqual(warm['values'][:50], [None]*50)
        self.assertEqual(warm['values'][-730:], scored['values'][:730])

    def test_missing_scored_value_or_changed_prefix_rejects(self):
        values, prefix = self.fixture(); values[3000] = None
        with self.assertRaises(ValueError):split_span('x', values, datetime(2019, 1, 1), 'sha', 'unit', prefix)
        values, prefix = self.fixture(); values[1344] = '9999'
        with self.assertRaises(ValueError):split_span('x', values, datetime(2019, 1, 1), 'sha', 'unit', prefix)


if __name__ == '__main__':unittest.main()
