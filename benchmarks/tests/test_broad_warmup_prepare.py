from datetime import datetime, timedelta
import unittest
from benchmarks.ledger_optimization.broad_warmup_prepare import collect


class Row(dict):
    def __getitem__(self, key):
        if key == 'Hourly_Counts' and self.get('sealed'):
            raise AssertionError('Unselected counts accessed')
        return super().__getitem__(key)


def row(sensor, at, sealed=True):
    return Row(Year=str(at.year), Month=at.strftime('%B'), Mdate=str(at.day),
        Time=str(at.hour), Day=at.strftime('%A'), Date_Time=at.strftime('%B %d, %Y %I:%M:%S %p'),
        Sensor_ID=str(sensor), Sensor_Name='fixed', Hourly_Counts='3', sealed=sealed)


class WarmupTest(unittest.TestCase):
    def test_metadata_does_not_access_any_counts(self):
        start = datetime(2019, 8, 1)
        positions, names, values = collect([row(1, start), row(2, start)], {1}, start, start+timedelta(hours=2))
        self.assertEqual(dict(positions[1]), {0: 1}); self.assertEqual(values, {1: {}})
        self.assertEqual(names, {1: {'fixed'}})

    def test_count_pass_excludes_foreign_identity_and_later_targets(self):
        start = datetime(2019, 8, 1); end = start+timedelta(hours=2)
        rows = [row(1, start, False), row(2, start), row(1, end)]
        _, _, values = collect(rows, {1}, start, end, counts=True)
        self.assertEqual(values, {1: {0: '3'}})

    def test_duplicate_hours_remain_visible(self):
        start = datetime(2019, 8, 1)
        positions, _, _ = collect([row(1, start), row(1, start)], {1}, start, start+timedelta(hours=1))
        self.assertEqual(positions[1][0], 2)


if __name__ == '__main__':
    unittest.main()
