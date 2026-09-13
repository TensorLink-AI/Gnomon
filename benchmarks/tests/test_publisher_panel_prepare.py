from datetime import timedelta
import unittest
from benchmarks.ledger_optimization.publisher_panel_prepare import selected_counts, START


class Row(dict):
    def __getitem__(self, key):
        if key == 'Hourly_Counts' and self.get('sealed'):
            raise AssertionError('Reserved/unselected count accessed')
        return super().__getitem__(key)


def row(sensor, offset, sealed=False):
    at = START + timedelta(hours=offset)
    return Row(Year=str(at.year), Month=at.strftime('%B'), Mdate=str(at.day),
               Time=str(at.hour), Day=at.strftime('%A'),
               Date_Time=at.strftime('%B %d, %Y %I:%M:%S %p'),
               Sensor_ID=str(sensor), Hourly_Counts='2', sealed=sealed)


class PublisherPreparationTest(unittest.TestCase):
    def test_only_selected_prefix_counts_accessed(self):
        rows = [row(1, 0), row(1, 1), row(1, 2, True), row(2, 0, True)]
        self.assertEqual(selected_counts(rows, [1], 2), {1: ['2', '2']})

    def test_missing_and_duplicate_reject(self):
        for rows in ([row(1, 0)], [row(1, 0), row(1, 0), row(1, 1)]):
            with self.assertRaises(ValueError):
                selected_counts(rows, [1], 2)


if __name__ == '__main__':
    unittest.main()
