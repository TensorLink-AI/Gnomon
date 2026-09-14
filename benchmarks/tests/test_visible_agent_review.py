import copy
import unittest
from benchmarks.ledger_optimization.visible_agent_review import visible_records


class Ledger:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def execution(self, eid):
        self.calls += 1
        if eid != self.value['execution_id']:
            raise KeyError(eid)
        return copy.deepcopy(self.value)


class VisibilityTest(unittest.TestCase):
    def setUp(self):
        self.task = {'origin': '2026-01-10T00:00:00+00:00'}
        request = {'history': [1, 2], 'horizon': 1, 'series_id': 's', 'unit': 'widgets',
                   'cutoff': '2026-01-02T00:00:00+00:00',
                   'timestamps': ['2026-01-01T00:00:00+00:00', '2026-01-02T00:00:00+00:00'],
                   'future_timestamps': ['2026-01-03T00:00:00+00:00']}
        self.execution = {'execution_id': 'e', 'recorded_at': self.task['origin'],
                          'provider': 'p', 'revision': 'r', 'fingerprint': 'fp',
                          'request': request, 'result': {'point': [2]}}
        self.row = {'event': 'result', 'kind': 'forecast', 'request': copy.deepcopy(request),
                    'execution': copy.deepcopy(self.execution), 'config_id': 'c', 'config': {}}
        self.db = Ledger(self.execution)

    def test_inclusive_cutoff_and_no_input_mutation(self):
        original = copy.deepcopy(self.row)
        rows, excluded, reads = visible_records(self.db, [self.row], self.task)
        self.assertEqual(rows[0]['execution'], self.execution)
        self.assertEqual(excluded, [])
        self.assertEqual(reads, 1)
        rows[0]['request']['history'][0] = 99
        self.assertEqual(self.row, original)

    def test_future_recording_uses_database_not_envelope(self):
        self.execution['recorded_at'] = '2026-01-10T00:00:00.000001+00:00'
        rows, excluded, _ = visible_records(self.db, [self.row], self.task)
        self.assertEqual(rows, [])
        self.assertEqual(excluded[0]['reason'], 'execution_not_recorded_by_query')

    def test_duplicate_reference_read_once(self):
        rows, _, reads = visible_records(self.db, [self.row, self.row], self.task)
        self.assertEqual(len(rows), 2)
        self.assertEqual(reads, 1)
        self.assertEqual(self.db.calls, 1)

    def test_unknown_reference_rejected(self):
        self.row['execution']['execution_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'unverifiable'):
            visible_records(self.db, [self.row], self.task)

    def test_tampered_envelope_rejected(self):
        for key, value in [('provider', 'other'), ('revision', 'r2'), ('fingerprint', 'other'),
                           ('request', {}), ('result', {'point': [99]})]:
            row = copy.deepcopy(self.row)
            row['execution'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                visible_records(self.db, [row], self.task)

    def test_tampered_event_task_rejected(self):
        for key, value in [('series_id', 'other'), ('unit', 'kg'), ('horizon', 2), ('history', [9]),
                           ('cutoff', '2026-01-01T00:00:00+00:00'),
                           ('timestamps', []), ('future_timestamps', [])]:
            row = copy.deepcopy(self.row)
            row['request'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                visible_records(self.db, [row], self.task)

    def test_equivalent_explicit_timezone_is_preserved(self):
        self.row['request']['cutoff'] = '2026-01-02T10:00:00+10:00'
        rows, _, _ = visible_records(self.db, [self.row], self.task)
        self.assertEqual(rows[0]['request']['cutoff'], self.execution['request']['cutoff'])

    def test_naive_recording_time_rejected(self):
        self.execution['recorded_at'] = '2026-01-10T00:00:00'
        with self.assertRaisesRegex(ValueError, 'timezone'):
            visible_records(self.db, [self.row], self.task)

    def test_nonproduction_events_not_discovered(self):
        self.row['kind'] = 'backtest'
        rows, excluded, reads = visible_records(self.db, [self.row], self.task)
        self.assertEqual((rows, excluded, reads), ([], [], 0))


if __name__ == '__main__':
    unittest.main()
