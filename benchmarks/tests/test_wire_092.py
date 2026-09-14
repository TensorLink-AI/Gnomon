"""Cost-evidence regressions for successful and failed 092 attempts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.ledger_optimization.audits.wire_092 import audit


class WireAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'session').mkdir()
        self.write('snapshot-manifest.json', {'completed_session_folders': ['session']})

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def attempt(self, index, status, usage=None):
        prefix = f'session/api-{index:02}'
        self.write(prefix + '-forwarded.json',
                   {'model': 'deepseek-v4.1-flash', 'max_tokens': 3072})
        self.write(prefix + '-receipt.json', {'status': status})
        response = ({'model': 'deepseek-v4.1-flash'} if status == 200
                    else {'error': {'message': 'URLError'}})
        if usage is not None:
            response['usage'] = usage
        self.write(prefix + '-response.json', response)

    def grade(self, calls, tokens, complete=True):
        self.write('session/grade.json', {'api_calls': calls, 'tokens': tokens,
                   'workflow_complete': complete, 'valid': complete})

    def test_transport_failure_consumes_budget_without_zero_cost_assumption(self):
        self.attempt(0, 200, {'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7})
        self.attempt(1, 502)
        self.grade(2, 7)
        result = audit(self.root)
        self.assertEqual(result['forwarded_attempts'], 2)
        self.assertEqual(result['failed_responses'], 1)
        self.assertEqual(result['reported_total_tokens'], 7)
        self.assertEqual(result['attempts_without_usage'], 1)
        self.assertFalse(result['usage_complete'])
        self.assertTrue(result['sessions'][0]['workflow_complete'])

    def test_failed_session_and_reported_failure_usage_are_retained(self):
        self.attempt(0, 500, {'prompt_tokens': 3, 'completion_tokens': 0, 'total_tokens': 3})
        self.grade(1, 3, complete=False)
        result = audit(self.root)
        self.assertEqual(result['reported_total_tokens'], 3)
        self.assertTrue(result['usage_complete'])
        self.assertFalse(result['sessions'][0]['valid_forecast'])
        self.assertEqual(result['successful_responses'], 0)

    def test_missing_usage_on_success_is_also_unknown(self):
        self.attempt(0, 200)
        self.grade(1, 0)
        self.assertFalse(audit(self.root)['usage_complete'])

    def test_missing_receipt_cannot_hide_an_attempt(self):
        self.attempt(0, 502)
        self.grade(1, 0)
        (self.root / 'session/api-00-receipt.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Every forwarded attempt'):
            audit(self.root)

    def test_grade_cannot_drop_a_failed_attempt(self):
        self.attempt(0, 502)
        self.grade(0, 0)
        with self.assertRaisesRegex(ValueError, 'attempt count'):
            audit(self.root)

    def test_inconsistent_usage_is_rejected(self):
        self.attempt(0, 200, {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 4})
        self.grade(1, 4)
        with self.assertRaisesRegex(ValueError, 'Usage arithmetic'):
            audit(self.root)


if __name__ == '__main__':
    unittest.main()
