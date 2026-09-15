"""An advisory reminder must not choose a model, reveal targets or grant budget."""
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.ledger_optimization.workflow_progress_097 import workflow_progress


class WorkflowProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.times = [(datetime(2020, 1, 1)+timedelta(days=i)).isoformat() for i in range(730)]
        (self.root/'task.json').write_text(json.dumps({'origin': self.times[-1]}))
        with (self.root/'history.csv').open('w') as stream:
            writer = csv.writer(stream); writer.writerow(['timestamp', 'value', 'onpromotion'])
            writer.writerows((t, 1, 0) for t in self.times)
        self.events = []

    def result(self, number=4, seconds=300):
        (self.root/'experiments.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in self.events))
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        result = workflow_progress(self.root, number, seconds)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})
        return result

    def backtest(self, cid, model, indices=(688, 702, 716), origin=None):
        for i in indices:
            self.events.append({'task_origin': origin or self.times[-1], 'event': 'result',
                'kind': 'backtest', 'config_id': cid, 'config': {'model': model},
                'request': {'cutoff': self.times[i-1]}, 'actual': ['HIDDEN_SENTINEL'],
                'point': ['HIDDEN_SENTINEL'], 'metrics': {'rmsle': 'HIDDEN_SENTINEL'}})

    def test_no_change_to_budgets_or_state(self):
        result = self.result(8)
        self.assertEqual(result['exploration_requests_including_this_one'], 5)
        self.assertEqual(result['total_requests_including_this_one'], 9)
        self.assertEqual(result['extra_requests_granted'], 0)
        self.assertEqual(result['fits_executed_by_reminder'], 0)
        self.assertIn('operation=start', result['next_step'])
        self.assertEqual(result['target_column'], 'value')

    def test_never_reopens_protected_phase(self):
        for number in (1, 3, 12, 13, 16, 17):
            self.assertIsNone(self.result(number))
        for seconds in (-1, 0, 89, 90):
            self.assertIsNone(self.result(8, seconds))

    def test_previous_origin_and_incomplete_folds_do_not_count(self):
        self.backtest('old', 'ridge', origin=self.times[-15])
        self.backtest('short', 'ridge', (688, 702))
        result = self.result()
        self.assertEqual(result['complete_current_configurations'], 0)
        self.assertFalse(result['ml_configuration_compared'])

    def test_baseline_prompts_agent_choice_without_settings(self):
        self.backtest('base', 'seasonal')
        result = self.result()
        self.assertIn('operation=backtest', result['next_step'])
        self.assertNotIn('alpha', json.dumps(result))
        self.assertNotIn('HIDDEN_SENTINEL', json.dumps(result))

    def test_three_wrong_origins_are_not_a_complete_config(self):
        self.backtest('wrong', 'ridge', (680, 695, 710))
        self.assertEqual(self.result()['complete_current_configurations'], 0)

    def test_selection_required_after_comparison(self):
        self.backtest('base', 'seasonal'); self.backtest('ml', 'ridge')
        result = self.result()
        self.assertEqual(result['complete_current_configurations'], 2)
        self.assertIn('operation=commit', result['next_step'])
        self.assertNotIn('HIDDEN_SENTINEL', json.dumps(result))
        self.events.append({'event': 'selection', 'task_origin': self.times[-1],
                            'selection_after_comparison': True})
        self.assertIn('optional', self.result()['next_step'])

    def test_last_selection_controls_reported_progress(self):
        self.backtest('base', 'seasonal'); self.backtest('ml', 'ridge')
        for value in (True, False):
            self.events.append({'event': 'selection', 'task_origin': self.times[-1],
                                'selection_after_comparison': value})
        self.assertIn('operation=commit', self.result()['next_step'])

    def test_malformed_state_produces_no_fabricated_success(self):
        (self.root/'task.json').write_text('invalid')
        result = self.result()
        self.assertFalse(result['progress_available'])
        self.assertIn('operation=status', result['next_step'])


if __name__ == '__main__':
    unittest.main()
