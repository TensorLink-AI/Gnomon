"""Mutation probes for independent reminder evidence and request accounting."""
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.ledger_optimization.workflow_audit_097 import audit_workflow_progress
from benchmarks.ledger_optimization.workflow_progress_097 import workflow_progress


class WorkflowAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name); self.work = self.folder/'project'; self.work.mkdir()
        times = [(datetime(2020, 1, 1)+timedelta(days=i)).isoformat() for i in range(730)]
        self.job = {'origin': times[-1], 'request': {'timestamps': times, 'past_covariate_names': []}}
        (self.work/'task.json').write_text(json.dumps({'origin': times[-1]}))
        with (self.work/'history.csv').open('w') as stream:
            writer = csv.writer(stream); writer.writerow(['timestamp', 'value'])
            writer.writerows((t, 1) for t in times)
        self.original = {'messages': [{'role': 'user', 'content': 'original task'}], 'stream': True}
        self.write('api-04-request.json', self.original)
        self.write('api-04-workflow-state.json', {'request_number': 4, 'seconds_remaining': 200})
        self.progress = workflow_progress(self.work, 4, 200)
        self.forward()

    def write(self, name, value):
        (self.folder/name).write_text(json.dumps(value))

    def forward(self):
        notice = json.dumps(self.progress, sort_keys=True, separators=(',', ':'))
        self.write('api-04-workflow-progress.json', {'progress': self.progress, 'notice': notice})
        self.write('api-04-forwarded.json', {**self.original,
            'messages': [*self.original['messages'], {'role': 'system', 'content': notice}],
            'model': 'deepseek-v4.1-flash', 'temperature': .2, 'seed': 7, 'max_tokens': 3072, 'stream': False})

    def audit(self):
        return audit_workflow_progress(self.folder, self.job)

    def test_valid_empty_prefix_and_later_appends(self):
        self.assertEqual(self.audit()['reminder_requests'], [4])
        event = {'event': 'attempt', 'task_origin': self.job['origin']}
        (self.work/'experiments.jsonl').write_text(json.dumps(event)+'\n')
        # The notice must still reflect its originally empty prefix, not later events.
        self.assertEqual(self.audit()['audited_prefixes'], 1)

    def test_false_completion_claim_rejected(self):
        self.progress['complete_current_configurations'] = 2; self.forward()
        with self.assertRaises(AssertionError): self.audit()

    def test_budget_inflation_rejected(self):
        self.progress['total_requests_including_this_one'] = 50; self.forward()
        with self.assertRaises(AssertionError): self.audit()

    def test_original_task_mutation_rejected(self):
        p = self.folder/'api-04-forwarded.json'; value = json.loads(p.read_text())
        value['messages'][0]['content'] = 'different task'; self.write(p.name, value)
        with self.assertRaises(AssertionError): self.audit()

    def test_outcome_field_rejected_even_when_receipt_agrees(self):
        self.progress['future_actuals'] = [99]*14; self.forward()
        with self.assertRaises(AssertionError): self.audit()

    def test_prefix_hash_cannot_be_forged_by_changing_recorded_length(self):
        (self.work/'experiments.jsonl').write_text('{}\n')
        self.progress['evidence_prefix_bytes'] = 3; self.forward()
        with self.assertRaises(AssertionError): self.audit()

    def test_protected_phase_cannot_get_exploration_reminder(self):
        self.write('api-04-workflow-state.json', {'request_number': 4, 'seconds_remaining': 80})
        with self.assertRaises(AssertionError): self.audit()

    def test_orphan_reminder_rejected(self):
        self.write('api-08-workflow-progress.json', {'extra': True})
        with self.assertRaises(AssertionError): self.audit()


if __name__ == '__main__':
    unittest.main()
