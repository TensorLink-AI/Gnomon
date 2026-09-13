import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.hermes_ml_checkpoint_v4.analyze import audit_orchestration
from benchmarks.hermes_ml_native_memory.analyze import native_calls
from benchmarks.ledger_optimization.ml_completed_opportunity import visible_history


class ShutdownAuditTest(unittest.TestCase):
    def test_wire_tool_calls_are_deduplicated_after_termination(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            call={'id':'call-1','function':{'name':'memory','arguments':'{}'}}
            data={'messages':[{'role':'assistant','tool_calls':[call]}]}
            for n in (1,2):
                (root/f'api-{n:02d}-request.json').write_text(json.dumps(data))
            self.assertEqual(native_calls(root)['memory'],1)

    def test_history_requires_prior_origin_and_recorded_visibility(self):
        event={'event':'matured','config_id':'a','forecast_origin':'2026-01-01T00:00:00Z',
               'outcome_recorded_at':'2026-01-15T00:00:00Z','point':[1]*14,
               'actual':[1]*14,'metrics':{'rmsle':0}}
        self.assertEqual(visible_history([event],'2026-01-14T00:00:00Z',{'a'}),{})
        self.assertEqual(len(visible_history([event],'2026-01-15T00:00:00Z',{'a'})),1)
        event['forecast_origin']='2026-01-15T00:00:00Z'
        self.assertEqual(visible_history([event],'2026-01-15T00:00:00Z',{'a'}),{})

    def fixture(self, root):
        for name, data in {'process.json': {'exit_code': -15}, 'command.json': {'timeout': 520},
                           'api-01-request.json': {'messages': [{'role': 'user', 'content': 'task'}]}}.items():
            (root/name).write_text(json.dumps(data))
        return {'exit_code': -15, 'seconds': 532, 'orchestration_stop': 'worker_failed', 'corrections': 0}

    def test_terminated_worker_keeps_checkpoint_separate(self):
        with TemporaryDirectory() as d:
            root=Path(d);row=self.fixture(root);before=dict(row)
            result=audit_orchestration(root,row)
            self.assertFalse(result['checkpoint_scoring_changed'])
            self.assertTrue(result['single_attempt_verified_from_wire'])
            self.assertEqual(row,before)
            self.assertFalse((root/'orchestration.json').exists())

    def test_missing_successful_shutdown_is_not_excused(self):
        with TemporaryDirectory() as d:
            root=Path(d);row=self.fixture(root);row['exit_code']=0
            with self.assertRaises(AssertionError):audit_orchestration(root,row)

    def test_unaccounted_correction_is_rejected(self):
        with TemporaryDirectory() as d:
            root=Path(d);row=self.fixture(root)
            (root/'api-01-request.json').write_text(json.dumps({'messages':[{'role':'user'}, {'role':'user'}]}))
            with self.assertRaises(AssertionError):audit_orchestration(root,row)

    def test_early_termination_is_not_assumed_budget_expiry(self):
        with TemporaryDirectory() as d:
            root=Path(d);row=self.fixture(root);row['seconds']=20
            with self.assertRaises(AssertionError):audit_orchestration(root,row)

    def test_normal_attempt_count_still_checked(self):
        with TemporaryDirectory() as d:
            root=Path(d);row=self.fixture(root)
            (root/'orchestration.json').write_text(json.dumps({'corrections':0,'attempts':1}))
            with self.assertRaises(AssertionError):audit_orchestration(root,row)
            (root/'hermes-attempt-01.json').write_text('{}')
            self.assertEqual(audit_orchestration(root,row)['status'],'shutdown_record_present')


if __name__=='__main__':unittest.main()
