from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization.collection_capsule_096 import build
from benchmarks.ledger_optimization.seed_capsule_093 import SOURCE


class CollectionCapsuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name) / 'capsule'
        self.manifest = build(root)
        self.events, self.calls = [], []
        self.phase, self.seconds, self.fail_end = 'exploration', 100, None
        self.padding = 0
        self.selected = {'original': 'checkpoint'}
        self.config = {'model': 'ridge', 'alpha': 10.0}

        def cid(config):
            return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]

        def request_at(end):
            return {'cutoff': str(end), 'horizon': 14, 'series_id': 'synthetic', 'unit': 'widgets',
                    'future_timestamps': [f'{end}+{i}' for i in range(14)]}, None if end == 730 else [0.] * 14

        def execute(config, end, kind):
            request, actual = request_at(end)
            self.calls.append((end, kind))
            self.events.append({'event': 'attempt', 'task_origin': '730'})
            if end == self.fail_end:
                raise RuntimeError('synthetic fit failure')
            point = [1.] * 14
            row = {'event': 'result', 'task_origin': '730', 'config': deepcopy(config),
                   'config_id': cid(config), 'request': request, 'kind': kind, 'point': point,
                   'actual': actual, 'metrics': {'rmsle': 1.} if kind == 'backtest' else None,
                   'execution': {'provider': config['model'] + '_' + cid(config),
                                 'revision': 'ml-lab-v1:' + cid(config), 'execution_id': str(len(self.calls)),
                                 'result': {'point': point}}}
            self.events.append(row)
            return row

        core = SimpleNamespace(read=lambda name: {'origin': '730'}, logs=lambda: self.events,
                               append=lambda name, row: self.events.append(row), execute=execute, request_at=request_at)
        numerical = SimpleNamespace(configuration=deepcopy, config_id=cid)
        path = root / 'benchmarks/hermes_ml_checkpoint_v6/lab.py'
        spec = importlib.util.spec_from_file_location('collection_test_lab', path)
        self.lab = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'core': core, 'numerical': numerical,
                                    'policy': SimpleNamespace(phase=lambda *a: self.phase, EXPLORATION_REQUESTS=12)}):
            spec.loader.exec_module(self.lab)
        self.lab.budget = lambda: {
            'phase': self.phase, 'seconds_remaining_approx': self.seconds,
            'numerical_remaining': 60 - self.padding - len(self.calls),
            'numerical_attempts': self.padding + len(self.calls)}
        self.lab.checkpoint = lambda: self.selected

        def publish(row, reason):
            self.selected = row['execution']['execution_id']
            return {'execution_id': self.selected}
        self.lab.publish = publish

    def test_only_lab_changes_and_source_is_immutable(self):
        self.assertEqual(self.manifest['changed_files'], ['lab.py'])
        self.assertEqual(len(self.manifest['sources']), 24)
        for name, digest in self.manifest['base_sources'].items():
            self.assertEqual(hashlib.sha256((SOURCE / name).read_bytes()).hexdigest(), digest)
        with self.assertRaises(ValueError):
            build(Path(self.tmp.name) / 'capsule')

    def test_four_fits_then_reuse_without_implicit_selection(self):
        before = deepcopy(self.selected)
        r = self.lab.backtest(self.config)
        self.assertEqual(self.calls, [(688, 'backtest'), (702, 'backtest'), (716, 'backtest'), (730, 'forecast')])
        self.assertEqual(r['collection']['new_fits'], 4)
        self.assertEqual(self.selected, before)
        again = self.lab.backtest(self.config)
        self.assertEqual(len(self.calls), 4)
        self.assertTrue(again['reused'])
        self.assertEqual(again['collection']['execution_id'], r['collection']['execution_id'])
        committed = self.lab.commit(self.config)
        self.assertEqual(committed['execution_id'], r['collection']['execution_id'])
        self.assertEqual(len(self.calls), 4)

    def test_production_failure_retries_only_missing_fit(self):
        self.fail_end = 730
        with self.assertRaises(self.lab.Rejected) as raised:
            self.lab.backtest(self.config)
        self.assertEqual(raised.exception.code, 'COLLECTION_FIT_FAILED')
        self.assertEqual(len(self.lab.tested()), 1)
        self.assertEqual(self.selected, {'original': 'checkpoint'})
        self.assertEqual(len(self.calls), 4)  # Failed fit counted.
        self.fail_end = None
        r = self.lab.backtest(self.config)
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(r['collection']['new_fits'], 1)
        self.assertEqual(r['collection']['reused_fits'], 3)

    def test_cv_failure_preserves_and_reuses_completed_fold(self):
        self.fail_end = 702
        with self.assertRaises(self.lab.Rejected):
            self.lab.backtest(self.config)
        self.assertEqual(len(self.calls), 2)
        self.fail_end = None
        result = self.lab.backtest(self.config)
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(result['collection']['reused_fits'], 1)
        self.assertEqual(self.calls.count((688, 'backtest')), 1)

    def test_insufficient_budget_or_selection_starts_zero_fits(self):
        self.padding = 56  # Four remaining, but candidate requires one final reserve.
        with self.assertRaises(self.lab.Rejected) as raised:
            self.lab.backtest(self.config)
        self.assertEqual(raised.exception.code, 'NUMERICAL_BUDGET_RESERVED')
        self.assertEqual(self.calls, [])
        self.padding = 0
        self.phase = 'selection'
        with self.assertRaises(self.lab.Rejected):
            self.lab.backtest(self.config)
        self.assertEqual(self.calls, [])

    def test_deadline_between_fits_and_resume(self):
        original = self.lab.core.execute
        def execute(*args):
            row = original(*args)
            self.seconds = 0
            return row
        self.lab.core.execute = execute
        with self.assertRaises(self.lab.Rejected) as raised:
            self.lab.backtest(self.config)
        self.assertEqual(raised.exception.code, 'COLLECTION_DEADLINE')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.selected, {'original': 'checkpoint'})
        # A real expired session cannot reset time; this is a synthetic recovery
        # scenario exercising reuse, not permission to extend a production run.
        self.seconds = 100
        self.lab.core.execute = original
        r = self.lab.backtest(self.config)
        self.assertEqual(r['collection']['new_fits'], 3)

    def test_reused_identity_and_no_future_actuals(self):
        self.lab.backtest(self.config)
        row = next(r for r in self.events if r['event'] == 'result' and r['kind'] == 'forecast')
        for field, bad in [('request', {'series_id': 'wrong'}), ('actual', [1.] * 14)]:
            original = deepcopy(row[field])
            row[field] = bad
            with self.assertRaises(self.lab.Rejected):
                self.lab.backtest(self.config)
            self.assertEqual(len(self.calls), 4)
            row[field] = original
        row['execution']['revision'] = 'different'
        with self.assertRaises(self.lab.Rejected):
            self.lab.backtest(self.config)

    def test_new_invalid_result_is_not_reported_complete(self):
        original = self.lab.core.execute
        def execute(*args):
            row = original(*args)
            if row['kind'] == 'forecast':
                row['point'] = [float('nan')] * 14
                row['execution']['result']['point'] = row['point']
            return row
        self.lab.core.execute = execute
        with self.assertRaises(self.lab.Rejected) as raised:
            self.lab.backtest(self.config)
        self.assertEqual(raised.exception.details['cause_code'], 'COLLECTION_IDENTITY_MISMATCH')
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.selected, {'original': 'checkpoint'})

    def test_initial_baseline_can_use_four_remaining_fits(self):
        self.padding = 56
        r = self.lab.backtest(self.config, initial_baseline=True)
        self.assertEqual(r['collection']['new_fits'], 4)
        self.assertEqual(self.lab.budget()['numerical_remaining'], 0)
        self.assertEqual(self.selected, {'original': 'checkpoint'})


if __name__ == '__main__':
    unittest.main()
