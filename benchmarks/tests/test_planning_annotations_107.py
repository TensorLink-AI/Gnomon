from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from benchmarks.ledger_optimization import planning_annotations_107 as a
from benchmarks.ledger_optimization.planning_sequence_107 import planning_sequence
from benchmarks.tests.test_planning_sequence_107 import inputs
from benchmarks.tests.test_paired_consistency_098 import encode


@contextmanager
def project(review=None, full=None):
    old = Path.cwd()
    with tempfile.TemporaryDirectory() as folder:
        os.chdir(folder)
        try:
            query = (review['query'] if review else
                     {'series_id': 'synthetic', 'unit': 'widgets', 'horizon': 14,
                      'origin': '2026-03-01T00:00:00+00:00'})
            task = dict(query, future_timestamps=['2026-03-02T00:00:00+00:00'])
            for name, value in [('task.json', task), ('backend.json', {'arm': 'ledger'}),
                                ('comparison-state.json', {'query': query, 'review': review})]:
                Path(name).write_text(json.dumps(value))
            Path('history.csv').write_text('timestamp,value\n')
            Path('future.csv').write_text('timestamp\n')
            if review: Path(review['full_evidence']['path']).write_bytes(encode(full))
            core = SimpleNamespace(read=lambda name: json.loads(Path(name).read_text()),
                                   configuration=deepcopy,
                                   request_at=lambda end: ({'cutoff': str(end)}, []))
            yield core
        finally:
            os.chdir(old)


class PlanningAnnotationTests(unittest.TestCase):
    def test_controls_do_not_read_checkpoint_or_history(self):
        for arm in ('plain', 'gnomon'):
            core = SimpleNamespace(read=Mock(return_value={'arm': arm}))
            getter = Mock(side_effect=AssertionError('unexpected checkpoint access'))
            self.assertIsNone(a.annotate_planning(core, 'review', {}, budget={}, checkpoint_getter=getter))
            getter.assert_not_called(); core.read.assert_called_once_with('backend.json')

    def test_commit_and_sync_do_not_emit_more_exploration(self):
        core = SimpleNamespace(read=Mock(side_effect=AssertionError('unexpected state access')))
        for operation in ('commit', 'sync', 'forecast'):
            self.assertIsNone(a.annotate_planning(core, operation, {}, budget={}, checkpoint_getter=Mock()))
        core.read.assert_not_called()

    def test_cold_artifact_is_content_addressed_without_execution_changes(self):
        with project() as core:
            before = Path('comparison-state.json').read_bytes()
            view = a.annotate_planning(core, 'review', {}, budget={'numerical_remaining': 60}, checkpoint_getter=lambda: None)
            self.assertEqual(view['status'], 'no_historical_catalog')
            self.assertIsNone(view['next_call'])
            raw = Path(view['evidence']['path']).read_bytes()
            self.assertEqual(len(raw), view['evidence']['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), view['evidence']['sha256'])
            full = json.loads(raw)
            self.assertEqual(full['execution_log_prefix'], {'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()})
            self.assertIsNone(full['plan']); self.assertIsNone(full['review_reference'])
            self.assertEqual(Path('comparison-state.json').read_bytes(), before)
            self.assertFalse(Path('experiments.jsonl').exists())
            again = a.annotate_planning(core, 'review', {}, budget={'numerical_remaining': 60}, checkpoint_getter=lambda: None)
            self.assertEqual(again, view)
            self.assertEqual(len(Path('planning-annotations.jsonl').read_text().splitlines()), 2)

    def test_existing_artifact_tampering_is_not_overwritten(self):
        with project() as core:
            view = a.annotate_planning(core, 'review', {}, budget={}, checkpoint_getter=lambda: None)
            path = Path(view['evidence']['path']); path.write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError, 'content-addressed evidence changed'):
                a.annotate_planning(core, 'review', {}, budget={}, checkpoint_getter=lambda: None)
            self.assertEqual(path.read_bytes(), b'altered')
            self.assertEqual(len(Path('planning-annotations.jsonl').read_text().splitlines()), 1)

    def test_old_task_review_is_not_reused(self):
        with project() as core:
            task = core.read('task.json'); task['origin'] = '2026-04-01T00:00:00+00:00'
            Path('task.json').write_text(json.dumps(task)); getter = Mock()
            self.assertIsNone(a.annotate_planning(core, 'start', {}, budget={}, checkpoint_getter=getter))
            getter.assert_not_called(); self.assertFalse(Path('planning-annotations.jsonl').exists())

    def test_checkpoint_identity_mismatch_rejected(self):
        with project() as core:
            bad = dict(core.read('task.json'), task_origin='2026-01-01T00:00:00+00:00')
            with self.assertRaisesRegex(ValueError, 'Checkpoint'):
                a.annotate_planning(core, 'review', {}, budget={}, checkpoint_getter=lambda: bad)

    def test_warm_compact_view_preserves_conditional_calls_and_full_context(self):
        b, f, c, _ = inputs()
        expected = planning_sequence(b, encode(f), c, deepcopy)
        with project(b, f) as core, patch.object(a, 'current_runs', return_value={'runs': []}) as validate:
            view = a.annotate_planning(core, 'review', b, budget=c['budget'], checkpoint_getter=lambda: None)
            artifact = json.loads(Path(view['evidence']['path']).read_bytes())
            self.assertEqual(artifact['plan'], expected)
            self.assertEqual(view['next_call'], expected['next_call'])
            self.assertTrue(view['optional']); self.assertTrue(view['shared_budget']['capacity_is_conditional'])
            self.assertEqual(view['shared_budget']['recipe_capacity'], 1)
            for brief, detail in zip(view['recipes'], expected['recipes'], strict=True):
                self.assertEqual(brief['arguments'], detail['next_call']['arguments'])
                self.assertFalse(brief['admissible_now'])
                self.assertTrue(brief['conditional_budget_feasible'])
            self.assertEqual(artifact['review_reference'], b['full_evidence'])
            self.assertEqual(view['additional_ledger_queries'], 0)
            validate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
