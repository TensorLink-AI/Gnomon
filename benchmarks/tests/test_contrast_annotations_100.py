from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from benchmarks.ledger_optimization.contrast_annotations_100 import annotate
from benchmarks.tests.test_contrast_records_100 import fixture


class ContrastAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.previous = Path.cwd(); os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.previous)
        self.task, self.rows, self.requests = fixture()
        self.arm = 'plain'
        self.write()
        def read(name): return json.loads(Path(name).read_text())
        def request_at(end): return self.requests[(688, 702, 716).index(end)], [2., 2.]
        self.core = SimpleNamespace(read=read, request_at=request_at, configuration=deepcopy)

    def write(self):
        Path('task.json').write_text(json.dumps(self.task))
        Path('backend.json').write_text(json.dumps({'arm': self.arm}))
        Path('history.csv').write_text('timestamp,value\n')
        Path('future.csv').write_text('timestamp\n')
        Path('experiments.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))

    def test_artifact_exists_and_prefix_is_unchanged_before_return(self):
        prefix = Path('experiments.jsonl').read_bytes()
        result = annotate(self.core, 'start', {})
        self.assertEqual(len(result['current_cv']['configurations']), 2)
        self.assertEqual(result['comparison_status'], 'available')
        ref = result['comparison']['evidence']
        raw = Path(ref['path']).read_bytes()
        self.assertEqual(len(raw), ref['bytes'])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), ref['sha256'])
        receipt = json.loads(Path('comparison-annotations.jsonl').read_text())
        self.assertEqual(receipt['annotation'], result)
        self.assertEqual(receipt['execution_log_prefix'], {'bytes': len(prefix), 'sha256': hashlib.sha256(prefix).hexdigest()})
        self.assertEqual(Path('experiments.jsonl').read_bytes(), prefix)
        self.assertEqual(result['annotation_diagnostics']['provider_calls'], 0)
        self.assertEqual(result['annotation_diagnostics']['additional_ledger_queries'], 0)

    def test_single_baseline_scores_are_visible_without_creating_a_pair(self):
        self.rows = self.rows[::2]; self.write()
        out = annotate(self.core, 'start', {})
        self.assertEqual(len(out['current_cv']['configurations']), 1)
        self.assertEqual(len(out['current_cv']['configurations'][0]['fold_rmsle']), 3)
        self.assertEqual(out['comparison_status'], 'fewer_than_two_complete_current_configurations')
        self.assertNotIn('comparison', out)

    def test_old_task_cache_never_becomes_current_history(self):
        self.arm = 'ledger'; self.write()
        Path('comparison-state.json').write_text(json.dumps({'query': {'origin': 'old'}, 'review': {'invalid': True}}))
        out = annotate(self.core, 'status', {})
        self.assertEqual(out['comparison']['history_status'], 'not_requested')

    def test_completed_cold_review_is_distinct_from_not_requesting_history(self):
        self.arm = 'ledger'; self.write()
        out = annotate(self.core, 'review', {'status': 'insufficient_evidence'})
        self.assertEqual(out['comparison']['history_status'], 'last_review_had_no_historical_catalog')
        later = annotate(self.core, 'status', {})
        self.assertEqual(later['comparison']['history_status'], 'last_review_had_no_historical_catalog')

    def test_current_table_is_equal_in_plain_gnomon_and_ledger(self):
        tables = []
        for arm in ('plain', 'gnomon', 'ledger'):
            self.arm = arm; self.write()
            tables.append(annotate(self.core, 'status', {})['current_cv'])
        self.assertEqual(tables[0], tables[1]); self.assertEqual(tables[0], tables[2])

    def test_new_backtest_focus_is_explicit_and_does_not_select_forecast(self):
        newest = self.rows[0]['config_id']
        out = annotate(self.core, 'backtest', {'config_id': newest})
        self.assertEqual(out['focus_rule'], 'new_backtest_vs_lowest_current_cv_alternative')
        self.assertFalse(out['comparison']['forecast_selection_made'])
        self.assertFalse(Path('checkpoint.json').exists())
        self.assertFalse(Path('forecast.json').exists())

    def test_unknown_explicit_pair_is_not_replaced_with_a_different_comparison(self):
        out = annotate(self.core, 'status', {}, requested_pair=['unknown', self.rows[0]['config_id']])
        self.assertEqual(out['comparison_status'], 'requested_pair_lacks_complete_current_cv')
        self.assertNotIn('comparison', out)

    def test_corrupted_immutable_artifact_is_rejected(self):
        out = annotate(self.core, 'status', {})
        Path(out['comparison']['evidence']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'content-addressed evidence changed'):
            annotate(self.core, 'status', {})

    def test_host_sync_is_not_annotated(self):
        self.assertIsNone(annotate(self.core, 'sync', {}))
        self.assertFalse(Path('comparison-annotations.jsonl').exists())


if __name__ == '__main__': unittest.main()
