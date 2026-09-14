import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization.collection_capsule_096 import build
from benchmarks.ledger_optimization import collection_plan_096 as plan


class CollectionPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.capsule = self.root/'capsule'
        build(self.capsule)
        self.source = self.root/'jobs.json'
        self.source.write_text(json.dumps({f's{s}':[{'series_id':f's{s}', 'round':n, 'origin':f't{n}'} for n in range(26)] for s in range(4)}))
        self.output = self.root/'plan.json'
        self.patch = patch.object(plan, 'SOURCE_SHA', hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def freeze(self):
        return plan.freeze(self.capsule, self.source, self.output)

    def test_full_matched_schedule_and_closed_dispatch(self):
        result = self.freeze()
        self.assertEqual(len(result['cases']), 104)
        self.assertEqual(sum(c['stage']=='pilot' for c in result['cases']), 12)
        self.assertEqual(result['planned']['total_sessions'], 312)
        self.assertEqual(result['arms'], ['plain','gnomon','ledger'])
        self.assertFalse(result['comparison']['old_093_controls_allowed'])
        self.assertFalse(result['comparison']['reuse_old_093_state'])
        self.assertIsNone(result['pilot_gate']['accuracy_threshold'])
        self.assertFalse(result['final_gate_opened'])
        self.assertEqual(result['budgets']['numerical_attempts'], 60)

    def test_input_change_rejected_before_plan(self):
        self.source.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'frozen reused'):
            self.freeze()
        self.assertFalse(self.output.exists())

    def test_capsule_change_rejected_before_plan(self):
        path=self.capsule/'benchmarks/hermes_ml_checkpoint_v6/lab.py'
        path.write_text(path.read_text()+'\n# unexpected change\n')
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            self.freeze()
        self.assertFalse(self.output.exists())

    def test_missing_source_cannot_be_hidden_by_manifest(self):
        path=self.capsule/'capsule.json';value=json.loads(path.read_text())
        value['sources'].pop('worker.py');path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Unrecognized frozen base'):
            self.freeze()

    def test_existing_plan_immutable(self):
        self.freeze();before=self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, 'fresh plan'):
            self.freeze()
        self.assertEqual(self.output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
