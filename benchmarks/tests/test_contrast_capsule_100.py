import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from benchmarks.ledger_optimization.contrast_capsule_100 import build, MODULES, AUDIT_MODULES


SOURCE = Path('results/workflow-097-offline-002/capsule')


@unittest.skipUnless(SOURCE.exists(), 'requires retained frozen 097 capsule')
class ContrastCapsuleTests(unittest.TestCase):
    def test_fresh_build_is_reproducible_and_all_added_code_is_protected(self):
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*.py')}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            first = build(SOURCE, root/'one'); second = build(SOURCE, root/'two')
            self.assertEqual(first, second)
            self.assertEqual(first['status'], 'offline_prototype_not_dispatch_ready')
            package = root/'one/benchmarks/hermes_ml_checkpoint_v6'
            tree = ast.parse((package/'run.py').read_text())
            node = next(n for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'PROJECT_FILES' for t in n.targets))
            protected = ast.literal_eval(node.value)
            self.assertTrue(set(MODULES) <= set(protected))
            for module in MODULES:
                raw = (package/module).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), first['sources'][module])
                compile(raw, module, 'exec')
            self.assertTrue(set(AUDIT_MODULES) <= set(first['sources']))
            self.assertIn('audit_annotations(p.parent)', (package/'analyze.py').read_text())
            self.assertNotIn('contrast_audit_100.py', protected)
            with self.assertRaises(ValueError): build(SOURCE, root/'one')
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*.py')})

    def test_wrong_parent_and_changed_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source = root/'source'
            shutil.copytree(SOURCE, source)
            (source/'benchmarks/hermes_ml_checkpoint_v6/lab.py').write_text('raise SystemExit(0)\n')
            with self.assertRaisesRegex(ValueError, 'sources changed'): build(source, root/'out')
            (source/'capsule.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'exact frozen'): build(source, root/'out')

    def test_unmanifested_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source = root/'source'
            shutil.copytree(SOURCE, source)
            (source/'benchmarks/hermes_ml_checkpoint_v6/extra').symlink_to(root)
            with self.assertRaisesRegex(ValueError, 'Unexpected frozen'): build(source, root/'out')


if __name__ == '__main__': unittest.main()
