"""Seed copies and scripted forwarding checks; no paid requests or real data."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from benchmarks.ledger_optimization import seed_capsule_093 as capsule


class CapsuleTests(unittest.TestCase):
    def test_exact_seed_seven_equivalence_and_seed_nineteen_only_two_changes(self):
        before = {p.name: p.read_bytes() for p in capsule.SOURCE.iterdir() if p.is_file()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = capsule.build(root / 'seven', 7)
            second = capsule.build(root / 'nineteen', 19)
            self.assertEqual(first['sources'], first['base_sources'])
            self.assertEqual(first['changed_files'], [])
            self.assertEqual(second['changed_files'], ['transport.py', 'worker.py'])
            for seed, name in ((7, 'seven'), (19, 'nineteen')):
                package = root / name / 'benchmarks/hermes_ml_checkpoint_v6'
                for filename, data in before.items():
                    actual = (package / filename).read_bytes()
                    if seed == 19 and filename == 'worker.py':
                        actual = actual.replace(b"'temperature':0.2,'seed':19", b"'temperature':0.2,'seed':7")
                    if seed == 19 and filename == 'transport.py':
                        actual = actual.replace(b'temperature=0.2, seed=19', b'temperature=0.2, seed=7')
                    self.assertEqual(actual, data, filename)
                self.assertFalse((root / name / '.env').exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in capsule.SOURCE.iterdir() if p.is_file()})

    def test_reject_overwrite_invalid_seed_and_altered_source_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (True, '19', -1, 42):
                with self.assertRaises(ValueError):
                    capsule.build(root / 'invalid', seed)
                self.assertFalse((root / 'invalid').exists())
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                capsule.build(root, 7)
            source = root / 'source'; source.mkdir()
            for p in capsule.SOURCE.iterdir():
                if p.is_file(): shutil.copyfile(p, source / p.name)
            with (source / 'policy.py').open('a') as f: f.write('\n# changed budget source\n')
            with self.assertRaisesRegex(ValueError, 'audited'):
                capsule.build(root / 'changed', 19, source)
            self.assertFalse((root / 'changed').exists())

    def test_actual_worker_constructor_and_local_proxy_enforce_each_seed(self):
        probe = Path(capsule.__file__).with_name('probe_seed_capsule_093.py')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (7, 19):
                folder = root / f'seed-{seed}'
                capsule.build(folder, seed)
                env = dict(os.environ)
                env.pop('PYTHONPATH', None)
                env['PYTHONDONTWRITEBYTECODE'] = '1'
                result = subprocess.run([sys.executable, str(probe), '--capsule', str(folder),
                                         '--output', str(root / f'probe-{seed}')],
                                        cwd=root, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                receipt = json.loads(result.stdout)
                self.assertEqual(receipt['requested_seed'], seed)
                self.assertEqual(receipt['worker_constructions'], 3)
                self.assertEqual(receipt['scripted_forwarded_requests'], 16)
                self.assertTrue(receipt['seventeenth_request_rejected'])
                self.assertEqual(receipt['engy_calls'], 0)


if __name__ == '__main__':
    unittest.main()
