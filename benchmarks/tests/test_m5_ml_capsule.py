import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from benchmarks.ledger_optimization.m5_ml_capsule import build, DEVELOPMENT_JOBS_SHA
from benchmarks.ledger_optimization.m5_ml_development_contrast import paired_series_contrast

SOURCE = Path('results/contrast-capsule-100-offline-003/capsule')
MANIFEST = Path('results/ledger-optimization/m5-panel-014/manifest.json')
JOBS = Path('results/m5-ml-development-prepare-001/prepared/development-jobs.json')


class DevelopmentContrastTests(unittest.TestCase):
    def pairs(self):
        return [({'series_id': 'a', 'round': i, 'origin': str(i), 'rmsle': a,
                  'valid': False, 'fallback_used': True},
                 {'series_id': 'a', 'round': i, 'origin': str(i), 'rmsle': b})
                for i, (a, b) in enumerate(((1., 2.), (3., 2.), (0., 0.)))]

    def test_all_pairs_including_fallbacks_and_no_independent_item_interval(self):
        result = paired_series_contrast(self.pairs())
        self.assertEqual((result['pairs'], result['wins'], result['losses'], result['ties']), (3, 1, 1, 1))
        self.assertEqual(result['relative_rmsle_reduction'], 0.)
        self.assertEqual(result['bootstrap_draws'], 0)
        self.assertIsNone(result['exploratory_series_bootstrap_95_interval'])
        self.assertFalse(result['target_established'])

    def test_zero_control_does_not_invent_relative_improvement(self):
        pairs = self.pairs()[-1:]
        result = paired_series_contrast(pairs)
        self.assertIsNone(result['relative_rmsle_reduction'])
        self.assertEqual(result['absolute_rmsle_reduction'], 0)
        pairs[0][0]['rmsle'] = 2.
        self.assertEqual(paired_series_contrast(pairs)['absolute_rmsle_reduction'], -2.)

    def test_invalid_scores_and_unmatched_or_duplicate_tasks_reject(self):
        for value in (True, float('nan'), float('inf'), -1.):
            rows = self.pairs(); rows[0][0]['rmsle'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                paired_series_contrast(rows)
        for field, value in (('series_id', 'other'), ('round', 3), ('origin', 'other')):
            rows = self.pairs(); rows[0][1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                paired_series_contrast(rows)
        rows = self.pairs()
        with self.assertRaises(ValueError): paired_series_contrast(rows+rows[:1])
        with self.assertRaises(ValueError): paired_series_contrast([])


@unittest.skipUnless(SOURCE.exists() and MANIFEST.exists() and JOBS.exists(),
                     'requires the retained source capsule and authorized development artifacts')
class M5CapsuleTests(unittest.TestCase):
    def test_reproducible_build_preserves_models_budgets_and_parent(self):
        parent = json.loads((SOURCE/'capsule.json').read_text())
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (SOURCE/'benchmarks/hermes_ml_checkpoint_v6').iterdir() if p.is_file()}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            a = build(SOURCE, MANIFEST, JOBS, root/'one')
            b = build(SOURCE, MANIFEST, JOBS, root/'two')
            self.assertEqual(a, b)
            self.assertEqual(a['planned_per_seed'], {'pilot': 72, 'continuation': 552, 'total': 624})
            self.assertFalse(a['execution_authorized']); self.assertFalse(a['selected_final_candidate'])
            self.assertFalse(a['final_gate_opened'])
            package = root/'one/benchmarks/hermes_ml_checkpoint_v6'
            for filename in ('numerical.py', 'core.py', 'lab.py', 'policy.py', 'transport.py',
                             'worker.py', 'maturation.py', 'execution_boundary_093.py'):
                self.assertEqual(a['sources'][filename], parent['sources'][filename])
            tree = ast.parse((package/'run.py').read_text())
            constant = next(n for n in tree.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == 'SOURCE_SHA' for t in n.targets))
            self.assertEqual(ast.literal_eval(constant.value), DEVELOPMENT_JOBS_SHA)
            self.assertIn('not evidence that promotions were absent', (package/'TASK.md').read_text())
            self.assertEqual(hashlib.sha256((root/'one/cohort-contract.json').read_bytes()).hexdigest(),
                             a['cohort_contract_sha256'])
            with self.assertRaises(ValueError): build(SOURCE, MANIFEST, JOBS, root/'one')
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (SOURCE/'benchmarks/hermes_ml_checkpoint_v6').iterdir() if p.is_file()})

    def test_altered_jobs_or_parent_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            wrong = root/'jobs.json'; wrong.write_bytes(JOBS.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError, 'Exact prepared development-job'):
                build(SOURCE, MANIFEST, wrong, root/'output')
            self.assertFalse((root/'output').exists())
            source = root/'parent'; shutil.copytree(SOURCE, source)
            (source/'benchmarks/hermes_ml_checkpoint_v6/numerical.py').write_text('pass\n')
            with self.assertRaisesRegex(ValueError, 'Parent sources changed'):
                build(source, MANIFEST, JOBS, root/'output')
            self.assertFalse((root/'output').exists())


if __name__ == '__main__':
    unittest.main()
