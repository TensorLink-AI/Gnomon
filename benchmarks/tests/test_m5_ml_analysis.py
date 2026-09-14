"""Synthetic-only tests; no reserved files, runtime imports or API calls."""
from copy import deepcopy
import math
import random
import unittest

from benchmarks.ledger_optimization import m5_ml_analysis as a


def fixture():
    panel = [{'series_id': f'series-{n:02}', 'store_id': f'store-{n//3}', 'item_id': f'item-{n}'}
             for n in range(24)]
    seeds = [7, 19]
    rows = [{'arm': arm, 'series_id': p['series_id'], 'round': r, 'seed': seed,
             'rmsle': {'plain': 1.2, 'gnomon': 1.0, 'ledger': .7}[arm],
             'valid': True, 'workflow_complete': True, 'fallback_used': False}
            for arm in a.ARMS for p in panel for r in range(26) for seed in seeds]
    return panel, seeds, rows


class AnalysisTests(unittest.TestCase):
    def test_known_reduction_all_cases_and_no_automatic_success_claim(self):
        panel, seeds, rows = fixture()
        for row in rows:
            if row['arm'] == 'ledger':
                row['rmsle'] = .8
        r = a.analyze(panel, seeds, rows)
        self.assertEqual(r['decisions'], 3744)
        self.assertEqual(r['matched_cases_per_arm'], 1248)
        primary = r['contrasts']['ledger_vs_gnomon']
        self.assertAlmostEqual(primary['relative_rmsle_reduction'], .2)
        for bound in primary['paired_95_interval']:
            self.assertAlmostEqual(bound, .2)
        self.assertTrue(r['numerical_primary_criteria_met'])
        self.assertFalse(r['target_established'])
        self.assertEqual(r['uncertainty']['store_clusters'], 8)

    def test_incomplete_duplicate_unexpected_and_invalid_rows_rejected(self):
        panel, seeds, rows = fixture()
        variants = [rows[:-1], rows + [rows[0]]]
        for field, value in [('arm', 'ledger_119'), ('series_id', 'replacement'),
                             ('seed', 8), ('round', 26), ('round', True), ('seed', True),
                             ('rmsle', math.nan), ('rmsle', math.inf), ('rmsle', -1),
                             ('rmsle', True), ('valid', 1), ('workflow_complete', None),
                             ('fallback_used', None), ('valid', False)]:
            changed = deepcopy(rows)
            changed[0][field] = value
            variants.append(changed)
        for variant in variants:
            with self.subTest(first=variant[0]), self.assertRaises(ValueError):
                a.analyze(panel, seeds, variant)

    def test_store_and_seed_structure_cannot_be_inferred_or_relaxed(self):
        panel, seeds, rows = fixture()
        with self.assertRaises(ValueError):
            a.analyze(panel[:-1], seeds, rows)
        for field, value in [('store_id', 'store-1'), ('item_id', 'item-1'),
                             ('series_id', 'series-01'), ('item_id', '')]:
            changed = deepcopy(panel)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                a.analyze(changed, seeds, rows)
        for invalid in ([7], [7, 7], [7, 19, 42], [True, 19]):
            with self.assertRaises(ValueError):
                a.analyze(panel, invalid, rows)

    def test_shared_circular_blocks_draw_stores_once_and_wrap(self):
        class PrescribedRandom:
            def __init__(self):
                self.values = iter([7, 7, 0, 1, 2, 3, 4, 5] + [25] * 7)
                self.bounds = []
            def randrange(self, bound):
                self.bounds.append(bound)
                return next(self.values)
        rng = PrescribedRandom()
        stores, origins = a._draw(rng)
        self.assertEqual(stores, [7, 7, 0, 1, 2, 3, 4, 5])
        self.assertEqual(origins, ([25, 0, 1, 2] * 7)[:26])
        self.assertEqual(rng.bounds, [8] * 8 + [26] * 7)

    def test_independent_heterogeneous_bootstrap_and_order_invariance(self):
        panel, seeds, rows = fixture()
        # Each store/round has distinct performance; items/seeds vary too.
        def score(arm, item, r, seed):
            base = 1 + item//3 + .01*r + .1*(item%3) + .001*seed
            return base * ({'gnomon': 1., 'plain': 1.1, 'ledger': .5 + .08*(item//3)}[arm])
        for row in rows:
            row['rmsle'] = score(row['arm'], int(row['series_id'].split('-')[1]), row['round'], row['seed'])
        # Failed workflow is deliberately high-error and must remain included.
        rows[-1].update(rmsle=50, valid=False, workflow_complete=False, fallback_used=True)
        r = a.analyze(panel, seeds, rows)
        means = {arm: math.fsum(row['rmsle'] for row in rows if row['arm'] == arm)/1248 for arm in a.ARMS}
        for arm in a.ARMS:
            self.assertAlmostEqual(r['mean_per_case_rmsle'][arm], means[arm])
        self.assertEqual(r['completion']['ledger']['fallback_used'], 1)
        self.assertEqual(r['completion']['ledger']['valid'], 1247)
        # Independent oracle: sum six raw cases per cell, then resample paired
        # store/time coordinates without using the implementation's draw helper.
        sums = {(arm, store, origin): math.fsum(row['rmsle'] for row in rows
                    if row['arm'] == arm and int(row['series_id'].split('-')[1])//3 == store
                    and row['round'] == origin)
                for arm in ('gnomon', 'ledger') for store in range(8) for origin in range(26)}
        rng = random.Random(20260912)
        reductions = []
        for _ in range(5000):
            stores = [rng.randrange(8) for _ in range(8)]
            starts = [rng.randrange(26) for _ in range(7)]
            origins = [(start+offset)%26 for start in starts for offset in range(4)][:26]
            total = {arm: math.fsum(sums[arm, s, origin] for s in stores for origin in origins)
                     for arm in ('gnomon', 'ledger')}
            reductions.append(1-total['ledger']/total['gnomon'])
        reductions.sort()
        expected = []
        for probability in (.025, .975):
            index = 4999*probability
            k = int(index)
            expected.append(reductions[k]*(1-(index-k))+reductions[k+1]*(index-k))
        for actual, wanted in zip(r['contrasts']['ledger_vs_gnomon']['paired_95_interval'], expected):
            self.assertAlmostEqual(actual, wanted, places=12)
        self.assertEqual(r, a.analyze(list(reversed(panel)), list(reversed(seeds)), list(reversed(rows))))

    def test_zero_control_does_not_claim_improvement_or_drop_draws(self):
        panel, seeds, rows = fixture()
        for row in rows:
            row['rmsle'] = 0.
        r = a.analyze(panel, seeds, rows)
        primary = r['contrasts']['ledger_vs_gnomon']
        self.assertIsNone(primary['relative_rmsle_reduction'])
        self.assertIsNone(primary['paired_95_interval'])
        self.assertEqual(primary['zero_control_bootstrap_draws'], 5000)
        self.assertFalse(r['numerical_primary_criteria_met'])


if __name__ == '__main__':
    unittest.main()
