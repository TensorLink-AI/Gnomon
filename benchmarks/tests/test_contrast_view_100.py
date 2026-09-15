from copy import deepcopy
import hashlib
import json
import unittest

from benchmarks.ledger_optimization.contrast_view_100 import contrast_view
from benchmarks.ledger_optimization.current_cv_pair_100 import current_cv_pair
from benchmarks.tests.test_current_history_contrast_100 import inputs
from benchmarks.tests.test_paired_consistency_098 import encode


class ContrastViewTests(unittest.TestCase):
    def test_common_current_evidence_is_identical_across_arms(self):
        b, f, c, ids = inputs()
        before = deepcopy((b, f, c))
        plain, plain_raw = contrast_view(c)
        ledger, ledger_raw = contrast_view(c, review=b, full_evidence_bytes=encode(f))
        self.assertEqual(plain['current_cv'], ledger['current_cv'])
        self.assertEqual(plain['history_status'], 'not_requested')
        self.assertNotIn('history', plain)
        models = {m['config_id']: m for m in plain['current_cv']['models']}
        self.assertEqual(models[ids['a']]['rank'], 2)
        self.assertEqual(models[ids['b']]['rank'], 1)
        self.assertEqual(models[ids['b']]['lower_error_folds'], 3)
        h = ledger['history']
        self.assertEqual(len(h), 1)
        self.assertEqual(h[0]['windows'], ['last_4_origins', 'last_12_origins', 'lifetime'])
        self.assertTrue(h[0]['cv_winner_disagrees'])
        self.assertEqual(h[0]['matched_origins'], 1)
        self.assertEqual((b, f, c), before)
        self.assertNotEqual(plain_raw, ledger_raw)

    def test_artifact_digest_and_all_original_current_references_survive(self):
        b, f, c, _ = inputs()
        view, raw = contrast_view(c, review=b, full_evidence_bytes=encode(f))
        ref = view['evidence']
        self.assertEqual(hashlib.sha256(raw).hexdigest(), ref['sha256'])
        self.assertEqual(len(raw), ref['bytes'])
        self.assertEqual(ref['path'], f"comparison-evidence/{ref['sha256']}.json")
        full = json.loads(raw)
        self.assertEqual(full['current'], c)
        self.assertEqual(full['current_summary'], current_cv_pair(c))
        self.assertEqual(full['historical_contrast']['full_evidence'], b['full_evidence'])
        self.assertEqual(contrast_view(c, review=b, full_evidence_bytes=encode(f)), (view, raw))

    def test_current_summary_needs_no_ledger_or_historical_catalog(self):
        _, _, c, _ = inputs()
        out = current_cv_pair(c)
        self.assertEqual(out['provider_calls'], 0)
        self.assertFalse(out['forecast_selection_made'])
        self.assertEqual(out['query'], c['query'])
        for mutation in [lambda q: q.update(horizon=True), lambda q: q.update(unit=''),
                         lambda q: q.update(series_id=''), lambda q: q.update(extra=1)]:
            bad = deepcopy(c); mutation(bad['query'])
            with self.assertRaises(ValueError): current_cv_pair(bad)

    def test_complete_ties_are_not_arbitrarily_ranked(self):
        _, _, c, _ = inputs()
        for l, r in zip(c['runs'][0]['folds'], c['runs'][1]['folds']): r['rmsle'] = l['rmsle']
        v, _ = contrast_view(c)
        self.assertEqual([m['rank'] for m in v['current_cv']['models']], [1, 1])
        self.assertEqual(v['current_cv']['tied_folds'], 3)
        self.assertEqual([m['lower_error_folds'] for m in v['current_cv']['models']], [0, 0])

    def test_no_match_and_absent_catalog_are_distinct(self):
        b, f, c, _ = inputs((), ())
        v, _ = contrast_view(c, review=b, full_evidence_bytes=encode(f))
        self.assertEqual(v['history_status'], 'pair_on_supplied_page')
        self.assertEqual(v['history'][0]['matched_origins'], 0)
        self.assertEqual(v['history'][0]['mean_rmsle'], {})
        self.assertIsNone(v['history'][0]['cv_winner_disagrees'])
        c['runs'][0].update(config={'model': 'new'})
        c['runs'][0]['config_id'] = hashlib.sha256(json.dumps(c['runs'][0]['config'], sort_keys=True).encode()).hexdigest()[:16]
        v, _ = contrast_view(c, review=b, full_evidence_bytes=encode(f))
        self.assertEqual(v['history_status'], 'configuration_not_in_historical_catalog')
        self.assertEqual(v['history'], [])

    def test_partial_or_invalid_historical_inputs_fail_closed(self):
        b, f, c, _ = inputs()
        for kwargs in ({'review': b}, {'full_evidence_bytes': encode(f)},
                       {'review': b, 'full_evidence_bytes': encode(f)+b' '}):
            with self.assertRaises(ValueError): contrast_view(c, **kwargs)


if __name__ == '__main__': unittest.main()
