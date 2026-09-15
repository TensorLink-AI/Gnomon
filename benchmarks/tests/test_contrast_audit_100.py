from copy import deepcopy
import json
import unittest

from benchmarks.ledger_optimization.contrast_audit_100 import review_at_operation
from benchmarks.ledger_optimization.contrast_view_100 import contrast_view
from benchmarks.tests.test_current_history_contrast_100 import inputs
from benchmarks.tests.test_paired_consistency_098 import encode, reseal


class CachedReviewAuditTests(unittest.TestCase):
    def test_distinct_recent_and_lifetime_groups_replay_exact_cache_order(self):
        brief, full, current, ids = inputs((0., 8., 8., 8., 8.), (100., 7., 7., 7., 7.))
        recent = full['cards'][0]['windows']['last_4_origins']
        recent['origins'] = recent['origins'][1:]
        recent.update(matched_origins=4, n=8, start=recent['origins'][0]['origin'])
        recent['models'][0]['rmsle'] = 8.
        recent['models'][1]['rmsle'] = 7.
        windows = brief['cards'][0]['windows']
        windows['last_12_origins'] = deepcopy(windows['last_4_origins'])
        windows['last_12_origins']['evidence_pointer'] = '/cards/0/windows/last_12_origins'
        windows['lifetime']['same_as'] = 'last_12_origins'
        windows['last_4_origins'].update(matched_origins=4, n=8, start=recent['start'],
            scores={ids['a']: 8., ids['b']: 7.}, lowest_error_config_ids=[ids['b']])
        reseal(brief, full)
        before = deepcopy(brief)
        direct, direct_raw = contrast_view(current, review=review_at_operation(brief, 'review'),
                                           full_evidence_bytes=encode(full))
        cached = json.loads(json.dumps({'query': brief['query'], 'review': brief},
                                       sort_keys=True))['review']
        expected, expected_raw = contrast_view(current, review=cached, full_evidence_bytes=encode(full))
        self.assertEqual([r['windows'] for r in direct['history']],
                         [['last_4_origins'], ['last_12_origins', 'lifetime']])
        self.assertEqual([r['windows'] for r in expected['history']],
                         [['last_12_origins', 'lifetime'], ['last_4_origins']])
        self.assertNotEqual(direct, expected)
        self.assertEqual(direct_raw, expected_raw)
        for operation in ('backtest', 'status', 'commit', 'start'):
            with self.subTest(operation=operation):
                self.assertEqual(contrast_view(current, review=review_at_operation(brief, operation),
                                              full_evidence_bytes=encode(full)), (expected, expected_raw))
        self.assertEqual(brief, before)

    def test_cache_replay_preserves_arrays_and_does_not_validate_false_facts(self):
        brief, full, current, ids = inputs()
        brief['cards'][0]['windows']['last_4_origins']['scores'][ids['a']] += .25
        replay = review_at_operation(brief, 'backtest')
        self.assertEqual(replay, brief)
        self.assertEqual(replay['cards'][0]['config_ids'], brief['cards'][0]['config_ids'])
        with self.assertRaises(ValueError):
            contrast_view(current, review=replay, full_evidence_bytes=encode(full))
        self.assertIsNone(review_at_operation(None, 'status'))


if __name__ == '__main__':
    unittest.main()
