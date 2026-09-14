from copy import deepcopy
import unittest

from benchmarks.ledger_optimization.sparse_review_094 import compact_unsupported_pairs


def card(count, index=0, loss=1.0):
    window = {'matched_origins': count, 'n': count * 14,
              'status': 'ok' if count else 'insufficient_evidence',
              'scores': {'a': loss, 'b': 2.0} if count else {},
              'lowest_error_config_ids': ['a' if loss < 2 else 'b'] if count else [],
              'evidence_pointer': f'/cards/{index}/windows/last_4_origins'}
    return {'config_ids': ['a', 'b'], 'excluded_count': 3,
            'recent_lifetime_disagreement': False,
            'evidence_pointer': f'/cards/{index}',
            'windows': {'last_4_origins': window,
                        'last_12_origins': {'same_as': 'last_4_origins'},
                        'lifetime': {'same_as': 'last_12_origins'}}}


def review(cards):
    return {'schema_version': 'agent-review-088', 'metric': 'rmsle',
            'provider_calls': 0, 'cards': cards,
            'configuration_index': {'a': {}, 'b': {}},
            'full_evidence': {'path': 'review.json', 'sha256': 'a' * 64},
            'pagination': {'shown_pairs': len(cards), 'offset': 12,
                           'total_pairs': 36, 'all_pairs_included': False,
                           'next_call': {'operation': 'review', 'arguments': {'offset': 24, 'limit': 12}}}}


class SparseReviewTests(unittest.TestCase):
    def test_supported_losses_and_single_origins_preserved(self):
        original = review([card(0, i) for i in range(8)] + [card(1, 8, loss=99.0), card(8, 9)])
        before = deepcopy(original)
        result = compact_unsupported_pairs(original)
        self.assertEqual(original, before)
        self.assertEqual(result['cards'], original['cards'][8:])
        self.assertEqual(result['pagination'], original['pagination'])
        self.assertEqual(result['full_evidence'], original['full_evidence'])
        self.assertEqual(result['configuration_index'], original['configuration_index'])
        self.assertEqual(result['unsupported_pairs'][0]['evidence_pointer'], '/cards/0')
        self.assertEqual(result['unsupported_pairs'][0]['excluded_count'], 3)

    def test_unsupported_recent_but_supported_lifetime_stays(self):
        item = card(0)
        item['windows']['lifetime'] = card(3)['windows']['last_4_origins']
        original = review([item])
        self.assertEqual(compact_unsupported_pairs(original), original)

    def test_all_unsupported_is_not_a_zero_score(self):
        result = compact_unsupported_pairs(review([card(0, i) for i in range(8)]))
        self.assertEqual(result['cards'], [])
        self.assertNotIn('scores', result['unsupported_pairs'][0])
        self.assertEqual(result['unsupported_pairs'][0]['matched_origins_all_windows'], 0)

    def test_small_response_is_not_expanded(self):
        original = review([card(0)])
        self.assertEqual(compact_unsupported_pairs(original), original)

    def test_no_empty_cards_leaves_payload_identical(self):
        for cards in ([], [card(1)], [card(1), card(4, 1)]):
            original = review(cards)
            self.assertEqual(compact_unsupported_pairs(original), original)

    def test_ambiguous_or_conflicting_support_rejected(self):
        variants = []
        for count in (None, True, -1):
            item = card(0)
            item['windows']['last_4_origins']['matched_origins'] = count
            variants.append(item)
        item = card(0)
        item['windows']['last_4_origins']['scores'] = {'a': 0.0}
        variants.append(item)
        item = card(0)
        item['windows']['lifetime']['same_as'] = 'lifetime'
        variants.append(item)
        item = card(0)
        del item['windows']['lifetime']
        variants.append(item)
        for item in variants:
            with self.subTest(item=item), self.assertRaises(ValueError):
                compact_unsupported_pairs(review([item]))


if __name__ == '__main__':
    unittest.main()
