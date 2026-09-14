from copy import deepcopy
import hashlib
import json
import unittest

from benchmarks.ledger_optimization.exploration_review_095 import exploration_review


def canonical(config):
    # Synthetic callable in place of the common numerical validator; no fits.
    if set(config) != {'model', 'alpha', 'lags'} or config['model'] != 'ridge':
        raise ValueError('Invalid test configuration')
    return config


def identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def fixture():
    configs = [{'model': 'ridge', 'alpha': a, 'lags': 14} for a in (1., 10., 100.)]
    ids = list(map(identity, configs))
    catalog = {cid: {'provider': 'ridge_' + cid, 'revision': 'v1:' + cid, 'config': c}
               for cid, c in zip(ids, configs)}
    cards = []
    for i, candidate in enumerate(ids[1:]):
        cards.append({'config_ids': [ids[0], candidate], 'evidence_pointer': f'/cards/{i}',
                      'windows': {
                          'last_4_origins': {'matched_origins': 1, 'n': 14,
                                            'scores': {ids[0]: 2., candidate: 1. if i == 0 else 3.}},
                          'last_12_origins': {'same_as': 'last_4_origins'},
                          'lifetime': {'matched_origins': 2, 'n': 28,
                                       'scores': {ids[0]: 2., candidate: 3. if i == 0 else 2.}}}})
    query = {'series_id': 'synthetic', 'unit': 'widgets', 'horizon': 14,
             'origin': '2026-01-01T00:00:00+00:00'}
    review = {'schema_version': 'agent-review-088', 'metric': 'rmsle', 'provider_calls': 0,
              'query': query, 'configuration_index': catalog, 'cards': cards,
              'full_evidence': {'path': 'evidence.json', 'sha256': 'a' * 64},
              'pagination': {'shown_pairs': 2, 'all_pairs_included': False,
                             'next_call': {'operation': 'review', 'arguments': {'offset': 2}}}}
    current = {'query': deepcopy(query), 'tested_configurations': [{'config_id': ids[0], 'config': configs[0]}],
               'budget': {'numerical_remaining': 4, 'phase': 'exploration'}}
    return review, current, ids


class ExplorationReviewTests(unittest.TestCase):
    def test_preserves_losses_disagreements_ties_and_source(self):
        review, current, ids = fixture()
        before = deepcopy((review, current))
        result = exploration_review(review, current, canonical)
        self.assertEqual((review, current), before)
        self.assertEqual({k: v for k, v in result.items() if k != 'exploration'}, review)
        neighbors = {n['candidate_config_id']: n for n in result['exploration']['neighbors']}
        self.assertEqual(neighbors[ids[1]]['windows']['last_4_origins']['comparison'], 'lower_error')
        self.assertEqual(neighbors[ids[1]]['windows']['lifetime']['comparison'], 'higher_error')
        self.assertEqual(neighbors[ids[2]]['windows']['last_4_origins']['comparison'], 'higher_error')
        self.assertEqual(neighbors[ids[2]]['windows']['lifetime']['comparison'], 'tie')
        for n in neighbors.values():
            call = n['next_call']
            self.assertEqual(call['valid_for_task'], current['query'])
            self.assertEqual(call['arguments']['config'], review['configuration_index'][n['candidate_config_id']]['config'])
            self.assertTrue(call['admissible_now'])

    def test_partial_page_does_not_invent_missing_comparison(self):
        review, current, ids = fixture()
        review['cards'].pop()
        review['pagination']['shown_pairs'] = 1
        result = exploration_review(review, current, canonical)
        n = next(n for n in result['exploration']['neighbors'] if n['candidate_config_id'] == ids[2])
        self.assertEqual(n['evidence_status'], 'pair_not_on_this_page')
        self.assertEqual(n['windows'], {})
        self.assertEqual(result['pagination'], review['pagination'])

    def test_zero_support_is_not_a_tie_or_parameter_benefit(self):
        review, current, _ = fixture()
        for card in review['cards']:
            for label in ('last_4_origins', 'lifetime'):
                card['windows'][label] = {'matched_origins': 0, 'n': 0, 'scores': {},
                                         'lowest_error_config_ids': [], 'status': 'insufficient_evidence'}
        result = exploration_review(review, current, canonical)
        for n in result['exploration']['neighbors']:
            self.assertTrue(all(w['comparison'] == 'unsupported' for w in n['windows'].values()))
        current['tested_configurations'] = []
        self.assertEqual(exploration_review(review, current, canonical)['exploration']['neighbors'], [])

    def test_no_repeat_of_current_backtest_and_no_multifield_invention(self):
        review, current, ids = fixture()
        current['tested_configurations'].append({'config_id': ids[1], 'config': review['configuration_index'][ids[1]]['config']})
        result = exploration_review(review, current, canonical)
        self.assertEqual({n['candidate_config_id'] for n in result['exploration']['neighbors']}, {ids[2]})
        config = {'model': 'ridge', 'alpha': 1000., 'lags': 28}
        review['configuration_index'][identity(config)] = {'provider': 'ridge_new', 'revision': 'v1:new', 'config': config}
        result = exploration_review(review, current, canonical)
        self.assertNotIn(identity(config), {n['candidate_config_id'] for n in result['exploration']['neighbors']})

    def test_budget_and_phase_cannot_authorize_extra_fits(self):
        for phase, remaining, expected in [('exploration', 3, False), ('selection', 60, False), ('exploration', 4, True)]:
            review, current, _ = fixture()
            current['budget'].update(phase=phase, numerical_remaining=remaining)
            result = exploration_review(review, current, canonical)
            for n in result['exploration']['neighbors']:
                self.assertEqual(n['next_call']['admissible_now'], expected)
                self.assertEqual(n['next_call']['required_fits'], 3)
                self.assertEqual(n['next_call']['final_fit_reserve'], 1)

    def test_wrong_task_or_identity_rejected(self):
        for field, value in [('series_id', 'elsewhere'), ('unit', 'kg'), ('horizon', 7), ('origin', '2027-01-01T00:00:00Z')]:
            review, current, _ = fixture()
            current['query'][field] = value
            with self.assertRaises(ValueError):
                exploration_review(review, current, canonical)
        review, current, ids = fixture()
        review['configuration_index'][ids[1]]['config']['alpha'] = 20.
        with self.assertRaises(ValueError):
            exploration_review(review, current, canonical)

    def test_malformed_evidence_never_becomes_guidance(self):
        for bad in ('cycle', 'count', 'nan', 'duplicate'):
            review, current, ids = fixture()
            if bad == 'cycle':
                review['cards'][0]['windows']['last_12_origins']['same_as'] = 'last_12_origins'
            elif bad == 'count':
                review['cards'][0]['windows']['lifetime']['n'] = 27
            elif bad == 'nan':
                review['cards'][0]['windows']['lifetime']['scores'][ids[0]] = float('nan')
            else:
                review['cards'][1] = deepcopy(review['cards'][0])
            with self.assertRaises(ValueError):
                exploration_review(review, current, canonical)


if __name__ == '__main__':
    unittest.main()
