from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from statistics import mean
import unittest

from benchmarks.ledger_optimization.paired_consistency_098 import paired_consistency


def encode(full):
    return json.dumps(full, separators=(',', ':'), allow_nan=False).encode()


def fixture(left=(1., 4.), right=(3., 1.)):
    providers = {'provider_a': 'rev_a', 'provider_b': 'rev_b'}
    ids = ['a', 'b']
    catalog = {c: {'provider': p, 'revision': providers[p], 'config': {'model': c}}
               for c, p in zip(ids, providers)}
    models = lambda x, y: [{'provider': p, 'revision': providers[p], 'rmsle': v}
                          for p, v in zip(providers, (x, y))]
    origins = [{'origin': (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=14*i)).isoformat(),
                'n': 2, 'models': models(x, y)} for i, (x, y) in enumerate(zip(left, right, strict=True))]
    window = {'status': 'ok' if origins else 'insufficient_evidence',
              'matched_origins': len(origins), 'n': 2*len(origins), 'origins': origins,
              'models': models(mean(left), mean(right)) if origins else [],
              'start': origins[0]['origin'] if origins else None,
              'end': origins[-1]['origin'] if origins else None}
    labels = ('last_4_origins', 'last_12_origins', 'lifetime')
    full = {'metric': 'rmsle', 'provider_calls': 0, 'as_of': '2026-03-01T00:00:00+00:00',
            'configuration_index': catalog, 'offset': 0, 'total_pairs': 1, 'next_offset': None,
            'cards': [{'left_config_id': 'a', 'right_config_id': 'b', 'providers': providers,
                       'windows': {k: deepcopy(window) for k in labels}}]}
    scores = dict(zip(ids, (mean(left), mean(right)))) if origins else {}
    brief_window = {k: deepcopy(window[k]) for k in ('status', 'matched_origins', 'n', 'start', 'end')}
    brief_window.update(scores=scores, lowest_error_config_ids=[c for c in ids if scores and scores[c] == min(scores.values())],
                        evidence_pointer='/cards/0/windows/last_4_origins')
    brief = {'schema_version': 'agent-review-088', 'metric': 'rmsle', 'provider_calls': 0,
             'query': {'series_id': 'sales', 'unit': 'widgets', 'horizon': 2, 'origin': full['as_of']},
             'configuration_index': deepcopy(catalog), 'forecast_selection_made': False,
             'pagination': {'shown_pairs': 1, 'offset': 0, 'total_pairs': 1,
                            'all_pairs_included': True, 'next_call': None},
             'full_evidence': {'path': 'evidence.json', 'sha256': hashlib.sha256(encode(full)).hexdigest()},
             'cards': [{'config_ids': ids, 'excluded_count': 7, 'evidence_pointer': '/cards/0',
                        'windows': {'last_4_origins': brief_window,
                                    **{k: {'same_as': 'last_4_origins', 'evidence_pointer': '/cards/0/windows/'+k}
                                       for k in labels[1:]}}}]}
    return brief, full


def render(brief, full):
    return paired_consistency(brief, encode(full), detailed=True)


def reseal(brief, full):
    brief['full_evidence']['sha256'] = hashlib.sha256(encode(full)).hexdigest()


class PairedConsistencyTests(unittest.TestCase):
    def test_default_counts_keep_the_evidence_small_and_unambiguous(self):
        b, f = fixture()
        result = paired_consistency(b, encode(f))
        self.assertEqual(result['cards'][0]['windows']['last_4_origins']['origin_consistency'],
                         {'left_wins': 1, 'right_wins': 1, 'ties': 0,
                          'aggregate_winner_has_origin_losses': True})
        self.assertLess(len(encode(result)), len(encode(render(b, f))))
        self.assertEqual(result['origin_consistency_semantics']['detail'], 'counts')
        with self.assertRaises(ValueError): paired_consistency(b, encode(f), detailed=1)

    def test_average_winner_loses_an_origin_and_inputs_are_unchanged(self):
        brief, full = fixture()
        before = deepcopy((brief, full))
        result = render(brief, full)
        stats = result['cards'][0]['windows']['last_4_origins']['origin_consistency']
        self.assertEqual((stats['left_wins'], stats['right_wins'], stats['ties']), (1, 1, 0))
        self.assertEqual(stats['mean_left_minus_right'], .5)
        self.assertEqual(stats['difference_range'], [-2., 3.])
        self.assertTrue(stats['aggregate_winner_has_origin_losses'])
        self.assertEqual(stats['lowest_on_every_origin_config_ids'], [])
        self.assertEqual(stats['latest_lowest_error_config_ids'], ['b'])
        self.assertEqual((brief, full), before)
        for key in ('query', 'configuration_index', 'pagination', 'full_evidence', 'provider_calls', 'forecast_selection_made'):
            self.assertEqual(result[key], brief[key])
        for key, value in brief['cards'][0]['windows']['last_4_origins'].items():
            self.assertEqual(result['cards'][0]['windows']['last_4_origins'][key], value)
        self.assertEqual(result['cards'][0]['windows']['lifetime'], brief['cards'][0]['windows']['lifetime'])

    def test_exact_origin_ties_are_separate_from_aggregate_ties(self):
        for left, right, expected in [((1., 2.), (2., 1.), (1, 1, 0, [])),
                                      ((1., 2.), (1., 2.), (0, 0, 2, ['a', 'b'])),
                                      ((0.,), (1.,), (1, 0, 0, ['a']))]:
            with self.subTest(left=left, right=right):
                b, f = fixture(left, right)
                s = render(b, f)['cards'][0]['windows']['last_4_origins']['origin_consistency']
                self.assertEqual((s['left_wins'], s['right_wins'], s['ties'], s['lowest_on_every_origin_config_ids']), expected)

    def test_no_evidence_is_not_zero_error_or_a_tie(self):
        b, f = fixture((), ())
        s = render(b, f)['cards'][0]['windows']['last_4_origins']['origin_consistency']
        self.assertEqual(s['status'], 'no_matched_origins')
        self.assertIsNone(s['mean_left_minus_right'])
        self.assertIsNone(s['aggregate_winner_has_origin_losses'])
        self.assertEqual(s['ties'], 0)

    def test_partial_page_navigation_and_exclusions_survive(self):
        b, f = fixture()
        f.update(offset=1, total_pairs=3, next_offset=2)
        b['pagination'].update(offset=1, total_pairs=3, all_pairs_included=False,
                              next_call={'operation': 'review', 'arguments': {'offset': 2, 'limit': 1}, 'valid_for_origin': f['as_of']})
        reseal(b, f)
        result = render(b, f)
        self.assertEqual(result['pagination'], b['pagination'])
        self.assertEqual(result['cards'][0]['excluded_count'], 7)
        self.assertEqual(len(result['cards']), 1)

    def test_recent_and_lifetime_opposite_winners_stay_separate(self):
        b, f = fixture((0., 8., 8., 8., 8.), (100., 7., 7., 7., 7.))
        windows = f['cards'][0]['windows']
        recent = windows['last_4_origins']
        recent['origins'] = recent['origins'][1:]
        recent.update(matched_origins=4, n=8, start=recent['origins'][0]['origin'])
        recent['models'][0]['rmsle'] = 8.
        recent['models'][1]['rmsle'] = 7.
        old = b['cards'][0]['windows']['last_4_origins']
        b['cards'][0]['windows']['last_12_origins'] = deepcopy(old)
        b['cards'][0]['windows']['last_12_origins']['evidence_pointer'] = '/cards/0/windows/last_12_origins'
        b['cards'][0]['windows']['lifetime']['same_as'] = 'last_12_origins'
        old.update(matched_origins=4, n=8, start=recent['start'], scores={'a': 8., 'b': 7.}, lowest_error_config_ids=['b'])
        reseal(b, f)
        result = render(b, f)['cards'][0]['windows']
        self.assertEqual(result['last_4_origins']['origin_consistency']['right_wins'], 4)
        self.assertFalse(result['last_4_origins']['origin_consistency']['aggregate_winner_has_origin_losses'])
        self.assertTrue(result['last_12_origins']['origin_consistency']['aggregate_winner_has_origin_losses'])
        self.assertEqual(result['lifetime']['same_as'], 'last_12_origins')

    def test_digest_identity_and_duplicate_json_keys_fail_closed(self):
        b, f = fixture()
        with self.assertRaises(ValueError):
            paired_consistency(b, encode(f)+b' ')
        raw = encode(f).replace(b'"metric":"rmsle"', b'"metric":"rmsle","metric":"rmsle"', 1)
        b['full_evidence']['sha256'] = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(ValueError):
            paired_consistency(b, raw)
        for change in ('revision', 'origin', 'page'):
            b, f = fixture()
            if change == 'revision': b['configuration_index']['a']['revision'] = 'other'
            if change == 'origin': b['query']['origin'] = '2026-04-01T00:00:00+00:00'
            if change == 'page': b['pagination']['next_call'] = {'operation': 'guess'}
            with self.subTest(change=change), self.assertRaises(ValueError): render(b, f)

    def test_inconsistent_or_nonhistorical_origin_evidence_rejected(self):
        mutations = [lambda w: w.update(matched_origins=True),
                     lambda w: w['origins'][0].update(n=1),
                     lambda w: w['origins'][0].update(origin=w['origins'][1]['origin']),
                     lambda w: w['origins'][0].update(origin='2026-03-01T00:00:00+00:00'),
                     lambda w: w['origins'][0].update(origin='2026-01-01'),
                     lambda w: w['origins'][0]['models'][0].update(revision='other'),
                     lambda w: w['origins'][0]['models'][0].update(rmsle=-1.),
                     lambda w: w['origins'][0]['models'][0].update(rmsle=True),
                     lambda w: w['models'][0].update(rmsle=42.),
                     lambda w: w.update(end='2026-02-01T00:00:00+00:00')]
        for mutate in mutations:
            b, f = fixture();mutate(f['cards'][0]['windows']['last_4_origins']);reseal(b, f)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): render(b, f)

    def test_brief_counts_scores_and_references_cannot_contradict_source(self):
        changes = [lambda b: b['cards'][0]['windows']['lifetime'].update(same_as='lifetime'),
                   lambda b: b['cards'][0]['windows']['last_4_origins'].update(matched_origins=3),
                   lambda b: b['cards'][0]['windows']['last_4_origins']['scores'].update(a=99.),
                   lambda b: b['cards'][0]['windows']['last_4_origins'].update(evidence_pointer='/wrong')]
        for change in changes:
            b, f = fixture();change(b)
            with self.subTest(change=change), self.assertRaises(ValueError): render(b, f)

    def test_observed_average_winner_has_one_win_and_one_loss(self):
        b, f = fixture((.6448454479465352, .5308334089651892),
                       (.6528094515964845, .48625357392095925))
        s = render(b, f)['cards'][0]['windows']['last_4_origins']['origin_consistency']
        self.assertEqual((s['left_wins'], s['right_wins']), (1, 1))
        self.assertTrue(s['aggregate_winner_has_origin_losses'])
        self.assertGreater(s['mean_left_minus_right'], 0.)


if __name__ == '__main__':
    unittest.main()
