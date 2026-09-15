from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
import unittest

from benchmarks.ledger_optimization.current_history_contrast_100 import current_history_contrast
from benchmarks.tests.test_paired_consistency_098 import fixture, encode, reseal


def inputs(left=(1.,), right=(2.,)):
    brief, full = fixture(left, right)
    mapping = {c: hashlib.sha256(json.dumps(v['config'], sort_keys=True).encode()).hexdigest()[:16]
               for c, v in brief['configuration_index'].items()}
    for item in (brief, full):
        item['configuration_index'] = {mapping[c]: v for c, v in item['configuration_index'].items()}
    for card in full['cards']:
        for key in ('left_config_id', 'right_config_id'):
            card[key] = mapping[card[key]]
    for card in brief['cards']:
        card['config_ids'] = [mapping[c] for c in card['config_ids']]
        for window in card['windows'].values():
            if 'same_as' not in window:
                window['scores'] = {mapping[c]: v for c, v in window['scores'].items()}
                window['lowest_error_config_ids'] = [mapping[c] for c in window['lowest_error_config_ids']]
    reseal(brief, full)
    runs = []
    for c, values in (('a', [3., 4., 5.]), ('b', [2., 3., 4.])):
        cid = mapping[c]
        folds = []
        for i, score in enumerate(values):
            origin = datetime.fromisoformat('2026-02-01T00:00:00+00:00') + timedelta(days=i*7)
            folds.append({'origin': origin.isoformat(), 'target_end': (origin+timedelta(days=2)).isoformat(),
                          'n': 2, 'rmsle': score, 'execution_id': f'{c}-{i}'})
        runs.append({'config_id': cid, **deepcopy(brief['configuration_index'][cid]), 'folds': folds})
    return brief, full, {'query': deepcopy(brief['query']), 'runs': runs}, mapping


def render(b, f, c):
    return current_history_contrast(b, encode(f), c)


class CurrentHistoryContrastTests(unittest.TestCase):
    def test_three_current_wins_and_one_old_opposing_origin_are_distinct(self):
        b, f, c, ids = inputs()
        before = deepcopy((b, f, c))
        result = render(b, f, c)
        cv = result['current_cv']
        self.assertEqual(cv['scores'], {ids['a']: 4., ids['b']: 3.})
        self.assertEqual(cv['fold_wins'], {ids['a']: 0, ids['b']: 3})
        self.assertEqual(cv['lower_error_on_every_fold'], [ids['b']])
        h = result['history']['last_4_origins']
        self.assertEqual(h['scores'], {ids['a']: 1., ids['b']: 2.})
        self.assertEqual(h['days_since_latest_matched_origin'], 59.)
        self.assertTrue(h['single_origin_only'])
        self.assertTrue(h['current_cv_and_history_winners_disagree'])
        self.assertEqual(h['n'], 2)
        self.assertEqual(cv['n'], 6)
        self.assertEqual(result['provider_calls'], 0)
        self.assertFalse(result['forecast_selection_made'])
        self.assertEqual((b, f, c), before)

    def test_tied_current_means_are_not_an_opposing_winner(self):
        b, f, c, ids = inputs()
        for fold, score in zip(c['runs'][1]['folds'], [4., 4., 4.]): fold['rmsle'] = score
        out = render(b, f, c)
        self.assertEqual(out['current_cv']['lowest_mean_config_ids'], sorted(ids.values()))
        self.assertEqual(out['current_cv']['fold_ties'], 1)
        self.assertFalse(out['history']['last_4_origins']['current_cv_and_history_winners_disagree'])

    def test_no_matched_origins_are_unknown_not_zero_or_disagreement(self):
        b, f, c, _ = inputs((), ())
        h = render(b, f, c)['history']['last_4_origins']
        self.assertEqual(h['scores'], {})
        self.assertEqual(h['matched_origins'], 0)
        self.assertIsNone(h['days_since_latest_matched_origin'])
        self.assertIsNone(h['current_cv_and_history_winners_disagree'])

    def test_unseen_configuration_does_not_inherit_old_history(self):
        b, f, c, _ = inputs()
        run = c['runs'][0]
        run['config'] = {'model': 'new'}
        run['config_id'] = hashlib.sha256(json.dumps(run['config'], sort_keys=True).encode()).hexdigest()[:16]
        run.update(provider='new_provider', revision='new_revision')
        out = render(b, f, c)
        self.assertEqual(out['history_status'], 'configuration_not_in_historical_catalog')
        self.assertEqual(out['history'], {})

    def test_known_pair_missing_from_page_is_not_no_evidence(self):
        b, f, c, _ = inputs()
        b['cards'] = []; f['cards'] = []
        f.update(offset=1, total_pairs=1, next_offset=None)
        b['pagination'].update(shown_pairs=0, offset=1, all_pairs_included=False)
        reseal(b, f)
        out = render(b, f, c)
        self.assertEqual(out['history_status'], 'pair_not_on_supplied_page')
        self.assertEqual(out['history'], {})

    def test_current_task_and_version_identity_must_match(self):
        mutations = [lambda c: c['query'].update(unit='kg'),
                     lambda c: c['query'].update(series_id='other'),
                     lambda c: c['query'].update(horizon=3),
                     lambda c: c['runs'][0].update(revision='other'),
                     lambda c: c['runs'][0].update(provider='other'),
                     lambda c: c['runs'][0].update(config_id='forged'),
                     lambda c: c['runs'].append(deepcopy(c['runs'][0])),
                     lambda c: c['runs'][1].update(config_id=c['runs'][0]['config_id'])]
        for change in mutations:
            b, f, c, _ = inputs(); change(c)
            with self.subTest(change=change), self.assertRaises(ValueError): render(b, f, c)

    def test_current_folds_must_be_matched_complete_past_and_distinct(self):
        mutations = [lambda r: r['folds'].pop(),
                     lambda r: r['folds'][0].update(target_end='2026-03-02T00:00:00+00:00'),
                     lambda r: r['folds'][0].update(target_end=r['folds'][0]['origin']),
                     lambda r: r['folds'][0].update(origin='2026-02-02T00:00:00+00:00'),
                     lambda r: r['folds'][0].update(target_end='2026-02-04T00:00:00+00:00'),
                     lambda r: r['folds'][0].update(origin='2026-02-01'),
                     lambda r: r['folds'][0].update(n=1),
                     lambda r: r['folds'][0].update(n=True),
                     lambda r: r['folds'][0].update(rmsle=float('nan')),
                     lambda r: r['folds'][0].update(rmsle=-1.),
                     lambda r: r['folds'][0].update(rmsle=True),
                     lambda r: r['folds'][0].update(execution_id=''),
                     lambda r: r['folds'][0].update(execution_id='b-0')]
        for change in mutations:
            b, f, c, _ = inputs(); change(c['runs'][0])
            with self.subTest(change=change), self.assertRaises(ValueError): render(b, f, c)

    def test_historical_source_and_summary_are_authenticated(self):
        b, f, c, _ = inputs()
        with self.assertRaises(ValueError): current_history_contrast(b, encode(f)+b' ', c)
        b['cards'][0]['windows']['last_4_origins']['matched_origins'] = 2
        with self.assertRaises(ValueError): render(b, f, c)


if __name__ == '__main__': unittest.main()
