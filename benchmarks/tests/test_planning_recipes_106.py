from copy import deepcopy
import hashlib
import json
import unittest

from benchmarks.ledger_optimization.planning_recipes_106 import planning_recipes
from benchmarks.tests.test_current_history_contrast_100 import inputs
from benchmarks.tests.test_paired_consistency_098 import encode, reseal


def fixture(count=4):
    brief, full, _, ids = inputs(tuple(range(1, count+1)), (9.,)*count)
    current = {'query': deepcopy(brief['query']), 'executed_configurations': [],
               'checkpoint_available': True,
               'budget': {'numerical_remaining': 9, 'fresh_backtest_fits': 4,
                          'reserved_final_fits': 1, 'phase': 'exploration'}}
    return brief, full, current, ids


def render(b, f, c):
    # Synthetic validator exercises the injected common-validator contract.
    # Production replay must use the frozen lab validator, not this fixture.
    return planning_recipes(b, encode(f), c, lambda config: deepcopy(config))


def extend_pair(b, f, letter):
    old = f['cards'][0]['right_config_id']
    row = {'config': {'model': letter}, 'provider': 'provider_'+letter, 'revision': 'rev_'+letter}
    cid = hashlib.sha256(json.dumps(row['config'], sort_keys=True).encode()).hexdigest()[:16]
    for item in (b, f): item['configuration_index'][cid] = deepcopy(row)
    fc, bc = deepcopy(f['cards'][0]), deepcopy(b['cards'][0])
    old_provider = f['configuration_index'][old]['provider']
    fc['right_config_id'] = cid
    fc['providers'].pop(old_provider); fc['providers'][row['provider']] = row['revision']
    for w in fc['windows'].values():
        for models in [w['models'], *(o['models'] for o in w['origins'])]:
            for m in models:
                if m['provider'] == old_provider:
                    m.update(provider=row['provider'], revision=row['revision'])
    index = len(f['cards']); f['cards'].append(fc)
    bc['config_ids'][1] = cid; bc['evidence_pointer'] = f'/cards/{index}'
    for name, w in bc['windows'].items():
        w['evidence_pointer'] = f'/cards/{index}/windows/{name}'
        if 'same_as' not in w:
            w['scores'][cid] = w['scores'].pop(old)
            w['lowest_error_config_ids'] = [cid if c == old else c for c in w['lowest_error_config_ids']]
    b['cards'].append(bc)
    f['total_pairs'] = len(f['cards'])
    b['pagination'].update(shown_pairs=len(b['cards']), total_pairs=len(b['cards']))
    reseal(b, f)


class PlanningRecipesTests(unittest.TestCase):
    def test_no_current_anchor_required_and_exact_task_config_preserved(self):
        b, f, c, ids = fixture(); before = deepcopy((b, f, c))
        result = render(b, f, c)
        self.assertEqual(len(result['recipes']), 2)
        self.assertEqual(result['sequential_capacity_at_snapshot'], 2)
        for row in result['recipes']:
            self.assertEqual(row['historical_origins_with_paired_evidence'], 4)
            self.assertEqual(row['next_call']['arguments'],
                             {'operation': 'backtest', 'config': b['configuration_index'][row['config_id']]['config']})
            self.assertEqual(row['next_call']['valid_for_task'], c['query'])
            self.assertTrue(row['next_call']['admissible_now'])
        self.assertEqual((b, f, c), before)
        self.assertFalse(result['forecast_selection_made'])
        self.assertEqual(result['provider_calls'], 0)

    def test_aliases_pairs_and_repeated_origins_do_not_inflate_support(self):
        b, f, c, ids = fixture()
        for letter in ('c', 'd', 'e'): extend_pair(b, f, letter)
        result = render(b, f, c)
        self.assertEqual(result['eligible_recipes_on_page'], 5)
        self.assertEqual(len(result['recipes']), 3)
        self.assertEqual(result['sequential_capacity_at_snapshot'], 2)
        self.assertEqual([x['config_id'] for x in result['recipes']], sorted(b['configuration_index'])[:3])
        self.assertTrue(all(x['historical_origins_with_paired_evidence'] == 4 for x in result['recipes']))

    def test_already_executed_recipe_excluded(self):
        b, f, c, ids = fixture()
        c['executed_configurations'] = [dict(b['configuration_index'][ids['a']], config_id=ids['a'])]
        result = render(b, f, c)
        self.assertEqual([x['config_id'] for x in result['recipes']], [ids['b']])
        self.assertEqual(result['already_executed_catalog_recipes'], 1)

    def test_insufficient_support_includes_empty_and_three_origins(self):
        for n in (0, 1, 3):
            b, f, c, _ = fixture(n)
            result = render(b, f, c)
            self.assertEqual(result['recipes'], [])
            self.assertEqual(result['insufficient_support_untried_recipes'], 2)
            self.assertEqual(result['sequential_capacity_at_snapshot'], 0)

    def test_scores_cannot_change_eligibility_or_order(self):
        b, f, c, _ = fixture()
        before = render(b, f, c)
        for data in f['cards'][0]['windows'].values():
            for models in [data['models'], *(o['models'] for o in data['origins'])]:
                for m in models: m['rmsle'] = 5.
        for w in b['cards'][0]['windows'].values():
            if 'same_as' not in w:
                w['scores'] = dict.fromkeys(w['scores'], 5.)
                w['lowest_error_config_ids'] = list(w['scores'])
        reseal(b, f)
        after = render(b, f, c)
        self.assertEqual(before['recipes'], after['recipes'])
        self.assertNotEqual(before['full_evidence']['sha256'], after['full_evidence']['sha256'])

    def test_shared_budget_checkpoint_and_phase_are_explicit(self):
        for checkpoint, phase, remaining, reason in [(False, 'exploration', 20, 'checkpoint_required'),
                (True, 'selection', 20, 'selection_phase'),
                (True, 'exploration', 4, 'insufficient_fits_with_final_reserve')]:
            b, f, c, _ = fixture(); c['checkpoint_available'] = checkpoint
            c['budget'].update(phase=phase, numerical_remaining=remaining)
            result = render(b, f, c)
            self.assertEqual(result['sequential_capacity_at_snapshot'], 0)
            self.assertTrue(all(x['next_call']['runnable'] and not x['next_call']['admissible_now']
                                and x['next_call']['blocked_reason'] == reason for x in result['recipes']))
        b, f, c, _ = fixture(); c['budget']['numerical_remaining'] = 5
        self.assertEqual(render(b, f, c)['sequential_capacity_at_snapshot'], 1)

    def test_unknown_or_malformed_budget_is_not_permission(self):
        for key in ('phase', 'numerical_remaining', 'fresh_backtest_fits', 'reserved_final_fits'):
            b, f, c, _ = fixture(); del c['budget'][key]
            with self.assertRaises(KeyError): render(b, f, c)
        b, f, c, _ = fixture(); c['checkpoint_available'] = None
        with self.assertRaises(ValueError): render(b, f, c)
        c['checkpoint_available'] = True; c['budget']['fresh_backtest_fits'] = True
        with self.assertRaises(ValueError): render(b, f, c)

    def test_changed_task_revision_or_canonical_config_rejected(self):
        b, f, c, ids = fixture(); c['query']['unit'] = 'tonnes'
        with self.assertRaises(ValueError): render(b, f, c)
        b, f, c, ids = fixture()
        c['executed_configurations'] = [dict(b['configuration_index'][ids['a']], config_id=ids['a'], revision='changed')]
        with self.assertRaises(ValueError): render(b, f, c)
        b, f, c, _ = fixture()
        with self.assertRaises(ValueError):
            planning_recipes(b, encode(f), c, lambda config: dict(config, extra='different'))

    def test_edited_evidence_and_nonhistorical_origins_rejected(self):
        b, f, c, _ = fixture(); f['cards'][0]['windows']['lifetime']['origins'][0]['origin'] = c['query']['origin']
        with self.assertRaises(ValueError): render(b, f, c)
        reseal(b, f)
        with self.assertRaises(ValueError): render(b, f, c)

    def test_partial_page_scope_and_next_call_preserved(self):
        b, f, c, _ = fixture()
        f.update(total_pairs=2, next_offset=1)
        b['pagination'].update(total_pairs=2, all_pairs_included=False,
             next_call={'operation': 'review', 'arguments': {'offset': 1, 'limit': 1}, 'valid_for_origin': c['query']['origin']})
        reseal(b, f)
        self.assertEqual(render(b, f, c)['source_pagination'], b['pagination'])


if __name__ == '__main__':
    unittest.main()
