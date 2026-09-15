from copy import deepcopy
import hashlib
import json
import unittest

from benchmarks.ledger_optimization.planning_sequence_107 import planning_sequence
from benchmarks.ledger_optimization.planning_recipes_106 import planning_recipes
from benchmarks.tests.test_planning_recipes_106 import fixture
from benchmarks.tests.test_paired_consistency_098 import encode, reseal


def inputs():
    b, f, c, ids = fixture()
    c['checkpoint_available'] = False
    c['budget'].update(agent_requests_remaining=15, exploration_requests_remaining=11)
    c['initial_checkpoint'] = {'config': {'model': 'seasonal', 'season': 7}, 'max_numerical_attempts': 4}
    return b, f, c, ids


def render(b, f, c):
    return planning_sequence(b, encode(f), c, deepcopy)


class PlanningSequenceTests(unittest.TestCase):
    def test_missing_checkpoint_is_prerequisite_not_fabricated_state(self):
        b, f, c, _ = inputs(); original = deepcopy((b, f, c))
        r = render(b, f, c)
        self.assertEqual(r['next_call']['arguments'], {'operation': 'start'})
        self.assertEqual(r['next_call']['valid_for_task'], c['query'])
        self.assertEqual(r['observed_state'], c)
        self.assertFalse(r['state_projected_as_executed'])
        self.assertFalse(r['forecast_selection_made'])
        self.assertEqual(r['numerical_attempts'], 0)
        self.assertEqual(r['sequential_capacity_upper_bound'], 1)
        for p in r['recipes']:
            self.assertFalse(p['next_call']['admissible_now'])
            self.assertTrue(p['next_call']['conditional_budget_feasible'])
            self.assertIn('fresh_time_phase_and_budget_admission', p['next_call']['requires'])
        self.assertEqual((b, f, c), original)

    def test_sequence_charges_initial_batch_once_plus_final_reserve(self):
        for remaining, expected in [(8, 0), (9, 1), (12, 1), (13, 2)]:
            b, f, c, _ = inputs(); c['budget']['numerical_remaining'] = remaining
            r = render(b, f, c)
            self.assertEqual(r['sequential_capacity_upper_bound'], expected)
            self.assertEqual(r['next_call'] is not None, expected > 0)
            self.assertEqual(r['checkpoint_max_numerical_attempts'], 4)

    def test_interaction_and_phase_limits(self):
        for total, exploration, phase, expected in [(2, 2, 'exploration', 0),
                (15, 1, 'exploration', 0), (15, 11, 'selection', 0),
                (3, 2, 'exploration', 1), (4, 3, 'exploration', 2)]:
            b, f, c, _ = inputs()
            c['budget'].update(numerical_remaining=60, agent_requests_remaining=total,
                               exploration_requests_remaining=exploration, phase=phase)
            self.assertEqual(render(b, f, c)['sequential_capacity_upper_bound'], expected)

    def test_existing_checkpoint_preserves_direct_calls(self):
        b, f, c, _ = inputs(); c['checkpoint_available'] = True
        del c['initial_checkpoint']
        direct = planning_recipes(b, encode(f), c, deepcopy); r = render(b, f, c)
        self.assertIsNone(r['prerequisite'])
        self.assertEqual(r['recipes'], direct['recipes'])
        self.assertEqual(r['sequential_capacity_upper_bound'], direct['sequential_capacity_at_snapshot'])
        self.assertEqual(r['checkpoint_max_numerical_attempts'], 0)

    def test_invalid_or_missing_prerequisite_metadata_is_not_assumed(self):
        b, f, c, _ = inputs(); del c['initial_checkpoint']
        with self.assertRaises(KeyError): render(b, f, c)
        for field, value in [('config', {'model': 'seasonal', 'season': 14}),
                             ('max_numerical_attempts', True), ('max_numerical_attempts', 0)]:
            b, f, c, _ = inputs(); c['initial_checkpoint'][field] = value
            with self.assertRaises(ValueError): render(b, f, c)
        b, f, c, _ = inputs(); c['budget']['agent_requests_remaining'] = True
        with self.assertRaises(ValueError): render(b, f, c)

    def test_no_supported_recipe_does_not_claim_meaningful_sequence(self):
        b, f, c, _ = inputs()
        c['executed_configurations'] = [dict(v, config_id=k) for k, v in b['configuration_index'].items()]
        r = render(b, f, c)
        self.assertEqual(r['recipes'], [])
        self.assertIsNone(r['next_call'])
        self.assertEqual(r['prerequisite']['blocked_reason'], 'no_supported_untried_recipe')

    def test_initial_baseline_is_excluded_without_claiming_it_executed(self):
        b, f, c, ids = inputs()
        old = ids['a']; config = c['initial_checkpoint']['config']
        cid = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        for item in (b, f):
            row = item['configuration_index'].pop(old); row['config'] = deepcopy(config)
            item['configuration_index'][cid] = row
        f['cards'][0]['left_config_id'] = cid
        b['cards'][0]['config_ids'][0] = cid
        for w in b['cards'][0]['windows'].values():
            if 'same_as' not in w:
                w['scores'][cid] = w['scores'].pop(old)
                w['lowest_error_config_ids'] = [cid if k == old else k for k in w['lowest_error_config_ids']]
        reseal(b, f)
        result = render(b, f, c)
        self.assertEqual([r['config_id'] for r in result['recipes']], [ids['b']])
        self.assertEqual(result['observed_state']['executed_configurations'], [])
        self.assertEqual(result['initial_checkpoint_config_id'], cid)

    def test_changed_task_and_provider_identity_rejected(self):
        b, f, c, _ = inputs(); c['query']['horizon'] += 1
        with self.assertRaises(ValueError): render(b, f, c)
        b, f, c, ids = inputs()
        c['executed_configurations'] = [dict(b['configuration_index'][ids['a']], config_id=ids['a'], revision='changed')]
        with self.assertRaises(ValueError): render(b, f, c)


if __name__ == '__main__':
    unittest.main()
