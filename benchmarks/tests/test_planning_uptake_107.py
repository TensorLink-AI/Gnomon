import json
import unittest

from benchmarks.ledger_optimization.planning_uptake_107 import config_id, summarize_events


class UptakeTests(unittest.TestCase):
    query = {'series_id': 's', 'unit': 'widgets', 'horizon': 14, 'origin': '2022-01-01T00:00:00Z'}
    config = {'model': 'ridge', 'window': 365, 'lags': 14, 'alpha': 10.0}

    def grade(self, **updates):
        return {'arm': 'ledger', 'series_id': 's', 'origin': self.query['origin'], 'round': 4,
                'valid': True, 'fallback_used': False, 'config': self.config, **updates}

    def requested(self, key, operation, **kwargs):
        return {'tool': 'lab', 'stage': 'requested', 'tool_call_id': key,
                'raw_arguments': json.dumps({'operation': operation, **kwargs})}

    def returned(self, key, *, result=None, view=None, ok=True):
        reply = {'status': 'ok' if ok else 'error', 'result': result or {}}
        if view is not None: reply['recipe_plan'] = view
        return {'tool': 'lab', 'stage': 'returned', 'tool_call_id': key,
                'result': {'status': 'ok', 'result': {'stdout': json.dumps(reply)}}}

    def view(self):
        return {'query': self.query, 'status': 'optional_experiments_available',
                'next_call': {'arguments': {'operation': 'start'}},
                'recipes': [{'config_id': config_id(self.config), 'arguments': {'operation': 'backtest', 'config': self.config}}]}

    def backtest(self):
        return {'config': self.config, 'config_id': config_id(self.config), 'reused': False}

    def test_actual_suggestion_checkpoint_test_and_selection(self):
        events = [self.requested('r', 'review'), self.returned('r', view=self.view()),
                  self.requested('s', 'start'), self.returned('s', result={'checkpoint_id': 'c'}),
                  self.requested('b', 'backtest', config=self.config), self.returned('b', result=self.backtest())]
        row = summarize_events(events, self.query, self.grade())
        self.assertEqual(row['suggested_checkpoints_published'], 1)
        self.assertTrue(row['selected_previously_proposed_and_backtested'])
        self.assertEqual(row['execution_traces'][0]['first_proposal_event'], 1)

    def test_proposal_in_own_response_is_not_prior_exposure(self):
        events = [self.requested('b', 'backtest', config=self.config),
                  self.returned('b', result=self.backtest(), view=self.view())]
        row = summarize_events(events, self.query, self.grade())
        self.assertEqual(row['proposed_then_successfully_backtested_configurations'], [])
        self.assertFalse(row['selected_previously_proposed_and_backtested'])

    def test_failed_call_is_requested_but_not_successful_uptake(self):
        events = [self.requested('r', 'review'), self.returned('r', view=self.view()),
                  self.requested('b', 'backtest', config=self.config), self.returned('b', ok=False)]
        row = summarize_events(events, self.query, self.grade())
        self.assertEqual(len(row['proposed_then_requested_configurations']), 1)
        self.assertEqual(row['proposed_then_successfully_backtested_configurations'], [])

    def test_inflight_request_not_influenced_by_later_parallel_reply(self):
        events = [self.requested('r', 'review'), self.requested('b', 'backtest', config=self.config),
                  self.returned('r', view=self.view()), self.returned('b', result=self.backtest())]
        self.assertEqual(summarize_events(events, self.query, self.grade())['proposed_then_successfully_backtested_configurations'], [])

    def test_duplicate_return_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            summarize_events([self.requested('r', 'review'), self.returned('r'), self.returned('r')], self.query, self.grade())

    def test_wrong_task_and_control_exposure_rejected(self):
        view = self.view(); view['query'] = {**self.query, 'series_id': 'other'}
        with self.assertRaisesRegex(ValueError, 'another task'):
            summarize_events([self.requested('r', 'review'), self.returned('r', view=view)], self.query, self.grade())
        with self.assertRaisesRegex(ValueError, 'Control'):
            summarize_events([self.requested('r', 'review'), self.returned('r', view=self.view())], self.query, self.grade(arm='plain'))

    def test_no_plan_control_remains_in_denominator(self):
        row = summarize_events([self.requested('b', 'backtest', config=self.config), self.returned('b', result=self.backtest())], self.query, self.grade(arm='plain'))
        self.assertEqual(row['plan_views'], 0)
        self.assertEqual(len(row['successful_explicit_backtest_configurations']), 1)

    def test_cold_compact_plan_has_no_action_or_query(self):
        view = {'status': 'no_historical_catalog', 'recipes': [], 'next_call': None}
        row = summarize_events([self.requested('r', 'review'), self.returned('r', view=view)], self.query, self.grade())
        self.assertEqual(row['plan_views'], 1)
        self.assertEqual(row['historical_plan_views'], 0)
        self.assertEqual(row['proposed_configurations'], [])
        view['next_call'] = {'arguments': {'operation': 'start'}}
        with self.assertRaisesRegex(ValueError, 'another task'):
            summarize_events([self.requested('r', 'review'), self.returned('r', view=view)], self.query, self.grade())

    def test_validated_response_canonicalizes_requested_config(self):
        submitted = {**self.config, 'alpha': 10}
        events = [self.requested('r', 'review'), self.returned('r', view=self.view()),
                  self.requested('b', 'backtest', config=submitted), self.returned('b', result=self.backtest())]
        row = summarize_events(events, self.query, self.grade())
        self.assertEqual(row['proposed_then_requested_configurations'], [config_id(self.config)])
        self.assertTrue(row['selected_previously_proposed_and_backtested'])

    def test_unreturned_and_fallback_not_credited_as_completion(self):
        events = [self.requested('r', 'review'), self.returned('r', view=self.view()),
                  self.requested('b', 'backtest', config=self.config)]
        row = summarize_events(events, self.query, self.grade(fallback_used=True))
        self.assertEqual(row['unreturned_requests'], ['b'])
        self.assertFalse(row['selected_previously_proposed_and_backtested'])


if __name__ == '__main__': unittest.main()
