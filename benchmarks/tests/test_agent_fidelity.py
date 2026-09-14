import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization.agent_fidelity import validate, grade, run, sha, save, MODEL


def answer():
    return {'comparisons': [{'pair_index': 0, 'windows': [
        {'window': name, 'matched_origins': 2, 'n': 4, 'start': '2026-01-01', 'end': '2026-01-02',
         'scores': [{'config_id': 'a', 'rmsle': 0.2}, {'config_id': 'b', 'rmsle': 0.2}],
         'lowest_error_config_ids': ['a', 'b']} for name in ('last_4_origins', 'lifetime')],
        'recent_lifetime_disagreement': False}],
        'pagination': {'total_pairs': 3, 'shown_pairs': 1, 'all_pairs_included': False, 'next_offset': 1},
        'global_ranking_supported': False, 'provider_calls': 0}


class Reply:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps({'model': MODEL, 'choices': [{'message': self.message}],
                           'usage': {'prompt_tokens': 200, 'completion_tokens': 100, 'total_tokens': 300}}).encode()


def tool_message(value):
    return {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'call1', 'type': 'function', 'function': {
            'name': 'submit_evidence', 'arguments': json.dumps(value)}}]}


class FidelityTest(unittest.TestCase):
    def test_exact_ties_and_reordering(self):
        ref = answer()
        got = copy.deepcopy(ref)
        got['comparisons'][0]['windows'].reverse()
        for window in got['comparisons'][0]['windows']:
            window['scores'].reverse()
            window['lowest_error_config_ids'].reverse()
        self.assertTrue(grade(got, ref)['all_facts_correct'])

    def test_wrong_rank_count_or_global_scope_fails_without_type_failure(self):
        for key in ('count', 'tie', 'global', 'metric'):
            got = answer()
            if key == 'count':
                got['comparisons'][0]['windows'][0]['n'] = 2
            elif key == 'tie':
                got['comparisons'][0]['windows'][0]['lowest_error_config_ids'] = ['a']
            elif key == 'metric':
                got['comparisons'][0]['windows'][0]['scores'][0]['rmsle'] = 2
            else:
                got['global_ranking_supported'] = True
            validate(got)
            self.assertFalse(grade(got, answer())['all_facts_correct'])

    def test_missing_pairs_preserve_field_denominator(self):
        got = answer()
        complete = grade(got, answer())
        got['comparisons'] = []
        incomplete = grade(got, answer())
        self.assertEqual(len(complete['checks']), len(incomplete['checks']))
        self.assertFalse(incomplete['all_facts_correct'])

    def test_reject_unknown_fields_bool_counts_and_nan(self):
        got = answer()
        got['provider_calls'] = True
        with self.assertRaises(ValueError):
            validate(got)
        got = answer()
        got['extra'] = 1
        with self.assertRaises(ValueError):
            validate(got)
        got = answer()
        got['comparisons'][0]['windows'][0]['scores'][0]['rmsle'] = float('nan')
        with self.assertRaises(ValueError):
            validate(got)

    def execute(self, replies):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        prepared = root / 'prepared'
        prepared.mkdir()
        save(prepared / 'cases.json', [{'case_id': 'c', 'question': {},
                                        'formats': {'brief': {'only': 'visible_evidence'}}, 'expected': answer()}])
        save(prepared / 'schedule.json', [{'case_id': 'c', 'seed': 7, 'format': 'brief', 'job_id': 'j'}])
        save(prepared / 'manifest.json', {'code': {}, 'cases_sha256': sha(prepared / 'cases.json'),
                                         'schedule_sha256': sha(prepared / 'schedule.json')})
        with patch('benchmarks.ledger_optimization.agent_fidelity.key', return_value='SYNTHETIC_TEST_KEY'), \
             patch('urllib.request.urlopen', side_effect=replies) as call, patch('builtins.print'):
            run(prepared, root / 'output')
        result = json.loads((root / 'output/j/result.json').read_text())
        sent = [json.loads(c.args[0].data) for c in call.call_args_list]
        return root, result, sent

    def test_success_needs_no_final_chat_and_no_reference_is_sent(self):
        root, result, sent = self.execute([Reply(tool_message(answer()))])
        self.assertTrue(result['all_facts_correct'])
        self.assertEqual(result['requests'], 1)
        user = json.loads(sent[0]['messages'][1]['content'])
        self.assertEqual(set(user), {'question', 'review'})
        self.assertNotIn('expected', user)
        self.assertEqual(sent[0]['model'], MODEL)
        self.assertNotIn('SYNTHETIC_TEST_KEY', (root / 'output/j/request-0.json').read_text())

    def test_syntax_correction_bounded_and_contains_no_factual_feedback(self):
        root, result, sent = self.execute([Reply({'role': 'assistant', 'content': 'prose'}),
                                          Reply(tool_message(answer()))])
        self.assertTrue(result['all_facts_correct'])
        self.assertEqual(result['requests'], 2)
        self.assertEqual(result['syntax_corrections'], 1)
        self.assertIn('exactly one submit_evidence', sent[1]['messages'][-1]['content'])

    def test_factual_error_is_scored_not_repaired(self):
        wrong = answer()
        wrong['provider_calls'] = 1
        _, result, sent = self.execute([Reply(tool_message(wrong))])
        self.assertTrue(result['schema_completed'])
        self.assertFalse(result['all_facts_correct'])
        self.assertEqual(len(sent), 1)

    def test_service_error_recorded_without_free_retry(self):
        _, result, sent = self.execute([TimeoutError('synthetic timeout')])
        self.assertTrue(result['service_error'])
        self.assertEqual(result['requests'], 1)
        self.assertFalse(result['schema_completed'])


if __name__ == '__main__':
    unittest.main()
