import copy
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.ledger_optimization.agent_fidelity import MODEL, SYSTEM, TOOL, grade, save, sha, compact_bytes
from benchmarks.ledger_optimization.agent_fidelity_report import independently_correct, report
from benchmarks.tests.test_agent_fidelity import answer, tool_message


class FidelityReportTest(unittest.TestCase):
    def test_independent_grader_agrees_for_missing_duplicate_wrong_and_reordered_facts(self):
        original = answer()
        variants = [copy.deepcopy(original) for _ in range(7)]
        variants[0]['comparisons'] = []
        variants[1]['comparisons'].append(copy.deepcopy(original['comparisons'][0]))
        variants[2]['comparisons'][0]['windows'][0]['scores'][0]['rmsle'] += 0.01
        variants[3]['comparisons'][0]['windows'][0]['lowest_error_config_ids'] = ['a']
        variants[4]['comparisons'][0]['windows'].reverse()
        variants[5]['pagination']['all_pairs_included'] = True
        variants[6]['comparisons'][0]['windows'][0]['scores'].reverse()
        for variant in variants:
            self.assertEqual(independently_correct(variant, original), grade(variant, original)['all_facts_correct'])

    def test_raw_response_audit_usage_and_incomplete_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, runs = root / 'prepared', root / 'run'
            prepared.mkdir()
            runs.mkdir()
            case = {'case_id': 'case', 'formats': {'original': {}, 'brief': {}}, 'question': {}, 'expected': answer()}
            schedule = [{'case_id': 'case', 'format': f, 'seed': 7, 'job_id': f} for f in ('original', 'brief')]
            save(prepared / 'cases.json', [case])
            save(prepared / 'schedule.json', schedule)
            save(prepared / 'manifest.json', {'code': {}, 'cases_sha256': sha(prepared / 'cases.json'),
                                             'schedule_sha256': sha(prepared / 'schedule.json')})
            for job in schedule:
                folder = runs / job['job_id']
                folder.mkdir()
                usage = {'prompt_tokens': 100 if job['format'] == 'original' else 40,
                         'completion_tokens': 20, 'total_tokens': 120 if job['format'] == 'original' else 60}
                save(folder / 'request-0.json', {'model': MODEL, 'seed': 7, 'temperature': .2, 'max_tokens': 2048,
                     'tools': [TOOL], 'messages': [{'role': 'system', 'content': SYSTEM},
                       {'role': 'user', 'content': compact_bytes({'question': {}, 'review': {}}).decode()}]})
                save(folder / 'started-0.json', {'unix_time': 1})
                save(folder / 'response-0.json', {'model': MODEL, 'choices': [{'message': tool_message(answer())}], 'usage': usage})
                save(folder / 'timing-0.json', {'wall_seconds': .5})
                save(folder / 'result.json', {**job, 'requests': 1, 'usage': [usage], 'service_error': False,
                      'syntax_corrections': 0, 'schema_completed': True, 'answer': answer(), **grade(answer(), answer())})
            summary = report(prepared, runs, root / 'report')
            self.assertFalse(summary['completed'])
            self.assertFalse(summary['adoption_screen_passed'])
            self.assertEqual(summary['formats']['brief']['prompt_tokens'], 40)
            self.assertEqual(summary['upstream_requests'], 2)
            self.assertEqual(summary['paired_sessions'], 1)
            self.assertIsNone(summary['billing_cost'])


if __name__ == '__main__':
    unittest.main()
