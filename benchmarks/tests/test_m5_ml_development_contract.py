"""Synthetic development tasks only; never read the real archive or final data."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization import m5_ml_development_contract as c


def fixture():
    a = c.adapter
    times = [datetime(2013, 5, 26, tzinfo=timezone.utc)+timedelta(days=n)
             for n in range(1094)]
    manifest = {'splits': {'development': [], 'reserved': []}}
    jobs = {}
    for n in range(8):
        values = [float(2+n+i % 7) for i in range(1094)]
        row = {'series_id': f'dev-{n}', 'store_id': f'dev-store-{n//4}',
               'item_id': f'dev-item-{n}',
               'initial_history_sha256': a.value_hash(values[364:730])}
        manifest['splits']['development'].append(row)
        jobs[row['series_id']] = a.build_series_jobs(row, values, times)
    for n in range(24):
        manifest['splits']['reserved'].append({
            'series_id': f'final-{n}', 'store_id': f'final-store-{n//3}',
            'item_id': f'final-item-{n}', 'initial_history_sha256': '0'*64})
    return manifest, jobs


class DevelopmentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.jobs = fixture()

    def test_exact_cohort_dimensions_without_execution_permission(self):
        result = c.describe_cohort(self.manifest, self.jobs)
        self.assertEqual(result['decisions_per_seed'],
                         {'pilot': 72, 'continuation': 552, 'total': 624})
        self.assertEqual((result['series'], result['stores'], result['host_tasks']), (8, 2, 208))
        self.assertFalse(result['execution_authorized'])
        self.assertFalse(result['final_gate_opened'])
        self.assertFalse(result['provenance_authenticated'])
        for series in self.jobs:
            cases = [r for r in result['cases'] if r['series_id'] == series]
            self.assertEqual([r['round'] for r in cases], list(range(26)))
            self.assertEqual(sum(r['stage'] == 'pilot' for r in cases), 3)
            for n, row in enumerate(cases):
                expected = datetime(2015, 5, 25, tzinfo=timezone.utc)+timedelta(days=14*n)
                self.assertEqual(row['origin'], expected.isoformat())
                self.assertEqual(row['outcome_recorded_at'], (expected+timedelta(days=14)).isoformat())
                self.assertNotIn('request', row)
                self.assertNotIn('actual', row)

    def test_rejects_missing_extra_and_reserved_series_before_values(self):
        class Poison:
            def __iter__(self):
                raise AssertionError('Unexpected numerical consumption')
        wrong = dict(self.jobs)
        wrong.pop('dev-0')
        for jobs in (wrong, {**wrong, 'final-0': Poison()}, {**self.jobs, 'final-0': Poison()}):
            with self.assertRaisesRegex(ValueError, 'exactly the eight'):
                c.describe_cohort(self.manifest, jobs)

    def test_task_identity_and_order_cannot_be_substituted(self):
        for mutation in (
                lambda rows: rows.pop(),
                lambda rows: rows.reverse(),
                lambda rows: rows[0].update(round=False),
                lambda rows: rows[0].update(series_id='dev-1'),
                lambda rows: rows[0].update(unexpected='field')):
            jobs = deepcopy(self.jobs)
            mutation(jobs['dev-0'])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                c.describe_cohort(self.manifest, jobs)

    def test_history_actuals_covariates_and_visibility_must_agree(self):
        mutations = [
            lambda rows: rows[1]['request']['history'].__setitem__(20, 9999.),
            lambda rows: rows[2]['actual'].__setitem__(0, 9999.),
            lambda rows: rows[1]['request']['past_covariates'][0].__setitem__(0, 1.),
            lambda rows: rows[1]['request'].update(known_time_cutoff=rows[0]['origin']),
            lambda rows: rows[1]['request'].update(recorded_time_cutoff=rows[0]['origin']),
            lambda rows: rows[1].update(outcome_recorded_at=rows[1]['origin']),
            lambda rows: rows[1]['request'].update(unit='different'),
            lambda rows: rows[1]['request'].update(horizon=7),
            lambda rows: rows[1]['future_timestamps'].pop(),
        ]
        for mutation in mutations:
            jobs = deepcopy(self.jobs)
            mutation(jobs['dev-0'])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                c.describe_cohort(self.manifest, jobs)

    def test_invalid_numbers_are_not_coerced_into_observations(self):
        for value in (True, -1., float('nan'), float('inf')):
            jobs = deepcopy(self.jobs)
            jobs['dev-0'][0]['actual'][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                c.describe_cohort(self.manifest, jobs)

    def test_individually_valid_series_cannot_shift_its_calendar(self):
        jobs = dict(self.jobs)
        records = jobs['dev-7']
        values = records[0]['request']['history']+[v for r in records for v in r['actual']]
        times = records[0]['request']['timestamps']+[v for r in records for v in r['future_timestamps']]
        shifted = [datetime.fromisoformat(t)+timedelta(days=1) for t in times]
        jobs['dev-7'] = c.adapter.build_series_jobs(self.manifest['splits']['development'][7], values, shifted)
        with self.assertRaisesRegex(ValueError, 'same origin/target grid'):
            c.describe_cohort(self.manifest, jobs)

    def test_mapping_order_invariance_and_no_input_mutation(self):
        before = json.dumps([self.manifest, self.jobs], sort_keys=True)
        altered = deepcopy(self.manifest)
        altered['splits']['development'].reverse()
        self.assertEqual(c.describe_cohort(self.manifest, self.jobs),
                         c.describe_cohort(altered, dict(reversed(list(self.jobs.items())))))
        self.assertEqual(before, json.dumps([self.manifest, self.jobs], sort_keys=True))

    def test_authentication_precedes_parsing_and_is_not_dispatch_authority(self):
        manifest = json.dumps(self.manifest).encode()
        jobs = json.dumps(self.jobs).encode()
        with patch.object(c.adapter, 'MANIFEST_SHA', hashlib.sha256(manifest).hexdigest()), \
             patch.object(c, 'DEVELOPMENT_JOBS_SHA', hashlib.sha256(jobs).hexdigest()):
            result = c.authenticated_contract(manifest, jobs)
            self.assertTrue(result['provenance_authenticated'])
            self.assertFalse(result['execution_authorized'])
            self.assertFalse(result['final_gate_opened'])
            with patch.object(c.json, 'loads', side_effect=AssertionError('Parsed before authentication')):
                for m, j in ((manifest+b' ', jobs), (manifest, jobs+b' ')):
                    with self.assertRaises(ValueError):
                        c.authenticated_contract(m, j)
        with self.assertRaisesRegex(ValueError, 'Already-loaded bytes'):
            c.authenticated_contract('/some/file', jobs)

    def test_path_like_series_identity_is_rejected(self):
        manifest = deepcopy(self.manifest)
        manifest['splits']['development'][0]['series_id'] = '../escape'
        jobs = dict(self.jobs)
        jobs['../escape'] = jobs.pop('dev-0')
        with self.assertRaisesRegex(ValueError, 'task-directory'):
            c.describe_cohort(manifest, jobs)


if __name__ == '__main__':
    unittest.main()
