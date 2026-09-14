"""Synthetic-only preparation checks; never access the pinned real archive."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization import m5_ml_adapter as a


class ReservedRow(dict):
    def __getitem__(self, key):
        if key.startswith('d_'):
            raise AssertionError('Reserved sales cell accessed')
        return super().__getitem__(key)


def fixture():
    rows=[];selected=[];reserved=[]
    calendar={f'd_{i}':date(2010,1,1)+timedelta(days=i-1) for i in range(a.FIRST,a.LAST+1)}
    for n in range(8):
        store=f'dev-store-{n//4}';item=f'dev-item-{n}';series=item+'_'+store
        row={'store_id':store,'item_id':item,
             **{f'd_{i}':str(float(n+1+i%7)) for i in range(a.FIRST,a.LAST+1)}}
        prefix=[float(row[f'd_{i}']) for i in range(a.m5_prepare.FIRST,a.m5_prepare.CUTOFF+1)]
        selected.append({'store_id':store,'item_id':item,'series_id':series,
            'initial_history_sha256':a.value_hash(prefix)})
        rows.append(row)
    for n in range(24):
        store=f'reserved-store-{n//3}';item=f'reserved-item-{n}'
        reserved.append({'store_id':store,'item_id':item,'series_id':item+'_'+store})
        rows.append(ReservedRow(store_id=store,item_id=item))
    return rows,calendar,{'splits':{'development':selected,'reserved':reserved}}


class AdapterTests(unittest.TestCase):
    def test_full_fixed_grid_and_reporting_day_covariates(self):
        rows,calendar,manifest=fixture();jobs=a.build_development_jobs(rows,calendar,manifest)
        self.assertEqual(len(jobs),8)
        self.assertEqual(sum(map(len,jobs.values())),208)
        for series,cases in jobs.items():
            for number,job in enumerate(cases):
                req=job['request'];self.assertEqual(job['round'],number)
                self.assertEqual(len(req['history']),730);self.assertEqual(len(req['timestamps']),730)
                self.assertEqual(len(req['past_covariates']),730);self.assertEqual(len(req['future_covariates']),14)
                self.assertEqual(len(job['actual']),14);self.assertEqual(req['series_id'],series)
                origin_day=1577+14*number
                self.assertEqual(date.fromisoformat(req['cutoff'][:10]),calendar[f'd_{origin_day}']+timedelta(days=1))
                self.assertEqual(date.fromisoformat(req['timestamps'][0][:10]),calendar[f'd_{origin_day-729}']+timedelta(days=1))
                self.assertEqual(date.fromisoformat(job['outcome_recorded_at'][:10]),calendar[f'd_{origin_day+14}']+timedelta(days=1))
                self.assertTrue(all(t.endswith('T00:00:00+00:00') for t in req['timestamps']+req['future_timestamps']))
                self.assertGreater(req['future_timestamps'][0],req['cutoff'])
                self.assertEqual(req['known_time_cutoff'],req['cutoff'])
                self.assertEqual(req['recorded_time_cutoff'],req['cutoff'])
                self.assertTrue(all(v[0]==0 for v in req['past_covariates']+req['future_covariates']))
                for offset in (687,701,715):
                    self.assertLessEqual(req['timestamps'][offset+14],req['cutoff'])
            self.assertEqual(cases[0]['request']['timestamps'][0],'2012-04-28T00:00:00+00:00')
            self.assertEqual(cases[0]['origin'],'2014-04-27T00:00:00+00:00')
            self.assertEqual(cases[-1]['future_timestamps'][-1],'2015-04-26T00:00:00+00:00')
            # First future observation is Sunday sales, represented Monday midnight.
            self.assertAlmostEqual(cases[0]['request']['future_covariates'][0][1],-.7818314824680298)
            self.assertAlmostEqual(cases[0]['request']['future_covariates'][0][2],.6234898018587334)

    def test_future_sales_do_not_change_earlier_requests(self):
        rows,calendar,manifest=fixture();before=a.build_development_jobs(rows,calendar,manifest)
        changed=deepcopy(rows);changed[0]['d_1578']='999999.0'
        after=a.build_development_jobs(changed,calendar,manifest)
        series=manifest['splits']['development'][0]['series_id']
        self.assertEqual(before[series][0]['request'],after[series][0]['request'])
        self.assertNotEqual(before[series][0]['actual'],after[series][0]['actual'])
        self.assertNotEqual(before[series][1]['request']['history'],after[series][1]['request']['history'])
        self.assertEqual(before[series][1]['request']['future_covariates'],after[series][1]['request']['future_covariates'])
        for other in set(before)-{series}:self.assertEqual(before[other],after[other])

    def test_extra_early_history_must_exist_no_padding_or_reselection(self):
        rows,calendar,manifest=fixture();del rows[0]['d_848']
        with self.assertRaisesRegex(ValueError,'no replacement'):a.build_development_jobs(rows,calendar,manifest)
        rows,calendar,manifest=fixture();rows[0]['d_848']='nan'
        with self.assertRaisesRegex(ValueError,'no replacement'):a.build_development_jobs(rows,calendar,manifest)

    def test_original_selection_hash_is_retained(self):
        rows,calendar,manifest=fixture();rows[0]['d_1300']='999.0'
        with self.assertRaisesRegex(ValueError,'selection-prefix hash changed'):
            a.build_development_jobs(rows,calendar,manifest)

    def test_missing_identity_or_calendar_is_not_silently_repaired(self):
        rows,calendar,manifest=fixture()
        with self.assertRaisesRegex(ValueError,'Missing selected'):a.build_development_jobs(rows[1:],calendar,manifest)
        calendar['d_1000']=calendar['d_999']
        with self.assertRaisesRegex(ValueError,'consecutive'):a.build_development_jobs(rows,calendar,manifest)

    def test_reserved_overlap_is_rejected(self):
        rows,calendar,manifest=fixture();manifest['splits']['reserved'][0]['store_id']='dev-store-0'
        with self.assertRaisesRegex(ValueError,'overlap'):a.build_development_jobs(rows,calendar,manifest)

    def test_same_visible_inputs_through_frozen_host_in_every_arm(self):
        from benchmarks.hermes_ml_checkpoint_v6 import run
        rows,calendar,manifest=fixture();rows[0]['d_1578']='999999.0'
        series=manifest['splits']['development'][0]['series_id']
        job=a.build_development_jobs(rows,calendar,manifest)[series][0]
        with tempfile.TemporaryDirectory() as tmp:
            seen=[]
            for arm in run.ARMS:
                work=Path(tmp)/arm;work.mkdir();run.prepare(work,job,[],arm)
                visible={name:(work/name).read_bytes() for name in ('history.csv','future.csv','task.json','previous_runs.json')}
                self.assertTrue(all(b'999999' not in value for value in visible.values()))
                self.assertEqual(len((work/'history.csv').read_text().splitlines()),731)
                self.assertEqual(len((work/'future.csv').read_text().splitlines()),15)
                seen.append(visible)
            self.assertEqual(seen[0],seen[1]);self.assertEqual(seen[1],seen[2])

    def test_shared_builder_rejects_unresolved_or_incomplete_inputs(self):
        rows,calendar,manifest=fixture();metadata=manifest['splits']['development'][0]
        values=[float(rows[0][f'd_{i}']) for i in range(a.FIRST,a.LAST+1)]
        stamps=[a.period_end(calendar[f'd_{i}']) for i in range(a.FIRST,a.LAST+1)]
        with self.assertRaisesRegex(ValueError,'complete 1094-day'):
            a.build_series_jobs(metadata,values[1:],stamps[1:])
        with self.assertRaisesRegex(ValueError,'series identity'):
            a.build_series_jobs({**metadata,'series_id':'__default__'},values,stamps)
        invalid=list(values);invalid[0]=True
        with self.assertRaisesRegex(ValueError,'Invalid'):
            a.build_series_jobs(metadata,invalid,stamps)
        naive=[t.replace(tzinfo=None) for t in stamps]
        with self.assertRaisesRegex(ValueError,'midnight UTC'):
            a.build_series_jobs(metadata,values,naive)
        shifted=list(stamps);shifted[0]+=timedelta(hours=1)
        with self.assertRaisesRegex(ValueError,'midnight UTC'):
            a.build_series_jobs(metadata,values,shifted)
        self.assertEqual(a.build_series_jobs(metadata,values,stamps),
            a.build_development_jobs(rows,calendar,manifest)[metadata['series_id']])

    def test_cli_preparation_refuses_overwrite_and_wrong_manifest_before_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'manifest.json';manifest.write_text('{}')
            with patch.object(a.m5_prepare,'verify_archive') as verify:
                with self.assertRaisesRegex(ValueError,'overwrite'):
                    a.prepare_development(root/'missing.zip',manifest,root)
                with self.assertRaisesRegex(ValueError,'pinned'):
                    a.prepare_development(root/'missing.zip',manifest,root/'new')
                verify.assert_not_called();self.assertFalse((root/'new').exists())


if __name__=='__main__':unittest.main()
