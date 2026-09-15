"""Synthetic fixtures only. Never open the real M5 archive or manifest."""
from copy import deepcopy
from datetime import date, timedelta
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization import m5_ml_panel as p


class UnselectedRow(dict):
    def __getitem__(self, key):
        if key.startswith('d_'):
            raise AssertionError('Unselected numerical cell accessed')
        return super().__getitem__(key)


def fixture():
    a=p.shared
    calendar={f'd_{i}':date(2010,1,1)+timedelta(days=i-1) for i in range(a.FIRST,a.LAST+1)}
    splits={'development':[], 'reserved':[]}; rows=[]
    for split, count, per_store in [('development',8,4),('reserved',24,3)]:
        for n in range(count):
            store=f'{split}-store-{n//per_store}';item=f'{split}-item-{n}'
            values={f'd_{i}':str(float(n+1+i%7)) for i in range(a.FIRST,a.LAST+1)}
            prefix=[float(values[f'd_{i}']) for i in range(a.m5_prepare.FIRST,a.m5_prepare.CUTOFF+1)]
            splits[split].append({'store_id':store,'item_id':item,'series_id':item+'_'+store,
                                 'initial_history_sha256':a.value_hash(prefix)})
            rows.append(dict(store_id=store,item_id=item,**values) if split=='reserved'
                        else UnselectedRow(store_id=store,item_id=item))
    return rows,calendar,{'splits':splits}


class PanelTests(unittest.TestCase):
    def test_balanced_624_tasks_use_exact_shared_constructor(self):
        rows,calendar,manifest=fixture();before=deepcopy(manifest)
        original=p.shared.build_series_jobs
        with patch.object(p.shared,'build_series_jobs',wraps=original) as builder:
            jobs=p.build_reserved_jobs(rows,calendar,manifest)
            self.assertEqual(builder.call_count,24)
            for call in builder.call_args_list:
                metadata,values,stamps=call.args
                self.assertEqual(jobs[metadata['series_id']],original(metadata,values,stamps))
        self.assertEqual(manifest,before);self.assertEqual(len(jobs),24)
        self.assertEqual(sum(map(len,jobs.values())),624)
        for series,cases in jobs.items():
            for r,job in enumerate(cases):
                self.assertEqual(job['round'],r)
                self.assertEqual(len(job['request']['history']),730)
                self.assertEqual(len(job['actual']),14)
                origin=1577+r*14
                self.assertEqual(job['origin'][:10],str(calendar[f'd_{origin}']+timedelta(days=1)))
                self.assertEqual(job['future_timestamps'][-1][:10],str(calendar[f'd_{origin+14}']+timedelta(days=1)))
                self.assertEqual(job['outcome_recorded_at'],job['future_timestamps'][-1])
                self.assertEqual(job['request']['known_time_cutoff'],job['origin'])
                self.assertEqual(job['request']['recorded_time_cutoff'],job['origin'])

    def test_future_sales_change_host_actuals_but_not_earlier_request(self):
        rows,calendar,manifest=fixture();before=p.build_reserved_jobs(rows,calendar,manifest)
        changed=deepcopy(rows);changed[8]['d_1578']='999999.0'
        after=p.build_reserved_jobs(changed,calendar,manifest)
        selected=manifest['splits']['reserved'][0]['series_id']
        self.assertEqual(before[selected][0]['request'],after[selected][0]['request'])
        self.assertNotEqual(before[selected][0]['actual'],after[selected][0]['actual'])
        self.assertNotEqual(before[selected][1]['request']['history'],after[selected][1]['request']['history'])
        for other in set(before)-{selected}:self.assertEqual(before[other],after[other])

    def test_manifest_errors_precede_any_numerical_consumption(self):
        rows,calendar,manifest=fixture()
        mutations=[lambda s:s['reserved'].pop(),
                   lambda s:s['reserved'][0].update(store_id=s['reserved'][3]['store_id']),
                   lambda s:s['reserved'][0].update(item_id=s['reserved'][1]['item_id']),
                   lambda s:s['reserved'][0].update(item_id=s['development'][0]['item_id']),
                   lambda s:s['reserved'][0].update(series_id='__default__'),
                   lambda s:s['reserved'][0].update(initial_history_sha256='not-a-hash')]
        poisoned=[UnselectedRow(store_id=r['store_id'],item_id=r['item_id']) for r in rows]
        for mutation in mutations:
            altered=deepcopy(manifest);mutation(altered['splits'])
            with self.subTest(manifest=altered):
                with self.assertRaises(ValueError):p.build_reserved_jobs(poisoned,calendar,altered)

    def test_lazy_reader_is_not_consumed(self):
        _,calendar,manifest=fixture()
        def forbidden():
            raise AssertionError('Archive reader invoked')
            yield
        with self.assertRaisesRegex(ValueError,'Already-loaded'):
            p.build_reserved_jobs(forbidden(),calendar,manifest)

    def test_missing_duplicate_invalid_data_and_changed_prefix_never_reselect(self):
        rows,calendar,manifest=fixture()
        for changed,pattern in [(rows[:8]+rows[9:],'Missing selected'),(rows+[rows[8]],'Duplicate selected')]:
            with self.assertRaisesRegex(ValueError,pattern):p.build_reserved_jobs(changed,calendar,manifest)
        for value in (True,'nan','-1','bad'):
            changed=deepcopy(rows);changed[8]['d_848']=value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):p.build_reserved_jobs(changed,calendar,manifest)
        changed=deepcopy(rows);changed[8]['d_1300']='123456'
        with self.assertRaisesRegex(ValueError,'selection-prefix'):
            p.build_reserved_jobs(changed,calendar,manifest)

    def test_row_order_does_not_change_panel_or_dates(self):
        rows,calendar,manifest=fixture()
        self.assertEqual(p.build_reserved_jobs(rows,calendar,manifest),
                         p.build_reserved_jobs(list(reversed(rows)),calendar,manifest))
        calendar['d_900']=calendar['d_899']
        with self.assertRaisesRegex(ValueError,'consecutive'):
            p.build_reserved_jobs(rows,calendar,manifest)


if __name__=='__main__':unittest.main()
