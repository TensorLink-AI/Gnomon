from copy import deepcopy
from datetime import datetime,timedelta,timezone
import unittest

from benchmarks.ledger_optimization.ml_continuous_prepare import reconstruct,canonical,FIELDS


def fixture():
    start=datetime(2014,1,1,tzinfo=timezone.utc);jobs=[]
    for number in (0,1,25):
        end=729+14*number;hist=range(end-729,end+1);future=range(end+1,end+15)
        times=lambda values:[(start+timedelta(days=i)).isoformat() for i in values]
        origin=times([end])[0]
        r={'history':[float(i%17) for i in hist],'timestamps':times(hist),
           'past_covariates':[[float(i%2)] for i in hist],
           'future_covariates':[[float(i%2)] for i in future],
           'past_covariate_names':['known_feature'],'future_covariate_names':['known_feature'],
           'future_timestamps':times(future),'cutoff':origin,'known_time_cutoff':origin,
           'series_id':'synthetic','unit':'items','frequency':'D','horizon':14,'season':7}
        jobs.append({'request':r,'actual':[float(i%17) for i in future],'origin':origin,
                     'outcome_recorded_at':r['future_timestamps'][-1],
                     'future_timestamps':r['future_timestamps'],'series_id':'synthetic','round':number})
    return {'synthetic':jobs}


class ContinuousPreparationTest(unittest.TestCase):
    def test_original_anchors_and_full_schedule(self):
        raw=fixture();before=deepcopy(raw);jobs,proof=reconstruct(raw)
        self.assertEqual(len(jobs['synthetic']),26)
        self.assertEqual(proof['synthetic']['original_tasks_reproduced'],3)
        self.assertEqual(raw,before)
        for original in raw['synthetic']:
            self.assertEqual(canonical(jobs['synthetic'][original['round']]),canonical(original))

    def test_conflicting_overlap_rejected(self):
        raw=fixture();raw['synthetic'][1]['request']['history'][0]+=1
        with self.assertRaisesRegex(ValueError,'Conflicting'):reconstruct(raw)

    def test_integer_history_float_actuals_preserve_anchor_bytes(self):
        raw=fixture()
        for anchor in raw['synthetic']:
            anchor['request']['history']=[int(v) for v in anchor['request']['history']]
        jobs,proof=reconstruct(raw)
        self.assertGreater(proof['synthetic']['equal_numeric_representation_matches'],0)
        for anchor in raw['synthetic']:
            self.assertEqual(canonical(jobs['synthetic'][anchor['round']]),canonical(anchor))

    def test_boolean_sales_rejected(self):
        raw=fixture();raw['synthetic'][0]['request']['history'][0]=True
        with self.assertRaisesRegex(ValueError,'Invalid sales'):reconstruct(raw)

    def test_export_mutation_cannot_change_source_or_other_tasks(self):
        raw=fixture();before=deepcopy(raw);jobs,_=reconstruct(raw)
        later=deepcopy(jobs['synthetic'][1])
        jobs['synthetic'][0]['request']['future_covariates'][0][0]=1234
        self.assertEqual(raw,before)
        self.assertEqual(jobs['synthetic'][1],later)

    def test_future_sales_do_not_change_earlier_request(self):
        raw=fixture();old,_=reconstruct(raw)
        target=raw['synthetic'][0]['future_timestamps'][0]
        for anchor in raw['synthetic']:
            for times,values in [(anchor['request']['timestamps'],anchor['request']['history']),
                                 (anchor['future_timestamps'],anchor['actual'])]:
                if target in times:values[times.index(target)]+=1000
        new,_=reconstruct(raw)
        self.assertEqual(new['synthetic'][0]['request'],old['synthetic'][0]['request'])
        self.assertNotEqual(new['synthetic'][0]['actual'],old['synthetic'][0]['actual'])
        self.assertNotEqual(new['synthetic'][1]['request']['history'],old['synthetic'][1]['request']['history'])

    def test_hindsight_metadata_not_exported(self):
        raw=fixture();old,_=reconstruct(raw)
        for anchor in raw['synthetic']:
            anchor.update(predictions={'future_winner':[999]},scores={'future_winner':0},card={'choice':'future_winner'})
        new,_=reconstruct(raw)
        self.assertEqual(new,old)
        self.assertTrue(all(set(job)==set(FIELDS) for job in new['synthetic']))

    def test_missing_endpoint_rejected(self):
        raw=fixture();raw['synthetic'].pop()
        with self.assertRaisesRegex(ValueError,'first and last'):reconstruct(raw)

    def test_changed_unit_rejected(self):
        raw=fixture();raw['synthetic'][1]['request']['unit']='other'
        with self.assertRaisesRegex(ValueError,'identity'):reconstruct(raw)

    def test_missing_date_rejected(self):
        raw=fixture();raw['synthetic'][0]['request']['timestamps'][0]='2013-12-31T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError,'Nonconsecutive'):reconstruct(raw)


if __name__=='__main__':unittest.main()
