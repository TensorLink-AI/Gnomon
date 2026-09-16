from datetime import date,timedelta
import json
import pytest
from .test_benchmark import case_fixture
from .models import CANDIDATES
from .data import dump
from .resume_ledger import build_history

@pytest.mark.parametrize('day',[date(2011,4,10),date(2011,11,13)])
def test_multi_origin_calendar_transition(day,tmp_path):
    case=tmp_path/'case';case_fixture(case,day)
    records=[]
    for offset in (28,14):
        origin=day-timedelta(days=offset)
        future=[str(origin+timedelta(days=i)) for i in range(1,15)]
        records.append({'origin':str(origin),'target_end':future[-1],'future_timestamps':future,
            'actual':[1]*14,'predictions':{p:[0]*14 for p in CANDIDATES},'models_sha256':'synthetic-v1'})
    dump(case/'matured-outcomes.json',{'as_of':str(day),'records':records})
    report=build_history(case,tmp_path/'ledger')
    assert report['matched_origins']==2
    assert report['numerical_refits']==0 and report['replayed_executions']==20
    comparisons=json.loads((tmp_path/'ledger/gnomon-comparisons.json').read_text())
    assert all(r['status']=='ok' and r['calendar_comparison']['independently_admitted_origins']==2 for r in comparisons)
    assert all(not r['calendar_comparison']['predictions_or_actuals_changed'] for r in comparisons)
    assert all(r['mean_rmsle']==pytest.approx(__import__('math').log(2)) for r in report['ranking'])


def test_real_task_mismatch_still_rejects():
    from .calendar_history import compare_calendar_history
    class Fake:
        def compare_history(self,**kwargs):
            return {'status':'incompatible_evidence','reason':'task_or_provider_identity_changed','observed_origins':2,
                'origins':[{'models':[{'execution_id':'first'}]},{'models':[{'execution_id':'second'}]}]}
        def execution(self,eid):
            return {'provider':'p','provider_identity':{'revision':'stable'},'request':{
                'frequency':'D','horizon':1,'season':1 if eid=='first' else 7,
                'timestamps':['2011-03-27T00:00:00+00:00'],'cutoff':'2011-03-27T23:59:59+01:00',
                'future_timestamps':['2011-03-28T00:00:00+01:00'],'related_series':[]}}
    assert compare_calendar_history(Fake(),horizon=1)['status']=='incompatible_evidence'
