from datetime import timedelta
import json
import pytest
from .data import START,FIRST_ORIGIN,END,cohort,origin
from .queue import predecessor_status


def test_51_complete_periods_from_two_week_start():
    assert (FIRST_ORIGIN-START).days+1==14
    assert origin(50)+timedelta(days=14)==END
    assert origin(51)+timedelta(days=14)>END


def test_cohort_uses_only_warmup_not_later_success():
    data={}
    offsets=[[0],[0,7,13],list(range(14))]
    for group,days in enumerate(offsets):
        for i in range(16):data[str(10000+group*100+i)]={START+timedelta(days=d):3 for d in days}
    before=cohort(data)
    assert len(before[0])==48
    for values in data.values():values[END]=1e9
    data['99999']={END:1e9}
    assert cohort(data)==before
    with pytest.raises(ValueError,match='Insufficient'):cohort({'99999':{END:1e9}})


def test_queue_requires_exact_terminal_success(tmp_path):
    (tmp_path/'launch-001').mkdir();(tmp_path/'paid-001').mkdir()
    assert predecessor_status(tmp_path)=='waiting'
    exit=tmp_path/'launch-001/exit.json';finish=tmp_path/'paid-001/FINISHED.json'
    exit.write_text(json.dumps({'exit_code':0}))
    assert predecessor_status(tmp_path)=='blocked'
    finish.write_text(json.dumps({'sessions':3744,'stub':False}))
    assert predecessor_status(tmp_path)=='ready'
    exit.write_text(json.dumps({'exit_code':2}))
    assert predecessor_status(tmp_path)=='blocked'


def test_cold_start_model_availability():
    from .models import availability, forecast
    a=availability(14)
    assert sum(v['available'] for v in a.values())==6
    assert not a['ridge_log']['available'] and a['ridge_log']['required']==84
    assert availability(28)['mean_28']['available']
    assert availability(84)['ridge_log']['available']
    p=forecast('ridge_log',list(range(14)),14,FIRST_ORIGIN)
    assert not p['available'] and p['fallback_used']
    assert p['executed_provider']=='seasonal_naive_7'
    assert p['point']==list(range(7,14))*2


def test_cold_start_export_execution_and_ledger(tmp_path):
    import pandas as pd
    from .data import dump,sha,protocol_sha
    from .baselines import run_baselines
    from .agent import execute
    from .ledger import build_history
    panel=tmp_path/'panel';panel.mkdir()
    selected={str(100+g*10+i):{'stratum':group} for g,group in enumerate(('sparse','intermittent','frequent')) for i in range(2)}
    pd.DataFrame([{'series_id':s,'date':(START+timedelta(days=i)).isoformat(),'value':i%7+1}
                  for s in selected for i in range(42)]).to_csv(panel/'host-panel.csv',index=False)
    dump(panel/'manifest.json',{'phase':'full_span','selected_products':selected,'panel_sha256':sha(panel/'host-panel.csv'),
         'protocol_sha256':protocol_sha(),'target':'synthetic','source_availability':'synthetic'})
    baseline=tmp_path/'baselines';report=run_baselines(panel,baseline,smoke=True)
    assert report['cases']==12
    assert report['methods']['ridge_log']['unavailable_cases']==12
    assert report['methods']['mean_28']['unavailable_cases']==6
    for index,folds in ((0,0),(1,1)):
        case=baseline/'agent-cases'/('100-'+origin(index).isoformat())
        task=json.loads((case/'task.json').read_text())
        cv=json.loads((case/'current-cv.json').read_text())
        assert task['cv_available_folds']==folds
        assert cv['candidates']['ridge_log']['mean_rmsle'] is None
        if index==0:
            assert cv['ranking']==[]
            assert all(r['fold_count']==0 for r in cv['candidates'].values())
        else:
            assert len(cv['ranking'])==6
        for provider in ('last_value','seasonal_naive_7','auto_ets_7','auto_arima_7','auto_theta_7','croston_sba'):
            direct=execute(case,tmp_path/'direct',provider,'direct')
            gnomon=execute(case,tmp_path/'gnomon',provider,'gnomon')
            assert direct['point']==gnomon['point']
        with pytest.raises(ValueError,match='requires 84'):execute(case,tmp_path/'invalid','ridge_log')
        assert not (tmp_path/'invalid').exists()
        ledger=build_history(case,tmp_path/f'ledger-{index}')
        assert ledger['matched_origins']==index
        assert 'ridge_log' in ledger['providers_without_valid_past_execution']
        assert all(r['provider']!='ridge_log' for r in ledger['ranking'])
        assert ledger['replayed_executions']==index*6


def test_unavailable_agent_call_preserves_numerical_budget(tmp_path):
    import time
    from .models import availability
    from .worker import RetailLab
    case=tmp_path/'case';case.mkdir()
    (case/'task.json').write_text(json.dumps({'model_availability':availability(14)}))
    lab=RetailLab(case,tmp_path/'output','unused','unused','hermes',time.time()+30)
    with pytest.raises(ValueError,match='Insufficient observed history'):
        lab.dispatch('retail',{'operation':'forecast','provider':'ridge_log'})
    assert lab.attempts==0
    assert not (tmp_path/'output').exists()
