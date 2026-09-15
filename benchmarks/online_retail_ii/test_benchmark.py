from datetime import date, datetime, timedelta
import json

import numpy as np
import pandas as pd
import pytest

from .agent import execute, resolve
from .data import classify, cohort, dump, origin, protocol_sha, sha, SELECTION_END, START
from .ledger import build_history
from .models import CANDIDATES, metrics, predict
from .run import fingerprint
from .score import score_submissions


def case_fixture(root, current=date(2011, 3, 28)):
    root.mkdir()
    dates = pd.date_range(current-timedelta(days=199), current)
    history = np.ones(len(dates))
    future = [(current+timedelta(days=i)).isoformat() for i in range(1, 15)]
    pd.DataFrame({'timestamp': dates, 'value': history}).to_csv(root/'history.csv', index=False)
    task = {'case_id': 'synthetic-'+str(current), 'series_id': '12345', 'unit': 'units',
            'origin': str(current), 'horizon': 14, 'future_timestamps': future,
            'request_fingerprint': fingerprint('12345', history, dates.strftime('%Y-%m-%d').tolist(), future)}
    dump(root/'task.json', task)
    dump(root/'matured-outcomes.json', {'as_of': str(current), 'records': []})
    return task


def test_filter_dates_before_future_fields():
    class Trap:
        def __str__(self):
            raise AssertionError('Future quantity or invoice was inspected')
    row = [Trap(), Trap(), None, Trap(), datetime(2011, 12, 1), Trap(), None, Trap()]
    assert classify(row, 'Year 2010-2011', date(2011, 6, 5))[1] == 'after_phase_cutoff_not_used'


def test_sheet_overlap_and_returns_are_explicit():
    row = ['123456', '12345A', None, 5, datetime(2010, 12, 2), 1.5, None, 'United Kingdom']
    assert classify(row, 'Year 2009-2010', date(2011, 6, 5))[1] == 'other_sheet_owns_date'
    assert classify(row, 'Year 2010-2011', date(2011, 6, 5))[0] == ('12345A', date(2010,12,2), 5)
    row[0] = 'C123456'
    assert classify(row, 'Year 2010-2011', date(2011, 6, 5))[1] == 'cancellation'


def test_cohort_does_not_depend_on_later_sales():
    daily = {'12345': {START+timedelta(days=i): 1 for i in range((SELECTION_END-START).days+1)}}
    before = cohort(daily)
    daily['12345'][date(2011,1,1)] = 1e12
    daily['99999'] = {date(2011,1,1): 1e15}
    assert cohort(daily) == before


def test_splits_are_disjoint_and_final_week_complete():
    assert origin(12)+timedelta(days=14) == origin(13)
    assert origin(18)+timedelta(days=14) == origin(19)
    assert origin(25)+timedelta(days=14) == date(2011,12,4)


def test_metric_arithmetic_and_undefined_scale():
    m = metrics([0,3],[0,0],[1]*14)
    assert m['rmsle'] == pytest.approx(np.log(4)/np.sqrt(2))
    assert m['mae'] == 1.5 and m['mase'] is None
    with pytest.raises(ValueError):
        metrics([float('nan')],[1],[1]*14)


@pytest.mark.parametrize('provider', CANDIDATES)
def test_each_candidate_has_finite_horizon(provider):
    p = predict(provider, np.tile(np.arange(1,8),30), 14, date(2011,1,2))
    assert p.shape == (14,) and np.isfinite(p).all()


def test_direct_and_published_gnomon_points_match(tmp_path):
    case = tmp_path/'case'; case_fixture(case)
    a = execute(case, tmp_path/'direct','ridge_log','direct')
    b = execute(case, tmp_path/'gnomon','ridge_log','gnomon')
    assert a['point'] == b['point']
    assert b['series_id'] == '12345' and b['unit'] == 'units'
    assert b['gnomon_distribution'] == '1.2.0'


def test_resolution_recovers_only_unambiguous_matching_execution(tmp_path):
    case = tmp_path/'case'; task = case_fixture(case)
    executions = tmp_path/'executions'
    a = execute(case, executions,'last_value','direct')
    assert resolve(case, executions)['status'] == 'recovered_single_execution'
    assert not resolve(case, executions, 'unrelated')['resolved']
    execute(case, executions,'last_value','direct')
    assert resolve(case, executions)['status'] == 'ambiguous_multiple_executions'
    assert resolve(case, executions, a['execution_id'])['status'] == 'explicit_selection'
    task['unit'] = 'dollars'; dump(case/'task.json', task)
    with pytest.raises(ValueError, match='unit or horizon'):
        resolve(case, executions)


def test_replayed_ledger_uses_visible_original_pairs_and_handles_dst(tmp_path):
    case = tmp_path/'case'; task = case_fixture(case)
    day = date.fromisoformat(task['origin']); previous = day-timedelta(days=14)
    future = [(previous+timedelta(days=i)).isoformat() for i in range(1,15)]
    record = {'origin': str(previous), 'target_end': str(day), 'future_timestamps': future,
              'actual': [1]*14, 'predictions': {p:[0]*14 for p in CANDIDATES}, 'models_sha256':'synthetic-v1'}
    dump(case/'matured-outcomes.json', {'as_of': str(day), 'records':[record]})
    result = build_history(case, tmp_path/'ledger')
    assert result['matched_origins'] == 1, result
    assert result['numerical_refits'] == 0 and result['replayed_executions'] == 10
    assert all(r['mean_rmsle'] == pytest.approx(np.log(2)) for r in result['ranking'])
    assert all(r['rank'] == 1 for r in result['ranking'])
    record['target_end'] = str(day+timedelta(days=1))
    dump(case/'matured-outcomes.json', {'as_of': str(day), 'records':[record]})
    with pytest.raises(ValueError, match='not matured'):
        build_history(case, tmp_path/'rejected-ledger')


def test_scoring_retains_missing_and_ambiguous_cases(tmp_path):
    panel = tmp_path/'panel'; panel.mkdir()
    baseline = tmp_path/'baseline'; (baseline/'agent-cases').mkdir(parents=True)
    case = baseline/'agent-cases'/('12345-'+str(origin(0)))
    task = case_fixture(case, origin(0)); task['case_id'] = case.name
    dump(case/'task.json', task)
    history = pd.read_csv(case/'history.csv')
    frame = history.rename(columns={'timestamp':'date'})
    frame = pd.concat([frame, pd.DataFrame({'date':task['future_timestamps'], 'value':[2]*14})])
    frame['series_id'] = '12345'
    frame.to_csv(panel/'host-panel.csv', index=False)
    dump(panel/'manifest.json', {'panel_sha256':sha(panel/'host-panel.csv'), 'protocol_sha256':protocol_sha()})
    dump(baseline/'plan.json', {'source_manifest_sha256':sha(panel/'manifest.json'),
        'series':['12345'], 'origins':[0], 'planned_cases':1})
    submissions = tmp_path/'submissions.json'; dump(submissions, [])
    missing = score_submissions(panel, baseline, submissions, tmp_path/'missing', 'hermes')
    assert missing['cases'] == 1 and missing['resolved'] == 0
    assert missing['summary']['hermes']['fallback_cases'] == 1
    executions = tmp_path/'executions'
    first = execute(case, executions, 'last_value', 'direct')
    execute(case, executions, 'mean_28', 'direct')
    submission = {'case_id':case.name, 'executions':str(executions)}
    dump(submissions, [submission])
    ambiguous = score_submissions(panel, baseline, submissions, tmp_path/'ambiguous', 'hermes')
    assert ambiguous['rows'][0]['resolution_status'] == 'ambiguous_multiple_executions'
    submission['execution_id'] = first['execution_id']; dump(submissions, [submission])
    chosen = score_submissions(panel, baseline, submissions, tmp_path/'chosen', 'hermes')
    assert chosen['resolved'] == 1 and chosen['summary']['hermes']['fallback_cases'] == 0
    wrong = score_submissions(panel, baseline, submissions, tmp_path/'wrong', 'gnomon')
    assert wrong['rows'][0]['resolution_status'] == 'wrong_execution_backend'
    frame.loc[frame.index[0], 'value'] = 100
    frame.to_csv(panel/'host-panel.csv', index=False)
    with pytest.raises(ValueError, match='Panel or protocol changed'):
        score_submissions(panel, baseline, submissions, tmp_path/'changed', 'hermes')
