from datetime import date,timedelta
import json
from pathlib import Path
import pytest
import pandas as pd
from benchmarks.online_retail_ii.full_span.models import CANDIDATES,availability,metrics
from benchmarks.online_retail_ii.full_span.baselines import fingerprint
from benchmarks.online_retail_ii.data import dump
from .client import Client,PROVIDERS,POLICIES,digest,parse_forecast
from .prepare import prepare_case
from .agent import execute,resolve
from .ledger import build_history


def source_case(root,index):
    start=date(2009,12,7);origin=start+timedelta(days=13+14*index)
    root.mkdir();dates=[str(start+timedelta(days=i)) for i in range(14+14*index)]
    future=[str(origin+timedelta(days=i)) for i in range(1,15)]
    pd.DataFrame({'timestamp':dates,'value':[1]*len(dates)}).to_csv(root/'history.csv',index=False)
    task={'case_id':'sku-'+str(origin),'series_id':'sku','unit':'units','origin':str(origin),'horizon':14,
        'future_timestamps':future,'request_fingerprint':fingerprint('sku',[1]*len(dates),dates,future),
        'candidates':list(CANDIDATES),'model_availability':availability(len(dates))}
    dump(root/'task.json',task)
    dump(root/'current-cv.json',{'candidates':{p:{'eligible':False,'mean_rmsle':None} for p in CANDIDATES},'ranking':[]})
    records=[]
    if index:
        first=start+timedelta(days=13);f=[str(first+timedelta(days=i)) for i in range(1,15)]
        records=[{'origin':str(first),'target_end':f[-1],'future_timestamps':f,'actual':[1]*14,
            'predictions':{p:[1]*14 for p in CANDIDATES},'fallbacks':{p:not v['available'] for p,v in availability(14).items()},
            'availability':availability(14),'scores':{p:0. for p in CANDIDATES},'models_sha256':'test-numerical'}]
    dump(root/'matured-outcomes.json',{'as_of':str(origin),'records':records})
    return task

class FakeClient:
    def __init__(self,output):self.output=output;self.calls=[]
    def forecast(self,history,label,provider):
        self.calls.append((list(history),provider));p=self.output/label;p.mkdir(parents=True)
        dump(p/'receipt.json',{'response_sha256':'synthetic','seconds':0,'request_sha256':'synthetic'})
        return {'forecasts':[{'quantiles':{'0.5':[3 if provider==PROVIDERS[0] else 4]*14}}],
                'meta':{'models_used':['synthetic'],'credits_charged':0}}


def test_origin_only_calls_cv_maturation_typed_choice_and_ledger(tmp_path):
    client=FakeClient(tmp_path/'api');store={};revision={p:'test/'+p for p in PROVIDERS}
    for index in (0,1):
        source=tmp_path/f'source-{index}';original=source_case(source,index);out=tmp_path/f'case-{index}'
        task=prepare_case(source,out,client,revision,store)
        assert task['request_fingerprint']==original['request_fingerprint']
        assert len(task['candidates'])==12
        cv=json.loads((out/'current-cv.json').read_text())
        for provider in PROVIDERS:
            assert cv['candidates'][provider]['fold_count']==index
            if index:assert cv['candidates'][provider]['folds'][0]['rmsle']>0
            result=execute(out,tmp_path/f'executions-{index}',provider)
            assert result['gnomon_distribution']=='1.2.0'
            assert result['point']==store['sku',task['origin'],provider]['point']
            assert resolve(out,tmp_path/f'executions-{index}',result['execution_id'])['resolved']
        assert not resolve(out,tmp_path/f'executions-{index}')['resolved']
        report=build_history(out,tmp_path/f'ledger-{index}')
        assert report['matched_origins']==index
        if index:
            ranked={r['provider']:r for r in report['ranking']}
            assert set(PROVIDERS)<=set(ranked)
            assert ranked[PROVIDERS[0]]['mean_rmsle']==pytest.approx(__import__('math').log(2))
    assert [len(h) for h,p in client.calls]==[14,14,28,28]
    assert len(client.calls)==4 # no repeated remote requests for CV or either seed
    frozen=tmp_path/'case-1'/f'{PROVIDERS[0]}-forecast.json'
    data=json.loads(frozen.read_text());data['point'][0]=999;dump(frozen,data)
    with pytest.raises(ValueError,match='changed'):execute(tmp_path/'case-1',tmp_path/'invalid',PROVIDERS[0])


@pytest.mark.parametrize('values',[[1]*13,[float('nan')]*14,[[1]]*14,[True]*14])
def test_invalid_medians_reject(values):
    with pytest.raises(ValueError):parse_forecast({'forecasts':[{'quantiles':{'0.5':values}}],'meta':{'models_used':['x']}},14)


def test_key_redaction_no_feedback_and_fixed_destination(tmp_path,monkeypatch):
    from urllib.error import HTTPError
    import io
    secret='cpk_test_only_secret';key=tmp_path/'key';key.write_text(secret)
    seen=[]
    def fail(request,timeout):
        seen.append(request)
        raise HTTPError(request.full_url,401,'bad',{},io.BytesIO(('failed '+secret).encode()))
    monkeypatch.setattr('urllib.request.urlopen',fail)
    client=Client(key,tmp_path/'out')
    with pytest.raises(ValueError):client.forecast([1]*14,'case',PROVIDERS[1])
    assert len(seen)==1 and seen[0].full_url.endswith('/forecast')
    payload=json.loads(seen[0].data);assert payload['mode']=='ensemble' and payload['top_k']==2
    for p in (tmp_path/'out').rglob('*'):
        if p.is_file():assert secret not in p.read_text()
    with pytest.raises(ValueError,match='not authorized'):client.call('feedback',{})


def test_advisory_catalog_failure_retains_receipt_without_retry(tmp_path,monkeypatch):
    from urllib.error import HTTPError
    import io
    key=tmp_path/'key';key.write_text('cpk_test_only_secret');calls=[]
    def fail(request,timeout):
        calls.append(request.full_url)
        raise HTTPError(request.full_url,404,'not found',{},io.BytesIO(b'{"error":"not found"}'))
    monkeypatch.setattr('urllib.request.urlopen',fail)
    client=Client(key,tmp_path/'out');result=client.catalog()
    assert not result['available'] and result['automatic_retries']==0
    assert len(calls)==1 and calls[0].endswith('/models')
    assert json.loads((client.output/result['receipt']).read_text())['http_status']==404


def test_direct_chronos_control_import_does_not_import_gnomon():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-c',
        'import sys; from benchmarks.online_retail_ii.tsfm.chronos_control import POLICY; '
        'assert POLICY["mode"]=="explicit" and POLICY["model"]=="chronos2"; '
        'assert "gnomon" not in sys.modules'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
