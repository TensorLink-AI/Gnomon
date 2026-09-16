import json
import time
from pathlib import Path
from unittest.mock import patch
import urllib.error
import urllib.request

import pytest

from .worker import RetailLab
from .transport import proxy
from benchmarks.online_retail_ii.data import dump
from benchmarks.online_retail_ii.models import CANDIDATES


def lab(tmp_path,arm='hermes'):
    case=tmp_path/'case';case.mkdir()
    dump(case/'current-cv.json',{'only_past':True})
    dump(case/'matured-outcomes.json',{'records':[]})
    out=tmp_path/'out';out.mkdir()
    return RetailLab(case,out,'numerical-python',str(tmp_path),arm,time.time()+30)


def test_no_arbitrary_tools_paths_or_forecast_alias(tmp_path):
    tool=lab(tmp_path)
    for name,args in [('terminal',{}),('retail',{'operation':'cv','path':'../host-panel.csv'}),
                      ('retail',{'operation':'forecast','selected_provider':'ridge_log'})]:
        with pytest.raises(ValueError):tool.dispatch(name,args)
    assert tool.attempts==0
    with pytest.raises(ValueError,match='disabled'):tool.dispatch('retail',{'operation':'ledger'})
    assert tool.dispatch('retail',{'operation':'cv'})['result']=={'only_past':True}


def test_attempt_budget_and_backend_are_enforced(tmp_path):
    tool=lab(tmp_path)
    result={k:None for k in ('provider','execution_id','point','series_id','unit','future_timestamps',
                            'request_fingerprint','fallback_used','executed_provider','backend')}
    with patch.object(tool,'numerical',return_value=result) as numerical:
        for provider in CANDIDATES[:4]:tool.dispatch('retail',{'operation':'forecast','provider':provider})
        assert numerical.call_count==4
        assert all('direct' in call.args[1] for call in numerical.call_args_list)
        with pytest.raises(ValueError,match='exhausted'):tool.dispatch('retail',{'operation':'forecast','provider':'ridge_log'})
    dump(tool.case/'current-cv.json',{'tampered':True})
    with pytest.raises(ValueError,match='Protected input'):tool.dispatch('retail',{'operation':'status'})


def test_proxy_overrides_seed_and_stops_before_forwarding(tmp_path):
    payload={'messages':[{'role':'user','content':'test'}],'seed':999,'model':'wrong','max_tokens':99999}
    def request(url):
        return urllib.request.urlopen(urllib.request.Request(url+'/chat/completions',data=json.dumps(payload).encode(),
            headers={'Content-Type':'application/json'})).read()
    with proxy(tmp_path,'',19,time.time()+30,stub=True) as url:
        request(url)
    forwarded=json.loads((tmp_path/'api-01-forwarded.json').read_text())
    assert forwarded['seed']==19 and forwarded['model']=='deepseek-v4.1-flash' and forwarded['max_tokens']==3072
    closed=tmp_path/'closed';closed.mkdir()
    with proxy(closed,'never-send',7,time.time()-1,stub=True) as url:
        with pytest.raises(urllib.error.HTTPError):request(url)
    assert not list(closed.glob('api-*-forwarded.json'))
