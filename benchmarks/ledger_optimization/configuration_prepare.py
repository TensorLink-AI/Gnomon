"""Synthetic-only064 catalogue and execution parity receipt; no task scoring."""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import time
import unittest
from unittest.mock import patch
from . import configured_hourly as configured
from . import hourly_numerical as original
from benchmarks.tests import test_configured_hourly as tests


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);here=Path(__file__).parent;started=time.monotonic();cpu=time.process_time();attempts=[]
    def save(n,v):(output/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    source_sha=hashlib.sha256((here/'hourly_numerical.py').read_bytes()).hexdigest()
    if source_sha!='df271a59538d0c644054fdbee3b9880113de7df917c11f2b377d10c017439369':raise ValueError('Original038 numerical module changed')
    def count(fn,surface,stage):
        def invoke(values,labels,future,config):
            canonical=original.RECIPES[config] if isinstance(config,str) else config
            attempt={'surface':surface,'stage':stage,'config':canonical,'config_id':configured.config_id(canonical),
                'history_count':len(values),'horizon':len(future),'estimator_fit':canonical['kind'] in ('ridge','forest'),'completed':False}
            attempts.append(attempt);begin=time.monotonic()
            try:
                point=fn(values,labels,future,config);attempt['completed']=True;return point
            except BaseException as e:attempt['error']=type(e).__name__+': '+str(e);raise
            finally:attempt['seconds']=time.monotonic()-begin
        return invoke
    manifest={'protocol':'CONFIGURATION_SPACE_064.md','code_sha256':{n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ('CONFIGURATION_SPACE_064.md','configuration_prepare.py','configured_hourly.py','hourly_numerical.py')},
        'test_sha256':hashlib.sha256(Path(tests.__file__).read_bytes()).hexdigest(),'adapter_revision':configured.revision(),
        'packages':{n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn')},'synthetic_only':True,
        'source_observations_read':False,'validation_or_final_access':False,'common_attempt_budget':60,'catalogue_size':78}
    save('manifest.json',manifest);save('catalogue.json',configured.catalog())
    try:
        stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
        with patch.object(tests,'predict',count(configured.predict,'configured','unit_tests')),patch.object(tests,'original_predict',count(original.predict,'original038','unit_tests')):
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (output/'tests.txt').write_text(stream.getvalue())
        if not result.wasSuccessful():raise AssertionError('Synthetic execution tests failed')
        start=datetime(2020,1,1,0,0,1,tzinfo=timezone.utc);parity=[]
        old=count(original.predict,'original038','parity');new=count(configured.predict,'configured','parity')
        for n in (658,730):
            values=[30+5*math.sin(2*math.pi*i/24)+3*math.cos(2*math.pi*i/168)+i/1000 for i in range(n)]
            labels=[start+timedelta(hours=i) for i in range(n+24)];before=values.copy()
            for name,config in original.RECIPES.items():
                expected=old(values,labels[:n],labels[n:],name);observed=new(values,labels[:n],labels[n:],config)
                if expected!=observed or before!=values:raise AssertionError('Original numerical recipe changed')
                parity.append({'recipe':name,'config':config,'history_count':n,'history_sha256':hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest(),
                    'original_point':expected,'configured_point':observed,'exact_equal':True})
        save('parity.json',parity)
        novel_configs=[{'kind':'ridge','window':336,'lags':24,'alpha':.01},{'kind':'ridge','window':730,'lags':168,'alpha':10000.},
            {'kind':'forest','window':336,'lags':24,'depth':3},{'kind':'forest','window':730,'lags':48,'depth':12}]
        smoke=[];new=count(configured.predict,'configured','novel_synthetic_smoke')
        for config in novel_configs:
            point=new(values,labels[:730],labels[730:],config)
            if len(point)!=24 or any(not math.isfinite(v) or v<0 for v in point):raise AssertionError('Invalid configured result')
            smoke.append({'config':config,'config_id':configured.config_id(config),'point':point,'synthetic_only':True})
        save('novel-synthetic.json',smoke);save('attempts.json',attempts)
        report={'status':'prepared_not_evaluated','configurations':len(configured.catalog()),'tests':result.testsRun,'test_failures':len(result.failures)+len(result.errors),
            'exact_parity_requests':len(parity),'new_synthetic_requests':len(smoke),'synthetic_forecast_computations':len(attempts),
            'synthetic_estimator_fits':sum(a['estimator_fit'] for a in attempts),'completed_computations':sum(a['completed'] for a in attempts),
            'source_forecast_computations':0,'api_calls':0,'seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,
            'limitation':'Common configuration preparation only; no scored source task, search policy or ledger accuracy result.'}
        save('report.json',report);return report
    except BaseException as e:
        save('attempts.json',attempts);save('FAILED.json',{'error':type(e).__name__,'message':str(e),'seconds':time.monotonic()-started});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');print(json.dumps(run(**vars(p.parse_args())),indent=2))
