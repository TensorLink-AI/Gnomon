"""Synthetic-only065 search preflight; never executes a source-data forecast."""
import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import time
import unittest
from unittest.mock import patch
from . import configuration_search as search
from .configured_hourly import catalog,revision
from benchmarks.tests import test_configuration_search as tests


def label(config,index):
    v=search.vector(config)
    return .1+.03*sum((a-b)**2 for a,b in zip(v,[0.,0.,1.,0.,.5,.5,.5,0.,0.],strict=True))+.001*index


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);here=Path(__file__).parent;start=time.monotonic();cpu=time.process_time();solves=[];stage='tests'
    save=lambda n,v:(output/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    save('manifest.json',{'protocol':'CONFIGURATION_SEARCH_065.md','code_sha256':{n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ('CONFIGURATION_SEARCH_065.md','configuration_search.py','configuration_search_prepare.py','configured_hourly.py')},
        'test_sha256':hashlib.sha256(Path(tests.__file__).read_bytes()).hexdigest(),'revision':revision(),
        'catalogue_receipt_sha256':hashlib.sha256((here/'evidence/common-configuration-064.json').read_bytes()).hexdigest(),
        'synthetic_only':True,'source_observations_read':False,'provider_calls':0,'api_calls':0})
    real_solve=search.np.linalg.solve
    def counted(a,b):
        record={'stage':stage,'training_rows':len(a),'right_hand_sides':b.shape[1],'completed':False};solves.append(record);begin=time.monotonic()
        try:result=real_solve(a,b);record['completed']=True;return result
        finally:record['seconds']=time.monotonic()-begin
    try:
        with patch.object(search.np.linalg,'solve',counted):
            stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
            (output/'tests.txt').write_text(stream.getvalue())
            if not result.wasSuccessful():raise AssertionError('Search unit tests failed')
            current,backtests,history=tests.fixture()
            for row in backtests:row['cv_rmsle']=label(row['config'],20)
            for i,episode in enumerate(history):
                for row in episode['backtests']:row['cv_rmsle']=label(row['config'],i)
            save('synthetic-inputs.json',{'current':current,'backtests':backtests,'history':history,'source':'synthetic_features_and_labels_only'})
            reports={}
            for arm in ('control','ledger'):
                stage='synthetic_search_'+arm;query={**current,'arm':arm};observed=copy.deepcopy(backtests);steps=[];used=24
                for number in range(11):
                    search.admit_backtest(used);proposal=search.suggest(query,observed,history if arm=='ledger' else [])
                    cv=label(proposal['next_config'],20)
                    steps.append({'step':number,'simulated_attempts_before':used,'proposal':proposal,'synthetic_cv_rmsle':cv})
                    observed.append({'config':proposal['next_config'],'config_id':proposal['next_config_id'],'cv_rmsle':cv});used+=3
                selected=search.select(observed);used+=1
                save(arm+'.json',{'arm':arm,'steps':steps,'backtests':observed,'selected':selected,'simulated_numerical_attempts':used,
                    'actual_forecast_computations':0,'synthetic_only':True})
                reports[arm]={'steps':len(steps),'tested_configurations':len(observed),'simulated_attempts':used,'selected_config_id':selected['config_id']}
        save('solves.json',solves)
        report={'status':'search_prepared_not_source_evaluated','tests':result.testsRun,'test_failures':len(result.failures)+len(result.errors),
            'arms':reports,'surrogate_solves':len(solves),'completed_solves':sum(r['completed'] for r in solves),'provider_calls':0,'api_calls':0,
            'seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
            'limitation':'Synthetic labels exercise acquisition, maturity and budget plumbing. No source-data model search or ledger accuracy result.'}
        save('report.json',report);return report
    except BaseException as e:
        save('solves.json',solves);save('FAILED.json',{'error':type(e).__name__,'message':str(e),'seconds':time.monotonic()-start});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');print(json.dumps(run(**vars(p.parse_args())),indent=2))
