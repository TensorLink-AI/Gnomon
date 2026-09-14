"""Independent inventory, AST parity and synthetic receipt audit for064."""
import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path


def audit(directory):
    directory=Path(directory);here=Path(__file__).parent;checks=0
    def check(ok,message):
        nonlocal checks
        checks+=1
        if not ok:raise AssertionError(message)
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=read(directory/'manifest.json')
    for n,h in manifest['code_sha256'].items():check(sha(here/n)==h,'Frozen code')
    check(sha(here.parent/'tests/test_configured_hourly.py')==manifest['test_sha256'],'Frozen tests')
    check(sha(here/'hourly_numerical.py')=='df271a59538d0c644054fdbee3b9880113de7df917c11f2b377d10c017439369','Original038 unchanged')
    check(manifest['adapter_revision']=='configured_hourly064/sha256:'+sha(here/'configured_hourly.py'),'Revision binds numerical adapter')
    check(manifest['synthetic_only'] and not manifest['source_observations_read'] and not manifest['validation_or_final_access'],'No source scoring')
    check(manifest['common_attempt_budget']==60,'Equal planned attempt budget')
    def funcs(path):return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    old,new=funcs(here/'hourly_numerical.py'),funcs(here/'configured_hourly.py')
    check(ast.dump(old['calendar_features'])==ast.dump(new['calendar_features']),'Unchanged calendar feature code')
    check([ast.dump(n) for n in old['predict'].body[1:]]==[ast.dump(n) for n in new['predict'].body[2:]],'Numerical body unchanged after parameter/label validation')
    original=[{'kind':'seasonal','season':24},{'kind':'seasonal','season':168},{'kind':'weekly_mean'},
        {'kind':'ridge','window':336,'lags':48,'alpha':10.},{'kind':'ridge','window':730,'lags':168,'alpha':10.},
        {'kind':'forest','window':730,'lags':48,'depth':6}]
    expected=original[:3]+[{'kind':'ridge','window':w,'lags':l,'alpha':a} for w,l,a in itertools.product((336,504,730),(24,48,168),(.01,.1,1.,10.,100.,1000.,10000.))]
    expected += [{'kind':'forest','window':w,'lags':l,'depth':d} for w,l,d in itertools.product((336,730),(24,48),(3,6,12))]
    encode=lambda r:json.dumps(r,sort_keys=True,separators=(',',':'))
    identity=lambda r:hashlib.sha256(encode(r).encode()).hexdigest()
    inventory=read(directory/'catalogue.json')
    check(len(inventory)==78 and len({r['config_id'] for r in inventory})==78,'Unique full inventory')
    check({encode(r['config']) for r in inventory}=={encode(r) for r in expected},'Exact prospective parameter grid')
    check([r['config'] for r in inventory[:6]]==original,'Original six preserved first')
    check([r['config_id'] for r in inventory[6:]]==sorted(r['config_id'] for r in inventory[6:]),'Remaining hash order, not score order')
    for row in inventory:check(row['config_id']==identity(row['config']),'Canonical identity')
    parity=read(directory/'parity.json');check(len(parity)==12,'All six recipes at both history lengths')
    names=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')
    for n in (658,730):
        rows=[r for r in parity if r['history_count']==n];check([r['recipe'] for r in rows]==list(names),'Complete synthetic cohort')
        values=[30+5*math.sin(2*math.pi*i/24)+3*math.cos(2*math.pi*i/168)+i/1000 for i in range(n)]
        history_hash=hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
        for row,config in zip(rows,original,strict=True):
            check(row['config']==config and row['history_sha256']==history_hash,'Declared synthetic request')
            check(row['original_point']==row['configured_point'] and row['exact_equal'],'Exact saved prediction parity')
            check(len(row['configured_point'])==24 and all(math.isfinite(v) and v>=0 for v in row['configured_point']),'Complete finite prediction')
            if config['kind']=='seasonal':check(row['configured_point']==[values[-config['season']+h%config['season']] for h in range(24)],'Independent seasonal arithmetic')
    smoke=read(directory/'novel-synthetic.json');check(len(smoke)==4,'Novel boundary smoke coverage')
    ids={r['config_id'] for r in inventory}
    for row in smoke:
        check(row['config_id']==identity(row['config']) and row['config_id'] in ids and row['config'] not in original,'New approved configuration')
        check(row['synthetic_only'] and len(row['point'])==24 and all(math.isfinite(v) and v>=0 for v in row['point']),'Synthetic result contract')
    attempts=read(directory/'attempts.json');report=read(directory/'report.json')
    check(len(attempts)==52 and sum(a['estimator_fit'] for a in attempts)==28,'All synthetic costs including repeated unit/parity checks')
    check(sum(a['stage']=='unit_tests' for a in attempts)==24 and sum(a['stage']=='parity' for a in attempts)==24 and sum(a['stage']=='novel_synthetic_smoke' for a in attempts)==4,'Stage-separated attempts')
    for a in attempts:
        check(a['completed'] and a['horizon']==24 and a['history_count'] in (658,730),'Completed synthetic computation')
        check(a['config_id']==identity(a['config']) and a['estimator_fit']==(a['config']['kind'] in ('ridge','forest')),'Attempt model/cost identity')
    check(report['tests']==4 and report['test_failures']==0 and 'OK' in (directory/'tests.txt').read_text(),'Passed synthetic tests')
    check(report['configurations']==78 and report['exact_parity_requests']==12 and report['new_synthetic_requests']==4,'Preparation coverage')
    check(report['synthetic_forecast_computations']==report['completed_computations']==52 and report['synthetic_estimator_fits']==28,'Reported synthetic cost')
    check(report['source_forecast_computations']==report['api_calls']==0 and report['status']=='prepared_not_evaluated','Not an accuracy result')
    result={'checks':checks,'failures':0,'report_sha256':sha(directory/'report.json'),'verifier_sha256':sha(Path(__file__)),
        'scope':'Independent full parameter inventory, canonical identities, AST numerical-body equality, synthetic parity records, seasonal arithmetic and complete cost accounting. No real-series scores or ledger benefit tested.'}
    target=directory/'verification.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');print(json.dumps(audit(**vars(p.parse_args())),indent=2))
