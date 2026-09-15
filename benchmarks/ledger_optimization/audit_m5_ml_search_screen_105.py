"""Independent request/score/selection audit of screen 105; never refits models."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from .m5_ml_development_contract import authenticated_contract
from .m5_ml_search_screen_105 import CONFIGS,POLICIES


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()


def metric(point,actual):
    if len(point)!=14 or len(actual)!=14:raise ValueError('Incomplete horizon')
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in point+actual):
        raise ValueError('Invalid numerical values')
    errors=[(math.log1p(p)-math.log1p(a))**2 for p,a in zip(point,actual,strict=True)]
    return math.sqrt(math.fsum(errors)/14)


def audit_series(folder,jobs):
    folder=Path(folder);checks=0;hashes={};prior=[];rows=[]
    def check(label,value):
        nonlocal checks
        checks+=1
        if not value:raise ValueError('Screen audit: '+label)
    def read(path):
        raw=path.read_bytes();hashes[str(path)]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    def near(label,a,b):check(label,abs(a-b)<=1e-12)
    for job in jobs:
        home=folder/f"round-{job['round']:02d}";base=job['request']
        r=read(home/'result.json');check('case complete',r['complete'] is True)
        for k in ('series_id','round','origin'):check('case '+k,r[k]==job[k])
        check('target end',r['target_end']==job['future_timestamps'][-1])
        check('recording time',r['recorded_at']==job['outcome_recorded_at'])
        all_times=base['timestamps']+base['future_timestamps'];cov=base['past_covariates']+base['future_covariates']
        requests={}
        for end in (688,702,716,730):
            saved=read(home/f'request-{end}.json')
            expected={'history':[float(v) for v in base['history'][:end]],'timestamps':all_times[:end],
                'future_timestamps':all_times[end:end+14],'past_covariates':cov[:end],
                'future_covariates':cov[end:end+14],'past_covariate_names':base['past_covariate_names'],
                'future_covariate_names':base['future_covariate_names'],'horizon':14,
                'series_id':job['series_id'],'unit':base['unit'],'cutoff':all_times[end-1],
                'frequency':'D','season':7,'known_time_cutoff':all_times[end-1]}
            check('exact frozen request',canonical(saved['request'])==canonical(expected))
            check('CV actuals',saved['actual']==(base['history'][end:end+14] if end<730 else None))
            check('CV target visibility',end==730 or datetime.fromisoformat(all_times[end+13])<=datetime.fromisoformat(job['origin']))
            requests[end]=saved
        path=home/'calls.jsonl';raw=path.read_bytes();hashes[str(path)]=hashlib.sha256(raw).hexdigest()
        events=[json.loads(line) for line in raw.splitlines()]
        check('all attempts and results retained',len(events)==88)
        points={};cv_computed=[]
        for i,config in enumerate(CONFIGS):
            folds=[]
            cid=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
            for j,end in enumerate((688,702,716,730)):
                a,b=events[8*i+2*j:8*i+2*j+2]
                check('attempt/result sequence',a['event']=='attempt' and b['event']=='result')
                check('recorded call duration',type(b['elapsed_seconds']) in (int,float)
                      and math.isfinite(b['elapsed_seconds']) and b['elapsed_seconds']>=0)
                for e in (a,b):
                    check('configuration identity',e['configuration_index']==i and e['config']==config and e['config_id']==cid)
                    check('request reference',e['end']==end and e['request_file']==f'request-{end}.json'
                          and e['request_sha256']==digest(requests[end]['request']))
                point=b['point'];metric(point,[0.]*14);points[i,end]=point
                if config['model']=='seasonal':
                    history=requests[end]['request']['history'];season=config['season']
                    check('independent seasonal calculation',point==[history[-season+k%season] for k in range(14)])
                else:
                    history=requests[end]['request']['history'][-config['window']:]
                    bound=max(max(history)*100,1000)
                    check('frozen overflow guard',max(point)<=bound+max(1e-9,bound*1e-12))
                if end<730:
                    value=metric(point,requests[end]['actual']);near('CV RMSLE',value,b['cv_rmsle']);folds.append(value)
                else:check('production target absent from call result',b['cv_rmsle'] is None)
            cv_computed.append(mean(folds))
        selection=read(home/'selection.json')
        check('same persisted selection',r['selection']=={k:v for k,v in selection.items() if k!='current_cv_scores'})
        check('same persisted CV scores',r['current_cv_scores']==selection['current_cv_scores'])
        for computed,recorded in zip(cv_computed,r['current_cv_scores'],strict=True):near('mean current CV score',computed,recorded)
        # Exact ranking uses the independently verified saved floating values.
        cv=min(range(11),key=lambda i:r['current_cv_scores'][i]);choices=dict.fromkeys(POLICIES,cv)
        now=datetime.fromisoformat(job['origin'])
        visible=[v for v in prior if datetime.fromisoformat(v['origin'])<now
                 and datetime.fromisoformat(v['target_end'])<=now and datetime.fromisoformat(v['recorded_at'])<=now]
        if len(visible)>=4:
            lifetime=[mean(v['production_scores'][i] for v in visible) for i in range(11)]
            recent=[mean(v['production_scores'][i] for v in visible[-4:]) for i in range(11)]
            best=min(range(11),key=lambda i:lifetime[i])
            choices['lifetime_history']=best;choices['recent_history']=min(range(11),key=lambda i:recent[i])
            wins=sum(v['production_scores'][best]<v['production_scores'][cv] for v in visible)
            if lifetime[best]<lifetime[cv] and wins*2>=len(visible):choices['lifetime_support']=best
        check('past-only choices',selection['choices']==choices)
        check('eligible historical origins',selection['visible_origins']==[v['origin'] for v in visible])
        check('matched history count',selection['matched_origins']==len(visible))
        exclusions={'source':sum(datetime.fromisoformat(v['target_end'])>now for v in prior),
                    'recording':sum(datetime.fromisoformat(v['recorded_at'])>now for v in prior),
                    'not_prior':sum(datetime.fromisoformat(v['origin'])>=now for v in prior)}
        check('visibility exclusions',selection['exclusions']==exclusions)
        check('exact tie policy',selection['tie_policy']=='first_in_frozen_configuration_order')
        check('no current production use',selection['current_production_outcomes_used'] is False)
        actual=read(home/'production-actuals.json');check('unaltered production target',actual==job['actual'])
        for i,value in enumerate(r['production_scores']):near('production RMSLE',metric(points[i,730],actual),value)
        for policy,index in choices.items():near('policy score',r['policy_scores'][policy],r['production_scores'][index])
        best=min(range(11),key=lambda i:r['production_scores'][i]);check('hindsight index',r['hindsight_index']==best)
        near('hindsight score',r['hindsight_score'],r['production_scores'][best])
        check('prediction uniqueness',r['unique_production_forecasts']==len({tuple(points[i,730]) for i in range(11)}))
        check('numerical and fit accounting',r['numerical_attempts']==44 and r['estimator_fits']==32 and r['failure_count']==0)
        rows.append(r);prior.append(r)
    for path,value in hashes.items():check('unchanged input bytes',hashlib.sha256(Path(path).read_bytes()).hexdigest()==value)
    return {'checks':checks,'rows':rows,'inputs':hashes}


def audit(root):
    root=Path(root);manifest=(root/'panel-manifest.json').read_bytes();raw_jobs=(root/'development-jobs.json').read_bytes()
    cohort=authenticated_contract(manifest,raw_jobs);jobs=json.loads(raw_jobs)
    launch=json.loads((root/'launch.json').read_text());report=json.loads((root/'report.json').read_text())
    if not report['complete']:raise ValueError('Incomplete screen is not a passing diagnostic')
    expected_sources={'run_m5_ml_search_screen_105.py','m5_ml_search_screen_105.py',
                      'm5_ml_development_contract.py','m5_ml_adapter.py','m5_ml_panel.py',
                      'm5_prepare.py','M5_ML_SEARCH_SCREEN_105.md'}
    if set(launch['sources'])!=expected_sources:raise ValueError('Incomplete source identity')
    for name,expected in launch['sources'].items():
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=expected:raise ValueError('Screen source changed')
    worker_path=Path('results/m5-ml-capsule-offline-001/synthetic-worker-002/manifest.json')
    worker_bytes=worker_path.read_bytes();worker_sha=hashlib.sha256(worker_bytes).hexdigest()
    if worker_sha!='123273702ac98c4eae16a2fde80043e252003ed95ce5b162de1cd44985077ebe':raise ValueError('Worker identity changed')
    numerical=Path('results/m5-ml-capsule-offline-001/capsule/benchmarks/hermes_ml_checkpoint_v6/numerical.py')
    if (launch['packages']!=json.loads(worker_bytes)['inventory']['plain']['packages']
            or launch['worker_manifest_sha256']!=worker_sha
            or launch['numerical_source_sha256']!='a71b75f06ad69fcca06275046c5e74565b15d7d6dd2538df0fe827c448fc0413'
            or hashlib.sha256(numerical.read_bytes()).hexdigest()!=launch['numerical_source_sha256']
            or launch['configs']!=CONFIGS or launch['ends']!=[688,702,716,730]
            or launch['cases']!=208 or launch['manifest_sha256']!=hashlib.sha256(manifest).hexdigest()
            or launch['jobs_sha256']!=hashlib.sha256(raw_jobs).hexdigest()):
        raise ValueError('Numerical runtime/configuration/input identity mismatch')
    expected={(c['series_id'],c['round']) for c in cohort['cases']}
    actual={(p.parent.parent.name,int(p.parent.name.split('-')[1])) for p in root.glob('series/*/round-*/result.json')}
    if actual!=expected or len(actual)!=208:raise ValueError('Exact complete development cohort required')
    rows=[];checks=0;inputs={}
    for series,records in sorted(jobs.items()):
        result=audit_series(root/'series'/series,records);rows.extend(result['rows']);checks+=result['checks'];inputs.update(result['inputs'])
    if rows!=report['rows']:raise ValueError('Aggregate rows differ from independently audited cases')
    groups=[(report['all'],rows),(report['cold_0_to_3'],[r for r in rows if r['round']<4]),
            (report['later_4_to_25'],[r for r in rows if r['round']>=4])]
    if set(report['by_series'])!=set(jobs) or set(report['by_round'])!={str(n) for n in range(26)}:
        raise ValueError('Subgroup membership')
    groups += [(report['by_series'][s],[r for r in rows if r['series_id']==s]) for s in jobs]
    groups += [(report['by_round'][str(n)],[r for r in rows if r['round']==n]) for n in range(26)]
    for saved,rs in groups:
        computed={p:math.fsum(r['policy_scores'][p] for r in rs)/len(rs) for p in POLICIES}
        computed['hindsight']=math.fsum(r['hindsight_score'] for r in rs)/len(rs)
        if saved['cases']!=len(rs):raise ValueError('Aggregate case count')
        for policy,value in computed.items():
            if abs(saved['mean_rmsle'][policy]-value)>1e-12:raise ValueError('Aggregate score')
            reduction=1-value/computed['current_cv'] if computed['current_cv'] else None
            recorded=saved['reduction_vs_current_cv'][policy]
            if (recorded is None)!=(reduction is None) or (reduction is not None and abs(recorded-reduction)>1e-12):
                raise ValueError('Aggregate reduction')
            checks+=2
    if (report['numerical_attempts']!=9152 or report['numerical_successes']!=9152 or report['estimator_fits']!=6656
            or report['numerical_failures'] or report['unmatched_attempts'] or report['unparsed_call_lines']):
        raise ValueError('Incomplete cost/attempt accounting')
    for name in ('launch.json','report.json','development-jobs.json','panel-manifest.json','FINISHED.json'):
        inputs[str(root/name)]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    terminal=json.loads((root/'FINISHED.json').read_text())
    if terminal['complete'] is not True or terminal['report_sha256']!=inputs[str(root/'report.json')]:
        raise ValueError('Terminal report identity mismatch')
    inputs[str(worker_path)]=worker_sha;inputs[str(numerical)]=launch['numerical_source_sha256']
    return {'passed':True,'cases':208,'checks':checks,'input_sha256':inputs,'refits':0,'engy_calls':0,
            'final_gate_opened':False,'objective_established':False,
            'scope':'Independent saved-request, metric, temporal-choice, seasonal-arithmetic and accounting audit. ML outputs are tied to frozen implementation and runtime; they are not refitted.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.root)
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))
