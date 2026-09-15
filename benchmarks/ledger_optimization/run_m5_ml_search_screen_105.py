"""Execute the frozen development numerical grid once; no agent/API interface."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time

from .m5_ml_development_contract import authenticated_contract
from .m5_ml_search_screen_105 import CONFIGS,ENDS,POLICIES,choose,requests_for_job,rmsle

NUMERICAL_SHA='a71b75f06ad69fcca06275046c5e74565b15d7d6dd2538df0fe827c448fc0413'
WORKER_MANIFEST_SHA='123273702ac98c4eae16a2fde80043e252003ed95ce5b162de1cd44985077ebe'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def fingerprint(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def dump(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def append(path,value):
    with Path(path).open('a') as stream:stream.write(canonical(value)+'\n')


def source_identity():
    here=Path(__file__).parent
    names=('run_m5_ml_search_screen_105.py','m5_ml_search_screen_105.py',
           'm5_ml_development_contract.py','m5_ml_adapter.py','m5_ml_panel.py',
           'm5_prepare.py','M5_ML_SEARCH_SCREEN_105.md')
    return {name:sha(here/name) for name in names}


def load_numerical(path):
    if sha(path)!=NUMERICAL_SHA:raise ValueError('Frozen common numerical implementation required')
    spec=importlib.util.spec_from_file_location('screen_105_frozen_numerical',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if [module.configuration(c) for c in CONFIGS]!=CONFIGS:
        raise ValueError('Frozen grid differs from numerical configuration contract')
    return module


def series_run(series,jobs,root,numerical_path):
    numerical=load_numerical(numerical_path);home=Path(root)/'series'/series
    home.mkdir(parents=True,exist_ok=False);prior=[];results=[]
    for job in jobs:
        started=time.monotonic();folder=home/f"round-{job['round']:02d}"
        folder.mkdir();requests=requests_for_job(job)
        for end,value in requests.items():dump(folder/f'request-{end}.json',value)
        predictions={};failures=[];attempts=0;fits=0
        for i,config in enumerate(CONFIGS):
            cid=numerical.config_id(config)
            for end,value in requests.items():
                request=deepcopy(value['request']);digest=fingerprint(request)
                attempt={'event':'attempt','configuration_index':i,'config':config,'config_id':cid,
                         'end':end,'request_file':f'request-{end}.json','request_sha256':digest,
                         'at':datetime.now(timezone.utc).isoformat()}
                append(folder/'calls.jsonl',attempt);attempts+=1
                fits+=int(config['model']!='seasonal');at=time.monotonic()
                try:
                    point=numerical.predict(request,deepcopy(config))
                    if fingerprint(request)!=digest:raise ValueError('Model mutated its request')
                    # Validate shape/values without reading current production targets.
                    rmsle(point,[0.]*14)
                    if any(p<0 for p in point):raise ValueError('Frozen numerical clipping contract violated')
                    predictions[i,end]=point
                    append(folder/'calls.jsonl',{**attempt,'event':'result','point':point,
                        'elapsed_seconds':time.monotonic()-at,
                        'cv_rmsle':rmsle(point,value['actual']) if end<730 else None})
                except Exception as exc:
                    failure={**attempt,'event':'failure','error_type':type(exc).__name__,
                             'message':str(exc),'elapsed_seconds':time.monotonic()-at}
                    failures.append(failure);append(folder/'calls.jsonl',failure)
        row={'series_id':series,'round':job['round'],'origin':job['origin'],
             'target_end':job['future_timestamps'][-1],'recorded_at':job['outcome_recorded_at'],
             'complete':not failures,'numerical_attempts':attempts,
             'estimator_fit_attempts':fits,'estimator_fits':fits if not failures else None,
             'failure_count':len(failures),'seconds':time.monotonic()-started}
        if not failures:
            cv=[mean(rmsle(predictions[i,e],requests[e]['actual']) for e in ENDS[:-1])
                for i in range(len(CONFIGS))]
            decision=choose(cv,prior,job['origin'],series_id=series)
            # This immutable choice record is written before production scoring.
            dump(folder/'selection.json',{'current_cv_scores':cv,**decision})
            actual=deepcopy(job['actual']);dump(folder/'production-actuals.json',actual)
            production=[rmsle(predictions[i,730],actual) for i in range(len(CONFIGS))]
            best=min(range(len(CONFIGS)),key=lambda i:production[i])
            row.update(current_cv_scores=cv,production_scores=production,selection=decision,
                policy_scores={p:production[i] for p,i in decision['choices'].items()},
                hindsight_index=best,hindsight_score=production[best],
                unique_production_forecasts=len({tuple(predictions[i,730]) for i in range(len(CONFIGS))}))
            prior.append({k:row[k] for k in ('series_id','origin','target_end','recorded_at')}|{'scores':production})
        dump(folder/'result.json',row);results.append(row)
    if sha(numerical_path)!=NUMERICAL_SHA:raise ValueError('Numerical source changed during execution')
    return results


def aggregate(rows):
    if not rows or any(not r['complete'] for r in rows):return None
    means={p:mean(r['policy_scores'][p] for r in rows) for p in POLICIES}
    means['hindsight']=mean(r['hindsight_score'] for r in rows)
    base=means['current_cv']
    return {'cases':len(rows),'mean_rmsle':means,
            'reduction_vs_current_cv':{p:1-v/base if base else None for p,v in means.items()}}


def call_totals(root):
    """Count saved attempts even if a worker never returned its case summary."""
    attempts={};terminal={};unparsed=[]
    for path in sorted(Path(root).glob('series/*/round-*/calls.jsonl')):
        for number,line in enumerate(path.read_text().splitlines(),1):
            try:
                event=json.loads(line);key=(str(path),event['configuration_index'],event['end'])
                target=attempts if event['event']=='attempt' else terminal
                if event['event'] not in ('attempt','result','failure') or key in target:
                    raise ValueError('Unknown or duplicate call event')
                target[key]=event
            except (ValueError,KeyError,TypeError):
                unparsed.append({'file':str(path),'line':number})
    missing=sorted(set(attempts)-set(terminal));orphans=sorted(set(terminal)-set(attempts))
    failed=sum(e['event']=='failure' for e in terminal.values())
    successful_fits=sum(e['event']=='result' and e['config']['model']!='seasonal' for e in terminal.values())
    return {'numerical_attempts':len(attempts),'numerical_successes':sum(e['event']=='result' for e in terminal.values()),
            'numerical_failures':failed,'estimator_fit_attempts':sum(e['config']['model']!='seasonal' for e in attempts.values()),
            'estimator_fits':successful_fits if not (failed or missing or orphans or unparsed) else None,
            'successful_estimator_fits':successful_fits,'unmatched_attempts':missing,
            'orphan_terminals':orphans,'unparsed_call_lines':unparsed,
            'fit_count_basis':'One fitted forecasting estimator per successful non-seasonal call; failed/interrupted calls leave the exact fit count unknown.'}


def run(output,numerical_path,worker_manifest):
    output=Path(output).absolute();numerical_path=Path(numerical_path).absolute()
    if output.exists() or output.is_symlink():raise ValueError('Fresh screen output required; no implicit resume')
    if sha(worker_manifest)!=WORKER_MANIFEST_SHA:raise ValueError('Original frozen worker manifest required')
    proof=json.loads(Path(worker_manifest).read_text())
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    if packages!=proof['inventory']['plain']['packages'] or importlib.util.find_spec('gnomon') is not None:
        raise ValueError('Run under the frozen plain numerical interpreter')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('Single-thread numerical libraries required: '+name)
    if sha(numerical_path)!=NUMERICAL_SHA:raise ValueError('Frozen common numerical source required')
    manifest=Path('results/ledger-optimization/m5-panel-014/manifest.json')
    source=Path('results/m5-ml-development-prepare-001/prepared/development-jobs.json')
    manifest_bytes=manifest.read_bytes();job_bytes=source.read_bytes()
    cohort=authenticated_contract(manifest_bytes,job_bytes);jobs=json.loads(job_bytes)
    before=source_identity();output.mkdir(parents=True);start=time.monotonic()
    dump(output/'launch.json',{'pid':os.getpid(),'at':datetime.now(timezone.utc).isoformat(),
        'sources':before,'numerical_source_sha256':NUMERICAL_SHA,'packages':packages,
        'interpreter':sys.executable,'configs':CONFIGS,'ends':ENDS,'cases':len(cohort['cases']),
        'manifest_sha256':sha(manifest),'jobs_sha256':sha(source),'worker_manifest_sha256':sha(worker_manifest),
        'parallel_series':2,'engy_calls':0,'gnomon_engine_calls':0,'final_gate_opened':False})
    (output/'development-jobs.json').write_bytes(job_bytes)
    (output/'panel-manifest.json').write_bytes(manifest_bytes)
    rows=[];worker_failures=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(series_run,s,values,output,numerical_path):s for s,values in sorted(jobs.items())}
        for future in as_completed(futures):
            series=futures[future]
            try:rows.extend(future.result())
            except Exception as exc:worker_failures.append({'series_id':series,'type':type(exc).__name__,'message':str(exc)})
            append(output/'progress.jsonl',{'completed_cases':len(rows),'series_completed':series,
                'worker_failures':worker_failures,'elapsed_seconds':time.monotonic()-start})
            print(json.dumps({'completed_cases':len(rows),'worker_failures':len(worker_failures)}),flush=True)
    rows.sort(key=lambda r:(r['series_id'],r['round']))
    if (source_identity()!=before or sha(source)!=hashlib.sha256(job_bytes).hexdigest()
            or sha(manifest)!=hashlib.sha256(manifest_bytes).hexdigest() or sha(numerical_path)!=NUMERICAL_SHA):
        worker_failures.append({'type':'source_changed'})
    totals=call_totals(output)
    complete=(len(rows)==208 and not worker_failures and all(r['complete'] for r in rows)
              and totals['numerical_attempts']==9152 and totals['numerical_successes']==9152
              and not any(totals[k] for k in ('numerical_failures','unmatched_attempts','orphan_terminals','unparsed_call_lines')))
    report={'complete':complete,'cases':len(rows),'worker_failures':worker_failures,
            **totals,'seconds':time.monotonic()-start,
            'all':aggregate(rows) if complete else None,
            'cold_0_to_3':aggregate([r for r in rows if r['round']<4]) if complete else None,
            'later_4_to_25':aggregate([r for r in rows if r['round']>=4]) if complete else None,
            'by_series':{s:aggregate([r for r in rows if r['series_id']==s]) for s in jobs} if complete else {},
            'by_round':{str(n):aggregate([r for r in rows if r['round']==n]) for n in range(26)} if complete else {},
            'rows':rows,'engy_calls':0,'gnomon_engine_calls':0,'api_billing_dollars':0,
            'local_compute_billing_dollars':None,'final_gate_opened':False,'objective_established':False,
            'scope':'Fixed-grid numerical development diagnostic, not an agent arm or a causal ledger-value estimate.'}
    dump(output/'report.json',report)
    dump(output/('FINISHED.json' if complete else 'INCOMPLETE.json'),
         {'complete':complete,'report_sha256':sha(output/'report.json'),'automatic_retry':False})
    if not complete:raise RuntimeError('Incomplete grid retained; no scores promoted or fits retried')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--numerical-source',type=Path,required=True);p.add_argument('--worker-manifest',type=Path,required=True)
    a=p.parse_args();r=run(a.output,a.numerical_source,a.worker_manifest)
    print(json.dumps({k:v for k,v in r.items() if k not in ('rows','by_round','by_series')},indent=2))
