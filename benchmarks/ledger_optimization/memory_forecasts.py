"""Generate fixed extra historical evidence with two auditable local workers."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import sys
import time
from .broad_screen import compute_case
from .warm_context import features

_SPANS=None
_OUTPUT=None


def save(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def initialize(source,output):
    global _SPANS,_OUTPUT
    root=Path(source);_OUTPUT=Path(output)
    _SPANS={kind:json.loads((root/name).read_text()) for kind,name in [('main','memory-spans.json'),('warm','warmup-spans.json')]}


def worker(task):
    series,round_number,index,kind,expected_origin=task
    stem=f'{series}-{round_number:03d}';started=time.monotonic();cpu=time.process_time()
    costs={'series_id':series,'round':round_number,'forecast_computations_started':0,'estimator_fits_started':0,'status':'started'}
    def charge(model):
        costs['forecast_computations_started']+=1;costs['estimator_fits_started']+=model in ('ridge_short','ridge_long','forest')
        costs['pending_model']=model;save(_OUTPUT/'jobs'/(stem+'.json'),costs)
    try:
        span=_SPANS[kind][series];row=compute_case(span,index,[],charge)
        row.update(series_id=series,round=round_number,role='additional_memory_training',included_in_scored_denominator=False)
        if expected_origin and row['origin'].replace('+00:00','')!=expected_origin:raise ValueError('Warm-up origin shifted')
        low,high=row['history_indices']
        context={**{k:row[k] for k in ('series_id','origin','last_target')},'domain':series.split(':')[0],
            'outcome_recorded_at':row['last_target'],'features':features(row['cv'],span['values'][low:high])}
        save(_OUTPUT/'raw'/(stem+'.json'),row);save(_OUTPUT/'contexts'/(stem+'.json'),context)
        costs.update(status='complete',seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
        save(_OUTPUT/'jobs'/(stem+'.json'),costs);return costs
    except BaseException as error:
        costs.update(status='failed',error=type(error).__name__,message=str(error),seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
        save(_OUTPUT/'jobs'/(stem+'.json'),costs);return costs


def run(source,output):
    source,output=Path(source),Path(output);here=Path(__file__).parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt_path=here/'evidence/broad-memory-source-058.json';receipt=json.loads(receipt_path.read_text())
    for name,h in receipt['files'].items():
        if sha(source/name)!=h:raise ValueError('Source058 changed: '+name)
    old_path=Path('results/broad-screen-038-001/manifest.json')
    original_receipt=json.loads((here/'evidence/broad-screen-038.json').read_text())
    if sha(old_path)!=original_receipt['files']['manifest.json']:raise ValueError('Original manifest changed')
    old=json.loads(old_path.read_text())
    if sha(here/'warm_context.py')!='592698661b84dbea7a5b27caf8310e42262595d62709f6107eaa2725203791fc':raise ValueError('Frozen context implementation changed')
    for name in ('broad_screen.py','hourly_numerical.py'):
        if sha(here/name)!=old['code_sha256'][name]:raise ValueError('Frozen forecasting implementation changed')
    warm=json.loads((source/'warmup-boundaries.json').read_text());main=json.loads((source/'memory-spans.json').read_text())
    tasks=[(t['series_id'],t['round'],t['round']+8,'warm',t['origin']) for t in warm if t['ready']]
    tasks += [(s,r,r,'main',None) for s in sorted(main) for r in range(25)]
    if len(tasks)!=525 or len({(t[0],t[1]) for t in tasks})!=525:raise ValueError('Exactly525 unique historical cases required')
    output.mkdir(parents=True,exist_ok=False)
    for name in ('raw','contexts','jobs'):(output/name).mkdir()
    save(output/'excluded-warmup.json',[t for t in warm if not t['ready']])
    save(output/'manifest.json',{'source_receipt_sha256':sha(receipt_path),'code_sha256':{n:sha(here/n) for n in
        ('memory_forecasts.py','MEMORY_FORECASTS_059.md','broad_screen.py','hourly_numerical.py','warm_context.py')},
        'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ('numpy','scikit-learn')},
        'workers':2,'start_method':'spawn','api_calls':0,'planned_cases':525,'planned_forecast_computations':12600,'planned_estimator_fits':6300,
        'validation_or_final_access':False,'role':'additional_memory_training','inherited_original_forecast_computations':12984})
    started=time.monotonic();completed=0;failure=None
    executor=ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(str(source),str(output)))
    futures=[]
    try:
        futures=[executor.submit(worker,t) for t in tasks]
        for future in as_completed(futures):
            result=future.result()
            if result['status']!='complete':raise RuntimeError('Historical case failed: '+json.dumps(result))
            completed+=1
            save(output/'status.json',{'phase':'forecasting','completed_cases':completed,'planned_cases':525,'workers':2,'seconds':time.monotonic()-started})
    except BaseException as error:
        failure={'error':type(error).__name__,'message':str(error)}
        for f in futures:f.cancel()
    finally:
        executor.shutdown(wait=True,cancel_futures=True)
    jobs=[json.loads(p.read_text()) for p in sorted((output/'jobs').glob('*.json'))]
    costs={'cases_started':len(jobs),'cases_completed':sum(r['status']=='complete' for r in jobs),
        'cases_failed':sum(r['status']=='failed' for r in jobs),'unfinished_cases':sum(r['status']=='started' for r in jobs),
        'worker_cpu_accounting_complete':all('cpu_seconds' in r for r in jobs),'forecast_computations_started':sum(r['forecast_computations_started'] for r in jobs),
        'estimator_fits_started':sum(r['estimator_fits_started'] for r in jobs),'worker_cpu_seconds':sum(r.get('cpu_seconds',0.) for r in jobs),
        'seconds':time.monotonic()-started,'api_calls':0,'role':'additional_memory_training'}
    if failure or costs['cases_completed']!=525 or costs['forecast_computations_started']!=12600 or costs['estimator_fits_started']!=6300:
        save(output/'FAILED.json',{**costs,**(failure or {'error':'CostOrCohortMismatch'})});raise RuntimeError('Historical generation failed; retained per-case artifacts')
    contexts=[json.loads(p.read_text()) for p in sorted((output/'contexts').glob('*.json'))]
    if len({(r['series_id'],r['origin']) for r in contexts})!=525:raise ValueError('Context identity mismatch')
    save(output/'contexts.json',contexts);save(output/'COMPLETED.json',costs);save(output/'status.json',{'phase':'complete',**costs});return costs

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('output')
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
