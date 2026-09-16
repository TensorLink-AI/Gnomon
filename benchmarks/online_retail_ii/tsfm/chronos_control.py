"""Direct Chronos-2 control: HTTP forecasts and arithmetic, no Gnomon execution or LLM."""
import argparse,json,sys
import importlib.metadata
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import date
from pathlib import Path
from .client import Client,parse_forecast,digest
from benchmarks.online_retail_ii.full_span.agent import load_case
from benchmarks.online_retail_ii.full_span.models import forecast,metrics
from benchmarks.online_retail_ii.full_span.baselines import summarize,fingerprint
from benchmarks.online_retail_ii.full_span.data import load_panel,sha,dump

POLICY={'mode':'explicit','model':'chronos2','freq':'D','horizon':14,'quantiles':[0.1,0.5,0.9],'return_members':True}


def run(args):
    root=Path(args.output);baseline=Path(args.baseline)
    if root.exists():raise ValueError('Fresh control output required')
    plan=json.loads((baseline/'plan.json').read_text())
    if plan['phase']!='full_span' or plan['planned_cases']!=2448:raise ValueError('Wrong benchmark')
    for name,version in plan['runtime'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Numerical runtime drift: '+name)
    source=Path(__file__).parents[1]/'full_span'
    for name,value in plan['code_sha256'].items():
        if sha(source/name)!=value:raise ValueError('Original source drift: '+name)
    manifest,panel=load_panel(args.panel)
    if sha(Path(args.panel)/'manifest.json')!=plan['source_manifest_sha256']:raise ValueError('Different panel')
    cases=sorted((baseline/'agent-cases').iterdir());tasks={c:load_case(c)[0] for c in cases}
    if len(cases)!=2448:raise ValueError('Missing tasks')
    origins=sorted({t['origin'] for t in tasks.values()});root.mkdir(parents=True)
    client=Client(args.chutes_key,root/'host-api');health=client.call('health')
    if health.get('status')!='ok':raise ValueError('TSFM service unhealthy')
    dump(root/'catalog-status.json',client.catalog())
    revision='paracast-explicit/'+digest({'policy':POLICY,'image':health.get('image'),'training_cutoff':'unknown'})
    dump(root/'plan.json',{'policy':POLICY,'planned_cases':2448,'seeds':'not applicable; one direct forecast per task',
        'baseline_plan_sha256':sha(baseline/'plan.json'),'revision':revision,'gnomon_execution':False,'llm_calls':0,
        'ledger':False,'native_memory':False,'feedback_enabled':False,'point':'API median clipped at zero',
        'training_cutoff':'unknown; retrospective test, training overlap not ruled out','service_image':health.get('image')})
    frames={s:f.assign(day=f.date.dt.strftime('%Y-%m-%d')) for s,f in panel.groupby('series_id')};rows=[]
    for task in tasks.values():
        frame=frames[task['series_id']];visible=frame[frame.day<=task['origin']].sort_values('date')
        if fingerprint(task['series_id'],visible.value,visible.day,task['future_timestamps'])!=task['request_fingerprint']:
            raise ValueError('Task history differs from reference panel')
    def one(c):
        task,history=load_case(c);label=task['case_id']+'-chronos2'
        error=None
        try:
            response=client.call('forecast',{**POLICY,'series':[list(map(float,history.value))]},label)
            value=parse_forecast(response,14)
            if value['api_meta']['models_used']!=['chronos2']:raise ValueError('Explicit Chronos-2 identity mismatch')
            fallback=False
        except (ValueError,KeyError,TypeError) as exc:
            value=forecast('seasonal_naive_7',history.value,14,date.fromisoformat(task['origin']));value['api_meta']={}
            error=type(exc).__name__;fallback=True
        f=frames[task['series_id']];actual=f[f.day.isin(task['future_timestamps'])].sort_values('date').value.tolist()
        if len(actual)!=14:raise ValueError('Incomplete target')
        receipt=json.loads((client.output/label/'receipt.json').read_text())
        return {'case_id':task['case_id'],'series_id':task['series_id'],'origin':task['origin'],'method':'chronos2_direct',
            'point':value['point'],'clipped_points':value['clipped_points'],'fallback_used':fallback,'error':error,
            'executed_provider':'seasonal_naive_7' if fallback else 'chronos2','request_fingerprint':task['request_fingerprint'],
            'api_meta':value['api_meta'],'api_seconds':receipt['seconds'],'api_response_sha256':receipt['response_sha256'],
            'metrics':metrics(value['point'],actual,history.value),'gnomon_provider_calls':0,'llm_calls':0}
    for day in origins:
        current=client.call('health',label='health-'+day)
        if current.get('status')!='ok' or current.get('image')!=health.get('image'):raise ValueError('TSFM health/image changed mid-run')
        selected=[c for c in cases if tasks[c]['origin']==day];batch=[];errors=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(one,c) for c in selected]):
                try:row=future.result()
                except Exception as exc:errors.append(type(exc).__name__);continue
                rows.append(row);batch.append(row)
                with (root/'scores.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                dump(root/'progress.json',{'completed_cases':len(rows),'planned_cases':2448,'summary':summarize(rows),
                    'credits_charged':sum(r['api_meta'].get('credits_charged',0) for r in rows),'gnomon_provider_calls':0,'llm_calls':0})
                print(json.dumps({'completed':len(rows),'case':row['case_id'],'fallback':row['fallback_used']}),flush=True)
        if errors:
            dump(root/'dispatch-errors.json',errors);raise ValueError('Control stage failed; completed forecasts retained')
        if sum(r['fallback_used'] for r in batch)>.1*len(batch):raise ValueError('TSFM error rate over 10 percent; retain failures and stop')
    if 'gnomon' in sys.modules:raise ValueError('Control unexpectedly imported Gnomon')
    dump(root/'FINISHED.json',{'cases':len(rows),'summary':summarize(rows),'gnomon_provider_calls':0,'llm_calls':0,
        'credits_charged':sum(r['api_meta'].get('credits_charged',0) for r in rows),'training_cutoff':'unknown'})


def main():
    p=argparse.ArgumentParser()
    for name in ('baseline','panel','output','chutes-key'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        if Path(args.output).exists():dump(Path(args.output)/'INCOMPLETE.json',{'cause':type(exc).__name__,'message':str(exc)})
        raise
if __name__=='__main__':main()
