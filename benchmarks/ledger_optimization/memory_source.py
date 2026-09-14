"""Bounded source extraction for the named057 memory-training cohort."""
import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import time
from .broad_panel_prepare import SOURCES,numbers,source_rows
from .broad_warmup_prepare import collect
from .broad_warmup_available import boundaries
from .pedestrian_timestamp_coverage import SHA,FIRST_ORIGIN,END
from .publisher_panel_prepare import csv_rows,digest

HOUR=timedelta(hours=1)
COMBINED=6130
MAIN=4786


def split(values,start,source_sha,unit,prefix):
    if len(values)!=COMBINED:raise ValueError('Exact6130-hour bounded span required')
    main=values[1344:];warm=values[:2074]
    if any(v is None for v in main):raise ValueError('Missing main memory observation')
    main=numbers(main)
    if hashlib.sha256(json.dumps(main[:730],separators=(',',':')).encode()).hexdigest()!=prefix:raise ValueError('Initial eligibility prefix changed')
    warm=[numbers([v])[0] if v is not None else None for v in warm]
    if warm[-730:]!=main[:730]:raise ValueError('Warmup overlap changed')
    base={'source_sha256':source_sha,'unit':unit,'role':'additional_memory_training','included_in_scored_denominator':False}
    first=start+2074*HOUR
    return ({**base,'start_label':(start+1344*HOUR).isoformat(),'first_origin':first.isoformat(),'values':main},
        {**base,'start_label':start.isoformat(),'first_origin':(start+730*HOUR).isoformat(),'first_scored_origin':first.isoformat(),'values':warm})


def prepare(electricity,pedestrian,identity,coverage,output):
    identity,coverage,output=Path(identity),Path(coverage),Path(output);here=Path(__file__).parent
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic();read=lambda p:json.loads(p.read_text())
    def save(n,v):(output/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    manifest={'code_sha256':digest(__file__),'protocol_sha256':digest(here/'MEMORY_BREADTH_057.md'),
        'helper_sha256':{n:digest(here/n) for n in ('broad_panel_prepare.py','broad_warmup_prepare.py','broad_warmup_available.py','pedestrian_timestamp_coverage.py','publisher_panel_prepare.py')},
        'combined_hours':COMBINED,'main_hours':MAIN,'api_calls':0,'forecast_computations':0,'validation_or_reserved_counts_parsed':0}
    save('manifest.json',manifest)
    try:
        receipt_path=here/'evidence/broad-memory-identity-057.json';receipt=read(receipt_path)
        if digest(identity/'selection.json')!=receipt['files']['selection.json']:raise ValueError('Frozen memory IDs changed')
        if manifest['protocol_sha256']!=receipt['manifest']['protocol_sha256']:raise ValueError('Memory protocol changed')
        if digest(coverage)!=read(here/'evidence/pedestrian-source-036.json')['coverage_receipt_sha256']:raise ValueError('Coverage metadata changed')
        if digest(electricity)!=SOURCES['electricity']['sha'] or digest(pedestrian)!=SHA:raise ValueError('Archive changed')
        selection=read(identity/'selection.json');manifest.update(identity_receipt_sha256=digest(receipt_path),selection_sha256=digest(identity/'selection.json'),
            coverage_sha256=digest(coverage),source_sha256={'electricity':digest(electricity),'pedestrian':digest(pedestrian)})
        save('manifest.json',manifest)
        chosen={r['sensor_id']:r for r in selection['pedestrian']['memory_training']}
        start=FIRST_ORIGIN-2074*HOUR;end=END-168*HOUR
        if (end-start)//HOUR!=COMBINED:raise ValueError('Bounded time geometry changed')
        expected={r['sensor_id']:set(r['sensor_names']) for r in read(coverage)['sensors']}
        positions,names,_=collect(csv_rows(pedestrian),chosen,start,end)
        grid=[]
        for sensor in chosen:
            grid.append({'sensor_id':sensor,'missing_main_positions':sorted(set(range(1344,COMBINED))-set(positions[sensor])),
                'missing_combined_positions':sorted(set(range(COMBINED))-set(positions[sensor])),
                'duplicate_hours':sum(n!=1 for n in positions[sensor].values()),'sensor_names':sorted(names[sensor]),'identity_matches':names[sensor]==expected[sensor]})
        save('metadata-coverage.json',grid)
        if any(r['missing_main_positions'] or r['duplicate_hours'] or not r['identity_matches'] for r in grid):raise ValueError('Memory source grid/identity failed; no replacement')
        save('status.json',{'phase':'metadata_passed','seconds':time.monotonic()-started})
        _,_,raw=collect(csv_rows(pedestrian),chosen,start,end,counts=True)
        main={};warm={}
        for sensor,r in chosen.items():
            key='pedestrian:'+r['series_name'];main[key],warm[key]=split([raw[sensor].get(i) for i in range(COMBINED)],start,SHA,'pedestrians_per_source_hour',r['initial_history_sha256'])
        wanted={r['series_name']:r for r in selection['electricity']['memory_training']}
        for name,begin,tokens in source_rows(electricity):
            if name not in wanted:continue
            r=wanted[name];low=r['first_origin_index']-2074;high=r['end_index']-168
            if low<0 or high>len(tokens) or high-low!=COMBINED:raise ValueError('Memory electricity range unavailable')
            key='electricity:'+name;main[key],warm[key]=split(tokens[low:high],begin+low*HOUR,SOURCES['electricity']['sha'],SOURCES['electricity']['unit'],r['initial_history_sha256'])
        wanted_ids={d+':'+r['series_name'] for d,p in selection.items() for r in p['memory_training']}
        if set(main)!=set(warm) or set(main)!=wanted_ids or len(main)!=16:raise ValueError('Incomplete frozen identities')
        tasks=[t for key,span in sorted(warm.items()) for t in boundaries(key,span)]
        save('memory-spans.json',main);save('warmup-spans.json',warm);save('warmup-boundaries.json',tasks)
        save('COMPLETED.json',{'memory_series':16,'main_memory_cases':400,'usable_warmup_cases':sum(t['ready'] for t in tasks),
            'unavailable_warmup_cases':sum(not t['ready'] for t in tasks),'warmup_attempts':128,
            'memory_spans_sha256':digest(output/'memory-spans.json'),'warmup_spans_sha256':digest(output/'warmup-spans.json'),
            'api_calls':0,'forecast_computations':0,'validation_or_reserved_counts_parsed':0,'seconds':time.monotonic()-started})
        save('status.json',{'phase':'complete','seconds':time.monotonic()-started})
    except BaseException as e:
        save('FAILED.json',{'error':type(e).__name__,'message':str(e),'seconds':time.monotonic()-started,'forecast_computations':0,'api_calls':0});raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('electricity','pedestrian','identity','coverage','output'):p.add_argument(n)
    prepare(**vars(p.parse_args()))
