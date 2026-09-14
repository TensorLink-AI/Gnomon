"""Independent audit of058 bounded memory spans and all planned tasks."""
import argparse
from datetime import datetime,timedelta
import hashlib
import json
import math
from pathlib import Path


def audit(identity,directory):
    identity,directory=map(Path,(identity,directory));read=lambda p:json.loads(p.read_text());checks=0
    def check(ok,message):
        nonlocal checks
        checks+=1
        if not ok:raise AssertionError(message)
    chosen=read(identity/'selection.json');main=read(directory/'memory-spans.json');warm=read(directory/'warmup-spans.json')
    tasks=read(directory/'warmup-boundaries.json');done=read(directory/'COMPLETED.json')
    expected={d+':'+r['series_name']:r for d,part in chosen.items() for r in part['memory_training']}
    check(set(main)==set(warm)==set(expected) and len(main)==16,'Exact memory identities')
    for domain,part in chosen.items():
        for field in ('original_development','reserved_final_unchanged','validation_unchanged'):
            check(not(set(main)&{domain+':'+n for n in part[field]}),'No protected identity')
    by_task={(t['series_id'],t['round']):t for t in tasks};check(len(by_task)==len(tasks)==128,'Unique warm tasks')
    usable=0
    for name,span in main.items():
        values=span['values'];earlier=warm[name]['values'];start=datetime.fromisoformat(span['start_label']);early=datetime.fromisoformat(warm[name]['start_label'])
        check(len(values)==4786 and len(earlier)==2074,'Bounded lengths')
        check(start==early+timedelta(hours=1344),'Warm offset')
        check(datetime.fromisoformat(span['first_origin'])==start+timedelta(hours=730),'First origin')
        check(values[:730]==earlier[-730:],'Shared prefix')
        check(hashlib.sha256(json.dumps(values[:730],separators=(',',':')).encode()).hexdigest()==expected[name]['initial_history_sha256'],'Frozen prefix hash')
        check(all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in values),'Finite main observations')
        check(all(v is None or (isinstance(v,(int,float)) and math.isfinite(v) and v>=0) for v in earlier),'Warm finite/missing')
        check(span['role']==warm[name]['role']=='additional_memory_training' and span['included_in_scored_denominator'] is False,'Training-only role')
        for i in range(25):
            stop=730+168*i;check(len(values[stop-730:stop])==730 and len(values[stop:stop+24])==24,'Complete main case')
            close=start+timedelta(hours=stop+24);next_scored=start+timedelta(hours=730+168*(i+1))
            check(close<next_scored,'Training outcome matures before next original scored origin')
        check(values[730+168*25:]==[],'Unused last origin absent')
        for i in range(8):
            stop=730+168*i;t=by_task[name,i-8];mh=sum(v is None for v in earlier[stop-730:stop]);mt=sum(v is None for v in earlier[stop:stop+24])
            check(t['missing_history']==mh and t['missing_targets']==mt and t['ready']==(mh+mt==0),'Warm availability')
            check(t['history_indices']==[stop-730,stop] and t['target_indices']==[stop,stop+24],'Warm slices')
            check(datetime.fromisoformat(t['origin'])==early+timedelta(hours=stop),'Warm dates')
            check(datetime.fromisoformat(t['last_target'])==early+timedelta(hours=stop+24)<datetime.fromisoformat(span['first_origin']),'Warm maturity')
            usable+=t['ready']
    for stem in ('memory','warmup'):
        check(hashlib.sha256((directory/(stem+'-spans.json')).read_bytes()).hexdigest()==done[stem+'_spans_sha256'],'Source span hash')
    check(done['main_memory_cases']==400 and done['usable_warmup_cases']==usable and done['unavailable_warmup_cases']==128-usable,'Task counts')
    check(done['api_calls']==done['forecast_computations']==done['validation_or_reserved_counts_parsed']==0,'Preparation only')
    result={'checks':checks,'failures':0,'memory_series':16,'main_memory_cases':400,'usable_warmup_cases':usable,'unavailable_warmup_cases':128-usable,
        'scope':'Named identities, protected partitions, prefix hashes, complete bounded task slices, missingness and temporal geometry; no forecast scores.'}
    (directory/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('identity');p.add_argument('directory');print(json.dumps(audit(**vars(p.parse_args())),indent=2))
