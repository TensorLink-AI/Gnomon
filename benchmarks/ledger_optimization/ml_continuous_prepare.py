"""Reconstruct all 26 development origins from overlapping frozen host histories.

No fitting/scoring, candidate selection, provider calls or final data access.
Only the seven task fields accepted by the common lab are exported.
"""
from copy import deepcopy
from datetime import datetime,timedelta
import hashlib
import json
import math
from pathlib import Path
import argparse

REPO=Path(__file__).resolve().parents[2]
SOURCE=REPO/'results/hermes-ml-checkpoint-120-v1/setup/host-jobs-original.json'
SOURCE_SHA='836e05229535f144acab46e94d48fb0fbddf2acc9b9bf4696369f5149efb9ec9'
FIELDS=('request','actual','origin','outcome_recorded_at','future_timestamps','series_id','round')
DYNAMIC={'history','timestamps','past_covariates','future_covariates','future_timestamps','cutoff','known_time_cutoff'}


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def reconstruct(raw):
    output={};proof={}
    for series,anchors in sorted(raw.items()):
        by_round={a['round']:a for a in anchors}
        if len(by_round)!=len(anchors) or 0 not in by_round or 25 not in by_round:
            raise ValueError('Unique first and last anchor rounds required')
        template=by_round[0]['request'];static={k:v for k,v in template.items() if k not in DYNAMIC}
        first=datetime.fromisoformat(by_round[0]['origin'])
        if first.tzinfo is None:raise ValueError('Explicit timezone required')
        values={};covariates={}
        def add(table,time,value):
            key=datetime.fromisoformat(time)
            if key.tzinfo is None:raise ValueError('Explicit timezone required')
            if key in table and canonical(table[key])!=canonical(value):
                raise ValueError('Conflicting overlapping source observation')
            table[key]=value
        for anchor in anchors:
            r=anchor['request'];origin=first+timedelta(days=14*anchor['round'])
            if (anchor['series_id']!=series or r['series_id']!=series or
                    anchor['origin']!=origin.isoformat() or r['cutoff']!=anchor['origin'] or
                    r['known_time_cutoff']!=anchor['origin'] or
                    {k:v for k,v in r.items() if k not in DYNAMIC}!=static):
                raise ValueError('Inconsistent task identity or request metadata')
            if (r['horizon']!=14 or len(r['history'])!=730 or
                    r['past_covariate_names']!=r['future_covariate_names'] or
                    len(r['future_timestamps'])!=14 or
                    r['future_timestamps']!=anchor['future_timestamps'] or
                    anchor['outcome_recorded_at']!=r['future_timestamps'][-1]):
                raise ValueError('Unsupported anchor shape or availability convention')
            expected_history=[(origin-timedelta(days=i)).isoformat() for i in range(729,-1,-1)]
            expected_future=[(origin+timedelta(days=i)).isoformat() for i in range(1,15)]
            if r['timestamps']!=expected_history or r['future_timestamps']!=expected_future:
                raise ValueError('Nonconsecutive anchor timestamps')
            for t,y,c in zip(r['timestamps'],r['history'],r['past_covariates'],strict=True):
                add(values,t,y);add(covariates,t,c)
            for t,y,c in zip(r['future_timestamps'],anchor['actual'],r['future_covariates'],strict=True):
                add(values,t,y);add(covariates,t,c)
        if any(not math.isfinite(y) or y<0 for y in values.values()):
            raise ValueError('Invalid sales value; do not delete the case')
        width=len(template['past_covariate_names'])
        if any(len(v)!=width or any(not math.isfinite(x) for x in v) for v in covariates.values()):
            raise ValueError('Invalid known covariates')
        jobs=[]
        for number in range(26):
            origin=first+timedelta(days=14*number)
            history=[origin-timedelta(days=i) for i in range(729,-1,-1)]
            future=[origin+timedelta(days=i) for i in range(1,15)]
            if any(t not in values or t not in covariates for t in history+future):
                raise ValueError('Missing source observation; interpolation forbidden')
            request=deepcopy(template)
            request.update(history=[values[t] for t in history],timestamps=[t.isoformat() for t in history],
                           past_covariates=[deepcopy(covariates[t]) for t in history],
                           future_covariates=[deepcopy(covariates[t]) for t in future],
                           future_timestamps=[t.isoformat() for t in future],
                           cutoff=origin.isoformat(),known_time_cutoff=origin.isoformat())
            job={'request':request,'actual':[values[t] for t in future],
                 'origin':origin.isoformat(),'outcome_recorded_at':future[-1].isoformat(),
                 'future_timestamps':request['future_timestamps'],'series_id':series,'round':number}
            if number in by_round:
                expected={k:by_round[number][k] for k in FIELDS}
                if canonical(job)!=canonical(expected):raise ValueError('Original anchor changed')
            jobs.append(job)
        output[series]=jobs
        proof[series]={'original_rounds':sorted(by_round),'original_tasks_reproduced':len(anchors),
                       'new_intermediate_origins':26-len(anchors),'unique_observations':len(values),
                       'start':min(values).isoformat(),'end':max(values).isoformat()}
    return output,proof


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    root=p.parse_args().output;root.mkdir(parents=True,exist_ok=False)
    blob=SOURCE.read_bytes();assert hashlib.sha256(blob).hexdigest()==SOURCE_SHA
    jobs,proof=reconstruct(json.loads(blob))
    data=canonical(jobs)+'\n';(root/'host-jobs.json').write_text(data)
    receipt={'scope':'Continuous development preparation only','source':str(SOURCE),'source_sha256':SOURCE_SHA,
             'output_sha256':hashlib.sha256(data.encode()).hexdigest(),'cases':sum(map(len,jobs.values())),
             'series':proof,'provider_calls':0,'api_calls':0,'ledger_writes':0,'final_data_opened':False,
             'covariate_assumption':'Retains provided future onpromotion/calendar convention; not measured publication vintages.',
             'launch_authorized_by_this_receipt':False}
    (root/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
