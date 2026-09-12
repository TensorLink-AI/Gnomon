"""Independent positional reconstruction audit for the continuous dev export.

Uses only the first/last anchor's arrays, without importing the exporter or its
timestamp-merge implementation. No model execution, scoring or final-data reads.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def verify(source, export):
    raw=json.loads(source.read_text());actual=json.loads(export.read_text());checks=0
    assert set(raw)==set(actual), 'Series inventory changed'
    for series, anchors in raw.items():
        first=next(x for x in anchors if x['round']==0)
        last=next(x for x in anchors if x['round']==25)
        # 25 fixed 14-day advances: append the last 350 observed days to the
        # original 730. The final 14 targets are never part of an earlier history.
        sales=first['request']['history']+last['request']['history'][-350:]+last['actual']
        dates=first['request']['timestamps']+last['request']['timestamps'][-350:]+last['future_timestamps']
        covariates=first['request']['past_covariates']+last['request']['past_covariates'][-350:]+last['request']['future_covariates']
        start=datetime.fromisoformat(dates[0])
        assert len(sales)==len(dates)==len(covariates)==1094
        assert dates==[(start+timedelta(days=i)).isoformat() for i in range(1094)]
        assert len(actual[series])==26
        checks+=3
        for number, job in enumerate(actual[series]):
            end=730+number*14; origin=dates[end-1]
            request=deepcopy(first['request'])
            request.update(history=sales[end-730:end],timestamps=dates[end-730:end],
                           past_covariates=covariates[end-730:end],future_covariates=covariates[end:end+14],
                           future_timestamps=dates[end:end+14],cutoff=origin,known_time_cutoff=origin)
            expected=dict(request=request,actual=[float(v) for v in sales[end:end+14]],
                          origin=origin,outcome_recorded_at=dates[end+13],
                          future_timestamps=dates[end:end+14],series_id=series,round=number)
            assert encoded(job)==encoded(expected), f'Independent task reconstruction failed: {series}/{number}'
            checks+=1
        for anchor in anchors:
            expected={k:anchor[k] for k in actual[series][anchor['round']]}
            assert encoded(actual[series][anchor['round']])==encoded(expected), 'Original anchor changed'
            checks+=1
    return {'passed':True,'checks':checks,'cases':sum(map(len,actual.values())),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'output_sha256':hashlib.sha256(export.read_bytes()).hexdigest(),
            'method':'Independent first/last positional reconstruction, all fields and original anchors',
            'provider_calls':0,'api_calls':0,'ledger_writes':0,'final_data_opened':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('export',type=Path);p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();result=verify(args.source,args.export)
    with args.receipt.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
