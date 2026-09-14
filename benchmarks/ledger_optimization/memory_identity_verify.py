"""Independently audit the extra training IDs and protected metadata partitions."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(panel,validation,directory):
    panel,validation,directory=map(Path,(panel,validation,directory));read=lambda p:json.loads(p.read_text());checks=0
    def check(ok,message):
        nonlocal checks
        checks+=1
        if not ok:raise AssertionError(message)
    out=read(directory/'selection.json');original=read(panel/'selection.json');valid=read(validation/'selection.json')
    for domain in ('electricity','pedestrian'):
        eligible=read(panel/(domain+'-eligibility.json'))['eligible']
        ranked=sorted((hashlib.sha256(('20260914:panel035:'+domain+':'+r['series_name']).encode()).hexdigest(),r['series_name']) for r in eligible)
        names=[n for _,n in ranked];chosen=out[domain];get=lambda rows:[r['series_name'] for r in rows]
        check(get(original[domain]['development'])==names[:8],'Original development')
        check(get(original[domain]['reserved'])==names[8:24],'Final reserve')
        check(get(valid[domain]['validation'])==names[24:32],'Validation protected')
        check(get(chosen['memory_training'])==names[32:40],'Exact next eight identities')
        check(len(names)>=40 and len(set(names))==len(names),'Sufficient unique IDs')
        for field,wanted in [('original_development',names[:8]),('reserved_final_unchanged',names[8:24]),('validation_unchanged',names[24:32])]:
            check(chosen[field]==wanted,'Protected record metadata');check(not(set(get(chosen['memory_training']))&set(wanted)),'No protected overlap')
        by_name={r['series_name']:r for r in eligible}
        for r in chosen['memory_training']:check(r==by_name[r['series_name']],'Exact metadata')
    completed=read(directory/'COMPLETED.json')
    check(completed['selection_sha256']==hashlib.sha256((directory/'selection.json').read_bytes()).hexdigest(),'Selection hash')
    check(completed['observation_values_read']==completed['forecast_computations']==completed['api_calls']==0,'Metadata only')
    result={'checks':checks,'failures':0,'memory_series':16,'protected_final_series':32,'protected_validation_series':16,'observation_values_read':0}
    (directory/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('panel','validation','directory'):p.add_argument(n)
    print(json.dumps(audit(**vars(p.parse_args())),indent=2))
