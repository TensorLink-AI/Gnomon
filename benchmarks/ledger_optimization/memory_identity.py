"""Select an extra memory-training cohort from frozen metadata only."""
import argparse
import hashlib
import json
from pathlib import Path
from .validation_identity import FIELDS


def choose(domain,eligible,original,validation):
    if len(eligible)<40 or len({r['series_name'] for r in eligible})!=len(eligible):raise ValueError('Forty unique eligible identities required')
    if any(set(r)-FIELDS for r in eligible):raise ValueError('Eligibility metadata only')
    ordered=sorted(eligible,key=lambda r:hashlib.sha256(f'20260914:panel035:{domain}:{r["series_name"]}'.encode()).hexdigest())
    names=lambda rows:[r['series_name'] for r in rows]
    if names(ordered[:8])!=names(original['development']) or names(ordered[8:24])!=names(original['reserved']) or names(ordered[24:32])!=names(validation['validation']):raise ValueError('Protected cohorts differ from frozen ordering')
    selected=ordered[32:40]
    if set(names(selected)) & set(names(ordered[:32])):raise ValueError('Protected identity overlap')
    return {'memory_training':selected,'original_development':names(ordered[:8]),'reserved_final_unchanged':names(ordered[8:24]),
        'validation_unchanged':names(ordered[24:32]),'eligible_count':len(eligible)}


def prepare(panel,validation,output):
    panel,validation,output=map(Path,(panel,validation,output));here=Path(__file__).parent
    output.mkdir(parents=True,exist_ok=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    read=lambda p:json.loads(p.read_text())
    def save(n,v):(output/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    manifest={'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(here/'MEMORY_BREADTH_057.md'),
        'files':{},'observation_values_read':0,'api_calls':0,'forecast_computations':0}
    save('manifest.json',manifest)
    try:
        old=read(here/'evidence/broad-panel-037.json');locked=read(here/'evidence/broad-validation-identity-052.json')
        for n in ('selection.json','electricity-eligibility.json','pedestrian-eligibility.json'):
            if sha(panel/n)!=old['files'][n]:raise ValueError('Original metadata changed: '+n)
            manifest['files']['panel/'+n]=sha(panel/n)
        if sha(validation/'selection.json')!=locked['files']['selection.json']:raise ValueError('Validation metadata changed')
        manifest['files']['validation/selection.json']=sha(validation/'selection.json')
        original=read(panel/'selection.json');valid=read(validation/'selection.json')
        selection={d:choose(d,read(panel/(d+'-eligibility.json'))['eligible'],original[d],valid[d]) for d in ('electricity','pedestrian')}
        save('selection.json',selection);save('manifest.json',manifest)
        save('COMPLETED.json',{'memory_training_series':16,'planned_main_memory_cases':400,'warmup_attempts':128,
            'scored_development_cases_unchanged':416,'reserved_series_unchanged':32,'validation_series_unchanged':16,
            'observation_values_read':0,'forecast_computations':0,'api_calls':0,'selection_sha256':sha(output/'selection.json')})
        return selection
    except BaseException as e:
        save('FAILED.json',{'error':type(e).__name__,'message':str(e),'observation_values_read':0});raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('panel','validation','output'):p.add_argument(n)
    r=prepare(**vars(p.parse_args()));print(json.dumps({d:[v['series_name'] for v in part['memory_training']] for d,part in r.items()},indent=2))
