"""Synthetic chronological preflight; no real provider or source-data scoring."""
from datetime import datetime,timedelta,timezone
import hashlib,json
from pathlib import Path
import sys,time
from .memory_strength import KEYS,blend_inputs,choose


def run(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);start=time.monotonic();records=[];first=datetime(2020,1,1,tzinfo=timezone.utc)
    for i in range(24):
        origin=(first+timedelta(days=i)).isoformat();close=(first+timedelta(days=i+1)).isoformat();tid='synthetic-'+str(i);best=.25 if i<12 else .75
        records.append({'task_id':tid,'series_id':'electricity:synthetic','arm':'ledger','domain':'electricity','origin':origin,'last_target':close,'source_available_at':close,'recorded_at':close,'actual':[1.]*24,'synthetic_fixture':True,
            'candidates':{k:{'task_id':tid,'arm':'ledger','requested_strength':float(k),'execution_id':tid+':'+k,'forecast_recorded_at':origin,'point':[1+abs(float(k)-best)*(1+i/24)]*24} for k in KEYS}})
    traces=[]
    for i in range(25):
        query={'series_id':'electricity:query','arm':'ledger','domain':'electricity','origin':(first+timedelta(days=i)).isoformat()};past=records[max(0,i-16):i];neighbors=[{k:r[k] for k in ('series_id','origin')} for r in past]
        traces.append({'query':query,'neighbors':neighbors,'selection':choose(query,records,neighbors)})
    current=[{'point':{},'actual':[],'label':'current'}]*3;past=[{'point':{},'actual':[],'label':'past'}]*16
    mass_cases=[blend_inputs(current,history,float(k)) for history in ([],past) for k in KEYS]
    here=Path(__file__).parent;manifest={'protocol':'MEMORY_STRENGTH_073.md','code_sha256':{n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ('MEMORY_STRENGTH_073.md','memory_strength.py','memory_strength_prepare.py')},
        'synthetic_only':True,'source_data_access':False,'provider_calls':0,'weight_fits':0,'api_calls':0}
    report={'status':'synthetic_selection_prepared_not_source_evaluated','chronological_queries':len(traces),'mature_queries':sum(r['selection']['ready'] for r in traces),'mass_cases':len(mass_cases),'provider_calls':0,'weight_fits':0,'api_calls':0,'seconds':time.monotonic()-start}
    for n,v in [('manifest.json',manifest),('records.json',records),('traces.json',traces),('mass-cases.json',mass_cases),('report.json',report)]: (root/n).write_text(json.dumps(v,indent=2)+'\n')
    return report


if __name__=='__main__':print(json.dumps(run(sys.argv[1]),indent=2))
