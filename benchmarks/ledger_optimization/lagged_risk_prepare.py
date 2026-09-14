"""Synthetic083 preparation; no source dataset or held-out accesses."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from . import lagged_risk as lr


def run(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    anchor=[1/6]*6;ids={m:'synthetic/'+m for m in lr.MODELS};pairs=[];metadata=[]
    def points(i):return {m:[float(2+3*j+.1*i)]*24 for j,m in enumerate(lr.MODELS)}
    def actual(i):
        low,high=[2.+.1*i]*12,[17.+.1*i]*12
        return low+high if i%2==0 else high+low
    def context(i):
        now=datetime(2026,1,2,tzinfo=timezone.utc)+timedelta(days=i)
        return now.isoformat(),{'origin':(now-timedelta(days=1)).isoformat(),
                                'last_target':now.isoformat(),'source_available_at':now.isoformat(),
                                'recorded_at':now.isoformat(),'config_ids':ids}
    for i in range(6):
        now,meta=context(i);meta=meta if i else None
        present=lr.validate_predecessor(meta,now,ids)
        prior={'point':points(i),'actual':actual(i)} if present else None
        pairs.append({'point':points(i),'actual':actual(i),'predecessor':prior})
        metadata.append({'origin':now,'predecessor':meta})
    now,meta=context(6);lr.validate_predecessor(meta,now,ids)
    query=points(6);prior={'point':points(6),'actual':actual(6)}
    masses=[1/6]*6;model,info=lr.fit(pairs,masses,anchor)
    matrices=lr.predict_matrices(model,query,anchor,prior);prediction=lr.combine(query,anchor,matrices)
    here=Path(__file__).parent
    manifest={'code_sha256':{n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in
              ('LAGGED_RISK_083.md','lagged_risk.py','lagged_risk_prepare.py','local_risk.py','guarded_correction.py','conditional_risk.py')},
              'synthetic_only':True,'source_data_access':False,'api_calls':0}
    training={'pairs':pairs,'masses':masses,'anchor':anchor,'metadata':metadata,'config_ids':ids}
    q={'point':query,'predecessor':prior,'origin':now,'predecessor_metadata':meta,
       'matrices':matrices.tolist(),'prediction':prediction}
    report={'synthetic_only':True,'training_cases':6,'training_rows':144,'features':17,
            'forest_fits':1,'trees':32,'weight_fits':24,'weight_iterations':prediction['weight_iterations'],
            'seconds':time.monotonic()-start,'api_calls':0,'base_forecasts':0,'source_evaluation_status':'not_started'}
    for n,v in (('manifest',manifest),('training',training),('model',info),('query',q),('report',report)):
        (root/(n+'.json')).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':print(json.dumps(run(*sys.argv[1:]),indent=2))
