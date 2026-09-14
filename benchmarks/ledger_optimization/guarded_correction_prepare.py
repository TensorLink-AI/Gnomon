"""Synthetic-only077tree and held-back choice preparation; no source data."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import sklearn
from .guarded_correction import MODELS,features,training,fit,apply,decide


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def run(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);here=Path(__file__).parent;started=time.monotonic();weights=[1/6]*6
    def fixture(offset):
        point={m:[float(2+i+.2*offset+h%4) for h in range(24)] for i,m in enumerate(MODELS)};_,base=features(point,weights);bias=np.array([.2 if h%4>=2 else -.1 for h in range(24)])
        return {'point':point,'actual':np.expm1(base+bias).tolist()}
    current=[fixture(i) for i in range(2)];past=[fixture(i) for i in range(2,6)];pairs,masses=training(current,past);model,info=fit(pairs,masses,weights);held=fixture(6);result=apply(held['point'],weights,model)
    favorable=decide(result['base_point'],result['point'],held['actual']);unfavorable=decide(result['base_point'],result['point'],result['base_point']);tie=decide(result['base_point'],result['base_point'],held['actual'])
    write(root/'manifest.json',{'code_sha256':{n:sha(here/n) for n in ('GUARDED_CORRECTION_077.md','guarded_correction.py','guarded_correction_prepare.py')},'synthetic_only':True,'sklearn_version':sklearn.__version__,'source_data_access':False,'api_calls':0})
    write(root/'training.json',{'pairs':pairs,'masses':masses,'weights':weights});write(root/'model.json',info);write(root/'held_back.json',{'fixture':held,'application':result,'favorable_decision':favorable,'unfavorable_decision':unfavorable,'tie_decision':tie})
    report={'synthetic_only':True,'forest_fits':1,'trees':32,'training_cases':6,'training_rows':144,'held_back_rows':24,'positive_correction_enabled':favorable['correction_enabled'],'negative_correction_enabled':unfavorable['correction_enabled'],'tie_correction_enabled':tie['correction_enabled'],'api_calls':0,'source_forecasts':0,'seconds':time.monotonic()-started,'source_evaluation_status':'not_started'}
    write(root/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');print(json.dumps(run(**vars(p.parse_args())),indent=2))
