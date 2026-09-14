"""Synthetic085case-norm-scaled PSD and simplex preparation."""
import argparse,hashlib,json,time
from pathlib import Path
from .norm_risk import MODELS,fit,predict_matrices,combine


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def run(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);here=Path(__file__).parent;start=time.monotonic();anchor=[1/6]*6
    def points(offset):return {m:[float(2+3*i+.1*offset)]*24 for i,m in enumerate(MODELS)}
    pairs=[{'point':points(i),'actual':[2.+.1*i]*12+[17.+.1*i]*12} for i in range(6)];masses=[1/6]*6;query=points(6);model,info=fit(pairs,masses,anchor);matrices=predict_matrices(model,query,anchor);prediction=combine(query,anchor,matrices)
    write(root/'manifest.json',{'code_sha256':{n:sha(here/n) for n in ('NORM_RISK_085.md','norm_risk.py','norm_risk_prepare.py','local_risk.py','guarded_correction.py','conditional_risk.py')},'synthetic_only':True,'source_data_access':False,'api_calls':0})
    write(root/'training.json',{'pairs':pairs,'masses':masses,'anchor':anchor});write(root/'model.json',info);write(root/'query.json',{'point':query,'matrices':matrices.tolist(),'prediction':prediction})
    report={'synthetic_only':True,'training_cases':6,'training_rows':144,'forest_fits':1,'trees':32,'weight_fits':24,'weight_iterations':prediction['weight_iterations'],'seconds':time.monotonic()-start,'api_calls':0,'base_forecasts':0,'source_evaluation_status':'not_started'};write(root/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');print(json.dumps(run(**vars(p.parse_args())),indent=2))
