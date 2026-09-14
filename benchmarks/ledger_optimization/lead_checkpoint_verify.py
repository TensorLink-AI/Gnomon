"""Audit an immutable completed prefix of live070; never report full-run success."""
import hashlib,json,math
from pathlib import Path
import sys,time
import numpy as np

NAMES=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest','search_selected')


def verify(root,incumbent):
    root,incumbent=Path(root),Path(incumbent);here=Path(__file__).parent;start=time.monotonic();checks=0
    def check(v,m):
        nonlocal checks
        checks+=1
        if not v:raise AssertionError(m)
    def load(p):return json.loads(Path(p).read_text())
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def near(a,b):check(np.allclose(a,b,atol=1e-9,rtol=1e-8),'Numerical disagreement')
    # The runner writes a complete case before advancing this count. Cases are
    # never modified afterward. A status read failure is an observation failure.
    count=load(root/'status.json')['completed_cases'];check(0<count<=416,'Valid finished-prefix count')
    receipt=load(here/'evidence/search-ensemble-068.json');old=[]
    for name,h in receipt['files'].items():
        if len(Path(name).stem)!=64:continue
        check(sha(incumbent/name)==h,'Frozen068identity');old.append(load(incumbent/name))
    check(len(old)==416,'Complete incumbent identity cohort')
    selected=sorted(old,key=lambda r:(r['origin'],r['series_id']))[:count];case_hashes={};max_gap=0.
    for previous in selected:
        name=previous['task_id']+'.json';p=root/name;before=sha(p);r=load(p)
        check(all(r[k]==previous[k] for k in ('task_id','series_id','origin','round','actual','retrieval','anchor_source')),'Identical task, retrieval and anchor')
        for arm in ('control','ledger'):
            rec=r['records'][arm];prior=previous['records'][arm]
            check({k:v for k,v in rec.items() if k!='fit'}=={k:v for k,v in prior.items() if k!='fit'},'All source pairs and semantic metadata unchanged')
            f=rec['fit'];check(f['input_sha256']==prior['fit']['input_sha256'],'Identical current and matured fit-input hash')
            a=np.array(f['anchor']);w=np.array(f['weights']);check(w.shape==(24,7) and np.isfinite(w).all() and w.min()>=0,'Hourly weight contract');near(w.sum(axis=1),np.ones(24))
            pairs=rec['pairs'];masses=rec['masses'];encoded=json.dumps({'pairs':pairs,'masses':masses,'anchor':f['anchor']},separators=(',',':')).encode()
            check(hashlib.sha256(encoded).hexdigest()==f['input_sha256'],'Actual input hash')
            def objective(weights):
                loss=.01/24*np.square(weights-a).sum();gradient=.02/24*(weights-a)
                for pair,mass in zip(pairs,masses,strict=True):
                    x=np.array([[math.log1p(pair['point'][m][h]) for m in NAMES] for h in range(24)]);y=np.log1p(pair['actual']);residual=np.sum(x*weights,axis=1)-y
                    norm=math.sqrt(float(np.mean(residual**2))+1e-12);loss+=mass*norm;gradient+=mass*x*residual[:,None]/(24*norm)
                return float(loss),gradient
            loss,g=objective(w);initial,_=objective(np.tile(a,(24,1)));gap=float(np.sum(w*g)-g.min(axis=1).sum());max_gap=max(max_gap,gap)
            near(f['objective'],loss);near(f['initial_objective'],initial);near(f['convex_gap_bound'],gap)
            check(f['success'] and gap<=1e-5+1e-9 and loss<=initial+1e-8,'Independent convergence certificate')
            point=[math.expm1(math.fsum(w[h,j]*math.log1p(rec['current_points'][m][h]) for j,m in enumerate(NAMES))) for h in range(24)]
            near(r['point'][arm],point)
            rmsle=math.sqrt(math.fsum((math.log1p(v)-math.log1p(y))**2 for v,y in zip(point,r['actual'],strict=True))/24);near(r['scores'][arm],rmsle)
        for arm in ('strong_block_cv','lifetime_ledger','search_control','search_ledger'):check(r['point'][arm]==previous['point'][arm] and r['scores'][arm]==previous['scores'][arm],'Comparators unchanged')
        check(sha(p)==before,'Case unchanged while read');case_hashes[name]=before
    result={'status':'verified_completed_prefix_only','full_run_verified':False,'completed_cases_audited':count,'expected_total_cases':416,'weight_fits_audited':count*2,
        'checks':checks,'failures':0,'maximum_recomputed_gap':max_gap,'seconds':time.monotonic()-start,'provider_calls':0,'api_calls':0,
        'case_sha256':case_hashes,'verifier_sha256':sha(__file__),'incumbent_receipt_sha256':sha(here/'evidence/search-ensemble-068.json'),
        'scope':'Completed chronological prefix only. Identical audited068 evidence, independent hourly objective and certificate, forecast/score reconstruction. No aggregate score or final gate claim.'}
    path=root/f'checkpoint-verification-{count:03d}.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+'\n');return {k:v for k,v in result.items() if k!='case_sha256'}


if __name__=='__main__':print(json.dumps(verify(*sys.argv[1:]),indent=2))
