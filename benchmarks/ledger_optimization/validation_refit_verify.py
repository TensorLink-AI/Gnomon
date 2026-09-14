"""Independent055 norm-dual certificate plus unchanged054 full validation audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from .ensemble_refinement import isolated

MODELS=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')


def exact_bound(w,pairs):
    value=1e-6*math.fsum((v-1/6)**2 for v in w);gradient=[2e-6*(v-1/6) for v in w];slack=0.
    for pair in pairs:
        x=[[math.log1p(pair['point'][m][h]) for m in MODELS] for h in range(24)]
        e=[math.fsum(w[j]*x[h][j] for j in range(6))-math.log1p(pair['actual'][h]) for h in range(24)]
        squared=math.fsum(v*v for v in e)/24;n=math.sqrt(squared);s=math.sqrt(squared+1e-12)
        value+=n/3;slack+=(n-squared/s)/3
        for j in range(6):gradient[j]+=math.fsum(x[h][j]*e[h] for h in range(24))/(72*s)
    return value,math.fsum(w[j]*gradient[j] for j in range(6))-min(gradient)+slack


def audit(source,directory):
    directory=Path(directory);here=Path(__file__).parent
    verifier=isolated('benchmarks.ledger_optimization._verify_055',here/'validation_verify.py')
    verifier.global_objective=exact_bound
    result=verifier.audit(source,directory)
    checks=0
    for path in sorted((directory/'cases').glob('*.json')):
        r=json.loads(path.read_text());c=r['fits']['global_cv']
        assert c['certificate_kind']=='exact_objective_norm_dual_support';checks+=1
        assert c['search_epsilon']==c['maximum_search_perturbation']==1e-6;checks+=1
        assert -1e-10<=c['search_objective']-c['objective']<=1e-6+1e-10;checks+=1
    note={'checks':checks,'failures':0,'base_audit_checks':result['checks'],
        'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Global exact-objective dual certificate replaces054 omitted-norm certificate; all other full-audit checks unchanged. Search perturbation and declared certificate checked.'}
    (directory/'numerical-amendment-verification.json').write_text(json.dumps(note,indent=2)+'\n');return {**result,'amendment_audit':note}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('directory');print(json.dumps(audit(**vars(p.parse_args())),indent=2))
