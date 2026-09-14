"""Independent076 feature, convex-certificate, application and score audit."""
from datetime import datetime,timedelta
import hashlib,json,math
from pathlib import Path
import sys,time
import numpy as np

NAMES=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest','search_selected')


def verify(root,source):
    root,source=Path(root),Path(source);here=Path(__file__).parent;started=time.monotonic();checks=0
    def load(p):return json.loads(Path(p).read_text())
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def check(v,m):
        nonlocal checks
        checks+=1
        if not v:raise AssertionError(m)
    def near(a,b):check(np.allclose(a,b,atol=1e-9,rtol=1e-8),'Numerical disagreement')
    def metric(p,y):return math.sqrt(math.fsum((math.log1p(a)-math.log1p(b))**2 for a,b in zip(p,y,strict=True))/24)
    def raw(point,w):
        logs=[[math.log1p(point[n][h]) for n in NAMES] for h in range(24)];base=[math.fsum(w[h//6][j]*logs[h][j] for j in range(7)) for h in range(24)]
        return [[base[h]]+[logs[h][j]-base[h] for j in range(7)] for h in range(24)],base
    def design(z,location,scale):return [[1.]+[(z[h][j]-location[j])/scale[j] for j in range(8)]+[math.sin(h*2*math.pi/24),math.cos(h*2*math.pi/24)] for h in range(24)]
    manifest=load(root/'manifest.json');receipt_path=here/'evidence/search-ensemble-068.json';receipt=load(receipt_path);check(sha(receipt_path)==manifest['source_receipt_sha256'],'068receipt identity')
    for n,h in manifest['code_sha256'].items():check(sha(here/n)==h,'Frozen candidate source')
    for n,h in load(root/'source_access.json').items():check(h==receipt['files'][n]==sha(source/n),'Full read-source identity')
    check(load(source/'verification.json')['failures']==0,'068source audit passed')
    rows=[];iterations=evaluations=0;clips=dict.fromkeys(('control','ledger'),0);outside=dict.fromkeys(clips,0)
    for n,h in receipt['files'].items():
        if not(n.endswith('.json') and len(n)==69):continue
        check(sha(source/n)==h,'Frozen training evidence');old=load(source/n);r=load(root/n);now=datetime.fromisoformat(old['origin'])
        check(r['source_file']==n and r['source_sha256']==h,'Bound source execution case')
        for k in ('task_id','series_id','origin','round','actual'):check(r[k]==old[k],'Unchanged task/outcome')
        for arm in ('control','ledger'):
            source_record=old['records'][arm];record=r['records'][arm];f=record['fit'];application=record['application'];pairs=source_record['pairs'];q=source_record['masses'];weights=source_record['fit']['weights'];refs=source_record['historical_references']
            check(record['source_record']==arm and record['historical_references']==refs,'Unchanged arm evidence')
            check((arm=='control' and not refs and len(pairs)==3 and q==[1/3]*3) or (arm=='ledger' and len(refs)==16 and len(pairs)==19 and q==[1/6]*3+[.5/16]*16),'Exact evidence budget')
            for ref in refs:
                at=datetime.fromisoformat(ref['origin']);sa=datetime.fromisoformat(ref['source_available_at']);ra=datetime.fromisoformat(ref['recorded_at']);check(at<now and at+timedelta(hours=24)<=now and sa<=now and ra<=now and at.timetz()==now.timetz(),'Causal mature and phase-matched history')
            encoded=json.dumps({'pairs':pairs,'masses':q,'weights':weights},separators=(',',':')).encode();check(hashlib.sha256(encoded).hexdigest()==f['input_sha256'] and f['weights']==weights,'Training input fingerprint')
            rawpairs=[raw(p['point'],weights) for p in pairs];location=[math.fsum(q[i]*z[h][j]/24 for i,(z,b) in enumerate(rawpairs) for h in range(24)) for j in range(8)]
            scale=[max(math.sqrt(math.fsum(q[i]*(z[h][j]-location[j])**2/24 for i,(z,b) in enumerate(rawpairs) for h in range(24))),.1) for j in range(8)];near(location,f['location']);near(scale,f['scale'])
            theta=np.array(f['coefficients']);check(theta.shape==(11,) and np.isfinite(theta).all(),'Finite coefficients');value=.1*float(theta@theta);initial=0.;gradient=.2*theta
            for i,(z,b) in enumerate(rawpairs):
                x=np.array(design(z,location,scale));residual=np.log1p(pairs[i]['actual'])-b;error=x@theta-residual;norm=math.sqrt(float(error@error)/24+1e-12);value+=q[i]*norm;initial+=q[i]*math.sqrt(float(residual@residual)/24+1e-12);gradient+=q[i]*(x.T@error)/(24*norm)
            bound=float(gradient@gradient)/.4;near(f['objective'],value);near(f['initial_objective'],initial);near(f['gradient'],gradient);near(f['suboptimality_bound'],bound);check(f['penalty']==.1 and f['strong_convexity']==.2,'Declared strong convexity');check(f['success'] and bound<=1e-8+1e-12 and value<=initial+1e-10,'Independent convergence certificate')
            iterations+=f['iterations'];evaluations+=f['function_evaluations']
            z,base=raw(source_record['current_points'],weights);x=np.array(design(z,location,scale));correction=x@theta;logs=np.array(base)+correction;point=[math.expm1(max(0.,v)) for v in logs];near(application['point'],point);near(application['base_log'],base);near(application['correction_log'],correction);near(application['raw_predicted_log'],logs);check(r['point'][arm]==application['point'],'Executed derived forecast')
            clipped=[i for i,v in enumerate(logs) if v<0];extrapolated=[i for i,v in enumerate(point) if v<min(source_record['current_points'][m][i] for m in NAMES)-1e-10 or v>max(source_record['current_points'][m][i] for m in NAMES)+1e-10];check(application['clipped_leads']==clipped and application['outside_model_range_leads']==extrapolated,'Clipping and extrapolation disclosed');clips[arm]+=len(clipped);outside[arm]+=len(extrapolated)
        for a,b in (('uncorrected_control','control'),('uncorrected_ledger','ledger'),('strong_block_cv','strong_block_cv'),('lifetime_ledger','lifetime_ledger')):check(r['point'][a]==old['point'][b],'Unchanged stronger guard')
        for a,p in r['point'].items():near(r['scores'][a],metric(p,r['actual']))
        rows.append(r)
    report=load(root/'report.json');check(len(rows)==report['completed_cases']==416 and report['correction_fits']==832,'Complete paired cases');check(iterations==report['iterations'] and evaluations==report['function_evaluations'],'Optimizer costs');check(clips==report['clipped_leads'] and outside==report['outside_model_range_leads'],'All transformed leads accounted')
    def summarize(group,saved):
        check(saved['cases']==len(group),'Case denominator');means={a:math.fsum(metric(r['point'][a],r['actual']) for r in group)/len(group) for a in group[0]['point']}
        for a,v in means.items():near(saved['mean_rmsle'][a],v)
        for a,v in means.items():
            if a!='ledger':near(saved['ledger_reduction'][a],1-means['ledger']/v)
    summarize(rows,report['overall'])
    for d in ('electricity','pedestrian'):summarize([r for r in rows if r['series_id'].startswith(d+':')],report['domains'][d])
    summarize([r for r in rows if r['round']<8],report['phases']['early_0_7']);summarize([r for r in rows if r['round']>=8],report['phases']['later_8_25'])
    g=report['overall']['ledger_reduction'];guards=('control','strong_block_cv','uncorrected_ledger','lifetime_ledger');passed=g['control']>=.2 and g['strong_block_cv']>=.2 and all(g[a]>0 for a in guards) and all(d['ledger_reduction'][a]>0 for d in report['domains'].values() for a in guards);check(report['development_gate_passed']==passed,'Frozen gate');check(report['new_provider_calls']==report['api_calls']==0 and manifest['validation_or_final_access'] is False,'No new source/API or protected access')
    for k,v in {'inherited_forecast_computations':49616,'inherited045anchor_fits':416,'inherited068blend_fits':832,'inherited_search_surrogate_solves':11902,'inherited_search_logical_attempts_per_arm':31378}.items():check(manifest[k]==v,'Inherited costs')
    result={'checks':checks,'failures':0,'cases':416,'certified_correction_fits':832,'seconds':time.monotonic()-started,'new_forecasts':0,'api_calls':0,'verifier_sha256':sha(__file__),'scope':'Independent training scalers/designs, temporal source references, objective/gradient/strong-convexity certificates, all transformed forecasts and clipping/extrapolation, score aggregates and inherited costs. Reuses audited068raw inputs.'}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(verify(*sys.argv[1:]),indent=2))
