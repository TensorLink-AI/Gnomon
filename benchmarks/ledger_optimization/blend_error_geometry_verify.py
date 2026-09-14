"""Independent vector audit of069 projection and pooled-energy accounting."""
import hashlib,json
from pathlib import Path
import sys
import numpy as np


def verify(root,source):
    root,source=Path(root),Path(source);here=Path(__file__).parent;checks=0;rows=[]
    def check(x,m):
        nonlocal checks
        checks+=1
        if not x:raise AssertionError(m)
    def near(a,b):check(np.allclose(a,b,atol=1e-10,rtol=1e-9),'Numerical disagreement')
    def load(p):return json.loads(Path(p).read_text())
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=load(root/'manifest.json');receipt=load(here/'evidence/search-ensemble-068.json')
    check(sha(here/'evidence/search-ensemble-068.json')==manifest['source_receipt_sha256'],'Source receipt identity')
    for name,h in manifest['code_sha256'].items():check(sha(here/name)==h,'Frozen diagnostic identity')
    for name,h in manifest['accessed_source_files'].items():
        check(receipt['files'][name]==h and sha(source/name)==h,'Source file identity');src=load(source/name);r=load(root/name)
        check(all(r[k]==src[k] for k in ('task_id','series_id','round','origin')),'Task identity')
        for arm in ('control','ledger'):
            records=src['records'][arm];blocks=[(r['arms'][arm]['production'],records['current_points'],src['actual'],src['point'][arm])]
            blocks += [(g,p['point'],p['actual'],None) for g,p in zip(r['arms'][arm]['current_cv'],records['pairs'][:3],strict=True)]
            for g,points,actual,blend in blocks:
                p=np.log1p(np.array(list(points.values())));y=np.log1p(actual);lo=p.min(axis=0);hi=p.max(axis=0);z=np.clip(y,lo,hi);range_error=z-y
                near(g['lower_log'],lo);near(g['upper_log'],hi);near(g['actual_log'],y);near(g['projected_log'],z);near(g['range_residual'],range_error);near(g['range_floor_rmsle'],np.sqrt(np.mean(range_error**2)))
                check(g['position']==np.where(y<lo,'below',np.where(y>hi,'above','inside')).tolist(),'Range position counts')
                if blend is not None:
                    p=np.log1p(blend);b=p-z;e=p-y;check(np.all(p>=lo-1e-10) and np.all(p<=hi+1e-10),'Convex range inclusion')
                    near(g['blend_log'],p);near(g['blending_residual'],b);near(g['error'],e);near(g['blend_rmsle'],np.sqrt(np.mean(e**2)))
                    for k,v in {'range_energy':np.sum(range_error**2),'blending_energy':np.sum(b**2),'cross_energy':2*np.dot(range_error,b),'total_energy':np.sum(e**2)}.items():near(g[k],v)
                    near(e**2,range_error**2+b**2+2*range_error*b);check(np.all(range_error*b>=-1e-10),'Nonnegative pointwise cross term')
        check(r['uses_current_future_actuals'] and r['diagnostic_only'],'Hindsight disclosure');rows.append(r)
    report=load(root/'report.json');check(len(rows)==416 and len({r['task_id'] for r in rows})==416,'Full denominator')
    def summary(group,saved):
        check(saved['cases']==len(group),'Group denominator')
        for arm in ('control','ledger'):
            out=saved['arms'][arm];g=[r['arms'][arm]['production'] for r in group];error=np.array([r['error'] for r in g]);range_error=np.array([r['range_residual'] for r in g]);blend_error=np.array([r['blending_residual'] for r in g]);total=np.sum(error**2)
            near(out['mean_case_rmsle'],np.mean(np.sqrt(np.mean(error**2,axis=1))));near(out['mean_range_floor_rmsle'],np.mean(np.sqrt(np.mean(range_error**2,axis=1))))
            near(out['mean_current_cv_range_floor'],np.mean([cv['range_floor_rmsle'] for r in group for cv in r['arms'][arm]['current_cv']]))
            near(out['mean_signed_log_error'],np.mean(error));near(out['mean_signed_range_residual'],np.mean(range_error));near(out['pooled_squared_error'],total)
            positions=[v for r in g for v in r['position']];check(out['point_count']==len(positions),'Point denominator')
            check(out['position_counts']=={k:positions.count(k) for k in ('below','inside','above')},'Complete positional counts')
            near(list(out['energy_shares'].values()),[np.sum(range_error**2)/total,np.sum(blend_error**2)/total,2*np.sum(range_error*blend_error)/total]);near(sum(out['energy_shares'].values()),1.)
            check(out['range_floor_scope']=='independent per-horizon convex mixtures, outside shared four-block action','Scope preserved')
    summary(rows,report['overall'])
    for d in ('electricity','pedestrian'):summary([r for r in rows if r['series_id'].startswith(d+':')],report['domains'][d])
    summary([r for r in rows if r['round']<8],report['phases']['early_0_7']);summary([r for r in rows if r['round']>=8],report['phases']['later_8_25'])
    check(report['provider_calls']==report['weight_fits']==report['api_calls']==0,'Read-only cost')
    result={'checks':checks,'failures':0,'cases':len(rows),'provider_calls':0,'weight_fits':0,'api_calls':0,'verifier_sha256':sha(__file__),
        'scope':'Every production and current-CV projection, pointwise energy identity, mean-case RMSLE, pooled energy shares and all counts; no predictive or agent claim.'}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(verify(*sys.argv[1:]),indent=2))
