"""Exact hindsight range geometry; no deployable policy or fit."""
import argparse
import hashlib,json,math,time
from pathlib import Path


def geometry(points,actual,blend=None):
    if not points or not actual or any(len(p)!=len(actual) for p in points):raise ValueError('Matched forecasts required')
    if any(not math.isfinite(v) or v<0 for row in [actual]+points+([] if blend is None else [blend]) for v in row):raise ValueError('Finite nonnegative values required')
    if blend is not None and len(blend)!=len(actual):raise ValueError('Complete blend required')
    lo=[min(math.log1p(p[h]) for p in points) for h in range(len(actual))];hi=[max(math.log1p(p[h]) for p in points) for h in range(len(actual))]
    y=[math.log1p(v) for v in actual];z=[min(u,max(l,v)) for l,u,v in zip(lo,hi,y,strict=True)]
    r=[p-v for p,v in zip(z,y,strict=True)];position=['below' if v<l else 'above' if v>u else 'inside' for l,u,v in zip(lo,hi,y,strict=True)]
    result={'lower_log':lo,'upper_log':hi,'actual_log':y,'projected_log':z,'range_residual':r,'position':position,
        'range_floor_rmsle':math.sqrt(math.fsum(v*v for v in r)/len(actual))}
    if blend is not None:
        p=[math.log1p(v) for v in blend]
        if any(v<l-1e-10 or v>u+1e-10 for l,u,v in zip(lo,hi,p,strict=True)):raise ValueError('Blend outside convex range')
        b=[v-u for v,u in zip(p,z,strict=True)];e=[v-u for v,u in zip(p,y,strict=True)]
        result.update(blend_log=p,blending_residual=b,error=e,blend_rmsle=math.sqrt(math.fsum(v*v for v in e)/len(actual)),
            range_energy=math.fsum(v*v for v in r),blending_energy=math.fsum(v*v for v in b),cross_energy=2*math.fsum(v*u for v,u in zip(r,b,strict=True)),total_energy=math.fsum(v*v for v in e))
        if result['cross_energy']< -1e-9 or not math.isclose(result['range_energy']+result['blending_energy']+result['cross_energy'],result['total_energy'],abs_tol=1e-9,rel_tol=1e-10):raise ValueError('Projection identity failed')
    return result


def summarize(rows):
    result={'cases':len(rows),'arms':{}}
    for arm in ('control','ledger'):
        gs=[r['arms'][arm]['production'] for r in rows];total=math.fsum(g['total_energy'] for g in gs);mean=lambda key:math.fsum(g[key] for g in gs)/len(gs)
        errors=[e for g in gs for e in g['error']];r=[e for g in gs for e in g['range_residual']];positions=[v for g in gs for v in g['position']]
        result['arms'][arm]={'mean_case_rmsle':mean('blend_rmsle'),'mean_range_floor_rmsle':mean('range_floor_rmsle'),
            'mean_current_cv_range_floor':math.fsum(math.fsum(c['range_floor_rmsle'] for c in row['arms'][arm]['current_cv'])/3 for row in rows)/len(rows),
            'point_count':len(positions),'position_counts':{k:positions.count(k) for k in ('below','inside','above')},
            'mean_signed_log_error':math.fsum(errors)/len(errors),'mean_signed_range_residual':math.fsum(r)/len(r),
            'pooled_squared_error':total,'energy_shares':{k:math.fsum(g[k] for g in gs)/total if total else None for k in ('range_energy','blending_energy','cross_energy')},
            'energy_share_denominator':'sum of squared log errors over all case-horizon points, not mean case RMSLE',
            'range_floor_scope':'independent per-horizon convex mixtures, outside shared four-block action'}
    return result


def run(source,output):
    source,output=Path(source),Path(output);here=Path(__file__).parent;receiptpath=here/'evidence/search-ensemble-068.json';receipt=json.loads(receiptpath.read_text());start=time.monotonic();output.mkdir(parents=True,exist_ok=False);rows=[];accessed={}
    for name,h in receipt['files'].items():
        if len(Path(name).stem)!=64:continue
        p=source/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('Changed068source')
        original=json.loads(p.read_text());accessed[name]=h;arms={}
        for arm in ('control','ledger'):
            rec=original['records'][arm];arms[arm]={'production':geometry(list(rec['current_points'].values()),original['actual'],original['point'][arm]),
                'current_cv':[geometry(list(pair['point'].values()),pair['actual']) for pair in rec['pairs'][:3]]}
        row={**{k:original[k] for k in ('task_id','series_id','round','origin')},'arms':arms,'uses_current_future_actuals':True,'diagnostic_only':True};rows.append(row)
        (output/name).write_text(json.dumps(row,indent=2)+'\n')
    if len(rows)!=416 or len({r['task_id'] for r in rows})!=416:raise ValueError('Full original development cohort required')
    report={'overall':summarize(rows),'domains':{d:summarize([r for r in rows if r['series_id'].startswith(d+':')]) for d in ('electricity','pedestrian')},
        'phases':{'early_0_7':summarize([r for r in rows if r['round']<8]),'later_8_25':summarize([r for r in rows if r['round']>=8])},
        'seconds':time.monotonic()-start,'provider_calls':0,'weight_fits':0,'api_calls':0,'diagnostic_only':True,'uses_current_future_actuals':True,
        'inherited_forecast_computations':49616,'inherited_068weight_fits':832,'limitation':'Projection uses future actuals and allows horizon-specific weights outside068action. Not a practical policy, causal explanation or held-out result.'}
    manifest={'protocol':'BLEND_ERROR_GEOMETRY_069.md','code_sha256':{n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ('BLEND_ERROR_GEOMETRY_069.md','blend_error_geometry.py')},
        'source_receipt_sha256':hashlib.sha256(receiptpath.read_bytes()).hexdigest(),'accessed_source_files':accessed,'validation_or_final_access':False,'policy_changed':False}
    for name,r in [('report.json',report),('manifest.json',manifest)]: (output/name).write_text(json.dumps(r,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('output');print(json.dumps(run(**vars(p.parse_args())),indent=2))
