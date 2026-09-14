"""Independent array-based audit of063 fixed-proposal hindsight ceilings."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

MODELS=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')
ARMS=('global_cv','block_cv','legacy_ledger','recent_ledger','lifetime_ledger','learned_ledger')
ORACLES=('provider_oracle','proposal_oracle','union_oracle','stepwise_envelope')


def audit(original,proposals,directory):
    original,proposals,directory=map(Path,(original,proposals,directory));here=Path(__file__).parent;checks=0
    def check(ok,message):
        nonlocal checks
        checks+=1
        if not ok:raise AssertionError(message)
    def near(a,b):check(bool(np.allclose(a,b,atol=1e-11,rtol=1e-10)),'Numerical disagreement')
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=read(directory/'manifest.json')
    for n,h in manifest['code_sha256'].items():check(sha(here/n)==h,'Frozen code')
    for n,h in manifest['source_receipts_sha256'].items():check(sha(here/'evidence'/n)==h,'Frozen receipt')
    check(manifest['diagnostic_only'] and manifest['uses_current_future_actuals'] and not manifest['validation_or_final_access'] and not manifest['forecasts_or_policies_changed'],'Hindsight scope')
    def load(root,receipt):
        rows={}
        for n,h in read(here/'evidence'/receipt)['files'].items():
            if not n.startswith(('electricity:','pedestrian:')):continue
            check(sha(root/n)==h,'Unmodified source');row=read(root/n);key=row['series_id'],row['origin']
            check(key not in rows,'Unique identity');rows[key]=row
        check(len(rows)==416,'Full source cohort');return rows
    original=load(original,'broad-screen-038.json');proposals=load(proposals,'broad-learned-retrieval-062.json')
    check(set(original)==set(proposals),'Same scored tasks');results=[]
    for key,source in sorted(original.items(),key=lambda p:(p[0][1],p[0][0])):
        prior=proposals[key];row=read(directory/f'{key[0]}-{source["round"]:02d}.json')
        for field in ('series_id','origin','round','last_target'):check(row[field]==source[field],'Task unchanged')
        check(prior['actual']==source['actual'],'Actual identity');actual=np.asarray(source['actual'],dtype=float)
        points=np.asarray([source['point'][m] for m in MODELS]+[prior['point'][a] for a in ARMS],dtype=float)
        check(points.shape==(12,24) and actual.shape==(24,),'Full forecast shapes')
        check(np.isfinite(points).all() and np.isfinite(actual).all() and (points>=0).all() and (actual>=0).all(),'Finite nonnegative evidence')
        errors=np.log1p(points)-np.log1p(actual)[None,:];scores=np.sqrt(np.mean(errors**2,axis=1))
        selected=int(np.argmin([source['cv'][m] for m in MODELS]));check(row['selected_cv_provider']==source['selection']['cv']==MODELS[selected],'Historical CV decision')
        near(row['scores']['selected_cv'],scores[selected])
        for i,m in enumerate(MODELS):near(row['provider_scores'][m],scores[i]);near(source['scores'][m],scores[i])
        for i,a in enumerate(ARMS):near(row['scores'][a],scores[6+i]);near(prior['scores'][a],scores[6+i])
        groups=[('provider_oracle',scores[:6],MODELS),('proposal_oracle',scores[6:],ARMS),('union_oracle',scores,tuple('provider:'+m for m in MODELS)+tuple('proposal:'+a for a in ARMS))]
        # Use stored scalar scores only for exact ties; array reduction can differ by ulps.
        scalar_values=[row['provider_scores'][m] for m in MODELS]+[row['scores'][a] for a in ARMS]
        for name,values,labels in groups:
            index=int(np.argmin(values));near(row['scores'][name],values[index]);near(row['hindsight'][name]['score'],values[index])
            own=dict(zip(labels,scalar_values[:6] if name=='provider_oracle' else scalar_values[6:] if name=='proposal_oracle' else scalar_values,strict=True))
            minimum=min(own.values());ties=[label for label,value in own.items() if value==minimum]
            check(row['hindsight'][name]['ties']==ties and row['hindsight'][name]['selected']==ties[0],'Exact reported score ties and deterministic ordering')
        lo=points[:6].min(axis=0);hi=points[:6].max(axis=0)
        projected=np.clip(actual,lo,hi);residual=np.log1p(projected)-np.log1p(actual);bound=float(np.sqrt(np.mean(residual**2)))
        near(row['stepwise_envelope']['log_residual'],residual);near(row['stepwise_envelope']['score'],bound);near(row['scores']['stepwise_envelope'],bound)
        check(row['stepwise_envelope']['scope']=='independent_horizon_weights_outside_shared_block_action','Loose envelope action scope')
        check(bound<=scores[:6].min()+1e-10,'Envelope lower-bounds provider choices')
        check(np.all(points[6:]>=lo-1e-8*(1+lo)) and np.all(points[6:]<=hi+1e-8*(1+hi)),'Saved mixtures lie inside six-provider envelope')
        check(bound<=scores.min()+1e-10,'Envelope lower-bounds all saved mixtures')
        check(row['diagnostic_only'] and row['uses_current_future_actuals'],'Every hindsight case labeled')
        results.append(row)
    report=read(directory/'report.json')
    def summary(observed,rows):
        check(observed['cases']==len(rows),'Aggregate denominator')
        means={a:float(np.mean([r['scores'][a] for r in rows])) for a in observed['mean_rmsle']}
        for a,value in means.items():near(observed['mean_rmsle'][a],value)
        for baseline in ('selected_cv','block_cv'):
            for oracle in ORACLES:
                c=observed['comparisons'][baseline][oracle];gain=1-means[oracle]/means[baseline]
                near(c['maximum_relative_improvement'],gain)
                check(c['twenty_percent_status']==('impossible_in_stated_class' if means[oracle]>.8*means[baseline] else 'not_ruled_out_not_established'),'Scoped feasibility classification')
                if gain>0:near(c['fraction_of_oracle_gain_required'],.2/gain)
                else:check(c['fraction_of_oracle_gain_required'] is None,'Undefined fraction disclosed')
    summary(report['overall'],results)
    for d in ('electricity','pedestrian'):summary(report['domains'][d],[r for r in results if r['series_id'].startswith(d+':')])
    summary(report['phases']['early_0_7'],[r for r in results if r['round']<8]);summary(report['phases']['later_8_25'],[r for r in results if r['round']>=8])
    check(report['provider_calls']==report['api_calls']==report['weight_fits']==0 and report['inherited_forecast_computations']==25584,'Analysis-only costs')
    check(report['diagnostic_only'] and report['uses_current_future_actuals'],'Report scope')
    result={'checks':checks,'failures':0,'cases':416,'report_sha256':sha(directory/'report.json'),'verifier_sha256':sha(Path(__file__)),
        'scope':'Independent array RMSLE, minima, scalar exact tie lists, observation-space envelope projection, all aggregates and scoped numerical ceilings. Hindsight diagnostic, not an agent result or uncertainty interval.'}
    target=directory/'verification.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('original','proposals','directory'):p.add_argument(n)
    print(json.dumps(audit(**vars(p.parse_args())),indent=2))
