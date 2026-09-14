"""Independent synthetic audit for073candidate scoring, timing and masses."""
from datetime import datetime
import hashlib,json,math
from pathlib import Path
import sys


def verify(root):
    root=Path(root);here=Path(__file__).parent;checks=0
    def load(n):return json.loads((root/n).read_text())
    def check(v,m):
        nonlocal checks
        checks+=1
        if not v:raise AssertionError(m)
    def near(a,b):check(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),'Numerical disagreement')
    manifest=load('manifest.json');records=load('records.json');traces=load('traces.json');keys=('0','0.25','0.5','0.75','1');tieorder=('0.5','0.25','0.75','0','1')
    for name,h in manifest['code_sha256'].items():check(hashlib.sha256((here/name).read_bytes()).hexdigest()==h,'Frozen source identity')
    check(len(records)==24 and len(traces)==25,'Full synthetic chronology')
    for r in records:
        check(r['synthetic_fixture'] and r['actual']==[1.]*24,'Synthetic source scope')
        check(datetime.fromisoformat(r['origin'])<datetime.fromisoformat(r['last_target'])<=datetime.fromisoformat(r['source_available_at']) and datetime.fromisoformat(r['last_target'])<=datetime.fromisoformat(r['recorded_at']),'Outcome timeline')
        check(set(r['candidates'])==set(keys),'Complete candidate grid')
        for k,v in r['candidates'].items():check(v['task_id']==r['task_id'] and v['arm']==r['arm'] and v['requested_strength']==float(k) and datetime.fromisoformat(v['forecast_recorded_at'])<=datetime.fromisoformat(r['origin']),'Executed-candidate identity and recording time')
    for t in traces:
        q=t['query'];now=datetime.fromisoformat(q['origin']);selected=t['selection'];pool={(r['series_id'],r['origin']):r for r in records if r['arm']==q['arm'] and r['domain']==q['domain'] and datetime.fromisoformat(r['origin'])<now and all(datetime.fromisoformat(r[k])<=now for k in ('last_target','source_available_at','recorded_at'))}
        cohort=[pool[n['series_id'],n['origin']] for n in t['neighbors']];ready=len(cohort)==16 and len({r['origin'] for r in cohort})>=3
        check(selected['ready']==ready and selected['eligible_records']==len(pool),'Only mature evidence contributes')
        check(selected['provider_calls']==selected['new_weight_fits']==0 and selected['missing_trial_records']==[],'Selection cost/cohort coverage')
        if not ready:
            check(selected['requested_strength']==.5 and selected['selection_basis']=='insufficient_matched_history' and selected['candidate_score_means'] is None and selected['cohort']==[],'Explicit insufficient-evidence fallback');continue
        means={}
        for k in keys:
            scores=[]
            for r in cohort:
                # Synthetic candidates are constant, so RMSLE independently
                # reduces to one absolute log-ratio rather than a norm routine.
                point=r['candidates'][k]['point'];check(len(point)==24 and all(v==point[0] for v in point),'Constant synthetic candidate')
                scores.append(abs(math.log((point[0]+1)/(r['actual'][0]+1))))
            means[k]=sum(scores)/len(scores);near(selected['candidate_score_means'][k],means[k])
        minimum=min(means.values());ties=[k for k in tieorder if abs(means[k]-minimum)<1e-13];check(str(selected['requested_strength']).rstrip('0').rstrip('.') in ties or selected['requested_strength']==0. and '0' in ties,'Optimal historical candidate selected')
        check(selected['selection_basis']=='lowest_matched_historical_candidate_rmsle','Selection meaning')
        check(selected['cohort']==[{k:r[k] for k in ('task_id','series_id','arm','domain','origin','last_target','source_available_at','recorded_at')} for r in cohort],'Full matched evidence provenance')
    mass_cases=load('mass-cases.json');check(len(mass_cases)==10,'Full mass grid')
    for i,r in enumerate(mass_cases):
        requested=float(keys[i%5]);effective=requested if i>=5 else 0.;check(r['requested_strength']==requested and r['effective_strength']==effective,'Requested/effective strength')
        near(sum(r['masses']),1.);check(all(m>0 for m in r['masses']),'Zero-mass inputs removed');near(sum(m for p,m in zip(r['pairs'],r['masses'],strict=True) if p['label']=='past'),effective)
        check(len(r['pairs'])==(3 if effective<1 else 0)+(16 if effective>0 else 0),'Effective training dimensions')
    report=load('report.json');check(report['mature_queries']==9 and report['chronological_queries']==25 and report['mass_cases']==10,'Report counts')
    check(report['provider_calls']==report['weight_fits']==report['api_calls']==0,'No real computation claims')
    result={'checks':checks,'failures':0,'chronological_queries':25,'candidate_risk_comparisons':45,'provider_calls':0,'weight_fits':0,'api_calls':0,'synthetic_only':True,'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(verify(sys.argv[1]),indent=2))
