"""Independent065 synthetic acquisition audit using scalar kernels and Cholesky."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve


def audit(directory,catalogue):
    directory,catalogue=Path(directory),Path(catalogue);here=Path(__file__).parent;checks=0;solves=0
    def check(ok,message):
        nonlocal checks
        checks+=1
        if not ok:raise AssertionError(message)
    def near(a,b):check(bool(np.allclose(a,b,atol=1e-9,rtol=1e-8)),'Numerical disagreement')
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=read(directory/'manifest.json')
    for n,h in manifest['code_sha256'].items():check(sha(here/n)==h,'Frozen source')
    check(sha(here.parent/'tests/test_configuration_search.py')==manifest['test_sha256'],'Frozen tests')
    receipt=read(here/'evidence/common-configuration-064.json')
    check(sha(here/'evidence/common-configuration-064.json')==manifest['catalogue_receipt_sha256'],'Frozen catalogue receipt')
    check(sha(catalogue)==receipt['files']['catalogue.json'],'Catalogue identity')
    inventory=read(catalogue);order={r['config_id']:i for i,r in enumerate(inventory)};starters=[r['config_id'] for r in inventory[:6]]
    fixture=read(directory/'synthetic-inputs.json');current=fixture['current'];history=fixture['history']
    def vector(r):
        onehot=[float(r['kind']==k) for k in ('seasonal','weekly_mean','ridge','forest')]
        return onehot+[(r.get('window',336)-336)/394,math.log(r.get('lags',24)/24)/math.log(7),(math.log10(r.get('alpha',.01))+2)/6,(r.get('depth',3)-3)/9,math.log(r.get('season',24)/24)/math.log(7)]
    def toy_label(config,index):return .1+.03*sum((x-y)**2 for x,y in zip(vector(config),[0.,0.,1.,0.,.5,.5,.5,0.,0.],strict=True))+.001*index
    for i,episode in enumerate(history):
        check(episode['synthetic_fixture'] and episode['evidence_kind']=='executed_backtests' and episode['revision']==manifest['revision'],'Declared synthetic prior evidence')
        check(episode['origin']==episode['source_available_at']==episode['recorded_at']<current['origin'],'Prior clocks precede query')
        for row in episode['backtests']:near(row['cv_rmsle'],toy_label(row['config'],i))
    for row in fixture['backtests']:near(row['cv_rmsle'],toy_label(row['config'],20))
    for arm in ('control','ledger'):
        report=read(directory/(arm+'.json'));tested={r['config_id']:r for r in fixture['backtests']};used=24
        check(report['synthetic_only'] and report['actual_forecast_computations']==0,'Simulated, not real fitting')
        check(len(report['steps'])==11,'All search steps')
        for step in report['steps']:
            proposal=step['proposal'];check(step['simulated_attempts_before']==used and used+4<=60,'Batch and final reservation')
            refs=[];c=[];x=[];y=[];mass=[]
            past=[];scale=[1.]*12
            if arm=='ledger':
                scale=np.maximum(.1,np.std(np.array([r['features'] for r in history]),axis=0)).tolist()
                distances=[math.fsum(((a-b)/s)**2 for a,b,s in zip(r['features'],current['features'],scale,strict=True)) for r in history]
                indices=sorted(range(16),key=lambda i:(distances[i],history[i]['origin'],history[i]['series_id']))
                past=[history[i] for i in indices]
                check([(r['series_id'],r['origin']) for r in proposal['retrieval']['selected']]==[(r['series_id'],r['origin']) for r in past],'Prior neighbors')
                for observed,i in zip(proposal['retrieval']['selected'],indices,strict=True):near(observed['distance'],distances[i])
            else:check(not proposal['retrieval']['selected'] and not proposal['retrieval']['ready'],'No-memory path')
            near(proposal['retrieval']['scale'],scale)
            def add(episode,records,total,kind):
                base=np.log1p([records[k]['cv_rmsle'] for k in starters]);center=float(base.mean());spread=max(.01,float(base.std()))
                for key,r in sorted(records.items()):
                    c.append(vector(r['config']));x.append(episode['features']);y.append((math.log1p(r['cv_rmsle'])-center)/spread);mass.append(total/len(records))
                    refs.append({'series_id':episode['series_id'],'origin':episode['origin'],'arm':episode['arm'],'config_id':key,'evidence_kind':kind})
            add({**current,'arm':arm},tested,.5 if past else 1.,'current_backtest')
            for episode in past:add(episode,{r['config_id']:r for r in episode['backtests']},.5/len(past),'prior_executed_backtest')
            training={'config_features':c,'context_features':x,'targets':y,'masses':mass,'refs':refs,'context_scale':scale}
            check(hashlib.sha256(json.dumps(training,separators=(',',':')).encode()).hexdigest()==proposal['training_sha256'],'Independent training reconstruction hash')
            check(proposal['training_records']==len(c) and proposal['current_backtests']==len(tested) and proposal['prior_episodes']==len(past),'Training counts')
            near(sum(mass[:len(tested)]),.5 if past else 1.);near(sum(mass),1.)
            available=[r for r in inventory if r['config_id'] not in tested]
            def k(a,b,ca,cb):return math.exp(-2*math.fsum((v-w)**2 for v,w in zip(a,b,strict=True))-.5*math.fsum(((v-w)/s)**2 for v,w,s in zip(ca,cb,scale,strict=True))/12)
            gram=np.array([[k(a,b,ca,cb) for b,cb in zip(c,x,strict=True)] for a,ca in zip(c,x,strict=True)])+np.diag([.05/w for w in mass])
            cross=np.array([[k(a,vector(r['config']),ca,current['features']) for r in available] for a,ca in zip(c,x,strict=True)])
            factor=cho_factor(gram,lower=True);solved=cho_solve(factor,np.column_stack((y,cross)));solves+=1
            expected_mean=cross.T@solved[:,0];expected_spread=np.sqrt(np.maximum(0,1-np.sum(cross*solved[:,1:],axis=0)))
            expected={r['config_id']:(float(m),float(s),float(m-.5*s)) for r,m,s in zip(available,expected_mean,expected_spread,strict=True)}
            check(len(proposal['ranking'])==len(available) and {r['config_id'] for r in proposal['ranking']}==set(expected),'Every untested candidate, no repeats')
            for r in proposal['ranking']:
                near([r['predicted_standardized_cv'],r['kernel_spread'],r['acquisition']],expected[r['config_id']])
                check(r['config']==inventory[order[r['config_id']]]['config'],'Ranked configuration identity')
            expected_order=sorted(expected,key=lambda cid:(expected[cid][2],order[cid]))
            check([r['config_id'] for r in proposal['ranking']]==expected_order,'Acquisition ordering')
            check(proposal['next_config_id']==expected_order[0],'Next proposal is acquisition minimum')
            near(step['synthetic_cv_rmsle'],toy_label(proposal['next_config'],20))
            tested[proposal['next_config_id']]={'config':proposal['next_config'],'config_id':proposal['next_config_id'],'cv_rmsle':step['synthetic_cv_rmsle']};used+=3
        check(used==57 and used+4>60 and report['simulated_numerical_attempts']==used+1==58,'No extra batch; final admitted')
        check({r['config_id']:r for r in report['backtests']}==tested and len(tested)==17,'All executed synthetic labels retained')
        selected=min(tested.values(),key=lambda r:(r['cv_rmsle'],order[r['config_id']]))
        check(report['selected']==selected,'Final selection uses observed current CV')
    report=read(directory/'report.json');logged=read(directory/'solves.json')
    check(report['tests']==8 and report['test_failures']==0,'Tests passed')
    check(report['surrogate_solves']==report['completed_solves']==len(logged)==30 and all(r['completed'] for r in logged),'Actual surrogate solve costs')
    check(sum(r['stage'].startswith('synthetic_search') for r in logged)==22,'Two eleven-step searches')
    check(report['provider_calls']==report['api_calls']==0 and report['status']=='search_prepared_not_source_evaluated','No numerical or agent accuracy claim')
    result={'checks':checks,'failures':0,'independent_cholesky_solves':solves,'report_sha256':sha(directory/'report.json'),'verifier_sha256':sha(Path(__file__)),
        'scope':'Independent synthetic labels, neighbor/scaling/mass reconstruction, training hashes, scalar kernels plus Cholesky posteriors, full acquisitions, final CV choices and simulated budgets. No source-data accuracy result.'}
    target=directory/'verification.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('catalogue');print(json.dumps(audit(**vars(p.parse_args())),indent=2))
