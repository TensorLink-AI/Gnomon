"""Independent configuration-search reconstruction with block Cholesky algebra."""
from datetime import datetime
import hashlib
import json
import math
import numpy as np
from scipy.linalg import cho_factor,cho_solve


def vector(r):
    return [float(r['kind']==k) for k in ('seasonal','weekly_mean','ridge','forest')]+[(r.get('window',336)-336)/394,
        math.log(r.get('lags',24)/24)/math.log(7),(math.log10(r.get('alpha',.01))+2)/6,(r.get('depth',3)-3)/9,math.log(r.get('season',24)/24)/math.log(7)]


class AcquisitionAudit:
    def __init__(self,current,history,inventory,check,near):
        self.current=current;self.inventory=inventory;self.check=check;self.near=near;now=datetime.fromisoformat(current['origin'])
        self.pool=sorted([r for r in history if r['domain']==current['domain'] and r['arm']==current['arm'] and datetime.fromisoformat(r['origin'])<now
            and datetime.fromisoformat(r['source_available_at'])<=now and datetime.fromisoformat(r['recorded_at'])<=now],key=lambda r:(r['origin'],r['series_id']))
        self.scale=np.maximum(.1,np.std(np.asarray([r['features'] for r in self.pool]),axis=0)).tolist() if self.pool else [1.]*12
        self.distances=[float(np.sum(((np.asarray(r['features'])-current['features'])/self.scale)**2)) for r in self.pool]
        self.ready=len(self.pool)>=16 and len({r['origin'] for r in self.pool})>=3
        order=sorted(range(len(self.pool)),key=lambda i:(self.distances[i],self.pool[i]['origin'],self.pool[i]['series_id']))[:16] if self.ready else []
        self.past=[self.pool[i] for i in order];self.selected=[{'series_id':self.pool[i]['series_id'],'origin':self.pool[i]['origin'],'arm':self.pool[i]['arm'],'distance':self.distances[i]} for i in order]
        self.historical=self.arrays([(r,r['backtests'],.5/len(self.past),'prior_executed_backtest') for r in self.past]) if self.past else ([],[],[],[],[])
        self.query=self.phi([vector(r['config']) for r in inventory],[current['features']]*len(inventory))
        self.hfactor=None;self.hsolved=None;self.hphi=None;self.hcross=None
        if self.past:
            c,x,y,mass,_=self.historical;self.hphi=self.phi(c,x)
            h=self.kernel(self.hphi,self.hphi)+np.diag([.05/w for w in mass]);self.hcross=self.kernel(self.hphi,self.query)
            self.hfactor=cho_factor(h,lower=True);self.hsolved=cho_solve(self.hfactor,np.column_stack((y,self.hcross)))
    def arrays(self,groups):
        c=[];x=[];y=[];mass=[];refs=[];starters=[r['config_id'] for r in self.inventory[:6]]
        for episode,backtests,total,kind in groups:
            rows={r['config_id']:r for r in backtests};self.check(len(rows)==len(backtests),'Unique training configurations')
            baseline=np.log1p([rows[k]['cv_rmsle'] for k in starters]);center=float(baseline.mean());spread=max(.01,float(baseline.std()))
            for key,r in sorted(rows.items()):
                c.append(vector(r['config']));x.append(list(episode['features']));y.append((math.log1p(r['cv_rmsle'])-center)/spread);mass.append(total/len(rows))
                refs.append({'series_id':episode['series_id'],'origin':episode['origin'],'arm':episode['arm'],'config_id':key,'evidence_kind':kind})
        return c,x,y,mass,refs
    def phi(self,c,x):
        # One transformed feature space gives the product kernel independently.
        return np.column_stack((math.sqrt(2)*np.asarray(c),(np.asarray(x)-self.current['features'])/np.asarray(self.scale)/math.sqrt(24)))
    @staticmethod
    def kernel(a,b):
        return np.exp(-np.sum((a[:,None,:]-b[None,:,:])**2,axis=2))
    def verify(self,proposal,backtests):
        check,near=self.check,self.near;ret=proposal['retrieval']
        check(ret['ready']==self.ready,'Retrieval readiness')
        expected=[{'series_id':r['series_id'],'origin':r['origin'],'arm':r['arm'],'distance':d} for r,d in zip(self.pool,self.distances,strict=True)]
        check(ret['eligible']==expected and ret['selected']==self.selected,'All and only causal eligible/selected studies');near(ret['scale'],self.scale)
        current=self.arrays([(self.current,backtests,.5 if self.past else 1.,'current_backtest')]);combined=[a+b for a,b in zip(current,self.historical,strict=True)]
        c,x,y,mass,refs=combined
        training={'config_features':c,'context_features':x,'targets':y,'masses':mass,'refs':refs,'context_scale':self.scale}
        check(hashlib.sha256(json.dumps(training,separators=(',',':')).encode()).hexdigest()==proposal['training_sha256'],'Full causal training hash')
        check(proposal['training_records']==len(y) and proposal['current_backtests']==len(backtests) and proposal['prior_episodes']==len(self.past),'Training counts')
        near(sum(mass),1.);near(sum(current[3]),.5 if self.past else 1.)
        cp=self.phi(current[0],current[1]);cgram=self.kernel(cp,cp)+np.diag([.05/w for w in current[3]]);cross=self.kernel(cp,self.query);rhs=np.column_stack((current[2],cross))
        if self.past:
            bridge=self.kernel(cp,self.hphi);hinv_bridge=cho_solve(self.hfactor,bridge.T)
            schur=cgram-bridge@hinv_bridge;rhs-=bridge@self.hsolved
            csolved=cho_solve(cho_factor(schur,lower=True),rhs);hsolved=self.hsolved-hinv_bridge@csolved
            predicted=cross.T@csolved[:,0]+self.hcross.T@hsolved[:,0]
            variance=1-np.sum(cross*csolved[:,1:],axis=0)-np.sum(self.hcross*hsolved[:,1:],axis=0)
        else:
            csolved=cho_solve(cho_factor(cgram,lower=True),rhs);predicted=cross.T@csolved[:,0];variance=1-np.sum(cross*csolved[:,1:],axis=0)
        check(variance.min()>=-1e-8 and variance.max()<=1+1e-8,'Independent predictive variance');spread=np.sqrt(np.clip(variance,0,1))
        tested={r['config_id'] for r in backtests};order={r['config_id']:i for i,r in enumerate(self.inventory)};available=[r for r in self.inventory if r['config_id'] not in tested]
        check(len(proposal['ranking'])==len(available) and {r['config_id'] for r in proposal['ranking']}=={r['config_id'] for r in available},'All untested candidates ranked')
        for r in proposal['ranking']:
            i=order[r['config_id']];check(r['config']==self.inventory[i]['config'],'Candidate identity')
            near([r['predicted_standardized_cv'],r['kernel_spread'],r['acquisition']],[predicted[i],spread[i],predicted[i]-.5*spread[i]])
        ranked=sorted(proposal['ranking'],key=lambda r:(r['acquisition'],order[r['config_id']]))
        check(ranked==proposal['ranking'] and proposal['next_config_id']==ranked[0]['config_id'] and proposal['next_config']==ranked[0]['config'],'Observed acquisition order/choice')
        minimum=min(predicted[order[r['config_id']]]-.5*spread[order[r['config_id']]] for r in available)
        check(predicted[order[proposal['next_config_id']]]-.5*spread[order[proposal['next_config_id']]]<=minimum+1e-8,'Independent acquisition minimum within numerical tolerance')
        check(proposal['provider_calls']==0 and proposal['surrogate_solves']==1 and proposal['kernel_ridge']==.05 and proposal['exploration']==.5,'Frozen search settings/cost')
        return len(y)
