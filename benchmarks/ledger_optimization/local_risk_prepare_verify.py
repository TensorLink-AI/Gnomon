"""Independent080weighted PSD evidence and per-step optimization audit."""
import hashlib,json,math
from pathlib import Path
import sys
import numpy as np

NAMES=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')


def verify(root):
    root=Path(root);here=Path(__file__).parent;checks=0
    def load(n):return json.loads((root/n).read_text())
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def check(v,m):
        nonlocal checks
        checks+=1
        if not v:raise AssertionError(m)
    def near(a,b):check(np.allclose(a,b,atol=1e-9,rtol=1e-8),'Independent numerical disagreement')
    manifest=load('manifest.json');training=load('training.json');model=load('model.json');query=load('query.json');anchor=np.array(training['anchor'])
    for n,h in manifest['code_sha256'].items():check(sha(here/n)==h,'Frozen source hash')
    check(manifest['synthetic_only'] and manifest['source_data_access'] is False,'Synthetic-only scope');check(model['input_sha256']==hashlib.sha256(json.dumps(training,separators=(',',':')).encode()).hexdigest(),'Training identity')
    def features(point):
        logs=np.array([[math.log1p(point[n][h]) for n in NAMES] for h in range(24)]);base=np.array([math.fsum(anchor[j]*logs[h,j] for j in range(6)) for h in range(24)]);location=math.fsum(base)/24;scale=max(math.sqrt(math.fsum((b-location)**2 for b in base)/24),.1)
        return np.array([[(v-location)/scale for v in logs[h]]+[base[h],scale,math.sin(2*math.pi*h/24),math.cos(2*math.pi*h/24)] for h in range(24)],dtype=np.float32)
    x=[];y=[];weights=[]
    for pair,mass in zip(training['pairs'],training['masses'],strict=True):
        x.extend(features(pair['point']));weights.extend([mass/24]*24)
        for h in range(24):
            e=np.array([math.log1p(pair['point'][n][h])-math.log1p(pair['actual'][h]) for n in NAMES]);y.append(np.outer(e,e).ravel())
    x=np.array(x,dtype=np.float32);y=np.array(y);weights=np.array(weights);qx=features(query['point']);predicted=[];check(x.shape==(144,10) and y.shape==(144,36),'Complete synthetic arrays');check(model['tree_count']==len(model['trees'])==32,'All trees recorded')
    for t in model['trees']:
        left=t['children_left'];right=t['children_right'];feature=t['feature'];threshold=t['threshold'];values=np.array(t['value'])[:,:,0];stack=[(0,np.arange(len(x)),0)];seen=set()
        while stack:
            node,indices,depth=stack.pop();seen.add(node);check(depth<=4 and len(indices)==t['n_node_samples'][node],'Tree depth/sample support');near(weights[indices].sum(),t['weighted_n_node_samples'][node]);near(np.sum(weights[indices,None]*y[indices],axis=0)/weights[indices].sum(),values[node])
            if left[node]==-1:check(right[node]==-1 and len(indices)>=24,'Minimum leaf support')
            else:
                mask=x[indices,feature[node]].astype(np.float64)<=threshold[node];stack.extend(((left[node],indices[mask],depth+1),(right[node],indices[~mask],depth+1)))
        check(seen==set(range(len(left))),'All nodes have support');p=[]
        for row in qx:
            node=0
            while left[node]!=-1:node=left[node] if float(row[feature[node]])<=threshold[node] else right[node]
            p.append(values[node])
        predicted.append(p)
    matrices=np.mean(predicted,axis=0).reshape(24,6,6);near(query['matrices'],matrices);check(np.linalg.eigvalsh(matrices).min()>=-1e-9,'Mixture of empirical PSD matrices');result=query['prediction'];iterations=0;forecast=[]
    for h,(g,f) in enumerate(zip(matrices,result['weight_fits'],strict=True)):
        w=np.array(f['weights']);check(w.shape==(6,) and w.min()>=0 and np.isfinite(w).all(),'Simplex weights');near(w.sum(),1.);near(f['matrix'],g);near(f['anchor'],anchor);value=float(w@g@w+.001*np.square(w-anchor).sum());initial=float(anchor@g@anchor);gradient=2*g@w+.002*(w-anchor);gap=float(w@gradient-gradient.min());near(f['objective'],value);near(f['initial_objective'],initial);near(f['convex_gap_bound'],gap);near(f['minimum_eigenvalue'],np.linalg.eigvalsh(g).min());check(f['success'] and gap<=1e-5+1e-9 and value<=initial+1e-8,'Independent convex certificate');iterations+=f['iterations']
        point=math.expm1(math.fsum(w[j]*math.log1p(query['point'][n][h]) for j,n in enumerate(NAMES)));forecast.append(point);check(min(query['point'][n][h] for n in NAMES)-1e-9<=point<=max(query['point'][n][h] for n in NAMES)+1e-9,'Forecast remains in model range')
    near(result['point'],forecast);check(result['weight_fit_count']==24 and result['weight_iterations']==iterations,'Actual solve counts');report=load('report.json');check(report['training_cases']==6 and report['training_rows']==144 and report['forest_fits']==1 and report['trees']==32 and report['weight_fits']==24 and report['weight_iterations']==iterations,'Complete numerical costs');check(report['synthetic_only'] and report['api_calls']==report['base_forecasts']==0 and report['source_evaluation_status']=='not_started','No source performance claim')
    audit={'checks':checks,'failures':0,'trees':32,'training_rows':144,'query_steps':24,'certified_weight_fits':24,'new_forests_or_weight_fits':0,'api_calls':0,'synthetic_only':True,'verifier_sha256':sha(__file__)};(root/'verification.json').write_text(json.dumps(audit,indent=2)+'\n');return audit


if __name__=='__main__':print(json.dumps(verify(sys.argv[1]),indent=2))
