"""Independent077 synthetic features, weighted tree leaves and gate audit."""
import hashlib,json,math
from pathlib import Path
import sys
import numpy as np

NAMES=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')


def verify(root):
    root=Path(root);here=Path(__file__).parent;checks=0
    def load(n):return json.loads((root/n).read_text())
    def check(v,m):
        nonlocal checks
        checks+=1
        if not v:raise AssertionError(m)
    def near(a,b):check(np.allclose(a,b,atol=1e-10,rtol=1e-9),'Numerical disagreement')
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=load('manifest.json');training=load('training.json');model=load('model.json');held=load('held_back.json');weights=training['weights']
    for n,h in manifest['code_sha256'].items():check(sha(here/n)==h,'Frozen source')
    check(manifest['synthetic_only'] and manifest['source_data_access'] is False and manifest['api_calls']==0,'Synthetic-only scope')
    check(model['training_sha256']==hashlib.sha256(json.dumps(training,separators=(',',':')).encode()).hexdigest(),'Training input hash')
    check(training['masses']==[.25,.25]+[.125]*4 and len(training['pairs'])==6,'Case mass allocation')
    def features(point):
        logs=[[math.log1p(point[n][h]) for n in NAMES] for h in range(24)];base=[math.fsum(weights[j]*logs[h][j] for j in range(6)) for h in range(24)];location=math.fsum(base)/24;scale=max(math.sqrt(math.fsum((b-location)**2 for b in base)/24),.1)
        x=[[(v-location)/scale for v in logs[h]]+[base[h],scale,math.sin(2*math.pi*h/24),math.cos(2*math.pi*h/24)] for h in range(24)]
        # sklearn's public tree prediction path converts input to float32.
        return np.asarray(x,dtype=np.float32),base
    xs=[];ys=[];ws=[]
    for pair,mass in zip(training['pairs'],training['masses'],strict=True):
        x,base=features(pair['point']);xs.extend(x);ys.extend(math.log1p(pair['actual'][h])-base[h] for h in range(24));ws.extend([mass/24]*24)
    x=np.asarray(xs,dtype=np.float32);y=np.asarray(ys);w=np.asarray(ws);hx,hbase=features(held['fixture']['point']);tree_predictions=[]
    check(x.shape==(144,10) and model['row_count']==144 and model['tree_count']==32,'Training dimensions')
    for tree in model['trees']:
        paths={};left=tree['children_left'];right=tree['children_right'];feature=tree['feature'];threshold=tree['threshold']
        def leaf(row):
            node=0;depth=0
            while left[node]!=-1:
                check(depth<4,'Bounded tree depth');node=left[node] if row[feature[node]]<=threshold[node] else right[node];depth+=1
            return node
        for i,row in enumerate(x):
            node=0
            while True:
                paths.setdefault(node,[]).append(i)
                if left[node]==-1:break
                node=left[node] if row[feature[node]]<=threshold[node] else right[node]
        check(len(paths)==len(left),'All saved nodes have support')
        for node,indices in paths.items():
            check(tree['n_node_samples'][node]==len(indices),'Exact node sample count');near(tree['weighted_n_node_samples'][node],w[indices].sum());near(tree['value'][node][0][0],np.dot(w[indices],y[indices])/w[indices].sum())
            if left[node]==-1:check(len(indices)>=24,'Minimum leaf observations')
        tree_predictions.append([tree['value'][leaf(row)][0][0] for row in hx])
    correction=np.mean(tree_predictions,axis=0);raw=np.asarray(hbase)+correction;point=np.expm1(np.maximum(raw,0.));application=held['application'];near(application['correction_log'],correction);near(application['raw_predicted_log'],raw);near(application['point'],point);near(application['base_point'],np.expm1(hbase));check(application['clipped_leads']==np.where(raw<0)[0].tolist(),'Clipping disclosed')
    def risk(point,actual):return float(np.linalg.norm(np.log1p(point)-np.log1p(actual))/math.sqrt(24))
    base=application['base_point']
    for name,pred,actual in (('favorable_decision',point,held['fixture']['actual']),('unfavorable_decision',point,base),('tie_decision',base,held['fixture']['actual'])):
        decision=held[name];before=risk(base,actual);after=risk(pred,actual);near(decision['baseline_rmsle'],before);near(decision['corrected_rmsle'],after);near(decision['improvement'],before-after);check(decision['correction_enabled']==(before-after>1e-12),'Held-back gate follows scores')
    report=load('report.json');check(report['forest_fits']==1 and report['trees']==32 and report['training_cases']==6 and report['training_rows']==144 and report['held_back_rows']==24,'Reported costs/dimensions');check(report['positive_correction_enabled'] and not report['negative_correction_enabled'] and not report['tie_correction_enabled'],'All synthetic decisions correct');check(report['api_calls']==report['source_forecasts']==0 and report['source_evaluation_status']=='not_started','No source evaluation claim')
    result={'checks':checks,'failures':0,'trees':32,'training_rows':144,'held_back_rows':24,'independent_leaf_predictions':768,'new_forest_fits':0,'api_calls':0,'synthetic_only':True,'verifier_sha256':sha(__file__)}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(verify(sys.argv[1]),indent=2))
