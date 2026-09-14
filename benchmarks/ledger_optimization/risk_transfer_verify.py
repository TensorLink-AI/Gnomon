"""Independent array-algebra audit of diagnostic082."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def verify(root, source):
    start=time.monotonic();root,source=Path(root),Path(source);checks=0
    load=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    def check(ok,message):
        nonlocal checks
        checks+=1
        if not ok:raise AssertionError(message)
    def near(a,b):
        check(np.allclose(a,b,rtol=1e-9,atol=1e-12),'Numerical mismatch')
    manifest=load(root/'manifest.json');receipt_path=Path(__file__).parent/'evidence/local-risk-081.json'
    receipt=load(receipt_path);check(sha(receipt_path)==manifest['source_receipt_sha256'],'Receipt identity')
    check(sha(Path(__file__).with_name('risk_transfer.py'))==manifest['code_sha256'],'Frozen diagnostic code')
    rows=load(root/'rows.json');check(len(rows)==len({r['task_id'] for r in rows})==416,'Full unique cases')
    for r in rows:
        name=r['task_id']+'.json';check(sha(source/name)==manifest['source_access'][name]==receipt['files'][name],'Source identity');s=load(source/name)
        for k in ('task_id','series_id','domain','round','origin'):check(r[k]==s[k],'Task metadata')
        target=np.log1p(s['actual']);losses={a:np.mean((np.log1p(s['point'][a])-target)**2) for a in ('control','ledger','uncorrected045','incumbent068')}
        weights={a:np.array([f['weights'] for f in s['records'][a]['weight_fits']]) for a in ('control','ledger')}
        anchor=np.array(s['anchor']['weights'])
        for arm in weights:
            g=np.array(s['records'][arm]['matrices']);expected={a:np.einsum('hi,hij,hj->h',w,g,w).mean() for a,w in weights.items()};base=np.einsum('i,hij,j->h',anchor,g,anchor).mean()
            for k,v in {'predicted_loss':expected[arm],'predicted_anchor_loss':base,'realised_loss':losses[arm],
                        'predicted_anchor_gain':base-expected[arm],'realised_anchor_gain':losses['uncorrected045']-losses[arm],
                        'predicted_ledger_vs_control':expected['control']-expected['ledger']}.items():near(r[arm][k],v)
        near(r['realised_ledger_vs_control'],losses['control']-losses['ledger']);near(r['realised_ledger_vs_incumbent'],losses['incumbent068']-losses['ledger'])
    def paired(pred,real,saved):
        p=np.array(pred);y=np.array(real);check(saved['cases']==len(p),'Pair denominator')
        near(saved['mean_predicted_gain'],p.mean());near(saved['mean_realised_gain'],y.mean())
        for k,v in {'realised_positive':(y>0).sum(),'realised_negative':(y<0).sum(),'realised_ties':(y==0).sum(),
                    'sign_agreement':(np.sign(p)==np.sign(y)).sum(),'predicted_positive':(p>0).sum(),
                    'realised_positive_when_predicted_positive':((p>0)&(y>0)).sum()}.items():check(saved[k]==v,'Sign count')
        if p.std() and y.std():near(saved['pearson'],np.corrcoef(p,y)[0,1])
        else:check(saved['pearson'] is None,'Constant correlation')
    def group(rs,s):
        check(s['cases']==len(rs),'Group denominator')
        for arm in ('control','ledger'):
            a=s['arms'][arm]
            paired([r[arm]['predicted_anchor_gain'] for r in rs],[r[arm]['realised_anchor_gain'] for r in rs],a['anchor_gain'])
            paired([r[arm]['predicted_ledger_vs_control'] for r in rs],[r['realised_ledger_vs_control'] for r in rs],a['ledger_vs_control'])
            near(a['mean_predicted_loss'],np.mean([r[arm]['predicted_loss'] for r in rs]));near(a['mean_realised_loss'],np.mean([r[arm]['realised_loss'] for r in rs]))
        near(s['mean_realised_ledger_vs_incumbent'],np.mean([r['realised_ledger_vs_incumbent'] for r in rs]))
    report=load(root/'report.json');group(rows,report['overall'])
    for d in ('electricity','pedestrian'):group([r for r in rows if r['domain']==d],report['domains'][d])
    group([r for r in rows if r['round']<8],report['phases']['early_0_7']);group([r for r in rows if r['round']>=8],report['phases']['later_8_25'])
    check(report['new_fits']==report['api_calls']==0 and manifest['protected_access'] is False,'Read-only scope')
    result={'checks':checks,'failures':0,'cases':416,'seconds':time.monotonic()-start,'new_fits':0,'api_calls':0,'verifier_sha256':sha(Path(__file__)),'scope':'All416source hashes, risk quadratic forms, realised losses, comparison counts and correlations; descriptive only.'}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(verify(*sys.argv[1:]),indent=2))
