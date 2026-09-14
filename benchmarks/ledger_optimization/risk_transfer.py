"""Descriptive transfer audit of frozen081 risks; never a selection policy."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def mean(values):
    return math.fsum(values) / len(values)


def quadratic(matrix, weights):
    return math.fsum(weights[i] * matrix[i][j] * weights[j]
                     for i in range(6) for j in range(6))


def loss(point, actual):
    return mean([(math.log1p(p)-math.log1p(y))**2
                 for p,y in zip(point, actual, strict=True)])


def comparison(predicted, realised):
    if len(predicted) != len(realised) or not predicted:
        raise ValueError('Nonempty matched pairs required')
    mp,mr=mean(predicted),mean(realised)
    xp=[v-mp for v in predicted];xr=[v-mr for v in realised]
    den=math.sqrt(math.fsum(v*v for v in xp)*math.fsum(v*v for v in xr))
    positive=[i for i,v in enumerate(predicted) if v>0]
    sign=lambda v: (v>0)-(v<0)
    return {'cases':len(predicted),'mean_predicted_gain':mp,'mean_realised_gain':mr,
            'realised_positive':sum(v>0 for v in realised),
            'realised_negative':sum(v<0 for v in realised),
            'realised_ties':sum(v==0 for v in realised),
            'sign_agreement':sum(sign(p)==sign(r) for p,r in zip(predicted,realised)),
            'pearson':math.fsum(p*r for p,r in zip(xp,xr))/den if den else None,
            'predicted_positive':len(positive),
            'realised_positive_when_predicted_positive':sum(realised[i]>0 for i in positive)}


def summarize(rows):
    result={'cases':len(rows),'arms':{}}
    for arm in ('control','ledger'):
        result['arms'][arm]={
            'anchor_gain':comparison([r[arm]['predicted_anchor_gain'] for r in rows],
                                     [r[arm]['realised_anchor_gain'] for r in rows]),
            'ledger_vs_control':comparison([r[arm]['predicted_ledger_vs_control'] for r in rows],
                                          [r['realised_ledger_vs_control'] for r in rows]),
            'mean_predicted_loss':mean([r[arm]['predicted_loss'] for r in rows]),
            'mean_realised_loss':mean([r[arm]['realised_loss'] for r in rows])}
    result['mean_realised_ledger_vs_incumbent']=mean([r['realised_ledger_vs_incumbent'] for r in rows])
    return result


def run(source, output):
    start=time.monotonic();source,output=Path(source),Path(output)
    output.mkdir(exist_ok=False)
    receipt_path=Path(__file__).parent/'evidence/local-risk-081.json'
    receipt=json.loads(receipt_path.read_text());rows=[];access={}
    for name,h in sorted(receipt['files'].items()):
        if len(Path(name).stem)!=64 or '/' in name:continue
        raw=(source/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Source hash mismatch')
        r=json.loads(raw);access[name]=h
        weights={a:[f['weights'] for f in r['records'][a]['weight_fits']] for a in ('control','ledger')}
        losses={a:loss(r['point'][a],r['actual']) for a in ('control','ledger','uncorrected045','incumbent068')}
        row={k:r[k] for k in ('task_id','series_id','origin','domain','round')}
        for arm in ('control','ledger'):
            gs=r['records'][arm]['matrices']
            expected={a:mean([quadratic(g,w) for g,w in zip(gs,weights[a],strict=True)]) for a in weights}
            anchor=mean([quadratic(g,r['anchor']['weights']) for g in gs])
            row[arm]={'predicted_loss':expected[arm],'predicted_anchor_loss':anchor,
                      'realised_loss':losses[arm],'predicted_anchor_gain':anchor-expected[arm],
                      'realised_anchor_gain':losses['uncorrected045']-losses[arm],
                      'predicted_ledger_vs_control':expected['control']-expected['ledger']}
        row['realised_ledger_vs_control']=losses['control']-losses['ledger']
        row['realised_ledger_vs_incumbent']=losses['incumbent068']-losses['ledger']
        rows.append(row)
    if len(rows)!=416:raise ValueError('Incomplete source cohort')
    report={'overall':summarize(rows),
            'domains':{d:summarize([r for r in rows if r['domain']==d]) for d in ('electricity','pedestrian')},
            'phases':{'early_0_7':summarize([r for r in rows if r['round']<8]),
                      'later_8_25':summarize([r for r in rows if r['round']>=8])},
            'seconds':time.monotonic()-start,'new_fits':0,'api_calls':0,
            'scope':'Descriptive squared-log risk transfer on reused development cases; no new selection or held-out claim.'}
    manifest={'source_receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_access':access,'protected_access':False}
    for name,data in (('rows',rows),('report',report),('manifest',manifest)):
        (output/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    return report


if __name__=='__main__':print(json.dumps(run(*sys.argv[1:]),indent=2))
