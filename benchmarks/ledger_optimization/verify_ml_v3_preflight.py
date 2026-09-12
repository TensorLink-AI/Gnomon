"""Independently verify saved maturation values from the v3 public-API preflight."""
import argparse
import json
import math
from pathlib import Path


def verify(root):
    checked=[]
    for arm in ('plain','gnomon','ledger'):
        work=root/arm
        events=[json.loads(line) for line in (work/'experiments.jsonl').read_text().splitlines()]
        results={e['execution']['execution_id']:e for e in events if e['event']=='result'}
        matured=[e for e in events if e['event']=='matured']
        assert len(results)==8 and sum(e['event']=='attempt' for e in events)==8
        assert len(matured)==2 and len({e['execution_id'] for e in matured})==2
        assert sum(e['selected_for_submission'] for e in matured)==1
        prior=json.loads((work/'previous_runs.json').read_text())[0]
        checkpoints=[json.loads(p.read_text()) for p in (work/'checkpoints').glob('*.json')]
        selected=[c for c in checkpoints if c['execution_id']==prior['execution_id']]
        assert len(selected)==1
        checkpoint=selected[0]
        for value in matured:
            original=results[value['execution_id']]
            assert original['kind']=='forecast' and original['actual'] is None
            assert value['point']==original['point'] and value['actual']==prior['actual']
            assert value['selected_for_submission']==(value['execution_id']==checkpoint['execution_id'])
            errors=[p-a for p,a in zip(value['point'],value['actual'],strict=True)]
            n=len(errors)
            expected={'mae':sum(abs(e) for e in errors)/n,
                      'rmse':math.sqrt(sum(e*e for e in errors)/n),
                      'bias':sum(errors)/n,
                      'rmsle':math.sqrt(sum((math.log1p(p)-math.log1p(a))**2
                                           for p,a in zip(value['point'],value['actual'],strict=True))/n)}
            for key,number in expected.items():
                assert abs(value['metrics'][key]-number)<1e-10
                if arm=='ledger' and key!='rmsle':
                    assert abs(value['ledger_score'][key]-number)<1e-10
            if arm=='ledger':assert value['ledger_score']['n']==n
            checked.append({'arm':arm,'execution_id':value['execution_id'],
                            'selected':value['selected_for_submission'],'n':n,'metrics':expected})
    return {'passed':True,'executions_checked':checked,'new_fits_during_maturation':0,
            'scope':'Synthetic preflight; six saved production executions across three arms; no paid agent claims.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=verify(args.root)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
