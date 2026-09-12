"""Separate service failures from agent completion on a finished ML pilot.

Read-only diagnosis: never changes grades, retries sessions or promotes a trial.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path


def classify(grade, outcomes):
    before_execution=not grade['workflow_complete'] and grade['numerical_attempts']==0
    return {
        'no_execution_service_failure':before_execution and bool(outcomes) and all(x=='service_error' for x in outcomes),
        'service_interrupted_before_execution':before_execution and len(outcomes)>=3 and outcomes[-3:]==['service_error']*3,
        'successful_api_responses':outcomes.count('success'),
    }


def audit(root):
    report=json.loads((root/'pilot/report.json').read_text())
    assert report['complete'] and not report['audit_failures']
    files=[root/'pilot/report.json',root/'GATE.json',root/'FINISHED.json']
    rows=[];http=Counter();totals=Counter();usage_responses=0
    for path in sorted((root/'pilot').glob('*/*/round-*/grade.json')):
        grade=json.loads(path.read_text());files.append(path);errors=[];tokens=0;outcomes=[]
        responses=sorted(path.parent.glob('api-*-response.json'))
        forwarded=list(path.parent.glob('api-*-forwarded.json'))
        assert len(responses)==grade['responses'] and len(forwarded)==grade['api_calls']
        for response in responses:
            try:r=json.loads(response.read_text())
            except ValueError:r={'error':{'type':'invalid_response'}}
            files.append(response)
            receipt=response.with_name(response.name.replace('-response','-receipt'))
            status=json.loads(receipt.read_text())['status'];files.append(receipt)
            if r.get('error'):
                e=r['error'];category=e.get('type') if isinstance(e,dict) else 'unstructured'
                errors.append({'http_status':status,'category':category});http[str(status)]+=1
                service=status in (408,429,500,502,503,504) or category in ('rate_limit_error','upstream_error')
                outcomes.append('service_error' if service else 'other_error')
            else:outcomes.append('success')
            if r.get('usage'):
                tokens+=r['usage'].get('total_tokens',0);usage_responses+=1
        assert len(errors)==grade['api_errors'] and tokens==grade['tokens']
        all_failed=bool(responses) and len(errors)==len(responses)
        row={k:grade[k] for k in ('arm','series_id','round','workflow_complete','valid',
                                'api_calls','api_errors','numerical_attempts','tokens','fallback_used')}
        row.update(all_responses_errors=all_failed,errors=errors,**classify(grade,outcomes))
        rows.append(row)
        for k in ('api_calls','api_errors','numerical_attempts','tokens'):totals[k]+=grade[k]
    assert len(rows)==36
    totals['responses']=sum(len(list(p.parent.glob('api-*-response.json'))) for p in (root/'pilot').glob('*/*/round-*/grade.json'))
    incomplete=[r for r in rows if not r['workflow_complete']]
    return {'source':str(root),'sessions':len(rows),'incomplete_sessions':len(incomplete),
            'no_execution_service_failures':sum(r['no_execution_service_failure'] for r in incomplete),
            'service_interrupted_before_execution':sum(r['service_interrupted_before_execution'] for r in incomplete),
            'incomplete_with_some_successful_responses':sum(r['successful_api_responses']>0 for r in incomplete),
            'incomplete_not_explained_by_pre_execution_service_stop':sum(not r['service_interrupted_before_execution'] for r in incomplete),
            'http_error_status_counts':dict(http),'totals':dict(totals),
            'responses_with_usage':usage_responses,'api_dollar_cost':None,
            'cost_note':'Preserved reported tokens; failed responses without usage are not evidence of zero billing.',
            'per_arm':{a:{'full':sum(r['workflow_complete'] for r in rows if r['arm']==a),
                          'tasks':sum(r['arm']==a for r in rows)} for a in ('plain','gnomon','ledger')},
            'failures':incomplete,'integrity_checks':report['audit_checks'],
            'gate':json.loads((root/'GATE.json').read_text()),
            'source_hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'grades_changed':False,'sessions_retried':0,'new_api_calls':0,
            'conclusion_scope':'Completion attribution only; original all-case scores and failed promotion are unchanged.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.root)
    with args.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','failures')},indent=2))
