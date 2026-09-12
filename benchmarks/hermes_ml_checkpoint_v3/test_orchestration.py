"""Offline fault injection plus real pinned Hermes/native-tool integration."""
from pathlib import Path
import json
import subprocess
import sys
import time
from unittest.mock import patch

from .policy import phase, correction_reason
from .worker import drive, bounded_agent_class
from .transport import dump, proxy
from .run import OTHER, HERE, prepare, environment, assess


def run_checks(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    assert phase(12,400)=='exploration' and phase(13,400)=='selection'
    assert phase(1,90)=='selection' and phase(1,91)=='exploration'
    assert correction_reason({'error':'HTTP 502'},False) is None
    assert correction_reason({'turn_exit_reason':'text_response(finish_reason=stop)'},True) is None

    def scenario(name, responses, *, initial=16, deadline=1000, advance=0, full=False):
        work=root/name;work.mkdir();(work/'TASK.md').write_text('Synthetic task')
        dump(work/'agent-budget.json',{'remaining_requests':initial,'deadline_epoch':deadline})
        calls=[];clock=[0]
        def factory(remaining,seconds):
            calls.append((remaining,seconds))
            class Fake:
                def run_conversation(self,**kwargs):
                    i=len(calls)-1
                    value=responses[min(i,len(responses)-1)]
                    dump(work/'agent-budget.json',{'remaining_requests':initial-len(calls),'deadline_epoch':deadline})
                    clock[0]+=advance
                    return value
            return Fake()
        result=drive(factory,work,work,now=lambda:clock[0],is_complete=lambda _:full)
        return result,calls

    prose={'turn_exit_reason':'text_response(finish_reason=stop)','messages':[]}
    repetition={'error':'Model output entered a repetition loop','messages':[]}
    for name,response in [('prose',prose),('repetition',repetition)]:
        result,calls=scenario(name,[response])
        assert result['corrections']==2 and len(calls)==3
        assert [c[0] for c in calls]==[16,15,14]
        assert result['stop_reason']=='correction_limit_reached'
    result,calls=scenario('exhausted',[prose],initial=1)
    assert result['corrections']==0 and len(calls)==1
    result,calls=scenario('deadline',[prose],deadline=1,advance=2)
    assert result['stop_reason']=='deadline_reached' and len(calls)==1
    result,calls=scenario('service-error',[{'error':'HTTP 502','failed':True}])
    assert result['corrections']==0 and len(calls)==1
    result,calls=scenario('already-complete',[prose],full=True)
    assert result['stop_reason']=='checkpoint_workflow_complete' and len(calls)==1
    class SummaryWouldCallNetwork:
        def _handle_max_iterations(self,*args):
            raise AssertionError('Unexpected network summary')
    assert 'checkpoint' in bounded_agent_class(SummaryWouldCallNetwork)()._handle_max_iterations([],16)

    # Real pinned Hermes, native terminal, our proxy, and numerical lab. Synthetic
    # model responses replace upstream HTTP only, never the native tool execution.
    job=next(iter(json.loads((OTHER/'setup/recovered-task-source/host-jobs.json').read_text()).values()))[0]
    work=root/'live-work';home=root/'live-home';out=root/'live-output'
    for path in (work,home,out):path.mkdir()
    prepare(work,job,[],'plain')
    dump(home/'config.yaml',{'terminal':{'backend':'local','cwd':str(work)},
                            'compression':{'enabled':False},'mcp_servers':{}})
    python=OTHER/'plain-venv/bin/python';env=environment(home,work,python)
    config=json.dumps({'model':'ridge','window':180,'lags':14,'alpha':10})
    commands={2:'python lab.py start',3:"python lab.py backtest --config '"+config+"'",
              13:"python lab.py backtest --config '"+config.replace('10','20')+"'",
              14:'python lab.py commit --execution-id missing',
              16:"python lab.py commit --config '"+config+"'"}
    upstream=[]
    class Reply:
        status=200
        def __init__(self,value):self.value=value
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps(self.value).encode()
    def fake_upstream(request,timeout):
        assert request.full_url=='https://api.engy.ai/v1/chat/completions'
        payload=json.loads(request.data);upstream.append(payload);n=len(upstream)
        assert n<=16
        if n==1:
            message={'role':'assistant','content':'I will start.'};finish='stop'
        else:
            message={'role':'assistant','content':None,'tool_calls':[{
                'id':f'call_{n}','type':'function','function':{'name':'terminal',
                'arguments':json.dumps({'command':commands.get(n,'python lab.py status'),'timeout':60})}}]}
            finish='tool_calls'
        return Reply({'id':f'chatcmpl-{n}','object':'chat.completion','created':int(time.time()),
                      'model':'deepseek-v4-flash-0731','choices':[{'index':0,'message':message,'finish_reason':finish}],
                      'usage':{'prompt_tokens':10,'completion_tokens':10,'total_tokens':20}})
    with patch('urllib.request.urlopen',side_effect=fake_upstream):
        with proxy(out,'synthetic-test-key',work=work,deadline=time.time()+480) as url:
            completed=subprocess.run([str(python),str(HERE/'worker.py'),str(work),str(out),url,'alone'],
                                     cwd=work,env=env,text=True,capture_output=True,timeout=180)
    (out/'stdout.txt').write_text(completed.stdout);(out/'stderr.txt').write_text(completed.stderr)
    assert completed.returncode==0,completed.stderr
    assert len(upstream)==16,len(upstream)
    assert not list(out.glob('blocked-request-*.json'))
    state=json.loads((out/'orchestration.json').read_text())
    assert state['corrections']==1 and state['attempts']==2 and state['stop_reason']=='checkpoint_workflow_complete',state
    assert len(list(out.glob('api-*-intervention.json')))==4
    grade=assess(work,job,{},b'');assert grade['valid'] and grade['workflow_complete'],grade
    assert grade['numerical_attempts']==8,grade
    # Check the actual native terminal response to the phase-blocked exploration.
    messages=json.loads((out/'hermes-result.json').read_text())['messages']
    assert any('SELECTION_PHASE_RESERVED' in str(m.get('content','')) for m in messages if m['role']=='tool')
    dump(root/'passed.json',{'passed':True,'offline_scenarios':9,'real_hermes_requests':16,
                            'corrections':1,'phase_interventions':4,'numerical_fits':8,
                            'upstream':'synthetic responses; no paid inference'})


if __name__=='__main__':run_checks(sys.argv[1])
