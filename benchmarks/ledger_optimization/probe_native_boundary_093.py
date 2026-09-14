"""Zero-API probe of the actual pinned Hermes dispatcher and native text memory.

Run in a fresh process with a disposable HERMES_HOME and the pinned Hermes on
PYTHONPATH. This tests real native tools and inherited dispatch, not conversation
or paid-provider behavior. All outbound socket connections are rejected.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.ledger_optimization.execution_boundary_093 import LabBoundary
from benchmarks.ledger_optimization.hermes_boundary_093 import (
    HermesBoundary, bounded_agent_class, native_callbacks,
)


def main():
    home = Path(os.environ['HERMES_HOME'])
    if home.exists():
        raise RuntimeError('Use a new nonexistent disposable HERMES_HOME.')
    home.mkdir(parents=True)
    # No external skills, autoloads, sync credentials, or plugins in this profile.
    (home/'config.yaml').write_text('memory:\n  memory_enabled: true\n  user_profile_enabled: true\nskills:\n  external_dirs: []\n')
    checks=[];network=[]
    def deny(*args, **kwargs):
        network.append('blocked_connection')
        raise RuntimeError('Network disabled for native boundary probe')
    def check(label, condition):
        checks.append({'assertion':label,'passed':bool(condition)})
        if not condition:raise AssertionError(label)
    with patch.object(socket.socket,'connect',deny), patch.object(socket,'create_connection',deny):
        import run_agent
        from tools.memory_tool import MemoryStore
        cls=bounded_agent_class(run_agent.AIAgent)
        agent=cls.__new__(cls)
        agent._flush_messages_to_session_db=None
        agent._memory_store=MemoryStore()
        agent._memory_store.load_from_disk()
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory);marker=work/'shell-executed'
            (work/'lab.py').write_text('import json;print(json.dumps({"operation":"status","provider_calls":0}))\n')
            lab=LabBoundary(work,sys.executable,{'lab.py':hashlib.sha256((work/'lab.py').read_bytes()).hexdigest()})
            events=[]
            agent.execution_boundary=HermesBoundary(lab,native=native_callbacks(agent),record=events.append)
            count=0
            def invoke(tool,args,method='_execute_tool_calls'):
                nonlocal count
                count+=1;ident=f'probe-{count}'
                call=SimpleNamespace(id=ident,function=SimpleNamespace(name=tool,arguments=json.dumps(args)))
                messages=[]
                getattr(agent,method)(SimpleNamespace(tool_calls=[call]),messages,'synthetic')
                check('tool ID preserved '+ident,messages[0]['tool_call_id']==ident)
                return json.loads(messages[0]['content'])
            def native(tool,args):
                r=invoke(tool,args);check('native wrapper '+tool,r['status']=='ok')
                return json.loads(r['result'])
            memory=native('memory',{'target':'memory','action':'add','content':'Synthetic observation: ridge was tested at origin one.'})
            check('native memory write succeeded',memory.get('success',False))
            fresh=MemoryStore();fresh.load_from_disk()
            check('memory survives fresh store','Synthetic observation' in fresh.format_for_system_prompt('memory'))
            lesson=f'---\nname: synthetic-lesson\ndescription: Synthetic dated modelling lesson\n---\n# Lesson\nKeep model comparisons dated.\n\n!`touch {marker}`\n'
            created=native('skill_manage',{'operations':[{'action':'create','name':'synthetic-lesson','content':lesson}]})
            check('native skill created',created.get('success',False))
            viewed=native('skill_view',{'name':'synthetic-lesson'})
            check('skill text remains uninterpolated',viewed.get('success',False) and f'!`touch {marker}`' in viewed.get('content',''))
            check('inline shell was not executed',not marker.exists())
            listed=native('skills_list',{})
            check('native skill discoverable','synthetic-lesson' in json.dumps(listed))
            for method in ('_execute_tool_calls','_execute_tool_calls_sequential','_execute_tool_calls_concurrent'):
                r=invoke('terminal',{'command':f'touch {marker}'},method)
                check('terminal blocked '+method,r['status']=='error' and not marker.exists())
                r=invoke('lab',{'operation':'status'},method)
                check('trusted lab reachable '+method,r['status']=='ok' and r['result']['exit_code']==0)
            check('only three lab subprocesses',sum(e.get('admitted',False) for e in lab.events)==3)
            check('no outbound connections attempted',not network)
            files=[Path(run_agent.__file__)]
            print(json.dumps({'checks':checks,'events':events,'network_attempts':network,
                              'api_calls':0,'model_fits':0,'scope':'actual pinned native tools and dispatch; no model conversation',
                              'hermes_sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))


if __name__=='__main__':main()
