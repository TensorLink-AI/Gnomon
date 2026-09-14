"""Prospective guarded Hermes worker. Not authorized for paid dispatch until frozen.

Reuses the frozen v5 continuation policy and host API budget. The new tool surface
is common to all arms. Sources and preparation inputs come from a host manifest;
agent-written notes or memory never select a forecast on the agent's behalf.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from .policy import MAX_CORRECTIONS, correction_reason
from .boundary_schemas_093 import attach
from .execution_boundary_093 import LabBoundary
from .hermes_boundary_093 import HermesBoundary, bounded_agent_class, native_callbacks


def read_budget(work):return json.loads((work/'agent-budget.json').read_text())


def drive(factory,work,output,*,is_complete):
    """Same v5 limits/termination policy, with corrections using structured tools."""
    corrections=0;history=[];attempts=[];prompt=(work/'TASK.md').read_text()
    while True:
        budget=read_budget(work);remaining=budget['remaining_requests'];seconds=budget['deadline_epoch']-time.time()
        if remaining<=0 or seconds<=0:
            stop='request_budget_exhausted' if remaining<=0 else 'deadline_reached';break
        result=factory(remaining,seconds).run_conversation(user_message=prompt,conversation_history=history)
        attempts.append(result)
        (output/f'hermes-attempt-{len(attempts):02d}.json').write_text(json.dumps(result,default=str,indent=2))
        full=is_complete(work)
        if full:stop='checkpoint_workflow_complete';break
        reason=correction_reason(result,full);budget=read_budget(work)
        if budget['remaining_requests']<=0 or budget['deadline_epoch']<=time.time():
            stop='request_budget_exhausted' if budget['remaining_requests']<=0 else 'deadline_reached';break
        if not reason:stop='incomplete_without_recoverable_termination';break
        if corrections>=MAX_CORRECTIONS:stop='correction_limit_reached';break
        corrections+=1
        with (output/'corrections.jsonl').open('a') as f:f.write(json.dumps({'correction':corrections,'cause':reason,'budget':budget})+'\n')
        history=result.get('messages',history)
        prompt=(f'Workflow correction ({reason}): no complete selected checkpoint was established. '
                'Use the actual lab tool with {"operation":"status"}; if needed use start, backtest an ML '
                'configuration while exploration is open, and explicitly commit your tested choice. '
                'Preserve existing evidence. The SAME remaining requests and time apply. '
                'Do not print tool markup or infer selection from prose.')
    if attempts:(output/'hermes-result.json').write_text(json.dumps(attempts[-1],default=str,indent=2))
    summary={'protocol':'metered-093','corrections':corrections,'attempts':len(attempts),
             'stop_reason':stop,'final_budget':read_budget(work)}
    (output/'orchestration.json').write_text(json.dumps(summary,indent=2))
    return summary


def run(work, output, base_url, manifest):
    work=Path(work).resolve(strict=True);output=Path(output).resolve(strict=True)
    manifest_path=Path(manifest).resolve(strict=True)
    if manifest_path.is_relative_to(work) or output.is_relative_to(work):
        raise ValueError('Host manifest and audit output must be outside agent-readable project files.')
    protected=json.loads(manifest_path.read_text())
    required={'lab.py','core.py','numerical.py','task.json','history.csv','future.csv','backend.json','previous_runs.json'}
    if not required.issubset(protected):raise ValueError('Incomplete protected source/input manifest.')
    boundary_lab=LabBoundary(work,sys.executable,protected)
    os.chdir(work);os.environ['TERMINAL_CWD']=str(work)
    from run_agent import AIAgent
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    cls=bounded_agent_class(AIAgent)
    # Append-only host audit. A failed intent append aborts dispatch before execution.
    def record(event):
        with (output/'boundary-events.jsonl').open('a') as f:
            f.write(json.dumps({'at':time.time(),**event},ensure_ascii=False,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
    def factory(remaining,seconds):
        agent=cls(model='deepseek-v4.1-flash',provider='custom',api_mode='chat_completions',
            base_url=base_url,api_key='local-evaluation-proxy',max_iterations=remaining,max_tokens=3072,
            request_overrides={'temperature':0.2,'seed':7},enabled_toolsets=['memory','skills'],quiet_mode=True,
            skip_context_files=True,load_soul_identity=False,skip_background_review=True,skip_memory=False,
            run_budget_seconds=seconds)
        agent._disable_streaming=True
        boundary=HermesBoundary(boundary_lab,native=native_callbacks(agent),record=record,
                                deadline=read_budget(work)['deadline_epoch'])
        attach(agent,boundary)
        (output/'tools.json').write_text(json.dumps(agent.tools,indent=2))
        return agent
    def complete(_):
        # Host status checks use the same verified interpreter, with no fits.
        result=boundary_lab.dispatch('lab',{'operation':'status'})
        record({'stage':'host_completion_check','result':result})
        if result['status']!='ok' or result['result']['exit_code']!=0:return False
        checkpoint=json.loads(result['result']['stdout'])['result']['checkpoint']
        return bool(checkpoint and checkpoint['selection_after_comparison'])
    started=time.monotonic()
    try:
        result=drive(factory,work,output,is_complete=complete)
        (output/'boundary-worker.json').write_text(json.dumps({
            'protocol':'metered-093','continuation_policy':'frozen-checkpoint-v5',
            'python':sys.executable,'protected':protected,'orchestration':result},indent=2))
        return result
    finally:
        shutdown_mcp_servers()
        (output/'worker-timing.json').write_text(json.dumps({'seconds':time.monotonic()-started}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--work',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--base-url',required=True)
    parser.add_argument('--manifest',required=True)
    args=parser.parse_args();run(args.work,args.output,args.base_url,args.manifest)


if __name__=='__main__':main()
