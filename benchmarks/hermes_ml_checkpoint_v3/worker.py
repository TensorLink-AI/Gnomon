"""Pinned Hermes with bounded continuations and local budget-exit reporting.

Only orchestration is adapted. Native tools and returned model messages are retained.
Each continuation is recorded; no forecast or selection is supplied by the host.
"""
import json
import os
from pathlib import Path
import sys
import time

try:
    from .policy import MAX_CORRECTIONS, correction_reason, correction_prompt
except ImportError:  # Invoked as the pinned worker script in an isolated interpreter.
    from policy import MAX_CORRECTIONS, correction_reason, correction_prompt


def read_budget(work):
    return json.loads((work / 'agent-budget.json').read_text())


def workflow_complete(work):
    """Read the lab's validated state without fitting or selecting anything.

    Final scoring independently validates all execution bindings and protected inputs.
    """
    import subprocess
    response = subprocess.run([sys.executable, 'lab.py', 'status'], cwd=work,
                              text=True, capture_output=True, timeout=20)
    if response.returncode:
        return False
    selected = json.loads(response.stdout)['result']['checkpoint']
    return bool(selected and selected['selection_after_comparison'])


def bounded_agent_class(base):
    class BoundedAgent(base):
        def _handle_max_iterations(self, messages, api_call_count):
            # Hermes otherwise spends a 17th request to summarize. Chat is not graded.
            return 'Request budget reached. The saved checkpoint remains authoritative.'
    return BoundedAgent


def drive(factory, work, output, *, now=time.time, is_complete=workflow_complete):
    corrections = 0
    history = []
    prompt = (work / 'TASK.md').read_text()
    attempts = []
    while True:
        budget = read_budget(work)
        remaining = budget['remaining_requests']
        seconds = budget['deadline_epoch'] - now()
        if remaining <= 0 or seconds <= 0:
            stop = 'request_budget_exhausted' if remaining <= 0 else 'deadline_reached'
            break
        agent = factory(remaining, seconds)
        # A fresh Hermes turn does not reset the host request count or wall-clock deadline.
        result = agent.run_conversation(user_message=prompt, conversation_history=history)
        (output / f'hermes-attempt-{len(attempts)+1:02d}.json').write_text(json.dumps(result, default=str, indent=2))
        attempts.append(result)
        full = is_complete(work)
        if full:
            stop = 'checkpoint_workflow_complete'
            break
        reason = correction_reason(result, full)
        budget = read_budget(work)
        if budget['remaining_requests'] <= 0 or budget['deadline_epoch'] <= now():
            stop = 'request_budget_exhausted' if budget['remaining_requests'] <= 0 else 'deadline_reached'
            break
        if not reason:
            stop = 'incomplete_without_recoverable_termination'
            break
        if corrections >= MAX_CORRECTIONS:
            stop = 'correction_limit_reached'
            break
        corrections += 1
        with (output / 'corrections.jsonl').open('a') as f:
            f.write(json.dumps({'correction':corrections,'cause':reason,'budget':budget}) + '\n')
        history = result.get('messages', history)
        prompt = correction_prompt(reason)
        # Hermes omits a discarded repetition from returned conversation history.
        # Its exact original response remains in API logs and this attempt record.
    if attempts:
        (output / 'hermes-result.json').write_text(json.dumps(attempts[-1], default=str, indent=2))
    summary = {'protocol':'checkpoint-v3','corrections':corrections,
               'attempts':len(attempts),'stop_reason':stop,'final_budget':read_budget(work)}
    (output / 'orchestration.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    work, output, base_url, arm = sys.argv[1:]
    work, output = Path(work), Path(output)
    os.chdir(work)
    os.environ['TERMINAL_CWD'] = str(work)
    from run_agent import AIAgent
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    bounded = bounded_agent_class(AIAgent)
    events = []

    def event(name, data):
        events.append({'event':name,'data':data})
        (output / 'hermes-events.json').write_text(json.dumps(events, default=str, indent=2))

    def factory(remaining, seconds):
        agent = bounded(
            model='deepseek-v4-flash-0731', provider='custom',
            api_mode='chat_completions', base_url=base_url,
            api_key='local-evaluation-proxy', max_iterations=remaining, max_tokens=3072,
            request_overrides={'temperature':0.2,'seed':7},
            enabled_toolsets=['terminal','file','memory','skills'], quiet_mode=True,
            skip_context_files=True, load_soul_identity=False,
            skip_background_review=True, skip_memory=False,
            run_budget_seconds=seconds, event_callback=event,
        )
        agent._disable_streaming = True
        (output / 'tools.json').write_text(json.dumps(agent.tools, indent=2))
        (output / 'discovery.json').write_text('[]\n')
        return agent

    started = time.monotonic()
    try:
        drive(factory, work, output)
    finally:
        shutdown_mcp_servers()
        (output / 'worker-timing.json').write_text(json.dumps({'seconds':time.monotonic()-started}))


if __name__ == '__main__':
    main()
