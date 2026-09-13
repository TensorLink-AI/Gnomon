"""Task-free, recorded service readiness before a fresh agent session.

Does not retry an agent, reset a running budget, score data or select a model.
"""
from datetime import datetime,timezone
import base64
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

PAYLOAD={'model':'deepseek-v4-flash-0731','messages':[{'role':'user','content':'Reply with the word READY.'}],
         'temperature':0,'seed':7,'max_tokens':16,'stream':False}
MAX_PROBES=10
RETRY_SECONDS=60
PROBE_TIMEOUT=30


def decision(status,raw):
    try:body=json.loads(raw)
    except (ValueError,UnicodeError):body={}
    error=body.get('error') if isinstance(body,dict) else None
    kind=error.get('type') if isinstance(error,dict) else None
    code=error.get('code') if isinstance(error,dict) else None
    if status in (408,429,500,502,503,504) or code in (408,429,500,502,503,504) or kind in ('upstream_error','rate_limit_error'):
        return {'ready':False,'retryable':True,'cause':'service_unavailable'}
    choices=body.get('choices') if isinstance(body,dict) else None
    message=choices[0].get('message') if isinstance(choices,list) and choices and isinstance(choices[0],dict) else None
    content=message.get('content') or message.get('reasoning_content') if isinstance(message,dict) else None
    ready=status==200 and not error and isinstance(content,str) and bool(content.strip())
    return {'ready':ready,'retryable':False,'cause':'completion_received' if ready else 'probe_contract_or_authorization_failure'}


def _network(api_key):
    if not isinstance(api_key,str) or not api_key.strip():raise ValueError('API credential required')
    request=urllib.request.Request('https://api.engy.ai/v1/chat/completions',data=json.dumps(PAYLOAD).encode(),
                headers={'Authorization':'Bearer '+api_key,'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=PROBE_TIMEOUT) as response:raw,status=response.read(),response.status
    except urllib.error.HTTPError as exc:raw,status=exc.read(),exc.code
    except (OSError,TimeoutError) as exc:
        raw=json.dumps({'error':{'type':'upstream_error','message':type(exc).__name__}}).encode();status=502
    return status,raw.replace(api_key.encode(),b'[REDACTED]')


def probe(api_key):
    """Bound the entire request with a process deadline, including response reads."""
    if not isinstance(api_key,str) or not api_key.strip():raise ValueError('API credential required')
    started=time.monotonic()
    process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--probe-worker'],
                             stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        stdout,stderr=process.communicate(json.dumps({'api_key':api_key}).encode(),timeout=PROBE_TIMEOUT)
        if process.returncode:raise ValueError('Probe worker failed')
        value=json.loads(stdout);raw=base64.b64decode(value['body']);status=value['status']
    except subprocess.TimeoutExpired:
        process.kill();process.communicate()
        raw=b'{"error":{"type":"upstream_error","message":"probe_wall_clock_deadline_exceeded"}}';status=504
    except (ValueError,KeyError):
        raw=b'{"error":{"type":"probe_worker_failure","message":"See probe worker contract"}}';status=400
    return status,raw.replace(api_key.encode(),b'[REDACTED]'),time.monotonic()-started


def wait_ready(output,perform_probe,*,sleep=time.sleep):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    attempts=[]
    for number in range(1,MAX_PROBES+1):
        (output/f'probe-{number:02}-request.json').write_text(json.dumps(PAYLOAD,indent=2)+'\n')
        status,raw,seconds=perform_probe()
        (output/f'probe-{number:02}-response.txt').write_bytes(raw)
        verdict=decision(status,raw)
        try:body=json.loads(raw)
        except (ValueError,UnicodeError):body={}
        record={'number':number,'at':datetime.now(timezone.utc).isoformat(),'status':status,'seconds':seconds,
                'usage':body.get('usage') if isinstance(body,dict) else None,**verdict}
        (output/f'probe-{number:02}-receipt.json').write_text(json.dumps(record,indent=2)+'\n');attempts.append(record)
        waiting=verdict['retryable'] and number<MAX_PROBES
        state={'ready':verdict['ready'],'probe_count':number,'attempts':attempts,
               'agent_started':False,'agent_requests':0,'numerical_attempts':0,
               'known_wait_seconds':(number-1)*RETRY_SECONDS,
               'waiting':waiting,
               'cost_scope':'All probes count toward experiment cost; separate from agent requests.',
               'terminal_cause':None if verdict['ready'] or waiting else ('service_admission_exhausted' if number==MAX_PROBES and verdict['retryable'] else verdict['cause'])}
        temporary=output/'status.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(output/'status.json')
        if verdict['ready'] or not state['waiting']:return state
        sleep(RETRY_SECONDS)
    raise AssertionError('Unreachable')


if __name__=='__main__':
    if sys.argv[1:]!=['--probe-worker']:raise SystemExit('Internal readiness worker only')
    status,raw=_network(json.load(sys.stdin)['api_key'])
    print(json.dumps({'status':status,'body':base64.b64encode(raw).decode()}))
