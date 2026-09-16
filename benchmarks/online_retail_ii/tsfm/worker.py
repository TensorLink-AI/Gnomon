"""Pinned Hermes worker: only bounded retail and native text-memory tools."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time

from benchmarks.ledger_optimization.hermes_boundary_093 import (
    HermesBoundary, NATIVE_TOOLS, bounded_agent_class, native_callbacks,
)
from benchmarks.ledger_optimization.execution_boundary_093 import fields, integer
from benchmarks.hermes_ml_checkpoint_v6.boundary_schemas_093 import tool_schemas, schema, obj
from benchmarks.online_retail_ii.data import dump, sha
from benchmarks.online_retail_ii.tsfm.agent import CANDIDATES


class RetailLab:
    def __init__(self, case, output, numerical_python, repo, arm, deadline):
        self.case, self.output = Path(case), Path(output)
        self.python, self.repo, self.arm, self.deadline = numerical_python, repo, arm, deadline
        self.attempts = 0
        self.protected = {p.name:sha(p) for p in self.case.iterdir() if p.is_file()}
        self.selected = None

    def numerical(self, operation, extra):
        seconds = self.deadline-time.time()
        if seconds <= 0:
            raise ValueError('Deadline reached; no numerical operation started')
        argv = [self.python, '-m', 'benchmarks.online_retail_ii.tsfm.cli', operation,
                '--case', str(self.case), *extra]
        p = subprocess.run(argv, cwd=self.repo, capture_output=True, text=True,
                           timeout=min(90, seconds), env={k:v for k,v in os.environ.items() if k != 'PYTHONPATH'})
        with (self.output/'numerical-calls.jsonl').open('a') as stream:
            stream.write(json.dumps({'argv':argv, 'exit_code':p.returncode,
                                    'stdout':p.stdout, 'stderr':p.stderr})+'\n')
        if p.returncode:
            raise ValueError('Numerical operation rejected; see host diagnostic reference numerical-calls.jsonl')
        return json.loads(p.stdout)

    def dispatch(self, tool, args):
        if {p.name:sha(p) for p in self.case.iterdir() if p.is_file()} != self.protected:
            raise ValueError('Protected input changed')
        if tool != 'retail':
            raise ValueError('Use retail or an advertised native text-memory tool; no shell or arbitrary code is available')
        fields(args, {'operation','provider','execution_id','offset','limit'}, {'operation'})
        op = args['operation']
        allowed = {'status':set(), 'history':{'offset','limit'}, 'cv':set(),
                   'outcomes':{'offset','limit'}, 'ledger':set(), 'forecast':{'provider'},
                   'select':{'execution_id'}}
        if op not in allowed:
            raise ValueError('Unknown retail operation')
        fields(args, {'operation'}|allowed[op], {'operation'}|({'provider'} if op=='forecast' else {'execution_id'} if op=='select' else set()))
        if op == 'forecast':
            if args['provider'] not in CANDIDATES:
                raise ValueError('provider must be one of the twelve task candidates')
            availability=json.loads((self.case/'task.json').read_text())['model_availability']
            if not availability[args['provider']]['available']:
                raise ValueError('Insufficient observed history: '+json.dumps(availability[args['provider']]))
            if self.attempts >= 4:
                raise ValueError('Four numerical attempts exhausted; select an existing execution')
            self.attempts += 1
            result = self.numerical('execute', ['--provider',args['provider'], '--backend',
                'direct' if self.arm=='hermes' else 'gnomon', '--output',str(self.output/'executions')])
            result = {k:result[k] for k in ('provider','execution_id','point','series_id','unit',
                'future_timestamps','request_fingerprint','fallback_used','executed_provider','backend')}
            result['next_action'] = 'retail select with this execution_id; retain numerical fallback disclosure'
        elif op == 'select':
            result = self.numerical('resolve', ['--executions',str(self.output/'executions'), '--selection',args['execution_id']])
            self.selected = args['execution_id']; dump(self.output/'selection.json', {'execution_id':self.selected})
            result = {'selected':True,'execution_id':self.selected,'provider':result['execution']['provider']}
        elif op == 'status':
            executions = [json.loads(p.read_text()) for p in (self.output/'executions').glob('*.json')]
            result = {'numerical_attempts':self.attempts, 'numerical_limit':4, 'selected_execution_id':self.selected,
                'executions':[{k:r[k] for k in ('provider','execution_id','fallback_used')} for r in executions]}
        elif op == 'cv':
            result = json.loads((self.case/'current-cv.json').read_text())
        elif op == 'ledger':
            if self.arm != 'ledger_tsfm':
                raise ValueError('Ledger interface is disabled in this arm; outcomes exposes identical raw historical facts')
            report = json.loads((self.output/'ledger/report.json').read_text())
            result = {k:v for k,v in report.items() if k not in ('references','excluded')}
            result['excluded_count'] = len(report.get('excluded',[]))
            result['reference_count'] = len(report.get('references',[]))
        else:
            offset = integer(args.get('offset',0),0,100000,'offset')
            limit = integer(args.get('limit',2 if op=='outcomes' else 56),1,2 if op=='outcomes' else 100,'limit')
            if op == 'history':
                import csv
                values = list(csv.DictReader((self.case/'history.csv').open()))
            else:
                values = json.loads((self.case/'matured-outcomes.json').read_text())['records']
            result = {'total':len(values),'offset':offset,'next_offset':min(len(values),offset+limit),
                      'records':values[offset:offset+limit]}
        return {'status':'ok','result':result}


def tools(native):
    retail = schema('retail', 'Inspect current CV/history or past matured outcomes; forecast one candidate; '
        'select a returned execution_id. ledger provides verified historical rankings only in the ledger arm. '
        'Forecasts do not reveal current actuals. Four numerical attempts maximum.', obj({
            'operation':{'type':'string','enum':['status','history','cv','outcomes','ledger','forecast','select']},
            'provider':{'type':'string','enum':list(CANDIDATES)}, 'execution_id':{'type':'string'},
            'offset':{'type':'integer','minimum':0}, 'limit':{'type':'integer','minimum':1,'maximum':100},
        }, ['operation']))
    return [retail]+[deepcopy(t) for t in tool_schemas(native) if t['function']['name'] in NATIVE_TOOLS]


def run(args):
    from run_agent import AIAgent
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    output = Path(args.output)
    deadline = args.deadline
    lab = RetailLab(args.case, output, args.numerical_python, args.repo, args.arm, deadline)
    def record(event):
        with (output/'boundary-events.jsonl').open('a') as f:
            f.write(json.dumps({'at':time.time(),**event},allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
    cls = bounded_agent_class(AIAgent)
    history = []; prompt = Path(args.prompt).read_text()
    try:
        for attempt in range(3):
            budget = json.loads((output/'agent-budget.json').read_text())
            if budget['remaining_requests']<=0 or time.time()>=deadline:
                break
            agent = cls(model='deepseek-v4.1-flash',provider='custom',api_mode='chat_completions',
                base_url=args.base_url,api_key='local-evaluation-proxy',max_iterations=budget['remaining_requests'],
                max_tokens=3072,request_overrides={'temperature':.2,'seed':args.seed},
                enabled_toolsets=['memory','skills'],quiet_mode=True,skip_context_files=True,
                load_soul_identity=False,skip_background_review=True,skip_memory=False,
                run_budget_seconds=max(1,deadline-time.time()))
            agent._disable_streaming=True
            boundary = HermesBoundary(lab,native=native_callbacks(agent),record=record,deadline=deadline)
            agent.tools=tools(agent.tools); agent.valid_tool_names={t['function']['name'] for t in agent.tools}
            agent.execution_boundary=boundary
            dump(output/'tools.json',agent.tools)
            result=agent.run_conversation(user_message=prompt,conversation_history=history)
            dump(output/f'hermes-attempt-{attempt+1}.json',result)
            if lab.selected or result.get('failed') or result.get('error'):
                break
            # A single matching execution is authoritative without parsing final prose.
            check=lab.numerical('resolve',['--executions',str(output/'executions')]) if len(list((output/'executions').glob('*.json')))==1 else None
            if check and check['resolved']:
                break
            history=result.get('messages',history)
            prompt='No unambiguous executed selection is saved. Use retail status, then forecast if none exists and select its execution_id. Same remaining budgets; do not print tool markup.'
        dump(output/'worker-result.json',{'numerical_attempts':lab.attempts,'selected_execution_id':lab.selected})
    finally:
        shutdown_mcp_servers()


def main():
    p=argparse.ArgumentParser()
    for field in ('case','output','numerical-python','repo','arm','base-url','prompt'):p.add_argument('--'+field,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--deadline',type=float,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
