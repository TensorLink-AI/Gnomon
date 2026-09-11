"""Bounded live-agent driver: isolated tools, persistent stores, reset chats."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import random
import sqlite3
import time

from benchmarks.ledger_optimization.agent_loop import MODEL, api_key, request_chat

from gnomon.contracts import GnomonError

from .scenario import canonical, digest, expected_provider, generate, matches, oracle
from .storage import Store, REFERENCE_SQL, execute, parameters

SYSTEM = '''Complete a longitudinal evidence checkpoint using the tools, not prose final JSON.
Both original and current queries are explicit tasks; preserve their series, unit, provider revisions,
context, origin range and source/recording cutoffs. Report matched complete origins and each provider's
mean per-origin RMSLE. Return ranking in ascending score; exact ties use provider input order.
When fewer than 3 current matched origins exist, choose last_value; otherwise choose the first current
ranked provider. Execute the chosen provider for the host-bound current request, then submit_decision
with that execution_id and both evidence answers: {matched_origins:int,scores:{provider:number},ranking:[provider]}.
An empty cohort means matched_origins=0,scores={},ranking=[]. Do not invent scores or change cutoffs.
You have 8 API turns, 12 tool attempts, 3 forecast attempts and 2 submission attempts per checkpoint.
Independent checkpoints reset chat; notebook entries and saved SQL survive. Store useful concise notes.
Historical descriptions are not causal explanations. Do not infer stockouts from zero observations.
The host supplies identical raw event ingestion to both arms, and no future scoring truth.
'''


def tool(name, description, properties, required=()):
    return {'type': 'function', 'function': dict(name=name, description=description,
        parameters=dict(type='object', additionalProperties=False, properties=properties, required=list(required)))}


def tools_for(arm):
    tools = [tool('forecast', 'Execute this provider for the exact current request. Returns typed completion.',
                  {'provider': {'type': 'string'}}, ['provider']),
             tool('submit_decision', 'Finish this checkpoint with an executed forecast and both evidence answers.',
                  {'execution_id': {'type': 'string'}, 'evidence': {'type': 'object'}, 'rationale': {'type': 'string'}},
                  ['execution_id', 'evidence']),
             tool('save_note', 'Persist a concise note or reusable instructions between checkpoints (max 4000 chars total).',
                  {'key': {'type': 'string'}, 'value': {'type': 'string'}}, ['key', 'value'])]
    if arm == 'gnomon':
        tools.append(tool('compare_context', 'Call public TemporalLedger.compare_context with the supplied query arguments. '
            'Returns calculated matched_origins, scores and ranking; no forecasts or writes.',
            {'arguments': {'type': 'object'}}, ['arguments']))
    else:
        tools.extend([tool('sql_query', 'Execute read-only SQLite. Default saved_query=matched_evidence computes exact matched '
            'RMSLE from raw forecasts/actuals. Pass the task query as params; dict-valued providers and context_filters '
            'are JSON-encoded automatically. Returns rows {provider,matched_origins,score}; [] means zero matched origins. '
            'Custom SQL also supports CTEs, window functions, LOG1P, SQRT. Max 500 rows/2 million VM steps.',
            {'params': {'type': 'object'}, 'saved_query': {'type': 'string'}, 'sql': {'type': 'string'}}, ['params']),
            tool('save_query', 'Persist a parameterized SQL statement for later use; no scoring data is precomputed.',
                 {'name': {'type': 'string'}, 'sql': {'type': 'string'}}, ['name', 'sql'])])
    return tools


class Boundary:
    def __init__(self, store, world, task):
        self.store, self.world, self.task = store, world, task
        self.calls = self.forecasts = self.submissions = 0
        self.executions = {}
        self.result = None
        self.errors = []

    def call(self, name, args):
        self.calls += 1
        if self.result is not None:
            return {'error': 'checkpoint_already_completed'}
        try:
            if self.calls > 12:
                raise ValueError('tool_attempt_budget_exhausted')
            if not isinstance(args, dict):
                raise ValueError('arguments_must_be_object')
            if name == 'compare_context' and self.store.arm == 'gnomon':
                return self.store.query(args['arguments'])
            if name == 'sql_query' and self.store.arm == 'sqlite':
                statement = args.get('sql') or self.store.saved_queries[args.get('saved_query', 'matched_evidence')]
                return self.store.sql(statement, parameters(args['params']))
            if name == 'save_query' and self.store.arm == 'sqlite':
                if len(args['sql']) > 12000 or len(self.store.saved_queries) >= 16:
                    raise ValueError('saved_query_limit')
                self.store.saved_queries[args['name']] = args['sql']
                return {'saved': args['name']}
            if name == 'save_note':
                proposed = {**self.store.notebook, args['key']: args['value']}
                if len(canonical(proposed)) > 4000:
                    raise ValueError('notebook_limit')
                self.store.notebook = proposed
                return {'saved': args['key']}
            if name == 'forecast':
                self.forecasts += 1
                if self.forecasts > 3:
                    raise ValueError('forecast_attempt_budget_exhausted')
                event = next((e for e in self.world['events'] if e['kind'] == 'forecast'
                    and e['request'] == self.task['request'] and e['provider'] == args['provider']), None)
                if event is None:
                    raise ValueError('unknown_provider_for_bound_task')
                execution = execute(event)
                completion = execution.completion()
                self.executions[execution.execution_id] = completion
                return completion
            if name == 'submit_decision':
                self.submissions += 1
                if self.submissions > 2:
                    raise ValueError('selection_attempt_budget_exhausted')
                selected = self.executions.get(args['execution_id'])
                if selected is None:
                    raise ValueError('execution_id_does_not_reference_successful_current_task')
                expected = {label: oracle(self.world['events'], query) for label, query in self.task['queries'].items()}
                supplied = args.get('evidence', {})
                invalid = [label for label in expected if not matches(supplied.get(label), expected[label])]
                if invalid:
                    raise ValueError('evidence_does_not_match_requested_query:' + ','.join(invalid))
                if selected['provider'] != expected_provider(expected['current']):
                    raise ValueError('selected_provider_does_not_follow_required_evidence_policy')
                self.result = dict(execution=selected, evidence=supplied, rationale=args.get('rationale', ''))
                return {'accepted': True, 'task_completed': True}
            raise ValueError('unknown_tool_for_arm')
        except (ValueError, KeyError, TypeError, RuntimeError, sqlite3.Error, GnomonError) as exc:
            error = {'tool': name, 'cause': str(exc), 'provider_calls': len(self.executions),
                     'task_completed': False, 'remaining_tool_attempts': max(0, 12 - self.calls)}
            self.errors.append(error)
            return {'error': error}


def checkpoint(store, world, task, seed, key, output, chat=request_chat):
    boundary = Boundary(store, world, task)
    prompt = dict(task=task, event_receipt=store.receipts[-1], notebook=store.notebook,
                  persistent_event_count=len(store.seen), storage='Gnomon ledger' if store.arm == 'gnomon' else 'SQLite')
    if store.arm == 'sqlite' and task['round'] == 0:
        prompt['installed_reference_sql'] = REFERENCE_SQL
        prompt['schema'] = {
            'predictions': 'event_id,provider,revision,series_id,unit,origin,recorded_at,horizon,step,valid_time,point,context(JSON)',
            'actuals': 'event_id,sequence,series_id,unit,valid_time,source_available_at,recorded_at,value'}
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': canonical(prompt)}]
    usage, api_errors, api_calls = [], [], 0
    started = time.perf_counter()
    path = output / f"{task['round']:03d}-{store.arm}.jsonl"
    with path.open('x') as log:
        def record(event):
            log.write(canonical(event) + '\n')
            log.flush()
        for turn in range(8):
            if boundary.calls >= 12 or boundary.submissions >= 2 or boundary.result:
                break
            record(dict(kind='request', turn=turn, messages=messages, tools=tools_for(store.arm), requested_seed=seed))
            api_calls += 1
            try:
                reply = chat(messages, tools_for(store.arm), seed, key)
            except Exception as exc:
                # No headers, credentials or unredacted remote response body in errors.
                api_errors.append(type(exc).__name__)
                record(dict(kind='api_error', category=type(exc).__name__))
                break
            record(dict(kind='response', reply=reply))
            usage.append(reply.get('usage'))
            try:
                message = reply['choices'][0]['message']
                messages.append(message)
                calls = message.get('tool_calls') or []
                if not calls:
                    messages.append({'role': 'user', 'content': 'Complete using forecast then submit_decision. Prose is not a task-bound execution selection.'})
                for call in calls:
                    try:
                        args = json.loads(call['function']['arguments'])
                    except (json.JSONDecodeError, TypeError):
                        args = None
                    result = boundary.call(call['function']['name'], args)
                    record(dict(kind='tool_result', call=call, result=result))
                    messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': canonical(result)})
            except (KeyError, TypeError, IndexError) as exc:
                api_errors.append('malformed_response:' + type(exc).__name__)
                break
    chosen = boundary.result['execution']['provider'] if boundary.result else 'last_value'
    event = next(e for e in world['events'] if e['kind'] == 'forecast'
                 and e['provider'] == chosen and e['request'] == task['request'])
    truth = world['truth'][f"{task['round']}/{task['request']['series_id']}"]
    rmsle = math.sqrt(sum((math.log1p(p) - math.log1p(a)) ** 2 for p, a in zip(event['point'], truth)) / len(truth))
    complete_usage = len(usage) == api_calls and all(isinstance(u, dict) and
        type(u.get('prompt_tokens')) is int and type(u.get('completion_tokens')) is int for u in usage)
    row = dict(world=world['seed'], agent_seed=seed, round=task['round'], family=task['family'], arm=store.arm,
        task_sha256=digest(task), event_receipt_sha256=digest(store.receipts[-1]),
        completed=boundary.result is not None, engine_execution_succeeded=bool(boundary.executions),
        evidence=boundary.result, chosen_provider=chosen, fallback_used=boundary.result is None,
        rmsle=rmsle, api_calls=api_calls, usage_complete=complete_usage,
        tokens=sum(u['prompt_tokens'] + u['completion_tokens'] for u in usage) if complete_usage else None,
        usd=None, tool_attempts=boundary.calls, provider_calls=len(boundary.executions),
        errors=boundary.errors, api_errors=api_errors, seconds=time.perf_counter() - started,
        transcript=str(path), transcript_sha256=__import__('hashlib').sha256(path.read_bytes()).hexdigest())
    (output / f"{task['round']:03d}-{store.arm}.decision.json").write_text(canonical(row) + '\n')
    return row


def run_world(world_seed, agent_seed, rounds, output, key):
    world = generate(world_seed, rounds)
    output.mkdir(parents=True, exist_ok=False)
    stores = {arm: Store(output / arm, arm) for arm in ('gnomon', 'sqlite')}
    rows = []
    rng = random.Random(world_seed * 100 + agent_seed)
    try:
        for task in world['tasks']:
            for store in stores.values():
                store.ingest(world['events'], task['now'])
            arms = list(stores)
            rng.shuffle(arms)
            for arm in arms:
                row = checkpoint(stores[arm], world, task, agent_seed, key, output)
                rows.append(row)
                print(canonical({k: row[k] for k in ('world', 'agent_seed', 'round', 'arm', 'completed', 'tokens')}), flush=True)
    finally:
        for store in stores.values():
            store.close()
    return rows


def source_manifest():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / 'src/gnomon').rglob('*.py')) + sorted(Path(__file__).parent.glob('*.py'))
    files += [Path(__file__).with_name('PROTOCOL.md'), root / 'benchmarks/ledger_optimization/agent_loop.py']
    return {str(p.relative_to(root)): __import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[100, 101, 102, 103])
    parser.add_argument('--agent-seeds', type=int, nargs='+', default=[7])
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    worlds = [generate(s, args.rounds) for s in args.seeds]  # reject reserved seeds before dispatch
    if set(args.agent_seeds) - {7, 19} or len(set(args.seeds)) != len(args.seeds) or len(set(args.agent_seeds)) != len(args.agent_seeds):
        raise ValueError('Use unique world seeds and unique agent seeds from 7,19')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(scope='development live-agent pilot', objective_achieved=False, model=MODEL,
        seeds=args.seeds, agent_seeds=args.agent_seeds, rounds=args.rounds, source=source_manifest(),
        worlds={str(w['seed']): dict(events_sha256=digest(w['events']), tasks_sha256=digest(w['tasks'])) for w in worlds},
        expected_decisions=len(worlds) * len(args.agent_seeds) * args.rounds * 2)
    (args.output / 'manifest.json').write_text(canonical(manifest) + '\n')
    key = api_key()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_world, s, a, args.rounds, args.output / f'{s}-{a}', key)
                for s in args.seeds for a in args.agent_seeds]
        rows = [row for job in jobs for row in job.result()]
    (args.output / 'decisions.json').write_text(canonical(rows) + '\n')
    from .report import summarize
    report = summarize(rows, manifest)
    (args.output / 'progress.json').write_text(canonical(report) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
