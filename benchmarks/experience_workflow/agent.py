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
from .storage import Store, REFERENCE_SQL, execute, parameters, sql_answer

SYSTEM = '''Complete a longitudinal evidence checkpoint using the tools, not prose final JSON.
Both original and current queries are explicit tasks; preserve their series, unit, provider revisions,
context, origin range and source/recording cutoffs. Report matched complete origins and each provider's
mean per-origin RMSLE. Return ranking in ascending score; exact ties use provider input order.
When fewer than 3 current matched origins exist, choose last_value; otherwise choose the first current
ranked provider. Execute the chosen provider for the host-bound current request, then submit_decision
with that execution_id and both evidence answers: {matched_origins:int,scores:{provider:number},ranking:[provider]}.
The submission shape is evidence={"original":{matched_origins,scores,ranking},"current":{matched_origins,scores,ranking}}.
Query original and current separately. Never put both query objects inside one compare_context call.
An empty cohort means matched_origins=0,scores={},ranking=[]. Do not invent scores or change cutoffs.
Empty cohorts during cold start are expected and valid: follow the last_value rule, without hunting for nonexistent history.
You have 8 API turns, 12 tool attempts, 3 forecast attempts and 2 submission attempts per checkpoint.
Independent checkpoints reset chat; notebook entries and saved SQL survive. Store useful concise notes.
Historical descriptions are not causal explanations. Do not infer stockouts from zero observations.
The host supplies identical raw event ingestion to both arms, and no future scoring truth.
'''

EVIDENCE_SCHEMA = dict(type='object', additionalProperties=False,
    required=['matched_origins', 'scores', 'ranking'], properties={
        'matched_origins': {'type': 'integer', 'minimum': 0},
        'scores': {'type': 'object', 'additionalProperties': {'type': 'number', 'minimum': 0}},
        'ranking': {'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True}})
QUERY_SCHEMA = dict(type='object', additionalProperties=False,
    required=['series_id', 'horizon', 'providers', 'start', 'end', 'source_as_of', 'recorded_as_of', 'context_filters'],
    properties={**{k: {'type': 'string'} for k in ('series_id', 'start', 'end', 'source_as_of', 'recorded_as_of')},
        'unit': {'type': ['string', 'null']}, 'horizon': {'type': 'integer', 'minimum': 1},
        'providers': {'type': 'object', 'additionalProperties': {'type': 'string'}},
        'context_filters': {'type': 'object', 'additionalProperties': {'type': 'string'}},
        'metric': {'enum': ['mae', 'rmsle'], 'default': 'mae',
                   'description': 'Public Gnomon default is MAE. This checkpoint requires explicit metric=rmsle.'},
        'recent_origins': {'type': 'integer', 'minimum': 1}})


def tool(name, description, properties, required=()):
    return {'type': 'function', 'function': dict(name=name, description=description,
        parameters=dict(type='object', additionalProperties=False, properties=properties, required=list(required)))}


def tools_for(arm):
    tools = [tool('forecast', 'Execute this provider for the exact current request. Returns typed completion.',
                  {'provider': {'type': 'string'}}, ['provider']),
             tool('submit_decision', 'Finish this checkpoint with an executed forecast and both evidence answers.',
                  {'execution_id': {'type': 'string'}, 'evidence': {'type': 'object', 'additionalProperties': False,
                   'required': ['original', 'current'], 'properties': {'original': EVIDENCE_SCHEMA, 'current': EVIDENCE_SCHEMA}},
                   'rationale': {'type': 'string'}},
                  ['execution_id', 'evidence']),
             tool('save_note', 'Persist a concise note or reusable instructions between checkpoints (max 4000 chars total).',
                  {'key': {'type': 'string'}, 'value': {'type': 'string'}}, ['key', 'value'])]
    if arm == 'gnomon':
        tools.append(tool('compare_context', 'Call public TemporalLedger.compare_context with the supplied query arguments. '
            'Set metric=rmsle explicitly for this task; the public default is mae. Returns effective metric/query and '
            'calculated matched_origins, scores and ranking; no forecasts or writes.',
            {'arguments': QUERY_SCHEMA}, ['arguments']))
    else:
        tools.extend([tool('sql_query', 'Execute read-only SQLite. Default saved_query=matched_evidence computes exact matched '
            'RMSLE from raw forecasts/actuals. Pass the task query as params; dict-valued providers and context_filters '
            'are JSON-encoded automatically. The default query returns matched_origins,scores,ranking, including an explicit empty cohort. '
            'Custom SQL also supports CTEs, window functions, LOG1P, SQRT. Max 500 rows/2 million VM steps.',
            {'params': {'type': 'object'}, 'saved_query': {'type': 'string'}, 'sql': {'type': 'string'}}, ['params']),
            tool('describe_query', 'Read a saved parameterized SQL statement. Saved queries live in the host registry, '
                 'not a SQL table. The default name is matched_evidence. This only returns code; it does not execute a query.',
                 {'name': {'type': 'string'}}, ['name']),
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
        self.query_results = []

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
                result = self.store.query(args['arguments'])
                self.query_results.append(result)
                return result
            if name == 'sql_query' and self.store.arm == 'sqlite':
                statement = args.get('sql') or self.store.saved_queries[args.get('saved_query', 'matched_evidence')]
                rows = self.store.sql(statement, parameters(args['params']))
                if statement == REFERENCE_SQL:
                    providers = args['params']['providers']
                    if isinstance(providers, str):
                        providers = json.loads(providers)
                    result = sql_answer(rows, providers)
                    result['query'] = {k: v for k, v in args['params'].items() if k not in ('metric', 'recent_origins')}
                    self.query_results.append(result)
                    return result
                return rows
            if name == 'save_query' and self.store.arm == 'sqlite':
                if len(args['sql']) > 12000 or len(self.store.saved_queries) >= 16:
                    raise ValueError('saved_query_limit')
                self.store.saved_queries[args['name']] = args['sql']
                self.store.persist()
                return {'saved': args['name']}
            if name == 'describe_query' and self.store.arm == 'sqlite':
                return {'name': args['name'], 'sql': self.store.saved_queries[args['name']],
                        'storage': 'host_registry_not_sql_table'}
            if name == 'save_note':
                proposed = {**self.store.notebook, args['key']: args['value']}
                if len(canonical(proposed)) > 4000:
                    raise ValueError('notebook_limit')
                self.store.notebook = proposed
                self.store.persist()
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
                if not isinstance(supplied, dict) or set(supplied) != {'original', 'current'}:
                    raise ValueError('evidence_requires_separate_original_and_current_objects; each needs matched_origins,scores,ranking')
                invalid = [label for label in expected if not matches(supplied.get(label), expected[label])]
                if invalid:
                    if any(result['metric'] != 'rmsle' for result in self.query_results[-2:]):
                        raise ValueError('evidence_metric_mismatch: a recent query returned MAE; rerun the affected query with metric=rmsle, preserving all other task fields')
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
    if store.arm == 'sqlite':
        prompt['schema'] = {
            'predictions': 'event_id,provider,revision,series_id,unit,origin,recorded_at,horizon,step,valid_time,point,context(JSON)',
            'actuals': 'event_id,sequence,series_id,unit,valid_time,source_available_at,recorded_at,value'}
        prompt['saved_queries'] = list(store.saved_queries)
        if task['round'] == 0:
            prompt['installed_reference_sql'] = REFERENCE_SQL
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


def run_world(world_seed, agent_seed, rounds, output, key, confirmation=False):
    world = generate(world_seed, rounds, _confirmation=confirmation)
    output.mkdir(parents=True, exist_ok=False)
    stores = {arm: Store(output / arm, arm) for arm in ('gnomon', 'sqlite')}
    rows = []
    rng = random.Random(world_seed * 100 + agent_seed)
    try:
        for task in world['tasks']:
            if (output.parent / 'STOP').exists():
                break
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
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--agent-seeds', type=int, nargs='+')
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--confirmation-freeze', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 24:
        raise ValueError('workers must be between 1 and 24')
    key = None
    confirmation = args.confirmation_freeze is not None
    if confirmation:
        if args.seeds is not None or args.agent_seeds is not None or args.rounds is not None:
            raise ValueError('Confirmation cannot override frozen seeds or rounds')
        key = api_key()  # A missing local credential must not consume the cohort.
        from .freeze import consume
        receipt = consume(args.confirmation_freeze, args.output)
        args.seeds, args.agent_seeds, args.rounds = receipt['seeds'], receipt['agent_seeds'], receipt['rounds']
    else:
        args.seeds = args.seeds or [100, 101, 102, 103]
        args.agent_seeds = args.agent_seeds or [7]
        args.rounds = args.rounds if args.rounds is not None else 12
    worlds = [generate(s, args.rounds, _confirmation=confirmation) for s in args.seeds]
    if set(args.agent_seeds) - {7, 19} or len(set(args.seeds)) != len(args.seeds) or len(set(args.agent_seeds)) != len(args.agent_seeds):
        raise ValueError('Use unique world seeds and unique agent seeds from 7,19')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(scope='confirmation' if confirmation else 'development live-agent pilot', objective_achieved=False, model=MODEL,
        confirmation_guard_passed=confirmation, prerequisites_passed=confirmation,
        seeds=args.seeds, agent_seeds=args.agent_seeds, rounds=args.rounds, source=source_manifest(),
        worlds={str(w['seed']): dict(events_sha256=digest(w['events']), tasks_sha256=digest(w['tasks'])) for w in worlds},
        expected_decisions=len(worlds) * len(args.agent_seeds) * args.rounds * 2)
    (args.output / 'manifest.json').write_text(canonical(manifest) + '\n')
    key = key or api_key()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_world, s, a, args.rounds, args.output / f'{s}-{a}', key, confirmation)
                for s in args.seeds for a in args.agent_seeds]
        rows = [row for job in jobs for row in job.result()]
    unchanged = manifest['source'] == source_manifest()
    (args.output / 'source_audit.json').write_text(canonical({'unchanged': unchanged}) + '\n')
    (args.output / 'decisions.json').write_text(canonical(rows) + '\n')
    from .report import summarize
    report = summarize(rows, {**manifest, 'source_unchanged': unchanged})
    (args.output / 'progress.json').write_text(canonical(report) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
