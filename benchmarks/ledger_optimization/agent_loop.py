"""Matched Engy development experiments using immutable candidate playback.

Actuals are scored only after decisions. Candidate points were computed by the
original StatsForecast run; playback through Gnomon validates their task binding.
This does not claim a fresh StatsForecast execution or a held-out evaluation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
import urllib.error
import urllib.request

from gnomon import ForecastResult, GnomonSession, InferenceEngine, TemporalLedger, resolve_final_selection
from gnomon.evidence_summary import rmsle
from gnomon.final_selection import forecast_request_fingerprint
from gnomon.build_info import build_info
from gnomon.contracts import GnomonError
from gnomon.forecast_adapter import AdapterCapabilities


MODEL = 'deepseek-v4-flash-0731'
ENDPOINT = 'https://api.engy.ai/v1/chat/completions'
ARMS = ('no_ledger', 'ledger_119', 'ledger_rmsle', 'ledger_blended', 'ledger_context', 'ledger_supported')
SYSTEM = '''Select a retail-demand forecast to minimize RMSLE over the next horizon.
All candidates have the same history and current CV evidence. Lower error is better.
You may execute up to three forecasts. Call gnomon_forecast with provider, then
select_forecast with the returned execution_id and a concise rationale. Do not
select a provider you have not executed. A single valid execution can be retained
if your final answer is prose, but multiple executions require explicit selection.
Historical evidence, when present, contains only matured outcomes as of this origin.
MAE and RMSLE may rank models differently. A rank is descriptive, not significance.
Do not infer stockouts, censored demand recovery or causal explanations from zeros.
All candidates were fitted without future target values. Keep the rationale short.
'''
FORECAST_TOOL = {'type': 'function', 'function': {'name': 'gnomon_forecast',
    'description': 'Execute a fixed candidate for this exact series/origin/horizon. The host binds the full request.',
    'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['provider'],
                   'properties': {'provider': {'type': 'string'}}}}}
SELECT_TOOL = {'type': 'function', 'function': {'name': 'select_forecast',
    'description': 'Finish by selecting a successfully executed forecast ID.',
    'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['execution_id'],
                   'properties': {'execution_id': {'type': 'string'}, 'rationale': {'type': 'string'}}}}}


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def api_key():
    key = os.environ.get('ENGY_API_KEY')
    if key:
        return key
    for line in Path('/root/Gnomon/.env').read_text().splitlines():
        if line.startswith('ENGY_API_KEY='):
            return line.split('=', 1)[1].strip().strip('\"\'')
    raise RuntimeError('ENGY_API_KEY not configured')


def request_chat(messages, tools, seed, key):
    payload = {'model': MODEL, 'messages': messages, 'tools': tools,
               'temperature': 0.2, 'max_tokens': 2048, 'seed': seed}
    request = urllib.request.Request(ENDPOINT, data=canonical(payload).encode(),
        headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=150) as response:
        return json.load(response)


def context(case, request, arm, card, memory=None):
    # This explicit allowlist prevents actuals or candidate hindsight losses from
    # entering an agent request when the host's case also contains evaluation data.
    result = {k: request[k] for k in ('series_id', 'unit', 'horizon', 'cutoff', 'frequency', 'season')}
    result['request_fingerprint'] = forecast_request_fingerprint(request)
    result['recent_history'] = request['history'][-112:]
    result['history_count'] = len(request['history'])
    result['current_cv'] = case['current_card']
    result['candidates'] = list(case['predictions'])
    if memory is not None:
        if memory['request_fingerprint'] != result['request_fingerprint']:
            raise ValueError('Historical memory belongs to another request')
        result['current_context'] = memory['current_context']
        result['raw_matched_history'] = memory['raw_history']
        result['information_contract'] = 'Every arm receives these same historical rows; ledger summaries only organize this evidence.'
    if arm == 'ledger_119':
        result['historical_evidence'] = case['legacy_evidence']
    elif arm in ('ledger_rmsle', 'ledger_blended'):
        result['historical_evidence'] = card
        if arm == 'ledger_blended':
            history = {m['provider']: m['score'] for m in card.get('lifetime', {}).get('ranking', [])}
            estimates = {p: (0.5*history[p]+0.5*cv['cv_rmsle']) if p in history else cv['cv_rmsle']
                         for p, cv in case['current_card'].items()}
            result['selection_support'] = {
                'kind': 'development_heuristic_not_validated_performance',
                'rule': 'Equal blend of current CV RMSLE and mean lifetime RMSLE; CV alone with no history.',
                'estimated_ranking': [{'provider': p, 'estimated_rmsle': estimates[p]}
                                      for p in sorted(estimates, key=estimates.__getitem__)],
                'guidance': 'Consider these estimates with sample size and regime changes. They are not measured future errors or a mandate.'}
    elif arm == 'ledger_context':
        result['historical_evidence'] = memory['context_retrieval'] if memory else {}
    elif arm == 'ledger_supported':
        from .support_screen import support_packet
        result['historical_evidence'] = support_packet(case, memory) if memory else {}
    return result


def play(case, request, revision, arm, card, seed, key, trace_path=None, memory=None):
    engine = InferenceEngine()
    expected = forecast_request_fingerprint(request)
    for name, points in case['predictions'].items():
        def predictor(req, points=points):
            if forecast_request_fingerprint(asdict(req)) != expected:
                raise ValueError('Playback task mismatch')
            return ForecastResult(point=tuple(points), series_id=req.series_id, unit=req.unit,
                                  timestamps=req.future_timestamps)
        engine.register(name, predictor, revision=revision, deterministic=True, lifecycle='stateless',
                        capabilities=AdapterCapabilities(past_covariates=True, future_covariates=True))
    session = GnomonSession(engine)
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': canonical(context(case, request, arm, card, memory))}]
    completions, wire, usage, errors = [], [], [], []
    final = None
    forecast_calls = 0
    resolution = None
    started = time.monotonic()
    for turn in range(6):
        if trace_path is not None:
            with trace_path.open('a') as stream:
                stream.write(canonical({'event': 'request_started', 'turn': turn, 'model': MODEL,
                    'messages_sha256': hashlib.sha256(canonical(messages).encode()).hexdigest()})+'\n')
        try:
            reply = request_chat(messages, ([FORECAST_TOOL] if forecast_calls < 3 else [])+[SELECT_TOOL], seed, key)
        except Exception as exc:
            errors.append({'category': type(exc).__name__, 'http_status': getattr(exc, 'code', None)})
            if trace_path is not None:
                with trace_path.open('a') as stream:
                    stream.write(canonical({'event': 'request_failed', 'turn': turn, **errors[-1]})+'\n')
            break
        if trace_path is not None:
            with trace_path.open('a') as stream:
                stream.write(canonical({'event': 'response_received', 'turn': turn, 'response': reply})+'\n')
        wire.append(reply)
        usage.append(reply.get('usage', {}))
        message = reply['choices'][0]['message']
        messages.append(message)
        calls = message.get('tool_calls') or []
        if not calls:
            final = message.get('content')
            resolution = resolve_final_selection(final_answer=final, successful_executions=completions,
                                                 expected_request=request)
            if resolution['resolved']:
                break
            messages.append({'role': 'user', 'content': canonical({
                'resolution': resolution, 'next_step': 'Execute a candidate if needed, then select its execution_id.'})})
            continue
        for call in calls:
            function = call['function']
            try:
                args = json.loads(function['arguments'])
                if not isinstance(args, dict):
                    raise ValueError('arguments must be object')
                if function['name'] == 'gnomon_forecast':
                    if forecast_calls >= 3:
                        result = {'status': 'error', 'cause': 'forecast_budget_exhausted',
                                  'next_step': 'select_forecast with an existing execution_id'}
                    else:
                        forecast_calls += 1
                        result = session.call('gnomon_forecast', {**args, 'request': request}, compact=False)
                        if result.get('status') == 'ok':
                            completions.append(result['completion'])
                            result = {'status': 'ok', 'completion': result['completion'],
                                      'next_call': {'name': 'select_forecast', 'arguments': {'execution_id': result['execution_id']}}}
                elif function['name'] == 'select_forecast':
                    final = args
                    resolution = resolve_final_selection(final_answer=final, successful_executions=completions,
                                                         expected_request=request)
                    result = resolution
                else:
                    result = {'status': 'error', 'cause': 'unknown_tool', 'accepted_tools': ['gnomon_forecast', 'select_forecast']}
            except GnomonError as exc:
                result = exc.to_dict()
            except (ValueError, TypeError, KeyError):
                result = {'status': 'error', 'cause': 'invalid_tool_arguments', 'required': 'JSON object matching tool schema'}
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': canonical(result)})
        if resolution and resolution['resolved']:
            break
    resolution = resolve_final_selection(final_answer=final, successful_executions=completions, expected_request=request)
    fallback = not resolution['resolved']
    executed = resolution.get('execution')
    selected = 'sf_seasonal_naive_7' if fallback else executed['provider']
    prediction = case['predictions'][selected] if fallback else executed['point']
    score, clipped = rmsle(zip(prediction, case['actual']), 'clip_zero')
    return {'case': {'series_id': case['series_id'], 'round': case['round'], 'origin': case['origin']},
        'arm': arm, 'seed': seed, 'rmsle': score, 'provider': selected, 'fallback_used': fallback,
        'clipped_predictions': clipped, 'forecast_attempts': forecast_calls,
        'successful_executions': len(completions), 'resolution': resolution,
        'api_calls': len(wire)+len(errors), 'api_usage': usage,
        'api_cost_usd': sum(u['cost'] for u in usage) if usage and all(isinstance(u.get('cost'), (int, float)) for u in usage) else None,
        'api_errors': errors, 'wall_seconds': time.monotonic()-started,
        'transcript': messages, 'responses': wire,
        'prediction_basis': 'task-bound playback of frozen StatsForecast candidate execution'}


def evidence_card(ledger, case, request, revision, start):
    query = ledger.compare_history(series_id=case['series_id'], horizon=request['horizon'], unit=request['unit'],
        providers={p: revision for p in case['predictions']}, start=start, end=case['origin'],
        source_as_of=case['origin'], recorded_as_of=case['origin'], metric='rmsle', recent_origins=4,
        negative_predictions='clip_zero')
    summary = query['evidence_summary']
    # Keep full pairwise/evidence payload in the ledger, not repeated in every prompt.
    return {k: v for k, v in summary.items() if k not in ('evidence_pointer', 'lifetime', 'recent')} | {
        window: {k: v for k, v in summary[window].items() if k != 'pairwise_differences'}
        for window in ('lifetime', 'recent')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', default='12')
    parser.add_argument('--seeds', default='7')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--arms', default=','.join(ARMS[:4]))
    parser.add_argument('--memory', type=Path, help='Prepared equal-information context-memory bundle')
    parser.add_argument('--confirmation-freeze', type=Path, help='Require the exact frozen confirmation implementation and full cohort')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.snapshot/'manifest.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text()).get('scope') == 'confirmation' and not args.confirmation_freeze:
        raise ValueError('Confirmation inputs require the frozen confirmation mode')
    cases = json.loads((args.snapshot/'cases.json').read_text())
    rounds = {int(v) for v in args.rounds.split(',')}
    seeds = [int(v) for v in args.seeds.split(',')]
    arms = tuple(args.arms.split(','))
    scope, frozen = 'development only', None
    if args.confirmation_freeze:
        from .confirmation import validate, validate_cases
        frozen, _ = validate(args.confirmation_freeze)
        scope = 'confirmation'
        if sorted(rounds) != frozen['rounds'] or seeds != frozen['seeds_requested'] or list(arms) != frozen['arms']:
            raise ValueError('Confirmation must run every frozen origin/seed/arm; no optional subset')
        if sorted({c['series_id'] for c in cases}) != frozen['series'] or len(cases) != 624:
            raise ValueError('Confirmation cases do not match the frozen partition')
        validate_cases(cases, frozen)
        receipt = json.loads((args.snapshot/'manifest.json').read_text())
        if receipt['scope'] != scope or receipt['freeze_sha256'] != hashlib.sha256(args.confirmation_freeze.read_bytes()).hexdigest():
            raise ValueError('Preparation was not authorized by this freeze')
        if receipt['cases_sha256'] != hashlib.sha256((args.snapshot/'cases.json').read_bytes()).hexdigest():
            raise ValueError('Prepared confirmation cases changed')
        if args.memory is None or receipt['memory_sha256'] != hashlib.sha256(args.memory.read_bytes()).hexdigest():
            raise ValueError('Prepared confirmation memory changed')
    if len(set(arms)) != len(arms) or not set(arms) <= set(ARMS) or 'no_ledger' not in arms:
        raise ValueError('Specify distinct supported arms, including no_ledger')
    if {'ledger_context', 'ledger_supported'} & set(arms) and args.memory is None:
        raise ValueError('Context/support arms require --memory and the same raw history in every arm')
    memory = {}
    if args.memory:
        bundle = json.loads(args.memory.read_text())
        if bundle['scope'] != scope or bundle['information_contract'] != 'same_raw_matched_history_in_every_arm':
            raise ValueError('Unsupported context-memory information contract')
        memory = {(p['series_id'], p['round']): p for p in bundle['packets']}
    manifest = {'scope': scope, 'model': MODEL, 'endpoint': ENDPOINT,
        'temperature': 0.2, 'max_tokens_per_turn': 2048, 'max_turns': 6, 'forecast_attempt_budget': 3,
        'seeds_requested': seeds, 'seed_honored_by_backend': 'unverified', 'rounds': sorted(rounds), 'arms': arms,
        'snapshot_sha256': hashlib.sha256((args.snapshot/'cases.json').read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_rule_sha256': hashlib.sha256(Path(__file__).with_name('support_screen.py').read_bytes()).hexdigest()
                               if 'ledger_supported' in arms else None,
        'gnomon_build': build_info(),
        'system_prompt_sha256': hashlib.sha256(SYSTEM.encode()).hexdigest(),
        'memory_sha256': hashlib.sha256(args.memory.read_bytes()).hexdigest() if args.memory else None,
        'historical_information_contract': 'same_raw_matched_history_in_every_arm' if args.memory else 'historical_records_only_in_ledger_arms',
        'series': sorted({c['series_id'] for c in cases}),
        'expected_decisions': sum(c['round'] in rounds for c in cases)*len(seeds)*len(arms),
        'selection_policy': 'Gnomon 1.1.9 execution-bound resolver in every arm; same bounded corrections',
        'fallback': 'sf_seasonal_naive_7; included in primary denominator'}
    if frozen is not None:
        for field in ('expected_decisions', 'model', 'temperature', 'max_tokens_per_turn', 'max_turns', 'forecast_attempt_budget',
                      'historical_information_contract'):
            if manifest[field] != frozen[field]:
                raise ValueError('Runtime confirmation settings differ from freeze: '+field)
        manifest['confirmation_freeze_sha256'] = hashlib.sha256(args.confirmation_freeze.read_bytes()).hexdigest()
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists() and canonical(json.loads(manifest_path.read_text())) != canonical(manifest):
        raise ValueError('Cannot resume with changed experiment manifest')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    ledger = TemporalLedger(args.ledger, create=False)
    tasks = []
    key = api_key()
    for case in cases:
        if case['round'] not in rounds:
            continue
        original = ledger.execution(case['execution_ids'][0])
        request, revision = original['request'], original['revision']
        card = evidence_card(ledger, case, request, revision, cases[0]['origin'])
        for seed in seeds:
            order = list(arms)
            random.Random(f'{seed}:{case["round"]}:{case["series_id"]}').shuffle(order)
            for arm in order:
                path = args.output/f'{case["round"]:04d}-{case["series_id"]}-{seed}-{arm}.json'
                if not path.exists():
                    packet = memory[case['series_id'], case['round']] if args.memory else None
                    tasks.append((path, case, request, revision, arm, card, seed, packet))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(play, case, request, revision, arm, card, seed, key, path.with_suffix('.wire.jsonl'), packet): path
                   for path, case, request, revision, arm, card, seed, packet in tasks}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                path.with_suffix('.harness-error.json').write_text(canonical({
                    'status': 'harness_failure', 'category': type(exc).__name__,
                    'message': str(exc), 'scored': False})+'\n')
                print(canonical({'case_path': str(path), 'harness_failure': type(exc).__name__}), flush=True)
                continue
            path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            print(canonical({k: result[k] for k in ('case', 'arm', 'rmsle', 'provider', 'fallback_used', 'api_calls')}), flush=True)


if __name__ == '__main__':
    main()
