"""Prepare comparable-context evidence from frozen DEVELOPMENT ledger requests."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from statistics import mean, pstdev

from gnomon import TemporalLedger
from gnomon.ids import FixedClock
from gnomon.final_selection import forecast_request_fingerprint


def labels(request):
    history = request['history'][-28:]
    if len(history) < 28:
        raise ValueError('Context protocol requires 28 observed history values')
    zero = sum(v == 0 for v in history)/len(history)
    before, after = mean(history[:14]), mean(history[14:])
    trend = ('rising' if after else 'stable') if before == 0 else 'falling' if after/before < .75 else 'rising' if after/before > 4/3 else 'stable'
    average = mean(history)
    names = request.get('future_covariate_names', [])
    promotion = 'unknown'
    if 'onpromotion' in names:
        index = names.index('onpromotion')
        values = [row[index] for row in request['future_covariates']]
        if len(values) != request['horizon']:
            raise ValueError('Incomplete future promotion plan')
        promotion = 'all' if all(values) else 'some' if any(values) else 'none'
    return {'sparsity': 'low' if zero < .1 else 'intermittent' if zero < .5 else 'high',
            'trend': trend, 'volatility': 'volatile' if average and pstdev(history)/average > 1 else 'stable',
            'promotion': promotion}


def candidates(current):
    return [{key: current[key] for key in keys} for keys in (
        ('sparsity', 'trend', 'promotion', 'volatility'), ('sparsity', 'trend', 'promotion'),
        ('sparsity', 'trend'), ('sparsity',), ())]


def compact_summary(summary):
    if summary is None:
        return None
    return {k: v for k, v in summary.items() if k not in ('lifetime', 'recent', 'evidence_pointer')} | {
        window: {k: v for k, v in summary[window].items() if k != 'pairwise_differences'}
        for window in ('lifetime', 'recent')}


def prepare(cases, ledger, output):
    if output.exists():
        raise ValueError('Use a new output directory; original evidence must remain unchanged')
    output.mkdir(parents=True)
    # SQLite backup is safe even if a WAL exists. Open the source read-only.
    with sqlite3.connect(ledger.resolve().as_uri()+'?mode=ro', uri=True) as source:
        with sqlite3.connect(output/'ledger.db') as destination:
            source.backup(destination)
    memory = TemporalLedger(output/'ledger.db', create=False)
    requests, contexts, decision_ids, revisions = {}, {}, {}, {}
    for case in cases:
        identity = (case['series_id'], case['round'])
        execution = memory.execution(case['execution_ids'][0])
        executions = [execution, *(memory.execution(eid) for eid in case['execution_ids'][1:])]
        revisions[identity] = {row['provider']: row['revision'] for row in executions}
        request = execution['request']
        requests[identity] = request
        current = labels(request)
        contexts[case['series_id'], datetime.fromisoformat(case['origin'])] = current
        memory.clock = FixedClock(datetime.fromisoformat(case['origin']))
        decision = memory.record_decision_summary(execution_id=execution['execution_id'],
            rationale='Synthetic reconstruction of forecast-time observable context; not a model preference.',
            assumptions=['Promotion plan is known as in the original recipe; context is not a causal explanation.'],
            invalidation_conditions=['Source data or promotion plan revised.'], context=[{
                'key': k, 'value': v, 'valid_from': case['origin'], 'valid_to': request['future_timestamps'][0],
                'source_available_at': case['origin'],
                'source_ref': 'derived_request_sha256:'+forecast_request_fingerprint(request)} for k, v in current.items()])
        decision_ids[identity] = decision['decision_id']
    packets, screens = [], []
    for case in cases:
        identity = (case['series_id'], case['round'])
        request = requests[identity]
        current = contexts[case['series_id'], datetime.fromisoformat(case['origin'])]
        providers = list(case['predictions'])
        query = dict(series_id=case['series_id'], horizon=request['horizon'], unit=request['unit'],
                     providers=revisions[identity],
                     start=cases[0]['origin'], end=case['origin'], source_as_of=case['origin'], recorded_as_of=case['origin'],
                     metric='rmsle', recent_origins=4, negative_predictions='clip_zero')
        retrieval = memory.retrieve_context(**query, context_candidates=candidates(current), min_origins=4)
        broad = memory.compare_history(**query)
        rows = []
        for origin in broad['origins']:
            models = {m['provider']: m for m in origin['models']}
            rows.append({'origin': origin['origin'], 'context': contexts[case['series_id'], datetime.fromisoformat(origin['origin'])],
                         'mae': [models[p]['mae'] for p in providers], 'rmsle': [models[p]['rmsle'] for p in providers]})
        # Same raw evidence in all arms, with numeric columns in declared order.
        raw = {'providers': providers, 'records': rows, 'metric_definition': 'Per-origin mean absolute error and RMSLE on complete matched horizons',
               'source_as_of': case['origin'], 'recorded_as_of': case['origin']}
        summary = {k: v for k, v in retrieval.items() if k not in ('comparison', 'cohorts')}
        summary['cohorts'] = [{k: v for k, v in cohort.items() if k != 'evidence_summary'} for cohort in retrieval['cohorts']]
        summary['selected_evidence'] = compact_summary(retrieval['comparison']['evidence_summary']) if retrieval['comparison'] else None
        summary['all_history'] = compact_summary(broad['evidence_summary'])
        packet = {'series_id': case['series_id'], 'round': case['round'], 'origin': case['origin'],
                  'request_fingerprint': forecast_request_fingerprint(request), 'current_context': current,
                  'raw_history': raw, 'context_retrieval': summary}
        packets.append(packet)
        cv = min(providers, key=lambda p: case['current_card'][p]['cv_rmsle'])
        selected = retrieval['comparison']
        choice = selected['evidence_summary']['lifetime']['ranking'][0]['provider'] if selected else cv
        scores = {'current_cv': case['scores'][cv], 'retrieved_context': case['scores'][choice]}
        screens.append({'series_id': case['series_id'], 'round': case['round'], 'rmsle': scores,
                        'context_choice': choice, 'current_cv_choice': cv, 'selected_index': retrieval['selected_index'],
                        'matched_origins': selected['matched_origins'] if selected else 0})
        path = output/f'{case["round"]:04d}-{case["series_id"]}.evidence.json'
        path.write_text(json.dumps({'retrieval': retrieval, 'unfiltered_comparison': broad}, indent=2, allow_nan=False)+'\n')
    payload = {'scope': 'development only', 'information_contract': 'same_raw_matched_history_in_every_arm',
               'context_contract': 'origin_request_only_v1', 'source_ledger': str(ledger),
               'source_ledger_sha256': hashlib.sha256(ledger.read_bytes()).hexdigest(),
               'provider_calls': 0, 'confirmation_opened': False, 'packets': packets}
    (output/'memory.json').write_text(json.dumps(payload, sort_keys=True, allow_nan=False)+'\n')
    report = {'scope': 'development automatic policy screen; not a live-agent or held-out result',
              'mean_case_rmsle': {key: mean(row['rmsle'][key] for row in screens) for key in screens[0]['rmsle']},
              'cases': screens, 'confirmation_opened': False, 'target_established': False}
    (output/'screen.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(json.loads(args.cases.read_text()), args.ledger, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
