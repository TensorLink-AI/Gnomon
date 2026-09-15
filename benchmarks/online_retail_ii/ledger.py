"""Reconstruct explicitly simulated ledger evidence from already matured cases.

The clock is a benchmark replay clock. These are not genuine 2010 production
recordings. No model is refitted and no current/future target is read.
"""
from datetime import date, datetime, time, timedelta
import importlib.metadata
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .agent import load_case
from .data import dump
from .models import CANDIDATES


def build_history(case, output):
    from gnomon import ForecastRequest, ForecastResult, InferenceEngine, TemporalLedger
    if importlib.metadata.version('gnomon-forecast') != '1.2.0':
        raise ValueError('Pinned Gnomon 1.2.0 is required')
    task, history = load_case(case)
    supplied = json.loads((Path(case)/'matured-outcomes.json').read_text())
    if supplied['as_of'] != task['origin']:
        raise ValueError('Historical query cutoff differs from task')
    output = Path(output)
    if output.exists():
        raise ValueError('Replay must use a fresh ledger; never rewrite production recording times')
    output.mkdir(parents=True)
    zone = ZoneInfo('Europe/London')
    def stamp(day, end=False):
        return datetime.combine(date.fromisoformat(day), time(23, 59, 59) if end else time(), zone)
    clock = [stamp(task['origin'], True)]
    class ReplayClock:
        def now(self):
            return clock[0].isoformat()
    ledger = TemporalLedger(output/'evidence.db', clock=ReplayClock())
    revisions = {}; count = 0
    previous_end = None
    observed = dict(zip(history.timestamp.dt.strftime('%Y-%m-%d'), history.value, strict=True))
    for past in supplied['records']:
        if past['origin'] >= task['origin'] or past['target_end'] > task['origin'] or any(d > task['origin'] for d in past['future_timestamps']):
            raise ValueError('Historical evidence has not matured by this task origin')
        expected_dates = [(date.fromisoformat(past['origin'])+timedelta(days=i)).isoformat() for i in range(1, 15)]
        if (past['future_timestamps'] != expected_dates or past['target_end'] != expected_dates[-1]
                or (previous_end is not None and past['origin'] < previous_end)
                or past['actual'] != [observed[d] for d in expected_dates]):
            raise ValueError('Historical actuals, dates or ordering differ from visible observations')
        previous_end = past['target_end']
        visible = history[history.timestamp.dt.date <= date.fromisoformat(past['origin'])]
        clock[0] = stamp(past['origin'], True)
        engine = InferenceEngine(ledger=ledger, cache_size=0)
        for provider in CANDIDATES:
            points = tuple(past['predictions'][provider])
            def replay(request, points=points):
                return ForecastResult(point=points, timestamps=request.future_timestamps,
                    series_id=request.series_id, unit=request.unit,
                    metadata={'evidence_kind': 'replayed_past_only_baseline_execution'})
            revision = 'online-retail-ii-v1/'+past['models_sha256']
            if provider in revisions and revisions[provider] != revision:
                raise ValueError('Provider revision changed across the requested history')
            revisions[provider] = revision
            engine.register(provider, replay, revision=revision, deterministic=True)
            req = ForecastRequest(history=tuple(map(float, visible.value)), horizon=14, season=7,
                frequency='D', series_id=task['series_id'], unit='units',
                timestamps=tuple(stamp(d.strftime('%Y-%m-%d')).isoformat() for d in visible.timestamp),
                future_timestamps=tuple(stamp(d).isoformat() for d in past['future_timestamps']),
                cutoff=clock[0].isoformat(), known_time_cutoff=clock[0].isoformat(), recorded_time_cutoff=clock[0].isoformat())
            engine.forecast(provider, req); count += 1
        for day, value in zip(past['future_timestamps'], past['actual'], strict=True):
            clock[0] = stamp(day, True)
            ledger.append_actual(series_id=task['series_id'], valid_time=stamp(day).isoformat(), value=value,
                source_available_at=clock[0].isoformat(), unit='units', source_ref='online-retail-ii:assumed-day-close')
    as_of = stamp(task['origin'], True).isoformat()
    if not supplied['records']:
        report = {'status': 'insufficient_evidence', 'matched_origins': 0, 'ranking': []}
    else:
        # The public API supports at most eight providers. Pair against one
        # common anchor, then require the SAME origins and actual IDs across
        # every pair before computing a ten-provider table.
        anchor = CANDIDATES[0]
        comparisons = [ledger.compare_history(series_id=task['series_id'], horizon=14,
            providers={p: revisions[p] for p in (anchor, provider)},
            start=stamp(supplied['records'][0]['origin'], True).isoformat(), end=as_of,
            source_as_of=as_of, recorded_as_of=as_of, unit='units') for provider in CANDIDATES[1:]]
        if any(c['status'] != 'ok' for c in comparisons):
            dump(output/'gnomon-comparisons.json', comparisons)
            raise ValueError('Gnomon did not admit all historical comparison pairs; inspect retained diagnostics')
        by_origin = [{o['origin']: o for o in c['origins']} for c in comparisons]
        common = set.intersection(*(set(c) for c in by_origin))
        merged = []
        for day in sorted(common):
            base = by_origin[0][day]; models = {}
            for group in by_origin:
                if group[day]['actual_ids'] != base['actual_ids'] or group[day]['matched_steps'] != base['matched_steps']:
                    raise ValueError('Comparison pairs do not share actual vintages and steps')
                for model in group[day]['models']:
                    if model['provider'] in models and model != models[model['provider']]:
                        raise ValueError('Shared comparison anchor changed')
                    models[model['provider']] = model
            merged.append({**base, 'models': list(models.values())})
        comparison = {'status': 'ok' if common else 'insufficient_evidence', 'origins': merged,
            'matched_origins': len(common), 'excluded': [e for c in comparisons for e in c['excluded']],
            'origins_excluded_by_pair_intersection': sorted(set.union(*(set(c) for c in by_origin))-common),
            'pair_query_count': len(comparisons), 'cohort_rule': 'intersection_of_all_pair_origins_and_exact_actual_ids'}
        dump(output/'gnomon-comparisons.json', comparisons)
        actuals = {r['actual_id']: r for r in ledger.actuals_as_of(task['series_id'], source_as_of=as_of,
                                                               recorded_as_of=as_of, unit='units')}
        scores = {p: [] for p in CANDIDATES}; references = []
        for matched in comparison['origins']:
            actual = np.asarray([actuals[a]['value'] for a in matched['actual_ids']])
            for model in matched['models']:
                execution = ledger.execution(model['execution_id'])
                point = np.asarray([execution['result']['point'][i] for i in matched['matched_steps']])
                rmsle = float(np.sqrt(np.mean((np.log1p(point)-np.log1p(actual))**2)))
                scores[model['provider']].append(rmsle)
                references.append({'execution_id': model['execution_id'], 'actual_ids': matched['actual_ids'],
                                   'provider': model['provider'], 'origin': matched['origin'], 'rmsle': rmsle})
        ranking = sorted([{'provider': p, 'origins': len(v), 'mean_rmsle': float(np.mean(v)),
                            'recent4_mean_rmsle': float(np.mean(v[-4:]))} for p, v in scores.items() if v],
                         key=lambda x: (x['mean_rmsle'], CANDIDATES.index(x['provider'])))
        for item in ranking:
            item['rank'] = 1+sum(r['mean_rmsle'] < item['mean_rmsle'] for r in ranking)
        report = {'status': comparison['status'], 'matched_origins': comparison['matched_origins'],
                  'ranking': ranking, 'references': references, 'excluded': comparison['excluded']}
        dump(output/'gnomon-comparison.json', comparison)
    report.update({'metric': 'mean_per_origin_rmsle', 'source': 'public_gnomon_matched_execution_and_actual_reads',
        'query_as_of': as_of, 'recording_semantics': 'simulated historical replay; not genuine recording timestamps',
        'gnomon_native_compare_history_metric': 'mae', 'rmsle_calculated_from_referenced_pairs': True,
        'numerical_refits': 0, 'replayed_executions': count, 'engy_calls': 0})
    dump(output/'report.json', report)
    return report
