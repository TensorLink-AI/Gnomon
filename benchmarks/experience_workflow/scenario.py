"""Deterministic worlds and a plain-Python scoring oracle, with no Gnomon imports."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import random
from statistics import mean

# Stable canonical JSON key order is also the declared exact-tie order.
PROVIDERS = ('historical_mean', 'last_value', 'seasonal_naive')
FAMILIES = ('clean', 'delayed', 'revisions', 'identity')
DEVELOPMENT = set(range(100, 108)) | set(range(200, 204))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def stamp(value):
    return value.astimezone(timezone.utc).isoformat(timespec='microseconds')


def shifted(value, days):
    return stamp(datetime.fromisoformat(value) + timedelta(days=days))


def predictions(history):
    return dict(last_value=[history[-1]] * 2,
                historical_mean=[mean(history)] * 2,
                seasonal_naive=history[-7:-5])


def generate(seed, rounds=24):
    if seed not in DEVELOPMENT:
        raise ValueError('Only preregistered development/validation seeds are enabled')
    if not 4 <= rounds <= 24:
        raise ValueError('Use 4..24 rounds; shorter runs are development probes')
    rng = random.Random(seed)
    family = FAMILIES[seed % 4]
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events, tasks, truth = [], [], {}
    # Generate demand without consulting candidate scores or treatment identity.
    series = {}
    for s in range(3):
        level, slope, amplitude = rng.uniform(15, 60), rng.uniform(-.08, .15), rng.uniform(0, 18)
        series[f'series-{s}'] = [max(0., round(level + slope * d + amplitude * math.sin(2 * math.pi * d / 7)
            + (10 if (d // 8 + s) % 2 else 0) + rng.gauss(0, 3), 4)) for d in range(4 * rounds + 40)]
    for r in range(rounds):
        day = 28 + r * 4
        origin = stamp(base + timedelta(days=day))
        revision = 'synthetic-recipes/2' if family == 'identity' and r >= 12 else 'synthetic-recipes/1'
        for s, values in series.items():
            context = {'promotion': 'planned' if (day // 8 + int(s[-1])) % 2 else 'none'}
            history = values[day - 27:day + 1]
            request = dict(history=history, horizon=2, season=7, series_id=s, unit='widgets',
                cutoff=origin, timestamps=[stamp(base + timedelta(days=d)) for d in range(day - 27, day + 1)],
                future_timestamps=[shifted(origin, i) for i in (1, 2)], frequency='D')
            for p, point in predictions(history).items():
                events.append(dict(kind='forecast', event_id=f'{seed}/{r}/{s}/{p}', recorded_at=origin,
                    provider=p, revision=revision, request=request, point=point, context=context))
            for step, target in enumerate(request['future_timestamps'], 1):
                final = values[day + step]
                delay = rng.randrange(5) if family != 'clean' else 0
                source = shifted(target, delay)
                recorded = shifted(source, rng.randrange(4) if family != 'clean' else 0)
                revised = family in ('revisions', 'identity') and (r + step) % 3 == 0
                common = dict(kind='actual', series_id=s, unit='widgets', valid_time=target)
                events.append(dict(common, event_id=f'{seed}/{r}/{s}/{step}/first', recorded_at=recorded,
                    source_available_at=source, value=max(0, final - 8) if revised else final))
                if revised:
                    events.append(dict(common, event_id=f'{seed}/{r}/{s}/{step}/revision',
                        recorded_at=shifted(recorded, 12), source_available_at=shifted(source, 10), value=final))
                if family == 'identity':
                    events.append(dict(common, unit='kg', event_id=f'{seed}/{r}/{s}/{step}/kg',
                        recorded_at=shifted(target, 1), source_available_at=target, value=final * 100))
                    # An early-recorded source-future revision tests both clocks independently.
                    events.append(dict(common, event_id=f'{seed}/{r}/{s}/{step}/future-source',
                        recorded_at=shifted(target, 1), source_available_at=shifted(target, 9), value=final + 4))
            truth[r, s] = values[day + 1:day + 3]
        s = f'series-{r % 3}'
        current = next(e for e in reversed(events) if e['kind'] == 'forecast' and e['request']['series_id'] == s)
        queries = {}
        for label, lag in [('original', 8), ('current', 0)]:
            cutoff = shifted(origin, -lag)
            queries[label] = dict(series_id=s, horizon=2, unit='widgets',
                providers={p: revision for p in PROVIDERS}, start=stamp(base),
                end=shifted(origin, -8), source_as_of=cutoff, recorded_as_of=cutoff,
                context_filters=current['context'], metric='rmsle', recent_origins=4)
        tasks.append(dict(task_id=f'{seed}/{r}', round=r, now=origin, queries=queries,
            request=current['request'], family=family))
    events.sort(key=lambda e: (e['recorded_at'], e['event_id']))
    # Arrival order is explicit; later source availability dominates arrival order.
    for seq, event in enumerate(events):
        event['sequence'] = seq
    return dict(seed=seed, family=family, rounds=rounds, events=events, tasks=tasks,
                truth={f'{r}/{s}': v for (r, s), v in truth.items()})


def oracle(events, query, *, mutation=None):
    """Independent joins/visibility/metrics. Never calls a Gnomon scoring method."""
    q = query
    visible = [e for e in events if mutation == 'ignore_recording' or e['recorded_at'] <= q['recorded_as_of']]
    forecasts = {}
    for e in visible:
        if e['kind'] != 'forecast':
            continue
        req = e['request']
        if req['series_id'] != q['series_id'] or req['unit'] != q['unit'] or req['horizon'] != q['horizon']:
            continue
        if not q['start'] <= req['cutoff'] <= q['end'] or e['revision'] != q['providers'].get(e['provider']):
            continue
        if mutation != 'ignore_context' and any(e['context'].get(k) != v for k, v in q['context_filters'].items()):
            continue
        forecasts.setdefault(req['cutoff'], {}).setdefault(e['provider'], e)
    actuals = {}
    for e in visible:
        if e['kind'] != 'actual' or e['series_id'] != q['series_id']:
            continue
        if mutation != 'ignore_unit' and e['unit'] != q['unit']:
            continue
        if mutation != 'ignore_source' and e['source_available_at'] > q['source_as_of']:
            continue
        key = e['valid_time']
        old = actuals.get(key)
        if old is None or (e['source_available_at'], e['sequence']) > (old['source_available_at'], old['sequence']):
            if mutation != 'ignore_revisions' or old is None:
                actuals[key] = e
    rows = []
    for origin, runs in sorted(forecasts.items()):
        if set(runs) != set(q['providers']):
            continue
        timestamps = next(iter(runs.values()))['request']['future_timestamps']
        if not all(t in actuals for t in timestamps):
            continue
        scores = {p: math.sqrt(mean((math.log1p(pred) - math.log1p(actuals[t]['value'])) ** 2
                  for pred, t in zip(run['point'], timestamps))) for p, run in runs.items()}
        rows.append(dict(origin=origin, scores=scores, actual_refs=[actuals[t]['event_id'] for t in timestamps]))
    if mutation == 'no_memory':
        rows = []
    scores = {p: mean(row['scores'][p] for row in rows) for p in q['providers']} if rows else {}
    return dict(matched_origins=len(rows), scores=scores,
                ranking=sorted(scores, key=scores.get), origins=rows)


def public_answer(answer):
    return {k: answer[k] for k in ('matched_origins', 'scores', 'ranking')}


def expected_provider(answer):
    return answer['ranking'][0] if answer['matched_origins'] >= 3 else 'last_value'


def matches(actual, expected):
    if not isinstance(actual, dict) or type(actual.get('matched_origins')) is not int or actual.get('matched_origins') != expected['matched_origins']:
        return False
    if not isinstance(actual.get('scores'), dict) or actual.get('ranking') != expected['ranking'] or set(actual['scores']) != set(expected['scores']):
        return False
    return all(type(actual['scores'][p]) in (float, int) and math.isfinite(actual['scores'][p])
               and math.isclose(actual['scores'][p], value, rel_tol=1e-9, abs_tol=1e-9)
               for p, value in expected['scores'].items())
