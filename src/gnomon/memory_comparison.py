"""Small verified memory cards; no model calls, writes, or prose interpretation."""
from itertools import combinations
from statistics import mean
import math

from .decision_memory import _fail
from .final_selection import forecast_request_fingerprint
from .ledger import _time


def check_comparison_claim(*, left, right, actuals, metric, relation):
    """Check arithmetic on supplied arrays, not their provenance or a causal claim.

    This is useful for hypothetical checks and explicit backtest claims. Arrays
    are caller-supplied, so a supported result does not attest to ledger evidence.
    """
    if metric not in ('mae', 'rmsle') or relation not in ('lower', 'equal', 'lower_or_equal'):
        _fail('metric/relation', 'Use mae/rmsle and lower/equal/lower_or_equal')
    arrays = [left, right, actuals]
    if any(not isinstance(v, (list, tuple)) or not 1 <= len(v) <= 10000 for v in arrays):
        _fail('arrays', 'Supply 1 to 10000 finite numbers per array')
    if len({len(v) for v in arrays}) != 1 or any(
            type(x) not in (int, float) or not math.isfinite(x) for v in arrays for x in v):
        _fail('arrays', 'Arrays must have equal length and finite numeric values')
    if metric == 'rmsle' and any(x < 0 for v in arrays for x in v):
        _fail('arrays', 'RMSLE requires nonnegative predictions and actuals; nothing is clipped')
    def score(pred):
        if metric == 'mae':
            return math.fsum(abs(p-a) for p, a in zip(pred, actuals)) / len(actuals)
        return math.hypot(*(math.log1p(p)-math.log1p(a) for p, a in zip(pred, actuals))) / math.sqrt(len(actuals))
    a, b = score(left), score(right)
    supported = {'lower': a < b, 'equal': a == b, 'lower_or_equal': a <= b}[relation]
    return {'status': 'supported' if supported else 'contradicted', 'metric': metric,
            'relation': relation, 'left_score': a, 'right_score': b, 'left_minus_right': a-b,
            'n': len(actuals), 'basis': 'arithmetic_on_caller_supplied_arrays',
            'source_verified': False, 'business_explanation_validated': False,
            'provider_calls': 0, 'ledger_writes': 0}


def comparison_card(bridge, *, execution_ids, source_as_of, recorded_as_of, metric='rmsle'):
    """Compare latest complete matched origin; retain exact versioned identities.

    Never pool changing custom-provider revisions. History is requested only for
    two or more distinct provider names with exactly one revision each.
    """
    if not isinstance(execution_ids, list) or len(execution_ids) > 100 or any(
            not isinstance(x, str) or not x for x in execution_ids):
        _fail('execution_ids', 'Supply at most 100 execution IDs')
    if metric not in ('mae', 'rmsle'):
        _fail('metric', 'Use mae or rmsle')
    source, recorded = _time(source_as_of), _time(recorded_as_of)
    groups, excluded = {}, []
    for eid in dict.fromkeys(execution_ids):
        ex = bridge.ledger.execution(eid)
        if ex['recorded_at'] > recorded:
            excluded.append({'execution_id': eid, 'reason': 'recorded_after_cutoff'})
            continue
        key = forecast_request_fingerprint(ex['request'])
        groups.setdefault(key, []).append(ex)
    ordered = sorted(groups.values(), key=lambda runs: runs[0]['request']['cutoff'] or '', reverse=True)
    for runs in ordered:
        if len(runs) < 2:
            excluded.append({'reason': 'single_execution_no_comparison', 'execution_id': runs[0]['execution_id']})
            continue
        if len(runs) > 8:
            excluded.append({'reason': 'more_than_eight_candidates_select_explicit_ids'})
            continue
        with bridge.ledger._connect() as conn:
            conn.execute('BEGIN')
            matched = bridge.ledger._compare(conn, runs, source, recorded)
            _, pairs = bridge.ledger._pairs(conn, runs[0]['request'], source, recorded)
            if metric == 'rmsle':
                for model, run in zip(matched['models'], runs):
                    checked = check_comparison_claim(left=[run['result']['point'][i] for i, _ in pairs],
                        right=[a['value'] for _, a in pairs], actuals=[a['value'] for _, a in pairs],
                        metric=metric, relation='lower_or_equal') if pairs else None
                    model[metric] = checked['left_score'] if checked else None
        if matched['n'] != matched['horizon']:
            excluded.append({'reason': 'incomplete_matching_actuals', 'origin': runs[0]['request']['cutoff'], 'n': matched['n']})
            continue
        models = matched['models']
        if any(m.get(metric) is None for m in models):
            continue
        ranking = [{**m, 'rank': 1+sum(other[metric] < m[metric] for other in models)}
                   for m in sorted(models, key=lambda v: v[metric])]
        differences = [{'left_execution_id': a['execution_id'], 'right_execution_id': b['execution_id'],
                        'left_minus_right': a[metric]-b[metric],
                        'relative_improvement': 1-a[metric]/b[metric] if b[metric] else None}
                       for a, b in combinations(models, 2)]
        q = runs[0]['request']
        identities = [(m['provider'], m['revision']) for m in models]
        historical = []
        if len(set(identities)) == len(identities):
            for group in ordered:
                request = group[0]['request']
                if any(request.get(k) != q.get(k) for k in ('series_id', 'unit', 'horizon')):
                    continue
                chosen = []
                for identity in identities:
                    matches = [r for r in group if (r['provider'], r['revision']) == identity]
                    if len(matches) != 1:
                        break
                    chosen.append(matches[0])
                if len(chosen) != len(identities):
                    continue
                with bridge.ledger._connect() as conn:
                    conn.execute('BEGIN')
                    comparison = bridge.ledger._compare(conn, chosen, source, recorded)
                    _, pairs = bridge.ledger._pairs(conn, request, source, recorded)
                if comparison['n'] != request['horizon']:
                    continue
                values = [check_comparison_claim(left=[r['result']['point'][i] for i, _ in pairs],
                    right=[a['value'] for _, a in pairs], actuals=[a['value'] for _, a in pairs],
                    metric=metric, relation='lower_or_equal')['left_score'] for r in chosen]
                historical.append({'origin': request['cutoff'], 'scores': values, 'n': len(pairs)})
        def window(rows):
            scores = [mean(row['scores'][i] for row in rows) for i in range(len(identities))] if rows else []
            return {'matched_origins': len(rows), 'n': sum(row['n'] for row in rows),
                'start': min((row['origin'] for row in rows), default=None),
                'end': max((row['origin'] for row in rows), default=None),
                'ranking': sorted([{'provider': ident[0], 'revision': ident[1], 'score': score,
                    'rank': 1+sum(v < score for v in scores)} for ident, score in zip(identities, scores)],
                    key=lambda m: m['score'])}
        historical.sort(key=lambda row: row['origin'])
        history = {'lifetime': window(historical), 'recent': window(historical[-3:])}
        history['recent_lifetime_disagreement'] = [(m['provider'], m['revision'], m['rank']) for m in history['lifetime']['ranking']] != [
            (m['provider'], m['revision'], m['rank']) for m in history['recent']['ranking']]
        return {'status': 'matched', 'metric': metric, 'origin': q['cutoff'], 'matched_origins': 1,
                'n': matched['n'], 'ranking': ranking, 'differences': differences,
                'actual_ids': matched['actual_ids'], 'history_summary': history,
                'history_status': 'matched' if historical else 'insufficient_exact_revision_matches',
                'source_as_of': source, 'recorded_as_of': recorded, 'excluded': excluded,
                'verification_call': {'execution_ids': [r['execution_id'] for r in runs],
                    'source_as_of': source, 'recorded_as_of': recorded, 'metric': metric},
                'uncertainty': 'Descriptive matched evidence; one origin does not establish superiority. Changing revisions are not pooled.',
                'provider_calls': 0, 'ledger_writes': 0}
    return {'status': 'insufficient_comparative_evidence', 'metric': metric, 'matched_origins': 0,
            'ranking': [], 'excluded': excluded, 'provider_calls': 0, 'ledger_writes': 0}
