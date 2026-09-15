"""Fixed offline selection diagnostics; no forecasts, target access or deployment.

The caller must authenticate scored-pair evidence, recording visibility and task
identity before supplying these aggregates. This function only checks temporal
order and numerical shape, then uses one common historical cohort per rule.
"""
from datetime import datetime
import math
from statistics import mean


def _instant(value):
    at = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if at.utcoffset() is None:
        raise ValueError('Explicit timezone required')
    return at


def _score(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative RMSLE required')
    return value


def selections(cv_scores, history, *, origin):
    """Return CV and fixed last-4/last-12/lifetime matched-history selections.

    cv_scores maps config ID to its current three-fold mean. history maps an
    earlier visible production origin to config-ID/RMSLE pairs, including other
    configurations when defining global recent windows. New-task actuals and
    forecasts are deliberately absent from this interface. Every current config
    must share at least two historical origins; otherwise use the CV minimum.
    """
    if not cv_scores or any(not isinstance(cid, str) or not cid for cid in cv_scores):
        raise ValueError('Nonempty identified current candidates required')
    current = {cid: _score(value) for cid, value in cv_scores.items()}
    at = _instant(origin)
    ordered = []
    seen = set()
    for timestamp, scores in history.items():
        previous = _instant(timestamp)
        if previous >= at or previous in seen:
            raise ValueError('Distinct earlier production origins required')
        seen.add(previous)
        if not scores or any(not isinstance(cid, str) or not cid for cid in scores):
            raise ValueError('Identified historical scores required')
        ordered.append((previous, timestamp, {cid: _score(v) for cid, v in scores.items()}))
    ordered.sort()
    default = min(current, key=lambda cid: (current[cid], cid))
    result = {'current_cv': {'config_id': default, 'basis': 'current_cv',
                             'scores': current, 'matched_origins': [],
                             'fallback_reason': None}}
    for name, size in (('last_4', 4), ('last_12', 12), ('lifetime', None)):
        window = ordered[-size:] if size is not None else ordered
        shared = [(timestamp, values) for _, timestamp, values in window
                  if set(current).issubset(values)]
        if len(shared) < 2:
            result[name] = {'config_id': default, 'basis': 'current_cv_fallback',
                            'scores': {}, 'matched_origins': [t for t, _ in shared],
                            'fallback_reason': 'fewer_than_two_all_candidate_matched_origins'}
        else:
            scores = {cid: mean(values[cid] for _, values in shared) for cid in current}
            selected = min(scores, key=lambda cid: (scores[cid], cid))
            result[name] = {'config_id': selected, 'basis': 'matched_production_history',
                            'scores': scores, 'matched_origins': [t for t, _ in shared],
                            'fallback_reason': None}
    return result
