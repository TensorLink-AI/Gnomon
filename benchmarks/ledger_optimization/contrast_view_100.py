"""Undeployed compact views with an exact, content-addressed evidence artifact.

The caller must store returned evidence_bytes at evidence.path before exposing
the view. This module performs no file writes, retrieval, model or agent calls.
"""
from copy import deepcopy
import hashlib
import json

from .current_cv_pair_100 import current_cv_pair
from .current_history_contrast_100 import current_history_contrast


def contrast_view(current, *, review=None, full_evidence_bytes=None):
    """Return (view, evidence_bytes); historical input is optional for all arms.

    The current_cv field is identical with and without a ledger review. An absent
    review is not treated as evidence that no historical matches exist.
    """
    if (review is None) != (full_evidence_bytes is None):
        raise ValueError('Historical review and its authenticated bytes must be supplied together')
    common = current_cv_pair(current)
    cv = common['current_cv']
    detailed = {'current': deepcopy(current), 'current_summary': common}
    view = {'schema_version': 'contrast-view-100', 'query': deepcopy(common['query']),
            'metric': 'rmsle', 'current_cv': {
                'matched_folds': cv['matched_folds'], 'predictions_per_config': cv['n'],
                'models': [{'config_id': cid, 'mean_rmsle': cv['scores'][cid],
                            'rank': 1 + sum(score < cv['scores'][cid] for score in cv['scores'].values()),
                            'lower_error_folds': cv['fold_wins'][cid]}
                           for cid in common['config_ids']],
                'tied_folds': cv['fold_ties']},
            'provider_calls': 0, 'forecast_selection_made': False}
    if review is not None:
        contrast = current_history_contrast(review, full_evidence_bytes, current)
        detailed['historical_contrast'] = contrast
        view['history_status'] = contrast['history_status']
        view['history'] = []
        groups = {}
        for label, window in contrast['history'].items():
            if 'same_as' in window:
                continue
            groups[label] = {
                'windows': [label], 'matched_origins': window['matched_origins'],
                'predictions_per_config': window['n'], 'mean_rmsle': deepcopy(window['scores']),
                'latest_origin': window['latest_matched_origin'],
                'days_since_latest_origin': window['days_since_latest_matched_origin'],
                'cv_winner_disagrees': window['current_cv_and_history_winners_disagree']}
        # The original validator permits aliases; resolve them without discarding
        # any window or treating repeated windows as additional observations.
        for label, window in contrast['history'].items():
            if 'same_as' not in window: continue
            target = window['same_as']
            while 'same_as' in contrast['history'][target]:
                target = contrast['history'][target]['same_as']
            groups[target]['windows'].append(label)
        view['history'] = list(groups.values())
        view['interpretation'] = 'Compare models within each cohort; differing CV/history winners do not dictate selection. One historical origin does not establish repeatability.'
    else:
        view['history_status'] = 'not_requested'
    raw = json.dumps(detailed, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    digest = hashlib.sha256(raw).hexdigest()
    path = f'comparison-evidence/{digest}.json'
    view['evidence'] = {'path': path, 'sha256': digest, 'bytes': len(raw)}
    return view, raw
