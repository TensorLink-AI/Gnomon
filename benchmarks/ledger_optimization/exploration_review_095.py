"""Undeployed ledger exploration aid; no fitting, selection, or new evidence.

Index one-setting differences between already catalogued configurations and
currently backtested configurations. All original review cards remain intact.
Only the upstream recording-visible review may supply historical identities.
"""
from copy import deepcopy
import hashlib
import json
import math

from .sparse_review_094 import WINDOWS, _support


def _window(windows, label):
    _support(windows, label)  # Reject cycles and inconsistent empty evidence.
    while 'same_as' in windows[label]:
        label = windows[label]['same_as']
    return windows[label]


def exploration_review(review, current, canonicalize):
    """Add unranked historical neighbors with task-bound current-backtest calls.

    current contains query, tested_configurations, and budget, captured together
    by a trusted caller for this task. canonicalize is the common lab's config
    validator, not a ledger-only model/configuration space. It must return the
    complete canonical configuration. This renderer does not establish upstream
    visibility or validate the actual saved ledger against its digest.
    """
    if review.get('schema_version') not in ('agent-review-087', 'agent-review-088'):
        raise ValueError('Unmodified 087/088 review required')
    if review.get('metric') != 'rmsle' or review.get('provider_calls') != 0:
        raise ValueError('Read-only RMSLE evidence required')
    query = review['query']
    if set(query) != {'series_id', 'unit', 'horizon', 'origin'} or current['query'] != query:
        raise ValueError('Current backtests and review must belong to exactly the same task')
    if type(query['horizon']) is not int or query['horizon'] <= 0:
        raise ValueError('Positive horizon required')
    evidence = review['full_evidence']
    digest = evidence.get('sha256', '')
    if (not evidence.get('path') or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('Full evidence path and SHA-256 required')
    page = review['pagination']
    if page['shown_pairs'] != len(review['cards']):
        raise ValueError('Page card count mismatch')

    def checked(cid, config):
        canonical = canonicalize(deepcopy(config))
        if canonical != config:
            raise ValueError('Complete canonical configurations required')
        expected = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]
        if cid != expected:
            raise ValueError('Configuration identity mismatch')
        return canonical

    catalog = review['configuration_index']
    for cid, row in catalog.items():
        checked(cid, row['config'])
        if not row.get('provider') or not row.get('revision'):
            raise ValueError('Versioned provider identity required')
    tested = {}
    for row in current['tested_configurations']:
        cid = row['config_id']
        if cid in tested:
            raise ValueError('Duplicate current configuration')
        tested[cid] = checked(cid, row['config'])
        if cid in catalog and tested[cid] != catalog[cid]['config']:
            raise ValueError('Current and historical configuration disagree')

    pairs = {}
    for card in review['cards']:
        ids = card['config_ids']
        if len(ids) != 2 or len(set(ids)) != 2 or any(c not in catalog for c in ids):
            raise ValueError('Pair must identify two catalog configurations')
        key = tuple(sorted(ids))
        if key in pairs or not card.get('evidence_pointer'):
            raise ValueError('Duplicate or unreferenced pair')
        if set(card['windows']) != set(WINDOWS):
            raise ValueError('All evidence windows required')
        for label in WINDOWS:
            w = _window(card['windows'], label)
            if w['n'] != w['matched_origins'] * query['horizon']:
                raise ValueError('Complete matched horizons required')
            if w['matched_origins'] and set(w['scores']) != set(ids):
                raise ValueError('Both scores required for supported pair')
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                   for v in w['scores'].values()):
                raise ValueError('Finite nonnegative scores required')
        pairs[key] = card

    budget = current['budget']
    remaining = budget['numerical_remaining']
    if type(remaining) is not int or remaining < 0:
        raise ValueError('Nonnegative numerical budget required')
    if budget['phase'] not in ('exploration', 'selection'):
        raise ValueError('Known lab phase required')
    admissible = budget['phase'] == 'exploration' and remaining >= 4
    reason = ('selection_phase' if budget['phase'] != 'exploration'
              else 'insufficient_fits_with_final_reserve' if remaining < 4 else None)
    neighbors = []
    for anchor, base in sorted(tested.items()):
        for cid, row in sorted(catalog.items()):
            config = row['config']
            if cid in tested or set(config) != set(base) or config['model'] != base['model']:
                continue
            changed = [k for k in config if config[k] != base[k]]
            if len(changed) != 1:
                continue
            field = changed[0]
            card = pairs.get(tuple(sorted((anchor, cid))))
            windows = {}
            if card:
                for label in WINDOWS:
                    w = _window(card['windows'], label)
                    windows[label] = {
                        'matched_origins': w['matched_origins'],
                        'comparison': ('unsupported' if not w['matched_origins'] else
                                       'lower_error' if w['scores'][cid] < w['scores'][anchor] else
                                       'higher_error' if w['scores'][cid] > w['scores'][anchor] else 'tie'),
                        'evidence_pointer': card['windows'][label].get('evidence_pointer', card['evidence_pointer']),
                    }
            neighbors.append({
                'anchor_config_id': anchor, 'candidate_config_id': cid,
                'parameter_change': {'field': field, 'from': base[field], 'to': config[field]},
                'evidence_status': 'pair_on_this_page' if card else 'pair_not_on_this_page',
                'windows': windows,
                'next_call': {'operation': 'backtest', 'arguments': {'config': deepcopy(config)},
                              'valid_for_task': deepcopy(query), 'runnable': True,
                              'admissible_now': admissible, 'blocked_reason': reason,
                              'required_fits': 3, 'final_fit_reserve': 1},
            })
    result = deepcopy(review)
    result['exploration'] = {
        'schema_version': 'historical-neighbors-095', 'neighbors': neighbors,
        'order': 'anchor_config_id, candidate_config_id; not score-ranked',
        'scope': 'Only catalogued one-setting neighbors of current backtests; not the full search space.',
        'interpretation': 'Historical association, not a causal parameter effect or forecast recommendation. '
                          'Missing page evidence is unknown, not a loss or zero error. Current backtests remain required.',
        'budget_at_render': deepcopy(budget),
        'budget_semantics': 'Admission is a snapshot; the lab must recheck live time, phase and budget before execution.',
        'provider_calls': 0, 'forecast_selection_made': False,
    }
    return result
