"""Undeployed planning-time index of exact historical backtest recipes.

Consumes trusted, recording-visible evidence; does not query, fit or select.
Counts describe recipe support, never a ranking across differing cohorts.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json

from .paired_consistency_098 import paired_consistency


def planning_recipes(review, full_evidence_bytes, current, canonicalize):
    """Propose up to three supported, untried recipes before a current backtest.

    The trusted caller captures current query, executed_configurations,
    checkpoint_available and budget together. Canonicalize must be the same
    validator used by all arms. Full evidence must already have passed the
    ledger's source/recording visibility checks; a digest alone proves neither.
    """
    paired_consistency(review, full_evidence_bytes)
    query = review['query']
    if (set(query) != {'series_id', 'unit', 'horizon', 'origin'}
            or current['query'] != query):
        raise ValueError('Exact current and historical task identity required')
    if type(current['checkpoint_available']) is not bool:
        raise ValueError('Explicit checkpoint availability required')
    budget = current['budget']
    remaining, cost, reserve = (budget[k] for k in
                                ('numerical_remaining', 'fresh_backtest_fits', 'reserved_final_fits'))
    if (any(type(n) is not int for n in (remaining, cost, reserve))
            or remaining < 0 or cost <= 0 or reserve < 0
            or budget['phase'] not in ('exploration', 'selection')):
        raise ValueError('Explicit nonnegative budget, positive batch cost and known phase required')

    def checked(cid, row):
        config = row['config']
        if canonicalize(deepcopy(config)) != config:
            raise ValueError('Complete common canonical configuration required')
        actual = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
        if cid != actual:
            raise ValueError('Configuration identity mismatch')
        for key in ('provider', 'revision'):
            if type(row.get(key)) is not str or not row[key]:
                raise ValueError('Explicit versioned provider identity required')

    catalog = review['configuration_index']
    for cid, row in catalog.items():
        checked(cid, row)
    executed = {}
    for row in current['executed_configurations']:
        cid = row['config_id']
        checked(cid, row)
        if cid in executed:
            raise ValueError('Duplicate current configuration')
        if cid in catalog and any(row[k] != catalog[cid][k] for k in ('config', 'provider', 'revision')):
            raise ValueError('Current and historical provider identities differ')
        executed[cid] = row

    full = json.loads(full_evidence_bytes)
    support = {cid: set() for cid in catalog}
    pointers = {cid: [] for cid in catalog}
    for index, card in enumerate(full['cards']):
        # The full artifact stores actual windows, even if the brief aliases them.
        origins = {datetime.fromisoformat(o['origin']) for o in card['windows']['lifetime']['origins']}
        for cid in (card['left_config_id'], card['right_config_id']):
            support[cid].update(origins)
            if origins:
                pointers[cid].append(f'/cards/{index}/windows/lifetime')
    eligible = sorted((cid for cid in catalog if cid not in executed and len(support[cid]) >= 4),
                      key=lambda cid: (-len(support[cid]), cid))
    blocked = ('checkpoint_required' if not current['checkpoint_available'] else
               'selection_phase' if budget['phase'] != 'exploration' else
               'insufficient_fits_with_final_reserve' if remaining < cost + reserve else None)
    capacity = max(0, (remaining-reserve)//cost) if blocked is None else 0
    recipes = []
    for cid in eligible[:3]:
        row = catalog[cid]
        dates = sorted(support[cid])
        recipes.append({'config_id': cid, 'provider': row['provider'], 'revision': row['revision'],
                        'historical_origins_with_paired_evidence': len(dates),
                        'earliest_origin': dates[0].isoformat(), 'latest_origin': dates[-1].isoformat(),
                        'evidence_pointers': pointers[cid],
                        'next_call': {'tool': 'lab', 'arguments': {'operation': 'backtest',
                                                                 'config': deepcopy(row['config'])},
                                      'valid_for_task': deepcopy(query), 'runnable': True,
                                      'admissible_now': blocked is None, 'blocked_reason': blocked,
                                      'required_fits': cost, 'final_fit_reserve': reserve}})
    return {'schema_version': 'planning-recipes-106', 'query': deepcopy(query),
            'recipes': recipes, 'eligible_recipes_on_page': len(eligible),
            'already_executed_catalog_recipes': len(set(executed) & set(catalog)),
            'insufficient_support_untried_recipes': sum(cid not in executed and len(support[cid]) < 4 for cid in catalog),
            'minimum_distinct_origins': 4, 'display_limit': 3,
            'order': 'distinct historical origin count descending, then config_id; not error-ranked',
            'sequential_capacity_at_snapshot': min(capacity, len(recipes)),
            'budget_at_render': deepcopy(budget),
            'budget_semantics': 'All proposals share this budget. Recheck live time, phase and remaining fits before each call.',
            'full_evidence': deepcopy(review['full_evidence']),
            'source_pagination': deepcopy(review['pagination']),
            'interpretation': 'Exact historical recipes to verify on current backtests, not selected forecasts. '
                              'Support is the union of visible paired origins for each recipe, not one shared comparison cohort. '
                              'Losses, ties and differing cohorts remain in referenced original evidence. '
                              'Unshown or unqueried evidence is unknown. Historical repetition does not establish future superiority.',
            'provider_calls': 0, 'forecast_selection_made': False}
