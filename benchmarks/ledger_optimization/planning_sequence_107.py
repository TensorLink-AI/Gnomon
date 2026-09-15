"""Undeployed explicit prerequisite plan; no projected state is an execution."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json

from .planning_recipes_106 import planning_recipes


def planning_sequence(review, full_evidence_bytes, current, canonicalize):
    source = planning_recipes(review, full_evidence_bytes, current, canonicalize)
    checkpoint = current['checkpoint_available']
    result = {k: deepcopy(source[k]) for k in
              ('query', 'full_evidence', 'source_pagination', 'minimum_distinct_origins', 'display_limit', 'order')}
    result.update(schema_version='planning-sequence-107', observed_state=deepcopy(current),
                  provider_calls=0, numerical_attempts=0, forecast_selection_made=False,
                  state_projected_as_executed=False)
    if checkpoint:
        recipes = deepcopy(source['recipes'])
        result.update(prerequisite=None, recipes=recipes,
                      next_call=deepcopy(recipes[0]['next_call']) if recipes and recipes[0]['next_call']['admissible_now'] else None,
                      eligible_recipes_on_page=source['eligible_recipes_on_page'],
                      sequential_capacity_upper_bound=source['sequential_capacity_at_snapshot'],
                      checkpoint_max_numerical_attempts=0,
                      interpretation='Checkpoint already exists. Direct calls retain 106 admission and evidence semantics.')
        return result

    prerequisite = current['initial_checkpoint']
    initial_config = prerequisite['config']
    if (initial_config != canonicalize({'model': 'seasonal', 'season': 7})
            or type(prerequisite['max_numerical_attempts']) is not int
            or prerequisite['max_numerical_attempts'] != 4):
        raise ValueError('Exact common seasonal-7 checkpoint and four-attempt upper bound required')
    initial_id = hashlib.sha256(json.dumps(initial_config, sort_keys=True).encode()).hexdigest()[:16]
    budget = current['budget']
    requests, exploratory = (budget[k] for k in ('agent_requests_remaining', 'exploration_requests_remaining'))
    if any(type(n) is not int or n < 0 for n in (requests, exploratory)):
        raise ValueError('Explicit nonnegative remaining interaction budgets required')
    remaining = budget['numerical_remaining']; cost = budget['fresh_backtest_fits']; reserve = budget['reserved_final_fits']
    start_cost = prerequisite['max_numerical_attempts']
    reason = ('selection_phase' if budget['phase'] != 'exploration' else
              'insufficient_interaction_budget' if requests < 3 or exploratory < 2 else
              'insufficient_sequence_numerical_budget' if remaining < start_cost+cost+reserve else None)
    # Subtract the prerequisite once; every displayed recipe shares these limits.
    capacity = (min(max(0, (remaining-start_cost-reserve)//cost), exploratory-1, requests-2)
                if reason is None else 0)
    full = json.loads(full_evidence_bytes); catalog = review['configuration_index']
    supports, pointers = {cid: set() for cid in catalog}, {cid: [] for cid in catalog}
    for index, card in enumerate(full['cards']):
        dates = {datetime.fromisoformat(o['origin']) for o in card['windows']['lifetime']['origins']}
        for cid in (card['left_config_id'], card['right_config_id']):
            supports[cid].update(dates)
            if dates: pointers[cid].append(f'/cards/{index}/windows/lifetime')
    unavailable = {initial_id, *(c['config_id'] for c in current['executed_configurations'])}
    eligible = sorted((cid for cid in catalog if cid not in unavailable and len(supports[cid]) >= 4),
                      key=lambda cid: (-len(supports[cid]), cid))
    recipes = []
    for cid in eligible[:3]:
        dates = sorted(supports[cid]); row = catalog[cid]
        recipes.append({'config_id': cid, 'provider': row['provider'], 'revision': row['revision'],
                        'historical_origins_with_paired_evidence': len(dates),
                        'earliest_origin': dates[0].isoformat(), 'latest_origin': dates[-1].isoformat(),
                        'evidence_pointers': pointers[cid],
                        'next_call': {'tool': 'lab', 'arguments': {'operation': 'backtest', 'config': deepcopy(row['config'])},
                                      'valid_for_task': deepcopy(source['query']), 'runnable': True,
                                      'admissible_now': False, 'blocked_reason': 'checkpoint_required',
                                      'conditional_budget_feasible': capacity > 0,
                                      'sequence_blocked_reason': reason, 'required_fits': cost,
                                      'requires': ['successful_task_matching_checkpoint', 'fresh_time_phase_and_budget_admission'],
                                      'final_fit_reserve': reserve}})
    first = {'tool': 'lab', 'arguments': {'operation': 'start'},
             'valid_for_task': deepcopy(source['query']), 'runnable': True,
             'admissible_now': bool(recipes) and capacity > 0,
             'max_numerical_attempts': start_cost,
             'blocked_reason': reason or ('no_supported_untried_recipe' if not recipes else None)}
    result.update(prerequisite=first, next_call=deepcopy(first) if first['admissible_now'] else None,
                  recipes=recipes, eligible_recipes_on_page=len(eligible),
                  initial_checkpoint_config_id=initial_id, checkpoint_max_numerical_attempts=start_cost,
                  sequential_capacity_upper_bound=min(capacity, len(recipes)),
                  interpretation='Run the common start prerequisite before any conditional backtest. '
                                 'No checkpoint or forecast has been created by this plan. '
                                 'The initial four-attempt bound is charged once, plus each fresh recipe and final reserve. '
                                 'Bounds are not observed costs or a guarantee of future admission. '
                                 'Historical support counts do not rank forecast accuracy; inspect the referenced losses and ties. '
                                 'Finish with explicit selection after current backtesting; no forecast is automatically selected.')
    return result
