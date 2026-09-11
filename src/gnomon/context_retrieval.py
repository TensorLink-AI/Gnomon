"""Retrieve progressively broader context cohorts without choosing by outcomes."""
from collections import Counter
from copy import deepcopy

from .decision_memory import validate_filters
from .forecast_adapter import ForecastAdapterError
from .ledger_history import compare_history


def retrieve_context(ledger, *, context_candidates, min_origins=4, **query):
    """Select the first caller-ordered cohort with enough complete matched origins.

    Each subsequent context must drop filters only; {} explicitly allows an
    unfiltered fallback. The threshold is a count rule, not statistical proof.
    All queries use one database read snapshot and the same evidence cutoffs.
    """
    if type(min_origins) is not int or not 1 <= min_origins <= 1000:
        raise ForecastAdapterError('min_origins must be an integer from 1 to 1000',
                                   details={'rejected_fields': ['min_origins']})
    if not isinstance(context_candidates, list) or not 1 <= len(context_candidates) <= 8:
        raise ForecastAdapterError('Supply 1 to 8 context candidates, most specific first',
                                   details={'rejected_fields': ['context_candidates']})
    candidates = []
    for filters in context_candidates:
        normalized = {} if isinstance(filters, dict) and not filters else validate_filters(filters)
        if candidates and (not set(normalized.items()) < set(candidates[-1].items())):
            raise ForecastAdapterError('Each later context must remove filters without changing their values; duplicates are not allowed',
                                       details={'rejected_fields': ['context_candidates']})
        candidates.append(normalized)
    query.setdefault('unit', None)
    comparisons = []
    with ledger._connect() as conn:
        conn.execute('BEGIN')
        for filters in candidates:
            comparisons.append(compare_history(ledger, context_filters=filters or None,
                                                _connection=conn, **query))
    selected = next((i for i, comparison in enumerate(comparisons)
                     if comparison['status'] == 'ok' and comparison['matched_origins'] >= min_origins), None)
    cohorts = []
    for index, (filters, comparison) in enumerate(zip(candidates, comparisons, strict=True)):
        cohorts.append({'index': index, 'context_filters': deepcopy(filters),
                        'status': comparison['status'], 'matched_origins': comparison['matched_origins'],
                        'meets_count_threshold': comparison['status'] == 'ok' and comparison['matched_origins'] >= min_origins,
                        'exclusion_counts': dict(Counter(row['reason'] for row in comparison['excluded'])),
                        'evidence_summary': comparison.get('evidence_summary')})
    return {'status': 'ok' if selected is not None else 'insufficient_evidence',
            'selection_rule': 'first_caller_ordered_cohort_meeting_count_threshold_not_best_observed_score',
            'min_origins': min_origins, 'selected_index': selected,
            'selected_filters': deepcopy(candidates[selected]) if selected is not None else None,
            'broadened': selected is not None and selected > 0,
            'context_specific': selected is not None and bool(candidates[selected]),
            'cohorts': cohorts, 'comparison': comparisons[selected] if selected is not None else None,
            'source_as_of': comparisons[0]['source_as_of'], 'recorded_as_of': comparisons[0]['recorded_as_of'],
            'provider_calls': 0, 'ledger_writes': 0,
            'read_consistency': 'single_sqlite_read_snapshot',
            'next_step': 'assess_relevance_and_sample_size_before_selecting_a_provider' if selected is not None
                         else 'use_current_evidence_or_collect_more_matched_outcomes',
            'limitation': 'Count eligibility is not confidence or forecast superiority. '
                          'Broader history may be less relevant; labels are caller assertions available at forecast time. '
                          'This operation retrieves evidence and does not select or execute a provider.'}
