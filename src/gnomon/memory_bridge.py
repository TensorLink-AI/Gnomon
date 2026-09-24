"""Execution-bound evidence for caller-owned memory systems; no model/network calls.

The ledger is authoritative. Narrative remains unverified data. Exports are not
signatures, and callers must not treat agent-supplied packets as ledger evidence.
"""
from copy import deepcopy
import hashlib
import json
import math

from .decision_memory import LESSON_KIND, SUMMARY_KIND, _fail, _text, validate_filters
from .final_selection import forecast_request_fingerprint
from .ledger import _json, _time


def _hash(value):
    return 'sha256:' + hashlib.sha256(_json(value).encode()).hexdigest()


def _metrics(review):
    result = deepcopy(review['metrics'])
    pairs = review['scored_pairs']
    if not pairs:
        result.update(rmsle=None, rmsle_status='no_scored_pairs')
    elif any(p['prediction'] < 0 or p['actual'] < 0 for p in pairs):
        result.update(rmsle=None, rmsle_status='negative_values_not_supported')
    else:
        result.update(rmsle=math.hypot(*(math.log1p(p['prediction']) - math.log1p(p['actual'])
                                      for p in pairs)) / math.sqrt(len(pairs)), rmsle_status='scored')
    return result


class EvidenceMemory:
    """Portable decisions, immutable lessons and bounded, exact-filter retrieval.

    ``ledger_ref`` is an application-owned identifier, not a filesystem path or
    authorization credential. Writes are explicit Python application actions.
    This bridge reuses TemporalLedger's decision/lesson APIs and schema.
    """

    def __init__(self, ledger, *, ledger_ref):
        self.ledger = ledger
        self.ledger_ref = _text(ledger_ref, 'ledger_ref', 128)

    def record_decision(self, *, execution_id, expected_request, rationale,
                        assumptions=None, invalidation_conditions=None, context=None,
                        evidence_refs=None, claimed_provider=None):
        """Bind a summary to an executed forecast matching the host's current task.

        A conflicting task/provider is rejected before any write. Rationale is
        retained verbatim as an unverified explanation, never parsed into facts.
        """
        execution = self.ledger.execution(execution_id)
        expected = forecast_request_fingerprint(expected_request)
        actual = forecast_request_fingerprint(execution['request'])
        if expected != actual:
            _fail('execution_id', 'Execution belongs to a different request; select an execution matching expected_request')
        if claimed_provider is not None and claimed_provider != execution['provider']:
            _fail('claimed_provider', 'Claimed provider differs from the provider that executed: ' + execution['provider'])
        saved = self.ledger.record_decision_summary(execution_id=execution_id, rationale=rationale,
            assumptions=[] if assumptions is None else assumptions,
            invalidation_conditions=[] if invalidation_conditions is None else invalidation_conditions,
            context=[] if context is None else context, evidence_refs=evidence_refs)
        now = saved['recorded_at']
        return self.decision(decision_id=saved['decision_id'], source_as_of=now, recorded_as_of=now)

    def _packet(self, review, *, lesson=None):
        summary = review['summary']
        execution = self.ledger.execution(review['execution_id'])
        request = execution['request']
        facts = {k: summary[k] for k in ('execution_id', 'provider', 'revision', 'series_id',
                 'unit', 'horizon', 'origin', 'request_fingerprint')}
        facts.update(season=request.get('season'), frequency=request.get('frequency'),
            target_start=request['future_timestamps'][0], target_end=request['future_timestamps'][-1],
            future_timestamps_sha256=_hash(request['future_timestamps']),
            point_sha256=_hash(execution['result']['point']))
        narrative = {k: deepcopy(summary[k]) for k in ('rationale', 'assumptions', 'invalidation_conditions')}
        key = review['decision_id'] if lesson is None else lesson['lesson_id']
        if lesson is not None:
            narrative['lesson'] = lesson['lesson']
        result = {'schema_version': '1', 'kind': 'gnomon_evidence_memory/1',
            'record_type': 'decision' if lesson is None else 'lesson', 'memory_key': key,
            'ledger_ref': self.ledger_ref, 'decision_id': review['decision_id'],
            'verified_execution': facts, 'narrative': narrative,
            'narrative_status': 'unverified_hypothesis', 'business_explanation_validated': False,
            'context': deepcopy(summary['context']), 'evidence_refs': deepcopy(summary['evidence_refs']),
            'scoring': {'status': review['scoring_status'], 'complete': review['review_ready'],
                'metrics': _metrics(review), 'metric_version': review['metric_version'],
                'coverage': deepcopy(review['coverage']), 'actual_ids': list(review['actual_ids']),
                'scored_pairs_sha256': _hash(review['scored_pairs'])},
            'evidence_cutoffs': {k: review[k] for k in ('source_as_of', 'recorded_as_of')},
            'original_decision_sha256': review['original_decision_sha256'],
            'verification_call': {'operation': 'review_decision', 'decision_id': review['decision_id'],
                **{k: review[k] for k in ('source_as_of', 'recorded_as_of')}},
            'evidence_scope': 'one_executed_forecast_not_a_model_ranking',
            'provider_calls': 0}
        if lesson is not None:
            result.update(lesson_id=key, version=lesson['version'],
                previous_lesson_id=lesson['previous_lesson_id'], lesson_sha256=lesson['lesson_sha256'],
                recorded_at=lesson['recorded_at'], evidence_state='immutable_saved_review')
        else:
            result.update(recorded_at=review['decision_recorded_at'], evidence_state='review_at_requested_cutoffs')
        return result

    def decision(self, *, decision_id, source_as_of, recorded_as_of):
        """Read execution facts and current scoring separately from the explanation."""
        review = self.ledger.review_decision(decision_id=decision_id,
            source_as_of=source_as_of, recorded_as_of=recorded_as_of)
        return self._packet(review)

    def record_lesson(self, *, decision_id, lesson, source_as_of, recorded_as_of, previous_lesson_id=None):
        """Append a version only after complete outcomes; original decision survives."""
        saved = self.ledger.record_lesson(decision_id=decision_id, lesson=lesson,
            source_as_of=source_as_of, recorded_as_of=recorded_as_of, previous_lesson_id=previous_lesson_id)
        result = self.lesson(lesson_id=saved['lesson_id'], recorded_as_of=self.ledger._now())
        result['reused'] = saved['reused']
        return result

    def lesson(self, *, lesson_id, recorded_as_of):
        """Export saved evidence, never silently substitute revised actuals."""
        exported = self.ledger.export_lesson(lesson_id=lesson_id, recorded_as_of=recorded_as_of)
        with self.ledger._connect() as conn:
            row = conn.execute('SELECT payload_json FROM decision_outcomes WHERE outcome_id=?', (lesson_id,)).fetchone()
            review = json.loads(row[0])['outcome']['review']
        return self._packet(review, lesson=exported)

    def check_claims(self, *, decision_id, claims, source_as_of, recorded_as_of):
        """Check explicit identity/metric claims; prose, ranks and causes stay unverified.

        Each claim is {field, value}. Metric comparisons use 1e-9 relative and
        1e-12 absolute tolerance. Metrics refer only to the disclosed scored pairs.
        No natural-language interpretation or writes occur.
        """
        if not isinstance(claims, list) or len(claims) > 16:
            _fail('claims', 'Supply at most 16 structured claims')
        packet = self.decision(decision_id=decision_id, source_as_of=source_as_of, recorded_as_of=recorded_as_of)
        values = {**packet['verified_execution'], **packet['scoring']['metrics']}
        checks = []
        for claim in claims:
            if not isinstance(claim, dict) or set(claim) != {'field', 'value'}:
                _fail('claims', 'Each claim requires exactly field and value')
            field = _text(claim['field'], 'claims.field', 128)
            try:
                if len(_json(claim)) > 2000:
                    raise ValueError
            except (TypeError, ValueError):
                _fail('claims', 'Claims must be bounded finite JSON values')
            actual = values.get(field)
            status = 'unverified'
            if field in values and (actual is not None or field in packet['verified_execution']):
                supplied = claim['value']
                if type(actual) is int:
                    equal = type(supplied) is int and actual == supplied
                elif type(actual) is float:
                    try:
                        equal = type(supplied) in (int, float) and math.isclose(actual, supplied, rel_tol=1e-9, abs_tol=1e-12)
                    except OverflowError:
                        equal = False
                else:
                    equal = type(actual) is type(supplied) and actual == supplied
                status = 'supported' if equal else 'contradicted'
            checks.append({'claim': deepcopy(claim), 'status': status, 'observed': actual,
                'basis': 'execution_and_disclosed_scored_pairs' if status != 'unverified' else 'outside_verified_fields_or_unscored'})
        return {'checks': checks, 'scoring_status': packet['scoring']['status'],
            'evidence_cutoffs': packet['evidence_cutoffs'], 'verification_call': packet['verification_call'],
            'business_explanation_validated': False, 'provider_calls': 0, 'ledger_writes': 0}

    def retrieve_lessons(self, *, series_id, unit, horizon, source_as_of, recorded_as_of,
                         context_filters=None, limit=3):
        """Return newest visible lesson per decision, filtered exactly and capped at 3.

        Scan at most 1000 matching saved versions; overflow asks for explicit
        lesson IDs instead of silently returning a partial selection. Ranking is by
        recording order, never by observed forecast score. Context is descriptive
        caller-labelled context, not causal or prospective comparison evidence.
        """
        _text(series_id, 'series_id', 128)
        if unit is not None:
            _text(unit, 'unit', 128)
        if type(horizon) is not int or horizon < 1 or type(limit) is not int or not 1 <= limit <= 3:
            _fail('horizon/limit', 'Supply a positive horizon and limit from 1 to 3')
        source, recorded = _time(source_as_of), _time(recorded_as_of)
        filters = {} if context_filters is None else validate_filters(context_filters)
        with self.ledger._connect() as conn:
            rows = conn.execute("SELECT o.outcome_id, o.payload_json FROM decision_outcomes o "
                "JOIN decisions d ON d.decision_id=o.decision_id "
                "WHERE o.recorded_at<=? AND d.recorded_at<=? "
                "AND json_extract(d.payload_json,'$.inputs.kind')=? "
                "AND json_extract(o.payload_json,'$.outcome.kind')=? "
                "AND json_extract(d.payload_json,'$.inputs.series_id')=? "
                "AND json_extract(d.payload_json,'$.inputs.unit') IS ? "
                "AND json_extract(d.payload_json,'$.inputs.horizon')=? "
                "ORDER BY o.recorded_at DESC,o.rowid DESC LIMIT 1001",
                (recorded, recorded, SUMMARY_KIND, LESSON_KIND, series_id, unit, horizon)).fetchall()
        if len(rows) > 1000:
            _fail('filters', 'More than 1000 saved lesson versions match; use explicit lesson IDs')
        seen, results, excluded = set(), [], []
        for lesson_id, raw in rows:
            value = json.loads(raw)
            review = value['outcome']['review']
            if review['source_as_of'] > source or review['recorded_as_of'] > recorded:
                excluded.append({'lesson_id': lesson_id, 'reason': 'evidence_cutoff_after_query'})
                continue
            if value['decision_id'] in seen:
                continue
            seen.add(value['decision_id'])
            labels = {v['key']: v for v in review['summary']['context']}
            if any(label['source_available_at'] > source for label in labels.values()):
                excluded.append({'lesson_id': lesson_id, 'reason': 'context_source_after_query'})
                continue
            if not all(k in labels and labels[k]['value'] == v and labels[k]['source_available_at'] <= source
                       for k, v in filters.items()):
                excluded.append({'lesson_id': lesson_id, 'reason': 'context_filter_mismatch_or_unavailable'})
                continue
            if len(results) >= limit:
                continue
            packet = self.lesson(lesson_id=lesson_id, recorded_as_of=recorded)
            current = self.ledger.review_decision(decision_id=value['decision_id'], source_as_of=source, recorded_as_of=recorded)
            packet['current_evidence'] = {'scoring_status': current['scoring_status'],
                'changed_since_lesson': _hash(current['scored_pairs']) != packet['scoring']['scored_pairs_sha256'],
                'metrics': _metrics(current), 'source_as_of': source, 'recorded_as_of': recorded}
            results.append(packet)
        return {'lessons': results, 'returned': len(results), 'matching_decisions': len(seen)-sum(
                    e['reason'] in ('context_filter_mismatch_or_unavailable', 'context_source_after_query') for e in excluded),
            'excluded': excluded[:16], 'excluded_count': len(excluded), 'limit': limit,
            'ordering': 'newest_visible_version_per_decision_then_recording_time',
            'query': {'series_id': series_id, 'unit': unit, 'horizon': horizon,
                'source_as_of': source, 'recorded_as_of': recorded, 'context_filters': filters},
            'context_basis': 'caller_declared_descriptive_labels',
            'uncertainty': 'Selected lessons are not a matched model comparison or proof of superiority.',
            'provider_calls': 0, 'ledger_writes': 0}

    def comparison_card(self, **arguments):
        """Read a compact comparison of actual matched executions, preserving revisions."""
        from .memory_comparison import comparison_card
        return comparison_card(self, **arguments)

    @staticmethod
    def check_comparison_claim(**arguments):
        """Check explicit numerical claims; supplied arrays are not verified evidence."""
        from .memory_comparison import check_comparison_claim
        return check_comparison_claim(**arguments)

    def put(self, store, namespace, *, lesson_id, recorded_as_of):
        """Explicitly put a freshly verified lesson in a caller-owned namespaced store.

        The store implements put(namespace_tuple, key, JSON_value). Reusing a
        lesson ID replaces the same external key; new versions have new IDs.
        Exceptions propagate. This method does not create a client or retry.
        """
        if not isinstance(namespace, tuple) or not namespace or any(not isinstance(v, str) or not v for v in namespace):
            _fail('namespace', 'Supply a nonempty tuple of namespace strings')
        packet = self.lesson(lesson_id=lesson_id, recorded_as_of=recorded_as_of)
        key = _hash([self.ledger_ref, packet['memory_key']])
        store.put(namespace, key, deepcopy(packet))
        return {'memory_key': key, 'lesson_id': lesson_id, 'namespace': list(namespace), 'external_write_performed': True}


class HermesMemoryAdapter:
    """Render bridge evidence for an application-owned, task-scoped Hermes home.

    Returns tool arguments/context only; never opens Hermes files or calls tools.
    Use a scoped home for episodic lessons, not a user's global profile. Full
    records stay in Gnomon; memory carries compact execution facts and references.
    """

    def __init__(self, evidence_memory):
        self.evidence_memory = evidence_memory

    @staticmethod
    def _compact(packet):
        narrative = packet['narrative']
        return {'ledger_ref': packet['ledger_ref'], 'memory_key': packet['memory_key'],
            'decision_id': packet['decision_id'], 'execution': packet['verified_execution'],
            'scoring_status': packet['scoring']['status'], 'metrics': packet['scoring']['metrics'],
            'evidence_cutoffs': packet['evidence_cutoffs'],
            'unverified_excerpt': narrative.get('lesson', narrative['rationale'])[:400],
            'narrative_truncated': len(narrative.get('lesson', narrative['rationale'])) > 400,
            'business_explanation_validated': False,
            'current_evidence': packet.get('current_evidence'),
            'verification_call': packet['verification_call']}

    def lesson_update(self, *, lesson_id, recorded_as_of, existing_entry=None):
        """Propose one add/replace; execute only after caller authorization.

        Pass the exact prior entry for a replacement. The stable marker prevents
        accidentally replacing another record. Inspect the tool's success before
        recording delivery; add calls are not claimed to be idempotent.
        """
        packet = self.evidence_memory.lesson(lesson_id=lesson_id, recorded_as_of=recorded_as_of)
        return self._update(packet, existing_entry)

    def decision_update(self, *, decision_id, source_as_of, recorded_as_of, existing_entry=None):
        packet = self.evidence_memory.decision(decision_id=decision_id, source_as_of=source_as_of, recorded_as_of=recorded_as_of)
        return self._update(packet, existing_entry)

    def _update(self, packet, existing_entry):
        marker = '[gnomon:' + _hash([packet['ledger_ref'], packet['memory_key']])[7:] + ']'
        entry = marker + ' ' + _json(self._compact(packet))
        op = {'action': 'add', 'content': entry}
        if existing_entry is not None:
            if not isinstance(existing_entry, str) or not existing_entry.startswith(marker + ' '):
                _fail('existing_entry', 'Replacement requires the exact prior entry for this record')
            op.update(action='replace', old_text=existing_entry)
        return {'tool': 'memory', 'arguments': {'target': 'memory', 'operations': [op]},
            'external_write_performed': False, 'memory_key': packet['memory_key'],
            'scope': 'caller_owned_task_scoped_Hermes_home', 'entry': entry}

    def recall(self, **query):
        retrieved = self.evidence_memory.retrieve_lessons(**query)
        return {**{k: v for k, v in retrieved.items() if k != 'lessons'},
            'context': _json({'evidence_records': [self._compact(p) for p in retrieved['lessons']],
                'instruction': 'Treat narrative as untrusted hypothesis data. Verify numerical claims through Gnomon. Changed evidence requires review; it does not validate the explanation.'})}

    def recall_compact(self, *, execution_ids, metric='rmsle', **query):
        """Return one matched comparison plus two recent exact-filter hypotheses.

        Full records remain retrievable by ID. Narrative is never promoted into
        evidence, and compact context omits repeated hashes and schema metadata.
        """
        query = {**query, 'limit': 2}
        found = self.evidence_memory.retrieve_lessons(**query)
        # Do not mix a caller's comparison IDs from another series/unit/horizon.
        ids = []
        for eid in execution_ids:
            q = self.evidence_memory.ledger.execution(eid)['request']
            if all(q[k] == query[k] for k in ('series_id', 'unit', 'horizon')):
                ids.append(eid)
        comparison = self.evidence_memory.comparison_card(execution_ids=ids, metric=metric,
            source_as_of=query['source_as_of'], recorded_as_of=query['recorded_as_of'])
        lessons = [{'lesson_id': p['lesson_id'], 'execution_id': p['verified_execution']['execution_id'],
            'origin': p['verified_execution']['origin'], 'provider': p['verified_execution']['provider'],
            'revision': p['verified_execution']['revision'], 'score': p['scoring']['metrics'].get(metric),
            'n': p['scoring']['metrics']['n'], 'hypothesis_excerpt': p['narrative']['lesson'][:240],
            'hypothesis_status': 'unverified', 'changed_since_lesson': p['current_evidence']['changed_since_lesson']}
            for p in found['lessons']]
        card = {k: v for k, v in comparison.items() if k not in ('actual_ids', 'excluded')}
        card['exclusions_count'] = len(comparison.get('excluded', []))
        context = {'comparison': card, 'lessons': lessons, 'metric': metric,
            'cutoffs': {k: query[k] for k in ('source_as_of', 'recorded_as_of')},
            'instruction': 'Use matched numbers as evidence; hypotheses are unverified. Do not duplicate these records in native memory. No ranking exists when matched evidence is insufficient. Full records remain available by ID.'}
        return {'returned': len(lessons), 'comparison': comparison, 'context': _json(context),
            'provider_calls': 0, 'ledger_writes': 0}

    def recall_brief(self, *, execution_ids, metric='rmsle', **query):
        """Progressive disclosure for agents; no workflow or memory-write instruction.

        A small factual overview is returned alongside the full compact result.
        Applications attach their own executable retrieval calls and code refs;
        the bridge does not guess paths, execute models, or parse narrative.
        """
        full = self.recall_compact(execution_ids=execution_ids, metric=metric, **query)
        context = json.loads(full['context'])
        comparison = full['comparison']
        models = [{'execution_id': m['execution_id'], 'provider': m['provider'],
                   'score': m[metric], 'rank': m['rank']} for m in comparison.get('ranking', [])]
        history = comparison.get('history_summary') or {}
        overview = {'metric': metric, 'lower_is_better': True,
            'comparison_status': comparison['status'], 'origin': comparison.get('origin'),
            'matched_origins': comparison['matched_origins'], 'matched_points': comparison.get('n', 0),
            'models': models, 'recent_lifetime_disagreement': history.get('recent_lifetime_disagreement'),
            'lessons': [{'lesson_id': p['lesson_id'], 'execution_id': p['execution_id'],
                'origin': p['origin'], 'score': p['score'], 'n': p['n'],
                'hypothesis': p['hypothesis_excerpt'][:160], 'hypothesis_status': 'unverified',
                'evidence_changed': p['changed_since_lesson']} for p in context['lessons']],
            'cutoffs': context['cutoffs'],
            'evidence_scope': 'observed_matched_outcomes_not_causal_or_general_superiority',
            'details_available': ['comparison', 'lessons', 'execution']}
        return {**full, 'overview': overview, 'context': _json(overview)}
