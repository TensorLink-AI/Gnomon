"""Immutable scoring of recorded predictions against later visible actuals.

Evidence cutoffs never move a forecast origin or reconstruct provider state.
No provider registry or model execution is needed to rescore a recorded study.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from uuid import uuid4

from .contracts import GnomonError
from .forecast_adapter import point_error_metrics
from .ids import content_id
from .ledger import _time
from .repair import historical_repair_blockers


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def ranking(scores, providers):
    ranked = sorted(providers, key=lambda p: (scores[p]['mae'], providers.index(p))) if any(s['n'] for s in scores.values()) else []
    groups = {}
    for p in ranked:
        groups.setdefault(scores[p]['mae'], []).append(p)
    return ranked, {'metric': 'mae', 'direction': 'ascending', 'tie_comparison': 'exact_unrounded_score',
        'tie_order': 'provider_input_order', 'ties': [{'mae': v, 'providers': ps} for v, ps in groups.items() if len(ps) > 1],
        'guidance': 'Equal scores do not establish a winner or future predictive superiority.'}


def rescore_study(ledger, references, data_ref, *, study_id, source_as_of, recorded_as_of,
                  allow_partial=True, max_folds=8):
    if ledger is None:
        raise GnomonError('LEDGER_REQUIRED', 'Rescoring requires the ledger containing the original study and executions.')
    if type(allow_partial) is not bool:
        raise GnomonError('INVALID_ARGUMENTS', 'allow_partial must be a boolean.')
    if not isinstance(study_id, str) or not study_id:
        raise GnomonError('INVALID_ARGUMENTS', 'study_id must be a nonempty recorded study ID.')
    source, recorded = _time(source_as_of), _time(recorded_as_of)
    original = ledger.study(study_id, recorded_as_of=recorded)
    if original.get('derived_from') or original.get('evidence') != 'rolling_origin_backtest':
        raise GnomonError('INVALID_ARGUMENTS', 'Select the original rolling-origin study, not an already derived rescore.')
    if len(original['folds']) > max_folds:
        raise GnomonError('INVALID_ARGUMENTS', 'Original study exceeds the operator fold limit.')
    frozen, name, _ = references._select(data_ref, original['series_id'])
    variable = original.get('variable', next((a['variable'] for f in original['folds'] for a in f['actuals']), None))
    if (frozen.unit != original['unit'] or frozen.loaded.frequency != original['frequency'] or
            (variable is not None and frozen.loaded.variable != variable)):
        raise GnomonError('TASK_IDENTITY_MISMATCH', 'Rescore actuals must use the original series, variable, unit and frequency.')
    if historical_repair_blockers(frozen.repairs):
        raise GnomonError('INVALID_ARGUMENTS', 'Rescoring requires observed actuals, not globally repaired history.')
    snapshot = frozen.loaded.snapshot.narrow(as_of=datetime.fromisoformat(source),
        recorded_as_of=None if frozen.loaded.snapshot.unknown_recorded_times else datetime.fromisoformat(recorded))
    truth = {r.valid_time: r for r in snapshot.series(name, frozen.loaded.variable)}
    providers = original.get('provider_order', [original['baseline'], *[p for p in original['providers'] if p != original['baseline']]])
    folds, exclusions = [], []
    for old in original['folds']:
        fold = deepcopy(old)
        reason = None
        if old['status'] != 'complete':
            reason = 'incomplete_original_fold'
        else:
            for p in providers:
                run = ledger.execution(old['runs'][p]['execution_id'])
                if run['recorded_at'] > recorded:
                    reason = 'execution_not_recorded_at_cutoff'
                if run['request'] != old['request'] or run['provider'] != p or run['revision'] != old['runs'][p]['revision'] or run['result']['point'] != old['runs'][p]['point']:
                    raise GnomonError('STUDY_EXECUTION_MISMATCH', 'Saved study predictions do not match their recorded executions.')
            times = [datetime.fromisoformat(t) for t in old['request']['future_timestamps']]
            if any(t not in truth for t in times):
                reason = 'actuals_unavailable_at_cutoffs'
            else:
                actuals = [{k: v.isoformat() if hasattr(v, 'isoformat') else v for k, v in asdict(truth[t]).items()} for t in times]
                if snapshot.unknown_recorded_times and actuals != old['actuals']:
                    reason = 'file_revision_availability_unknown'
                elif not reason:
                    fold['actuals'] = actuals
        if reason:
            exclusions.append({'origin': old['origin'], 'reason': reason})
            fold['status'] = 'excluded_from_rescore'
            fold['exclusion_reason'] = reason
        folds.append(fold)
    matched = [f for f in folds if f['status'] == 'complete']
    complete = len(matched) == original['usage']['requested_folds']
    if not complete and not allow_partial:
        raise GnomonError('INCOMPLETE_RESCORE', 'Not all original folds have recorded predictions and visible actuals.',
                          {'excluded_folds': exclusions, 'matched_folds': len(matched), 'required_folds': original['usage']['requested_folds']})
    scores = {p: point_error_metrics((point, actual['value']) for f in matched
               for point, actual in zip(f['runs'][p]['point'], f['actuals'])) for p in providers}
    ranked, policy = ranking(scores, providers)
    if 'provider_order' not in original:
        policy['tie_order'] = 'baseline_then_provider_name_legacy'
    identity = str(uuid4())
    result = {**deepcopy(original), 'study_id': identity, 'original_study_id': study_id,
        'rescore_study_id': identity, 'derived_from': study_id, 'rescore_only': True,
        'original_study_sha256': digest(original), 'original_unchanged': True,
        'original_snapshot_id': original['snapshot_id'], 'snapshot_id': snapshot.snapshot_id,
        'source_fingerprint': snapshot.source_ref, 'folds': folds, 'excluded_folds': exclusions,
        'scores': scores, 'ranking': ranked, 'ranking_policy': policy, 'provider_calls': 0,
        'predictions_reused_exactly': True, 'status': 'complete' if complete else 'partial' if matched else 'unscored',
        'source_as_of': source, 'recorded_as_of': recorded,
        'cutoff_scopes': {'forecast_origins': 'unchanged original fold.origin and fold.request.cutoff',
            'observation_valid_times': 'unchanged original future_timestamps',
            'actual_source_available_as_of': source, 'actual_recorded_as_of': recorded,
            'snapshot_source_as_of': snapshot.as_of.isoformat(),
            'snapshot_recorded_as_of': snapshot.recorded_as_of.isoformat() if snapshot.recorded_as_of else None,
            'study_evidence_recorded_as_of': recorded},
        'original_usage': original['usage'], 'usage': {**original['usage'], 'provider_calls': 0,
            'internal_model_calls': 0, 'matched_folds': len(matched), 'elapsed_seconds': 0, 'stop_reason': None},
        'original_budget': original['budget'], 'budget': {'max_folds': max_folds, 'max_calls': 0,
            'max_seconds': None, 'max_providers': len(providers)},
        'diagnostics': {p: {'attempted': 0, 'succeeded': 0, 'failed': 0, 'reused': len(matched)} for p in providers},
        'issues': exclusions, 'full_study': {'tool': 'gnomon_evaluate', 'arguments': {'study_id': identity}},
        'routing_readiness': {'ready': False, 'issues': [{'action': 'route_original_study',
            'description': 'This is a rescore, not a new forecasting task. Route using the original study and appropriate input/origin cutoffs.'}]}}
    result.pop('recorded_at', None)
    result['evaluation_status'] = result['status']
    result['routing_status'] = 'rescore_not_a_new_route_task'
    refresh_derivations(result, matched, providers)
    result['cohort_id'] = content_id('cohort', {'folds': [{'request': f['request'], 'actuals': f['actuals']} for f in matched]}, length=64)
    if digest(ledger.study(study_id)) != result['original_study_sha256']:
        raise GnomonError('STUDY_INTEGRITY', 'Original study changed during rescoring.')
    ledger.record_study(result)
    return result


def compare_studies(ledger, *, original_study_id, rescored_study_id):
    if ledger is None:
        raise GnomonError('LEDGER_REQUIRED', 'Study comparison requires the evidence ledger.')
    if any(not isinstance(v, str) or not v for v in (original_study_id, rescored_study_id)):
        raise GnomonError('INVALID_ARGUMENTS', 'Both study IDs must be nonempty strings.')
    old, new = ledger.study(original_study_id), ledger.study(rescored_study_id)
    if new.get('derived_from') != original_study_id:
        raise GnomonError('INVALID_ARGUMENTS', 'The second study must be derived from the specified original.')
    original_folds = {f['origin']: f for f in old['folds']}
    changed, reused = [], True
    for f in new['folds']:
        prior = original_folds.get(f['origin'])
        if prior is None:
            reused = False
            continue
        reused &= prior['request'] == f['request'] and prior['runs'] == f['runs']
        if prior['actuals'] != f['actuals'] or prior['status'] != f['status']:
            changed.append({'origin': f['origin'], 'old_actuals': prior['actuals'], 'new_actuals': f['actuals'],
                            'old_status': prior['status'], 'new_status': f['status']})
    expected = new.get('original_study_sha256')
    return {'schema_version': '1', 'status': 'ok', 'operation': 'compare_studies',
        'original_study_id': original_study_id, 'rescored_study_id': rescored_study_id,
        'old_scores': old['scores'], 'new_scores': new['scores'], 'old_ranking': old['ranking'],
        'new_ranking': new['ranking'], 'old_ties': old['ranking_policy']['ties'], 'new_ties': new['ranking_policy']['ties'],
        'changed_folds': changed, 'original_study_sha256': digest(old),
        'changed_actual_ids': [{'origin': f['origin'], 'old': [digest(a) for a in f['old_actuals']],
                                'new': [digest(a) for a in f['new_actuals']]} for f in changed if f['old_actuals'] != f['new_actuals']],
        'actual_id_basis': 'SHA-256 of canonical serialized temporal observation including revision/source/recording identity',
        'original_unchanged': digest(old) == expected if expected is not None else None,
        'original_integrity_basis': 'recorded_rescore_parent_hash' if expected else 'parent_hash_unavailable_for_legacy_rescore',
        'predictions_reused_exactly': reused, 'provider_calls': 0}


def refresh_derivations(result, folds, providers):
    if 'score_derivations' in result:
        from .diagnostics import scored_pairs
        result['score_derivations'] = {p: scored_pairs((point, actual['value']) for fold in folds
            for point, actual in zip(fold['runs'][p]['point'], fold['actuals'])) for p in providers}
