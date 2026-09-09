"""Resolve agent selections against host-owned forecast evidence, without calls.

Executions must come from the host's trusted tool/engine transcript, never from
the agent's final answer. Request hashes bind tasks; they are not signatures.
No natural-language extraction, provider fallback, writes or network access.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import re

from .forecast_adapter import ForecastAdapterError


PROVIDER_ALIASES = ('selected_provider', 'candidate')
COMPLETION_FIELDS = ('operation', 'operation_succeeded', 'provider', 'execution_id',
    'revision', 'series_id', 'unit', 'horizon', 'future_timestamps', 'point', 'request_fingerprint')
FINAL_SELECTION_GUIDANCE = {
    'completion_pointer': '/completion',
    'selection_fields': ['execution_id', 'provider'],
    'required_any_of': ['execution_id', 'provider'],
    'execution_id_required_when': 'More than one successful execution uses the selected provider.',
    'preserve_completion_fields': list(COMPLETION_FIELDS),
    'host_resolver': 'gnomon.resolve_final_selection',
    'host_rule': 'Resolve against trusted executions and the expected request; never replace a successful matching execution solely for a prose final.',
}


def forecast_request_fingerprint(request):
    """SHA-256 of the full canonical request; excludes provider/revision identity.

    Uses the engine's validated numeric, container, default and time conventions.
    Calendar offsets remain significant when a frequency is declared.
    """
    from .inference import _freeze_request, _json
    return 'sha256:' + hashlib.sha256(_json(asdict(_freeze_request(request))).encode()).hexdigest()


def forecast_completion(execution):
    """Return the canonical point-forecast completion of a ForecastExecution."""
    from .inference import ForecastExecution, _freeze_request
    if not isinstance(execution, ForecastExecution):
        raise ForecastAdapterError('forecast_completion requires a trusted ForecastExecution')
    request = _freeze_request(execution.request)
    execution.result.validate(request)
    result = {
        'operation': 'forecast', 'operation_succeeded': True,
        'provider': execution.provider, 'execution_id': execution.execution_id,
        'revision': execution.revision, 'series_id': request.series_id, 'unit': request.unit,
        'horizon': request.horizon, 'future_timestamps': list(request.future_timestamps),
        'point': [float(v) for v in execution.result.point],
        'request_fingerprint': forecast_request_fingerprint(request),
    }
    if not _valid_completion(result):
        raise ForecastAdapterError('execution does not contain a valid forecast identity and point result')
    return result


def _valid_completion(value):
    if not isinstance(value, dict) or not set(COMPLETION_FIELDS) <= value.keys():
        return False
    if value['operation'] != 'forecast' or value['operation_succeeded'] is not True:
        return False
    if any(not isinstance(value[k], str) or not value[k].strip() for k in ('provider', 'execution_id')):
        return False
    if any(value[k] is not None and (not isinstance(value[k], str) or not value[k].strip()) for k in ('revision', 'series_id', 'unit')):
        return False
    if type(value['horizon']) is not int or value['horizon'] < 1:
        return False
    if not isinstance(value['request_fingerprint'], str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', value['request_fingerprint']):
        return False
    points, times = value['point'], value['future_timestamps']
    try:
        return (isinstance(points, (list, tuple)) and len(points) == value['horizon']
            and all(type(p) in (float, int) and math.isfinite(p) for p in points)
            and isinstance(times, (list, tuple)) and len(times) in (0, value['horizon'])
            and all(isinstance(t, str) and t for t in times))
    except (ValueError, OverflowError):
        return False


def _execution(value):
    from .inference import ForecastExecution
    if isinstance(value, ForecastExecution):
        return forecast_completion(value)
    if not isinstance(value, dict):
        raise ValueError('invalid_execution_evidence')
    if value.get('status') in ('error', 'failed', 'cancelled', 'unscored') or value.get('operation_succeeded') is False or value.get('isError') is True:
        return None
    if 'completion' not in value and 'request' in value and 'result' in value and value.get('result_contract_validated') is True:
        return _completion_from_record(value)
    completion = value.get('completion', value)
    if not _valid_completion(completion):
        raise ValueError('incomplete_or_invalid_execution_evidence')
    if completion is not value:
        # Reject contradictory envelopes, not just contradictory final answers.
        if value.get('status', 'ok') != 'ok' or value.get('result_contract_validated') is not True:
            raise ValueError('execution_not_validated')
        if any(k in value and value[k] != completion[k] for k in ('provider', 'execution_id', 'revision')):
            raise ValueError('conflicting_execution_envelope')
        result = value.get('result', {})
        if any(k in result and list(result[k]) != list(completion[c]) for k, c in (('point', 'point'), ('timestamps', 'future_timestamps'))):
            raise ValueError('conflicting_execution_envelope')
    result = {k: deepcopy(completion[k]) for k in COMPLETION_FIELDS}
    result['point'] = list(result['point'])
    result['future_timestamps'] = list(result['future_timestamps'])
    return result


def _completion_from_record(record):
    """Reconstruct a completion from an integrity-checked ledger execution."""
    from .inference import ForecastExecution, _freeze_request
    from .forecast_adapter import ForecastResult
    result = deepcopy(record['result'])
    if result.get('quantiles') is not None:
        result['quantiles'] = [{float(k): v for k, v in row.items()} for row in result['quantiles']]
    execution = ForecastExecution(record['execution_id'], record['fingerprint'], record['provider'],
        record['revision'], _freeze_request(record['request']), ForecastResult(**result))
    return forecast_completion(execution)


class _InvalidJSON(ValueError):
    pass


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidJSON('duplicate_final_field')
        result[key] = value
    return result


def _invalid_constant(value):
    raise _InvalidJSON('nonfinite_final_value')


def _final(answer):
    if answer is None or isinstance(answer, str) and not answer.strip():
        return {}, False, [], None
    if isinstance(answer, str):
        try:
            answer = json.loads(answer, object_pairs_hook=_object,
                parse_constant=_invalid_constant)
        except (ValueError, RecursionError):
            # Prose/tables are absence of a machine-readable selection, not
            # instructions to extract names. Malformed JSON objects fail closed.
            return {}, False, [], 'invalid_final_json' if answer.lstrip().startswith(('{', '[')) else None
    if not isinstance(answer, dict):
        return {}, False, [], 'final_answer_must_be_object'
    answer = deepcopy(answer)
    unknown = set(answer) - set(COMPLETION_FIELDS) - set(PROVIDER_ALIASES) - {'rationale'}
    if unknown:
        return answer, False, [], 'unsupported_final_fields'
    normalizations = []
    for alias in PROVIDER_ALIASES:
        if alias in answer:
            if 'provider' in answer and answer['provider'] != answer[alias]:
                return answer, False, normalizations, 'conflicting_provider_aliases'
            answer['provider'] = answer.pop(alias)
            normalizations.append({'from': alias, 'to': 'provider'})
    for key in ('provider', 'execution_id', 'rationale'):
        if key in answer and (not isinstance(answer[key], str) or not answer[key].strip()):
            return answer, False, normalizations, 'invalid_final_field_type'
    if 'operation' in answer and answer['operation'] != 'forecast' or 'operation_succeeded' in answer and answer['operation_succeeded'] is not True:
        return answer, False, normalizations, 'conflicting_final_operation'
    for key in ('series_id', 'unit', 'revision'):
        if key in answer and answer[key] is not None and (not isinstance(answer[key], str) or not answer[key].strip()):
            return answer, False, normalizations, 'invalid_final_field_type'
    if 'horizon' in answer and (type(answer['horizon']) is not int or answer['horizon'] < 1):
        return answer, False, normalizations, 'invalid_final_field_type'
    for key in ('point', 'future_timestamps'):
        if key in answer and not isinstance(answer[key], list):
            return answer, False, normalizations, 'invalid_final_field_type'
    if 'future_timestamps' in answer and any(not isinstance(t, str) or not t for t in answer['future_timestamps']):
        return answer, False, normalizations, 'invalid_final_field_type'
    if 'point' in answer:
        try:
            if any(type(p) not in (int, float) or not math.isfinite(p) for p in answer['point']):
                return answer, False, normalizations, 'invalid_final_field_type'
        except OverflowError:
            return answer, False, normalizations, 'invalid_final_field_type'
    if 'request_fingerprint' in answer and (not isinstance(answer['request_fingerprint'], str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', answer['request_fingerprint'])):
        return answer, False, normalizations, 'invalid_final_field_type'
    conformant = not normalizations and bool(set(answer) & {'provider', 'execution_id'})
    return answer, conformant, normalizations, None


def resolve_final_selection(*, final_answer, successful_executions, expected_request):
    """Resolve a final JSON selection, or recover one task-matching execution.

    Accept host-owned ForecastExecution objects, canonical completions, or full
    CLI/Python/MCP payloads containing completion. Retrieve retained payloads first.
    Explicit failures may be included and are ignored; malformed success evidence
    fails closed. No prose extraction. Documented provider aliases are
    selected_provider and candidate; normalization never counts as strict final
    conformance. Multiple execution IDs for one provider require execution_id.
    An explicit conflicting/failed/unrelated selection is never auto-recovered.

    expected_request must be the full ForecastRequest/dict used for dispatch,
    including snapshot metadata for a data_ref call. Incorrect API argument types
    raise ForecastAdapterError. Limit: 1,000 evidence entries per resolution.
    """
    from .inference import _freeze_request
    from .diagnostics import EXECUTION_SCOPE
    expected = _freeze_request(expected_request)
    fingerprint = forecast_request_fingerprint(expected)
    if not isinstance(successful_executions, (list, tuple)) or len(successful_executions) > 1000:
        raise ForecastAdapterError('successful_executions must be a list/tuple of at most 1000 trusted execution records')
    executions, invalid, failed = {}, [], 0
    for index, record in enumerate(successful_executions):
        try:
            completion = _execution(record)
            if completion is None:
                failed += 1
                continue
            identity = completion['execution_id']
            if identity in executions and executions[identity] != completion:
                raise ValueError('conflicting_duplicate_execution_id')
            executions[identity] = completion
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            safe_causes = ('invalid_execution_evidence', 'incomplete_or_invalid_execution_evidence',
                'execution_not_validated', 'conflicting_execution_envelope', 'conflicting_duplicate_execution_id')
            invalid.append({'index': index, 'cause': str(exc) if type(exc) is ValueError and str(exc) in safe_causes else 'invalid_execution_evidence'})
    matching = {key: c for key, c in executions.items() if c['request_fingerprint'] == fingerprint
        and c['series_id'] == expected.series_id and c['unit'] == expected.unit
        and c['horizon'] == expected.horizon and list(c['future_timestamps']) == list(expected.future_timestamps)}
    selection, conformant, normalizations, problem = _final(final_answer)
    def result(status, cause, execution=None):
        resolved = execution is not None
        return {'status': status, 'resolution_status': status, 'resolved': resolved,
            'engine_execution_succeeded': bool(executions), 'final_answer_conformant': conformant,
            'end_to_end_completed': resolved, 'execution': deepcopy(execution), 'cause': cause,
            'normalizations': normalizations, 'successful_execution_count': len(executions),
            'task_matching_execution_count': len(matching), 'ignored_failed_execution_count': failed,
            'invalid_evidence': invalid, 'expected_request_fingerprint': fingerprint,
            'execution_diagnostics': dict(provider_calls=0, forecast_calls=0, ledger_writes=0, source_mutations=0, scope=EXECUTION_SCOPE)}
    if not executions:
        return result('no_successful_execution', 'invalid_execution_evidence' if invalid else 'no_successful_execution')
    if invalid or problem:
        return result('conflicting_final_selection', 'invalid_execution_evidence' if invalid else problem)
    explicit = bool(set(selection) & {'provider', 'execution_id'})
    if 'execution_id' in selection:
        chosen = executions.get(selection['execution_id'])
        if chosen is None:
            return result('conflicting_final_selection', 'selected_execution_not_successful')
    elif 'provider' in selection:
        choices = [c for c in executions.values() if c['provider'] == selection['provider']]
        if not choices:
            return result('conflicting_final_selection', 'selected_provider_not_successful')
        if len(choices) > 1:
            return result('ambiguous_multiple_executions', 'execution_id_required')
        chosen = choices[0]
    else:
        if not matching:
            return result('task_mismatch', 'no_execution_matches_expected_request')
        if len(matching) > 1:
            return result('ambiguous_multiple_executions', 'final_answer_missing_explicit_selection')
        chosen = next(iter(matching.values()))
    if chosen['execution_id'] not in matching:
        return result('conflicting_final_selection', 'selected_execution_task_mismatch')
    for key, value in selection.items():
        if key != 'rationale' and (key not in chosen or value != chosen[key]):
            return result('conflicting_final_selection', 'final_fields_conflict_with_execution')
    return result('explicit_selection' if explicit else 'recovered_single_execution',
        'explicit_selection_matches_execution' if explicit else 'final_answer_missing_explicit_selection', chosen)
