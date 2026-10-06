"""Canonical hosted action schemas, portable discovery and structural validation."""
from copy import deepcopy
import jsonschema
from gnomon.session import REQUEST_SCHEMA
from .storage import ServiceError

FIELDS = {'idempotency_key': {'type': 'string'},
 'decision_id': {'type': 'string'},
 'lesson_id': {'type': 'string'},
 'request_id': {'type': 'string'},
 'export_id': {'type': 'string'},
 'source_as_of': {'type': 'string'},
 'recorded_as_of': {'type': 'string'},
 'query': {'type': 'string'},
 'memory_id': {'type': 'string'},
 'provider': {'type': 'string'},
 'revision': {'type': ['string', 'null'],
              'description': 'Caller-claimed model version; null means unknown.'},
 'computed_at': {'type': 'string'},
 'snapshot_id': {'type': 'string'},
 'analysis_id': {'type': 'string'},
 'method': {'type': 'string'},
 'method_version': {'type': 'string'},
 'lesson': {'type': 'string'},
 'reference': {'type': 'object'},
 'export_to_ditto': {'type': 'boolean',
                     'description': 'analysis.submit only: atomically store analysis and enqueue Ditto '
                                    'export; requires memory.export and configured Ditto.'},
 'metrics': {'type': 'object',
             'description': 'Client-computed named finite scalar metrics or null. Not server-verified.'},
 'target_action': {'type': 'string'}}
FIELDS["request"] = deepcopy(REQUEST_SCHEMA)
FIELDS["result"] = {'type': 'object',
 'additionalProperties': False,
 'required': ['point'],
 'description': 'Client forecast. Shape is validated; provider execution is not attested.',
 'properties': {'point': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 1, 'maxItems': 1000},
                'quantiles': {'type': ['array', 'null'],
                              'items': {'type': 'object', 'additionalProperties': {'type': 'number'}}},
                'timestamps': {'type': 'array', 'items': {'type': 'string'}},
                'series_id': {'type': ['string', 'null']},
                'unit': {'type': ['string', 'null']},
                'metadata': {'type': 'object'},
                'sample_paths': {'type': ['array', 'null'],
                                 'items': {'type': 'array', 'items': {'type': 'number'}}}}}

for _field in ('provider', 'method', 'method_version'):
    FIELDS[_field] = {'type': 'string', 'minLength': 1, 'maxLength': 512, 'pattern': r'\S'}
FIELDS['revision'] = {'type': ['string', 'null'], 'minLength': 1, 'maxLength': 512, 'pattern': r'\S'}
FIELDS['lesson'] = {'type': 'string', 'minLength': 1, 'maxLength': 8000, 'pattern': r'\S'}
FIELDS['idempotency_key'] = {'type': 'string', 'minLength': 1, 'maxLength': 128}
FIELDS['metrics'] = {'type': 'object', 'maxProperties': 32,
    'propertyNames': {'minLength': 1, 'maxLength': 128, 'pattern': r'\S'},
    'additionalProperties': {'type': ['number', 'null']}}

# action: (required fields, optional fields). Action itself is always required.
CONTRACTS = {
    'info': ((), ()),
    'schema.get': (('target_action',), ()),
    'decision.status': (('decision_id',), ()),
    'forecast.submit': (('provider', 'request', 'result', 'idempotency_key'), ('revision', 'computed_at')),
    'snapshot.save': (('decision_id', 'source_as_of', 'recorded_as_of', 'idempotency_key'), ()),
    'analysis.submit': (('snapshot_id', 'method', 'method_version', 'metrics', 'lesson', 'idempotency_key'), ('export_to_ditto',)),
    'dataset.put': (('request', 'idempotency_key'), ()),
    'review.save': (('decision_id', 'source_as_of', 'recorded_as_of', 'idempotency_key'), ()),
    'request.get': (('request_id',), ()),
    'resolve': (('reference',), ('recorded_as_of',)),
    'export.enqueue': (('recorded_as_of', 'idempotency_key'), ('lesson_id', 'analysis_id')),
    'export.get': (('export_id',), ()),
    'export.deliver': (('export_id',), ()),
    'export.reconcile': (('export_id', 'memory_id'), ()),
    'memory.recall': (('query', 'source_as_of', 'recorded_as_of'), ()),
}
DESCRIPTIONS = {
    'info': 'Read versions, project permissions, configured providers and Ditto setup without external calls.',
    'schema.get': 'Get the exact action schema used by runtime validation; set target_action from info.actions.',
    'decision.status': 'Read the current decision lifecycle, actual coverage, saved analyses/reviews and Ditto exports. No writes or external calls.',
    'snapshot.save': 'Freeze decision, forecast and actuals at explicit source/recording cutoffs; returns snapshot_id.',
    'analysis.submit': 'Save client-computed metrics and lesson linked to snapshot_id. Optional export_to_ditto atomically enqueues delivery; arithmetic remains unverified.',
    'forecast.submit': 'Record a locally computed forecast. Requires a named series and history/target timestamps. No dataset or snapshot prerequisite.',
    'export.enqueue': 'Enqueue exactly one lesson_id or analysis_id at a recording cutoff; does not deliver automatically.',
}


def action_schema(action):
    if action not in CONTRACTS:
        raise ServiceError('INVALID_ARGUMENTS', 'Unknown action. Discover supported actions with info.')
    required, optional = CONTRACTS[action]
    schema = {'type': 'object', 'additionalProperties': False,
              'required': ['action', *required],
              'properties': {'action': {'type': 'string', 'enum': [action]},
                             **{k: deepcopy(FIELDS[k]) for k in (*required, *optional)}},
              'description': DESCRIPTIONS.get(action, action)}
    if action == 'schema.get':
        schema['properties']['target_action']['enum'] = list(CONTRACTS)
    if action == 'forecast.submit':
        request = schema['properties']['request']
        request['required'] = ['history', 'horizon', 'series_id', 'timestamps', 'future_timestamps']
        request['properties']['series_id'] = {'type': 'string', 'minLength': 1}
        for key in ('timestamps', 'future_timestamps'):
            request['properties'][key] = {**request['properties'][key], 'minItems': 1}
    if action == 'export.enqueue':
        schema['oneOf'] = [{'required': ['lesson_id']}, {'required': ['analysis_id']}]
    return schema


def union_schema():
    # Flat discovery stays compatible with harnesses that strip top-level oneOf.
    return {'type': 'object', 'additionalProperties': False, 'required': ['action'],
            'properties': {'action': {'type': 'string', 'enum': list(CONTRACTS)}, **deepcopy(FIELDS)},
            'description': 'Discovery index only: use schema.get for exact per-action required/allowed fields. '
                           'Structural schemas do not replace evidence, authorization or numerical consistency checks.'}


def validate_action(arguments):
    if not isinstance(arguments, dict):
        raise ServiceError('INVALID_ARGUMENTS', 'Arguments must be an object.')
    action = arguments.get('action')
    if not isinstance(action, str) or action not in CONTRACTS:
        raise ServiceError('INVALID_ARGUMENTS', 'Unknown or missing action. Discover supported actions with info.')
    schema = action_schema(action)
    missing = sorted(set(schema['required']) - set(arguments))
    unsupported = set(arguments) - set(schema['properties'])
    details = []
    if missing:
        details.append('Missing required fields: ' + ', '.join(missing) + '.')
    if unsupported:
        details.append('Unsupported fields supplied. Allowed fields for this operation: '
                       + ', '.join(sorted(schema['properties'])) + '.')
    if details:
        raise ServiceError('INVALID_ARGUMENTS', ' '.join(details))
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(arguments))
    if errors:
        if action == 'export.enqueue' and errors[0].validator == 'oneOf':
            message = 'Supply exactly one lesson_id or analysis_id.'
        else:
            # Only reflect a server-declared top-level field name, not secret values/keys.
            field = next(iter(errors[0].absolute_path), 'arguments')
            field = field if field in schema['properties'] else 'arguments'
            message = f'Invalid structure for {field}; use schema.get with target_action={action}.'
        raise ServiceError('INVALID_ARGUMENTS', message)
