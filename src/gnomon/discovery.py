"""Task-oriented discovery from this session's configuration; no execution."""
from .forecast_adapter import ForecastAdapterError

DESCRIPTION = 'Inspect time series, forecast with your models, compare forecasts, and review observed outcomes.'
TASKS = {
    'inspect': ('Inspect time-series data', 'gnomon_inspect', None, {}, ['input']),
    'describe': ('Calculate an observed statistic', 'gnomon_describe', None, {}, ['data_ref', 'statistic']),
    'forecast': ('Forecast a series', 'gnomon_forecast', '0', {}, ['provider', 'request']),
    'compare': ('Compare forecasting models', 'gnomon_evaluate', '0', {}, ['data_ref', 'candidates', 'baseline', 'horizon']),
    'route': ('Select from a saved model comparison', 'gnomon_route', None, {}, ['data_ref', 'study_id', 'source_as_of', 'recorded_as_of']),
    'review': ('Score observed forecast outcomes', 'gnomon_ledger', 'evaluate', {'operation': 'evaluate'}, ['execution_id', 'source_as_of', 'recorded_as_of']),
    'recall': ('Recall recorded evidence and lessons', 'gnomon_memory', None, {'operation': 'recall'}, ['series_id', 'unit', 'horizon', 'source_as_of', 'recorded_as_of']),
    'submit_actuals': ('Submit observed outcomes', 'gnomon_ledger', 'append_actual', {'operation': 'append_actual'}, ['series_id', 'unit', 'valid_time', 'value', 'source_available_at']),
    'temporal': ('Calculate dates and time intervals', 'gnomon_temporal', None, {}, ['operation']),
}


def task_index(session):
    providers = session.engine.capabilities()
    status = {name: 'available' for name in TASKS}
    if not providers:
        status['forecast'] = 'register_provider'
    if len(providers) < 2:
        status['compare'] = 'register_two_providers'
    if session.ledger is None:
        for name in ('route', 'review', 'recall', 'submit_actuals'):
            status[name] = 'open_configured_ledger'
    elif not session.allow_outcome_writes:
        status['submit_actuals'] = 'enable_outcome_writes'
    if not session.enable_temporal:
        status['temporal'] = 'enable_temporal'
    return status


def describe_task(session, task):
    if not isinstance(task, str) or task not in TASKS:
        raise ForecastAdapterError('task must be one of: ' + ', '.join(TASKS))
    title, tool, variant, arguments, required = TASKS[task]
    state = task_index(session)[task]
    exposed = {t['name'] for t in session.tools()}
    enabled = tool in exposed and state not in {'open_configured_ledger', 'enable_outcome_writes'}
    schema = {'tool': 'gnomon_capabilities', 'arguments': {'schema_tool': tool}}
    if variant is not None:
        schema['arguments']['schema_variant'] = variant
    templates = {'tool': tool, 'arguments': dict(arguments), 'requires': list(required)}
    providers = list(session.engine.capabilities())
    if task == 'forecast' and 'last_value' in providers:
        templates = {'tool': tool, 'arguments': {'provider': 'last_value', 'request': {
            'history': [10, 12, 11], 'horizon': 2}}, 'requires': [], 'example_scope': 'synthetic_smoke_not_user_forecast'}
    return {'schema_version': '1', 'status': 'ok', 'task': task, 'purpose': title,
            'availability': state, 'availability_scope': 'configured_session_not_data_or_evidence_readiness',
            'configured_providers': providers if task in {'forecast', 'compare'} else [],
            'configured_routers': list(session.routers) if task == 'forecast' else [],
            'schema_call': schema if enabled else None, 'call_template': templates if enabled else None,
            'setup': None if state == 'available' else {
                'register_provider': 'Register a callable/factory or load an operator provider configuration.',
                'register_two_providers': 'Configure a baseline and at least one candidate provider.',
                'open_configured_ledger': 'Open a session with an operator-configured ledger; configuration-only discovery does not open it.',
                'enable_outcome_writes': 'The operator must enable allow_outcome_writes for an authorized outcome-submission task.',
                'enable_temporal': 'Enable temporal tools at session startup; standalone gnomon temporal is also available.',
            }[state],
            'effect': 'ledger_write' if task in {'review', 'submit_actuals'} else 'provider_calls' if task in {'forecast', 'compare'} else 'read',
            'guidance': 'Fill requires from the task. A template does not authorize execution or spending. Inspect the exact schema for optional fields and budgets.',
            'guide': 'https://github.com/TensorLink-AI/Gnomon/blob/main/skills/use-gnomon/SKILL.md'}
