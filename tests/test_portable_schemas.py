from copy import deepcopy

import pytest

from gnomon import GnomonSession, TemporalLedger
from gnomon.contracts import GnomonError
from gnomon.tool_schemas import portable_schema


def test_declarations_survive_branch_stripping_and_preserve_exact_schema(tmp_path):
    with GnomonSession.from_config(ledger=TemporalLedger(tmp_path / 'ledger.db')) as session:
        exact = {t['name']: t['inputSchema'] for t in session.tools(portable=False)}
        for tool in session.tools():
            schema = tool['inputSchema']
            branches = exact[tool['name']].get('oneOf', [])
            if not branches:
                continue
            assert schema['oneOf'] == branches
            # Emulate the legacy host's removal of top-level combinators.
            stripped = {k: v for k, v in schema.items() if k not in ('oneOf', 'anyOf', 'allOf')}
            assert set(stripped['properties']) == set().union(*(b['properties'] for b in branches))
            assert set(stripped['required']) == set.intersection(*(set(b.get('required', [])) for b in branches))
        result = session.call('gnomon_capabilities', {'schema_tool':'gnomon_ledger',
            'schema_variant':'review_decision'})
        assert result['schema']['properties']['operation']['const'] == 'review_decision'
        assert 'decision_id' in result['schema']['required']
        with pytest.raises(GnomonError):
            session.call('gnomon_capabilities', {'schema_tool':'gnomon_ledger','schema_variant':'append_actual'})
        with pytest.raises(GnomonError):
            session.call('gnomon_forecast', {'provider':'last_value','request':{'history':[1,2],'horizon':1},
                                             'data_ref':'invalid','horizon':1})


def test_portable_union_does_not_mutate_or_restrict_valid_variants():
    import jsonschema
    canonical = {'type':'object', 'oneOf':[
        {'type':'object','additionalProperties':False,'required':['operation','value'],
         'properties':{'operation':{'const':'a'},'value':{'type':'number','minimum':0}}},
        {'type':'object','additionalProperties':False,'required':['operation','value'],
         'properties':{'operation':{'const':'b'},'value':{'type':'string','enum':['x']}}}]}
    original = deepcopy(canonical)
    portable = portable_schema(canonical)
    assert canonical == original
    for payload in ({'operation':'a','value':1}, {'operation':'b','value':'x'}):
        jsonschema.validate(payload, portable)
    for payload in ({'operation':'a','value':-1}, {'operation':'b','value':'bad'}, {'operation':'unknown','value':1}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, portable)
