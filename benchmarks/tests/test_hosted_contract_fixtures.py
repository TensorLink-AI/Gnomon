"""Validate design reference examples, not nonexistent hosted authorization."""
import json
from pathlib import Path

import jsonschema
import pytest

CONTRACTS = Path(__file__).resolve().parents[2] / 'docs/hosting/contracts'


def test_reference_schema_and_design_vectors():
    schema = json.loads((CONTRACTS / 'evidence-reference.schema.json').read_text())
    fixtures = json.loads((CONTRACTS / 'acceptance-fixtures.json').read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(fixtures['reference'])
    for change in ({'resource_id': '../other-ledger'}, {'resource_type': 'result_ref'},
                   {'token': 'not-a-real-token'}, {'service_id': 'https://untrusted.example'},
                   {'content_sha256': 'wrong'}, {'schema_version': '2'}):
        with pytest.raises(jsonschema.ValidationError):
            validator.validate({**fixtures['reference'], **change})
    principals = {p['id'] for p in fixtures['principals']}
    assert all(c['principal'] in principals for c in fixtures['authorization_cases'])
    # Permission expectations are deliberately not executed against a toy ACL.
    # They become service conformance tests when the real boundary exists in B.
    assert fixtures['status'] == 'contract_vectors'
