import json

import pytest

from benchmarks.ledger_optimization import confirmation


@pytest.fixture
def frozen(tmp_path):
    series = [f's{i:02d}' for i in range(24)]
    panel = tmp_path/'panel.json'
    panel.write_text(json.dumps({'splits': {'confirmation': {'series': series}}}))
    freeze = {'schema': 'ledger_confirmation_freeze/1', 'implementation_files': confirmation.implementation_files(),
              'gnomon_source_sha256': confirmation.build_info()['source_sha256'],
              'panel_manifest': str(panel), 'panel_manifest_sha256': confirmation.digest(panel),
              'series': series, 'rounds': list(range(26)), 'seeds_requested': [7, 19],
              'expected_decisions': 3744, 'arms': ['no_ledger', 'ledger_119', 'ledger_supported']}
    path = tmp_path/'freeze.json'
    path.write_text(json.dumps(freeze))
    return path, freeze, panel


def test_freeze_validation_reads_only_manifest_not_outcomes(frozen):
    path, freeze, _ = frozen
    # The fixture contains no data path or target observations at all.
    assert confirmation.validate(path)[0] == freeze


@pytest.mark.parametrize('field,value', [('rounds', [0, 8, 17, 25]), ('seeds_requested', [7]),
                                       ('expected_decisions', 192), ('arms', ['no_ledger', 'ledger_supported']),
                                       ('gnomon_source_sha256', 'changed'), ('implementation_files', {})])
def test_confirmation_rejects_changed_implementation_or_partial_cohort(frozen, field, value):
    path, freeze, _ = frozen
    freeze[field] = value
    path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError):
        confirmation.validate(path)


def test_confirmation_rejects_changed_partition(frozen):
    path, _, panel = frozen
    panel.write_text('{}')
    with pytest.raises(ValueError, match='manifest changed'):
        confirmation.validate(path)


def test_confirmation_grid_validation_rejects_duplicates_and_wrong_forecasts():
    freeze = {'series': ['s'], 'rounds': [0], 'candidate_names': ['p'], 'horizon': 1}
    case = {'series_id': 's', 'round': 0, 'origin': '2026-01-01T00:00:00Z',
            'actual': [1], 'future_timestamps': ['2026-01-02T00:00:00Z'], 'predictions': {'p': [1]}}
    confirmation.validate_cases([case], freeze)
    with pytest.raises(ValueError, match='unique'):
        confirmation.validate_cases([case, case], freeze)
    with pytest.raises(ValueError, match='candidates'):
        confirmation.validate_cases([{**case, 'predictions': {'other': [1]}}], freeze)
