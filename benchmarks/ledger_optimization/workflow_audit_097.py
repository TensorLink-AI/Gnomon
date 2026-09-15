"""Independent audit of prospective workflow reminders and forwarded requests."""
import hashlib
import json
from pathlib import Path


def audit_workflow_progress(folder, job):
    """Reconstruct each reminder from its hashed append-only visible prefix.

    Does not call the reminder generator. Original payloads, execution prefixes,
    configured budgets and final recorded messages provide independent evidence.
    """
    folder = Path(folder)
    event_path = folder/'project/experiments.jsonl'
    event_bytes = event_path.read_bytes() if event_path.exists() else b''
    expected_folds = {job['request']['timestamps'][i-1] for i in (688, 702, 716)}
    seen, checks = [], 0
    metadata_files = {p.name for p in folder.glob('api-*-workflow-state.json')}
    receipt_files = {p.name for p in folder.glob('api-*-workflow-progress.json')}
    for forwarded_path in sorted(folder.glob('api-*-forwarded.json')):
        number = int(forwarded_path.name.split('-')[1])
        prefix = f'api-{number:02d}'
        original = json.loads((folder/(prefix+'-request.json')).read_text())
        forwarded = json.loads(forwarded_path.read_text())
        state_name = prefix+'-workflow-state.json'
        state = json.loads((folder/state_name).read_text())
        assert state['request_number'] == number
        assert state['seconds_remaining'] is None or isinstance(state['seconds_remaining'], (int, float))
        metadata_files.remove(state_name)
        due = number in (4, 8, 11) and (state['seconds_remaining'] is None or state['seconds_remaining'] > 90)
        receipt_name = prefix+'-workflow-progress.json'
        assert (receipt_name in receipt_files) == due
        messages = list(original['messages'])
        if due:
            receipt = json.loads((folder/receipt_name).read_text())
            receipt_files.remove(receipt_name)
            progress = receipt['progress']
            assert json.loads(receipt['notice']) == progress
            assert progress['request_number'] == number
            assert progress['exploration_requests_including_this_one'] == 13-number
            assert progress['total_requests_including_this_one'] == 17-number
            assert progress['extra_requests_granted'] == progress['fits_executed_by_reminder'] == 0
            assert progress['progress_available'] is True
            length = progress['evidence_prefix_bytes']
            assert type(length) is int and 0 <= length <= len(event_bytes)
            raw = event_bytes[:length]
            assert not raw or raw.endswith(b'\n')
            assert hashlib.sha256(raw).hexdigest() == progress['evidence_prefix_sha256']
            events = [json.loads(line) for line in raw.splitlines()]
            assert len(events) == progress['evidence_events_read']
            current = [e for e in events if e.get('task_origin') == job['origin']]
            model_folds, models = {}, {}
            attempts = sum(e['event'] == 'attempt' for e in current)
            for event in current:
                if event['event'] == 'result' and event.get('kind') == 'backtest':
                    key = event['config_id']
                    model_folds.setdefault(key, set()).add(event['request']['cutoff'])
                    models[key] = event['config']['model']
            complete = [key for key, folds in model_folds.items() if folds == expected_folds]
            assert progress['complete_current_configurations'] == len(complete)
            assert progress['ml_configuration_compared'] == any(models[key] in ('ridge', 'random_forest') for key in complete)
            assert progress['baseline_compared'] == any(models[key] == 'seasonal' for key in complete)
            assert progress['numerical_attempts'] == attempts
            assert progress['numerical_attempts_remaining'] == max(0, 60-attempts)
            selections = [e for e in current if e['event'] == 'selection']
            assert progress['explicit_selection_after_comparison_recorded'] == bool(selections and selections[-1].get('selection_after_comparison') is True)
            assert progress['history_columns'] == ['timestamp', 'value', *job['request']['past_covariate_names']]
            assert progress['target_column'] == 'value'
            # A whitelist prevents unnoticed forecast/target/metric data in an advisory notice.
            assert set(progress) == {'kind', 'request_number', 'exploration_requests_including_this_one',
                'total_requests_including_this_one', 'extra_requests_granted', 'fits_executed_by_reminder',
                'scope', 'history_columns', 'target_column', 'evidence_prefix_bytes', 'evidence_prefix_sha256',
                'evidence_events_read', 'complete_current_configurations', 'ml_configuration_compared',
                'baseline_compared', 'numerical_attempts', 'numerical_attempts_remaining',
                'explicit_selection_after_comparison_recorded', 'next_step', 'progress_available', 'guidance'}
            messages.append({'role': 'system', 'content': receipt['notice']})
            seen.append(number)
            checks += 19
        selection = number >= 13 or state['seconds_remaining'] is not None and state['seconds_remaining'] <= 90
        if selection:
            notice = json.loads((folder/(prefix+'-intervention.json')).read_text())
            assert notice['phase'] == 'selection'
            messages.append({'role': 'system', 'content': notice['notice']})
        expected = {**original, 'messages': messages, 'model': 'deepseek-v4.1-flash',
                    'temperature': .2, 'seed': 7, 'max_tokens': 3072, 'stream': False}
        expected.pop('stream_options', None)
        expected.pop('max_completion_tokens', None)
        assert forwarded == expected
        checks += 5
    assert not metadata_files and not receipt_files
    return {'checks': checks, 'reminder_requests': seen,
            'audited_prefixes': len(seen), 'provider_calls_from_reminders': 0}
