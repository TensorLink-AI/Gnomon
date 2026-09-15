"""Undeployed common workflow reminders derived solely from agent-visible files.

This supplies no model choice, accuracy evidence, execution, or extra budget.
It is intended for a new frozen three-arm experiment, never a live-run patch.
"""
import csv
import json
from pathlib import Path


def workflow_progress(work, request_number, remaining_seconds):
    """Describe unfinished comparison steps before exploration closes.

    Only requests 4, 8 and 11 receive a reminder. The original 12/16 request
    split and 90-second protected phase are unchanged; reminders cannot reopen
    exploration. Counts concern current-origin recorded folds, not score quality.
    """
    if request_number not in (4, 8, 11) or remaining_seconds is not None and remaining_seconds <= 90:
        return None
    work = Path(work)
    result = {
        'kind': 'common_workflow_progress', 'request_number': request_number,
        'exploration_requests_including_this_one': 13-request_number,
        'total_requests_including_this_one': 17-request_number,
        'extra_requests_granted': 0, 'fits_executed_by_reminder': 0,
        'scope': 'Current task and recorded workflow progress; not model advice or a verified forecast result.',
    }
    try:
        task = json.loads((work/'task.json').read_text())
        with (work/'history.csv').open(newline='') as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            times = [row['timestamp'] for row in reader]
        if len(times) != 730:
            raise ValueError('Unexpected history length')
        expected = {times[i-1] for i in (688, 702, 716)}
        folds, configs = {}, {}
        selection_after_comparison = False
        attempts = 0
        events = work/'experiments.jsonl'
        if events.exists():
            with events.open() as stream:
                for line in stream:
                    event = json.loads(line)
                    if event.get('task_origin') != task['origin']:
                        continue
                    if event['event'] == 'attempt':
                        attempts += 1
                    elif event['event'] == 'result' and event.get('kind') == 'backtest':
                        cid = event['config_id']
                        folds.setdefault(cid, set()).add(event['request']['cutoff'])
                        configs[cid] = event['config']
                    elif event['event'] == 'selection':
                        selection_after_comparison = event.get('selection_after_comparison') is True
        eligible = {cid for cid, origins in folds.items() if origins == expected}
        ml = any(configs[cid]['model'] in ('ridge', 'random_forest') for cid in eligible)
        baseline = any(configs[cid]['model'] == 'seasonal' for cid in eligible)
        comparison = len(eligible) >= 2 and ml
        result.update(history_columns=columns, target_column='value',
            complete_current_configurations=len(eligible), ml_configuration_compared=ml,
            baseline_compared=baseline, numerical_attempts=attempts,
            numerical_attempts_remaining=max(0, 60-attempts),
            explicit_selection_after_comparison_recorded=selection_after_comparison)
        if not baseline:
            result['next_step'] = 'Create the baseline with lab operation=start before further optional inspection.'
        elif not comparison:
            result['next_step'] = ('Use lab operation=backtest with an ML configuration you choose from the declared bounds. '
                                   'Choose its settings from your available evidence; this reminder chooses none.')
        elif not selection_after_comparison:
            result['next_step'] = ('Use lab operation=commit to select a tested configuration or execution_id after comparison. '
                                   'You choose which forecast; lower CV error is not a guarantee about future outcomes.')
        else:
            result['next_step'] = 'Comparison and explicit selection are recorded; further iteration is optional within the unchanged budget.'
        result['progress_available'] = True
    except (OSError, ValueError, TypeError, KeyError):
        result.update(progress_available=False,
            next_step='Use lab operation=status to inspect the current workflow; progress could not be read safely.')
    result['guidance'] = ('Complete baseline, an ML comparison, and explicit selection before exploration closes. '
        'Source-code reading and optional data exploration consume the same request budget. '
        'The remaining protected phase cannot backtest a new ML configuration. '
        'Keep the selected forecast and concise decision record; do not infer causal explanations from zeros.')
    return result
