"""Arm-independent prospective orchestration limits; no target or model selection."""
REQUEST_LIMIT = 16
SELECTION_REQUESTS = 4
EXPLORATION_REQUESTS = REQUEST_LIMIT - SELECTION_REQUESTS
TIME_LIMIT = 480
SELECTION_SECONDS = 90
MAX_CORRECTIONS = 2


def phase(forwarded, seconds_remaining=None):
    if forwarded > EXPLORATION_REQUESTS or (
        seconds_remaining is not None and seconds_remaining <= SELECTION_SECONDS
    ):
        return 'selection'
    return 'exploration'


SELECTION_NOTICE = (
    'Protected final-selection phase: stop exploring new configurations. '
    'Use your existing backtests to explicitly select a configuration with '
    '`python lab.py commit --config ...` or an exact existing execution ID. '
    'If needed, `python lab.py status` shows your evidence. A status call alone '
    'does not select anything. You may keep the baseline by explicitly committing '
    'it after comparison. If no checkpoint exists, `python lab.py start` can '
    'save a baseline, but baseline-only survival is not full completion. '
    'No new exploratory backtests are admitted in this phase. '
    'Do not infer success from final prose; checkpoint.json is authoritative.'
)


def correction_reason(result, full):
    if full:
        return None
    if 'repetition' in str(result.get('error', '')).lower():
        return 'repetitive_output'
    if result.get('failed') or result.get('error'):
        return None  # Runtime/service failures are not silently retried as agent repair.
    if result.get('turn_exit_reason', '').startswith('text_response'):
        return 'premature_text_termination'
    return None


def correction_prompt(reason):
    return (
        f'Workflow correction ({reason}): the task has not completed. '
        'Your previous response did not establish a complete selected checkpoint. '
        'Use actual terminal tool calls, not printed tool markup. '
        'Inspect `python lab.py status`; if needed run `start`, compare an ML '
        'configuration while exploration remains open, and explicitly `commit` '
        'your backtested choice. Preserve prior evidence and checkpoints. '
        'This continuation uses the SAME remaining request and time budget. '
        'Do not redo completed work or infer a selection from prose.'
    )
