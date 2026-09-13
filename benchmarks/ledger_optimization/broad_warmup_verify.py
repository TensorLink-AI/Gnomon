"""Audit earlier spans against unchanged scored identities and temporal boundaries."""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path


def audit(panel, output):
    panel, output = Path(panel), Path(output)
    destination = output/'verification.json'
    if destination.exists():raise FileExistsError(destination)
    original = json.loads((panel/'development-spans.json').read_text())
    selection = json.loads((panel/'selection.json').read_text())
    spans = json.loads((output/'warmup-spans.json').read_text())
    tasks = json.loads((output/'boundaries.json').read_text())
    completed = json.loads((output/'COMPLETED.json').read_text())
    receipt = json.loads(Path(__file__).with_name('evidence').joinpath('broad-panel-037.json').read_text())
    checks = 0
    def check(ok, message):
        nonlocal checks
        checks += 1
        if not ok:raise AssertionError(message)
    for name in ('development-spans.json', 'selection.json'):
        check(hashlib.sha256((panel/name).read_bytes()).hexdigest() == receipt['files'][name], 'original panel hash')
    check(set(spans) == set(original) and len(spans) == 16, 'all scored series retained')
    reserved = {source+':'+r['series_name'] for source, groups in selection.items() for r in groups['reserved']}
    check(not set(spans)&reserved, 'reserved identities excluded')
    for series, span in spans.items():
        values = span['values']; before = datetime.fromisoformat(span['start_label'])
        first = datetime.fromisoformat(original[series]['first_origin'])
        check(len(values) == 2074, 'preperiod size')
        check(all(v is None or (math.isfinite(v) and v >= 0) for v in values), 'valid observed counts or explicit missing')
        check(values[-730:] == original[series]['values'][:730], 'scored-history overlap unchanged')
        check(span['unit'] == original[series]['unit'], 'unit')
        check(span['source_sha256'] == original[series]['source_sha256'], 'source identity')
        check(before+timedelta(hours=2074) == first, 'preperiod endpoint')
        check(span['first_scored_origin'] == original[series]['first_origin'], 'unchanged origin')
        own = sorted([t for t in tasks if t['series_id'] == series], key=lambda t: t['round'])
        check([t['round'] for t in own] == list(range(-8, 0)), 'all warmup attempts retained')
        for task in own:
            i = task['round']+8; stop = 730+168*i
            check(task['history_indices'] == [stop-730, stop], 'full history slice')
            check(task['target_indices'] == [stop, stop+24], 'full target slice')
            check(datetime.fromisoformat(task['origin']) == before+timedelta(hours=stop), 'origin coordinate')
            close = datetime.fromisoformat(task['last_target'])
            check(close == before+timedelta(hours=stop+24) and close < first, 'targets close before scored period')
            missing_history = sum(v is None for v in values[stop-730:stop])
            missing_target = sum(v is None for v in values[stop:stop+24])
            check(task['missing_history'] == missing_history, 'missing history accounting')
            check(task['missing_targets'] == missing_target, 'missing target accounting')
            check(task['ready'] == (missing_history+missing_target == 0), 'entire cohort admissibility')
    check(len(tasks) == 128, '128 proposed warmup tasks')
    check(completed['usable_warmup_tasks'] == sum(t['ready'] for t in tasks), 'usable count')
    check(completed['unavailable_warmup_tasks'] == sum(not t['ready'] for t in tasks), 'unavailable count')
    check(completed['scored_tasks_unchanged'] == 416, 'scored cohort count')
    check(hashlib.sha256((output/'warmup-spans.json').read_bytes()).hexdigest() == completed['spans_sha256'], 'export hash')
    result = {'checks': checks, 'failures': 0,
        'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Fixed identities, source hashes, original overlap, every warmup history/target boundary, missing counts and cohort admissibility. Does not establish actual historical publication times.',
        'ready_at_first_scored_origin': {s: sum(t['ready'] for t in tasks if t['series_id'] == s) for s in sorted(spans)}}
    destination.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel'); parser.add_argument('output')
    args = parser.parse_args()
    print(json.dumps(audit(args.panel, args.output), indent=2))
