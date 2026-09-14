"""Verify088 against the saved synthetic public-runtime database, read-only."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from .agent_review import review as old_review
from .visible_agent_review import review, visible_records
from benchmarks.hermes_ml_checkpoint_v4 import dev_evidence_summary


class Reads:
    def __init__(self, db):
        self.db = db
        self.calls = Counter()

    def execution(self, *args, **kwargs):
        self.calls['execution'] += 1
        return self.db.execution(*args, **kwargs)

    def compare_history(self, *args, **kwargs):
        self.calls['compare_history'] += 1
        return self.db.compare_history(*args, **kwargs)

    def actuals_as_of(self, *args, **kwargs):
        self.calls['actuals_as_of'] += 1
        return self.db.actuals_as_of(*args, **kwargs)


def run(source, output):
    from gnomon import TemporalLedger
    from gnomon.build_info import build_info
    if importlib.metadata.version('gnomon-forecast') != '1.2.0':
        raise ValueError('Pinned1.2.0 required')
    build = build_info()
    if build['commit'] != 'a38cd0cad35383e5f10021abf3aa20d4c16923be':
        raise ValueError('Wrong runtime build')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    source = Path(source)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    inventory = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    records = json.loads((source / 'records.json').read_text())
    task = json.loads((source / 'task.json').read_text())
    before = json.loads((source / 'before.json').read_text())
    db = Reads(TemporalLedger(source / 'base/ledger.db'))
    checks = []

    def require(condition, name):
        checks.append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)

    require(len(records) == 11, 'eleven frozen synthetic forecasts')
    visible, excluded, reads = visible_records(db, records, task)
    require(len(visible) == 9 and len(excluded) == 2 and reads == 11, 'future-recorded events excluded')
    packets = []
    for offset in range(3):
        count = sum(db.calls.values())
        packet = review(db, records, task, dev_evidence_summary, root / f'full-{offset}.json',
                        limit=1, offset=offset)
        require(packet['ledger_queries'] == sum(db.calls.values()) - count, f'page{offset} exact read accounting')
        require(packet['provider_calls'] == 0, f'page{offset} no provider calls')
        require(packet['full_evidence']['sha256'] == sha(root / f'full-{offset}.json'), f'page{offset} full hash')
        require(set(packet['configuration_index']) == {'a', 'b', 'c'}, f'page{offset} hidden providers absent')
        require(packet['pagination']['total_pairs'] == 3, f'page{offset} visible pair count')
        # Independently compare with the old adapter on only the original nine
        # records. The underlying database still physically contains later data.
        expected = old_review(db, records[:9], task, dev_evidence_summary, root / f'expected-{offset}.json',
                              limit=1, offset=offset)
        require(packet['cards'] == expected['cards'], f'page{offset} exact original cards')
        require(packet['pagination'] == expected['pagination'], f'page{offset} exact original pagination')
        require(packet['global_past_origin_count'] == expected['global_past_origin_count'],
                f'page{offset} original global windows')
        packets.append(packet)
    full = json.loads((root / 'full-0.json').read_text())
    require(full['cards'] == before['cards'], 'cards unchanged from before late insertions')
    require(full['global_past_origins'] == before['global_past_origins'], 'origin membership unchanged')
    require(packets[0]['status'] == 'evidence_available', 'first page remains useful')
    require(packets[2]['pagination']['next_call'] is None, 'pagination terminates')
    inclusive_task = {**task, 'origin': '2026-01-30T00:00:00+00:00'}
    inclusive, excluded, _ = visible_records(db, records, inclusive_task)
    require(len(inclusive) == 11 and not excluded, 'recording boundary inclusive')
    late = review(db, records, inclusive_task, dev_evidence_summary, root / 'inclusive.json',
                  pair=['late_d', 'late_e'])
    require(late['status'] == 'insufficient_evidence', 'visible late forecasts do not become eligible scores')
    late_full = json.loads((root / 'inclusive.json').read_text())
    require(bool(late_full['cards'][0]['excluded']), 'public comparison explains late forecast exclusion')
    bad = deepcopy(records[:1])
    bad[0]['request']['unit'] = 'kg'
    rejected = False
    try:
        review(db, bad, task, dev_evidence_summary, root / 'must-not-exist.json')
    except ValueError as exc:
        rejected = 'unit' in str(exc)
    require(rejected and not (root / 'must-not-exist.json').exists(), 'tampering rejected before evidence save')
    saved_hash = sha(root / 'full-0.json')
    try:
        review(db, records, task, dev_evidence_summary, root / 'full-0.json', limit=1)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Existing evidence was overwritten')
    require(sha(root / 'full-0.json') == saved_hash, 'existing evidence immutable')
    require(inventory == {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()},
            'all original source artifacts and ledger byte-identical')
    report = {'passed': True, 'checks': checks, 'check_count': len(checks),
              'runtime_build': build, 'source_files': inventory,
              'public_reads': dict(db.calls), 'new_provider_calls': 0, 'api_calls': 0,
              'protected_access': False, 'accuracy_claim': False}
    for name, value in [('report', report), ('packets', packets)]:
        (root / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    return report


if __name__ == '__main__':
    print(json.dumps(run(*sys.argv[1:]), indent=2))
