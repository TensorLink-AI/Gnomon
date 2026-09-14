"""091: paired public/patched history queries with independent scored-pair audit."""
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

from . import history_091
from .agent_review_audit import INVENTORY_SHA
from .visible_agent_review import visible_records
from .ml_ledger_cards import development_cards
from benchmarks.hermes_ml_checkpoint_v4 import dev_evidence_summary


class Query:
    def __init__(self, db, modified=False):
        self.db = db
        self.modified = modified
        self.calls = Counter()

    def execution(self, *args, **kwargs):
        self.calls['execution'] += 1
        return self.db.execution(*args, **kwargs)

    def actuals_as_of(self, *args, **kwargs):
        self.calls['actuals_as_of'] += 1
        return self.db.actuals_as_of(*args, **kwargs)

    def compare_history(self, **kwargs):
        self.calls['compare_history'] += 1
        return history_091.compare_history(self.db, **kwargs) if self.modified else self.db.compare_history(**kwargs)


def run(source, prepared, output):
    import gnomon
    from gnomon import TemporalLedger, InferenceEngine
    from gnomon.build_info import build_info
    build = build_info()
    if importlib.metadata.version('gnomon-forecast') != '1.2.0' or build['commit'] != 'a38cd0cad35383e5f10021abf3aa20d4c16923be':
        raise ValueError('Exact published1.2.0storage/engine required')
    started = time.monotonic()
    source, prepared, output = Path(source), Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(source / 'SHA256SUMS.json') != INVENTORY_SHA:
        raise ValueError('Original inventory mismatch')
    inventory = json.loads((source / 'SHA256SUMS.json').read_text())
    frozen = json.loads((prepared / 'manifest.json').read_text())
    if sha(prepared / 'cases.json') != frozen['cases_sha256']:
        raise ValueError('Frozen sixteen queries changed')
    cases = json.loads((prepared / 'cases.json').read_text())
    access = {}
    checks = []
    results = []
    totals = Counter()
    verification_reads = Counter()
    def check(ok, name):
        checks.append({'check': name, 'passed': bool(ok)})
        if not ok:
            raise AssertionError(name)
    def read(name):
        if name not in inventory or sha(source / name) != inventory[name]:
            raise ValueError('Original source mismatch: ' + name)
        access[name] = inventory[name]
        return source / name
    stamp = datetime.fromisoformat
    def pairs(full):
        return {(c['left_config_id'], c['right_config_id']): c for c in full['cards']}
    def origin_map(card):
        return {stamp(o['origin']): o for o in card['windows']['lifetime']['origins']}
    try:
        with patch.object(InferenceEngine, 'forecast', side_effect=AssertionError('Query executed a provider')):
            for series in sorted({c['series_id'] for c in cases}):
                folder = output / series
                folder.mkdir()
                prefix = f'evaluation/ledger/{series}/round-25/project/'
                records = [json.loads(line) for line in read(prefix + 'experiments.jsonl').read_text().splitlines()]
                dbfile = folder / 'ledger.db'
                shutil.copyfile(read(prefix + 'ledger.db'), dbfile)
                db = TemporalLedger(dbfile)
                initialized_hash = sha(dbfile)
                for case in (c for c in cases if c['series_id'] == series):
                    old_saved = json.loads(read(case['source']).read_text())
                    task = case['formats']['brief']['query']
                    index_reader = Query(db)
                    visible, excluded, read_count = visible_records(index_reader, records, task)
                    check(read_count == index_reader.calls['execution'], 'Catalogue execution-read accounting')
                    totals.update(index_reader.calls)
                    controls, changed = Query(db), Query(db, modified=True)
                    before = development_cards(controls, visible, task, dev_evidence_summary,
                                               offset=old_saved['offset'], limit=12)
                    after = development_cards(changed, visible, task, dev_evidence_summary,
                                              offset=old_saved['offset'], limit=12)
                    totals.update(controls.calls)
                    totals.update(changed.calls)
                    for name, value in [('baseline', before), ('modified', after)]:
                        (folder / f'{case["case_id"]}-{name}.json').write_text(json.dumps(value, indent=2) + '\n')
                    check(before['provider_calls'] == after['provider_calls'] == 0, 'No provider calls')
                    for field in ('configuration_index', 'global_past_origins', 'total_pairs', 'offset', 'next_offset'):
                        check(before[field] == after[field] == old_saved[field], 'Same catalogue/page: ' + field)
                    new_pairs = pairs(after)
                    check(new_pairs.keys() == pairs(before).keys() == pairs(old_saved).keys(), 'Same queried pairs')
                    preserved_saved = preserved_baseline = recovered = saved_gain = 0
                    for key, card in new_pairs.items():
                        now = origin_map(card)
                        previous = origin_map(pairs(before)[key])
                        original = origin_map(pairs(old_saved)[key])
                        check(previous.keys() <= now.keys(), 'No baseline scored origin lost')
                        check(original.keys() <= now.keys(), 'No original saved scored origin lost')
                        for old in (previous, original):
                            for origin, value in old.items():
                                check(value == now[origin], 'Existing scored origin/evidence/metrics unchanged')
                        preserved_baseline += len(previous)
                        preserved_saved += len(original)
                        recovered += len(now.keys() - previous.keys())
                        saved_gain += len(now.keys() - original.keys())
                    # Independent scored-pair reconstruction from authoritative
                    # executions and cutoff-filtered actuals, without using the
                    # proposed eligibility helper or its metric implementation.
                    actuals = db.actuals_as_of(series, unit=task['unit'], source_as_of=task['origin'], recorded_as_of=task['origin'])
                    verification_reads['actuals_as_of'] += 1
                    by_id = {a['actual_id']: a for a in actuals}
                    executions = {}
                    audited = set()
                    for card in after['cards']:
                        for window in card['windows'].values():
                            check(window['matched_origins'] == len(window['origins']), 'Window origin count')
                            for origin in window['origins']:
                                actual = [by_id[aid] for aid in origin['actual_ids']]
                                by_time = {stamp(a['valid_time']): a['value'] for a in actual}
                                check(len(by_time) == len(actual) == task['horizon'] == origin['n'], 'Complete unique actual horizon')
                                for a in actual:
                                    check(stamp(a['recorded_at']) <= stamp(task['origin']) and stamp(a['source_available_at']) <= stamp(task['origin']), 'Actual cutoffs respected')
                                    check(a['series_id'] == series and a['unit'] == task['unit'] and a['value'] >= 0, 'Actual task identity')
                                requests = []
                                for model in origin['models']:
                                    eid = model['execution_id']
                                    if eid not in executions:
                                        executions[eid] = db.execution(eid)
                                        verification_reads['execution'] += 1
                                    e = executions[eid]
                                    r = e['request']
                                    requests.append(r)
                                    future = [stamp(t) for t in r['future_timestamps']]
                                    check(stamp(e['recorded_at']) < min(future) and stamp(e['recorded_at']) <= stamp(task['origin']), 'Scored execution ex-ante and recording-visible')
                                    check(stamp(r['cutoff']) == stamp(origin['origin']) <= stamp(e['recorded_at']), 'Forecast origin bound')
                                    check(max(stamp(t) for t in r['timestamps']) <= stamp(origin['origin']), 'No future history timestamps')
                                    check(not r['known_time_cutoff'] or stamp(r['known_time_cutoff']) <= stamp(origin['origin']), 'No future source cutoff')
                                    check(not r['recorded_time_cutoff'] or stamp(r['recorded_time_cutoff']) <= stamp(e['recorded_at']), 'No future recording cutoff')
                                    check(r['series_id'] == series and r['unit'] == task['unit'] and r['horizon'] == task['horizon'], 'Execution task identity')
                                    check(e['provider'] == model['provider'] and e['revision'] == model['revision'], 'Provider revision bound')
                                    check(set(future) == set(by_time) and len(future) == len(e['result']['point']), 'Prediction target identity')
                                    computed = math.sqrt(sum((math.log1p(max(0, p)) - math.log1p(by_time[t])) ** 2
                                                         for p, t in zip(e['result']['point'], future)) / len(future))
                                    check(abs(computed - model['rmsle']) <= 1e-12, 'Independently reconstructed RMSLE')
                                    audited.add((eid, tuple(origin['actual_ids'])))
                                check(all(r == requests[0] for r in requests), 'Models have matched request inputs')
                    outcome = {'case_id': case['case_id'], 'baseline_pair_origins': preserved_baseline,
                               'original_saved_pair_origins': preserved_saved,
                               'modified_pair_origins': preserved_baseline + recovered,
                               'recovered_from_baseline': recovered, 'gained_over_original_saved': saved_gain,
                               'distinct_execution_actual_pairs_audited': len(audited),
                               'baseline_status': before['status'], 'modified_status': after['status'],
                               'recording_excluded_events': len(excluded)}
                    results.append(outcome)
                    check(sha(dbfile) == initialized_hash, 'Queries leave working ledger byte-identical')
                    print(json.dumps(outcome), flush=True)
        for name, hash_ in access.items():
            check(sha(source / name) == hash_, 'Original source byte-identical')
        summary = {'passed': True, 'queries': len(results), 'results': results, 'check_count': len(checks),
                   'query_reads': dict(totals), 'independent_verification_reads': dict(verification_reads),
                   'source_access': access, 'runtime': build, 'package_path': gnomon.__file__,
                   'development_history_sha256': sha(Path(history_091.__file__)),
                   'seconds': time.monotonic() - started, 'new_forecasts': 0, 'api_calls': 0,
                   'accuracy_claim': False, 'protected_access': False}
        (output / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')
        (output / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
        return summary
    except Exception as exc:
        (output / 'failure.json').write_text(json.dumps({'error_type': type(exc).__name__, 'error': str(exc),
            'results_so_far': results, 'checks': checks, 'source_access': access}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    run(*sys.argv[1:])
