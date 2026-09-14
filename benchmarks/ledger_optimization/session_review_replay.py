"""Replay090 against copies of original030 session logs and public ledgers."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

from .agent_review import brief
from .agent_review_audit import INVENTORY_SHA
from .visible_agent_review import review
from benchmarks.hermes_ml_checkpoint_v4 import dev_evidence_summary


def run(source, prepared, output):
    from gnomon import TemporalLedger, InferenceEngine
    from gnomon.build_info import build_info
    build = build_info()
    if importlib.metadata.version('gnomon-forecast') != '1.2.0' or build['commit'] != 'a38cd0cad35383e5f10021abf3aa20d4c16923be':
        raise ValueError('Exact published1.2.0runtime required')
    source, prepared, output = Path(source), Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(source / 'SHA256SUMS.json') != INVENTORY_SHA:
        raise ValueError('Original inventory mismatch')
    inventory = json.loads((source / 'SHA256SUMS.json').read_text())
    prepared_manifest = json.loads((prepared / 'manifest.json').read_text())
    if digest(prepared / 'cases.json') != prepared_manifest['cases_sha256']:
        raise ValueError('Prepared cases changed')
    cases = json.loads((prepared / 'cases.json').read_text())
    source_access = {}
    def original(name):
        path = source / name
        if name not in inventory or digest(path) != inventory[name]:
            raise ValueError('Original source mismatch')
        source_access[name] = inventory[name]
        return path
    results = []
    try:
        with patch.object(InferenceEngine, 'forecast', side_effect=AssertionError('Replay executed a provider')):
            for series in sorted({c['series_id'] for c in cases}):
                prefix = f'evaluation/ledger/{series}/round-25/project/'
                log = original(prefix + 'experiments.jsonl')
                records = [json.loads(line) for line in log.read_text().splitlines()]
                ledger = original(prefix + 'ledger.db')
                folder = output / series
                folder.mkdir()
                copy = folder / 'ledger.db'
                shutil.copyfile(ledger, copy)
                db = TemporalLedger(copy)
                opened_hash = digest(copy)
                for case in (c for c in cases if c['series_id'] == series):
                    path = original(case['source'])
                    full = json.loads(path.read_text())
                    task = case['formats']['brief']['query']
                    expected = brief(full, task, case['formats']['brief']['full_evidence']['path'], digest(path))
                    target = folder / (case['case_id'] + '.json')
                    actual = review(db, records, task, dev_evidence_summary, target,
                                    offset=full['offset'], limit=12)
                    fields = ('cards', 'configuration_index', 'pagination', 'query', 'status',
                              'global_past_origin_count', 'provider_calls')
                    mismatches = [field for field in fields if actual[field] != expected[field]]
                    outcome = {'case_id': case['case_id'], 'passed': not mismatches, 'mismatched_fields': mismatches,
                               'ledger_queries': actual['ledger_queries'], 'catalog_visibility': actual['catalog_visibility'],
                               'full_sha256': digest(target), 'provider_calls': actual['provider_calls']}
                    results.append(outcome)
                    (folder / (case['case_id'] + '-brief.json')).write_text(json.dumps(actual, indent=2) + '\n')
                    if mismatches:
                        raise AssertionError('Original query differs: ' + ', '.join(mismatches))
                    if digest(copy) != opened_hash:
                        raise AssertionError('Query changed working ledger bytes')
                    print(json.dumps(outcome), flush=True)
        for name, hash_ in source_access.items():
            if digest(source / name) != hash_:
                raise AssertionError('Original source changed')
        report = {'passed': True, 'queries': len(results), 'results': results, 'source_access': source_access,
                  'runtime': build, 'new_forecasts': 0, 'api_calls': 0, 'protected_access': False,
                  'accuracy_claim': False, 'original_source_unchanged': True}
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        return report
    except Exception as exc:
        (output / 'failure.json').write_text(json.dumps({'error_type': type(exc).__name__, 'error': str(exc),
            'results_so_far': results, 'source_access': source_access}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    run(*sys.argv[1:])
