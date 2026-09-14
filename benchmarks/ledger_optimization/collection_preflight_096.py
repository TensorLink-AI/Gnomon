"""Adapt fully audited collection worker evidence to the frozen pilot interface.

Read-only verification; writes only a new acceptance receipt. Does not authorize
paid dispatch, read credentials, change runtimes, or execute forecast providers.
"""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accept(capsule, evidence, receipt, output):
    capsule, evidence, receipt, output = map(Path, (capsule, evidence, receipt, output))
    if output.exists() or output.is_symlink():
        raise ValueError('Fresh preflight output required')
    trusted = json.loads(receipt.read_text())
    if trusted['status'] != 'local_exact_runtime_hermes_collection_integration_passed':
        raise ValueError('Require completed collection worker integration')
    hashes = trusted['files']
    def pinned(name):
        candidates = [h for p,h in hashes.items() if Path(p).name == name]
        if len(candidates) != 1 or digest(evidence/name) != candidates[0]:
            raise ValueError('Worker evidence hash mismatch: '+name)
        return json.loads((evidence/name).read_text())
    passed, report, files = pinned('passed.json'), pinned('report.json'), pinned('SHA256SUMS.json')
    for name, expected in files.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or (evidence/rel).is_symlink() or digest(evidence/rel) != expected:
            raise ValueError('Retained worker evidence changed: '+name)
    manifest=json.loads((capsule/'capsule.json').read_text())
    sources=manifest['sources']
    package=capsule/'benchmarks/hermes_ml_checkpoint_v6'
    actual={p.name:digest(p) for p in package.iterdir() if p.is_file()}
    if sources != actual or sources != passed['sources']:
        raise ValueError('Current capsule differs from tested worker sources')
    if (passed.get('passed') is not True or not passed['checks']
            or not all(c['passed'] is True for c in passed['checks'])
            or passed['engy_calls'] != 0 or passed['numerical_attempts'] != 48
            or passed['scripted_model_responses'] != 30):
        raise ValueError('Worker checks incomplete or execution accounting differs')
    if report.get('complete') is not True or report['audit_failures'] or report['shutdown_record_gaps']:
        raise ValueError('Independent audit incomplete')
    rows=report['rows']
    expected={(a,n) for a in ('plain','gnomon','ledger') for n in (0,1)}
    if len(rows)!=6 or {(r['arm'],r['round']) for r in rows}!=expected:
        raise ValueError('Require all six matched synthetic workflows')
    for row in rows:
        if not (row['valid'] and row['workflow_complete'] and row['api_calls']==5
                and row['numerical_attempts']==8 and row['collection_audit']['production_results']==2):
            raise ValueError('Incomplete collection workflow')
    tested_runtime=json.loads((evidence/'manifest.json').read_text())
    if (tested_runtime['build']['package_version']!='1.2.0'
            or tested_runtime['build']['source_sha256']!='9723394ccb6d9e11991b312e01bac47c767c69407b6b33d36971cb6e48b6a22e'):
        raise ValueError('Wrong published runtime')
    result={'passed':True,'tested_sources':sources,'checks':passed['checks'],
            'engy_calls':0,'synthetic_upstream_responses':30,'numerical_attempts':48,
            'verification_new_fits':0,'independent_audit_checks':report['audit_checks'],
            'tested_inventory':tested_runtime['inventory'],'tested_build':tested_runtime['build'],
            'integration_receipt_sha256':digest(receipt),'evidence_inventory_sha256':digest(evidence/'SHA256SUMS.json'),
            'files_verified':len(files),'dispatch_authorized':False,
            'scope':'Verified existing full Hermes integration; compatible receipt for the frozen pilot runner. Terminal predecessor, current runtime, source, fresh-state and paid launch gates still apply.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    return result
