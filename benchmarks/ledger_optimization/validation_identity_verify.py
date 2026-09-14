"""Independently reproduce validation membership from frozen metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def audit(panel, directory):
    panel, directory = Path(panel), Path(directory); output = directory/'verification.json'
    if output.exists():raise FileExistsError(output)
    here = Path(__file__).parent; checks = 0
    def check(ok, message):
        nonlocal checks
        checks += 1
        if not ok:raise AssertionError(message)
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((directory/'manifest.json').read_text())
    check(sha(here/'validation_identity.py') == manifest['code_sha256'], 'code')
    check(sha(here/'BROAD_VALIDATION_052.md') == manifest['protocol_sha256'], 'protocol')
    receipt_path = here/'evidence/broad-panel-037.json'
    check(sha(receipt_path) == manifest['source_receipt_sha256'], 'source receipt')
    receipt = json.loads(receipt_path.read_text())
    for name, digest in manifest['metadata_files'].items():
        check(sha(panel/name) == digest == receipt['files'][name], 'unchanged metadata')
    check(set(manifest['metadata_files']) == {'selection.json', 'electricity-eligibility.json', 'pedestrian-eligibility.json'}, 'metadata-only inputs')
    selection = json.loads((directory/'selection.json').read_text()); previous = json.loads((panel/'selection.json').read_text())
    for domain in ('electricity', 'pedestrian'):
        eligible = json.loads((panel/(domain+'-eligibility.json')).read_text())['eligible']
        ranks = {r['series_name']: hashlib.sha256(('20260914:panel035:'+domain+':'+r['series_name']).encode()).hexdigest() for r in eligible}
        ordered = sorted(ranks, key=ranks.__getitem__)
        chosen = [r['series_name'] for r in selection[domain]['validation']]
        old = [r['series_name'] for r in previous[domain]['development']]
        reserved = [r['series_name'] for r in previous[domain]['reserved']]
        check(chosen == ordered[24:32] and len(chosen) == 8, 'next eight fixed identities')
        check(old == ordered[:8] and reserved == ordered[8:24], 'original partition')
        check(set(chosen).isdisjoint(old+reserved), 'disjoint validation')
        check(selection[domain]['original_development'] == old and selection[domain]['reserved_final_unchanged'] == reserved, 'reserved preserved')
        by_name = {r['series_name']: r for r in eligible}
        for row in selection[domain]['validation']:check(row == by_name[row['series_name']], 'unchanged eligibility evidence')
    check(manifest['locked_implementation_commit'] == '11af992', 'chosen recipe commit')
    for name, digest in manifest['locked_implementation_sha256'].items():
        original = subprocess.run(['git', 'show', '11af992:benchmarks/ledger_optimization/'+name], check=True, capture_output=True).stdout
        check(sha(here/name) == digest == hashlib.sha256(original).hexdigest(), 'locked numerical implementation')
    completed = json.loads((directory/'COMPLETED.json').read_text())
    check(completed['validation_series'] == 16 and completed['planned_scored_cases'] == 416 and completed['reserved_final_series_unchanged'] == 32, 'fixed cohort size')
    check(completed['scored_outcome_values_read'] == completed['reserved_future_values_read'] == 0, 'metadata-only stage')
    check(completed['selection_sha256'] == sha(directory/'selection.json'), 'selection hash')
    result = {'checks': checks, 'failures': 0, 'validation_series': 16, 'reserved_final_series_unchanged': 32,
        'selection_sha256': sha(directory/'selection.json'), 'verifier_sha256': sha(Path(__file__)),
        'scope': 'Independent deterministic ranking, disjoint identities, original reserve, metadata hashes and locked 050 code. No outcome values accessed.'}
    output.write_text(json.dumps(result, indent=2)+'\n'); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('panel'); p.add_argument('directory')
    print(json.dumps(audit(**vars(p.parse_args())), indent=2))
