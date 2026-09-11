"""Freeze a validated candidate and reserve one untouched confirmation attempt."""
import argparse
import hashlib
import json
from pathlib import Path

from .scenario import canonical

RESERVED_SEEDS = list(range(9000, 9024))
REGISTRY = Path(__file__).resolve().parents[2] / 'results/experience-workflow/confirmation-9000-9023.used.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_receipt(validation, audit):
    from .agent import source_manifest
    from .report import summarize
    manifest = json.loads((validation / 'manifest.json').read_text())
    rows = json.loads((validation / 'decisions.json').read_text())
    audit_report = json.loads(audit.read_text())
    source = source_manifest()
    if (sorted(manifest['seeds']) != [200, 201, 202, 203] or sorted(manifest['agent_seeds']) != [7, 19]
            or manifest['rounds'] != 24 or manifest['expected_decisions'] != 384):
        raise ValueError('Freeze requires the full preregistered validation grid, not selected pilot checkpoints')
    if manifest['source'] != source or audit_report.get('source') != source:
        raise ValueError('Validation and deterministic audit must exercise the current unchanged source')
    if not json.loads((validation / 'source_audit.json').read_text())['unchanged']:
        raise ValueError('Source changed during validation')
    result = summarize(rows, manifest)
    gates = result['gates']
    if any(gates[k] != 'met_in_this_sample' for k in ('complete_paired_run', 'completion_floor', 'cost_point')):
        raise ValueError('Validation must complete with >=95% correctness and <=0.80 token-per-correct ratio')
    if result['point']['rmsle_ratio'] is None or result['point']['rmsle_ratio'] > 1.02:
        raise ValueError('Validation forecast quality misses the preregistered noninferiority point threshold')
    if not audit_report['passed'] or not all(audit_report['mutation_failures_detected'].values()) or sum(
            w.get('feature_visibility_checks', 0) for w in audit_report['worlds']) < 864:
        raise ValueError('Complete independent parity, feature visibility and mutation audits are required')
    references = [validation / name for name in ('manifest.json', 'decisions.json', 'source_audit.json')] + [audit]
    receipt = dict(kind='experience-workflow-confirmation/1', source=source,
        seeds=RESERVED_SEEDS, agent_seeds=[7, 19], rounds=24,
        validation_run=str(validation.resolve()), audit_path=str(audit.resolve()),
        expected_decisions=2304, prerequisites_passed=True,
        prerequisites={str(p.resolve()): sha(p) for p in references})
    return receipt


def create(validation, audit, output):
    if output.exists() or REGISTRY.exists():
        raise ValueError('Use a new freeze path; the reserved cohort must be unspent')
    receipt = _validated_receipt(validation, audit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(canonical(receipt) + '\n')
    return receipt


def consume(path, output):
    """Read-only verification, then one exclusive global cohort claim before generation."""
    from .agent import source_manifest
    receipt = json.loads(path.read_text())
    if (receipt.get('kind') != 'experience-workflow-confirmation/1' or receipt.get('seeds') != RESERVED_SEEDS
            or receipt.get('agent_seeds') != [7, 19] or receipt.get('rounds') != 24
            or not receipt.get('prerequisites_passed') or len(receipt.get('prerequisites', {})) != 4):
        raise ValueError('Invalid confirmation freeze')
    if receipt['source'] != source_manifest():
        raise ValueError('Code/prompts/protocol changed after freeze; confirmation refused')
    for name, expected in receipt['prerequisites'].items():
        if sha(Path(name)) != expected:
            raise ValueError('Frozen prerequisite artifact changed')
    if receipt != _validated_receipt(Path(receipt['validation_run']), Path(receipt['audit_path'])):
        raise ValueError('Frozen validation no longer satisfies the registered prerequisites')
    if output.exists():
        raise ValueError('Confirmation output directory must be new')
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open('x') as stream:
        stream.write(canonical(dict(freeze=str(path.resolve()), freeze_sha256=sha(path),
                                   run=str(output.resolve()), status='cohort_consumed_before_dispatch')) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-run', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(args.validation_run, args.audit, args.output), indent=2))


if __name__ == '__main__':
    main()
