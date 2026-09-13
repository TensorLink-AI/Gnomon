"""Verify an exact model-only amendment, then exercise the native Hermes handoff.

Unchanged numerical/ledger checks inherit the recorded full preflight. This is
explicitly a differential validation, not a claim those checks were rerun.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from benchmarks.hermes_ml_checkpoint_v4.run import HERE, ARMS, BUILD_SHA
from benchmarks.hermes_ml_checkpoint_v4.transport import MODEL, dump, sha
from benchmarks.hermes_ml_checkpoint_v4.service_admission import PAYLOAD


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline-preflight', type=Path, required=True)
    p.add_argument('--baseline-source', type=Path, required=True)
    p.add_argument('--admission-tests', type=Path, required=True)
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(args.baseline_preflight.read_text())
    assert baseline['passed']
    assert MODEL == PAYLOAD['model'] == 'deepseek-v4.1-flash'
    assert ARMS == ('plain', 'gnomon', 'ledger')
    from gnomon.build_info import build_info
    build = build_info()
    assert build['package_version'] == '1.2.0' and build['source_sha256'] == BUILD_SHA
    frozen = {p.name: sha(p) for p in HERE.iterdir() if p.is_file()}
    assert set(frozen) == set(baseline['tested_sources'])
    changes = []
    for name, digest in baseline['tested_sources'].items():
        original = args.baseline_source / name
        assert sha(original) == digest
        expected = original.read_bytes().replace(b'deepseek-v4-flash-0731', b'deepseek-v4.1-flash')
        assert (HERE / name).read_bytes() == expected, name
        if frozen[name] != digest:
            changes.append(name)
    dump(root / 'differential.json', {
        'only_change': 'Exact model identifier replacement', 'changed_files': changes,
        'baseline_preflight_sha256': sha(args.baseline_preflight),
        'baseline_source': str(args.baseline_source), 'tested_sources': frozen,
        'inherited_checks': baseline['checks'], 'inherited_checks_rerun': False,
        'build': build, 'new_model': MODEL,
    })
    command = [sys.executable, str(args.admission_tests.resolve())]
    result = subprocess.run(command, capture_output=True, text=True)
    dump(root / 'admission-command.json', {'argv': command, 'exit_code': result.returncode})
    (root / 'admission.stdout').write_text(result.stdout)
    (root / 'admission.stderr').write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    from benchmarks.hermes_ml_checkpoint_v4.test_orchestration import run_checks
    run_checks(root / 'orchestration-tests')
    requests = list((root / 'orchestration-tests/live-output').glob('api-*-forwarded.json'))
    assert len(requests) == 16
    assert all(json.loads(p.read_text())['model'] == MODEL for p in requests)
    assert frozen == {p.name: sha(p) for p in HERE.iterdir() if p.is_file()}
    receipt = {
        'passed': True, 'validation_kind': 'model_only_differential_plus_native_handoff',
        'checks': ['Exact model-only source delta against passing full preflight',
                   'Published Gnomon 1.2.0 runtime and three arms unchanged',
                   'Service admission regression tests',
                   'Native Hermes handoff, correction and request budgets with new model ID'],
        'inherited_preflight_sha256': sha(args.baseline_preflight),
        'differential_sha256': sha(root / 'differential.json'),
        'tested_sources': frozen, 'paid_api_calls': 0,
    }
    dump(root / 'passed.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
