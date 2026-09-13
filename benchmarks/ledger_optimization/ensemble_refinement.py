"""Execute frozen 044 experiment with a separately recorded solver tolerance.

Load an isolated copy of the 044 runner and numerical module so neither frozen
source is edited. The only changed numerical option is ftol=1e-12.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from scipy.optimize import minimize


def isolated(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(original, warm, output):
    here = Path(__file__).parent
    numerical = isolated('benchmarks.ledger_optimization._ensemble_045', here/'evidence_ensemble.py')
    runner = isolated('benchmarks.ledger_optimization._ensemble_runner_045', here/'broad_ensemble.py')
    def refined_minimize(*args, **kwargs):
        kwargs['options'] = {**kwargs['options'], 'ftol': 1e-12}
        return minimize(*args, **kwargs)
    numerical.minimize = refined_minimize
    runner.fit = numerical.fit
    output = Path(output)
    # The runner owns output creation. Preserve wrapper identity on success or
    # failure, without changing any artifacts in the original 044 directory.
    if output.exists():raise FileExistsError(output)
    try:
        return runner.run(original, warm, output)
    finally:
        if output.is_dir():
            (output/'refinement.json').write_text(json.dumps({
                'protocol': 'BROAD_ENSEMBLE_045.md', 'ftol': 1e-12, 'maxiter': 500,
                'convex_gap_threshold': 1e-5,
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'protocol_sha256': hashlib.sha256((here/'BROAD_ENSEMBLE_045.md').read_bytes()).hexdigest(),
                'frozen_044_sources_edited': False}, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('original'); p.add_argument('warm'); p.add_argument('output'); a = p.parse_args()
    r = run(a.original, a.warm, a.output)
    print(json.dumps({k: v for k, v in r.items() if k != 'rows'}, indent=2))
