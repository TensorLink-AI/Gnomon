import argparse
import json

from .agent import execute, resolve
from .data import prepare
from .ledger import build_history
from .run import run_baselines
from .score import score_submissions


def main():
    parser = argparse.ArgumentParser(description='Online Retail II real-sales benchmark, development-first.')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--archive', required=True); p.add_argument('--output', required=True)
    p.add_argument('--phase', choices=['development','validation','final'], default='development')
    p.add_argument('--release')
    p = sub.add_parser('run-baselines')
    p.add_argument('--panel', required=True); p.add_argument('--output', required=True); p.add_argument('--smoke', action='store_true')
    p = sub.add_parser('execute')
    p.add_argument('--case', required=True); p.add_argument('--output', required=True)
    p.add_argument('--provider', required=True); p.add_argument('--backend', choices=['direct','gnomon'], default='gnomon')
    p = sub.add_parser('resolve')
    p.add_argument('--case', required=True); p.add_argument('--executions', required=True); p.add_argument('--selection')
    p = sub.add_parser('ledger')
    p.add_argument('--case', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('score')
    for name in ('panel','baseline-run','submissions','output','arm'):
        p.add_argument('--'+name, required=True)
    args = vars(parser.parse_args()); command = args.pop('command')
    fn = {'prepare':prepare, 'run-baselines':run_baselines, 'execute':execute,
          'resolve':resolve, 'ledger':build_history, 'score':score_submissions}[command]
    result = fn(**args)
    print(json.dumps(result, indent=2, allow_nan=False))
    if command == 'resolve' and not result['resolved']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
