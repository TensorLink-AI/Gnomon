import argparse
import json
from .data import prepare
from .baselines import run_baselines
from .agent import execute, resolve
from .ledger import build_history

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare');a.add_argument('--archive',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('baselines');a.add_argument('--panel',required=True);a.add_argument('--output',required=True)
    a.add_argument('--smoke',action='store_true')
    a=s.add_parser('execute');a.add_argument('--case',required=True);a.add_argument('--output',required=True)
    a.add_argument('--provider',required=True);a.add_argument('--backend',default='gnomon')
    a=s.add_parser('resolve');a.add_argument('--case',required=True);a.add_argument('--executions',required=True);a.add_argument('--selection')
    a=s.add_parser('ledger');a.add_argument('--case',required=True);a.add_argument('--output',required=True)
    args=vars(p.parse_args());command=args.pop('command')
    result={'prepare':prepare,'baselines':run_baselines,'execute':execute,'resolve':resolve,'ledger':build_history}[command](**args)
    print(json.dumps(result,indent=2,allow_nan=False))
    if command=='resolve' and not result['resolved']:raise SystemExit(2)

if __name__=='__main__':main()
