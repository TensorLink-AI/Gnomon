import argparse
import json
from .data import prepare
from .baselines import run_baselines

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare');a.add_argument('--archive',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('baselines');a.add_argument('--panel',required=True);a.add_argument('--output',required=True)
    a.add_argument('--smoke',action='store_true')
    args=vars(p.parse_args());command=args.pop('command')
    print(json.dumps((prepare if command=='prepare' else run_baselines)(**args),indent=2,allow_nan=False))

if __name__=='__main__':main()
