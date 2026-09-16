import argparse,json
from .agent import execute,resolve

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('execute')
    for name in ('case','output','provider'):a.add_argument('--'+name,required=True)
    a.add_argument('--backend',default='gnomon')
    a=s.add_parser('resolve');a.add_argument('--case',required=True);a.add_argument('--executions',required=True);a.add_argument('--selection')
    args=vars(p.parse_args());op=args.pop('command');result={'execute':execute,'resolve':resolve}[op](**args)
    print(json.dumps(result,allow_nan=False))
    if op=='resolve' and not result['resolved']:raise SystemExit(2)
if __name__=='__main__':main()
