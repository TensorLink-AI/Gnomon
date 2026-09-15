"""Archive one candidate-100 pilot; never retry, continue, or open final data."""
import argparse
import json
from pathlib import Path
import sys

from .control_collection_096 import supervise

FIELDS=('parent-plan','capsule','worker-proof-root','plan','preflight','task-source','runtime',
        'previous-root','previous-launch','previous-capsule','predecessor-audit','output','credentials-file')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-directory',type=Path,required=True)
    for field in FIELDS:parser.add_argument('--'+field,type=Path,required=True)
    args=parser.parse_args()
    launch=args.launch_directory.resolve();audit=args.predecessor_audit.resolve()
    if launch not in audit.parents:
        raise ValueError('Predecessor audit must be inside the fresh controller directory so it is archived')
    command=[sys.executable,'-m','benchmarks.ledger_optimization.launch_contrast_100']
    for field in FIELDS:command+=['--'+field,str(getattr(args,field.replace('-','_')).absolute())]
    result=supervise(command,args.launch_directory,args.output,args.credentials_file)
    print(json.dumps(result))
    if not result['complete']:raise SystemExit(1)


if __name__=='__main__':main()
