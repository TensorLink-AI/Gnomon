"""Offline readiness checks and a separately marked synthetic nine-model CPU smoke."""
import argparse
import json
import math
from pathlib import Path

from .common import PROTOCOL, identity, read, resources, write
from .models import FAMILIES
from .run import bounded_forecast


def preflight(smoke=False):
    protocol=read(PROTOCOL)
    result={'resources':resources(),'identity':identity(),'models':FAMILIES,
            'paid_requests':0,'benchmark_outcomes':False}
    if any(v is None for v in result['identity']['dependencies'].values()):
        raise RuntimeError('Install benchmark requirements before running preflight')
    if smoke:
        history=[10 + .01*i + 2*math.sin(i*2*math.pi/7) for i in range(140)]
        result['synthetic_smoke']={name:bounded_forecast({'name':name,'history':history,'horizon':8,
            'season':7,'seed':protocol['seed']},protocol['model_timeout_seconds']) for name in protocol['models']}
        result['ready']=all(v['status']=='ok' for v in result['synthetic_smoke'].values())
    else:
        result['ready']=False
        result['reason']='resource/dependency inspection only; request --smoke to exercise models'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=preflight(args.smoke)
    write(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='identity'},indent=2))
    if args.smoke and not result['ready']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
