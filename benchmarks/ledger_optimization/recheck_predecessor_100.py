"""Re-audit only the complete 097 development run using its frozen analyzer."""
import argparse
import json
from pathlib import Path

from .contrast_plan_100 import PARENT_PLAN_SHA, read, sha
from .continue_collection_096 import configure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('capsule','parent-plan','runtime','root','output'):parser.add_argument('--'+field,type=Path,required=True)
    args=parser.parse_args()
    parent=read(args.parent_plan)
    if sha(args.parent_plan)!=PARENT_PLAN_SHA or sha(args.capsule/'capsule.json')!=parent['capsule_sha256']:
        raise ValueError('Exact 097 plan and capsule required')
    if args.output.exists() or args.output.is_symlink():raise ValueError('Fresh audit output required')
    original=read(args.root/'report.json')
    if len(original['rows'])!=312 or not original['complete']:raise ValueError('Complete predecessor required')
    helper=configure(args.capsule,args.runtime)
    result=helper.analyze(args.root,args.output)
    if result['audit_failures'] or result['shutdown_record_gaps'] or not result['complete']:
        raise ValueError('Independent predecessor audit failed')
    if result['rows']!=original['rows'] or result['arms']!=original['arms']:
        raise ValueError('Re-audit disagrees with original report')
    passed={'passed':True,'checks':result['audit_checks'],'sessions':312,'original_report_sha256':sha(args.root/'report.json'),
            'rechecked_report_sha256':sha(args.output/'report.json'),'engy_calls':0,'provider_calls':0,'final_gate_opened':False}
    (args.output/'passed.json').write_text(json.dumps(passed,indent=2)+'\n')
    print(json.dumps(passed))


if __name__=='__main__':main()
