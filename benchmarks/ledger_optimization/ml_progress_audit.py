"""Audit a closed-session snapshot without changing a live or archived trial.

Uses the trial's frozen auditor; links only completed immutable round directories.
Never promotes a partial run or sends a provider/API request.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types


def audit(source,output):
    source=source.resolve();output=output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text())
    frozen=source/'frozen-source'
    assert all((frozen/n).is_file() and hashlib.sha256((frozen/n).read_bytes()).hexdigest()==h
               for n,h in manifest['sources'].items()), 'Frozen auditor source differs from trial manifest'
    files=[frozen/n for n in manifest['sources']];included=[];incomplete=[]
    for name in ('manifest.json','host-jobs.json'):
        shutil.copyfile(source/name,output/name);files.append(source/name)
    for grade in sorted(source.glob('*/*/round-*/grade.json')):
        directory=grade.parent
        # memory.json is written after the host copies the agent's project.
        if not (directory/'memory.json').is_file():
            incomplete.append(str(directory.relative_to(source)));continue
        expected=json.loads((directory/'input-hashes.json').read_text())
        if any(not (directory/'project'/n).is_file() for n in expected):
            incomplete.append(str(directory.relative_to(source)));continue
        target=output/directory.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        target.symlink_to(directory,target_is_directory=True)
        included.append(str(directory.relative_to(source)))
        files.extend(p for p in directory.rglob('*') if p.is_file())
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    # Import from saved frozen-source, not the possibly changing development tree.
    package='_frozen_ml_progress_'+hashlib.sha256(str(source).encode()).hexdigest()[:12]
    module=types.ModuleType(package);module.__path__=[str(source/'frozen-source')]
    sys.modules[package]=module
    spec=importlib.util.spec_from_file_location(package+'.analyze',source/'frozen-source/analyze.py')
    analyzer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=analyzer;spec.loader.exec_module(analyzer)
    report=analyzer.analyze(output)
    assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in hashes.items()), 'Source changed during snapshot audit'
    receipt={'at':datetime.now(timezone.utc).isoformat(),'source':str(source),'included':included,
             'not_yet_closed':incomplete,'source_files_unchanged':True,'source_hashes':hashes,
             'audit_checks':report['audit_checks'],'audit_failures':report['audit_failures'],
             'complete':report['complete'],'promotion_evaluated':False,
             'note':'Progress snapshot only. Frozen pipeline owns promotion after all planned sessions.'}
    (output/'snapshot-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=audit(args.source,args.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ('source_hashes','included')},indent=2))
