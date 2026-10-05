"""Run a serial CPU suite; the scored stage requires a matching successful pilot."""
import argparse
import fcntl
import os
import signal
import json
from pathlib import Path
import subprocess
import sys
import time

from .common import PROTOCOL, digest, identity, read, resources, write
from .data import download, prepare
from .report import report


def execute(root,cache,mode,pilot=None):
    root,cache=Path(root),Path(cache)
    protocol=read(PROTOCOL)
    pinned=identity()
    host={k:v for k,v in resources().items() if k!='memory.current'}
    if mode=='evaluation':
        if pilot is None:
            raise ValueError('Scored stage requires --pilot pointing to a completed pilot suite')
        prior=read(Path(pilot)/'suite.json')
        if prior['runtime']!=pinned or prior['protocol_sha256']!=digest(protocol):
            raise ValueError('Pilot identity differs; rerun feasibility before scoring')
        evidence=read(Path(pilot)/'report.json')
        finished=read(Path(pilot)/'FINISHED.json')
        if finished['report_sha256']!=digest(evidence):
            raise ValueError('Pilot report changed')
        if prior['host']!=host:
            raise ValueError('Pilot must run on the same CPU host/allocation')
        if not evidence['pilot_feasible']:
            raise ValueError('Pilot did not meet coverage/failure gate')
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'suite.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(root/'suite.json',{'runtime':pinned,'protocol_sha256':digest(protocol),'mode':mode,'host':host})
    if (root/'FINISHED.json').exists():
        finished=read(root/'FINISHED.json')
        if finished['report_sha256']!=digest(read(root/'report.json')):
            raise ValueError('Completed report changed')
        print('Suite already finished; retained immutable outputs')
        return
    started=time.monotonic()
    for spec in protocol['datasets']:
        if mode=='pilot' and spec['id'] not in protocol['pilot_datasets']:
            continue
        name=spec['id'].replace('/','_')
        path,source=download(protocol,spec,cache)
        manifest=prepare(protocol,spec,path,source,mode)
        manifest_path=root/'manifests'/(name+'.json')
        write(manifest_path,manifest)
        output=root/'runs'/name
        if mode=='evaluation' and spec['id'] in protocol['pilot_datasets']:
            pilot_manifest=read(Path(pilot)/'manifests'/(name+'.json'))
            if {s['id'] for s in pilot_manifest['series']} & {s['id'] for s in manifest['series']}:
                raise ValueError('Pilot and scored series overlap')
        cap=protocol['pilot_wall_seconds'] if mode=='pilot' else protocol['evaluation_wall_seconds']
        steps=[['benchmarks.cpu_eval.run','--manifest',str(manifest_path),'--output',str(output)],
               ['benchmarks.cpu_eval.analyze','--run',str(output)]]
        for index,argv in enumerate(steps):
            if identity()!=pinned:
                raise ValueError('Frozen runtime changed')
            log=root/(name+f'.{index}.log')
            print(json.dumps({'dataset':spec['id'],'step':argv[0],'mode':mode,'log':str(log)}),flush=True)
            with log.open('a') as stream:
                process=subprocess.Popen([sys.executable,'-m',*argv],stdout=stream,stderr=subprocess.STDOUT,
                                         start_new_session=True)
                try:
                    returncode=process.wait(timeout=cap+120)
                except BaseException:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
                    raise
            if returncode:
                raise RuntimeError(f'{argv[0]} failed; inspect {log}; no automatic retry')
    result=report(root/'runs',mode)
    write(root/'report.json',result)
    write(root/'FINISHED.json',{'runtime':pinned,'protocol_sha256':digest(protocol),
                              'elapsed_seconds':time.monotonic()-started,'report_sha256':digest(result)})
    print(json.dumps({'complete':result['complete'],'pilot_feasible':result['pilot_feasible'],
                      'report':str(root/'report.json')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--mode',choices=['pilot','evaluation'],required=True)
    parser.add_argument('--pilot',type=Path)
    args=parser.parse_args()
    execute(args.root.resolve(),args.cache.resolve(),args.mode,args.pilot)


if __name__=='__main__':
    main()
